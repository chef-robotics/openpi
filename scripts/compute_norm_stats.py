"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import json

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=8,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def apply_gripper_transform(values: np.ndarray, gripper_indices: list[int], gripper_range: tuple[float, float]) -> np.ndarray:
    """Transform gripper joints from physical range to [1, 0] range.
    
    Args:
        values: Array of joint values with shape (..., num_joints)
        gripper_indices: List of indices corresponding to gripper joints
        gripper_range: Tuple of (min_val, max_val) for the physical gripper range
    
    Returns:
        Transformed values where gripper joints are mapped to [1, 0] (closed -> 1, open -> 0)
    """
    values = values.copy()
    min_val, max_val = gripper_range
    for idx in gripper_indices:
        # Map [min_val, max_val] -> [1, 0]
        # closed (min_val) -> 1, open (max_val) -> 0
        values[..., idx] = 1.0 - (values[..., idx] - min_val) / (max_val - min_val)
    return values


def main(config_name: str, max_frames: int | None = None, gripper_indices: list[int] | None = None, gripper_range: tuple[float, float] = (0.0, 0.045)):
    """Compute normalization statistics for a config.
    
    Args:
        config_name: Name of the training config
        max_frames: Maximum number of frames to use for computing stats
        gripper_indices: List of gripper joint indices to apply custom transform (e.g., [6, 13] for bimanual arms).
                        If None, no gripper transform is applied.
        gripper_range: Physical range of gripper joints as (min, max). Default is (0.0, 0.045).
    """
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            values = np.asarray(batch[key][0])
            
            # Apply gripper transformation if specified
            if gripper_indices is not None:
                values = apply_gripper_transform(values, gripper_indices, gripper_range)
            
            stats[key].update(values.reshape(-1, values.shape[-1]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    
    # Store gripper transformation metadata as a custom dict (not NormStats)
    if gripper_indices is not None:
        norm_stats["gripper_metadata"] = {
            "indices": gripper_indices,
            "physical_range": list(gripper_range),  # [min, max] in meters
        }
        print(f"Applied gripper transform to indices {gripper_indices} with range {gripper_range}")

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    
    # Save with custom serialization to handle the gripper_metadata
    path = output_path / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert norm_stats to serializable format
    serializable_stats = {}
    for key, value in norm_stats.items():
        if key == "gripper_metadata":
            serializable_stats[key] = value
        else:
            serializable_stats[key] = {
                "mean": value.mean.tolist(),
                "std": value.std.tolist(),
                "q01": value.q01.tolist() if value.q01 is not None else None,
                "q99": value.q99.tolist() if value.q99 is not None else None,
            }
    
    path.write_text(json.dumps({"norm_stats": serializable_stats}, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
