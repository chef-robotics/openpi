# Gripper Normalization Guide

This guide explains how to use the custom gripper normalization feature for bimanual WidowX arms where gripper joints have a physical range of 0-0.045m and need to be mapped to [1, 0] in the policy space.

## Overview

The gripper normalization handles the mapping:
- **Physical space**: 0 (closed) to 0.045 (open)
- **Policy space**: 1 (closed) to 0 (open)

This ensures the policy learns a consistent representation of gripper states across different robot configurations.

## Step 1: Compute Normalization Statistics

When computing normalization statistics for your dataset, specify the gripper joint indices and their physical range:

```bash
python scripts/compute_norm_stats.py \
    your_config_name \
    --gripper-indices [6, 13] \
    --gripper-range "(0.0, 0.045)"
```

### Parameters:
- `--gripper-indices`: List of joint indices for gripper joints
  - For bimanual WidowX: typically `[6, 13]` (left gripper, right gripper)
  - For single arm: typically `[6]`
- `--gripper-range`: Tuple of (min, max) physical gripper values
  - Default: `(0.0, 0.045)` meters
  - Adjust if your gripper has a different physical range

### What happens:
1. Gripper joint values in your dataset are transformed from [0, 0.045] to [1, 0]
2. Statistics (mean, std, q01, q99) are computed on the transformed values
3. The transformation metadata is saved in `norm_stats.json` under the `gripper_metadata` key

### Example norm_stats.json:
```json
{
  "norm_stats": {
    "state": {
      "mean": [...],
      "std": [...],
      "q01": [...],
      "q99": [...]
    },
    "actions": {
      "mean": [...],
      "std": [...],
      "q01": [...],
      "q99": [...]
    },
    "gripper_metadata": {
      "indices": [6, 13],
      "physical_range": [0.0, 0.045]
    }
  }
}
```

## Step 2: Training

Train your policy as usual. The training code will use the computed normalization statistics, and the gripper joints will be in the [1, 0] range during training.

## Step 3: Inference with Reverse Transformation

When running inference on the real robot, provide the path to `norm_stats.json` so the policy outputs can be transformed back to physical gripper values:

```bash
python examples/trossen_ai/main.py \
    --mode autonomous \
    --task_prompt "grab and handover red cube" \
    --norm_stats_path /path/to/norm_stats.json
```

### Parameters:
- `--norm_stats_path`: Path to the `norm_stats.json` file containing gripper transformation metadata

### What happens:
1. The script loads the gripper transformation metadata from `norm_stats.json`
2. Policy outputs actions in the [1, 0] range for gripper joints
3. Before executing, gripper joints are transformed back to [0, 0.045] physical range
4. Actions are sent to the robot in physical units

## Implementation Details

### Forward Transform (Training):
```python
# Map physical range [0, 0.045] to policy range [1, 0]
normalized = 1.0 - (physical_value - min_val) / (max_val - min_val)
```

### Reverse Transform (Inference):
```python
# Map policy range [1, 0] back to physical range [0, 0.045]
physical_value = min_val + (1.0 - normalized) * (max_val - min_val)
```

## Example Workflow

### 1. Compute normalization stats:
```bash
cd /home/sherry/ChefResearch/sandi/third_party/openpi
python scripts/compute_norm_stats.py \
    trossen_bimanual_config \
    --gripper-indices [6, 13] \
    --gripper-range "(0.0, 0.045)"
```

### 2. Train your policy (standard training):
```bash
python train.py --config trossen_bimanual_config
```

### 3. Run inference with gripper transformation:
```bash
python examples/trossen_ai/main.py \
    --mode autonomous \
    --task_prompt "pick up the red cube" \
    --norm_stats_path /path/to/checkpoint/assets/trossen/norm_stats.json
```

## Troubleshooting

### Issue: Actions seem inverted
- Check that the gripper indices are correct
- Verify the physical range matches your robot (0 = closed, 0.045 = open)

### Issue: Gripper doesn't move smoothly
- Ensure the gripper transformation is applied BEFORE other action processing
- Check that the physical range in norm_stats matches your robot's actual range

### Issue: norm_stats.json doesn't have gripper_metadata
- Make sure you ran `compute_norm_stats.py` with `--gripper-indices` specified
- If using pre-computed stats, you may need to recompute them

## Notes

- The transformation is only applied to the specified gripper joint indices
- All other joints remain unchanged
- The transformation is bi-directional: forward during training, reverse during inference
- If no `--norm_stats_path` is provided, actions are executed without transformation (backward compatible)

