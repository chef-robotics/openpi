# openpi (Chef Fork)

Fork of [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) for the SANDI legacy (v2) workflow.

> **Note:** This repo is currently public but should be made private.

## Upstream Documentation

For full documentation including all models, PyTorch support, and fine-tuning guides, see:
- [Upstream README](./UPSTREAM_README.md)
- [Pi0 Paper](https://www.physicalintelligence.company/blog/pi0)
- [Pi0-FAST Paper](https://www.physicalintelligence.company/research/fast)
- [Pi0.5 Paper](https://www.physicalintelligence.company/blog/pi05)

This README covers Chef-specific information for the Trossen AI workflow.

## Workflow Context

This repo is part of the **legacy v2 workflow**:

| Workflow | Dataset Format | Repos |
|----------|---------------|-------|
| **New (v3)** | LeRobotDataset v3 | chef-lerobot + chef-lerobot_trossen |
| **Legacy (v2)** | LeRobotDataset v2.1 | trossen-lerobot + **openpi** |

Use this repo when:
- Training on existing v2.1 format datasets
- Using JAX-based Pi0/Pi0-FAST/Pi0.5 models
- Following Trossen's OpenPI integration docs

## Branch Structure

| Branch | Purpose |
|--------|---------|
| `trossen-ai` | **Working branch** - Chef development |
| Other branches | Feature branches (sherry/*, hw-eval, etc.) |

## Chef Modifications

- Hierarchical semi-autonomous pi0 prototype (#7)
- Updated camera serial numbers
- Trossen arm driver updates
- Graceful shutdown on Ctrl-C (#6)

## Setup

This repo has its own virtual environment:

```bash
cd ChefResearch/sandi/third_party/openpi

# Install with submodules
git submodule update --init --recursive

# Install dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

**Note:** `GIT_LFS_SKIP_SMUDGE=1` prevents downloading large files from LeRobot dependency.

## GPU Requirements

| Mode | Memory | Example GPU |
|------|--------|-------------|
| Inference | > 8 GB | RTX 4090 |
| LoRA Fine-tuning | > 22.5 GB | RTX 4090 |
| Full Fine-tuning | > 70 GB | A100/H100 |

## Training (Trossen AI)

```bash
# From openpi root
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_trossen_transfer_block \
    --exp-name=my_experiment \
    --overwrite
```

### Custom Training Config

Edit `src/openpi/training/config.py` to add custom configurations:

```python
TrainConfig(
    name="pi0_my_task",
    model=pi0.Pi0Config(...),
    data=LeRobotAlohaDataConfig(
        repo_id="your/dataset",
        default_prompt="your task description",
        ...
    ),
    num_train_steps=20_000,
    batch_size=2,
)
```

For full fine-tuning documentation including norm stats computation, see [UPSTREAM_README.md](./UPSTREAM_README.md).

## Inference (Trossen AI)

### 1. Start Policy Server

```bash
# From openpi root
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_trossen_transfer_block \
    --policy.dir=checkpoints/pi0_trossen_transfer_block/my_experiment/19999
```

### 2. Start Client

```bash
# From examples/trossen_ai (uses separate venv with LeRobot 0.3.2)
cd examples/trossen_ai
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv run main.py --mode autonomous --task_prompt "grab red cube"
```

### Important Version Note

OpenPI uses **two LeRobot versions**:
- **Root venv (LeRobot 0.1.0)**: For training and policy server
- **examples/trossen_ai venv (LeRobot 0.3.2)**: For robot client

Always run training commands from the root directory.

## Checkpoints

Checkpoints are saved to `checkpoints/`. Download pretrained checkpoints:
- [OpenPi Fine-Tuned Checkpoint](https://huggingface.co/shantanu-tr/open_pi_finetune_checkpoint)

For all available base and fine-tuned checkpoints (pi0, pi0-FAST, pi0.5), see [UPSTREAM_README.md](./UPSTREAM_README.md).

## Camera Configuration

Edit camera settings in `examples/trossen_ai/main.py`:

```python
bi_widowx_ai_config = BiWidowXAIFollowerConfig(
    left_arm_ip_address="192.168.1.5",
    right_arm_ip_address="192.168.1.4",
    cameras={
        "cam_high": RealSenseCameraConfig(serial_number_or_name="..."),
        "cam_low": RealSenseCameraConfig(serial_number_or_name="..."),
        ...
    }
)
```

## Inference Parameters

### Rate of Inference

Controls how often the policy is queried:
- **rate = 50**: Smoother motion, less responsive
- **rate = 25**: More responsive, jerkier

```python
self.rate_of_inference = 50  # Control steps per policy query
```

### Temporal Ensembling

Disabled by default (Pi0 paper recommends against it):
```python
self.temporal_ensemble_coefficient = None
```

## Development Workflow

### Making Changes

```bash
git checkout trossen-ai
# ... make changes ...
git add . && git commit -m "Description"
git push origin trossen-ai
```

### Update Parent Repo

```bash
cd ../..  # to sandi/
git add third_party/openpi
git commit -m "Update openpi submodule"
```

### Syncing with Upstream

```bash
git remote add upstream https://github.com/Physical-Intelligence/openpi.git
git fetch upstream
git checkout trossen-ai
git merge upstream/main
git push origin trossen-ai
```

## Key Directories

```
openpi/
├── scripts/
│   ├── train.py              # Training script
│   └── serve_policy.py       # Policy server
├── src/openpi/
│   ├── models/               # Pi0, PaLI-Gemma, etc.
│   ├── policies/             # Policy configs
│   ├── training/             # Training config, data loading
│   └── serving/              # WebSocket policy server
├── examples/
│   └── trossen_ai/           # Trossen AI client (separate venv)
│       ├── main.py           # Robot client
│       └── README.md         # Full workflow docs
└── packages/
    └── openpi-client/        # Client package
```

## Related Submodules

- **trossen-lerobot**: Provides datasets for this repo
- **chef-lerobot**: New v3 workflow (separate, not used with this)
- **chef-lerobot_trossen**: New v3 workflow (separate, not used with this)

## TODO

- [ ] Make this repository private
