# SkyRL Modal Integration

This integration allows you to run SkyRL commands inside a containerized environment hosted by [Modal](https://modal.com/), providing seamless access to GPU resources without managing infrastructure.

## Overview

The Modal integration:
- **Runs commands** from your local SkyRL repo inside Modal's cloud infrastructure
- **Sets up a container** based on the SkyRL base image (`novaskyai/skyrl-train-ray-2.48.0-py3.12-cu12.8`)
- **Mounts your local SkyRL repository** to `/root/SkyRL` in the container
- **Attaches persistent volumes** for data storage at `/root/data` and `/home/ray/data`
- **Initializes Ray** automatically for distributed computing
- **Streams live output** from your commands back to your local terminal

## Prerequisites

1. Install Modal: `pip install modal`
2. Set up Modal authentication: `modal setup`
3. Ensure you have the SkyRL repository cloned locally

## Usage

### Basic Command

```bash
modal run main.py --command "your-command-here"
```

### Examples

#### 1. Test GPU availability
```bash
modal run main.py --command "nvidia-smi"
```

#### 2. Generate GSM8K dataset
```bash
modal run main.py --command "uv run examples/gsm8k/gsm8k_dataset.py --output_dir /root/data/gsm8k"
```

#### 3. Run training script
```bash
modal run main.py --command "bash examples/gsm8k/run_generation_gsm8k.sh"
```

#### 4. Run from a different directory in the repo
```bash
modal run main.py \
  --command "uv run search/run_skyrl_train_search.py" \
  --path-in-skyrl "skyrl-train/examples"
```

## Configuration

### Command Parameters

- `--command`: The command to execute in the container (default: `"nvidia-smi"`)
- `--path-in-skyrl`: Directory within the SkyRL repo where the command should run (default: `"skyrl-train"`)

### Environment Variables

Configure the integration using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODAL_APP_NAME` | Name of your Modal app (useful for team collaboration) | `"my_skyrl_app"` |
| `WANDB_API_KEY` | Weights & Biases API key for experiment tracking | None |

### Example with Environment Variables

```bash
MODAL_APP_NAME=benji_skyrl_app \
WANDB_API_KEY=your_wandb_key_here \
modal run main.py --command "bash examples/gsm8k/run_generation_gsm8k.sh"
```

## How It Works

### 1. Image Creation
The `create_modal_image()` function:
- Pulls the SkyRL base Docker image
- Sets environment variables (`SKYRL_REPO_ROOT`, `SKYRL_USING_MODAL`, `WANDB_API_KEY`)
- Mounts your local SkyRL repository to `/root/SkyRL` in the container
- Excludes unnecessary files (`.venv`, `.git`, `__pycache__`, etc.)

### 2. Volume Management
The `create_modal_volume()` function:
- Creates or attaches a persistent volume named `"skyrl-data"`
- Mounts it at `/root/data` for data persistence across runs
- Creates a symlink at `/home/ray/data` pointing to `/root/data`

### 3. Command Execution
The `run_script()` function:
- Changes to the specified directory within the SkyRL repo
- Ensures `skyrl-gym` is available for dependencies
- Starts a Ray cluster with `ray start --head`
- Executes your command with live output streaming
- Returns exit codes for error handling

## Resource Configuration

By default, the integration uses:
- **GPU**: 1x NVIDIA L4 GPU
- **Timeout**: 3600 seconds (1 hour)

To modify resources, edit the `@app.function()` decorator in `main.py`:

```python
@app.function(
    image=image,
    gpu="A100:1",  # Change GPU type/count
    volumes=volume,
    timeout=7200,  # Change timeout (in seconds)
)
```

## Data Persistence

Data stored in `/root/data` (or `/home/ray/data`) persists across Modal runs. This is useful for:
- Storing generated datasets
- Saving model checkpoints
- Caching intermediate results

## Notes

- The local SkyRL repository is automatically detected by traversing parent directories
- Files in `.gitignore` are excluded from the container mount
- Ray is automatically configured to use `127.0.0.1:6379`
- All output is streamed in real-time to your terminal

## Support

For issues with:
- **Modal platform**: See [Modal documentation](https://modal.com/docs)
- **SkyRL integration**: Check SkyRL repository issues or documentation

