import modal
import os
from pathlib import Path


def _find_local_repo_root() -> Path:
    """Computes full path of local SkyRL repo robustly

    Raises:
        Exception: if cannot find local SkyRL repo

    Returns:
        Path: path object describing full path of local SkyRL repo
    """
    candidates = [Path(__file__).resolve(), Path.cwd()]
    for start in candidates:
        for base in [start] + list(start.parents):
            if (base / "skyrl-train").exists() and (base / "skyrl-gym").exists():
                return base
    print("Warning: returning cwd as fall back")
    return Path.cwd()


def create_modal_image() -> modal.Image:
    """Creates a Modal image for Modal container. This uses the SkyRL container as
    a base image. It also mounts the local SkyRL repo to the container

    Returns:
        modal.Image: container image
    """

    local_repo_path = _find_local_repo_root()
    print(f"Root path: {local_repo_path}")

    envs = {
        "SKYRL_REPO_ROOT": "/root/SkyRL",  # where to put SkyRL in container
    }
    if os.getenv("WANDB_API_KEY", False):
        envs["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

    return (
        modal.Image.from_registry("novaskyai/skyrl-train-ray-2.48.0-py3.12-cu12.8")
        .env(envs)
        .add_local_dir(
            local_path=str(local_repo_path),
            remote_path="/root/SkyRL",
            ignore=[
                ".venv",
                "*.pyc",
                "__pycache__",
                ".git",
                "*.egg-info",
                ".pytest_cache",
                "node_modules",
                ".DS_Store",
            ],
        )
    )


def create_modal_volume(volume_name: str = "skyrl-data") -> dict[str, modal.Volume]:
    """Creates volume to attach to container.

    Args:
        volume_name (str, optional): Name of volume. Creates a new
        volume if given name does not exist. Defaults to "skyrl-data".

    Returns:
        dict[str, modal.Volume]: location in container to attach the volume & volume itself
    """
    data_volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    return {"/root/data": data_volume}  # mounts volume at /root/data inside container


app = modal.App(os.getenv("MODAL_APP_NAME", "my_skyrl_app"))
image = create_modal_image()
volume = create_modal_volume()


@app.function(
    image=image,
    gpu="L4:1",
    volumes=volume,
    timeout=3600,  # 1 hour
)
def run_script(command: str, skyrl_path: str):
    """
    Run any command from the SkyRL repo.
    Example: run_script.remote("uv run examples/gsm8k/gsm8k_dataset.py --output_dir /root/data/gsm8k")
    """
    import subprocess
    import os

    # The repo root is already set in the image environment
    repo_root = os.environ.get("SKYRL_REPO_ROOT", "/root/SkyRL")

    # Print current environment for debugging
    print(f"Container repo root: {repo_root}")
    print(f"Initial working directory: {os.getcwd()}")

    # Change to the skyrl-train directory
    run_command_dir = os.path.join(repo_root, skyrl_path.lstrip("/"))
    os.chdir(run_command_dir)
    print(f"Changed to directory: {os.getcwd()}")

    # Ensure skyrl-gym exists inside working_dir so uv can resolve editable path
    gym_src = os.path.join("..", "skyrl-gym")
    gym_dst = os.path.join(".", "skyrl-gym")
    if not os.path.exists(gym_dst):
        if os.path.exists(gym_src):
            print("Copying ../skyrl-gym into working_dir for uv packaging")
            subprocess.run(
                f"cp -r {gym_src} {gym_dst}",
                shell=True,
                check=True,
            )
        else:
            raise Exception("Cannot find skyrl-gym source")

    print("Initializing ray cluster in command line")
    subprocess.run(
        "ray start --head",
        shell=True,
        check=True,
    )
    # Use 'auto' to automatically detect the Ray cluster instead of hardcoded IP
    os.environ["RAY_ADDRESS"] = "auto"

    # Create symlink so /home/ray/data points to /root/data (where volume is mounted)
    print("\n=== Setting up data directory symlink ===")
    os.makedirs("/home/ray", exist_ok=True)
    if os.path.islink("/home/ray/data"):
        print("Symlink /home/ray/data already exists")
    elif os.path.exists("/home/ray/data"):
        print("Warning: /home/ray/data exists but is not a symlink")
    else:
        os.symlink("/root/data", "/home/ray/data")
        print("Created symlink: /home/ray/data -> /root/data")

    print(f"Running command: {command}")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 60)

    # Run the command with live output streaming
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True,
    )

    # Stream output line by line
    for line in process.stdout:
        print(line, end="")  # Print each line as it comes

    # Wait for process to complete
    returncode = process.wait()

    print("=" * 60)
    if returncode != 0:
        raise Exception(f"Command failed with exit code {returncode}")


@app.local_entrypoint()
def main(command: str = "nvidia-smi", path_in_skyrl: str = "skyrl-train"):
    """Main entry-point for running a command in Modal-integrated SkyRL environmenmt

    Args:
        command (str, optional): Command to run. Defaults to "nvidia-smi".
        path_in_skyrl (str, optional): Directory in which to run the command.
            This must be a valid directory within your local SkyRL repo. Defaults to "skyrl-train".

    Examples:
        modal run main.py --command "uv run examples/gsm8k/gsm8k_dataset.py --output_dir /root/data/gsm8k"
        MODAL_APP_NAME=benji_skyrl_app WANDB_API_KEY=... modal run main.py --command "bash examples/gsm8k/run_generation_gsm8k.sh"
    """
    print(f"{'=' * 5} Submitting command to Modal: {command} {'=' * 5}")
    print(f"{'=' * 5} Running inside: {path_in_skyrl} {'=' * 5}")
    run_script.remote(command, path_in_skyrl)
    print(f"\n{'=' * 5} Command completed successfully {'=' * 5}")
