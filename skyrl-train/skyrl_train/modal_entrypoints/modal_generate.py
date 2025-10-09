import modal
from pathlib import Path

app = modal.App("benji-skyrl-app")

# Compute the SkyRL repo root robustly (works when this file is mounted standalone)
def _find_repo_root() -> Path:
    candidates = [Path(__file__).resolve(), Path.cwd()]
    for start in candidates:
        for base in [start] + list(start.parents):
            if (base / "skyrl-train").exists() and (base / "skyrl-gym").exists():
                return base
    # Fallback: if not found, return current working directory
    return Path.cwd()

repo_path = _find_repo_root()
print(f"Root path: {repo_path}")

# This syncs your local code to /root/SkyRL in the container
image = (
    modal.Image.from_registry("novaskyai/skyrl-train-ray-2.48.0-py3.12-cu12.8")
    .env({"SKYRL_REPO_ROOT": "/root/SkyRL"})  # Set this environment variable in the image
    .add_local_dir(
        local_path=str(repo_path),
        remote_path="/root/SkyRL",
        ignore=[".venv", "*.pyc", "__pycache__", ".git", "*.egg-info", ".pytest_cache", "node_modules", ".DS_Store"]
    )

)

# Create external volume for datasets

data_volume = modal.Volume.from_name("skyrl-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100:1",
    volumes={"/root/data": data_volume},
    timeout=60 * 60, # 1 hour
)
def run_script(command: str):
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
    skyrl_train_dir = os.path.join(repo_root, "skyrl-train")
    os.chdir(skyrl_train_dir)
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
            print("Warning: ../skyrl-gym not found; uv editable path may fail")

    # Rewrite pyproject to reference ./skyrl-gym (relative to working_dir)
    try:
        pyproject_path = os.path.join(os.getcwd(), "pyproject.toml")
        with open(pyproject_path, "r", encoding="utf-8") as f:
            py_text = f.read()
        new_text = py_text.replace('path = "../skyrl-gym"', 'path = "./skyrl-gym"')
        if new_text != py_text:
            with open(pyproject_path, "w", encoding="utf-8") as f:
                f.write(new_text)
            print("Updated pyproject.toml to use ./skyrl-gym for uv sources")
    except Exception as e:
        print(f"Warning: failed to rewrite pyproject.toml for uv sources: {e}")
    # Ensure Ray points to a local head inside this container
    for var in ["RAY_ADDRESS", "RAY_HEAD_NODE", "RAY_GCS_ADDRESS"]:
        os.environ.pop(var, None)
    try:
        subprocess.run(
            "ray status --address 127.0.0.1:6379",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        print("Initializing ray cluster in command line")
        subprocess.run(
            "ray start --head --disable-usage-stats --port 6379",
            shell=True,
            check=True,
        )
    os.environ["RAY_ADDRESS"] = "127.0.0.1:6379"

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
        universal_newlines=True
    )

    # Stream output line by line
    for line in process.stdout:
        print(line, end='')  # Print each line as it comes

    # Wait for process to complete
    returncode = process.wait()

    print("=" * 60)
    if returncode != 0:
        raise Exception(f"Command failed with exit code {returncode}")



@app.local_entrypoint()
def main(command: str = "nvidia-smi"):
    """
    Local entrypoint - runs on your Mac and calls the remote function.
    Usage: modal run main/run.py --command "uv run examples/gsm8k/gsm8k_dataset.py --output_dir /root/data/gsm8k"
    """
    print(f"Submitting command to Modal: {command}")
    result = run_script.remote(command)
    print("\n=== Command completed successfully ===")
    return result

