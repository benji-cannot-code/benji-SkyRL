# # import modal

# # app = modal.App("example-get-started")

# # skyrl_image = modal.Image.from_registry("novaskyai/skyrl-train-ray-2.48.0-py3.12-cu12.8")

# # def func():
# #     print("hi this does not hv decorator")


# # @app.function()
# # def square(x):
# #     func()
# #     print("This code is running on a remote worker!")
# #     return x ** 2

# # @app.function(image=skyrl_image, gpu="L4:1")
# # def func_image():
# #     import fastapi
# #     import uvicorn
# #     from fastapi import Request
# #     from fastapi.middleware.cors import CORSMiddleware
# #     from fastapi.responses import JSONResponse
# #     from pydantic import BaseModel
# #     import asyncio
# #     import math
# #     import os
# #     import shutil
# #     from typing import Any, List, Optional, Dict, Tuple, Union
# #     # from jaxtyping import Float
# #     from pathlib import Path
# #     import ray
# #     from ray import ObjectRef
# #     import torch
# #     from loguru import logger
# #     from omegaconf import DictConfig
# #     from ray.util.placement_group import PlacementGroup, placement_group
# #     from tqdm import tqdm
# #     from transformers import AutoTokenizer
# #     print("All imports done!")

# # @app.local_entrypoint()
# # def main():
# #     print("the square is", square.remote(42))

# #     func_image.remote()


# # import modal

# # app = modal.App("skyrl-modal")

# # # pull their registry image as base
# # skyrl_image = (
# #     modal.Image.from_registry("novaskyai/skyrl-train-ray-2.48.0-py3.12-cu12.8")
# #     .apt_install("git", "gcc", "build-essential")   # system packages if needed
# #     .run_commands(
# #         # clone only the needed subdir (shallow clone) and install python deps
# #         "git clone --depth=1 https://github.com/novasky-ai/SkyRL.git /workspace/SkyRL || true",
# #         "cd /workspace/SkyRL/skyrl-train",
# #         # "cd /workspace/SkyRL/skyrl-train && pip install -r requirements.txt || true"
# #         "uv venv --python 3.12 /root/venvs/skyrl",
# #         "source /root/venvs/skyrl/bin/activate",
# #         "uv sync --active --extra vllm",
# #     )
# #     # optionally add specific pip packages if requirements.txt is incomplete
# #     .pip_install("ray==2.48.0", "torch")   # adapt versions to what SkyRL expects
# # )

# # def func():
# #     print("hi this does not hv decorator")


# # @app.function()
# # def square(x):
# #     func()
# #     print("This code is running on a remote worker!")
# #     return x ** 2

# # @app.function(image=skyrl_image, gpu="L4:1")
# # def func_image():
# #     # now imports that depend on SkyRL and its deps should work
# #     import fastapi
# #     import uvicorn
# #     from fastapi import Request
# #     from fastapi.middleware.cors import CORSMiddleware
# #     from fastapi.responses import JSONResponse
# #     from pydantic import BaseModel
# #     import asyncio
# #     import math
# #     import os
# #     import shutil
# #     from typing import Any, List, Optional, Dict, Tuple, Union
# #     # from jaxtyping import Float
# #     from pathlib import Path
# #     import ray
# #     from ray import ObjectRef
# #     import torch
# #     from loguru import logger
# #     from omegaconf import DictConfig
# #     from ray.util.placement_group import PlacementGroup, placement_group
# #     from tqdm import tqdm
# #     from transformers import AutoTokenizer
# #     print("All imports done!")
# #     print("All imports done on Modal GPU!")

# # @app.local_entrypoint()
# # def main():
# #     print("the square is", square.remote(42))
# #     func_image.remote()


# import modal

# app = modal.App("skyrl-uv-modal")

# skyrl_image = (
#     modal.Image.from_registry("novaskyai/skyrl-train-ray-2.48.0-py3.12-cu12.8")
#     .add_local_dir("/Users/bjx/Documents/SkyRL", remote_path="/root/SkyRL")
#     # optionally install small system deps if uv needs them (git usually present)
#     # .run_commands(
#     #     # clone repo
#     #     "git clone --depth=1 https://github.com/novasky-ai/SkyRL.git /workspace/SkyRL || true",
#     #     "cd /workspace/SkyRL/skyrl-train || true",
#     #     # # ensure uv CLI available: try basic install command if the image doesn't already include uv
#     #     # # replace this with your preferred uv-install method if different
#     #     # "python -m pip install --upgrade pip setuptools || true",
#     #     # "python -m pip install uv || true",
#     #     # # run uv to sync/install packages from UV lockfile / manifest
#     #     # # If your project expects a different uv command (e.g. `uv install`), change it here.
#     #     # "cd /workspace/SkyRL/skyrl-train && uv sync || true",
#     #     # # optionally pre-run any build steps (compile extensions, etc.)
#     #     # "cd /workspace/SkyRL/skyrl-train && uv run -- python -c \"print('uv sync done')\" || true"
#     # )
# )

# def func():
#     print("hi this does not hv decorator")


# @app.function()
# def square(x):
#     func()
#     print("This code is running on a remote worker!")
#     return x ** 2

# @app.function(image=skyrl_image, gpu="L4:1")
# def func_image():
#     # run under uv environment to validate imports
#     # uv run -- <cmd> runs the command under the uv-managed env
#     import os, sys, subprocess
#     project_dir = "/workspace/SkyRL/skyrl-train"
#     print("Running import test under uv-managed environment...")
#     print(f"cwd for uv: {project_dir}")
#     # 1) Minimal uv sanity check (should succeed even if heavy deps aren't present)
#     uv_minimal = subprocess.run(
#         [
#             sys.executable,
#             "-m",
#             "uv",
#             "run",
#             "--",
#             "python",
#             "-c",
#             ("import sys; print('uv ok, python:', sys.executable)")
#         ],
#         cwd=project_dir,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         text=True,
#         check=False,
#     )
#     print("uv minimal stdout:", uv_minimal.stdout)

# @app.local_entrypoint()
# def main():
#     square.remote(50)
#     func_image.remote()




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

