# modal_app.py
import modal
import os
import subprocess
import sys

# Choose python version matching your pyproject (your pyproject requires >=3.13)
PYTHON_VERSION = "3.13"

# Use NVIDIA CUDA image for maximum performance
cuda_tag = "12.1.0-devel-ubuntu22.04"  # Match CUDA version with PyTorch 2.9.0
image = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_tag}", add_python=PYTHON_VERSION)
    .pip_install(
        "torch==2.9.0",
        "torchvision==0.24.0",
        extra_index_url="https://download.pytorch.org/whl/cu121"
    )
    # Install system dependencies
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxrender1", 
        "libxext6",
        "wget",  # For ROMs
        "unzip",
    )
    # Install Python packages
    .pip_install(
        "stable-baselines3[extra]==2.7.0",
        "gymnasium==1.2.1",
        "ale-py==0.11.2",
        "pyyaml==6.0.3",
        "opencv-python-headless",  # Faster than opencv-python
        "tensorboard",
        "pynvml",  # For GPU monitoring
    )
    # copy the repository into the image
    .add_local_dir(".", remote_path="/root/project", copy=True)
)

# Option B (recommended if you need full CUDA toolkit / native builds):
# uncomment and use this instead of Option A when you need nvcc, tensorrt, or manual CUDA toolchain.
# cuda_tag = "12.8.1-devel-ubuntu24.04"
# image = (
#     modal.Image.from_registry(f"nvidia/cuda:{cuda_tag}", add_python=PYTHON_VERSION)
#     .entrypoint([])  # optional: silence base image entrypoint
#     .apt_install("git", "ffmpeg")  # example system deps
#     .pip_install("torch==2.9.0", "torchvision==0.24.0", "stable-baselines3[extra]==2.7.0")
#     .add_local_dir(".", remote_path="/root/project", copy=True)
# )

app = modal.App("atari-trainer")

# Persistent volume for runs/models/logs
runs_volume = modal.Volume.from_name("atari_trainer_runs", create_if_missing=True)

# A quick GPU sanity check function
@app.function(gpu="A100", image=image)
def gpu_check():
    import torch, subprocess, textwrap
    out = {}
    out["torch_version"] = torch.__version__
    out["cuda_available"] = torch.cuda.is_available()
    if out["cuda_available"]:
        try:
            out["device_name"] = torch.cuda.get_device_name(0)
        except Exception as e:
            out["device_name"] = f"error: {e}"
    else:
        out["device_name"] = None
    # also run nvidia-smi
    try:
        out["nvidia-smi"] = subprocess.check_output(["nvidia-smi"], text=True)
    except Exception as e:
        out["nvidia-smi"] = f"nvidia-smi failed: {e}"
    print(out)
    return out

# The main training launcher runs your existing train.py as a subprocess.
# It uses gpu="A100" as an example — change to T4, L40S, H100, B200, etc. as needed.
@app.function(gpu="A100", image=image, volumes={"/root/project/runs": runs_volume}, timeout=28800)
def run_training():
    import os, subprocess, sys
    os.chdir("/root/project")
    # Make sure the train.py CLI uses "cuda" as device so Torch will use GPU inside container
    cmd = [
        sys.executable, "train.py",
        "--config", "configs/pacman.yaml",
        "--device", "cuda",
        "--log-dir", "/root/project/runs",
        # add/override other CLI args as needed, e.g. "--algo", "PPO", "--total-timesteps", "1000000"
    ]
    # For longer runs you may want to stream logs (modal will capture stdout/stderr).
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    print("modal app module; use 'modal run modal_app.py:gpu_check' or 'modal run modal_app.py:run_training'")