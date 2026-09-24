#!/usr/bin/env bash
set -euo pipefail

CONDA_PREFIX_PATH=/root/pc/conda-envs/gqa-process-consistency
OUTPUT_DIR=/root/pc/gpc_dynaloop_storage/results/environment_baseline

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_PREFIX_PATH"
mkdir -p "$OUTPUT_DIR"

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUTPUT_DIR/captured_at_utc.txt"
uname -a > "$OUTPUT_DIR/uname.txt"
cat /etc/os-release > "$OUTPUT_DIR/os-release.txt"
nvidia-smi > "$OUTPUT_DIR/nvidia-smi.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$OUTPUT_DIR/gpu.csv"
conda list --explicit > "$OUTPUT_DIR/conda-explicit.txt"
python -m pip freeze > "$OUTPUT_DIR/pip-freeze.txt"
vulkaninfo --summary > "$OUTPUT_DIR/vulkan-summary.txt" 2>&1

python - <<'PY' > "$OUTPUT_DIR/python-runtime.txt"
from importlib.metadata import version
import torch

print("python_package_ai2thor:", version("ai2thor"))
print("python_package_accelerate:", version("accelerate"))
print("python_package_peft:", version("peft"))
print("python_package_transformers:", version("transformers"))
print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

printf 'ENVIRONMENT_BASELINE_CAPTURED=%s\n' "$OUTPUT_DIR"
