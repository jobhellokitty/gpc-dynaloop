#!/usr/bin/env bash
set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate /root/pc/conda-envs/gqa-process-consistency

export PYTHONPATH=/root/pc/gpc_dynaloop/src
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export XDG_RUNTIME_DIR=/tmp/xdg-runtime-root
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

python -m unittest discover -s tests/unit -p 'test_*.py'
vulkaninfo --summary 2>&1 | grep -q 'deviceName.*NVIDIA A800-SXM4-80GB'
python tests/integration/run_stage1_protocol_suite.py
