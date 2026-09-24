#!/usr/bin/env bash
set -uo pipefail

cd /root/pc/gpc_dynaloop
protocol_status=0
bash scripts/run_stage1_protocol_suite.sh || protocol_status=$?

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate /root/pc/conda-envs/gqa-process-consistency
export PYTHONPATH=/root/pc/gpc_dynaloop/src
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export XDG_RUNTIME_DIR=/tmp/xdg-runtime-root

model_status=0
python tests/integration/run_gpc_sft_he_multi_image.py || model_status=$?

if [[ "$protocol_status" -eq 0 && "$model_status" -eq 0 ]]; then
  python scripts/verify_stage1_gate.py
else
  printf 'STAGE1_COMPONENT_FAILURE protocol=%s model=%s\n' "$protocol_status" "$model_status"
  exit 1
fi
