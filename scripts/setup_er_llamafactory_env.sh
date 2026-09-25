#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT=/root/pc/gpc_dynaloop_storage/tools/llamafactory-embodied-venv
PYTHON=/root/pc/conda-envs/gqa-process-consistency/bin/python
LLAMAFACTORY=/root/pc/LLaMA-Factory-Embodied

if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  "$PYTHON" -m venv --system-site-packages "$ENV_ROOT"
fi

source "$ENV_ROOT/bin/activate"
python -m pip install --upgrade pip
python -m pip install --requirement "$LLAMAFACTORY/requirements.txt"
python -m pip install --no-deps --editable "$LLAMAFACTORY"

python - <<'PY'
import accelerate
import datasets
import peft
import torch
import transformers
import trl

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("accelerate", accelerate.__version__)
print("peft", peft.__version__)
print("trl", trl.__version__)
print("cuda_available", torch.cuda.is_available())
PY

llamafactory-cli version
echo ER_LLAMAFACTORY_ENV_OK
