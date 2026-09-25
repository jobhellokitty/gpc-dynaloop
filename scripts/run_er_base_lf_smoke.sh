#!/usr/bin/env bash
set -euo pipefail

cd /root/pc/gpc_dynaloop
source /root/pc/gpc_dynaloop_storage/tools/llamafactory-embodied-venv/bin/activate
export PYTHONPATH=/root/pc/gpc_dynaloop/src

python scripts/export_er_llamafactory_dataset.py
llamafactory-cli train configs/er_base_lora_smoke.yaml

test -f /root/pc/gpc_dynaloop_storage/checkpoints/gpc_embodied_base_smoke/adapter_config.json
echo ER_LLAMAFACTORY_SMOKE_OK
