#!/usr/bin/env bash
set -euo pipefail

cd /root/pc/gpc_dynaloop
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate /root/pc/conda-envs/gqa-process-consistency
export PYTHONPATH=/root/pc/gpc_dynaloop/src

python -m unittest discover -s tests/unit -p 'test_*.py'
python scripts/audit_er_base_subset.py
