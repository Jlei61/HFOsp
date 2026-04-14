#!/bin/bash
# Epilepsiae full batch HFO detection with GPU
# Expected runtime: ~60-70 hours for 19 subjects (139 auto-skipped by Nyquist)
# Subject 548 already done, will be skipped by --skip-existing

set -e

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate cuda_env

export PYTHONUNBUFFERED=1
cd /home/honglab/leijiaxin/HFOsp

echo "=== Epilepsiae batch HFO detection ==="
echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader)"
echo ""

python scripts/run_hfo_detection.py \
    --dataset epilepsiae \
    --all \
    --gpu \
    --skip-existing \
    --output-dir results/hfo_detection \
    --output-summary results/hfo_detection/epilepsiae_batch_summary.json

echo ""
echo "=== DONE ==="
echo "End: $(date)"
