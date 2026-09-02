#!/bin/bash
# Everything downstream of the model grid, in the order the results depend on.
# Safe to re-run: every stage is idempotent by result hash.
set -euo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
WT=/tmp/hfosp_group_event_state_v02_c
TAG=${1:-main}
cd "$WT"

SUBJ=$($PY - <<'P'
import json
d = json.load(open('results/epi_prssm/group_event_state/v0_2/h3/support/coverage_support_primary.json'))
print(' '.join(r['subject'] for r in sorted(d['subjects'], key=lambda r: -r['usable_hours_after_seizure_cuts'])))
P
)

for STAGE in impulse perturbation innovation; do
  EXTRA=()
  [ "$STAGE" = perturbation ] && EXTRA=(--include-secondary)
  $PY scripts/topic5_group_event_state_h3/queue_runner.py --stage "$STAGE" \
      --subjects $SUBJ --seeds 0 1 2 --tag "$TAG" --gpus 0 1 --slots-per-gpu 3 "${EXTRA[@]}"
done

$PY scripts/topic5_group_event_state_h3/aggregate_h3.py --tag "$TAG"
$PY scripts/topic5_group_event_state_h3/aggregate_downstream.py --tag "$TAG"
$PY scripts/topic5_group_event_state_h3/publish_registry.py --tag "$TAG"
$PY scripts/topic5_group_event_state_h3/make_figures.py --tag "$TAG"
$PY scripts/topic5_group_event_state_h3/write_status.py --tag "$TAG"
