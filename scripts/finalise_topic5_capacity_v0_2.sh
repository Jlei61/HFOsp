#!/usr/bin/env bash
# Phase F4 -> J for Topic 5.2D v0.2, in the order the plan fixes.
# Every step is resumable; nothing here retrains a spatial model.
set -u
PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
R=results/topic5_capacity_constrained_history_motif_v0_2
LOG=$R/run_logs
mkdir -p "$LOG"
export TOPIC5_TORCH_THREADS=1

step () {
  name=$1; shift
  echo "=== $name  $(date -Is) ==="
  "$@" > "$LOG/final_$name.log" 2>&1
  echo "    exit=$? tail:"; tail -3 "$LOG/final_$name.log" | sed 's/^/    /'
}

# 1. what the frozen models actually use at test time (+ STOP, + basis transplant)
step usephase   $PY scripts/run_topic5_capacity_usephase_v0_2.py --workers "${W:-12}" --blocks CORE1,CORE2
# 2. representation ceiling (cheap, independent of the ordered models)
step ceiling    $PY scripts/run_topic5_capacity_basis_ceiling_v0_2.py --workers "${W:-12}"
# 3. closed-loop generation, secondary
step rollout    $PY scripts/run_topic5_capacity_rollout_v0_2.py --workers "${W:-10}"
# 4. patient-first aggregation and the cohort evidence matrix
step aggregate  $PY scripts/aggregate_topic5_capacity_v0_2.py
# 5. the only code allowed to read the model-unseen split, after everything is frozen
step confirm    $PY scripts/confirm_topic5_capacity_split_minus_one_v0_2.py --workers "${W:-10}" --confirm
# 6. supplementary figure, its README, source data, metadata and visual QA
step figure     $PY scripts/plot_topic5_capacity_supp_figure_v0_2.py
# 7. engineering / scientific-contract / figure audits
step audit      $PY scripts/audit_topic5_capacity_closeout_v0_2.py
# 8. both reports, generated from the artefacts
step reports    $PY scripts/write_topic5_capacity_reports_v0_2.py --date "${REPORT_DATE:-$(date +%F)}"
echo "=== finalise done $(date -Is) ==="
