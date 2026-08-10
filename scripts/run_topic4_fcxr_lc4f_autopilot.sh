#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4f_x_depth_closure"
PY="/home/honglab/leijiaxin/anaconda3/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$OUT"

run_stage() {
  local stage="$1" hours="$2"
  setsid "$PY" "$ROOT/scripts/run_topic4_fcxr_lc4f.py" --confirm-run --stage "$stage" &
  local pid=$!
  echo "$pid" > "$OUT/${stage}.pid"
  local deadline=$(( $(date +%s) + hours * 3600 ))
  while kill -0 "$pid" 2>/dev/null; do
    if (( $(date +%s) > deadline )); then
      kill -TERM "$pid" 2>/dev/null || true
      wait "$pid" || true
      printf '{"status":"FAILED","reason":"wall guard","stage":"%s"}\n' "$stage" > "$OUT/AUTOPILOT_FAILED.json"
      exit 1
    fi
    sleep 30
  done
  wait "$pid"
}

run_stage screen 4
if ! grep -q 'X_DEPTH_OFFSET_CANDIDATE' "$OUT/X1_DONE.json"; then
  printf '{"status":"STOP","reason":"X1 did not produce an offset candidate","stage":"X1"}\n' > "$OUT/AUTOPILOT_STOP.json"
  exit 0
fi
run_stage nominal 8
if ! grep -q '"eligible_for_frozen_D": true' "$OUT/nominal_lifecycle.json"; then
  printf '{"status":"STOP","reason":"nominal lifecycle incomplete","stage":"X2_NOMINAL"}\n' > "$OUT/AUTOPILOT_STOP.json"
  exit 0
fi
run_stage confirm 3
printf '{"status":"DONE","stage":"X2_CONFIRM"}\n' > "$OUT/AUTOPILOT_DONE.json"
