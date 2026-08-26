#!/usr/bin/env bash
set -euo pipefail

repo=/home/honglab/leijiaxin/HFOsp
python_bin=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
root="$repo/results/epi_prssm/continuous_marked_state/r1/r1_3"
mkdir -p "$root/logs/fits" "$root/manifests"

write_status() {
  local stage="$1"
  local status="$2"
  PYTHONPATH="$repo" "$python_bin" - "$root" "$stage" "$status" <<'PY'
import json, os, sys
from pathlib import Path
root, stage, status = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
fits = sorted((root / "fits").glob("*/*/result.json"))
payload = {
    "status": status,
    "stage": stage,
    "completed_fits": len(fits),
    "expected_fits": 18,
    "sealed_opened": False,
}
target = root / "RUN_STATUS.json"
tmp = target.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
os.replace(tmp, target)
PY
}

subjects=(epilepsiae_620 epilepsiae_958 yuquan_huanghanwen)
while true; do
  ready=0
  for subject in "${subjects[@]}"; do
    [[ -s "$root/cache/$subject/manifest.json" ]] && ready=$((ready + 1))
  done
  [[ "$ready" -eq 3 ]] && break
  write_status waiting_for_observation_caches RUNNING
  sleep 30
done

run_fit() {
  local subject="$1"
  local arm="$2"
  local seed="$3"
  local output="$root/fits/$subject/${arm}_seed_${seed}/result.json"
  local log="$root/logs/fits/${subject}_${arm}_seed_${seed}.log"
  if [[ -s "$output" ]]; then
    return 0
  fi
  PYTHONPATH="$repo" OMP_NUM_THREADS=1 "$python_bin" \
    "$repo/scripts/topic5_continuous_marked_state_r1/run_r1_3_target_observer.py" \
    --subject "$subject" --arm "$arm" --seed "$seed" --device cuda \
    --observer-epochs 2 --joint-epochs 2 --chunk-anchors 8 \
    --output-root "$root" --observation-cache-root "$root/cache" \
    >"$log" 2>&1
}

run_stage() {
  local arm="$1"
  local workers="$2"
  local subject seed
  for subject in "${subjects[@]}"; do
    for seed in 0 1 2; do
      while [[ "$(jobs -rp | wc -l)" -ge "$workers" ]]; do
        wait -n
      done
      run_fit "$subject" "$arm" "$seed" &
    done
  done
  wait
}

write_status explicit RUNNING
run_stage explicit 3
write_status explicit_raw RUNNING
run_stage explicit_raw 2
write_status aggregation RUNNING
PYTHONPATH="$repo" "$python_bin" \
  "$repo/scripts/topic5_continuous_marked_state_r1/aggregate_r1_3.py" \
  >"$root/logs/aggregate.log" 2>&1
write_status complete COMPLETE
