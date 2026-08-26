#!/usr/bin/env bash
set -euo pipefail

repo=/home/honglab/leijiaxin/HFOsp
python_bin=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
root="$repo/results/epi_prssm/continuous_marked_state/r1/r1_2b"
mkdir -p "$root/logs/persistent_diagnostics"

run_subject() {
  local subject="$1"
  local arm seed output log
  for arm in joint_explicit joint_explicit_raw; do
    for seed in 0 1 2; do
      output="$root/diagnostics/$subject/${arm}_seed_${seed}/result.json"
      log="$root/logs/persistent_diagnostics/${subject}_${arm}_seed_${seed}.log"
      if [[ -s "$output" ]]; then
        continue
      fi
      PYTHONPATH="$repo" OMP_NUM_THREADS=1 "$python_bin" \
        "$repo/scripts/topic5_continuous_marked_state_r1/run_r1_2b_persistent_diagnostics.py" \
        --subject "$subject" --arm "$arm" --seed "$seed" --device cuda \
        >"$log" 2>&1
    done
  done
}

run_subject epilepsiae_620 &
pid_620=$!
run_subject epilepsiae_958 &
pid_958=$!
run_subject yuquan_huanghanwen &
pid_hh=$!

status=COMPLETE
for pid in "$pid_620" "$pid_958" "$pid_hh"; do
  if ! wait "$pid"; then
    status=FAILED
  fi
done

PYTHONPATH="$repo" "$python_bin" - "$root" "$status" <<'PY'
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
status = sys.argv[2]
files = sorted((root / "diagnostics").glob("*/*/result.json"))
payload = {
    "status": status if len(files) == 18 else "INCOMPLETE",
    "completed": len(files),
    "expected": 18,
    "sealed_opened": False,
}
target = root / "manifests" / "PERSISTENT_DIAGNOSTICS_QUEUE.json"
target.parent.mkdir(parents=True, exist_ok=True)
tmp = target.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
os.replace(tmp, target)
if payload["status"] != "COMPLETE":
    raise SystemExit(1)
PY
