#!/usr/bin/env bash
set -euo pipefail

repo=/home/honglab/leijiaxin/HFOsp
python_bin=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
root="$repo/results/epi_prssm/continuous_marked_state/r1"
t2="$root/t2_s1_long_scale"
mkdir -p "$t2/logs" "$t2/human" "$root/r1_3/fits"

while [[ ! -s "$t2/reports/t2_s1_summary.json" ]]; do
  r13_count=$(find "$root/r1_3/fits" -name result.json 2>/dev/null | wc -l)
  t2_count=$(find "$t2/human" -name result.json 2>/dev/null | wc -l)
  PYTHONPATH="$repo" "$python_bin" - "$t2" "$r13_count" "$t2_count" <<'PY'
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
payload = {
    "status": "RUNNING",
    "stage": "waiting_for_r1_3" if int(sys.argv[2]) < 18 else "human_t2_s1",
    "r1_3_completed_fits": int(sys.argv[2]),
    "r1_3_expected_fits": 18,
    "t2_completed_fits": int(sys.argv[3]),
    "t2_expected_fits": 12,
    "sealed_opened": False,
}
target = root / "RUN_STATUS.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
os.replace(temporary, target)
PY
  sleep 60
done

PYTHONPATH="$repo" "$python_bin" \
  "$repo/scripts/topic5_continuous_marked_state_r1/finalize_r1_3_t2_reports.py" \
  > "$t2/logs/finalize.log" 2>&1
PYTHONPATH="$repo" "$python_bin" - "$t2" <<'PY'
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
payload = {
    "status": "COMPLETE", "stage": "complete",
    "r1_3_completed_fits": 18, "r1_3_expected_fits": 18,
    "t2_completed_fits": 12, "t2_expected_fits": 12,
    "sealed_opened": False,
}
target = root / "RUN_STATUS.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
os.replace(temporary, target)
PY
