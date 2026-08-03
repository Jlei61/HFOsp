#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <exact-e1-launcher-pid>" >&2
  exit 2
fi

e1_pid="$1"
result_root="results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability"
python_bin="/home/honglab/leijiaxin/anaconda3/bin/python"
runner="scripts/run_topic4_fcxr_lc3_geometry.py"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$result_root"
printf '%s\n' "$$" > "$result_root/geometry_autopilot.pid"

on_error() {
  code="$?"
  "$python_bin" - "$result_root" "$code" <<'PY'
import datetime, json, os, sys
root, code = sys.argv[1], int(sys.argv[2])
path = os.path.join(root, "GEOMETRY_AUTOPILOT_FAILED.json")
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump({"status": "FAILED", "exit_code": code,
               "failed": datetime.datetime.now(datetime.timezone.utc).isoformat()}, f, indent=2)
    f.flush(); os.fsync(f.fileno())
os.replace(tmp, path)
PY
  exit "$code"
}
trap on_error ERR

while kill -0 "$e1_pid" 2>/dev/null; do
  sleep 30
done

"$python_bin" - "$result_root" <<'PY'
import json, os, sys
path = os.path.join(sys.argv[1], "d_field_lock.json")
if not os.path.isfile(path) or json.load(open(path)).get("status") != "PASS":
    raise SystemExit("E1 ended without a PASS d_field_lock.json")
PY

"$python_bin" "$runner" field-audit
"$python_bin" "$runner" lock
"$python_bin" "$runner" prepare --point H1 --state low --confirm-run
"$python_bin" "$runner" prepare --point H1 --state high --confirm-run
"$python_bin" "$runner" prepare --point H6 --state low --confirm-run
"$python_bin" "$runner" prepare --point H6 --state high --confirm-run
"$python_bin" "$runner" manifest
"$python_bin" "$runner" map --confirm-run

"$python_bin" - "$result_root" <<'PY'
import datetime, json, os, sys
root = sys.argv[1]
path = os.path.join(root, "GEOMETRY_AUTOPILOT_DONE.json")
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump({"status": "DONE",
               "finished": datetime.datetime.now(datetime.timezone.utc).isoformat()}, f, indent=2)
    f.flush(); os.fsync(f.fileno())
os.replace(tmp, path)
PY
