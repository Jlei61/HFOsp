#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="/home/honglab/leijiaxin/anaconda3/bin/python"
OUT="$ROOT/results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4_lifecycle_gate"
RUNNER="$ROOT/scripts/run_topic4_fcxr_lc4_gate.py"
PURE="$ROOT/src/topic4_fcxr_lc4_gate.py"
mkdir -p "$OUT"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

swap_mib() {
  awk '/SwapTotal:/ {t=$2} /SwapFree:/ {f=$2} END {printf "%.3f", (t-f)/1024.0}' /proc/meminfo
}

mem_available_gib() {
  awk '/MemAvailable:/ {printf "%.3f", $2/1024.0/1024.0}' /proc/meminfo
}

BASE_SWAP="$(swap_mib)"
RUNNER_SHA="$(sha256sum "$RUNNER" | awk '{print $1}')"
PURE_SHA="$(sha256sum "$PURE" | awk '{print $1}')"

"$PY" - "$OUT/autopilot_lock.json" "$BASE_SWAP" "$RUNNER_SHA" "$PURE_SHA" <<'PY'
import json, os, sys, time
p, swap, runner, pure = sys.argv[1:]
tmp = p + f".{os.getpid()}.tmp"
with open(tmp, "w") as f:
    json.dump(dict(status="LOCKED", pid=os.getppid(), swap_baseline_mib=float(swap),
                   runner_sha256=runner, pure_contract_sha256=pure,
                   locked_epoch=time.time()), f, indent=2)
    f.flush(); os.fsync(f.fileno())
os.replace(tmp, p)
PY

baseline_pid="$($PY - "$OUT/BASELINE_RUNNING.json" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1]))["pid"]))
PY
)"

while kill -0 "$baseline_pid" 2>/dev/null; do
  now_swap="$(swap_mib)"
  delta="$($PY - "$now_swap" "$BASE_SWAP" <<'PY'
import sys
print(float(sys.argv[1]) - float(sys.argv[2]))
PY
)"
  hard="$($PY - "$delta" <<'PY'
import sys
print(int(float(sys.argv[1]) >= 512.0))
PY
)"
  if [[ "$hard" == "1" ]]; then
    kill -TERM "$baseline_pid" 2>/dev/null || true
    "$PY" - "$OUT/AUTOPILOT_FAILED.json" "$delta" <<'PY'
import json, os, sys, time
p, delta = sys.argv[1:]
with open(p, "w") as f:
    json.dump(dict(status="FAILED", reason="swap delta reached +512 MiB during baseline",
                   swap_delta_mib=float(delta), epoch=time.time()), f, indent=2)
raise SystemExit(0)
PY
    exit 1
  fi
  sleep 20
done

if [[ ! -f "$OUT/BASELINE_DONE.json" || ! -f "$OUT/baseline_verdict.json" ]]; then
  "$PY" - "$OUT/AUTOPILOT_FAILED.json" <<'PY'
import json, sys, time
with open(sys.argv[1], "w") as f:
    json.dump(dict(status="FAILED", reason="baseline process ended without complete verdict",
                   epoch=time.time()), f, indent=2)
PY
  exit 1
fi

selected="$($PY - "$OUT/baseline_verdict.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
print("yes" if d.get("selected_candidate") is not None else "no")
PY
)"
if [[ "$selected" != "yes" ]]; then
  "$PY" - "$OUT/AUTOPILOT_STOP.json" <<'PY'
import json, sys, time
with open(sys.argv[1], "w") as f:
    json.dump(dict(status="STOP", stage="F0", reason="no baseline-preserving Hill candidate",
                   epoch=time.time()), f, indent=2)
PY
  exit 0
fi

if [[ "$(sha256sum "$RUNNER" | awk '{print $1}')" != "$RUNNER_SHA" \
   || "$(sha256sum "$PURE" | awk '{print $1}')" != "$PURE_SHA" ]]; then
  "$PY" - "$OUT/AUTOPILOT_FAILED.json" <<'PY'
import json, sys, time
with open(sys.argv[1], "w") as f:
    json.dump(dict(status="FAILED", reason="F1 source hash drift after F0",
                   epoch=time.time()), f, indent=2)
PY
  exit 1
fi

now_swap="$(swap_mib)"
now_mem="$(mem_available_gib)"
may_submit="$($PY - "$now_swap" "$BASE_SWAP" "$now_mem" <<'PY'
import sys
swap_delta=float(sys.argv[1])-float(sys.argv[2]); mem=float(sys.argv[3])
print(int(swap_delta < 256.0 and mem >= 128.0))
PY
)"
if [[ "$may_submit" != "1" ]]; then
  "$PY" - "$OUT/AUTOPILOT_STOP.json" "$now_swap" "$BASE_SWAP" "$now_mem" <<'PY'
import json, sys, time
p, now, base, mem = sys.argv[1:]
with open(p, "w") as f:
    json.dump(dict(status="STOP", stage="F0", reason="resource gate blocks F1 submission",
                   swap_delta_mib=float(now)-float(base), mem_available_gib=float(mem),
                   epoch=time.time()), f, indent=2)
PY
  exit 0
fi

"$PY" "$RUNNER" --confirm-run --stage onset > "$OUT/nohup_onset.log" 2>&1 &
onset_pid=$!
printf '%s\n' "$onset_pid" > "$OUT/onset.pid"
wait "$onset_pid"

if [[ ! -f "$OUT/ONSET_DONE.json" || ! -f "$OUT/onset_surface_verdict.json" ]]; then
  "$PY" - "$OUT/AUTOPILOT_FAILED.json" <<'PY'
import json, sys, time
with open(sys.argv[1], "w") as f:
    json.dump(dict(status="FAILED", reason="onset process ended without complete verdict",
                   epoch=time.time()), f, indent=2)
PY
  exit 1
fi

"$PY" - "$OUT/onset_surface_verdict.json" "$OUT/AUTOPILOT_DONE.json" <<'PY'
import json, sys, time
d=json.load(open(sys.argv[1])); g=d["gate"]
with open(sys.argv[2], "w") as f:
    json.dump(dict(status="DONE", stage="F1", verdict=g["verdict"],
                   lifecycle_authorized=bool(g["passed"]), epoch=time.time()), f, indent=2)
PY
