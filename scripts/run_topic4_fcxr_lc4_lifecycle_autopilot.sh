#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="/home/honglab/leijiaxin/anaconda3/bin/python"
OUT="$ROOT/results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4_lifecycle_gate"
RUNNER="$ROOT/scripts/run_topic4_fcxr_lc4_lifecycle.py"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

LOCKED_FILES=(
  "scripts/run_topic4_fcxr_lc4_lifecycle.py"
  "src/topic4_fcxr_lc4_lifecycle.py"
  "src/snn_engine/mz_slow_vars.py"
  "src/topic4_fcxr_lc3.py"
  "src/topic4_fcxr_lc3_geometry.py"
  "src/topic4_fcxr_lc3_ledger.py"
  "src/topic4_fcxr_lc3_stage.py"
  "src/topic4_fcxr_lc3_statefork.py"
  "src/topic4_mz_fcxr_lifecycle.py"
  "scripts/run_m4_phaseplane.py"
  "scripts/run_topic4_fcxr_lc3.py"
  "scripts/run_topic4_fcxr_lc3_geometry.py"
  "scripts/run_topic4_fcxr_lc3_phase_map.py"
  "scripts/run_topic4_mz_fcxr_lifecycle.py"
  "scripts/run_topic4_mz_slowvars.py"
  "docs/superpowers/specs/2026-08-09-topic4-fcxr-lc4-functional-selectivity-design.md"
  "docs/superpowers/plans/2026-08-09-topic4-fcxr-lc4-functional-selectivity.md"
)

swap_mib() {
  awk '/SwapTotal:/ {t=$2} /SwapFree:/ {f=$2} END {printf "%.3f", (t-f)/1024.0}' /proc/meminfo
}

mem_available_gib() {
  awk '/MemAvailable:/ {printf "%.3f", $2/1024.0/1024.0}' /proc/meminfo
}

source_digest() {
  sha256sum "${LOCKED_FILES[@]}" | sha256sum | awk '{print $1}'
}

write_json() {
  local path="$1"; shift
  "$PY" - "$path" "$@" <<'PY'
import json, os, sys, time
p, *args = sys.argv[1:]
payload = dict(status=args[0], reason=args[1] if len(args) > 1 else None,
               epoch=time.time())
if len(args) > 2:
    payload["stage"] = args[2]
tmp = p + f".{os.getpid()}.tmp"
with open(tmp, "w") as f:
    json.dump(payload, f, indent=2)
    f.flush(); os.fsync(f.fileno())
os.replace(tmp, p)
PY
}

SOURCE_SHA="$(source_digest)"
GIT_HEAD="$(git rev-parse HEAD)"
"$PY" - "$OUT/f2_autopilot_lock.json" "$SOURCE_SHA" "$GIT_HEAD" <<'PY'
import json, os, sys, time
p, sha, head = sys.argv[1:]
tmp = p + f".{os.getpid()}.tmp"
with open(tmp, "w") as f:
    json.dump(dict(status="LOCKED", pid=os.getppid(), source_digest=sha,
                   git_head=head, locked_epoch=time.time()), f, indent=2)
    f.flush(); os.fsync(f.fileno())
os.replace(tmp, p)
PY

# F0/F1 are owned by the preceding gate autopilot.  Wait by its recorded PID, never pgrep -f.
if [[ ! -f "$OUT/autopilot.pid" ]]; then
  write_json "$OUT/F2_AUTOPILOT_FAILED.json" FAILED "missing F0/F1 autopilot PID" WAIT_F1
  exit 1
fi
gate_pid="$(tr -dc '0-9' < "$OUT/autopilot.pid")"
while [[ -n "$gate_pid" ]] && kill -0 "$gate_pid" 2>/dev/null; do
  state="$(ps -o stat= -p "$gate_pid" 2>/dev/null | tr -d ' ' || true)"
  [[ "$state" == Z* ]] && break
  sleep 20
done

if [[ ! -f "$OUT/AUTOPILOT_DONE.json" ]]; then
  reason="F0/F1 stopped or failed; F2 is not authorised"
  [[ -f "$OUT/AUTOPILOT_STOP.json" ]] && reason="F0/F1 scientific/resource stop; F2 is not authorised"
  write_json "$OUT/F2_AUTOPILOT_STOP.json" STOP "$reason" WAIT_F1
  exit 0
fi
authorised="$($PY - "$OUT/AUTOPILOT_DONE.json" <<'PY'
import json, sys
print("yes" if json.load(open(sys.argv[1])).get("lifecycle_authorized") else "no")
PY
)"
if [[ "$authorised" != "yes" ]]; then
  write_json "$OUT/F2_AUTOPILOT_STOP.json" STOP "F1 onset surface did not authorise lifecycle" F1
  exit 0
fi

if [[ "$(source_digest)" != "$SOURCE_SHA" ]]; then
  write_json "$OUT/F2_AUTOPILOT_FAILED.json" FAILED "F2 locked source drift before nominal" F2_NOMINAL
  exit 1
fi

run_stage() {
  local stage="$1" log="$2" pidfile="$3"
  local base_swap now_swap delta mem soft_written=0
  base_swap="$(swap_mib)"
  mem="$(mem_available_gib)"
  if ! "$PY" - "$mem" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= 128.0 else 1)
PY
  then
    write_json "$OUT/F2_AUTOPILOT_STOP.json" STOP "MemAvailable below 128 GiB" "$stage"
    return 20
  fi

  setsid nohup "$PY" "$RUNNER" --confirm-run --stage "$stage" > "$log" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" > "$pidfile"
  while kill -0 "$pid" 2>/dev/null; do
    local stat
    stat="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    [[ "$stat" == Z* ]] && break
    now_swap="$(swap_mib)"
    delta="$($PY - "$now_swap" "$base_swap" <<'PY'
import sys
print(float(sys.argv[1]) - float(sys.argv[2]))
PY
)"
    if "$PY" - "$delta" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= 512.0 else 1)
PY
    then
      kill -TERM "$pid" 2>/dev/null || true
      write_json "$OUT/F2_AUTOPILOT_FAILED.json" FAILED "swap delta reached +512 MiB; newest task terminated" "$stage"
      wait "$pid" 2>/dev/null || true
      return 21
    fi
    if [[ "$soft_written" == 0 ]] && "$PY" - "$delta" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= 256.0 else 1)
PY
    then
      write_json "$OUT/F2_RESOURCE_SOFT.json" SOFT "swap delta reached +256 MiB; no new task will be submitted" "$stage"
      soft_written=1
    fi
    sleep 20
  done
  local rc=0
  wait "$pid" || rc=$?
  return "$rc"
}

if ! run_stage nominal "$OUT/nohup_f2_nominal.log" "$OUT/f2_nominal.pid"; then
  [[ -f "$OUT/F2_AUTOPILOT_FAILED.json" ]] || \
    write_json "$OUT/F2_AUTOPILOT_FAILED.json" FAILED "nominal runner exited nonzero" F2_NOMINAL
  exit 1
fi
if [[ ! -f "$OUT/F2_NOMINAL_DONE.json" || ! -f "$OUT/nominal_lifecycle.json" ]]; then
  write_json "$OUT/F2_AUTOPILOT_FAILED.json" FAILED "nominal ended without complete artifacts" F2_NOMINAL
  exit 1
fi
eligible="$($PY - "$OUT/nominal_lifecycle.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
print("yes" if d.get("nominal_gate", {}).get("eligible_for_frozen_D") else "no")
PY
)"
if [[ "$eligible" != "yes" ]]; then
  write_json "$OUT/F2_AUTOPILOT_STOP.json" STOP "nominal lifecycle incomplete; frozen-D confirmation locked" F2_NOMINAL
  exit 0
fi

if [[ "$(source_digest)" != "$SOURCE_SHA" ]]; then
  write_json "$OUT/F2_AUTOPILOT_FAILED.json" FAILED "F2 locked source drift before frozen-D confirmation" F2_CONFIRM
  exit 1
fi
if [[ -f "$OUT/F2_RESOURCE_SOFT.json" ]]; then
  write_json "$OUT/F2_AUTOPILOT_STOP.json" STOP "resource soft gate blocks frozen-D submission" F2_CONFIRM
  exit 0
fi
if ! run_stage confirm "$OUT/nohup_f2_confirm.log" "$OUT/f2_confirm.pid"; then
  [[ -f "$OUT/F2_AUTOPILOT_FAILED.json" ]] || \
    write_json "$OUT/F2_AUTOPILOT_FAILED.json" FAILED "frozen-D runner exited nonzero" F2_CONFIRM
  exit 1
fi
if [[ ! -f "$OUT/F2_CONFIRM_DONE.json" || ! -f "$OUT/lifecycle_verdict.json" ]]; then
  write_json "$OUT/F2_AUTOPILOT_FAILED.json" FAILED "confirmation ended without lifecycle verdict" F2_CONFIRM
  exit 1
fi
write_json "$OUT/F2_AUTOPILOT_DONE.json" DONE "F2 nominal and gated frozen-D stages finished" F2_CONFIRM
