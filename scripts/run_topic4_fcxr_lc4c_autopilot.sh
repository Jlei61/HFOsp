#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="/home/honglab/leijiaxin/anaconda3/bin/python"
OUT="$ROOT/results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4c_entry_offset_alignment"
RUNNER="$ROOT/scripts/run_topic4_fcxr_lc4c.py"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

LOCKED_FILES=(
  "scripts/run_topic4_fcxr_lc4c_autopilot.sh"
  "scripts/run_topic4_fcxr_lc4c.py"
  "scripts/lock_topic4_fcxr_lc4c.py"
  "scripts/run_topic4_fcxr_lc4_lifecycle.py"
  "src/topic4_fcxr_lc4c.py"
  "src/topic4_fcxr_lc4b_deadzone.py"
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
  "docs/superpowers/specs/2026-08-10-topic4-fcxr-lc4c-entry-offset-alignment-design.md"
  "docs/superpowers/plans/2026-08-10-topic4-fcxr-lc4c-entry-offset-alignment.md"
  "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4c_entry_offset_alignment/candidate_lock.json"
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
  local path="$1" status="$2" reason="$3" stage="$4"
  "$PY" - "$path" "$status" "$reason" "$stage" <<'PY'
import json, os, sys, time
p, status, reason, stage = sys.argv[1:]
tmp = p + f".{os.getpid()}.tmp"
with open(tmp, "w") as f:
    json.dump(dict(status=status, reason=reason, stage=stage, epoch=time.time()), f, indent=2)
    f.flush(); os.fsync(f.fileno())
os.replace(tmp, p)
PY
}

BASE_SWAP="$(swap_mib)"
SOURCE_SHA="$(source_digest)"
GIT_HEAD="$(git rev-parse HEAD)"
"$PY" - "$OUT/autopilot_lock.json" "$SOURCE_SHA" "$GIT_HEAD" "$BASE_SWAP" <<'PY'
import json, os, sys, time
p, sha, head, swap = sys.argv[1:]
tmp = p + f".{os.getpid()}.tmp"
with open(tmp, "w") as f:
    json.dump(dict(status="LOCKED", pid=os.getppid(), source_digest=sha, git_head=head,
                   swap_baseline_mib=float(swap), locked_epoch=time.time()), f, indent=2)
    f.flush(); os.fsync(f.fileno())
os.replace(tmp, p)
PY

SOFT_BLOCK=0

run_stage() {
  local stage="$1" runner_stage="$2" log="$3" pidfile="$4" max_wall_s="$5"
  local mem now delta started stat rc=0
  if [[ "$(source_digest)" != "$SOURCE_SHA" ]]; then
    write_json "$OUT/AUTOPILOT_FAILED.json" FAILED "locked source drift before submission" "$stage"
    return 31
  fi
  if [[ "$SOFT_BLOCK" == 1 ]]; then
    write_json "$OUT/AUTOPILOT_STOP.json" STOP "resource soft gate blocks new submission" "$stage"
    return 32
  fi
  mem="$(mem_available_gib)"
  if ! "$PY" - "$mem" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= 128.0 else 1)
PY
  then
    write_json "$OUT/AUTOPILOT_STOP.json" STOP "MemAvailable below 128 GiB" "$stage"
    return 33
  fi
  setsid nohup "$PY" "$RUNNER" --confirm-run --stage "$runner_stage" > "$log" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" > "$pidfile"
  started="$(date +%s)"
  while kill -0 "$pid" 2>/dev/null; do
    stat="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    [[ "$stat" == Z* ]] && break
    now="$(swap_mib)"
    delta="$($PY - "$now" "$BASE_SWAP" <<'PY'
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
      write_json "$OUT/AUTOPILOT_FAILED.json" FAILED "swap delta reached +512 MiB; newest task terminated" "$stage"
      wait "$pid" 2>/dev/null || true
      return 34
    fi
    if [[ "$SOFT_BLOCK" == 0 ]] && "$PY" - "$delta" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= 256.0 else 1)
PY
    then
      SOFT_BLOCK=1
      write_json "$OUT/RESOURCE_SOFT.json" SOFT "swap delta reached +256 MiB; no later submission" "$stage"
    fi
    if (( $(date +%s) - started > max_wall_s )); then
      kill -TERM "$pid" 2>/dev/null || true
      write_json "$OUT/AUTOPILOT_FAILED.json" FAILED "registered wall-time guard exceeded" "$stage"
      wait "$pid" 2>/dev/null || true
      return 35
    fi
    sleep 20
  done
  wait "$pid" || rc=$?
  return "$rc"
}

if ! run_stage C1 entry "$OUT/nohup_c1.log" "$OUT/c1.pid" 10800; then
  [[ -f "$OUT/AUTOPILOT_FAILED.json" || -f "$OUT/AUTOPILOT_STOP.json" ]] || \
    write_json "$OUT/AUTOPILOT_FAILED.json" FAILED "C1 runner exited nonzero" C1
  exit 1
fi
if [[ ! -f "$OUT/C1_DONE.json" || ! -f "$OUT/entry_gate.json" ]]; then
  write_json "$OUT/AUTOPILOT_FAILED.json" FAILED "C1 ended without complete verdict" C1
  exit 1
fi
c1_pass="$($PY - "$OUT/entry_gate.json" <<'PY'
import json, sys
print("yes" if json.load(open(sys.argv[1])).get("gate", {}).get("passed") else "no")
PY
)"
if [[ "$c1_pass" != yes ]]; then
  write_json "$OUT/AUTOPILOT_STOP.json" STOP "aligned candidate failed C1 entry gate" C1
  exit 0
fi

if ! run_stage C2_NOMINAL nominal "$OUT/nohup_c2_nominal.log" "$OUT/c2_nominal.pid" 25200; then
  [[ -f "$OUT/AUTOPILOT_FAILED.json" || -f "$OUT/AUTOPILOT_STOP.json" ]] || \
    write_json "$OUT/AUTOPILOT_FAILED.json" FAILED "C2 nominal runner exited nonzero" C2_NOMINAL
  exit 1
fi
if [[ ! -f "$OUT/F2_NOMINAL_DONE.json" || ! -f "$OUT/nominal_lifecycle.json" ]]; then
  write_json "$OUT/AUTOPILOT_FAILED.json" FAILED "C2 nominal ended without complete verdict" C2_NOMINAL
  exit 1
fi
eligible="$($PY - "$OUT/nominal_lifecycle.json" <<'PY'
import json, sys
print("yes" if json.load(open(sys.argv[1])).get("nominal_gate", {}).get("eligible_for_frozen_D") else "no")
PY
)"
if [[ "$eligible" != yes ]]; then
  write_json "$OUT/AUTOPILOT_STOP.json" STOP "C2 nominal incomplete; exact-D continuation locked" C2_NOMINAL
  exit 0
fi

if ! run_stage C2_CONFIRM confirm "$OUT/nohup_c2_confirm.log" "$OUT/c2_confirm.pid" 10800; then
  [[ -f "$OUT/AUTOPILOT_FAILED.json" || -f "$OUT/AUTOPILOT_STOP.json" ]] || \
    write_json "$OUT/AUTOPILOT_FAILED.json" FAILED "C2 confirmation runner exited nonzero" C2_CONFIRM
  exit 1
fi
if [[ ! -f "$OUT/F2_CONFIRM_DONE.json" || ! -f "$OUT/lifecycle_verdict.json" ]]; then
  write_json "$OUT/AUTOPILOT_FAILED.json" FAILED "C2 confirmation ended without final verdict" C2_CONFIRM
  exit 1
fi
write_json "$OUT/AUTOPILOT_DONE.json" DONE "C1-C2 conditional chain finished" C2_CONFIRM
