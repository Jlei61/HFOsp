#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="/home/honglab/leijiaxin/anaconda3/bin/python"
OUT="$ROOT/results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4e_spatially_shared_terminator"
RUNNER="$ROOT/scripts/run_topic4_fcxr_lc4e.py"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

LOCKED_FILES=(
  "scripts/run_topic4_fcxr_lc4e_autopilot.sh"
  "scripts/run_topic4_fcxr_lc4e.py"
  "scripts/run_topic4_fcxr_lc4d.py"
  "scripts/run_topic4_fcxr_lc4_lifecycle.py"
  "src/topic4_fcxr_lc4e.py"
  "src/topic4_fcxr_lc4d.py"
  "src/snn_engine/mz_slow_vars.py"
  "docs/superpowers/specs/2026-08-10-topic4-fcxr-lc4e-spatially-shared-terminator-design.md"
  "docs/superpowers/plans/2026-08-10-topic4-fcxr-lc4e-spatially-shared-terminator.md"
  "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4e_spatially_shared_terminator/candidate_lock.json"
)

swap_mib() { awk '/SwapTotal:/ {t=$2} /SwapFree:/ {f=$2} END {printf "%.3f", (t-f)/1024.0}' /proc/meminfo; }
mem_gib() { awk '/MemAvailable:/ {printf "%.3f", $2/1024.0/1024.0}' /proc/meminfo; }
digest() { sha256sum "${LOCKED_FILES[@]}" | sha256sum | awk '{print $1}'; }

write_stop() {
  local file="$1" status="$2" reason="$3" stage="$4"
  "$PY" - "$file" "$status" "$reason" "$stage" <<'PY'
import json, os, sys, time
p,status,reason,stage=sys.argv[1:]
tmp=p+f".{os.getpid()}.tmp"
with open(tmp,"w") as f:
    json.dump(dict(status=status,reason=reason,stage=stage,epoch=time.time()),f,indent=2)
    f.flush(); os.fsync(f.fileno())
os.replace(tmp,p)
PY
}

BASE_SWAP="$(swap_mib)"
SOURCE_SHA="$(digest)"
"$PY" - "$OUT/autopilot_lock.json" "$SOURCE_SHA" "$(git rev-parse HEAD)" "$BASE_SWAP" <<'PY'
import json, os, sys, time
p,sha,head,swap=sys.argv[1:]
tmp=p+f".{os.getpid()}.tmp"
with open(tmp,"w") as f:
    json.dump(dict(status="LOCKED",pid=os.getppid(),source_digest=sha,git_head=head,
                   swap_baseline_mib=float(swap),locked_epoch=time.time()),f,indent=2)
    f.flush(); os.fsync(f.fileno())
os.replace(tmp,p)
PY

SOFT_BLOCK=0
run_stage() {
  local label="$1" stage="$2" maxwall="$3" log="$4" pidfile="$5"
  [[ "$(digest)" == "$SOURCE_SHA" ]] || { write_stop "$OUT/AUTOPILOT_FAILED.json" FAILED "source drift" "$label"; return 31; }
  [[ "$SOFT_BLOCK" == 0 ]] || { write_stop "$OUT/AUTOPILOT_STOP.json" STOP "soft resource block" "$label"; return 32; }
  "$PY" - "$(mem_gib)" <<'PY' || { write_stop "$OUT/AUTOPILOT_STOP.json" STOP "MemAvailable below 128 GiB" "$label"; return 33; }
import sys
raise SystemExit(0 if float(sys.argv[1]) >= 128.0 else 1)
PY
  setsid nohup "$PY" "$RUNNER" --confirm-run --stage "$stage" >"$log" 2>&1 &
  local pid=$! start now delta rc=0 stat
  printf '%s\n' "$pid" >"$pidfile"
  start="$(date +%s)"
  while kill -0 "$pid" 2>/dev/null; do
    stat="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    [[ "$stat" == Z* ]] && break
    now="$(swap_mib)"
    delta="$($PY - "$now" "$BASE_SWAP" <<'PY'
import sys
print(float(sys.argv[1])-float(sys.argv[2]))
PY
)"
    if "$PY" - "$delta" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= 512.0 else 1)
PY
    then
      kill -TERM "$pid" 2>/dev/null || true
      write_stop "$OUT/AUTOPILOT_FAILED.json" FAILED "swap +512 MiB; newest task terminated" "$label"
      wait "$pid" 2>/dev/null || true; return 34
    fi
    if [[ "$SOFT_BLOCK" == 0 ]] && "$PY" - "$delta" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= 256.0 else 1)
PY
    then SOFT_BLOCK=1; write_stop "$OUT/RESOURCE_SOFT.json" SOFT "swap +256 MiB; later submission blocked" "$label"; fi
    if (( $(date +%s)-start > maxwall )); then
      kill -TERM "$pid" 2>/dev/null || true
      write_stop "$OUT/AUTOPILOT_FAILED.json" FAILED "wall guard exceeded" "$label"
      wait "$pid" 2>/dev/null || true; return 35
    fi
    sleep 20
  done
  wait "$pid" || rc=$?
  return "$rc"
}

run_stage E1 screen 14400 "$OUT/nohup_e1.log" "$OUT/e1.pid" || exit $?
[[ -f "$OUT/E1_DONE.json" && -f "$OUT/architecture_verdict.json" ]] || {
  write_stop "$OUT/AUTOPILOT_FAILED.json" FAILED "E1 missing terminal verdict" E1; exit 1; }
e1="$($PY - "$OUT/architecture_verdict.json" <<'PY'
import json,sys
print("yes" if json.load(open(sys.argv[1])).get("passed") else "no")
PY
)"
if [[ "$e1" != yes ]]; then write_stop "$OUT/AUTOPILOT_STOP.json" STOP "shared executor failed E1" E1; exit 0; fi

run_stage E2_NOMINAL nominal 25200 "$OUT/nohup_e2_nominal.log" "$OUT/e2_nominal.pid" || exit $?
eligible="$($PY - "$OUT/nominal_lifecycle.json" <<'PY'
import json,sys
print("yes" if json.load(open(sys.argv[1])).get("nominal_gate",{}).get("eligible_for_frozen_D") else "no")
PY
)"
if [[ "$eligible" != yes ]]; then write_stop "$OUT/AUTOPILOT_STOP.json" STOP "nominal lifecycle incomplete" E2_NOMINAL; exit 0; fi

run_stage E2_CONFIRM confirm 10800 "$OUT/nohup_e2_confirm.log" "$OUT/e2_confirm.pid" || exit $?
[[ -f "$OUT/F2_CONFIRM_DONE.json" && -f "$OUT/lifecycle_verdict.json" ]] || {
  write_stop "$OUT/AUTOPILOT_FAILED.json" FAILED "confirmation missing terminal verdict" E2_CONFIRM; exit 1; }
write_stop "$OUT/AUTOPILOT_DONE.json" DONE "conditional chain finished" E2_CONFIRM
