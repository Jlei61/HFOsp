#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <exact-e1-pid>" >&2
  exit 2
fi

target_pid="$1"
result_root="results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability"
swap_base_kib="$(awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print t-f}' /proc/meminfo)"
last_swap_kib="$swap_base_kib"
start_epoch="$(date +%s)"
printf '%s\n' "$$" > "$result_root/e1_watchdog.pid"

while kill -0 "$target_pid" 2>/dev/null; do
  sleep 30
  current_swap_kib="$(awk '/^SwapTotal:/ {t=$2} /^SwapFree:/ {f=$2} END {print t-f}' /proc/meminfo)"
  delta_kib="$((current_swap_kib - swap_base_kib))"
  if (( delta_kib >= 524288 && current_swap_kib > last_swap_kib )); then
    kill -TERM "$target_pid"
    exit 70
  fi
  # Original archived runtimes imply ~1.5 h remaining at watchdog launch;
  # four hours is a conservative runaway guard, not a scientific time limit.
  if (( $(date +%s) - start_epoch > 14400 )); then
    kill -TERM "$target_pid"
    exit 71
  fi
  last_swap_kib="$current_swap_kib"
done
