#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:-$ROOT/results/topic5_minimal_sequence_kernel_closeout/rank_tolerance_v0_2}"
MAX_WORKERS="${MAX_WORKERS:-4}"
DEVICE="${DEVICE:-cpu}"
mkdir -p "$OUT_ROOT/logs"

mapfile -t SUBJECTS < <(
  find "$ROOT/results/topic5_interictal_rank_distribution/dataset_v0_4/per_subject" \
    -maxdepth 1 -type f -name '*.npz' -printf '%f\n' \
    | sed 's/\.npz$//' | sort
)

run_subject() {
  local subject="$1"
  local output="$OUT_ROOT/$subject"
  local log="$OUT_ROOT/logs/${subject}.log"
  if [[ -f "$output/summary.json" ]]; then
    return 0
  fi
  if [[ -d "$output" ]]; then
    mv "$output" "${output}.incomplete.$(date +%Y%m%dT%H%M%S)"
  fi
  conda run -n cuda_env python \
    "$ROOT/scripts/analyze_topic5_rank_tolerance_subject_v0_2.py" \
    --subject "$subject" \
    --output-dir "$output" \
    --device "$DEVICE" \
    --batch-size 512 \
    --cpu-threads 4 \
    >"$log" 2>&1
}
export ROOT OUT_ROOT DEVICE
export -f run_subject

printf '%s\n' "${SUBJECTS[@]}" \
  | xargs -P "$MAX_WORKERS" -n 1 bash -c 'run_subject "$0"'

complete="$(find "$OUT_ROOT" -name summary.json -type f | wc -l)"
if [[ "$complete" -ne "${#SUBJECTS[@]}" ]]; then
  echo "expected ${#SUBJECTS[@]} complete subjects, found $complete" >&2
  exit 1
fi
printf '{"status":"COMPLETE","expected_subjects":%d,"complete_subjects":%d}\n' \
  "${#SUBJECTS[@]}" "$complete" >"$OUT_ROOT/LAUNCHER_DONE.json"
