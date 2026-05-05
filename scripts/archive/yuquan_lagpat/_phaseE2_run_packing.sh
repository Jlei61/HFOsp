#!/usr/bin/env bash
# Yuquan 24-subject same-source lagPat / packedTimes batch packing.
#
# Subject list is the single source of truth in
# `scripts/run_yuquan_lagpat_backfill.py::YUQUAN_SAME_SOURCE_SUBJECTS`
# (3 reference + 10 main cohort with legacy lagPat + 8 backfill-only).
# Pack-stage parameters are read from `config/subject_params.json`.
#
# Modes:
#   default     - write into each subject's raw EDF dir (legacy lagPat is
#                 first moved into `<subject>/.legacy_backup/`).
#   DRY_RUN=1   - write into $DRY_RUN_OUT_DIR (default: results/lagpat_dryrun);
#                 raw dir is never touched.
#
# Failure policy: continue on per-subject failure but track a non-zero exit
# code at the end so CI / orchestrators can still detect partial failure.
#
# This is a CPU/IO job (no GPU); safe to run in parallel with GPU-bound
# detection.
cd "$(dirname "$0")/.."

PY="${PY:-/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python}"
LOG_DIR="${LOG_DIR:-logs/yuquan_lagpat_pack}"
mkdir -p "$LOG_DIR"

DRY_RUN="${DRY_RUN:-0}"
DRY_RUN_OUT_DIR="${DRY_RUN_OUT_DIR:-results/lagpat_dryrun}"

# Subject list comes from the same Python constant the script uses, so the
# bash file can never silently drift from the Python cohort definition.
SUBJECT_LIST_RAW=$(
  PYTHONPATH=scripts:. "$PY" -c \
    'from run_yuquan_lagpat_backfill import YUQUAN_SAME_SOURCE_SUBJECTS; print(" ".join(YUQUAN_SAME_SOURCE_SUBJECTS))'
)
if [ -z "$SUBJECT_LIST_RAW" ]; then
  echo "FATAL: failed to load YUQUAN_SAME_SOURCE_SUBJECTS from python" >&2
  exit 2
fi
read -r -a SUBJECTS <<< "$SUBJECT_LIST_RAW"

if [ "${1:-}" = "--list" ]; then
  printf '%s\n' "${SUBJECTS[@]}"
  exit 0
fi

if [ -n "${ONLY_SUBJECTS:-}" ]; then
  read -r -a SUBJECTS <<< "$ONLY_SUBJECTS"
fi

EXTRA_ARGS=()
if [ "$DRY_RUN" = "1" ]; then
  mkdir -p "$DRY_RUN_OUT_DIR"
  EXTRA_ARGS+=(--dry-run-out-dir "$DRY_RUN_OUT_DIR")
  echo "=== DRY RUN MODE: writing to $DRY_RUN_OUT_DIR ==="
fi

echo "=== [$(date '+%F %T')] Yuquan same-source packing start: ${#SUBJECTS[@]} subjects ==="
echo "subjects: ${SUBJECTS[*]}"
echo "log_dir: $LOG_DIR"
echo

failed=()
ok=()
for s in "${SUBJECTS[@]}"; do
  echo "=== [$(date '+%F %T')] starting $s ==="
  if "$PY" scripts/run_yuquan_lagpat_backfill.py \
    --subject "$s" \
    "${EXTRA_ARGS[@]}" \
    > "$LOG_DIR/${s}.log" 2>&1; then
    ok+=("$s")
    tail -n 1 "$LOG_DIR/${s}.log"
    echo "=== [$(date '+%F %T')] finished $s OK ==="
  else
    failed+=("$s")
    echo "=== [$(date '+%F %T')] $s FAILED, see $LOG_DIR/${s}.log ==="
    tail -n 20 "$LOG_DIR/${s}.log"
  fi
done

echo
echo "=== [$(date '+%F %T')] Yuquan same-source packing DONE ==="
echo "ok=${#ok[@]} failed=${#failed[@]}"
if [ ${#failed[@]} -gt 0 ]; then
  echo "FAILED: ${failed[*]}"
  exit 1
fi
exit 0
