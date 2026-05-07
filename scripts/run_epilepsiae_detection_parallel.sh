#!/usr/bin/env bash
# Parallel wrapper for `scripts/run_hfo_detection.py --dataset epilepsiae`.
# Per-subject Python process, xargs -P, output strictly to results/hfo_detection/
# (NEVER /mnt/epilepsia_data — guard via mandatory --output-dir).
#
# 2026-05-03: created for Path A detector rebuild (commit 6027281). Each
# subject runs with the new defaults (legacy_align=True, FIR-801 notch,
# 5-freq notch, 200s chunks, no overlap). Per-subject hyperparameters
# (pick_k, pack_win_sec) come from config/subject_params.json — they only
# affect synRefine + lagPat-stage selection, not detector. Detector params
# are dataset _defaults (rel_thresh=2, abs_thresh=2, side_thresh=2).
set -euo pipefail

cd "$(dirname "$0")/.."

N_JOBS=${N_JOBS:-1}  # v2: GPU 单卡，串行最稳；多卡可调
SUBJECTS="${SUBJECTS:-253 548 139 384 1077 1084 442 818 916 922 958 583 590 620 635 1073 1096 1125 1146 1150}"
OUTPUT_DIR="${OUTPUT_DIR:-results/hfo_detector_v2}"
CONDA_ENV="${CONDA_ENV:-cuda_env}"

# Hard-fail if OUTPUT_DIR points anywhere under /mnt/epilepsia_data — that
# would risk overwriting legacy 2021 _gpu.npz files.
case "$OUTPUT_DIR" in
    /mnt/epilepsia_data/*)
        echo "FATAL: OUTPUT_DIR='$OUTPUT_DIR' would overwrite legacy data." >&2
        exit 1
        ;;
esac

# v2 cohort PREFLIGHT: hard-fail if cuda_env's cupy + cusignal aren't importable.
# Without this gate, src/utils/bqk_utils.py:329 silently falls back to CPU
# (RuntimeWarning: GPU requested but CuPy/cusignal not available. Using CPU.)
# and a "GPU cohort" run becomes a 10×-slower CPU run with subtle numerical
# differences from the documented v2 detector spec.
if ! conda run -n "$CONDA_ENV" python -c \
        "import cupy, cusignal; print(f'cupy={cupy.__version__} cusignal={cusignal.__version__}')" \
        >/dev/null 2>&1; then
    echo "FATAL: conda env '$CONDA_ENV' missing cupy or cusignal. v2 cohort requires GPU." >&2
    echo "  Verify with: conda run -n $CONDA_ENV python -c 'import cupy, cusignal'" >&2
    exit 2
fi
echo "[preflight] cuda_env GPU stack OK: $(conda run -n "$CONDA_ENV" python -c 'import cupy, cusignal; print(f"cupy={cupy.__version__} cusignal={cusignal.__version__}")')"

mkdir -p "$OUTPUT_DIR"
mkdir -p "${OUTPUT_DIR}/_logs"

worker() {
    local subj="$1"
    local out_dir="${OUTPUT_DIR}/${subj}"
    mkdir -p "$out_dir"
    local log="${OUTPUT_DIR}/_logs/${subj}_detect.log"
    {
        echo "[detect] subject=${subj} start $(date)"
        conda run -n "$CONDA_ENV" --no-capture-output python scripts/run_hfo_detection.py \
            --dataset epilepsiae \
            --subject "$subj" \
            --output-dir "$OUTPUT_DIR" \
            --skip-existing
        echo "[detect] subject=${subj} done $(date)"
    } >>"$log" 2>&1
    echo "[detect] subject=${subj} done"
}
export -f worker
export OUTPUT_DIR CONDA_ENV

# shellcheck disable=SC2086
printf '%s\n' $SUBJECTS \
    | xargs -n1 -P "$N_JOBS" -I {} bash -c 'worker "$@"' _ {}

echo "[detect] cohort done $(date)"
