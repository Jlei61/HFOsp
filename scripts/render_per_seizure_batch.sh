#!/bin/bash
# Batch-render per-seizure atlas PNGs for v2.3 cohort + sentinels.
#
# For each subject with a v2.3 per-subject JSON, render every seizure.
# Skips ones already rendered (atlas script's default --skip-existing).
#
# Usage: bash scripts/render_per_seizure_batch.sh [SUBJECT_LIST]
#        Default subject list = all *.json under per_subject/ + _sentinel/

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYBIN="${PYBIN:-python}"

PER_SUBJECT_DIR="$ROOT/results/data_driven_soz/layer_a_ictal_er_rank/per_subject"
SENTINEL_DIR="$ROOT/results/data_driven_soz/layer_a_ictal_er_rank/_sentinel"

# Build candidate list: every epilepsiae_*.json that is v2.3
declare -A SEEN
SUBJECTS=()
for d in "$PER_SUBJECT_DIR" "$SENTINEL_DIR"; do
    [ -d "$d" ] || continue
    for f in "$d"/epilepsiae_*.json; do
        [ -f "$f" ] || continue
        sid=$(basename "$f" .json)
        sid_short="${sid#epilepsiae_}"
        # Skip cohort_summary / sanity_report / etc.
        case "$sid_short" in cohort_*|sanity_*) continue ;; esac
        # Match epilepsiae_<digits>.json only
        case "$sid_short" in [0-9]*) ;;  *) continue ;; esac
        # v2.3 schema check
        sv=$($PYBIN -c "import json,sys; d=json.load(open('$f')); print(d.get('schema_version',''))" 2>/dev/null)
        if [ "$sv" = "pr_t3_1_layer_a_v2_3_timing" ] && [ -z "${SEEN[$sid_short]:-}" ]; then
            SUBJECTS+=("$sid_short")
            SEEN[$sid_short]=1
        fi
    done
done

# Sort ascending
IFS=$'\n' SUBJECTS=($(sort -n <<<"${SUBJECTS[*]}"))
unset IFS

echo "[batch] rendering per-seizure PNGs for ${#SUBJECTS[@]} subjects:"
printf "  %s\n" "${SUBJECTS[@]}"

t_start=$(date +%s)
total_seizures=0
total_skipped=0
total_failed=0

for sid in "${SUBJECTS[@]}"; do
    subject="epilepsiae/$sid"
    # Read n_seizures_total
    json="$PER_SUBJECT_DIR/epilepsiae_${sid}.json"
    [ -f "$json" ] || json="$SENTINEL_DIR/epilepsiae_${sid}.json"
    n_sz=$($PYBIN -c "import json; print(json.load(open('$json'))['n_seizures_total'])" 2>/dev/null)
    [ -z "$n_sz" ] && { echo "  [skip] $subject: cannot read n_seizures_total"; continue; }
    src_flag=""
    [ -f "$PER_SUBJECT_DIR/epilepsiae_${sid}.json" ] || src_flag="--from-sentinel"
    echo ""
    echo "=== $subject  n_seizures=$n_sz  $src_flag  ==="
    for ((sz=0; sz<n_sz; sz++)); do
        out="$ROOT/results/data_driven_soz/layer_a_ictal_er_rank/atlas_v2_3/figures/per_seizure/epilepsiae_${sid}_seizure_$(printf %02d $sz).png"
        if [ -f "$out" ]; then
            total_skipped=$((total_skipped + 1))
            continue
        fi
        if $PYBIN scripts/plot_ictal_er_atlas.py per-seizure \
            --subject "$subject" --seizure-idx "$sz" $src_flag 2>&1 | tail -1; then
            total_seizures=$((total_seizures + 1))
        else
            echo "  [fail] $subject seizure $sz"
            total_failed=$((total_failed + 1))
        fi
    done
done

t_end=$(date +%s)
echo ""
echo "[batch] DONE in $((t_end - t_start))s : rendered=$total_seizures  skipped=$total_skipped  failed=$total_failed"
