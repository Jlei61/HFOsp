#!/usr/bin/env bash
# Post-PR-2 sequence for Slice A2 (Path D).
# Runs PR-2.5 + PR-6 + double-track cohort aggregation.
# Idempotent: each step writes its own log + output, no destructive mutations.

set -euo pipefail
cd /home/honglab/leijiaxin/HFOsp

LOG_DIR="results/run_logs"
mkdir -p "$LOG_DIR"

ALL_NEW="zhangjiaqi gaolan wangyiyang zhangkexuan pengzihang songzishuo zhangbichen zhaochenxi zhaojinrui zhourongxuan"
PATH_D="zhangkexuan pengzihang songzishuo zhangbichen zhaochenxi zhaojinrui zhourongxuan"

CONDA="conda run -n cuda_env --no-capture-output"

# If the script aborts mid-run while PR-6 outputs are stashed, restore them.
STASH=/tmp/pr6_pathd_stash_2026-05-07
PR6_DIR=results/interictal_propagation/template_anchoring
_restore_stash() {
    if [ -d "$STASH" ]; then
        shopt -s nullglob
        for f in "$STASH"/*.json; do
            mv "$f" "$PR6_DIR/per_subject/" 2>/dev/null || true
        done
        shopt -u nullglob
    fi
}
trap _restore_stash EXIT

echo "=== [0/7] Pre-flight: detect PR-2 failures (advisor BLOCKER 2 guard) ==="
EXCLUDE_LIST=""
for s in $PATH_D; do
    p="results/interictal_propagation/per_subject/yuquan_${s}.json"
    if [ ! -f "$p" ]; then
        echo "  ✗ MISSING: $p"
        EXCLUDE_LIST="$EXCLUDE_LIST yuquan/${s}"
        continue
    fi
    bad=$(python -c "
import json
d = json.load(open('$p'))
if 'error' in d:
    print('top_error')
elif 'error' in (d.get('adaptive_cluster') or {}):
    print('adaptive_cluster_error')
elif 'propagation_stereotypy' not in d or 'mixture' not in d:
    print('missing_required_field')
else:
    print('ok')
")
    if [ "$bad" != "ok" ]; then
        echo "  ✗ FAILED ($bad): yuquan/${s}"
        EXCLUDE_LIST="$EXCLUDE_LIST yuquan/${s}"
    else
        echo "  ✓ ok: yuquan/${s}"
    fi
done
EXCLUDE_LIST=$(echo $EXCLUDE_LIST | xargs)
if [ -n "$EXCLUDE_LIST" ]; then
    echo "  Will exclude from n=40 aggregation: $EXCLUDE_LIST"
fi

echo "=== [1/7] PR-2.5 reproducibility on 10 new subjects ==="
$CONDA python scripts/run_interictal_propagation.py \
    --pr25 --dataset yuquan --subjects $ALL_NEW \
    > "$LOG_DIR/path_d_pr25_2026-05-07.log" 2>&1
echo "  PR-2.5 done -> $LOG_DIR/path_d_pr25_2026-05-07.log"

echo "=== [2/7] Aggregate primary cohort (n=33, manifest n33) ==="
$CONDA python scripts/aggregate_propagation_cohort.py \
    --manifest results/interictal_propagation/cohort_manifest_n33_2026-05-06.txt \
    > "$LOG_DIR/path_d_aggregate_n33_2026-05-07.log" 2>&1
echo "  -> pr1_cohort_summary.json (default path = primary)"

echo "=== [3/7] Aggregate extended cohort (n=40 minus failures, manifest n40) ==="
EXCLUDE_ARG=""
if [ -n "$EXCLUDE_LIST" ]; then
    EXCLUDE_ARG="--exclude $EXCLUDE_LIST"
fi
$CONDA python scripts/aggregate_propagation_cohort.py \
    --manifest results/interictal_propagation/cohort_manifest_n40_2026-05-07.txt \
    $EXCLUDE_ARG \
    --out-summary results/interictal_propagation/pr1_subject_summary_n40.json \
    --out-cohort  results/interictal_propagation/pr1_cohort_summary_n40.json \
    > "$LOG_DIR/path_d_aggregate_n40_2026-05-07.log" 2>&1
echo "  -> pr1_cohort_summary_n40.json (extended)"

echo "=== [4/7] PR-6 driver --all (writes per-subject + cohort summary at default paths) ==="
$CONDA python scripts/run_pr6_template_anchoring.py --all \
    > "$LOG_DIR/path_d_pr6_2026-05-07.log" 2>&1
cp "$PR6_DIR/cohort_summary.json" "$PR6_DIR/cohort_summary_n40.json"
echo "  PR-6 n=40 cohort -> $PR6_DIR/cohort_summary_n40.json"

echo "=== [5/7] PR-6 cohort recompute on n=33 (stash 7 path-D outputs aside) ==="
mkdir -p "$STASH"
for s in $PATH_D; do
    if [ -f "$PR6_DIR/per_subject/yuquan_${s}.json" ]; then
        mv "$PR6_DIR/per_subject/yuquan_${s}.json" "$STASH/"
    fi
done
$CONDA python scripts/run_pr6_template_anchoring.py --cohort \
    >> "$LOG_DIR/path_d_pr6_2026-05-07.log" 2>&1
cp "$PR6_DIR/cohort_summary.json" "$PR6_DIR/cohort_summary_n33.json"
echo "  PR-6 n=33 cohort -> $PR6_DIR/cohort_summary_n33.json"

# Restore 7 path-D outputs (default cohort_summary.json now stays at n=33 primary,
# matching pr1_cohort_summary.json default convention).
for f in "$STASH"/*.json; do
    [ -f "$f" ] && mv "$f" "$PR6_DIR/per_subject/"
done
echo "  Restored 7 path-D anchoring outputs to per_subject/. Default cohort_summary.json stays at n=33."

echo "=== [6/7] PR-6 valid_mask_source audit (advisor BLOCKER 1 guard) ==="
python scripts/_audit_pr6_valid_mask_source.py "$PR6_DIR/per_subject" \
    2>&1 | tee -a "$LOG_DIR/path_d_pr6_2026-05-07.log"

echo "=== [7/7] Verification ==="
python -c "
import json
print()
print('=== PR-1/PR-2 Cohort Summaries ===')
print('Primary (n=33) ', end='');
d = json.load(open('results/interictal_propagation/pr1_cohort_summary.json'))
print(f'n={d[\"n_subjects\"]} mixture={d[\"n_strict_mixture\"]}/{d[\"n_possible_mixture\"]} mean_tau_med={d[\"mean_tau_median\"]:.4f} bias_med={d[\"bias_fraction_median\"]:.4f}')
print('Extended       ', end='')
d = json.load(open('results/interictal_propagation/pr1_cohort_summary_n40.json'))
print(f'n={d[\"n_subjects\"]} mixture={d[\"n_strict_mixture\"]}/{d[\"n_possible_mixture\"]} mean_tau_med={d[\"mean_tau_median\"]:.4f} bias_med={d[\"bias_fraction_median\"]:.4f}')

print()
print('=== PR-6 Cohort Summaries ===')
import os
for tag in ('n33', 'n40'):
    p = f'results/interictal_propagation/template_anchoring/cohort_summary_{tag}.json'
    if os.path.exists(p):
        d = json.load(open(p))
        h1 = d.get('h1_pooled', {})
        print(f'PR-6 {tag}: H1 pooled n={h1.get(\"n\")} median_delta={h1.get(\"median\")} wilcoxon_p={h1.get(\"wilcoxon_p\")}')
"

echo ""
echo "=== Done. ==="
