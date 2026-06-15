#!/usr/bin/env bash
# Topic5 A-line ENHANCED sweep v2 (2026-06-14, review-corrected): 4 nulls (channel /
# within-shaft / anchor-matched / joint) + effective-shuffle + leave-one-subject-out, then
# figures, then the tiered + FDR-controlled aggregate. The cache already has baseline-activity
# (bact) + ei_like, so this is analysis-only (no reload/augment). broadband = primary metric;
# HFA/ramp = sensitivity; EI-like = exploratory.
set -u
cd /home/honglab/leijiaxin/HFOsp || exit 1
echo "[sweep-v2] $(date) START — enhanced 4-null A-line"
for m in broadband hfa ramp ei; do for B in 1000 2000; do
  echo "[sweep-v2] $(date) A-line activation=$m B=$B"
  python scripts/run_topic5_axis_alignment.py --activation "$m" --B "$B" || echo "[sweep-v2] FAILED $m B=$B"
done; done
echo "[sweep-v2] $(date) figures"
python scripts/plot_topic5_axis_alignment.py 1000 || echo "[sweep-v2] fig1000 failed"
python scripts/plot_topic5_axis_alignment.py 2000 || echo "[sweep-v2] fig2000 failed"
echo "[sweep-v2] $(date) aggregate (tiered + FDR + LOSO + degeneracy)"
python scripts/aggregate_topic5_axis_alignment.py || echo "[sweep-v2] aggregate failed"
echo "SWEEP_V2_DONE $(date)"
