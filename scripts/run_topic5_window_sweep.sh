#!/usr/bin/env bash
# Topic5 A-line §3.2 window sweep driver. Run AFTER the v2 trace cache (t0_feature_cache_v2_windows)
# is fully built (needs --store-bb-zt --pre-feature-window 130). Slices all v2 subjects into 6
# per-window standard caches, then runs the alignment runner per window x {broadband, hfa}, B=1000.
# Distal pre [-120,-90] is the load-bearing negative control (should fall well below the ictal windows).
set -euo pipefail
cd "$(dirname "$0")/.."
WC=results/topic5_ictal_recruitment/axis_alignment/window_caches
OUT=results/topic5_ictal_recruitment/axis_alignment/window
mkdir -p "$OUT"

echo "[window-sweep] slicing v2 trace cache -> per-window caches"
python scripts/build_topic5_window_caches.py

for win in post_0_5 post_5_10 post_0_10 post_0_20 pre_prox_m10_0 pre_distal_m120_m90; do
  for act in broadband hfa; do
    echo "[window-sweep] $win $act"
    python scripts/run_topic5_axis_alignment.py --cache-dir "$WC/$win" \
      --activation "$act" --B 1000 \
      --out "$OUT/win_${win}_${act}_B1000.json"
  done
done
echo WINDOW_SWEEP_ALL_DONE
