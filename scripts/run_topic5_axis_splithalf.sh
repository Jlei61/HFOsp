#!/usr/bin/env bash
# Topic5 A-line §3.1 split-half robustness driver.
# Half-axis records (real_subjects_splithalf/{sid}_t_a_{mode}_half{N}.json) are produced by
# run_contact_plane_readout.py --event-split. The alignment runner expects {sid}_t_a.json, so we
# symlink each half record under a per-(mode,half) dir and point --axis-dir there. Ictal side
# (--cache-dir) is UNCHANGED (same 0-10s t0 cache); only the interictal AXIS varies by half.
# broadband(PRIMARY) + hfa(sensitivity) only; B=1000. Output -> axis_alignment/splithalf/.
set -euo pipefail
cd "$(dirname "$0")/.."
OBS=results/spatial_modulation/propagation_geometry/observation_readout
SH="$OBS/real_subjects_splithalf"
OUT=results/topic5_ictal_recruitment/axis_alignment/splithalf
mkdir -p "$OUT"

for mode in first_second odd_even; do
  for half in 1 2; do
    DIR="$OBS/real_subjects_${mode}_half${half}"
    rm -rf "$DIR"; mkdir -p "$DIR"
    n=0
    for f in "$SH"/*_t_a_${mode}_half${half}.json; do
      [ -e "$f" ] || continue
      base=$(basename "$f")
      sid="${base%_t_a_${mode}_half${half}.json}"      # e.g. epilepsiae_1077
      ln -sf "$(realpath "$f")" "$DIR/${sid}_t_a.json"
      n=$((n+1))
    done
    echo "[splithalf] ${mode} half${half}: linked ${n} half-axis records -> $DIR"
    for act in broadband hfa; do
      python scripts/run_topic5_axis_alignment.py --axis-dir "$DIR" \
        --activation "$act" --B 1000 \
        --out "$OUT/sh_${mode}_half${half}_${act}_B1000.json"
    done
  done
done
echo SPLITHALF_ALL_DONE
