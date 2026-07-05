#!/usr/bin/env bash
# M3A-A2 axis-break + self-limit combined sweep. The smoke showed two_tank breaks the axis only when
# the GLOBAL tank actually drains (needs drive ~0.9 + k high -> q_global -> ~0.1), but full drain goes
# TONIC. drive 0.6 barely drains (q_global ~0.85 -> stays axial). So this sweep focuses the axis-break
# zone (drive 0.8-1.0) x the termination levers (gk_max, tau_k, tau_rec) + drain depth (q_min), looking
# for a DISCRETE off-axis self-limiting event. core_only controls isolate the global tank's role.
# off-axis = biggest event isotropy>0.7 AND reach_perp>8 (round + wide perpendicular = broke corridor).
# Usage:  bash scripts/run_a2_axisbreak_sweep.sh <out_dir> [T=8000] [max_jobs=14]
set -u
OUT="${1:?out dir}"; T="${2:-8000}"; MAXJ="${3:-14}"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
RUN=scripts/run_a2_axisbreak_sweep.py
throttle(){ while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 2; done; }
# launch <tag> <mode> <k_use> <drive> <gk_max> <q_min> <tau_k> <tau_rec>   (skips if summary exists)
launch(){ local tag="$1"
  if [ -f "$OUT/summary_${tag}.json" ]; then echo "SKIP $tag (already done)"; return; fi
  throttle; shift
  echo "LAUNCH $tag :: mode=$1 k=$2 drive=$3 gk=$4 qmin=$5 tk=$6 trec=$7"
  python $RUN --T "$T" --a2-mode "$1" --a2-k-use "$2" --drive "$3" --a2-gk-max "$4" --a2-q-min "$5" \
    --a2-tau-k "$6" --a2-tau-rec "$7" --tag "$tag" --out "$OUT" >"$OUT/log_${tag}.txt" 2>&1 & }

# axis-break reference (smoke-like, expect off_axis_TONIC) then add termination
launch tonic_ref      two_tank 0.8 0.9 0.08 0.1 2000 2000
launch gk15_tk1_tr1   two_tank 0.8 0.9 0.15 0.1 1000 1000
launch gk25_tk1_tr1   two_tank 0.8 0.9 0.25 0.1 1000 1000
launch gk40_tk5_tr5   two_tank 0.8 0.9 0.40 0.1  500  500
launch gk40_tk1_tr1   two_tank 0.8 0.9 0.40 0.1 1000 1000
# partial drain (q_min 0.2) -> milder disinhibition, may stay discrete
launch q2_gk25_tk1    two_tank 0.8 0.9 0.25 0.2 1000 1000
launch q2_gk40_tk5    two_tank 0.8 0.9 0.40 0.2  500  500
# less drive (0.8): does it still break the axis, more terminable?
launch d8_q1_gk25     two_tank 0.8 0.8 0.25 0.1 1000 1000
launch d8_q2_gk25     two_tank 0.8 0.8 0.25 0.2 1000 1000
# more drive / faster drain + strongest brake
launch d10_gk40_tk5   two_tank 0.8 1.0 0.40 0.1  500  500
launch k15_gk40_tk5   two_tank 1.5 0.9 0.40 0.1  500  500
launch mid_all        two_tank 1.0 0.95 0.30 0.15 700 700
# controls
launch ctl_core_d9    core_only 0.8 0.9 0.25 0.1 1000 1000
launch ctl_best_ref   core_only 0.2 0.6 0.03 0.25 2000 2000

wait
echo "A2 axis-break combined sweep DONE -> $OUT  ($(ls "$OUT"/summary_*.json 2>/dev/null | wc -l) cells)"
