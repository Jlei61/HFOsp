#!/usr/bin/env bash
# M3A-A2 batch 5 — A2-P propagation-gate CONFIRMATION at the full propagation instrument.
#
# The T=8000 first pass (scripts/audit_a2p_propagation.py) gave GATE=FAIL on the coarse
# bin-based readout: the g_K rho>1.35 excursion is a population-rate (timing) oscillation,
# not an interictal<->seizure propagation switch. This batch re-runs the best point + seeds
# + the two phenotype endpoints (baseline interictal, depletion-only runaway) + two more
# excitable anchors (substrate-specificity probe), all with --dump-fullfield (source-space
# spatial field, the rich r95_mm / onset map) at LONGER T (more bouts -> more statistical
# weight on the FAIL), so the verdict is nailed on the full instrument, not just bins.
#
# Usage:  bash scripts/run_a2_batch5_fullfield.sh <out_dir> [T=20000] [max_jobs=12]
set -u
OUT="${1:?out dir}"; T="${2:-20000}"; MAXJ="${3:-12}"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
RUN=scripts/run_sef_hfo_snn_cm_spontaneous_readout.py
BASE="--L 20 --density 100 --theta 45 --core-mean 17.5 --core-std 1.0 --core-r 1.5 --sep-frac 0.7 \
--drive 0.6 --lesion twoend_equal --T $T"
INSTR="--dump-fullfield --dump-a2-trace"               # full source-space spatial readout + rho trace
DEPL="--a2-mode core_only --a2-tau-rec 2000 --a2-tau-a 100 --a2-q-min 0.25 --a2-boundary 1.35"
GK="$DEPL --a2-gk-max 0.03 --a2-tau-k 2000"            # depletion + the g_K brake (best-point pack)

throttle(){ while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 2; done; }
# launch <tag> <extra args...>
launch(){ throttle; local tag="$1"; shift; echo "LAUNCH $tag :: $*"; \
  python $RUN $BASE "$@" --out "$OUT" --tag "$tag" >"$OUT/log_${tag}.txt" 2>&1 & }

# anchor knob packs
L0="--core-ei-scale 1.0 --core-ee-gain 1.0 --global-ei-scale 1.0"
L1="--core-ei-scale 0.85 --core-ee-gain 1.15 --global-ei-scale 1.3"
L2="--core-ei-scale 0.70 --core-ee-gain 1.30 --global-ei-scale 1.6"

# --- core gate: best point, 3 seeds (seed robustness of the FAIL, rich instrument) ---
for s in 1 2 3; do
  launch l0_gk_best_s${s} $L0 $GK --a2-k-use 0.2 --seed $s $INSTR
done
# --- phenotype endpoints at matched instrument/T ---
launch l0_base_s1        $L0 $DEPL --a2-k-use 0.0 --seed 1 $INSTR             # interictal reference (no depletion)
launch l0_runaway_k0.4_s1 $L0 $DEPL --a2-k-use 0.4 --seed 1 $INSTR            # depletion-only runaway (no g_K)
# --- substrate-specificity probe: more excitable anchors + same g_K brake ---
launch l1_gk_best_s1 $L1 $GK --a2-k-use 0.2 --seed 1 $INSTR
launch l2_gk_best_s1 $L2 $GK --a2-k-use 0.2 --seed 1 $INSTR

wait
echo "A2 batch5 fullfield DONE -> $OUT  ($(ls "$OUT"/readout_*.json 2>/dev/null | wc -l) readouts)"
