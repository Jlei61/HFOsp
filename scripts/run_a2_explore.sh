#!/usr/bin/env bash
# M3A-A2 parallel exploration driver. Launches the run matrix with threads pinned (single-thread per
# run) and a concurrency cap, so a wide grid finishes in a few waves on a many-core box. Usage:
#   bash scripts/run_a2_explore.sh <out_dir> <T> <max_jobs>
set -u
OUT="${1:?out dir}"; T="${2:-8000}"; MAXJ="${3:-40}"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
RUN=scripts/run_sef_hfo_snn_cm_spontaneous_readout.py
BASE="--L 20 --density 100 --theta 45 --core-mean 17.5 --core-std 1.0 --core-r 1.5 --sep-frac 0.7 \
--drive 0.6 --lesion twoend_equal --T $T"

# anchor: "name cei cee gei lgr"
ANCHORS=("l0_g1.0 1.0 1.0 1.0 1.00" "l1_g1.3 0.85 1.15 1.3 1.04" "l2_g1.6 0.70 1.30 1.6 1.16")
KUSE=(0.05 0.1 0.2 0.4 0.8)            # core_only dynamic ladder (wide: stay->excursion->runaway)
TAUREC=2000

throttle(){ while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 2; done; }
launch(){ throttle; python $RUN $BASE "$@" >/dev/null 2>&1 & }

for A in "${ANCHORS[@]}"; do
  set -- $A; name=$1; cei=$2; cee=$3; gei=$4; lgr=$5
  K="--core-ei-scale $cei --core-ee-gain $cee --global-ei-scale $gei"
  # Task-0 baselines (k_use=0) seeds 1-2  -> a_bar + clean baseline
  for s in 1 2; do
    launch $K --seed $s --a2-mode core_only --a2-k-use 0.0 --dump-a2-trace \
      --out "$OUT" --tag ${name}_base_s${s}
  done
  # Task-0b frozen-q at the A1b-predicted seizure / runaway products (q_core = lgr/1.35, lgr/1.86)
  for rho in 1.35 1.86; do
    q=$(python -c "print(round($lgr/$rho,4))")
    launch $K --seed 1 --a2-mode core_only --a2-frozen --a2-frozen-qcore $q --a2-frozen-qglobal 1.0 \
      --dump-a2-trace --out "$OUT" --tag ${name}_frzq${q}_s1
  done
  # Dynamic core_only sweep over k_use (seed 1; provisional --a2-boundary 1.35, refined post-hoc from rho_bin)
  for k in "${KUSE[@]}"; do
    launch $K --seed 1 --a2-mode core_only --a2-k-use $k --a2-tau-rec $TAUREC --a2-tau-a 100 \
      --a2-q-min 0.25 --a2-boundary 1.35 --dump-a2-trace --out "$OUT" --tag ${name}_dyn_k${k}_t${TAUREC}_s1
  done
done
wait
echo "A2 explore batch DONE -> $OUT  ($(ls "$OUT"/readout_*.json 2>/dev/null | wc -l) readouts)"
