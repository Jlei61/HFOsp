#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
BASE=results/topic4_sef_hfo/data_driven_zm_ictal_transition/fig5_workpoint_etoi005
REPLAY=$BASE/replay/joint_04_control_seed_1801_frames.npz
SNAPSHOT=$BASE/mode_snapshot/seed1801_early_runaway_mode_snapshot.npz
LOW_CHECKPOINT=$BASE/checkpoints/joint_04_control_seed_1801_low_activity.npz
START_CHECKPOINT=$BASE/state_contrast/seed1801_low_vs_post120-post-checkpoint.npz
SELECTED_CHECKPOINT=$BASE/checkpoints/joint_04_control_seed_1801_mode_matched_runaway.npz
OUT=$BASE/global_perturbation
LOG=$OUT/logs
STATUS=$OUT/STATUS

mkdir -p "$OUT/chunks" "$LOG"
printf 'RUNNING\n' > "$STATUS"
trap 'printf "FAILED\n" > "$STATUS"' ERR

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kib < 62914560 )); then
  printf 'REFUSED_LOW_MEMORY available_kib=%s\n' "$available_kib" > "$STATUS"
  exit 1
fi

monitor() {
  while sleep 600; do
    {
      date --iso-8601=seconds
      awk '/MemAvailable:/ {print}' /proc/meminfo
      df -h "$ROOT" | tail -n 1
      find "$OUT/chunks" -maxdepth 1 -name '*.json' -type f | wc -l
    } >> "$LOG/monitor.log"
  done
}
monitor &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

"$PY" scripts/prepare_topic4_zm_fig5_selected_checkpoint.py \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json \
  --replay "$REPLAY" \
  --mode-snapshot "$SNAPSHOT" \
  --start-checkpoint "$START_CHECKPOINT" \
  --out "$SELECTED_CHECKPOINT" \
  > "$LOG/selected_checkpoint.log" 2>&1

pids=()
for state in low_activity runaway; do
  checkpoint=$LOW_CHECKPOINT
  if [[ "$state" == runaway ]]; then
    checkpoint=$SELECTED_CHECKPOINT
  fi
  for chunk in 0 1 2 3; do
    first=$(( chunk * 4 ))
    indices=("$first" "$((first + 1))" "$((first + 2))" "$((first + 3))")
    "$PY" scripts/run_topic4_zm_fig5_global_perturbation_worker.py \
      --config config/topic4_data_driven_zm_ictal_transition_v1.json \
      --replay "$REPLAY" \
      --mode-snapshot "$SNAPSHOT" \
      --checkpoint "$checkpoint" \
      --state-label "$state" \
      --site-indices "${indices[@]}" \
      --n-side 4 \
      --site-seed 20260820 \
      --dose-cells 16 \
      --window-ms 200 \
      --out "$OUT/chunks/${state}_chunk${chunk}.npz" \
      > "$LOG/${state}_chunk${chunk}.log" 2>&1 &
    pids+=("$!")
  done
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PY" scripts/aggregate_topic4_zm_fig5_global_perturbation.py \
  --low-chunks \
    "$OUT/chunks/low_activity_chunk0.npz" \
    "$OUT/chunks/low_activity_chunk1.npz" \
    "$OUT/chunks/low_activity_chunk2.npz" \
    "$OUT/chunks/low_activity_chunk3.npz" \
  --runaway-chunks \
    "$OUT/chunks/runaway_chunk0.npz" \
    "$OUT/chunks/runaway_chunk1.npz" \
    "$OUT/chunks/runaway_chunk2.npz" \
    "$OUT/chunks/runaway_chunk3.npz" \
  --mode-snapshot "$SNAPSHOT" \
  --out "$OUT/seed1801_global_random_state_contrast.npz" \
  > "$LOG/aggregate.log" 2>&1

"$PY" scripts/paper_figures/plot_fig5_data_driven_zm_main.py \
  --config config/topic4_data_driven_zm_ictal_transition_v1.json \
  --replay "$REPLAY" \
  --mode-snapshot "$SNAPSHOT" \
  --global-state-contrast "$OUT/seed1801_global_random_state_contrast.npz" \
  --allow-exploratory-workpoint \
  --stem fig5-data-driven-zm-etoi005-main-v5 \
  > "$LOG/plot.log" 2>&1

printf 'COMPLETE\n' > "$STATUS"
printf 'Fig5 global perturbation pipeline complete: %s\n' "$OUT"
