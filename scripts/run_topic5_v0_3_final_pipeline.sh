#!/usr/bin/env bash
# Unattended tail of the v0.3 Figure 6 chain.
#
# Waits for the 306-unit formal training to finish, then runs the interictal
# summary, both source-pool field freezes, both immutable manifests, and only
# then unseals the early-ictal target.  Figures run last.
#
# Fail-fast: any non-zero step writes PIPELINE_FAILED.json and stops, so a
# broken step can never be mistaken for a completed chain.
set -u -o pipefail

W=/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-structured-rnn-fig6
PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
CFG=config/topic5_source_conditioned_shared_scaffold_rnn_v0_3_final.yaml
RCFG=config/topic5_source_conditioned_ictal_readout_v0_3.yaml
OUT=results/topic5_patient_specific_source_conditioned_rnn_v0_3_final
LOGS=$W/$OUT/pipeline_logs
cd "$W" || exit 1
mkdir -p "$LOGS"

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

fail() {
  printf '{"status":"FAILED","step":"%s","utc":"%s","log":"%s"}\n' \
    "$1" "$(stamp)" "$LOGS/$1.log" > "$W/$OUT/PIPELINE_FAILED.json"
  echo "[$(stamp)] FAILED at step $1 (see $LOGS/$1.log)"
  exit 1
}

step() {
  local name=$1; shift
  echo "[$(stamp)] START $name"
  if ! "$@" > "$LOGS/$name.log" 2>&1; then fail "$name"; fi
  echo "[$(stamp)] OK    $name"
}

# ---------------------------------------------------------------- 0. wait
echo "[$(stamp)] waiting for 306-unit training"
while :; do
  done_n=$(find "$W/$OUT/per_subject" -name DONE.json 2>/dev/null | wc -l)
  fail_n=$(find "$W/$OUT/per_subject" -name FAILED.json 2>/dev/null | wc -l)
  if [ "$fail_n" -gt 0 ]; then fail "training_unit_failed"; fi
  if [ "$done_n" -ge 306 ]; then break; fi
  if ! tmux has-session -t topic5_final_train 2>/dev/null; then
    echo "[$(stamp)] launcher gone at $done_n/306 units"
    fail "training_launcher_exited_early"
  fi
  sleep 120
done
echo "[$(stamp)] training complete: $done_n/306"
tmux kill-session -t topic5_final_watch 2>/dev/null

# --------------------------------------------------- 1. interictal summary
step interictal_summary \
  $PY scripts/analyze_topic5_shared_scaffold_interictal_v0_2.py \
    --config "$CFG" --output-root "$OUT"

# ------------------------------- 2. target-free fields, both source rules
# Primary = the model's own learned signed axis.
step rollouts_learned_axis \
  $PY scripts/launch_topic5_shared_scaffold_rollouts_v0_2.py \
    --config "$CFG" --models structured ordinary_gru \
    --source-pool-rule learned_axis --workers 10 --resume

step manifest_learned_axis \
  $PY scripts/freeze_topic5_shared_scaffold_field_manifest_v0_2.py \
    --config "$CFG" --models structured ordinary_gru \
    --source-pool-rule learned_axis

# Sensitivity = the pre-registered diffusion-graph split.
step rollouts_diffusion_graph \
  $PY scripts/launch_topic5_shared_scaffold_rollouts_v0_2.py \
    --config "$CFG" --models structured ordinary_gru \
    --source-pool-rule normalized_laplacian --workers 10 --resume

step manifest_diffusion_graph \
  $PY scripts/freeze_topic5_shared_scaffold_field_manifest_v0_2.py \
    --config "$CFG" --models structured ordinary_gru \
    --source-pool-rule normalized_laplacian

# ------------------------------------- 3. target unseal (manifests exist)
step score_learned_axis \
  $PY scripts/score_topic5_shared_scaffold_early_ictal_v0_2.py \
    --readout-config "$RCFG" --training-config "$CFG" \
    --output-root "$OUT" --source-pool-rule learned_axis

step score_diffusion_graph \
  $PY scripts/score_topic5_shared_scaffold_early_ictal_v0_2.py \
    --readout-config "$RCFG" --training-config "$CFG" \
    --output-root "$OUT" --source-pool-rule normalized_laplacian

# --------------------------------------------------------- 4. Figure 6
step figure6 \
  $PY scripts/paper_figures/plot_topic5_figure6_source_conditioned_rnn.py \
    --training-config "$CFG" --readout-config "$RCFG" --output-root "$OUT"

printf '{"status":"COMPLETE","utc":"%s"}\n' "$(stamp)" > "$W/$OUT/PIPELINE_COMPLETE.json"
echo "[$(stamp)] PIPELINE COMPLETE"
