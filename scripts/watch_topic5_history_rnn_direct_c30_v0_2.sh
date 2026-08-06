#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-history-rnn-early-ictal}"
ARTIFACT_ROOT="${2:-/home/honglab/leijiaxin/HFOsp}"
REFIT_ROOT="results/topic5_history_rnn_direct_early_ictal_transfer_v0_2/g1_refit_c30"
DIRECT_ROOT="results/topic5_history_rnn_direct_c30_candidate_v0_2"
WATCH_ROOT="results/topic5_history_rnn_direct_early_ictal_transfer_v0_2/watchers/c30"

cd "${REPO_ROOT}"
mkdir -p "${WATCH_ROOT}"

while [[ ! -f "${REFIT_ROOT}/REFIT_SUMMARY.json" ]]; do
  sleep 30
done

python - "${REFIT_ROOT}/REFIT_SUMMARY.json" <<'PY'
import json, pathlib, sys
summary = json.loads(pathlib.Path(sys.argv[1]).read_text())
if summary.get("status") != "COMPLETE":
    raise SystemExit("c30 refit summary is not complete")
if int(summary.get("n_completed_folds", 0)) != 16:
    raise SystemExit("c30 refit does not contain 16 completed folds")
if summary.get("target_values_read") is not False:
    raise SystemExit("c30 refit violated the target seal")
PY

# Prevent a reconnecting interactive agent and the watcher from running the
# same direct-transfer batch simultaneously.
exec 9>"${WATCH_ROOT}/postprocess.lock"
if ! flock -n 9; then
  exit 0
fi

if [[ ! -f "${DIRECT_ROOT}/DIRECT_TRANSFER_SUMMARY.json" ]]; then
  bash scripts/run_topic5_history_rnn_direct_early_ictal_transfer_v0_2.sh \
    "${ARTIFACT_ROOT}" 4 "${REFIT_ROOT}" "${DIRECT_ROOT}"
fi

conda run -n cuda_env python \
  scripts/plot_topic5_history_rnn_direct_early_ictal_transfer_v0_2.py \
  --root "${DIRECT_ROOT}"

conda run -n cuda_env python \
  scripts/compare_topic5_history_rnn_direct_training_budget_v0_2.py \
  --short-root results/topic5_history_rnn_direct_c10_candidate_v0_2 \
  --long-root "${DIRECT_ROOT}" \
  --output-dir "${DIRECT_ROOT}/training_budget_comparison_c10_to_c30"

python - "${WATCH_ROOT}/WATCHER_DONE.json" <<'PY'
import json, pathlib, sys, time
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "status": "COMPLETE",
    "contract": "topic5_history_rnn_direct_c30_watcher_v0_2",
    "refit_folds": 16,
    "direct_folds": 16,
    "time_epoch": time.time(),
}, indent=2) + "\n")
PY
