#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
primary_root="${repo_root}/results/topic5_interictal_scaffold_reliability_history_necessity/history_runs_v0_1"
confirm_root="${repo_root}/results/topic5_interictal_scaffold_reliability_history_necessity/history3_rank_shuffle_runs_v0_1"
mkdir -p "${confirm_root}"

while true; do
  if [[ -f "${primary_root}/DONE.json" ]] && \
     python - "${primary_root}/DONE.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
raise SystemExit(0 if value.get("status") == "complete" else 1)
PY
  then
    break
  fi
  if ! tmux has-session -t topic5_history_v01 2>/dev/null; then
    echo "Primary tmux ended without a complete DONE.json" >&2
    exit 5
  fi
  sleep 30
done

bash "${repo_root}/scripts/run_topic5_history3_rank_shuffle_formal_v0_1.sh" \
  > "${confirm_root}/launcher.log" 2>&1

conda run --no-capture-output -n cuda_env \
  python "${repo_root}/scripts/paper_figures/plot_topic5_scaffold_reliability_history_necessity_v0_1.py" \
  > "${confirm_root}/figure.log" 2>&1
