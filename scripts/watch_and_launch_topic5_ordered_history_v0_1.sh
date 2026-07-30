#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
formal_root="${repo_root}/results/topic5_ordered_history_architecture_audit/formal/architecture_controls_formal_20260729"
analysis_root="${repo_root}/results/topic5_ordered_history_architecture_audit/analysis"
shuffle_tag="selected_architecture_rank_shuffle_20260729"
shuffle_root="${repo_root}/results/topic5_ordered_history_architecture_audit/rank_shuffle/${shuffle_tag}"
parameter_tag="parameter_matched_formal_20260729"
parameter_root="${repo_root}/results/topic5_ordered_history_architecture_audit/parameter_matched/${parameter_tag}"
status_root="${repo_root}/results/topic5_ordered_history_architecture_audit/watcher"
mkdir -p "${status_root}"
python - "${status_root}" "$$" <<'PY'
import sys
from pathlib import Path
Path(sys.argv[1], "PID").write_text(sys.argv[2] + "\n")
PY

while [[ ! -f "${formal_root}/LAUNCHER_DONE.json" ]]; do
  sleep 30
done
formal_status="$(
  python - "${formal_root}/LAUNCHER_DONE.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["status"])
PY
)"
if [[ "${formal_status}" != "COMPLETE" ]]; then
  echo "formal architecture launcher did not complete" >&2
  exit 3
fi

conda run --no-capture-output -n cuda_env \
  python "${repo_root}/scripts/summarize_topic5_ordered_history_architecture_v0_1.py" \
    --formal-root "${formal_root}" \
    --output "${analysis_root}" \
  > "${status_root}/architecture_summary.log" 2>&1
selected="$(
  python - "${analysis_root}/ARCHITECTURE_SUMMARY.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["target_blind_best_non_gru"]["control"])
PY
)"

bash "${repo_root}/scripts/run_topic5_selected_architecture_shuffle_v0_1.sh" \
  "${selected}" "${shuffle_tag}" \
  > "${status_root}/selected_shuffle_launcher.log" 2>&1

conda run --no-capture-output -n cuda_env \
  python "${repo_root}/scripts/summarize_topic5_ordered_history_architecture_v0_1.py" \
    --formal-root "${formal_root}" \
    --shuffle-root "${shuffle_root}" \
    --output "${analysis_root}" \
  > "${status_root}/architecture_with_shuffle_summary.log" 2>&1

bash "${repo_root}/scripts/run_topic5_parameter_matched_architecture_sensitivity_v0_1.sh" \
  "${parameter_tag}" \
  > "${status_root}/parameter_matched_launcher.log" 2>&1

conda run --no-capture-output -n cuda_env \
  python "${repo_root}/scripts/summarize_topic5_parameter_matched_architecture_v0_1.py" \
    --root "${parameter_root}" \
    --output "${analysis_root}" \
  > "${status_root}/parameter_matched_summary.log" 2>&1

bash "${repo_root}/scripts/run_topic5_selected_history_interventions_v0_1.sh" \
  "selected_history_interventions_20260729" \
  > "${status_root}/selected_history_interventions.log" 2>&1

conda run --no-capture-output -n cuda_env \
  python "${repo_root}/scripts/summarize_topic5_selected_history_interventions_v0_1.py" \
    --root "${repo_root}/results/topic5_ordered_history_architecture_audit/interventions/selected_history_interventions_20260729" \
    --output "${analysis_root}" \
  > "${status_root}/history_intervention_summary.log" 2>&1

conda run --no-capture-output -n cuda_env \
  python "${repo_root}/scripts/analyze_topic5_ordered_history_early_ictal_increment_v0_1.py" \
    --intervention-root "${repo_root}/results/topic5_ordered_history_architecture_audit/interventions/selected_history_interventions_20260729" \
    --output "${analysis_root}" \
  > "${status_root}/early_ictal_conditional_summary.log" 2>&1

conda run --no-capture-output -n cuda_env \
  python "${repo_root}/scripts/paper_figures/plot_topic5_ordered_history_architecture_audit_v0_1.py" \
  > "${status_root}/paper_figure.log" 2>&1

conda run --no-capture-output -n cuda_env \
  python "${repo_root}/scripts/build_topic5_ordered_history_final_acceptance_v0_1.py" \
  > "${status_root}/final_acceptance.log" 2>&1

python - "${status_root}" "${selected}" <<'PY'
import json,sys
from pathlib import Path
root = Path(sys.argv[1])
(root / "WATCHER_DONE.json").write_text(
    json.dumps(
        {
            "status": "FULL_ORDERED_HISTORY_AUDIT_COMPLETE",
            "selected_architecture": sys.argv[2],
            "parameter_matched_sensitivity_complete_before_target_read": True,
        },
        indent=2,
    )
    + "\n"
)
PY
