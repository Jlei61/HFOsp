#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root="${1:-${repo_root}/results/topic5_structured_axis_graph/formal_persistent_path_mode_v1_0}"
launcher_pid="${2:-4091866}"
config="${3:-${repo_root}/config/topic5_persistent_path_mode_rnn_v1_0.yaml}"
state="${run_root}/POSTPROCESS_STATE.json"

write_state() {
  local status="$1"
  local stage="$2"
  local message="$3"
  python - "${state}" "${status}" "${stage}" "${message}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "message": sys.argv[4],
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}
path.write_text(json.dumps(payload, indent=2))
PY
}

write_state "WAITING" "formal_training" "waiting for formal cohort analyzer"
while true; do
  formal_complete="false"
  if [[ -f "${run_root}/analysis/formal_gate_summary.json" ]] \
    && [[ -f "${run_root}/RUN_MANIFEST.json" ]]; then
    formal_complete="$(
      python - "${run_root}/RUN_MANIFEST.json" <<'PY'
import json
import sys
print("true" if json.load(open(sys.argv[1])).get("status") == "COMPLETE" else "false")
PY
    )"
  fi
  if [[ "${formal_complete}" == "true" ]]; then
    break
  fi
  if ! kill -0 "${launcher_pid}" 2>/dev/null; then
    complete_runs="$(
      find "${run_root}" -name run_state.json -print0 \
        | xargs -0 -r grep -l '"status": "COMPLETE"' \
        | wc -l
    )"
    if [[ "${complete_runs}" -eq 510 ]]; then
      write_state "RUNNING" "formal_analysis" \
        "all 510 runs complete; recovering analyzer after launcher exit"
      conda run --no-capture-output -n cuda_env \
        python "${repo_root}/scripts/analyze_topic5_persistent_path_formal.py" \
          --root "${run_root}" \
          > "${run_root}/analysis.log" 2>&1
      continue
    fi
    write_state "FAILED" "formal_training" \
      "formal launcher exited with only ${complete_runs}/510 complete runs"
    exit 5
  fi
  sleep 30
done

write_state "RUNNING" "internal_dynamics" \
  "summarizing target-sealed recurrent state trajectories"
conda run --no-capture-output -n cuda_env \
  python "${repo_root}/scripts/analyze_topic5_persistent_path_internal_dynamics.py" \
    --root "${run_root}" \
    --config "${config}" \
    --max-events-per-run 512 \
    --batch-size 64

gate_pass="$(
  python - "${run_root}/analysis/formal_gate_summary.json" <<'PY'
import json
import sys
print(
    "true"
    if json.load(open(sys.argv[1]))["formal_interictal_gate_pass"]
    else "false"
)
PY
)"

if [[ "${gate_pass}" == "true" ]]; then
  write_state "RUNNING" "cross_state_features" \
    "building frozen node-rank probability fields"
  conda run --no-capture-output -n cuda_env \
    python "${repo_root}/scripts/build_topic5_persistent_path_cross_state_features.py" \
      --root "${run_root}" \
      --config "${config}" \
      --device cuda:0

  write_state "RUNNING" "clinical_onset_readout" \
    "reading the preregistered clinical-onset BB150 static target"
  conda run --no-capture-output -n cuda_env \
    python "${repo_root}/scripts/run_topic5_persistent_path_frozen_ictal_readout.py" \
      --root "${run_root}" \
      --n-perm 5000 \
      --seed 20260726
else
  write_state "RUNNING" "bounded_negative" \
    "interictal gate failed; ictal target remains sealed"
fi

write_state "RUNNING" "paper_figure" \
  "rendering the six-panel paper-ready figure"
conda run --no-capture-output -n cuda_env \
  python "${repo_root}/scripts/paper_figures/plot_fig6_structured_rank_rnn.py" \
    --root "${run_root}"

write_state "COMPLETE" "complete" \
  "formal analysis, internal dynamics, conditional target gate, and figure finished"
