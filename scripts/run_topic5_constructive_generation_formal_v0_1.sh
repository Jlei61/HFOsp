#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/results/topic5_ordered_history_architecture_audit/formal/architecture_controls_formal_20260729/linear_state}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/topic5_constructive_event_generation/formal_v0_1}"
MAX_WORKERS="${MAX_WORKERS:-12}"
DEVICE="${DEVICE:-cuda}"
CPU_THREADS="${CPU_THREADS:-2}"
GPU_MEMORY_FRACTION="${GPU_MEMORY_FRACTION:-0.06}"
LOG_ROOT="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_ROOT}"
manifest="${OUTPUT_ROOT}/launch_manifest.tsv"
printf "seed\tsubject\tcheckpoint\toutput\tlog\n" > "${manifest}"

run_cell() {
    local checkpoint="$1"
    local seed="$2"
    local subject="$3"
    local output="$4"
    local log="$5"

    if [[ -f "${output}/run_summary.json" ]] && \
       python - "${output}/run_summary.json" <<'PY'
import json
import sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["status"] == "COMPLETE" else 1)
PY
    then
        return 0
    fi
    if [[ -e "${output}" ]]; then
        mv "${output}" "${output}.incomplete.$(date +%Y%m%d_%H%M%S)"
    fi
    conda run -n cuda_env python \
        "${ROOT}/scripts/run_topic5_constructive_generation_cell_v0_1.py" \
        --checkpoint "${checkpoint}" \
        --subject "${subject}" \
        --seed "${seed}" \
        --out-dir "${output}" \
        --device "${DEVICE}" \
        --cpu-threads "${CPU_THREADS}" \
        --gpu-memory-fraction "${GPU_MEMORY_FRACTION}" \
        > "${log}" 2>&1
}
export -f run_cell
export ROOT DEVICE CPU_THREADS GPU_MEMORY_FRACTION

while IFS= read -r checkpoint; do
    seed_dir="$(basename "$(dirname "$(dirname "${checkpoint}")")")"
    seed="${seed_dir#seed_}"
    subject="$(basename "$(dirname "${checkpoint}")")"
    output="${OUTPUT_ROOT}/seed_${seed}/${subject}"
    log="${LOG_ROOT}/seed_${seed}__${subject}.log"
    printf "%s\t%s\t%s\t%s\t%s\n" \
        "${seed}" "${subject}" "${checkpoint}" "${output}" "${log}" >> "${manifest}"
done < <(find "${CHECKPOINT_ROOT}" -name linear_state_checkpoint.pt | sort)

tail -n +2 "${manifest}" |
    xargs -P "${MAX_WORKERS}" -n 5 bash -c \
    'run_cell "$2" "$0" "$1" "$3" "$4"'

python - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summaries = list(root.glob("seed_*/*/run_summary.json"))
complete = []
for path in summaries:
    try:
        complete.append(json.loads(path.read_text())["status"] == "COMPLETE")
    except Exception:
        complete.append(False)
state = {
    "status": "COMPLETE" if len(summaries) == 102 and all(complete) else "INCOMPLETE",
    "expected_cells": 102,
    "found_summaries": len(summaries),
    "complete_cells": int(sum(complete)),
}
(root / "formal_run_state.json").write_text(json.dumps(state, indent=2) + "\n")
print(json.dumps(state))
raise SystemExit(0 if state["status"] == "COMPLETE" else 1)
PY
