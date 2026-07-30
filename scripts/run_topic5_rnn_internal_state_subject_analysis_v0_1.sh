#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
out_root="${repo_root}/results/topic5_rnn_internal_state_reduction"
log_root="${out_root}/logs/subject_analysis"
mkdir -p "${log_root}"

"${python_bin}" - <<'PY' |
from pathlib import Path
import pandas as pd

frame = pd.read_csv(
    Path("results/topic5_interictal_rank_distribution/dataset_v0_4")
    / "subject_audit.csv"
)
subjects = sorted(frame.loc[frame.status.astype(str) == "ok", "subject"].astype(str))
if len(subjects) != 34:
    raise SystemExit(f"expected 34 subjects, found {len(subjects)}")
for subject in subjects:
    print(subject)
PY
  xargs -P 16 -n 1 bash -c '
    subject="$1"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "'"${python_bin}"'" \
      "'"${repo_root}"'/scripts/analyze_topic5_rnn_internal_state_subject_v0_1.py" \
      --subject "${subject}" \
      > "'"${log_root}"'/${subject}.log" 2>&1
  ' _

"${python_bin}" - <<'PY'
import json
from pathlib import Path

path = Path(
    "results/topic5_rnn_internal_state_reduction/"
    "SUBJECT_ANALYSIS_DONE.json"
)
temporary = path.with_suffix(".json.tmp")
temporary.write_text(
    json.dumps(
        {
            "status": "COMPLETE",
            "expected_subjects": 34,
            "target_values_read": False,
        },
        indent=2,
    )
    + "\n"
)
temporary.replace(path)
PY
