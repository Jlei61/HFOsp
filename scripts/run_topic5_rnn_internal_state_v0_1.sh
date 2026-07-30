#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
out_root="${repo_root}/results/topic5_rnn_internal_state_reduction"
log_root="${out_root}/logs/cells"
mkdir -p "${log_root}"

force_arg=""
if [[ "${1:-}" == "--force" ]]; then
  force_arg="--force"
fi

"${python_bin}" - <<'PY' |
from pathlib import Path
import pandas as pd

root = Path("results/topic5_interictal_rank_distribution/dataset_v0_4")
frame = pd.read_csv(root / "subject_audit.csv")
subjects = sorted(frame.loc[frame.status.astype(str) == "ok", "subject"].astype(str))
if len(subjects) != 34:
    raise SystemExit(f"expected 34 subjects, found {len(subjects)}")
for seed_dir in ("seed_20260725", "seed_20260726", "seed_20260727"):
    for subject in subjects:
        print(seed_dir, subject)
PY
  xargs -P 12 -n 2 bash -c '
    seed_dir="$1"
    subject="$2"
    log_path="'"${log_root}"'/${seed_dir}__${subject}.log"
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      "'"${python_bin}"'" \
      "'"${repo_root}"'/scripts/extract_topic5_rnn_internal_states_v0_1.py" \
      --subject "${subject}" \
      --seed-dir "${seed_dir}" \
      --device cpu \
      --train-events 2048 \
      --validation-events 1024 \
      --heldout-events 2048 \
      --batch-size 256 \
      '"${force_arg}"' \
      > "${log_path}" 2>&1
  ' _

"${python_bin}" - <<'PY'
import json
from pathlib import Path

path = Path("results/topic5_rnn_internal_state_reduction/EXTRACTION_DONE.json")
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(".json.tmp")
temporary.write_text(
    json.dumps(
        {
            "status": "COMPLETE",
            "expected_cells": 102,
            "target_values_read": False,
        },
        indent=2,
    )
    + "\n"
)
temporary.replace(path)
PY
