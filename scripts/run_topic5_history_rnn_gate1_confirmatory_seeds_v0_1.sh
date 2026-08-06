#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${1:-/home/honglab/leijiaxin/HFOsp}"
BASE="results/topic5_history_rnn_early_ictal_field/g1_sequential_formal_v0_1"
SEED1="${BASE}/seed_20260725/G1_SUMMARY.json"
if [[ ! -f "${SEED1}" ]]; then
  echo "missing first-seed G1 summary" >&2
  exit 2
fi
readarray -t seed1_gate < <(python - "${SEED1}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(d["status"])
print(d["primary"]["median_chronological_increment"])
PY
)
status="${seed1_gate[0]}"
direction="${seed1_gate[1]}"
if ! python - "${direction}" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > 0 else 1)
PY
then
  echo "[locked] first-seed primary chronological direction was non-positive"
  python - "${BASE}/G1_MULTI_SEED_SUMMARY.json" "${status}" "${direction}" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "status": "G1_SEED1_NONPOSITIVE_KEEP_ICTAL_TARGET_SEALED",
    "seed1_status": sys.argv[2],
    "seed1_primary_median_chronological_increment": float(sys.argv[3]),
    "target_values_read": False,
    "n_seeds_run": 1,
    "next_action": "STOP_G2_G3",
}, indent=2) + "\n")
PY
  exit 0
fi

# Two fold processes per seed give four concurrent jobs.  The frozen model
# uses <1 GiB per process on the 24-GiB device; scientific settings are unchanged.
bash scripts/run_topic5_history_rnn_gate1_sequential_formal_v0_1.sh \
  "${ARTIFACT_ROOT}" 2 20260726 &
pid_a=$!
bash scripts/run_topic5_history_rnn_gate1_sequential_formal_v0_1.sh \
  "${ARTIFACT_ROOT}" 2 20260727 &
pid_b=$!
failed=0
wait "${pid_a}" || failed=1
wait "${pid_b}" || failed=1
if (( failed != 0 )); then
  echo "confirmatory seed execution failed" >&2
  exit 1
fi

conda run -n cuda_env python \
  scripts/summarize_topic5_history_rnn_gate1_multiseed_v0_1.py \
  --input-dir "${BASE}"
