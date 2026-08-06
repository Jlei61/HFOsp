#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${1:-/home/honglab/leijiaxin/HFOsp}"
WORKERS="${2:-2}"
SEED="${3:-20260725}"
DEVELOPMENT_ROOT="results/topic5_history_rnn_early_ictal_field/g1_sequential_development_selection_v0_1"
SELECTION="${DEVELOPMENT_ROOT}/DEVELOPMENT_SELECTION.json"
OUTPUT_ROOT="results/topic5_history_rnn_early_ictal_field/g1_sequential_formal_v0_1/seed_${SEED}"

if [[ ! -f "${SELECTION}" ]]; then
  echo "missing frozen development selection: ${SELECTION}" >&2
  exit 2
fi

readarray -t selected < <(python - "${SELECTION}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
name = d["selected_configuration"]
mapping = {
    "h32_half2_lr3e4": (32, 2.0, "3e-4", 3, 256),
    "h32_half0p5_lr3e4": (32, 0.5, "3e-4", 3, 256),
    "h32_half2_lr1e3": (32, 2.0, "1e-3", 3, 256),
    "h16_half2_lr3e4_c3_k256": (16, 2.0, "3e-4", 3, 256),
    "h32_half6_lr3e4_c3_k256": (32, 6.0, "3e-4", 3, 256),
    "h32_half2_lr3e4_c3_k128": (32, 2.0, "3e-4", 3, 128),
    "h32_half2_lr3e4_c6_k256": (32, 2.0, "3e-4", 6, 256),
}
if name not in mapping:
    raise SystemExit(f"unknown frozen configuration: {name}")
print(name)
print(mapping[name][0])
print(mapping[name][1])
print(mapping[name][2])
print(mapping[name][3])
print(mapping[name][4])
PY
)
CONFIG_NAME="${selected[0]}"
HISTORY_DIM="${selected[1]}"
HALF_LIFE="${selected[2]}"
LEARNING_RATE="${selected[3]}"
CYCLES="${selected[4]}"
BPTT_CHUNK="${selected[5]}"

mkdir -p "${OUTPUT_ROOT}/logs"
# Write the manifest without reading any early-ictal target values.
python - "${OUTPUT_ROOT}/RUN_MANIFEST.json" "${SELECTION}" "${CONFIG_NAME}" "${WORKERS}" "${SEED}" <<'PY'
import hashlib, json, pathlib, sys, time
out = pathlib.Path(sys.argv[1])
selection = pathlib.Path(sys.argv[2])
payload = {
    "contract": "topic5_history_rnn_early_ictal_field_v0_1_g1_formal",
    "created_epoch": time.time(),
    "selected_configuration": sys.argv[3],
    "workers": int(sys.argv[4]),
    "seed": int(sys.argv[5]),
    "development_selection": str(selection.resolve()),
    "development_selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
    "target_values_read": False,
}
out.write_text(json.dumps(payload, indent=2) + "\n")
PY

readarray -t subjects < <(python - "${ARTIFACT_ROOT}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = root / "results/topic5_interictal_rank_distribution/dataset_v0_4/dataset_manifest.json"
for subject in json.loads(manifest.read_text())["cohort_subjects"]:
    print(subject)
PY
)

run_one() {
  local subject="$1"
  local output="${OUTPUT_ROOT}/${subject}"
  local log="${OUTPUT_ROOT}/logs/${subject}.log"
  local failed="${OUTPUT_ROOT}/${subject}.FAILED.json"
  if [[ -f "${output}/DONE.json" && -f "${output}/ORDER_CONTROLS.json" ]]; then
    echo "[skip] ${subject}"
    return 0
  fi
  if [[ -d "${output}" && ! -f "${output}/DONE.json" ]]; then
    local archived="${output}.incomplete.$(date +%Y%m%dT%H%M%S)"
    mv "${output}" "${archived}"
    echo "[archive-incomplete] ${subject} -> ${archived}"
  fi
  if [[ ! -f "${output}/DONE.json" ]]; then
    echo "[start-train] ${subject}"
    if conda run -n cuda_env python \
      scripts/run_topic5_history_rnn_gate1_sequential_fold_v0_1.py \
      --heldout-subject "${subject}" \
      --seed "${SEED}" \
      --artifact-root "${ARTIFACT_ROOT}" \
      --output-dir "${output}" \
      --device cuda:0 \
      --embedding-batch-size 8192 \
      --history-dim "${HISTORY_DIM}" \
      --initial-half-life-hours "${HALF_LIFE}" \
      --matched-cycles "${CYCLES}" \
      --history-cycles "${CYCLES}" \
      --segment-batch-size 16 \
      --bptt-chunk "${BPTT_CHUNK}" \
      --learning-rate "${LEARNING_RATE}" \
      >"${log}" 2>&1; then
      :
    else
      status=$?
      python - "${failed}" "${subject}" "${status}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "status": "FAILED",
    "subject": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "time_epoch": time.time(),
}, indent=2) + "\n")
PY
      echo "[failed-train] ${subject} exit=${status}" >&2
      return "${status}"
    fi
  fi
  echo "[start-controls] ${subject}"
  if conda run -n cuda_env python \
    scripts/audit_topic5_history_rnn_gate1_order_controls_v0_1.py \
    --fold-dir "${output}" \
    --artifact-root "${ARTIFACT_ROOT}" \
    --window 64 \
    --batch-size 256 \
    >>"${log}" 2>&1; then
    :
  else
    status=$?
    python - "${failed}" "${subject}" "${status}" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "status": "FAILED",
    "stage": "order_controls",
    "subject": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "time_epoch": time.time(),
}, indent=2) + "\n")
PY
    echo "[failed-controls] ${subject} exit=${status}" >&2
    return "${status}"
  fi
  rm -f "${failed}"
  echo "[done] ${subject}"
}

active=0
failed=0
for subject in "${subjects[@]}"; do
  run_one "${subject}" &
  active=$((active + 1))
  if (( active >= WORKERS )); then
    if ! wait -n; then failed=1; fi
    active=$((active - 1))
  fi
done
while (( active > 0 )); do
  if ! wait -n; then failed=1; fi
  active=$((active - 1))
done

if (( failed != 0 )); then
  echo "one or more formal G1 folds failed; see *.FAILED.json" >&2
  exit 1
fi

conda run -n cuda_env python \
  scripts/summarize_topic5_history_rnn_gate1_sequential_formal_v0_1.py \
  --input-dir "${OUTPUT_ROOT}"
