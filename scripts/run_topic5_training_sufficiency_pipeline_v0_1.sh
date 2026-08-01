#!/usr/bin/env bash
# Unattended, resumable Topic 5 sufficiency pipeline.
#
#   run_topic5_training_sufficiency_pipeline_v0_1.sh <stage> [workers]
#
# Stages run in order and each one is idempotent: a manifest cell that already
# carries DONE.json is skipped, so re-running after a dropped shell continues
# rather than restarting.  The frozen training budget is read from the Phase B
# selection files, never typed twice.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
stage="${1:?usage: run_topic5_training_sufficiency_pipeline_v0_1.sh <b2|b3|b1c|c|d> [workers]}"
workers="${2:-6}"
# per-process GPU cap; the chain script sets this so that
# workers x fraction stays below the physical device
gpu_fraction="${TOPIC5_GPU_FRACTION:-0.06}"
root="results/topic5_rnn_training_sufficiency_v0_1"
manifests="${root}/development/manifests"
logs="${root}/logs"
mkdir -p "${manifests}" "${logs}"

run() { conda run --no-capture-output -n cuda_env python "$@"; }

# cycles, updates-per-patient and hidden size come from the budget sweep; the
# learning rate, optimizer and weight decay come from B2 once it has run, so
# every downstream LOSO phase uses the frozen configuration rather than the
# stale published learning rate.
budget() {
  run - <<'PY'
import json, pathlib
root = pathlib.Path("results/topic5_rnn_training_sufficiency_v0_1/analysis")
extended = root / "b1x_selection.json"
selection = json.loads((extended if extended.is_file() else root / "b1_selection.json").read_text())
chosen = selection["selected"]
learning_rate, optimizer, weight_decay = 1e-3, "adamw", 1e-4
b2 = root / "b2_selection.json"
if b2.is_file():
    tuned = json.loads(b2.read_text())["selected"]
    learning_rate = tuned["learning_rate"]
    optimizer = tuned["optimizer"]
    weight_decay = tuned["weight_decay"]
print(
    f"{chosen['coverage_cycle']} {chosen['updates_per_patient']} "
    f"{chosen['hidden_size']} {learning_rate:g} {optimizer} {weight_decay:g}"
)
PY
}

case "${stage}" in
  b2)
    read -r cycles updates hidden lr opt wd < <(budget)
    run scripts/build_topic5_training_sufficiency_manifest_v0_1.py \
      --phase b2 --out "${manifests}/b2.json" --cycles "${cycles}" --updates "${updates}"
    run scripts/launch_topic5_training_sufficiency_v0_1.py \
      --manifest "${manifests}/b2.json" --workers "${workers}" --cpu-threads 3 \
      > "${logs}/b2_launcher.log" 2>&1
    run scripts/analyze_topic5_training_sufficiency_development_v0_1.py \
      --phase b2 --root "${root}/development/b2_learning_rate" > "${logs}/b2_analysis.log" 2>&1
    ;;
  b3)
    read -r cycles updates hidden lr opt wd < <(budget)
    run scripts/build_topic5_training_sufficiency_manifest_v0_1.py \
      --phase b3 --out "${manifests}/b3.json" \
      --learning-rate "${lr}" --optimizer "${opt}" --weight-decay "${wd}" --cycles "${cycles}" --updates "${updates}"
    run scripts/launch_topic5_training_sufficiency_v0_1.py \
      --manifest "${manifests}/b3.json" --workers 2 --cpu-threads 4 \
      > "${logs}/b3_launcher.log" 2>&1
    run scripts/analyze_topic5_training_sufficiency_development_v0_1.py \
      --phase b3 --root "${root}/development/b3_chunk_parity" > "${logs}/b3_analysis.log" 2>&1
    ;;
  b1c)
    read -r cycles updates hidden lr opt wd < <(budget)
    # three arms: the selected budget, an intermediate budget that guards
    # against the selected one overfitting the 33 training patients, and the
    # published budget, which is the contrast the freeze actually turns on
    intermediate=$(( cycles > 3 ? (cycles + 1) / 2 : cycles ))
    run scripts/build_topic5_training_sufficiency_manifest_v0_1.py \
      --phase b1c --out "${manifests}/b1c.json" \
      --gpu-memory-fraction "${gpu_fraction}" \
      --learning-rate "${lr}" --optimizer "${opt}" --weight-decay "${wd}" \
      --budget "${cycles}:${updates}" "${intermediate}:${updates}" "1:8"
    run scripts/launch_topic5_training_sufficiency_v0_1.py \
      --manifest "${manifests}/b1c.json" --workers "${workers}" --cpu-threads 2 --retry-once \
      > "${logs}/b1c_launcher.log" 2>&1
    run scripts/analyze_topic5_training_sufficiency_loso_v0_1.py \
      --phase b1c --root "${root}/development/b1c_loso_confirm" > "${logs}/b1c_analysis.log" 2>&1
    run scripts/freeze_topic5_training_sufficiency_v0_1.py --kind hyperparameters
    ;;
  c)
    read -r cycles updates hidden lr opt wd < <(budget)
    offset=$(run - <<'PY'
import json, pathlib
frozen = json.loads(
    pathlib.Path(
        "results/topic5_rnn_training_sufficiency_v0_1/development/HYPERPARAMETER_FREEZE.json"
    ).read_text()
)
print(frozen["selected"]["heldout_offset_calibration_cycles"])
PY
)
    run scripts/build_topic5_training_sufficiency_manifest_v0_1.py \
      --phase c --out "${manifests}/c.json" \
      --gpu-memory-fraction "${gpu_fraction}" \
      --learning-rate "${lr}" --optimizer "${opt}" --weight-decay "${wd}" \
      --cycles "${cycles}" --updates "${updates}" --offset-cycles "${offset}"
    run scripts/launch_topic5_training_sufficiency_v0_1.py \
      --manifest "${manifests}/c.json" --workers "${workers}" --cpu-threads 2 --retry-once \
      > "${logs}/c_launcher.log" 2>&1
    run scripts/analyze_topic5_training_sufficiency_loso_v0_1.py \
      --phase c --root "${root}/development/c_objectives" > "${logs}/c_analysis.log" 2>&1
    run scripts/freeze_topic5_training_sufficiency_v0_1.py --kind objective
    ;;
  d)
    read -r cycles updates hidden lr opt wd < <(budget)
    read -r offset objective < <(run - <<'PY'
import json, pathlib
root = pathlib.Path("results/topic5_rnn_training_sufficiency_v0_1/development")
frozen = json.loads((root / "HYPERPARAMETER_FREEZE.json").read_text())
objective = json.loads((root / "OBJECTIVE_FREEZE.json").read_text())
print(
    frozen["selected"]["heldout_offset_calibration_cycles"],
    objective["selected_rollout_aware_objective"],
)
PY
)
    run scripts/build_topic5_training_sufficiency_manifest_v0_1.py \
      --phase d --out "${manifests}/d.json" \
      --gpu-memory-fraction "${gpu_fraction}" \
      --learning-rate "${lr}" --optimizer "${opt}" --weight-decay "${wd}" \
      --cycles "${cycles}" --updates "${updates}" --offset-cycles "${offset}" \
      --objective "${objective}"
    run scripts/launch_topic5_training_sufficiency_v0_1.py \
      --manifest "${manifests}/d.json" --workers "${workers}" --cpu-threads 2 --retry-once \
      > "${logs}/d_launcher.log" 2>&1
    run scripts/analyze_topic5_training_sufficiency_loso_v0_1.py \
      --phase d --root "${root}/formal" > "${logs}/d_analysis.log" 2>&1
    ;;
  *)
    echo "unknown stage: ${stage}" >&2
    exit 2
    ;;
esac

echo "{\"stage\": \"${stage}\", \"status\": \"COMPLETE\"}"
