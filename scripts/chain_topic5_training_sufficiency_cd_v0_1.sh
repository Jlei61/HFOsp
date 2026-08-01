#!/usr/bin/env bash
# Unattended tail of the Topic 5 sufficiency pipeline: B3 -> B1c -> freeze ->
# Phase C -> freeze -> Phase D -> machine acceptance.  Every stage is resumable,
# so re-running this script after a dropped shell continues where it stopped.
#
#   chain_topic5_training_sufficiency_cd_v0_1.sh [workers] [gpu_fraction]
#
# Keep ``workers * gpu_fraction`` below 1.0 so that even the worst case, where
# every process reaches its cap at once, cannot exhaust the device.
set -uo pipefail
cd /home/honglab/leijiaxin/HFOsp

workers="${1:-14}"
fraction="${2:-0.06}"
root=results/topic5_rnn_training_sufficiency_v0_1
pipeline=scripts/run_topic5_training_sufficiency_pipeline_v0_1.sh
export TOPIC5_GPU_FRACTION="${fraction}"

if ! python - "$workers" "$fraction" <<'PY'
import sys
workers, fraction = int(sys.argv[1]), float(sys.argv[2])
if workers * fraction >= 0.95:
    raise SystemExit(
        f"refusing to launch: {workers} workers x {fraction} would oversubscribe"
    )
print(f"launch guard ok: {workers} workers, cap {fraction * 24576:.0f} MiB each")
PY
then
  exit 1
fi

wait_for() {  # an upstream stage may still be running in another shell
  local marker="$1" label="$2" limit="${3:-720}" waited=0
  until [ -f "$marker" ]; do
    waited=$((waited + 1))
    if [ "$waited" -gt "$limit" ]; then
      echo "timed out waiting for ${label}"; return 1
    fi
    sleep 60
  done
  echo "${label} ready at $(date -Is)"
}

stage() {
  local name="$1"
  echo "=== ${name} starting at $(date -Is) with ${workers} workers ==="
  if ! bash "${pipeline}" "${name}" "${workers}"; then
    echo "stage ${name} failed at $(date -Is)"
    return 1
  fi
  echo "=== ${name} complete at $(date -Is) ==="
}

wait_for "${root}/analysis/b2_selection.json" "B2 selection" || exit 1
stage b3 || exit 1
stage b1c || exit 1
wait_for "${root}/development/HYPERPARAMETER_FREEZE.json" "hyperparameter freeze" 10 || exit 1
stage c || exit 1
wait_for "${root}/development/OBJECTIVE_FREEZE.json" "objective freeze" 10 || exit 1
stage d || exit 1
conda run --no-capture-output -n cuda_env python \
  scripts/build_topic5_training_sufficiency_acceptance_v0_1.py
echo "pipeline finished at $(date -Is)"
