#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="${repo_root}/config/topic5_shared_scaffold_propagation_rnn_v0_2.yaml"
python_bin="/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
output_root="${repo_root}/results/topic5_patient_specific_shared_scaffold_rnn_v0_2"
monitor_root="${output_root}/monitor"
mkdir -p "${monitor_root}"

stage="${1:-status}"
workers="${2:-12}"

case "${stage}" in
  audit)
    exec "${python_bin}" "${repo_root}/scripts/audit_topic5_shared_scaffold_inputs_v0_2.py" --config "${config_path}"
    ;;
  smoke)
    "${python_bin}" "${repo_root}/scripts/audit_topic5_shared_scaffold_inputs_v0_2.py" --config "${config_path}"
    nohup "${python_bin}" "${repo_root}/scripts/watch_topic5_shared_scaffold_rnn_v0_2.py" \
      --config "${config_path}" --smoke --watch \
      >"${monitor_root}/smoke_watcher.nohup.log" 2>&1 &
    echo "$!" >"${monitor_root}/smoke_watcher.pid"
    nohup "${python_bin}" "${repo_root}/scripts/launch_topic5_shared_scaffold_rnn_v0_2.py" \
      --config "${config_path}" --smoke --resume --workers "${workers}" \
      >"${monitor_root}/smoke_launcher.nohup.log" 2>&1 &
    echo "$!" >"${monitor_root}/smoke_launcher.pid"
    echo "smoke launcher pid=$(<"${monitor_root}/smoke_launcher.pid") watcher pid=$(<"${monitor_root}/smoke_watcher.pid")"
    ;;
  formal)
    "${python_bin}" "${repo_root}/scripts/audit_topic5_shared_scaffold_inputs_v0_2.py" --config "${config_path}"
    nohup "${python_bin}" "${repo_root}/scripts/watch_topic5_shared_scaffold_rnn_v0_2.py" \
      --config "${config_path}" --watch \
      >"${monitor_root}/watcher.nohup.log" 2>&1 &
    echo "$!" >"${monitor_root}/watcher.pid"
    nohup "${python_bin}" "${repo_root}/scripts/launch_topic5_shared_scaffold_rnn_v0_2.py" \
      --config "${config_path}" --resume --workers "${workers}" \
      >"${monitor_root}/launcher.nohup.log" 2>&1 &
    echo "$!" >"${monitor_root}/launcher.pid"
    echo "formal launcher pid=$(<"${monitor_root}/launcher.pid") watcher pid=$(<"${monitor_root}/watcher.pid")"
    ;;
  rollout)
    nohup "${python_bin}" "${repo_root}/scripts/launch_topic5_shared_scaffold_rollouts_v0_2.py" \
      --config "${config_path}" --resume --workers "${workers}" \
      >"${monitor_root}/rollout_launcher.nohup.log" 2>&1 &
    echo "$!" >"${monitor_root}/rollout_launcher.pid"
    echo "rollout launcher pid=$(<"${monitor_root}/rollout_launcher.pid")"
    ;;
  watch)
    exec "${python_bin}" "${repo_root}/scripts/watch_topic5_shared_scaffold_rnn_v0_2.py" \
      --config "${config_path}" --watch
    ;;
  status)
    exec "${python_bin}" "${repo_root}/scripts/watch_topic5_shared_scaffold_rnn_v0_2.py" \
      --config "${config_path}"
    ;;
  *)
    echo "usage: $0 {audit|smoke|formal|rollout|watch|status} [workers]" >&2
    exit 2
    ;;
esac
