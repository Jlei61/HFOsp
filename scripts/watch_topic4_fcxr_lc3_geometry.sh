#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <geometry-autopilot-pid>" >&2
  exit 2
fi

geometry_pid="$1"
result_root="results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability"
running="$result_root/GEOMETRY_RUNNING.json"
done_file="$result_root/GEOMETRY_DONE.json"
watch_done="$result_root/GEOMETRY_WATCHDOG_DONE.json"
watch_failed="$result_root/GEOMETRY_WATCHDOG_FAILED.json"
python_bin="/home/honglab/leijiaxin/anaconda3/bin/python"
max_wall_s=64800

mkdir -p "$result_root"
printf '%s\n' "$$" > "$result_root/geometry_watchdog.pid"

write_json() {
  local path="$1"
  local status="$2"
  local reason="$3"
  "$python_bin" - "$path" "$status" "$reason" <<'PY'
import datetime, json, os, sys
path, status, reason = sys.argv[1:]
tmp = path + f".{os.getpid()}.tmp"
with open(tmp, "w") as f:
    json.dump({"status": status, "reason": reason,
               "finished": datetime.datetime.now(datetime.timezone.utc).isoformat()},
              f, indent=2)
    f.flush(); os.fsync(f.fileno())
os.replace(tmp, path)
PY
}

# Wait for the map stage itself, not merely the already detached autopilot.  The
# preparation runs before it are individually shorter than 20 simulated seconds.
while [[ ! -f "$running" ]]; do
  if ! kill -0 "$geometry_pid" 2>/dev/null; then
    if [[ -f "$done_file" ]]; then
      write_json "$watch_done" DONE geometry_completed_before_watch
      exit 0
    fi
    write_json "$watch_failed" FAILED geometry_autopilot_ended_before_map
    exit 1
  fi
  sleep 30
done

start_epoch="$(date +%s)"
while kill -0 "$geometry_pid" 2>/dev/null; do
  if [[ -f "$done_file" ]]; then
    write_json "$watch_done" DONE geometry_completed
    exit 0
  fi
  if (( $(date +%s) - start_epoch > max_wall_s )); then
    # Resolve only descendants of the registered LC3 autopilot.  Send TERM from
    # leaves upward, then the parent; never use pgrep -f or touch sibling jobs.
    mapfile -t descendants < <("$python_bin" - "$geometry_pid" <<'PY'
import subprocess, sys
root = int(sys.argv[1])
rows = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True).splitlines()
children = {}
for row in rows:
    pid, ppid = map(int, row.split())
    children.setdefault(ppid, []).append(pid)
out = []
def walk(pid):
    for child in children.get(pid, []):
        walk(child)
        out.append(child)
walk(root)
print("\n".join(map(str, out)))
PY
    )
    for pid in "${descendants[@]}"; do
      kill -TERM "$pid" 2>/dev/null || true
    done
    kill -TERM "$geometry_pid" 2>/dev/null || true
    write_json "$watch_failed" FAILED geometry_wall_guard_64800s
    exit 1
  fi
  sleep 30
done

if [[ -f "$done_file" ]]; then
  write_json "$watch_done" DONE geometry_completed
  exit 0
fi
write_json "$watch_failed" FAILED geometry_autopilot_ended_without_done
exit 1
