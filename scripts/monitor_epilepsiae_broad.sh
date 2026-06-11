#!/bin/bash
# Detached monitor for the 5 Epilepsiae broad re-pack strategies.
# Appends a progress snapshot every 30 min to results/lagpat_broad_epilepsiae_MONITOR.log
# Survives SSH/network drops (run via nohup, no session dependency).
# Stops automatically once all 5 strategy processes have exited.
cd /home/honglab/leijiaxin/HFOsp || exit 1
LOG=results/lagpat_broad_epilepsiae_MONITOR.log
DIRS=(lagpat_broad_epilepsiae lagpat_broad_epilepsiae_topn40 lagpat_broad_epilepsiae_k05 lagpat_broad_epilepsiae_k00 lagpat_broad_epilepsiae_km10)

while true; do
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  nproc=$(ps aux | grep "broad_lagpat_repack_epilepsiae" | grep -v grep | wc -l)
  {
    echo "===== $ts | live processes: $nproc / 5 ====="
    for d in "${DIRS[@]}"; do
      [ -d "results/$d" ] || continue
      # count subjects whose log line reports a completed subject (done=N/N)
      done_n=$(grep -cE "subject=.* done=[0-9]+/[0-9]+ " "results/$d/repack.log" 2>/dev/null)
      cur=$(grep -E "=== epilepsiae" "results/$d/repack.log" 2>/dev/null | tail -1 | grep -oE "(re-pack: |: )[0-9]+" | grep -oE "[0-9]+" | tail -1)
      nrec=$(grep -E "subject=.* done=" "results/$d/repack.log" 2>/dev/null | tail -1)
      printf "  %-32s done_subjects=%-3s  now=%-6s\n" "$d" "${done_n:-0}" "${cur:-?}"
    done
  } >> "$LOG"
  [ "$nproc" -eq 0 ] && { echo "===== all 5 strategies finished at $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG"; break; }
  sleep 1800
done
