#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python3.11"
RESULT="$ROOT/results/topic5_history_conditioned_field_refinement_v0_4"
STAGE="${1:-status}"
WORKERS="${2:-8}"
mkdir -p "$RESULT/logs" "$RESULT/watchers"

subjects() {
  "$PY" - "$RESULT/INPUT_MANIFEST.json" <<'PY'
import json, sys
for value in json.load(open(sys.argv[1]))["cohort"]["primary_subjects"]:
    print(value)
PY
}

case "$STAGE" in
  cache)
    subjects | xargs -P "$WORKERS" -I{} bash -c '
      subject="$1"; root="$2"; py="$3"; result="$4"
      "$py" "$root/scripts/build_topic5_history_conditioned_field_cache_v0_4.py" \
        --heldout-subject "$subject" --device cuda:0 \
        >"$result/logs/cache_${subject}.log" 2>&1
    ' _ {} "$ROOT" "$PY" "$RESULT"
    ;;
  train)
    units="$RESULT/formal_units.tsv"
    : > "$units"
    while read -r subject; do
      for seed in 11 29 47; do
        printf '%s\t%s\n' "$subject" "$seed" >> "$units"
      done
    done < <(subjects)
    export ROOT PY RESULT
    xargs -P "$WORKERS" -n 2 bash -c '
      subject="$1"; seed="$2"
      log="$RESULT/logs/train_${subject}_seed${seed}.log"
      if "$PY" "$ROOT/scripts/run_topic5_history_conditioned_field_fold_v0_4.py" \
          --heldout-subject "$subject" --seed "$seed" --device cuda:0 \
          >"$log" 2>&1; then
        exit 0
      fi
      out="$RESULT/per_subject/seed_${seed}/${subject}"
      mkdir -p "$out"
      "$PY" - "$out/FAILED.json" "$subject" "$seed" "$log" <<'PY'
import json, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "status":"FAILED", "subject":sys.argv[2], "seed":int(sys.argv[3]),
    "log":sys.argv[4]}, indent=2)+"\n")
PY
      exit 1
    ' _ < "$units"
    ;;
  monitor)
    while true; do
      date --iso-8601=seconds
      "$PY" "$ROOT/scripts/monitor_topic5_history_conditioned_field_v0_4.py"
      done_count="$($PY - "$RESULT" <<'PY'
from pathlib import Path
import sys
print(len(list((Path(sys.argv[1])/'per_subject').glob('seed_*/epilepsiae_*/DONE.json'))))
PY
)"
      failed_count="$($PY - "$RESULT" <<'PY'
from pathlib import Path
import sys
print(len(list((Path(sys.argv[1])/'per_subject').glob('seed_*/epilepsiae_*/FAILED.json'))))
PY
)"
      if [[ "$((done_count + failed_count))" -ge 45 ]]; then
        break
      fi
      sleep 60
    done
    ;;
  summarize)
    "$PY" "$ROOT/scripts/summarize_topic5_history_conditioned_field_v0_4.py"
    ;;
  plot)
    "$PY" "$ROOT/scripts/plot_topic5_history_conditioned_field_v0_4.py"
    ;;
  downstream)
    "$PY" "$ROOT/scripts/extract_topic5_history_conditioned_field_diagnostics_v0_4.py"
    "$PY" "$ROOT/scripts/accept_topic5_history_conditioned_field_v0_4.py"
    "$PY" "$ROOT/scripts/summarize_topic5_history_conditioned_field_v0_4.py"
    "$PY" "$ROOT/scripts/plot_topic5_history_conditioned_field_v0_4.py"
    "$PY" "$ROOT/scripts/report_topic5_history_conditioned_field_v0_4.py"
    ;;
  finalize)
    while true; do
      done_count="$($PY - "$RESULT" <<'PY'
from pathlib import Path
import sys
print(len(list((Path(sys.argv[1])/'per_subject').glob('seed_*/epilepsiae_*/DONE.json'))))
PY
)"
      failed_count="$($PY - "$RESULT" <<'PY'
from pathlib import Path
import sys
print(len(list((Path(sys.argv[1])/'per_subject').glob('seed_*/epilepsiae_*/FAILED.json'))))
PY
)"
      if [[ "$failed_count" -gt 0 ]]; then
        echo "formal run has $failed_count failed unit(s); downstream analysis not started" >&2
        exit 1
      fi
      if [[ "$done_count" -ge 45 ]]; then
        "$PY" "$ROOT/scripts/extract_topic5_history_conditioned_field_diagnostics_v0_4.py"
        "$PY" "$ROOT/scripts/accept_topic5_history_conditioned_field_v0_4.py"
        "$PY" "$ROOT/scripts/summarize_topic5_history_conditioned_field_v0_4.py"
        "$PY" "$ROOT/scripts/plot_topic5_history_conditioned_field_v0_4.py"
        "$PY" "$ROOT/scripts/report_topic5_history_conditioned_field_v0_4.py"
        break
      fi
      sleep 60
    done
    ;;
  status)
    "$PY" "$ROOT/scripts/monitor_topic5_history_conditioned_field_v0_4.py"
    ;;
  *)
    echo "usage: $0 {cache|train|monitor|status|summarize|plot|downstream|finalize} [workers]" >&2
    exit 2
    ;;
esac
