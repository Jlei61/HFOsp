#!/usr/bin/env bash
# One-shot status snapshot for the detached Topic-5 T0 eligibility audit.
# Usage: bash scripts/monitor_t0.sh   (run anytime, including after reconnecting)
cd "$(dirname "$0")/.." || exit 1
LOG=results/topic5_ictal_recruitment/t0_audit.log
CSV=results/topic5_ictal_recruitment/t0_eligibility_audit.csv
SUMM=results/topic5_ictal_recruitment/t0_eligibility_summary.json
COHORT=25

pid=$(pgrep -f "python scripts/run_topic5_t0_eligibility" | head -1)
done=$(grep -c "seizures=" "$LOG" 2>/dev/null || true); done=${done:-0}

if [ -n "$pid" ]; then
  secs=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
  printf "STATUS : RUNNING (pid=%s, elapsed=%dm%02ds)\n" "$pid" $((secs/60)) $((secs%60))
  if [ "$done" -gt 0 ] && [ -n "$secs" ]; then
    eta=$(( secs * (COHORT - done) / done ))
    printf "ETA    : ~%dm for remaining %d subjects (rough, rate so far)\n" $((eta/60)) $((COHORT-done))
  fi
elif [ -f "$SUMM" ]; then
  echo "STATUS : DONE (summary written)"
else
  echo "STATUS : NOT RUNNING + no summary -> died/stopped. Re-run to resume:"
  echo "         ( setsid bash -c 'python scripts/run_topic5_t0_eligibility.py --out $CSV' </dev/null >>$LOG 2>&1 & )"
fi

echo "SUBJECTS: $done / $COHORT completed"
if [ -f "$CSV" ]; then
  python3 - "$CSV" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
b = lambda v: str(v).strip().lower() in ("true", "1", "yes")
ca = sum(1 for r in rows if b(r["cacheable"]))
el = sum(1 for r in rows if b(r["analysis_eligible"]))
be = sum(1 for r in rows if b(r["b_eligible"]))
print(f"SO FAR : {len(rows)} seizures | cacheable={ca} analysis_eligible={el} b_eligible(HFA)={be}")
PY
fi
echo "LAST   : $(tail -1 "$LOG" 2>/dev/null)"
if [ -f "$SUMM" ]; then echo "----- ROLL-UP -----"; cat "$SUMM"; fi
