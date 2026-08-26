#!/usr/bin/env bash
set -euo pipefail

REPO=/home/honglab/leijiaxin/HFOsp
PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
ROOT="$REPO/results/epi_prssm/continuous_marked_state/r1/r1_2b"
LOG="$ROOT/logs"
STATUS="$ROOT/RUN_STATUS.json"
mkdir -p "$LOG"
cd "$REPO"
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

write_status() {
  local stage="$1"
  local detail="$2"
  "$PY" -c 'import json,os,sys,time; p,s,d=sys.argv[1:]; t=p+".tmp"; open(t,"w").write(json.dumps({"status":"RUNNING","stage":s,"detail":d,"updated_epoch":time.time(),"sealed_opened":False},indent=2,sort_keys=True)); os.replace(t,p)' "$STATUS" "$stage" "$detail"
}

is_complete() {
  local path="$1"
  "$PY" -c 'import json,sys; x=json.load(open(sys.argv[1])); raise SystemExit(0 if x.get("status")=="COMPLETE" and x.get("sealed_opened") is False else 1)' "$path" 2>/dev/null
}

for subject in yuquan_huanghanwen epilepsiae_620 epilepsiae_958; do
  manifest="$ROOT/cache/$subject/manifest.json"
  if [[ -f "$manifest" ]] && is_complete "$manifest"; then
    continue
  fi
  batch=16
  [[ "$subject" == "epilepsiae_958" ]] && batch=8
  write_status cache "$subject"
  "$PY" scripts/topic5_continuous_marked_state_r1/build_r1_2b_cache.py \
    --subject "$subject" --device cuda --anchor-batch-size "$batch" \
    > "$LOG/cache_${subject}.log" 2>&1
done

for subject in yuquan_huanghanwen epilepsiae_620 epilepsiae_958; do
  for arm in joint_explicit joint_explicit_raw; do
    for seed in 0 1 2; do
      result="$ROOT/joint/$subject/${arm}_seed_${seed}/result.json"
      if [[ -f "$result" ]] && is_complete "$result"; then
        continue
      fi
      write_status fit "$subject/$arm/seed_$seed"
      "$PY" scripts/topic5_continuous_marked_state_r1/run_r1_2b_joint.py \
        --subject "$subject" --arm "$arm" --seed "$seed" --device cuda \
        --epochs 4 --chunk-anchors 128 --horizon-starts 64 \
        > "$LOG/joint_${subject}_${arm}_seed_${seed}.log" 2>&1
    done
  done
done

write_status aggregate all_fits_complete
"$PY" scripts/topic5_continuous_marked_state_r1/aggregate_r1_2b.py \
  > "$LOG/aggregate.log" 2>&1
write_status audit reports_complete
"$PY" scripts/topic5_continuous_marked_state_r1/audit_r1_2b_package.py \
  > "$LOG/audit.log" 2>&1
"$PY" -c 'import json,os,sys,time; p=sys.argv[1]; t=p+".tmp"; open(t,"w").write(json.dumps({"status":"COMPLETE","stage":"done","detail":"18 fits + aggregate + audit","updated_epoch":time.time(),"sealed_opened":False},indent=2,sort_keys=True)); os.replace(t,p)' "$STATUS"
