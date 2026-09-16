#!/usr/bin/env python3
"""Detached multi-GPU supervisor for v0.3.7 dual observer H1."""

from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
PY=Path('/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python')
RUN=REPO/'scripts/run_group_event_state_v037_h1_dual.py'
OUT=Path('/data/hfosp_group_event_state_v0_3_7/h1_dual_budget_complete')
SUBJECTS=('epilepsiae_253','epilepsiae_958','epilepsiae_1077','epilepsiae_1125')
SEEDS=(20260903,20260904,20260905,20260906,20260907)

def write(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--workers-per-gpu',type=int,default=2); ap.add_argument('--poll-seconds',type=float,default=10); a=ap.parse_args()
 sup=OUT/'supervisor'; logs=sup/'logs'; logs.mkdir(parents=True,exist_ok=True)
 pending=[(s,z) for z in SEEDS for s in SUBJECTS]; slots=[(g,k) for g in (0,1) for k in range(a.workers_per_gpu)]
 running={}; failures=[]; complete=0
 while pending or running:
  for key in list(running):
   proc,subject,seed,handle=running[key]; code=proc.poll()
   if code is None: continue
   handle.close(); del running[key]
   if code==0 and (OUT/subject/f'seed{seed}'/'card.json').exists(): complete+=1
   else: failures.append({'subject':subject,'seed':seed,'returncode':code,'physical_gpu':key[0]})
  for key in slots:
   if key in running or not pending: continue
   subject,seed=pending.pop(0); handle=(logs/f'{subject}__seed{seed}.log').open('a')
   env=dict(os.environ); env['CUDA_VISIBLE_DEVICES']=str(key[0]); env.setdefault('OMP_NUM_THREADS','2')
   proc=subprocess.Popen([str(PY),str(RUN),'--subject',subject,'--seed',str(seed),'--device','cuda:0','--out-root',str(OUT)],cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT)
   running[key]=(proc,subject,seed,handle)
  write(sup/'queue_status.json',{'format':'group_event_state_v0_3_7_h1_dual_budget_complete_queue_v2','status':'RUNNING' if pending or running else ('FAILED' if failures else 'COMPLETE'),'total':20,'complete':complete,'pending':len(pending),'running':[{'subject':s,'seed':z,'pid':p.pid,'physical_gpu':g,'slot_on_gpu':k} for (g,k),(p,s,z,_h) in sorted(running.items())],'failures':failures,'development_targets_read':False,'seizure_targets_read':False,'sealed_partition_opened':False})
  if pending or running: time.sleep(a.poll_seconds)
 write(sup/'queue_status.json',{'format':'group_event_state_v0_3_7_h1_dual_budget_complete_queue_v2','status':'FAILED' if failures else 'COMPLETE','total':20,'complete':complete,'pending':0,'running':[],'failures':failures,'development_targets_read':False,'seizure_targets_read':False,'sealed_partition_opened':False})
 raise SystemExit(1 if failures else 0)

if __name__=='__main__': main()
