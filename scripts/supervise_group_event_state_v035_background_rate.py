#!/usr/bin/env python3
"""Persistent CPU queue for all registered background-rate comparisons."""

from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.topic5_group_event_state.v035.contracts import LOCKED_SEEDS,OUTPUT_ROOT,V035_SUBJECTS,atomic_json  # noqa: E402
PY=Path('/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python')
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--rate-root',type=Path,default=OUTPUT_ROOT/'dynamic_rate')
    ap.add_argument('--out-root',type=Path,default=OUTPUT_ROOT/'background_rate')
    ap.add_argument('--control-root',type=Path,default=OUTPUT_ROOT/'background_rate_supervisor')
    ap.add_argument('--three-seeds',action='store_true')
    ap.add_argument('--priority-five-seeds',action='store_true')
    a=ap.parse_args()
    root=a.control_root; logs=root/'logs'; logs.mkdir(parents=True,exist_ok=True)
    pending=[]
    for s in V035_SUBJECTS:
        if a.priority_five_seeds:
            seeds = LOCKED_SEEDS if s in {'epilepsiae_253','epilepsiae_922','epilepsiae_1096'} else LOCKED_SEEDS[:3]
        elif a.three_seeds:
            seeds = LOCKED_SEEDS[:3]
        else:
            seeds = LOCKED_SEEDS if s not in {'epilepsiae_384','epilepsiae_1125'} else LOCKED_SEEDS[:3]
        for seed in seeds: pending.append({'subject':s,'seed':seed,'out':str(a.out_root/s/f'seed{seed}'/'card.json')})
    running={}; complete=[]; failed=[]; env=os.environ.copy()
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): env[k]='1'
    while pending or running:
        for slot,row in list(running.items()):
            code=row['p'].poll()
            if code is None: continue
            row['h'].close(); job=row['job']
            (complete if code==0 and Path(job['out']).exists() else failed).append(job if code==0 else {**job,'returncode':code,'log':row['log']})
            del running[slot]
        for slot in range(a.workers):
            if slot in running or not pending: continue
            job=pending.pop(0)
            if Path(job['out']).exists(): complete.append(job); continue
            log=logs/f"{job['subject']}_seed{job['seed']}.log"; h=log.open('a',encoding='utf-8')
            p=subprocess.Popen([str(PY),str(ROOT/'scripts/run_group_event_state_v035_background_rate.py'),'--subject',job['subject'],'--seed',str(job['seed']),'--rate-root',str(a.rate_root),'--out-root',str(a.out_root)],cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
            running[slot]={'p':p,'h':h,'job':job,'log':str(log),'started':time.time()}
        atomic_json(root/'queue_state.json',{'format':'group_event_state_v0_3_5_background_rate_queue_v1','updated_epoch':time.time(),'pending':len(pending),'complete':len(complete),'failed':failed,'running':{str(k):{'pid':v['p'].pid,'job':v['job'],'elapsed_seconds':time.time()-v['started']} for k,v in running.items()}})
        if pending or running: time.sleep(10)
    atomic_json(root/'queue_done.json',{'format':'group_event_state_v0_3_5_background_rate_done_v1','complete':complete,'failed':failed})
if __name__=='__main__': main()
