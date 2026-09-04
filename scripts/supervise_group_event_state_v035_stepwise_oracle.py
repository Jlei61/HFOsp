#!/usr/bin/env python3
"""Persistent two-GPU queue for the per-step future-oracle sensitivity assay."""
from __future__ import annotations
import argparse,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.topic5_group_event_state.v035.contracts import LOCKED_SEEDS,OUTPUT_ROOT,atomic_json  # noqa: E402
PY=Path('/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python'); SUBJECTS=('epilepsiae_253','epilepsiae_548','epilepsiae_583')
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--gpus',default='0,1'); a=ap.parse_args(); gpus=[x.strip() for x in a.gpus.split(',')]
    root=OUTPUT_ROOT/'stepwise_oracle_supervisor'; logs=root/'logs'; logs.mkdir(parents=True,exist_ok=True); pending=[]
    for s in SUBJECTS:
        for d,z in enumerate(LOCKED_SEEDS[:3]): pending.append({'subject':s,'decoder_seed':d,'state_seed':z,'out':str(OUTPUT_ROOT/'stepwise_oracle'/s/f'decoder_seed{d}_state_seed{z}'/'card.json')})
    env=os.environ.copy(); running={}; complete=[]; failed=[]
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): env[k]='1'
    while pending or running:
        for gpu,row in list(running.items()):
            code=row['p'].poll()
            if code is None: continue
            row['h'].close(); job=row['job']; (complete if code==0 and Path(job['out']).exists() else failed).append(job if code==0 else {**job,'returncode':code,'log':row['log']}); del running[gpu]
        for gpu in gpus:
            if gpu in running or not pending: continue
            job=pending.pop(0)
            if Path(job['out']).exists(): complete.append(job); continue
            log=logs/f"{job['subject']}_decoder{job['decoder_seed']}_state{job['state_seed']}_gpu{gpu}.log"; h=log.open('a',encoding='utf-8'); cmd=[str(PY),str(ROOT/'scripts/run_group_event_state_v035_stepwise_oracle.py'),'--subject',job['subject'],'--decoder-seed',str(job['decoder_seed']),'--state-seed',str(job['state_seed']),'--device',f'cuda:{gpu}']; p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True); running[gpu]={'p':p,'h':h,'job':job,'log':str(log),'started':time.time()}
        atomic_json(root/'queue_state.json',{'format':'group_event_state_v0_3_5_stepwise_oracle_queue_v1','updated_epoch':time.time(),'pending':len(pending),'complete':len(complete),'failed':failed,'running':{gpu:{'pid':row['p'].pid,'job':row['job'],'elapsed_seconds':time.time()-row['started']} for gpu,row in running.items()}})
        if pending or running: time.sleep(10)
    atomic_json(root/'queue_done.json',{'format':'group_event_state_v0_3_5_stepwise_oracle_done_v1','complete':complete,'failed':failed})
if __name__=='__main__': main()
