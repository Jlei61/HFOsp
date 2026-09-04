#!/usr/bin/env python3
"""Persistent multi-recipe trainability search for the full-event state.

All recipe and epoch choices use FIT/INNER only.  SELECTION targets are held
unread by every search job and are opened only after a single cohort recipe is
registered by the companion selector.
"""

from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.topic5_group_event_state.v035.contracts import LOCKED_SEEDS,OUTPUT_ROOT,atomic_json  # noqa: E402
PY=Path('/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python')
SUBJECTS=('epilepsiae_253','epilepsiae_548','epilepsiae_583','epilepsiae_1096')
RECIPES=tuple(sorted((ROOT/'config/group_event_state_v035_search').glob('*.json')))

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--gpus',default='0,1')
    ap.add_argument('--workers-per-gpu',type=int,default=2)
    ap.add_argument('--wait-for',type=Path,default=OUTPUT_ROOT/'expansion_supervisor'/'queue_done.json'); a=ap.parse_args()
    gpus=[v.strip() for v in a.gpus.split(',') if v.strip()]
    if a.workers_per_gpu < 1: raise ValueError('workers-per-gpu must be positive')
    slots=[(f'{gpu}:{worker}',gpu) for gpu in gpus for worker in range(a.workers_per_gpu)]
    root=OUTPUT_ROOT/'full_mark_search_supervisor'; logs=root/'logs'; logs.mkdir(parents=True,exist_ok=True)
    while not a.wait_for.exists():
        atomic_json(root/'queue_state.json',{'format':'group_event_state_v0_3_5_full_mark_search_queue_v1','status':'WAITING_FOR_EXPANSION','wait_for':str(a.wait_for),'updated_epoch':time.time()}); time.sleep(30)
    pending=[]
    for recipe in RECIPES:
        for subject in SUBJECTS:
            for decoder_seed,state_seed in enumerate(LOCKED_SEEDS[:3]):
                out=OUTPUT_ROOT/'full_mark_search'/recipe.stem/subject/f'decoder_seed{decoder_seed}_state_seed{state_seed}'/'card.json'
                pending.append({'recipe':recipe.stem,'config':str(recipe),'subject':subject,'decoder_seed':decoder_seed,'state_seed':state_seed,'chunk_events':256,'retries':0,'out':str(out)})
    running={}; complete=[]; failed=[]; env=os.environ.copy()
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): env[k]='1'
    while pending or running:
        for slot,row in list(running.items()):
            code=row['p'].poll()
            if code is None: continue
            row['h'].close(); job=row['job']; body=Path(row['log']).read_text(encoding='utf-8',errors='replace')[-20000:]
            if code==0 and Path(job['out']).exists(): complete.append(job)
            elif 'out of memory' in body.lower() and job['chunk_events']>32 and job['retries']<3:
                job['chunk_events']//=2; job['retries']+=1; pending.insert(0,job)
            else: failed.append({**job,'returncode':code,'log':row['log'],'tail':body[-3000:]})
            del running[slot]
        for slot,gpu in slots:
            if slot in running or not pending: continue
            job=pending.pop(0)
            if Path(job['out']).exists(): complete.append(job); continue
            out_root=OUTPUT_ROOT/'full_mark_search'/job['recipe']
            log=logs/f"{job['recipe']}_{job['subject']}_decoder{job['decoder_seed']}_state{job['state_seed']}_gpu{gpu}.log"; h=log.open('a',encoding='utf-8')
            cmd=[str(PY),str(ROOT/'scripts/run_group_event_state_v035_full_mark_state.py'),'--subject',job['subject'],'--decoder-seed',str(job['decoder_seed']),'--state-seed',str(job['state_seed']),'--chunk-events',str(job['chunk_events']),'--config-json',job['config'],'--hold-selection','--out-root',str(out_root),'--device',f'cuda:{gpu}']
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
            running[slot]={'p':p,'h':h,'job':job,'log':str(log),'started':time.time(),'gpu':gpu}
        atomic_json(root/'queue_state.json',{'format':'group_event_state_v0_3_5_full_mark_search_queue_v1','status':'RUNNING','updated_epoch':time.time(),'planned':len(RECIPES)*len(SUBJECTS)*3,'workers_per_gpu':a.workers_per_gpu,'pending':len(pending),'complete':len(complete),'failed':failed,'running':{slot:{'gpu':row['gpu'],'pid':row['p'].pid,'job':row['job'],'elapsed_seconds':time.time()-row['started']} for slot,row in running.items()},'selection_targets_read':False})
        if pending or running: time.sleep(15)
    atomic_json(root/'queue_done.json',{'format':'group_event_state_v0_3_5_full_mark_search_done_v1','complete':complete,'failed':failed,'selection_targets_read':False})
if __name__=='__main__': main()
