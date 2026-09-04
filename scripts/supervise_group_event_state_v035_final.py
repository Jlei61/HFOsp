#!/usr/bin/env python3
"""Lock the INNER-selected recipe, open SELECTION once, and run final W4--W6."""

from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.topic5_group_event_state.v035.contracts import LOCKED_SEEDS,OUTPUT_ROOT,atomic_json  # noqa: E402
PY=Path('/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python'); SEARCH_SUBJECTS=('epilepsiae_253','epilepsiae_548','epilepsiae_583','epilepsiae_1096'); EXTRA=('epilepsiae_384','epilepsiae_1125','epilepsiae_1146')
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--gpus',default='0,1'); ap.add_argument('--workers-per-gpu',type=int,default=4); ap.add_argument('--wait-for',type=Path,default=OUTPUT_ROOT/'full_mark_search_supervisor'/'queue_done.json'); a=ap.parse_args(); gpus=[v.strip() for v in a.gpus.split(',') if v.strip()]
    if a.workers_per_gpu < 1: raise ValueError('workers-per-gpu must be positive')
    slots=[(f'{gpu}:{worker}',gpu) for gpu in gpus for worker in range(a.workers_per_gpu)]
    root=OUTPUT_ROOT/'final_supervisor'; logs=root/'logs'; logs.mkdir(parents=True,exist_ok=True)
    while not a.wait_for.exists(): atomic_json(root/'queue_state.json',{'format':'group_event_state_v0_3_5_final_queue_v1','status':'WAITING_FOR_SEARCH','updated_epoch':time.time(),'wait_for':str(a.wait_for)}); time.sleep(30)
    subprocess.run([str(PY),str(ROOT/'scripts/select_group_event_state_v035_full_mark_recipe.py')],cwd=ROOT,check=True)
    lock=json.loads((OUTPUT_ROOT/'full_mark_search'/'selected_recipe.json').read_text(encoding='utf-8')); recipe=lock['selected_recipe']; config=ROOT/'config/group_event_state_v035_search'/f'{recipe}.json'
    # A recipe whose INNER optimum sits at the final epoch has not received a
    # fair trainability test.  Extend the single locked recipe on the same
    # subjects/seeds before opening SELECTION; never extend only a favourable
    # patient.
    subprocess.run([str(PY),str(ROOT/'scripts/ensure_group_event_state_v035_full_mark_budget.py'),
                    '--gpus',','.join(gpus),'--workers-per-gpu',str(a.workers_per_gpu)],cwd=ROOT,check=True)
    budget=json.loads((OUTPUT_ROOT/'full_mark_search_budget_extension'/'budget_audit.json').read_text(encoding='utf-8'))
    config=Path(budget['final_config']); source_root=Path(budget['final_source_root'])
    jobs=[]
    for subject in SEARCH_SUBJECTS:
        for decoder_seed,state_seed in enumerate(LOCKED_SEEDS[:3]):
            source=source_root/subject/f'decoder_seed{decoder_seed}_state_seed{state_seed}'
            jobs.append({'kind':'state_rescore','subject':subject,'decoder_seed':decoder_seed,'state_seed':state_seed,'source':str(source),'batch_events':96,'retries':0})
    for subject in EXTRA:
        for decoder_seed,state_seed in enumerate(LOCKED_SEEDS[:3]): jobs.append({'kind':'state_train','subject':subject,'decoder_seed':decoder_seed,'state_seed':state_seed,'config':str(config),'chunk_events':256,'batch_events':96,'retries':0})
    env=os.environ.copy()
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): env[k]='1'
    pending=jobs; running={}; complete=[]; failed=[]
    def out(job):
        tag=f"decoder_seed{job['decoder_seed']}_state_seed{job['state_seed']}"
        return OUTPUT_ROOT/('full_mark_final' if job['kind'].startswith('state') else 'final_downstream')/job['subject']/tag/'card.json'
    while pending or running:
        for slot,row in list(running.items()):
            code=row['p'].poll()
            if code is None: continue
            row['h'].close(); job=row['job']; body=Path(row['log']).read_text(encoding='utf-8',errors='replace')[-30000:]
            if code==0 and out(job).exists():
                if job['kind'].startswith('state'):
                    pending.append({**job,'kind':'downstream','retries':0})
                else: complete.append(job)
            elif 'out of memory' in body.lower() and job['retries']<3:
                key='chunk_events' if job['kind']=='state_train' else 'batch_events'; job[key]=max(12,job.get(key,96)//2); job['retries']+=1; pending.insert(0,job)
            else: failed.append({**job,'failed_kind':job['kind'],'returncode':code,'log':row['log'],'tail':body[-4000:]})
            del running[slot]
        for slot,gpu in slots:
            if slot in running or not pending: continue
            job=pending.pop(0)
            if out(job).exists():
                if job['kind'].startswith('state'): pending.append({**job,'kind':'downstream','retries':0})
                else: complete.append(job)
                continue
            common=['--subject',job['subject'],'--decoder-seed',str(job['decoder_seed']),'--state-seed',str(job['state_seed'])]
            if job['kind']=='state_rescore': cmd=[str(PY),str(ROOT/'scripts/rescore_group_event_state_v035_full_mark.py'),*common,'--source-unit',job['source'],'--device',f'cuda:{gpu}']
            elif job['kind']=='state_train': cmd=[str(PY),str(ROOT/'scripts/run_group_event_state_v035_full_mark_state.py'),*common,'--config-json',job['config'],'--chunk-events',str(job['chunk_events']),'--out-root',str(OUTPUT_ROOT/'full_mark_final'),'--device',f'cuda:{gpu}']
            else: cmd=[str(PY),str(ROOT/'scripts/run_group_event_state_v035_final_downstream.py'),*common,'--batch-events',str(job['batch_events']),'--device',f'cuda:{gpu}']
            log=logs/f"{job['kind']}_{job['subject']}_decoder{job['decoder_seed']}_state{job['state_seed']}_gpu{gpu}_slot{slot.replace(':','_')}.log"; h=log.open('a',encoding='utf-8'); p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True); running[slot]={'p':p,'h':h,'job':job,'log':str(log),'started':time.time(),'gpu':gpu}
        atomic_json(root/'queue_state.json',{'format':'group_event_state_v0_3_5_final_queue_v1','status':'RUNNING','selected_recipe':recipe,'budget_status':budget['status'],'final_config':str(config),'workers_per_gpu':a.workers_per_gpu,'updated_epoch':time.time(),'pending':len(pending),'complete':len(complete),'failed':failed,'running':{slot:{'gpu':row['gpu'],'pid':row['p'].pid,'job':row['job'],'elapsed_seconds':time.time()-row['started']} for slot,row in running.items()}})
        if pending or running: time.sleep(15)
    # Explicit assay limitations are preserved as machine records, not converted to biological negatives.
    for subject,reason in {'epilepsiae_922':'mature decoder has no scorable event in the registered evaluation window'}.items():
        atomic_json(OUTPUT_ROOT/'full_mark_final'/subject/'NOT_ESTIMABLE.json',{'format':'group_event_state_v0_3_5_final_not_estimable_v1','subject':subject,'status':'NOT_ESTIMABLE','reason':reason,'development_targets_read':False,'sealed_partition_opened':False})
    atomic_json(root/'queue_done.json',{'format':'group_event_state_v0_3_5_final_done_v1','selected_recipe':recipe,'budget_audit':str(OUTPUT_ROOT/'full_mark_search_budget_extension'/'budget_audit.json'),'budget_status':budget['status'],'final_config':str(config),'complete':complete,'failed':failed})
if __name__=='__main__': main()
