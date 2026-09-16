#!/usr/bin/env python
"""Window close-out: task ledger, exact resume commands and measured ETA."""
import json,glob,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
ROOT=Path('/data/hfosp_group_event_state_rich_event_identification_v0311')
REPO=Path(__file__).resolve().parents[1]
PY_BIN='/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python'


def measured_rates():
    rates={}
    for f in glob.glob(str(ROOT/'runs*'/'*.card.json')):
        c=json.load(open(f))
        if not c.get('updates'):continue
        arm='state' if c['config']['arm']=='state' else 'reference'
        rates.setdefault(arm,[]).append(c['seconds']/max(c['updates'],1))
    return {k:dict(median_seconds_per_update=float(np.median(v)),n=len(v)) for k,v in rates.items()}


if __name__=='__main__':
    plan=json.load(open(ROOT/'execution_plan.json'))
    cards={}
    for d in ('runs','runs_extended','runs_history','runs_reference_variants'):
        for f in glob.glob(str(ROOT/d/'*.card.json'))+glob.glob(str(ROOT/d/'*'/'*.card.json')):
            cards[Path(f).name.replace('.card.json','')]=dict(dir=d,path=f)
    claims={}
    for f in glob.glob(str(ROOT/'claims'/'*.done.json')):
        d=json.load(open(f));claims[d['task']['subject']+'|'+d['task']['inputs']+'|'+d['task']['family']+'|'+d['task']['arm']]=d['status']
    ledger=[]
    for t in plan['tasks']:
        st='COMPLETE' if t['tag'] in cards else 'NOT_RUN'
        if st=='NOT_RUN':
            k=t['subject']+'|'+t['inputs']+'|'+t['family']+'|'+t['arm']
            if claims.get(k)=='FAILED':st='FAILED'
        ledger.append(dict(priority=t['priority'],tag=t['tag'],status=st,
                           **{k:t[k] for k in ('subject','split','inputs','family','arm',
                                               'target_ablation','crossview_ablation')}))
    todo=[r for r in ledger if r['status']!='COMPLETE']
    rates=measured_rates()
    def cmd(r,updates=1200,out='runs'):
        c=[PY_BIN,'scripts/run_group_event_state_v0311_cell.py','--subject',r['subject'],
           '--split',r['split'],'--inputs',r['inputs'],'--family',r['family'],'--arm',r['arm'],
           '--max-updates',str(updates),'--extended-updates',str(updates),
           '--device','cuda:0','--out-dir',str(ROOT/out)]
        if r.get('target_ablation'):c.append('--target-ablation')
        if r.get('crossview_ablation'):c.append('--crossview-ablation')
        return ' '.join(c)
    est=0.
    for r in todo:
        s=rates.get('state' if r['arm']=='state' else 'reference',{}).get('median_seconds_per_update',2.0)
        est+=s*1200
    out=dict(generated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
             git_commit=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,capture_output=True,text=True).stdout.strip(),
             n_planned=len(ledger),n_complete=sum(1 for r in ledger if r['status']=='COMPLETE'),
             n_remaining=len(todo),measured_rates=rates,
             estimated_remaining_worker_hours=round(est/3600,1),
             estimated_wall_hours_at_6_workers=round(est/3600/6,1),
             resume=dict(
                 scheduler=(f'{PY_BIN} scripts/schedule_group_event_state_v0311.py --device cuda:0 '
                            f'--worker g0w0 --updates 1200 --deadline-epoch <epoch>'),
                 note='claims are atomic; start one worker per GPU slot and they will divide the queue',
                 rescore=(f'{PY_BIN} scripts/rescore_group_event_state_v0311.py --device cuda:0 '
                          f'--paths 64 --dirs runs runs_extended runs_history runs_reference_variants'),
                 postprocess=(f'{PY_BIN} scripts/postprocess_group_event_state_v0311.py --device cpu '
                              f'--deadline-epoch <epoch>'),
                 aggregate=f'{PY_BIN} scripts/aggregate_group_event_state_v0311.py',
                 figures=f'{PY_BIN} scripts/plot_group_event_state_v0311.py'),
             next_priority_pairings=[dict(tag=r['tag'],priority=r['priority'],command=cmd(r)) for r in todo[:12]],
             remaining=ledger)
    (ROOT/'handoff.json').write_text(json.dumps(out,indent=1,default=str))
    print(json.dumps({k:out[k] for k in ('n_planned','n_complete','n_remaining','measured_rates',
                                         'estimated_remaining_worker_hours',
                                         'estimated_wall_hours_at_6_workers')},indent=1,default=str))
