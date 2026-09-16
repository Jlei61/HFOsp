"""Current-interval-independent parameter fitting for joint pre-seizure readout."""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.renewal import prepare
from scripts.patient_state_v1.joint_two_state import fit_joint
from scripts.patient_state_v1.joint_two_state_filter import evaluate
OUT=RUN/'joint_preseizure_v1_22'

def worker(j):
    p=OUT/'fits'/f"sz{j['sz']:02d}_c{int(j['coupled'])}_{j['initial']}.json"
    if p.exists():return json.loads(p.read_text())
    start=time.time()
    try:
        full=prepare(5,.25);d={k:v[:j['end']].copy() for k,v in full.items() if np.ndim(v)>0};q=d['y'].sum()/d['n'].sum();b=np.log(q/(1-q));a=np.log(d['n'].sum()/d['exposure'].sum());initial=np.r_[b,a,([j['c0']] if j['coupled'] else []),np.log(j['tau0']),np.log(.6),np.log(.1),np.log(1.5)];r=fit_joint(d,j['coupled'],initial);f=evaluate(r['theta'],full,j['coupled'],80,True);r.update(status='COMPLETE',job=j,elapsed=time.time()-start);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p.with_suffix('.npz'),**f)
    except Exception:r=dict(status='FAILED',job=j,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(p,r);return r

def main():
    d=prepare(5,.25);inv=json.loads((RUN/'seizures.json').read_text());wins=pd.read_csv(RUN/'frozen_seizure_windows.csv');jobs=[]
    for sz in sorted(wins.sz.unique()):
        cutoff=inv[int(sz)-2]['offset'];end=int(np.searchsorted(d['hi'],cutoff,side='right'));assert end>0
        for coupled in [False,True]:
            for i,(tau,c) in enumerate([(.2,-.5),(2.,.5)]):jobs.append(dict(sz=int(sz),cutoff_epoch=cutoff,end=end,coupled=coupled,initial=i,tau0=tau,c0=c))
    write_json(OUT/'contract.json',dict(question='Do the same two TB-source cases show a common upward mode-state trajectory when event timing participates in inference?',n_seizures=12,n_tb=2,training='Parameters use only observations before the preceding seizure offset; current interval updates latent states only',models='Independent and observation-coupled activity/mode OU, same 5-second effective renewal likelihood',fit='Two training-only starts; Gaussian quadrature filter with 80 nodes; no seizure type enters fitting',readout='State levels and changes, total activity and mode probability distinguished; not a seizure hazard model',limits='Only two TB-source seizures, both early; numerical and observation-model approximation remain',n_jobs=len(jobs)))
    with ProcessPoolExecutor(max_workers=16) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(done=i+1,total=len(jobs),sz=r['job']['sz'],coupled=r['job']['coupled'],status=r['status'],elapsed=r['elapsed'],success=r.get('success'))),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs)))
if __name__=='__main__':main()
