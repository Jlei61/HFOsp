"""Fit parameters before the preceding seizure offset, then filter current interval.

These are observational state readouts, not a fitted seizure-type classifier.
"""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import fit,slice_data

def worker(job):
    path=RUN/'preseizure_forward_fits'/f"sz{job['sz']:02d}_{job['model']}.json"
    if path.exists():return json.loads(path.read_text())
    data=slice_data(dict(np.load(RUN/'observations.npz')),0,job['end']);p=np.clip(data['y'].mean(),1e-5,1-1e-5);nc=3 if job['model']=='ou_cycle' else 1
    initial=[np.r_[np.log(p/(1-p)),np.zeros(nc-1),np.log(tau),np.log(.6)] for tau in (.1,1.,6.)]
    fits=[fit(data,job['model'],t,maxiter=150) for t in initial];result=dict(max(fits,key=lambda r:r['loglik']));result.update(job=job,all_initial_fits=fits,status='COMPLETE')
    write_json(path,result);return result

def main():
    iv=json.loads((RUN/'seizures.json').read_text());wins=pd.read_csv(RUN/'frozen_seizure_windows.csv');e=pd.read_csv(RUN/'events.csv');jobs=[]
    for sz in sorted(wins.sz.unique()):
        cutoff=iv[sz-2]['offset'];end=int(np.searchsorted(e.start_epoch,cutoff))
        assert np.all(e.end_epoch.to_numpy()[:end]<=cutoff)
        for model in ('ou','ou_cycle'):jobs.append(dict(sz=int(sz),model=model,end=end,cutoff_epoch=cutoff,n_train_events=end,training_excludes_current_interval=True))
    write_json(RUN/'preseizure_forward_queue.json',jobs)
    with ProcessPoolExecutor(max_workers=16) as ex:
        for f in as_completed([ex.submit(worker,j) for j in jobs]):
            r=f.result();print(json.dumps(dict(sz=r['job']['sz'],model=r['model'],tau=r['tau_hours'],sd=r['stationary_sd'],status=r['status'])),flush=True)
    write_json(RUN/'preseizure_forward_status.json',dict(status='COMPLETE',n_jobs=len(jobs),finished_unix=time.time()))

if __name__=='__main__':main()
