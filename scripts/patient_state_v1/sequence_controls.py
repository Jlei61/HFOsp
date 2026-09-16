"""Destroy temporal label information at specified scales while preserving actual times/counts."""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import fit,slice_data,filter_adf
from scripts.patient_state_v1.analyze_first import metrics
from scripts.patient_state_v1.round2 import beta_fit_predict

def worker(job):
    path=RUN/'sequence_controls'/f"{job['id']}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.time()
    try:
        data=dict(np.load(RUN/'observations.npz'));events=pd.read_csv(RUN/'events.csv');rng=np.random.default_rng(job['seed'])
        coverage=events.coverage_segment.to_numpy()*1000+data['epoch']
        if job['scale']=='epoch':groups=data['epoch']
        elif job['scale']=='coverage':groups=coverage
        else:
            minutes=float(job['scale']);groups=coverage*1000000+np.floor(data['t']*60/minutes).astype(int)
        y=data['y'].copy()
        for group in np.unique(groups):
            ix=np.flatnonzero(groups==group);y[ix]=rng.permutation(y[ix])
        data['y']=y;folds=json.loads((RUN/'splits.json').read_text());rows=[]
        for f in folds:
            d=slice_data(data,0,f['train_end']);r=fit(d,'ou',maxiter=100)
            p=filter_adf(np.array(r['theta']),data)['predict_tb']
            control,_=beta_fit_predict(data,coverage,f['train_end'])
            a,b=f['test_start'],f['test_end'];m=metrics(p[a:b],y[a:b],data['n'][a:b]);base=metrics(control[a:b],y[a:b],data['n'][a:b])
            rows.append(dict(fold=f['fold'],tau_hours=r['tau_hours'],stationary_sd=r['stationary_sd'],success=r['success'],
                             ou_loglik=m['loglik'],coverage_loglik=base['loglik'],n_events=m['n_events'],gain_per_event=(m['loglik']-base['loglik'])/m['n_events']))
        result=dict(status='COMPLETE',job=job,rows=rows,elapsed=time.time()-start)
    except Exception:result=dict(status='FAILED',job=job,traceback=traceback.format_exc(),elapsed=time.time()-start)
    write_json(path,result);return result

def main():
    jobs=[dict(id=f'shuffle_{scale}_{rep:03d}',scale=scale,seed=265000+1000*i+rep) for i,scale in enumerate(['epoch','coverage','15','1']) for rep in range(128)]
    write_json(RUN/'sequence_control_queue.json',jobs)
    with ProcessPoolExecutor(max_workers=28) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(complete=i+1,total=len(jobs),id=r['job']['id'],status=r['status'],elapsed=r['elapsed'])),flush=True)
    write_json(RUN/'sequence_control_status.json',dict(status='COMPLETE',n_jobs=len(jobs),finished_unix=time.time()))

if __name__=='__main__':main()
