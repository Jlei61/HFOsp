"""Does mode memory survive adjustment for causal recent total activity?

Use 1/5-minute exponential count histories, reset at missing-coverage starts.
Only completed previous event windows contribute. No current/future label or
seizure-type covariate, no reinterpretation as a causal physiological coupling.
"""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.special import expit
from numba import njit
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import fit,filter_adf,slice_data
from scripts.patient_state_v1.advanced_controls import history_data
from scripts.patient_state_v1.analyze_first import metrics

OUT=RUN/'activity_adjusted_marks_v1_7'

@njit(cache=True)
def causal_activity(t,group,baseline,hours):
    values=np.empty(len(t));r=baseline
    for i in range(len(t)):
        if i==0 or group[i]!=group[i-1]:r=baseline
        else:r=(r+1/hours)*np.exp(-(t[i]-t[i-1])/hours)
        values[i]=np.log(max(r,1e-12)/baseline)
    return values

def prepare(end,minutes):
    d=dict(np.load(RUN/'observations.npz'));ev=pd.read_csv(RUN/'events.csv');ex=pd.read_csv(RUN/'exposure.csv');cutoff=ev.end_epoch.iloc[end-1]
    hours=np.maximum(0,np.minimum(ex.end_epoch.to_numpy(),cutoff)-ex.start_epoch.to_numpy()).sum()/3600;baseline=end/hours
    groups=ev.coverage_segment.to_numpy()*1000+d['epoch'];x=causal_activity(d['t'],groups,baseline,minutes/60);h=history_data(d)['x'][:,1]
    assert np.all(ev.end_epoch.to_numpy()[:-1]<=ev.start_epoch.to_numpy()[1:])
    d['x']=np.column_stack([np.ones(len(x)),x,h]);return d,dict(baseline_rate_per_hour=baseline,activity_window_minutes=minutes,activity_scale='log exponentially weighted past total count rate / training observed rate',coverage_start='initialize rate at training mean, do not interpret missing time as zero count')

def worker(job):
    path=OUT/'fits'/f"{job['id']}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.time()
    try:
        d,info=prepare(job['end'],job['minutes']);train=slice_data(d,0,job['end']);p=train['y'].mean();initial=np.r_[np.log(p/(1-p)),0.,.8,np.log(job['tau0']),np.log(.6)]
        # Reuse the generic three-coefficient interface; x contains no clock cycle.
        r=fit(train,'ou_cycle' if job['latent'] else 'cycle',initial if job['latent'] else None);r['model']='ou_activity_history' if job['latent'] else 'activity_history'
        if job['latent']:
            out=filter_adf(np.array(r['theta']),d,True);prediction=out['predict_tb'];state=out['mean']
        else:prediction=expit(d['x']@r['theta']);state=np.zeros(len(prediction))
        r.update(status='COMPLETE',job=job,covariate_info=info,elapsed=time.time()-start)
        if job['scope']!='full':
            lo,hi=job['test_start'],job['test_end'];r['forward']=metrics(prediction[lo:hi],d['y'][lo:hi],d['n'][lo:hi])
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),predict_tb=prediction,state=state)
    except Exception:r=dict(status='FAILED',job=job,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(path,r);return r

def main():
    data=np.load(RUN/'observations.npz');folds=json.loads((RUN/'splits.json').read_text());scopes=[dict(scope='full',end=len(data['y']))]+[dict(scope=f"fold{f['fold']}",end=f['train_end'],test_start=f['test_start'],test_end=f['test_end']) for f in folds];jobs=[]
    for scope in scopes:
        for minutes in (1,5):
            for latent in (False,True):
                for i,tau in enumerate((.2,2.) if latent else (.2,)):
                    jobs.append(dict(id=f"{scope['scope']}_w{minutes}_latent{int(latent)}_{i}",minutes=minutes,latent=latent,tau0=tau,**scope))
    write_json(OUT/'queue.json',jobs)
    with ProcessPoolExecutor(max_workers=20) as ex:
        for f in as_completed([ex.submit(worker,j) for j in jobs]):
            r=f.result();print(json.dumps(dict(id=r['job']['id'],status=r['status'],tau=r.get('tau_hours'),loglik=r.get('loglik'),forward=r.get('forward'))),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs),finished_unix=time.time()))

if __name__=='__main__':main()
