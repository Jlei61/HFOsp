"""Generate event times with the patient's observation and ictal exclusions fixed.

This tests the fitted effective observation model on matching exposure, not an
autonomous seizure model. Interictal event times and marks are never replayed.
"""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.generate import simulate,summaries,build_configs

OUT=RUN/'matched_exposure_generation_v1_12'

def worker(job):
    path=OUT/'runs'/f"{job['version']}_{job['model']}_{job['rep']:03d}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.time()
    try:
        d=np.load(RUN/'observations.npz');origin=float(d['origin_epoch']);ex=pd.read_csv(RUN/'exposure.csv');inv=json.loads((RUN/'seizures.json').read_text());extent=ex.end_epoch.max()-origin;audit=json.loads((RUN/'data_audit.json').read_text());intervals=[];lo=0.
        for seizure in inv:
            hi=seizure['onset']-origin
            if hi>lo:intervals.append((lo,hi))
            lo=seizure['offset']-origin
        if extent>lo:intervals.append((lo,extent))
        times=[];ys=[];groups=[]
        for epoch,(lo,hi) in enumerate(intervals):
            if hi<=0:continue
            lo=max(lo,0);cfg=dict(job['config']);cfg['hours']=(hi-lo)/3600;t,y,_,_,_=simulate(seed=job['seed']+epoch*10000,**cfg);t+=lo
            for segment,row in enumerate(ex.itertuples()):
                aa,bb=row.start_epoch-origin,row.end_epoch-origin
                if bb<=lo or aa>=hi:continue
                keep=(t>=aa)&(t+cfg['deadtime']<=bb);times.append(t[keep]);ys.append(y[keep]);groups.append(np.full(keep.sum(),epoch*1000+segment,int))
        t=np.concatenate(times);y=np.concatenate(ys);g=np.concatenate(groups);order=np.argsort(t);t=t[order];y=y[order];g=g[order];assert np.all(np.diff(t)>=.25-1e-6)
        result=summaries(t,y,audit['observed_interictal_hours'],segments=g,exposure=ex[['start_epoch','end_epoch']].to_numpy()-origin,seizures=np.array([[s['onset'],s['offset']] for s in inv])-origin);result.update(status='COMPLETE',job=job,elapsed=time.time()-start)
        if job['rep']==0:
            path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),event_seconds=t,label_tb=y,group=g)
    except Exception:result=dict(status='FAILED',job=job,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(path,result);return result

def main():
    cfg,old=build_configs(1);fits=[]
    for p in (RUN/'renewal_grid_refit_v1_7/fits').glob('g1_full_m512_*.json'):
        r=json.loads(p.read_text())
        if r['status']=='COMPLETE':fits.append(r)
    refit=max(fits,key=lambda r:r['loglik']);a,lt,ls=refit['theta'];configs={};jobs=[]
    for version in ['laplace','state_grid']:
        for model in ['activity_only','independent_history']:
            c=dict(cfg[model])
            if version=='state_grid':c.update(a=a,tau_r=np.exp(lt),sd_r=np.exp(ls))
            configs[version+'_'+model]=c
            for rep in range(64):jobs.append(dict(version=version,model=model,rep=rep,seed=380000+rep+(1000 if model=='independent_history' else 0),config=c))
    write_json(OUT/'contract.json',dict(scope='Conditional on actual coverage and ictal exclusions; interictal event times and labels autonomously generated',initial_law='Independent stationary latent initial distributions at interictal interval boundaries, as in fitted models; not a biological reset claim',observation='Only generated 250-ms windows fully inside actual coverage retained; gaps remain missing',configs=configs,grid_refit_source=refit,n_replicates=64,not_claimed='No seizure time, clinical type, or exact event-path prediction'))
    with ProcessPoolExecutor(max_workers=20) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(complete=i+1,total=len(jobs),version=r['job']['version'],model=r['job']['model'],rep=r['job']['rep'],status=r['status'],rate=r.get('rate_per_hour'),elapsed=r['elapsed'])),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_runs=len(jobs),finished_unix=time.time()))

if __name__=='__main__':main()
