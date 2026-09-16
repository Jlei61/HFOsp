"""Isolate mark dynamics by generating labels at fixed observed event times.

Only mode summaries are evaluated. Event rates, intervals, and packing support
are conditioned on and therefore cannot count as successful generated outcomes.
"""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from numba import njit
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.analyze_first import best_fits
from scripts.patient_state_v1.review_predictions import selected
from scripts.patient_state_v1.generate import summaries

OUT=RUN/'conditional_mark_generation_v1_10'

@njit(cache=True)
def labels(dt,reset,b,gamma,tf,sdf,ts,sds,seed):
    np.random.seed(seed);s=0.;background=0.;previous=.5;y=np.empty(len(dt),np.int8)
    for i in range(len(dt)):
        if reset[i]:s=np.random.normal()*sdf;background=np.random.normal()*sds;h=0.
        else:
            a=np.exp(-dt[i]/tf);c=np.exp(-dt[i]/ts);s=a*s+sdf*np.sqrt(-np.expm1(-2*dt[i]/tf))*np.random.normal();background=c*background+sds*np.sqrt(-np.expm1(-2*dt[i]/ts))*np.random.normal();h=(previous-.5)*np.exp(-dt[i]*3600)
        p=1/(1+np.exp(-(b+s+background+gamma*h)));y[i]=1 if np.random.random()<p else 0;previous=y[i]
    return y

def worker(job):
    dest=OUT/'runs'/f"{job['model']}_{job['rep']:03d}.json"
    if dest.exists():return json.loads(dest.read_text())
    d=np.load(RUN/'observations.npz');ev=pd.read_csv(RUN/'events.csv');ex=pd.read_csv(RUN/'exposure.csv');iv=json.loads((RUN/'seizures.json').read_text());audit=json.loads((RUN/'data_audit.json').read_text());origin=float(d['origin_epoch']);y=labels(d['dt'],d['reset'],*job['parameters'],job['seed']);r=summaries(ev.start_epoch.to_numpy()-origin,y,audit['observed_interictal_hours'],segments=ev.coverage_segment.to_numpy()*1000+ev.interictal_epoch.to_numpy(),exposure=ex[['start_epoch','end_epoch']].to_numpy()-origin,seizures=np.array([[s['onset'],s['offset']] for s in iv])-origin)
    # Remove conditioned timing summaries from delivered measures.
    result=dict(status='COMPLETE',job=job,tb_fraction=r['tb_fraction'],adjacent_same=r['adjacent_same'],adjacent_excess=r['adjacent_excess'],windows={k:{key:value for key,value in v.items() if key.startswith('tb_share') or key=='n_share_windows'} for k,v in r['windows'].items()})
    if job['rep']==0:
        dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest.with_suffix('.npz'),label_tb=y)
    write_json(dest,result);return result

def main():
    best=best_fits();t=best['full','ou']['theta'];h=selected(RUN/'advanced_controls_v1_2/fits','full','ou_history')[0]['theta'];d=np.load(RUN/'observations.npz');p=d['y'].mean();cfg={'constant':[np.log(p/(1-p)),0,1,0,1,0],'ou':[t[0],0,np.exp(t[1]),np.exp(t[2]),1,0],'ou_history':[h[0],h[1],np.exp(h[2]),np.exp(h[3]),1,0]}
    for history in (False,True):
        rows=[]
        for path in (RUN/'two_timescale_marks_v1_9/fits').glob('full_slow6_hist'+str(int(history))+'_*.json'):rows.append(json.loads(path.read_text()))
        r=max(rows,key=lambda x:x['loglik']);t=r['theta'];cfg['two_ou6h_history' if history else 'two_ou6h']=[t[0],t[1] if history else 0,np.exp(t[-3]),np.exp(t[-2]),6,np.exp(t[-1])]
    jobs=[dict(model=m,rep=i,seed=350000+k*1000+i,parameters=t) for k,(m,t) in enumerate(cfg.items()) for i in range(128)];write_json(OUT/'contract.json',dict(scope='Conditional label generation at exact observed event times, boundaries and coverage',n_replicates=128,parameters=cfg,not_evaluated='Event rate, interval and count distributions are fixed inputs, not generated evidence',two_timescale_limit='v1.9 fast correlation time touches its lower bound; included as a provisional statistical comparator'))
    with ProcessPoolExecutor(max_workers=24) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(complete=i+1,total=len(jobs),model=r['job']['model'],rep=r['job']['rep'],tb=r['tb_fraction'],adjacent_excess=r['adjacent_excess'])),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_runs=len(jobs),finished_unix=time.time()))

if __name__=='__main__':main()
