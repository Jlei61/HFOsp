"""Conditional label generation with the observed causal activity covariate."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from numba import njit
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.two_scale_activity import dataset
from scripts.patient_state_v1.generate import summaries
OUT=RUN/'activity_mark_generation_v1_16'
@njit(cache=True)
def simulate(dt,reset,sdt,sreset,cov,t,seed):
    np.random.seed(seed);b,gamma,coef=t[:3];tf,sdf,sds,ts=np.exp(t[3:]);f=0.;s=0.;previous=.5;y=np.empty(len(dt),np.int64)
    for i in range(len(dt)):
        if reset[i]:f=np.random.normal()*sdf;h=0.
        else:
            a=np.exp(-dt[i]/tf);f=a*f+sdf*np.sqrt(-np.expm1(-2*dt[i]/tf))*np.random.normal();h=(previous-.5)*np.exp(-dt[i]*3600)
        if sreset[i]:s=np.random.normal()*sds
        else:
            a=np.exp(-sdt[i]/ts);s=a*s+sds*np.sqrt(-np.expm1(-2*sdt[i]/ts))*np.random.normal()
        p=1/(1+np.exp(-(b+gamma*h+coef*cov[i]+f+s)));y[i]=int(np.random.random()<p);previous=y[i]
    return y

def worker(j):
    p=OUT/'runs'/f"{j['model']}_{j['rep']:03d}.json"
    if p.exists():return json.loads(p.read_text())
    d,_=dataset(j['end'],j['minutes'],j['carry']);t=np.array(j['theta']);y=simulate(d['dt'],d['reset'],d['slow_dt'],d['slow_reset'],d['x'][:,2],t,j['seed']);ex=pd.read_csv(RUN/'exposure.csv');ev=pd.read_csv(RUN/'events.csv');iv=json.loads((RUN/'seizures.json').read_text());audit=json.loads((RUN/'data_audit.json').read_text());origin=float(np.load(RUN/'observations.npz')['origin_epoch']);r=summaries(ev.start_epoch.to_numpy()-origin,y,audit['observed_interictal_hours'],segments=ev.coverage_segment.to_numpy()*1000+ev.interictal_epoch.to_numpy(),exposure=ex[['start_epoch','end_epoch']].to_numpy()-origin,seizures=np.array([[s['onset'],s['offset']] for s in iv])-origin);result=dict(status='COMPLETE',job=j,tb_fraction=r['tb_fraction'],adjacent_excess=r['adjacent_excess'],windows={k:{kk:vv for kk,vv in v.items() if kk.startswith('tb_share') or kk=='n_share_windows'} for k,v in r['windows'].items()});write_json(p,result)
    if j['rep']==0:np.savez_compressed(p.with_suffix('.npz'),label_tb=y)
    return result

def main():
    best={}
    for p in (RUN/'two_scale_activity_v1_16/fits').glob('full_*.json'):
        r=json.loads(p.read_text());j=r['job'];k=(j['carry'],j['minutes'])
        if r['status']=='COMPLETE' and r['feasible'] and (k not in best or r['loglik']>best[k]['loglik']):best[k]=r
    assert len(best)==4;jobs=[]
    for k,((carry,minutes),r) in enumerate(sorted(best.items())):
        for rep in range(64):jobs.append(dict(model=f'activity{minutes}m_carry{int(carry)}',carry=carry,minutes=minutes,end=r['job']['end'],theta=r['theta'],rep=rep,seed=520000+k*1000+rep))
    write_json(OUT/'contract.json',dict(scope='Generate labels conditional on actual event times, coverage, exclusions and causal past activity; no event-time generation',n_replicates_per_model=64,history='Generated previous labels, not replayed observed labels',question='Does measured past activity remove excess persistence in generated mode fractions?'))
    with ProcessPoolExecutor(max_workers=24) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(complete=i+1,total=len(jobs),model=r['job']['model'],adjacent_excess=r['adjacent_excess'])),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_runs=len(jobs)))
if __name__=='__main__':main()
