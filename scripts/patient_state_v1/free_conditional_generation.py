"""Conditional-label checks of the numerically audited free two-scale models."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.gpu_two_timescale import dataset
from scripts.patient_state_v1.two_scale_prefix import simulate
from scripts.patient_state_v1.generate import summaries
OUT=RUN/'free_conditional_generation_v1_15'
def worker(j):
    p=OUT/'runs'/f"{j['model']}_{j['rep']:03d}.json"
    if p.exists():return json.loads(p.read_text())
    d=dataset(j['carry']);t=j['theta'];y=simulate(d['dt'],d['reset'],d['slow_dt'],d['slow_reset'],t[0],t[1],np.exp(t[2]),np.exp(t[3]),np.exp(t[5]),np.exp(t[4]),j['seed']);ex=pd.read_csv(RUN/'exposure.csv');ev=pd.read_csv(RUN/'events.csv');iv=json.loads((RUN/'seizures.json').read_text());audit=json.loads((RUN/'data_audit.json').read_text());origin=float(np.load(RUN/'observations.npz')['origin_epoch']);r=summaries(ev.start_epoch.to_numpy()-origin,y,audit['observed_interictal_hours'],segments=ev.coverage_segment.to_numpy()*1000+ev.interictal_epoch.to_numpy(),exposure=ex[['start_epoch','end_epoch']].to_numpy()-origin,seizures=np.array([[s['onset'],s['offset']] for s in iv])-origin)
    result=dict(status='COMPLETE',job=j,tb_fraction=r['tb_fraction'],adjacent_same=r['adjacent_same'],adjacent_excess=r['adjacent_excess'],windows={k:{kk:vv for kk,vv in v.items() if kk.startswith('tb_share') or kk=='n_share_windows'} for k,v in r['windows'].items()});write_json(p,result)
    if j['rep']==0:np.savez_compressed(p.with_suffix('.npz'),label_tb=y)
    return result

def main():
    best={}
    for p in (RUN/'two_timescale_free_tau_v1_13/fits').glob('full_*hist1_*.json'):
        r=json.loads(p.read_text());j=r['job'];k=(j['carry'],j['method'])
        if r['status']=='COMPLETE' and r['feasible'] and (k not in best or r['loglik']>best[k]['loglik']):best[k]=r
    jobs=[]
    for k,((carry,method),r) in enumerate(sorted(best.items())):
        for rep in range(128):jobs.append(dict(model=f'{method}_carry{int(carry)}',carry=carry,theta=r['theta'],rep=rep,seed=480000+k*1000+rep))
    write_json(OUT/'contract.json',dict(scope='Conditional label generation at actual event times and observation/ictal boundaries',replicates=128,models='Free two-OU timescales plus 1-second label history; Gaussian ADF and Laplace fits; background carry sensitivity',not_claimed='No event timing, seizure type, or physical state trajectory generated from patient data',likelihood_audit='free_timescale_particle_audit_v1_14'))
    with ProcessPoolExecutor(max_workers=16) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(complete=i+1,total=len(jobs),model=r['job']['model'],tb=r['tb_fraction'],adjacent_excess=r['adjacent_excess'])),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_runs=len(jobs)))
if __name__=='__main__':main()
