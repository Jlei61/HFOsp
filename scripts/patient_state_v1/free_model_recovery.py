"""Conditional synthetic recovery of free two-scale statistical parameters."""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.gpu_two_timescale import dataset
from scripts.patient_state_v1.two_scale_prefix import simulate
from scripts.patient_state_v1.advanced_controls import history_data
from scripts.patient_state_v1.free_timescale import fit_free
OUT=RUN/'free_model_recovery_v1_15'
def worker(j):
    p=OUT/'fits'/f"carry{int(j['carry'])}_{j['rep']:03d}.json"
    if p.exists():return json.loads(p.read_text())
    start=time.time()
    try:
        d=dataset(j['carry']);t=np.array(j['theta']);d['y']=simulate(d['dt'],d['reset'],d['slow_dt'],d['slow_reset'],t[0],t[1],np.exp(t[2]),np.exp(t[3]),np.exp(t[5]),np.exp(t[4]),j['seed']);d=history_data(d);r=[]
        for i in range(2):
            initial=t.copy();initial[0]=np.log((d['y'].mean()+1e-6)/(1-d['y'].mean()+1e-6))
            if i:initial[2]+=np.log(3);initial[3]-=.3;initial[5]+=np.log(2)
            r.append(fit_free(d,True,'adf',initial))
        good=[x for x in r if x['feasible']];best=max(good,key=lambda x:x['loglik']);result=dict(status='COMPLETE',job=j,fit=best,all_fits=r,elapsed=time.time()-start)
    except Exception:result=dict(status='FAILED',job=j,traceback=traceback.format_exc(),elapsed=time.time()-start)
    write_json(p,result);return result

def main():
    jobs=[];truth={}
    for carry in [False,True]:
        rr=[]
        for p in (RUN/'two_timescale_free_tau_v1_13/fits').glob(f'full_carry{int(carry)}_hist1_adf_*.json'):
            r=json.loads(p.read_text())
            if r['status']=='COMPLETE' and r['feasible']:rr.append(r)
        t=max(rr,key=lambda x:x['loglik'])['theta'];truth[str(carry)]=t
        for rep in range(64):jobs.append(dict(carry=carry,rep=rep,theta=t,seed=490000+carry*1000+rep))
    write_json(OUT/'contract.json',dict(question='Can the free background time and diffusion amplitudes be recovered under the fitted statistical model?',replicates_per_boundary=64,truth=truth,observations='Generate labels at actual event times; rebuild short-history input from generated labels',fitting='Two starts per generated full record using Gaussian ADF likelihood; optimization failures and boundary hits retained',limits='Conditional synthetic recovery checks the method under its own model; not evidence that the physical SNN state has these parameters'))
    with ProcessPoolExecutor(max_workers=24) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(complete=i+1,total=len(jobs),status=r['status'],carry=r['job']['carry'],elapsed=r['elapsed'],success=r.get('fit',{}).get('success'))),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs)))
if __name__=='__main__':main()
