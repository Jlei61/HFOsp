"""Distinguish non-Gaussian drift from evidence specifically for two wells."""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.joint_counts import prepare
from scripts.patient_state_v1.nonlinear_drift import evaluate
from scripts.patient_state_v1.nonlinear_calibration import best_real
OUT=RUN/'nonlinear_shape_profile_v1_23'
def worker(j):
    path=OUT/'fits'/f"{j['id']}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.time()
    try:
        d=prepare(j['seconds']);initial=np.asarray(j['initial']);bounds=[(-3,1),(np.log(1/3600),np.log(48)),(np.log(.05),np.log(5))]
        def f(t):return -evaluate(np.r_[t,j['k']],d,'quartic')['loglik']
        opt=minimize(f,initial,method='L-BFGS-B',bounds=bounds,options={'maxiter':150,'ftol':1e-10,'eps':2e-5,'maxls':25});r=dict(status='COMPLETE',job=j,elapsed=time.time()-start,theta=np.r_[opt.x,j['k']],loglik=-opt.fun,success=bool(opt.success),message=str(opt.message))
    except Exception:r=dict(status='FAILED',job=j,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(path,r);return r
def main():
    best=best_real();jobs=[];ks=[-2,-1.5,-1,-.75,-.5,-.25,0,.25,.5,1,2,4,8,20]
    for sec in [15,60]:
        for shape in ks:
            for i in range(2):
                t=np.array(best[(sec,'full','quartic')]['theta'][:3])
                if i==1:
                    ou=best[(sec,'full','ou')]['theta'];scale=max(shape,1.);t=np.array([ou[0],ou[1]+np.log(scale),ou[2]+.5*np.log(scale)])
                jobs.append(dict(id=f'b{sec}_k{shape}_i{i}',seconds=sec,k=shape,initial=t))
    write_json(OUT/'contract.json',dict(question='Does the likelihood require negative curvature, or only departure from a Gaussian stationary OU?',shape='k<0 allows two wells; k>=0 is nonlinear but single-well',method='Fix k and refit baseline, time and amplitude using two starts; identical observation bins/grid',readout='Profile support on both sides of zero; rejecting OU alone does not establish bistability',limits='Model-conditional likelihood profile, not a calibrated confidence interval or neural potential',n_jobs=len(jobs)))
    with ProcessPoolExecutor(max_workers=16) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(done=i+1,total=len(jobs),id=r['job']['id'],status=r['status'],elapsed=r['elapsed'],success=r.get('success'))),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs)))
if __name__=='__main__':main()
