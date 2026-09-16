"""Test whether joint-generation failure depends on Laplace parameter fitting."""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.renewal import prepare
from scripts.patient_state_v1.joint_two_state_filter import evaluate
OUT=RUN/'joint_adf_refit_v1_23'
def worker(j):
    path=OUT/'fits'/f"{j['id']}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.time()
    try:
        r=json.loads(Path(j['source']).read_text());job=r['job'];full=prepare(5,.25);d={k:v[:job['end']].copy() for k,v in full.items() if np.ndim(v)>0};coupled=job['coupled'];initial=np.asarray(r['theta']);initial[0]+=j['shift'];bounds=[(-8,8),(np.log(.01),np.log(1e7))]+([(-5,5)] if coupled else [])+[(np.log(1/60),np.log(24)),(np.log(.01),np.log(5)),(np.log(1/3600),np.log(24)),(np.log(.01),np.log(5))]
        def objective(t):return -evaluate(t,d,coupled,80,True)['loglik']
        opt=minimize(objective,initial,method='L-BFGS-B',bounds=bounds,options={'maxiter':160,'ftol':1e-10,'eps':2e-5,'maxls':25});f=evaluate(opt.x,full,coupled,80,True);ans=dict(status='COMPLETE',job=j,source_job=job,theta=opt.x,loglik=-opt.fun,success=bool(opt.success),message=str(opt.message),nfev=opt.nfev,elapsed=time.time()-start);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),**f)
    except Exception:ans=dict(status='FAILED',job=j,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(path,ans);return ans
def main():
    best={}
    for p in (RUN/'joint_two_state_v1_20/fits').glob('b5_*.json'):
        r=json.loads(p.read_text());j=r['job'];key=(j['scope'],j['coupled'])
        if r['status']=='COMPLETE' and (key not in best or r['loglik']>best[key][1]['loglik']):best[key]=(p,r)
    assert len(best)==8;jobs=[]
    for (scope,coupled),(p,r) in best.items():
        for k,shift in enumerate([-.3,0.,.3]):jobs.append(dict(id=f'{scope}_c{int(coupled)}_i{k}',scope=scope,coupled=coupled,source=str(p),shift=shift))
    write_json(OUT/'contract.json',dict(question='Does optimizing a different marginal-likelihood approximation change the coupled mode/rate parameters and generative mismatch?',same_model='Same two independent OU states and optional state-to-rate observation coupling; all data, strict cutoffs and support unchanged',fit='Sequential Gaussian moment filter, rate then mark, 80-point quadrature; three training-only starts around the same-scope Laplace fit',limits='ADF is still approximate; quadrature/projection-order checks and new generated sequences required before interpretation',n_jobs=len(jobs)))
    with ProcessPoolExecutor(max_workers=16) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(done=i+1,total=len(jobs),id=r['job']['id'],status=r['status'],elapsed=r['elapsed'],success=r.get('success'))),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs)))
if __name__=='__main__':main()
