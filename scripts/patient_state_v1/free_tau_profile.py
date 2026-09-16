"""Profile background correlation time, without seizure outcome selection."""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.gpu_two_timescale import dataset
from scripts.patient_state_v1.two_timescale import laplace2,filter2
OUT=RUN/'free_tau_profile_v1_14'
def worker(j):
    path=OUT/'fits'/f"carry{int(j['carry'])}_{j['method']}_tau{j['tau']:.6f}_init{j['init']}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.time()
    try:
        d=dataset(j['carry']);t=np.array(j['initial']);t[2]=min(t[2],np.log(j['tau']/4));bounds=[(-8,8),(-8,8),(np.log(1/3600),np.log(min(3,j['tau']/4))),(np.log(.01),np.log(3)),(np.log(.001),np.log(5))]
        def obj(t):return -(laplace2(t,d,j['tau'],True) if j['method']=='laplace' else filter2(t,d,j['tau'],True)['loglik'])
        opt=minimize(obj,t,method='L-BFGS-B',bounds=bounds,options={'maxiter':150,'ftol':1e-10,'eps':2e-5,'maxls':30});r=dict(status='COMPLETE',job=j,theta=opt.x,loglik=-opt.fun,success=opt.success,message=opt.message,elapsed=time.time()-start,adf_at_solution=filter2(opt.x,d,j['tau'],True)['loglik'],laplace_at_solution=laplace2(opt.x,d,j['tau'],True))
    except Exception:r=dict(status='FAILED',job=j,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(path,r);return r

def main():
    taus=np.array([.25,.4,.65,1,1.5,2.25,3.5,6,10,16,24,48,96]);best={}
    for path in (RUN/'two_timescale_free_tau_v1_13/fits').glob('full_*hist1_*.json'):
        r=json.loads(path.read_text());j=r['job'];k=(j['carry'],j['method'])
        if r['status']=='COMPLETE' and r['feasible'] and (k not in best or r['loglik']>best[k]['loglik']):best[k]=r
    jobs=[]
    for (carry,method),r in best.items():
        for tau in taus:
            for i in range(2):
                t=np.array(r['theta'][:-1]);t[2]+=np.log(2) if i else 0
                jobs.append(dict(carry=carry,method=method,tau=float(tau),init=i,initial=t))
    write_json(OUT/'contract.json',dict(question='Does the record constrain slower mean reversion after short-time dependence is represented?',history=True,background_tau_hours=taus,likelihood_methods=['Laplace','Gaussian ADF'],limits='Approximate profile likelihood; differences between approximations audited independently. No clinical seizure label enters model selection.',jobs=len(jobs)))
    with ProcessPoolExecutor(max_workers=20) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(complete=i+1,total=len(jobs),status=r['status'],carry=r['job']['carry'],method=r['job']['method'],tau=r['job']['tau'],elapsed=r['elapsed'])),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs)))
if __name__=='__main__':main()
