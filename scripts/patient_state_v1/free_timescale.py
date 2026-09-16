"""v1.13: estimate background time, and compare likelihood approximations.

Chronological fits use only their training prefixes. The two OU components are
ordered in correlation time, with a minimum fourfold separation. This is a model
restriction for identifiability, not a discovered biological separation.
"""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.two_timescale import laplace2,filter2
from scripts.patient_state_v1.gpu_two_timescale import dataset
from scripts.patient_state_v1.model import slice_data
from scripts.patient_state_v1.analyze_first import metrics

OUT=RUN/'two_timescale_free_tau_v1_13'

def fit_free(d,history,method,initial,maxiter=150):
    nc=2 if history else 1;bounds=[(-8,8)]*nc+[(np.log(1/3600),np.log(3)),(np.log(.01),np.log(3)),(np.log(.001),np.log(5)),(np.log(.25),np.log(96))];cache={}
    def objective(t):
        key=np.asarray(t).tobytes()
        if key not in cache:
            try:
                ll=laplace2(t[:-1],d,np.exp(t[-1]),history) if method=='laplace' else filter2(t[:-1],d,np.exp(t[-1]),history)['loglik'];cache[key]=-ll
            except (ValueError,np.linalg.LinAlgError,FloatingPointError):cache[key]=1e20
        return cache[key]
    # Linear inequality in log-time coordinates.
    constraint=dict(type='ineq',fun=lambda t:t[-1]-t[nc]-np.log(4))
    opt=minimize(objective,initial,method='SLSQP',bounds=bounds,constraints=[constraint],options={'maxiter':maxiter,'ftol':1e-7,'eps':2e-5})
    return dict(theta=opt.x,loglik=-opt.fun,success=opt.success,message=opt.message,nfev=opt.nfev,feasible=bool(constraint['fun'](opt.x)>=-1e-6),method=method)

def worker(j):
    path=OUT/'fits'/f"{j['id']}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.time()
    try:
        data=dataset(j['carry']);d=slice_data(data,0,j['end']);nc=2 if j['history'] else 1;p=d['y'].mean();initial=np.r_[np.log(p/(1-p)),np.ones(nc-1)*.8,np.log(j['fast_tau']),np.log(.4),np.log(j['slow_sd']),np.log(j['slow_tau'])];r=fit_free(d,j['history'],j['method'],initial);theta=r['theta'];f=filter2(theta[:-1],data,np.exp(theta[-1]),j['history']);r.update(status='COMPLETE',job=j,elapsed=time.time()-start,laplace_at_solution=laplace2(theta[:-1],d,np.exp(theta[-1]),j['history']),adf_at_solution=filter2(theta[:-1],d,np.exp(theta[-1]),j['history'])['loglik'])
        if j['scope']!='full':
            lo,hi=j['end'],j['test_end'];r['forward']=metrics(f['predict_tb'][lo:hi],data['y'][lo:hi],data['n'][lo:hi])
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),**f)
    except Exception:r=dict(status='FAILED',job=j,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(path,r);return r

def main():
    d=np.load(RUN/'observations.npz');folds=json.loads((RUN/'splits.json').read_text());scopes=[dict(scope='full',end=len(d['y']))]+[dict(scope=f"fold{f['fold']}",end=f['train_end'],test_end=f['test_end']) for f in folds];jobs=[]
    for scope in scopes:
        for carry in [False,True]:
            for history in [False,True]:
                for method in ['laplace','adf']:
                    for i,(tf,ts,sds) in enumerate([(.003,1.,.4),(.03,6.,.6),(.03,24.,1.)]):jobs.append(dict(id=f"{scope['scope']}_carry{int(carry)}_hist{int(history)}_{method}_{i}",carry=carry,history=history,method=method,fast_tau=tf,slow_tau=ts,slow_sd=sds,**scope))
    write_json(OUT/'queue.json',jobs);write_json(OUT/'contract.json',dict(question='Is the slower correlation time constrained by the record, and is its estimate sensitive to likelihood approximation?',parameter_order='baseline, optional history, log fast tau, log fast SD, log background SD, log background tau',time_units='hours',constraints='fast tau 1 second to 3h; background tau 15min to 96h; background tau >= 4 * fast tau',scope='Version comparisons are exploratory development-data analyses; no untouched clinical test claim',next_gate='Compare fitted solutions using independent particles before accepting quantitative time constants'))
    with ProcessPoolExecutor(max_workers=24) as ex:
        for f in as_completed([ex.submit(worker,j) for j in jobs]):
            r=f.result();print(json.dumps(dict(id=r['job']['id'],status=r['status'],success=r.get('success'),elapsed=r['elapsed'],loglik=r.get('loglik'))),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs),finished_unix=time.time()))

if __name__=='__main__':main()
