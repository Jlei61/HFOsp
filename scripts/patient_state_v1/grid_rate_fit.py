"""Refit the same renewal OU model using deterministic state integration."""
import sys,json,time,argparse,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.grid_renewal import evaluate
from scripts.patient_state_v1.renewal import prepare

OUT=RUN/'renewal_grid_refit_v1_7'

def worker(job):
    path=OUT/'fits'/f"{job['id']}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.time();calls=0;trace=[]
    try:
        full=prepare(job['seconds'],.25);d={k:v[:job['end']].copy() for k,v in full.items() if np.ndim(v)>0}
        def objective(theta):
            nonlocal calls
            calls+=1;r=evaluate(d,theta,job['points']);trace.append(dict(call=calls,theta=theta.copy(),loglik=r['loglik'],boundary_loss=r['max_transition_mass_loss'],elapsed=time.time()-start))
            if calls%5==0:write_json(OUT/'progress'/f"{job['id']}.json",dict(status='RUNNING',calls=calls,best=max(trace,key=lambda t:t['loglik']),elapsed=time.time()-start))
            return -r['loglik']
        opt=minimize(objective,job['initial'],method='L-BFGS-B',bounds=[(np.log(.01),np.log(1e7)),(np.log(1/60),np.log(24)),(np.log(.01),np.log(5))],options={'maxiter':100,'ftol':1e-10,'eps':.0005,'maxls':15})
        result=dict(status='COMPLETE',job=job,theta=opt.x,loglik=-opt.fun,success=opt.success,message=opt.message,nfev=opt.nfev,trace=trace,elapsed=time.time()-start)
        verify=evaluate(d,opt.x,job['points']*2);mean=verify.pop('mean');variance=verify.pop('variance');result['refined_grid']=verify
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),mean=mean,variance=variance)
    except Exception:result=dict(status='FAILED',job=job,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(path,result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=10);args=ap.parse_args();fits={}
    for p in (RUN/'renewal_observation_v1_3/fits').glob('*.json'):
        r=json.loads(p.read_text());j=r['job'];key=(j['seconds'],j['scope'])
        if r['status']=='COMPLETE' and j['deadtime']==.25 and j['model']=='rate' and j['seconds'] in (1,5):
            if key not in fits or r['loglik']>fits[key]['loglik']:fits[key]=r
    jobs=[]
    for (seconds,scope),r in fits.items():
        if seconds==1 and scope!='full':continue
        for i,shift in enumerate((0.,-.5)):
            t=np.array(r['theta']);t[0]+=shift;t[2]+=(-.1 if i else 0.)
            jobs.append(dict(id=f'g{seconds}_{scope}_m512_{i}',seconds=seconds,scope=scope,end=r['job']['end'],points=512,initial=t,laplace_theta=r['theta'],laplace_loglik=r['loglik']))
    write_json(OUT/'queue.json',jobs)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(worker,j) for j in jobs]):
            r=f.result();print(json.dumps(dict(id=r['job']['id'],status=r['status'],elapsed=r['elapsed'],theta=r.get('theta'),loglik=r.get('loglik')),default=lambda x:x.tolist()),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs),finished_unix=time.time()))

if __name__=='__main__':main()
