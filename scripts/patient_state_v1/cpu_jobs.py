"""Bounded parallel fit/synthetic queue. Each result is written immediately."""
import sys,json,time,argparse,traceback,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import fit,slice_data,simulate,filter_adf,laplace

def worker(job):
    start=time.time();out=RUN/'cpu_fits'/f"{job['id']}.json"
    if out.exists():return json.loads(out.read_text())
    try:
        data=dict(np.load(RUN/'observations.npz'));d=slice_data(data,0,job['end'])
        if job.get('synthetic'):
            syn=job['synthetic'];y,s=simulate(d,syn['b'],syn['tau'],syn['sd'],syn['seed'],syn.get('cyclic',(0.,0.)));d['y']=y
        nc=3 if 'cycle' in job['model'] else 1
        p=np.clip(d['y'].sum()/d['n'].sum(),1e-4,1-1e-4)
        init=np.r_[np.log(p/(1-p)),np.zeros(nc-1),np.log(job.get('tau0',1)),np.log(job.get('sd0',.6))]
        result=fit(d,job['model'],init,maxiter=job.get('maxiter',150))
        if job['model'].startswith('ou'):
            a=filter_adf(np.array(result['theta']),d,'cycle' in job['model'])
            result['adf_loglik']=a['loglik']
            result['laplace_minus_adf']=result['loglik']-a['loglik']
        result.update(job=job,elapsed_seconds=time.time()-start,status='COMPLETE')
    except Exception:
        result=dict(job=job,status='FAILED',traceback=traceback.format_exc(),elapsed_seconds=time.time()-start)
    write_json(out,result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wave',choices=['real','synthetic'],default='real');ap.add_argument('--workers',type=int,default=20);args=ap.parse_args()
    data=np.load(RUN/'observations.npz');length=len(data['y']);folds=json.loads((RUN/'splits.json').read_text())
    scopes=[('full',length)]+[(f"fold{f['fold']}",f['train_end']) for f in folds]
    jobs=[]
    if args.wave=='real':
        for scope,end in scopes:
            for model in ('constant','cycle','ou','ou_cycle'):
                initial=[(.05,.25),(.25,.6),(1.,1.2),(6.,.6),(20.,2.)] if model.startswith('ou') else [(1.,.6)]
                for i,(tau,sd) in enumerate(initial):jobs.append(dict(id=f'{scope}_{model}_init{i}',scope=scope,end=end,model=model,tau0=tau,sd0=sd))
    else:
        scenarios=[('constant',1.,0.,(0.,0.)),('cycle_null',1.,0.,(.6,-.4))]
        scenarios += [(f'ou_t{tau}_s{sd}',tau,sd,(0.,0.)) for tau in (1/6,1.,6.) for sd in (.3,1.)]
        for scenario,tau,sd,cyclic in scenarios:
            for rep in range(6):
                syn=dict(b=-.9,tau=tau,sd=sd,cyclic=cyclic,seed=26091000+rep+100*len(jobs))
                for model in ('constant','cycle','ou','ou_cycle'):
                    initial=[(.15,.4),(3.,1.2)] if model.startswith('ou') else [(1.,.6)]
                    for i,(t0,s0) in enumerate(initial):jobs.append(dict(id=f'syn_{scenario}_rep{rep}_{model}_init{i}',scope='synthetic',end=length,model=model,tau0=t0,sd0=s0,synthetic=syn,maxiter=100))
    write_json(RUN/f'{args.wave}_queue.json',jobs)
    print(json.dumps(dict(wave=args.wave,total=len(jobs),workers=args.workers,pid=os.getpid())),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fs={ex.submit(worker,j):j for j in jobs}
        for i,f in enumerate(as_completed(fs)):
            r=f.result();print(json.dumps(dict(complete=i+1,total=len(jobs),id=r['job']['id'],status=r['status'],loglik=r.get('loglik'),tau=r.get('tau_hours'),sd=r.get('stationary_sd'),elapsed=r['elapsed_seconds'])),flush=True)
    write_json(RUN/f'{args.wave}_queue_status.json',dict(status='COMPLETE',n_jobs=len(jobs),finished_unix=time.time()))

if __name__=='__main__':main()
