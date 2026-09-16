"""Particle marginal MH: unbiased likelihood estimates retained upon rejection.

64 independent chains per GPU, adaptive proposal during warmup only. Parameter
priors regularize state amplitude; no seizure type or future covariate is used.
"""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from numba import cuda
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import laplace
from scripts.patient_state_v1.analyze_first import best_fits
from scripts.patient_state_v1.advanced_controls import history_data
from scripts.patient_state_v1.gpu_particles import evaluate

OUT=RUN/'particle_posterior_v1_2'

def logprior(t):
    # Coefficient N(0,2^2); log tau uniform over computational envelope;
    # stationary SD half-normal(1), truncated to [.01,5], including log-SD Jacobian.
    values=-.5*np.sum((t[:,:-2]/2)**2,axis=1)+t[:,-1]-.5*np.exp(2*t[:,-1])
    ok=(t[:,-2]>=np.log(1/60))&(t[:,-2]<=np.log(24))&(t[:,-1]>=np.log(.01))&(t[:,-1]<=np.log(5))
    values[~ok]=-np.inf;return values

def expand(theta,model):
    if model=='ou':return np.column_stack([theta[:,0],np.zeros(len(theta)),np.zeros(len(theta)),theta[:,1:]])
    return np.column_stack([theta[:,:2],np.zeros(len(theta)),theta[:,2:]])

def initial_and_cov(data,model):
    if model=='ou':center=np.array(best_fits()['full','ou']['theta'])
    else:
        rows=[json.loads(f.read_text()) for f in (RUN/'advanced_controls_v1_2/fits').glob('full_ou_history*.json')]
        center=np.array(max(rows,key=lambda r:r['loglik'])['theta'])
    def f(t):
        e=expand(t[None,:],model)[0]
        ll=laplace(t,data) if model=='ou' else laplace(e,data,True)
        return -ll-logprior(t[None,:])[0]
    dim=len(center);H=np.empty((dim,dim));h=.002;f0=f(center)
    for i in range(dim):
        ei=np.eye(dim)[i]*h;H[i,i]=(f(center+ei)-2*f0+f(center-ei))/h**2
        for j in range(i):
            ej=np.eye(dim)[j]*h;H[i,j]=H[j,i]=(f(center+ei+ej)-f(center+ei-ej)-f(center-ei+ej)+f(center-ei-ej))/(4*h*h)
    val,vec=np.linalg.eigh(H);cov=(vec/np.maximum(val,1e-3))@vec.T
    return center,cov,H

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--model',choices=['ou','ou_history'],required=True)
    ap.add_argument('--chains',type=int,default=64);ap.add_argument('--iterations',type=int,default=4000);ap.add_argument('--warmup',type=int,default=1000);ap.add_argument('--particles',type=int,default=1024)
    ap.add_argument('--max-hours',type=float,default=5.5);ap.add_argument('--rng-stream',type=int,default=None,help='Preserve the original random stream when resuming on a different physical GPU');args=ap.parse_args();cuda.select_device(args.gpu);stream=args.gpu if args.rng_stream is None else args.rng_stream
    out=OUT/args.model;out.mkdir(parents=True,exist_ok=True);data=dict(np.load(RUN/'observations.npz'))
    if args.model=='ou_history':data=history_data(data)
    center,cov,H=initial_and_cov(data,args.model);dim=len(center);rng=np.random.default_rng(271000+stream);started=time.time();hard_end=min(started+args.max_hours*3600,1788968996+8.75*3600)
    checkpoint=out/'checkpoint.npz';jsonpath=out/'checkpoint.json'
    shape=(args.iterations+1,args.chains,dim);samples=np.full(shape,np.nan);loglikes=np.full((args.iterations+1,args.chains),np.nan);accepted=np.zeros((args.iterations,args.chains),bool)
    scale=2.38/np.sqrt(dim)
    if checkpoint.exists() and jsonpath.exists():
        z=np.load(checkpoint);meta=json.loads(jsonpath.read_text());iteration=int(meta['iteration']);samples[:iteration+1]=z['samples'];loglikes[:iteration+1]=z['loglikes'];accepted[:iteration]=z['accepted'];theta=samples[iteration].copy();ll=loglikes[iteration].copy();cov=z['proposal_cov'];scale=float(meta['scale']);rng.bit_generator.state=meta['rng_state'];initial_started=meta['initial_started']
    else:
        iteration=0;theta=center+rng.multivariate_normal(np.zeros(dim),cov*2,args.chains)
        theta[:,-2]=np.clip(theta[:,-2],np.log(1/60)+.01,np.log(24)-.01);theta[:,-1]=np.clip(theta[:,-1],np.log(.01)+.01,np.log(5)-.01)
        ll=evaluate(data,expand(theta,args.model),args.particles,seed=270000+stream*10000,batch=args.chains);samples[0]=theta;loglikes[0]=ll;initial_started=started
    prior=logprior(theta)
    write_json(out/'contract.json',dict(model=args.model,chains=args.chains,iterations=args.iterations,warmup=args.warmup,particles=args.particles,
        parameter_names=['baseline_log_odds']+(['fast_history_coefficient'] if args.model=='ou_history' else [])+['log_tau_hours','log_stationary_sd'],
        priors='coefficients Normal(0,2); log tau uniform(log(1/60),log(24)); stationary SD half-normal(1), truncated [.01,5]',
        likelihood='bootstrap-particle unbiased likelihood estimate; reject retains old estimate; exact irregular OU transitions',
        adaptation='global random-walk covariance and scale adjusted in warmup only',goal_start_unix=1788968996,hard_end_unix=hard_end,execution_gpu=args.gpu,rng_stream=stream,
        claim_scope='conditional mark-process parameter posterior under one model; not clinical prediction or physical drift identification'))
    def save(status):
        tmp=out/'checkpoint.tmp.npz';np.savez_compressed(tmp,samples=samples[:iteration+1],loglikes=loglikes[:iteration+1],accepted=accepted[:iteration],proposal_cov=cov,initial_center=center,initial_hessian=H);tmp.replace(checkpoint)
        write_json(jsonpath,dict(status=status,iteration=iteration,warmup=args.warmup,chains=args.chains,scale=scale,rng_state=rng.bit_generator.state,
             initial_started=initial_started,last_saved=time.time(),elapsed_total=time.time()-initial_started,acceptance=float(accepted[:iteration].mean()) if iteration else None))
    save('RUNNING')
    while iteration<args.iterations and time.time()<hard_end:
        proposal=theta+rng.multivariate_normal(np.zeros(dim),cov,args.chains)*scale;lp=logprior(proposal);valid=np.isfinite(lp);pll=np.full(args.chains,-np.inf)
        if valid.any():pll[valid]=evaluate(data,expand(proposal[valid],args.model),args.particles,seed=280000+iteration*10+stream,batch=args.chains)
        accept=np.log(rng.uniform(size=args.chains))<(pll+lp-ll-prior)
        theta[accept]=proposal[accept];ll[accept]=pll[accept];prior[accept]=lp[accept];accepted[iteration]=accept;iteration+=1;samples[iteration]=theta;loglikes[iteration]=ll
        if iteration<=args.warmup and iteration%25==0:
            rate=accepted[max(0,iteration-100):iteration].mean();scale=float(np.clip(scale*np.exp(.3*(rate-.23)),.05,4))
            if iteration>=100:
                pool=samples[max(0,iteration//2):iteration+1].reshape(-1,dim);cov=.9*np.cov(pool,rowvar=False)+.1*cov+np.eye(dim)*1e-7
        if iteration%25==0:
            save('RUNNING');print(json.dumps(dict(iteration=iteration,target=args.iterations,model=args.model,acceptance=float(accepted[max(0,iteration-100):iteration].mean()),elapsed_seconds=time.time()-initial_started,scale=scale)),flush=True)
    save('COMPLETE' if iteration==args.iterations else 'CHECKPOINTED_TIME_LIMIT')

if __name__=='__main__':main()
