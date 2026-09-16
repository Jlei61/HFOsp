"""Independent-particle posterior for the two-scale mark model.

Only the proposal is initialized from ADF curvature. The target uses an unbiased
average of independent particle likelihood estimates, retained on rejection.
"""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.special import logsumexp
from numba import cuda
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.gpu_two_timescale import dataset,evaluate
from scripts.patient_state_v1.two_timescale import filter2
OUT=RUN/'two_scale_particle_posterior_v1_15'
def logprior(t):
    val=-.5*np.sum((t[:,:2]/2)**2,axis=1)+t[:,3]+t[:,4]-.5*(np.exp(2*t[:,3])+np.exp(2*t[:,4]));valid=(t[:,2]>=np.log(1/3600))&(t[:,2]<=np.log(3))&(t[:,5]>=np.log(.25))&(t[:,5]<=np.log(96))&(t[:,5]-t[:,2]>=np.log(4))&(t[:,3]>=np.log(.01))&(t[:,3]<=np.log(3))&(t[:,4]>=np.log(.001))&(t[:,4]<=np.log(5));val[~valid]=-np.inf;return val

def center_cov(d):
    rr=[]
    for p in (RUN/'two_timescale_free_tau_v1_13/fits').glob('full_carry1_hist1_adf_*.json'):
        r=json.loads(p.read_text())
        if r['status']=='COMPLETE' and r['feasible']:rr.append(r)
    c=np.array(max(rr,key=lambda r:r['loglik'])['theta']);dim=len(c);h=.002;H=np.empty((dim,dim))
    def f(t):return -filter2(t[:-1],d,np.exp(t[-1]),True)['loglik']-logprior(t[None,:])[0]
    f0=f(c)
    for i in range(dim):
        ei=np.eye(dim)[i]*h;H[i,i]=(f(c+ei)-2*f0+f(c-ei))/h**2
        for j in range(i):
            ej=np.eye(dim)[j]*h;H[i,j]=H[j,i]=(f(c+ei+ej)-f(c+ei-ej)-f(c-ei+ej)+f(c-ei-ej))/(4*h*h)
    val,vec=np.linalg.eigh(H);cov=(vec/np.maximum(val,.02))@vec.T;return c,cov,H

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--chains',type=int,default=32);ap.add_argument('--iterations',type=int,default=2400);ap.add_argument('--warmup',type=int,default=800);ap.add_argument('--likelihood-replicates',type=int,default=4);ap.add_argument('--output-name',default='two_scale_particle_posterior_v1_15');args=ap.parse_args();OUT=RUN/args.output_name;cuda.select_device(args.gpu);OUT.mkdir(exist_ok=True);d=dataset(True);center,cov,H=center_cov(d);rng=np.random.default_rng(450901);dim=len(center);scale=1.;start=time.time();hard_end=1788968996+8.8*3600;samples=np.full((args.iterations+1,args.chains,dim),np.nan);likes=np.full((args.iterations+1,args.chains),np.nan);accepted=np.zeros((args.iterations,args.chains),bool)
    def ll(t,seed):
        pars=np.repeat(t,args.likelihood_replicates,axis=0);v=evaluate(d,pars,1,particles=1024,seed=seed,batch=len(pars));return logsumexp(v.reshape(len(t),args.likelihood_replicates),axis=1)-np.log(args.likelihood_replicates)
    checkpoint=OUT/'checkpoint.npz';meta=OUT/'checkpoint.json'
    if checkpoint.exists():
        z=np.load(checkpoint);m=json.loads(meta.read_text());iteration=m['iteration'];samples[:iteration+1]=z['samples'];likes[:iteration+1]=z['loglikes'];accepted[:iteration]=z['accepted'];theta=samples[iteration].copy();lik=likes[iteration].copy();cov=z['proposal_cov'];scale=m['scale'];rng.bit_generator.state=m['rng_state'];start=m['initial_started']
    else:
        theta=np.empty((args.chains,dim))
        for i in range(args.chains):
            while True:
                t=center+rng.multivariate_normal(np.zeros(dim),cov*2)
                if np.isfinite(logprior(t[None,:])[0]):theta[i]=t;break
        iteration=0;lik=ll(theta,450000);samples[0]=theta;likes[0]=lik
    prior=logprior(theta)
    write_json(OUT/'contract.json',dict(model='Two independent OU components with scalar sum readout plus 1-second label history',boundary='Fast component independent prior at ictal exclusions; slow background carried continuously across real elapsed time',parameter_names=['baseline','history','log_fast_tau_hours','log_fast_sd','log_background_sd','log_background_tau_hours'],priors='Coefficients Normal(0,2); amplitude SDs truncated half-Normal(1), log-SD Jacobians included; uniform ordered log-times on fast 1s to 3h and background 15min to 96h, background >=4 fast',chains=args.chains,iterations=args.iterations,warmup=args.warmup,particles_per_filter=1024,independent_filters_per_likelihood=args.likelihood_replicates,likelihood='Arithmetic average of independent unbiased likelihood estimates, implemented with logsumexp; old value retained on rejection',proposal='ADF curvature for initialization only; covariance/scale adapt during warmup only',hard_end_unix=hard_end,acceptance_gate='Inspect rank-normalized split Rhat, ESS and stickiness; computation completion alone is not posterior acceptance',scope='Exploratory fixed-label development record; no SNN modification or seizure outcome likelihood'))
    def save(status):
        p=OUT/'checkpoint.tmp.npz';np.savez_compressed(p,samples=samples[:iteration+1],loglikes=likes[:iteration+1],accepted=accepted[:iteration],proposal_cov=cov,initial_center=center,initial_hessian=H);p.replace(checkpoint);write_json(meta,dict(status=status,iteration=iteration,warmup=args.warmup,chains=args.chains,scale=scale,rng_state=rng.bit_generator.state,initial_started=start,last_saved=time.time(),elapsed_total=time.time()-start,acceptance=float(accepted[:iteration].mean()) if iteration else None))
    save('RUNNING')
    while iteration<args.iterations and time.time()<hard_end:
        prop=theta+rng.multivariate_normal(np.zeros(dim),cov,args.chains)*scale;lp=logprior(prop);valid=np.isfinite(lp);newll=np.full(args.chains,-np.inf)
        if valid.any():newll[valid]=ll(prop[valid],460000+iteration*1000)
        ok=np.log(rng.random(args.chains))<newll+lp-lik-prior;theta[ok]=prop[ok];lik[ok]=newll[ok];prior[ok]=lp[ok];accepted[iteration]=ok;iteration+=1;samples[iteration]=theta;likes[iteration]=lik
        if iteration<=args.warmup and iteration%25==0:
            rate=accepted[max(0,iteration-100):iteration].mean();scale=float(np.clip(scale*np.exp(.4*(rate-.2)),.04,3))
            if iteration>=100:cov=.9*np.cov(samples[iteration//2:iteration+1].reshape(-1,dim),rowvar=False)+.1*cov+np.eye(dim)*1e-7
        if iteration%25==0:save('RUNNING');print(json.dumps(dict(iteration=iteration,target=args.iterations,acceptance=float(accepted[max(0,iteration-100):iteration].mean()),elapsed=time.time()-start,scale=scale)),flush=True)
    save('COMPLETE' if iteration==args.iterations else 'CHECKPOINTED_TIME_LIMIT')
if __name__=='__main__':main()
