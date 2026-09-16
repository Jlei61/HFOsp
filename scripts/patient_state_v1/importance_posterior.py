"""Independent hyperparameter importance sampling with unbiased particle weights.

A frozen, explicitly evaluated Student-t mixture proposal may be constructed
from earlier MCMC draws without requiring those draws to have converged. New
independent proposal draws and new particle randomness target the same posterior.
CPU Laplace weights are a separate approximation comparison, never substituted
for the particle weights. Importance diagnostics and independent batches govern
acceptance; a completed array alone does not establish adequate importance ESS.
"""
import sys,json,time,argparse,multiprocessing
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.stats import multivariate_t
from scipy.special import logsumexp
from concurrent.futures import ProcessPoolExecutor,as_completed
from numba import cuda
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.particle_mcmc import logprior,expand
from scripts.patient_state_v1.advanced_controls import history_data
from scripts.patient_state_v1.model import laplace
from scripts.patient_state_v1.gpu_particles import evaluate
OUT=RUN/'independent_importance_posterior_v1_28'

def laplace_chunk(job):
    model,theta=job;d=dict(np.load(RUN/'observations.npz'));d=history_data(d) if model=='ou_history' else d;answer=np.full(len(theta),-np.inf);valid=np.isfinite(logprior(theta))
    for i in np.flatnonzero(valid):answer[i]=laplace(expand(theta[i:i+1],model)[0],d,True) if model=='ou_history' else laplace(theta[i],d)
    return answer

def proposal(model,draws,replicates):
    root=RUN/'particle_posterior_v1_2'/model;z=dict(np.load(root/'checkpoint.npz'));meta=json.loads((root/'checkpoint.json').read_text());contract=json.loads((root/'contract.json').read_text());samples=z['samples'];iteration=min(meta['iteration'],len(samples)-1);warmup=contract['warmup'];pool=samples[max(warmup+1,iteration-999):iteration+1].reshape(-1,samples.shape[-1]);assert len(pool)>1000;mu=pool.mean(0);cov=np.cov(pool,rowvar=False);values,vectors=np.linalg.eigh(cov);cov=(vectors*np.maximum(values,1e-8))@vectors.T;means=[mu,z['initial_center']];scales=[cov*1.5,cov*8.];weights=np.array([.85,.15]);theta=[];reps=[]
    for rep in range(replicates):
        rng=np.random.default_rng(981000+(10000 if model=='ou_history' else 0)+rep);group=rng.choice(2,draws,p=weights);x=np.empty((draws,len(mu)))
        for k in range(2):
            ix=np.flatnonzero(group==k);noise=rng.multivariate_normal(np.zeros(len(mu)),scales[k],len(ix));x[ix]=means[k]+noise/np.sqrt(rng.chisquare(5,len(ix))/5)[:,None]
        theta.append(x);reps.append(np.full(draws,rep))
    theta=np.concatenate(theta);rep=np.concatenate(reps);logq=logsumexp(np.array([np.log(weights[k])+multivariate_t.logpdf(theta,loc=means[k],shape=scales[k],df=5) for k in range(2)]),axis=0)
    return theta,rep,logq,dict(source_checkpoint=meta,source_iteration=iteration,source_window_start=max(warmup+1,iteration-999),means=means,scales=scales,mixture_weights=weights,student_df=5,parameter_names=contract['parameter_names'],independence='Proposal is frozen before generating all new independent parameter draws and all new particle random numbers; source-chain convergence is not assumed')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['ou','ou_history'],required=True);ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--replicates',type=int,default=4);ap.add_argument('--draws',type=int,default=2048);args=ap.parse_args();root=OUT/args.model;root.mkdir(parents=True,exist_ok=True);path=root/'proposal_draws.npz';started=time.time()
    if path.exists():
        z=dict(np.load(path));theta,rep,logq=z['theta'],z['replicate'],z['logq'];proposal_info=json.loads((root/'proposal.json').read_text())
    else:
        theta,rep,logq,proposal_info=proposal(args.model,args.draws,args.replicates);np.savez_compressed(path,theta=theta,replicate=rep,logq=logq);write_json(root/'proposal.json',proposal_info)
    assert len(theta)==args.draws*args.replicates;prior=logprior(theta);N=len(theta);ll=np.full(N,np.nan);lp=np.full(N,np.nan)
    write_json(root/'contract.json',dict(model=args.model,n_proposals=N,independent_replicates=args.replicates,draws_per_replicate=args.draws,particles_per_likelihood=1024,proposal='Frozen explicit mixture of multivariate Student t(df5), full real support; priors enforce same envelope as PMMH',target='Same conditional mark-process posterior and priors as PMMH; weight = unbiased particle likelihood * prior / proposal density',cpu_comparison='Independent Laplace likelihood at every identical parameter point, reported separately',acceptance='Assess raw importance ESS, maximum normalized weight, Pareto tail diagnostic, weighted Monte Carlo errors, and independent-replicate consistency; do not accept on completion alone',scope='Numerical posterior under one stated model; not patient model adequacy or physiological identification',source=proposal_info))
    chunks=[(i,min(i+128,N)) for i in range(0,N,128)]
    with ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(laplace_chunk,(args.model,theta[lo:hi])):(lo,hi) for lo,hi in chunks};cuda.select_device(args.gpu);d=dict(np.load(RUN/'observations.npz'));d=history_data(d) if args.model=='ou_history' else d
        for lo in range(0,N,64):
            hi=min(lo+64,N);batchpath=root/'particle_batches'/f'{lo:05d}.npz'
            if batchpath.exists():ll[lo:hi]=np.load(batchpath)['loglik']
            else:
                valid=np.isfinite(prior[lo:hi]);values=np.full(hi-lo,-np.inf)
                if valid.any():values[valid]=evaluate(d,expand(theta[lo:hi][valid],args.model),1024,seed=990000+(100000 if args.model=='ou_history' else 0)+lo,batch=64)
                ll[lo:hi]=values;batchpath.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(batchpath,loglik=values)
            if (lo//64+1)%8==0:print(json.dumps(dict(model=args.model,proposals_done=hi,total=N,elapsed=time.time()-started)),flush=True);write_json(root/'progress.json',dict(status='RUNNING',proposals_done=hi,total=N,elapsed=time.time()-started))
        for future in as_completed(futures):
            lo,hi=futures[future];lp[lo:hi]=future.result()
    assert not np.isnan(ll).any() and not np.isnan(lp).any();np.savez_compressed(root/'weighted_samples.npz',theta=theta,replicate=rep,logq=logq,logprior=prior,particle_loglik=ll,laplace_loglik=lp,particle_logweight=ll+prior-logq,laplace_logweight=lp+prior-logq);write_json(root/'status.json',dict(status='COMPLETE',n_proposals=N,n_particle_evaluations=int(np.isfinite(prior).sum()),elapsed=time.time()-started,diagnostics='PENDING'));print(json.dumps(dict(model=args.model,status='COMPLETE',elapsed=time.time()-started)),flush=True)

if __name__=='__main__':main()
