"""Add fixed-size independent groups using the original frozen proposal."""
import sys,json,time,argparse,multiprocessing
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.stats import multivariate_t
from scipy.special import logsumexp
from concurrent.futures import ProcessPoolExecutor,as_completed
from numba import cuda
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.importance_posterior import OUT,laplace_chunk
from scripts.patient_state_v1.particle_mcmc import logprior,expand
from scripts.patient_state_v1.advanced_controls import history_data
from scripts.patient_state_v1.gpu_particles import evaluate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['ou','ou_history'],required=True);ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--extension',type=int,default=1);args=ap.parse_args();assert args.extension>=1;source=OUT/args.model;root=source/f'extension_{args.extension}';root.mkdir(exist_ok=True);info=json.loads((source/'proposal.json').read_text());contract=json.loads((source/'contract.json').read_text());theta=[];reps=[];started=time.time();means=[np.array(x) for x in info['means']];scales=[np.array(x) for x in info['scales']];mix=np.array(info['mixture_weights'])
    for rep in range(4*args.extension,4*args.extension+4):
        rng=np.random.default_rng(981000+(10000 if args.model=='ou_history' else 0)+rep);group=rng.choice(2,2048,p=mix);x=np.empty((2048,len(means[0])))
        for k in range(2):
            ix=np.flatnonzero(group==k);noise=rng.multivariate_normal(np.zeros(len(means[0])),scales[k],len(ix));x[ix]=means[k]+noise/np.sqrt(rng.chisquare(5,len(ix))/5)[:,None]
        theta.append(x);reps.append(np.full(2048,rep))
    theta=np.concatenate(theta);rep=np.concatenate(reps);logq=logsumexp(np.array([np.log(mix[k])+multivariate_t.logpdf(theta,loc=means[k],shape=scales[k],df=5) for k in range(2)]),axis=0);prior=logprior(theta);N=len(theta);ll=np.full(N,np.nan);lp=np.full(N,np.nan);np.savez_compressed(root/'proposal_draws.npz',theta=theta,replicate=rep,logq=logq);write_json(root/'contract.json',dict(**contract,extension_number=args.extension,extension_reason='Initial particle-importance diagnostics did not pass; frozen gate and proposal retained, fresh independent groups added',proposal_source=str(source/'proposal.json'),global_replicates=sorted(np.unique(rep).tolist())))
    with ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(laplace_chunk,(args.model,theta[lo:min(lo+128,N)])):(lo,min(lo+128,N)) for lo in range(0,N,128)};cuda.select_device(args.gpu);d=dict(np.load(RUN/'observations.npz'));d=history_data(d) if args.model=='ou_history' else d
        for lo in range(0,N,64):
            hi=min(lo+64,N);p=root/'particle_batches'/f'{lo:05d}.npz'
            if p.exists():ll[lo:hi]=np.load(p)['loglik']
            else:
                valid=np.isfinite(prior[lo:hi]);values=np.full(hi-lo,-np.inf)
                if valid.any():values[valid]=evaluate(d,expand(theta[lo:hi][valid],args.model),1024,seed=990000+args.extension*1000000+(100000 if args.model=='ou_history' else 0)+lo,batch=64)
                ll[lo:hi]=values;p.parent.mkdir(exist_ok=True);np.savez_compressed(p,loglik=values)
            if (lo//64+1)%8==0:print(json.dumps(dict(model=args.model,extension=args.extension,proposals_done=hi,total=N,elapsed=time.time()-started)),flush=True);write_json(root/'progress.json',dict(status='RUNNING',proposals_done=hi,total=N))
        for future in as_completed(futures):
            lo,hi=futures[future];lp[lo:hi]=future.result()
    assert not np.isnan(ll).any() and not np.isnan(lp).any();np.savez_compressed(root/'weighted_samples.npz',theta=theta,replicate=rep,logq=logq,logprior=prior,particle_loglik=ll,laplace_loglik=lp,particle_logweight=ll+prior-logq,laplace_logweight=lp+prior-logq);write_json(root/'status.json',dict(status='COMPLETE',n_proposals=N,elapsed=time.time()-started));print(json.dumps(dict(status='COMPLETE',elapsed=time.time()-started)),flush=True)

if __name__=='__main__':main()
