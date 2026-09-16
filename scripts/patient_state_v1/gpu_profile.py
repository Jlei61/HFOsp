"""Monte Carlo likelihood profile check, independent of the CPU Laplace approximation."""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.special import logsumexp
from numba import cuda
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.gpu_particles import evaluate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--model',choices=['ou','ou_cycle'],required=True);ap.add_argument('--replicates',type=int,default=192);ap.add_argument('--particles',type=int,default=1024);args=ap.parse_args()
    cuda.select_device(args.gpu);data=dict(np.load(RUN/'observations.npz'));rng=np.random.default_rng(20310+args.gpu)
    out=RUN/'gpu_profiles'/args.model;out.mkdir(parents=True,exist_ok=True);started=time.time()
    # Wait only for this bounded CPU profile wave; never poll forever.
    pending=list(range(61))
    while pending and time.time()-started<7200:
        worked=False
        for i in pending.copy():
            target=out/f'{i:03d}.json'
            if target.exists():pending.remove(i);continue
            source=RUN/'round2'/f'profile_full_{args.model}_{i:03d}.json'
            if not source.exists():continue
            r=json.loads(source.read_text())
            if r.get('status')!='COMPLETE':pending.remove(i);continue
            t=r['theta'];par=t if args.model=='ou_cycle' else [t[0],0,0,t[1],t[2]]
            ll=evaluate(data,np.tile(par,(args.replicates,1)),args.particles,seed=263000+1000*i+args.gpu)
            estimate=logsumexp(ll)-np.log(len(ll));boots=[]
            for _ in range(1000):boots.append(logsumexp(rng.choice(ll,len(ll),replace=True))-np.log(len(ll)))
            result=dict(status='COMPLETE',model=args.model,tau_hours=r['job']['tau'],theta=t,particle_loglik=estimate,
                        bootstrap_mc_interval=np.quantile(boots,[.025,.975]),laplace_loglik=r['loglik'],loglik_replicates=ll,
                        particles=args.particles,replicates=args.replicates,source=str(source),estimator='log of mean unbiased particle likelihood; finite-replicate log bias remains')
            write_json(target,result);pending.remove(i);worked=True
            print(json.dumps(dict(complete=61-len(pending),total=61,tau=r['job']['tau'],laplace=r['loglik'],particle=float(estimate),elapsed=time.time()-started)),flush=True)
        if not worked and pending:time.sleep(5)
    write_json(out/'status.json',dict(status='COMPLETE' if not pending else 'INCOMPLETE_TIMEOUT',pending=pending,elapsed=time.time()-started))

if __name__=='__main__':main()
