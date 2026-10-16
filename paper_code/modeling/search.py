"""Portable proposal step from scripts/topic4_core_count_optimization/search.py.

The GP, numerical nugget, local box, acquisition and uncertainty exploration
are the paper implementation. Filesystem scheduling is replaced by arguments.
Only completed training observations enter a proposal batch.
"""
import numpy as np
from scipy.stats import qmc


def bounds(k, radius):
    return (np.r_[np.full(2*k,radius+.05),.9,-60.],
            np.r_[np.full(2*k,20-radius-.05),1.6,60.])


def valid(k, vector, radius):
    lo,hi=bounds(k,radius);x=np.asarray(vector);xy=x[:2*k].reshape(k,2)
    return bool(np.isfinite(x).all() and np.all(x>=lo) and np.all(x<=hi) and
                all(np.linalg.norm(xy[i]-xy[j])>=2*radius+.05 for i in range(k) for j in range(i)))


def initial_vectors(k,restart,count,config):
    lo,hi=bounds(k,config['radius_mm'])
    pool=qmc.Sobol(2*k+2,scramble=True,seed=config['optimizer_seed']+100*restart+k).random_base2(12)
    return [lo+z*(hi-lo) for z in pool if valid(k,lo+z*(hi-lo),config['radius_mm'])][:count]


def propose(k,restart,batch,rows,count,config):
    dim=2*k+2;lo,hi=bounds(k,config['radius_mm'])
    seed=config['optimizer_seed']+10000*k+1000*restart+batch
    good=[r for r in rows if r['J'] is not None and np.isfinite(r['J'])]
    pool=qmc.Sobol(dim,scramble=True,seed=seed).random_base2(13)
    pool=np.array([z for z in pool if valid(k,lo+z*(hi-lo),config['radius_mm'])])
    used=(np.asarray([r['vector'] for r in rows])-lo)/(hi-lo)
    selected=[];roles=[]
    def novel(z):
        all_used=np.vstack([used,np.asarray(selected).reshape(-1,dim)])
        return np.min(np.linalg.norm(all_used-z,axis=1))>1e-5 and valid(k,lo+z*(hi-lo),config['radius_mm'])
    if len(good)>=config['feedback_min_scorable']:
        import torch
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition.logei import qLogExpectedImprovement
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.optim import optimize_acqf
        from gpytorch.kernels import ScaleKernel,MaternKernel
        from gpytorch.mlls import ExactMarginalLogLikelihood
        torch.set_num_threads(2);torch.manual_seed(seed)
        X=torch.tensor((np.asarray([r['vector'] for r in good])-lo)/(hi-lo),dtype=torch.float64)
        Y=torch.tensor([[-r['J']] for r in good],dtype=torch.float64)
        Yvar=torch.full_like(Y,max(float(Y.var()),1e-12)*1e-5)
        model=SingleTaskGP(X,Y,train_Yvar=Yvar,covar_module=ScaleKernel(MaternKernel(nu=2.5,ard_num_dims=dim)),outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood,model),optimizer_kwargs={'options':{'maxiter':160}})
        center=X[int(Y.argmax())];box=torch.stack([(center-.35).clamp_min(0),(center+.35).clamp_max(1)])
        acq=qLogExpectedImprovement(model,best_f=Y.max(),sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64]),seed=seed))
        z,_=optimize_acqf(acq,bounds=box,q=min(6,count),num_restarts=8,raw_samples=128,sequential=True,options={'maxiter':80,'batch_limit':2})
        for x in z.detach().numpy():
            if novel(x):selected.append(x);roles.append('qLogEI')
        with torch.no_grad():
            std=np.concatenate([model.posterior(torch.tensor(p,dtype=torch.float64)).variance[:,0].sqrt().numpy() for p in np.array_split(pool,32)])
        for i in np.argsort(-std):
            if len(selected)==count:break
            if novel(pool[i]):selected.append(pool[i]);roles.append('global_posterior_uncertainty')
    else:
        for z in pool:
            if len(selected)==count:break
            if novel(z):selected.append(z);roles.append('space_filling_insufficient_scorable')
    if len(selected)!=count:raise RuntimeError('Insufficient feasible novel proposals')
    return [(lo+z*(hi-lo),role) for z,role in zip(selected,roles)]
