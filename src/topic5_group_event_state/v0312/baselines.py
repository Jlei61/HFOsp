"""Independent NB fits with exposure offset and published-history covariates."""
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from . import data as D
from .train import load_run,atomic_json,atomic_torch,evaluate
from .engine import build_model
import torch


def covariates(prep,queries,role):
    pk=prep.payload['packets'];mask=D.input_mask(prep.split,role);out=[]
    for q in queries:
        t=pk['end'][q];ix=np.flatnonzero(mask&(pk['release']<=t)&(pk['end']<=t)&
                          (np.arange(prep.n_packets)>=prep.split['episode_start'][q]))
        ages=(t-pk['end'][ix])/3600
        expo=pk['exposure'][ix]/3600.;counts=pk['event_hi'][ix]-pk['event_lo'][ix]
        rates=[]
        for tau in (1/60,5/60,.5,2.,8.):
            w=np.exp(-ages/tau);rates.append(np.log1p(np.dot(w,counts)/max(np.dot(w,expo),1e-12)))
        out.append([1.,*D.clock_features(np.array([t]))[0],*rates,float(len(ix)==0),np.log1p(ages.min()) if len(ix) else 0.])
    return np.asarray(out)


def logp(par,X,y,expo):
    lr=np.clip(X@par[:-1],-20,20);mu=np.exp(lr)*expo;r=np.exp(par[-1])
    return gammaln(y+r)-gammaln(r)-gammaln(y+1)+r*(np.log(r)-np.log(r+mu))+y*(np.log(mu.clip(1e-100))-np.log(r+mu))


def run_baselines(cfg,out_path):
    _,prep=load_run(cfg);role='outer' if cfg.stage in ('outer','sid') else 'inner'
    tables={r:D.target_table(prep.payload,prep.split,r,stride=1 if r=='fit' else cfg.eval_stride) for r in ('fit',role)}

    features={r:covariates(prep,t[:,2],r) for r,t in tables.items()};results={}
    for r,t in tables.items():
        features[r][:,1:3]=D.clock_features((prep.packet_start[t[:,0]]+prep.packet_end[t[:,0]])/2)
        features[r]=np.column_stack((features[r],np.log1p(t[:,1]/60.)))
    for mode,cols in [('fixed_nb',[0]),('clock_nb',[0,1,2]),('recent_nb',[0,1,2,5,8,9,10]),('multiscale_rate_nb',list(range(11)))]:
        f=tables['fit'];y=prep.count.cpu().numpy()[f[:,0]];expo=prep.exposure_hours.cpu().numpy()[f[:,0]]
        X=features['fit'][:,cols];center=X.mean(0);scale=X.std(0).clip(.1);center[0]=0;scale[0]=1;X=(X-center)/scale
        init=np.zeros(len(cols)+1);init[0]=np.log(prep.scaling['base_rate_per_hour']);init[-1]=np.log(prep.scaling['nb_size'])
        fit=minimize(lambda p:-logp(p,X,y,expo).mean()+1e-4*np.square(p[1:-1]).sum(),init,method='L-BFGS-B',
                     bounds=[(-20,20)]*len(cols)+[(-12,15)],options={'maxiter':2000,'ftol':1e-12,'gtol':1e-7})
        if mode=='fixed_nb':fit.x=init;fit.success=True;fit.fun=-logp(init,X,y,expo).mean();fit.message='independent scalar FIT MLE on every physical FIT packet'
        t=tables[role];xx=(features[role][:,cols]-center)/scale;yy=prep.count.cpu().numpy()[t[:,0]];ee=prep.exposure_hours.cpu().numpy()[t[:,0]]
        lp=logp(fit.x,xx,yy,ee)
        results[mode]=dict(fit_success=bool(fit.success),fit_message=str(fit.message),fit_objective=float(fit.fun),
                          parameters=fit.x,center=center,scale=scale,columns=cols,
                          scores={h:float(-lp[t[:,1]==h].mean()) if (t[:,1]==h).any() else None for h in (1,5,30,120)},
                          rows=[dict(packet=int(i),horizon=int(h),query=int(q),logp=float(v)) for (i,h,q),v in zip(t,lp)])
    fixed=build_model(prep,'P_stats',arm='intercept',seed=cfg.seed)
    with torch.no_grad():
        final=fixed.readout.net[-1];final.weight.zero_();final.bias.zero_()
        fixed_result=evaluate(fixed,prep,cfg,role,collect=True)
    fixed_path=Path(out_path).with_suffix('.fixed_distributions.pt');atomic_torch(fixed_result,fixed_path)
    result=dict(fixed_distribution_path=str(fixed_path),status='COMPLETE',config=vars(cfg),split_id=prep.split['split_id'],transform_id=prep.scaling['transform_id'],
                role=role,target_digest=D.digest(tables[role]),models=results,
                rule='Unsuccessful solver is unqualified. Paired benchmark floors use identical target IDs; no OUTER-selected deployed predictor.')
    atomic_json(result,out_path);return result
