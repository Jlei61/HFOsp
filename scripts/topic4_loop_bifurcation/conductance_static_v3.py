#!/usr/bin/env python3
"""Repair the diagnosed parent-domain constraint; independently test g=0 too."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time
import numpy as np
from numba import set_num_threads
import torch
from torch import nn
from campaign import ROOT,read,write,sha
from conductance_static_v2 import design
from calibrate_topic4_loop_conductance_static import base_features
from audit_topic4_loop_conductance_response import monte_carlo

OUT=ROOT/'conductance_static_v3'
PARENT=ROOT/'conductance_static_v2'


class Response(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(4,96),nn.Tanh(),nn.Linear(96,96),nn.Tanh(),
                                  nn.Linear(96,96),nn.Tanh(),nn.Linear(96,1))
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)

    def forward(self,features,base_logits):
        return 500*torch.sigmoid(base_logits+self.layers(features).squeeze(-1)+np.log(20.))


def fresh_training():
    p1,g1=design(9,927371);p2,g2=design(9,927372,True)
    p3,g3=design(8,927373);p4,g4=design(8,927374,True)
    # Sobol g coordinate is uniform after its inverse mapping; remap to[0,.3].
    g3=np.expm1(np.log1p(g3)/np.log(33.)*np.log(1.3))
    g4=np.expm1(np.log1p(g4)/np.log(33.)*np.log(1.3))
    return np.concatenate([p1,p2,p3,p4]),np.r_[np.zeros(1024),g3,g4]


def validation():
    pars=[];gs=[];labels=[]
    for i,(label,low,zero) in enumerate([('broad',False,False),('low_mean',True,False),
                                       ('zero_g_broad',False,True),('zero_g_low_mean',True,True)]):
        p,g=design(8,927381+i,low)
        if zero:g.fill(0.)
        pars.append(p);gs.append(g);labels.extend([label]*256)
    return np.concatenate(pars),np.concatenate(gs),np.array(labels)


def main():
    assert read(PARENT/'result.json')['status']=='STATIC_VALIDATION_FAIL'
    diagnostic=read(ROOT/'parent_domain_diagnostic/result.json')
    assert diagnostic['status']=='COMPLETE' and abs(diagnostic['error_Hz'][0])>10.
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    write(OUT/'contract.json',dict(status='LOCKED_BEFORE_NEW_TARGETS',created_epoch=time.time(),
        question='Can independently calibrated zero/small-conductance data repair the inherited table-domain error without changing the native SNN?',
        identified_failure='v2 index270 lies beyond parent sigmaE6.5 table; atg0 measured16.34Hz,parent2.37Hz. The forced g0-parent invariant prevents correcting this inherited error.',
        change='Same4-96-96-96-1 residual architecture, remove g/(1+g) residual gate. Oldparent pluslog(1+g) is only a numerical baseline. No claim of exact old-rate closure atg0; directg0 independent MC replaces that invalid invariant.',
        training=dict(reused_v1_v2_training=5120,known_v2_validation_now_development=512,
                      fresh_g0=1024,fresh_small_g=512,replicates=512,
                      design_seeds=[927371,927372,927373,927374],noise_seed=927375,
                      record_ms=2000,burn_ms=500),
        fit=dict(seed=927376,steps=30000,batch=512,loss='Squared asinh(rate/2Hz) error',
                 optimizer='AdamW',weight_decay=1e-6,gradient_clip=10.,
                 learning_rate_schedule=[[0,.001],[12000,.0003],[24000,.0001]],
                 selection='Fixed final step only; no validation selection.'),
        validation=dict(total=1024,strata=['broad256','low_mean256','zero_g_broad256','zero_g_low_mean256'],
                        design_seeds=[927381,927382,927383,927384],noise_seed=927385,
                        replicates=1024,burn_ms=500,record_ms=2000),
        gate='Unchanged error criteria, separately everystratum andoverall: >=90% withinmax(2Hz,10%MC,3SEM), everypoint withinmax(10Hz,25%MC,5SEM). Predictions/model frozen before newtesttargets.',
        source_sha256=sha(__file__),
        limits='A static pass does not establish gain, dynamic conductance, shared/private noise, spatial propagation or bifurcation. Parent spline boundary derivatives and monotonicity must be inspected before stability claims.',
        reflection='If this cause-specific repair still fails, diagnose actual native domain and approximation assumptions before another model expansion. No indefinite capacity sweep.'))
    start=time.time();set_num_threads(24);torch.manual_seed(927376)
    def progress(stage,**kw):
        write(OUT/'progress.json',dict(status=stage,pid=os.getpid(),elapsed_s=time.time()-start,**kw))
    p,g=fresh_training();f,b=base_features(p,g)
    np.savez_compressed(OUT/'fresh_training_design.npz',pars=p,g=g,features=f,base_logits=b)
    counts=[]
    for lo in range(0,len(p),256):
        progress('FRESH_G0_SMALL_G_TRAINING',completed_points=lo,total_points=len(p))
        c=monte_carlo(p[lo:lo+256],g[lo:lo+256],512,20000,5000,927375)
        np.savez_compressed(OUT/f'train_counts_{lo:04d}.npz',counts=c);counts.append(c)
    targets=np.concatenate(counts).mean(1)/2.
    with np.load(PARENT/'training_arrays.npz') as d:
        f=np.concatenate([f,d['features']]);b=np.r_[b,d['base_logits']];targets=np.r_[targets,d['rate_Hz']]
    known_counts=np.concatenate([np.load(PARENT/f'validation_counts_{lo:04d}.npz')['counts'] for lo in range(0,512,128)])
    with np.load(PARENT/'validation_predictions_locked.npz') as d:
        f=np.concatenate([f,d['features']]);b=np.r_[b,d['base_logits']]
    targets=np.r_[targets,known_counts.mean(1)/2.]
    assert len(targets)==7168
    np.savez_compressed(OUT/'training_arrays.npz',features=f,base_logits=b,rate_Hz=targets)
    torch.set_num_threads(4);model=Response().double();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-6)
    f=torch.tensor(f);b=torch.tensor(b);y=torch.asinh(torch.tensor(targets)/2.)
    history=[]
    for step in range(30000):
        if step in [12000,24000]:
            for group in optimizer.param_groups:group['lr']=.0003 if step==12000 else .0001
        ix=torch.randint(len(f),(512,));pred=model(f[ix],b[ix]);loss=((torch.asinh(pred/2.)-y[ix])**2).mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
        if (step+1)%1000==0:
            history.append([step+1,float(loss.detach())]);progress('TRAIN_ONLY_FIT',steps=step+1,history=history)
    model.eval();torch.save(dict(model=model.state_dict(),architecture=[4,96,96,96,1]),OUT/'locked_model.pt')
    write(OUT/'fit_locked.json',dict(status='FROZEN_BEFORE_NEW_VALIDATION_TARGETS',model_sha256=sha(OUT/'locked_model.pt'),history=history))
    pv,gv,labels=validation();fv,bv=base_features(pv,gv)
    with torch.no_grad():pred=model(torch.tensor(fv),torch.tensor(bv)).numpy()
    np.savez_compressed(OUT/'validation_predictions_locked.npz',pars=pv,g=gv,features=fv,base_logits=bv,prediction_Hz=pred,strata=labels)
    counts=[]
    for lo in range(0,len(pv),128):
        progress('FRESH_VALIDATION',completed_points=lo,total_points=len(pv))
        c=monte_carlo(pv[lo:lo+128],gv[lo:lo+128],1024,20000,5000,927385)
        np.savez_compressed(OUT/f'validation_counts_{lo:04d}.npz',counts=c);counts.append(c)
    rates=np.concatenate(counts)/2.;mean=rates.mean(1);sem=rates.std(1,ddof=1)/32.
    error=abs(pred-mean);tol=np.maximum.reduce([np.full(len(mean),2.),.1*mean,3*sem])
    cap=np.maximum.reduce([np.full(len(mean),10.),.25*mean,5*sem]);passed=error<=tol;bounded=error<=cap
    np.savez_compressed(OUT/'validation_scored.npz',mean_Hz=mean,SEM_Hz=sem,prediction_Hz=pred,
                        error_Hz=error,tolerance_Hz=tol,broad_error_cap_Hz=cap,passed=passed,within_cap=bounded,strata=labels)
    rows=[]
    for label in ['overall','broad','low_mean','zero_g_broad','zero_g_low_mean']:
        ix=np.ones(len(labels),bool) if label=='overall' else labels==label
        rows.append(dict(label=label,total=int(ix.sum()),passed=int(passed[ix].sum()),
             within_cap=int(bounded[ix].sum()),maximum_error_Hz=float(error[ix].max()),
             median_absolute_error_Hz=float(np.median(error[ix])),
             pass_gate=bool(passed[ix].mean()>=.9 and bounded[ix].all())))
    result=dict(status='STATIC_VALIDATION_PASS' if all(r['pass_gate'] for r in rows) else 'STATIC_VALIDATION_FAIL',
                rows=rows,elapsed_s=time.time()-start,zero_g_old_parent_invariant_intentionally_removed=True,
                derivative_validated=False,transient_validated=False,spatial_validated=False,formal_bifurcation_allowed=False)
    write(OUT/'result.json',result);progress(result['status'],result=result);print(result,flush=True)


if __name__=='__main__':main()
