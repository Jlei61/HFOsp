#!/usr/bin/env python3
"""Freshly validated conductance response; old failures remain development data."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import time
import numpy as np
from scipy.stats import qmc
from numba import set_num_threads
import torch
from torch import nn
from campaign import ROOT, PREVIOUS, read, write, sha
from audit_topic4_loop_conductance_response import monte_carlo, condition
from calibrate_topic4_loop_conductance_static import base_features

OUT = ROOT / 'conductance_static_v2'
OLD = PREVIOUS / 'conductance_response/static_calibration_v1'


def design(power, seed, low_mean=False):
    u = qmc.Sobol(4, scramble=True, seed=seed).random_base2(power)
    upper = 1. if low_mean else 30.
    x = np.sinh(np.arcsinh(-10.) + (np.arcsinh(upper)-np.arcsinh(-10.))*u[:,0])
    se = np.expm1(np.log(13.)*u[:,1])
    si = np.expm1(np.log(13.)*u[:,2])
    g = np.expm1(np.log(33.)*u[:,3])
    pars = np.array([condition(11+7*a,18.,(7*b)**2,(7*c)**2,'E') for a,b,c in zip(x,se,si)])
    return pars, g


class Response(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4,96), nn.Tanh(), nn.Linear(96,96), nn.Tanh(),
                                    nn.Linear(96,96), nn.Tanh(), nn.Linear(96,1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features, base_logits):
        gate = -torch.expm1(-3*features[:,3])
        correction = gate*self.layers(features).squeeze(-1)
        return 500*torch.sigmoid(base_logits+correction+np.log(20.))


def score(prediction, rates, label):
    mean = rates.mean(1)
    sem = rates.std(1,ddof=1)/np.sqrt(rates.shape[1])
    strict = np.maximum.reduce([np.full(len(mean),2.), .1*mean, 3*sem])
    broad = np.maximum.reduce([np.full(len(mean),10.), .25*mean, 5*sem])
    error = abs(prediction-mean)
    np.savez_compressed(OUT/f'{label}_scored.npz', mean_Hz=mean, SEM_Hz=sem,
          predicted_Hz=prediction, error_Hz=error, tolerance_Hz=strict,
          broad_error_cap_Hz=broad, pass_point=error<=strict, within_cap=error<=broad)
    return dict(label=label, passed=int((error<=strict).sum()), total=len(mean),
                within_broad_cap=int((error<=broad).sum()),
                maximum_error_Hz=float(error.max()), median_absolute_error_Hz=float(np.median(error)),
                pass_gate=bool((error<=strict).mean()>=.9 and (error<=broad).all()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'contract.json').exists(), 'One frozen attempt; preserve prior evidence.'
    set_num_threads(24)
    torch.set_num_threads(4)
    torch.manual_seed(927273)
    contract = dict(status='LOCKED_BEFORE_FRESH_TARGETS', created_epoch=time.time(),
        source_sha256=sha(__file__),
        question='Does broader sampling with deliberate low-effective-mean coverage repair the documented recovery-relevant static response errors without relaxing acceptance?',
        changes='4096 fresh training conditions (half broad, half subthreshold effective mean) plus original1024 training conditions; 4-96-96-96-1 tanh residual; fixed20000-step train-only schedule.',
        development_information='All previous failures are known development evidence. Previous validation is diagnostic only; this version has a fresh independent test design and noise seed.',
        domain=dict(normalized_mean=[-10,30], noise_sigma_EI=[0,12], total_g=[0,32],
                    low_mean_stratum=[-10,1]),
        train=dict(fresh_points=4096, reused_training_points=1024, design_seeds=[927271,927272],
                   noise_seed=927274, replicates=512, burn_ms=500, record_ms=2000),
        validation=dict(points=512, strata=['broad256','low_mean256'], design_seeds=[927281,927282],
                        noise_seed=927284, replicates=1024, burn_ms=500, record_ms=2000),
        fit=dict(seed=927273, optimizer='AdamW', weight_decay=1e-6, gradient_clip=10.,
                 steps=20000, batch=512, lr_schedule=[[0,.001],[8000,.0003],[16000,.0001]],
                 loss='Mean squared asinh(rate/2Hz) error; final fixed-step weights only.'),
        gate='Same original gate, applied separately to both fresh strata and overall: >=90% within max(2Hz,10%MC,3SEM), all within max(10Hz,25%MC,5SEM). No test-target refit, threshold relaxation or checkpoint selection.',
        stationary_boundary='Exact g=0 parent; static pass does not certify derivatives, transient response, spatial correspondence or bifurcations.',
        physical_mapping='Constant total g changes tau_m to20/(1+g); effective current mean and both colored-current channel amplitudes incorporate division by1+g. Native filter times and absolute refractory steps unchanged.',
        future='After independent static pass, assess local gain/derivative and dynamics, then same native spatial slices; inspect actual domain coverage and hidden spatial/history differences.')
    write(OUT/'contract.json', contract)
    start = time.time()
    def progress(stage, **kwargs):
        write(OUT/'progress.json', dict(status=stage, pid=os.getpid(), elapsed_s=time.time()-start, **kwargs))
    p1,g1=design(11,927271)
    p2,g2=design(11,927272,True)
    p,g=np.concatenate([p1,p2]),np.r_[g1,g2]
    f,b=base_features(p,g)
    np.savez_compressed(OUT/'fresh_training_design.npz',pars=p,g=g,features=f,base_logits=b)
    counts=[]
    for lo in range(0,len(p),256):
        progress('TRAINING_TARGET_MC',completed_points=lo,total_points=len(p))
        c=monte_carlo(p[lo:lo+256],g[lo:lo+256],512,20000,5000,927274)
        np.savez_compressed(OUT/f'train_counts_{lo:04d}.npz',counts=c)
        counts.append(c)
    rates=np.concatenate(counts).mean(1)/2.
    with np.load(OLD/'training_design.npz') as old, np.load(OLD/'training_targets.npz') as targets:
        f=np.concatenate([f,old['features']]);b=np.r_[b,old['base_logits']]
        rates=np.r_[rates,targets['rate_Hz']]
    np.savez_compressed(OUT/'training_arrays.npz',features=f,base_logits=b,rate_Hz=rates)
    net=Response().double()
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-6)
    f=torch.tensor(f);b=torch.tensor(b);target=torch.asinh(torch.tensor(rates)/2.)
    history=[]
    for step in range(20000):
        if step in [8000,16000]:
            for group in opt.param_groups:group['lr']=.0003 if step==8000 else .0001
        ix=torch.randint(len(f),(512,))
        pred=net(f[ix],b[ix])
        loss=torch.mean((torch.asinh(pred/2.)-target[ix])**2)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10.);opt.step()
        if (step+1)%1000==0:
            history.append([step+1,float(loss.detach())])
            progress('FITTING_TRAIN_ONLY',steps=step+1,history=history)
    net.eval()
    torch.save(dict(model=net.state_dict(),architecture=[4,96,96,96,1]),OUT/'locked_model.pt')
    write(OUT/'fit_locked.json',dict(status='FROZEN_BEFORE_FRESH_VALIDATION_TARGETS',
          model_sha256=sha(OUT/'locked_model.pt'),history=history))
    pv1,gv1=design(8,927281);pv2,gv2=design(8,927282,True)
    pv,gv=np.concatenate([pv1,pv2]),np.r_[gv1,gv2]
    fv,bv=base_features(pv,gv)
    with torch.no_grad():prediction=net(torch.tensor(fv),torch.tensor(bv)).numpy()
    np.savez_compressed(OUT/'validation_predictions_locked.npz',pars=pv,g=gv,features=fv,
                        base_logits=bv,prediction_Hz=prediction)
    validation=[]
    for lo in range(0,len(pv),128):
        progress('INDEPENDENT_VALIDATION_MC',completed_points=lo,total_points=len(pv))
        c=monte_carlo(pv[lo:lo+128],gv[lo:lo+128],1024,20000,5000,927284)
        np.savez_compressed(OUT/f'validation_counts_{lo:04d}.npz',counts=c)
        validation.append(c/2.)
    measured=np.concatenate(validation)
    rows=[score(prediction,measured,'overall'),score(prediction[:256],measured[:256],'broad'),
          score(prediction[256:],measured[256:],'low_mean')]
    # Physical zero-conductance base logits are recalculated, not reused from g>0.
    f0,b0=base_features(pv,np.zeros(len(pv)))
    with torch.no_grad():
        zero=net(torch.tensor(f0),torch.tensor(b0)).numpy()
        parent=(500*torch.sigmoid(torch.tensor(b0)+np.log(20.))).numpy()
    assert np.array_equal(zero,parent)
    result=dict(status='STATIC_VALIDATION_PASS' if all(r['pass_gate'] for r in rows) else 'STATIC_VALIDATION_FAIL',
                rows=rows,zero_conductance_parent_exact=True,transient_validated=False,
                spatial_validated=False,derivatives_validated=False,formal_bifurcation_allowed=False,
                elapsed_s=time.time()-start,source_sha256=sha(__file__))
    write(OUT/'result.json',result)
    progress(result['status'],result=result)
    print(result,flush=True)


if __name__=='__main__':
    main()
