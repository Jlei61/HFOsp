#!/usr/bin/env python3
"""One frozen E-cell conductance calibration, independent local validation."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key]='1'
import json
import hashlib
import time
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from numba import set_num_threads
import torch
from torch import nn
from audit_topic4_loop_conductance_response import (monte_carlo, condition, load_models,
                                                    normalized_input, SCALE, OUT as AUDIT)

OUT=AUDIT/'static_calibration_v1'


def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def design(power,seed):
    u=qmc.Sobol(4,scramble=True,seed=seed).random_base2(power)
    x=np.sinh(np.arcsinh(-10.)+(np.arcsinh(30.)-np.arcsinh(-10.))*u[:,0])
    se=np.expm1(np.log(13.)*u[:,1]);si=np.expm1(np.log(13.)*u[:,2])
    g=np.expm1(np.log(33.)*u[:,3])
    p=np.array([condition(11+7*a,18.,(7*b)**2,(7*c)**2,'E') for a,b,c in zip(x,se,si)])
    return p,g


def base_features(pars,g):
    nets,bases,_=load_models();p=pars[:,[0,2,3]]
    f=np.zeros((len(pars),39));f[:,:3]=normalized_input(p,18.)/SCALE
    with torch.no_grad():
        ell=nets['E'].logits(torch.tensor(f),torch.tensor(bases['E'].evaluate(p,18.))).numpy()
    features=np.c_[f[:,:3],np.log1p(g)/3.]
    return features,ell+np.log1p(g)


class StaticCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(4,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)

    def forward(self,features,base_logits):
        g=torch.expm1(features[:,3]*3.)
        correction=(g/(1+g))*self.layers(features).squeeze(-1)
        return 500*torch.sigmoid(base_logits+correction+np.log(20.))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'contract.json').exists(), 'Calibration is a single frozen attempt'
    set_num_threads(8);torch.set_num_threads(2);torch.manual_seed(929243)
    write('contract.json',dict(status='LOCKED_BEFORE_TARGET_GENERATION',created_epoch=time.time(),
        question='Does one independently calibrated static conductance correction repair the rejected time-compression approximation over the actual conductance scale?',
        source_failure=str(AUDIT/'result.json'),
        class_definition='Original conditioned39 E stationary log hazard + log(1+g) + g/(1+g)*MLP(asinh(x)/3,log1p(sigmaE^2)/2,log1p(sigmaI^2)/2,log1p(g)/3). Fixed parent atg0.',
        data_definition='Independent colored-LIF MonteCarlo with native filter/refractory/exponential membrane order; gconstant. These are local test conditions, not nativeSNN trajectories or training to onset times.',
        train=dict(sobol_points=1024,design_seed=929241,noise_seed=929244,replicates=512,duration_ms=2000,burn_ms=500),
        validation=dict(sobol_points=256,design_seed=929242,noise_seed=929245,replicates=1024,duration_ms=2000,burn_ms=500),
        domain=dict(normalized_mean_x=[-10,30],normalized_noise_sigma_EI=[0,12],g=[0,32],sampling='asinh mean,log1p sigma,log1p g'),
        fit=dict(architecture=[4,64,64,1],activation='tanh',optimizer='AdamW',learning_rate=.001,weight_decay=1e-6,
                 steps=8000,batch=256,seed=929243,loss='Squared asinh(rate/2Hz) error; final fixed-step model only.'),
        gate='At least90% of256 independent validation points withinmax(2Hz,10%measured,3SEM), and every point withinmax(10Hz,25%measured,5SEM). Zero-conductance parent invariant exact. No validation-selected refit.',
        next_gate='A static pass is only a candidate. Validate conductance transients and same spatial conditional dynamics before any formal branch or loop claim.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    p,g=design(10,929241);features,base_logits=base_features(p,g)
    np.savez_compressed(OUT/'training_design.npz',pars=p,g=g,features=features,base_logits=base_logits)
    start=time.time();write('progress.json',dict(status='TRAINING_TARGET_MC',pid=os.getpid(),started_epoch=start))
    counts=monte_carlo(p,g,512,20000,5000,929244)
    rate=counts.mean(1)/2.
    np.savez_compressed(OUT/'training_targets.npz',counts=counts,rate_Hz=rate)
    net=StaticCorrection().double();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-6)
    f=torch.tensor(features);b=torch.tensor(base_logits);target=torch.asinh(torch.tensor(rate)/2.)
    history=[]
    for step in range(8000):
        ix=torch.randint(len(f),(256,));pred=net(f[ix],b[ix])
        loss=torch.mean((torch.asinh(pred/2.)-target[ix])**2)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10.);opt.step()
        if (step+1)%1000==0:
            history.append([step+1,float(loss.detach())])
            write('progress.json',dict(status='FITTING_TRAIN_ONLY',pid=os.getpid(),steps=step+1,history=history,elapsed_s=time.time()-start))
    net.eval();torch.save(dict(model=net.state_dict(),architecture=[4,64,64,1]),OUT/'locked_model.pt')
    write('fit_locked.json',dict(status='FROZEN_BEFORE_VALIDATION_TARGETS',history=history,
          model_sha256=hashlib.sha256((OUT/'locked_model.pt').read_bytes()).hexdigest()))
    pv,gv=design(8,929242);fv,bv=base_features(pv,gv)
    with torch.no_grad():prediction=net(torch.tensor(fv),torch.tensor(bv)).numpy()
    np.savez_compressed(OUT/'validation_predictions_locked.npz',pars=pv,g=gv,features=fv,base_logits=bv,prediction_Hz=prediction)
    write('progress.json',dict(status='INDEPENDENT_VALIDATION_MC',pid=os.getpid(),elapsed_s=time.time()-start))
    measured=monte_carlo(pv,gv,1024,20000,5000,929245)/2.
    mean=measured.mean(1);sem=measured.std(1,ddof=1)/np.sqrt(1024)
    tol=np.maximum.reduce([np.full(len(mean),2.),mean*.1,sem*3])
    cap=np.maximum.reduce([np.full(len(mean),10.),mean*.25,sem*5])
    error=abs(prediction-mean);passed=error<=tol;bounded=error<=cap
    np.savez_compressed(OUT/'validation_scored.npz',mean_Hz=mean,SEM_Hz=sem,predicted_Hz=prediction,
                        error_Hz=error,tolerance_Hz=tol,broad_error_cap_Hz=cap,pass_point=passed,within_cap=bounded)
    f0=fv.copy();f0[:,3]=0.
    with torch.no_grad():
        observed0=net(torch.tensor(f0),torch.tensor(bv)).numpy()
    expected0=(500*torch.sigmoid(torch.tensor(bv)+np.log(20.))).numpy()
    assert np.array_equal(observed0,expected0)
    result=dict(status='STATIC_VALIDATION_PASS' if passed.mean()>=.9 and bounded.all() else 'STATIC_VALIDATION_FAIL',
        passed=int(passed.sum()),total=len(passed),within_broad_cap=int(bounded.sum()),
        maximum_error_Hz=float(error.max()),median_absolute_error_Hz=float(np.median(error)),
        zero_conductance_parent_exact=True,transient_validated=False,spatial_validated=False,
        formal_bifurcation_allowed=False,seconds=time.time()-start)
    write('result.json',result);write('progress.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
