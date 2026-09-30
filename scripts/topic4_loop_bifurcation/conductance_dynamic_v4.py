#!/usr/bin/env python3
"""Cause-specific conductance gain calibration with a single stationary function.

This is a local response candidate, never a replacement for native spatial QA.
Only constant conductance is calibrated here; time-varying conductance remains
a separate required qualification. All prior failed assays are development.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import time
import numpy as np
from scipy.stats import qmc
import torch
from torch import nn
from campaign import ROOT,read,write,sha
from conductance_static_v3 import Response
from conductance_static_v2 import design
from calibrate_topic4_loop_conductance_static import base_features
from audit_topic4_loop_conductance_response import condition,load_models,normalized_input,SCALE
from nonlinear_rate_response import normalized_jacobian
from refractory_rate_response import transfer_factors
import lif_mc

OUT=ROOT/'conductance_dynamic_v4'
STATIC=ROOT/'conductance_static_v3'


class ResponseV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.static=Response().double()
        self.history=nn.Sequential(nn.Linear(4,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,36)).double()
        nn.init.zeros_(self.history[-1].weight);nn.init.zeros_(self.history[-1].bias)

    def linear(self,d,create_graph=False):
        u=d['features'].detach().requires_grad_(True)
        correction=self.static.layers(u).squeeze(-1)
        grad=torch.autograd.grad(correction.sum(),u,create_graph=create_graph)[0]
        ell=d['base_logits']+correction
        p=torch.sigmoid(ell);rate=500*torch.sigmoid(ell+np.log(20.))
        ix=torch.arange(len(u));ch=d['channel']
        direct=d['base_gradient'][ix,ch]+grad[ix,ch]*d['input_gradient'][ix,ch]
        gate=-torch.expm1(-3*u[:,3])
        weights=d['parent_history']+gate[:,None]*self.history(u)
        memory=(weights.reshape(-1,3,12)[ix,ch]*d['bank']).sum(1)*d['input_gradient'][ix,ch]
        gain=rate*(1-p)*(direct+memory)*d['cov'][ix,ch]/((1-p)+p*d['K']/.1)
        return rate,gain


def compressed_bank(frequency,g):
    bank,cov,K=transfer_factors('E',frequency)
    for i,(f,gg) in enumerate(zip(frequency,g)):
        z=np.exp(-2j*np.pi*f/1000.*.1)
        for j,tau in enumerate(np.array([1.,4.,16.,64.])/(1+gg)):
            v=.1/tau;e=np.exp(-v)
            L=e*np.array([[1,0,0],[v,1,0],[.5*v*v,v,1.]])
            B=np.array([1-e,1-e*(1+v),1-e*(1+v+.5*v*v)])
            bank[i,3*j:3*j+3]=np.linalg.solve(np.eye(3)-L*z,B)-1
    return bank,cov,K


def workpoints(power,seed):
    p1,g1=design(power-1,seed);p2,g2=design(power-1,seed+1,True)
    p=np.concatenate([p1,p2]);g=np.r_[g1,g2]
    # Explicit zero-conductance coverage, selected before observing targets.
    g[::8]=0.
    return p,g


def rows_for(p,g,frequencies,half=False):
    rows=[];pars=[]
    for point,(pp,gg) in enumerate(zip(p,g)):
        for ch in [0,1,2]:
            for f in frequencies:
                for scale in ([1.,.5] if half and point%8==0 else [1.]):
                    amp=(.2 if ch==0 else .05)*scale
                    row=dict(point=point,g=float(gg),x=float((pp[0]-11)/7),sigma_E=float(np.sqrt(pp[2])/7),sigma_I=float(np.sqrt(pp[3])/7),channel=ch,frequency_Hz=f,amplitude=amp,primary=scale==1.)
                    q=condition(pp[0],18.,pp[2],pp[3],'E',amp,f,ch);q[18]=np.exp(-.1*(1+gg)/20.)
                    rows.append(row);pars.append(q)
    return rows,np.array(pars)


def arrays(rows,pars):
    torch.set_num_threads(2);nets,bases,_=load_models()
    physical=pars[:,[0,2,3]];g=np.array([r['g'] for r in rows]);frequency=np.array([r['frequency_Hz'] for r in rows])
    f=np.zeros((len(rows),39));f[:,:3]=normalized_input(physical,18.)/SCALE
    ft=torch.tensor(f,requires_grad=True);b,bg=bases['E'].evaluate(physical,18.,True)
    ell=nets['E'].logits(ft,torch.tensor(b));grad=torch.autograd.grad(ell.sum(),ft)[0].detach().numpy()
    du=normalized_jacobian(physical,18.);bank,cov,K=compressed_bank(frequency,g)
    return dict(features=np.c_[f[:,:3],np.log1p(g)/3.],base_logits=ell.detach().numpy()+np.log1p(g),
                base_gradient=bg+grad[:,:3]*du,input_gradient=du,parent_history=grad[:,3:],
                bank=bank,cov=cov,K=K,channel=np.array([r['channel'] for r in rows]))


def register():
    assert read(STATIC/'result.json')['status']=='STATIC_VALIDATION_PASS'
    assert not (OUT/'contract.json').exists()
    OUT.mkdir(exist_ok=True)
    contract=dict(status='FROZEN_BEFORE_TARGETS',created_epoch=time.time(),source_sha256=sha(__file__),
        question='Can physically motivated history time compression plus conductance-dependent history weights repair dynamic gain while the same stationary function fits rates and DC derivatives?',
        cause='Staticv3 passes rate tests but original histories fail atg>0; known diagnostic compression improves46to68of81. One DC derivative fails despite accurate mean rate.',
        candidate='Same static4-96-96-96-1 residual fine-tuned with DC derivative labels. Preserve parent nonlinear39-feature hazard; history taus become[1,4,16,64]/(1+g)ms. Add g/(1+g)*36 history weights from4-64-64-36tanh. Added history is zero at equilibrium andg0. Native current covariance filters and absolute2ms refractory unchanged.',
        training=dict(sobol_workpoints=256,design_seed=927451,structured_known_workpoints=9,
                      frequencies_Hz=[0,10,30,80,150],replicates=8192,record_ms=2000,burn_ms=500,noise_seed=927453,
                      reused_static_points=7168,steps=18000,batch_static=256,batch_linear=192,
                      seed=927454,learning_rates=[[0,.0003],[6000,.0001],[12000,.00003]],
                      loss='mean((asinh(r/2)-asinh(target/2))/.05)^2 + mean(abs((H-target)/tolerance))^2 +1e-5*mean(newhistoryweights^2). Gain normalization same10%DC/15%highfrequency and2SEM; nonidentifiable gains excluded and reported.'),
        validation=dict(workpoints=64,design_seed=927461,frequencies_Hz=[0,7,25,60],
                        half_amplitude_every_eighth_point=True,replicates=16384,record_ms=4000,burn_ms=500,noise_seed=927463,
                        static_points=512,static_design_seed=927471,static_noise_seed=927473),
        gate='Every estimable primary complex gain must be withinmax(10%absDC,2SEM,1e-7) at<=25Hz or15%absDC at60Hz. DC SNR>=10 determines estimability, never pass an unmeasured derivative. Half-amplitude sensitivity must passcombined2SEM/10%DC. Static criteria unchanged:>=90%withinmax(2Hz,10%,3SEM),ALLwithinmax(10Hz,25%,5SEM),separate broad/lowmean. No fitting or checkpoint selection with fresh validation targets.',
        interpretation='Passing constantg gain is only a local gate. Nonestimable points remain uncertified; time-varyingg, nonlinear waveform and native spatial correspondence still required. No bifurcation permission from this local result.',
        bounded='One frozen architecture/training schedule. Diagnose any failure before further changes; no automatic capacity sweep.',
        static_source_sha256=sha(STATIC/'locked_model.pt'))
    write(OUT/'contract.json',contract)
    p,g=workpoints(8,927451)
    extra=[];eg=[]
    for gg in [0.,2.,8.]:
        for x in [-.5,.5,1.2]:extra.append(condition(11+7*x,18.,196.,441.,'E'));eg.append(gg)
    p=np.concatenate([p,np.array(extra)]);g=np.r_[g,eg]
    rows,pars=rows_for(p,g,[0.,10.,30.,80.,150.])
    np.savez_compressed(OUT/'training_inputs.npz',pars=pars,**arrays(rows,pars));write(OUT/'training_rows.json',rows)
    p,g=workpoints(6,927461);rows,pars=rows_for(p,g,[0.,7.,25.,60.],half=True)
    np.savez_compressed(OUT/'validation_inputs.npz',pars=pars,**arrays(rows,pars));write(OUT/'validation_rows.json',rows)


def acquire(split):
    c=read(OUT/'contract.json')[split];rows=read(OUT/f'{split}_rows.json');pars=np.load(OUT/f'{split}_inputs.npz')['pars']
    folder=OUT/f'{split}_mc';folder.mkdir(exist_ok=True)
    if split=='validation':assert (OUT/'fit_locked.json').exists()
    seconds=c['record_ms']/1000.;targets=[];sems=[];rates=[];allrep=[];start=time.time()
    for lo in range(0,len(pars),8):
        path=folder/f'{lo:05d}.npz'
        if path.exists():obs=np.load(path)['observed']
        else:
            obs=lif_mc.run(pars[lo:lo+8],c['replicates'],c['record_ms'],c['burn_ms'],c['noise_seed'],device=1,batch=8)
            np.savez_compressed(path,observed=obs)
        for i,o in enumerate(obs):
            row=rows[lo+i];amp=row['amplitude']*(1. if row['channel']==0 else pars[lo+i,row['channel']+1])
            h=(o[:,0]+1j*o[:,1])/(seconds*amp);mu=h.mean();sem=np.sqrt(np.mean(abs(h-mu)**2)/len(h))
            targets.append(mu);sems.append(sem);rates.append(o[:,2:].mean()/seconds);allrep.append(h)
        write(OUT/'progress.json',dict(status=split.upper()+'_MC',pid=os.getpid(),completed=min(lo+8,len(rows)),total=len(rows),elapsed_s=time.time()-start))
    dc={(r['point'],r['channel']):i for i,r in enumerate(rows) if r['primary'] and r['frequency_Hz']==0}
    target=np.array(targets);sem=np.array(sems);norm=[];estimable=[]
    for i,r in enumerate(rows):
        j=dc[r['point'],r['channel']];d=abs(target[j]);ident=d/max(sem[j],1e-15)>=10
        estimable.append(ident);norm.append(max((.1 if r['frequency_Hz']<=25 else .15)*d,2*sem[i],1e-7))
    np.savez_compressed(OUT/f'{split}_targets.npz',target=target,SEM=sem,rate_Hz=rates,norm=norm,estimable=estimable)
    if split=='validation':
        half=[]
        for i,r in enumerate(rows):
            if r['primary']:continue
            j=next(j for j,q in enumerate(rows) if q['primary'] and all(q[k]==r[k] for k in ['point','channel','frequency_Hz']))
            difference=allrep[i]-allrep[j];sem=np.sqrt(np.mean(abs(difference-difference.mean())**2)/len(difference))
            tol=max(.1*abs(target[dc[r['point'],r['channel']]]),2*sem,1e-7)
            half.append(dict(point=r['point'],channel=r['channel'],frequency_Hz=r['frequency_Hz'],difference=float(abs(difference.mean())),paired_SEM=float(sem),tolerance=float(tol),passed=bool(abs(difference.mean())<=tol)))
        write(OUT/'amplitude_sensitivity.json',half)


def tensors(path):
    with np.load(path) as z:return {k:torch.tensor(z[k]) for k in z.files if k!='pars'}


def train():
    c=read(OUT/'contract.json')['training'];torch.set_num_threads(4);torch.manual_seed(c['seed'])
    assert not (OUT/'locked_model.pt').exists()
    model=ResponseV4();model.static.load_state_dict(torch.load(STATIC/'locked_model.pt',map_location='cpu',weights_only=False)['model'])
    data=tensors(OUT/'training_inputs.npz');targets=tensors(OUT/'training_targets.npz');s=tensors(STATIC/'training_arrays.npz')
    valid=torch.nonzero(targets['estimable']).flatten();assert len(valid)>0
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=1e-6);trace=[];start=time.time()
    for step in range(c['steps']):
        if step in [6000,12000]:
            for group in opt.param_groups:group['lr']=.0001 if step==6000 else .00003
        ix=valid[torch.randint(len(valid),(c['batch_linear'],))];si=torch.randint(len(s['rate_Hz']),(c['batch_static'],))
        d={k:v[ix] for k,v in data.items()};_,gain=model.linear(d,True)
        lg=(abs((gain-targets['target'][ix])/targets['norm'][ix])**2).mean()
        rate=model.static(s['features'][si],s['base_logits'][si])
        ls=(((torch.asinh(rate/2.)-torch.asinh(s['rate_Hz'][si]/2.))/.05)**2).mean()
        penalty=1e-5*(model.history(d['features'])**2).mean();loss=ls+lg+penalty
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
        if (step+1)%1000==0:
            trace.append(dict(step=step+1,loss=float(loss.detach()),static=float(ls.detach()),linear=float(lg.detach())))
            write(OUT/'progress.json',dict(status='TRAIN_ONLY_FIT',pid=os.getpid(),trace=trace,elapsed_s=time.time()-start))
    torch.save(dict(model=model.state_dict(),source_sha256=sha(__file__)),OUT/'locked_model.pt')
    model.eval();d=tensors(OUT/'validation_inputs.npz');r,h=model.linear(d)
    np.savez_compressed(OUT/'validation_predictions_locked.npz',rate_Hz=r.detach().numpy(),gain=h.detach().numpy())
    write(OUT/'fit_locked.json',dict(status='FROZEN_BEFORE_FRESH_TARGETS',model_sha256=sha(OUT/'locked_model.pt'),trace=trace,
                                   training_identifiable_gains=len(valid),training_total_gains=len(targets['target'])))


def static_validate():
    model=ResponseV4();model.load_state_dict(torch.load(OUT/'locked_model.pt',map_location='cpu',weights_only=False)['model']);model.eval()
    p,g=workpoints(9,927471);f,b=base_features(p,g)
    with torch.no_grad():pred=model.static(torch.tensor(f),torch.tensor(b)).numpy()
    np.savez_compressed(OUT/'static_validation_predictions_locked.npz',pars=p,g=g,prediction_Hz=pred)
    pars=p.copy();pars[:,18]=np.exp(-.1*(1+g)/20.)
    # Static kernel's unmodulated side1 remains zero; use side0 only.
    obs=lif_mc.run(pars,2048,2000,500,927473,device=1,batch=8);rate=obs[:,:,2]/2.
    mu=rate.mean(1);sem=rate.std(1,ddof=1)/np.sqrt(rate.shape[1]);error=abs(pred-mu)
    tol=np.maximum.reduce([np.full(len(mu),2.),.1*mu,3*sem]);cap=np.maximum.reduce([np.full(len(mu),10.),.25*mu,5*sem])
    rows=[]
    for label,ix in [('broad',np.arange(256)),('low_mean',np.arange(256,512)),('zero_g',np.flatnonzero(g==0))]:
        rows.append(dict(label=label,total=len(ix),strict=int((error[ix]<=tol[ix]).sum()),within_cap=int((error[ix]<=cap[ix]).sum()),passed=bool(np.mean(error[ix]<=tol[ix])>=.9 and np.all(error[ix]<=cap[ix]))))
    np.savez_compressed(OUT/'static_validation_scored.npz',rate_Hz=mu,SEM_Hz=sem,prediction_Hz=pred,error_Hz=error,tolerance_Hz=tol,cap_Hz=cap)
    write(OUT/'static_validation.json',dict(status='PASS' if all(r['passed'] for r in rows) else 'FAIL',rows=rows))


def score():
    r=read(OUT/'validation_rows.json');p=np.load(OUT/'validation_predictions_locked.npz');t=np.load(OUT/'validation_targets.npz')
    error=abs(p['gain']-t['target']);ident=t['estimable'];primary=np.array([q['primary'] for q in r]);chosen=ident&primary
    rows=[]
    for i,row in enumerate(r):
        rows.append(dict(**row,measured_gain=[float(t['target'][i].real),float(t['target'][i].imag)],predicted_gain=[float(p['gain'][i].real),float(p['gain'][i].imag)],SEM=float(t['SEM'][i]),tolerance=float(t['norm'][i]),error=float(error[i]),estimable=bool(ident[i]),passed=bool(error[i]<=t['norm'][i]) if ident[i] else None))
    sensitivity=read(OUT/'amplitude_sensitivity.json');static=read(OUT/'static_validation.json')
    passed=bool(np.all(error[chosen]<=t['norm'][chosen]) and all(r['passed'] for r in sensitivity) and static['status']=='PASS')
    result=dict(status='LOCAL_CONSTANT_G_ESTIMABLE_GAINS_PASS' if passed else 'LOCAL_CONSTANT_G_NOT_QUALIFIED',
        estimable=int(chosen.sum()),primary=int(primary.sum()),passed=int(np.sum(error[chosen]<=t['norm'][chosen])),nonestimable=int((primary&~ident).sum()),
        amplitude_sensitivity_pass=sum(r['passed'] for r in sensitivity),amplitude_sensitivity_total=len(sensitivity),static=static,rows=rows,
        full_domain_derivatives_certified=False,dynamic_conductance_validated=False,nonlinear_waveform_validated=False,native_spatial_validated=False,formal_bifurcation_allowed=False)
    write(OUT/'result.json',result);write(OUT/'progress.json',{k:v for k,v in result.items() if k!='rows'})
    print({k:v for k,v in result.items() if k not in ['rows','static']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','acquire-training','train','validate','all']);a=p.parse_args()
    if a.command in ['register','all']:register()
    if a.command in ['acquire-training','all']:acquire('training')
    if a.command in ['train','all']:train()
    if a.command in ['validate','all']:
        acquire('validation');static_validate();score()
