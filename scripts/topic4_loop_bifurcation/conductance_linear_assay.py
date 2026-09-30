#!/usr/bin/env python3
"""Fresh paired local gain assay of the frozen static-corrected hazard candidate.

Constant conductance only: changing G/K is a separate missing qualification.
This candidate retains the parent's input-history states, not measured spikes.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time
import numpy as np
import torch
from campaign import ROOT, read, write, sha
from conductance_static_v2 import Response
from calibrate_topic4_loop_conductance_static import base_features
from audit_topic4_loop_conductance_response import condition,load_models,normalized_input,SCALE
from refractory_rate_response import transfer_factors
from nonlinear_rate_response import normalized_jacobian
import lif_mc

OUT=ROOT/'conductance_linear_v2'
STATIC=ROOT/'conductance_static_v2'


def predictions(rows):
    torch.set_num_threads(2)
    nets,bases,_=load_models();parent=nets['E'];base=bases['E']
    correction=Response().double()
    correction.load_state_dict(torch.load(STATIC/'locked_model.pt',map_location='cpu',weights_only=False)['model'])
    correction.eval()
    physical=np.array([[11+7*r['x'],(7*r['sigma_E'])**2,(7*r['sigma_I'])**2] for r in rows])
    g=np.array([r['g'] for r in rows]);freq=np.array([r['frequency_Hz'] for r in rows])
    channel=np.array([r['channel'] for r in rows]);n=len(rows)
    f=np.zeros((n,39));f[:,:3]=normalized_input(physical,18.)/SCALE
    ft=torch.tensor(f,requires_grad=True)
    baseline,base_grad=base.evaluate(physical,18.,True)
    ell0=parent.logits(ft,torch.tensor(baseline))
    pg=torch.autograd.grad(ell0.sum(),ft)[0].detach().numpy()
    u=torch.tensor(np.c_[f[:,:3],np.log1p(g)/3.],requires_grad=True)
    residual=(-torch.expm1(-3*u[:,3]))*correction.layers(u).squeeze(-1)
    cg=torch.autograd.grad(residual.sum(),u)[0].detach().numpy()
    ell=(ell0.detach().numpy()+np.log1p(g)+residual.detach().numpy())
    p=1/(1+np.exp(-ell));r=500/(1+np.exp(-ell)/20.)
    bank,cov,K=transfer_factors('E',freq)
    ix=np.arange(n);du=normalized_jacobian(physical,18.)
    direct=base_grad[ix,channel]+(pg[ix,channel]+cg[ix,channel])*du[ix,channel]
    memory=(pg[:,3:].reshape(n,3,12)[ix,channel]*bank).sum(1)*du[ix,channel]
    gain=r*(1-p)*(direct+memory)*cov[ix,channel]/((1-p)+p*K/.1)
    with torch.no_grad():
        check=correction(torch.tensor(np.c_[f[:,:3],np.log1p(g)/3.]),
                         torch.tensor(ell0.detach().numpy()+np.log1p(g))).numpy()
    assert np.allclose(check,r,rtol=1e-13,atol=1e-13)
    # DC gain must be the derivative of the same stationary candidate.
    dc=np.flatnonzero(freq==0.)
    plus=physical[dc].copy();minus=plus.copy()
    h=1e-5*np.maximum(1.,abs(plus[np.arange(len(dc)),channel[dc]]))
    plus[np.arange(len(dc)),channel[dc]]+=h
    minus[np.arange(len(dc)),channel[dc]]-=h
    pp=np.concatenate([plus,minus]);gg=np.tile(g[dc],2)
    pars=np.array([condition(a,18.,b,c,'E') for a,b,c in pp])
    ff,bb=base_features(pars,gg)
    with torch.no_grad():rr=correction(torch.tensor(ff),torch.tensor(bb)).numpy()
    finite=(rr[:len(dc)]-rr[len(dc):])/(2*h)
    assert np.allclose(finite,gain[dc].real,rtol=2e-4,atol=2e-6), (finite,gain[dc])
    assert np.max(abs(gain[dc].imag))<1e-12
    return r,gain


def main():
    assert read(STATIC/'result.json')['status']=='STATIC_VALIDATION_PASS', 'No dependent qualification before static pass.'
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'contract.json').exists(), 'Preserve this frozen attempt.'
    rows=[];pars=[]
    for g in [0.,2.,8.]:
        for x in [-.5,.5,1.2]:
            for channel in [0,1,2]:
                for frequency in [0.,10.,50.]:
                    for amplitude_factor in ([1.,.5] if g==2. and x==.5 else [1.]):
                        amplitude=(.2 if channel==0 else .05)*amplitude_factor
                        row=dict(g=g,x=x,sigma_E=2.,sigma_I=3.,channel=channel,
                                 frequency_Hz=frequency,amplitude=amplitude,
                                 primary=amplitude_factor==1.)
                        p=condition(11+7*x,18.,14.**2,21.**2,'E',amplitude,frequency,channel)
                        p[18]=np.exp(-.1*(1+g)/20.)
                        rows.append(row);pars.append(p)
    rate,gain=predictions(rows)
    for row,r,h in zip(rows,rate,gain):
        row.update(predicted_rate_Hz=float(r),predicted_gain=[float(h.real),float(h.imag)])
    write(OUT/'contract.json',dict(status='FROZEN_BEFORE_FRESH_TARGETS',created_epoch=time.time(),
        question='Does adding the static conductance residual to the existing conditioned input-history hazard preserve local DC and dynamic gains at constant conductance?',
        candidate='Exact staticv2 at equilibrium, original39 parent features/history and own refractory integral. Residual responds to instantaneous effective current features; no fitted transient parameter.',
        limitation='Constant g only; no dynamic-conductance, shot-noise correlation, heterogeneous spatial or native propagation qualification.',
        design=dict(primary_points=81,linearity_sensitivity_points=9,replicates=8192,
                    burn_ms=500,record_ms=2000,dt_ms=.1,seed=927293,device=1),
        gate='Use measured paired DC gain SNR>=10. For every estimable primary channel/frequency, complex error<=max(10%abs(DC),2SEM,1e-7) at0/10Hz and15% at50Hz. Nonestimable channels are reported, never silently passed. Half-amplitude agreement at the center uses combined2SEM and10%DC to expose nonlinear finite-difference bias.',
        unit='One synthetic colored-LIF input/g condition; replicas are independent noise histories. Not native-network seeds.',
        source_sha256=sha(__file__),static_weights_sha256=sha(STATIC/'locked_model.pt'),
        MC_source=str(lif_mc.__file__),MC_source_sha256=sha(lif_mc.__file__),
        kernel_change='Existing exact colored-current assay; only p18=exp(-dt*(1+g)/20). Reset, refractory, current filters and paired noise unchanged.',
        resources='One existing GPU context onGPU1, batches3 conditions, small output buffers; no native worker, no native physics modification.',
        rows=rows))
    start=time.time();observed=[]
    for lo in range(0,len(rows),3):
        write(OUT/'progress.json',dict(status='PAIRED_MC',pid=os.getpid(),
              completed=lo,total=len(rows),elapsed_s=time.time()-start))
        out=lif_mc.run(pars[lo:lo+3],8192,2000,500,927293,device=1,batch=3)
        np.savez_compressed(OUT/f'mc_{lo:03d}.npz',observed=out)
        observed.append(out)
    observed=np.concatenate(observed)
    for i,row in enumerate(rows):
        amplitude=row['amplitude']*(1. if row['channel']==0 else pars[i][row['channel']+1])
        z=(observed[i,:,0]+1j*observed[i,:,1])/(2.*amplitude)
        mu=z.mean();sem=np.sqrt(np.mean(abs(z-mu)**2)/len(z))
        row.update(measured_gain=[float(mu.real),float(mu.imag)],complex_SEM=float(sem),
                   measured_rate_Hz=float(observed[i,:,2:].mean()/2.))
    dc={(r['g'],r['x'],r['channel']):r for r in rows if r['frequency_Hz']==0 and r['primary']}
    for row in rows:
        baseline=dc[row['g'],row['x'],row['channel']]
        d=abs(complex(*baseline['measured_gain']));snr=d/max(baseline['complex_SEM'],1e-15)
        tol=max((.1 if row['frequency_Hz']<=25 else .15)*d,2*row['complex_SEM'],1e-7)
        error=abs(complex(*row['predicted_gain'])-complex(*row['measured_gain']))
        row.update(DC_SNR=snr,estimable=snr>=10.,tolerance=tol,complex_error=error,
                   passed=bool(error<=tol) if snr>=10. else None)
    sensitivity=[]
    for i,row in enumerate(rows):
        if row['primary']:continue
        j=next(j for j,q in enumerate(rows) if q['primary'] and all(q[k]==row[k] for k in ['g','x','channel','frequency_Hz']))
        # Paired amplitude comparison retains covariance between the common replicas.
        amp_i=row['amplitude']*(1. if row['channel']==0 else pars[i][row['channel']+1])
        amp_j=rows[j]['amplitude']*(1. if row['channel']==0 else pars[j][row['channel']+1])
        a=(observed[i,:,0]+1j*observed[i,:,1])/(2*amp_i)
        b=(observed[j,:,0]+1j*observed[j,:,1])/(2*amp_j)
        delta=a-b;sem=float(np.sqrt(np.mean(abs(delta-delta.mean())**2)/len(delta)))
        tolerance=max(.1*abs(complex(*dc[row['g'],row['x'],row['channel']]['measured_gain'])),2*sem,1e-7)
        sensitivity.append(dict(channel=row['channel'],frequency_Hz=row['frequency_Hz'],
             difference=float(abs(delta.mean())),paired_SEM=sem,tolerance=tolerance,
             passed=bool(abs(delta.mean())<=tolerance)))
    primary=[r for r in rows if r['primary']];estimable=[r for r in primary if r['estimable']]
    passed=all(r['passed'] for r in estimable) and len(estimable)==len(primary) and all(r['passed'] for r in sensitivity)
    result=dict(status='LOCAL_LINEAR_PASS' if passed else 'LOCAL_LINEAR_NOT_QUALIFIED',
        estimable=len(estimable),total=len(primary),passed=sum(r['passed'] for r in estimable),
        rows=rows,linearity_sensitivity=sensitivity,elapsed_s=time.time()-start,
        dynamic_conductance_validated=False,spatial_validated=False,formal_bifurcation_allowed=False)
    write(OUT/'result.json',result);write(OUT/'progress.json',dict(status=result['status'],elapsed_s=time.time()-start))
    print({k:v for k,v in result.items() if k not in ['rows','linearity_sensitivity']},flush=True)


if __name__=='__main__':main()
