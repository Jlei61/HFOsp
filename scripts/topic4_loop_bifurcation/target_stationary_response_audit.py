#!/usr/bin/env python3
"""Check the frozen static response at the validated target-resolved K9 state.

This is not a root, continuation, or stability computation. It asks whether
the old static surrogate can even represent the current relevant working point.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time
from types import SimpleNamespace
import numpy as np
import torch
from scipy import sparse
from campaign import ROOT,read,write,sha
from conditional_density_inputs import OPS
from prepare_target_density import OUT as TARGET
from stationary_native_diagnostic import Stationary,Response,load_models

OUT=ROOT/'target_stationary_response_audit'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(TARGET/'individual/result.json')['status']=='COMPLETE'
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_LOCAL_STATIC_SCORING',created_epoch=time.time(),
        question='At the now spatially corresponding actual-field K9 high state, does the frozen v3 static response return the observed density group rates?',
        design='Read-only one working-point audit, no root, fit or network simulation. Preserve individual target weights/Z/K/M and groupthresholds; feed sourcegroup meanrates andexpectedexternaldrive from elapsed5-10s. G is computed from the exact discrete causalR DC relation, and observedmeanG compared.',
        adaptation='Use observed endpoint individualM, not a hidden groupaverage; compare its E rate equivalent with measuredgroup rates. This is an input-response residual audit, not a selfconsistent equilibrium.',
        limits='v3 static response was validated on its own exact-colored-current assay; dynamic/DC validation previouslyfailed. Here both spatialdomain and local surrogate errors remain inspectable; no automatic continuationpermission.',
        model_sha256=sha(ROOT/'conductance_static_v3/locked_model.pt'),producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    started=time.time();torch.set_num_threads(3)
    geo=dict(np.load(OPS/'geometry.npz'));prep=read(OPS/'prepared.json');p=prep['params']
    group=geo['cell_group'];P=len(geo['group_size']);N=len(group);E=np.arange(N)<32000
    with np.load(TARGET/'individual/trajectory.npz') as z:
        measured=z['group_output'][5000:10000].mean(0);observedG=float(30*z['global_s'][5000:10000].mean())
    with np.load(TARGET/'individual/final_state.npz') as z:
        state=z['state'].mean(1)
    r=measured[0]/1000.;M=state[:,5];Z=state[:,6];K=state[:,7]
    nu=np.load(ROOT/'exit_branch_density/drive_0p1ms.npy',mmap_mode='r')[50000:100000].mean(0)[group]
    tm=np.where(E,p['tau_m_E'],p['tau_m_I']);jext=np.where(E,p['J_ext_E'],p['J_ext_I'])
    area=np.array([.1/(p[n]*(1-np.exp(-.1/p[n]))) for n in ['tau_r_AMPA','tau_r_GABA']])
    input_channels=[];operator_checks=[]
    for name in ['mean_ampa','mean_gaba','variance_ampa','variance_gaba']:
        a=sparse.load_npz(TARGET/f'{name}.npz').tocoo()
        w=sparse.coo_matrix((a.data,(a.row,a.col%P)),shape=(N,P)).tocsr()
        value=w@r
        # Independent delayed-matrix action at a constant source history.
        oracle=a.tocsr()@np.tile(r,prep['max_delay_steps'])
        error=float(abs(oracle-value).max());assert error<1e-9
        input_channels.append(value);operator_checks.append(dict(channel=name,error=error))
        sparse.save_npz(OUT/f'{name}_dc.npz',w)
        del a,w,oracle
    meanE=float(np.average(r[geo['population']==0],weights=geo['group_size'][geo['population']==0])*1000)
    causal_factor=.1/(15*(-np.expm1(-.1/15)))
    causalR=meanE*causal_factor;G=30*np.clip((causalR-200)/300,0,1)
    g=E*Z*G+K;h=1+g
    IE=tm*area[0]*(input_channels[0]+jext*nu);II=tm*area[1]*input_channels[1]
    physical=np.c_[(IE-Z*II-.0005*M+E*Z*G*(-17.662847938268442)-30*K)/h,
         tm*area[0]**2*(input_channels[2]+jext**2*nu)/h**2,
         tm*(Z*area[1])**2*input_channels[3]/h**2]
    assert np.isfinite(physical).all() and physical[:,1:].min()>=0
    model=Stationary.__new__(Stationary);model.s=SimpleNamespace(P=N,E=E,theta=geo['threshold_mv'][group])
    model.nets,model.bases,_=load_models();model.response=Response().double()
    model.response.load_state_dict(torch.load(ROOT/'conductance_static_v3/locked_model.pt',map_location='cpu',weights_only=False)['model']);model.response.eval()
    rate,grad=model.phi(physical,g)
    aggregate=lambda v:np.bincount(group,weights=v,minlength=P)/geo['group_size']
    predicted=aggregate(rate)*1000;size=geo['group_size'];pop=geo['population'];region=geo['group_region'];rows=[]
    for label,mask in [('allE',pop==0),('coreA',(pop==0)&(region==0)),('coreB',(pop==0)&(region==1)),('surround',(pop==0)&(region==2)),('I',pop==1)]:
        error=predicted[mask]-measured[0,mask]
        rows.append(dict(region=label,measured_density_rate_Hz=float(np.average(measured[0,mask],weights=size[mask])),
             frozen_static_rate_Hz=float(np.average(predicted[mask],weights=size[mask])),
             group_weighted_RMS_Hz=float(np.sqrt(np.average(error**2,weights=size[mask]))),
             group_weighted_MAE_Hz=float(np.average(abs(error),weights=size[mask])),max_group_error_Hz=float(abs(error).max())))
    x=(physical[E,0]-11)/(model.s.theta[E]-11);sig=np.sqrt(physical[E,1:])/(model.s.theta[E,None]-11)
    outside=(x < -10)|(x>30)|(sig>12).any(1)|(g[E]>32)
    write(OUT/'implementation_qa.json',dict(status='PASS',operator_checks=operator_checks,discrete_causal_rate_factor=float(causal_factor),
          G_from_stationary_mean=float(G),observed_mean_G=observedG))
    np.savez_compressed(OUT/'response.npz',physical=physical,g=g,predicted_target_rate_Hz=rate*1000,
          predicted_group_rate_Hz=predicted,measured_group_rate_Hz=measured[0],gradient=grad,Z=Z,K=K,M=M,
          mean_input_E=IE,mean_input_I=II,mean_source_rate_per_ms=r,mean_external_rate_per_ms=nu,
          group_mean_M=aggregate(M),projected_mean_IE=aggregate(IE),observed_mean_IE=measured[4],
          projected_mean_applied_II=aggregate(Z*II),observed_mean_applied_II=measured[5])
    result=dict(status='COMPLETE_ONE_POINT_RESPONSE_AUDIT',rows=rows,G_from_mean_rate=float(G),observed_G=observedG,
          E_fraction_outside_static_training_domain=float(outside.mean()),
          E_M_rate_equivalent_Hz=float(M[E].mean()),measured_E_rate_Hz=meanE,elapsed_s=time.time()-started,
          root_established=False,formal_bifurcation_allowed=False)
    write(OUT/'result.json',result);print(result,flush=True)


if __name__=='__main__':main()
