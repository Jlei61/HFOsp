#!/usr/bin/env python3
"""Check joint local/source linearisation and prepare the full target proposal."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from scipy import sparse
from campaign import ROOT,read,write,sha
from conditional_density_inputs import OPS

OUT=ROOT/'all_target_dc_direct'
EG=-17.662847938268442


def main(wait):
    while not (OUT/'analysis.json').exists():
        if not wait:return
        time.sleep(20)
    result=read(OUT/'analysis.json');assert result['status']=='COMPLETE_DC_OPERATOR_AND_UNVALIDATED_STEP'
    geo=dict(np.load(OPS/'geometry.npz'));params=read(OPS/'prepared.json')['params'];group=geo['cell_group'];sizes=geo['group_size'];N=len(group);P=len(sizes)
    E=np.arange(N)<32000;S=sparse.coo_matrix((1/sizes[group],(group,np.arange(N))),shape=(P,N)).tocsr()
    with np.load(OUT/'measured_dc.npz') as z:chi=z['gain'][:,:,0]
    with np.load(ROOT/'all_target_stationary_direct/inputs.npz') as z:M=z['M']
    with np.load(ROOT/'target_stationary_response_audit/response.npz') as z:
        physical=z['physical'];g=z['g'];Z=z['Z'];K=z['K'];IE=z['mean_input_E'];II=z['mean_input_I']
    with np.load(OUT/'newton_proposal.npz') as z:
        r=z['source_rate_Hz'];v=z['step_Hz'];local=z['local_adaptation_corrected_rate_Hz'];predicted=z['predicted_linear_residual_Hz'];D=z['adaptation_denominator']
    W=[sparse.load_npz(ROOT/'target_stationary_response_audit'/f'{name}_dc.npz').tocsr() for name in ['mean_ampa','mean_gaba','variance_ampa','variance_gaba']]
    h=1+g;tm=np.where(E,params['tau_m_E'],params['tau_m_I']);area=np.array([.1/(params[n]*(-np.expm1(-.1/params[n]))) for n in ['tau_r_AMPA','tau_r_GABA']])
    weight=np.where(geo['population']==0,sizes/32000,0.);causal=.1/(15*(-np.expm1(-.1/15)));R=causal*(weight@r)
    assert 200<R<500
    G=.1*(R-200);dG=.1*causal*weight
    # Recover the derivative with fixed effective inputs for a same-equation
    # finite difference. PhysicalG already contains the other three changes.
    partialg=chi[:,3]-chi[:,0]*(EG-physical[:,0])/h+2*(chi[:,1]*physical[:,1]+chi[:,2]*physical[:,2])/h
    def raw_action(direction):
        a,b,qA,qI=[w@direction for w in W]
        return chi[:,0]*tm*(area[0]*a-Z*area[1]*b)/(1000*h)+chi[:,1]*tm*area[0]**2*qA/(1000*h**2)+chi[:,2]*tm*(Z*area[1])**2*qI/(1000*h**2)+E*Z*chi[:,3]*(dG@direction)
    rng=np.random.default_rng(928891);checks=[]
    for scale in [1.,.1]:
        direction=r*rng.standard_normal(P)*.01*scale;localdirection=(M+1)*rng.standard_normal(N)*.01*scale
        va,vb,vqa,vqi=[w@direction for w in W];dg=float(dG@direction);eps=1e-4
        changes=[]
        for sign in [1,-1]:
            step=sign*eps;gg=E*Z*(G+step*dg)+K;hh=1+gg
            mu=(IE+step*tm*area[0]*va/1000-Z*(II+step*tm*area[1]*vb/1000)-.0005*(M+step*E*localdirection)+E*Z*(G+step*dg)*EG-30*K)/hh
            ve=(physical[:,1]*h*h+step*tm*area[0]**2*vqa/1000)/(hh*hh)
            vi=(physical[:,2]*h*h+step*tm*(Z*area[1])**2*vqi/1000)/(hh*hh)
            changes.append(chi[:,0]*(mu-physical[:,0])+chi[:,1]*(ve-physical[:,1])+chi[:,2]*(vi-physical[:,2])+partialg*(gg-g)-step*localdirection)
        measured=(changes[0]-changes[1])/(2*eps)
        expected=raw_action(direction)-D*localdirection
        error=float(np.linalg.norm(measured-expected)/max(np.linalg.norm(expected),1e-15))
        assert error<2e-5,error
        checks.append(dict(scale=scale,joint_local_source_JVP_relative_error=error))
    target=local+raw_action(v)/D
    constraint=S@target-(r+v)
    error=float(abs(constraint-predicted).max());assert error<1e-8,error
    cap=np.where(E,500.,1000.)
    np.savez_compressed(OUT/'joint_newton_proposal.npz',proposed_source_rate_Hz=r+v,
        proposed_target_rate_Hz=target,proposed_M=np.where(E,target,0.),source_consistency_residual_Hz=constraint,
        original_M=M,Z=Z,K=K)
    qa=dict(status='PASS_SAME_EQUATION_LINEARISATION',rows=checks,source_constraint_max_error_Hz=error,
        targets_below_zero=int((target<0).sum()),minimum_target_rate_Hz=float(target.min()),targets_above_cap=int((target>cap).sum()),
        minimum_proposed_E_M=float(target[E].min()),gmres_info=result['gmres_info'],
        clarification='The earlier local_adaptation_corrected_rate was at the old source input. This file also includes the proposed source-current and G changes. No proposed M/rate has been accepted without fresh nonlinear evaluation.',
        independent_response_validation=False,formal_bifurcation_allowed=False,producer_sha256=sha(__file__))
    write(OUT/'joint_operator_qa.json',qa);print(qa,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)
