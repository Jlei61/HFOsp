#!/usr/bin/env python3
"""Collect measured DC operators and propose one independently testable step.

The proposed Newton step is diagnostic, not an accepted new root. No dynamic
stability labels or bifurcation claims are computed from this static matrix.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator,gmres
from campaign import ROOT,read,write,sha
from conditional_density_inputs import OPS
from measure_all_target_dc import OUT,SOURCE


def main(wait):
    if not (OUT/'analysis_contract.json').exists():
        write(OUT/'analysis_contract.json',dict(status='REGISTERED_BEFORE_COMPLETE_DC_ARRAY',created_epoch=time.time(),
            task='Collect every target/channel/amplitude, retain estimator uncertainty, construct the same graph DC selfconsistency derivative with implicit localM and stationaryG. At most one bounded GMRES correction proposal; no nonlinear iteration or eigenvalue interpretation.',
            units='Source and target rates inHz; divide presynaptic rate by1000 in all current mean/variance actions. Physical appliedG derivative already includes voltage/current noise rescaling. I has no G or M feedback.',
            limits='A measured local stationary near-root is not an exact equilibrium. Proposed correction must be tested with fresh direct local input-response arrays before acceptance. Static matrix is not full delayed dynamical stability.',
            producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    while not all((OUT/f'part_{part}/result.json').exists() for part in range(2)):
        write(OUT/'analysis_progress.json',dict(status='WAITING_TWO_FIXED_PARTITIONS',pid=os.getpid(),updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    assert all(read(OUT/f'part_{part}/result.json')['status']=='COMPLETE' for part in range(2))
    write(OUT/'analysis_progress.json',dict(status='ASSEMBLING',pid=os.getpid(),updated_epoch=time.time()))
    N=40000;gain=np.full((N,4,2),np.nan);sem=gain.copy();meanrate=gain.copy();block=np.full((N,4,2,8),np.nan)
    dsem=np.full((N,4),np.nan);passed=np.zeros((N,4),bool);coverage=np.zeros((N,4,2),int)
    for part in range(2):
        for path in sorted((OUT/f'part_{part}').glob('batch_*.npz')):
            with np.load(path) as z:
                cell=z['cell'];ch=z['channel'];a=np.arange(len(cell))%2
                assert np.array_equal(cell[::2],cell[1::2]) and np.array_equal(ch[::2],ch[1::2])
                assert not coverage[cell,ch,a].any()
                gain[cell,ch,a]=z['gain'];sem[cell,ch,a]=z['SEM'];block[cell,ch,a]=z['replicate_block_gain']
                meanrate[cell,ch,a]=z['mean_rate_Hz'];coverage[cell,ch,a]+=1
                dsem[cell[::2],ch[::2]]=z['amplitude_difference_SEM'];passed[cell[::2],ch[::2]]=z['amplitude_pass']
    E=np.arange(N)<32000;expected=np.ones((N,4,2),int);expected[~E,3]=0
    assert np.array_equal(coverage,expected),np.unique(coverage-expected,return_counts=True)
    assert np.isfinite(gain[expected>0]).all()
    # Structurally absent I conductance response is exactly zero in the model;
    # weak measured components elsewhere are never zeroed.
    gain[~E,3]=0;sem[~E,3]=0;block[~E,3]=0
    np.savez_compressed(OUT/'measured_dc.npz',gain=gain,SEM=sem,replicate_block_gain=block,
        paired_amplitude_SEM=dsem,amplitude_pass=passed,mean_rate_Hz=meanrate,coverage=coverage)
    geo=dict(np.load(OPS/'geometry.npz'));p=read(OPS/'prepared.json')['params'];group=geo['cell_group'];sizes=geo['group_size'];P=len(sizes)
    S=sparse.coo_matrix((1/sizes[group],(group,np.arange(N))),shape=(P,N)).tocsr()
    with np.load(SOURCE/'inputs.npz') as z:g=z['g'];M=z['M'];r=z['source_per_ms']*1000
    with np.load(ROOT/'target_stationary_response_audit/response.npz') as z:Z=z['Z'];K=z['K']
    with np.load(SOURCE/'response.npz') as z:rho=z['cell_rate_Hz'];rho_sem=z['cell_SEM_Hz']
    W=[sparse.load_npz(ROOT/'target_stationary_response_audit'/f'{name}_dc.npz').tocsr()
        for name in ['mean_ampa','mean_gaba','variance_ampa','variance_gaba']]
    h=1+g;tm=np.where(E,p['tau_m_E'],p['tau_m_I'])
    area=np.array([.1/(p[n]*(-np.expm1(-.1/p[n]))) for n in ['tau_r_AMPA','tau_r_GABA']])
    weights=np.where(geo['population']==0,sizes/32000,0.);causal=.1/(15*(-np.expm1(-.1/15)))
    R=float(weights@r*causal);dG=.1*causal*weights if 200<R<500 else np.zeros(P)
    def coefficients(chi):
        D=1+.0005*E*chi[:,0]/h
        if D.min()<=.5:raise RuntimeError('Measured local adaptation denominator is not safely positive.')
        C=np.array([chi[:,0]*tm*area[0]/(1000*h),-chi[:,0]*Z*tm*area[1]/(1000*h),
            chi[:,1]*tm*area[0]**2/(1000*h**2),chi[:,2]*tm*(Z*area[1])**2/(1000*h**2)])/D
        u=E*Z*chi[:,3]/D
        return C,u,D
    C,u,D=coefficients(gain[:,:,0])
    def action(v):return S@(sum(c*(w@v) for c,w in zip(C,W))+u*(dG@v))-v
    J=LinearOperator((P,P),matvec=action,dtype=float)
    diagonal=-np.ones(P)
    for c,w in zip(C,W):diagonal+=S@(c*np.asarray(w[np.arange(N),group]).ravel())
    diagonal+=(S@u)*dG
    safe=np.where(abs(diagonal)>.05,diagonal,-1.)
    pre=LinearOperator((P,P),matvec=lambda v:v/safe,dtype=float)
    a=.0005*E*gain[:,0,0]/h
    local_corrected=rho+a*(M-rho)/D
    residual=S@local_corrected-r
    errors=[]
    step,info=gmres(J,-residual,M=pre,rtol=1e-7,atol=1e-9,restart=100,maxiter=3,
        callback=lambda e:errors.append(float(e)),callback_type='pr_norm')
    cap=np.where(geo['population']==0,500.,1000.)
    proposal=np.clip(r+step,0,cap*(1-1e-12));actual_step=proposal-r
    unclipped=float(abs(actual_step-step).max());linear_residual=residual+action(actual_step)
    # Eight independent replica blocks preserve within-channel amplitude pairing.
    # This estimates sampling uncertainty of the matrix action, not native noise.
    inputs=[w@actual_step for w in W];globalchange=float(dG@actual_step);actions=[]
    for b in range(8):
        cb,ub,_=coefficients(block[:,:,0,b]);actions.append(S@(sum(c*v for c,v in zip(cb,inputs))+ub*globalchange)-actual_step)
    action_sem=np.array(actions).std(0,ddof=1)/np.sqrt(8)
    counts=geo['group_size'];groupE=geo['population']==0;cellregion=geo['group_region'][group]
    summaries=[]
    for label,mask in [('E',E),('I',~E)]:
        for channel,name in enumerate(['effective_mean','varianceE','varianceI','physicalG']):
            if label=='I' and channel==3:continue
            estimable=abs(gain[mask,channel,0])/np.maximum(sem[mask,channel,0],1e-15)>=10
            summaries.append(dict(population=label,channel=name,targets=int(mask.sum()),estimable=int(estimable.sum()),
                amplitude_pass_among_estimable=int(passed[mask,channel][estimable].sum()),
                amplitude_fail_among_estimable=int((~passed[mask,channel][estimable]).sum()),
                nonestimable=int((~estimable).sum())))
    result=dict(status='COMPLETE_DC_OPERATOR_AND_UNVALIDATED_STEP',summary=summaries,
        minimum_adaptation_denominator=float(D.min()),maximum_fixedM_corrected_residual_Hz=float(abs(residual).max()),
        E_weighted_residual_RMS_Hz=float(np.sqrt(np.average(residual[groupE]**2,weights=counts[groupE]))),
        I_weighted_residual_RMS_Hz=float(np.sqrt(np.average(residual[~groupE]**2,weights=counts[~groupE]))),
        gmres_info=int(info),gmres_iterations=len(errors),maximum_proposed_group_change_Hz=float(abs(actual_step).max()),
        clipping_maximum_Hz=unclipped,maximum_predicted_linear_residual_Hz=float(abs(linear_residual).max()),
        maximum_matrix_action_MC_SEM_Hz=float(action_sem.max()),
        full_nonlinear_validation_required=True,dynamic_stability_established=False,formal_bifurcation_allowed=False,
        interpretation='One measured DC operator and one bounded proposal, not an accepted new root. All amplitude failures remain; eight-block sampling uncertainty is separate from approximation and native fluctuations.',
        producer_sha256=sha(__file__))
    np.savez_compressed(OUT/'newton_proposal.npz',source_rate_Hz=r,proposed_source_rate_Hz=proposal,
        step_Hz=actual_step,local_adaptation_corrected_rate_Hz=local_corrected,residual_Hz=residual,
        predicted_linear_residual_Hz=linear_residual,matrix_action_MC_SEM_Hz=action_sem,
        direct_cell_rate_Hz=rho,direct_cell_SEM_Hz=rho_sem,Z=Z,K=K,g=g,
        source_diagonal=diagonal,adaptation_denominator=D)
    write(OUT/'analysis.json',result);write(OUT/'analysis_progress.json',dict(status=result['status'],updated_epoch=time.time()))
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)
