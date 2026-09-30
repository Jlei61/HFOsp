#!/usr/bin/env python3
"""Current-point static K sensitivity, not a stability or branch certificate."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator,gmres
from campaign import ROOT,read,write,sha
from held_direct_moments import HeldInputs

OUT=ROOT/'held_exit_K9p35_tangent'
DC=ROOT/'held_exit_phase_dc_operator_K9p35'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(DC/'joint_operator_qa.json')['status']=='PASS_SAME_EQUATION_LINEARISATION'
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_CURRENT_POINT_K_TANGENT',created_epoch=time.time(),
        question='At the relevant held K9.35 state, which population/feedback component responds to K, and how close is the local prediction to the R=200 feedback activation corner?',
        method='Exact current point input derivatives, measured40000targetDC withimplicitM andstationaryG. Independent finite-difference check ofK reversalterm. One bounded linear solve; report residual, field/coreactions and8block matrixaction uncertainty.',
        limits='Workingpoint is near-selfconsistent but independent nonlinear correction stillrunning. Tangent is a local static susceptibility, not a timeeigenvalue, stablebranch or bifurcation. Linear estimate of reaching the piecewiseGcorner is not a measured criticalK. Compare prior9.2→9.35finitechange descriptively only.',
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    e=HeldInputs();r=e.reference['source_rate_Hz'];M=e.reference['M'];g=e.reference['g'];physical=e.reference['physical'];K=e.K.copy()
    with np.load(DC/'measured_dc.npz') as z:chi=z['gain'][:,:,0];blocks=z['replicate_block_gain'][:,:,0]
    h=1+g;D=1+.0005*e.E*chi[:,0]/h;Kshape=K/9.35
    fk=Kshape*(chi[:,3]+chi[:,0]*(-30+17.662847938268442)/h)/D
    partialg=chi[:,3]-chi[:,0]*(-17.662847938268442-physical[:,0])/h+2*np.sum(chi[:,1:3]*physical[:,1:3],axis=1)/h
    values=[];eps=1e-4
    for sign in [1,-1]:
        e.K=Kshape*(9.35+sign*eps);p,gg,_,_=e.moments(r,M)
        values.append(np.sum(chi[:,:3]*(p-physical),axis=1)+partialg*(gg-g))
    e.K=K
    fd=(values[0]-values[1])/(2*eps);rel=float(np.linalg.norm(fd-fk*D)/np.linalg.norm(fk*D));assert rel<2e-5
    C=np.array([chi[:,0]*e.tm*e.area[0]/(1000*h),-chi[:,0]*e.Z*e.tm*e.area[1]/(1000*h),
        chi[:,1]*e.tm*e.area[0]**2/(1000*h**2),chi[:,2]*e.tm*(e.Z*e.area[1])**2/(1000*h**2)])/D
    u=e.E*e.Z*chi[:,3]/D;dG=.1*e.causal*e.weights
    assert 200<float(e.reference['stationary_R_Hz'])<500
    def action(v):return sum(c*(w@v) for c,w in zip(C,e.W))+u*(dG@v)
    J=LinearOperator((e.P,e.P),matvec=lambda v:e.S@action(v)-v,dtype=float)
    diagonal=-np.ones(e.P)
    for c,w in zip(C,e.W):diagonal+=e.S@(c*np.asarray(w[np.arange(e.N),e.group]).ravel())
    diagonal+=(e.S@u)*dG;diagonal=np.where(abs(diagonal)>.05,diagonal,-1.)
    pre=LinearOperator((e.P,e.P),matvec=lambda v:v/diagonal,dtype=float);hist=[]
    dr,info=gmres(J,-(e.S@fk),M=pre,rtol=1e-7,atol=1e-9,restart=100,maxiter=3,
        callback=lambda x:hist.append(float(x)),callback_type='pr_norm')
    target=action(dr)+fk;residual=e.S@target-dr
    region=e.geo['group_region'];masks=[e.groupE]+[e.groupE&(region==j) for j in range(3)]+[~e.groupE]
    regional=[float(np.average(dr[m],weights=e.sizes[m])) for m in masks]
    dR=float(e.causal*(e.weights@dr));dGraw=.1*dR;R=float(e.reference['stationary_R_Hz'])
    matrix=[]
    for b in range(8):
        value,Db=e.target_action(blocks[:,:,b],dr)
        value+=Kshape*(blocks[:,3,b]+blocks[:,0,b]*(-30+17.662847938268442)/h)/Db
        matrix.append(e.S@value-dr)
    matrixsem=np.array(matrix).std(0,ddof=1)/np.sqrt(8)
    previous=[read(ROOT/'carried_exit_lower_holds/analysis'/f'{n}.json') for n in ['held_K9p2','held_K9p35']]
    slope=(np.array(previous[1]['final3s_rates_allE_A_B_surround_Hz'])-previous[0]['final3s_rates_allE_A_B_surround_Hz'])/.15
    np.savez_compressed(OUT/'tangent.npz',source_Hz_per_K=dr,target_Hz_per_K=target,
        conditional_constraint_residual_Hz_per_K=residual,measured_matrix_action_SEM_Hz_per_K=matrixsem,
        source_K9p35=r,local_K_partial_Hz_per_K=fk)
    result=dict(status='COMPLETE_STATIC_K_SENSITIVITY' if info==0 else 'BOUNDED_LINEAR_SOLVE_UNRESOLVED',
        gmres_info=int(info),gmres_iterations=len(hist),K_parameter_JVP_relative_error=rel,
        maximum_linear_constraint_residual_Hz_per_K=float(abs(residual).max()),
        regional_Hz_per_K_allE_A_B_surround_I=regional,current_R_Hz=R,dR_Hz_per_K=dR,dGraw_per_K=dGraw,
        linear_deltaK_to_R200=float((200-R)/dR) if dR<0 else None,
        previous_held9p2_to9p35_finite_slope_allE_A_B_surround_Hz_per_K=slope.tolist(),
        maximum_measured_matrix_action_SEM_Hz_per_K=float(matrixsem.max()),
        scope='Local conditional static susceptibility before independentrootvalidation. ReachingR200 changes the stationaryGderivative formula, not automatically stability or equilibriumexistence. Finite0.15Khistorydifference is not an independentinfinitesimal derivative check. Eightblock actionSEM is not total tangent/closure uncertainty.',
        formal_bifurcation_allowed=False,producer_sha256=sha(__file__))
    write(OUT/'result.json',result);print(result,flush=True)


if __name__=='__main__':main()
