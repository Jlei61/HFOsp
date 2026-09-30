#!/usr/bin/env python3
"""Separate spatial recruitment modes from the rank-one global G response.

These are sampled feedback operators, not time-growth eigenvalues or a stability
certificate. The unchanged base conductance is present when its variation is held.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import LinearOperator,eigs,splu
from campaign import ROOT,read,write,sha
from direct_response_system import DirectDC
from direct_frequency_operator import coefficients,NAMES,OUT as OPERATORS

OUT=ROOT/'frequency_mode_review'


def complex_pair(x):return [float(np.real(x)),float(np.imag(x))]


def main(f):
    dest=OUT/f'frequency_{f:g}Hz';dest.mkdir(parents=True,exist_ok=True);assert not (dest/'contract.json').exists()
    write(dest/'contract.json',dict(status='REGISTERED_BEFORE_MODE_AND_GLOBAL_FEEDBACK_DECOMPOSITION',created_epoch=time.time(),
        question='Are the near-unit sampled modes concentrated in recruitment edges, and how much does dynamic globalG contribute to them?',
        method='Build L=A+u*w^T from all measured target channels. A retains the same baselineGraw but holds its perturbation fixed. Compute right/left nearunit eigenvectors, global-rate participation and regional mode mass. Project eight replica-block matrix actions through biorthogonal modes, and compare full/half amplitudes. Separately compute eta=w^T*(I-A)^-1*u, the exact rankone determinant factor1-eta at this frequency.',
        limits='Eigenvalues of L(iomega) are feedback gains, not time-growth rates. eta alone cannot certify full stability because A may have its own modes/poles. Firstorder replica-block SEM excludes finite-record, sourcegroup, fixedM and amplitude errors. No stability or bifurcation labels.',
        frequency_Hz=f,producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    e=DirectDC()
    if f==0:
        with np.load(ROOT/'all_target_dc_direct/measured_dc.npz') as z:allchi=z['gain'].astype(complex);blocks=z['replicate_block_gain'][:,:,0].astype(complex)
        W=e.W
    else:
        index={1.:0,5.:1}[f]
        with np.load(ROOT/f'all_target_frequency_direct/frequency_{index:02d}/measured_response.npz') as z:allchi=z['gain'];blocks=z['replicate_block_gain'][:,:,0]
        W=[sparse.load_npz(OPERATORS/f'frequency_{f:g}Hz'/f'{name}.npz').tocsr() for name in NAMES]
    def assemble(chi):
        C,u,D=coefficients(e,chi,f)
        A=sum(e.S@(w.multiply(c[:,None])) for c,w in zip(C,W)).tocsr();us=e.S@u
        L=(A+sparse.csr_matrix(np.outer(us,e.weights))).tocsr()
        return A,L,C,u,us
    A,L,C,u,us=assemble(allchi[:,:,0]);lu=splu((L-sparse.eye(e.P)).tocsc())
    inverse=LinearOperator(L.shape,matvec=lu.solve,dtype=complex)
    values,vectors=eigs(L,k=8,sigma=1.,OPinv=inverse,tol=1e-8,maxiter=400)
    left_inverse=LinearOperator(L.shape,matvec=lambda v:lu.solve(v,trans='H'),dtype=complex)
    lv,left=eigs(L.getH(),k=8,sigma=1.,OPinv=left_inverse,tol=1e-8,maxiter=400)
    rr,cc=linear_sum_assignment(abs(values[:,None]-lv[None,:].conj()));assert np.array_equal(rr,np.arange(8))
    assert max(abs(values-lv[cc].conj()))<1e-6
    left=left[:,cc];overlap=np.sum(left.conj()*vectors,axis=0)
    residual=np.array([np.linalg.norm(L@vectors[:,j]-values[j]*vectors[:,j]) for j in range(8)])
    lres=np.array([np.linalg.norm(L.getH()@left[:,j]-values[j].conj()*left[:,j]) for j in range(8)])
    assert max(residual)<1e-6 and max(lres)<1e-6
    B=splu((sparse.eye(e.P)-A).tocsc());s=B.solve(us);eta=e.weights@s
    adj=B.solve(e.weights.astype(complex),trans='H')
    WX=[w@vectors for w in W];Ws=[w@s for w in W];full_action=L@vectors;projections=[];eta_shifts=[]
    for b in range(8):
        cb,ub,_=coefficients(e,blocks[:,:,b],f)
        action=e.S@(sum(c[:,None]*x for c,x in zip(cb,WX))+ub[:,None]*(e.weights@vectors))
        projections.append(np.sum(left.conj()*(action-full_action),axis=0)/overlap)
        da_s=e.S@sum((c1-c0)*x for c1,c0,x in zip(cb,C,Ws))
        du=e.S@(ub-u);eta_shifts.append(adj.conj()@(da_s+du))
    projections=np.array(projections);sampling_sem=np.std(projections,axis=0,ddof=1)/np.sqrt(8)
    eta_sem=float(np.std(np.array(eta_shifts),ddof=1)/np.sqrt(8))
    Ah,Lh,*_=assemble(allchi[:,:,1]);Jh=splu((Lh-sparse.eye(e.P)).tocsc())
    halfvals,halfvec=eigs(Lh,k=8,sigma=1.,OPinv=LinearOperator(Lh.shape,matvec=Jh.solve,dtype=complex),tol=1e-8,maxiter=400)
    rr,cc=linear_sum_assignment(abs(values[:,None]-halfvals[None,:]));halfvals=halfvals[cc]
    Ch,uh,Dh=coefficients(e,allchi[:,:,1],f);eta_half=e.weights@splu((sparse.eye(e.P)-Ah).tocsc()).solve(e.S@uh)
    frozen_lu=splu((A-sparse.eye(e.P)).tocsc())
    frozen_values,_=eigs(A,k=8,sigma=1.,OPinv=LinearOperator(A.shape,matvec=frozen_lu.solve,dtype=complex),tol=1e-8,maxiter=400)
    rows=[];region=e.geo['group_region'];masks=[e.groupE&(region==j) for j in range(3)]+[~e.groupE]
    for j,value in enumerate(values):
        v=vectors[:,j];mass=e.sizes*abs(v);den=np.sqrt(np.average(abs(v[e.groupE])**2,weights=e.sizes[e.groupE]))
        global_action=us*(e.weights@v);sensitivity=left[:,j].conj()@global_action/overlap[j]
        rows.append(dict(eigenvalue=complex_pair(value),distance_to_one=float(abs(value-1)),
            eigen_residual=float(residual[j]),left_eigen_residual=float(lres[j]),eigenvalue_condition=float(1/abs(overlap[j])),
            firstorder_MC_eigenvalue_SEM=float(sampling_sem[j]),half_amplitude_eigenvalue=complex_pair(halfvals[j]),
            amplitude_eigenvalue_difference=float(abs(halfvals[j]-value)),
            global_rate_participation=float(abs(e.weights@v)/max(den,1e-30)),
            source_mode_absolute_mass_A_B_surround_I=[float(mass[m].sum()/mass.sum()) for m in masks],
            sensitivity_to_global_feedback_multiplier=complex_pair(sensitivity)))
    np.savez_compressed(dest/'modes.npz',eigenvalue=values,right=vectors,left=left,overlap=overlap,
        block_firstorder_eigenvalue_shifts=projections,half_amplitude_eigenvalue=halfvals,
        global_resolvent_source=s,global_resolvent_adjoint=adj,eta=eta,eta_block_firstorder_shifts=eta_shifts)
    result=dict(status='COMPLETE_SAMPLED_MODE_DECOMPOSITION',frequency_Hz=f,rows=rows,
        rank_one_global_feedback=dict(eta=complex_pair(eta),one_minus_eta=complex_pair(1-eta),
            firstorder_MC_SEM=eta_sem,half_amplitude_eta=complex_pair(eta_half),amplitude_difference=float(abs(eta-eta_half))),
        fixed_G_perturbation_loop_eigenvalues=[complex_pair(v) for v in frozen_values],
        dynamic_stability_established=False,formal_bifurcation_allowed=False)
    write(dest/'analysis.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frequency',type=float,choices=[0,1,5],required=True);a=p.parse_args();main(a.frequency)
