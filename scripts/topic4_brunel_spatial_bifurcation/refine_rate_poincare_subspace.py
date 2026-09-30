"""Resolve individual modes inside an already computed return subspace.

Polynomial filtering can leave inaccurate individual weak modes when other
multipliers are much larger. Re-diagonalize the original projected return
operator on a real orthonormal basis and check each lifted eigenvector by a
fresh full-history propagation. Preserve the original spectrum and its
residuals. This does not add a missing invariant subspace or locate crossings.
"""
from rate_floquet_poincare import *
from scipy.linalg import svd, eig
from threadpoolctl import threadpool_limits


def refine(source,device=1):
    source=Path(source);q=read(source);arrays=np.load(source.with_suffix('.npz'))
    s=RateField()
    m=Monodromy(s,q['orbit'],q['dt_ms']*(1+1e-12),device,stream_harmonics=True)
    assert abs(m.dt-q['dt_ms'])<1e-13
    phase=m.phase_vector();phase/=np.linalg.norm(phase)
    phase_defect=float(np.linalg.norm(m.matvec(phase)-phase))
    v=np.r_[arrays['local_vectors'],arrays['history_vectors']]
    assert len(v)==m.dim
    v-=phase[:,None]*(phase@v)[None,:]
    with threadpool_limits(limits=4,user_api='blas'):
        basis,singular,_=svd(np.c_[v.real,v.imag],full_matrices=False,check_finite=False)
    rank=int(np.sum(singular>singular[0]*1e-11))
    assert rank==v.shape[1], ('Non-real or rank-deficient source subspace',rank,v.shape[1],singular)
    basis=basis[:,:rank].copy();del v
    images=np.empty_like(basis)
    for j in range(rank):
        y=m.matvec(basis[:,j]);images[:,j]=y-phase*(phase@y)
        print('RETURN SUBSPACE COLUMN',j+1,rank,flush=True)
    with threadpool_limits(limits=4,user_api='blas'):
        mu,coeff=eig(basis.T@images)
        vectors=basis@coeff
    vectors/=np.linalg.norm(vectors,axis=0)
    order=np.argsort(abs(mu))[::-1];mu=mu[order];vectors=vectors[:,order]
    residuals=[];betas=[];fullres=[]
    for j,z in enumerate(mu):
        v=vectors[:,j]
        mv=m.matvec(v.real)+1j*m.matvec(v.imag)
        beta=np.vdot(phase,mv)
        residual=float(np.linalg.norm(mv-beta*phase-z*v))
        residuals.append(residual);fullres.append(residual);betas.append(beta)
        print('DIRECT RITZ CHECK',j,z,residual,flush=True)
    rho=q['polynomial_filter_rho'];transformed=mu*(mu-rho)
    result=dict(q,multipliers=mu,residuals=residuals,
        phase_tangent_relative_defect=phase_defect,phase_overlap=abs(phase@vectors),
        section_time_shift_coefficients=np.asarray(betas),full_state_generalized_eigen_residuals=fullres,
        transformed_eigenvalues=transformed,
        smallest_returned_transformed_modulus=float(min(abs(transformed))),
        source_spectrum=str(source),subspace_singular_values=singular,
        subspace_rank=rank,
        method='Original full-history Poincare operator re-diagonalized in the real orthonormal source subspace; each lifted mode checked by a fresh propagation.',
        limitation='Refinement of the existing numerically selected subspace only; no missing-mode or branch-interval completeness proof.')
    output=source.with_name(source.stem+'_ritz.json')
    write(output,result)
    save_periodic_array(output.with_suffix('.npz'),multipliers=mu,
        local_vectors=vectors[:9*s.P],history_vectors=vectors[9*s.P:],
        section_time_shift_coefficients=np.asarray(betas),dt=m.dt)
    print('REFINED RETURN SPECTRUM',output,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--device',type=int,default=1)
    a=p.parse_args();refine(a.source,a.device)
