"""Recover monodromy eigenpairs in a filtered Arnoldi invariant subspace.

The polynomial M(M-rho I) can cluster distinct M eigenvalues. Diagonalizing
the small projected M, instead of separate Rayleigh quotients of each filter
vector, resolves mixing when the saved subspace contains both directions.
Every lifted eigenpair is checked against the full variational delay system.
"""
from rate_floquet import *
from rate_stability_coverage import assess


def main():
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--device',type=int,default=0)
    p.add_argument('--dt',type=float,default=.1);p.add_argument('--survey-site')
    a=p.parse_args();source=Path(a.source);q=read(source);z=np.load(source.with_suffix('.npz'))
    assert 'polynomial_filter_rho' in q
    V=np.vstack([z['local_vectors'],z['rate_history_vectors']]);s=RateField();m=Monodromy(s,q['orbit'],a.dt,a.device)
    assert V.shape[0]==m.dim and abs(m.dt-q['dt_ms'])<1e-12
    Q,_=np.linalg.qr(V);MQ=np.column_stack([m.matvec(v.real)+1j*m.matvec(v.imag) for v in Q.T])
    B=Q.conj().T@MQ;vals,U=np.linalg.eig(B);vec=Q@U;mv=MQ@U
    ix=np.argsort(abs(vals))[::-1];vals=vals[ix];vec=vec[:,ix];mv=mv[:,ix]
    residual=np.linalg.norm(mv-vec*vals,axis=0)/np.linalg.norm(vec,axis=0)
    phase=m.phase_vector();phase_error=float(np.linalg.norm(m.matvec(phase)-phase)/np.linalg.norm(phase))
    overlaps=abs(vec.conj().T@phase)/(np.linalg.norm(vec,axis=0)*np.linalg.norm(phase));neutral=int(np.argmin(abs(vals-1)))
    if abs(vals[neutral]-1)>.01 or overlaps[neutral]<.995:neutral=None
    nontrivial=np.delete(vals,neutral) if neutral is not None else vals
    refined=dict(q);refined.update(multipliers=vals,residuals=residual,phase_tangent_relative_defect=phase_error,
        phase_eigenvector_overlaps=overlaps,identified_neutral_index=neutral,nontrivial_multipliers=nontrivial,
        phase_multiplier_error=float(min(abs(vals-1))),largest_computed_nontrivial_modulus=float(max(abs(nontrivial))),
        raw_multipliers_above_1p001=int(np.sum(abs(vals)>1.001)),ritz_source=str(source),
        subspace_relative_invariance_defect=float(np.linalg.norm(MQ-Q@B)/np.linalg.norm(MQ)),
        method='Full-delay monodromy Rayleigh-Ritz in the saved filtered Arnoldi subspace; full lifted eigenpair residual checks.')
    dest=source.with_name(source.stem+'_ritz.json');write(dest,refined)
    np.savez_compressed(dest.with_suffix('.npz'),multipliers=vals,local_vectors=vec[:9*s.P],rate_history_vectors=vec[9*s.P:],dt=m.dt)
    verdict=assess(refined);write(dest.with_name(dest.stem+'_verdict.json'),dict(**verdict,source=str(dest)))
    if a.survey_site and verdict['status'] in ['NUMERICALLY_STABLE','UNSTABLE']:
        target=PERIODIC_OUT/'stability_coverage'/(a.survey_site+'.json');old=read(target)
        assert Path(old['analyzed_orbit']).resolve()==Path(q['orbit']).resolve()
        old.setdefault('original_monodromy_classification',{k:old.get(k) for k in ['status','margin','reliable_outside_count','floquet_source']})
        old.update(**verdict,floquet_source=str(dest),ritz_refinement=str(dest),
            classification_method=refined['method']);write(target,old)
    print('RITZ REFINEMENT',dest.name,verdict,'residuals',residual,'mu',vals,flush=True)


if __name__=='__main__':main()
