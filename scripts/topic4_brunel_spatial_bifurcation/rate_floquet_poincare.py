"""Full-delay first-return linearization, removing the autonomous phase mode.

For the exact monodromy M and phase tangent f, P M P restricted to f-perp
has the nontrivial Floquet multipliers, P=I-ff*/(f*f). This avoids mistaking
the universal phase multiplier for a cycle-fold mode near +1. Integration
error still matters: paired time steps, the phase defect and the original
full-state generalized eigen-equation are reported explicitly.
"""
from rate_floquet import *
from scipy.optimize import linear_sum_assignment


def compute_return(path,dt=.1,nev=8,device=0,ncv=24,stream_harmonics=False,output_label=None):
    s=RateField();m=Monodromy(s,path,dt,device,stream_harmonics=stream_harmonics);phase=m.phase_vector();phase/=np.linalg.norm(phase)
    mp=m.matvec(phase);phase_defect=float(np.linalg.norm(mp-phase))
    def project(v):return v-phase*np.dot(phase,v)
    def apply(v):return project(m.matvec(project(v)))
    rho=np.exp(-m.T/1000)
    def polynomial(v):
        av=apply(v);return apply(av)-rho*av
    target=LinearOperator((m.dim,m.dim),matvec=polynomial,dtype=float)
    v0=project(np.random.default_rng(7301).normal(size=m.dim))
    transformed,vec=eigs(target,k=nev,which='LM',ncv=ncv,tol=2e-7,maxiter=120,v0=v0)
    vals=[];res=[];betas=[];fullres=[]
    for i in range(nev):
        v=vec[:,i];mv=m.matvec(v.real)+1j*m.matvec(v.imag)
        av=mv-phase*np.vdot(phase,mv);mu=np.vdot(v,av)/np.vdot(v,v)
        beta=np.vdot(phase,mv)
        vals.append(mu);betas.append(beta)
        res.append(np.linalg.norm(av-mu*v)/np.linalg.norm(v))
        fullres.append(np.linalg.norm(mv-mu*v-beta*phase)/np.linalg.norm(v))
    ix=np.argsort(abs(np.asarray(vals)))[::-1]
    vals=np.array(vals)[ix];vec=vec[:,ix];res=np.array(res)[ix];betas=np.array(betas)[ix];fullres=np.array(fullres)[ix]
    row=dict(orbit=str(path),J_EE_core=m.J,T_ms=m.T,dt_ms=m.dt,history_dimension=m.dim,
        minimum_occupied_delay_ms=m.minimum_occupied_delay_ms,
        bounded_harmonic_indices=m.bounded_harmonic_indices,
        multipliers=vals,residuals=res,phase_tangent_relative_defect=phase_defect,
        phase_overlap=abs(phase@vec),section_time_shift_coefficients=betas,
        full_state_generalized_eigen_residuals=fullres,
        polynomial_filter_rho=rho,transformed_eigenvalues=transformed,
        filter_coverage_threshold=1-rho,smallest_returned_transformed_modulus=float(min(abs(transformed))),
        method='Full nine-state plus delay history Poincare return linearization P M P; autonomous phase removed by a section.',
        seconds=time.time()-m.start,
        limitation='Numerical spectrum at this orbit only. Near-unit conclusions require time-step refinement; no continuum completeness claim.')
    folder=PERIODIC_OUT/'poincare_floquet';folder.mkdir(exist_ok=True)
    name=f'{output_label or Path(path).stem}_dt{dt:g}'
    write(folder/(name+'.json'),row)
    save_periodic_array(folder/(name+'.npz'),multipliers=vals,local_vectors=vec[:9*s.P],
        history_vectors=vec[9*s.P:],section_time_shift_coefficients=betas,dt=m.dt)
    print('RETURN FLOQUET',row,flush=True);return row


def values(q):
    v=np.asarray(q['multipliers'])
    return v[:,0]+1j*v[:,1] if v.ndim==2 else v.astype(complex)


def filter_spectrum_covered(q,safety=.9):
    """A missing/null coverage bound is unresolved, never a stable spectrum.

    JSON stores a deliberately failed infinite bound as null.  Preserve
    that negative verdict when a saved Ritz result is read back.
    """
    bound=q.get('smallest_returned_transformed_modulus')
    threshold=q.get('filter_coverage_threshold')
    return bool(bound is not None and threshold is not None and
                np.isfinite(bound) and np.isfinite(threshold) and
                0<=bound<safety*threshold and threshold>0)


def paired_verdict(coarse,fine):
    for q in [coarse,fine]:
        if q['dt_ms']>q.get('minimum_occupied_delay_ms',.1)*(1+1e-12):
            raise ValueError('Step comparison contains a Heun step exceeding the minimum physical delay')
    a,b=values(coarse),values(fine)
    ii,jj=linear_sum_assignment(abs(a[:,None]-b[None,:]))
    changes=np.full(len(b),np.inf);changes[jj]=abs(a[ii]-b[jj])
    relevant=abs(b)>.8
    delta=float(max(changes[relevant])) if relevant.any() else float(max(changes))
    margin=max(2e-5,6*delta,4*fine['phase_tangent_relative_defect'])
    res=np.array(fine['residuals'])/np.maximum(1,abs(b));reliable=res<1e-6
    covered=all(filter_spectrum_covered(q) for q in [coarse,fine])
    outside=reliable&(abs(b)>1+margin)
    verdict='UNRESOLVED'
    if outside.any():verdict='UNSTABLE'
    elif covered and reliable.all() and max(fine['phase_overlap'])<1e-6 and max(abs(b))<1-margin:
        verdict='NUMERICALLY_STABLE'
    return dict(status=verdict,reliable_outside_count=int(outside.sum()),margin=margin,
        matched_multiplier_changes=changes,largest_relevant_change=delta,
        maximum_nontrivial_modulus=float(max(abs(b))),filter_coverage=covered,
        meaning='Paired-step numerical classification with a safety margin; not a rigorous spectrum enclosure.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--dt',type=float,nargs='+',default=[.1,.05])
    p.add_argument('--device',type=int,default=0);p.add_argument('--nev',type=int,default=8);p.add_argument('--ncv',type=int,default=24)
    p.add_argument('--stream-harmonics',action='store_true',help='Retain exact harmonics with bounded operator-bank memory')
    a=p.parse_args();rows=[]
    import gc
    for dt in a.dt:
        f=PERIODIC_OUT/'poincare_floquet'/f'{Path(a.orbit).stem}_dt{dt:g}.json'
        rows.append(read(f) if f.exists() else compute_return(a.orbit,dt,a.nev,a.device,a.ncv,a.stream_harmonics))
        gc.collect()
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    if len(rows)>=2:
        q=paired_verdict(rows[-2],rows[-1]);q['orbit']=str(a.orbit);q['dt_ms']=[r['dt_ms'] for r in rows[-2:]]
        write(PERIODIC_OUT/'poincare_floquet'/f'{Path(a.orbit).stem}_step_check.json',q)
        print('RETURN STEP CHECK',q,flush=True)
