"""Resolve the remaining spectrum after locking checked invariant directions.

For an invariant orthonormal basis V, A=PMP has blocks [B C; 0 D].
Arnoldi acts on D=Q A Q, Q=I-VV*, while eigenvectors of D are lifted by
(B-mu I)y=-V*A*v.  We retain B, all returned D modes, and verify every
lifted eigenvector with the original full-history monodromy.  Approximate
invariance is checked explicitly; no convergence tolerance is relaxed.
"""
from refine_rate_poincare_ritz import *


def lift_complement(basis, block, vector, image, multiplier):
    matrix = block - multiplier*np.eye(len(block))
    condition = float(np.linalg.cond(matrix))
    if condition > 1e10:
        raise ValueError(('Locked and complementary spectra are not separated', condition))
    coefficient = np.linalg.solve(matrix, -(basis.T @ image))
    lifted = vector + basis @ coefficient
    return lifted/np.linalg.norm(lifted), condition


def compute_complement(source, selected, device, label, nev=4):
    original = read(source)
    with np.load(Path(source).with_suffix('.npz')) as stored:
        seed = np.vstack([stored['local_vectors'][:, selected],
                          stored['history_vectors'][:, selected]])
    s = RateField()
    m = Monodromy(s, original['orbit'], original['dt_ms']*(1+1e-10),
                  device, stream_harmonics=True)
    assert m.dim == seed.shape[0] and abs(m.dt-original['dt_ms']) < 1e-12
    phase = m.phase_vector(); phase /= np.linalg.norm(phase)
    phase_defect = float(np.linalg.norm(m.matvec(phase)-phase))
    def project(v):
        return v-phase*np.dot(phase, v)
    def apply(v):
        return project(m.matvec(project(v)))
    locked, _ = section_basis(seed, phase)
    images = np.column_stack([apply(v) for v in locked.T])
    block = locked.T @ images
    invariance = float(np.linalg.norm(images-locked@block))
    if invariance > 1e-6:
        raise ValueError(('Locked subspace requires tighter invariance', invariance))
    def quotient(v):
        v = project(v)
        return v-locked@(locked.T@v)
    def complement(v):
        return quotient(apply(quotient(v)))
    rho = original['polynomial_filter_rho']
    def filtered(v):
        av = complement(v)
        return complement(av)-rho*av
    operator = LinearOperator((m.dim, m.dim), matvec=filtered, dtype=float)
    initial = quotient(np.random.default_rng(8126).normal(size=m.dim))
    transformed, vectors = eigs(operator, k=nev, which='LM',
        ncv=max(16, 3*nev+4), tol=2e-9, maxiter=120, v0=initial)
    # Diagonalize the original quotient operator within the entire returned
    # subspace, so polynomial degeneracies cannot mix the reported modes.
    vectors -= locked@(locked.T@vectors)
    basis, _ = section_basis(vectors, phase)
    images = np.column_stack([apply(v) for v in basis.T])
    small = basis.T @ images
    vals, coefficients = np.linalg.eig(small)
    all_vectors, conditions = [], []
    for mu, coefficient in zip(vals, coefficients.T):
        vector, condition = lift_complement(locked, block, basis@coefficient,
                                             images@coefficient, mu)
        all_vectors.append(vector); conditions.append(condition)
    locked_vals, locked_coefficients = np.linalg.eig(block)
    all_vectors += [v/np.linalg.norm(v) for v in (locked@locked_coefficients).T]
    complement_transformed = vals*(vals-rho)
    ii, jj = linear_sum_assignment(abs(transformed[:, None]-complement_transformed[None, :]))
    filter_change = abs(transformed[ii]-complement_transformed[jj])/np.maximum(1, abs(transformed[ii]))
    consistent = bool(max(filter_change) < 1e-5)
    vals = np.r_[vals, locked_vals]
    vectors = np.column_stack(all_vectors)
    order = np.argsort(abs(vals))[::-1]
    vals, vectors = vals[order], vectors[:, order]
    residuals, betas = [], []
    for mu, vector in zip(vals, vectors.T):
        mv = m.matvec(vector.real)+1j*m.matvec(vector.imag)
        beta = np.vdot(phase, mv)
        residuals.append(float(np.linalg.norm(mv-phase*beta-mu*vector)))
        betas.append(beta)
    row = dict(orbit=original['orbit'], J_EE_core=m.J, T_ms=m.T, dt_ms=m.dt,
        minimum_occupied_delay_ms=m.minimum_occupied_delay_ms,
        history_dimension=m.dim, multipliers=vals, residuals=residuals,
        phase_overlap=abs(phase@vectors), phase_tangent_relative_defect=phase_defect,
        full_state_generalized_eigen_residuals=residuals,
        section_time_shift_coefficients=betas, polynomial_filter_rho=rho,
        transformed_eigenvalues=vals*(vals-rho), filter_coverage_threshold=1-rho,
        smallest_returned_transformed_modulus=(max(min(abs(transformed)),
            min(abs(complement_transformed))) if consistent else float('inf')),
        complement_filter_relative_changes=filter_change,
        locked_subspace_dimension=locked.shape[1], locked_subspace_invariance_defect=invariance,
        complementary_subspace_dimension=nev, lift_condition_numbers=conditions,
        source=str(source), locked_source_indices=selected,
        method='Full Poincare spectrum: checked invariant block plus filtered Arnoldi on its orthogonal quotient; every lifted mode checked with the original full-delay operator.',
        seconds=time.time()-m.start,
        limitation='Numerical, paired-step spectrum only. Approximate invariant-subspace and full original residual checks are required; not a rigorous enclosure or an interval-wide statement.')
    output = PERIODIC_OUT/'poincare_ritz'/(label+'.json')
    save_periodic_array(output.with_suffix('.npz'), multipliers=vals,
        local_vectors=vectors[:9*s.P], history_vectors=vectors[9*s.P:],
        section_time_shift_coefficients=np.asarray(betas), dt=m.dt)
    write(output, row)
    print('COMPLEMENT SPECTRUM', vals, 'residuals', residuals,
          'invariance', invariance, flush=True)
    return output, row


def main():
    p=argparse.ArgumentParser();p.add_argument('sources', nargs=2)
    p.add_argument('--device', type=int, default=0);p.add_argument('--label', required=True)
    p.add_argument('--after-pids', type=int, nargs='*', default=[])
    p.add_argument('--min-free-gib', type=float, default=4.)
    p.add_argument('--lock',choices=['growing','all-reliable'],default='growing',
        help='Optionally retain checked stable modes in the locked block to expose the remaining filter spectrum')
    p.add_argument('--nev',type=int,default=4,help='Number of additional quotient modes')
    a=p.parse_args(); assert Path(a.label).name == a.label
    assert a.nev>=2
    folder=DEST/'ritz_checks';folder.mkdir(exist_ok=True)
    worker=folder/(a.label+'_worker.json')
    def status(state, **kw):
        write(worker, dict(status=state, pid=os.getpid(), timestamp=time.time(), **kw))
    dependencies={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.after_pids
                  if Path(f'/proc/{pid}/cmdline').exists()}
    while dependencies:
        for pid, identity in list(dependencies.items()):
            f=Path(f'/proc/{pid}/cmdline')
            if not f.exists() or f.read_bytes()!=identity:dependencies.pop(pid)
        if dependencies:status('WAITING_DEPENDENCIES', dependencies=list(dependencies));time.sleep(30)
    originals=[read(source) for source in a.sources]
    assert Path(originals[0]['orbit']).resolve()==Path(originals[1]['orbit']).resolve()
    verdict=paired_modes(*originals)
    coarse, fine=map(values, originals)
    ii, jj=linear_sum_assignment(abs(coarse[:,None]-fine[None,:]))
    lookup=dict(zip(jj,ii))
    mask=np.array(verdict['outside_unit_disk_mask'] if a.lock=='growing'
                  else verdict['reliable_mode_mask'],dtype=bool)
    for j in np.flatnonzero(mask):
        if max(originals[0]['phase_overlap'][lookup[j]],originals[1]['phase_overlap'][j])>=1e-6:
            mask[j]=False
    selected_fine=np.flatnonzero(mask)
    assert len(selected_fine)>0
    selected_coarse=np.array([lookup[j] for j in selected_fine])
    write(folder/(a.label+'_selection.json'),dict(lock_rule=a.lock,quotient_modes_requested=a.nev,
        sources=a.sources,selected_source_indices=[selected_coarse,selected_fine],
        source_paired_classification=verdict,
        scope='Selection for invariant-block checks only. All selected eigenvalues are retained in the final spectrum, including stable ones; no classification is assumed from the selection.'))
    rows=[];sources=[]
    for index, selected in enumerate([selected_coarse, selected_fine]):
        release(a.device)
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=a.min_free_gib*1024:break
            status('WAITING_RESOURCE', free_mib=free, index=index);time.sleep(30)
        status('COMPLEMENT_SPECTRUM', index=index, locked_mode_count=len(selected),lock_rule=a.lock,nev=a.nev)
        output=PERIODIC_OUT/'poincare_ritz'/f'{a.label}_{index}.json'
        if output.exists():
            row=read(output)
            assert Path(row['source']).resolve()==Path(a.sources[index]).resolve()
            assert row['locked_source_indices']==selected.tolist()
            assert row['complementary_subspace_dimension']==a.nev
        else:
            output,row=compute_complement(a.sources[index], selected, a.device, f'{a.label}_{index}',nev=a.nev)
        rows.append(row);sources.append(str(output))
    classification=paired_modes(*rows)
    write(folder/(a.label+'.json'), dict(sources=sources, **classification))
    status('COMPLEMENT_SPECTRUM_COMPLETE', classification=classification)


if __name__=='__main__':main()
