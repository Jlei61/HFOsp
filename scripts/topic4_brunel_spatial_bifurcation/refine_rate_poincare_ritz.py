"""Re-extract full-delay eigenpairs from a saved filtered Poincare subspace.

The model, orbit, integration step and number of returned directions stay
fixed.  Real Rayleigh--Ritz on P M P separates eigenvectors mixed by the
polynomial filter.  No mode is discarded, and every lifted vector is applied
to the original monodromy again.  Old outputs are never overwritten.
"""
from complete_rate_positive_stability import *
from scipy.linalg import qr
import subprocess


def section_basis(vectors, phase):
    """A real orthonormal basis of the same section-projected complex span."""
    columns = np.concatenate([vectors.real, vectors.imag], axis=1)
    columns -= np.outer(phase, phase @ columns)
    basis, triangular, _ = qr(columns, mode='economic', pivoting=True)
    diagonal = abs(np.diag(triangular))
    rank = int(np.sum(diagonal > 1e-10 * diagonal.max()))
    if rank != vectors.shape[1]:
        raise ValueError(('Saved subspace is not full rank or conjugate closed',
                          rank, vectors.shape[1], diagonal.tolist()))
    basis = basis[:, :rank]
    assert np.max(abs(phase @ basis)) < 1e-10
    assert np.linalg.norm(basis.T @ basis - np.eye(rank)) < 1e-10
    return basis, diagonal[:rank]


def extract(basis, images, phase):
    """Galerkin eigenpairs; images are M Q, including the phase component."""
    section_images = images - np.outer(phase, phase @ images)
    small = basis.T @ section_images
    multipliers, coefficients = np.linalg.eig(small)
    order = np.argsort(abs(multipliers))[::-1]
    multipliers, coefficients = multipliers[order], coefficients[:, order]
    vectors = basis @ coefficients
    vectors /= np.linalg.norm(vectors, axis=0)
    return multipliers, vectors, small, section_images


def refine(source, device, output_label):
    source = Path(source)
    original = read(source)
    assert 'section_time_shift_coefficients' in original
    assert 'polynomial_filter_rho' in original
    with np.load(source.with_suffix('.npz')) as saved:
        vectors = np.vstack([saved['local_vectors'], saved['history_vectors']])
    s = RateField()
    m = Monodromy(s, original['orbit'], original['dt_ms'] * (1 + 1e-10),
                  device, stream_harmonics=True)
    assert vectors.shape == (m.dim, len(values(original)))
    assert abs(m.dt - original['dt_ms']) < 1e-12
    phase = m.phase_vector()
    phase /= np.linalg.norm(phase)
    phase_defect = float(np.linalg.norm(m.matvec(phase) - phase))
    basis, diagonal = section_basis(vectors, phase)
    del vectors
    images = np.column_stack([m.matvec(v) for v in basis.T])
    multipliers, vectors, small, section_images = extract(basis, images, phase)
    invariance = np.linalg.norm(section_images - basis @ small)
    relative_invariance = invariance / np.linalg.norm(section_images)
    del basis, images, section_images
    residuals, betas, full_residuals = [], [], []
    # Direct re-application checks the lifted vectors independently of Q*M*Q.
    for mu, vector in zip(multipliers, vectors.T):
        mv = m.matvec(vector.real) + 1j * m.matvec(vector.imag)
        beta = np.vdot(phase, mv)
        av = mv - phase * beta
        residuals.append(float(np.linalg.norm(av - mu * vector)))
        betas.append(beta)
        full_residuals.append(float(np.linalg.norm(mv - mu * vector - phase * beta)))
    rho = original['polynomial_filter_rho']
    transformed = multipliers * (multipliers - rho)
    old = np.asarray(original['transformed_eigenvalues'])
    old = old[:, 0] + 1j * old[:, 1] if old.ndim == 2 else old.astype(complex)
    ii, jj = linear_sum_assignment(abs(old[:, None] - transformed[None, :]))
    change = abs(old[ii] - transformed[jj]) / np.maximum(1., abs(old[ii]))
    filter_consistent = bool(max(change) < 1e-5)
    row = dict(original)
    row.update(multipliers=multipliers, residuals=residuals,
        phase_overlap=abs(phase @ vectors), phase_tangent_relative_defect=phase_defect,
        section_time_shift_coefficients=betas,
        full_state_generalized_eigen_residuals=full_residuals,
        transformed_eigenvalues=transformed, ritz_source=str(source),
        ritz_subspace_dimension=len(multipliers), ritz_basis_diagonal=diagonal,
        subspace_invariance_defect=float(invariance),
        subspace_relative_invariance_defect=float(relative_invariance),
        matched_filter_relative_changes=change,
        inherited_filter_spectrum_consistent=filter_consistent,
        smallest_returned_transformed_modulus=(max(
            original['smallest_returned_transformed_modulus'],
            float(min(abs(transformed)))) if filter_consistent else float('inf')),
        method='Full-history Poincare Rayleigh-Ritz on the entire saved filtered subspace; phase projection, no dropped modes, direct lifted full-monodromy residual checks.',
        seconds=time.time()-m.start,
        limitation='Numerical re-extraction at the same orbit and delay step. Filter coverage is inherited only after spectrum matching; paired steps remain mandatory and this is not a rigorous enclosure.')
    folder = PERIODIC_OUT/'poincare_ritz'
    folder.mkdir(exist_ok=True)
    output = folder/(output_label + '.json')
    save_periodic_array(output.with_suffix('.npz'), multipliers=multipliers,
        local_vectors=vectors[:9*s.P], history_vectors=vectors[9*s.P:],
        section_time_shift_coefficients=np.asarray(betas), dt=m.dt)
    write(output, row)
    print('POINCARE RITZ', str(output), 'mu', multipliers, 'residuals', residuals,
          'filter_consistent', filter_consistent, flush=True)
    return output, row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sources', nargs=2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--label', required=True)
    parser.add_argument('--min-free-gib', type=float, default=4.)
    parser.add_argument('--after-pids', type=int, nargs='*', default=[])
    parser.add_argument('--wait-for-sources', action='store_true',
        help='Wait for the specified JSON and vector files from an already running spectrum calculation')
    args = parser.parse_args()
    assert Path(args.label).name == args.label
    folder = DEST/'ritz_checks'
    folder.mkdir(exist_ok=True)
    dependencies = {pid: Path(f'/proc/{pid}/cmdline').read_bytes()
                    for pid in args.after_pids if Path(f'/proc/{pid}/cmdline').exists()}
    while dependencies:
        for pid, identity in list(dependencies.items()):
            process = Path(f'/proc/{pid}/cmdline')
            if not process.exists() or process.read_bytes() != identity:
                dependencies.pop(pid)
        if dependencies:
            write(folder/(args.label+'_worker.json'), dict(status='WAITING_DEPENDENCIES',
                pid=os.getpid(), dependencies=list(dependencies), timestamp=time.time()))
            time.sleep(30)
    rows, sources = [], []
    for index, source in enumerate(args.sources):
        if args.wait_for_sources:
            while not (Path(source).exists() and Path(source).with_suffix('.npz').exists()):
                write(folder/(args.label+'_worker.json'), dict(status='WAITING_SOURCE',
                    pid=os.getpid(), source=source, index=index, timestamp=time.time()))
                time.sleep(30)
        release(args.device)
        while True:
            free = float(subprocess.check_output(['nvidia-smi', '-i', str(args.device),
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True))
            if free >= args.min_free_gib * 1024:
                break
            print('WAITING_RESOURCE', free, flush=True)
            time.sleep(30)
        write(folder/(args.label+'_worker.json'), dict(status='REEXTRACTING',
            pid=os.getpid(), source=source, index=index, timestamp=time.time()))
        output = PERIODIC_OUT/'poincare_ritz'/f'{args.label}_{index}.json'
        if output.exists():
            row = read(output)
            assert Path(row['ritz_source']).resolve() == Path(source).resolve()
        else:
            output, row = refine(source, args.device, f'{args.label}_{index}')
        sources.append(str(output))
        rows.append(row)
    assert Path(rows[0]['orbit']).resolve() == Path(rows[1]['orbit']).resolve()
    assert rows[0]['dt_ms'] > rows[1]['dt_ms']
    classification = paired_modes(*rows)
    write(folder/(args.label+'.json'), dict(sources=sources, **classification))
    write(folder/(args.label+'_worker.json'), dict(status='REEXTRACTION_COMPLETE',
        pid=os.getpid(), timestamp=time.time(), classification=classification))
    print('PAIRED RITZ', classification, flush=True)


if __name__ == '__main__':
    main()
