"""CPU full-state direction diagnostic near the candidate TR2 saddle cycle.

Use all spatial groups and nine local states. The default basis is at a
nearby J; --exact-j requires the paired spectrum of the matched target.
Either calculation is a direction diagnostic, not a manifold connection.
"""
from check_rate_torus_full_state_approach import coefficients
from rate_field import RateField, ROOT, read, write, np
from pathlib import Path
import math
import time
import argparse

PER = ROOT/'results/topic4_sef_hfo/interictal_spatial_rate_only_20260917/periodic_completion'
OUT = Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920/TR2_saddle_directions')


def state_coefficients(s, c, lam, J, name):
    """Retain every harmonic; exploit only the real-field conjugate symmetry."""
    path = OUT/(name+'_local_coefficients.npy')
    shape = (9, len(c), s.P)
    meta = dict(J=float(J), shape=list(shape), frequencies=lam.imag.tolist())
    if path.exists() and path.with_suffix('.json').exists():
        assert read(path.with_suffix('.json')) == meta
        return np.load(path, mmap_mode='r')
    out = np.lib.format.open_memmap(path, mode='w+', dtype=complex, shape=shape)
    assert np.allclose(c, c[::-1].conj(), rtol=0, atol=1e-17)
    assert np.allclose(lam, -lam[::-1], rtol=0, atol=1e-14)
    for i in range((len(c)+1)//2):
        out[:, i] = s.eigenstate(None, J, lam[i], c[i])
        out[:, -1-i] = out[:, i].conj()
        if i % 128 == 0:
            print('LOCAL COEFFICIENTS', name, i, len(c), flush=True)
    out.flush()
    write(path.with_suffix('.json'), meta)
    return out


def periodic_state(local, c, k, omega, ages, derivative=False):
    factor = 1j*k*omega if derivative else np.ones(len(k))
    y = np.einsum('akp,k->ap', local, factor).real
    h = (np.exp(-1j*ages[:, None]*k*omega)@(c*factor[:, None])).real
    return np.r_[y.ravel(), h.ravel()]


def torus_state(local, c, k, ell, omega, nu, theta, psi, ages):
    fast = np.exp(1j*k*theta)
    slow = np.exp(1j*ell*psi)
    y = np.einsum('aklp,k,l->ap', local, fast, slow, optimize=True).real
    # This Taylor expansion evaluates the tiny slow-phase shift over the
    # physical history window. Its explicit absolute error bound controls
    # numerical evaluation; it drops no stored temporal or spatial mode.
    x = float(max(abs(nu*ell))*max(ages))
    order = 0
    while math.exp(x)*x**(order+1)/math.factorial(order+1) > 1e-18:
        order += 1
    kernel = np.exp(1j*k[None, :]*(theta-omega*ages[:, None]))
    history = np.zeros((len(ages), c.shape[-1]))
    for n in range(order+1):
        collapsed = np.einsum('klp,l->kp', c, slow*(-1j*nu*ell)**n, optimize=True)
        history += (kernel@collapsed).real*(ages[:, None]**n/math.factorial(n))
    bound = math.exp(x)*x**(order+1)/math.factorial(order+1)*np.max(np.sum(abs(c), axis=(0, 1)))
    return np.r_[y.ravel(), history.ravel()], order, float(bound)


def diagnostics(difference, phase, modes, scales):
    d = difference-phase*np.dot(phase, difference)
    v = modes-phase[:, None]*(phase@modes)[None, :]
    d = d*scales
    v = v*scales[:, None]
    norms = np.linalg.norm(v, axis=0)
    v /= norms
    coef = np.linalg.lstsq(v, d, rcond=None)[0]
    dn = np.linalg.norm(d)
    if dn == 0:
        return dict(norm=0., coefficients=coef.tolist(), relative_two_mode_residual=None,
            angles_to_unstable_stable_degrees=[None,None],
            normalized_basis_condition=float(np.linalg.cond(v)),
            direction_status='UNDEFINED_ZERO_DISPLACEMENT')
    angles = []
    for mode in v.T:
        dot = np.dot(mode, d)
        square = np.dot(mode, mode)
        orthogonal = np.linalg.norm(d-mode*(dot/square))
        angles.append(float(np.degrees(np.arctan2(orthogonal, abs(dot)/np.sqrt(square)))))
    return dict(norm=float(dn), coefficients=coef.tolist(),
        relative_two_mode_residual=float(np.linalg.norm(d-v@coef)/dn),
        angles_to_unstable_stable_degrees=angles,
        normalized_basis_condition=float(np.linalg.cond(v)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--exact-j',action='store_true',
        help='Require the completed paired spectrum of the identical-J target; preserve the nearby-J diagnostic')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    destination=OUT/'exactJ_diagnostic' if args.exact_j else OUT
    destination.mkdir(exist_ok=True)
    s = RateField()
    source = read(PER/'TR2_same_parameter_saddle_approach.json')
    row = source['rows'][-1]
    torus, target = np.load(row['torus']), np.load(row['target'])
    basis_path = PER/'poincare_floquet/TR2_endpoint_middle_N128_dt0.05.npz'
    coarse_path = PER/'poincare_floquet/TR2_endpoint_middle_N128_dt0.1.npz'
    exact_evidence=None
    if args.exact_j:
        from complete_rate_positive_stability import paired_modes, values
        exact_evidence=OUT/'exactJ_spectrum/result.json'
        completed=read(exact_evidence)
        assert Path(completed['orbit']).resolve()==Path(row['target']).resolve()
        pair=[read(path) for path in completed['sources']]
        assert all(Path(q['orbit']).resolve()==Path(row['target']).resolve() for q in pair)
        verdict=paired_modes(*pair)
        mu_all=values(pair[-1])
        unstable=np.flatnonzero((abs(mu_all.imag)<1e-12)&(mu_all.real>1)&(mu_all.real<1.05))
        stable=np.flatnonzero((abs(mu_all.imag)<1e-12)&(mu_all.real<1)&(mu_all.real>.95))
        assert len(unstable)==len(stable)==1
        selected=[int(unstable[0]),int(stable[0])]
        assert all(verdict['reliable_mode_mask'][i] for i in selected)
        assert verdict['outside_unit_disk_mask'][selected[0]]
        assert abs(mu_all[selected[1]])<1-verdict['per_mode_margin'][selected[1]]
        assert verdict['section_projection_checked']
        coarse_path,basis_path=[Path(path).with_suffix('.npz') for path in completed['sources']]
    else:selected=[0,1]
    basis = np.load(basis_path)
    spectrum = read(basis_path.with_suffix('.json'))
    original = np.load(ROOT/spectrum['orbit'])
    if args.exact_j:
        assert float(original['J'])==float(target['J'])==float(torus['J'])
        assert np.array_equal(original['r'],target['r'])
    mu = basis['multipliers'][selected]
    assert max(abs(mu.imag)) < 1e-12 and mu[0].real > 1 and 0 < mu[1].real < 1
    modes = np.r_[basis['local_vectors'][:, selected], basis['history_vectors'][:, selected]]
    assert np.linalg.norm(modes.imag) < 1e-10
    modes = modes.real
    D = basis['history_vectors'].shape[0]//s.P
    ages = np.arange(1, D+1)*float(basis['dt'])
    cc, kt, kp = coefficients(torus['r'])
    sc, ks, _ = coefficients(target['r'][:, None])
    oc, ko, _ = coefficients(original['r'][:, None])
    omega, nu = 2*np.pi/float(torus['T']), float(torus['nu'])
    st = state_coefficients(s, sc[:, 0], 2j*np.pi*ks/float(target['T']), float(target['J']), 'matched_target')
    mode_name='mode_source_exactJ' if args.exact_j else 'mode_source'
    so = state_coefficients(s, oc[:, 0], 2j*np.pi*ko/float(original['T']), float(original['J']), mode_name)
    lc = state_coefficients(s, cc.reshape(-1, s.P), 1j*(kt[:, None]*omega+kp[None, :]*nu).ravel(),
                            float(torus['J']), 'torus_a0038').reshape(9, len(kt), len(kp), s.P)
    baseline = periodic_state(st, sc[:, 0], ks, 2*np.pi/float(target['T']), ages)
    reference = periodic_state(so, oc[:, 0], ko, 2*np.pi/float(original['T']), ages)
    phase = periodic_state(so, oc[:, 0], ko, 2*np.pi/float(original['T']), ages, derivative=True)
    phase /= np.linalg.norm(phase)
    assert max(abs(phase@modes)) < 1e-8
    # Weight all populations by their actual neuron counts and average the
    # history rather than letting its denser sampling dominate this norm.
    w = np.sqrt(s.geo['group_size']/s.geo['group_size'].sum())
    local_scales = np.array([1000, 1000, 1, 1, 1, 1, .1, .1, 1])[:, None]*w
    history_scales = np.broadcast_to(1000*w/np.sqrt(D), (D, s.P))
    weighted = np.r_[local_scales.ravel(), history_scales.ravel()]
    metrics = dict(raw=np.ones(len(phase)), weighted=weighted)
    center = int(np.argmin(row['distances_Hz']))
    indices = sorted(set(range(0, 256, 8)) | set(range(center-32, center+33, 2)))
    rows = []
    for index in indices:
        shift = row['fast_phase_shifts_cycles'][index]
        value, order, bound = torus_state(lc, cc, kt, kp, omega, nu, 2*np.pi*shift, 2*np.pi*index/256, ages)
        rate = s.output(value[:9*s.P].reshape(9, s.P))
        exact_rate = np.einsum('klp,k,l->p', cc, np.exp(2j*np.pi*kt*shift),
                              np.exp(2j*np.pi*kp*index/256), optimize=True).real
        assert max(abs(rate-exact_rate)) < 1e-13
        rows.append(dict(slow_index=index, slow_phase_cycles=index/256,
            time_s=index/256*row['slow_period_s'], rate_phase_averaged_distance_Hz=row['distances_Hz'][index],
            history_slow_shift_Taylor_order=order, history_absolute_evaluation_bound_per_ms=bound,
            matched_target={key: diagnostics(value-baseline, phase, modes, scale) for key, scale in metrics.items()},
            original_target={key: diagnostics(value-reference, phase, modes, scale) for key, scale in metrics.items()}))
        print('DIRECTION', index, rows[-1]['matched_target']['weighted'], flush=True)
    result = dict(status=('FINITE_EXACT_PARAMETER_DIRECTION_DIAGNOSTIC' if args.exact_j else
                          'FINITE_NEARBY_PARAMETER_DIRECTION_DIAGNOSTIC'), timestamp=time.time(),
        torus=row['torus'], matched_target=row['target'], eigenvector_source=str(basis_path),
        coarse_eigenvector_source=str(coarse_path),selected_mode_indices=selected,
        exact_parameter_spectrum_evidence=str(exact_evidence) if exact_evidence else None,
        torus_local_coefficients=str(OUT/'torus_a0038_local_coefficients.npy'),
        mode_source_local_coefficients=str(OUT/(mode_name+'_local_coefficients.npy')),
        J_torus=float(torus['J']), J_target=float(target['J']), J_eigenvectors=float(original['J']),
        parameter_offset=float(torus['J']-original['J']), slow_period_s=row['slow_period_s'],
        spatial_groups=s.P, local_states=9*s.P, history_samples=D, history_end_ms=float(ages[-1]),
        multipliers=[[float(v.real), float(v.imag)] for v in mu],
        mode_phase_overlap=abs(phase@modes).tolist(), closest_fast_averaged_rate_index=center,
        target_offset={key: diagnostics(baseline-reference, phase, modes, scale) for key, scale in metrics.items()},
        rows=rows,
        norm='Raw Poincare-section projection first; additionally neuron-weighted local scales '
             '[1000,1000,1,1,1,1,.1,.1,1] and history rate in Hz averaged across the stored delay samples.',
        scope=('Full 935-group state and history projection on two exact-J right Floquet vectors. '
               if args.exact_j else 'Full 935-group state and history projection on two nearby-J right Floquet vectors. ')+
              'Least-squares coefficients are not adjoint modal coordinates. Common fast-phase alignment '
              'followed by linear section projection is not a solved nonlinear Poincare return. '
              +('This does not establish invariant-manifold connection or torus stability.' if args.exact_j else
               'This does not establish invariant-manifold connection, exact-J eigendirections or torus stability.'))
    write(destination/'result.json', result)
    print('FINISHED', destination/'result.json', flush=True)


if __name__ == '__main__':
    main()
