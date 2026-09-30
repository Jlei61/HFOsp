"""Check that physical PD1 children depart along the accepted flip mode.

The half-period odd projection removes the repeated parent and all even
corrections. This is a rate-profile identity check, not a new full-history
Floquet computation or a nonlinear criticality verdict.
"""
from complete_rate_positive_stability import DEST, RateField, read, write, np
from scipy.signal import resample


def main():
    source = DEST / 'physical_children/PD_double_low/result.json'
    branch = read(source)
    parent = branch['parent']
    assert parent['status'] == 'FILTER_AND_CRITICAL_MODE_RECHECKED'
    assert parent['continuous_check']['filter_state_check']['positive']
    assert parent['continuous_check']['maximum_group_defect_Hz'] < 1e-6
    assert parent['antiperiodic_relative_residual'] < 1e-8
    s = RateField()
    mass = np.asarray(s.geo['group_size'], float)
    weights = mass / mass.sum()
    assert s.P == 935 and len(np.unique(s.geo['group_cell'])) == 400

    def inner(x, y):
        return float(np.mean(np.sum(x * y * weights, axis=1)))

    def norm(x):
        return np.sqrt(inner(x, x))

    with np.load(parent['orbit']) as saved:
        parent_rate = saved['r'] * 1000
    with np.load(parent['mode']) as saved:
        mode = saved['u'].copy()
    assert parent_rate.shape == mode.shape == (parent['N'], s.P)
    doubled_mode = np.concatenate([mode, -mode])
    doubled_mode /= np.max(abs(doubled_mode))
    doubled_parent = np.concatenate([parent_rate, parent_rate])
    rows = []
    for child in sorted(branch['rows'], key=lambda x: x['amplitude_hz']):
        check = child['physical_check']
        assert child['physical_pass'] and check['filter_state_check']['positive']
        assert check['maximum_group_defect_Hz'] < 1e-6
        assert check['minimum_rate_Hz'] >= -1e-9
        with np.load(child['orbit']) as saved:
            r = saved['r'] * 1000
            J, T = float(saved['J']), float(saved['T'])
        assert r.shape[1] == 935 and len(r) % 2 == 0
        assert abs(J - child['J_EE_core']) < 1e-12
        assert abs(T - child['T_ms']) < 1e-8
        half = len(r) // 2
        odd = (r - np.roll(r, half, axis=0)) / 2
        even = (r + np.roll(r, half, axis=0)) / 2
        reference = resample(doubled_parent, len(r), axis=0)
        seed = resample(doubled_mode, len(r), axis=0)
        assert norm(odd + np.roll(odd, half, axis=0)) < 1e-10
        assert norm(even - np.roll(even, half, axis=0)) < 1e-10
        assert norm(r - even - odd) < 1e-10
        amplitude = child['amplitude_hz']
        coefficient = inner(odd, seed) / inner(seed, seed)
        # No fitted phase or sign is used: the same branch-switch phase
        # condition and signed amplitude define the comparison.
        cosine = inner(odd, seed) / (norm(odd) * norm(seed))
        error = norm(odd / amplitude - seed) / norm(seed)
        first = 2 * np.fft.rfft(r, axis=0)[1] / len(r)
        first_rms = float(np.sqrt(np.sum(weights * abs(first)**2) / 2))
        assert first_rms > 1000 * check['maximum_group_defect_Hz']
        rows.append(dict(amplitude_Hz=amplitude, J_EE_core=J, T_ms=T,
            orbit=child['orbit'], N=len(r),
            J_shift=J - parent['J_EE_core'],
            J_shift_per_amplitude_squared=(J-parent['J_EE_core'])/amplitude**2,
            period_shift_from_twice_parent_ms=T-2*parent['T_ms'],
            weighted_mode_cosine=cosine,
            odd_over_amplitude_relative_error=error,
            projected_amplitude_Hz=coefficient,
            projected_over_prescribed_amplitude=coefficient/amplitude,
            odd_weighted_RMS_Hz=norm(odd),
            even_correction_weighted_RMS_Hz=norm(even-reference),
            first_harmonic_weighted_RMS_Hz=first_rms,
            physical_check=check))

    amplitudes = np.array([q['amplitude_Hz'] for q in rows])
    errors = np.array([q['odd_over_amplitude_relative_error'] for q in rows])
    even_errors = np.array([q['even_correction_weighted_RMS_Hz'] for q in rows])
    odd_orders = np.diff(np.log(errors)) / np.diff(np.log(amplitudes))
    even_orders = np.diff(np.log(even_errors)) / np.diff(np.log(amplitudes))
    coeff = np.array([q['J_shift_per_amplitude_squared'] for q in rows])
    passed = bool(np.all(coeff > 0) and np.ptp(coeff)/np.mean(coeff) < .1
        and np.all((odd_orders > 1.8) & (odd_orders < 2.2))
        and all(q['weighted_mode_cosine'] > .999 for q in rows)
        and errors[0] < .001)
    output = dict(status='CHILD_DEPARTURE_IDENTITY_CHECKED' if passed else
        'CHILD_DEPARTURE_IDENTITY_UNRESOLVED', source=str(source),
        parent_orbit=parent['orbit'], parent_mode=parent['mode'],
        parent_J_EE_core=parent['J_EE_core'], parent_T_ms=parent['T_ms'],
        rows=rows, odd_over_amplitude_error_orders=odd_orders,
        even_correction_orders=even_orders,
        coefficient_relative_spread=float(np.ptp(coeff)/abs(np.mean(coeff))),
        state_contract=dict(spatial_cells=400, populations=935,
            changed_equations=False, changed_delays=False),
        comparison='All 935 population rate profiles weighted by original neuron counts; same phase condition; no fitted phase or sign. Odd projection is (r(theta)-r(theta+1/2))/2.',
        criterion_scope='Numerical shape and local scaling diagnostics, not a rigorous enclosure.',
        scope='Physical 2T children approach the accepted antiperiodic critical rate mode. This does not by itself establish child radial Floquet-mode identity, parent-side stability, nonlinear PD criticality, stable attractors, or global connections.')
    write(DEST/'physical_children/PD_double_low/departure_identity.json', output)
    print(output['status'], flush=True)
    for row in rows:
        print({k: row[k] for k in ['amplitude_Hz', 'weighted_mode_cosine',
            'odd_over_amplitude_relative_error', 'even_correction_weighted_RMS_Hz',
            'first_harmonic_weighted_RMS_Hz']}, flush=True)
    print('Error orders', odd_orders, 'even correction orders', even_orders, flush=True)


if __name__ == '__main__':
    main()
