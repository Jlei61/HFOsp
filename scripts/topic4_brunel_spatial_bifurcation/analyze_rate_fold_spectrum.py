"""Identify a validated fold mode before classifying the remaining spectrum.

The autonomous phase is already removed in the Poincare operator. Match its
near-+1 eigenvector to the independently reconstructed BVP fold tangent in the
same full local-state plus delay-history coordinates. Only that matched mode
may be excluded when checking the rest of the paired-step numerical spectrum.
"""
from complete_rate_positive_stability import paired_modes
from rate_floquet_poincare import values
from validate_rate_mean_fold import validation_matches_latest_root
from rate_periodic import Path, np, read, write, PERIODIC_OUT, argparse
from audit_rate_filter_states import filter_state_minima
from rate_field import RateField

DEST = Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920')


def analyze(label, steps, refined_steps=()):
    assert validation_matches_latest_root(label), 'Current physical fold validation required'
    validation_path = PERIODIC_OUT / (label + '_validation.json')
    validation = read(validation_path)
    root = validation['mesh_checks'][-1]
    physical=validation.get('continuous_defect',{}).get('filter_state_check')
    if physical is None:
        z=np.load(root['orbit'])
        physical=filter_state_minima(RateField(),z['r'],float(z['T']))
    assert physical['positive'], 'Current constituent filters must be resolved'
    rows, identities = [], []
    for step in steps:
        prefix = PERIODIC_OUT / 'poincare_floquet' / f'{Path(root["orbit"]).stem}_dt{step:g}'
        # dt values contain dots, hence all output paths append extensions.
        source = Path(str(prefix) + '.json')
        if step in refined_steps:
            source = Path(str(prefix) + '_ritz.json')
        q = read(source)
        assert Path(q['orbit']).resolve() == Path(root['orbit']).resolve()
        assert abs(q['J_EE_core'] - root['J_EE_core']) < 1e-12
        assert abs(q['T_ms'] - root['T_ms']) < 1e-8
        vector_source = source.with_suffix('.npz')
        tangent_source = PERIODIC_OUT / f'{label}_monodromy_check_N{root["N"]}_dt{step:g}_analytic.npz'
        with np.load(vector_source) as vectors, np.load(tangent_source) as tangent:
            assert abs(float(vectors['dt']) - float(tangent['dt'])) < 1e-13
            phase = np.r_[tangent['phase_local'], tangent['phase_history'].ravel()]
            phase /= np.linalg.norm(phase)
            shape = np.r_[tangent['local'], tangent['history'].ravel()]
            shape -= phase * (phase @ shape)
            shape /= np.linalg.norm(shape)
            modes = np.r_[vectors['local_vectors'], vectors['history_vectors']]
            assert modes.shape[0] == len(shape)
            overlap = abs(shape @ modes) / np.linalg.norm(modes, axis=0)
        mu = values(q)
        index = int(np.argmin(abs(mu - 1)))
        identities.append(dict(requested_dt_ms=step, source=str(source),
            tangent_source=str(tangent_source), critical_mode_index=index,
            multiplier=mu[index], overlap_with_projected_fold_tangent=float(overlap[index]),
            all_tangent_overlaps=overlap, highest_overlap_mode=int(np.argmax(overlap)),
            identity_checked=bool(overlap[index] > .999 and index == int(np.argmax(overlap)))))
        rows.append(q)
    paired = paired_modes(*rows)
    mu = values(rows[-1])
    margin = np.asarray(paired['per_mode_margin'])
    reliable = np.asarray(paired['reliable_mode_mask'])
    critical = identities[-1]['critical_mode_index']
    mask = np.arange(len(mu)) != critical
    identified = all(q['identity_checked'] for q in identities)
    near_plus_one = bool(abs(mu[critical] - 1) < margin[critical] and reliable[critical])
    inside = reliable[mask] & (abs(mu[mask]) < 1 - margin[mask])
    rest_inside = bool(np.all(inside))
    rest_outside = reliable[mask] & (abs(mu[mask]) > 1 + margin[mask])
    spectrum_checked = bool(paired['filter_coverage'] and paired['section_projection_checked'])
    accepted = identified and near_plus_one and rest_inside and spectrum_checked
    unstable=identified and near_plus_one and bool(rest_outside.any()) and paired['section_projection_checked']
    remaining_classified=bool(spectrum_checked and np.all(inside | rest_outside))
    result = dict(label=label, orbit=root['orbit'], J_EE_core=root['J_EE_core'], T_ms=root['T_ms'],
        status=('FOLD_WITH_REMAINING_SPECTRUM_NUMERICALLY_INSIDE' if accepted else
                'FOLD_WITH_VERIFIED_UNSTABLE_MODES' if unstable else 'SPECTRUM_REVIEW_PENDING'),
        fold_validation=str(validation_path), critical_mode_identity=identities,
        physical_profile_check=physical,
        critical_mode_near_plus_one=near_plus_one, paired_spectrum=paired,
        remaining_multipliers=mu[mask], remaining_maximum_modulus=float(max(abs(mu[mask]))),
        remaining_reliable_outside_count=int(rest_outside.sum()),
        numerical_unstable_dimension_excluding_fold=int(rest_outside.sum()) if remaining_classified else None,
        remaining_spectrum_classified=remaining_classified,
        remaining_spectrum_checked=bool(rest_inside and spectrum_checked),
        scope='Paired-step numerical spectrum at this validated fold. This does not certify whole adjacent intervals or global branch connections.')
    output = DEST / 'primary_folds' / f'{label}_root_spectrum_assessment.json'
    if refined_steps and output.exists():
        previous=output.with_name(output.stem+'_before_ritz.json')
        if not previous.exists():write(previous,read(output))
    write(output, result)
    print(label, result['status'], 'remaining modulus', result['remaining_maximum_modulus'], flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('labels', nargs='+')
    p.add_argument('--dt', type=float, nargs=2, default=[.05, .025])
    p.add_argument('--refined-steps',type=float,nargs='*',default=[],
                   help='Use explicitly checked Rayleigh-Ritz files for these time steps')
    a = p.parse_args()
    for label in a.labels:
        analyze(label, a.dt, a.refined_steps)
