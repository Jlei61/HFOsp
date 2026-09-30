"""Validate the exact PD1 parent witnesses before adding display markers."""
from complete_rate_positive_stability import DEST, PERIODIC_OUT, read, np, paired_modes
from pathlib import Path


def verified_parent_witnesses():
    source = DEST / 'PD1_parent_witnesses/result.json'
    if not source.exists():
        return []
    result = read(source)
    if result['status'] != 'PARENT_SIDES_CHECKED':
        return []
    root = read(PERIODIC_OUT / 'PD_double_low_validation.json')
    assert root['full_acceptance']
    assert abs(root['J_EE_core'] - result['critical_J']) < 1e-12
    rows = []
    for witness in result['rows']:
        orbit = Path(witness['analyzed_orbit'])
        physical = witness['resolution']
        assert Path(physical['orbit']).resolve() == orbit.resolve()
        assert physical['status'] == 'RESOLUTION_CHECKED'
        assert physical['filter_state_check']['positive']
        assert physical['maximum_group_defect_Hz'] < 1e-6
        assert witness['relative_waveform_refinement_change'] < .001
        assert witness['relative_period_change'] < 1e-6
        classifications = []
        for attempt in witness['attempts']:
            pair = [read(p) for p in attempt['sources']]
            assert len(pair) == 2
            assert all(Path(p['orbit']).resolve() == orbit.resolve() for p in pair)
            assert all(abs(p['J_EE_core'] - witness['J_EE_core']) < 1e-12 for p in pair)
            classifications.append(paired_modes(*pair))
        accepted = [q for q in classifications if q['status'] != 'UNRESOLVED']
        assert accepted and {q['status'] for q in accepted} == {witness['status']}
        assert all(q['section_projection_checked'] for q in accepted)
        expected_dimension = 1 if witness['side'] == 'below' else 0
        assert accepted[-1]['numerical_unstable_dimension'] == expected_dimension
        delta = witness['J_EE_core'] - result['critical_J']
        assert (delta < 0) == (witness['side'] == 'below')
        with np.load(orbit) as data:
            assert data['r'].shape == (physical['N'], 935)
            assert abs(float(data['J']) - witness['J_EE_core']) < 1e-12
            assert abs(float(data['T']) - witness['T_ms']) < 1e-8
        meta = read(orbit.with_suffix('.json'))
        assert abs(meta['J_EE_core'] - witness['J_EE_core']) < 1e-12
        assert abs(meta['T_ms'] - witness['T_ms']) < 1e-8
        rows.append(dict(family='double', side=witness['side'], status=witness['status'],
            orbit=str(orbit), source=str(source), J_EE_core=witness['J_EE_core'],
            mean_rates_hz=meta['mean_rates_hz'], classification=accepted[-1]))
    assert {q['side'] for q in rows} == {'below', 'above'}
    return rows
