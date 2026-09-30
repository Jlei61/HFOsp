"""Combine PD1 physical departure, parent orientation and radial-mode identity.

No numerical classification is published until every independent component
has passed on the same accepted parent and the same physical child.
"""
from complete_rate_positive_stability import DEST, PERIODIC_OUT, np, read, write, values, paired_modes
from check_rate_PD1_parent_display import verified_parent_witnesses
from pathlib import Path
import argparse
import time


def assess():
    folder = DEST / 'physical_children/PD_double_low'
    source = folder / 'result.json'
    branch = read(source)
    validation = PERIODIC_OUT / 'PD_double_low_validation.json'
    parent = read(validation)
    followup = parent['filter_state_followup']
    assert parent['full_acceptance'] and parent['status'] == 'VALIDATED_PD'
    assert followup['status'] == 'FILTER_AND_CRITICAL_MODE_RECHECKED'
    assert Path(parent['accepted_parent_orbit']).resolve() == Path(branch['parent']['orbit']).resolve()
    assert Path(followup['mode']).resolve() == Path(branch['parent']['mode']).resolve()
    assert abs(parent['J_EE_core'] - branch['parent']['J_EE_core']) < 1e-12
    assert followup['continuous_check']['filter_state_check']['positive']
    assert followup['continuous_check']['maximum_group_defect_Hz'] < 1e-6
    assert followup['antiperiodic_relative_residual'] < 1e-7
    checks = sorted(followup['monodromy_checks'], key=lambda q: -q['dt_ms'])
    errors = np.array([q['minus_one_relative_defect'] for q in checks])
    assert len(errors) >= 3 and errors[-1] < 1e-4
    assert np.all(errors[:-1] / errors[1:] > 3)
    assert abs(parent['crossing_border_slope']) > 1e-6
    witnesses = verified_parent_witnesses()
    below = next(q for q in witnesses if q['side'] == 'below')
    above = next(q for q in witnesses if q['side'] == 'above')
    assert below['status'] == 'UNSTABLE' and above['status'] == 'NUMERICALLY_STABLE'
    for q in witnesses:
        spectrum = values(q['classification'])
        k = int(np.argmin(abs(spectrum + 1)))
        assert abs(spectrum[k].imag) < 1e-10
        assert q['classification']['reliable_mode_mask'][k]
        margin = q['classification']['per_mode_margin'][k]
        if q['side'] == 'below':
            assert spectrum[k].real < -1 - margin
        else:
            assert -1 + margin < spectrum[k].real < 0
    departure_source = folder / 'departure_identity.json'
    departure = read(departure_source)
    assert departure['status'] == 'CHILD_DEPARTURE_IDENTITY_CHECKED'
    assert Path(departure['parent_orbit']).resolve() == Path(parent['accepted_parent_orbit']).resolve()
    assert Path(departure['parent_mode']).resolve() == Path(followup['mode']).resolve()
    assert branch['full_physical_child_checks']
    children = sorted(branch['rows'], key=lambda q: q['amplitude_hz'])
    assert len(children) == 3
    coefficients = []
    for child in children:
        physical = child['physical_check']
        shape = next(q for q in departure['rows'] if Path(q['orbit']).resolve() == Path(child['orbit']).resolve())
        assert child['physical_pass'] and physical['filter_state_check']['positive']
        assert physical['maximum_group_defect_Hz'] < 1e-6
        assert Path(physical['orbit']).resolve() == Path(child['orbit']).resolve()
        assert shape['weighted_mode_cosine'] > .999
        assert parent['J_EE_core'] < child['J_EE_core'] < above['J_EE_core']
        assert abs(child['T_ms'] / (2 * followup['T_ms']) - 1) < .001
        coefficients.append((child['J_EE_core'] - parent['J_EE_core']) / child['amplitude_hz']**2)
    spread = float(np.ptp(coefficients) / np.mean(coefficients))
    assert min(coefficients) > 0 and spread < .05
    assert np.all(np.abs(np.asarray(departure['odd_over_amplitude_error_orders']) - 2) < .1)
    assert np.all(np.abs(np.asarray(departure['even_correction_orders']) - 2) < .1)
    selected = children[-1]
    spectra_sources = [PERIODIC_OUT / f'poincare_floquet/PD1_physical_20260920_child_k6_dt{dt}.json'
        for dt in ['0.05', '0.025']]
    spectra = [read(p) for p in spectra_sources]
    assert all(Path(q['orbit']).resolve() == Path(selected['orbit']).resolve() for q in spectra)
    verdict = paired_modes(*spectra)
    assert verdict['status'] == 'UNSTABLE' and verdict['section_projection_checked']
    growing = np.flatnonzero(verdict['outside_unit_disk_mask'])
    assert len(growing) == 1
    index = int(growing[0]); multiplier = values(spectra[-1])[index]
    assert abs(multiplier.imag) < 1e-10
    assert multiplier.real > 1 + verdict['per_mode_margin'][index]
    radial_source = folder / 'radial_history_identity/result.json'
    result = dict(status='RADIAL_IDENTITY_PENDING', timestamp=time.time(), label='PD1',
        parent_validation=str(validation), parent_orbit=parent['accepted_parent_orbit'],
        parent_mode=followup['mode'], parent_J_EE_core=parent['J_EE_core'],
        parent_T_ms=followup['T_ms'], parent_witnesses=witnesses,
        physical_children_source=str(source), full_physical_child_checks=True, rows=children,
        child_orbit=selected['orbit'], child_mu=[float(multiplier.real), 0.],
        child_side='HIGHER_J_PARENT_STABLE_SIDE', child_stability='UNSTABLE',
        departure_identity_source=str(departure_source),
        departure_coefficient_relative_spread=spread, departure_coefficients=coefficients,
        spectra_sources=list(map(str, spectra_sources)), child_spectrum_classification=verdict,
        radial_history_identity_source=str(radial_source),
        model=dict(spatial_cells=400, populations=935, local_states=8415,
                   original_equations_and_delays_retained=True),
        global_branch_completeness=False, canonical_criticality_promoted=False)
    if not radial_source.exists():
        return result
    radial = read(radial_source)
    if radial['status'] != 'RADIAL_HISTORY_IDENTITY_PASS':
        result['status'] = 'RADIAL_IDENTITY_UNRESOLVED'
        return result
    assert Path(radial['child_orbit']).resolve() == Path(selected['orbit']).resolve()
    assert [Path(p).resolve() for p in radial['spectra_sources']] == [p.resolve() for p in spectra_sources]
    assert radial['fine_mode_index'] == index
    assert abs(radial['multiplier'] - multiplier.real) < 1e-12
    assert radial['parent_antiperiodic_full_state_history_cosine'] > .95
    assert radial['paired_time_step_mode_cosine'] > .999
    assert all(q['full_state_history_tangent_cosine'] > .95 for q in radial['tangent_comparisons'])
    assert len(radial['tangent_comparisons']) == 2
    result.update(status='SUBCRITICAL_PD', criticality='SUBCRITICAL_PD',
        canonical_criticality_promoted=(parent.get('criticality')=='SUBCRITICAL_PD' and
            Path(parent.get('child_classification_source','')).resolve()==
            (PERIODIC_OUT/'PD_double_low_child_validation.json').resolve()),
        scope='Local subcritical flip: physical 2T children depart toward the numerically stable parent side, '
              'with quadratic parameter scaling and an identified growing radial return direction. '
              'The checked child is unstable; its total unstable dimension remains unresolved. '
              'This does not establish an attracting irregular state, a propagation-template switch, '
              'a global branch connection or an exhaustive interval crossing count.')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--promote', action='store_true')
    args = parser.parse_args()
    result = assess()
    folder = DEST / 'physical_children/PD_double_low'
    if args.promote:
        assert result['status'] == 'SUBCRITICAL_PD', result['status']
        path = PERIODIC_OUT / 'PD_double_low_validation.json'
        parent = read(path)
        backup = folder / 'before_physical_child_classification.json'
        if not backup.exists():
            write(backup, parent)
        destination = PERIODIC_OUT / 'PD_double_low_child_validation.json'
        result['canonical_criticality_promoted'] = True
        write(destination, result)
        parent.update(criticality=result['criticality'], child_stability='UNSTABLE',
            child_validation_status='PHYSICAL_CHILD_AND_RADIAL_MODE_CHECKED',
            child_classification_source=str(destination), child_multiplier=result['child_mu'],
            child_side=result['child_side'], meaning=result['scope'],
            accepted_parent_mesh_N=parent['filter_state_followup']['N'], accepted_mode=result['parent_mode'])
        write(path, parent)
        orientation_path = DEST / 'PD1_parent_witnesses/local_child_orientation.json'
        orientation = read(orientation_path)
        orientation.update(canonical_criticality_promoted=True, remaining_check=None,
            combined_classification_source=str(destination),
            scope=result['scope'])
        write(orientation_path, orientation)
    write(folder / 'combined_classification_check.json', result)
    print('PD1 PHYSICAL CHILD', result['status'], 'promoted', result['canonical_criticality_promoted'], flush=True)


if __name__ == '__main__':
    main()
