"""Merge checked PD4 spectra without overwriting live worker snapshots.

The direct worker retains its in-memory earlier attempts while quotient
checks finish separately. Recompute every paired verdict on the same
physical orbit before adopting a resolved numerical unstable dimension.
"""
from complete_rate_positive_stability import DEST, PERIODIC_OUT, read, write, paired_modes
from pathlib import Path
import time


def main():
    folder = DEST/'H2_local_PD'
    profiles = read(folder/'physical_children.json')['rows']
    rows = []
    for index, profile in enumerate(profiles):
        physical = profile['physical_check']
        assert physical['filter_state_check']['positive']
        assert physical['maximum_group_defect_Hz'] < 1e-6
        orbit = Path(profile['orbit']).resolve()
        assert Path(physical['orbit']).resolve() == orbit
        attempts = []
        raw = folder/f'child_full_spectrum_{index}.json'
        if raw.exists():
            attempts.extend(dict(**attempt, provider=str(raw))
                            for attempt in read(raw)['attempts'])
        source = DEST/f'ritz_checks/PD4_child{index}_complement.json'
        if source.exists():
            attempts.append(dict(sources=read(source)['sources'], provider=str(source)))
        checked = []
        for attempt in attempts:
            pair = [read(p) for p in attempt['sources']]
            assert len(pair) == 2
            assert all(Path(q['orbit']).resolve() == orbit for q in pair)
            assert all(abs(q['J_EE_core']-profile['J_EE_core']) < 1e-12 for q in pair)
            for q in pair:
                if 'locked_subspace_invariance_defect' in q:
                    assert q['locked_subspace_invariance_defect'] < 1e-6
                    assert max(q['complement_filter_relative_changes']) < 1e-5
                    assert max(q['lift_condition_numbers']) < 1e10
            checked.append(dict(sources=attempt['sources'], provider=attempt['provider'],
                                classification=paired_modes(*pair)))
        accepted = [q['classification'] for q in checked
                    if q['classification']['status'] != 'UNRESOLVED']
        verdicts = {q['status'] for q in accepted}
        dimensions = {q['numerical_unstable_dimension'] for q in accepted
                      if q['numerical_unstable_dimension'] is not None}
        lower_bound = max([q['reliable_outside_count'] for q in accepted], default=0)
        conflict = len(verdicts)>1 or len(dimensions)>1
        dimension = next(iter(dimensions)) if len(dimensions)==1 else None
        if dimension is not None and lower_bound>dimension:
            conflict = True
        status = ('CONFLICT_REVIEW' if conflict else next(iter(verdicts)) if verdicts else
                  'UNRESOLVED' if checked else 'PHYSICAL_PROFILE_ONLY')
        if conflict:
            dimension = None
        rows.append(dict(child_index=index, amplitude_Hz=profile['amplitude_hz'],
            J_EE_core=profile['J_EE_core'], orbit=str(orbit), physical_check=physical,
            status=status, numerical_unstable_dimension=dimension,
            verified_unstable_dimension_lower_bound=lower_bound,
            attempts=checked, interval_completeness=False))
    output = folder/'current_child_spectrum_evidence.json'
    write(output, dict(status='CURRENT_PAIRED_SPECTRA_MERGED', timestamp=time.time(), rows=rows,
        direct_worker_snapshots_modified=False, interval_completeness=False,
        scope='Separate direct and invariant-complement calculations on each identical physical child are combined after recomputing their paired-step classifications. Resolved dimensions apply only to their sampled cycles. Equal endpoint dimensions do not exclude intervening crossings.'))
    first = rows[0]
    if first['status']=='UNSTABLE' and first['numerical_unstable_dimension']==4:
        canonical = PERIODIC_OUT/'PD_H2_after_LPC13_child_validation.json'
        q = read(canonical)
        assert q['status']=='LOCALLY_CHECKED_PD_CHILD' and q['criticality']=='SUBCRITICAL_PD'
        assert Path(q['child_orbit']).resolve() == Path(first['orbit']).resolve()
        backup = folder/'PD4_child_validation_before_full_spectrum.json'
        if not backup.exists():
            write(backup, q)
        q['child_full_spectrum'] = dict(source=str(output), child_index=0,
            numerical_unstable_dimension=4, paired_time_steps_checked=True,
            phase_removed=True, filter_coverage_checked=True,
            scope='Numerical full-delay Poincare count at this physical child only; no additional bifurcation or interval completeness established.')
        write(canonical, q)
    print('PD4 CURRENT SPECTRA', [(q['amplitude_Hz'], q['status'],
        q['numerical_unstable_dimension'], q['verified_unstable_dimension_lower_bound'])
        for q in rows], flush=True)


if __name__ == '__main__':
    main()
