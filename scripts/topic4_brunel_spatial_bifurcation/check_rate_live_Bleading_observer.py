"""Check newly saved B-leading cycles without publishing the live branch.

CPU reconstruction retains all 935 populations, nine local states and
physical delays. Frozen prefix evidence and contact readouts are saved
separately; the continuation and its accepted-prefix file are never edited.
"""
from audit_current_rate_model import full_rhs_check, RateField, PER, read, write
from audit_rate_filter_states import filter_state_minima
from audit_rate_survey_filter_states import fingerprint
from plot_rate_sameJ_burst_pair import observe_cycle, OLD
from pathlib import Path
import argparse
import os
import time
import numpy as np


LABEL = 'arcBleadingConnection_20260920'
DEST = Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920/Bleading_extension/live_CPU_observer')


def identity(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes() or None
    except (FileNotFoundError, ProcessLookupError):
        return None


def atomic_write(path, result):
    temp = path.with_suffix('.tmp.json')
    write(temp, result)
    temp.replace(path)


def physical_pass(row):
    return (row['between_nodes_integrated_error_Hz'] < .001 and
            row['full_RHS_rate_error_Hz_per_ms'] < .001 and
            row['full_RHS_linear_state_max_abs'] < 1e-8 and
            row['minimum_interpolated_group_rate_Hz'] >= -1e-9 and
            row['refractory_rate_bound_excess_Hz'] < 1e-8 and
            row['filter_state_check']['positive'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--after-pid', type=int, required=True)
    parser.add_argument('--control-source', type=Path, required=True)
    parser.add_argument('--continuation-pid', type=int, required=True)
    parser.add_argument('--last-index', type=int, required=True)
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    continuation_identity = identity(args.continuation_pid)
    dependency_identity = identity(args.after_pid)
    rows, observations, brackets = [], [], []

    def record(status, **kwargs):
        atomic_write(DEST/'worker.json', dict(status=status, pid=os.getpid(),
            control_source=str(args.control_source), continuation_pid=args.continuation_pid,
            requested_last_index=args.last_index, completed_new_points=len(observations),
            accepted_prefix_modified=False, **kwargs))

    while dependency_identity and identity(args.after_pid) == dependency_identity:
        record('WAITING_CONTROL_CHECK')
        time.sleep(20)
    control = read(args.control_source)
    assert control['status'] == 'SELECTED_POINTS_PASS'
    assert all(physical_pass(row) for row in control['rows'])
    first = max(control['rows'], key=lambda row: row['index'])
    rows.append(first)
    start = first['index'] + 1
    assert start <= args.last_index
    model = RateField()
    assert model.P == 935
    contract_source = OLD/'observer_firing.json'
    contract = read(contract_source)

    def readout(row):
        z = np.load(row['path'])
        observed = observe_cycle(z['r'], float(z['T']), model, contract)
        keys = ['bin_origin_ms', 'detected_events', 'qualified_events',
                'SCL_qualified_events', 'sustained_contact_names',
                'sustained_SCL_contact_names', 'contact_peak_to_threshold', 'metrics']
        return dict(index=row['index'], orbit=row['path'], J_EE_core=row['J'],
            T_ms=row['T_ms'], records=[{k:r[k] for k in keys} for r in observed['records']],
            interior_cycles=observed['interior_cycles'])

    control_observation = readout(first)
    previous = control_observation
    for index in range(start, args.last_index + 1):
        path = PER/'orbits'/f'{LABEL}_{index:04d}_N4096.npz'
        metadata_path = path.with_suffix('.json')
        while not metadata_path.exists():
            if not continuation_identity or identity(args.continuation_pid) != continuation_identity:
                record('CHECKED_ALL_CURRENTLY_SAVED_POINTS', last_checked_index=index-1,
                       requested_range_complete=False)
                return
            record('WAITING_SAVED_ORBIT', index=index)
            time.sleep(20)
        metadata = read(metadata_path)
        assert metadata['status'] == 'CONVERGED'
        assert Path(metadata['path']).resolve() == path.resolve()
        before = fingerprint(path)
        record('CPU_FULL_RHS_CHECK', index=index, orbit=str(path))
        destination = DEST/f'point_{index:04d}.json'
        cached = read(destination) if destination.exists() else None
        if cached and cached.get('profile_fingerprint') == before:
            row, observed = cached['check'], cached.get('readout')
        else:
            z = np.load(path)
            assert z['r'].shape == (4096, 935)
            assert abs(float(z['J'])-metadata['J_EE_core']) < 1e-12
            assert abs(float(z['T'])-metadata['T_ms']) < 1e-8
            row = full_rhs_check(model, path, factor=4)
            row.update(index=index, filter_state_check=filter_state_minima(model, z['r'], float(z['T'])))
            row['status'] = 'POINT_CHECK_PASS' if physical_pass(row) else 'POINT_CHECK_UNRESOLVED'
            assert fingerprint(path) == before
            observed = readout(row) if physical_pass(row) else None
            atomic_write(destination, dict(profile_fingerprint=before, check=row, readout=observed,
                observer_source=str(contract_source)))
        if not physical_pass(row):
            record('PHYSICAL_PROFILE_UNRESOLVED', index=index, evidence=str(destination))
            return
        assert observed is not None
        rows.append(row)
        observations.append(observed)
        prefix = DEST/f'through_{index:04d}_CPU_checks.json'
        atomic_write(prefix, dict(status='SELECTED_POINTS_PASS', rows=rows,
            selected_orbits=[r['path'] for r in rows], control_source=str(args.control_source),
            scope='Frozen selected points, with the previous accepted control. Does not publish branch acceptance or establish stability/global connections.'))

        def robust_contacts(q):
            sets = [tuple(r['sustained_SCL_contact_names']) for r in q['records']]
            return sets[0] if len(set(sets)) == 1 else None

        left, right = robust_contacts(previous), robust_contacts(observed)
        if left is not None and right is not None and left != right:
            brackets.append(dict(left_index=previous['index'], right_index=index,
                J_EE_core=[previous['J_EE_core'], observed['J_EE_core']],
                left_SCL_contacts=left, right_SCL_contacts=right,
                kind='Frozen-observer recruitment change, not a dynamical bifurcation'))
        atomic_write(DEST/'observer_summary.json', dict(status='FROZEN_CHECKED_PREFIX',
            control=control_observation, rows=observations, recruitment_change_brackets=brackets,
            contact_names=contract['contact_names'], observer_source=str(contract_source),
            physical_evidence=str(prefix),
            statistical_unit='One periodic solution per point. Repeated cycles and bin origins are not independent trials.',
            scope='Includes branches of unclassified stability. A threshold crossing is not evidence of a dynamical bifurcation; no exhaustive parameter window is established.'))
        previous = observed
        print('CHECKED OBSERVER', index, row['J'], right, flush=True)
        record('POINT_CHECKED', index=index, physical_evidence=str(prefix))
    record('REQUESTED_SAVED_POINTS_CHECKED', last_checked_index=args.last_index,
           requested_range_complete=True)


if __name__ == '__main__':
    main()
