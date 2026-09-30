"""Frozen contact readout and consecutive-half peak changes of PD3.

Exact unstable periodic solutions are repeated only to evaluate the same
observer at all phases. These repeats are not autonomous trajectories.
"""
from plot_rate_sameJ_burst_pair import observe_cycle, OLD
from plot_rate_branch_completion import DATA, PERIODIC_OUT, RateField, read, write, np
from pathlib import Path


def main():
    display = read(DATA/'PD3_physical_child.json')
    snapshot = read(display['source'])
    parent = read(PERIODIC_OUT/'PD_A_return_validation.json')
    assert parent['full_acceptance'] and parent['criticality'] == 'SUPERCRITICAL_PD'
    parent_check = parent['filter_state_followup']['continuous_orbit_check']
    child = next(q for q in snapshot['rows'] if
                 Path(q['orbit']).resolve() == Path(display['orbit']).resolve())
    contract_source = OLD/'observer_firing.json'
    contract = read(contract_source)
    s = RateField()
    records = []
    for name, path, check in [('critical_parent', parent['accepted_parent_orbit'], parent_check),
                               ('displayed_child', display['orbit'], child['physical_check'])]:
        assert Path(check['orbit']).resolve() == Path(path).resolve()
        assert check['filter_state_check']['positive']
        assert check['maximum_group_defect_Hz'] < 1e-6
        with np.load(path) as z:
            r, T, J = z['r'].copy(), float(z['T']), float(z['J'])
        if name == 'displayed_child':
            r = np.roll(r, -display['phase_shift_samples'], axis=0)
        observed = observe_cycle(r, T, s, contract)
        retained = [{k: row[k] for k in ['bin_origin_ms', 'detected_events',
            'qualified_events', 'SCL_qualified_events', 'sustained_contact_names',
            'sustained_SCL_contact_names', 'contact_peak_to_threshold', 'exclusions',
            'metrics']} for row in observed['records']]
        record = dict(name=name, orbit=path, J_EE_core=J, T_ms=T,
            physical_check=check, observer_records=retained,
            interior_cycles=observed['interior_cycles'],
            required_unique_contacts=int(np.ceil(len(contract['contact_names'])*
                contract['channel_fraction'])),
            sustained_contact_counts=[len(row['sustained_contact_names']) for row in retained],
            group_absence_reason=('INSUFFICIENT_RECRUITED_CONTACTS' if all(
                len(row['sustained_contact_names']) < np.ceil(len(contract['contact_names'])*
                    contract['channel_fraction']) and row['detected_events']==0
                for row in retained) else 'SEE_EVENT_AND_QUALIFICATION_RECORDS'))
        if name == 'displayed_child':
            regional = np.array([s.regional_rates(row) for row in r])
            half = len(r)//2
            peaks = []
            for k, region in enumerate(['Core A', 'Core B', 'Surround']):
                indices = [int(np.argmax(regional[i*half:(i+1)*half, k]))+i*half
                           for i in range(2)]
                assert all(i*half < ix < (i+1)*half-1 for i, ix in enumerate(indices))
                times = np.array(indices)*T/len(r)
                peaks.append(dict(region=region, half_peak_time_ms=times,
                    half_peak_rate_Hz=regional[indices, k],
                    alternating_peak_intervals_ms=[times[1]-times[0], T-times[1]+times[0]],
                    maximum_abs_equal_phase_half_difference_Hz=float(
                        abs(regional[half:,k]-regional[:half,k]).max())))
            record['regional_peaks'] = peaks
            record['peak_definition'] = ('Maximum within each half of the exact displayed cycle; '
                'these regional waveform peaks are not observer-qualified group-event intervals.')
        records.append(record)
        print(name, [(row['qualified_events'], row['sustained_SCL_contact_names'])
                     for row in retained], flush=True)
    output = dict(status='PD3_PHYSICAL_READOUT_CHECKED', figure_source=str(DATA/'PD3_physical_child.json'),
        observer_source=str(contract_source), contact_names=contract['contact_names'], rows=records,
        statistical_unit='One exact critical parent and one exact unstable doubled child; repeats and four bin origins are not independent realizations.',
        comparison='Parent is at the critical J and child is slightly away on its branch; this is not a same-J causal comparison. Consecutive-half changes are measured within the one child at identical J.',
        missingness='Three group-event metrics are undefined when no events meet the original qualification criteria. Individual contact recruitment remains separately reported.',
        scope='Local periodic waveform and rate-readout effects of PD3. No stable attractor, stochastic irregularity, new propagation template or native-SNN correspondence is inferred.')
    write(DATA/'PD3_child_followup/spatial_contact_readout.json', output)
    print(records[-1]['regional_peaks'], flush=True)


if __name__ == '__main__':
    main()
