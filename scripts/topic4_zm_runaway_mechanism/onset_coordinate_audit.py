"""Locate the existing autonomous rate onset in its own recorded Z/M field.

This reads the original trajectory, preserves the registered onset definition,
and reports the two recorded 10-ms state endpoints around that time. It does
not substitute a native-checkpoint conditional path or a core-only Z mean.
"""
from common import *
from scipy.ndimage import uniform_filter1d


def first_high_time(time_ms, global_rate):
    sm = uniform_filter1d(global_rate, 10, mode='nearest')
    edges = np.diff(np.r_[0, (sm >= 200).astype(int), 0])
    pairs = zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
    return next((float(time_ms[a]) for a, b in pairs if b-a >= 200), None)


def main():
    s = model()
    rows = []
    for label in ['A4_det_meandrive', 'A4_stoch_seed9108401', 'A4_stoch_meandrive']:
        source = BASE/'runs'/label/'trajectory.npz'
        data = np.load(source)
        result = read(source.parent/'result.json')
        contract = read(source.parent/'contract.json')
        t = data['time_ms']
        assert np.array_equal(t, np.arange(1, len(t)+1))
        assert len(t) == 10*len(data['Z']) == 10*len(data['D'])
        assert contract['dynamic_Z'] and contract['dynamic_M']
        # Producer saves the state at every tenth 1-ms endpoint: 10, 20, ... ms.
        state_times = t[9::10]
        rates = data['field_E_hz'] @ (data['cell_counts']/data['cell_counts'].sum())
        onset = first_high_time(t, rates)
        assert onset == result['high_onset_ms'], (label, onset, result['high_onset_ms'])
        zs = data['Z'].astype(float)
        rebuilt_D = 1-zs[:, s.E] @ s.mean_weights
        max_D_error = float(np.max(abs(rebuilt_D-data['D'])))
        assert max_D_error < 4e-8  # Saved Z is float32; D was computed before casting.
        right = int(np.searchsorted(state_times, onset))
        indices = sorted(set([max(0, right-1), min(right, len(state_times)-1)]))
        endpoints = []
        for index in indices:
            reg = s.geo['group_region']
            means = [float(np.average(zs[index, s.E & (reg == k)],
                                      weights=s.sizes[s.E & (reg == k)])) for k in range(3)]
            endpoints.append(dict(index=index, time_ms=float(state_times[index]),
                D=float(data['D'][index]), global_Z=float(1-data['D'][index]),
                Z_coreA_coreB_surround=means,
                global_M_current_mV_equiv=float(data['M_current'][index, s.E] @ s.mean_weights)))
        assert endpoints[0]['time_ms'] <= onset <= endpoints[-1]['time_ms']
        rows.append(dict(label=label, source=str(source), onset_ms=onset,
            endpoint_records=endpoints, reconstructed_D_max_error=max_D_error,
            Z='dynamic', M='dynamic', dt_ms=contract['dt_ms'],
            numerical_scheme='Original stored A4 simulation; not a new endpoint-RK4 run',
            onset_definition='First 1-ms bin of >=200 ms with 10-ms smoothed global E rate >=200 Hz'))
    native = read(OUT/'native_same_history_feedback/late_clamp_result.json')
    nrow = next(r for r in native['rows'] if r['clamp_time_ms'] == 9870)
    turn = read(OUT/'periodic/native_turn_G8505_M65536/result.json')
    out = dict(status='COMPLETE', rate_autonomous_rows=rows,
        native_context=dict(checkpoint_ms=9870, global_Z=nrow['global_Z_at_clamp'],
            held_Z_high_entry_s=nrow['held']['high_entry_s'],
            interpretation='A native checkpoint and a paired finite continuation; not an exact global-Z coordinate for the separately binned native onset'),
        native_conditional_periodic_turn=dict(D=turn['D'], global_Z=1-turn['D'],
            Z='held native-checkpoint spatial path', M='dynamic',
            type=turn['bifurcation_type']),
        state_time_convention='Z/M/D index j is the state at t=10*(j+1) ms. No independent regional time shift or interpolation is applied.',
        scope='Coordinates of existing onset readouts only. Different paths, mean Z values, or core means are not identified as the same bifurcation.',
        lineage_note='Original a4_acceptance.py used indices t/10 for some checkpoint D comparisons, a +10 ms offset. This audit uses saved-state endpoint times and does not overwrite the original acceptance decisions.')
    write(OUT/'onset_coordinate_audit.json', out)
    for row in rows:
        log(row['label'], row['onset_ms'], row['endpoint_records'])


if __name__ == '__main__':
    main()
