"""Native voltage/input description only after exact recording replay passes."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'
DEST = OUT / 'native_voltage_observation'
REFERENCE = ROOT / 'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/runs/eta0.0005_s9108401'


def main():
    read = lambda p: json.loads(p.read_text())
    audit = read(DEST / 'replay_audit.json')
    assert audit['status'] == 'NATIVE_VOLTAGE_REPLAY_PASS'
    c = read(OUT / 'native_voltage_observation_contract.json')
    folder = DEST / 'runs/native_t9000_voltage_observe'
    chunks = [np.load(f) for f in sorted((folder / 'voltage').glob('*.npz'))]
    t = np.concatenate([z['time_ms'] for z in chunks])
    m = np.concatenate([z['moments'] for z in chunks])
    counts = np.concatenate([z['spikes'] for z in chunks])
    sizes = chunks[0]['cell_counts']
    assert all(np.array_equal(z['cell_counts'], sizes) for z in chunks)
    assert sizes[:3].sum() == 32000 and counts.shape == (13700, 4)
    reference_global = []
    for start in [90000, 95000, 100000]:
        z = np.load(REFERENCE / 'chunks' / f'{start:010d}_{start+5000:010d}.npz')
        n = min(500, (103700-start)//10)
        reference_global.append(z['spikes_1ms'][:n, 0])
    reference_global = np.concatenate(reference_global)
    assert np.array_equal(counts[:, :3].sum(1).reshape(-1, 10).sum(1), reference_global)
    assert np.max(abs(m[:, :, 7] - (m[:, :, 3]-m[:, :, 5]-m[:, :, 6]))) < 1e-9
    assert np.min(m[:, :, 1] - m[:, :, 0]**2) > -1e-8
    assert np.all((m[:, :, 2] >= 0) & (m[:, :, 2] <= 1))
    rate = counts.reshape(-1, 100, 4).sum(1) / sizes * 100
    rate_times = 9000 + (np.arange(len(rate))+.5)*10
    params_source = ROOT / 'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators/g20/prepared.json'
    reset = read(params_source)['params']['V_reset']
    assert reset == 11
    windows = [[9000, 9420], [9420, 9870], [9870, 10370]]
    rows = []
    for lo, hi in windows:
        keep = (t >= lo) & (t < hi)
        steps = slice((lo-9000)*10, (hi-9000)*10)
        for j, label in enumerate(c['populations']):
            v = m[keep, j, 0]
            rows.append(dict(window_ms=[lo,hi], population=label, cells=int(sizes[j]),
                mean_rate_hz=float(counts[steps,j].sum()/sizes[j]/(hi-lo)*1000),
                min_group_mean_voltage_relative_reset_mv=float(v.min()-reset),
                median_group_mean_voltage_relative_reset_mv=float(np.median(v)-reset),
                time_fraction_group_mean_voltage_below_reset=float(np.mean(v < reset)),
                mean_cell_fraction_below_reset=float(m[keep,j,2].mean()),
                minimum_net_current_mv=float(m[keep,j,7].min()),
                maximum_net_current_mv=float(m[keep,j,7].max())))
    np.savez_compressed(DEST / 'observations.npz', time_ms=t, moments=m, spike_counts=counts,
                         cell_counts=sizes, rate_10ms_hz=rate, rate_time_ms=rate_times, reset_mv=reset)
    q = dict(status='NATIVE_VOLTAGE_OBSERVATION_AUDITED', rows=rows,
        observation_columns=['V', 'V_squared', 'fraction_V_below_reset', 'AMPA', 'raw_GABA', 'Z_GABA', 'eta_M_M', 'net_current', 'Z'],
        global_spike_counts_all1370ms_bitwise_reference=True,
        current_product_and_variance_checks=True, reset_mv=reset, reset_parameter_source=str(params_source),
        exact_replay=str(DEST / 'replay_audit.json'), statistical_unit=c['statistical_unit'],
        clock=c['ordering'], scope=c['scope'], onset_type='NOT_ESTABLISHED', model_promoted=False)
    (DEST / 'result.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(q,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
