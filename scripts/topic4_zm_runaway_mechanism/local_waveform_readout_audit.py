"""Independent count-level audit for the two bounded local waveform assays."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'


def read(path):
    return json.loads(path.read_text())


def main(name):
    assert name in ('in_domain_waveform', 'factorial_waveform')
    c = read(OUT / f'{name}_contract.json')
    folder = OUT / name
    q = read(folder / 'result.json')
    preparation = read(folder / 'preparation.json')
    z = np.load(folder / 'response.npz')
    p = np.load(folder / 'prepared.npz')
    original = np.load(OUT / c['source'])
    period = float(p['T_ms'])
    assert period == float(original['T_ms'])
    rows, bins, replicas = len(preparation['rows']), c['phase_bins'], c['replicates']
    counts = z['counts']
    assert counts.shape == (12, replicas, bins) and rows == 12
    assert np.issubdtype(counts.dtype, np.unsignedinteger)
    assert preparation['status'] == 'PREDICTIONS_LOCKED_BEFORE_ACQUISITION'
    assert (folder / 'preparation.json').stat().st_mtime_ns < (folder / 'response.npz').stat().st_mtime_ns
    steps = round(c['record_cycles'] * period / c['dt_ms'])
    phases = (((np.arange(steps) + 1) * c['dt_ms']) % period) / period
    indices = np.minimum(np.floor(phases * bins).astype(int), bins - 1)
    occupancy = np.bincount(indices, minlength=bins) * c['dt_ms']
    assert np.max(abs(occupancy - z['occupancy_ms'])) < 1e-10
    measured = counts.sum(axis=1, dtype=np.uint64) / replicas / occupancy * 1000
    assert np.max(abs(measured - z['measured_hz'])) < 1e-10
    params = read(ROOT / 'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators/g20/prepared.json')['params']
    observed = []
    worst = 0.
    for j, group in enumerate(preparation['rows']):
        assert group == {k: q['rows'][j][k] for k in group}
        pop = group['pop']
        ref = params['tau_ref_' + pop]
        assert abs(p['pars'][j, 18] - np.exp(-c['dt_ms'] / params['tau_m_' + pop])) < 1e-15
        assert p['pars'][j, 19] == round(ref / c['dt_ms'])
        assert p['pars'][j, 1] == group['theta_mv'] and p['pars'][j, 21] == params['V_reset']
        assert counts[j].sum(axis=1).max() <= int(np.ceil(steps * c['dt_ms'] / ref)) + 1
        source = original['wave'][j // 3]
        if name == 'factorial_waveform':
            expected = source.copy()
            if group['condition'] == 'mean_only':
                expected[1:] = np.mean(source[1:], axis=1, keepdims=True)
            elif group['condition'] == 'variances_only':
                expected[0] = np.mean(source[0])
        else:
            a = group['amplitude']
            centered = source - source.mean(axis=1, keepdims=True)
            centered /= abs(centered).max(axis=1, keepdims=True)
            span = c['threshold_mv'] - c['reset_mv']
            expected = np.array([
                c['reset_mv'] + span * (c['baseline_normalized_mean'] + a * centered[0]),
                (span * c['baseline_sigma_E']) ** 2 * (1 + a * centered[1]),
                (span * c['baseline_sigma_I']) ** 2 * (1 + a * centered[2])])
        assert np.array_equal(expected, p['wave'][j])
        mean = float(counts[j].sum(dtype=np.uint64) / replicas / (steps * c['dt_ms']) * 1000)
        assert abs(mean - q['rows'][j]['MC_mean_hz']) < 1e-10
        for k, v in enumerate(q['rows'][j]['rows']):
            pred = z['predicted_hz'][j, k]
            l2 = np.linalg.norm(pred - measured[j]) / max(np.linalg.norm(measured[j]), np.sqrt(bins))
            bias = abs(np.average(pred, weights=occupancy) - mean) / max(mean, 1.)
            worst = max(worst, abs(l2 - v['waveform_L2']), abs(bias - v['relative_mean_error']))
            gate = c['acceptance']
            assert v['passed'] == bool(l2 <= gate['normalized_waveform_RMSE_max'] and bias <= gate['relative_cycle_mean_error_max'])
        observed.append(dict(source_label=group['source_label'], condition=group.get('condition', group.get('amplitude')),
                             MC_rate_min_hz=float(measured[j].min()), MC_rate_max_hz=float(measured[j].max()), mean_hz=mean))
    assert worst < 1e-10
    result = dict(status='COUNT_LEVEL_AUDIT_PASS', maximum_readout_difference=worst,
                  cases=rows, observations=observed,
                  predictions_locked_before_observations=True, exact_counterfactual_input_identity=True,
                  scope='Validates assay inputs, clock, raw-count readouts and unchanged local gates. No model, SNN or bifurcation promotion.')
    (folder / 'independent_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(result['status'], name, 'readout error', worst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', choices=['in_domain_waveform', 'factorial_waveform'])
    main(parser.parse_args().name)
