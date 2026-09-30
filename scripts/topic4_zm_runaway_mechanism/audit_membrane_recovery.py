"""Independent aggregation of recording-only local membrane diagnostics."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'


def main():
    read = lambda p: json.loads(p.read_text())
    dest = OUT / 'membrane_recovery_diagnostic'
    c = read(OUT / 'membrane_recovery_diagnostic_contract.json')
    result = read(dest / 'result.json')
    source = np.load(OUT / c['source'] / 'response.npz')
    z = np.load(dest / 'response.npz')
    sim = c['inherited_simulation']
    period = float(z['T_ms'])
    r, b, dt = sim['replicates'], sim['phase_bins'], sim['dt_ms']
    steps = round(sim['record_cycles'] * period / dt)
    exposure = np.zeros(b, np.int64)
    for k in range(1, steps+1):
        bin_index = min(int((k*dt/period % 1)*b), b-1)
        exposure[bin_index] += 1
    assert np.array_equal(exposure, z['exposure_steps'])
    errors = []
    for j, index in enumerate(c['selected_indices']):
        assert np.array_equal(z['counts'][j], source['counts'][index])
        per_path = z['moment_sums'][j] / exposure[:, None]
        mean = np.sum(per_path, axis=0) / r
        sem = np.sqrt(np.sum((per_path - mean)**2, axis=0) / (r*(r-1)))
        errors += [np.max(abs(mean-z['moments_mean'][j])),
                   np.max(abs(sem-z['moments_sem'][j]))]
        assert np.isfinite(per_path).all()
        # Flags are accumulated as exact integers, independent of floating moments.
        for col in [2, 5]:
            raw = z['moment_sums'][j, :, :, col]
            assert np.array_equal(raw, raw.astype(np.int64))
            assert np.all((raw >= 0) & (raw <= exposure))
        assert np.min(per_path[:, :, 1] - per_path[:, :, 0]**2) > -1e-8
        reset = float(z['pars'][j, 21])
        active = (mean[:, 0] - reset*mean[:, 2]) / (1-mean[:, 2])
        errors.append(np.max(abs(active-z['nonrefractory_mean_voltage_mv'][j])))
        over = z['predicted_hz'][j, 1] - z['measured_hz'][j]
        i = int(over.argmax())
        row = result['rows'][j]
        assert row['largest_rate_overprediction_hz'] == float(over[i])
        assert row['voltage_at_largest_overprediction_mv'] == float(mean[i, 0])
        assert row['refractory_at_largest_overprediction'] == float(mean[i, 2])
        assert np.all(mean[:, 0] <= z['pars'][j, 1])
    assert max(errors) < 1e-10
    q = dict(status='COUNT_AND_MOMENT_AUDIT_PASS', max_aggregation_difference=float(max(errors)),
             paths_per_condition=r, conditions=len(c['selected_indices']),
             spike_counts_bitwise=True, exposure_independent_scalar_clock=True,
             moment_inequalities=True, integer_flag_checks=True,
             scope='Checks saved counts and moments; associations are descriptive, not proof of a sufficient closure or onset bifurcation.')
    (dest / 'independent_audit.json').write_text(json.dumps(q, indent=2)+'\n')
    print(json.dumps(q))


if __name__ == '__main__':
    main()
