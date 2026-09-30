"""Independent count, clock, occupancy, and scalar recurrence verification."""
from pathlib import Path
import json
import numpy as np

OUT = Path(__file__).resolve().parents[2] / 'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'


def main():
    dest = OUT / 'refractory_current_closure'
    c = json.loads((OUT / 'refractory_current_closure_contract.json').read_text())
    q = json.loads((dest / 'result.json').read_text())
    data = np.load(dest / 'observations.npz')
    prior = np.load(OUT / c['source'] / 'response.npz')
    assert np.array_equal(data['counts'], prior['counts'][c['indices']])
    R = c['inherited']['replicates']; B = c['inherited']['phase_bins']
    dt, period = float(data['dt_ms']), float(data['T_ms'])
    raw = data['block_sums']; n = raw.shape[2]
    bins = np.minimum((((np.arange(n)+1)*dt/period) % 1*B).astype(int), B-1)
    exposure = np.bincount(bins, minlength=B)
    assert np.array_equal(exposure*dt, prior['occupancy_ms'])
    errors, rows = [], []
    for j, p in enumerate(data['pars']):
        assert np.array_equal(raw[j].sum(axis=0)/R, data['ensemble_mean'][j])
        v, current, clamped, fired, reset_q, clamp_q, previous = data['ensemble_mean'][j].T
        a, vr, theta = p[18], p[21], p[1]
        # Sum spike probabilities at the actual clock, rather than infer them
        # from reset charge (which includes finite-step threshold overshoot).
        reconstructed_counts = np.bincount(bins, weights=fired*R, minlength=B)
        assert np.array_equal(reconstructed_counts, data['counts'][j].sum(axis=0))
        block_error = raw[j,:,:,0]-a*raw[j,:,:,6]-(1-a)*raw[j,:,:,1]+raw[j,:,:,4]+raw[j,:,:,5]
        assert abs(block_error).max() < 1e-8
        nref = int(p[19]); history = np.zeros(n)
        for lag in range(1, nref):
            history[lag:] += fired[:-lag]
        assert np.array_equal(history[nref:], clamped[nref:])
        operands = [(reset_q, clamp_q), (reset_q, (1-a)*(current-vr)*clamped),
                    (fired*(theta-vr), clamp_q), (fired*(theta-vr), (1-a)*(current-vr)*clamped)]
        select = np.arange(n)*dt >= period
        for k, (rq, cq) in enumerate(operands):
            prediction = np.empty(n); state = previous[0]
            for i in range(n):
                state = a*state+(1-a)*current[i]-rq[i]-cq[i]
                prediction[i] = state
            error = float(abs(prediction-data['reconstructed_voltage'][j,k]).max())
            assert error < 1e-8; errors.append(error)
            rms = float(np.sqrt(np.mean((prediction[select]-v[select])**2)))
            assert abs(rms-q['rows'][j]['summaries'][k]['voltage_RMS_error_mv']) < 1e-9
        rows.append(dict(condition=q['rows'][j]['condition'], max_block_balance_error=float(abs(block_error).max()),
            per_phase_counts_exact=True, refractory_history_exact=True))
    result = dict(status='REFRACTORY_CURRENT_INDEPENDENT_AUDIT_PASS', rows=rows,
        max_recurrence_difference_mv=max(errors), original_counts_bitwise=True,
        scope='Measured firing history supplied; this is not a predictive population response validation.')
    (dest/'independent_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
