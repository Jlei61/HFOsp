"""Independent readout of the matched natural-history Core-A Z step.

Reconstruct regional rates from saved group rates. Neither event duration
nor a finite-window contrast is promoted to a bifurcation certificate.
"""
from common import OUT, np, read, write, model
from onset_state_continuation import regional_weights
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import argparse


def intervals(mask):
    edges = np.flatnonzero(np.r_[True, mask[1:] != mask[:-1], True])
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if mask[a]]


def main(destination):
    base = Path(destination).resolve()
    s = model(40)
    W = regional_weights(s)
    core = s.E & (s.geo['group_region'] == 0)
    rows = []
    contracts = []
    for label in ['held_short_field', 'depleted_entry_field']:
        folder = base / label
        assert read(folder / 'jobs.json')['status'] == 'COMPLETE'
        contract = read(folder / 'contract.json')
        contracts.append(contract)
        z = np.load(folder / 'trajectory.npz')
        r = z['group_rate_hz'].astype(float)
        regional = r @ W.T
        saved = z['regional_rate_hz']
        # Storage is float32; its rounding cannot exceed this weighted bound.
        bound = np.abs(np.spacing(z['group_rate_hz']).astype(float)) @ W.T
        error = abs(regional - saved)
        assert np.all(error <= bound + 1e-10)
        sm = uniform_filter1d(regional, 10, axis=0, mode='nearest')
        original = read(folder / 'result.json')
        regions = []
        for j, name in enumerate(['Global E', 'Core A', 'Core B', 'Surround']):
            quiet = [(a, b) for a, b in intervals(sm[:, j] < 5) if b-a >= 20]
            assert quiet == [tuple(q) for q in original['rows'][j]['quiet_intervals_ms']]
            edges = [(0, 0), *quiet, (len(r), len(r))]
            active = [dict(start_ms=b, end_ms=c, duration_ms=c-b,
                           left_censored=b == 0, right_censored=c == len(r))
                      for (_, b), (c, _) in zip(edges[:-1], edges[1:]) if c > b]
            assert active == original['rows'][j]['activities']
            complete = [q for q in active if not q['left_censored'] and not q['right_censored']]
            regions.append(dict(region=name, quiet_fraction=sum(b-a for a, b in quiet)/len(r),
                                complete_activities=len(complete),
                                max_complete_ms=max([q['duration_ms'] for q in complete], default=None),
                                complete_at_least_1s=[q for q in complete if q['duration_ms'] >= 1000],
                                activities=active, quiet_intervals_ms=quiet))
        source = np.load(contract['source'])
        final = np.load(folder / 'final_state.npz')
        assert np.array_equal(z['Z'], final['syn'][5])
        assert np.array_equal(z['Z'][~core], source['syn'][5, ~core])
        assert np.all(final['parameters'][19] == 0)
        assert np.all(final['parameters'][20] == 1)
        initial_tick = round(int(source['clock'][0]) * contract['source_dt_ms'] / contract['dt_ms'])
        assert int(final['clock'][0]) - initial_tick == round(len(r) / contract['dt_ms'])
        rows.append(dict(label=label, source=str(folder), duration_ms=len(r),
                         Z_A=float(np.average(z['Z'][core], weights=s.sizes[core])),
                         regional_reconstruction_max_error_hz=float(error.max()),
                         regions=regions,
                         spatial_persistent_fraction_last1s=original['spatial_persistent_fraction_last1s']))
    assert contracts[0]['source'] == contracts[1]['source']
    assert contracts[0]['dt_ms'] == contracts[1]['dt_ms']
    intervention = read(base / 'depleted_entry_field/Z_intervention.json')
    assert intervention['all_initial_fast_M_history_bitwise_unchanged']
    write(base / 'independent_audit.json', dict(status='PASS', rows=rows,
          initial_fast_M_history_identical=True, intervention=intervention,
          unit='One deterministic matched-history pair; repeated events are not independent trials.',
          scope='Original fixed-Z, dynamic-M spatial rate. Finite-duration observation only; no asymptotic attractor, coexistence, bifurcation type, or native onset attribution.',
          target_entry_type='NOT_ESTABLISHED', model_promoted=False))
    for row in rows:
        a = row['regions'][1]
        print(row['label'], row['Z_A'], 'max_complete_ms', a['max_complete_ms'],
              'complete_ge1s', a['complete_at_least_1s'], 'last', a['activities'][-1:], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('destination')
    main(p.parse_args().destination)
