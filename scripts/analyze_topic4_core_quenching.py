#!/usr/bin/env python3
"""Describe local core quenching without relabelling it as network recovery."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import time
from pathlib import Path
import numpy as np
import analyze_topic4_autonomous_recovery as common


def main(root, name):
    folder = root / 'runs' / name
    job = common.read(root / 'jobs' / (name + '.json'))
    a = common.load(folder, ['time_ms', 'regions_1ms', 'slow_time_ms',
                             'Z', 'regional_currents'])
    if a is None:
        raise ValueError('No complete observation chunk')
    nr = np.load(root / 'geometry.npz')['region_counts'][:3]
    n = len(a['time_ms']) // 10
    rates = a['regions_1ms'][:n * 10, :3].reshape(n, 10, 3).sum(1) / nr / .01
    ts = a['slow_time_ms'] / 1000
    pool = None
    files = sorted((folder / 'pool_chunks').glob('*.npz'))
    if files:
        q = {key: [] for key in ['time_ms', 'rate_Hz', 'raw_global_current']}
        for path in files:
            if '.tmp.' in path.name:
                continue
            with np.load(path) as data:
                for key in q:
                    q[key].append(data[key])
        pool = {key: np.concatenate(values) for key, values in q.items()}
        assert np.allclose(np.diff(pool['time_ms']), 5.)
    rows = []
    for quiet, active in [(0, 1), (1, 0)]:
        mask = (rates[:, quiet] < 5) & (rates[:, active] >= 200)
        edges = np.diff(np.r_[False, mask, False].astype(int))
        for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if hi - lo < 20:
                continue
            start, end = lo * .01, hi * .01
            selected = (ts >= start) & (ts < end)
            assert selected.any()
            row = dict(quiet_core='AB'[quiet], active_core='AB'[active],
                interval_s=[start, end], duration_s=end - start,
                mean_regional_rates_Hz=rates[lo:hi].mean(0).tolist(),
                min_active_core_rate_Hz=float(rates[lo:hi, active].min()),
                max_quiet_core_rate_Hz=float(rates[lo:hi, quiet].max()),
                regional_current_IE_rawII_deliveredII_effectiveII_M=
                    a['regional_currents'][selected].mean(0).tolist(),
                regional_Z_mean=a['Z'][selected][:, [5, 6, 7]].mean(0).tolist(),
                mean_Z_start_end=a['Z'][selected][[0, -1], 0].tolist())
            if pool is not None:
                pt = pool['time_ms'] / 1000
                # Include the preceding sample and use the known decay between
                # samples. Spikes can only increase Rg above this lower bound.
                enclosing = (pt < end) & (pt + .005 > start)
                assert enclosing.any() and pt[0] <= start and pt[-1] + .005 >= end - 1e-9
                lower = job['pool_gain'] * np.maximum(
                    pool['rate_Hz'] * np.exp(-.005 / job['pool_tau_s'])
                    - job['pool_threshold_Hz'], 0)
                bound = float(lower[enclosing].min())
                row['continuous_global_current_lower_bound'] = bound
                row['threshold_mV_equivalent'] = job['threshold']
                row['global_input_alone_guarantees_b_zero'] = bound >= job['threshold']
            rows.append(row)
    common.write(folder / 'core_quenching_review.json', dict(updated_at=time.time(),
        source=str(folder), job=job, observed_s=n * .01, intervals=rows,
        definition='Actual10ms regional counts: one core<5Hz and the other>=200Hz for>=200ms. Descriptive local-quenching diagnostic; does not change the high/return acceptance rule.',
        region_definition='All E neurons in fixed1.75mm observation neighborhoods; same denominators as original regional rate observations.',
        global_bound='The minimum of causal5ms Rg decay bounds over every interval intersecting the quiet-core window; nonnegative spike increments can only increase Rg. A bound above Ith guarantees b_i=1[J_i<Ith]=0 for every E cell.',
        interpretation='A quiet core with a persistently active partner is not network recovery. For nativeZ (rho=0), b=0 gives tauZ*dZ/dt=-Z; for revisedZ, b=0 gives tauZ*dZ/dt=rho-(1+rho)*Z, so the native depletion interpretation cannot be copied to rho>0. Regional current means are not single-neuron stability tests.',
        human_review='PENDING'))
    print([(r['quiet_core'], r['interval_s'], r.get('continuous_global_current_lower_bound')) for r in rows])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=common.OUT / 'activity_global_pool_round3')
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    main(args.root, args.name)
