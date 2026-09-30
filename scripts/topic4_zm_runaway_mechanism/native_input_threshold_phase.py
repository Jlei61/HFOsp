"""Fixed physical threshold-mixture control for the millisecond input assay.

Preserves each original cell threshold and conditional input law; no fitted
parameters, network simulation, timing alignment, or acceptance promotion.
"""
from common import OUT, np, read, write, log
from native_input_local_lif import simulate, condition
from native_input_millisecond_assay import metrics
from datetime import datetime
import argparse, os

SOURCE = OUT / 'native_input_bridge'
PARENT = OUT / 'native_current_memory/millisecond_assay'
DEST = OUT / 'native_current_memory/threshold_phase'


def run(device):
    DEST.mkdir(parents=True, exist_ok=True)
    assert not (DEST / 'contract.json').exists()
    c = read(PARENT / 'contract.json')
    write(DEST / 'contract.json', dict(
        created_local=datetime.now().astimezone().isoformat(),
        question='Does restoring original within-group threshold heterogeneity explain the over-concentrated millisecond spike counts?',
        only_change='Replace group mean threshold by equally weighted actual original cell thresholds. Same native prestep mean, private variance forcing, observed Z, initialization, path seeds and time steps.',
        groups=c['groups'], dt_ms=c['dt_ms'], seed=c['seed'],
        windows_ms=c['windows_ms'], bin_ms=[1, 5, 50],
        comparisons='Reuse all four predeclared count metrics and matched-step conditional predictive ranges. Paired mean and threshold-mixture references; no time shift or response fitting.',
        scope='Local supplied-input diagnostic only. Conditional independent Gaussian inputs are not the actual network input distribution. No model promotion or bifurcation conclusion.',
        budget='Six groups at two time steps only.'))
    z = np.load(SOURCE / 'selected_input_history.npz')
    raw = np.load(SOURCE / 'selected_raw_variance_forcing.npz')
    geo = np.load(SOURCE / 'membership.npz')
    old = np.load(PARENT / 'dt0.1.npz')
    nrep = old['replicates']; R = int(nrep.max()); G = len(c['groups'])
    assert np.array_equal(z['groups'], c['groups'])
    assert np.array_equal(z['time_ms'], raw['time_ms'])
    m = {k: z['moments'][:, j] for j, k in enumerate(z['moment_names'])}
    wave = np.stack([m['net'], raw['raw_variances'][:, 2], raw['raw_variances'][:, 3], m['z']], axis=1).transpose(2, 1, 0).copy()
    theta = np.empty((G, R)); threshold_rows = []
    for j, g in enumerate(c['groups']):
        original = geo['actual_threshold_mv'][geo['cell_group'] == g]
        N = int(z['group_size'][j]); nr = int(nrep[j])
        assert len(original) == N and nr % N == 0
        assert abs(original.mean() - z['theta'][j]) < 1e-12
        theta[j] = np.resize(original, R)
        assert np.array_equal(theta[j, :nr].reshape(-1, N), np.broadcast_to(original, (nr // N, N)))
        threshold_rows.append(dict(group=g, N=N, mean=float(original.mean()), std=float(original.std()), min=float(original.min()), max=float(original.max())))
    jobs = dict(status='RUNNING', pid=os.getpid(), expected=2, completed=[])
    write(DEST / 'jobs.json', jobs)
    for dt in c['dt_ms']:
        pars = np.array([condition(0., z['theta'][j], 1., 1., 'EI'[int(z['population'][j])], dt=dt) for j in range(G)])
        counts = simulate(pars, wave, theta, nrep, dt, 1000., 1350, 1., c['seed'], device)
        assert counts.max() <= 1
        # Homogeneous groups must reproduce the existing reference bitwise.
        reference = np.load(PARENT / f'dt{dt:g}.npz')
        homogeneous = [j for j, r in enumerate(threshold_rows) if r['std'] == 0.]
        for j in homogeneous:
            assert np.array_equal(counts[j], reference['counts'][j])
        np.savez_compressed(DEST / f'dt{dt:g}.npz', counts=counts.astype('u1'), replicates=nrep,
                            group_sizes=z['group_size'], groups=z['groups'], dt_ms=dt, thresholds=theta)
        jobs['completed'].append(dt); write(DEST / 'jobs.json', jobs)
        log('THRESHOLD PHASE COMPLETE', dt, 'homogeneous parity', homogeneous)
    jobs.update(status='COMPLETE', homogeneous_groups_bitwise=True)
    write(DEST / 'jobs.json', jobs); write(DEST / 'thresholds.json', threshold_rows)
    score()


def score():
    assert read(DEST / 'jobs.json')['status'] == 'COMPLETE'
    c = read(DEST / 'contract.json'); native = np.load(PARENT / 'native_counts.npz'); rows = []
    for dt in c['dt_ms']:
        z = np.load(DEST / f'dt{dt:g}.npz')
        for j, g in enumerate(c['groups']):
            N = int(z['group_sizes'][j]); nr = int(z['replicates'][j])
            trials = z['counts'][j, :nr].reshape(-1, N, 1350).sum(1).astype(float)
            cut = len(trials) // 2
            for bin_ms in c['bin_ms']:
                nb = 1350 // bin_ms; tt = np.arange(nb) * bin_ms + 9000
                trial = trials.reshape(-1, nb, bin_ms).sum(2)
                observed = native['counts'][:, j].reshape(nb, bin_ms).sum(1).astype(float)
                for lo, hi in c['windows_ms']:
                    keep = (tt >= lo) & (tt + bin_ms <= hi)
                    a = trial[:cut, keep]; b = trial[cut:, keep]
                    mu = a.mean(0); var = a.var(0, ddof=1)
                    actual = metrics(observed[keep][None], mu, var)
                    for key, samples in metrics(b, mu, var).items():
                        low, median, high = np.quantile(samples, [.025, .5, .975])
                        value = float(actual[key][0])
                        rows.append(dict(group=g, N=N, dt_ms=dt, bin_ms=bin_ms, window_ms=[lo, hi],
                                         metric=key, observed=value, predictive_low=float(low), predictive_median=float(median),
                                         predictive_high=float(high), outside_this_step=bool(value < low or value > high),
                                         complete_bins=int(keep.sum()), training_reference_trials=cut, predictive_trials=len(b)))
    paired = []
    for row in rows:
        if row['dt_ms'] != .1: continue
        other = next(x for x in rows if x['dt_ms'] == .05 and all(x[k] == row[k] for k in ['group', 'bin_ms', 'window_ms', 'metric']))
        lower = row['observed'] < row['predictive_low'] and other['observed'] < other['predictive_low']
        upper = row['observed'] > row['predictive_high'] and other['observed'] > other['predictive_high']
        paired.append(dict(group=row['group'], bin_ms=row['bin_ms'], window_ms=row['window_ms'], metric=row['metric'],
                           side='low' if lower else 'high' if upper else None,
                           outside_same_side_both_steps=bool(lower or upper)))
    write(DEST / 'result.json', dict(status='THRESHOLD_PHASE_DIAGNOSTIC_COMPLETE', rows=rows, paired=paired,
                                    model_promoted=False, scope=c['scope']))
    log('THRESHOLD PHASE PREENTRY', [(r['group'], r['metric'], r['side']) for r in paired if r['bin_ms'] == 1 and r['window_ms'][0] == 9000])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--device', type=int, default=0)
    run(p.parse_args().device)
