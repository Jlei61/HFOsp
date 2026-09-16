"""Read-only counterfactual observations; never replace the frozen XY objective."""
import numpy as np


def window_onsets(envelope, windows, dt_ms, *, margin=.1, timing=.5, bar_scope='event'):
    """Same peak/half-peak rule as the root readout, with no lineage mask.

    Event windows and output rows are preserved, including completely unreadable rows.
    This is an observation diagnostic, not evidence that concurrent roots are causal.
    """
    env = np.asarray(envelope, float)
    if env.ndim != 2 or not np.isfinite(env).all() or dt_ms <= 0 or bar_scope not in ('event', 'run'):
        raise ValueError('invalid observation input')
    out = np.full((len(windows), len(env)), np.nan)
    run_floor = env.min(); run_bar = run_floor + margin * (env.max() - run_floor)
    for i, (lo, hi) in enumerate(windows):
        start = max(0, int(round(lo / dt_ms))); stop = min(env.shape[1], int(round(hi / dt_ms)))
        segment = env[:, start:stop]
        if not segment.size:
            continue
        floor = segment.min()
        bar = floor + margin * (segment.max() - floor) if bar_scope == 'event' else run_bar
        peaks = segment.max(axis=1)
        for c in np.flatnonzero(peaks > bar):
            hit = np.flatnonzero(segment[c] >= timing * peaks[c])
            if len(hit): out[i, c] = (start + hit[0]) * dt_ms
    return out


def support_summary(onsets, groups, pairs):
    mask = np.isfinite(onsets)
    n = len(mask)
    if not n:
        return {'n_events': 0, 'both_shafts_fraction': None, 'mean_contacts': None,
                'eligible_cross_pairs': 0, 'positive_cross_pairs': 0,
                'conditional_gate': False, 'unreadable_fraction': None}
    p = np.asarray(pairs['ICL-SCL'])
    joint = (mask[:, p[:, 0]] & mask[:, p[:, 1]]).sum(axis=0)
    counts = mask.sum(axis=1)
    both = mask[:, groups['ICL']].any(axis=1) & mask[:, groups['SCL']].any(axis=1)
    gate = True
    for cp in pairs.values():
        cp = np.asarray(cp)
        j = (mask[:, cp[:, 0]] & mask[:, cp[:, 1]]).sum(axis=0)
        gate &= np.mean(j >= 5) >= .5
    return {'n_events': int(n), 'both_shafts_fraction': float(both.mean()),
            'mean_contacts': float(counts.mean()), 'median_contacts': float(np.median(counts)),
            'contact_count_histogram': (np.bincount(counts, minlength=mask.shape[1]+1) / n).tolist(),
            'eligible_cross_pairs': int(np.sum(joint >= 5)),
            'positive_cross_pairs': int(np.sum(joint > 0)),
            'cross_pair_joint_counts': joint.tolist(),
            'conditional_gate': bool(gate), 'unreadable_fraction': float(np.mean(counts == 0)),
            'readable_fraction_5contacts': float(np.mean(counts >= 5)),
            'recruitment_per_contact': mask.mean(axis=0).tolist()}


def support_gate_probability(onsets_by_seed, groups, pairs, total_n, draws, seed):
    """Stratified empirical bootstrap projection, NOT new observations or validation."""
    rng = np.random.default_rng(seed)
    source = [np.isfinite(x) for x in onsets_by_seed]
    if any(len(m) == 0 for m in source): return None
    passes = []
    for _ in range(draws):
        chunks = [m[rng.integers(0, len(m), total_n // len(source) + (i < total_n % len(source)))]
                  for i, m in enumerate(source)]
        mask = np.concatenate(chunks)
        ok = True
        for cp in pairs.values():
            cp = np.asarray(cp)
            count = (mask[:, cp[:, 0]] & mask[:, cp[:, 1]]).sum(axis=0)
            ok &= np.mean(count >= 5) >= .5
        passes.append(ok)
    return float(np.mean(passes))
