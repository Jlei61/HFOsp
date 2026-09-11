"""Auditable contact identity endpoints; no model fitting or cohort selection."""
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.special import logsumexp


@lru_cache(None)
def subsets(c, k):
    result = np.zeros((len(list(combinations(range(c), k))), c), dtype=float)
    for i, cols in enumerate(combinations(range(c), k)):
        result[i, list(cols)] = 1
    return result


def exact_set_scores(logits, target, available):
    """p(S | |S|=K, available) proportional to exp(sum_{c in S} logit_c).

    This is the independent Bernoulli decoder conditioned on its observed K.
    K=0 and K=number available are deterministic and excluded from identity.
    """
    logits = np.asarray(logits, dtype=float)
    target = np.asarray(target, dtype=bool)
    available = np.asarray(available, dtype=bool)
    if logits.shape != target.shape or target.shape != available.shape:
        raise ValueError('shape mismatch')
    if np.any(target & ~available) or not np.isfinite(logits).all():
        raise ValueError('invalid target support or nonfinite logits')
    shape = logits.shape[:-1]
    c = logits.shape[-1]
    z, y, a = [v.reshape(-1, c) for v in (logits, target, available)]
    k = y.sum(1)
    informative = (k > 0) & (k < a.sum(1))
    nll = np.full(len(z), np.nan)
    accuracy = nll.copy()
    for size in np.unique(k[informative]):
        rows = np.flatnonzero(informative & (k == size))
        candidates = subsets(c, int(size))
        for start in range(0, len(rows), 1024):
            rr = rows[start:start + 1024]
            allowed = (~a[rr]).astype(float) @ candidates.T == 0
            scores = z[rr] @ candidates.T
            scores[~allowed] = -np.inf
            nll[rr] = logsumexp(scores, axis=1) - (z[rr] * y[rr]).sum(1)
            chosen = candidates[scores.argmax(1)].astype(bool)
            accuracy[rr] = (chosen == y[rr]).all(1)
    return {'set_nll': nll.reshape(shape), 'set_accuracy': accuracy.reshape(shape),
            'informative': informative.reshape(shape)}


def prefix_key(rank, prefix=2, include_k=False):
    value = tuple(tuple(np.flatnonzero(rank == g)) for g in range(prefix))
    return value + (int(np.sum(rank == prefix)),) if include_k else value


def fit_branch_dictionary(ranks, fit_rows, minimum=5):
    """Choose ambiguous prefix+K strata using FIT only, before heldout labels."""
    groups = {}
    for i in fit_rows:
        r = ranks[i]
        k = int((r == 2).sum())
        n_available = int((~((r >= 0) & (r < 2))).sum())
        if k == 0 or k >= n_available or not np.any(r == 1):
            continue
        key = prefix_key(r, include_k=True)
        groups.setdefault(key, []).append(tuple(np.flatnonzero(r == 2)))
    return {key: {'count': len(values), 'branches': len(set(values))}
            for key, values in groups.items() if len(values) >= minimum and len(set(values)) >= 2}


def matched_donors(ranks, times, segments, rows, *, include_k=True,
                   minimum_events=5, minimum_seconds=7200):
    """Diagnostic wrong-time donors, with exact prefix, segment, and optional K."""
    donors = np.full(len(times), -1, dtype=int)
    groups = {}
    for i in rows:
        key = (int(segments[i]), prefix_key(ranks[i], include_k=include_k))
        groups.setdefault(key, []).append(int(i))
    for members in groups.values():
        order = np.array(sorted(members, key=lambda i: times[i]), dtype=int)
        if len(order) < minimum_events:
            continue
        source = np.roll(order, len(order) // 2)
        keep = np.abs(times[order] - times[source]) >= minimum_seconds
        donors[order[keep]] = source[keep]
    return donors


def event_mean(values, mask):
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(values)
    counts = mask.sum(1)
    return np.divide(np.where(mask, values, 0).sum(1), counts,
                     out=np.full(len(values), np.nan), where=counts > 0)


def paired_summary(correct, control, times, segments, *, block_seconds=7200):
    """Descriptive paired events and equal-weight physical bins; no pseudo CI."""
    correct, control = np.asarray(correct), np.asarray(control)
    keep = np.isfinite(correct) & np.isfinite(control)
    if not keep.any():
        return {'status': 'NOT_ESTIMABLE', 'n_events': 0, 'n_blocks': 0}
    delta = control[keep] - correct[keep]
    local = []
    for seg in np.unique(segments[keep]):
        ss = (segments == seg) & keep
        # Absolute physical time bins do not reset at an outcome-dependent event.
        bins = np.floor(times[ss] / block_seconds).astype(np.int64)
        values = control[ss] - correct[ss]
        for b in np.unique(bins):
            m = bins == b
            local.append({'segment': int(seg), 'bin': int(b), 'n_events': int(m.sum()),
                          'gain': float(values[m].mean())})
    gains = np.array([r['gain'] for r in local])
    return {'status': 'ESTIMATED', 'n_events': int(keep.sum()), 'n_blocks': len(local),
            'n_segments': int(len(np.unique(segments[keep]))),
            'event_weighted_gain': float(delta.mean()),
            'block_equal_gain': float(gains.mean()),
            'positive_blocks': int((gains > 1e-6).sum()), 'blocks': local}
