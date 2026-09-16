"""Observable group events and joint rank/space fitting, without an axis target."""
import numpy as np


def observable_groups(envelope, dt_ms, *, window_ms=250., margin=.1, extension_ms=30.,
                      channel_fraction=.5, timing='centroid', burnin_ms=500.):
    """Apply the shared patient grouping algorithm to a firing-density proxy.

    This is not a HFO or LFP forward model. The inherited global amplitude bar
    defines local detections; grouping/padding/non-overlap reuse the legacy port.
    No family/mode/axis label influences selection. All grouped rows are kept.
    """
    from src.group_event_analysis import build_windows_from_detections
    env = np.asarray(envelope, float)
    if env.ndim != 2 or not env.size or not np.isfinite(env).all() or dt_ms <= 0:
        raise ValueError('invalid envelope')
    if timing not in ('centroid', 'half_peak'):
        raise ValueError('unknown timing')
    stable = env[:, int(round(burnin_ms/dt_ms)):]
    if stable.size == 0: raise ValueError('no post-burnin observation')
    floor = stable.min(); bar = floor + margin * (stable.max() - floor)
    detected = env > bar
    detections = {}
    for c, row in enumerate(detected):
        edges = np.diff(np.r_[False, row, False].astype(int))
        detections[str(c)] = np.column_stack([np.flatnonzero(edges == 1),
                                              np.flatnonzero(edges == -1)]) * dt_ms / 1000.
    # The general port omits empty channels from its denominator. Keep the
    # frozen montage denominator, as legacy's explicit chns_num does: otherwise
    # a single active contact would qualify as a 'half-montage' group.
    nonempty = sum(len(v) > 0 for v in detections.values())
    effective_fraction = channel_fraction * len(env) / max(nonempty, 1)
    groups = [] if effective_fraction > 1 or nonempty == 0 else build_windows_from_detections(
        detections, window_sec=window_ms/1000., ext_ms=extension_ms,
        chns_thr=effective_fraction, time_axis_hz=1000./dt_ms,
        t_max_sec=env.shape[1]*dt_ms/1000.)
    complete = [w for w in groups if w.start*1000. >= burnin_ms and w.end*1000. <= env.shape[1]*dt_ms]
    out = np.full((len(complete), len(env)), np.nan)
    for i, w in enumerate(complete):
        a, b = int(round(w.start*1000./dt_ms)), int(round(w.end*1000./dt_ms))
        for c in np.flatnonzero(detected[:, a:b].any(axis=1)):
            segment = env[c, a:b]
            if timing == 'centroid':
                weight = np.maximum(segment, 0.)
                if weight.sum() > 0:
                    out[i, c] = np.dot(np.arange(a, b)*dt_ms, weight) / weight.sum()
            else:
                out[i, c] = (a + np.flatnonzero(segment >= .5*segment.max())[0])*dt_ms
    return out, {'n_groups': len(out), 'n_boundary_censored_groups': len(groups)-len(complete),
                 'n_local_detection_intervals': sum(len(v) for v in detections.values()),
                 'threshold': float(bar), 'timing': timing, 'fixed_montage_denominator': len(env),
                 'windows_ms': [[w.start*1000., w.end*1000.] for w in complete],
                 'physical_claim': 'firing_density_proxy_not_HFO_or_LFP'}


def joint_features(times_ms, xy, groups, lag_cap_ms=180.):
    """All contacts, masks, within-event ranks, lags, and spatial rank moments.

    Four blocks get equal Euclidean scaling. No fitted PCA, classifier, K=2
    condition, core angle, or directional target is used. Missingness is explicit.
    """
    t = np.asarray(times_ms, float); xy = np.asarray(xy, float)
    if t.ndim != 2 or xy.shape != (t.shape[1], 2) or np.isinf(t).any():
        raise ValueError('contact/time shape or values invalid')
    if not np.isfinite(xy).all() or lag_cap_ms <= 0:
        raise ValueError('invalid spatial contract')
    mask = np.isfinite(t); counts = mask.sum(axis=1)
    # Average ranks among PARTICIPATING contacts only; no phantom ranks.
    valid_pair = mask[:, :, None] & mask[:, None, :]
    delta = t[:, :, None] - t[:, None, :]
    lower = ((delta > 1e-9) & valid_pair).sum(axis=2)
    equal = ((np.abs(delta) <= 1e-9) & valid_pair).sum(axis=2)
    rank = np.where(mask, (lower + .5*(equal-1))/np.maximum(counts-1, 1)[:, None], 0.)
    first = np.min(np.where(mask, t, np.inf), axis=1, initial=np.inf)
    first = np.where(counts, first, 0.)
    lag = np.where(mask, np.clip((t-first[:, None])/lag_cap_ms, 0, 1), 0.)
    positions = xy/20.
    moments = []
    for weights in (mask.astype(float), mask*(1-rank), mask*rank):
        moments.append(weights @ positions / np.maximum(weights.sum(axis=1), 1e-12)[:, None])
    spatial = np.column_stack(moments + [mask[:, groups[g]].mean(axis=1) for g in ('ICL', 'SCL')])
    blocks = [mask.astype(float), np.where(mask, (1+rank)/2., 0.),
              np.where(mask, (1+lag)/2., 0.), spatial]
    return np.column_stack([b / np.sqrt(b.shape[1]) for b in blocks]) / 2.


def projections(n_features, seed=2026090601, count=128):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_features, count))
    return x / np.linalg.norm(x, axis=0)


def projected_quantiles(features, axes, count=256):
    x = np.asarray(features)
    if x.ndim != 2 or len(x) == 0: return None
    return np.quantile(x @ axes, (np.arange(count)+.5)/count, axis=0)


def joint_distance(features, axes, reference_quantiles):
    q = projected_quantiles(features, axes, len(reference_quantiles))
    return None if q is None else float(np.abs(q-reference_quantiles).mean())


def proposal_rng(master_seed, round_number, stream=0):
    return np.random.default_rng(np.random.SeedSequence([master_seed, round_number, stream]))


def round_action(history, best, support_ratio, n_events, *, tolerance=.01):
    """Bounded evidence-based adaptation; never silently rewrite the loss."""
    if n_events < 16:
        return {'diagnosis': 'LOW_OBSERVABLE_EVENT_YIELD', 'action': 'longer_paired_screen',
                'explanation': 'Too few observed groups; no conclusion on geometry capacity.'}
    if len(history) >= 3 and min(history[-3:]) - best <= tolerance * max(min(history[-3:]), 1e-6):
        return {'diagnosis': 'SEARCH_PLATEAU_CAPACITY_VS_OPTIMIZATION_UNRESOLVED',
                'action': 'increase_random_restart_fraction',
                'explanation': 'Broaden independent starts; current evidence does not distinguish loss inadequacy from model capacity.'}
    return {'diagnosis': 'JOINT_DISTRIBUTION_MISMATCH' if support_ratio > 1 else 'CONTINUE_JOINT_REFINEMENT',
            'action': 'multi_anchor_local_plus_random',
            'explanation': 'Continue finite joint loss; conditional support never blocks exploratory proposals.'}
