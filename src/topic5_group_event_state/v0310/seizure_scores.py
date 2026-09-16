"""S-A / S-B / S-C on frozen states (spec section 7.3).

The state score is outcome blind: FIT PCA (<=4 components ordered by FIT
variance) then a shrunk Mahalanobis distance to the FIT centre. No seizure
label selects a dimension, a window or a control.
"""
from __future__ import annotations

import numpy as np
import torch

from ..v039.transition import EventTransition
from .history import build_history
from .seizure import PRE_ICTAL_WINDOW, SECONDARY_WINDOWS, TRAJECTORY, window_profile

MAX_COMPONENTS = 4
SHRINKAGE = 0.9
VARIANCE_FLOOR = 1e-6


def query_states(model, data, times, hours, device='cpu', batch=128):
    """Observer state at arbitrary query times, from the frozen replay blocks."""
    blocks = data['event_replay_blocks']
    built = [build_history(blocks, float(t), hours, data['input_dim']) for t in times]
    steps = max(len(x) for x, _ in built)
    out = []
    with torch.no_grad():
        for start in range(0, len(built), batch):
            chunk = built[start:start + batch]
            x = np.zeros((len(chunk), steps, data['input_dim']), np.float32)
            dt = np.zeros((len(chunk), steps), np.float32)
            for i, (a, b) in enumerate(chunk):
                x[i, :len(a)] = a; dt[i, :len(b)] = b
            out.append(model.scan(torch.from_numpy(x).to(device),
                                  torch.from_numpy(dt).to(device), checkpoint_chunk=0).cpu().numpy())
    events = np.array([x[:, 0].sum() for x, _ in built]) * float(data['normalization']['count_scale'])
    return np.concatenate(out), events


def fit_reference(states_fit):
    centre = states_fit.mean(0)
    centred = states_fit - centre
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    k = int(min(MAX_COMPONENTS, vt.shape[0], (s > 1e-9).sum()))
    if k < 1:
        raise ValueError('FIT states are degenerate; distance is not estimable')
    basis = vt[:k]
    projected = centred @ basis.T
    covariance = np.cov(projected, rowvar=False).reshape(k, k)
    shrunk = SHRINKAGE * covariance + (1 - SHRINKAGE) * np.trace(covariance) / k * np.eye(k)
    shrunk = shrunk + VARIANCE_FLOOR * np.eye(k)
    return dict(centre=centre, basis=basis, precision=np.linalg.inv(shrunk), n_components=k,
                explained_variance_ratio=(s[:k] ** 2 / max((s ** 2).sum(), 1e-12)).tolist())


def mahalanobis(states, reference):
    z = (states - reference['centre']) @ reference['basis'].T
    return np.sqrt(np.einsum('ij,jk,ik->i', z, reference['precision'], z))


def window_mean(values, times, centre, window, eligible):
    mask = (times >= centre + window[0]) & (times < centre + window[1]) & eligible
    return (float(values[mask].mean()) if mask.any() else None), int(mask.sum())


def empirical_quantile(sample, value):
    sample = np.asarray(sample, float)
    if value is None or not sample.size:
        return None
    return float((sample <= value).mean())


def cluster_scores(distances, events, times, eligible, centres, phases, fit_mask):
    """Window-mean distance per centre plus the FIT null of the same statistic."""
    null = []
    for t in times[fit_mask]:
        value, n = window_mean(distances, times, t, PRE_ICTAL_WINDOW, eligible)
        if value is not None and n >= 3:
            null.append(value)
    null = np.asarray(null, float)
    rows = []
    for centre, label in centres:
        entry = dict(centre=float(centre), label=label)
        entry['primary'], entry['n_primary_queries'] = window_mean(distances, times, centre,
                                                                   PRE_ICTAL_WINDOW, eligible)
        for name, window in SECONDARY_WINDOWS.items():
            entry[name], entry['n_' + name] = window_mean(distances, times, centre, window, eligible)
        entry['event_load'], _ = window_mean(events, times, centre, PRE_ICTAL_WINDOW, eligible)
        entry['fit_quantile'] = empirical_quantile(null, entry['primary'])
        rows.append(entry)
    return rows, null


def trajectory_series(distances, times, eligible, centre, step=1800.):
    edges = np.arange(TRAJECTORY[0], TRAJECTORY[1] + step, step)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (times >= centre + lo) & (times < centre + hi) & eligible
        out.append(dict(offset_hours=float(lo / 3600), n=int(mask.sum()),
                        mean_distance=float(distances[mask].mean()) if mask.any() else None))
    return out


def leave_one_cluster_out(case_rows, control_rows):
    """S-B exploratory direction consistency; never a stable-generalisation claim."""
    clusters = sorted({r['cluster_id'] for r in case_rows})
    if len(clusters) < 2:
        return dict(status='NOT_ESTIMABLE', reason='fewer than two eligible clusters', n_clusters=len(clusters))
    folds = []
    for held in clusters:
        train_case = [r['primary'] for r in case_rows if r['cluster_id'] != held and r['primary'] is not None]
        train_ctrl = [r['primary'] for r in control_rows if r['cluster_id'] != held and r['primary'] is not None]
        test_case = [r['primary'] for r in case_rows if r['cluster_id'] == held and r['primary'] is not None]
        test_ctrl = [r['primary'] for r in control_rows if r['cluster_id'] == held and r['primary'] is not None]
        if not train_case or not train_ctrl or not test_case:
            folds.append(dict(held_out_cluster=held, status='NOT_ESTIMABLE')); continue
        direction = float(np.mean(train_case) - np.mean(train_ctrl))
        observed = float(np.mean(test_case) - np.mean(test_ctrl)) if test_ctrl else None
        folds.append(dict(held_out_cluster=held, train_direction=direction, test_difference=observed,
                          consistent=None if observed is None else bool(np.sign(direction) == np.sign(observed)),
                          n_test_controls=len(test_ctrl)))
    usable = [f for f in folds if f.get('consistent') is not None]
    return dict(status='EXPLORATORY' if usable else 'NOT_ESTIMABLE', folds=folds,
                n_consistent=sum(f['consistent'] for f in usable), n_folds=len(usable),
                note='training folds include data recorded after the held-out cluster, so this is '
                     'retrospective cross-cluster transfer, not prospective generalisation')


def forward_in_time(case_rows, control_rows, phases, l2=1.0):
    """S-C: a readout fitted only on earlier clusters, applied to a later one.

    Everything upstream (training, validation selection) already precedes the
    SELECTION partition, so a cluster there is a genuinely forward test as long
    as this readout also sees only earlier clusters. With one or two test
    seizures the numbers below are a description of those predictions, not a
    detection rate.
    """
    train = [(r, 1) for r in case_rows if phases[r['cluster_id']] in ('FIT', 'INNER')]
    train += [(r, 0) for r in control_rows if phases[r['cluster_id']] in ('FIT', 'INNER')]
    test = [(r, 1) for r in case_rows if phases[r['cluster_id']] == 'SELECTION']
    test += [(r, 0) for r in control_rows if phases[r['cluster_id']] == 'SELECTION']
    train = [(r, y) for r, y in train if r['primary'] is not None]
    test = [(r, y) for r, y in test if r['primary'] is not None]
    if not test or sum(y for _, y in train) == 0 or sum(1 - y for _, y in train) == 0:
        return dict(status='NOT_ESTIMABLE',
                    reason='needs at least one case and one control in the earlier clusters and at least '
                           'one window in the later partition',
                    n_train_cases=sum(y for _, y in train), n_train_controls=sum(1 - y for _, y in train),
                    n_test_windows=len(test))
    x = np.array([[r['primary']] for r, _ in train], float)
    y = np.array([v for _, v in train], float)
    centre, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-9, scale, 1.)
    z = (x - centre) / scale
    weight = np.zeros(z.shape[1]); bias = float(np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6)))
    for _ in range(300):                       # fixed budget, fixed L2, no tuning
        p = 1 / (1 + np.exp(-(z @ weight + bias)))
        grad_w = z.T @ (p - y) / len(y) + l2 * weight / len(y)
        grad_b = float((p - y).mean())
        weight -= 0.5 * grad_w; bias -= 0.5 * grad_b
    predictions = []
    for r, label in test:
        value = float(1 / (1 + np.exp(-(((np.array([r['primary']]) - centre) / scale) @ weight + bias))))
        predictions.append(dict(cluster_id=r['cluster_id'], is_case=bool(label), risk=value,
                                brier=float((value - label) ** 2),
                                log_score=float(-(label * np.log(max(value, 1e-9))
                                                  + (1 - label) * np.log(max(1 - value, 1e-9))))))
    cases = [p for p in predictions if p['is_case']]
    controls = [p for p in predictions if not p['is_case']]
    return dict(status='EXPLORATORY', n_train_cases=int(sum(y)), n_train_controls=int(len(y) - sum(y)),
                n_test_cases=len(cases), n_test_controls=len(controls),
                predictions=predictions,
                mean_brier=float(np.mean([p['brier'] for p in predictions])),
                mean_log_score=float(np.mean([p['log_score'] for p in predictions])),
                case_minus_control_risk=(None if not cases or not controls else
                                         float(np.mean([p['risk'] for p in cases])
                                               - np.mean([p['risk'] for p in controls]))),
                readout='logistic on the single outcome-blind distance, fixed L2=1, fixed feature scale, '
                        'fixed 300 gradient steps; no held-out split is spent on tuning',
                limit='one or two forward test seizures cannot give a detection rate or a false-alarm rate '
                      'per hour; these are the individual predictions and their scores')
