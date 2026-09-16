"""Prospective whole-event kernels for patient rank, timing, and spatial patterns.

No axis target or model-selected normalization. Fit scales/maps on training data
before ranking model geometries. This module does not modify the running v1 loss.
"""
import numpy as np
from src.topic4_joint_xy import joint_features


def event_kernel_features(times_ms, xy, groups, time_scale_ms):
    t = np.asarray(times_ms, float)
    if not np.isfinite(time_scale_ms) or time_scale_ms <= 0:
        raise ValueError('positive training time scale required')
    f = joint_features(t, xy, groups); c = t.shape[1]
    mask = np.isfinite(t).astype(float)
    rank = f[:, c:2*c]*4*np.sqrt(c)-1
    r = np.where(mask, 2*rank-1, 0.)/np.sqrt(c)
    first = np.min(np.where(mask, t, np.inf), axis=1, initial=np.inf)
    first = np.where(mask.any(axis=1), first, 0.)
    lag = np.where(mask, (t-first[:, None])/time_scale_ms, 0.)/np.sqrt(c)
    m = mask/np.sqrt(c)
    # f's spatial block retains the frozen contact XY and early/late rank moments.
    spatial = f[:, 3*c:]*2
    return {'support': m,
            'rank_space': np.column_stack([m, r, spatial])/np.sqrt(3),
            'timing_space': np.column_stack([m, lag, spatial])/np.sqrt(3),
            'joint': np.column_stack([m, r, lag, spatial])/2}


def fit_kernel_maps(features, *, seed=2026090605, n_fourier=1024):
    rng = np.random.default_rng(seed); maps = {}
    for name, x in features.items():
        if len(x) < 2 or not np.isfinite(x).all(): raise ValueError('finite training events required')
        a = rng.integers(len(x), size=4096); b = rng.integers(len(x), size=4096)
        distance = np.linalg.norm(x[a]-x[b], axis=1)
        positive = distance[distance > 1e-12]
        bandwidth = float(np.median(positive)) if len(positive) else 1.
        maps[name] = {'bandwidth': bandwidth,
                      'weights': rng.normal(size=(x.shape[1], n_fourier))/bandwidth,
                      'phases': rng.uniform(0, 2*np.pi, size=n_fourier)}
    return maps


def kernel_map(values, specification):
    weights = np.asarray(specification['weights']); phase = np.asarray(specification['phases'])
    out = np.empty((len(values), len(phase)), dtype=np.float32)
    for start in range(0, len(values), 1024):
        out[start:start+1024] = np.sqrt(2/len(phase))*np.cos(values[start:start+1024]@weights+phase)
    return out


def mapped_distance(values, reference_mean):
    if not len(values): return None
    return float(np.sum((values.mean(axis=0, dtype=float)-reference_mean)**2))
