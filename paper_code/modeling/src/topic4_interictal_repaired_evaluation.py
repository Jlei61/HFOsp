"""Participation-aware rank features for the fitting objective."""
import numpy as np
from scipy.stats import rankdata
from analysis.src.lagpat_rank_audit import build_masked_kmeans_features

def validate(t):
    t = np.asarray(t, float)
    if t.ndim != 2 or np.isinf(t).any():
        raise ValueError('finite-or-missing events x contacts required')
    return t

def rank_features(t):
    t = validate(t)
    mask = np.isfinite(t)
    ranks = np.zeros_like(t)
    for i, row in enumerate(t):
        ranks[i, mask[i]] = rankdata(row[mask[i]], method='average') - 1
    return build_masked_kmeans_features(ranks.T, mask.T, impute='event_median')
