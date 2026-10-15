"""Joint 15-contact participation-mask distribution term (design 2026-09-10 §6; checklist C11).

Mixture kernel over binary masks k(b,b') = mean_h exp(-Hamming(b,b')/h), h in {1,3,6}.
D_mask_off = model self term without the diagonal - 2 * model-patient cross term + fixed
patient self term. The patient target is the full FIT natural mask frequency; the positive
scale a_mask is the median of the non-negative (diagonal-included) matched-count distance
over block-contiguous FIT draws. Negative D_mask_off values are retained.
"""
from __future__ import annotations
import numpy as np

BANDWIDTHS = (1., 3., 6.)
MINIMUM_EVENTS = 16


def participation_masks(times):
    t = np.asarray(times, float)
    if t.ndim != 2:
        raise ValueError('events x contacts required')
    return np.isfinite(t)


def _hamming(a, b):
    a = np.asarray(a, bool).astype(np.int32); b = np.asarray(b, bool).astype(np.int32)
    return a @ (1 - b).T + (1 - a) @ b.T


def mask_kernel(a, b, bandwidths=BANDWIDTHS):
    h = _hamming(a, b).astype(float)
    return np.mean([np.exp(-h / w) for w in bandwidths], axis=0)


def _patterns(masks):
    masks = np.asarray(masks, bool)
    codes = masks.astype(np.int64) @ (1 << np.arange(masks.shape[1], dtype=np.int64))
    unique, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
    return masks[np.unique(inverse, return_index=True)[1]], counts


class MaskReference:
    """Fixed patient target: unique mask patterns, their natural frequencies and the self term."""

    def __init__(self, patient_masks, bandwidths=BANDWIDTHS):
        masks = np.asarray(patient_masks, bool)
        if masks.ndim != 2 or len(masks) < 2:
            raise ValueError('patient masks (events x contacts) with at least two events required')
        self.n_contacts = masks.shape[1]; self.n_events = int(len(masks)); self.bandwidths = tuple(bandwidths)
        self.patterns, counts = _patterns(masks)
        self.weights = counts / counts.sum()
        K = mask_kernel(self.patterns, self.patterns, self.bandwidths)
        n = self.n_events
        total = n * n * float(self.weights @ K @ self.weights)
        self.self_biased = total / (n * n)
        self.self_off = (total - n) / (n * (n - 1))            # K(x,x)=1 on the diagonal
        self.contact_frequency = masks.mean(0)
        self.a_mask = None; self.calibration = None

    def cross(self, masks):
        return mask_kernel(np.asarray(masks, bool), self.patterns, self.bandwidths) @ self.weights


def off_diagonal_mask_distance(masks, reference):
    m = np.asarray(masks, bool)
    if m.ndim != 2 or m.shape[1] != reference.n_contacts:
        raise ValueError('masks must be events x patient contacts')
    n = len(m)
    if n < 2:
        return None
    K = mask_kernel(m, m, reference.bandwidths)
    self_off = (K.sum() - np.trace(K)) / (n * (n - 1))
    return float(self_off - 2. * reference.cross(m).mean() + reference.self_off)


def biased_mask_distance(masks, reference):
    m = np.asarray(masks, bool); n = len(m)
    if n < 1:
        return None
    K = mask_kernel(m, m, reference.bandwidths)
    return float(K.mean() - 2. * reference.cross(m).mean() + reference.self_biased)


def calibrate_scale(patient_masks, blocks, reference, *, sample_count=MINIMUM_EVENTS, n_samples=128, seed=20260910):
    """a_mask: median non-negative matched-count distance over block-contiguous draws (design §6)."""
    masks = np.asarray(patient_masks, bool); blocks = np.asarray(blocks)
    if len(masks) != len(blocks):
        raise ValueError('one block id per patient event required')
    rng = np.random.default_rng(int(seed)); unique = np.unique(blocks); draws = []; values = []
    for _ in range(int(n_samples)):
        chosen, pool = [], []
        for block in rng.permutation(unique):
            pool.append(np.flatnonzero(blocks == block)); chosen.append(int(block))
            if sum(len(x) for x in pool) >= sample_count:
                break
        index = np.concatenate(pool)
        if len(index) < sample_count:
            raise RuntimeError('patient partition too small for the matched count')
        pick = rng.choice(index, sample_count, replace=False)
        value = biased_mask_distance(masks[pick], reference)
        values.append(value); draws.append(dict(blocks=chosen, n_events_available=int(len(index)), value=value))
    a_mask = float(np.median(values))
    if not np.isfinite(a_mask) or a_mask <= 0:
        raise RuntimeError('patient mask scale is not positive')
    return dict(a_mask=a_mask, statistic=f'biased_non_negative_matched_{sample_count}', sample_count=int(sample_count),
                n_samples=int(n_samples), seed=int(seed), draws=draws, median=a_mask,
                q05=float(np.quantile(values, .05)), q95=float(np.quantile(values, .95)))


def score_masks(masks, reference, minimum_events=MINIMUM_EVENTS):
    m = np.asarray(masks, bool); n = len(m)
    if n < minimum_events:
        return dict(status='INSUFFICIENT_EVENTS', n_events=n, minimum_events=minimum_events, D_mask_off=None, D_mask_biased=None,
                    D_mask_off_scaled=None)
    d = off_diagonal_mask_distance(m, reference); b = biased_mask_distance(m, reference)
    return dict(status='ESTIMABLE', n_events=n, minimum_events=minimum_events, D_mask_off=d, D_mask_biased=b,
                D_mask_off_scaled=None if reference.a_mask is None else d / reference.a_mask,
                contact_frequency=m.mean(0).tolist(), both_rods=None)


def score_times(times, reference, minimum_events=MINIMUM_EVENTS):
    return score_masks(participation_masks(times), reference, minimum_events)


def combined_search_loss(loss_off, d_mask_off, a_mask):
    """L_search = 0.5 L_off + 0.5 D_mask_off / a_mask (design weights, not a recovery rate)."""
    if loss_off is None or d_mask_off is None or a_mask is None or a_mask <= 0:
        return None
    return float(.5 * loss_off + .5 * d_mask_off / a_mask)
