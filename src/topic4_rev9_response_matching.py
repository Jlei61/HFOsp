"""Pure scoring helpers for exploratory rev9 Node-to-Edge response matching."""
from __future__ import annotations

import numpy as np


def robust_scale(values, floor=1e-6):
    """MAD scale with a standard-deviation fallback for discrete responses."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float(floor)
    median = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - median))
    if not np.isfinite(scale) or scale <= floor:
        scale = np.std(values)
    return float(max(scale, floor))


def pseudo_huber_squared(z):
    """Quadratic near zero and linear for large standardized differences."""
    z = np.asarray(z, float)
    return 2.0 * (np.sqrt(1.0 + z ** 2) - 1.0)


def positive_map_js_distance(left, right):
    """Square-root JS distance normalized to [0, 1] for non-negative maps."""
    left = np.clip(np.asarray(left, float).ravel(), 0.0, np.inf)
    right = np.clip(np.asarray(right, float).ravel(), 0.0, np.inf)
    if left.shape != right.shape or np.any(~np.isfinite(left + right)):
        return None
    left_total, right_total = float(left.sum()), float(right.sum())
    if left_total == 0.0 and right_total == 0.0:
        return 0.0
    if left_total == 0.0 or right_total == 0.0:
        return 1.0
    left, right = left / left_total, right / right_total
    midpoint = 0.5 * (left + right)
    kl_left = np.sum(left[left > 0.0] * np.log(
        left[left > 0.0] / midpoint[left > 0.0]))
    kl_right = np.sum(right[right > 0.0] * np.log(
        right[right > 0.0] / midpoint[right > 0.0]))
    return float(np.sqrt((0.5 * kl_left + 0.5 * kl_right) / np.log(2.0)))


def scalar_pair_loss(node_values, edge_values, scales):
    """Mean pseudo-Huber loss across available scalar response features."""
    node_values = np.asarray(node_values, float)
    edge_values = np.asarray(edge_values, float)
    scales = np.asarray(scales, float)
    valid = (np.isfinite(node_values) & np.isfinite(edge_values)
             & np.isfinite(scales) & (scales > 0.0))
    if not valid.any():
        return None, 0
    losses = pseudo_huber_squared(
        (edge_values[valid] - node_values[valid]) / scales[valid])
    return float(np.mean(losses)), int(valid.sum())
