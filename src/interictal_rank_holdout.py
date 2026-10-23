"""Out-of-sample two-template rank reconstruction, conditional on participation.

No model is fitted here. Test sequences choose their closest frozen template;
this is quantization/reconstruction error, not prediction of unseen channels.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

METHODS = ("rank_sequence", "rank_trajectory")


def normalized_order(values, mask):
    """Average ranks of participating entries, scaled by their count minus one."""
    values, mask = np.asarray(values, float), np.asarray(mask, bool)
    if values.shape != mask.shape or values.ndim != 2:
        raise ValueError("expected matching event-by-channel matrices")
    if np.any(mask & ~np.isfinite(values)):
        raise ValueError("nonfinite participating value")
    count = mask.sum(axis=1)
    if np.any(count < 2):
        raise ValueError("at least two participating contacts are needed")
    ordered = rankdata(np.where(mask, values, np.inf), axis=1, method="average")
    return np.where(mask, (ordered - 1) / (count[:, None] - 1), 0.)


def prepare_reconstruction(ranks, bools, templates):
    """Use entire participating sequences supported by all four training templates.

    Unsupported sequences are reported and excluded jointly rather than silently
    evaluating a different channel subset for each model. No direction-QC mask.
    """
    values = np.asarray(ranks, float).T
    mask = np.asarray(bools, bool).T
    if values.shape != mask.shape:
        raise ValueError("rank and participation shapes differ")
    stack = np.asarray([templates[m] for m in METHODS], float)
    if stack.shape != (2, 2, values.shape[1]):
        raise ValueError("expected two templates per method in source channel order")
    supported = np.isfinite(stack).all(axis=(0, 1))
    valid_values = np.all(~mask | np.isfinite(values), axis=1)
    enough = mask.sum(axis=1) >= 2
    supported_events = np.all(~mask | supported[None, :], axis=1)
    eligible = valid_values & enough & supported_events
    eligible_mask = mask[eligible]
    if not eligible.any():
        raise ValueError("no complete held-out sequences supported by both models")
    truth = normalized_order(values[eligible], eligible_mask)
    predictions = np.stack([
        np.stack([normalized_order(np.broadcast_to(t, truth.shape), eligible_mask)
                  for t in stack[i]], axis=1) for i in range(2)
    ])
    coverage = dict(n_test=len(mask), n_evaluated=int(eligible.sum()),
                    n_fewer_than_two=int((~enough).sum()),
                    n_nonfinite_participating=int((~valid_values).sum()),
                    n_unsupported_sequence=int((~supported_events).sum()),
                    n_participating_contacts=int(eligible_mask.sum()))
    return truth, eligible_mask, predictions, eligible, coverage


def closest_template_error(truth, mask, predictions):
    """MAE chooses the nearest template; return the same assignment's MSE too."""
    difference = predictions - truth[None, :, None, :]
    count = mask.sum(axis=1)
    mae = np.sum(np.abs(difference) * mask[None, :, None, :], axis=3) / count[None, :, None]
    mse = np.sum(difference**2 * mask[None, :, None, :], axis=3) / count[None, :, None]
    chosen = np.argmin(mae, axis=2)
    return dict(mae=np.take_along_axis(mae, chosen[..., None], axis=2)[..., 0],
                mse=np.take_along_axis(mse, chosen[..., None], axis=2)[..., 0],
                assignments=chosen)


def shuffle_participating_orders(truth, mask, rng):
    """Uniform independent permutation within each event; preserve missingness/ties."""
    source = np.argsort(np.where(mask, rng.random(mask.shape), np.inf), axis=1, kind="stable")
    position = np.maximum(np.cumsum(mask, axis=1) - 1, 0)
    shuffled = np.take_along_axis(np.take_along_axis(truth, source, axis=1), position, axis=1)
    return np.where(mask, shuffled, 0.)
