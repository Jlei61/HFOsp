"""Held-out directional coverage by two frozen three-dimensional template axes."""
from __future__ import annotations

import numpy as np

from src.dab_gradient_axis import RCOND


def nearest_direction_errors(directions, axes):
    """Select and score in signed 3D direction space; no rank assignment input.

    Directions have shape (events, 3), axes (methods, templates, 3).
    All vectors are unit length. Opposite directions remain 180 degrees apart.
    """
    directions, axes = np.asarray(directions, float), np.asarray(axes, float)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError('expected event-by-3 directions')
    if axes.ndim != 3 or axes.shape[1:] != (2, 3):
        raise ValueError('expected method-by-2-by-3 axes')
    for values in (directions, axes):
        if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=-1), 1, atol=1e-7):
            raise ValueError('finite unit directions required')
    cosines = np.clip(np.einsum('ec,mkc->mek', directions, axes), -1, 1)
    assignments = cosines.argmax(axis=2)
    closest = cosines.max(axis=2)
    return dict(angular_mae_deg=np.degrees(np.arccos(closest)),
                cosine_loss=1-closest, assignments=assignments)


def shuffled_template_axes(templates, coords, n_draws, seed):
    """Shuffle each training template's rank-to-contact map, keeping geometry.

    A shared random key per contact/draw defines a uniform permutation restricted
    to each template's finite, coordinate-mapped support. Identical supports use
    identical permutations across all four templates. Missing supports remain
    missing. Refit the same signed least-squares gradient (rcond=0.05).
    """
    templates, coords = np.asarray(templates, float), np.asarray(coords, float)
    if templates.ndim != 3 or templates.shape[1] != 2 or coords.shape != (templates.shape[2], 3):
        raise ValueError('template/contact-coordinate dimensions do not match')
    supports, values, inverses = [], [], []
    for template in templates.reshape(-1, templates.shape[-1]):
        support = np.flatnonzero(np.isfinite(template) & np.isfinite(coords).all(axis=1))
        if len(support) < 3 or np.std(template[support]) < 1e-12:
            raise ValueError('undefined training template gradient')
        xyz = coords[support]
        supports.append(support)
        values.append(template[support])
        inverses.append(np.linalg.pinv(xyz-xyz.mean(axis=0), rcond=RCOND))
    rng = np.random.default_rng(seed)
    accepted, rejected, proposed = [], 0, 0
    while sum(len(a) for a in accepted) < n_draws:
        need = n_draws-sum(len(a) for a in accepted)
        keys = rng.random((need, len(coords)))
        gradients = []
        for support, rank, inverse in zip(supports, values, inverses):
            permuted = rank[np.argsort(keys[:, support], axis=1)]
            gradients.append((permuted-permuted.mean(axis=1, keepdims=True)) @ inverse.T)
        gradients = np.stack(gradients, axis=1)
        norms = np.linalg.norm(gradients, axis=2)
        valid = np.isfinite(norms).all(axis=1) & (norms >= 1e-12).all(axis=1)
        accepted.append(gradients[valid]/norms[valid, :, None])
        rejected += int((~valid).sum()); proposed += need
        if proposed > max(1000, n_draws*20):
            raise ValueError('too many undefined shuffled template gradients')
    axes = np.concatenate(accepted).reshape(n_draws, templates.shape[0], 2, 3)
    return axes, dict(n_draws=n_draws, seed=int(seed), rejected_degenerate_draws=rejected,
                      support_counts=[len(s) for s in supports],
                      identical_supports=all(np.array_equal(supports[0], s) for s in supports),
                      rcond=RCOND)


def shuffled_direction_errors(directions, null_axes, chunk_size=2048):
    """Mean nearest angle for every null draw/method, using all held-out events."""
    directions, null_axes = np.asarray(directions, float), np.asarray(null_axes, float)
    if len(directions) == 0:
        raise ValueError('no held-out directions')
    total = np.zeros(null_axes.shape[:2])
    for start in range(0, len(directions), chunk_size):
        cosines = np.einsum('ec,dmkc->edmk', directions[start:start+chunk_size], null_axes, optimize=True)
        nearest = np.clip(cosines.max(axis=3), -1, 1)
        total += np.degrees(np.arccos(nearest)).sum(axis=0)
    return (total/len(directions)).T
