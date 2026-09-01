"""Early ictal field: per-contact recruitment descriptors for H2b B2.

The field itself is baseline-robust-z band power from the already-validated
Topic 5 primitives (``src/topic5_ictal_recruitment.band_power_trace`` +
``baseline_robust_z``), sliced with the existing
``src/topic5_t0_features.window_activation``. Only the descriptors the H2b spec
§1 names on top of the field live here:

    first recruited group / early propagation path -> :func:`first_crossing_time`
    spatial entropy / extent                       -> :func:`spatial_entropy`
    laterality                                     -> :func:`laterality_index`

Anchoring note (topic5 caveat 9): every time axis here is relative to the
**EEG onset**, not the clinical onset and not the cache's ``relt=0``.
"""

from __future__ import annotations

import numpy as np


def first_crossing_time(
    z_trace: np.ndarray,
    rel_t: np.ndarray,
    threshold: float,
    t0: float,
    t1: float,
) -> np.ndarray:
    """Per-contact time of first threshold crossing inside ``[t0, t1]``.

    Crossings outside the window are ignored, so pre-onset activity can never be
    read as early ictal recruitment. A contact that never crosses gets ``NaN``
    rather than the window end -- "recruited last" and "never recruited" are
    different statements.
    """

    z = np.asarray(z_trace, float)
    t = np.asarray(rel_t, float)
    inside = (t >= t0) & (t <= t1)
    out = np.full(z.shape[0], np.nan)
    if not inside.any():
        return out
    idx = np.flatnonzero(inside)
    sub = z[:, idx]
    hit = np.isfinite(sub) & (sub >= threshold)
    any_hit = hit.any(axis=1)
    first = np.argmax(hit, axis=1)
    out[any_hit] = t[idx][first[any_hit]]
    return out


def normalize_field(field: np.ndarray) -> np.ndarray:
    """Positive part of the field, normalised to sum to one.

    An all-suppressed field returns all-``NaN``: it carries no recruitment mass,
    and returning a uniform distribution would assert spatial spread that the
    data does not show.
    """

    x = np.asarray(field, float)
    pos = np.where(np.isfinite(x) & (x > 0.0), x, 0.0)
    total = pos.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.full(x.shape, np.nan)
    return pos / total


def spatial_entropy(field: np.ndarray) -> float:
    """Shannon entropy of the normalised field, scaled to ``[0, 1]``.

    Contacts without coverage (``NaN``) are dropped from both the distribution
    and the normalising ``log n``, so extent is never diluted by absent channels.
    """

    x = np.asarray(field, float)
    keep = np.isfinite(x)
    n = int(keep.sum())
    if n <= 1:
        return float("nan")
    p = normalize_field(x[keep])
    if not np.all(np.isfinite(p)):
        return float("nan")
    nz = p[p > 0.0]
    h = -np.sum(nz * np.log(nz))
    return float(h / np.log(n))


def laterality_index(field: np.ndarray, hemisphere: np.ndarray) -> float:
    """``(left - right) / (left + right)`` on the positive part of the field.

    ``hemisphere`` is ``-1`` (left), ``+1`` (right) or ``0`` (unmapped);
    unmapped contacts are excluded and an entirely unmapped montage yields
    ``NaN`` rather than a fabricated zero.
    """

    x = np.asarray(field, float)
    h = np.asarray(hemisphere, int)
    keep = np.isfinite(x) & (h != 0)
    if not keep.any():
        return float("nan")
    pos = np.where(x[keep] > 0.0, x[keep], 0.0)
    left = pos[h[keep] < 0].sum()
    right = pos[h[keep] > 0].sum()
    total = left + right
    if total <= 0.0:
        return float("nan")
    return float((left - right) / total)


def save_npz_atomic(path, arrays: dict) -> None:
    """Write an ``.npz`` and rename it into place in one step.

    ``np.savez`` appends ``.npz`` to a *filename* argument, so the obvious
    "write foo.npz.tmp then rename" scheme silently produces ``foo.npz.tmp.npz``
    and the rename fails. Passing an open handle suppresses that behaviour, so
    the temp file really is the path we later rename.
    """

    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays)
    tmp.replace(path)
