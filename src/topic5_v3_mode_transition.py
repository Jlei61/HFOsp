"""Topic 5 V3a mode-transition helpers.

This module intentionally stays on pure configuration for Task 0. Later
tasks add event-window extraction, geometry, dynamics, and avalanche-flux
estimators on top of this config contract. See
docs/superpowers/plans/2026-07-02-topic5-v3a-mode-transition.md for the
full task list; treat this line as exploratory pending the pilot-lock gate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import yaml
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CFG = _ROOT / "config" / "topic5_v3.yaml"


def load_v3_config(path: str | Path | None = None) -> dict:
    """Load the V3a mode-transition YAML config as a plain dict."""
    cfg_path = Path(path) if path is not None else _DEFAULT_CFG
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, Mapping):
        raise ValueError(f"V3a config must be a mapping: {cfg_path}")
    return dict(cfg)


def _window_index_range(relt: np.ndarray, lo: float, hi: float) -> tuple[int, int] | None:
    """Half-open ``(start, stop)`` sample indices where ``lo <= relt <= hi``.

    ``relt`` is monotone increasing, so the mask is contiguous. Returns
    ``None`` if the window is empty. Local copy of the pattern in
    ``scripts/_topic5_v2_crit_io.py::window_index_range``.
    """
    relt = np.asarray(relt, dtype=float)
    mask = (relt >= float(lo)) & (relt <= float(hi))
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)
    return int(idx[0]), int(idx[-1] + 1)


def i1_range(
    eeg_onset_rel: float, eeg_offset_rel: float, duration: float, cfg: dict
) -> tuple[float, float, bool]:
    """Early-ictal I1 window relative to eeg onset.

    Primary (``duration >= I1_min_duration_sec``): ``[onset+I1_rel[0],
    onset+I1_rel[1]]``. Short-seizure fallback: ``[onset+I1_rel[0],
    offset - I1_post_guard_sec]`` — offset-based, never ``0.25*duration``
    (plan rev2). ``i1_eligible`` requires at least one full ``window_sec``.
    """
    ph = cfg["phases"]
    onset = float(eeg_onset_rel)
    offset = float(eeg_offset_rel)
    dur = float(duration)
    lo = onset + ph["I1_rel"][0]
    if dur >= ph["I1_min_duration_sec"]:
        hi = onset + ph["I1_rel"][1]
    else:
        hi = offset - ph["I1_post_guard_sec"]
    i1_eligible = bool((hi - lo) >= ph["window_sec"])
    return lo, hi, i1_eligible


def phase_bin_range(
    relt: np.ndarray,
    eeg_onset_rel: float,
    eeg_offset_rel: float,
    duration: float,
    phase: str,
    cfg: dict,
    onset_shift: float = 0.0,
) -> tuple[int, int] | None:
    """Half-open sample-index range for one named phase bin.

    Anchored on ``eeg_onset_rel + onset_shift`` (onset jitter perturbs the
    anchor) for P0..O and I1; I2/I3 are ictal-fraction of ``[anchor,
    offset]`` (offset itself does not shift); Post is relative to
    ``eeg_offset_rel`` only and is never shifted by onset jitter.
    """
    ph = cfg["phases"]
    anchor = float(eeg_onset_rel) + float(onset_shift)
    offset = float(eeg_offset_rel)

    if phase == "P0":
        lo, hi = anchor - 120.0, anchor - 90.0
    elif phase == "P1":
        lo, hi = anchor - 90.0, anchor - 60.0
    elif phase == "P2":
        lo, hi = anchor - 60.0, anchor - 30.0
    elif phase == "P3":
        lo, hi = anchor + ph["P3_rel"][0], anchor + ph["P3_rel"][1]
    elif phase == "O":
        lo, hi = anchor + ph["O_rel"][0], anchor + ph["O_rel"][1]
    elif phase == "I1":
        lo, hi, _ = i1_range(anchor, offset, duration, cfg)
    elif phase == "I2":
        lo, hi = anchor + 0.25 * (offset - anchor), anchor + 0.75 * (offset - anchor)
    elif phase == "I3":
        lo, hi = anchor + 0.75 * (offset - anchor), offset
    elif phase == "Post":
        lo, hi = offset, offset + ph["span_post_sec"]
    else:
        raise ValueError(f"unknown phase: {phase!r}")

    return _window_index_range(relt, lo, hi)


def sliding_windows(
    relt: np.ndarray, start: int, stop: int, window_sec: float, step_sec: float
) -> list[tuple[int, int]]:
    """Sliding ``(window_start_idx, window_end_idx)`` half-open pairs over ``[start, stop)``.

    Samples-per-second is derived from the median spacing of ``relt``. Only
    full-length windows are emitted: a window is kept only if it spans the
    complete ``window_sec`` within ``[start, stop)``; the partial trailing
    tail is dropped rather than clipped to ``stop``. The ``>= 3``-sample
    guard is kept as a defensive floor (subsumed for realistic configs, but
    still checked).
    """
    relt = np.asarray(relt, dtype=float)
    dt = float(np.median(np.diff(relt)))
    window_n = int(round(window_sec / dt))
    step_n = int(round(step_sec / dt))
    windows: list[tuple[int, int]] = []
    ws = start
    while ws + window_n <= stop:
        we = ws + window_n
        if we - ws >= 3:
            windows.append((ws, we))
        ws += step_n
    return windows


def rank_forward(ta_rank: dict) -> dict:
    """Rescale interictal ``typical_rank`` to a signed forward-order axis.

    Only names with a finite ``typical_rank`` are kept (non-participating
    contacts carry non-finite values upstream and are dropped, not
    remapped). Finite values are linearly rescaled to ``[-1, +1]``
    (earliest rank -> -1, latest rank -> +1); if every finite value ties
    (``rmax == rmin``) every name maps to ``0.0`` instead of dividing by
    zero.
    """
    finite = {name: float(r) for name, r in ta_rank.items() if np.isfinite(r)}
    if not finite:
        return {}
    values = np.array(list(finite.values()))
    rmin, rmax = float(values.min()), float(values.max())
    if rmax == rmin:
        return {name: 0.0 for name in finite}
    return {name: 2.0 * (r - rmin) / (rmax - rmin) - 1.0 for name, r in finite.items()}


def beta_axis(metric_by_name: dict, rank_forward: dict) -> float:
    """Signed Spearman correlation between ``metric_by_name`` and ``rank_forward``.

    Restricted to names present in both dicts with finite values in both —
    these are the axis contacts. Returns ``nan`` if fewer than 4 valid pairs
    (Spearman on <4 points is not meaningful).
    """
    xs: list[float] = []
    ys: list[float] = []
    for name, rank_val in rank_forward.items():
        if name not in metric_by_name:
            continue
        metric_val = metric_by_name[name]
        if np.isfinite(metric_val) and np.isfinite(rank_val):
            xs.append(float(metric_val))
            ys.append(float(rank_val))
    if len(xs) < 4:
        return float("nan")
    r, _ = spearmanr(xs, ys)
    return float(r)


def classify_contacts(
    all_clean: list, axis_template_names: list, hfo_participation: dict, thresh: float
) -> dict:
    """Partition ``all_clean`` into three disjoint classes.

    ``axis_template_names`` is membership only — the caller has already
    decided which names count as axis (finite typical_rank OR in
    axis_partition source/mid/end); this does not recompute that. The
    remaining clean contacts split on ``hfo_participation`` vs ``thresh``:
    below is non-axis-strict (feeds ``P_N``), at/above is ambiguous_hfo
    (stays in the all-clean VAR state ``X`` but is excluded from both
    ``P_A`` and ``P_N``).
    """
    axis_set = set(axis_template_names)
    is_axis: list = []
    is_nonaxis_strict: list = []
    is_ambiguous_hfo: list = []
    for name in all_clean:
        if name in axis_set:
            is_axis.append(name)
        elif hfo_participation.get(name, 0.0) >= thresh:
            is_ambiguous_hfo.append(name)
        else:
            is_nonaxis_strict.append(name)
    is_axis.sort()
    is_nonaxis_strict.sort()
    is_ambiguous_hfo.sort()
    return {
        "is_axis": is_axis,
        "is_nonaxis_strict": is_nonaxis_strict,
        "is_ambiguous_hfo": is_ambiguous_hfo,
        "n_axis": len(is_axis),
        "n_nonaxis": len(is_nonaxis_strict),
        "n_ambiguous": len(is_ambiguous_hfo),
    }


def subspace_projectors(
    names: list, axis_names: list, nonaxis_names: list
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonal 0/1 projection matrices onto the axis / non-axis-strict subspaces.

    Ambiguous contacts (in ``names`` but in neither ``axis_names`` nor
    ``nonaxis_names``) are 0 in both ``P_A`` and ``P_N``.
    """
    axis_set = set(axis_names)
    nonaxis_set = set(nonaxis_names)
    a_diag = np.array([1.0 if name in axis_set else 0.0 for name in names])
    n_diag = np.array([1.0 if name in nonaxis_set else 0.0 for name in names])
    return np.diag(a_diag), np.diag(n_diag)


def axis_nonaxis_vectors(
    names: list, rank_forward: dict, axis_names: list, nonaxis_names: list
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three length-``len(names)`` vectors ordered by ``names``.

    ``e_axis_mean``/``e_nonaxis_mean`` are UNIFORM unit indicators over
    their respective positions (not participation-weighted); ambiguous
    positions are 0 in both. ``e_nonaxis_mean`` is then Gram-Schmidt
    orthogonalized against ``e_axis_mean`` and renormalized — a no-op in
    value since axis/non-axis are disjoint (already orthogonal), but keeps
    the construction explicit for future non-disjoint use. ``e_axis_grad``
    weights axis contacts by ``rank_forward`` (0 elsewhere) and
    L2-normalizes; an all-zero weighting (e.g. all-zero ``rank_forward``,
    or no axis contacts) returns the zero vector rather than dividing by
    zero.
    """
    axis_set = set(axis_names)
    nonaxis_set = set(nonaxis_names)
    is_axis = np.array([name in axis_set for name in names])
    is_nonaxis = np.array([name in nonaxis_set for name in names])

    n_axis = int(is_axis.sum())
    e_axis_mean = is_axis.astype(float)
    if n_axis > 0:
        e_axis_mean = e_axis_mean / np.sqrt(n_axis)

    n_nonaxis = int(is_nonaxis.sum())
    e_nonaxis_mean = is_nonaxis.astype(float)
    if n_nonaxis > 0:
        e_nonaxis_mean = e_nonaxis_mean / np.sqrt(n_nonaxis)
    e_nonaxis_mean = e_nonaxis_mean - (e_nonaxis_mean @ e_axis_mean) * e_axis_mean
    norm_nonaxis = np.linalg.norm(e_nonaxis_mean)
    if norm_nonaxis > 0:
        e_nonaxis_mean = e_nonaxis_mean / norm_nonaxis

    grad_weights = np.array(
        [rank_forward.get(name, 0.0) if is_axis[i] else 0.0 for i, name in enumerate(names)]
    )
    norm_grad = np.linalg.norm(grad_weights)
    e_axis_grad = grad_weights / norm_grad if norm_grad > 0 else np.zeros(len(names))

    return e_axis_mean, e_axis_grad, e_nonaxis_mean


def geometry_sufficient(
    n_axis: int, n_nonaxis: int, shafts_with_both: int, cfg: dict
) -> tuple[bool, str]:
    """Whether axis/non-axis geometry is sufficient for downstream VAR/subspace work.

    Checks ``n_axis``/``n_nonaxis`` against ``cfg["geometry"]["min_n_axis"]``/
    ``min_n_nonaxis`` and requires at least one shaft carrying both classes
    (so the non-axis subspace isn't confounded with shaft identity). Returns
    a human-readable ``reason`` for the feasibility CSV: ``"ok"`` when
    sufficient, else the first failing condition.
    """
    geo = cfg["geometry"]
    min_axis = geo["min_n_axis"]
    min_nonaxis = geo["min_n_nonaxis"]
    if n_axis < min_axis:
        return False, f"n_axis<{min_axis}"
    if n_nonaxis < min_nonaxis:
        return False, f"n_nonaxis<{min_nonaxis}"
    if shafts_with_both < 1:
        return False, "no_shaft_with_both"
    return True, "ok"


def _coerce_rng(rng) -> np.random.Generator:
    """Accept an ``np.random.Generator`` or an int seed; return a ``Generator``."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def shaft_constrained_permute(
    values_by_name: dict, shaft_by_name: dict, rng
) -> dict:
    """Shaft-spatial null: shuffle values, not positions, within each shaft.

    Groups ``values_by_name`` keys by ``shaft_by_name[name]``; within each
    shaft, the multiset of values is randomly reassigned across that
    shaft's own contact names. A contact never receives a value from a
    different shaft, and single-contact shafts are unchanged (nothing to
    permute against).
    """
    rng = _coerce_rng(rng)
    shafts: dict[str, list] = {}
    for name in values_by_name:
        shafts.setdefault(shaft_by_name[name], []).append(name)

    out: dict = {}
    for names in shafts.values():
        values = [values_by_name[name] for name in names]
        order = rng.permutation(len(values))
        for name, idx in zip(names, order):
            out[name] = values[idx]
    return out


def rate_preserving_shuffle(active_bool: np.ndarray, rng) -> np.ndarray:
    """Rate-preserving null: permute each contact's time bins independently.

    ``active_bool`` has shape ``(n_contacts, n_time)``. Each row is
    reordered along the time axis using its own independent random
    permutation, so each row's activation count (per-contact rate) is
    preserved exactly while cross-contact temporal alignment is destroyed.
    Returns a new array; ``active_bool`` is never written to.
    """
    rng = _coerce_rng(rng)
    arr = np.asarray(active_bool, dtype=bool)
    out = np.empty_like(arr)
    n_time = arr.shape[1]
    for i in range(arr.shape[0]):
        out[i] = arr[i, rng.permutation(n_time)]
    return out


def label_permute(
    axis_names: list, nonaxis_names: list, shaft_by_name: dict, rng
) -> tuple[list, list]:
    """Axis/non-axis label null: shuffle labels within each shaft.

    Restricted to the union of ``axis_names`` and ``nonaxis_names``. Groups
    those names by ``shaft_by_name[name]``; within each shaft, the
    axis/non-axis label vector is randomly permuted across that shaft's own
    names, which preserves that shaft's axis count exactly (and therefore
    the global axis/non-axis counts, since they are just the sum over
    shafts). All-axis or all-non-axis shafts are unaffected because
    permuting a uniform label vector is a no-op — no swap is possible.
    """
    rng = _coerce_rng(rng)
    axis_set = set(axis_names)
    shafts: dict[str, list] = {}
    for name in list(axis_names) + list(nonaxis_names):
        shafts.setdefault(shaft_by_name[name], []).append(name)

    new_axis: list = []
    new_nonaxis: list = []
    for names in shafts.values():
        labels = [name in axis_set for name in names]
        order = rng.permutation(len(labels))
        for name, idx in zip(names, order):
            (new_axis if labels[idx] else new_nonaxis).append(name)
    new_axis.sort()
    new_nonaxis.sort()
    return new_axis, new_nonaxis
