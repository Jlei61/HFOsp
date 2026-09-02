"""Explicit history baseline ``log mu_H`` and endpoint eligibility (design §8).

Two providers, never mixed silently:

* ``agent2_registry`` -- the evaluation agent's ``H_strong`` registry.  This
  module only *reads and aligns* it; it does not redefine ``H``.
* ``provisional_local`` -- a stop-gap fitted from the v0.2 ``B_multiscale``
  features (seizure columns removed) with the reviewed v0.3 ridge recipe.  It
  exists so unit tests, synthetic assays and pathway diagnostics can run before
  the registry appears.  Every artefact built on it carries ``h_source``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.topic5_group_event_state.v03.evaluate import _eligible_baseline_columns, _fit_count_ridge
from src.topic5_group_event_state.v03.partition import PHASE_NAMES

from .readout import fit_nb_log_dispersion

ANCHOR_TIME_TOLERANCE_SECONDS = 1e-3
LOG_MU_KEYS = ("log_mu_h", "log_mu_H", "log_mu")
ANCHOR_KEYS = ("anchor_time", "t_anchor", "anchor_epoch")


@dataclass
class HistoryBaseline:
    log_mu: dict[int, np.ndarray]
    nb_log_dispersion: dict[int, float | None]
    source: str
    meta: dict[str, Any] = field(default_factory=dict)

    def has_horizon(self, horizon: float) -> bool:
        return int(horizon) in self.log_mu


def _phase_anchor_mask(timeline, partition, phase: str, horizon: float) -> np.ndarray:
    grid = timeline.grid
    h_i = list(timeline.config.horizons_seconds).index(float(horizon))
    labels = partition.labels_of(grid.t_anchor)
    _lo, hi = partition.bounds(phase)
    return (
        (labels == PHASE_NAMES.index(phase))
        & np.asarray(grid.eligible[:, h_i], dtype=bool)
        & (grid.t_anchor + float(horizon) <= hi + 1e-6)
    )


def fit_provisional_history_baseline(
    timeline, partition, horizons: Sequence[float], *, seed: int = 0
) -> HistoryBaseline:
    """Ridge on B_multiscale (no seizure columns): fit state_train, select on dev_val."""

    del seed  # the ridge solve is deterministic; kept for signature stability
    keep = _eligible_baseline_columns(tuple(timeline.baseline.names))
    x = np.asarray(timeline.baseline.x, dtype=np.float64)[:, keep]
    log_mu: dict[int, np.ndarray] = {}
    dispersion: dict[int, float | None] = {}
    fits: dict[str, Any] = {}
    for horizon in horizons:
        h_i = list(timeline.config.horizons_seconds).index(float(horizon))
        counts = (
            np.asarray(timeline.grid.window_hi[:, h_i]) - np.asarray(timeline.grid.window_lo[:, h_i])
        ).astype(np.int64)
        train = _phase_anchor_mask(timeline, partition, "state_train", horizon)
        val = _phase_anchor_mask(timeline, partition, "dev_val", horizon)
        if train.sum() < 5 or val.sum() < 1:
            log_mu[int(horizon)] = np.full(x.shape[0], np.nan)
            dispersion[int(horizon)] = None
            fits[str(int(horizon))] = {"status": "insufficient_anchors",
                                       "n_train": int(train.sum()), "n_val": int(val.sum())}
            continue
        pred, fit = _fit_count_ridge(x[train], counts[train], x[val], counts[val], x)
        pred = np.clip(pred, 1e-3, None)
        log_mu[int(horizon)] = np.log(pred)
        dispersion[int(horizon)] = fit_nb_log_dispersion(counts[train], pred[train])
        fits[str(int(horizon))] = {**{k: v for k, v in fit.items() if k != "path"},
                                   "status": "ok", "n_train": int(train.sum()), "n_val": int(val.sum())}
    return HistoryBaseline(
        log_mu=log_mu,
        nb_log_dispersion=dispersion,
        source="provisional_local",
        meta={
            "definition": "log1p-target ridge on B_multiscale without seizure columns; "
                          "fit on state_train anchors, ridge selected on dev_val; "
                          "NB dispersion MLE on state_train",
            "feature_names_used": [n for n, k in zip(timeline.baseline.names, keep) if k],
            "fits": fits,
            "warning": "provisional stop-gap; not the Agent 2 H_strong definition",
        },
    )


def _first_key(payload, keys: Sequence[str]) -> str | None:
    for key in keys:
        if key in payload:
            return key
    return None


def load_agent2_history_baseline(
    registry_path: Path,
    subject: str,
    t_anchor: np.ndarray,
    horizons: Sequence[float] | float,
) -> tuple[HistoryBaseline | None, str]:
    """Read ``H_strong`` for one subject and align it on anchor time.

    Returns ``(None, reason)`` on any incompatibility; the caller decides whether
    a provisional fallback is permitted and must record the source.
    """

    if isinstance(horizons, (int, float)):
        horizons = (float(horizons),)
    path = Path(registry_path)
    if not path.exists():
        return None, f"missing registry {path}"
    try:
        registry = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return None, f"unreadable registry {path}: {exc}"
    subjects = registry.get("subjects", registry)
    entry = subjects.get(subject)
    if entry is None:
        return None, f"subject {subject} absent from registry"
    per_horizon = entry.get("horizons", entry)
    t_mine = np.asarray(t_anchor, dtype=np.float64)
    log_mu: dict[int, np.ndarray] = {}
    dispersion: dict[int, float | None] = {}
    meta: dict[str, Any] = {"registry_path": str(path), "registry_format": registry.get("format"),
                            "horizon_meta": {}}
    for horizon in horizons:
        key = str(int(horizon))
        spec = per_horizon.get(key) or per_horizon.get(f"{key}s") or per_horizon.get(float(horizon))
        if spec is None:
            return None, f"horizon {key} absent for {subject}"
        arrays_path = spec.get("arrays") or spec.get("path") or spec.get("npz")
        if arrays_path is None or not Path(arrays_path).exists():
            return None, f"arrays file missing for {subject} horizon {key}"
        with np.load(arrays_path, allow_pickle=False) as data:
            a_key = _first_key(data, ANCHOR_KEYS)
            m_key = _first_key(data, LOG_MU_KEYS)
            if a_key is None or m_key is None:
                return None, f"arrays for {subject} horizon {key} lack anchor/log_mu keys"
            t_theirs = np.asarray(data[a_key], dtype=np.float64)
            values = np.asarray(data[m_key], dtype=np.float64)
        if t_theirs.shape != values.shape:
            return None, f"anchor/log_mu length mismatch for {subject} horizon {key}"
        order = np.argsort(t_theirs, kind="stable")
        t_sorted, v_sorted = t_theirs[order], values[order]
        pos = np.searchsorted(t_sorted, t_mine)
        pos = np.clip(pos, 0, max(t_sorted.size - 1, 0))
        left = np.clip(pos - 1, 0, max(t_sorted.size - 1, 0))
        pick = np.where(
            np.abs(t_sorted[pos] - t_mine) <= np.abs(t_sorted[left] - t_mine), pos, left
        )
        if t_sorted.size == 0 or np.any(np.abs(t_sorted[pick] - t_mine) > ANCHOR_TIME_TOLERANCE_SECONDS):
            return None, f"registry anchors do not align with the model anchor grid for {subject} horizon {key}"
        aligned = v_sorted[pick]
        if not np.isfinite(aligned).all():
            return None, f"non-finite log_mu_H for {subject} horizon {key}"
        log_mu[int(horizon)] = aligned
        nb = spec.get("nb_log_dispersion")
        dispersion[int(horizon)] = None if nb is None else float(nb)
        meta["horizon_meta"][key] = {k: v for k, v in spec.items() if k not in ("arrays",)}
    return HistoryBaseline(log_mu=log_mu, nb_log_dispersion=dispersion,
                           source="agent2_registry", meta=meta), "ok"


def load_endpoint_eligibility(path: Path, subject: str) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    subjects = payload.get("subjects", payload)
    entry = subjects.get(subject)
    return None if entry is None else dict(entry)
