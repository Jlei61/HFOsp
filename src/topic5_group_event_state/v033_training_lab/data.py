"""Training-laboratory data view (design §3).

Only two anchor sets are ever exposed to a trainer: ``train`` (nested
``state_train``, 20-70 % of recorded time) and ``inner_val`` (nested
``dev_val``, 70-80 %, chronologically after TRAIN).  Every other anchor's
targets are wiped to ``-1`` at construction so a development-evaluation number
cannot leak into selection by accident.

Contract clauses (plan Task 2):
  [D1] bin counts on ``(0, h)`` equal the bundle's cumulative window counts;
  [D2] only train / inner_val indices exist; other rows are -1; ``assert_no_dev_test`` raises;
  [D3] split_hash depends on subject + partition + horizons only; input_hash on features + scaling;
  [D4] robust scaling statistics use TRAIN events only;
  [D5] event_balanced weights = (1 + events in the lookback window) normalised to mean 1 on TRAIN (A4);
  [D6] a registry-backed H is never patched with provisional bins (missing bins are reported);
  [D7] inner-validation blocks never cross a segment and span < block length.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np

from src.topic5_group_event_state.v03.evaluate import _eligible_baseline_columns, _fit_count_ridge
from src.topic5_group_event_state.v03.partition import PHASE_NAMES
from src.topic5_group_event_state.v032_model.data import SubjectBundle
from src.topic5_group_event_state.v032_model.readout import fit_nb_log_dispersion

from .paths import payload_hash

DEFAULT_BINS: tuple[tuple[float, float], ...] = ((0.0, 300.0), (300.0, 900.0), (900.0, 1800.0))
PHASES: dict[str, str] = {"train": "state_train", "inner_val": "dev_val"}
SCALINGS = ("zscore", "robust")
SAMPLINGS = ("anchor_balanced", "event_balanced")
MIN_BLOCK_SECONDS = 1800.0


@dataclass
class DataView:
    subject: str
    bins: tuple[tuple[float, float], ...]
    horizon: float
    event_times: np.ndarray
    event_segment: np.ndarray
    x_scaled: np.ndarray
    train_event_mask: np.ndarray
    t_anchor: np.ndarray
    anchor_segment: np.ndarray
    last_event_pos: np.ndarray
    segment_bounds: np.ndarray
    phase_index: dict[str, np.ndarray]
    counts: np.ndarray
    log_mu_h: np.ndarray
    log_r_h: np.ndarray
    h_source: str
    missing_h_bins: list[int]
    split_hash: str
    input_hash: str
    scaling: str
    feature_names: tuple[str, ...]
    fingerprint: dict[str, Any]
    scaler_stats: dict[str, Any]
    h_meta: dict[str, Any] = field(default_factory=dict)
    bundle: Any = field(default=None, repr=False)

    # ------------------------------------------------------------------ sizes
    @property
    def n_bins(self) -> int:
        return len(self.bins)

    @property
    def n_features(self) -> int:
        return int(self.x_scaled.shape[1])

    def n(self, phase: str) -> int:
        return int(self.phase_index[phase].size)

    # ---------------------------------------------------------------- guards
    def assert_no_dev_test(self, idx: np.ndarray) -> None:
        """[D2] Any index outside train / inner_val is a leak; refuse loudly."""

        idx = np.asarray(idx, dtype=np.int64)
        if idx.size and (self.counts[idx] < 0).any():
            raise ValueError("anchor index outside train/inner_val requested; dev_test is not exposed")

    # ---------------------------------------------------------------- blocks
    def blocks(self, phase: str, block_seconds: float | None = None) -> np.ndarray:
        """[D7] Contiguous within-segment time bins of length max(horizon, 1800 s)."""

        length = float(block_seconds or max(self.horizon, MIN_BLOCK_SECONDS))
        idx = self.phase_index[phase]
        seg = self.anchor_segment[idx]
        start = self.segment_bounds[seg, 0]
        local = np.floor((self.t_anchor[idx] - start) / length).astype(np.int64)
        key = seg.astype(np.int64) * (int(local.max()) + 2 if local.size else 1) + local
        _unique, inverse = np.unique(key, return_inverse=True)
        return inverse.astype(np.int64)

    def effective_independent_windows(self, phase: str) -> int:
        return int(self.bundle.effective_independent_windows(PHASES[phase], self.horizon))

    # --------------------------------------------------------------- weights
    def lookback_event_counts(self, phase: str, lookback_seconds: float) -> np.ndarray:
        idx = self.phase_index[phase]
        t = self.t_anchor[idx]
        seg_start = self.segment_bounds[self.anchor_segment[idx], 0]
        lo = np.searchsorted(self.event_times, np.maximum(t - float(lookback_seconds), seg_start), side="left")
        hi = np.searchsorted(self.event_times, t, side="left")
        return (hi - lo).astype(np.int64)

    def sample_weights(self, phase: str, mode: str, *, lookback_seconds: float) -> np.ndarray:
        """[D5] anchor_balanced -> 1; event_balanced -> (1 + lookback events) / TRAIN mean."""

        if mode not in SAMPLINGS:
            raise ValueError(f"unknown sampling mode {mode!r}; allowed {SAMPLINGS}")
        n = self.n(phase)
        if mode == "anchor_balanced":
            return np.ones(n, dtype=np.float64)
        raw = 1.0 + self.lookback_event_counts(phase, lookback_seconds).astype(np.float64)
        train_raw = 1.0 + self.lookback_event_counts("train", lookback_seconds).astype(np.float64)
        return raw / float(train_raw.mean())

    def summary(self) -> dict[str, Any]:
        return {
            "subject": self.subject, "bins_seconds": [list(b) for b in self.bins], "horizon_seconds": self.horizon,
            "n_events": int(self.event_times.size), "n_features": self.n_features,
            "n_train_events": int(self.train_event_mask.sum()),
            "n_anchors": {p: self.n(p) for p in self.phase_index},
            "effective_independent_windows": {p: self.effective_independent_windows(p) for p in self.phase_index},
            "h_source": self.h_source, "missing_h_bins": list(self.missing_h_bins),
            "split_hash": self.split_hash, "input_hash": self.input_hash, "scaling": self.scaling,
            "dev_test_exposed": False,
        }


# ------------------------------------------------------------------- helpers
def bin_counts(event_times: np.ndarray, t_anchor: np.ndarray, bins: Sequence[Sequence[float]]) -> np.ndarray:
    """[D1] Events in ``[t+a, t+b)`` per anchor, same ``searchsorted(side='left')`` rule as v0.2 windows."""

    events = np.asarray(event_times, dtype=np.float64)
    t = np.asarray(t_anchor, dtype=np.float64)
    out = np.zeros((t.size, len(bins)), dtype=np.int64)
    for j, (a, b) in enumerate(bins):
        lo = np.searchsorted(events, t + float(a), side="left")
        hi = np.searchsorted(events, t + float(b), side="left")
        out[:, j] = hi - lo
    return out


def robust_scale_fit(x: np.ndarray, train_mask: np.ndarray) -> dict[str, Any]:
    """[D4] Median / (IQR / 1.349) per column from TRAIN rows only; MAD fallback; degenerate -> 1."""

    rows = np.asarray(x, dtype=np.float64)[np.asarray(train_mask, dtype=bool)]
    if rows.shape[0] == 0:
        raise ValueError("robust scaler needs at least one TRAIN row")
    with np.errstate(all="ignore"):
        center = np.nanmedian(rows, axis=0)
        q75, q25 = np.nanpercentile(rows, [75, 25], axis=0)
        scale = (q75 - q25) / 1.349
        mad = np.nanmedian(np.abs(rows - center), axis=0) * 1.4826
    scale = np.where(scale > 1e-9, scale, mad)
    degenerate = ~np.isfinite(center) | ~np.isfinite(scale) | (scale <= 1e-9)
    center = np.where(degenerate, 0.0, center)
    scale = np.where(degenerate, 1.0, scale)
    return {"method": "robust", "center": center.tolist(), "scale": scale.tolist(),
            "degenerate": degenerate.tolist()}


def robust_scale_apply(x: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    center = np.asarray(stats["center"], dtype=np.float64)
    scale = np.asarray(stats["scale"], dtype=np.float64)
    degenerate = np.asarray(stats["degenerate"], dtype=bool)
    z = (np.asarray(x, dtype=np.float64) - center) / scale
    z[:, degenerate] = 0.0
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.astype(np.float32)


def provisional_bin_history(bundle: SubjectBundle, bins: Sequence[Sequence[float]], *, horizon: float
                            ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Provisional ``log mu_H`` per bin: v0.3 ridge on B_multiscale (no seizure columns),
    fitted on state_train anchors, ridge chosen on dev_val, dispersion MLE on state_train.
    Toy / synthetic use only (``h_source='provisional_local'``)."""

    if bundle.baseline_x is None:
        raise ValueError("provisional per-bin H needs the bundle's B_multiscale baseline features")
    keep = _eligible_baseline_columns(tuple(bundle.baseline_names))
    x = np.asarray(bundle.baseline_x, dtype=np.float64)[:, keep]
    counts = bin_counts(bundle.event_times, bundle.t_anchor, bins)
    train = bundle.anchor_mask("state_train", horizon)
    val = bundle.anchor_mask("dev_val", horizon)
    log_mu = np.full(counts.shape, np.nan)
    log_r = np.full(len(bins), np.nan)
    fits: dict[str, Any] = {}
    for j, (a, b) in enumerate(bins):
        pred, fit = _fit_count_ridge(x[train], counts[train, j], x[val], counts[val, j], x)
        pred = np.clip(pred, 1e-3, None)
        log_mu[:, j] = np.log(pred)
        log_r[j] = fit_nb_log_dispersion(counts[train, j], pred[train])
        fits[f"{int(a)}-{int(b)}"] = {k: v for k, v in fit.items() if k != "path"}
    meta = {"definition": "log1p-target ridge on B_multiscale without seizure columns per bin; "
                          "fit on state_train, ridge on dev_val, NB dispersion MLE on state_train",
            "fits": fits, "warning": "provisional stop-gap; not an Agent A/2 H definition"}
    return log_mu, log_r, meta


def _registry_bin(bundle: SubjectBundle, a: float, b: float, *, train_idx: np.ndarray) -> tuple[np.ndarray, float] | None:
    """A registry / bundle H exists only for cumulative windows ``(0, h)`` with ``h`` in the bundle."""

    if float(a) != 0.0 or not bundle.history.has_horizon(b):
        return None
    log_mu = np.asarray(bundle.history.log_mu[int(b)], dtype=np.float64)
    given = bundle.history.nb_log_dispersion.get(int(b))
    if given is not None and math.isfinite(float(given)):
        log_r = float(given)
    else:
        h_i = bundle.horizon_index(float(b))
        log_r = fit_nb_log_dispersion(bundle.counts[train_idx, h_i], np.exp(log_mu[train_idx]))
    return log_mu, log_r


def build_view(
    bundle: SubjectBundle,
    *,
    bins: Sequence[Sequence[float]] = DEFAULT_BINS,
    scaling: str = "zscore",
) -> DataView:
    bins_t = tuple((float(a), float(b)) for a, b in bins)
    if any(b <= a or a < 0 for a, b in bins_t):
        raise ValueError("bins must be [a, b) with 0 <= a < b")
    horizon = max(b for _a, b in bins_t)
    if float(horizon) not in tuple(float(h) for h in bundle.horizons):
        raise ValueError(f"horizon {horizon} s is not one of the bundle horizons {bundle.horizons}")
    if scaling not in SCALINGS:
        raise ValueError(f"unknown scaling {scaling!r}; allowed {SCALINGS}")

    train_idx = np.flatnonzero(bundle.anchor_mask(PHASES["train"], horizon))
    val_idx = np.flatnonzero(bundle.anchor_mask(PHASES["inner_val"], horizon))
    phase_index = {"train": train_idx, "inner_val": val_idx}

    counts = bin_counts(bundle.event_times, bundle.t_anchor, bins_t)               # [D1]
    exposed = np.zeros(bundle.n_anchors, dtype=bool)
    exposed[train_idx] = True
    exposed[val_idx] = True
    counts[~exposed] = -1                                                           # [D2]

    log_mu_h = np.full(counts.shape, np.nan)
    log_r_h = np.full(len(bins_t), np.nan)
    missing: list[int] = []
    for j, (a, b) in enumerate(bins_t):
        got = _registry_bin(bundle, a, b, train_idx=train_idx)
        if got is None:
            missing.append(j)
        else:
            log_mu_h[:, j], log_r_h[j] = got
    h_source = bundle.history.source
    h_meta: dict[str, Any] = {"bundle_h_meta": dict(bundle.history.meta)}
    if missing and h_source != "agent2_registry":                                   # [D6]
        prov_mu, prov_r, meta = provisional_bin_history(bundle, bins_t, horizon=horizon)
        for j in missing:
            log_mu_h[:, j] = prov_mu[:, j]
            log_r_h[j] = prov_r[j]
        h_meta["provisional_bins"] = {"bins": missing, **meta}
        h_source = "provisional_local"
        missing = []

    train_event_mask = bundle.train_event_mask()
    if scaling == "zscore":
        x_scaled = np.asarray(bundle.x_std, dtype=np.float32)
        scaler_stats = {"method": "zscore", **bundle.standardizer.to_dict()}
    else:
        scaler_stats = robust_scale_fit(bundle.x_raw, train_event_mask)             # [D4]
        x_scaled = robust_scale_apply(bundle.x_raw, scaler_stats)

    finite_lower = [float(v) if np.isfinite(v) else None for v in bundle.phase_lower]
    finite_upper = [float(v) if np.isfinite(v) else None for v in bundle.phase_upper]
    split_hash = payload_hash({                                                     # [D3]
        "subject": bundle.subject, "phase_names": list(PHASE_NAMES),
        "phase_lower": finite_lower, "phase_upper": finite_upper,
        "horizons": [float(h) for h in bundle.horizons], "train_phase": PHASES["train"],
        "inner_val_phase": PHASES["inner_val"],
    })
    input_hash = payload_hash({
        "fingerprint": bundle.fingerprint, "feature_names": list(bundle.feature_names),
        "scaling": scaling, "scaler_stats": {k: (np.round(np.asarray(v, dtype=np.float64), 6).tolist()
                                              if isinstance(v, list) else v)
                                             for k, v in scaler_stats.items()},
        "n_events": int(bundle.n_events),
    })
    return DataView(
        subject=bundle.subject, bins=bins_t, horizon=float(horizon),
        event_times=np.asarray(bundle.event_times, dtype=np.float64),
        event_segment=np.asarray(bundle.event_segment, dtype=np.int64),
        x_scaled=np.ascontiguousarray(x_scaled), train_event_mask=np.asarray(train_event_mask, dtype=bool),
        t_anchor=np.asarray(bundle.t_anchor, dtype=np.float64),
        anchor_segment=np.asarray(bundle.anchor_segment, dtype=np.int64),
        last_event_pos=np.asarray(bundle.last_event_pos, dtype=np.int64),
        segment_bounds=np.asarray(bundle.segment_bounds, dtype=np.float64),
        phase_index=phase_index, counts=counts, log_mu_h=log_mu_h, log_r_h=log_r_h,
        h_source=h_source, missing_h_bins=missing, split_hash=split_hash, input_hash=input_hash,
        scaling=scaling, feature_names=tuple(bundle.feature_names), fingerprint=dict(bundle.fingerprint),
        scaler_stats=scaler_stats, h_meta=h_meta, bundle=bundle,
    )
