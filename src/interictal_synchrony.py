"""
PR4: interictal synchrony metrics on lagPat/groupAnalysis assets.

This module is intentionally additive:
- it consumes existing `groupAnalysis.npz` outputs
- it does not change the detection / packing / lag pipeline
- it exposes a compact contract for downstream PR5/PR6 slicing and stats
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .epilepsiae_dataset import (
    EpilepsiaePaths,
    load_epilepsiae_sync_subject_manifest,
    survey_epilepsiae_dataset,
)
from .group_event_analysis import load_group_analysis_results
from .preprocessing import epoch_to_local_hour, read_edf_record_info


_EMPTY_STRATUM = "empty"
_CORE_ONLY_STRATUM = "core_only"
_PENUMBRA_ONLY_STRATUM = "penumbra_only"
_MIXED_STRATUM = "mixed"
EVENT_SYNC_SCHEMA_VERSION = "event_sync_v1"
BLOCK_SYNC_COMPAT_SCHEMA_VERSION = "block_sync_compat_v1"


@dataclass
class InterictalSynchronyResult:
    """Synchrony features computed from an existing group-analysis asset."""

    ch_names: List[str]
    event_windows: np.ndarray
    core_mask: np.ndarray
    event_n_participating: np.ndarray
    event_n_core: np.ndarray
    event_n_penumbra: np.ndarray
    event_stratum: np.ndarray
    sync_phase_global: np.ndarray
    sync_phase_core: np.ndarray
    sync_phase_penumbra: np.ndarray
    sync_legacy_global: np.ndarray
    sync_legacy_core: np.ndarray
    sync_legacy_penumbra: np.ndarray
    sync_span_global: np.ndarray
    sync_span_core: np.ndarray
    sync_span_penumbra: np.ndarray
    jaccard_global: np.ndarray
    jaccard_core: np.ndarray
    jaccard_penumbra: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, float] = field(default_factory=dict)


def _as_bool_matrix(x: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    return arr.astype(bool, copy=False)


def _as_float_matrix(x: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    return arr


def _as_event_windows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"event_windows must have shape (n_events, 2), got {arr.shape}")
    return arr


def select_core_penumbra_mask(
    ch_names: Sequence[str],
    core_channels: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    Return a boolean mask marking channels treated as core.

    If `core_channels` is omitted **or** none of the given core channels appear
    in `ch_names`, all channels are treated as core (full-core fallback).
    """
    names = [str(x) for x in ch_names]
    if core_channels is None:
        return np.ones((len(names),), dtype=bool)
    core_set = {str(x) for x in core_channels}
    mask = np.array([name in core_set for name in names], dtype=bool)
    if not np.any(mask):
        import warnings
        warnings.warn(
            f"No overlap between core_channels ({len(core_set)}) and "
            f"ch_names ({len(names)}); falling back to all-core.",
            stacklevel=2,
        )
        return np.ones((len(names),), dtype=bool)
    return mask


def compute_event_synchrony_metrics(
    lag_raw: np.ndarray,
    events_bool: np.ndarray,
    event_windows: np.ndarray,
    *,
    channel_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute three per-event synchrony metrics on the selected channel subset.

    Metrics:
    - `phase`: half-circle phase-order score in [0, 1]
    - `legacy`: 1 - std(lag) / (max(lag) - min(lag))
    - `span`:   1 - (max(lag) - min(lag)) / window_duration
    """
    lag = _as_float_matrix(lag_raw, name="lag_raw")
    eb = _as_bool_matrix(events_bool, name="events_bool")
    windows = _as_event_windows(event_windows)
    if lag.shape != eb.shape:
        raise ValueError(f"lag_raw shape {lag.shape} != events_bool shape {eb.shape}")
    if lag.shape[1] != windows.shape[0]:
        raise ValueError(
            "Event dimension mismatch between lag_raw/events_bool and event_windows."
        )

    n_ch, n_events = lag.shape
    if channel_mask is None:
        use_mask = np.ones((n_ch,), dtype=bool)
    else:
        use_mask = np.asarray(channel_mask, dtype=bool)
        if use_mask.shape != (n_ch,):
            raise ValueError(f"channel_mask must have shape ({n_ch},), got {use_mask.shape}")

    phase = np.full((n_events,), np.nan, dtype=np.float64)
    legacy = np.full((n_events,), np.nan, dtype=np.float64)
    span = np.full((n_events,), np.nan, dtype=np.float64)
    n_participating = np.zeros((n_events,), dtype=np.int64)

    eps = 1e-12
    durations = np.asarray(windows[:, 1] - windows[:, 0], dtype=np.float64)
    if np.any(durations <= 0):
        raise ValueError("event_windows must have positive duration.")

    for ev in range(n_events):
        members = use_mask & eb[:, ev]
        vals = lag[members, ev]
        vals = vals[np.isfinite(vals)]
        n_participating[ev] = int(vals.size)
        if vals.size == 0:
            continue
        if vals.size == 1:
            phase[ev] = 1.0
            legacy[ev] = 1.0
            span[ev] = 1.0
            continue

        lag_min = float(np.min(vals))
        lag_max = float(np.max(vals))
        lag_span = lag_max - lag_min
        if lag_span <= eps:
            phase[ev] = 1.0
            legacy[ev] = 1.0
            span[ev] = 1.0
            continue

        phases = np.pi * (vals - lag_min) / lag_span
        phase[ev] = float(np.abs(np.mean(np.exp(1j * phases))))
        legacy[ev] = float(np.clip(1.0 - (np.std(vals) / lag_span), 0.0, 1.0))
        span[ev] = float(np.clip(1.0 - (lag_span / durations[ev]), 0.0, 1.0))

    return {
        "phase": phase,
        "legacy": legacy,
        "span": span,
        "n_participating": n_participating,
    }


def compute_adjacent_jaccard(
    events_bool: np.ndarray,
    *,
    channel_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute adjacent-event Jaccard similarity on participation sets."""
    eb = _as_bool_matrix(events_bool, name="events_bool")
    n_ch, n_events = eb.shape
    if channel_mask is None:
        use_mask = np.ones((n_ch,), dtype=bool)
    else:
        use_mask = np.asarray(channel_mask, dtype=bool)
        if use_mask.shape != (n_ch,):
            raise ValueError(f"channel_mask must have shape ({n_ch},), got {use_mask.shape}")
    if n_events <= 1:
        return np.zeros((0,), dtype=np.float64)

    out = np.full((n_events - 1,), np.nan, dtype=np.float64)
    for ev in range(n_events - 1):
        left = use_mask & eb[:, ev]
        right = use_mask & eb[:, ev + 1]
        union = int(np.sum(left | right))
        if union == 0:
            continue
        inter = int(np.sum(left & right))
        out[ev] = float(inter / union)
    return out


def _classify_event_strata(
    events_bool: np.ndarray,
    core_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    eb = _as_bool_matrix(events_bool, name="events_bool")
    mask = np.asarray(core_mask, dtype=bool)
    if mask.shape != (eb.shape[0],):
        raise ValueError(f"core_mask must have shape ({eb.shape[0]},), got {mask.shape}")

    n_core = np.sum(eb & mask[:, None], axis=0).astype(np.int64, copy=False)
    n_penumbra = np.sum(eb & (~mask)[:, None], axis=0).astype(np.int64, copy=False)
    n_participating = np.sum(eb, axis=0).astype(np.int64, copy=False)

    strata = np.full((eb.shape[1],), _EMPTY_STRATUM, dtype=object)
    strata[(n_core > 0) & (n_penumbra == 0)] = _CORE_ONLY_STRATUM
    strata[(n_core == 0) & (n_penumbra > 0)] = _PENUMBRA_ONLY_STRATUM
    strata[(n_core > 0) & (n_penumbra > 0)] = _MIXED_STRATUM

    return {
        "n_core": n_core,
        "n_penumbra": n_penumbra,
        "n_participating": n_participating,
        "strata": strata,
    }


def _safe_nanmean(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _normalize_metadata(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if metadata is None:
        return {}
    return {str(key): value for key, value in dict(metadata).items()}


def _event_metric_or_none(arr: np.ndarray, idx: int) -> Optional[float]:
    value = float(arr[idx])
    if np.isnan(value):
        return None
    return value


def build_event_rows_from_result(result: InterictalSynchronyResult) -> List[Dict[str, object]]:
    """
    Flatten a per-block synchrony result into event-level rows.

    The exported rows are the primary PR4 artifact for downstream PR5/PR6.
    """
    metadata = dict(result.metadata)
    block_start_epoch = metadata.get("block_start_epoch")
    block_start = None if block_start_epoch in (None, "") else float(block_start_epoch)
    windows = np.asarray(result.event_windows, dtype=np.float64)
    n_events = int(windows.shape[0])
    rows: List[Dict[str, object]] = []

    for event_idx in range(n_events):
        start_rel = float(windows[event_idx, 0])
        end_rel = float(windows[event_idx, 1])
        center_rel = 0.5 * (start_rel + end_rel)
        if block_start is None:
            event_start_epoch = None
            event_end_epoch = None
            event_center_epoch = None
        else:
            event_start_epoch = block_start + start_rel
            event_end_epoch = block_start + end_rel
            event_center_epoch = block_start + center_rel

        row: Dict[str, object] = {
            **metadata,
            "schema_version": EVENT_SYNC_SCHEMA_VERSION,
            "n_channels": int(len(result.ch_names)),
            "n_core_channels": int(np.sum(result.core_mask)),
            "n_penumbra_channels": int(np.sum(~result.core_mask)),
            "event_index": event_idx,
            "event_start_sec_rel": start_rel,
            "event_end_sec_rel": end_rel,
            "event_center_sec_rel": center_rel,
            "event_duration_sec": float(end_rel - start_rel),
            "event_start_epoch": event_start_epoch,
            "event_end_epoch": event_end_epoch,
            "event_center_epoch": event_center_epoch,
            "n_participating": int(result.event_n_participating[event_idx]),
            "n_core": int(result.event_n_core[event_idx]),
            "n_penumbra": int(result.event_n_penumbra[event_idx]),
            "event_stratum": str(result.event_stratum[event_idx]),
            "sync_phase_global": _event_metric_or_none(result.sync_phase_global, event_idx),
            "sync_phase_core": _event_metric_or_none(result.sync_phase_core, event_idx),
            "sync_phase_penumbra": _event_metric_or_none(result.sync_phase_penumbra, event_idx),
            "sync_legacy_global": _event_metric_or_none(result.sync_legacy_global, event_idx),
            "sync_legacy_core": _event_metric_or_none(result.sync_legacy_core, event_idx),
            "sync_legacy_penumbra": _event_metric_or_none(result.sync_legacy_penumbra, event_idx),
            "sync_span_global": _event_metric_or_none(result.sync_span_global, event_idx),
            "sync_span_core": _event_metric_or_none(result.sync_span_core, event_idx),
            "sync_span_penumbra": _event_metric_or_none(result.sync_span_penumbra, event_idx),
            "jaccard_global_with_next": None,
            "jaccard_core_with_next": None,
            "jaccard_penumbra_with_next": None,
        }
        if event_idx < len(result.jaccard_global):
            row["jaccard_global_with_next"] = _event_metric_or_none(result.jaccard_global, event_idx)
            row["jaccard_core_with_next"] = _event_metric_or_none(result.jaccard_core, event_idx)
            row["jaccard_penumbra_with_next"] = _event_metric_or_none(result.jaccard_penumbra, event_idx)
        rows.append(row)
    return rows


def build_block_summary_from_event_rows(event_rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """
    Build the compatibility block-level summary from event rows.

    This keeps old consumers alive while making clear that block summaries are
    derived views, not primary metrics.
    """
    rows = [dict(r) for r in event_rows]
    if not rows:
        return {"schema_version": BLOCK_SYNC_COMPAT_SCHEMA_VERSION}

    def _mean_of(key: str) -> float:
        values = [
            float(r[key])
            for r in rows
            if r.get(key) not in (None, "", "nan", "NaN")
        ]
        if not values:
            return float("nan")
        return float(np.mean(values))

    n_events = float(len(rows))
    frac_core_only = float(np.mean([str(r.get("event_stratum")) == _CORE_ONLY_STRATUM for r in rows]))
    frac_penumbra_only = float(
        np.mean([str(r.get("event_stratum")) == _PENUMBRA_ONLY_STRATUM for r in rows])
    )
    frac_mixed = float(np.mean([str(r.get("event_stratum")) == _MIXED_STRATUM for r in rows]))
    summary = {
        "schema_version": BLOCK_SYNC_COMPAT_SCHEMA_VERSION,
        "n_events": n_events,
        "n_channels": float(rows[0].get("n_channels") or np.nan),
        "n_core_channels": float(rows[0].get("n_core_channels") or np.nan),
        "n_penumbra_channels": float(rows[0].get("n_penumbra_channels") or np.nan),
        "mean_sync_phase_global": _mean_of("sync_phase_global"),
        "mean_sync_phase_core": _mean_of("sync_phase_core"),
        "mean_sync_phase_penumbra": _mean_of("sync_phase_penumbra"),
        "mean_sync_legacy_global": _mean_of("sync_legacy_global"),
        "mean_sync_legacy_core": _mean_of("sync_legacy_core"),
        "mean_sync_legacy_penumbra": _mean_of("sync_legacy_penumbra"),
        "mean_sync_span_global": _mean_of("sync_span_global"),
        "mean_sync_span_core": _mean_of("sync_span_core"),
        "mean_sync_span_penumbra": _mean_of("sync_span_penumbra"),
        "mean_jaccard_global": _mean_of("jaccard_global_with_next"),
        "mean_jaccard_core": _mean_of("jaccard_core_with_next"),
        "mean_jaccard_penumbra": _mean_of("jaccard_penumbra_with_next"),
        "frac_core_only_events": frac_core_only,
        "frac_penumbra_only_events": frac_penumbra_only,
        "frac_mixed_events": frac_mixed,
    }
    return summary


def compute_interictal_synchrony(
    group_analysis: Mapping[str, Any],
    *,
    core_channels: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> InterictalSynchronyResult:
    """Compute PR4 synchrony features from a loaded group-analysis mapping."""
    ch_names = [str(x) for x in group_analysis["ch_names"]]
    event_windows = _as_event_windows(group_analysis["event_windows"])
    lag_raw = _as_float_matrix(group_analysis["lag_raw"], name="lag_raw")
    events_bool = _as_bool_matrix(group_analysis["events_bool"], name="events_bool")
    if lag_raw.shape != events_bool.shape:
        raise ValueError("lag_raw and events_bool shape mismatch.")
    if lag_raw.shape[0] != len(ch_names):
        raise ValueError("Channel dimension mismatch between matrices and ch_names.")

    core_mask = select_core_penumbra_mask(ch_names, core_channels=core_channels)
    strata = _classify_event_strata(events_bool, core_mask)

    global_sync = compute_event_synchrony_metrics(lag_raw, events_bool, event_windows)
    core_sync = compute_event_synchrony_metrics(
        lag_raw, events_bool, event_windows, channel_mask=core_mask
    )
    penumbra_sync = compute_event_synchrony_metrics(
        lag_raw, events_bool, event_windows, channel_mask=~core_mask
    )

    jaccard_global = compute_adjacent_jaccard(events_bool)
    jaccard_core = compute_adjacent_jaccard(events_bool, channel_mask=core_mask)
    jaccard_penumbra = compute_adjacent_jaccard(events_bool, channel_mask=~core_mask)

    summary = {
        "n_events": float(event_windows.shape[0]),
        "n_channels": float(len(ch_names)),
        "n_core_channels": float(np.sum(core_mask)),
        "n_penumbra_channels": float(np.sum(~core_mask)),
        "mean_sync_phase_global": _safe_nanmean(global_sync["phase"]),
        "mean_sync_phase_core": _safe_nanmean(core_sync["phase"]),
        "mean_sync_phase_penumbra": _safe_nanmean(penumbra_sync["phase"]),
        "mean_sync_legacy_global": _safe_nanmean(global_sync["legacy"]),
        "mean_sync_legacy_core": _safe_nanmean(core_sync["legacy"]),
        "mean_sync_legacy_penumbra": _safe_nanmean(penumbra_sync["legacy"]),
        "mean_sync_span_global": _safe_nanmean(global_sync["span"]),
        "mean_sync_span_core": _safe_nanmean(core_sync["span"]),
        "mean_sync_span_penumbra": _safe_nanmean(penumbra_sync["span"]),
        "mean_jaccard_global": _safe_nanmean(jaccard_global),
        "mean_jaccard_core": _safe_nanmean(jaccard_core),
        "mean_jaccard_penumbra": _safe_nanmean(jaccard_penumbra),
        "frac_core_only_events": float(np.mean(strata["strata"] == _CORE_ONLY_STRATUM)),
        "frac_penumbra_only_events": float(np.mean(strata["strata"] == _PENUMBRA_ONLY_STRATUM)),
        "frac_mixed_events": float(np.mean(strata["strata"] == _MIXED_STRATUM)),
    }

    return InterictalSynchronyResult(
        ch_names=ch_names,
        event_windows=event_windows,
        core_mask=core_mask,
        event_n_participating=strata["n_participating"],
        event_n_core=strata["n_core"],
        event_n_penumbra=strata["n_penumbra"],
        event_stratum=strata["strata"],
        sync_phase_global=global_sync["phase"],
        sync_phase_core=core_sync["phase"],
        sync_phase_penumbra=penumbra_sync["phase"],
        sync_legacy_global=global_sync["legacy"],
        sync_legacy_core=core_sync["legacy"],
        sync_legacy_penumbra=penumbra_sync["legacy"],
        sync_span_global=global_sync["span"],
        sync_span_core=core_sync["span"],
        sync_span_penumbra=penumbra_sync["span"],
        jaccard_global=jaccard_global,
        jaccard_core=jaccard_core,
        jaccard_penumbra=jaccard_penumbra,
        metadata=_normalize_metadata(metadata),
        params={
            "metrics": ["phase", "legacy", "span"],
            "spatial_stability": "adjacent_jaccard",
        },
        summary=summary,
    )


def load_legacy_lagpat_group_analysis(
    lagpat_npz_path: str,
    packed_times_path: str,
) -> Dict[str, Any]:
    """
    Convert legacy `lagPat + packedTimes` assets into the group-analysis contract.

    `lagPatRaw` does not need absolute alignment to `packedTimes`; PR4 metrics only
    depend on event membership and within-window lag span/order.
    """
    lag = np.load(lagpat_npz_path, allow_pickle=True)
    missing = [key for key in ("lagPatRaw", "eventsBool", "chnNames") if key not in lag]
    if missing:
        raise ValueError(
            f"Legacy lagPat asset missing required keys {missing}: {lagpat_npz_path}"
        )

    lag_raw = _as_float_matrix(np.asarray(lag["lagPatRaw"], dtype=np.float64), name="lagPatRaw")
    events_bool = _as_bool_matrix(np.asarray(lag["eventsBool"]) > 0, name="eventsBool")
    event_windows = _as_event_windows(np.load(packed_times_path, allow_pickle=True))
    ch_names = [str(x) for x in np.asarray(lag["chnNames"]).tolist()]

    if lag_raw.shape != events_bool.shape:
        raise ValueError("Legacy lagPatRaw and eventsBool shape mismatch.")
    if lag_raw.shape[0] != len(ch_names):
        raise ValueError("Legacy chnNames length does not match lagPatRaw rows.")
    if lag_raw.shape[1] != event_windows.shape[0]:
        raise ValueError("packedTimes event count does not match lagPat columns.")

    out: Dict[str, Any] = {
        "ch_names": ch_names,
        "event_windows": event_windows,
        "lag_raw": lag_raw,
        "events_bool": events_bool,
    }
    if "start_t" in lag:
        out["start_t"] = float(np.asarray(lag["start_t"]).reshape(-1)[0])
    return out


def build_interictal_synchrony_from_legacy_lagpat(
    lagpat_npz_path: str,
    packed_times_path: str,
    *,
    core_channels: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> InterictalSynchronyResult:
    group_analysis = load_legacy_lagpat_group_analysis(lagpat_npz_path, packed_times_path)
    return compute_interictal_synchrony(
        group_analysis,
        core_channels=core_channels,
        metadata=metadata,
    )


def build_interictal_synchrony(
    group_analysis_npz_path: str,
    *,
    core_channels: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> InterictalSynchronyResult:
    """Load `groupAnalysis.npz` and compute PR4 synchrony features."""
    group_analysis = load_group_analysis_results(group_analysis_npz_path)
    return compute_interictal_synchrony(
        group_analysis,
        core_channels=core_channels,
        metadata=metadata,
    )


def save_interictal_synchrony_result(result: InterictalSynchronyResult, npz_path: str) -> str:
    """Persist the PR4 synchrony result in a compact NPZ contract."""
    np.savez_compressed(
        npz_path,
        ch_names=np.asarray(result.ch_names, dtype=object),
        event_windows=np.asarray(result.event_windows, dtype=np.float64),
        core_mask=np.asarray(result.core_mask, dtype=bool),
        event_n_participating=np.asarray(result.event_n_participating, dtype=np.int64),
        event_n_core=np.asarray(result.event_n_core, dtype=np.int64),
        event_n_penumbra=np.asarray(result.event_n_penumbra, dtype=np.int64),
        event_stratum=np.asarray(result.event_stratum, dtype=object),
        sync_phase_global=np.asarray(result.sync_phase_global, dtype=np.float64),
        sync_phase_core=np.asarray(result.sync_phase_core, dtype=np.float64),
        sync_phase_penumbra=np.asarray(result.sync_phase_penumbra, dtype=np.float64),
        sync_legacy_global=np.asarray(result.sync_legacy_global, dtype=np.float64),
        sync_legacy_core=np.asarray(result.sync_legacy_core, dtype=np.float64),
        sync_legacy_penumbra=np.asarray(result.sync_legacy_penumbra, dtype=np.float64),
        sync_span_global=np.asarray(result.sync_span_global, dtype=np.float64),
        sync_span_core=np.asarray(result.sync_span_core, dtype=np.float64),
        sync_span_penumbra=np.asarray(result.sync_span_penumbra, dtype=np.float64),
        jaccard_global=np.asarray(result.jaccard_global, dtype=np.float64),
        jaccard_core=np.asarray(result.jaccard_core, dtype=np.float64),
        jaccard_penumbra=np.asarray(result.jaccard_penumbra, dtype=np.float64),
        metadata=np.array([result.metadata], dtype=object),
        params=np.array([result.params], dtype=object),
        summary=np.array([result.summary], dtype=object),
    )
    return npz_path


def save_interictal_synchrony_summary(rows: Sequence[Mapping[str, object]], csv_path: str) -> str:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = [dict(row) for row in rows]
    if not rows_list:
        path.write_text("", encoding="utf-8")
        return str(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_list[0].keys()))
        writer.writeheader()
        writer.writerows(rows_list)
    return str(path)


def _classify_day_night(
    epoch_ts: float,
    *,
    timezone_name: str,
    day_start_hour: int,
    night_start_hour: int,
) -> str:
    local_hour = epoch_to_local_hour(float(epoch_ts), timezone_name)
    if int(day_start_hour) <= int(local_hour) < int(night_start_hour):
        return "day"
    return "night"


def run_epilepsiae_interictal_synchrony_from_manifest(
    manifest_csv_path: str,
    output_dir: str,
    *,
    tier: str = "ready_full_artifacts",
    artifact_root: Optional[str] = None,
    core_channels_by_subject: Optional[Mapping[str, Sequence[str]]] = None,
    event_rows_csv_path: Optional[str] = None,
) -> List[Dict[str, object]]:
    manifest_rows = load_epilepsiae_sync_subject_manifest(manifest_csv_path)
    selected = [row for row in manifest_rows if row.get("tier") == tier]
    if not selected:
        raise ValueError(f"No Epilepsiae subjects found in tier={tier!r}.")

    artifact_dir_root = Path(artifact_root) if artifact_root else EpilepsiaePaths().artifact_root
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    inventory = survey_epilepsiae_dataset()
    block_meta_by_key = {
        (str(block["subject"]), str(block["block_stem"])): {
            "subject": str(block["subject"]),
            "patient_code": str(block.get("patient_code", "")),
            "recording_id": str(block.get("recording_id", "")),
            "block_stem": str(block["block_stem"]),
            "block_start_epoch": float(block["block_start_epoch"]),
            "block_end_epoch": float(block["block_end_epoch"]),
            "block_start_day_night": str(block.get("block_start_day_night", "")),
            "block_end_day_night": str(block.get("block_end_day_night", "")),
            "timezone_name": str(block.get("timezone_name", "")),
        }
        for block in inventory.block_rows
        if block.get("block_start_epoch") is not None and block.get("block_end_epoch") is not None
    }

    summary_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []
    for row in selected:
        subject = str(row["subject"])
        subject_dir = artifact_dir_root / subject / "all_recs"
        if not subject_dir.exists():
            raise ValueError(f"Artifact directory missing for subject {subject}: {subject_dir}")

        core_channels = None
        if core_channels_by_subject is not None:
            core_channels = core_channels_by_subject.get(subject)

        for lagpat_path in sorted(subject_dir.glob("*_lagPat.npz")):
            stem = lagpat_path.stem.removesuffix("_lagPat")
            packed_times_path = subject_dir / f"{stem}_packedTimes.npy"
            if not packed_times_path.exists():
                continue
            block_metadata = block_meta_by_key.get((subject, stem))
            if block_metadata is None:
                raise ValueError(
                    f"Missing block metadata for subject={subject} block_stem={stem}."
                )
            result_metadata = {
                **block_metadata,
                "tier": tier,
                "day_night_rule": row.get("day_night_rule", ""),
            }

            result = build_interictal_synchrony_from_legacy_lagpat(
                str(lagpat_path),
                str(packed_times_path),
                core_channels=core_channels,
                metadata=result_metadata,
            )
            subject_out_dir = out_root / subject
            subject_out_dir.mkdir(parents=True, exist_ok=True)
            npz_path = subject_out_dir / f"{stem}_interictal_sync.npz"
            save_interictal_synchrony_result(result, str(npz_path))

            summary_rows.append(
                {
                    **result.metadata,
                    "lagpat_npz_path": str(lagpat_path),
                    "packed_times_path": str(packed_times_path),
                    "output_npz_path": str(npz_path),
                    **result.summary,
                    "summary_schema_version": BLOCK_SYNC_COMPAT_SCHEMA_VERSION,
                    "event_schema_version": EVENT_SYNC_SCHEMA_VERSION,
                }
            )
            event_rows.extend(build_event_rows_from_result(result))

    if event_rows_csv_path:
        save_interictal_synchrony_summary(event_rows, event_rows_csv_path)

    return summary_rows


def run_yuquan_interictal_synchrony(
    output_dir: str,
    *,
    artifact_root: str = "/mnt/yuquan_data/yuquan_24h_edf",
    subjects: Optional[Sequence[str]] = None,
    core_channels_by_subject: Optional[Mapping[str, Sequence[str]]] = None,
    timezone_name: str = "Asia/Shanghai",
    day_start_hour: int = 8,
    night_start_hour: int = 20,
    event_rows_csv_path: Optional[str] = None,
) -> List[Dict[str, object]]:
    artifact_dir_root = Path(artifact_root)
    if not artifact_dir_root.exists():
        raise ValueError(f"Yuquan artifact root missing: {artifact_dir_root}")

    if subjects is None:
        selected_subjects = sorted(x.name for x in artifact_dir_root.iterdir() if x.is_dir())
    else:
        selected_subjects = sorted(str(x) for x in subjects)
    if not selected_subjects:
        raise ValueError("No Yuquan subjects selected for interictal synchrony.")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []
    day_night_rule = (
        f"day={int(day_start_hour):02d}:00-{int(night_start_hour):02d}:00 local"
    )

    for subject in selected_subjects:
        subject_dir = artifact_dir_root / subject
        if not subject_dir.exists():
            raise ValueError(f"Yuquan subject directory missing: {subject_dir}")

        core_channels = None
        if core_channels_by_subject is not None:
            core_channels = core_channels_by_subject.get(subject)

        for lagpat_path in sorted(subject_dir.glob("*_lagPat.npz")):
            stem = lagpat_path.stem.removesuffix("_lagPat")
            packed_times_path = subject_dir / f"{stem}_packedTimes.npy"
            if not packed_times_path.exists():
                raise ValueError(
                    f"Missing packedTimes for subject={subject} block_stem={stem}: {packed_times_path}"
                )
            edf_path = subject_dir / f"{stem}.edf"
            if not edf_path.exists():
                raise ValueError(
                    f"Missing EDF for subject={subject} block_stem={stem}: {edf_path}"
                )
            record_info = read_edf_record_info(edf_path)
            block_start_epoch = float(record_info["start_epoch"])
            block_end_epoch = float(record_info["end_epoch"])
            result_metadata = {
                "subject": subject,
                "patient_code": subject,
                "recording_id": stem,
                "block_stem": stem,
                "block_start_epoch": block_start_epoch,
                "block_end_epoch": block_end_epoch,
                "block_start_day_night": _classify_day_night(
                    block_start_epoch,
                    timezone_name=timezone_name,
                    day_start_hour=day_start_hour,
                    night_start_hour=night_start_hour,
                ),
                "block_end_day_night": _classify_day_night(
                    block_end_epoch,
                    timezone_name=timezone_name,
                    day_start_hour=day_start_hour,
                    night_start_hour=night_start_hour,
                ),
                "timezone_name": timezone_name,
                "day_night_rule": day_night_rule,
            }

            result = build_interictal_synchrony_from_legacy_lagpat(
                str(lagpat_path),
                str(packed_times_path),
                core_channels=core_channels,
                metadata=result_metadata,
            )
            subject_out_dir = out_root / subject
            subject_out_dir.mkdir(parents=True, exist_ok=True)
            npz_path = subject_out_dir / f"{stem}_interictal_sync.npz"
            save_interictal_synchrony_result(result, str(npz_path))
            summary_rows.append(
                {
                    **result.metadata,
                    "lagpat_npz_path": str(lagpat_path),
                    "packed_times_path": str(packed_times_path),
                    "edf_path": str(edf_path),
                    "output_npz_path": str(npz_path),
                    **result.summary,
                    "summary_schema_version": BLOCK_SYNC_COMPAT_SCHEMA_VERSION,
                    "event_schema_version": EVENT_SYNC_SCHEMA_VERSION,
                }
            )
            event_rows.extend(build_event_rows_from_result(result))

    if event_rows_csv_path:
        save_interictal_synchrony_summary(event_rows, event_rows_csv_path)

    return summary_rows


def load_interictal_synchrony_result(npz_path: str) -> InterictalSynchronyResult:
    """Load a saved PR4 synchrony result."""
    d = np.load(npz_path, allow_pickle=True)
    return InterictalSynchronyResult(
        ch_names=[str(x) for x in np.asarray(d["ch_names"]).tolist()],
        event_windows=np.asarray(d["event_windows"], dtype=np.float64),
        core_mask=np.asarray(d["core_mask"], dtype=bool),
        event_n_participating=np.asarray(d["event_n_participating"], dtype=np.int64),
        event_n_core=np.asarray(d["event_n_core"], dtype=np.int64),
        event_n_penumbra=np.asarray(d["event_n_penumbra"], dtype=np.int64),
        event_stratum=np.asarray(d["event_stratum"], dtype=object),
        sync_phase_global=np.asarray(d["sync_phase_global"], dtype=np.float64),
        sync_phase_core=np.asarray(d["sync_phase_core"], dtype=np.float64),
        sync_phase_penumbra=np.asarray(d["sync_phase_penumbra"], dtype=np.float64),
        sync_legacy_global=np.asarray(d["sync_legacy_global"], dtype=np.float64),
        sync_legacy_core=np.asarray(d["sync_legacy_core"], dtype=np.float64),
        sync_legacy_penumbra=np.asarray(d["sync_legacy_penumbra"], dtype=np.float64),
        sync_span_global=np.asarray(d["sync_span_global"], dtype=np.float64),
        sync_span_core=np.asarray(d["sync_span_core"], dtype=np.float64),
        sync_span_penumbra=np.asarray(d["sync_span_penumbra"], dtype=np.float64),
        jaccard_global=np.asarray(d["jaccard_global"], dtype=np.float64),
        jaccard_core=np.asarray(d["jaccard_core"], dtype=np.float64),
        jaccard_penumbra=np.asarray(d["jaccard_penumbra"], dtype=np.float64),
        metadata=(
            dict(np.asarray(d["metadata"], dtype=object).ravel()[0])
            if "metadata" in d
            else {}
        ),
        params=dict(np.asarray(d["params"], dtype=object).ravel()[0]),
        summary=dict(np.asarray(d["summary"], dtype=object).ravel()[0]),
    )
