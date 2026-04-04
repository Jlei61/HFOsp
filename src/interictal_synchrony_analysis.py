"""
PR6: interval-first interictal synchrony statistics and visualization.

Consumes annotated PR5 event rows as raw inputs, but the formal hypothesis tests
operate at the seizure-interval level:
- fixed-window: mean within subject × seizure_interval × window_position, then paired test
- trajectory: per-interval Spearman rho, then one-sample test on rho distribution
- pooled event-level correlations are kept only as exploratory reference
- Figures A-E and interval-level audit tables
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
from scipy import stats as sp_stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from .preprocessing import epoch_to_local_hour

# ── constants ──────────────────────────────────────────────────────────────

SYNC_METRICS: Dict[str, str] = {
    "phase_all": "sync_phase_global",
    "phase_core": "sync_phase_core",
    "legacy_all": "sync_legacy_global",
    "legacy_core": "sync_legacy_core",
    "span_all": "sync_span_global",
    "span_core": "sync_span_core",
}

METRIC_FAMILIES: Dict[str, Tuple[str, str]] = {
    "phase": ("sync_phase_global", "sync_phase_core"),
    "legacy": ("sync_legacy_global", "sync_legacy_core"),
    "span": ("sync_span_global", "sync_span_core"),
}

DEFAULT_PRIMARY_METRIC = "sync_legacy_global"
DEFAULT_MIN_INTERVAL_SEC = 10800.0  # 3h
DEFAULT_WINDOW_SEC = 3600.0  # 1h
DEFAULT_N_BINS = 10

WINDOW_COLORS = {"post": "#4C72B0", "mid": "#55A868", "pre": "#C44E52"}
FIGURE_A_WINDOW_COLORS = {"post": WINDOW_COLORS["post"], "pre": WINDOW_COLORS["pre"]}
DATASET_COLORS = {"epilepsiae": "#17becf", "yuquan": "#ff7f0e"}
SEIZURE_COLOR = "#d62728"
OTHER_BLOCK_COLOR = "#7f7f7f"
DAY_BAR_COLOR = "#ffffff"
NIGHT_BAR_COLOR = "#111111"
FIGURE_A_FACET_HOURS = 12.0

# ── CSV I/O ────────────────────────────────────────────────────────────────

_FLOAT_COLS = frozenset({
    "event_index", "event_start_sec_rel", "event_end_sec_rel",
    "event_center_sec_rel", "event_duration_sec",
    "event_start_epoch", "event_end_epoch", "event_center_epoch",
    "n_participating", "n_core", "n_penumbra",
    "sync_phase_global", "sync_phase_core", "sync_phase_penumbra",
    "sync_legacy_global", "sync_legacy_core", "sync_legacy_penumbra",
    "sync_span_global", "sync_span_core", "sync_span_penumbra",
    "jaccard_global_with_next", "jaccard_core_with_next", "jaccard_penumbra_with_next",
    "block_start_epoch", "block_end_epoch", "block_duration_sec",
    "gap_from_prev_observed_sec",
    "continuous_segment_id",
    "prev_eeg_onset_epoch", "prev_eeg_offset_epoch",
    "next_eeg_onset_epoch", "next_eeg_offset_epoch",
    "clean_between_seizures_sec", "post_ictal_available_sec",
    "interictal_available_sec", "n_channels", "n_core_channels", "n_penumbra_channels",
    "sync_phase_i", "sync_legacy_i", "sync_span_i", "n_i",
    "sync_phase_l", "sync_legacy_l", "sync_span_l", "n_l",
    "sync_phase_e", "sync_legacy_e", "sync_span_e", "n_e",
})

_BOOL_COLS = frozenset({
    "phase_eligible", "diurnal_eligible",
    "starts_new_continuous_segment", "has_gap_before_event",
    "overlaps_complete_eeg_seizure",
})


def _parse_csv_value(key: str, raw: str) -> object:
    if key in _FLOAT_COLS:
        if raw in ("", "nan", "NaN", "None"):
            return None
        return float(raw)
    if key in _BOOL_COLS:
        return str(raw).strip().lower() == "true"
    return raw


def load_annotated_event_rows(
    csv_path: str | Path,
    *,
    dataset: str = "",
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        for raw_row in csv.DictReader(fh):
            parsed = {k: _parse_csv_value(k, v) for k, v in raw_row.items()}
            if dataset:
                parsed["dataset"] = dataset
            rows.append(parsed)
    return rows


def load_event_rows(
    csv_path: str | Path,
    *,
    dataset: str = "",
) -> List[Dict[str, object]]:
    """Load annotated PR4/PR5 event rows (raw inputs for interval-level PR6)."""
    return load_annotated_event_rows(csv_path, dataset=dataset)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return str(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


# ── helpers ────────────────────────────────────────────────────────────────


def _float_or_none(v: object) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _time_center(row: Mapping) -> Optional[float]:
    event_center = _float_or_none(row.get("event_center_epoch"))
    if event_center is not None:
        return event_center
    start = _float_or_none(row.get("event_start_epoch"))
    end = _float_or_none(row.get("event_end_epoch"))
    if start is not None and end is not None:
        return 0.5 * (start + end)
    return None


def _metric_value(row: Mapping, metric_col: str) -> Optional[float]:
    return _float_or_none(row.get(metric_col))


def _is_assigned(row: Mapping) -> bool:
    return str(row.get("interval_assignment_status")) == "assigned"


def _figure_a_window_position(
    row: Mapping,
    *,
    window_sec: float = DEFAULT_WINDOW_SEC,
) -> Optional[str]:
    """Highlight only onset-adjacent events for Figure A.

    Figure A is a full-timeline visualization, not the PR6 fixed-window test.
    We therefore color only the events close to a seizure onset:
    - `post`: within `window_sec` after the previous seizure onset
    - `pre`:  within `window_sec` before the next seizure onset

    If a short interval makes both windows overlap, pick the closer onset.
    """
    bc = _time_center(row)
    if bc is None:
        return None

    candidates: List[Tuple[float, str]] = []
    prev_onset = _float_or_none(row.get("prev_eeg_onset_epoch"))
    next_onset = _float_or_none(row.get("next_eeg_onset_epoch"))
    window_sec = float(window_sec)

    if prev_onset is not None:
        dt_post = bc - prev_onset
        if 0.0 <= dt_post <= window_sec:
            candidates.append((dt_post, "post"))
    if next_onset is not None:
        dt_pre = next_onset - bc
        if 0.0 <= dt_pre <= window_sec:
            candidates.append((dt_pre, "pre"))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], 0 if item[1] == "post" else 1))
    return candidates[0][1]


def _safe_median(arr: Sequence[float]) -> Optional[float]:
    if not arr:
        return None
    return float(np.median(arr))


def _metric_axis_label(metric_col: str) -> str:
    labels = {
        "sync_legacy_global": "Legacy synchronization",
        "sync_legacy_core": "Legacy synchronization, core only",
        "sync_phase_global": "Phase-order synchronization",
        "sync_phase_core": "Phase-order synchronization, core only",
        "sync_span_global": "Span synchronization",
        "sync_span_core": "Span synchronization, core only",
    }
    return labels.get(metric_col, metric_col.replace("sync_", "").replace("_", " "))


# ── fixed-window assignment ────────────────────────────────────────────────


def assign_fixed_window_positions(
    annotated_rows: Sequence[Mapping[str, object]],
    *,
    min_interval_sec: float = DEFAULT_MIN_INTERVAL_SEC,
    window_sec: float = DEFAULT_WINDOW_SEC,
) -> List[Dict[str, object]]:
    """
    Assign Post/Mid/Pre to events in intervals >= min_interval_sec.

    Post: event center in [clean_start, clean_start + window_sec]
    Pre:  event center in [clean_end - window_sec, clean_end]
    Mid:  event center in [midpoint - window_sec/2, midpoint + window_sec/2]
    """
    by_interval: Dict[str, List[Dict]] = defaultdict(list)
    for row in annotated_rows:
        if not _is_assigned(row):
            continue
        iv_id = str(row.get("seizure_interval_id", ""))
        if iv_id:
            by_interval[iv_id].append(dict(row))

    results: List[Dict[str, object]] = []
    for iv_id, blocks in by_interval.items():
        sample = blocks[0]
        clean_sec = _float_or_none(sample.get("clean_between_seizures_sec"))
        prev_offset = _float_or_none(sample.get("prev_eeg_offset_epoch"))
        next_onset = _float_or_none(sample.get("next_eeg_onset_epoch"))
        if clean_sec is None or clean_sec < min_interval_sec:
            continue
        if prev_offset is None or next_onset is None:
            continue

        clean_start = prev_offset
        clean_end = next_onset
        midpoint = (clean_start + clean_end) / 2.0

        post_boundary = clean_start + window_sec
        pre_boundary = clean_end - window_sec
        mid_lo = midpoint - window_sec / 2.0
        mid_hi = midpoint + window_sec / 2.0
        windows_overlap = post_boundary > mid_lo or mid_hi > pre_boundary

        for event in blocks:
            bc = _time_center(event)
            if bc is None:
                continue
            position: Optional[str] = None
            if clean_start <= bc <= post_boundary:
                position = "post"
            elif pre_boundary <= bc <= clean_end:
                position = "pre"
            elif mid_lo <= bc <= mid_hi:
                position = "mid"
            if position is None:
                continue
            out = dict(event)
            out["window_position"] = position
            out["windows_overlap"] = windows_overlap
            results.append(out)

    return results


# ── normalized trajectory ─────────────────────────────────────────────────


def compute_normalized_trajectory(
    annotated_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Compute norm_t in [0, 1] for each assigned event within its interval."""
    results: List[Dict[str, object]] = []
    for row in annotated_rows:
        if not _is_assigned(row):
            continue
        prev_offset = _float_or_none(row.get("prev_eeg_offset_epoch"))
        next_onset = _float_or_none(row.get("next_eeg_onset_epoch"))
        if prev_offset is None or next_onset is None:
            continue
        span = next_onset - prev_offset
        if span <= 0:
            continue
        bc = _time_center(row)
        if bc is None:
            continue
        norm_t = max(0.0, min(1.0, (bc - prev_offset) / span))
        out = dict(row)
        out["norm_t"] = norm_t
        results.append(out)
    return results


# ── statistics ─────────────────────────────────────────────────────────────


def _aggregate_to_interval_window_means(
    rows: Sequence[Mapping],
    metric_col: str,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Group by (subject, interval_id, window_position), average metric.
    Returns {(subject, interval_id): {window_position: mean_value}}.
    """
    accum: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for row in rows:
        val = _metric_value(row, metric_col)
        if val is None:
            continue
        key = (
            str(row["subject"]),
            str(row["seizure_interval_id"]),
            str(row["window_position"]),
        )
        accum[key].append(val)

    out: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    for (subj, iv_id, win_pos), values in accum.items():
        out[(subj, iv_id)][win_pos] = float(np.mean(values))
    return dict(out)


def paired_window_test(
    fixed_window_rows: Sequence[Mapping],
    *,
    metric_col: str = DEFAULT_PRIMARY_METRIC,
    pair: Tuple[str, str] = ("post", "pre"),
) -> Dict[str, object]:
    """
    Paired Wilcoxon signed-rank test between two window positions.

    Effect size: rank-biserial correlation r_rb = 1 - 2T / (n(n+1)/2).
    """
    interval_means = _aggregate_to_interval_window_means(
        fixed_window_rows, metric_col
    )
    a_name, b_name = pair
    a_vals: List[float] = []
    b_vals: List[float] = []
    paired_subjects: List[str] = []
    for (subj, _), windows in sorted(interval_means.items()):
        if a_name in windows and b_name in windows:
            a_vals.append(windows[a_name])
            b_vals.append(windows[b_name])
            paired_subjects.append(subj)

    result: Dict[str, object] = {
        "metric": metric_col,
        "pair": f"{a_name}_vs_{b_name}",
        "n_pairs": len(a_vals),
        "n_unique_subjects": len(set(paired_subjects)),
        "statistical_unit": "seizure_interval",
        "interval_summary": "mean_across_events_per_window",
    }

    if len(a_vals) < 2:
        result.update({
            "statistic": None, "p_value": None, "effect_size_r": None,
            "median_a": _safe_median(a_vals), "median_b": _safe_median(b_vals),
            "median_diff": None,
        })
        return result

    a_arr = np.array(a_vals, dtype=np.float64)
    b_arr = np.array(b_vals, dtype=np.float64)
    diff = b_arr - a_arr
    nonzero_diff = diff[diff != 0]

    if len(nonzero_diff) < 2:
        result.update({
            "statistic": None, "p_value": None, "effect_size_r": None,
            "median_a": float(np.median(a_arr)),
            "median_b": float(np.median(b_arr)),
            "median_diff": float(np.median(diff)),
        })
        return result

    wtest = sp_stats.wilcoxon(nonzero_diff, alternative="two-sided")
    n = len(nonzero_diff)
    T = float(wtest.statistic)
    R = n * (n + 1) / 2.0
    r_unsigned = 1.0 - 2.0 * T / R
    positive_dominant = np.sum(nonzero_diff > 0) >= np.sum(nonzero_diff < 0)
    r_effect = r_unsigned if positive_dominant else -r_unsigned

    result.update({
        "statistic": T,
        "p_value": float(wtest.pvalue),
        "effect_size_r": r_effect,
        "median_a": float(np.median(a_arr)),
        "median_b": float(np.median(b_arr)),
        "median_diff": float(np.median(diff)),
        "mean_a": float(np.mean(a_arr)),
        "mean_b": float(np.mean(b_arr)),
        "mean_diff": float(np.mean(diff)),
    })
    return result


def trajectory_trend_test(
    trajectory_rows: Sequence[Mapping],
    *,
    metric_col: str = DEFAULT_PRIMARY_METRIC,
    n_bins: int = DEFAULT_N_BINS,
) -> Dict[str, object]:
    """Pooled Spearman correlation of norm_t vs metric, plus binned summary.

    WARNING: this pools events across intervals and subjects, which can
    produce Simpson's-paradox artifacts. Prefer ``within_interval_trend_test``
    for hypothesis tests about within-interval dynamics.
    """
    norm_ts: List[float] = []
    metric_vals: List[float] = []
    for row in trajectory_rows:
        t = _float_or_none(row.get("norm_t"))
        v = _metric_value(row, metric_col)
        if t is not None and v is not None:
            norm_ts.append(t)
            metric_vals.append(v)

    result: Dict[str, object] = {
        "metric": metric_col,
        "n_events": len(norm_ts),
        "statistical_unit": "pooled_event_rows_exploratory",
        "is_exploratory": True,
    }
    if len(norm_ts) < 3:
        result.update({"spearman_r": None, "spearman_p": None, "bins": []})
        return result

    t_arr = np.array(norm_ts)
    v_arr = np.array(metric_vals)
    if np.ptp(t_arr) == 0 or np.ptp(v_arr) == 0:
        rho, p_val = float("nan"), float("nan")
    else:
        rho, p_val = sp_stats.spearmanr(t_arr, v_arr)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins: List[Dict[str, object]] = []
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (t_arr >= bin_edges[i]) & (t_arr < bin_edges[i + 1])
        else:
            mask = (t_arr >= bin_edges[i]) & (t_arr <= bin_edges[i + 1])
        bv = v_arr[mask]
        bins.append({
            "bin_center": float((bin_edges[i] + bin_edges[i + 1]) / 2),
            "n": int(mask.sum()),
            "mean": float(np.mean(bv)) if len(bv) > 0 else None,
            "sem": float(sp_stats.sem(bv)) if len(bv) > 1 else None,
            "median": float(np.median(bv)) if len(bv) > 0 else None,
            "q25": float(np.percentile(bv, 25)) if len(bv) > 0 else None,
            "q75": float(np.percentile(bv, 75)) if len(bv) > 0 else None,
        })

    result.update({
        "spearman_r": float(rho),
        "spearman_p": float(p_val),
        "bins": bins,
    })
    return result


def within_interval_trend_test(
    trajectory_rows: Sequence[Mapping],
    *,
    metric_col: str = DEFAULT_PRIMARY_METRIC,
    min_events_per_interval: int = 3,
) -> Dict[str, object]:
    """Per-interval Spearman ρ, then one-sample Wilcoxon on the ρ distribution.

    Avoids Simpson's paradox by testing within-interval trends independently,
    then aggregating the per-interval effect sizes. This is the correct test
    when the hypothesis operates at the interval level (e.g. sync resets after
    seizure and rebuilds toward the next).
    """
    by_interval: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)
    for row in trajectory_rows:
        t = _float_or_none(row.get("norm_t"))
        v = _metric_value(row, metric_col)
        iv_id = str(row.get("seizure_interval_id", ""))
        subj = str(row.get("subject", ""))
        if t is not None and v is not None and iv_id:
            by_interval[(subj, iv_id)].append((t, v))

    per_interval_rho: List[float] = []
    per_interval_details: List[Dict[str, object]] = []
    for (subj, iv_id), points in sorted(by_interval.items()):
        if len(points) < min_events_per_interval:
            continue
        ts = np.array([p[0] for p in points])
        vs = np.array([p[1] for p in points])
        if np.ptp(ts) == 0 or np.ptp(vs) == 0:
            continue
        rho, p = sp_stats.spearmanr(ts, vs)
        if not np.isfinite(rho):
            continue
        per_interval_rho.append(float(rho))
        per_interval_details.append({
            "subject": subj,
            "seizure_interval_id": iv_id,
            "n_events": len(points),
            "rho": float(rho),
            "p": float(p),
        })

    result: Dict[str, object] = {
        "metric": metric_col,
        "n_intervals_tested": len(per_interval_rho),
        "min_events_per_interval": min_events_per_interval,
        "statistical_unit": "seizure_interval",
        "interval_effect": "spearman_rho_within_interval",
    }

    if len(per_interval_rho) < 3:
        result.update({
            "median_rho": None, "mean_rho": None, "std_rho": None,
            "wilcoxon_p": None, "n_positive": None, "n_negative": None,
        })
        return result

    rho_arr = np.array(per_interval_rho)
    n_pos = int(np.sum(rho_arr > 0))
    n_neg = int(np.sum(rho_arr < 0))
    nonzero = rho_arr[rho_arr != 0]
    w_p: Optional[float] = None
    if len(nonzero) >= 2:
        wtest = sp_stats.wilcoxon(nonzero, alternative="two-sided")
        w_p = float(wtest.pvalue)

    result.update({
        "median_rho": float(np.median(rho_arr)),
        "mean_rho": float(np.mean(rho_arr)),
        "std_rho": float(np.std(rho_arr)),
        "wilcoxon_p": w_p,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_zero": int(np.sum(rho_arr == 0)),
        "per_interval_details": per_interval_details,
    })
    return result


def event_rate_paired_window_test(
    fixed_window_rows: Sequence[Mapping],
    *,
    window_sec: float = DEFAULT_WINDOW_SEC,
    pair: Tuple[str, str] = ("post", "pre"),
) -> Dict[str, object]:
    """Paired Wilcoxon on event rate (events/hour) per interval-window."""
    count: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for row in fixed_window_rows:
        key = (
            str(row["subject"]),
            str(row["seizure_interval_id"]),
            str(row["window_position"]),
        )
        count[key] += 1

    interval_rates: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    rate_scale = 3600.0 / float(window_sec)
    for (subj, iv_id, win_pos), n in count.items():
        interval_rates[(subj, iv_id)][win_pos] = float(n) * rate_scale

    a_name, b_name = pair
    a_vals: List[float] = []
    b_vals: List[float] = []
    for (subj, iv_id), windows in sorted(interval_rates.items()):
        if a_name in windows and b_name in windows:
            a_vals.append(windows[a_name])
            b_vals.append(windows[b_name])

    result: Dict[str, object] = {
        "metric": "event_rate_per_hour",
        "pair": f"{a_name}_vs_{b_name}",
        "n_pairs": len(a_vals),
        "statistical_unit": "seizure_interval",
        "window_sec": window_sec,
    }
    if len(a_vals) < 2:
        result.update({
            "statistic": None, "p_value": None,
            "median_a": _safe_median(a_vals), "median_b": _safe_median(b_vals),
        })
        return result

    a_arr = np.array(a_vals)
    b_arr = np.array(b_vals)
    diff = b_arr - a_arr
    nonzero = diff[diff != 0]
    if len(nonzero) < 2:
        result.update({
            "statistic": None, "p_value": None,
            "median_a": float(np.median(a_arr)),
            "median_b": float(np.median(b_arr)),
            "median_diff": float(np.median(diff)),
        })
        return result

    wtest = sp_stats.wilcoxon(nonzero, alternative="two-sided")
    result.update({
        "statistic": float(wtest.statistic),
        "p_value": float(wtest.pvalue),
        "median_a": float(np.median(a_arr)),
        "median_b": float(np.median(b_arr)),
        "median_diff": float(np.median(diff)),
        "mean_a": float(np.mean(a_arr)),
        "mean_b": float(np.mean(b_arr)),
    })
    return result


def _detect_available_region_metrics(
    rows: Sequence[Mapping],
) -> Dict[str, Tuple[str, ...]]:
    """Discover which region-stratified columns have data (i/l/e)."""
    region_families: Dict[str, Tuple[str, ...]] = {}
    step = max(1, len(rows) // 500)
    sample = rows[::step]
    for region in ("i", "l", "e"):
        col = f"sync_phase_{region}"
        has_data = any(_float_or_none(r.get(col)) is not None for r in sample)
        if has_data:
            region_families[f"phase_{region}"] = (f"sync_phase_{region}",)
    return region_families


# ── cohort summary ─────────────────────────────────────────────────────────


def build_cohort_summary(
    annotated_rows: Sequence[Mapping],
    fixed_window_rows: Sequence[Mapping],
    trajectory_rows: Sequence[Mapping],
) -> Dict[str, object]:
    subjects_all = {str(r["subject"]) for r in annotated_rows}
    subjects_assigned = {
        str(r["subject"]) for r in annotated_rows if _is_assigned(r)
    }
    intervals_assigned = {
        str(r["seizure_interval_id"])
        for r in annotated_rows
        if _is_assigned(r) and r.get("seizure_interval_id")
    }
    intervals_fixed = {
        str(r["seizure_interval_id"]) for r in fixed_window_rows
    }
    subjects_fixed = {str(r["subject"]) for r in fixed_window_rows}

    exclusion_counts: Counter[str] = Counter()
    for r in annotated_rows:
        reasons = str(r.get("exclusion_reasons", ""))
        for part in reasons.split("|"):
            if part.strip():
                exclusion_counts[part.strip()] += 1

    fw_by_pos: Counter[str] = Counter()
    for r in fixed_window_rows:
        fw_by_pos[str(r["window_position"])] += 1

    return {
        "n_events_total": len(annotated_rows),
        "n_events_assigned": sum(1 for r in annotated_rows if _is_assigned(r)),
        "n_subjects_total": len(subjects_all),
        "n_subjects_with_intervals": len(subjects_assigned),
        "n_subjects_with_fixed_windows": len(subjects_fixed),
        "n_intervals_assigned": len(intervals_assigned),
        "n_intervals_fixed_window": len(intervals_fixed),
        "n_fixed_window_events": len(fixed_window_rows),
        "fixed_window_events_by_position": dict(fw_by_pos),
        "n_trajectory_events": len(trajectory_rows),
        "exclusion_reason_counts": dict(exclusion_counts),
    }


def build_fixed_window_interval_means_table(
    fixed_window_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Materialize the PR6 fixed-window analysis unit: one row per interval."""
    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in fixed_window_rows:
        subject = str(row.get("subject", ""))
        interval_id = str(row.get("seizure_interval_id", ""))
        if subject and interval_id:
            grouped[(subject, interval_id)].append(row)

    metric_cols = sorted(set(SYNC_METRICS.values()))
    out: List[Dict[str, object]] = []
    for (subject, interval_id), rows in sorted(grouped.items()):
        per_window_rows: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
        for row in rows:
            per_window_rows[str(row["window_position"])].append(row)

        sample = rows[0]
        result: Dict[str, object] = {
            "subject": subject,
            "dataset": str(sample.get("dataset", "")),
            "seizure_interval_id": interval_id,
            "clean_between_seizures_sec": _float_or_none(
                sample.get("clean_between_seizures_sec")
            ),
        }
        for position in ("post", "mid", "pre"):
            window_rows = per_window_rows.get(position, [])
            result[f"{position}_n_events"] = len(window_rows)
            for metric_col in metric_cols:
                vals = [
                    val
                    for val in (_metric_value(r, metric_col) for r in window_rows)
                    if val is not None
                ]
                result[f"{metric_col}_{position}_mean"] = (
                    float(np.mean(vals)) if vals else None
                )
        out.append(result)
    return out


def build_trajectory_interval_stats_table(
    trajectory_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Materialize the PR6 trajectory analysis unit: one row per interval."""
    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in trajectory_rows:
        subject = str(row.get("subject", ""))
        interval_id = str(row.get("seizure_interval_id", ""))
        if subject and interval_id:
            grouped[(subject, interval_id)].append(row)

    metric_cols = sorted(set(SYNC_METRICS.values()))
    out: List[Dict[str, object]] = []
    for (subject, interval_id), rows in sorted(grouped.items()):
        sample = rows[0]
        result: Dict[str, object] = {
            "subject": subject,
            "dataset": str(sample.get("dataset", "")),
            "seizure_interval_id": interval_id,
            "n_events": len(rows),
            "clean_between_seizures_sec": _float_or_none(
                sample.get("clean_between_seizures_sec")
            ),
        }
        ts = np.array(
            [
                t
                for t in (_float_or_none(row.get("norm_t")) for row in rows)
                if t is not None
            ],
            dtype=np.float64,
        )
        for metric_col in metric_cols:
            paired = [
                (t, v)
                for row in rows
                for t, v in [(_float_or_none(row.get("norm_t")), _metric_value(row, metric_col))]
                if t is not None and v is not None
            ]
            result[f"{metric_col}_n_events"] = len(paired)
            if len(paired) < 3:
                result[f"{metric_col}_spearman_rho"] = None
                result[f"{metric_col}_spearman_p"] = None
                continue
            metric_ts = np.array([t for t, _ in paired], dtype=np.float64)
            metric_vs = np.array([v for _, v in paired], dtype=np.float64)
            if np.ptp(metric_ts) == 0 or np.ptp(metric_vs) == 0:
                result[f"{metric_col}_spearman_rho"] = None
                result[f"{metric_col}_spearman_p"] = None
                continue
            rho, p_val = sp_stats.spearmanr(metric_ts, metric_vs)
            result[f"{metric_col}_spearman_rho"] = float(rho)
            result[f"{metric_col}_spearman_p"] = float(p_val)
        out.append(result)
    return out


# ── figures ────────────────────────────────────────────────────────────────


def _extract_seizure_epochs(
    annotated_rows: Sequence[Mapping],
    subject: str,
) -> List[Tuple[float, float]]:
    """Recover unique seizure (onset, offset) from annotations for one subject."""
    seen: Dict[str, Tuple[float, float]] = {}
    for r in annotated_rows:
        if str(r.get("subject")) != subject:
            continue
        for prefix in ("prev", "next"):
            onset = _float_or_none(r.get(f"{prefix}_eeg_onset_epoch"))
            offset = _float_or_none(r.get(f"{prefix}_eeg_offset_epoch"))
            sz_id = str(r.get(f"{prefix}_seizure_id", ""))
            if onset is not None and offset is not None and sz_id:
                seen[sz_id] = (onset, offset)
    return sorted(seen.values())


def _classify_day_night(hour: int, *, day_start_hour: int = 8, night_start_hour: int = 20) -> str:
    return "day" if int(day_start_hour) <= int(hour) < int(night_start_hour) else "night"


def _next_day_night_transition_epoch(
    epoch_ts: float,
    timezone_name: str,
    *,
    day_start_hour: int = 8,
    night_start_hour: int = 20,
) -> float:
    dt = datetime.fromtimestamp(float(epoch_ts), ZoneInfo(str(timezone_name)))
    label = _classify_day_night(
        epoch_to_local_hour(float(epoch_ts), str(timezone_name)),
        day_start_hour=day_start_hour,
        night_start_hour=night_start_hour,
    )
    if label == "day":
        next_dt = dt.replace(hour=int(night_start_hour), minute=0, second=0, microsecond=0)
        if next_dt <= dt:
            next_dt = next_dt + timedelta(days=1)
    else:
        next_dt = dt.replace(hour=int(day_start_hour), minute=0, second=0, microsecond=0)
        if next_dt <= dt:
            next_dt = next_dt + timedelta(days=1)
    return float(next_dt.timestamp())


def _build_day_night_segments(
    start_epoch: float,
    end_epoch: float,
    timezone_name: str,
    *,
    day_start_hour: int = 8,
    night_start_hour: int = 20,
) -> List[Tuple[float, float, str]]:
    segments: List[Tuple[float, float, str]] = []
    cursor = float(start_epoch)
    end_epoch = float(end_epoch)
    while cursor < end_epoch:
        label = _classify_day_night(
            epoch_to_local_hour(cursor, timezone_name),
            day_start_hour=day_start_hour,
            night_start_hour=night_start_hour,
        )
        next_boundary = _next_day_night_transition_epoch(
            cursor + 1e-6,
            timezone_name,
            day_start_hour=day_start_hour,
            night_start_hour=night_start_hour,
        )
        seg_end = min(end_epoch, next_boundary)
        segments.append((cursor, seg_end, label))
        cursor = seg_end
    return segments


def _prepare_figure_a_subject_context(
    annotated_rows: Sequence[Mapping],
    subject: str,
    *,
    metric_col: str = DEFAULT_PRIMARY_METRIC,
    metric_label: str = "",
    facet_hours: float = FIGURE_A_FACET_HOURS,
    highlight_window_sec: float = DEFAULT_WINDOW_SEC,
) -> Dict[str, object]:
    subject_rows = [
        r for r in annotated_rows if str(r.get("subject")) == subject
    ]
    events = [
        r
        for r in subject_rows
        if _is_assigned(r) and not bool(r.get("overlaps_complete_eeg_seizure"))
    ]
    events.sort(key=lambda r: float(_time_center(r) or 0.0))
    seizures = _extract_seizure_epochs(subject_rows, subject)

    if events:
        origin = min(float(_time_center(r) or 0.0) for r in events)
    elif seizures:
        origin = min(onset for onset, _ in seizures)
    else:
        origin = 0.0

    max_hour = 0.0
    if events:
        max_hour = max(
            max_hour,
            max((float(_time_center(r) or origin) - origin) / 3600.0 for r in events),
        )
    if seizures:
        max_hour = max(max_hour, max((offset - origin) / 3600.0 for _, offset in seizures))
    facet_hours = float(facet_hours)
    x_limit = max(facet_hours, float(np.ceil(max_hour / facet_hours) * facet_hours))
    n_facets = max(1, int(np.ceil(x_limit / facet_hours)))

    xs, ys = [], []
    colors = []
    by_segment: Dict[Tuple[str, int], List[Tuple[float, float]]] = defaultdict(list)
    for event in events:
        bc = _time_center(event)
        val = _metric_value(event, metric_col)
        if bc is None or val is None:
            continue
        x_rel = (bc - origin) / 3600.0
        xs.append(x_rel)
        ys.append(val)
        pos = _figure_a_window_position(event, window_sec=highlight_window_sec)
        colors.append(FIGURE_A_WINDOW_COLORS.get(str(pos), OTHER_BLOCK_COLOR))
        segment_key = (
            str(event.get("seizure_interval_id", "")),
            int(_float_or_none(event.get("continuous_segment_id")) or 0),
        )
        by_segment[segment_key].append((x_rel, val))

    if ys:
        y_min = float(min(ys))
        y_max = float(max(ys))
    else:
        y_min, y_max = 0.0, 1.0
    y_pad = max(0.02, 0.08 * max(y_max - y_min, 1e-6))
    bar_h = max(0.012, 0.05 * max(y_max - y_min, 1e-6))
    bar_y = y_min - y_pad - bar_h
    y_lo = bar_y - 0.6 * y_pad
    y_hi = y_max + 0.6 * y_pad

    timezone_name = next((str(r.get("timezone_name", "")) for r in subject_rows if r.get("timezone_name")), "")
    day_night_segments: List[Tuple[float, float, str]] = []
    if timezone_name:
        day_night_segments = [
            ((seg_start - origin) / 3600.0, (seg_end - origin) / 3600.0, label)
            for seg_start, seg_end, label in _build_day_night_segments(
                origin,
                origin + x_limit * 3600.0,
                timezone_name,
            )
        ]

    return {
        "subject": subject,
        "subject_rows": subject_rows,
        "seizures": seizures,
        "xs": xs,
        "ys": ys,
        "colors": colors,
        "by_segment": by_segment,
        "origin": origin,
        "x_limit": x_limit,
        "n_facets": n_facets,
        "facet_hours": facet_hours,
        "y_lo": y_lo,
        "y_hi": y_hi,
        "bar_y": bar_y,
        "bar_h": bar_h,
        "label": metric_label or _metric_axis_label(metric_col),
        "day_night_segments": day_night_segments,
    }


def _render_figure_a_panel(
    ax: plt.Axes,
    ctx: Mapping[str, object],
    facet_idx: int,
    *,
    show_title: bool,
    show_xlabel: bool,
) -> None:
    subject = str(ctx["subject"])
    facet_hours = float(ctx["facet_hours"])
    x_limit = float(ctx["x_limit"])
    x0 = facet_idx * facet_hours
    x1 = min(x_limit, x0 + facet_hours)
    origin = float(ctx["origin"])
    seizures = ctx["seizures"]
    xs = ctx["xs"]
    ys = ctx["ys"]
    colors = ctx["colors"]
    by_segment = ctx["by_segment"]
    day_night_segments = ctx["day_night_segments"]

    for onset, offset in seizures:
        onset_h = (float(onset) - origin) / 3600.0
        offset_h = (float(offset) - origin) / 3600.0
        if offset_h <= x0 or onset_h >= x1:
            continue
        ax.axvspan(max(onset_h, x0), min(offset_h, x1), color=SEIZURE_COLOR, alpha=0.14, zorder=0)
        if x0 <= onset_h <= x1:
            ax.axvline(onset_h, color=SEIZURE_COLOR, linewidth=1.2, alpha=0.9, zorder=1)

    facet_pts = [(x, y, c) for x, y, c in zip(xs, ys, colors) if x0 <= x <= x1]
    if facet_pts:
        ax.scatter(
            [p[0] for p in facet_pts],
            [p[1] for p in facet_pts],
            c=[p[2] for p in facet_pts],
            s=34,
            zorder=3,
            edgecolors="k",
            linewidths=0.2,
        )
    for pts in by_segment.values():
        seg_pts = sorted([p for p in pts if x0 <= p[0] <= x1], key=lambda x: x[0])
        if len(seg_pts) <= 1:
            continue
        ax.plot(
            [p[0] for p in seg_pts],
            [p[1] for p in seg_pts],
            color="#5a5a5a",
            linewidth=0.85,
            alpha=0.60,
            zorder=2,
        )

    bar_y = float(ctx["bar_y"])
    bar_h = float(ctx["bar_h"])
    for seg_start_h, seg_end_h, label in day_night_segments:
        if seg_end_h <= x0 or seg_start_h >= x1:
            continue
        left = max(seg_start_h, x0)
        right = min(seg_end_h, x1)
        ax.broken_barh(
            [(left, right - left)],
            (bar_y, bar_h),
            facecolors=DAY_BAR_COLOR if label == "day" else NIGHT_BAR_COLOR,
            edgecolors="k",
            linewidth=0.8,
            zorder=4,
            clip_on=False,
        )
        if (right - left) >= 1.2:
            ax.text(
                0.5 * (left + right),
                bar_y + 0.5 * bar_h,
                "Day" if label == "day" else "Night",
                ha="center",
                va="center",
                fontsize=8,
                color="k" if label == "day" else "w",
                zorder=5,
            )

    ax.set_xlim(x0, x1)
    ax.set_ylim(float(ctx["y_lo"]), float(ctx["y_hi"]))
    ax.xaxis.set_major_locator(MultipleLocator(2.0))
    ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax.grid(axis="x", which="major", color="#d0d0d0", linewidth=0.6, alpha=0.6)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.text(
        0.01,
        0.95,
        f"{x0:.0f}-{x1:.0f} h",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#444444",
    )
    if show_title:
        ax.set_title(f"{subject} | {ctx['label']}", fontsize=15)
    ax.set_xlabel("Hours from first clean assigned event" if show_xlabel else "", fontsize=13)
    ax.set_ylabel(str(ctx["label"]), fontsize=12)


def plot_figure_a_subject_timeline(
    annotated_rows: Sequence[Mapping],
    subject: str,
    *,
    metric_col: str = DEFAULT_PRIMARY_METRIC,
    metric_label: str = "",
    facet_index: Optional[int] = None,
    facet_hours: float = FIGURE_A_FACET_HOURS,
    highlight_window_sec: float = DEFAULT_WINDOW_SEC,
) -> Figure:
    """Figure A: event timeline; optionally render one 12h panel only."""
    ctx = _prepare_figure_a_subject_context(
        annotated_rows,
        subject,
        metric_col=metric_col,
        metric_label=metric_label,
        facet_hours=facet_hours,
        highlight_window_sec=highlight_window_sec,
    )
    n_facets = int(ctx["n_facets"])
    if facet_index is not None:
        if facet_index < 0 or facet_index >= n_facets:
            raise ValueError(f"facet_index {facet_index} out of range for {n_facets} facets.")
        fig, ax = plt.subplots(1, 1, figsize=(14.0, 4.2))
        _render_figure_a_panel(ax, ctx, int(facet_index), show_title=True, show_xlabel=True)
        legend_anchor_ax = ax
    else:
        fig, axes = plt.subplots(
            n_facets,
            1,
            figsize=(14.0, max(4.8, 3.2 * n_facets)),
            sharey=True,
            squeeze=False,
        )
        axes_list = list(axes[:, 0])
        for idx, ax in enumerate(axes_list):
            _render_figure_a_panel(
                ax,
                ctx,
                idx,
                show_title=(idx == 0),
                show_xlabel=(idx == n_facets - 1),
            )
        legend_anchor_ax = axes_list[0]

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=WINDOW_COLORS["post"],
               markeredgecolor="k", markeredgewidth=0.35, markersize=7,
               label=f"Post event (<={highlight_window_sec / 3600:.0f}h after onset)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=WINDOW_COLORS["pre"],
               markeredgecolor="k", markeredgewidth=0.35, markersize=7,
               label=f"Pre event (<={highlight_window_sec / 3600:.0f}h before onset)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=OTHER_BLOCK_COLOR,
               markeredgecolor="k", markeredgewidth=0.35, markersize=7, label="Other clean assigned event"),
        Line2D([0], [0], color=SEIZURE_COLOR, linewidth=1.2, label="Seizure onset"),
        Patch(facecolor=SEIZURE_COLOR, alpha=0.14, label="Seizure duration"),
        Patch(facecolor=DAY_BAR_COLOR, edgecolor="k", label="Day"),
        Patch(facecolor=NIGHT_BAR_COLOR, edgecolor="k", label="Night"),
    ]
    legend_anchor_ax.legend(handles=legend_handles, loc="upper left", fontsize=9, frameon=False, ncol=4)
    fig.tight_layout()
    return fig


def _two_level_binned_ribbon(
    trajectory_rows: Sequence[Mapping],
    metric_col: str,
    n_bins: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Compute bin medians via two-level averaging (per-interval, then across).

    Returns (centers, medians, q25s, q75s). For bins where fewer than 2
    intervals contribute, IQR values are None-filled.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)

    by_interval: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for r in trajectory_rows:
        t = _float_or_none(r.get("norm_t"))
        v = _metric_value(r, metric_col)
        iv_id = str(r.get("seizure_interval_id", ""))
        if t is not None and v is not None and iv_id:
            by_interval[iv_id].append((t, v))

    centers: List[float] = []
    medians: List[float] = []
    q25s: List[float] = []
    q75s: List[float] = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        c = float((lo + hi) / 2.0)
        interval_bin_means: List[float] = []
        for pts in by_interval.values():
            if i < n_bins - 1:
                bv = [v for t, v in pts if lo <= t < hi]
            else:
                bv = [v for t, v in pts if lo <= t <= hi]
            if bv:
                interval_bin_means.append(float(np.mean(bv)))
        if interval_bin_means:
            centers.append(c)
            medians.append(float(np.median(interval_bin_means)))
            if len(interval_bin_means) >= 4:
                q25s.append(float(np.percentile(interval_bin_means, 25)))
                q75s.append(float(np.percentile(interval_bin_means, 75)))
            else:
                q25s.append(float(np.median(interval_bin_means)))
                q75s.append(float(np.median(interval_bin_means)))

    return centers, medians, q25s, q75s


def plot_figure_b_trajectory_ribbon(
    trajectory_rows: Sequence[Mapping],
    *,
    metric_col: str = DEFAULT_PRIMARY_METRIC,
    n_bins: int = DEFAULT_N_BINS,
    metric_label: str = "",
    title_suffix: str = "",
    within_interval_stat: Optional[Dict[str, object]] = None,
) -> Figure:
    """Figure B: interval-aware normalized trajectory ribbon with median ± IQR.

    The ribbon is computed via two-level averaging (per-interval means first,
    then median across intervals) to avoid Simpson's-paradox artifacts.
    Annotations show both the within-interval test (primary) and the pooled
    event-level Spearman only as exploratory reference.
    """
    by_interval: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for r in trajectory_rows:
        t = _float_or_none(r.get("norm_t"))
        v = _metric_value(r, metric_col)
        iv_id = str(r.get("seizure_interval_id", ""))
        if t is not None and v is not None and iv_id:
            by_interval[iv_id].append((t, v))

    fig, ax = plt.subplots(figsize=(8, 5))

    for iv_id, pts in by_interval.items():
        pts.sort()
        ts = [p[0] for p in pts]
        vs = [p[1] for p in pts]
        ax.plot(ts, vs, color="#cccccc", linewidth=0.5, alpha=0.5, zorder=1)

    centers, medians, q25s, q75s = _two_level_binned_ribbon(
        trajectory_rows, metric_col, n_bins
    )
    if centers:
        if len(centers) == len(q25s) == len(q75s):
            ax.fill_between(centers, q25s, q75s, color="#4C72B0", alpha=0.25, zorder=2)
        ax.plot(centers, medians, color="#4C72B0", linewidth=2.0, zorder=3, label="median")

    annotation_lines: List[str] = []
    if within_interval_stat is not None:
        med_rho = within_interval_stat.get("median_rho")
        w_p = within_interval_stat.get("wilcoxon_p")
        n_iv = within_interval_stat.get("n_intervals_tested", 0)
        n_pos = within_interval_stat.get("n_positive", 0)
        n_neg = within_interval_stat.get("n_negative", 0)
        if med_rho is not None:
            annotation_lines.append(
                f"within-interval: median ρ={med_rho:+.3f}, "
                f"p={w_p:.2e}, +/{n_pos} −/{n_neg} (n={n_iv})"
            )
    pooled = trajectory_trend_test(
        trajectory_rows, metric_col=metric_col, n_bins=n_bins
    )
    rho = pooled.get("spearman_r")
    p = pooled.get("spearman_p")
    if rho is not None:
        annotation_lines.append(f"pooled event ref (exploratory): ρ={rho:.3f}, p={p:.2e}")

    if annotation_lines:
        ax.text(
            0.02, 0.98,
            "\n".join(annotation_lines),
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.8),
        )

    label = metric_label or metric_col.replace("sync_", "").replace("_", " ")
    ax.set_xlabel("normalized interval position")
    ax.set_ylabel(label)
    ax.set_xlim(0, 1)
    title = "Normalized interictal trajectory"
    if title_suffix:
        title += f"  ({title_suffix})"
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_figure_c_fixed_window_comparison(
    fixed_window_rows: Sequence[Mapping],
    *,
    metric_col_all: str = "sync_legacy_global",
    metric_col_core: str = "sync_legacy_core",
    metric_label: str = "legacy sync",
) -> Figure:
    """Figure C: interval-mean Post/Mid/Pre comparison; all vs core side by side."""
    positions = ("post", "mid", "pre")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax_idx, (metric_col, col_label) in enumerate([
        (metric_col_all, "sync_all (global)"),
        (metric_col_core, "sync_core_only"),
    ]):
        ax = axes[ax_idx]
        interval_means = _aggregate_to_interval_window_means(
            fixed_window_rows, metric_col
        )
        data_by_pos: Dict[str, List[float]] = {p: [] for p in positions}
        paired: List[Dict[str, float]] = []
        for (_, _), windows in sorted(interval_means.items()):
            for pos in positions:
                if pos in windows:
                    data_by_pos[pos].append(windows[pos])
            if all(pos in windows for pos in positions):
                paired.append(windows)

        bp_data = [data_by_pos[p] for p in positions]
        bp = ax.boxplot(
            bp_data,
            positions=list(range(len(positions))),
            widths=0.5,
            patch_artist=True,
            showfliers=False,
        )
        for patch, pos in zip(bp["boxes"], positions):
            patch.set_facecolor(WINDOW_COLORS[pos])
            patch.set_alpha(0.5)

        for interval_vals in paired:
            y_vals = [interval_vals[p] for p in positions]
            ax.plot(
                range(len(positions)), y_vals,
                color="#888888", linewidth=0.6, alpha=0.4, zorder=1,
            )

        for i, pos in enumerate(positions):
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(data_by_pos[pos]))
            ax.scatter(
                i + jitter, data_by_pos[pos],
                color=WINDOW_COLORS[pos], s=18, alpha=0.6, zorder=2,
                edgecolors="k", linewidths=0.3,
            )

        stat = paired_window_test(
            fixed_window_rows, metric_col=metric_col, pair=("post", "pre")
        )
        p_val = stat.get("p_value")
        if p_val is not None:
            y_max = max(
                (max(vals) for vals in data_by_pos.values() if vals), default=1.0
            )
            ax.annotate(
                f"p={p_val:.3e}" if p_val < 0.001 else f"p={p_val:.3f}",
                xy=(1, y_max * 1.02), fontsize=8, ha="center",
            )

        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels([p.capitalize() for p in positions])
        ax.set_title(col_label)

    axes[0].set_ylabel(metric_label)
    fig.suptitle("Fixed-window Post / Mid / Pre comparison (interval means)", fontsize=12)
    fig.tight_layout()
    return fig


def plot_figure_d_robustness(
    fixed_window_rows: Sequence[Mapping],
    *,
    datasets: Sequence[str] = ("epilepsiae", "yuquan", "combined"),
) -> Figure:
    """Figure D: metric families × datasets; Post-vs-Pre effect bars."""
    families = list(METRIC_FAMILIES.keys())
    n_rows = len(families)
    n_cols = len(datasets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col_idx, ds_label in enumerate(datasets):
        if ds_label == "combined":
            ds_rows = list(fixed_window_rows)
        else:
            ds_rows = [
                r for r in fixed_window_rows
                if str(r.get("dataset", "")) == ds_label
            ]
        for row_idx, family in enumerate(families):
            ax = axes[row_idx, col_idx]
            metric_all, metric_core = METRIC_FAMILIES[family]
            bar_labels = ["all", "core"]
            bar_vals: List[float] = []
            bar_errs: List[float] = []
            bar_pvals: List[Optional[float]] = []

            for metric_col in (metric_all, metric_core):
                stat = paired_window_test(
                    ds_rows, metric_col=metric_col, pair=("post", "pre")
                )
                md = _float_or_none(stat.get("median_diff"))
                bar_vals.append(md if md is not None else 0.0)
                bar_errs.append(0.0)
                bar_pvals.append(_float_or_none(stat.get("p_value")))

            x = np.arange(len(bar_labels))
            bars = ax.bar(
                x, bar_vals, width=0.5,
                color=["#4C72B0", "#C44E52"], alpha=0.7,
            )
            for i, pv in enumerate(bar_pvals):
                if pv is not None:
                    label = f"p={pv:.2e}" if pv < 0.01 else f"p={pv:.2f}"
                    ax.text(x[i], bar_vals[i], label, ha="center", va="bottom", fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels(bar_labels, fontsize=8)
            ax.axhline(0, color="k", linewidth=0.5)
            if col_idx == 0:
                ax.set_ylabel(f"{family}\nmedian(Pre−Post)")
            if row_idx == 0:
                ax.set_title(ds_label, fontsize=10)

    fig.suptitle("Robustness: Pre−Post effect across metrics and datasets", fontsize=12)
    fig.tight_layout()
    return fig


def plot_figure_e_coverage_audit(
    annotated_rows: Sequence[Mapping],
    fixed_window_rows: Sequence[Mapping],
    trajectory_rows: Sequence[Mapping],
) -> Figure:
    """Figure E: cohort coverage bar chart + exclusion pie."""
    summary = build_cohort_summary(annotated_rows, fixed_window_rows, trajectory_rows)

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(13, 5))

    categories = [
        ("Events total", summary["n_events_total"]),
        ("Events assigned", summary["n_events_assigned"]),
        ("Fixed-window\nevents", summary["n_fixed_window_events"]),
        ("Trajectory\nevents", summary["n_trajectory_events"]),
        ("Subjects total", summary["n_subjects_total"]),
        ("Subjects w/\nintervals", summary["n_subjects_with_intervals"]),
        ("Subjects w/\nfixed windows", summary["n_subjects_with_fixed_windows"]),
        ("Intervals\nassigned", summary["n_intervals_assigned"]),
        ("Intervals\nfixed window", summary["n_intervals_fixed_window"]),
    ]
    labels = [c[0] for c in categories]
    values = [c[1] for c in categories]
    colors_bar = ["#4C72B0"] * 4 + ["#55A868"] * 3 + ["#C44E52"] * 2
    ax_bar.barh(range(len(labels)), values, color=colors_bar, alpha=0.7)
    ax_bar.set_yticks(range(len(labels)))
    ax_bar.set_yticklabels(labels, fontsize=8)
    ax_bar.set_xlabel("count")
    ax_bar.set_title("Cohort coverage")
    for i, v in enumerate(values):
        ax_bar.text(v + 0.5, i, str(v), va="center", fontsize=8)

    excl = summary.get("exclusion_reason_counts", {})
    if excl:
        pie_labels = list(excl.keys())
        pie_sizes = list(excl.values())
        wedges, texts, autotexts = ax_pie.pie(
            pie_sizes, labels=pie_labels, autopct="%1.0f%%",
            textprops={"fontsize": 7},
        )
        ax_pie.set_title("Event exclusion reasons")
    else:
        ax_pie.text(0.5, 0.5, "No exclusions", ha="center", va="center")
        ax_pie.set_title("Event exclusion reasons")

    fig.tight_layout()
    return fig


def plot_figure_f_event_rate(
    fixed_window_rows: Sequence[Mapping],
    *,
    window_sec: float = DEFAULT_WINDOW_SEC,
) -> Figure:
    """Figure F: event rate (events/hour) in Post / Mid / Pre windows."""
    count: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for row in fixed_window_rows:
        key = (
            str(row["subject"]),
            str(row["seizure_interval_id"]),
            str(row["window_position"]),
        )
        count[key] += 1

    rate_scale = 3600.0 / float(window_sec)
    by_window: Dict[str, List[float]] = defaultdict(list)
    intervals = {(s, i) for s, i, _ in count.keys()}
    for subj, iv_id in intervals:
        for wp in ("post", "mid", "pre"):
            n = count.get((subj, iv_id, wp), 0)
            by_window[wp].append(float(n) * rate_scale)

    fig, ax = plt.subplots(figsize=(5, 4))
    positions = {"post": 0, "mid": 1, "pre": 2}
    colors = [WINDOW_COLORS.get(wp, "#999") for wp in ("post", "mid", "pre")]
    data = [by_window.get(wp, []) for wp in ("post", "mid", "pre")]
    bp = ax.boxplot(data, positions=[0, 1, 2], patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Post", "Mid", "Pre"])
    ax.set_ylabel("Events / hour")
    ax.set_title("Interictal group event rate by window position")
    medians = [float(np.median(d)) if d else 0.0 for d in data]
    for i, med in enumerate(medians):
        ax.text(i, med, f"{med:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


# ── orchestration ──────────────────────────────────────────────────────────


def run_pr6_analysis(
    *,
    epilepsiae_events_csv: Optional[str | Path] = None,
    yuquan_events_csv: Optional[str | Path] = None,
    output_dir: str | Path,
    min_interval_sec: float = DEFAULT_MIN_INTERVAL_SEC,
    window_sec: float = DEFAULT_WINDOW_SEC,
    n_bins: int = DEFAULT_N_BINS,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
    example_subjects: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """
    Full PR6 analysis pipeline.

    Loads event annotations, computes fixed windows and trajectories,
    runs statistics, generates Figures A–E, writes summary tables.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    if epilepsiae_events_csv is not None:
        epi_rows = load_event_rows(epilepsiae_events_csv, dataset="epilepsiae")
        all_rows.extend(epi_rows)
    if yuquan_events_csv is not None:
        yq_rows = load_event_rows(yuquan_events_csv, dataset="yuquan")
        all_rows.extend(yq_rows)
    if not all_rows:
        raise ValueError("No event annotations loaded.")

    fixed = assign_fixed_window_positions(
        all_rows, min_interval_sec=min_interval_sec, window_sec=window_sec
    )
    traj = compute_normalized_trajectory(all_rows)
    fixed_interval_means = build_fixed_window_interval_means_table(fixed)
    trajectory_interval_stats = build_trajectory_interval_stats_table(traj)

    _write_csv(out / "pr6_fixed_window_events.csv", fixed)
    _write_csv(out / "pr6_trajectory_events.csv", traj)
    _write_csv(out / "pr6_fixed_window_interval_means.csv", fixed_interval_means)
    _write_csv(out / "pr6_trajectory_interval_stats.csv", trajectory_interval_stats)

    stats_results: Dict[str, object] = {}
    within_interval_stats: Dict[str, Dict[str, object]] = {}
    stats_results["analysis_contract"] = {
        "formal_statistical_unit": "seizure_interval",
        "raw_input_unit": "event_row",
        "fixed_window_primary_test": (
            "paired Wilcoxon on subject×interval window means"
        ),
        "trajectory_primary_test": (
            "within-interval Spearman rho, then one-sample Wilcoxon across intervals"
        ),
        "pooled_event_level_trend": "exploratory_reference_only",
    }
    for family, (metric_all, metric_core) in METRIC_FAMILIES.items():
        for label, mc in [("all", metric_all), ("core", metric_core)]:
            key = f"{family}_{label}"
            stats_results[f"post_vs_pre_{key}"] = paired_window_test(
                fixed, metric_col=mc, pair=("post", "pre")
            )
            stats_results[f"post_vs_mid_{key}"] = paired_window_test(
                fixed, metric_col=mc, pair=("post", "mid")
            )
            pooled = trajectory_trend_test(traj, metric_col=mc, n_bins=n_bins)
            stats_results[f"trajectory_pooled_{key}"] = pooled
            wi = within_interval_trend_test(traj, metric_col=mc)
            stats_results[f"trajectory_within_interval_{key}"] = wi
            within_interval_stats[mc] = wi

    # ── event rate analysis ──
    if fixed:
        stats_results["event_rate_post_vs_pre"] = event_rate_paired_window_test(
            fixed, window_sec=window_sec, pair=("post", "pre")
        )
        stats_results["event_rate_post_vs_mid"] = event_rate_paired_window_test(
            fixed, window_sec=window_sec, pair=("post", "mid")
        )

    # ── region-stratified analysis (i/l/e, if columns present) ──
    region_families = _detect_available_region_metrics(all_rows)
    for rkey, (rcol,) in region_families.items():
        stats_results[f"post_vs_pre_{rkey}"] = paired_window_test(
            fixed, metric_col=rcol, pair=("post", "pre")
        )
        stats_results[f"trajectory_within_interval_{rkey}"] = (
            within_interval_trend_test(traj, metric_col=rcol)
        )

    cohort = build_cohort_summary(all_rows, fixed, traj)
    stats_results["cohort_summary"] = cohort

    summary_path = out / "pr6_statistics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(stats_results, fh, indent=2, ensure_ascii=False, default=str)

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved_figures: List[str] = []

    enriched_rows = [dict(r) for r in all_rows]

    subjects_for_fig_a = list(example_subjects or [])
    if not subjects_for_fig_a:
        subjects_for_fig_a = sorted({str(r["subject"]) for r in enriched_rows if _is_assigned(r)})
    figure_a_metric_specs = [
        ("legacy", "sync_legacy_global"),
        ("phase", "sync_phase_global"),
        ("span", "sync_span_global"),
    ]
    subject_timeline_dirs: Dict[str, str] = {}
    for subj in subjects_for_fig_a:
        subject_dir = fig_dir / "subjects" / str(subj)
        subject_dir.mkdir(parents=True, exist_ok=True)
        subject_timeline_dirs[str(subj)] = str(subject_dir)
        for metric_slug, metric_col in figure_a_metric_specs:
            fig = plot_figure_a_subject_timeline(
                enriched_rows,
                subj,
                metric_col=metric_col,
                highlight_window_sec=window_sec,
            )
            combined_path = subject_dir / f"figure_a_{metric_slug}_combined.png"
            fig.savefig(str(combined_path), dpi=150)
            n_panels = len(fig.axes)
            plt.close(fig)
            saved_figures.append(str(combined_path))
            for facet_idx in range(n_panels):
                panel_fig = plot_figure_a_subject_timeline(
                    enriched_rows,
                    subj,
                    metric_col=metric_col,
                    facet_index=facet_idx,
                    highlight_window_sec=window_sec,
                )
                hour_lo = int(facet_idx * FIGURE_A_FACET_HOURS)
                hour_hi = int((facet_idx + 1) * FIGURE_A_FACET_HOURS)
                panel_path = subject_dir / (
                    f"figure_a_{metric_slug}_panel_{facet_idx:02d}_{hour_lo:03d}-{hour_hi:03d}h.png"
                )
                panel_fig.savefig(str(panel_path), dpi=150)
                plt.close(panel_fig)

    wi_primary = within_interval_stats.get(primary_metric)
    if traj:
        fig = plot_figure_b_trajectory_ribbon(
            traj, metric_col=primary_metric, n_bins=n_bins,
            title_suffix="all eligible intervals",
            within_interval_stat=wi_primary,
        )
        p = fig_dir / "figure_b_trajectory_all.png"
        fig.savefig(str(p), dpi=150)
        plt.close(fig)
        saved_figures.append(str(p))

        long_traj = [
            r for r in traj
            if (_float_or_none(r.get("clean_between_seizures_sec")) or 0)
            >= min_interval_sec
        ]
        if long_traj:
            wi_long = within_interval_trend_test(
                long_traj, metric_col=primary_metric
            )
            fig = plot_figure_b_trajectory_ribbon(
                long_traj, metric_col=primary_metric, n_bins=n_bins,
                title_suffix=f"≥{min_interval_sec / 3600:.0f}h intervals",
                within_interval_stat=wi_long,
            )
            p = fig_dir / "figure_b_trajectory_long.png"
            fig.savefig(str(p), dpi=150)
            plt.close(fig)
            saved_figures.append(str(p))

    if fixed:
        fig = plot_figure_c_fixed_window_comparison(fixed)
        p = fig_dir / "figure_c_fixed_window.png"
        fig.savefig(str(p), dpi=150)
        plt.close(fig)
        saved_figures.append(str(p))

    if fixed:
        ds_present = sorted({
            str(r.get("dataset", "")) for r in fixed if r.get("dataset")
        })
        datasets_for_d = ds_present + ["combined"] if len(ds_present) > 1 else ds_present
        if not datasets_for_d:
            datasets_for_d = ["combined"]
        fig = plot_figure_d_robustness(fixed, datasets=datasets_for_d)
        p = fig_dir / "figure_d_robustness.png"
        fig.savefig(str(p), dpi=150)
        plt.close(fig)
        saved_figures.append(str(p))

    fig = plot_figure_e_coverage_audit(all_rows, fixed, traj)
    p = fig_dir / "figure_e_coverage.png"
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    saved_figures.append(str(p))

    if fixed:
        fig = plot_figure_f_event_rate(fixed, window_sec=window_sec)
        p = fig_dir / "figure_f_event_rate.png"
        fig.savefig(str(p), dpi=150)
        plt.close(fig)
        saved_figures.append(str(p))

    output_summary = {
        "output_dir": str(out),
        "n_annotated_event_rows": len(all_rows),
        "n_fixed_window_events": len(fixed),
        "n_trajectory_events": len(traj),
        "n_fixed_window_intervals": len(fixed_interval_means),
        "n_trajectory_intervals": len(trajectory_interval_stats),
        "statistics_json": str(summary_path),
        "figures": saved_figures,
        "subject_timeline_dirs": subject_timeline_dirs,
        "cohort_summary": cohort,
    }
    return output_summary
