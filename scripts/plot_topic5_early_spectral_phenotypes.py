#!/usr/bin/env python3
"""Classify overlapping early-seizure spectral support and render a pie chart.

This is a descriptive seizure-level summary.  It deliberately assigns only
frequency-defined phenotypes from power.  True LVFA, HYP, rhythmic sharp,
spike-wave, polyspike, delta-brush, and burst-suppression diagnoses still require
raw-trace review.

The classifier uses the six common bands spanning 1--150 Hz.  When a seizure has
an accepted T_spectral_best, band support is measured in the five seconds after
that event-specific onset.  Otherwise, broadband, fast, and slow category-specific
change points are searched near EEG onset.  Unaccepted candidates remain visible
in the audit table but never act as timing anchors.  No category is forced when
its empirical distal-baseline gate fails.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_energy_timing import (  # noqa: E402
    detect_multiband_recruitment_onset,
    detect_sustained_enhancement,
    smooth_trace,
)


CACHE_ROOT = ROOT / "results/topic5_ictal_recruitment/v2_band_scan/cache"
EPI_TIMING = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/epilepsiae/refinement_v1p2/per_seizure_subject_refined_onset.csv"
)
YUQUAN_TIMING = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/yuquan/refinement_v1p2/per_seizure_subject_refined_onset.csv"
)
EPI_ONSET_BANDS = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/epilepsiae/source_cache/sensitivity_extended_1_250hz/band_seizure_level_onset_energy.csv"
)
ALIGNED_ROOTS = {
    "epilepsiae": ROOT
    / "results/topic5_ictal_recruitment/v2_band_scan/cache_tspectral_v1p2_common_1_80hz",
    "yuquan": ROOT
    / "results/topic5_ictal_recruitment/v2_band_scan/cache_tspectral_v1p2_yuquan_common_1_80hz",
}
OUT_ROOT = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/early_spectral_phenotype"
)

BANDS = (
    "delta_HYP_slow",
    "theta_preictal_PAC",
    "alpha_sharp_leq13",
    "beta_LVFA_low",
    "gamma_LVFA",
    "hg_low_ripple",
)
LOW_BANDS = BANDS[:3]
FAST_BANDS = BANDS[3:]
BAND_LABELS = {
    "delta_HYP_slow": "delta 1-4 Hz",
    "theta_preictal_PAC": "theta 4-8 Hz",
    "alpha_sharp_leq13": "alpha 8-13 Hz",
    "beta_LVFA_low": "beta 13-30 Hz",
    "gamma_LVFA": "gamma 30-80 Hz",
    "hg_low_ripple": "high-gamma 80-150 Hz",
}

CATEGORIES = (
    "broadband_gamma_low_overlap",
    "broadband_low_no_gamma",
    "gamma_low_nonbroadband",
    "gamma_only",
    "low_frequency_only",
    "neither_defined_support",
)
CATEGORY_LABELS = {
    "broadband_gamma_low_overlap": "Broadband + gamma + low-frequency",
    "broadband_low_no_gamma": "Broadband + low-frequency; no gamma",
    "gamma_low_nonbroadband": "Gamma + low-frequency; non-broadband",
    "gamma_only": "Gamma; no broadband/low-frequency",
    "low_frequency_only": "Low-frequency; no broadband/gamma",
    "neither_defined_support": "None / other",
}
CATEGORY_COLORS = {
    "broadband_gamma_low_overlap": "#A33A2B",
    "broadband_low_no_gamma": "#7F4A3A",
    "gamma_low_nonbroadband": "#8C6BB1",
    "gamma_only": "#D98C2B",
    "low_frequency_only": "#4C78A8",
    "neither_defined_support": "#C7C7C7",
}

SIMPLE_CATEGORIES = (
    "broadband_1_150",
    "gamma_nonbroadband",
    "low_frequency_only",
    "other",
)
SIMPLE_LABELS = {
    "broadband_1_150": "Broadband increase (1–150 Hz)",
    "gamma_nonbroadband": "Gamma enhancement (30–80 Hz; non-broadband)",
    "low_frequency_only": "Low-frequency enhancement (1–13 Hz)",
    "other": "Other / no defined early pattern",
}
SIMPLE_COLORS = {
    "broadband_1_150": "#A33A2B",
    "gamma_nonbroadband": "#D98C2B",
    "low_frequency_only": "#4C78A8",
    "other": "#C7C7C7",
}

BASELINE = (-120.0, -90.0)
CATEGORY_SEARCH = (-15.0, 20.0)
POST_SEC = 5.0
SMOOTH_SEC = 2.0
BASELINE_QUANTILE = 0.99
SUSTAIN_SEC = 2.0
SPATIAL_QUANTILE = 0.75


def _truth(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def is_early_anchor(value: float) -> bool:
    """Return whether an anchor is inside the locked peri-EEG-onset domain."""
    return bool(
        np.isfinite(value) and CATEGORY_SEARCH[0] <= value <= CATEGORY_SEARCH[1]
    )


def classify_band_hits(hits: dict[str, bool]) -> str:
    """Map six sustained band-hit flags to one conservative phenotype."""
    missing = [band for band in BANDS if band not in hits]
    if missing:
        raise ValueError(f"missing band-hit flags: {missing}")
    low = sum(bool(hits[band]) for band in LOW_BANDS)
    fast = sum(bool(hits[band]) for band in FAST_BANDS)
    total = low + fast
    if total >= 5 and low >= 2 and fast >= 2:
        return "broadband_1_150"
    if bool(hits["gamma_LVFA"]) and fast >= 2 and low <= 1:
        return "fast_frequency_dominant_13_150"
    if low >= 2 and fast <= 1:
        return "low_frequency_dominant_le13"
    return "other"


def classify_overlap_state(hits: dict[str, bool]) -> str:
    """Map non-exclusive broadband, gamma, and low support to intersections."""
    missing = [band for band in BANDS if band not in hits]
    if missing:
        raise ValueError(f"missing band-hit flags: {missing}")
    n_low = sum(bool(hits[band]) for band in LOW_BANDS)
    n_fast = sum(bool(hits[band]) for band in FAST_BANDS)
    broadband = n_low >= 2 and n_fast >= 2 and (n_low + n_fast) >= 5
    gamma = bool(hits["gamma_LVFA"])
    low = n_low >= 2
    if broadband and gamma:
        return "broadband_gamma_low_overlap"
    if broadband:
        return "broadband_low_no_gamma"
    if gamma and low:
        return "gamma_low_nonbroadband"
    if gamma:
        return "gamma_only"
    if low:
        return "low_frequency_only"
    return "neither_defined_support"


def classify_simple_state(hits: dict[str, bool]) -> str:
    """Assign one display class using broadband > gamma > low priority."""
    missing = [band for band in BANDS if band not in hits]
    if missing:
        raise ValueError(f"missing band-hit flags: {missing}")
    n_low = sum(bool(hits[band]) for band in LOW_BANDS)
    n_fast = sum(bool(hits[band]) for band in FAST_BANDS)
    broadband = n_low >= 2 and n_fast >= 2 and (n_low + n_fast) >= 5
    if broadband:
        return "broadband_1_150"
    if bool(hits["gamma_LVFA"]):
        return "gamma_nonbroadband"
    if n_low >= 2:
        return "low_frequency_only"
    return "other"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _finite_spatial_trace(z: np.ndarray, rel_t: np.ndarray) -> np.ndarray:
    """Return the finite-data equivalent of the existing q75 timing trace.

    Committed v2 cache arrays used here are finite.  ``np.quantile`` is much
    faster than ``np.nanquantile`` for these contact-by-time matrices, while the
    fail-closed finite check prevents a silent contract change.
    """
    z = np.asarray(z, dtype=float)
    rel_t = np.asarray(rel_t, dtype=float)
    if z.ndim != 2 or rel_t.ndim != 1 or z.shape[1] != rel_t.size:
        raise ValueError("z must be [contact,time] and match rel_t")
    if not np.isfinite(z).all() or not np.isfinite(rel_t).all():
        raise ValueError("early phenotype classifier requires finite cache arrays")
    baseline = (rel_t >= BASELINE[0]) & (rel_t < BASELINE[1])
    if not np.any(baseline):
        raise ValueError("distal baseline is absent from cache time grid")
    delta = z - np.median(z[:, baseline], axis=1, keepdims=True)
    trace = np.quantile(delta, SPATIAL_QUANTILE, axis=0)
    return smooth_trace(trace, rel_t, smooth_sec=SMOOTH_SEC)


def _hits_at_anchor(
    traces: dict[str, np.ndarray], rel_t: np.ndarray, anchor_sec: float
) -> tuple[dict[str, bool], dict[str, dict]]:
    anchor = float(anchor_sec)
    post_hi = min(float(rel_t[-1]), anchor + POST_SEC)
    hits: dict[str, bool] = {}
    diagnostics: dict[str, dict] = {}
    for band in BANDS:
        if not np.isfinite(anchor) or post_hi - anchor < SUSTAIN_SEC:
            hits[band] = False
            diagnostics[band] = {
                "peak_minus_q99": float("nan"),
                "longest_above_sec": 0.0,
                "rise_sec": float("nan"),
            }
            continue
        result = detect_sustained_enhancement(
            traces[band],
            rel_t,
            baseline=BASELINE,
            search=(anchor, post_hi),
            baseline_quantile=BASELINE_QUANTILE,
            sustain_sec=SUSTAIN_SEC,
        )
        hits[band] = bool(result.detected)
        diagnostics[band] = {
            "peak_minus_q99": float(result.peak_value - result.threshold),
            "longest_above_sec": float(result.longest_above_sec),
            "rise_sec": float(result.rise_sec),
        }
    return hits, diagnostics


def _category_search(
    traces: dict[str, np.ndarray], rel_t: np.ndarray
) -> tuple[str, float, dict[str, bool], dict[str, dict], str]:
    """Search broadband, selective fast, and selective slow onset candidates."""
    broad = detect_multiband_recruitment_onset(
        np.vstack([traces[band] for band in BANDS]),
        rel_t,
        baseline=BASELINE,
        search=CATEGORY_SEARCH,
        majority_required=5,
        post_sec=POST_SEC,
        flank_sec=2.0,
        baseline_quantile=BASELINE_QUANTILE,
        sustain_sec=SUSTAIN_SEC,
    )
    fast = detect_multiband_recruitment_onset(
        np.vstack([traces[band] for band in FAST_BANDS]),
        rel_t,
        baseline=BASELINE,
        search=CATEGORY_SEARCH,
        majority_required=2,
        post_sec=POST_SEC,
        flank_sec=2.0,
        baseline_quantile=BASELINE_QUANTILE,
        sustain_sec=SUSTAIN_SEC,
    )
    slow = detect_multiband_recruitment_onset(
        np.vstack([traces[band] for band in LOW_BANDS]),
        rel_t,
        baseline=BASELINE,
        search=CATEGORY_SEARCH,
        majority_required=2,
        post_sec=POST_SEC,
        flank_sec=2.0,
        baseline_quantile=BASELINE_QUANTILE,
        sustain_sec=SUSTAIN_SEC,
    )
    broad_hits, broad_diag = _hits_at_anchor(traces, rel_t, broad.onset_sec)
    fast_hits, fast_diag = _hits_at_anchor(traces, rel_t, fast.onset_sec)
    slow_hits, slow_diag = _hits_at_anchor(traces, rel_t, slow.onset_sec)
    broad_label = (
        broad.detected and classify_band_hits(broad_hits) == "broadband_1_150"
    )
    fast_label = (
        fast.detected
        and classify_band_hits(fast_hits) == "fast_frequency_dominant_13_150"
    )
    slow_label = (
        slow.detected
        and classify_band_hits(slow_hits) == "low_frequency_dominant_le13"
    )
    if broad_label:
        return (
            "broadband_1_150",
            float(broad.onset_sec),
            broad_hits,
            broad_diag,
            "broad_specific_change_point",
        )
    if fast_label and not slow_label:
        return (
            "fast_frequency_dominant_13_150",
            float(fast.onset_sec),
            fast_hits,
            fast_diag,
            "fast_specific_change_point",
        )
    if slow_label and not fast_label:
        return (
            "low_frequency_dominant_le13",
            float(slow.onset_sec),
            slow_hits,
            slow_diag,
            "slow_specific_change_point",
        )
    # Both selective patterns, isolated bands, and absent rises remain Other.
    empty = {band: False for band in BANDS}
    empty_diag = {
        band: {
            "peak_minus_q99": float("nan"),
            "longest_above_sec": 0.0,
            "rise_sec": float("nan"),
        }
        for band in BANDS
    }
    reason = "ambiguous_fast_and_slow" if fast_label and slow_label else "no_selective_gate"
    return "other", float("nan"), empty, empty_diag, reason


def _timing_rows() -> dict[tuple[str, int], dict]:
    frames = [pd.read_csv(EPI_TIMING), pd.read_csv(YUQUAN_TIMING)]
    timing = pd.concat(frames, ignore_index=True, sort=False)
    return {
        (str(row["subject"]), int(row["seizure_idx"])): row.to_dict()
        for _, row in timing.iterrows()
    }


def _eeg_offsets() -> dict[tuple[str, int], float]:
    table = pd.read_csv(EPI_ONSET_BANDS).drop_duplicates(
        ["subject", "seizure_idx"]
    )
    return {
        (str(row["subject"]), int(row["seizure_idx"])): float(
            row["eeg_onset_rel_clinical_sec"]
        )
        for _, row in table.iterrows()
    }


def _anchor_from_timing(row: dict | None) -> tuple[float, str]:
    if row is None:
        return float("nan"), "none"
    if _truth(row.get("has_accepted_t_best")):
        value = float(row.get("t_spectral_best_rel_eeg_sec", np.nan))
        if np.isfinite(value):
            return value, "accepted_t_spectral_best"
    # Candidate-only times are not refined enough for downstream alignment.
    # Their existence remains visible in the per-seizure audit columns.
    return float("nan"), "none"


def _process_subject(
    meta_path: Path,
    timing: dict[tuple[str, int], dict],
    eeg_offsets: dict[tuple[str, int], float],
) -> tuple[list[dict], list[dict]]:
    subject = meta_path.stem
    dataset = subject.split("_", 1)[0]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    aligned_meta_path = ALIGNED_ROOTS[dataset] / meta_path.name
    if not aligned_meta_path.exists():
        return [], [
            {
                "subject": subject,
                "seizure_idx": "",
                "reason": "missing_tspectral_channel_contract",
            }
        ]
    aligned_meta = json.loads(aligned_meta_path.read_text(encoding="utf-8"))
    analysis_channels = [str(value) for value in aligned_meta["analysis_channels"]]

    obj = np.load(meta_path.with_suffix(".npz"), allow_pickle=False)
    cache_channels = [str(value) for value in obj["channels"]]
    absent = [name for name in analysis_channels if name not in cache_channels]
    if absent:
        obj.close()
        return [], [
            {
                "subject": subject,
                "seizure_idx": "",
                "reason": f"timing_channels_missing:{absent}",
            }
        ]
    channel_idx = np.asarray(
        [cache_channels.index(name) for name in analysis_channels], dtype=int
    )

    rows: list[dict] = []
    exclusions: list[dict] = []
    for seizure_idx in meta["seizure_idxs"]:
        idx = int(seizure_idx)
        if not all(f"{band}__zt__{idx}" in obj.files for band in BANDS):
            exclusions.append(
                {
                    "subject": subject,
                    "seizure_idx": idx,
                    "reason": "missing_one_or_more_1_150hz_bands",
                }
            )
            continue
        rel_cache = np.asarray(obj[f"{BANDS[0]}__relt__{idx}"], dtype=float)
        if dataset == "epilepsiae":
            key = (subject, idx)
            if key not in eeg_offsets:
                exclusions.append(
                    {
                        "subject": subject,
                        "seizure_idx": idx,
                        "reason": "missing_eeg_onset_offset",
                    }
                )
                continue
            rel_eeg = rel_cache - eeg_offsets[key]
        else:
            rel_eeg = rel_cache
        traces = {
            band: _finite_spatial_trace(
                np.asarray(obj[f"{band}__zt__{idx}"], dtype=float)[channel_idx],
                rel_eeg,
            )
            for band in BANDS
        }

        timing_row = timing.get((subject, idx))
        accepted_anchor, accepted_anchor_source = _anchor_from_timing(timing_row)
        accepted_in_early_window = is_early_anchor(accepted_anchor)
        if accepted_in_early_window:
            anchor = accepted_anchor
            anchor_source = accepted_anchor_source
            hits, diagnostics = _hits_at_anchor(traces, rel_eeg, anchor)
            category = classify_band_hits(hits)
            classification_reason = "tspectral_anchored_band_support"
        else:
            (
                category,
                anchor,
                hits,
                diagnostics,
                classification_reason,
            ) = _category_search(traces, rel_eeg)
            if category != "other":
                anchor_source = classification_reason
            elif np.isfinite(accepted_anchor):
                anchor_source = (
                    "accepted_t_spectral_outside_early_window_no_early_pattern"
                )
            else:
                anchor_source = "no_accepted_or_candidate_tspectral"

        n_low = int(sum(bool(hits[band]) for band in LOW_BANDS))
        n_fast = int(sum(bool(hits[band]) for band in FAST_BANDS))
        overlap_state = classify_overlap_state(hits)
        simple_state = classify_simple_state(hits)
        strict_broadband = bool(
            n_low >= 2 and n_fast >= 2 and (n_low + n_fast) >= 5
        )
        gamma_band_support = bool(hits["gamma_LVFA"])
        low_frequency_support = bool(n_low >= 2)
        row = {
            "analysis_version": "topic5_early_spectral_overlap_v3",
            "dataset": dataset,
            "subject": subject,
            "seizure_idx": idx,
            "seizure_id": (
                timing_row.get("seizure_id", "") if timing_row is not None else ""
            ),
            "phenotype": overlap_state,
            "phenotype_label": CATEGORY_LABELS[overlap_state],
            "simple_phenotype": simple_state,
            "simple_phenotype_label": SIMPLE_LABELS[simple_state],
            "detection_gate_category": category,
            "anchor_rel_eeg_sec": anchor,
            "anchor_source": anchor_source,
            "classification_reason": classification_reason,
            "timing_status": (
                timing_row.get("timing_status", "") if timing_row is not None else ""
            ),
            "has_tspectral_candidate": bool(
                timing_row is not None and _truth(timing_row.get("has_candidate_t"))
            ),
            "has_accepted_tspectral": bool(
                timing_row is not None
                and _truth(timing_row.get("has_accepted_t_best"))
            ),
            "accepted_tspectral_rel_eeg_sec": accepted_anchor,
            "accepted_tspectral_in_early_window": accepted_in_early_window,
            "n_analysis_contacts": int(len(analysis_channels)),
            "n_low_band_hits": n_low,
            "n_fast_band_hits": n_fast,
            "n_total_band_hits": n_low + n_fast,
            "strict_broadband_5of6": strict_broadband,
            "gamma_band_30_80_support": gamma_band_support,
            "low_frequency_1_13_support": low_frequency_support,
        }
        for band in BANDS:
            row[f"{band}__hit"] = bool(hits[band])
            row[f"{band}__peak_minus_q99"] = diagnostics[band]["peak_minus_q99"]
            row[f"{band}__longest_above_sec"] = diagnostics[band][
                "longest_above_sec"
            ]
            row[f"{band}__rise_rel_eeg_sec"] = diagnostics[band]["rise_sec"]
        rows.append(row)
    obj.close()
    return rows, exclusions


def _cohort_rows(events: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    groups = [("combined_descriptive", events)] + list(events.groupby("dataset"))
    for dataset, use in groups:
        n = int(len(use))
        n_subjects = int(use["subject"].nunique())
        for category in CATEGORIES:
            count = int(np.sum(use["phenotype"] == category))
            rows.append(
                {
                    "dataset": dataset,
                    "phenotype": category,
                    "phenotype_label": CATEGORY_LABELS[category].replace("\n", " "),
                    "n_seizures": count,
                    "fraction_seizures": count / n if n else float("nan"),
                    "denominator_seizures": n,
                    "denominator_subjects": n_subjects,
                }
            )
    return rows


def _simple_cohort_rows(events: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    groups = [("combined_descriptive", events)] + list(events.groupby("dataset"))
    for dataset, use in groups:
        n = int(len(use))
        n_subjects = int(use["subject"].nunique())
        for category in SIMPLE_CATEGORIES:
            count = int(np.sum(use["simple_phenotype"] == category))
            rows.append(
                {
                    "dataset": dataset,
                    "phenotype": category,
                    "phenotype_label": SIMPLE_LABELS[category],
                    "n_seizures": count,
                    "fraction_seizures": count / n if n else float("nan"),
                    "denominator_seizures": n,
                    "denominator_subjects": n_subjects,
                }
            )
    return rows


def _subject_rows(events: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for (dataset, subject), use in events.groupby(["dataset", "subject"], sort=True):
        counts = Counter(use["phenotype"])
        row = {
            "dataset": dataset,
            "subject": subject,
            "n_seizures": int(len(use)),
        }
        for category in CATEGORIES:
            n = int(counts.get(category, 0))
            row[f"n__{category}"] = n
            row[f"fraction__{category}"] = n / len(use)
        rows.append(row)
    return rows


def _render(events: pd.DataFrame, out_root: Path) -> tuple[Path, Path]:
    figures = out_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    total = int(len(events))
    counts = [int(np.sum(events["phenotype"] == value)) for value in CATEGORIES]
    colors = [CATEGORY_COLORS[value] for value in CATEGORIES]

    fig, ax = plt.subplots(figsize=(8.0, 4.1))
    fig.subplots_adjust(left=0.015, right=0.53, top=0.88, bottom=0.02)
    wedges, _ = ax.pie(
        counts,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.48, "edgecolor": "white", "linewidth": 1.5},
    )
    ax.text(
        0,
        0.04,
        f"n={total}",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0,
        -0.14,
        f"{events['subject'].nunique()} subjects",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#555555",
    )
    legend_labels = [
        f"{CATEGORY_LABELS[category]}  n={count} ({100.0 * count / total:.1f}%)"
        for category, count in zip(CATEGORIES, counts)
    ]
    fig.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.55, 0.49),
        frameon=False,
        fontsize=8.6,
        ncol=1,
        labelspacing=0.78,
    )
    fig.suptitle(
        "Early spectral-support overlap",
        fontsize=13,
        fontweight="bold",
        y=0.965,
    )
    png = figures / "early_seizure_spectral_overlap_pie.png"
    pdf = figures / "early_seizure_spectral_overlap_pie.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return png, pdf


def _render_simple(events: pd.DataFrame, out_root: Path) -> tuple[Path, Path]:
    figures = out_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    total = int(len(events))
    counts = [
        int(np.sum(events["simple_phenotype"] == value))
        for value in SIMPLE_CATEGORIES
    ]
    colors = [SIMPLE_COLORS[value] for value in SIMPLE_CATEGORIES]

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    fig.subplots_adjust(left=0.015, right=0.54, top=0.88, bottom=0.02)
    wedges, _ = ax.pie(
        counts,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.48, "edgecolor": "white", "linewidth": 1.5},
    )
    ax.text(
        0,
        0.04,
        f"n={total}",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0,
        -0.14,
        f"{events['subject'].nunique()} subjects",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#555555",
    )
    legend_labels = [
        f"{SIMPLE_LABELS[category]}  n={count} ({100.0 * count / total:.1f}%)"
        for category, count in zip(SIMPLE_CATEGORIES, counts)
    ]
    fig.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.56, 0.49),
        frameon=False,
        fontsize=8.8,
        ncol=1,
        labelspacing=0.88,
    )
    fig.suptitle(
        "Early spectral phenotypes",
        fontsize=13,
        fontweight="bold",
        y=0.965,
    )
    png = figures / "early_seizure_spectral_simple_pie.png"
    pdf = figures / "early_seizure_spectral_simple_pie.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return png, pdf


def run(cache_root: Path, out_root: Path) -> Path:
    timing = _timing_rows()
    eeg_offsets = _eeg_offsets()
    rows: list[dict] = []
    exclusions: list[dict] = []
    for meta_path in sorted(cache_root.glob("*.json")):
        subject_rows, subject_exclusions = _process_subject(
            meta_path, timing, eeg_offsets
        )
        rows.extend(subject_rows)
        exclusions.extend(subject_exclusions)
    if not rows:
        raise RuntimeError("no 1-150 Hz eligible seizures were classified")

    out_root.mkdir(parents=True, exist_ok=True)
    _write_csv(out_root / "per_seizure_spectral_overlap_state.csv", rows)
    _write_csv(out_root / "classification_exclusions.csv", exclusions)
    events = pd.DataFrame(rows)
    cohort = _cohort_rows(events)
    simple_cohort = _simple_cohort_rows(events)
    subjects = _subject_rows(events)
    _write_csv(out_root / "cohort_spectral_overlap_summary.csv", cohort)
    _write_csv(out_root / "cohort_spectral_simple_summary.csv", simple_cohort)
    _write_csv(out_root / "subject_spectral_overlap_composition.csv", subjects)
    png, pdf = _render(events, out_root)
    simple_png, simple_pdf = _render_simple(events, out_root)

    combined = {
        row["phenotype"]: {
            "n_seizures": int(row["n_seizures"]),
            "fraction_seizures": float(row["fraction_seizures"]),
        }
        for row in cohort
        if row["dataset"] == "combined_descriptive"
    }
    simple_combined = {
        row["phenotype"]: {
            "n_seizures": int(row["n_seizures"]),
            "fraction_seizures": float(row["fraction_seizures"]),
        }
        for row in simple_cohort
        if row["dataset"] == "combined_descriptive"
    }
    contract = {
        "analysis_version": "topic5_early_spectral_overlap_v3",
        "status": "descriptive_exploratory",
        "denominator": (
            "committed source-cache seizures with all six 1-150 Hz bands and "
            "the fixed lagPat timing-contact set"
        ),
        "n_subjects": int(events["subject"].nunique()),
        "n_seizures": int(len(events)),
        "n_seizures_by_dataset": events.groupby("dataset").size().astype(int).to_dict(),
        "n_exclusions": int(len(exclusions)),
        "bands": {band: BAND_LABELS[band] for band in BANDS},
        "distal_baseline_rel_eeg_sec": list(BASELINE),
        "spatial_quantile": SPATIAL_QUANTILE,
        "smooth_sec": SMOOTH_SEC,
        "baseline_quantile": BASELINE_QUANTILE,
        "sustain_sec": SUSTAIN_SEC,
        "post_anchor_sec": POST_SEC,
        "selective_category_search_rel_eeg_sec": list(CATEGORY_SEARCH),
        "accepted_tspectral_use": (
            "used directly only inside the locked -15 to +20 s peri-EEG-onset "
            "domain; accepted times outside it remain audit fields and trigger a "
            "fresh early-window category search"
        ),
        "classification": {
            "low_frequency_support": ">=2/3 delta-theta-alpha bands sustained",
            "gamma_band_support": "gamma 30-80 Hz sustained",
            "overlap_states": list(CATEGORIES),
            "strict_broadband_5of6": (
                ">=5/6 sustained bands, including >=2/3 low and >=2/3 fast bands"
            ),
            "simple_display_priority": (
                "broadband first, then gamma 30-80 Hz among non-broadband events, "
                "then low-frequency support among remaining events, then other"
            ),
        },
        "claim_boundary": (
            "The pie partitions every observed intersection of strict broadband, "
            "gamma 30-80 Hz, and low-frequency 1-13 Hz support exactly once. Gamma "
            "support is not a clinical LVFA diagnosis because voltage and morphology "
            "are not classified."
        ),
        "combined_descriptive": combined,
        "simple_combined_descriptive": simple_combined,
        "figure_png": str(png.relative_to(ROOT)),
        "figure_pdf": str(pdf.relative_to(ROOT)),
        "simple_figure_png": str(simple_png.relative_to(ROOT)),
        "simple_figure_pdf": str(simple_pdf.relative_to(ROOT)),
    }
    (out_root / "contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    (out_root / "README.md").write_text(
        "# Early-seizure spectral phenotype\n\n"
        "该目录保留同一合同的两张图：完整 overlap 饼图用于审计，简单互斥饼图用于展示；其余为逐发作状态表、dataset/cohort 汇总和逐患者组成表。分母是 committed cache 中同时具备 1–150 Hz 六频带、固定 lagPat timing-contact 集合和完整远端 baseline 的 seizure。\n\n"
        "早期窗口锁定为 EEG onset 的 −15 至 +20 s。accepted `T_spectral` 只有落在该窗口内才直接用于分型；窗口外时间仍保留在逐发作审计字段中，并在早期窗口内重新搜索，避免把晚期 recruitment 混入 onset phenotype。\n\n"
        "三个非互斥 flag 分别为：严格 broadband（至少 5/6 频带，且低频和快速频率各至少 2/3）、gamma 30–80 Hz 持续增强、低频 1–13 Hz 支持（delta/theta/alpha 至少 2/3）。饼图把实际出现的交集拆成互斥扇区，因此每次 seizure 只计数一次，同时明确显示 overlap。\n\n"
        "不把 LVFA、HYP、rhythmic sharp、spike-wave、polyspike、delta-brush 或 burst-suppression 等 morphology 标签直接赋给功率状态。后续统计必须以 subject 为单位，并分别报告 Epilepsiae 与 Yuquan。\n",
        encoding="utf-8",
    )
    (out_root / "figures" / "README.md").write_text(
        "### early_seizure_spectral_overlap_pie.png\n\n"
        "单一环图把严格 broadband、gamma 30–80 Hz 和低频 1–13 Hz 三个非互斥 flag 的实际交集拆成互斥扇区。右侧 legend 同时报告每个交集的 seizure 数和比例；图中没有第二个 panel 或 footnote。\n\n"
        "**关注点**：主要 overlap 是 broadband、gamma 和低频三者共同增强；gamma 只表示频带能量，不等于临床 LVFA morphology。\n\n"
        "### early_seizure_spectral_simple_pie.png\n\n"
        "简单版按 broadband → gamma → low-frequency → other 的固定优先级将每次 seizure 放入一个扇区，不展开 overlap。Gamma 扇区明确限定为 non-broadband，完整交集仍由上一张审计图保留。\n\n"
        "**关注点**：用于正文或汇报时读取四类总体构成；需要解释共同增强时回看 overlap 版。\n",
        encoding="utf-8",
    )
    return out_root / "contract.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    print(run(args.cache_root.resolve(), args.out_root.resolve()))


if __name__ == "__main__":
    main()
