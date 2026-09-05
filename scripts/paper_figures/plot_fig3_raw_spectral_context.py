#!/usr/bin/env python3
"""Raw peri-onset EEG + standard-band spectrum context for Topic 5 Fig3a.

This figure is a deliberately low-level bridge before z-ER / field projection:

  panel A: two representative seizure examples, each with stacked raw traces and
  a baseline-normalized TFR on the same compact baseline/onset axis;
  panel B: continuous phenotype-colored low-band, gamma, high-gamma, and
  broadband trajectories from those two seizures.

It is explanatory material, not a cohort statistic.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.signal import spectrogram

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paper_figure_typography import (  # noqa: E402
    LOCKED_PANEL_TYPOGRAPHY_POLICY,
    SIGNAL_CONTEXT_TYPOGRAPHY,
    apply_panel_aware_figure_typography,
)

from src.ictal_onset_extraction import extract_seizure_window  # noqa: E402
from src.plot_style import savefig_pub, style_panel  # noqa: E402
from src.topic5_ictal_recruitment import bipolar_alias_label  # noqa: E402
from scripts.paper_figures.patient_public_labels import (  # noqa: E402
    public_patient_label,
)


OUT_DIR = ROOT / "results/paper-ready-figure/fig3a_raw_spectral_context/figures"
T0_CACHE = ROOT / "results/topic5_ictal_recruitment/t0_feature_cache_v2_windows"
LAGPAT_CHANNEL_SOURCES = (
    ROOT / "results/interictal_propagation_masked/rank_displacement/per_subject",
    ROOT / "results/interictal_propagation_masked_broad/rank_displacement/per_subject",
)

ANALYSIS_BANDS = (
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("low bands", 1.0, 30.0),
    ("gamma", 30.0, 80.0),
    ("high-gamma", 80.0, 150.0),
    ("broadband", 1.0, 150.0),
)

# Alpha and beta remain only in the representative-channel selection audit.
# The paper-facing Fig3a canvas displays four non-overlapping/summary bands.
DISPLAY_BAND_NAMES = ("low bands", "gamma", "high-gamma", "broadband")
DISPLAY_BANDS = tuple(band for band in ANALYSIS_BANDS if band[0] in DISPLAY_BAND_NAMES)
SELECTION_BAND_NAMES = ("alpha", "beta", "gamma", "high-gamma", "broadband")

WINDOW_COLORS = {
    "baseline": "#4C78A8",
    "eeg_onset": "#D98C52",
    "early_ictal": "#B2182B",
}

BAND_LINE_COLORS = {
    "alpha": "#59A14F",
    "beta": "#ECA82C",
    "low bands": "#ECA82C",
    "gamma": "#D98C52",
    "high-gamma": "#B2182B",
    "broadband": "#5B5B5B",
}

PHENOTYPE_STATE_CSV = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/"
    "early_spectral_phenotype/per_seizure_spectral_overlap_state.csv"
)
LOCKED_EXAMPLE_PHENOTYPE_ROWS = {
    ("epilepsiae_1146", 7): {
        "subject": "epilepsiae_1146",
        "seizure_idx": "7",
        "simple_phenotype": "broadband_1_150",
        "simple_phenotype_label": "Broadband increase (1–150 Hz)",
        "classification_reason": "tspectral_anchored_band_support",
        "n_low_band_hits": "3",
        "n_fast_band_hits": "3",
        "n_total_band_hits": "6",
        "strict_broadband_5of6": "True",
    },
    ("epilepsiae_635", 7): {
        "subject": "epilepsiae_635",
        "seizure_idx": "7",
        "simple_phenotype": "gamma_nonbroadband",
        "simple_phenotype_label": "Gamma enhancement (30–80 Hz; non-broadband)",
        "classification_reason": "fast_specific_change_point",
        "n_low_band_hits": "0",
        "n_fast_band_hits": "2",
        "n_total_band_hits": "2",
        "strict_broadband_5of6": "False",
    },
}
PHENOTYPE_COLORS = {
    "broadband_1_150": "#8D9FCD",
    "gamma_nonbroadband": "#62BE9F",
}
PHENOTYPE_LABELS = {
    "broadband_1_150": "Broadband-type",
    "gamma_nonbroadband": "Gamma-type",
}
PHENOTYPE_LEGEND_LABELS = {
    "broadband_1_150": "Broadband",
    "gamma_nonbroadband": "Gamma",
}
BAND_YLABEL = "dB"
MAIN_DISPLAY_WINDOWS = ((-110.0, -90.0), (-10.0, 20.0))
MAIN_DISPLAY_GAP = 14.0
EXAMPLE_IDENTITY_TITLE_FONTSIZE = 14.0
OMITTED_INTERVAL_FACE = "#F2F2F2"
OMITTED_INTERVAL_TEXT = "…"


def _ds_sid(subject: str) -> str:
    return subject.replace("/", "_")


def _loader_subject(subject: str) -> str:
    if "/" in subject:
        return subject
    if "_" not in subject:
        raise ValueError("subject must look like 'epilepsiae_1146' or 'epilepsiae/1146'")
    dataset, sid = subject.split("_", 1)
    return f"{dataset}/{sid}"


def _alias_index(ch_names: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, ch in enumerate(ch_names):
        out[str(ch)] = i
        out[bipolar_alias_label(str(ch))] = i
    return out


def _finite_window(t_axis: np.ndarray, lo: float, hi: float) -> np.ndarray:
    idx = np.where((t_axis >= float(lo)) & (t_axis < float(hi)))[0]
    if idx.size == 0:
        raise ValueError(f"empty time window [{lo}, {hi})")
    return idx


def _phenotype_row(ds_sid: str, seizure_idx: int) -> dict[str, str]:
    """Return the frozen mutually exclusive phenotype row, failing closed."""
    locked = LOCKED_EXAMPLE_PHENOTYPE_ROWS.get((str(ds_sid), int(seizure_idx)))
    if not PHENOTYPE_STATE_CSV.exists():
        if locked is None:
            raise FileNotFoundError(PHENOTYPE_STATE_CSV)
        return dict(locked)
    matches: list[dict[str, str]] = []
    with PHENOTYPE_STATE_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                str(row.get("subject")) == str(ds_sid)
                and int(row["seizure_idx"]) == int(seizure_idx)
            ):
                matches.append(dict(row))
    if len(matches) != 1:
        raise RuntimeError(
            f"{ds_sid} seizure {seizure_idx}: expected one phenotype row, "
            f"found {len(matches)}"
        )
    if locked is not None:
        for key, expected in locked.items():
            if str(matches[0].get(key)) != str(expected):
                raise RuntimeError(
                    f"{ds_sid} seizure {seizure_idx}: locked phenotype field "
                    f"{key} drifted from {expected!r} to {matches[0].get(key)!r}"
                )
    return matches[0]


def _phenotype_detection_counts(row: dict[str, str]) -> dict[str, int | bool]:
    return {
        "n_low_band_hits": int(row["n_low_band_hits"]),
        "n_fast_band_hits": int(row["n_fast_band_hits"]),
        "n_total_band_hits": int(row["n_total_band_hits"]),
        "strict_broadband_5of6": str(row["strict_broadband_5of6"]).lower()
        == "true",
    }


def _compressed_segments(
    times: np.ndarray,
    windows: Sequence[tuple[float, float]],
    *,
    gap: float = MAIN_DISPLAY_GAP,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Map disjoint real-time windows onto one compact display axis."""
    times = np.asarray(times, dtype=float)
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    cursor = 0.0
    for index, (lo, hi) in enumerate(windows):
        if not hi > lo:
            raise ValueError(f"invalid display window {(lo, hi)}")
        mask = (times >= float(lo)) & (times <= float(hi))
        mapped = cursor + (times[mask] - float(lo))
        segments.append((mask, mapped))
        cursor += float(hi) - float(lo)
        if index < len(windows) - 1:
            cursor += float(gap)
    return segments


def _compressed_position(
    value: float,
    windows: Sequence[tuple[float, float]],
    *,
    gap: float = MAIN_DISPLAY_GAP,
) -> float:
    cursor = 0.0
    for index, (lo, hi) in enumerate(windows):
        if float(lo) <= float(value) <= float(hi):
            return cursor + float(value) - float(lo)
        cursor += float(hi) - float(lo)
        if index < len(windows) - 1:
            cursor += float(gap)
    raise ValueError(f"time {value} is outside compact display windows {windows}")


def _configure_compressed_time_axis(
    ax: plt.Axes,
    windows: Sequence[tuple[float, float]],
    *,
    gap: float = MAIN_DISPLAY_GAP,
) -> None:
    ticks: list[float] = []
    labels: list[str] = []
    cursor = 0.0
    gap_centers: list[float] = []
    for index, (lo, hi) in enumerate(windows):
        if float(lo) == -10.0 and float(hi) == 20.0:
            values = (-10.0, 0.0, 10.0, 20.0)
        else:
            midpoint = 0.5 * (float(lo) + float(hi))
            values = (float(lo), midpoint, float(hi))
        for value in values:
            ticks.append(cursor + value - float(lo))
            labels.append(f"{value:g}".replace("-", "−"))
        cursor += float(hi) - float(lo)
        if index < len(windows) - 1:
            gap_centers.append(cursor + 0.5 * float(gap))
            cursor += float(gap)
    ax.set_xlim(0.0, cursor)
    ax.set_xticks(ticks, labels)
    for center in gap_centers:
        gap_left = center - 0.5 * float(gap)
        gap_right = center + 0.5 * float(gap)
        omitted = ax.axvspan(
            gap_left,
            gap_right,
            color=OMITTED_INTERVAL_FACE,
            alpha=0.82,
            lw=0,
            zorder=0.4,
        )
        omitted.set_gid("compressed-time-omitted-region")
        label = ax.text(
            center,
            0.52,
            OMITTED_INTERVAL_TEXT,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=15.0,
            fontweight="bold",
            color="0.48",
            clip_on=False,
            zorder=7,
        )
        label.set_gid("compressed-time-omitted-ellipsis")
        # A conventional paired slash on the x-axis reads as a deliberate
        # time-axis break; the pale band and ellipsis repeat that omission
        # through the otherwise empty raw/TFR data region.
        for offset in (-1.15, 1.15):
            marker, = ax.plot(
                [center + offset - 0.55, center + offset + 0.55],
                [-0.025, 0.025],
                transform=ax.get_xaxis_transform(),
                color="0.28",
                lw=1.45,
                solid_capstyle="round",
                clip_on=False,
                zorder=8,
            )
            marker.set_gid("compressed-time-axis-break")


def _configure_continuous_band_time_axis(
    ax: plt.Axes,
    x_window: tuple[float, float],
) -> None:
    """Keep Fig3-B on physical time; never inherit panel A's compression."""
    ax.set_xlim(*x_window)
    candidate_ticks = (-120.0, -80.0, -40.0, 0.0, 20.0)
    ticks = [tick for tick in candidate_ticks if x_window[0] <= tick <= x_window[1]]
    ax.set_xticks(ticks)


def _draw_clinical_onset_marker(ax: plt.Axes) -> None:
    marker = ax.axvline(
        0.0,
        color="black",
        lw=0.9,
        ls="--",
        zorder=4,
    )
    marker.set_gid("clinical-onset-marker")


def _plot_compact_curve(
    ax: plt.Axes,
    times: np.ndarray,
    values: np.ndarray,
    windows: Sequence[tuple[float, float]],
    *,
    color: str,
    linewidth: float,
    alpha: float = 1.0,
) -> None:
    for mask, mapped in _compressed_segments(times, windows):
        if np.any(mask):
            ax.plot(
                mapped,
                np.asarray(values)[mask],
                color=color,
                lw=linewidth,
                alpha=alpha,
            )


def _shade_compact_interval(
    ax: plt.Axes,
    interval: tuple[float, float],
    windows: Sequence[tuple[float, float]],
    *,
    color: str,
    alpha: float,
) -> None:
    for lo, hi in windows:
        overlap_lo = max(float(interval[0]), float(lo))
        overlap_hi = min(float(interval[1]), float(hi))
        if overlap_hi <= overlap_lo:
            continue
        ax.axvspan(
            _compressed_position(overlap_lo, windows),
            _compressed_position(overlap_hi, windows),
            color=color,
            alpha=alpha,
            lw=0,
        )


def _cache_ranked_channels(ds_sid: str, seizure_idx: int, n_channels: int) -> list[str]:
    npz_path = T0_CACHE / f"{ds_sid}.npz"
    if not npz_path.exists():
        return []
    z = np.load(npz_path, allow_pickle=True)
    key = f"bb_auc__{int(seizure_idx)}"
    if key not in z.files or "channels" not in z.files:
        return []
    channels = [str(x) for x in z["channels"]]
    vals = np.asarray(z[key], dtype=np.float64)
    order = np.argsort(np.where(np.isfinite(vals), vals, -np.inf))[::-1]
    return [channels[i] for i in order[: int(n_channels)] if np.isfinite(vals[i])]


def _load_lagpat_channels(ds_sid: str) -> tuple[list[str], str | None]:
    for root in LAGPAT_CHANNEL_SOURCES:
        path = root / f"{ds_sid}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        pairs = d.get("pairs") or []
        if pairs:
            pair = pairs[0]
            names = [str(x) for x in pair.get("channel_names", [])]
            valid = pair.get("joint_valid", [True] * len(names))
            channels = [ch for ch, ok in zip(names, valid, strict=False) if bool(ok)]
        else:
            channels = [str(x) for x in d.get("channel_names", [])]
        if channels:
            return channels, str(path.relative_to(ROOT))
    return [], None


def _select_channels(
    sw,
    ds_sid: str,
    seizure_idx: int,
    n_channels: int,
    requested: Sequence[str] | None,
    *,
    channel_source: str,
) -> tuple[list[int], str, list[str]]:
    lookup = _alias_index(sw.ch_names)
    if requested:
        idx = []
        for ch in requested:
            if ch not in lookup:
                raise ValueError(f"requested channel {ch!r} not in seizure window")
            idx.append(lookup[ch])
        return idx, "manual", list(requested)

    if channel_source == "lagpat":
        lagpat_channels, source_path = _load_lagpat_channels(ds_sid)
        idx = [lookup[ch] for ch in lagpat_channels if ch in lookup]
        if idx:
            return idx[: int(n_channels)], f"lagpat:{source_path}", lagpat_channels

    ranked = _cache_ranked_channels(ds_sid, seizure_idx, n_channels)
    idx = [lookup[ch] for ch in ranked if ch in lookup]
    if len(idx) >= max(1, min(n_channels, 3)):
        return idx[:n_channels], "ictal_bb_auc", ranked

    early = _finite_window(sw.t_axis, 0.0, min(10.0, float(sw.t_axis[-1])))
    robust_scale = np.nanpercentile(np.abs(sw.signal[:, early] - np.nanmedian(sw.signal[:, early], axis=1, keepdims=True)), 95, axis=1)
    order = np.argsort(np.where(np.isfinite(robust_scale), robust_scale, -np.inf))[::-1]
    idx = [int(i) for i in order[:n_channels]]
    return idx, "early_ictal_robust_scale_fallback", [str(sw.ch_names[i]) for i in idx]


def _eeg_rel_sec(sw) -> float | None:
    if sw.eeg_onset_epoch is None:
        return None
    return float(sw.eeg_onset_epoch - sw.clin_onset_epoch)


def _plot_continuous_stacked(
    ax: plt.Axes,
    sw,
    ch_idx: Sequence[int],
    window: tuple[float, float],
    *,
    scale: float,
    display_windows: Sequence[tuple[float, float]] | None = None,
) -> None:
    idx = _finite_window(sw.t_axis, *window)
    decim = max(1, int(round(float(sw.fs) / 180.0)))
    idx = idx[::decim]
    t = sw.t_axis[idx]
    x = sw.signal[np.asarray(ch_idx), :][:, idx]
    offsets = np.arange(len(ch_idx), dtype=float)[::-1] * scale
    ymin = np.inf
    ymax = -np.inf
    for row, ci in enumerate(ch_idx):
        y = x[row] - np.nanmedian(x[row])
        yy = y + offsets[row]
        ymin = min(ymin, float(np.nanmin(yy)))
        ymax = max(ymax, float(np.nanmax(yy)))
        if display_windows is None:
            ax.plot(t, yy, lw=0.38, color="0.20", alpha=0.85)
        else:
            _plot_compact_curve(
                ax,
                t,
                yy,
                display_windows,
                color="0.20",
                linewidth=0.38,
                alpha=0.85,
            )
    if display_windows is None:
        ax.set_xlim(float(window[0]), float(window[1]))
    else:
        _configure_compressed_time_axis(ax, display_windows)
    span = ymax - ymin
    if np.isfinite(span) and span > 0.0:
        bottom_pad = 0.01 * span
        top_pad = 0.06 * span
    else:
        bottom_pad = 0.01 * float(scale)
        top_pad = 0.06 * float(scale)
    ax.set_ylim(ymin - bottom_pad, ymax + top_pad)
    ax.set_yticks(offsets)
    ax.set_yticklabels([bipolar_alias_label(str(sw.ch_names[i])) for i in ch_idx], fontsize=6)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_xlabel("Time (s)", fontsize=9)


def _shade_windows(
    ax: plt.Axes,
    baseline: tuple[float, float],
    eeg_rel: float | None,
    post_window: tuple[float, float],
    *,
    display_windows: Sequence[tuple[float, float]] | None = None,
) -> None:
    if display_windows is None:
        ax.axvspan(float(baseline[0]), float(baseline[1]), color=WINDOW_COLORS["baseline"], alpha=0.10, lw=0)
        ax.axvspan(float(post_window[0]), float(post_window[1]), color=WINDOW_COLORS["early_ictal"], alpha=0.15, lw=0)
        return
    _shade_compact_interval(
        ax,
        baseline,
        display_windows,
        color=WINDOW_COLORS["baseline"],
        alpha=0.10,
    )
    _shade_compact_interval(
        ax,
        post_window,
        display_windows,
        color=WINDOW_COLORS["early_ictal"],
        alpha=0.15,
    )


def _label_shaded_windows(
    ax: plt.Axes,
    baseline: tuple[float, float],
    eeg_rel: float | None,
    post_window: tuple[float, float],
    *,
    y: float = 0.96,
    post_label: str = "CLINICAL ONSET",
    display_windows: Sequence[tuple[float, float]] | None = None,
) -> None:
    items: list[tuple[float, str, str, float]] = [
        ((float(baseline[0]) + float(baseline[1])) / 2.0, "BASELINE", WINDOW_COLORS["baseline"], y),
    ]
    items.append(
        (
            0.5 * (float(post_window[0]) + float(post_window[1])),
            post_label,
            WINDOW_COLORS["early_ictal"],
            y,
        )
    )
    for x, label, color, yy in items:
        if display_windows is not None:
            interval = baseline if label == "BASELINE" else post_window
            overlaps = [
                (max(float(interval[0]), float(lo)), min(float(interval[1]), float(hi)))
                for lo, hi in display_windows
            ]
            overlaps = [(lo, hi) for lo, hi in overlaps if hi > lo]
            if not overlaps:
                continue
            visible_lo, visible_hi = max(overlaps, key=lambda pair: pair[1] - pair[0])
            x = 0.5 * (visible_lo + visible_hi)
            x = _compressed_position(x, display_windows)
        ax.text(
            x,
            yy,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.0,
            fontweight="bold",
            color=color,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": color, "linewidth": 0.7, "alpha": 0.88},
            clip_on=False,
        )


def _channel_tfr(sw, channel_idx: int, x_window: tuple[float, float], baseline: tuple[float, float]):
    idx = _finite_window(sw.t_axis, *x_window)
    sig = np.asarray(sw.signal[int(channel_idx), idx], dtype=np.float64)
    sig = sig - np.nanmedian(sig)
    fs = float(sw.fs)
    nperseg = int(round(1.0 * fs))
    noverlap = int(round(0.9 * fs))
    freqs, t, pxx = spectrogram(
        sig,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="psd",
        detrend="constant",
    )
    rel_t = t + float(x_window[0])
    keep = (freqs >= 1.0) & (freqs <= 150.0)
    freqs = freqs[keep]
    pxx = pxx[keep, :]
    bl = (rel_t >= float(baseline[0])) & (rel_t < float(baseline[1]))
    if not np.any(bl):
        raise ValueError("TFR baseline has no frames")
    base = np.nanmedian(pxx[:, bl], axis=1, keepdims=True)
    db = 10.0 * np.log10(np.maximum(pxx, 1e-30) / np.maximum(base, 1e-30))
    return freqs, rel_t, db, pxx, base


def _broadband_db_curve(freqs: np.ndarray, rel_t: np.ndarray, pxx: np.ndarray, baseline: tuple[float, float]) -> np.ndarray:
    mask = (freqs >= 1.0) & (freqs <= 150.0)
    band_power = np.nanmean(pxx[mask, :], axis=0)
    bl = (rel_t >= float(baseline[0])) & (rel_t < float(baseline[1]))
    base = np.nanmedian(band_power[bl])
    return 10.0 * np.log10(np.maximum(band_power, 1e-30) / max(base, 1e-30))


def _select_spectral_channel(
    sw,
    ch_idx: Sequence[int],
    x_window: tuple[float, float],
    baseline: tuple[float, float],
    eeg_rel: float | None,
    post_window: tuple[float, float],
    requested: str | None,
    selection_profile: str = "broadband",
) -> tuple[int, dict]:
    lookup = _alias_index(sw.ch_names)
    if requested:
        if requested not in lookup:
            raise ValueError(f"requested spectral channel {requested!r} not in seizure window")
        ci = int(lookup[requested])
        return ci, {"method": "manual", "score_window_sec": None, "score_95pct_db": None}

    score_lo = float(eeg_rel) - 5.0 if eeg_rel is not None else float(post_window[0])
    score_hi = max(float(post_window[1]), float(eeg_rel) + 5.0 if eeg_rel is not None else float(post_window[1]))
    best: tuple[float, float, int, dict[str, float]] | None = None
    for ci in ch_idx:
        freqs, rel_t, _db, pxx, _base = _channel_tfr(sw, int(ci), x_window, baseline)
        curves = _band_enhancement(freqs, rel_t, pxx, baseline)
        win = (rel_t >= score_lo) & (rel_t < score_hi)
        if not np.any(win):
            continue
        band_p95 = {
            name: float(np.nanpercentile(curves[name][win], 95))
            for name in SELECTION_BAND_NAMES
        }
        if selection_profile == "gamma":
            gamma = band_p95["gamma"]
            off_band = max(
                band_p95["alpha"],
                band_p95["beta"],
                band_p95["high-gamma"],
            )
            score = gamma - 0.5 * max(off_band, 0.0)
            min_p95 = gamma
        else:
            min_p95 = min(band_p95.values())
            mean_p95 = float(np.mean(list(band_p95.values())))
            score = min_p95 + 0.25 * mean_p95
        if best is None or score > best[0]:
            best = (score, min_p95, int(ci), band_p95)
    if best is None:
        raise RuntimeError("could not select spectral channel")
    return best[2], {
        "method": (
            "max_gamma_95pct_db_minus_half_max_low_or_high_gamma"
            if selection_profile == "gamma"
            else "max_min_95pct_db_across_alpha_beta_gamma_high_gamma_broadband_in_onset_to_early_ictal_window"
        ),
        "selection_profile": selection_profile,
        "score_window_sec": [score_lo, score_hi],
        "score": best[0],
        "score_min_band_95pct_db": best[1],
        "band_95pct_db": best[3],
    }


def _band_enhancement(freqs: np.ndarray, rel_t: np.ndarray, pxx: np.ndarray, baseline: tuple[float, float]) -> dict[str, np.ndarray]:
    bl = (rel_t >= float(baseline[0])) & (rel_t < float(baseline[1]))
    out: dict[str, np.ndarray] = {}
    for name, lo, hi in ANALYSIS_BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            out[name] = np.full(rel_t.shape, np.nan)
            continue
        band_power = np.nanmean(pxx[mask, :], axis=0)
        base = np.nanmedian(band_power[bl])
        out[name] = 10.0 * np.log10(np.maximum(band_power, 1e-30) / max(base, 1e-30))
    return out


def _window_mean_by_band(curves: dict[str, np.ndarray], rel_t: np.ndarray, win: tuple[float, float]) -> dict[str, float]:
    idx = (rel_t >= float(win[0])) & (rel_t < float(win[1]))
    return {k: float(np.nanmean(v[idx])) if np.any(idx) else float("nan") for k, v in curves.items()}


def _smooth_curve(y: np.ndarray, rel_t: np.ndarray, *, smooth_sec: float = 2.0) -> np.ndarray:
    vals = np.asarray(y, dtype=np.float64)
    if vals.size < 3:
        return vals
    dt = float(np.nanmedian(np.diff(rel_t)))
    n = max(1, int(round(float(smooth_sec) / max(dt, 1e-6))))
    if n <= 1:
        return vals
    kernel = np.ones(n, dtype=np.float64)
    finite = np.isfinite(vals)
    num = np.convolve(np.where(finite, vals, 0.0), kernel, mode="same")
    den = np.convolve(finite.astype(np.float64), kernel, mode="same")
    return np.divide(num, den, out=np.full_like(vals, np.nan), where=den > 0)


def _make_figure(
    sw,
    ch_idx: Sequence[int],
    spectral_idx: int,
    *,
    baseline: tuple[float, float],
    x_window: tuple[float, float],
    post_window: tuple[float, float],
    channel_source_label: str,
    spectral_selection: dict,
    display_subject: str,
    display_windows: Sequence[tuple[float, float]] | None = None,
    primary_phenotype: str | None = None,
    comparison: dict | None = None,
    return_panel_axes: bool = False,
) -> tuple[plt.Figure, dict] | tuple[plt.Figure, dict, dict[str, list[plt.Axes]]]:
    # Panel A contains the two representative seizure examples. Panel B is a separate
    # continuous-time quantitative comparison; its x axis is intentionally not
    # compressed even when the two example panels use a broken display axis.
    compact_main = display_windows is not None
    fig = plt.figure(figsize=(12.4, 3.35) if compact_main else (12.4, 4.1))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.72, 1.0], wspace=0.14)
    left = outer[0, 0].subgridspec(
        2,
        4,
        width_ratios=[1.0, 0.45, 1.0, 0.022],
        height_ratios=[1.65, 1.0],
        hspace=0.50 if compact_main else 0.62,
        wspace=0.06,
    )
    right = outer[0, 1].subgridspec(2, 2, hspace=0.58, wspace=0.20)
    eeg_rel = _eeg_rel_sec(sw)

    freqs, rel_t, db, pxx, _base = _channel_tfr(sw, spectral_idx, x_window, baseline)
    band_curves = _band_enhancement(freqs, rel_t, pxx, baseline)
    spectral_channel = bipolar_alias_label(str(sw.ch_names[int(spectral_idx)]))

    comparison_curves = None
    comparison_rel_t = None
    comparison_db = None
    comparison_freqs = None
    if comparison is not None:
        comparison_freqs, comparison_rel_t, comparison_db, comp_pxx, _comp_base = _channel_tfr(
            comparison["sw"],
            int(comparison["spectral_idx"]),
            x_window,
            baseline,
        )
        comparison_curves = _band_enhancement(
            comparison_freqs, comparison_rel_t, comp_pxx, baseline
        )

    vmax_arrays = [db]
    if comparison_db is not None:
        vmax_arrays.append(comparison_db)
    vmax = max(float(np.nanpercentile(np.abs(values), 98)) for values in vmax_arrays)
    vmax = max(3.0, min(vmax, 14.0))

    def _draw_example(
        column: int,
        example_sw,
        example_ch_idx: Sequence[int],
        example_spectral_idx: int,
        example_eeg_rel: float | None,
        example_post_window: tuple[float, float],
        example_freqs: np.ndarray,
        example_rel_t: np.ndarray,
        example_db: np.ndarray,
        example_label: str,
        phenotype: str | None,
    ) -> tuple[plt.Axes, plt.Axes, object]:
        grid_column = 0 if column == 0 else 2
        idx = _finite_window(example_sw.t_axis, *x_window)
        decimation = max(1, int(round(float(example_sw.fs) / 180.0)))
        trace_values = example_sw.signal[np.asarray(example_ch_idx), :][:, idx[::decimation]]
        centered = trace_values - np.nanmedian(trace_values, axis=1, keepdims=True)
        trace_scale = max(
            40.0,
            float(np.nanpercentile(np.abs(centered), 95) * 3.0),
        )

        ax_raw_example = fig.add_subplot(left[0, grid_column])
        _shade_windows(
            ax_raw_example,
            baseline,
            example_eeg_rel,
            example_post_window,
            display_windows=display_windows,
        )
        _plot_continuous_stacked(
            ax_raw_example,
            example_sw,
            example_ch_idx,
            x_window,
            scale=trace_scale,
            display_windows=display_windows,
        )
        ax_raw_example.set_xlabel("")
        _label_shaded_windows(
            ax_raw_example,
            baseline,
            example_eeg_rel,
            example_post_window,
            display_windows=display_windows,
        )
        title_color = PHENOTYPE_COLORS.get(str(phenotype), "0.15")
        phenotype_label = PHENOTYPE_LABELS.get(str(phenotype), "")
        title = f"{example_label}  {phenotype_label}" if phenotype_label else example_label
        ax_raw_example.set_title(
            title,
            fontsize=9.2,
            fontweight="bold",
            color=title_color,
            loc="left",
            pad=5,
        )
        style_panel(ax_raw_example)
        ax_raw_example.tick_params(axis="x", labelsize=8, width=0.9, length=4)
        ax_raw_example.tick_params(axis="y", labelsize=5.5, length=0)

        ax_tfr_example = fig.add_subplot(left[1, grid_column])
        _shade_windows(
            ax_tfr_example,
            baseline,
            example_eeg_rel,
            example_post_window,
            display_windows=display_windows,
        )
        if display_windows is None:
            example_mesh = ax_tfr_example.pcolormesh(
                example_rel_t,
                example_freqs,
                example_db,
                shading="auto",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            ax_tfr_example.set_xlim(*x_window)
        else:
            example_mesh = None
            for mask, mapped in _compressed_segments(example_rel_t, display_windows):
                if np.count_nonzero(mask) < 2:
                    continue
                example_mesh = ax_tfr_example.pcolormesh(
                    mapped,
                    example_freqs,
                    example_db[:, mask],
                    shading="auto",
                    cmap="RdBu_r",
                    vmin=-vmax,
                    vmax=vmax,
                )
            if example_mesh is None:
                raise RuntimeError("no TFR frames in compact display windows")
            _configure_compressed_time_axis(ax_tfr_example, display_windows)
        ax_tfr_example.set_ylim(1.0, 150.0)
        ax_tfr_example.set_xlabel("Time (s)", fontsize=8)
        spectral_label = bipolar_alias_label(
            str(example_sw.ch_names[int(example_spectral_idx)])
        )
        ax_tfr_example.set_title(
            f"TFR on {spectral_label}",
            fontsize=8.0,
            fontweight="bold",
            loc="left",
            pad=4,
        )
        style_panel(ax_tfr_example)
        ax_tfr_example.tick_params(labelsize=7, width=0.9, length=4)
        return ax_raw_example, ax_tfr_example, example_mesh

    ax_raw, ax_tfr, mesh = _draw_example(
        0,
        sw,
        ch_idx,
        spectral_idx,
        eeg_rel,
        post_window,
        freqs,
        rel_t,
        db,
        display_subject,
        primary_phenotype,
    )
    example_raw_axes = [ax_raw]
    example_tfr_axes = [ax_tfr]
    if comparison is not None:
        comparison_post_window = (
            0.0,
            min(10.0, float(comparison["sw"].t_axis[-1])),
        )
        ax_raw_comparison, ax_tfr_comparison, mesh = _draw_example(
            1,
            comparison["sw"],
            comparison["ch_idx"],
            int(comparison["spectral_idx"]),
            _eeg_rel_sec(comparison["sw"]),
            comparison_post_window,
            comparison_freqs,
            comparison_rel_t,
            comparison_db,
            comparison["display_subject"],
            str(comparison["phenotype"]),
        )
        example_raw_axes.append(ax_raw_comparison)
        example_tfr_axes.append(ax_tfr_comparison)
        ax_raw_comparison.tick_params(axis="y", left=False, labelleft=False)
        ax_tfr_comparison.tick_params(axis="y", left=False, labelleft=False)
        ax_tfr_comparison.set_ylabel("")
    else:
        placeholder_top = fig.add_subplot(left[0, 2])
        placeholder_bottom = fig.add_subplot(left[1, 2])
        placeholder_top.set_axis_off()
        placeholder_bottom.set_axis_off()

    ax_tfr.set_ylabel("frequency (Hz)", fontsize=8)
    cax = fig.add_subplot(left[1, 3])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.ax.set_title(
        "TFR\n(dB)", fontsize=6.5, pad=3, fontweight="bold", loc="left"
    )
    cbar.ax.tick_params(labelsize=6, length=2)

    band_axes = []
    band_ylim_candidates: list[tuple[float, float]] = []
    for j, (name, lo, hi) in enumerate(DISPLAY_BANDS):
        row = j // 2
        col = j % 2
        sharey_ax = band_axes[j - 1] if col == 1 else None
        ax = fig.add_subplot(right[row, col], sharey=sharey_ax)
        _shade_windows(
            ax,
            baseline,
            eeg_rel,
            post_window,
            display_windows=None,
        )
        smoothed = _smooth_curve(band_curves[name], rel_t, smooth_sec=2.0)
        primary_color = (
            PHENOTYPE_COLORS[primary_phenotype]
            if comparison is not None and primary_phenotype in PHENOTYPE_COLORS
            else BAND_LINE_COLORS[name]
        )
        ax.plot(rel_t, smoothed, color=primary_color, lw=1.55)
        comparison_smoothed = None
        if comparison_curves is not None and comparison_rel_t is not None:
            comparison_smoothed = _smooth_curve(
                comparison_curves[name], comparison_rel_t, smooth_sec=2.0
            )
            comparison_color = PHENOTYPE_COLORS[str(comparison["phenotype"])]
            ax.plot(
                comparison_rel_t,
                comparison_smoothed,
                color=comparison_color,
                lw=1.55,
            )
        ax.axhline(0.0, color="0.35", lw=0.6)
        _draw_clinical_onset_marker(ax)
        _configure_continuous_band_time_axis(ax, x_window)
        ax.margins(x=0)
        finite = smoothed[np.isfinite(smoothed)]
        if finite.size:
            lo_y = float(np.nanpercentile(finite, 1))
            hi_y = float(np.nanpercentile(finite, 99.5))
            span = max(1.0, hi_y - lo_y)
            ylim_candidate = (min(-1.0, lo_y - 0.08 * span), max(2.0, hi_y + 0.08 * span))
        else:
            ylim_candidate = (-1.0, 2.0)
        if comparison_smoothed is not None:
            comp_finite = comparison_smoothed[np.isfinite(comparison_smoothed)]
            if comp_finite.size:
                comp_lo = float(np.nanpercentile(comp_finite, 1))
                comp_hi = float(np.nanpercentile(comp_finite, 99.5))
                comp_span = max(1.0, comp_hi - comp_lo)
                ylim_candidate = (
                    min(ylim_candidate[0], comp_lo - 0.08 * comp_span),
                    max(ylim_candidate[1], comp_hi + 0.08 * comp_span),
                )
        band_ylim_candidates.append(ylim_candidate)
        ax.set_title(f"{name}\n({lo:g}-{hi:g} Hz)", fontsize=7.5, pad=2.5, fontweight="bold", linespacing=0.95)
        ax.tick_params(labelsize=6, length=2.5, width=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        if col == 0:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.set_ylabel(BAND_YLABEL, fontsize=6.5, labelpad=2)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_xlabel("Time (s)" if row == 1 else "", fontsize=7, labelpad=1)
        band_axes.append(ax)

    if comparison is not None:
        handles = [
            Line2D(
                [0],
                [0],
                color=PHENOTYPE_COLORS[str(primary_phenotype)],
                lw=2.2,
                label=PHENOTYPE_LEGEND_LABELS[str(primary_phenotype)],
            ),
            Line2D(
                [0],
                [0],
                color=PHENOTYPE_COLORS[str(comparison["phenotype"])],
                lw=2.2,
                label=PHENOTYPE_LEGEND_LABELS[str(comparison['phenotype'])],
            ),
        ]
        # The upper-left of the low-band axis is empty at baseline and keeps
        # the key away from the onset-associated gamma/broadband peaks.
        band_axes[0].legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            ncol=1,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.86,
            handlelength=1.4,
            labelspacing=0.35,
            borderpad=0.25,
        )

    for row in range(2):
        row_idx = (2 * row, 2 * row + 1)
        row_lo = min(band_ylim_candidates[i][0] for i in row_idx)
        row_hi = max(band_ylim_candidates[i][1] for i in row_idx)
        band_axes[row_idx[0]].set_ylim(row_lo, row_hi)
    fig.subplots_adjust(left=0.074, right=0.98, top=0.865, bottom=0.10)
    apply_panel_aware_figure_typography(
        fig,
        spec=SIGNAL_CONTEXT_TYPOGRAPHY,
        policy=LOCKED_PANEL_TYPOGRAPHY_POLICY,
        dense_axes=example_raw_axes,
        colorbar_axes=[cax],
        enforce_atomic_axis_gate=False,
    )
    for ax in example_raw_axes:
        # The event identity/type is the primary header. Keep it visibly above
        # the boxed BASELINE / CLINICAL ONSET interval annotations.
        ax._left_title.set_fontsize(EXAMPLE_IDENTITY_TITLE_FONTSIZE)
    if comparison is not None:
        # Keep the two-entry phenotype key compact inside the low-band axis.
        legend = band_axes[0].get_legend()
        for text in legend.get_texts():
            text.set_fontsize(11.0)
        for line in legend.get_lines():
            line.set_linewidth(2.5)

    eeg_win = (float(eeg_rel) - 5.0, float(eeg_rel) + 5.0) if eeg_rel is not None else (float("nan"), float("nan"))
    summary = {
        "selected_channels": [bipolar_alias_label(str(sw.ch_names[i])) for i in ch_idx],
        "channel_source": channel_source_label,
        "spectral_channel": spectral_channel,
        "spectral_channel_selection": spectral_selection,
        "spectral_summary": "single representative lagPat channel PSD, then dB vs baseline",
        "x_window_sec": list(map(float, x_window)),
        "paper_role": (
            "Fig3-A/B representative spectral-phenotype context"
            if comparison is not None
            else "Fig3-A raw spectral context"
        ),
        "layout": (
            "two compact broken-axis raw/TFR seizure examples in panel A; continuous phenotype-colored low/gamma/high-gamma/broadband trajectories in panel B"
            if display_windows is not None and comparison is not None
            else "raw/TFR seizure examples in panel A; continuous low/gamma/high-gamma/broadband trajectories in panel B"
        ),
        "display_windows_sec": (
            [list(map(float, window)) for window in display_windows]
            if display_windows is not None
            else [list(map(float, x_window))]
        ),
        "omitted_display_interval_sec": (
            [float(display_windows[0][1]), float(display_windows[1][0])]
            if display_windows is not None and len(display_windows) == 2
            else None
        ),
        "right_axis_contract": "continuous -120 to +20 s axis with black dashed 0-s markers; row-shared y limits; y ticks and compact dB label shown only on the left panel of each row; Broadband/Gamma legend occupies the curve-free upper-left of the low-band axis",
        "displayed_bands": [name for name, _lo, _hi in DISPLAY_BANDS],
        "sidecar_only_bands": [name for name, _lo, _hi in ANALYSIS_BANDS if name not in DISPLAY_BAND_NAMES],
        "band_enhancement_mean_db": {
            "baseline": _window_mean_by_band(band_curves, rel_t, baseline),
            "eeg_onset_neighborhood": _window_mean_by_band(band_curves, rel_t, eeg_win),
            "clinical_0_10": _window_mean_by_band(band_curves, rel_t, post_window),
        },
    }
    if comparison is not None:
        summary["phenotype_comparison"] = {
            "color_encodes": "seizure phenotype, not frequency band",
            "primary": {
                "display_subject": display_subject,
                "phenotype": primary_phenotype,
                "color": PHENOTYPE_COLORS[str(primary_phenotype)],
                "spectral_channel": spectral_channel,
            },
            "comparison": {
                "display_subject": comparison["display_subject"],
                "phenotype": comparison["phenotype"],
                "color": PHENOTYPE_COLORS[str(comparison["phenotype"])],
                "spectral_channel": bipolar_alias_label(
                    str(comparison["sw"].ch_names[int(comparison["spectral_idx"])])
                ),
                "spectral_channel_selection": comparison["spectral_selection"],
                "clinical_0_10_mean_db": _window_mean_by_band(
                    comparison_curves,
                    comparison_rel_t,
                    post_window,
                ),
            },
        }
    if return_panel_axes:
        return fig, summary, {
            "a": [*example_raw_axes, *example_tfr_axes, cax],
            "b": list(band_axes),
        }
    return fig, summary


def _save_independent_panel_crops(
    fig: plt.Figure,
    panel_axes: dict[str, list[plt.Axes]],
    out_dir: Path,
) -> dict[str, list[str]]:
    """Save vector/raster crops; panel letters are added only in full layout."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outputs: dict[str, list[str]] = {}
    all_axes = list(fig.axes)
    for panel_id in ("a", "b"):
        target_axes = panel_axes[panel_id]
        for ax in all_axes:
            ax.set_visible(ax in target_axes)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [ax.get_tightbbox(renderer) for ax in target_axes]
        bbox = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted())
        pad = 0.10
        bbox = Bbox.from_extents(
            bbox.x0 - pad, bbox.y0 - pad, bbox.x1 + pad, bbox.y1 + pad,
        )
        png = out_dir / f"fig3-panel{panel_id}.png"
        pdf = out_dir / f"fig3-panel{panel_id}.pdf"
        fig.savefig(png, dpi=600, bbox_inches=bbox, facecolor="white")
        fig.savefig(pdf, bbox_inches=bbox, facecolor="white")
        outputs[panel_id] = [
            str(png.relative_to(ROOT)),
            str(pdf.relative_to(ROOT)),
        ]
    for ax in all_axes:
        ax.set_visible(True)
    return outputs


def _write_readme(
    out_dir: Path,
    out_png: Path,
    out_pdf: Path,
    display_label: str,
) -> None:
    readme = out_dir / "README.md"
    readme.write_text(
        "# Fig3-A/B Spectral Phenotype Context\n\n"
        f"### {out_png.name} / {out_pdf.name}\n\n"
        f"这张图使用 {display_label} 及另一例代表性表型发作，在进入 z-ER、field projection 和 maxAB 相似性之前，展示远端 baseline 与 clinical onset 邻域的原始发作信号。"
        "主图模式下 A 横向并列两个发作的 raw SEEG 与 baseline-normalized TFR；两例都只显示 20 s baseline（−110 至 −90 s）和 −10 至 +20 s，中间 −90 至 −10 s 用成对斜线断轴跳过。"
        "B 的 2×2 小图依次展示 low bands (1-30 Hz)、gamma (30-80 Hz)、high-gamma (80-150 Hz) 和 broadband (1-150 Hz)，并连续显示 −120 至 +20 s；有比较发作时，颜色只编码 broadband-type / gamma-type，不编码频带。"
        "B 的四图都在 0 s 画黑色竖直虚线；同一行共用 y 轴范围，数值 ticks 与简写 dB 标签只放在每行左图。legend 在 low-bands 图左上角的无曲线区纵向排列，只写 Broadband / Gamma。图面不标 EEG onset；alpha 与 beta 只保留在 summary JSON 的审计中。"
        "它只承担解释和质控作用，不是 cohort 统计，也不证明 timing-order replay 或机制。\n\n"
        "**关注点**：A 中每个示例的 raw SEEG 与 TFR 断轴必须严格对齐；成对斜线表示未显示区间，不是数据缺失；B 不得断轴，颜色语义必须保持为发作表型。\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    loader_subject = _loader_subject(args.subject)
    ds_sid = _ds_sid(loader_subject)
    sw = extract_seizure_window(
        loader_subject,
        int(args.seizure_idx),
        pre_sec=float(args.pre_sec),
        post_sec=float(args.post_sec),
        results_root=ROOT / "results",
        reference=args.reference,
    )
    baseline = (float(args.baseline_start), float(args.baseline_end))
    x_window = (float(args.baseline_start), min(float(args.post_sec), max(10.0, float(args.post_sec))))
    post_window = (0.0, min(10.0, float(sw.t_axis[-1])))
    ch_idx, channel_source_label, source_channels = _select_channels(
        sw,
        ds_sid,
        int(args.seizure_idx),
        int(args.n_channels),
        args.channels,
        channel_source=args.channel_source,
    )
    if not ch_idx:
        raise RuntimeError("no channels selected")
    eeg_rel = None
    if sw.eeg_onset_epoch is not None:
        eeg_rel = float(sw.eeg_onset_epoch - sw.clin_onset_epoch)
    spectral_idx, spectral_selection = _select_spectral_channel(
        sw,
        ch_idx,
        x_window,
        baseline,
        eeg_rel,
        post_window,
        args.spectral_channel,
        selection_profile=str(getattr(args, "spectral_profile", "broadband")),
    )
    display_dataset, display_raw_subject = ds_sid.split("_", 1)
    public_label = getattr(args, "display_subject_label", None) or public_patient_label(
        display_dataset, display_raw_subject
    )
    public_seizure_label = f"SZ{int(args.seizure_idx) + 1}"
    display_subject = f"{public_label} | {public_seizure_label}"

    display_windows = (
        MAIN_DISPLAY_WINDOWS if bool(getattr(args, "compact_main", False)) else None
    )
    primary_phenotype = None
    primary_phenotype_row = None
    comparison = None
    comparison_subject = getattr(args, "comparison_subject", None)
    comparison_seizure_idx = getattr(args, "comparison_seizure_idx", None)
    if comparison_subject is not None or comparison_seizure_idx is not None:
        if comparison_subject is None or comparison_seizure_idx is None:
            raise ValueError(
                "comparison-subject and comparison-seizure-idx must be provided together"
            )
        primary_phenotype_row = _phenotype_row(ds_sid, int(args.seizure_idx))
        primary_phenotype = str(primary_phenotype_row["simple_phenotype"])
        if primary_phenotype != "broadband_1_150":
            raise RuntimeError(
                f"primary example is {primary_phenotype}, expected broadband_1_150"
            )

        comparison_loader_subject = _loader_subject(str(comparison_subject))
        comparison_ds_sid = _ds_sid(comparison_loader_subject)
        comparison_sw = extract_seizure_window(
            comparison_loader_subject,
            int(comparison_seizure_idx),
            pre_sec=float(args.pre_sec),
            post_sec=float(args.post_sec),
            results_root=ROOT / "results",
            reference=args.reference,
        )
        comparison_ch_idx, comparison_channel_source, _ = _select_channels(
            comparison_sw,
            comparison_ds_sid,
            int(comparison_seizure_idx),
            int(args.n_channels),
            args.channels,
            channel_source=args.channel_source,
        )
        comparison_spectral_idx, comparison_selection = _select_spectral_channel(
            comparison_sw,
            comparison_ch_idx,
            x_window,
            baseline,
            _eeg_rel_sec(comparison_sw),
            (0.0, min(10.0, float(comparison_sw.t_axis[-1]))),
            getattr(args, "comparison_spectral_channel", None),
            selection_profile=str(
                getattr(args, "comparison_spectral_profile", "gamma")
            ),
        )
        comparison_phenotype_row = _phenotype_row(
            comparison_ds_sid, int(comparison_seizure_idx)
        )
        comparison_phenotype = str(
            comparison_phenotype_row["simple_phenotype"]
        )
        if comparison_phenotype != "gamma_nonbroadband":
            raise RuntimeError(
                f"comparison example is {comparison_phenotype}, "
                "expected gamma_nonbroadband"
            )
        comparison_dataset, comparison_raw_subject = comparison_ds_sid.split("_", 1)
        comparison_public_label = public_patient_label(
            comparison_dataset, comparison_raw_subject
        )
        comparison_display_subject = (
            f"{comparison_public_label} | SZ{int(comparison_seizure_idx) + 1}"
        )
        comparison = {
            "sw": comparison_sw,
            "ch_idx": list(comparison_ch_idx),
            "ds_sid": comparison_ds_sid,
            "seizure_idx": int(comparison_seizure_idx),
            "display_subject": comparison_display_subject,
            "phenotype": comparison_phenotype,
            "phenotype_row": comparison_phenotype_row,
            "spectral_idx": int(comparison_spectral_idx),
            "spectral_selection": comparison_selection,
            "channel_source": comparison_channel_source,
        }

    independent_only = bool(getattr(args, "independent_only", False))
    made = _make_figure(
        sw,
        ch_idx,
        spectral_idx,
        baseline=baseline,
        x_window=x_window,
        post_window=post_window,
        channel_source_label=channel_source_label,
        spectral_selection=spectral_selection,
        display_subject=display_subject,
        display_windows=display_windows,
        primary_phenotype=primary_phenotype,
        comparison=comparison,
        return_panel_axes=independent_only,
    )
    if independent_only:
        fig, summary, panel_axes = made
    else:
        fig, summary = made
    out_dir = Path(getattr(args, "output_dir", None) or OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{ds_sid}_seizure_{int(args.seizure_idx):02d}_raw_spectral_context"
    if independent_only:
        independent_outputs = _save_independent_panel_crops(
            fig, panel_axes, out_dir,
        )
        plt.close(fig)
        out_png = out_dir / "fig3-panela.png"
        out_pdf = out_dir / "fig3-panela.pdf"
    else:
        out_png = out_dir / f"{stem}.png"
        out_pdf = out_dir / f"{stem}.pdf"
        savefig_pub(fig, out_png, dpi=300)
        # Rebuild so PDF is saved from a live figure, not a closed one.
        fig_pdf, _ = _make_figure(
            sw,
            ch_idx,
            spectral_idx,
            baseline=baseline,
            x_window=x_window,
            post_window=post_window,
            channel_source_label=channel_source_label,
            spectral_selection=spectral_selection,
            display_subject=display_subject,
            display_windows=display_windows,
            primary_phenotype=primary_phenotype,
            comparison=comparison,
        )
        savefig_pub(fig_pdf, out_pdf, dpi=300)

    summary.update(
        {
            "subject": ds_sid,
            "public_patient_label": public_label,
            "public_seizure_label": public_seizure_label,
            "display_label": display_subject,
            "loader_subject": loader_subject,
            "seizure_idx": int(args.seizure_idx),
            "seizure_id": sw.seizure_id,
            "fs": float(sw.fs),
            "reference": args.reference,
            "requested_channel_source": args.channel_source,
            "source_channels": source_channels,
            "clinical_onset_sec": 0.0,
            "eeg_onset_rel_sec": eeg_rel,
            "baseline_window_sec": list(map(float, baseline)),
            "eeg_onset_neighborhood_sec": [float(eeg_rel) - 5.0, float(eeg_rel) + 5.0] if eeg_rel is not None else None,
            "clinical_field_input_window_sec": list(map(float, post_window)),
            "outputs": {
                "png": str(out_png.relative_to(ROOT)),
                "pdf": str(out_pdf.relative_to(ROOT)),
            },
            "tier": (
                "paper-ready Fig3-A/B descriptive phenotype context; not a within-patient or cohort statistic"
                if comparison is not None
                else "paper-ready Fig3-A single-seizure explanatory context; not a cohort statistic"
            ),
        }
    )
    if comparison is not None:
        summary["phenotype_comparison"].update(
            {
                "classification_source": str(PHENOTYPE_STATE_CSV.relative_to(ROOT)),
                "same_patient": comparison["ds_sid"] == ds_sid,
                "same_spectral_channel": (
                    summary["spectral_channel"]
                    == summary["phenotype_comparison"]["comparison"][
                        "spectral_channel"
                    ]
                ),
                "primary_seizure_idx": int(args.seizure_idx),
                "comparison_seizure_idx": int(comparison["seizure_idx"]),
                "primary_classification_reason": primary_phenotype_row[
                    "classification_reason"
                ],
                "comparison_classification_reason": comparison[
                    "phenotype_row"
                ]["classification_reason"],
                "primary_detection_counts": _phenotype_detection_counts(
                    primary_phenotype_row
                ),
                "comparison_detection_counts": _phenotype_detection_counts(
                    comparison["phenotype_row"]
                ),
            }
        )
    if independent_only:
        summary["outputs"] = independent_outputs
        summary["panel_letters_in_individual_files"] = False
    out_json = out_dir / f"{stem}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not independent_only:
        _write_readme(out_dir, out_png, out_pdf, display_subject)
    print(out_png)
    print(out_pdf)
    print(out_json)
    return out_png, out_pdf, out_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="epilepsiae_1146")
    ap.add_argument("--seizure-idx", type=int, default=7)
    ap.add_argument("--reference", default="car", choices=("car", "bipolar", "none"))
    ap.add_argument("--pre-sec", type=float, default=130.0)
    ap.add_argument("--post-sec", type=float, default=20.0)
    ap.add_argument("--baseline-start", type=float, default=-120.0)
    ap.add_argument("--baseline-end", type=float, default=-90.0)
    ap.add_argument("--display-sec", type=float, default=10.0)
    ap.add_argument("--n-channels", type=int, default=15)
    ap.add_argument("--channel-source", default="lagpat", choices=("lagpat", "ictal"))
    ap.add_argument("--spectral-channel", default=None)
    ap.add_argument(
        "--spectral-profile",
        default="broadband",
        choices=("broadband", "gamma"),
        help="channel-selection profile; gamma is for the supplemental gamma example",
    )
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument(
        "--display-subject-label",
        default=None,
        help="optional paper-facing subject label; does not change the data subject",
    )
    ap.add_argument(
        "--independent-only",
        action="store_true",
        help="write label-free fig3-panela/b crops instead of the legacy combined canvas",
    )
    ap.add_argument(
        "--compact-main",
        action="store_true",
        help="show only 20 s baseline [-110,-90] and [-10,20] s with an explicit omitted interval",
    )
    ap.add_argument("--comparison-subject", default=None)
    ap.add_argument("--comparison-seizure-idx", type=int, default=None)
    ap.add_argument("--comparison-spectral-channel", default=None)
    ap.add_argument(
        "--comparison-spectral-profile",
        default="gamma",
        choices=("broadband", "gamma"),
    )
    ap.add_argument("--channels", nargs="*", default=None)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
