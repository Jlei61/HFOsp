#!/usr/bin/env python3
"""Raw peri-onset EEG + standard-band spectrum context for Topic 5 Fig3a.

This figure is a deliberately low-level bridge before z-ER / field projection:

  left top: stacked raw intracranial traces on one continuous peri-onset axis;
  left bottom: baseline-normalized TFR on the same continuous axis;
  right 2x2: low-band, gamma, high-gamma, and broadband trajectories.

It is explanatory material, not a cohort statistic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.signal import spectrogram

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
        ax.plot(t, yy, lw=0.38, color="0.20", alpha=0.85)
    ax.set_xlim(float(window[0]), float(window[1]))
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
    ax.set_xlabel("time from clinical onset (s)", fontsize=9)


def _shade_windows(ax: plt.Axes, baseline: tuple[float, float], eeg_rel: float | None, post_window: tuple[float, float]) -> None:
    ax.axvspan(float(baseline[0]), float(baseline[1]), color=WINDOW_COLORS["baseline"], alpha=0.10, lw=0)
    ax.axvspan(float(post_window[0]), float(post_window[1]), color=WINDOW_COLORS["early_ictal"], alpha=0.15, lw=0)


def _label_shaded_windows(
    ax: plt.Axes,
    baseline: tuple[float, float],
    eeg_rel: float | None,
    post_window: tuple[float, float],
    *,
    y: float = 0.96,
    post_label: str = "CLINICAL ONSET",
) -> None:
    items: list[tuple[float, str, str, float]] = [
        ((float(baseline[0]) + float(baseline[1])) / 2.0, "BASELINE", WINDOW_COLORS["baseline"], y),
    ]
    items.append((float(post_window[0]), post_label, WINDOW_COLORS["early_ictal"], y))
    for x, label, color, yy in items:
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
) -> tuple[plt.Figure, dict]:
    # Compact Fig3a layout: raw/TFR share the wide left column; the four
    # band-energy trajectories occupy a balanced 2x2 block on the right.
    fig = plt.figure(figsize=(12.1, 4.1))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.12, 1.0], wspace=0.11)
    left = outer[0, 0].subgridspec(
        2,
        2,
        width_ratios=[1.0, 0.018],
        height_ratios=[1.65, 1.0],
        hspace=0.62,
        wspace=0.025,
    )
    right = outer[0, 1].subgridspec(2, 2, hspace=0.48, wspace=0.20)
    eeg_rel = _eeg_rel_sec(sw)

    idx = _finite_window(sw.t_axis, *x_window)
    x = sw.signal[np.asarray(ch_idx), :][:, idx[:: max(1, int(round(float(sw.fs) / 180.0)))]]
    trace_scale = max(40.0, float(np.nanpercentile(np.abs(x - np.nanmedian(x, axis=1, keepdims=True)), 95) * 3.0))

    ax_raw = fig.add_subplot(left[0, 0])
    _shade_windows(ax_raw, baseline, eeg_rel, post_window)
    _plot_continuous_stacked(ax_raw, sw, ch_idx, x_window, scale=trace_scale)
    _label_shaded_windows(ax_raw, baseline, eeg_rel, post_window)
    ax_raw.set_title(display_subject, fontsize=10.0, fontweight="bold", loc="left", pad=5)
    style_panel(ax_raw)
    ax_raw.tick_params(axis="x", labelsize=9, width=0.9, length=4)
    ax_raw.tick_params(axis="y", labelsize=6, length=0)

    freqs, rel_t, db, pxx, _base = _channel_tfr(sw, spectral_idx, x_window, baseline)
    band_curves = _band_enhancement(freqs, rel_t, pxx, baseline)

    ax_tfr = fig.add_subplot(left[1, 0])
    _shade_windows(ax_tfr, baseline, eeg_rel, post_window)
    vmax = float(np.nanpercentile(np.abs(db), 98))
    vmax = max(3.0, min(vmax, 14.0))
    mesh = ax_tfr.pcolormesh(rel_t, freqs, db, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    _label_shaded_windows(ax_tfr, baseline, eeg_rel, post_window, y=0.94)
    ax_tfr.set_xlim(*x_window)
    ax_tfr.set_ylim(1.0, 150.0)
    ax_tfr.set_ylabel("frequency (Hz)", fontsize=9)
    ax_tfr.set_xlabel("time from clinical onset (s)", fontsize=9)
    spectral_channel = bipolar_alias_label(str(sw.ch_names[int(spectral_idx)]))
    ax_tfr.set_title(f"TFR on {spectral_channel}", fontsize=10.0, fontweight="bold", loc="left", pad=5)
    style_panel(ax_tfr)
    ax_tfr.tick_params(labelsize=8, width=0.9, length=4)
    cax = fig.add_subplot(left[1, 1])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.ax.set_title("TFR\n(dB)", fontsize=6.5, pad=3, fontweight="bold")
    cbar.ax.tick_params(labelsize=6, length=2)

    band_axes = []
    band_ylim_candidates: list[tuple[float, float]] = []
    for j, (name, lo, hi) in enumerate(DISPLAY_BANDS):
        row = j // 2
        col = j % 2
        sharey_ax = band_axes[j - 1] if col == 1 else None
        ax = fig.add_subplot(right[row, col], sharey=sharey_ax)
        _shade_windows(ax, baseline, eeg_rel, post_window)
        smoothed = _smooth_curve(band_curves[name], rel_t, smooth_sec=2.0)
        ax.plot(rel_t, smoothed, color=BAND_LINE_COLORS[name], lw=1.45)
        ax.axhline(0.0, color="0.35", lw=0.6)
        ax.set_xlim(*x_window)
        ax.margins(x=0)
        finite = smoothed[np.isfinite(smoothed)]
        if finite.size:
            lo_y = float(np.nanpercentile(finite, 1))
            hi_y = float(np.nanpercentile(finite, 99.5))
            span = max(1.0, hi_y - lo_y)
            ylim_candidate = (min(-1.0, lo_y - 0.08 * span), max(2.0, hi_y + 0.08 * span))
        else:
            ylim_candidate = (-1.0, 2.0)
        band_ylim_candidates.append(ylim_candidate)
        ax.set_title(f"{name}\n({lo:g}-{hi:g} Hz)", fontsize=7.5, pad=2.5, fontweight="bold", linespacing=0.95)
        ax.tick_params(labelsize=6, length=2.5, width=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        if col == 0:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.set_ylabel("dB vs baseline", fontsize=6.5, labelpad=2)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_xlabel("Time (s)", fontsize=7, labelpad=1)
        band_axes.append(ax)

    for row in range(2):
        row_idx = (2 * row, 2 * row + 1)
        row_lo = min(band_ylim_candidates[i][0] for i in row_idx)
        row_hi = max(band_ylim_candidates[i][1] for i in row_idx)
        band_axes[row_idx[0]].set_ylim(row_lo, row_hi)
    fig.subplots_adjust(left=0.074, right=0.98, top=0.865, bottom=0.10)

    eeg_win = (float(eeg_rel) - 5.0, float(eeg_rel) + 5.0) if eeg_rel is not None else (float("nan"), float("nan"))
    summary = {
        "selected_channels": [bipolar_alias_label(str(sw.ch_names[i])) for i in ch_idx],
        "channel_source": channel_source_label,
        "spectral_channel": spectral_channel,
        "spectral_channel_selection": spectral_selection,
        "spectral_summary": "single representative lagPat channel PSD, then dB vs baseline",
        "x_window_sec": list(map(float, x_window)),
        "paper_role": "Fig3-A raw spectral context",
        "layout": "raw/TFR aligned on the left; low bands/gamma/high-gamma/broadband trajectories in a right-side 2x2 block",
        "right_axis_contract": "row-shared y limits; y ticks and dB label shown only on the left panel of each row",
        "displayed_bands": [name for name, _lo, _hi in DISPLAY_BANDS],
        "sidecar_only_bands": [name for name, _lo, _hi in ANALYSIS_BANDS if name not in DISPLAY_BAND_NAMES],
        "band_enhancement_mean_db": {
            "baseline": _window_mean_by_band(band_curves, rel_t, baseline),
            "eeg_onset_neighborhood": _window_mean_by_band(band_curves, rel_t, eeg_win),
            "clinical_0_10": _window_mean_by_band(band_curves, rel_t, post_window),
        },
    }
    return fig, summary


def _write_readme(
    out_dir: Path,
    out_png: Path,
    out_pdf: Path,
    ds_sid: str,
    seizure_idx: int,
) -> None:
    readme = out_dir / "README.md"
    readme.write_text(
        "# Fig3-A Raw Spectral Context\n\n"
        f"### {out_png.name} / {out_pdf.name}\n\n"
        f"这张图使用 `{ds_sid}` 的 seizure `{seizure_idx}`，在进入 z-ER、field projection 和 maxAB 相似性之前，展示远端 baseline 与 clinical onset 附近的原始发作信号。"
        "左侧上排是连续时间轴上的 lagPat 电极原始波形，左侧下排是同一时间轴上的代表性 lagPat 单通道 baseline-normalized TFR；右侧 2×2 小图依次展示同一代表通道 low bands (1-30 Hz)、gamma (30-80 Hz)、high-gamma (80-150 Hz) 和 broadband (1-150 Hz) 相对 baseline 的能量增强轨迹。"
        "右侧同一行共用 y 轴范围，数值 ticks 与 dB 标签只放在每行左图。图面不标 EEG onset，也不画 onset 虚线；alpha 与 beta 只保留在 summary JSON 的通道选择审计中。"
        "它只承担解释和质控作用，不是 cohort 统计，也不证明 timing-order replay 或机制。\n\n"
        "**关注点**：raw SEEG 与 TFR 的时间轴必须严格对齐；baseline 是标准化参考，不等于发作前最后几秒；clinical-onset 阴影表示早期 ictal field input，而不是原始 z-ER 图本身。\n",
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
    display_subject = public_patient_label(
        display_dataset, display_raw_subject
    )

    fig, summary = _make_figure(
        sw,
        ch_idx,
        spectral_idx,
        baseline=baseline,
        x_window=x_window,
        post_window=post_window,
        channel_source_label=channel_source_label,
        spectral_selection=spectral_selection,
        display_subject=display_subject,
    )
    out_dir = Path(getattr(args, "output_dir", None) or OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{ds_sid}_seizure_{int(args.seizure_idx):02d}_raw_spectral_context"
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
    )
    savefig_pub(fig_pdf, out_pdf, dpi=300)

    summary.update(
        {
            "subject": ds_sid,
            "public_patient_label": display_subject,
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
            "tier": "paper-ready Fig3-A single-seizure explanatory context; not a cohort statistic",
        }
    )
    out_json = out_dir / f"{stem}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(out_dir, out_png, out_pdf, ds_sid, int(args.seizure_idx))
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
    ap.add_argument("--channels", nargs="*", default=None)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
