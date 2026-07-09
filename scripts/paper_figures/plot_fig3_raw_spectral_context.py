#!/usr/bin/env python3
"""Raw peri-onset EEG + standard-band spectrum context for Topic 5 Fig3.

This figure is a deliberately low-level bridge before z-ER / field projection:

  a. stacked raw intracranial traces on one continuous peri-onset axis;
  b. baseline-normalized TFR on the same continuous axis;
  c. standard-band energy enhancement trajectories relative to baseline.

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
from matplotlib.lines import Line2D
import numpy as np
from scipy.signal import spectrogram

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ictal_onset_extraction import extract_seizure_window  # noqa: E402
from src.plot_style import FS_LABEL, FS_TICK, savefig_pub, style_panel  # noqa: E402
from src.topic5_ictal_recruitment import bipolar_alias_label  # noqa: E402


OUT_DIR = ROOT / "results/paper-ready-figure/fig3_sup2_raw_spectral_context/figures"
T0_CACHE = ROOT / "results/topic5_ictal_recruitment/t0_feature_cache_v2_windows"
LAGPAT_CHANNEL_SOURCES = (
    ROOT / "results/interictal_propagation_masked/rank_displacement/per_subject",
    ROOT / "results/interictal_propagation_masked_broad/rank_displacement/per_subject",
)

STANDARD_BANDS = (
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 80.0),
    ("HFA", 80.0, 150.0),
    ("1-150", 1.0, 150.0),
)

WINDOW_COLORS = {
    "baseline": "#4C78A8",
    "eeg_onset": "#D98C52",
    "early_ictal": "#B2182B",
}

BAND_LINE_COLORS = {
    "alpha": "#59A14F",
    "beta": "#ECA82C",
    "gamma": "#D98C52",
    "HFA": "#B2182B",
    "1-150": "#5B5B5B",
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
    pad = 0.08 * max(1.0, ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_yticks(offsets)
    ax.set_yticklabels([bipolar_alias_label(str(sw.ch_names[i])) for i in ch_idx], fontsize=6)
    ax.tick_params(axis="x", labelsize=FS_TICK - 4)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_xlabel("time from clinical onset (s)", fontsize=FS_LABEL - 2)


def _shade_windows(ax: plt.Axes, baseline: tuple[float, float], eeg_rel: float | None, post_window: tuple[float, float]) -> None:
    ax.axvspan(float(baseline[0]), float(baseline[1]), color=WINDOW_COLORS["baseline"], alpha=0.10, lw=0)
    if eeg_rel is not None:
        ax.axvspan(float(eeg_rel) - 5.0, float(eeg_rel) + 5.0, color=WINDOW_COLORS["eeg_onset"], alpha=0.08, lw=0)
    ax.axvspan(float(post_window[0]), float(post_window[1]), color=WINDOW_COLORS["early_ictal"], alpha=0.08, lw=0)


def _mark_onsets(ax: plt.Axes, sw, *, label_lines: bool = True) -> None:
    eeg_rel = _eeg_rel_sec(sw)
    for x, label, color, ls in (
        (eeg_rel, "EEG", "#7A4F9A", ":"),
        (0.0, "clinical", "0.18", "--"),
    ):
        if x is None:
            continue
        ax.axvline(float(x), color=color, lw=1.0, ls=ls)
        if label_lines:
            ax.text(float(x), 0.98, label, transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=7, color=color, rotation=90)


def _label_shaded_windows(
    ax: plt.Axes,
    baseline: tuple[float, float],
    eeg_rel: float | None,
    post_window: tuple[float, float],
    *,
    y: float = 0.96,
) -> None:
    items: list[tuple[float, str, str, float]] = [
        ((float(baseline[0]) + float(baseline[1])) / 2.0, "BASELINE", WINDOW_COLORS["baseline"], y),
    ]
    clinical_center = (float(post_window[0]) + float(post_window[1])) / 2.0
    clinical_y = y
    if eeg_rel is not None:
        items.append((float(eeg_rel), "EEG ONSET", "#7A4F9A", y))
        if abs(clinical_center - float(eeg_rel)) < 12.0:
            clinical_y = y - 0.14
    items.append((clinical_center, "CLINICAL 0-10 s", WINDOW_COLORS["early_ictal"], clinical_y))
    for x, label, color, yy in items:
        ax.text(
            x,
            yy,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.5,
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
        band_p95 = {name: float(np.nanpercentile(vals[win], 95)) for name, vals in curves.items()}
        min_p95 = min(band_p95.values())
        mean_p95 = float(np.mean(list(band_p95.values())))
        score = min_p95 + 0.25 * mean_p95
        if best is None or score > best[0]:
            best = (score, min_p95, int(ci), band_p95)
    if best is None:
        raise RuntimeError("could not select spectral channel")
    return best[2], {
        "method": "max_min_95pct_db_across_alpha_beta_gamma_HFA_1_150Hz_in_onset_to_early_ictal_window",
        "score_window_sec": [score_lo, score_hi],
        "score": best[0],
        "score_min_band_95pct_db": best[1],
        "band_95pct_db": best[3],
    }


def _band_enhancement(freqs: np.ndarray, rel_t: np.ndarray, pxx: np.ndarray, baseline: tuple[float, float]) -> dict[str, np.ndarray]:
    bl = (rel_t >= float(baseline[0])) & (rel_t < float(baseline[1]))
    out: dict[str, np.ndarray] = {}
    for name, lo, hi in STANDARD_BANDS:
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
) -> tuple[plt.Figure, dict]:
    fig = plt.figure(figsize=(11.4, 9.4))
    gs = fig.add_gridspec(3, 5, height_ratios=[2.05, 1.25, 1.8], hspace=0.60, wspace=0.46)
    eeg_rel = _eeg_rel_sec(sw)

    idx = _finite_window(sw.t_axis, *x_window)
    x = sw.signal[np.asarray(ch_idx), :][:, idx[:: max(1, int(round(float(sw.fs) / 180.0)))]]
    trace_scale = max(40.0, float(np.nanpercentile(np.abs(x - np.nanmedian(x, axis=1, keepdims=True)), 95) * 3.0))

    ax_raw = fig.add_subplot(gs[0, :])
    _shade_windows(ax_raw, baseline, eeg_rel, post_window)
    _plot_continuous_stacked(ax_raw, sw, ch_idx, x_window, scale=trace_scale)
    _mark_onsets(ax_raw, sw, label_lines=False)
    _label_shaded_windows(ax_raw, baseline, eeg_rel, post_window)
    ax_raw.set_title("raw intracranial traces on one continuous onset axis", fontsize=FS_LABEL, pad=6)
    style_panel(ax_raw, "a", label_x=-0.055, label_y=1.04)

    freqs, rel_t, db, pxx, _base = _channel_tfr(sw, spectral_idx, x_window, baseline)
    band_curves = _band_enhancement(freqs, rel_t, pxx, baseline)

    ax_tfr = fig.add_subplot(gs[1, :])
    _shade_windows(ax_tfr, baseline, eeg_rel, post_window)
    vmax = float(np.nanpercentile(np.abs(db), 98))
    vmax = max(3.0, min(vmax, 14.0))
    mesh = ax_tfr.pcolormesh(rel_t, freqs, db, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    _mark_onsets(ax_tfr, sw, label_lines=False)
    _label_shaded_windows(ax_tfr, baseline, eeg_rel, post_window, y=0.94)
    ax_tfr.set_xlim(*x_window)
    ax_tfr.set_ylim(1.0, 150.0)
    ax_tfr.set_ylabel("frequency (Hz)", fontsize=FS_LABEL - 2)
    ax_tfr.set_xlabel("time from clinical onset (s)", fontsize=FS_LABEL - 2)
    spectral_channel = bipolar_alias_label(str(sw.ch_names[int(spectral_idx)]))
    ax_tfr.set_title(f"TFR on representative lagPat channel {spectral_channel}: dB vs baseline", fontsize=FS_LABEL, pad=6)
    style_panel(ax_tfr, "b", label_x=-0.055, label_y=1.04)
    cbar = fig.colorbar(mesh, ax=ax_tfr, pad=0.012, fraction=0.018)
    cbar.set_label("dB vs baseline", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    band_axes = []
    for j, (name, lo, hi) in enumerate(STANDARD_BANDS):
        ax = fig.add_subplot(gs[2, j])
        _shade_windows(ax, baseline, eeg_rel, post_window)
        smoothed = _smooth_curve(band_curves[name], rel_t, smooth_sec=2.0)
        ax.plot(rel_t, smoothed, color=BAND_LINE_COLORS[name], lw=1.7)
        _mark_onsets(ax, sw)
        ax.axhline(0.0, color="0.35", lw=0.7)
        ax.set_xlim(*x_window)
        finite = smoothed[np.isfinite(smoothed)]
        if finite.size:
            lo_y = float(np.nanpercentile(finite, 1))
            hi_y = float(np.nanpercentile(finite, 99.5))
            span = max(1.0, hi_y - lo_y)
            ax.set_ylim(min(-1.0, lo_y - 0.08 * span), max(2.0, hi_y + 0.08 * span))
        ax.set_title(f"{name}\n{lo:g}-{hi:g} Hz", fontsize=8.5, pad=4)
        ax.tick_params(labelsize=7, length=3)
        ax.spines[["top", "right"]].set_visible(False)
        if j == 0:
            ax.set_ylabel("dB vs baseline", fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("s", fontsize=8)
        band_axes.append(ax)
    fig.text(0.012, 0.318, "c", fontsize=24, fontweight="bold", va="bottom", ha="left")

    handles = [
        Line2D([0], [0], color=WINDOW_COLORS["baseline"], lw=7, alpha=0.25, label="baseline window"),
        Line2D([0], [0], color=WINDOW_COLORS["eeg_onset"], lw=7, alpha=0.25, label="EEG-onset neighborhood"),
        Line2D([0], [0], color=WINDOW_COLORS["early_ictal"], lw=7, alpha=0.25, label="clinical 0-10 s field input"),
        Line2D([0], [0], color="#7A4F9A", lw=1.2, ls=":", label="EEG onset"),
        Line2D([0], [0], color="0.18", lw=1.2, ls="--", label="clinical onset"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.52, 0.01), ncol=5)
    fig.subplots_adjust(left=0.085, right=0.965, top=0.94, bottom=0.15)

    eeg_win = (float(eeg_rel) - 5.0, float(eeg_rel) + 5.0) if eeg_rel is not None else (float("nan"), float("nan"))
    summary = {
        "selected_channels": [bipolar_alias_label(str(sw.ch_names[i])) for i in ch_idx],
        "channel_source": channel_source_label,
        "spectral_channel": spectral_channel,
        "spectral_channel_selection": spectral_selection,
        "spectral_summary": "single representative lagPat channel PSD, then dB vs baseline",
        "x_window_sec": list(map(float, x_window)),
        "band_enhancement_mean_db": {
            "baseline": _window_mean_by_band(band_curves, rel_t, baseline),
            "eeg_onset_neighborhood": _window_mean_by_band(band_curves, rel_t, eeg_win),
            "clinical_0_10": _window_mean_by_band(band_curves, rel_t, post_window),
        },
    }
    return fig, summary


def _write_readme(out_png: Path, out_pdf: Path, ds_sid: str, seizure_idx: int) -> None:
    readme = OUT_DIR / "README.md"
    readme.write_text(
        "# Fig3-Sup2 Raw Spectral Context\n\n"
        f"### {out_png.name} / {out_pdf.name}\n\n"
        f"这张图使用 `{ds_sid}` 的 seizure `{seizure_idx}`，在进入 z-ER、field projection 和 maxAB 相似性之前，先展示同一批原始发作数据的三个层次：远端 baseline、EEG-onset 附近、clinical onset 后 0-10 s。"
        "上排是连续时间轴上的 lagPat 电极原始波形，中排是一个代表性 lagPat 单通道的 baseline-normalized TFR，下排是同一代表通道的 alpha/beta/gamma/HFA/1-150 Hz 相对 baseline 的能量增强轨迹。"
        "它只承担解释和质控作用，不是 cohort 统计，也不证明 timing-order replay 或机制。\n\n"
        "**关注点**：baseline、EEG onset 和 clinical onset 必须在同一条 x 轴上读；baseline 是 z 标准化参考，不等于发作前最后几秒；0-10 s field projection 消费的是早期 ictal 能量场，而不是原始 z-ER 图本身。\n",
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
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{ds_sid}_seizure_{int(args.seizure_idx):02d}_raw_spectral_context"
    out_png = OUT_DIR / f"{stem}.png"
    out_pdf = OUT_DIR / f"{stem}.pdf"
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
    )
    savefig_pub(fig_pdf, out_pdf, dpi=300)

    summary.update(
        {
            "subject": ds_sid,
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
            "tier": "single-seizure explanatory context; not a cohort statistic",
        }
    )
    out_json = OUT_DIR / f"{stem}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(out_png, out_pdf, ds_sid, int(args.seizure_idx))
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
    ap.add_argument("--channels", nargs="*", default=None)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
