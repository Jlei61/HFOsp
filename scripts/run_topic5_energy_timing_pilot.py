#!/usr/bin/env python3
"""Per-seizure multi-band peri-onset energy-timing pilot.

This pilot answers only the timing question.  For each eligible seizure it
reads the committed multi-band baseline-robust-z cache, restricts to one fixed
set of lagPat-valid contacts, explicitly subtracts the distal [-120,-90] s
baseline, and extracts per-band rise/peak/duration from a spatial-Q75 trace.
It never reads A/B ranks or correlations.

E1146 has two otherwise eligible seizures missing from the long-duration band
cache for reasons irrelevant to a peri-onset analysis.  Those short windows are
recomputed with the same spectrogram, line-mask, and robust-z contract and
stored under this pilot's per_subject/cache directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_topic5_ictal_field_long_cache import GUARD_SEC, MIN_BASELINE_SEC  # noqa: E402
from scripts.paper_figures.plot_fig3_raw_spectral_context import (  # noqa: E402
    _alias_index,
    _label_shaded_windows,
    _load_lagpat_channels,
    _mark_onsets,
    _plot_continuous_stacked,
    _shade_windows,
)
from scripts.run_topic5_t0_eligibility import ICTAL_REFERENCE, _inventory_rows  # noqa: E402
from src.ictal_onset_extraction import extract_seizure_window, resolve_baseline_window  # noqa: E402
from src.topic5_energy_timing import (  # noqa: E402
    band_energy_timing,
    detect_sustained_enhancement,
    max_upward_transition,
)
from src.topic5_ictal_recruitment import _spectrogram_on_hop, bipolar_alias_label  # noqa: E402
from src.topic5_v2_band_scan import (  # noqa: E402
    band_bin_selection,
    line_noise_bin_mask,
    load_phase1_config,
    robust_z_with_flags,
)


CACHE_ROOT = ROOT / "results/topic5_ictal_recruitment/v2_band_scan/cache"
AUDIT_CSV = ROOT / "results/topic5_ictal_recruitment/t0_eligibility_audit.csv"
DEFAULT_OUT_ROOT = ROOT / "results/topic5_ictal_recruitment/peri_onset_energy_timing"

BASELINE = (-120.0, -90.0)
SEARCH = (-60.0, 20.0)
PRE_SEC = 130.0
POST_SEC = 20.0
SPATIAL_Q = 0.75
SMOOTH_SEC = 2.0
BASELINE_QUANTILE = 0.99
SUSTAIN_SEC = 2.0
SENSITIVITY_QUANTILE = 0.95
TRANSITION_FLANK_SEC = 2.0

BAND_LABELS = {
    "delta_HYP_slow": "delta 1-4 Hz",
    "theta_preictal_PAC": "theta 4-8 Hz",
    "alpha_sharp_leq13": "alpha 8-13 Hz",
    "beta_LVFA_low": "beta 13-30 Hz",
    "gamma_LVFA": "gamma 30-80 Hz",
    "hg_low_ripple": "ripple 80-150 Hz",
    "ripple_high": "fast ripple 150-250 Hz",
    "multiband_consensus": "multiband consensus",
}

BAND_SHORT = {
    "delta_HYP_slow": "δ",
    "theta_preictal_PAC": "θ",
    "alpha_sharp_leq13": "α",
    "beta_LVFA_low": "β",
    "gamma_LVFA": "γ",
    "hg_low_ripple": "R",
    "ripple_high": "FR",
    "multiband_consensus": "ALL",
}

BAND_COLORS = {
    "delta_HYP_slow": "#4C78A8",
    "theta_preictal_PAC": "#72B7B2",
    "alpha_sharp_leq13": "#59A14F",
    "beta_LVFA_low": "#ECA82C",
    "gamma_LVFA": "#F28E2B",
    "hg_low_ripple": "#D62728",
    "ripple_high": "#9C2F45",
    "multiband_consensus": "#3A3A3A",
}


def _eligible_indices(ds_sid: str) -> list[int]:
    rows = list(csv.DictReader(AUDIT_CSV.open(encoding="utf-8")))
    return sorted(
        int(r["seizure_idx"])
        for r in rows
        if r["subject_id"] == ds_sid
        and str(r["analysis_eligible"]).strip().lower() in {"true", "1", "yes"}
    )


def _primary_band_specs() -> list[tuple[str, float, float]]:
    cfg = load_phase1_config()
    return [(str(name), float(lo), float(hi)) for name, lo, hi in cfg["bands"]["primary"]]


def _eeg_rel(row: dict) -> float:
    return float(row["eeg_onset_epoch"]) - float(row["clin_onset_epoch"])


def _seizure_id(row: dict, idx: int) -> str:
    return str(row.get("seizure_id") or row.get("seizure") or idx)


def _fallback_cache_paths(out_root: Path, ds_sid: str) -> tuple[Path, Path]:
    root = out_root / "per_subject" / "cache"
    return root / f"{ds_sid}_peri_onset_missing_bands.npz", root / f"{ds_sid}_peri_onset_missing_bands.json"


def _compute_missing_bands(
    ds_sid: str,
    missing: list[int],
    inv_rows: list[dict],
    specs: list[tuple[str, float, float]],
    out_root: Path,
) -> tuple[dict[str, np.ndarray], dict]:
    npz_path, json_path = _fallback_cache_paths(out_root, ds_sid)
    if npz_path.exists() and json_path.exists():
        z = np.load(npz_path, allow_pickle=True)
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        if sorted(int(x) for x in meta.get("seizure_idxs", [])) == sorted(missing):
            return {k: np.asarray(z[k]) for k in z.files}, meta

    cfg = load_phase1_config()
    ln = cfg["line_noise"]
    spec_win = float(cfg["power"]["spectrogram_win_sec"])
    spec_hop = float(cfg["power"]["spectrogram_hop_sec"])
    dataset, sid = ds_sid.split("_", 1)
    arrays: dict[str, np.ndarray] = {}
    meta = {
        "subject": ds_sid,
        "seizure_idxs": [],
        "drops": [],
        "pre_sec": PRE_SEC,
        "post_sec": POST_SEC,
        "spectrogram_win_sec": spec_win,
        "spectrogram_hop_sec": spec_hop,
        "source": "raw short-window fallback for long-cache-ineligible seizures",
    }
    channels: list[str] | None = None
    for idx in missing:
        try:
            sw = extract_seizure_window(
                f"{dataset}/{sid}",
                idx,
                pre_sec=PRE_SEC,
                post_sec=POST_SEC,
                results_root=ROOT / "results",
                reference=ICTAL_REFERENCE[dataset],
            )
            names = [bipolar_alias_label(str(x)) for x in sw.ch_names]
            if channels is None:
                channels = names
            elif names != channels:
                raise ValueError("fallback channel ordering changed across seizures")
            f, t, sxx = _spectrogram_on_hop(sw.signal, sw.fs, spec_win, spec_hop)
            rel_t = np.asarray(t, float) - float(sw.pre_sec)
            line_mask = line_noise_bin_mask(f, ln["harmonics_hz"], ln["halfwidth_hz"])
            eeg_rel = _eeg_rel(inv_rows[idx])
            for band, lo, hi in specs:
                bmask, _, _ = band_bin_selection(f, lo, hi, line_mask, half_open=True)
                if not np.any(bmask):
                    raise ValueError(f"{band}: no usable FFT bins")
                power = np.asarray(sxx[:, bmask, :], float).sum(axis=1)
                logp = np.log(np.maximum(power, 1e-30))
                bl = resolve_baseline_window(
                    logp.shape[1],
                    hop_sec=spec_hop,
                    pre_sec=sw.pre_sec,
                    buffer_sec=GUARD_SEC,
                    eeg_onset_rel_sec=eeg_rel,
                    min_baseline_valid_sec=MIN_BASELINE_SEC,
                )
                zt, _ = robust_z_with_flags(
                    logp,
                    (bl.start_idx, bl.end_idx),
                    spec_hop,
                    MIN_BASELINE_SEC,
                )
                arrays[f"{band}__zt__{idx}"] = zt.astype(np.float32)
                arrays[f"{band}__relt__{idx}"] = rel_t.astype(np.float32)
            meta["seizure_idxs"].append(idx)
            del sxx
        except Exception as exc:  # noqa: BLE001 - fail-closed provenance
            meta["drops"].append({"seizure_idx": idx, "reason": f"{type(exc).__name__}:{exc}"})
    if channels is not None:
        arrays["channels"] = np.asarray(channels)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return arrays, meta


def _timing_row(
    ds_sid: str,
    idx: int,
    seizure_id: str,
    band: str,
    source: str,
    n_contacts: int,
    eeg_rel: float,
    timing,
    sensitivity,
    transition,
) -> dict:
    rise_eeg = timing.rise_sec - eeg_rel if timing.detected else float("nan")
    peak_eeg = timing.peak_sec - eeg_rel
    return {
        "subject": ds_sid,
        "seizure_idx": int(idx),
        "seizure_id": seizure_id,
        "band": band,
        "band_label": BAND_LABELS[band],
        "source": source,
        "n_contacts": int(n_contacts),
        "eeg_onset_rel_clinical_sec": float(eeg_rel),
        "detected_q99_sustain2s": bool(timing.detected),
        "rise_rel_clinical_sec": float(timing.rise_sec),
        "rise_rel_eeg_sec": float(rise_eeg),
        "peak_rel_clinical_sec": float(timing.peak_sec),
        "peak_rel_eeg_sec": float(peak_eeg),
        "peak_value_delta_z": float(timing.peak_value),
        "threshold_q99_delta_z": float(timing.threshold),
        "total_above_threshold_sec": float(timing.total_above_sec),
        "longest_above_threshold_sec": float(timing.longest_above_sec),
        "detected_q95_sustain2s": bool(sensitivity.detected),
        "rise_q95_rel_clinical_sec": float(sensitivity.rise_sec),
        "rise_q95_rel_eeg_sec": float(sensitivity.rise_sec - eeg_rel) if sensitivity.detected else float("nan"),
        "transition_detected_q99": bool(transition.detected),
        "transition_rel_clinical_sec": float(transition.transition_sec),
        "transition_rel_eeg_sec": float(transition.transition_sec - eeg_rel),
        "transition_step_delta_z": float(transition.step_delta),
        "transition_threshold_q99_delta_z": float(transition.threshold),
        "transition_flank_sec": float(transition.flank_sec),
    }


def _finite(values) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def _quartiles(values) -> tuple[float, float, float]:
    x = _finite(values)
    if not x.size:
        return float("nan"), float("nan"), float("nan")
    return tuple(float(v) for v in np.percentile(x, [25, 50, 75]))


def _subject_band_summary(rows: list[dict], band_order: list[str], n_eligible: int) -> list[dict]:
    out = []
    for band in band_order:
        rr = [r for r in rows if r["band"] == band]
        detected = [r for r in rr if r["detected_q99_sustain2s"]]
        transitions = [r for r in rr if r["transition_detected_q99"]]
        c25, c50, c75 = _quartiles(r["rise_rel_clinical_sec"] for r in detected)
        e25, e50, e75 = _quartiles(r["rise_rel_eeg_sec"] for r in detected)
        p25, p50, p75 = _quartiles(r["peak_rel_eeg_sec"] for r in rr)
        tc25, tc50, tc75 = _quartiles(r["transition_rel_clinical_sec"] for r in transitions)
        te25, te50, te75 = _quartiles(r["transition_rel_eeg_sec"] for r in transitions)
        out.append(
            {
                "band": band,
                "band_label": BAND_LABELS[band],
                "n_eligible": int(n_eligible),
                "n_computed": len(rr),
                "n_detected_q99_sustain2s": len(detected),
                "sustained_detection_fraction": len(detected) / n_eligible if n_eligible else float("nan"),
                "rise_clinical_q25_sec": c25,
                "rise_clinical_median_sec": c50,
                "rise_clinical_q75_sec": c75,
                "rise_eeg_q25_sec": e25,
                "rise_eeg_median_sec": e50,
                "rise_eeg_q75_sec": e75,
                "peak_eeg_q25_sec": p25,
                "peak_eeg_median_sec": p50,
                "peak_eeg_q75_sec": p75,
                "n_detected_q95_sustain2s": sum(bool(r["detected_q95_sustain2s"]) for r in rr),
                "n_transition_detected_q99": len(transitions),
                "transition_detection_fraction": len(transitions) / n_eligible if n_eligible else float("nan"),
                "transition_clinical_q25_sec": tc25,
                "transition_clinical_median_sec": tc50,
                "transition_clinical_q75_sec": tc75,
                "transition_eeg_q25_sec": te25,
                "transition_eeg_median_sec": te50,
                "transition_eeg_q75_sec": te75,
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mark_reference_lines(ax, eeg_rel: float) -> None:
    ax.axvline(0.0, color="0.15", lw=0.9, ls="--")
    ax.axvline(float(eeg_rel), color="#7A4F9A", lw=1.0, ls=":")


def _plot_seizure(
    ds_sid: str,
    idx: int,
    seizure_id: str,
    eeg_rel: float,
    band_order: list[str],
    rel_t: np.ndarray,
    traces: dict[str, np.ndarray],
    timing_by_band: dict[str, object],
    transition_by_band: dict[str, object],
    raw_sw,
    raw_channel_idx: list[int],
    out_path: Path,
) -> None:
    plot_bands = band_order + ["multiband_consensus"]
    fig = plt.figure(figsize=(12.4, 13.0))
    gs = fig.add_gridspec(
        4,
        4,
        height_ratios=[1.75, 1.10, 1.0, 1.0],
        hspace=0.62,
        wspace=0.28,
    )

    raw_window = (BASELINE[0], SEARCH[1])
    raw_mask = (raw_sw.t_axis >= raw_window[0]) & (raw_sw.t_axis < raw_window[1])
    raw_decim = max(1, int(round(float(raw_sw.fs) / 180.0)))
    raw_x = raw_sw.signal[np.asarray(raw_channel_idx), :][:, np.flatnonzero(raw_mask)[::raw_decim]]
    trace_scale = max(
        40.0,
        float(
            np.nanpercentile(
                np.abs(raw_x - np.nanmedian(raw_x, axis=1, keepdims=True)),
                95,
            )
            * 3.0
        ),
    )
    ax_raw = fig.add_subplot(gs[0, :])
    _shade_windows(ax_raw, BASELINE, eeg_rel, (0.0, 10.0))
    _plot_continuous_stacked(
        ax_raw,
        raw_sw,
        raw_channel_idx,
        raw_window,
        scale=trace_scale,
    )
    _mark_onsets(ax_raw, raw_sw, label_lines=False)
    _label_shaded_windows(ax_raw, BASELINE, eeg_rel, (0.0, 10.0), y=0.97)
    ax_raw.set_title(
        f"raw intracranial traces on the same {len(raw_channel_idx)} timing contacts",
        fontsize=11,
        pad=6,
    )

    ax_hm = fig.add_subplot(gs[1, :])
    hm = np.vstack([traces[b] for b in plot_bands])
    use = (rel_t >= BASELINE[0]) & (rel_t <= SEARCH[1])
    vmax = float(np.nanpercentile(np.abs(hm[:, use]), 98))
    vmax = max(2.0, min(vmax, 8.0))
    mesh = ax_hm.imshow(
        hm[:, use],
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=[float(rel_t[use][0]), float(rel_t[use][-1]), len(plot_bands) - 0.5, -0.5],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax_hm.set_yticks(np.arange(len(plot_bands)))
    ax_hm.set_yticklabels([BAND_SHORT[b] for b in plot_bands])
    ax_hm.axvspan(*BASELINE, color="#4C78A8", alpha=0.10, lw=0)
    ax_hm.axvline(SEARCH[0], color="0.5", lw=0.8, ls="--")
    _mark_reference_lines(ax_hm, eeg_rel)
    ax_hm.set_xlim(BASELINE[0], SEARCH[1])
    ax_hm.set_xlabel("time from clinical onset (s)")
    ax_hm.set_title("distal-baseline-referenced multi-band spatial Q75")
    cb = fig.colorbar(mesh, ax=ax_hm, pad=0.01, fraction=0.02)
    cb.set_label("delta robust-z")

    axes = []
    for pos, band in enumerate(plot_bands):
        ax = fig.add_subplot(gs[2 + pos // 4, pos % 4])
        axes.append(ax)
        y = traces[band]
        timing = timing_by_band[band]
        transition = transition_by_band[band]
        ax.plot(rel_t, y, color=BAND_COLORS[band], lw=1.2)
        ax.axhline(float(timing.threshold), color="0.35", lw=0.8, ls="--")
        ax.axvspan(*BASELINE, color="#4C78A8", alpha=0.08, lw=0)
        ax.axvline(SEARCH[0], color="0.7", lw=0.7, ls="--")
        _mark_reference_lines(ax, eeg_rel)
        if timing.detected:
            yr = float(np.interp(timing.rise_sec, rel_t, y))
            ax.scatter([timing.rise_sec], [yr], s=24, color="#2CA02C", zorder=5)
        yt = float(np.interp(transition.transition_sec, rel_t, y))
        ax.scatter(
            [transition.transition_sec],
            [yt],
            s=28,
            marker="D",
            facecolor="#CC79A7" if transition.detected else "none",
            edgecolor="#CC79A7",
            linewidth=0.9,
            zorder=5,
        )
        ax.scatter([timing.peak_sec], [timing.peak_value], s=24, marker="x", color="#D62728", zorder=5)
        ax.set_xlim(BASELINE[0], SEARCH[1])
        finite = y[(rel_t >= BASELINE[0]) & (rel_t <= SEARCH[1]) & np.isfinite(y)]
        if finite.size:
            lo, hi = np.percentile(finite, [1, 99.5])
            pad = 0.12 * max(1.0, hi - lo)
            ax.set_ylim(min(lo - pad, timing.threshold - pad), max(hi + pad, timing.threshold + pad))
        status = f"rise {timing.rise_sec:.1f}s" if timing.detected else "no sustained rise"
        step_status = f"step {transition.transition_sec:.1f}s" + ("" if transition.detected else " (unconfirmed)")
        ax.set_title(f"{BAND_LABELS[band]}\n{status}; {step_status}", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        if pos // 4 == 1:
            ax.set_xlabel("s from clinical onset")
        if pos % 4 == 0:
            ax.set_ylabel("delta z")

    fig.text(
        0.5,
        0.015,
        "purple dotted = EEG onset; black dashed = clinical onset; green dot = sustained rise; magenta diamond = largest 2-s upward step",
        ha="center",
        fontsize=9,
    )
    fig.suptitle(
        f"{ds_sid} seizure {idx:02d} ({seizure_id}) — raw signal and multi-band energy timing",
        fontsize=14,
        y=0.992,
    )
    fig.subplots_adjust(left=0.08, right=0.96, top=0.965, bottom=0.055)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_subject_summary(
    ds_sid: str,
    eligible: list[int],
    band_order: list[str],
    rows: list[dict],
    seizure_traces: dict[int, tuple[np.ndarray, dict[str, np.ndarray]]],
    out_path: Path,
) -> None:
    row_map = {(int(r["seizure_idx"]), r["band"]): r for r in rows}
    common_t = np.arange(SEARCH[0], SEARCH[1] + 0.05, 0.1)
    consensus = []
    for idx in eligible:
        rel_t, traces = seizure_traces[idx]
        consensus.append(np.interp(common_t, rel_t, traces["multiband_consensus"], left=np.nan, right=np.nan))
    consensus_arr = np.vstack(consensus)

    fig, axs = plt.subplots(2, 2, figsize=(12.2, 8.8), gridspec_kw={"hspace": 0.42, "wspace": 0.28})
    ax = axs[0, 0]
    vmax = max(2.0, min(float(np.nanpercentile(np.abs(consensus_arr), 98)), 8.0))
    im = ax.imshow(
        consensus_arr,
        aspect="auto",
        origin="upper",
        extent=[SEARCH[0], SEARCH[1], len(eligible) - 0.5, -0.5],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_yticks(np.arange(len(eligible)))
    ax.set_yticklabels([str(i) for i in eligible], fontsize=7)
    ax.axvline(0.0, color="0.15", lw=0.9, ls="--")
    ax.set_xlabel("time from clinical onset (s)")
    ax.set_ylabel("seizure index")
    ax.set_title("a  Multiband-consensus energy (unsorted seizures)", loc="left")
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03, label="delta robust-z")

    ax = axs[0, 1]
    x = np.arange(len(eligible))
    eeg = np.array([row_map[(idx, band_order[0])]["eeg_onset_rel_clinical_sec"] for idx in eligible], float)
    transition = np.array([
        row_map[(idx, "multiband_consensus")]["transition_rel_clinical_sec"] for idx in eligible
    ], float)
    ax.axhline(0.0, color="0.15", lw=0.9, ls="--", label="clinical onset")
    ax.plot(x, eeg, color="#7A4F9A", marker="o", ms=3.5, lw=0.9, label="EEG onset")
    trans_ok = np.array([
        bool(row_map[(idx, "multiband_consensus")]["transition_detected_q99"]) for idx in eligible
    ])
    ax.scatter(
        x[trans_ok], transition[trans_ok], color="#CC79A7", marker="D", s=26,
        label="largest upward step", zorder=4,
    )
    ax.scatter(
        x[~trans_ok], transition[~trans_ok], facecolor="none", edgecolor="#CC79A7", marker="D", s=26,
        label="step below baseline Q99", zorder=4,
    )
    for xi, e, tr in zip(x, eeg, transition):
        if np.isfinite(tr):
            ax.plot([xi, xi], [e, tr], color="0.75", lw=0.7, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in eligible], rotation=90, fontsize=7)
    ax.set_ylabel("seconds from clinical onset")
    ax.set_xlabel("seizure index")
    ax.set_title("b  Data-derived upward step versus annotated onsets", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axs[1, 0]
    transition_eeg = np.full((len(band_order), len(eligible)), np.nan)
    for bi, band in enumerate(band_order):
        for si, idx in enumerate(eligible):
            row = row_map[(idx, band)]
            if row["transition_detected_q99"]:
                transition_eeg[bi, si] = float(row["transition_rel_eeg_sec"])
    im2 = ax.imshow(transition_eeg, aspect="auto", cmap="coolwarm", vmin=-30.0, vmax=30.0)
    im2.cmap.set_bad("#E5E5E5")
    ax.set_yticks(np.arange(len(band_order)))
    ax.set_yticklabels([BAND_SHORT[b] for b in band_order])
    ax.set_xticks(np.arange(len(eligible)))
    ax.set_xticklabels([str(i) for i in eligible], rotation=90, fontsize=7)
    ax.set_xlabel("seizure index")
    ax.set_title("c  Largest upward-step latency relative to EEG onset", loc="left")
    fig.colorbar(im2, ax=ax, pad=0.01, fraction=0.03, label="s")

    ax = axs[1, 1]
    plot_bands = band_order + ["multiband_consensus"]
    fractions = []
    medians = []
    q25 = []
    q75 = []
    for band in plot_bands:
        rr = [row_map[(idx, band)] for idx in eligible]
        vals = [r["transition_rel_eeg_sec"] for r in rr if r["transition_detected_q99"]]
        a, b, c = _quartiles(vals)
        fractions.append(sum(bool(r["transition_detected_q99"]) for r in rr) / len(eligible))
        q25.append(a)
        medians.append(b)
        q75.append(c)
    xx = np.arange(len(plot_bands))
    ax.bar(xx, fractions, color=[BAND_COLORS[b] for b in plot_bands], alpha=0.78)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("detected fraction")
    ax.set_xticks(xx)
    ax.set_xticklabels([BAND_SHORT[b] for b in plot_bands])
    ax.set_title("d  Detection prevalence and timing dispersion", loc="left")
    ax2 = ax.twinx()
    med = np.asarray(medians, float)
    lo = med - np.asarray(q25, float)
    hi = np.asarray(q75, float) - med
    ax2.errorbar(xx, med, yerr=np.vstack([lo, hi]), fmt="ko", ms=3.5, lw=0.9, capsize=2)
    ax2.axhline(0.0, color="0.25", lw=0.7, ls=":")
    ax2.set_ylabel("upward step relative EEG onset, median [IQR] (s)")
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    fig.suptitle(
        f"{ds_sid} peri-onset multi-band energy-timing pilot — {len(eligible)} eligible seizures",
        y=0.99,
        fontsize=14,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_readme(fig_dir: Path, ds_sid: str, n_eligible: int) -> None:
    (fig_dir / "README.md").write_text(
        "# Peri-onset multi-band energy timing pilot\n\n"
        f"### {ds_sid}_energy_timing_subject_summary.png\n\n"
        f"E1146 pilot 汇总 {n_eligible} 次 T0-eligible seizures。Panel a 按原 seizure index（不按结果排序）显示多频带共识能量；"
        "Panel b 对比数据驱动的最大向上阶跃时间与 EEG/clinical onset；Panel c 展示各频带相对 EEG onset 的阶跃 latency；"
        "Panel d 同时给出检出比例和 subject 内 timing IQR。所有能量先减去每次 seizure 的远期 `[-120,-90] s` baseline。\n\n"
        "**关注点**：这里是 E1146 内部 exploratory timing pilot，不是 cohort statistic；灰格表示主规则下没有持续增强，不能当作零 latency。\n\n"
        "### per_seizure/*.png\n\n"
        "每张图对应一次 seizure。最上排使用原始波形脚本的连续 stacked-trace 逻辑，展示与 timing 完全相同的固定 15 个 lagPat-valid contacts；"
        "第二排是同一 contact 集上的七频带空间 Q75 和 multiband consensus 热图，"
        "下两排逐频带显示远期 baseline 阈值、首次持续增强和最大向上阶跃。紫色点线是 EEG onset，黑色虚线是 clinical onset。\n\n"
        "**关注点**：绿色点要求超过远期 baseline 99% 阈值并持续至少 2 s；紫红菱形是在宽搜索窗内不参考 onset 标签选出的最大前后 2 s 阶跃。\n",
        encoding="utf-8",
    )


def run(ds_sid: str, out_base: Path) -> Path:
    specs = _primary_band_specs()
    band_order = [name for name, _, _ in specs]
    eligible = _eligible_indices(ds_sid)
    if not eligible:
        raise RuntimeError(f"no T0-eligible seizures for {ds_sid}")
    dataset, sid = ds_sid.split("_", 1)
    inv_rows, _ = _inventory_rows(dataset, sid)

    cache_npz_path = CACHE_ROOT / f"{ds_sid}.npz"
    cache_json_path = CACHE_ROOT / f"{ds_sid}.json"
    if not (cache_npz_path.exists() and cache_json_path.exists()):
        raise FileNotFoundError(f"multi-band cache missing for {ds_sid}")
    cache_npz = np.load(cache_npz_path, allow_pickle=True)
    cache_meta = json.loads(cache_json_path.read_text(encoding="utf-8"))
    cache_channels = [str(x) for x in cache_npz["channels"]]
    cached_idx = {int(x) for x in cache_meta.get("seizure_idxs", [])}

    lagpat_channels, lagpat_source = _load_lagpat_channels(ds_sid)
    timing_channels = [x for x in lagpat_channels if x in set(cache_channels)]
    if len(timing_channels) < 6:
        raise RuntimeError(f"only {len(timing_channels)} lagPat-valid contacts match the band cache")
    cache_index = {name: i for i, name in enumerate(cache_channels)}
    timing_idx = np.asarray([cache_index[name] for name in timing_channels], dtype=int)

    out_root = out_base / f"pilot_{ds_sid}"
    missing = sorted(set(eligible) - cached_idx)
    fallback_arrays, fallback_meta = _compute_missing_bands(ds_sid, missing, inv_rows, specs, out_root)
    fallback_channels = [str(x) for x in fallback_arrays.get("channels", [])]
    if fallback_channels and fallback_channels != cache_channels:
        raise RuntimeError("fallback and committed cache channel order differ")

    rows: list[dict] = []
    trace_arrays: dict[str, np.ndarray] = {"channels": np.asarray(timing_channels)}
    seizure_traces: dict[int, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    fig_dir = out_root / "figures"
    per_seizure_dir = fig_dir / "per_seizure"
    computed = []
    drops = []

    for idx in eligible:
        row = inv_rows[idx]
        eeg_rel = _eeg_rel(row)
        seizure_id = _seizure_id(row, idx)
        traces: dict[str, np.ndarray] = {}
        timing_by_band = {}
        transition_by_band = {}
        rel_ref = None
        source = "committed_v2_band_cache" if idx in cached_idx else "short_window_raw_fallback"
        source_obj = cache_npz if idx in cached_idx else fallback_arrays
        try:
            for band in band_order:
                zkey = f"{band}__zt__{idx}"
                tkey = f"{band}__relt__{idx}"
                if zkey not in source_obj or tkey not in source_obj:
                    raise KeyError(f"missing {zkey}/{tkey}")
                z = np.asarray(source_obj[zkey], float)[timing_idx]
                rel_t = np.asarray(source_obj[tkey], float)
                if rel_ref is None:
                    rel_ref = rel_t
                elif rel_t.shape != rel_ref.shape or not np.allclose(rel_t, rel_ref, atol=1e-6):
                    raise ValueError(f"{band}: time grid differs within seizure {idx}")
                trace, timing = band_energy_timing(
                    z,
                    rel_t,
                    baseline=BASELINE,
                    search=SEARCH,
                    spatial_q=SPATIAL_Q,
                    smooth_sec=SMOOTH_SEC,
                    baseline_quantile=BASELINE_QUANTILE,
                    sustain_sec=SUSTAIN_SEC,
                )
                sensitivity = detect_sustained_enhancement(
                    trace,
                    rel_t,
                    baseline=BASELINE,
                    search=SEARCH,
                    baseline_quantile=SENSITIVITY_QUANTILE,
                    sustain_sec=SUSTAIN_SEC,
                )
                _, transition = max_upward_transition(
                    trace,
                    rel_t,
                    baseline=BASELINE,
                    search=SEARCH,
                    flank_sec=TRANSITION_FLANK_SEC,
                    baseline_quantile=BASELINE_QUANTILE,
                )
                traces[band] = trace
                timing_by_band[band] = timing
                transition_by_band[band] = transition
                rows.append(
                    _timing_row(
                        ds_sid,
                        idx,
                        seizure_id,
                        band,
                        source,
                        len(timing_channels),
                        eeg_rel,
                        timing,
                        sensitivity,
                        transition,
                    )
                )
            assert rel_ref is not None
            consensus = np.nanmedian(np.vstack([traces[b] for b in band_order]), axis=0)
            timing = detect_sustained_enhancement(
                consensus,
                rel_ref,
                baseline=BASELINE,
                search=SEARCH,
                baseline_quantile=BASELINE_QUANTILE,
                sustain_sec=SUSTAIN_SEC,
            )
            sensitivity = detect_sustained_enhancement(
                consensus,
                rel_ref,
                baseline=BASELINE,
                search=SEARCH,
                baseline_quantile=SENSITIVITY_QUANTILE,
                sustain_sec=SUSTAIN_SEC,
            )
            _, transition = max_upward_transition(
                consensus,
                rel_ref,
                baseline=BASELINE,
                search=SEARCH,
                flank_sec=TRANSITION_FLANK_SEC,
                baseline_quantile=BASELINE_QUANTILE,
            )
            traces["multiband_consensus"] = consensus
            timing_by_band["multiband_consensus"] = timing
            transition_by_band["multiband_consensus"] = transition
            rows.append(
                _timing_row(
                    ds_sid,
                    idx,
                    seizure_id,
                    "multiband_consensus",
                    source,
                    len(timing_channels),
                    eeg_rel,
                    timing,
                    sensitivity,
                    transition,
                )
            )
            seizure_traces[idx] = (rel_ref, traces)
            trace_arrays[f"relt__{idx}"] = rel_ref.astype(np.float32)
            for band, trace in traces.items():
                trace_arrays[f"{band}__q75_delta_z__{idx}"] = trace.astype(np.float32)
            raw_sw = extract_seizure_window(
                f"{dataset}/{sid}",
                idx,
                pre_sec=PRE_SEC,
                post_sec=POST_SEC,
                results_root=ROOT / "results",
                reference=ICTAL_REFERENCE[dataset],
            )
            raw_lookup = _alias_index(raw_sw.ch_names)
            raw_missing = [name for name in timing_channels if name not in raw_lookup]
            if raw_missing:
                raise ValueError(f"raw timing contacts missing for seizure {idx}: {raw_missing}")
            raw_channel_idx = [int(raw_lookup[name]) for name in timing_channels]
            _plot_seizure(
                ds_sid,
                idx,
                seizure_id,
                eeg_rel,
                band_order,
                rel_ref,
                traces,
                timing_by_band,
                transition_by_band,
                raw_sw,
                raw_channel_idx,
                per_seizure_dir / f"{ds_sid}_seizure_{idx:02d}_multiband_energy_timing.png",
            )
            del raw_sw
            computed.append(idx)
        except Exception as exc:  # noqa: BLE001 - fail-closed subject inventory
            drops.append({"seizure_idx": idx, "reason": f"{type(exc).__name__}:{exc}"})

    if sorted(computed) != sorted(eligible):
        raise RuntimeError(f"pilot failed to compute all eligible seizures; drops={drops}")

    band_with_consensus = band_order + ["multiband_consensus"]
    summary_rows = _subject_band_summary(rows, band_with_consensus, len(eligible))
    _write_csv(out_root / "per_seizure_timing.csv", rows)
    _write_csv(out_root / "subject_band_summary.csv", summary_rows)
    trace_path = out_root / "per_subject" / f"{ds_sid}_energy_timing_traces.npz"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trace_path, **trace_arrays)
    _plot_subject_summary(
        ds_sid,
        eligible,
        band_order,
        rows,
        seizure_traces,
        fig_dir / f"{ds_sid}_energy_timing_subject_summary.png",
    )

    consensus_summary = next(r for r in summary_rows if r["band"] == "multiband_consensus")
    consensus_transition_rows = [
        r for r in rows
        if r["band"] == "multiband_consensus" and r["transition_detected_q99"]
    ]
    consensus_eeg_delta = np.asarray(
        [r["transition_rel_eeg_sec"] for r in consensus_transition_rows], dtype=float
    )
    consensus_clin_delta = np.asarray(
        [r["transition_rel_clinical_sec"] for r in consensus_transition_rows], dtype=float
    )
    consensus_alignment = {
        "n_transition_confirmed": len(consensus_transition_rows),
        "n_within_1s_of_eeg": int(np.sum(np.abs(consensus_eeg_delta) <= 1.0)),
        "n_within_2s_of_eeg": int(np.sum(np.abs(consensus_eeg_delta) <= 2.0)),
        "n_within_5s_of_eeg": int(np.sum(np.abs(consensus_eeg_delta) <= 5.0)),
        "median_abs_distance_to_eeg_sec": (
            float(np.median(np.abs(consensus_eeg_delta))) if consensus_eeg_delta.size else float("nan")
        ),
        "median_abs_distance_to_clinical_sec": (
            float(np.median(np.abs(consensus_clin_delta))) if consensus_clin_delta.size else float("nan")
        ),
        "outliers_abs_gt_5s_from_eeg": [
            {
                "seizure_idx": int(r["seizure_idx"]),
                "transition_rel_eeg_sec": float(r["transition_rel_eeg_sec"]),
            }
            for r in consensus_transition_rows
            if abs(float(r["transition_rel_eeg_sec"])) > 5.0
        ],
    }
    summary = {
        "subject": ds_sid,
        "tier": "exploratory within-subject timing pilot; not a cohort statistic",
        "n_t0_eligible": len(eligible),
        "eligible_seizure_idxs": eligible,
        "n_computed": len(computed),
        "computed_seizure_idxs": computed,
        "long_cache_missing_idxs_recomputed": missing,
        "fallback_drops": fallback_meta.get("drops", []),
        "timing_contacts": timing_channels,
        "n_timing_contacts": len(timing_channels),
        "timing_contact_source": lagpat_source,
        "contract": {
            "distal_baseline_sec": list(BASELINE),
            "search_sec": list(SEARCH),
            "spatial_summary": f"Q{int(SPATIAL_Q * 100)} across fixed lagPat-valid contacts",
            "smooth_sec": SMOOTH_SEC,
            "primary_threshold": f"distal baseline Q{int(BASELINE_QUANTILE * 100)}",
            "sustain_sec": SUSTAIN_SEC,
            "sensitivity_threshold": f"distal baseline Q{int(SENSITIVITY_QUANTILE * 100)}",
            "transition": (
                f"largest mean([t,t+{TRANSITION_FLANK_SEC:g})) - "
                f"mean([t-{TRANSITION_FLANK_SEC:g},t)) step; distal-baseline Q99 confirmation"
            ),
            "bands": [{"name": n, "lo": lo, "hi": hi} for n, lo, hi in specs],
            "consensus": "pointwise median of seven smoothed primary-band Q75 traces",
            "onset_markers": "clinical onset is time zero; EEG onset retained as noisy reference",
            "raw_waveform": (
                "same fixed lagPat-valid contacts as timing; CAR reference; original continuous stacked-trace helper"
            ),
        },
        "subject_band_summary": summary_rows,
        "consensus_summary": consensus_summary,
        "consensus_transition_alignment": consensus_alignment,
        "outputs": {
            "per_seizure_timing_csv": str((out_root / "per_seizure_timing.csv").relative_to(ROOT)),
            "subject_band_summary_csv": str((out_root / "subject_band_summary.csv").relative_to(ROOT)),
            "trace_npz": str(trace_path.relative_to(ROOT)),
            "subject_figure": str((fig_dir / f"{ds_sid}_energy_timing_subject_summary.png").relative_to(ROOT)),
            "per_seizure_figure_dir": str(per_seizure_dir.relative_to(ROOT)),
        },
    }
    summary_path = out_root / "pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(fig_dir, ds_sid, len(eligible))
    return summary_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="epilepsiae_1146")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = ap.parse_args()
    out = run(args.subject, args.out_root)
    print(out)


if __name__ == "__main__":
    main()
