#!/usr/bin/env python3
"""Render all-seizure raw QC figures for low EEG-onset-alignment subjects."""
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

from scripts.paper_figures.plot_fig3_raw_spectral_context import (  # noqa: E402
    _alias_index,
    _label_shaded_windows,
    _load_lagpat_channels,
    _mark_onsets,
    _plot_continuous_stacked,
    _shade_windows,
)
from scripts.run_topic5_energy_timing_pilot import BAND_COLORS, BAND_LABELS, BAND_SHORT  # noqa: E402
from scripts.run_topic5_onset_energy_cohort import (  # noqa: E402
    BASELINE_QUANTILE,
    BROAD_CONTEXT,
    CACHE_ROOT,
    DISTAL_BASELINE,
    EEG_ONSET_WINDOW,
    HALF_WIDTH_SEC,
    SMOOTH_SEC,
    SUSTAIN_SEC,
    _compute_missing_bands,
    _eligible_map,
    _tier_contract,
)
from scripts.run_topic5_t0_eligibility import ICTAL_REFERENCE, _inventory_rows  # noqa: E402
from src.ictal_onset_extraction import extract_seizure_window  # noqa: E402
from src.topic5_energy_timing import (  # noqa: E402
    band_energy_timing,
    detect_centered_window_enhancement,
    detect_multiband_recruitment_onset,
)


COHORT_ROOT = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/epilepsiae/source_cache"
)
PRIMARY_ROOT = COHORT_ROOT / "primary_common_1_80hz"
DEFAULT_OUT = COHORT_ROOT / "raw_qc_low_eeg_alignment"
DEFAULT_SUBJECTS = ("epilepsiae_583", "epilepsiae_916", "epilepsiae_442")
RAW_POST_SEC = 15.0


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _reference_lines(
    ax,
    eeg_rel: float,
    energy_rel_clin: float,
    energy_detected: bool,
    *,
    marker_q05_rel_clin: float | None = None,
    marker_q95_rel_clin: float | None = None,
    marker_color: str | None = None,
    clinical_onset_available: bool = True,
) -> None:
    if clinical_onset_available:
        ax.axvline(0.0, color="0.15", lw=0.9, ls="--")
    ax.axvline(eeg_rel, color="#7A4F9A", lw=1.0, ls=":")
    if (
        marker_q05_rel_clin is not None
        and marker_q95_rel_clin is not None
        and np.isfinite(marker_q05_rel_clin)
        and np.isfinite(marker_q95_rel_clin)
    ):
        ax.axvspan(
            marker_q05_rel_clin,
            marker_q95_rel_clin,
            color=marker_color or "#1F77B4",
            alpha=0.09,
            lw=0,
        )
    ax.axvline(
        energy_rel_clin,
        color=marker_color or ("#2CA02C" if energy_detected else "0.55"),
        lw=1.0,
        ls="-" if energy_detected else "--",
    )


def _plot_one(
    *,
    ds_sid: str,
    idx: int,
    seizure_id: str,
    eeg_rel: float,
    timing_channels: list[str],
    rel_eeg: np.ndarray,
    traces: dict[str, np.ndarray],
    recruitment,
    eeg_window_hit: bool,
    n_band_eeg_hits: int,
    raw_sw,
    raw_channel_idx: list[int],
    out_path: Path,
    marker_label: str = "T_energy",
    marker_status: str | None = None,
    marker_q05_rel_eeg: float | None = None,
    marker_q95_rel_eeg: float | None = None,
    marker_color: str | None = None,
    show_eeg_hit_context: bool = True,
    clinical_onset_available: bool = True,
) -> None:
    band_order = list(traces)
    consensus = np.nanmedian(np.vstack([traces[name] for name in band_order]), axis=0)
    plot_names = band_order + ["multiband_consensus"]
    plot_traces = {**traces, "multiband_consensus": consensus}
    t_clin = rel_eeg + eeg_rel
    baseline_clin = (eeg_rel + DISTAL_BASELINE[0], eeg_rel + DISTAL_BASELINE[1])
    energy_rel_clin = recruitment.onset_sec + eeg_rel
    marker_q05_rel_clin = (
        eeg_rel + marker_q05_rel_eeg if marker_q05_rel_eeg is not None else None
    )
    marker_q95_rel_clin = (
        eeg_rel + marker_q95_rel_eeg if marker_q95_rel_eeg is not None else None
    )
    desired_right = (
        max(RAW_POST_SEC, float(energy_rel_clin) + 8.0)
        if marker_color is not None
        else RAW_POST_SEC
    )
    x_window = (
        float(baseline_clin[0]),
        min(desired_right, float(raw_sw.t_axis[-1]), float(t_clin[-1])),
    )

    raw_mask = (raw_sw.t_axis >= x_window[0]) & (raw_sw.t_axis < x_window[1])
    raw_decim = max(1, int(round(float(raw_sw.fs) / 180.0)))
    raw_x = raw_sw.signal[np.asarray(raw_channel_idx), :][:, np.flatnonzero(raw_mask)[::raw_decim]]
    trace_scale = float(
        np.nanpercentile(
            np.abs(raw_x - np.nanmedian(raw_x, axis=1, keepdims=True)), 95
        )
        * 3.0
    )
    if not np.isfinite(trace_scale) or trace_scale <= 0.0:
        trace_scale = 1.0

    fig = plt.figure(figsize=(12.4, 12.8))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.9, 1.0, 1.0, 1.0], hspace=0.65, wspace=0.30)

    ax_raw = fig.add_subplot(gs[0, :])
    _shade_windows(ax_raw, baseline_clin, eeg_rel, (0.0, 10.0))
    _plot_continuous_stacked(ax_raw, raw_sw, raw_channel_idx, x_window, scale=trace_scale)
    if clinical_onset_available:
        _mark_onsets(ax_raw, raw_sw, label_lines=False)
    else:
        ax_raw.axvline(eeg_rel, color="#7A4F9A", lw=1.0, ls=":")
    ax_raw.axvline(
        energy_rel_clin,
        color=marker_color or ("#2CA02C" if recruitment.detected else "0.55"),
        lw=1.1,
        ls="-" if recruitment.detected else "--",
    )
    if (
        marker_q05_rel_clin is not None
        and marker_q95_rel_clin is not None
        and np.isfinite(marker_q05_rel_clin)
        and np.isfinite(marker_q95_rel_clin)
    ):
        ax_raw.axvspan(
            marker_q05_rel_clin,
            marker_q95_rel_clin,
            color=marker_color or "#1F77B4",
            alpha=0.09,
            lw=0,
        )
    _label_shaded_windows(
        ax_raw,
        baseline_clin,
        eeg_rel,
        (0.0, 10.0),
        y=0.97,
        post_label=("CLINICAL 0-10 s" if clinical_onset_available else "EEG 0-10 s"),
    )
    axis_reference = "clinical onset" if clinical_onset_available else "EEG onset"
    ax_raw.set_xlabel(f"time from {axis_reference} (s)")
    ax_raw.set_title(
        f"raw intracranial traces on the same {len(timing_channels)} timing contacts",
        fontsize=11,
        pad=6,
    )

    ax_hm = fig.add_subplot(gs[1, :])
    hm = np.vstack([plot_traces[name] for name in plot_names])
    use = (t_clin >= x_window[0]) & (t_clin <= x_window[1])
    vmax = max(2.0, min(float(np.nanpercentile(np.abs(hm[:, use]), 98)), 8.0))
    mesh = ax_hm.imshow(
        hm[:, use],
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=[float(t_clin[use][0]), float(t_clin[use][-1]), len(plot_names) - 0.5, -0.5],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax_hm.set_yticks(np.arange(len(plot_names)))
    ax_hm.set_yticklabels(
        [BAND_SHORT[name] for name in band_order] + ["ALL"], fontsize=8
    )
    ax_hm.axvspan(*baseline_clin, color="#4C78A8", alpha=0.10, lw=0)
    _reference_lines(
        ax_hm,
        eeg_rel,
        energy_rel_clin,
        recruitment.detected,
        marker_q05_rel_clin=marker_q05_rel_clin,
        marker_q95_rel_clin=marker_q95_rel_clin,
        marker_color=marker_color,
        clinical_onset_available=clinical_onset_available,
    )
    ax_hm.set_xlim(x_window)
    ax_hm.set_xlabel(f"time from {axis_reference} (s)")
    ax_hm.set_title("EEG-relative distal-baseline-referenced spatial Q75 energy")
    fig.colorbar(mesh, ax=ax_hm, pad=0.01, fraction=0.02, label="delta robust-z")

    for pos, name in enumerate(plot_names):
        ax = fig.add_subplot(gs[2 + pos // 3, pos % 3])
        y = plot_traces[name]
        if name == "multiband_consensus":
            color = "#3A3A3A"
            label = "multiband consensus"
        else:
            color = BAND_COLORS[name]
            label = BAND_LABELS[name]
        timing = detect_centered_window_enhancement(
            y,
            rel_eeg,
            center_sec=0.0,
            half_width_sec=HALF_WIDTH_SEC,
            baseline=DISTAL_BASELINE,
            baseline_quantile=BASELINE_QUANTILE,
            sustain_sec=SUSTAIN_SEC,
        )
        ax.plot(t_clin, y, color=color, lw=1.1)
        ax.axhline(timing.threshold, color="0.40", lw=0.8, ls="--")
        ax.axvspan(*baseline_clin, color="#4C78A8", alpha=0.08, lw=0)
        ax.axvspan(eeg_rel + EEG_ONSET_WINDOW[0], eeg_rel + EEG_ONSET_WINDOW[1], color="#7A4F9A", alpha=0.06, lw=0)
        _reference_lines(
            ax,
            eeg_rel,
            energy_rel_clin,
            recruitment.detected,
            marker_q05_rel_clin=marker_q05_rel_clin,
            marker_q95_rel_clin=marker_q95_rel_clin,
            marker_color=marker_color,
            clinical_onset_available=clinical_onset_available,
        )
        ax.set_xlim(x_window)
        finite = y[use & np.isfinite(y)]
        if finite.size:
            lo, hi = np.percentile(finite, [1.0, 99.5])
            pad = 0.12 * max(1.0, hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
        context = f"\nEEG-window Q99 hit={timing.detected}" if show_eeg_hit_context else ""
        ax.set_title(f"{label}{context}", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        if pos // 3 == 1:
            ax.set_xlabel(f"s from {axis_reference}")
        if pos % 3 == 0:
            ax.set_ylabel("delta z")

    status = marker_status or (
        "confirmed" if recruitment.detected else "unconfirmed candidate"
    )
    hit_context = (
        f"EEG-window hit={eeg_window_hit}, bands={n_band_eeg_hits}/5; "
        if show_eeg_hit_context
        else ""
    )
    fig.suptitle(
        f"{ds_sid} seizure {idx:02d} ({seizure_id}) — {hit_context}"
        f"{marker_label} {status} at {recruitment.onset_sec:+.1f} s from EEG",
        fontsize=13,
        y=0.993,
    )
    if marker_color is not None:
        footer = "purple dotted = EEG onset; "
        footer += (
            "black dashed = clinical onset; "
            if clinical_onset_available
            else "Yuquan EEG-only annotation; "
        )
        footer += f"blue = {marker_label} and 90% resampling interval"
    else:
        footer = (
            "purple dotted = EEG onset; black dashed = clinical onset; "
            "green = confirmed T_energy; gray dashed = unconfirmed candidate"
        )
    fig.text(
        0.5,
        0.012,
        footer,
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.96, top=0.965, bottom=0.055)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_subject(ds_sid: str, out_root: Path) -> list[dict]:
    specs, majority_required, _, tier = _tier_contract("common_1_80hz")
    band_order = [name for name, _, _ in specs]
    all_subjects = sorted(path.stem for path in CACHE_ROOT.glob("epilepsiae_*.json"))
    eligible_map, _ = _eligible_map(all_subjects, max_band_hz=80.0)
    eligible = eligible_map[ds_sid]
    dataset, sid = ds_sid.split("_", 1)
    inv_rows, _ = _inventory_rows(dataset, sid)

    cache_obj = np.load(CACHE_ROOT / f"{ds_sid}.npz", allow_pickle=True)
    cache_channels = [str(x) for x in cache_obj["channels"]]
    cache_lookup = {name: i for i, name in enumerate(cache_channels)}
    cache_meta = json.loads((CACHE_ROOT / f"{ds_sid}.json").read_text())
    cached_idx = {int(x) for x in cache_meta["seizure_idxs"]}
    lagpat_channels, _ = _load_lagpat_channels(ds_sid)
    timing_channels = [name for name in lagpat_channels if name in cache_lookup]
    timing_idx = np.asarray([cache_lookup[name] for name in timing_channels], dtype=int)
    missing = sorted(set(eligible) - cached_idx)
    fallback, fallback_meta = _compute_missing_bands(
        ds_sid,
        missing,
        inv_rows,
        specs,
        timing_channels,
        PRIMARY_ROOT,
        "common_1_80hz",
    )
    if fallback_meta.get("drops"):
        raise RuntimeError(f"{ds_sid}: fallback drops={fallback_meta['drops']}")

    cohort_rows = {
        (row["subject"], int(row["seizure_idx"])): row
        for row in csv.DictReader((PRIMARY_ROOT / "seizure_level_onset_energy.csv").open())
    }
    rows = []
    subject_fig_dir = out_root / "figures" / ds_sid
    for idx in eligible:
        inv = inv_rows[idx]
        eeg_rel = float(inv["eeg_onset_epoch"]) - float(inv["clin_onset_epoch"])
        source = cache_obj if idx in cached_idx else fallback
        traces = {}
        rel_ref = None
        for band in band_order:
            z = np.asarray(source[f"{band}__zt__{idx}"], dtype=float)
            if idx in cached_idx:
                z = z[timing_idx]
            rel_clin = np.asarray(source[f"{band}__relt__{idx}"], dtype=float)
            rel_eeg = rel_clin - eeg_rel
            trace, _ = band_energy_timing(
                z,
                rel_eeg,
                baseline=DISTAL_BASELINE,
                search=BROAD_CONTEXT,
                spatial_q=0.75,
                smooth_sec=SMOOTH_SEC,
                baseline_quantile=BASELINE_QUANTILE,
                sustain_sec=SUSTAIN_SEC,
            )
            traces[band] = trace
            rel_ref = rel_eeg if rel_ref is None else rel_ref
        assert rel_ref is not None
        recruitment = detect_multiband_recruitment_onset(
            np.vstack([traces[name] for name in band_order]),
            rel_ref,
            baseline=DISTAL_BASELINE,
            search=BROAD_CONTEXT,
            majority_required=majority_required,
            baseline_quantile=BASELINE_QUANTILE,
            sustain_sec=SUSTAIN_SEC,
        )

        pre_sec = max(120.0, 120.0 - eeg_rel)
        raw_sw = extract_seizure_window(
            f"{dataset}/{sid}",
            idx,
            pre_sec=pre_sec,
            post_sec=RAW_POST_SEC,
            results_root=ROOT / "results",
            reference=ICTAL_REFERENCE[dataset],
        )
        raw_lookup = _alias_index(raw_sw.ch_names)
        absent = [name for name in timing_channels if name not in raw_lookup]
        if absent:
            raise ValueError(f"{ds_sid} seizure {idx}: raw contacts missing: {absent}")
        raw_idx = [int(raw_lookup[name]) for name in timing_channels]
        metrics = cohort_rows[(ds_sid, idx)]
        out_path = subject_fig_dir / f"{ds_sid}_seizure_{idx:02d}_raw_energy_qc.png"
        _plot_one(
            ds_sid=ds_sid,
            idx=idx,
            seizure_id=inv["seizure_id"],
            eeg_rel=eeg_rel,
            timing_channels=timing_channels,
            rel_eeg=rel_ref,
            traces=traces,
            recruitment=recruitment,
            eeg_window_hit=metrics["consensus_eeg_window_extreme_hit"] == "True",
            n_band_eeg_hits=int(metrics["n_band_eeg_window_extreme_hits"]),
            raw_sw=raw_sw,
            raw_channel_idx=raw_idx,
            out_path=out_path,
        )
        rows.append(
            {
                "subject": ds_sid,
                "seizure_idx": idx,
                "seizure_id": inv["seizure_id"],
                "eeg_window_hit": metrics["consensus_eeg_window_extreme_hit"],
                "n_band_eeg_hits": metrics["n_band_eeg_window_extreme_hits"],
                "energy_recruitment_detected": recruitment.detected,
                "energy_recruitment_rel_eeg_sec": recruitment.onset_sec,
                "figure": str(out_path.relative_to(ROOT)),
            }
        )
        del raw_sw
        print(f"[raw-qc] {ds_sid} seizure {idx}", flush=True)
    cache_obj.close()
    (subject_fig_dir / "README.md").write_text(
        f"# {ds_sid} raw energy QC\n\n"
        f"本目录包含该患者全部 {len(eligible)} 次 common 1–80 Hz eligible seizures。每张图最上方为固定 lagPat-valid contacts 的原始波形，下面为同一 contacts 的五频带与 consensus 能量。\n\n"
        "**关注点**：目视区分 EEG onset 附近真实能量跃迁、提前的短暂事件、持续高基线和原始信号 artifact。\n",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    all_rows = []
    for subject in args.subjects:
        all_rows.extend(render_subject(subject, args.out_root))
    _write_csv(args.out_root / "raw_qc_index.csv", all_rows)
    fig_dir = args.out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    (fig_dir / "README.md").write_text(
        "# Low EEG-onset-alignment raw QC\n\n"
        "### epilepsiae_583/\n\nE583 在 EEG onset ±5 s 的 consensus Q99 命中为 0/22。\n\n**关注点**：检查低采样率段、持续高基线或晚期 recruitment。\n\n"
        "### epilepsiae_916/\n\nE916 的命中为 0/48。\n\n**关注点**：检查固定 6 contacts 是否漏掉临床可见的发作能量变化。\n\n"
        "### epilepsiae_442/\n\nE442 为最低非零对照，命中 4/22。\n\n**关注点**：比较少数 near-EEG positives 与其余 seizures 的原始形态。\n",
        encoding="utf-8",
    )
    print(args.out_root / "raw_qc_index.csv")


if __name__ == "__main__":
    main()
