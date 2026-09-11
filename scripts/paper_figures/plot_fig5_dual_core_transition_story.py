#!/usr/bin/env python3
"""Build the six-panel dual-core Figure 5 transition narrative.

The figure deliberately separates three evidence levels:

* A--C: one exact OU-driven 40,000-neuron dual-core SNN trajectory;
* D: the already frozen clinical cohort spatial-concordance statistic;
* E--F: deterministic spatial-Z continuation and the existing SNN timescale
  library on the same ``dualcore_s39 + Joint=1.25`` model substrate.

No simulation is run here.  The producer validates and renders immutable
artifacts, and records every cross-panel semantic in JSON metadata.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig5_dual_core_spatial_z_panels import (  # noqa: E402
    _plot_panel_c as _plot_spatial_z_branch_atlas,
)


ICL = "#F1783A"
SCL = "#29A6B5"
EVENT = "#E6A15A"
PRE = "#8D6AAE"
ONSET = "#C7254E"
ENERGY = "#6D3A9C"
Z_CORE = "#245B9B"
Z_SURROUND = "#87A7C7"
ADAPT = "#E36A2E"
COHORT = "#9C3C87"
NULL = "#A8A8A8"
CORE_A = "#F28E2B"
CORE_B = "#2FA7B8"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.stem + ".", suffix=".json")
    os.close(descriptor)
    try:
        Path(temporary).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _source_checkout() -> Path:
    """Return the checkout that owns the ignored, immutable source results."""
    worker = (
        ROOT / "results/topic4_sef_hfo/data_driven_dual_core_zm_transition/"
        "timescale/workers/rev21_ts_tz3000_ta500_topology_2542_dynamics_2642.npz"
    )
    if worker.exists():
        return ROOT
    if ROOT.parent.name == ".worktrees":
        candidate = ROOT.parent.parent
        if (candidate / worker.relative_to(ROOT)).exists():
            return candidate
    raise FileNotFoundError(
        "cannot locate the frozen rev21 source results; pass --source-repo")


def _panel_label(axis: plt.Axes, label: str, *, x=-0.13, y=1.12) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        clip_on=False,
    )


def _style_axis(axis: plt.Axes, *, top=False, right=False) -> None:
    axis.spines["top"].set_visible(top)
    axis.spines["right"].set_visible(right)
    axis.tick_params(labelsize=7.2, width=0.75, length=2.8)


def _contact_order(names: np.ndarray) -> np.ndarray:
    def number(name: str) -> int:
        match = re.search(r"(\d+)$", str(name))
        return int(match.group(1)) if match else -1

    return np.asarray(
        sorted(
            range(len(names)),
            key=lambda index: (
                0 if str(names[index]).startswith("SCL") else 1,
                number(str(names[index])),
            ),
        ),
        dtype=int,
    )


def _bandpass(raw: np.ndarray, dt_ms: float, band_hz: tuple[float, float]) -> np.ndarray:
    raw = np.asarray(raw, float)
    fs_hz = 1000.0 / float(dt_ms)
    if fs_hz <= 2.0 * float(band_hz[1]):
        raise ValueError("virtual-contact sampling rate violates the Nyquist gate")
    sos = butter(4, band_hz, btype="bandpass", fs=fs_hz, output="sos")
    return sosfiltfilt(sos, raw, axis=0)


def _select_interictal_event(meta: dict, arrays: dict[str, np.ndarray]) -> dict:
    """Latest returned event with complete contact onset readout."""
    events = list(meta["events"])
    onsets = np.asarray(arrays["onsets"], float)
    eligible = []
    for event in events:
        index = int(event["event_index"])
        if bool(event["returned"]) and np.all(np.isfinite(onsets[index])):
            eligible.append(event)
    if not eligible:
        raise RuntimeError("no returned interictal event has complete contact readout")
    return max(eligible, key=lambda row: float(row["t_on_ms"]))


def _candidate_eta_m(meta: dict, manifest: dict) -> float:
    matches = [row for row in manifest['candidates']
               if row['candidate_id'] == meta['candidate_id']]
    if len(matches) != 1:
        raise RuntimeError('exactly one matching candidate is required for M gain')
    gain = float(matches[0]['slow_variables']['eta_m'])
    if not np.isfinite(gain) or gain < 0:
        raise RuntimeError('candidate M gain must be finite and nonnegative')
    return gain


def _stage_contract(meta: dict, selected_event: dict) -> dict:
    landmarks = meta["model_ictal_rev21"]["landmarks"]
    onset_ms = float(meta["model_ictal_rev21"]["scientific_onset_ms"])
    return {
        "display_ms": [0.0, float(meta["simulation"]["recorded_duration_ms"])],
        "interictal_ms": [
            float(selected_event["t_on_ms"]),
            float(selected_event["t_off_ms"]),
        ],
        "returned_interictal_ms": [
            [float(event["t_on_ms"]), float(event["t_off_ms"])]
            for event in meta["events"] if bool(event["returned"])
        ],
        "pre_onset_ms": [float(value) for value in landmarks["w_pre_ms"]],
        "early_ictal_ms": [
            float(landmarks["w_early_ms"][0]),
            min(float(landmarks["w_early_ms"][0]) + 400.0,
                float(landmarks["w_early_ms"][1])),
        ],
        "onset_ms": onset_ms,
    }


def _shade_stages(axis: plt.Axes, stages: dict, *, label=False) -> None:
    event_lo, event_hi = np.asarray(stages["interictal_ms"]) / 1000.0
    pre_lo, pre_hi = np.asarray(stages["pre_onset_ms"]) / 1000.0
    early_lo, early_hi = np.asarray(stages["early_ictal_ms"]) / 1000.0
    onset_s = float(stages["onset_ms"]) / 1000.0
    stop_s = float(stages["display_ms"][1]) / 1000.0
    for lo_ms, hi_ms in stages["returned_interictal_ms"]:
        axis.axvspan(
            lo_ms / 1000.0, hi_ms / 1000.0,
            color=EVENT, alpha=0.055, lw=0, zorder=0)
    axis.axvspan(event_lo, event_hi, color=EVENT, alpha=0.12, lw=0, zorder=0)
    axis.axvspan(pre_lo, pre_hi, color=PRE, alpha=0.10, lw=0, zorder=0)
    axis.axvspan(onset_s, stop_s, color=ONSET, alpha=0.055, lw=0, zorder=0)
    axis.axvspan(early_lo, early_hi, color=ONSET, alpha=0.075, lw=0, zorder=0)
    axis.axvline(onset_s, color=ONSET, lw=0.95, ls="--", zorder=5)
    if label:
        y = 1.02
        transform = axis.get_xaxis_transform()
        axis.text(
            0.5 * (event_lo + event_hi), y, "selected interictal event",
            transform=transform, color="#B86A25", fontsize=6.5,
            ha="center", va="bottom", clip_on=False,
        )
        axis.text(
            0.5 * (pre_lo + pre_hi), y, "pre-onset",
            transform=transform, color=PRE, fontsize=6.5,
            ha="center", va="bottom", clip_on=False,
        )
        axis.text(
            onset_s + 0.02, y, "sustained recruited state",
            transform=transform, color=ONSET, fontsize=6.5,
            ha="left", va="bottom", clip_on=False,
        )


def _plot_panel_a(
    rate_axis: plt.Axes,
    trace_axis: plt.Axes,
    arrays: dict[str, np.ndarray],
    stages: dict,
) -> dict:
    names = np.asarray(arrays["contact_names"]).astype(str)
    shafts = np.asarray(arrays["shaft_ids"]).astype(str)
    order = _contact_order(names)
    dt_ms = float(arrays["transition_lfp_dt_ms"])
    signed = _bandpass(
        arrays["transition_lfp_trace"], dt_ms, (30.0, 80.0))
    time_s = np.arange(signed.shape[0], dtype=float) * dt_ms / 1000.0
    onset_s = float(stages["onset_ms"]) / 1000.0
    pre = time_s < onset_s
    scale = float(np.percentile(np.abs(signed[pre]), 99.0))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Panel A has no finite pre-onset virtual-SEEG scale")
    trace = 0.66 * signed[:, order] / scale
    offsets = np.arange(len(order), dtype=float) * 1.0
    _shade_stages(trace_axis, stages)
    for row, index in enumerate(order):
        color = ICL if shafts[index] == "ICL" else SCL
        trace_axis.plot(
            time_s, trace[:, row] + offsets[row], color=color,
            lw=0.62, alpha=0.96, rasterized=True,
        )
    stop_s = float(stages["display_ms"][1]) / 1000.0
    trace_axis.set_xlim(0.0, stop_s)
    trace_axis.set_ylim(-0.70, offsets[-1] + 0.75)
    trace_axis.set_yticks(offsets)
    trace_axis.set_yticklabels(names[order], fontsize=6.5)
    for tick, index in zip(trace_axis.get_yticklabels(), order):
        tick.set_color(ICL if shafts[index] == "ICL" else SCL)
    trace_axis.tick_params(axis="x", labelbottom=False, bottom=False)
    trace_axis.set_ylabel("Virtual-SEEG proxy\n(30–80 Hz)", fontsize=7.8)
    _style_axis(trace_axis)

    recruitment_time_s = (
        np.asarray(arrays["transition_recruitment_time_1mm_ms"], float) / 1000.0)
    active = 100.0 * np.asarray(
        arrays["transition_active_E_fraction_1mm"], float)
    sheet = 100.0 * np.asarray(
        arrays["transition_sheet_fraction_1mm"], float)
    _shade_stages(rate_axis, stages, label=True)
    rate_axis.plot(recruitment_time_s, active, color="0.18", lw=0.85,
                   label="active E neurons")
    rate_axis.plot(recruitment_time_s, sheet, color="#708090", lw=0.85,
                   label="recruited sheet")
    rate_axis.axhline(50.0, color=ONSET, lw=0.75, ls=":")
    rate_axis.set_xlim(0.0, stop_s)
    rate_axis.set_ylim(0.0, 102.0)
    rate_axis.set_yticks([0.0, 50.0, 100.0])
    rate_axis.set_ylabel("global\nrecruitment (%)", fontsize=6.8)
    rate_axis.tick_params(axis="x", labelbottom=False, bottom=False)
    rate_axis.legend(
        frameon=False, fontsize=6.4, ncol=2, loc="upper left",
        handlelength=1.6, columnspacing=0.9, borderaxespad=0.15,
    )
    rate_axis.set_title(
        "Spontaneous transition under continuous OU drive",
        fontsize=9.4, fontweight="bold", loc="left", pad=12,
    )
    _style_axis(rate_axis)
    return {
        "readout": "current-based virtual-contact proxy, signed 30–80 Hz",
        "scaling": "one p99 amplitude scale frozen on all pre-onset samples",
        "recruitment": (
            "active-E fraction and fraction of 1-mm bins with at least 50% "
            "local recruitment"
        ),
        "majority_threshold": 0.5,
    }


def _plot_panel_b(
    axes: tuple[plt.Axes, plt.Axes, plt.Axes],
    arrays: dict[str, np.ndarray],
    stages: dict,
    eta_m: float,
) -> dict:
    energy_axis, z_axis, m_axis = axes
    for axis in axes:
        _shade_stages(axis, stages)
    activity_time_s = (
        np.asarray(arrays["transition_spatial_frame_time_ms"], float) / 1000.0)
    population_rate = np.asarray(arrays["transition_rate_E_hz_20ms"], float)
    energy_axis.plot(activity_time_s, population_rate, color=ENERGY, lw=0.95)
    energy_axis.axhline(120.0, color=ONSET, lw=0.65, ls=":")
    energy_axis.set_ylabel("population E rate\n(20-ms bins, Hz)", fontsize=6.8,
                           color=ENERGY)
    energy_axis.tick_params(axis="y", colors=ENERGY)
    energy_axis.set_title(
        "Population activity and slow variables",
        fontsize=9.4, fontweight="bold", loc="left", pad=5,
    )

    slow_time_s = np.asarray(arrays["slow_time_ms"], float) / 1000.0
    z_core = np.asarray(arrays["slow_z_core_mean"], float)
    z_surround = np.asarray(arrays["slow_z_surround_mean"], float)
    z_axis.plot(slow_time_s, z_core, color=Z_CORE, lw=1.00, label="core")
    z_axis.plot(slow_time_s, z_surround, color=Z_SURROUND, lw=0.85,
                label="surround")
    z_axis.set_ylim(min(0.38, float(z_core.min()) - 0.03), 1.03)
    z_axis.set_ylabel("inhibitory efficacy\n$Z$", fontsize=6.8, color=Z_CORE)
    z_axis.tick_params(axis="y", colors=Z_CORE)
    z_axis.legend(frameon=False, fontsize=6.0, ncol=2, loc="lower left",
                  borderaxespad=0.15, handlelength=1.3, columnspacing=0.8)

    adaptation = eta_m * np.asarray(arrays["slow_m_core_mean"], float)
    m_axis.plot(slow_time_s, adaptation, color=ADAPT, lw=0.95)
    m_axis.set_ylim(0.0, max(0.1, float(np.percentile(adaptation, 99.5)) * 1.08))
    m_axis.set_ylabel("adaptation\n" + r"$A=\eta_m m$", fontsize=6.8, color=ADAPT)
    m_axis.tick_params(axis="y", colors=ADAPT)
    stop_s = float(stages["display_ms"][1]) / 1000.0
    for axis in axes:
        axis.set_xlim(0.0, stop_s)
        _style_axis(axis)
    energy_axis.tick_params(axis="x", labelbottom=False, bottom=False)
    z_axis.tick_params(axis="x", labelbottom=False, bottom=False)
    m_axis.set_xlabel("Time in the same SNN trajectory (s)", fontsize=7.8)
    return {
        "activity": "population E firing rate in the frozen 20-ms bins",
        "tonic_boundary": (
            "the accepted endpoint is a tonic recruited plateau; 10–250 Hz "
            "band-limited virtual-contact power decreases after onset and is "
            "therefore not used as a surrogate for runaway magnitude"
        ),
        "z": "core and surround mean inhibitory efficacy",
        "adaptation": "core mean A=eta_m*m",
        "eta_m": eta_m,
    }


def _fit_plane_gradient(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    finite = np.isfinite(values) & np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite)) < 3:
        raise ValueError("at least three finite spatial samples are required")
    design = np.column_stack((np.ones(int(np.sum(finite))), x[finite], y[finite]))
    coefficients = np.linalg.lstsq(design, values[finite], rcond=None)[0]
    gradient = np.asarray(coefficients[1:], float)
    norm = float(np.linalg.norm(gradient))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("spatial gradient is degenerate")
    return gradient / norm


def _spatial_rate_maps(
    arrays: dict[str, np.ndarray],
    stages: dict,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    positions = np.asarray(arrays["positions_E"], float)
    frame_time = np.asarray(arrays["transition_spatial_frame_time_ms"], float)
    # Producer stores (time, y, x); histogram2d below returns (x, y).
    # Convert once before occupancy normalization, then keep maps in (x, y).
    counts = np.asarray(
        arrays["transition_spatial_spike_count_20ms"], float).swapaxes(1, 2)
    bin_mm = float(arrays["transition_spatial_bin_mm"])
    sheet_mm = 20.0
    n_bins = int(round(sheet_mm / bin_mm))
    occupancy, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=n_bins,
        range=((0.0, sheet_mm), (0.0, sheet_mm)),
    )
    windows = [
        stages["interictal_ms"],
        stages["pre_onset_ms"],
        stages["early_ictal_ms"],
    ]
    maps = []
    for start, stop in windows:
        keep = (frame_time >= float(start)) & (frame_time < float(stop))
        if not np.any(keep):
            raise RuntimeError(f"no spatial frames in window {start}--{stop} ms")
        mean_counts = np.mean(counts[keep], axis=0)
        rate = np.divide(
            mean_counts,
            occupancy * 0.020,
            out=np.zeros_like(mean_counts),
            where=occupancy > 0,
        )
        maps.append(rate)
    centers = (np.arange(n_bins, dtype=float) + 0.5) * bin_mm
    return maps, centers, occupancy


def _plot_panel_c(
    axes: list[plt.Axes],
    arrays: dict[str, np.ndarray],
    stages: dict,
    selected_event: dict,
    core_centers: np.ndarray,
    colorbar_axis: plt.Axes,
) -> dict:
    maps, centers, occupancy = _spatial_rate_maps(arrays, stages)
    xx, yy = np.meshgrid(centers, centers, indexing="ij")
    event_index = int(selected_event["event_index"])
    # event_source_onset_maps also stores (y, x).
    onset_map = np.asarray(arrays["source_onset_maps_ms"][event_index], float).T
    onset_centers = np.arange(onset_map.shape[0], dtype=float) + 0.5
    ox, oy = np.meshgrid(onset_centers, onset_centers, indexing="ij")
    reference_axis = _fit_plane_gradient(onset_map, ox, oy)
    transformed = [np.log10(1.0 + rate) for rate in maps]
    vmax = float(np.percentile(np.concatenate([x.ravel() for x in transformed]), 99.5))
    vmax = max(vmax, 1e-6)
    positions = np.asarray(arrays["positions_E"], float)
    contacts = np.asarray(arrays["contact_xy_mm"], float)
    h = np.asarray(arrays["h"], float) > 0
    core_assignment = np.argmin(
        np.sum(np.square(positions[h, None, :] - core_centers[None, :, :]), axis=2),
        axis=1,
    )
    core_fraction = []
    for core_index in range(2):
        selected_positions = positions[h][core_assignment == core_index]
        core_count, _, _ = np.histogram2d(
            selected_positions[:, 0], selected_positions[:, 1],
            bins=len(centers), range=((0.0, 20.0), (0.0, 20.0)),
        )
        core_fraction.append(np.divide(
            core_count, occupancy, out=np.zeros_like(core_count), where=occupancy > 0))
    titles = ("Interictal event", "Pre-onset amplification", "Early recruited state")
    window_keys = ("interictal_ms", "pre_onset_ms", "early_ictal_ms")
    alignments = []
    image = None
    for index, (axis, rate, field, title, window_key) in enumerate(
            zip(axes, maps, transformed, titles, window_keys)):
        image = axis.imshow(
            field.T, origin="lower", extent=(0.0, 20.0, 0.0, 20.0),
            cmap="magma", norm=Normalize(0.0, vmax), interpolation="bilinear",
            aspect="equal",
        )
        for fraction, color in zip(core_fraction, (CORE_A, CORE_B)):
            axis.contour(
                centers, centers, fraction.T, levels=[0.10], colors=[color],
                linewidths=0.85, alpha=0.95,
            )
        axis.scatter(
            contacts[:, 0], contacts[:, 1], s=7.5, facecolor="white",
            edgecolor="0.35", lw=0.35, alpha=0.78, zorder=4,
        )
        for core_index, (center_xy, color) in enumerate(
                zip(core_centers, (CORE_A, CORE_B))):
            axis.scatter(
                center_xy[0], center_xy[1], s=20, facecolor="white",
                edgecolor=color, lw=0.8, zorder=5,
            )
            axis.text(
                center_xy[0], center_xy[1], "AB"[core_index],
                color=color, fontsize=5.2, fontweight="bold",
                ha="center", va="center", zorder=6,
            )
        half_length = 8.0
        center = np.asarray([10.0, 10.0])
        start = center - half_length * reference_axis
        stop = center + half_length * reference_axis
        axis.annotate(
            "", xy=stop, xytext=start,
            arrowprops={"arrowstyle": "-|>", "color": "white", "lw": 1.0,
                        "linestyle": "--", "mutation_scale": 8},
            zorder=6,
        )
        gradient = _fit_plane_gradient(np.log1p(rate), xx, yy)
        alignment = float(abs(np.dot(reference_axis, gradient)))
        alignments.append(alignment)
        lo, hi = np.asarray(stages[window_key], float) / 1000.0
        axis.set_title(
            f"{title}\n{lo:.2f}–{hi:.2f} s  |cos Δθ|={alignment:.2f}",
            fontsize=6.8, fontweight="bold", pad=3,
        )
        axis.set_xlim(0.0, 20.0)
        axis.set_ylim(0.0, 20.0)
        axis.set_xticks([0.0, 10.0, 20.0])
        axis.set_yticks([0.0, 10.0, 20.0])
        axis.set_xlabel("sheet x (mm)", fontsize=6.4)
        if index == 0:
            axis.set_ylabel("sheet y (mm)", fontsize=6.4)
        else:
            axis.set_yticklabels([])
        _style_axis(axis, top=True, right=True)
    axes[0].text(
        0.02, 0.98, "white arrow: interictal onset gradient",
        transform=axes[0].transAxes, ha="left", va="top", fontsize=5.4,
        color="white",
    )
    colorbar = axes[0].figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(r"log$_{10}$(1 + local E rate [Hz])", fontsize=6.0)
    colorbar.ax.tick_params(labelsize=5.8, width=0.65, length=2.2)
    return {
        "map_measure": "mean local E spike rate in 0.5-mm bins, log10(1+Hz)",
        "shared_normalization": [0.0, vmax],
        "reference_axis": (
            "least-squares spatial gradient of the selected returned "
            "interictal event onset map; positive direction is early-to-late"
        ),
        "reference_axis_unit_xy": reference_axis.tolist(),
        "rate_gradient": "least-squares gradient of log(1+local E rate)",
        "absolute_cosine_alignment": alignments,
        "windows_ms": [stages[key] for key in window_keys],
        "core_contours": (
            "actual union-core neurons split by nearest frozen core center; "
            "10% within-bin membership contour"
        ),
    }


def _load_clinical_cohort(subject_csv: Path, cohort_csv: Path) -> tuple[list[dict], dict]:
    with subject_csv.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["group_id"] == "strict_broadband"
        ]
    with cohort_csv.open(newline="", encoding="utf-8") as handle:
        cohort_rows = [
            row for row in csv.DictReader(handle)
            if row["group_id"] == "strict_broadband"
        ]
    if len(cohort_rows) != 1:
        raise RuntimeError("strict-broadband cohort summary is not unique")
    cohort = cohort_rows[0]
    if len(rows) != int(cohort["n_subjects"]):
        raise RuntimeError("clinical cohort subject and summary denominators differ")
    parsed = [{
        "subject": row["subject"],
        "data": float(row["data"]),
        "null": float(row["channel_null_median"]),
        "n_seizures": int(row["n_seizures"]),
        "field_plane": row["field_plane"],
    } for row in sorted(rows, key=lambda item: item["subject"])]
    return parsed, {
        "n_subjects": int(cohort["n_subjects"]),
        "n_seizures": int(cohort["n_seizures"]),
        "n_data_gt_null": int(cohort["n_data_gt_null"]),
        "wilcoxon_one_sided_data_gt_null_p": float(
            cohort["wilcoxon_one_sided_data_gt_null_p"]),
        "data_median": float(cohort["data_median"]),
        "null_median": float(cohort["null_median"]),
        "n_shared_subjects": int(cohort["n_shared_subjects"]),
        "n_own_fallback_subjects": int(cohort["n_own_fallback_subjects"]),
    }


def _plot_panel_d(axis: plt.Axes, rows: list[dict], summary: dict) -> dict:
    data = np.asarray([row["data"] for row in rows], float)
    null = np.asarray([row["null"] for row in rows], float)
    rng = np.random.default_rng(20260904)
    jitter = rng.normal(0.0, 0.035, size=len(rows))
    positions = [0.0, 1.0]
    for values, x, color, edge in (
        (data, positions[0], COHORT, "#713064"),
        (null, positions[1], "#D0D0D0", "#858585"),
    ):
        if len(values) >= 2 and not np.allclose(values, values[0]):
            violin = axis.violinplot(
                [values], positions=[x], widths=0.65,
                showmeans=False, showmedians=False, showextrema=False,
            )["bodies"][0]
            violin.set_facecolor(color)
            violin.set_edgecolor("none")
            violin.set_alpha(0.45 if x == 0 else 0.58)
        axis.boxplot(
            [values], positions=[x], widths=0.30, patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.25},
            boxprops={"facecolor": color, "edgecolor": edge,
                      "linewidth": 0.9, "alpha": 0.72},
            whiskerprops={"color": edge, "linewidth": 0.8},
            capprops={"color": edge, "linewidth": 0.8},
        )
    for dx, observed, shuffled in zip(jitter, data, null):
        axis.plot(
            [positions[0] + dx, positions[1] + dx], [observed, shuffled],
            color="0.45", lw=0.55, alpha=0.28, zorder=2,
        )
    axis.scatter(
        positions[0] + jitter, data, s=17, facecolor=COHORT,
        edgecolor="white", lw=0.55, zorder=4,
    )
    axis.scatter(
        positions[1] + jitter, null, s=17, facecolor="#8F8F8F",
        edgecolor="white", lw=0.55, zorder=4,
    )
    ymax = max(float(np.max(data)), float(np.max(null))) + 0.010
    bracket_y = min(1.015, ymax + 0.010)
    axis.plot([0.0, 0.0, 1.0, 1.0],
              [bracket_y, bracket_y + 0.018, bracket_y + 0.018, bracket_y],
              color="black", lw=0.85, clip_on=False)
    p_value = float(summary["wilcoxon_one_sided_data_gt_null_p"])
    axis.text(
        0.5, bracket_y + 0.022, f"p={p_value:.3f}",
        fontsize=7.2, ha="center", va="bottom",
    )
    axis.text(
        0.5, 0.04,
        f"{summary['n_data_gt_null']}/{summary['n_subjects']} subjects above null",
        transform=axis.transAxes, fontsize=6.4, ha="center", va="bottom",
        color="#5E2354",
    )
    axis.set_xlim(-0.58, 1.58)
    axis.set_ylim(0.0, 1.08)
    axis.set_xticks(positions)
    axis.set_xticklabels(["Interictal field\nvs early ictal", "Channel-shuffle\nnull median"],
                         fontsize=6.6)
    axis.set_ylabel("Field concordance  |r|", fontsize=7.6)
    axis.set_title(
        "Clinical cohort spatial concordance",
        fontsize=8.7, fontweight="bold", pad=15,
    )
    _style_axis(axis)
    return {
        **summary,
        "metric": (
            "within-subject absolute spatial correlation between the frozen "
            "interictal A/B propagation field and clinical-onset 0–10 s, "
            "1–150 Hz early-ictal energy field; seizures are collapsed within "
            "subject before the cohort statistic"
        ),
        "null": (
            "per-subject median after shuffling field values across all contacts "
            "while preserving the implanted contact geometry"
        ),
        "inference": "one-sided paired Wilcoxon, observed > channel-null median",
        "boundary": (
            "clinical data bridge, not a cohort of individualized two-core SNNs; "
            "channel shuffle is weaker than a within-shaft null"
        ),
    }


def _timescale_latency_matrix(aggregate: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    records = []
    pattern = re.compile(r"rev21_ts_tz(\d+)_ta(\d+)$")
    for row in aggregate["candidate_summaries"]:
        match = pattern.match(str(row["candidate_id"]))
        if not match:
            continue
        tau_z = float(match.group(1)) / 1000.0
        tau_m = float(match.group(2)) / 1000.0
        onset = row["operational_onset_ms"]
        records.append((
            tau_z, tau_m, float(onset["median"]) / 1000.0,
            int(onset["n"]),
        ))
    tau_z_values = np.asarray(sorted({row[0] for row in records}), float)
    tau_m_values = np.asarray(sorted({row[1] for row in records}), float)
    latency = np.full((len(tau_z_values), len(tau_m_values)), np.nan)
    count = np.zeros_like(latency, dtype=int)
    for tau_z, tau_m, value, n in records:
        iz = int(np.flatnonzero(np.isclose(tau_z_values, tau_z))[0])
        im = int(np.flatnonzero(np.isclose(tau_m_values, tau_m))[0])
        latency[iz, im] = value
        count[iz, im] = n
    if np.any(~np.isfinite(latency)) or np.any(count <= 0):
        raise RuntimeError("timescale latency map is incomplete")
    return tau_z_values, tau_m_values, latency, count


def _plot_panel_f(axis: plt.Axes, aggregate: dict, colorbar_axis: plt.Axes) -> dict:
    tau_z, tau_m, latency, count = _timescale_latency_matrix(aggregate)
    image = axis.imshow(
        latency, origin="lower", cmap="turbo", interpolation="nearest",
        aspect="auto", vmin=float(np.min(latency)), vmax=float(np.max(latency)),
    )
    midpoint = 0.5 * (float(np.min(latency)) + float(np.max(latency)))
    for iz in range(latency.shape[0]):
        for im in range(latency.shape[1]):
            color = "white" if latency[iz, im] >= midpoint else "black"
            axis.text(
                im, iz, f"{latency[iz, im]:.2f} s\n{count[iz, im]}/4",
                ha="center", va="center", fontsize=6.4, color=color,
            )
    axis.set_xticks(np.arange(len(tau_m)))
    axis.set_xticklabels([f"{value:g}" for value in tau_m])
    axis.set_yticks(np.arange(len(tau_z)))
    axis.set_yticklabels([f"{value:g}" for value in tau_z])
    axis.set_xlabel(r"adaptation recovery  $\tau_m$ (s)", fontsize=7.6)
    axis.set_ylabel(r"inhibitory recovery  $\tau_z$ (s)", fontsize=7.6)
    axis.set_title(
        "Slow-state timescales set runaway latency",
        fontsize=8.8, fontweight="bold", pad=5,
    )
    _style_axis(axis, top=True, right=True)
    colorbar = axis.figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("median operational\nrunaway latency (s)", fontsize=6.4)
    colorbar.ax.tick_params(labelsize=6.0, width=0.65, length=2.2)
    return {
        "tau_z_s": tau_z.tolist(),
        "tau_m_s": tau_m.tolist(),
        "median_operational_latency_s": latency.tolist(),
        "n_transition_runs": count.tolist(),
        "cell_denominator": 4,
        "detector": "20-ms causal EMA of E rate >=120 Hz for >=100 ms",
        "boundary": (
            "existing 3x3 timescale library; every cell has four operational "
            "transitions, but only one run passed the stricter rev21 model-ictal "
            "qualification and the library did not freeze a formal work point"
        ),
    }


def _save_all(figure: plt.Figure, stem: Path) -> dict:
    outputs = {}
    for suffix, options in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        path = stem.with_suffix("." + suffix)
        figure.savefig(
            path, bbox_inches="tight", pad_inches=0.025,
            facecolor="white", **options,
        )
        outputs[suffix] = {"path": str(path), "sha256": _sha256(path)}
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, default=None)
    parser.add_argument(
        "--config", type=Path,
        default=Path("config/topic4_rev21_dual_core_zm_transition.json"),
    )
    parser.add_argument(
        "--worker", type=Path, default=Path(
            "results/topic4_sef_hfo/data_driven_dual_core_zm_transition/"
            "timescale/workers/rev21_ts_tz3000_ta500_topology_2542_dynamics_2642.json"),
    )
    parser.add_argument(
        "--bifurcation", type=Path, default=Path(
            "/data/hfosp_topic4_fig45_artifacts/fig5/"
            "data_driven_dual_core_spatial_z/bifurcation/"
            "dualcore_spatial_z_bifurcation.json"),
    )
    parser.add_argument(
        "--branch-atlas", type=Path, default=Path(
            "/data/hfosp_topic4_fig45_artifacts/fig5/"
            "data_driven_dual_core_spatial_z/bifurcation/branch_atlas/"
            "dualcore_spatial_z_branch_atlas.json"),
    )
    parser.add_argument(
        "--stability-assay", type=Path, default=Path(
            "/data/hfosp_topic4_fig45_artifacts/fig5/"
            "data_driven_dual_core_spatial_z/bifurcation/stability_assay/"
            "delay_ou_operating_section.json"),
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("results/paper-ready-figure/fig5_dual_core_transition_story/figures"),
    )
    args = parser.parse_args()

    source_repo = (args.source_repo.resolve() if args.source_repo
                   else _source_checkout().resolve())
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    worker_json = args.worker if args.worker.is_absolute() else source_repo / args.worker
    worker_npz = worker_json.with_suffix(".npz")
    meta = json.loads(worker_json.read_text(encoding="utf-8"))
    candidate_manifest_path = worker_json.parent.parent / 'candidate_manifest.json'
    eta_m = _candidate_eta_m(
        meta, json.loads(candidate_manifest_path.read_text(encoding='utf-8')))
    with np.load(worker_npz, allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    if meta.get("candidate_id") != "rev21_ts_tz3000_ta500":
        raise RuntimeError("Figure 5 requires the frozen rev21 timescale candidate")
    if (int(meta.get("topology_seed", -1)) != 2542
            or int(meta.get("dynamics_seed", -1)) != 2642):
        raise RuntimeError("Figure 5 requires topology 2542 / dynamics 2642")
    if meta["mechanism_freeze"].get("Z_M") != "z_plus_m":
        raise RuntimeError("Figure 5 worker does not have both Z and M active")

    selected_event = _select_interictal_event(meta, arrays)
    stages = _stage_contract(meta, selected_event)

    bifurcation_path = args.bifurcation.resolve()
    bifurcation = json.loads(bifurcation_path.read_text(encoding="utf-8"))
    if bifurcation.get("status") != "DUAL_CORE_SPATIAL_Z_FOLD_CHAIN_ESTABLISHED":
        raise RuntimeError("Panel E requires the verified spatial-Z fold result")
    if bifurcation["substrate"]["identity"] != "dualcore_s39 + Joint=1.25":
        raise RuntimeError("Panel E and A--C use different model substrates")
    bifurcation_arrays = np.load(bifurcation_path.with_suffix(".npz"), allow_pickle=False)
    branch_path = args.branch_atlas.resolve()
    branch_payload = json.loads(branch_path.read_text(encoding="utf-8"))
    if branch_payload.get("status") != "DUAL_CORE_SPATIAL_Z_MULTIBRANCH_ATLAS_COMPLETE":
        raise RuntimeError("Panel E requires the completed spatial-Z branch atlas")
    branch_arrays = np.load(branch_path.with_suffix(".npz"), allow_pickle=False)
    stability_path = args.stability_assay.resolve()
    stability_payload = json.loads(stability_path.read_text(encoding="utf-8"))
    if stability_payload.get("status") != "DUAL_CORE_SPATIAL_Z_DELAY_OU_ASSAY_COMPLETE":
        raise RuntimeError("Panel E requires the completed delay-aware OU assay")

    subject_csv = (
        source_repo / "results/topic5_ictal_recruitment/tspectral_field_concordance/"
        "clinical_onset_gradient_field_cohort_stat_subject.csv")
    cohort_csv = subject_csv.with_name(
        "clinical_onset_gradient_field_cohort_stat_cohort.csv")
    clinical_rows, clinical_summary = _load_clinical_cohort(subject_csv, cohort_csv)
    timescale_path = (
        source_repo / "results/topic4_sef_hfo/data_driven_dual_core_zm_transition/"
        "timescale/aggregate.json")
    timescale = json.loads(timescale_path.read_text(encoding="utf-8"))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    output_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(15.8, 10.1), facecolor="white")
    figure.suptitle(
        "Slow inhibitory depletion links recurrent interictal propagation "
        "to runaway recruitment",
        fontsize=15.0, fontweight="bold", y=0.988,
    )
    outer = figure.add_gridspec(
        2, 1, height_ratios=[1.56, 0.90], left=0.055, right=0.976,
        bottom=0.065, top=0.910, hspace=0.22,
    )
    upper = outer[0].subgridspec(
        1, 2, width_ratios=[1.55, 1.0], wspace=0.18)
    left = upper[0, 0].subgridspec(
        2, 1, height_ratios=[1.28, 0.92], hspace=0.20)
    right = upper[0, 1].subgridspec(
        2, 1, height_ratios=[1.13, 0.86], hspace=0.30)
    lower = outer[1].subgridspec(
        1, 2, width_ratios=[1.35, 1.0], wspace=0.22)

    a_grid = left[0, 0].subgridspec(2, 1, height_ratios=[0.25, 1.0], hspace=0.035)
    ax_a_rate = figure.add_subplot(a_grid[0, 0])
    ax_a_trace = figure.add_subplot(a_grid[1, 0], sharex=ax_a_rate)
    panel_a = _plot_panel_a(ax_a_rate, ax_a_trace, arrays, stages)
    _panel_label(ax_a_rate, "A", x=-0.085, y=1.46)

    b_grid = left[1, 0].subgridspec(3, 1, hspace=0.06)
    ax_b_energy = figure.add_subplot(b_grid[0, 0])
    ax_b_z = figure.add_subplot(b_grid[1, 0], sharex=ax_b_energy)
    ax_b_m = figure.add_subplot(b_grid[2, 0], sharex=ax_b_energy)
    panel_b = _plot_panel_b(
        (ax_b_energy, ax_b_z, ax_b_m), arrays, stages, eta_m)
    _panel_label(ax_b_energy, "B", x=-0.085, y=1.27)

    c_grid = right[0, 0].subgridspec(
        1, 4, width_ratios=[1.0, 1.0, 1.0, 0.055], wspace=0.11)
    ax_c = [figure.add_subplot(c_grid[0, index]) for index in range(3)]
    ax_cbar = figure.add_subplot(c_grid[0, 3])
    panel_c = _plot_panel_c(
        ax_c, arrays, stages, selected_event,
        np.asarray(bifurcation["substrate"]["centers_mm"], float), ax_cbar)
    ax_c[0].text(
        -0.26, 1.30, "C", transform=ax_c[0].transAxes, fontsize=16,
        fontweight="bold", ha="left", va="top", clip_on=False)
    ax_c[0].text(
        0.0, 1.30, "Shared spatial axis across the transition",
        transform=ax_c[0].transAxes, fontsize=8.8, fontweight="bold",
        ha="left", va="top", clip_on=False)

    ax_d = figure.add_subplot(right[1, 0])
    panel_d = _plot_panel_d(ax_d, clinical_rows, clinical_summary)
    _panel_label(ax_d, "D", x=-0.10, y=1.12)

    ax_e = figure.add_subplot(lower[0, 0])
    panel_e = _plot_spatial_z_branch_atlas(
        ax_e, bifurcation_arrays, bifurcation, branch_arrays,
        branch_payload, stability_payload)
    ax_e.set_title(
        "Spatial-Z fixed-point branches organize runaway susceptibility",
        fontsize=9.2, fontweight="bold", pad=6)
    _panel_label(ax_e, "E", x=-0.10, y=1.10)

    f_grid = lower[0, 1].subgridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.10)
    ax_f = figure.add_subplot(f_grid[0, 0])
    ax_fbar = figure.add_subplot(f_grid[0, 1])
    panel_f = _plot_panel_f(ax_f, timescale, ax_fbar)
    _panel_label(ax_f, "F", x=-0.15, y=1.10)

    stem = output_dir / "fig5-dual-core-transition-story-v2"
    outputs = _save_all(figure, stem)
    plt.close(figure)
    bifurcation_arrays.close()
    branch_arrays.close()

    source_entries = {
        "frozen_rev21_config": config_path,
        "worker_json": worker_json,
        "worker_npz": worker_npz,
        "candidate_manifest": candidate_manifest_path,
        "bifurcation_json": bifurcation_path,
        "bifurcation_npz": bifurcation_path.with_suffix(".npz"),
        "branch_atlas_json": branch_path,
        "branch_atlas_npz": branch_path.with_suffix(".npz"),
        "stability_assay_json": stability_path,
        "clinical_subject_csv": subject_csv,
        "clinical_cohort_csv": cohort_csv,
        "timescale_aggregate_json": timescale_path,
    }
    metadata = {
        "status": "FIG5_DUAL_CORE_TRANSITION_STORY_V2_CANDIDATE",
        "substrate": "dualcore_s39 + Joint=1.25",
        "representative_run": {
            "candidate_id": meta["candidate_id"],
            "topology_seed": int(meta["topology_seed"]),
            "dynamics_seed": int(meta["dynamics_seed"]),
            "continuous_ou": True,
            "z_and_m_active": True,
            "model_state_eligible_rev21": bool(meta["model_ictal_rev21"]["eligible"]),
        },
        "time_contract": {
            **stages,
            "A_B_axes_identical": True,
            "C_windows_are_exact_subsets_of_A_B": True,
            "onset_definition": (
                "rev21 scientific model-state onset = operational detector time "
                "minus the frozen 100-ms offset"
            ),
        },
        "selected_interictal_event": {
            **selected_event,
            "selection_rule": "latest returned event with complete 15-contact onset readout",
        },
        "panel_A": panel_a,
        "panel_B": panel_b,
        "panel_C": panel_c,
        "panel_D": panel_d,
        "panel_E": panel_e,
        "panel_F": panel_f,
        "sources": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in source_entries.items()
        },
        "outputs": outputs,
        "claim_boundary": (
            "A--C are one development-only E1146 dual-core SNN trajectory; D is "
            "a separate clinical cohort data bridge, not individualized SNN "
            "prediction; E is a 2-mm deterministic frozen fast-subsystem branch "
            "atlas; F is an existing 3x3 operational-latency screen. The figure "
            "does not establish a clinical seizure mechanism, a thermodynamic "
            "phase transition, or delay-aware stability along every branch."
        ),
    }
    metadata_path = output_dir / "fig5-dual-core-transition-story-v2-metadata.json"
    _atomic_json(metadata, metadata_path)
    (output_dir / "README.md").write_text(
        "### fig5-dual-core-transition-story-v2.png / .pdf / .svg\n\n"
        "Fig.5 六联候选图。A/B 来自同一条 `dualcore_s39 + Joint=1.25`、topology 2542 / dynamics 2642、连续 OU 驱动且 Z/M 同时开启的 40,000-cell SNN 轨迹，并严格共用 0–3.815 s 时间轴。A 显示返回型间期事件、pre-onset 放大和持续招募态；B 显示同一时刻的 population E rate、core/surround 的 `Z` 与 core adaptation `A=eta_m*m`。当前工作点是 tonic plateau，10–250 Hz band-limited virtual-contact power 在 onset 后下降，因此没有把它伪装成 runaway magnitude；runaway 用直接的群体放电率和招募比例定义。\n\n"
        "C 使用 A/B 中精确标出的三个时间窗，统一画 0.5-mm local-E rate 的 `log10(1+Hz)`。白色箭头由规则选中的返回型间期事件 onset map 拟合，并原样复制到 pre-onset 与 early recruited map；图下的 `|cos Δtheta|` 是各窗口 `log(1+rate)` 空间梯度与该间期 onset-gradient 的绝对余弦。橙/青轮廓是同一 realized network 的两个冻结 core，空心点是虚拟触点。\n\n"
        "D 是独立的临床 cohort 桥，不是 16 个患者各自跑了 two-core SNN。指标是每名患者冻结间期 A/B propagation field 与 clinical onset 后 0–10 s、1–150 Hz early-ictal energy field 的绝对空间相关，先在患者内折叠，再与保留触点几何的 all-contact channel-shuffle 中位数配对比较；strict-broadband 组为 12/16 高于 null，单侧 paired Wilcoxon `p=0.0193`。该 null 比 within-shaft null 弱。\n\n"
        "E 复用经过 pseudo-arclength 折返与 fixed-point Jacobian 零特征值共同核验的 spatial-Z 多分支图谱。线型不编码 delay-aware 稳定性；OU-on 竖线只投影 100 个 active runs 的中位工作截面。F 使用现有 `tau_z x tau_m` 3x3 网格，每格 2 topology x 2 dynamics；格内上行为 median operational runaway latency、下行为发生转变的次数。\n\n"
        "**关注点**：这是 layout 与证据链都已落到真实 artifact 的 v2，不含示意数字。当前最需要补的是 F 的 `depletion strength x tau_z` 专扫，以及把 C 的空间轴一致性在多个 topology/dynamics seed 上确认；D 目前承担临床数据桥，不能写成跨患者模型预测。\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": metadata["status"],
        "figure": outputs,
        "metadata": str(metadata_path),
        "clinical_n_above_null": panel_d["n_data_gt_null"],
        "clinical_n": panel_d["n_subjects"],
        "spatial_alignment": panel_c["absolute_cosine_alignment"],
    }, indent=2))


if __name__ == "__main__":
    main()
