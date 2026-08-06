#!/usr/bin/env python3
"""Render the existing R4 regional lifecycle without inventing SEEG or field data.

The source trajectory, hybrid recovery checkpoints, same-basin sentinel, and
recovered-state challenge are distinct registered artifacts.  This renderer
keeps those boundaries visible: analytic recovery segments are dashed and the
late challenge is labelled as a fork.  The spatial snapshots are the exact
piecewise-constant P=3 projection of regional rates, not a continuous field.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfosp_mpl_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_mz_spatial_reduction import canonical_m3b_core_annulus_bath  # noqa: E402


SOURCE_TRACE = ROOT / "results/topic4_sef_hfo/mz_m_gated_reserve_coupled_canary/segment_a_center_canary_trace.npz"
R4_TRACE = ROOT / "results/topic4_sef_hfo/mz_actual_entry_lifecycle_closure/representative_traces_dt0p125.npz"
R4_ENDPOINTS = ROOT / "results/topic4_sef_hfo/mz_actual_entry_lifecycle_closure/hybrid_endpoint_table.csv"
R4_SUMMARY = ROOT / "results/topic4_sef_hfo/mz_actual_entry_lifecycle_closure/actual_entry_lifecycle_closure_summary.json"
OUTPUT = ROOT / "results/paper-ready-figure/fig5_mz_r4_regional_lifecycle_diagnostic/figures"

PATCH_NAMES = ("core", "annulus", "bath")
PATCH_COLORS = ("#B2182B", "#EF8A62", "#2166AC")
Q_COLOR = "#2166AC"
P_COLOR = "#4D9221"
M_COLOR = "#B2182B"
FOLD_Q = 0.8558315843088748
Q_RESET = 0.885
EVENT_ONSETS_MS = np.asarray([1000.0, 3122.0, 5044.0, 6321.0, 7531.0, 10915.0])
ENTRY_MS = 7620.0
LATCH_SET_MS = 10234.0
TERMINATION_MS = 11172.0
RESET_MS = 74117.30225332972
A_MAX_MV = 1.6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _load_endpoints(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not np.isclose(float(row["dt_ms"]), 0.125):
            continue
        selected.append({
            **row,
            "absolute_time_ms": float(row["absolute_time_ms"]),
            "q": float(row["q"]),
            "p": float(row["p"]),
            "m": float(row["m"]),
            "A_mv": float(row["A_mv"]),
            "rE_max_hz": float(row["rE_max_hz"]),
            "latch_on": row["latch_on"] == "True",
        })
    if not selected:
        raise RuntimeError("base-dt R4 endpoint rows are missing")
    return selected


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8, width=0.7, length=3)


def _plot_rates(ax: plt.Axes, time_s: np.ndarray, rates_hz: np.ndarray) -> None:
    for patch, (name, color) in enumerate(zip(PATCH_NAMES, PATCH_COLORS)):
        ax.plot(time_s, rates_hz[:, patch], color=color, lw=0.9, label=name)
    _style_axis(ax)


def _nearest(time_ms: np.ndarray, value_ms: float) -> int:
    return int(np.argmin(np.abs(np.asarray(time_ms, dtype=float) - float(value_ms))))


def _window_peak(time_ms: np.ndarray, rates_hz: np.ndarray, start_ms: float, stop_ms: float) -> int:
    mask = (time_ms >= start_ms) & (time_ms <= stop_ms)
    indices = np.flatnonzero(mask)
    if not indices.size:
        raise RuntimeError("snapshot peak window is empty")
    return int(indices[np.argmax(rates_hz[indices, 0])])


def _arrow_along(ax: plt.Axes, x: np.ndarray, y: np.ndarray, start: int, stop: int, color: str) -> None:
    ax.annotate(
        "",
        xy=(float(x[stop]), float(y[stop])),
        xytext=(float(x[start]), float(y[start])),
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": color, "mutation_scale": 9},
    )


def render(output: Path = OUTPUT) -> tuple[Path, Path]:
    for required in (SOURCE_TRACE, R4_TRACE, R4_ENDPOINTS, R4_SUMMARY):
        if not required.is_file():
            raise FileNotFoundError(required)
    summary = json.loads(R4_SUMMARY.read_text(encoding="utf-8"))
    if summary.get("status") != "R4_ACTUAL_ENTRY_REGIONAL_HYBRID_LIFECYCLE_CENTER_SUPPORTED":
        raise RuntimeError("R4 source status is not the accepted center closure")

    source = _load_npz(SOURCE_TRACE)
    r4 = _load_npz(R4_TRACE)
    endpoints = _load_endpoints(R4_ENDPOINTS)
    output.mkdir(parents=True, exist_ok=True)

    source_time_ms = np.asarray(source["time_ms"], dtype=float)
    source_time_s = source_time_ms / 1000.0
    source_fast_hz = 1000.0 * np.asarray(source["rE_fast_khz"], dtype=float)
    source_q = np.mean(np.asarray(source["q"], dtype=float)[:, :2], axis=1)
    source_p = np.mean(np.asarray(source["persistence"], dtype=float)[:, :2], axis=1)
    source_m = np.mean(np.asarray(source["m"], dtype=float)[:, :2], axis=1)
    source_a = A_MAX_MV * source_m
    source_latch = np.any(np.asarray(source["latch"], dtype=bool)[:, :2], axis=1)

    final_time_s = np.asarray(r4["final_time_ms"], dtype=float) / 1000.0
    final_fast_hz = 1000.0 * np.asarray(r4["final_rE_fast_khz"], dtype=float)
    late_time_ms = np.asarray(r4["late_time_ms"], dtype=float)
    late_time_s = late_time_ms / 1000.0
    late_fast_hz = 1000.0 * np.asarray(r4["late_rE_fast_khz"], dtype=float)

    fig = plt.figure(figsize=(18.5, 12.0), constrained_layout=True)
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.015, 0.045, 0.985, 0.965), h_pad=0.04, w_pad=0.04)
    grid = fig.add_gridspec(3, 3, height_ratios=(1.15, 1.02, 0.90), hspace=0.16, wspace=0.12)

    # A: three explicitly separated pieces of the evidence chain.
    ax_a1 = fig.add_subplot(grid[0, 0])
    mask = (source_time_s >= 0.65) & (source_time_s <= 11.45)
    _plot_rates(ax_a1, source_time_s[mask], source_fast_hz[mask])
    for event in EVENT_ONSETS_MS[:5] / 1000.0:
        ax_a1.axvline(event, color="0.82", lw=0.55, zorder=0)
    ax_a1.axvspan(ENTRY_MS / 1000.0, TERMINATION_MS / 1000.0, color="#F4A6A6", alpha=0.18, zorder=-2)
    ax_a1.axvline(ENTRY_MS / 1000.0, color="#B2182B", ls="--", lw=1.0)
    ax_a1.axvline(LATCH_SET_MS / 1000.0, color="black", ls=":", lw=0.9)
    ax_a1.axvline(TERMINATION_MS / 1000.0, color="black", ls="--", lw=0.9)
    ax_a1.text(0.015, 0.95, "A", transform=ax_a1.transAxes, va="top", fontweight="bold", fontsize=12)
    ax_a1.text(0.05, 0.82, "events 1–4\nreturn", transform=ax_a1.transAxes, color="#2166AC", fontsize=8)
    ax_a1.text(0.68, 0.90, "event 5 entry\n+ 4 autonomous returns", transform=ax_a1.transAxes, color="#9E0142", fontsize=8)
    ax_a1.set(xlim=(0.65, 11.45), ylim=(0, 118), xlabel="source time (s)", ylabel="$r_E^{fast}$ (Hz)", title="Registered IED-like train → bounded ictal-like bursting → exit")
    ax_a1.legend(frameon=False, fontsize=7, ncol=3, loc="upper left", bbox_to_anchor=(0.18, 1.0))

    ax_a2 = fig.add_subplot(grid[0, 1])
    _plot_rates(ax_a2, final_time_s, final_fast_hz)
    ax_a2.axhspan(0, 1.0, color="#D9EF8B", alpha=0.18, zorder=-2)
    ax_a2.text(0.015, 0.95, "B", transform=ax_a2.transAxes, va="top", fontweight="bold", fontsize=12)
    ax_a2.text(0.5, 0.73, "same-basin sentinel\n$\max r_E^{fast}=0.823$ Hz", transform=ax_a2.transAxes, ha="center", fontsize=8)
    ax_a2.set(xlim=(0, 4), ylim=(0, 2.2), xlabel="final full integration (s)", ylabel="$r_E^{fast}$ (Hz)", title="After hybrid recovery: stable low state")

    ax_a3 = fig.add_subplot(grid[0, 2])
    late_mask = (late_time_s >= 0.65) & (late_time_s <= 4.30)
    _plot_rates(ax_a3, late_time_s[late_mask], late_fast_hz[late_mask])
    for event in EVENT_ONSETS_MS[:2] / 1000.0:
        ax_a3.axvline(event, color="0.78", lw=0.7, zorder=0)
    ax_a3.text(0.015, 0.95, "C", transform=ax_a3.transAxes, va="top", fontweight="bold", fontsize=12)
    ax_a3.text(0.52, 0.84, "returning responses restored", transform=ax_a3.transAxes, ha="center", color="#2166AC", fontsize=8)
    ax_a3.set(xlim=(0.65, 4.30), ylim=(0, 118), xlabel="recovered-state challenge time (s)", ylabel="$r_E^{fast}$ (Hz)", title="Same-basin fork: interictal-like events resume")

    # D: slow variables during the actual entry and fast exit.
    ax_d = fig.add_subplot(grid[1, 0])
    onset_mask = (source_time_s >= 0.65) & (source_time_s <= 11.45)
    line_q, = ax_d.plot(source_time_s[onset_mask], source_q[onset_mask], color=Q_COLOR, lw=1.4, label="$q$")
    ax_d.axhline(FOLD_Q, color=Q_COLOR, ls="--", lw=0.8, label="regional entry fold")
    ax_d.set_ylim(0.842, 0.903)
    ax_d.set_xlabel("source time (s)")
    ax_d.set_ylabel("inhibitory resource $q$", color=Q_COLOR)
    ax_d.tick_params(axis="y", labelcolor=Q_COLOR)
    ax_d2 = ax_d.twinx()
    line_p, = ax_d2.plot(source_time_s[onset_mask], source_p[onset_mask], color=P_COLOR, lw=1.0, label="$p$")
    line_m, = ax_d2.plot(source_time_s[onset_mask], source_m[onset_mask], color=M_COLOR, lw=1.15, label="$m$")
    line_l, = ax_d2.plot(source_time_s[onset_mask], 0.24 * source_latch[onset_mask], color="black", lw=0.9, drawstyle="steps-post", label="$0.24L$")
    ax_d2.set_ylim(-0.005, 0.255)
    ax_d2.set_ylabel("$p$, $m$ and scaled latch")
    ax_d.axvline(ENTRY_MS / 1000.0, color="#B2182B", ls="--", lw=0.9)
    ax_d.axvline(LATCH_SET_MS / 1000.0, color="black", ls=":", lw=0.9)
    ax_d.axvline(TERMINATION_MS / 1000.0, color="black", ls="--", lw=0.9)
    ax_d.text(0.015, 0.95, "D", transform=ax_d.transAxes, va="top", fontweight="bold", fontsize=12)
    ax_d.set_title("Slow ordering: q opens entry; p sets latch; m supplies exit")
    _style_axis(ax_d)
    ax_d.spines["right"].set_visible(False)
    ax_d2.spines["top"].set_visible(False)
    ax_d.legend([line_q, line_p, line_m, line_l], ["q", "p", "m", "latch (scaled)"], frameon=False, fontsize=7, ncol=2, loc="lower left")

    # E: actual slow path over the frozen entry/exit landmarks.
    ax_e = fig.add_subplot(grid[1, 1])
    source_stride = 10
    ax_e.plot(source_q[::source_stride], source_a[::source_stride], color="#7B3294", lw=1.35, label="coupled source path")
    endpoint_q = np.asarray([row["q"] for row in endpoints], dtype=float)
    endpoint_a = np.asarray([row["A_mv"] for row in endpoints], dtype=float)
    ax_e.plot(endpoint_q, endpoint_a, color="#008837", lw=1.2, ls="--", label="hybrid recovery path")
    full = np.asarray([str(row["kind"]).startswith("full") or row["kind"] == "final_4s_full" for row in endpoints])
    ax_e.scatter(endpoint_q[~full], endpoint_a[~full], facecolors="white", edgecolors="#008837", s=24, lw=0.8, zorder=3)
    ax_e.scatter(endpoint_q[full], endpoint_a[full], color="#008837", edgecolors="white", s=25, lw=0.5, zorder=4)
    ax_e.scatter([FOLD_Q], [0.0], marker="*", s=125, color="#1A9850", edgecolor="white", lw=0.6, zorder=5, label="regional fold at $A=0$")
    ax_e.scatter([0.855, 0.850], [0.020, 0.120], marker="s", s=38, color="#FDAE61", edgecolor="#A6611A", lw=0.7, zorder=4, label="frozen delayed-exit brackets")
    idx_entry = _nearest(source_time_ms, ENTRY_MS)
    idx_exit = _nearest(source_time_ms, TERMINATION_MS)
    ax_e.scatter([source_q[idx_entry], source_q[idx_exit]], [source_a[idx_entry], source_a[idx_exit]], color=["#B2182B", "black"], s=35, zorder=6)
    ax_e.annotate("entry", (source_q[idx_entry], source_a[idx_entry]), xytext=(8, 9), textcoords="offset points", fontsize=7)
    ax_e.annotate("finite exit", (source_q[idx_exit], source_a[idx_exit]), xytext=(-48, -13), textcoords="offset points", fontsize=7)
    reset_row = min(endpoints, key=lambda row: abs(row["absolute_time_ms"] - RESET_MS))
    final_row = endpoints[-1]
    ax_e.scatter([reset_row["q"], final_row["q"]], [reset_row["A_mv"], final_row["A_mv"]], color=["#F46D43", "#313695"], s=36, zorder=6)
    ax_e.annotate("latch reset", (reset_row["q"], reset_row["A_mv"]), xytext=(6, -13), textcoords="offset points", fontsize=7)
    ax_e.annotate("same basin", (final_row["q"], final_row["A_mv"]), xytext=(-54, 8), textcoords="offset points", fontsize=7)
    _arrow_along(ax_e, source_q, source_a, _nearest(source_time_ms, 6300), _nearest(source_time_ms, 7650), "#7B3294")
    _arrow_along(ax_e, endpoint_q, endpoint_a, 7, min(11, len(endpoint_q) - 1), "#008837")
    ax_e.text(0.015, 0.95, "E", transform=ax_e.transAxes, va="top", fontweight="bold", fontsize=12)
    ax_e.set(xlim=(0.841, 0.902), ylim=(-0.008, 0.415), xlabel="regional inhibitory resource $q$", ylabel="additive exit current $A_M$ (mV)", title="Frozen landmarks + actual q–A slow loop")
    _style_axis(ax_e)
    ax_e.legend(frameon=False, fontsize=6.6, loc="center", bbox_to_anchor=(0.59, 0.54))

    # F: long recovery shown as hybrid, not dense continuous integration.
    ax_f = fig.add_subplot(grid[1, 2])
    endpoint_t_s = np.asarray([row["absolute_time_ms"] for row in endpoints]) / 1000.0
    ax_f.plot(source_time_s, source_q, color=Q_COLOR, lw=1.0)
    ax_f.plot(endpoint_t_s, endpoint_q, color=Q_COLOR, ls="--", lw=1.1, label="$q$ (hybrid checkpoints)")
    ax_f.scatter(endpoint_t_s[~full], endpoint_q[~full], facecolors="white", edgecolors=Q_COLOR, s=20, lw=0.7, zorder=3)
    ax_f.scatter(endpoint_t_s[full], endpoint_q[full], color=Q_COLOR, edgecolors="white", s=21, lw=0.4, zorder=4)
    ax_f.axhline(Q_RESET, color=Q_COLOR, ls=":", lw=0.8)
    ax_f.set_ylim(0.842, 0.903)
    ax_f.set_xlabel("hybrid absolute time (s)")
    ax_f.set_ylabel("$q$", color=Q_COLOR)
    ax_f.tick_params(axis="y", labelcolor=Q_COLOR)
    ax_f2 = ax_f.twinx()
    ax_f2.plot(source_time_s, source_a, color=M_COLOR, lw=1.0)
    ax_f2.plot(endpoint_t_s, endpoint_a, color=M_COLOR, ls="--", lw=1.1, label="$A_M$")
    ax_f2.scatter(endpoint_t_s[~full], endpoint_a[~full], facecolors="white", edgecolors=M_COLOR, s=20, lw=0.7, zorder=3)
    ax_f2.scatter(endpoint_t_s[full], endpoint_a[full], color=M_COLOR, edgecolors="white", s=21, lw=0.4, zorder=4)
    ax_f2.set_ylim(-0.01, 0.415)
    ax_f2.set_ylabel("$A_M$ (mV)", color=M_COLOR)
    ax_f2.tick_params(axis="y", labelcolor=M_COLOR)
    ax_f.axvspan(LATCH_SET_MS / 1000.0, RESET_MS / 1000.0, color="#F4A6A6", alpha=0.15, zorder=-3)
    ax_f.axvline(RESET_MS / 1000.0, color="black", ls=":", lw=0.9)
    ax_f.text(RESET_MS / 1000.0 + 3, 0.9005, "latch reset", fontsize=7, va="top")
    ax_f.text(0.015, 0.95, "F", transform=ax_f.transAxes, va="top", fontweight="bold", fontsize=12)
    ax_f.text(0.56, 0.13, "dashed = analytic bridge\nfilled = full-fast sentinel", transform=ax_f.transAxes, fontsize=7, ha="center")
    ax_f.set_title("Protected recovery → reset → natural M release")
    _style_axis(ax_f)
    ax_f.spines["right"].set_visible(False)
    ax_f2.spines["top"].set_visible(False)

    # G: exact P=3 piecewise-constant display of selected rate states.
    snapshot_grid = grid[2, :].subgridspec(1, 7, wspace=0.05)
    source_baseline = source_fast_hz[0]
    ied1 = _window_peak(source_time_ms, source_fast_hz, 1000.0, 1160.0)
    entry = _nearest(source_time_ms, ENTRY_MS)
    burst1 = _window_peak(source_time_ms, source_fast_hz, 8750.0, 8950.0)
    burst4 = _window_peak(source_time_ms, source_fast_hz, 10720.0, 10890.0)
    post_exit = _nearest(source_time_ms, TERMINATION_MS)
    late_ied1 = _window_peak(late_time_ms, late_fast_hz, 1000.0, 1160.0)
    snapshots = [
        ("IED-like peak", source_fast_hz[ied1], "source", float(source_time_ms[ied1])),
        ("entry", source_fast_hz[entry], "source", float(source_time_ms[entry])),
        ("burst 1", source_fast_hz[burst1], "source", float(source_time_ms[burst1])),
        ("burst 4", source_fast_hz[burst4], "source", float(source_time_ms[burst4])),
        ("post-exit", source_fast_hz[post_exit], "source", float(source_time_ms[post_exit])),
        ("same-basin rest", final_fast_hz[-1], "final", float(r4["final_time_ms"][-1])),
        ("recovered IED-like", late_fast_hz[late_ied1], "late_fork", float(late_time_ms[late_ied1])),
    ]
    reduction = canonical_m3b_core_annulus_bath(grid_n=48, grid_L_mm=12.0, core_radius_mm=1.5, theta_rad=np.deg2rad(45.0))
    proxy_vectors = [np.log10(1.0 + np.maximum(values - source_baseline, 0.0) ** 2) for _, values, _, _ in snapshots]
    norm = Normalize(vmin=0.0, vmax=max(float(np.max(values)) for values in proxy_vectors))
    snapshot_axes: list[plt.Axes] = []
    image = None
    for index, ((label, rates, source_name, time_ms), proxy) in enumerate(zip(snapshots, proxy_vectors)):
        ax = fig.add_subplot(snapshot_grid[0, index])
        snapshot_axes.append(ax)
        field = np.zeros_like(reduction.masks[0], dtype=float)
        for patch, region in enumerate(reduction.masks):
            field[region] = proxy[patch]
        image = ax.imshow(field, origin="lower", cmap="magma", norm=norm, extent=(-6, 6, -6, 6), interpolation="nearest")
        ax.contour(reduction.masks[0].astype(float), levels=[0.5], colors="white", linewidths=0.55, origin="lower", extent=(-6, 6, -6, 6))
        ax.contour((reduction.masks[0] | reduction.masks[1]).astype(float), levels=[0.5], colors="white", linewidths=0.45, linestyles="--", origin="lower", extent=(-6, 6, -6, 6))
        ax.set_title(label, fontsize=8.2, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
        if index == 0:
            ax.text(-0.12, 1.06, "G", transform=ax.transAxes, va="top", fontweight="bold", fontsize=12)
            ax.set_ylabel("exact P=3\nstate projection", fontsize=8)
        ax.text(0.5, -0.08, f"{source_name}, {time_ms / 1000.0:.3f} s", transform=ax.transAxes, ha="center", va="top", fontsize=6.4)
    assert image is not None
    colorbar = fig.colorbar(image, ax=snapshot_axes, orientation="vertical", shrink=0.80, pad=0.008)
    colorbar.set_label(r"regional rate-energy proxy  $\log_{10}[1+(r_E^{fast}-r_0)_+^2]$", fontsize=7.5)
    colorbar.ax.tick_params(labelsize=7)

    fig.suptitle("R4 regional q–p–M hybrid lifecycle: fast bursting inside a slow recovery loop", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.002,
        "Diagnostic boundary: P=3 rate-patch model; registered core pulses trigger entry. Not Virtual-SEEG/LFP, not a continuous field, and the 20–312 s recovery contains analytic bridges plus full-fast sentinels; the late train is a same-basin fork.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#8C2D1D",
    )

    png = output / "fig5_mz_r4_regional_lifecycle_diagnostic.png"
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "title": "R4 regional q-p-M hybrid lifecycle diagnostic",
        "status": summary["status"],
        "scientific_layer": "three_patch_fixed_bath_rate_model_diagnostic",
        "inputs": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (SOURCE_TRACE, R4_TRACE, R4_ENDPOINTS, R4_SUMMARY)
        },
        "outputs": [str(png.relative_to(ROOT)), str(pdf.relative_to(ROOT))],
        "landmarks": {
            "regional_entry_fold_q": FOLD_Q,
            "entry_ms": ENTRY_MS,
            "latch_set_ms": LATCH_SET_MS,
            "termination_complete_ms": TERMINATION_MS,
            "latch_reset_absolute_ms": RESET_MS,
            "same_basin_absolute_ms": float(summary["dt_runs"][0]["same_basin_absolute_time_ms"]),
        },
        "snapshot_proxy": "log10(1 + max(rE_fast_hz - source_baseline_hz, 0)^2), displayed piecewise-constant on the exact P3 masks",
        "snapshots": [
            {"label": label, "source": source_name, "time_ms": time_ms, "rE_fast_hz": np.asarray(rates).tolist()}
            for label, rates, source_name, time_ms in snapshots
        ],
        "claim_boundary": [
            "regional rates are not virtual SEEG or LFP",
            "piecewise-constant P3 snapshots are not a continuous emergent field or wavefront",
            "registered core pulses trigger entry; zero-input spontaneous onset was not tested",
            "the long recovery contains analytic bridges bracketed by full-fast sentinels",
            "the recovered challenge is a same-basin state fork, not an uninterrupted spontaneous train",
            "frozen regional exit evidence is sampled at q=.855/.850; the precise dynamic exit fold near q=.845 was not continued",
        ],
    }
    metadata_path = output / "fig5_mz_r4_regional_lifecycle_diagnostic_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (output / "README.md").write_text(
        "### fig5_mz_r4_regional_lifecycle_diagnostic.png / .pdf\n\n"
        "这张图把 R4 已有的三层证据放到同一条可审阅链上：A 展示前四个可返回的 IED-like 响应、event 5 后四次无脉冲自主回返与有限退出；B–C 显式分开展示 hybrid 恢复后的同 basin 静息和 recovered-state challenge 中重新出现的 IED-like 响应。D–F 展示 q、p、m、latch 的时序、实际 q–A 慢环与 frozen fold/exit landmarks 的关系，以及带解析 bridge 和 full-fast sentinel 标记的长恢复。G 只是把三个 patch 的 rate-energy proxy 原样回填到 locked core–annulus–bath mask，不能解释成连续空间场。\n\n"
        "**关注点**：这是 fixed-bath P=3 rate-model diagnostic，不是 Virtual-SEEG/LFP，也不是零输入自发发作。entry 由登记的 core pulse train 触发，20–312 s 段包含解析 slow bridge，恢复后挑战是从同 basin 末态建立的 fork；只有把该机制移植到 continuous field/full SNN 并由同一连续轨迹输出 contact readout 后，才可升级为 Figure 5 的完整 SEEG lifecycle candidate。\n",
        encoding="utf-8",
    )
    return png, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    png, metadata = render(args.output.resolve())
    print(json.dumps({"figure": str(png), "metadata": str(metadata)}, indent=2))


if __name__ == "__main__":
    main()
