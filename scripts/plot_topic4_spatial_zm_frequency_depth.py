#!/usr/bin/env python3
"""Results figure: rhythm frequency versus how much the firing actually moves.

Three questions, one panel each:

a. Does any explored state land in the qualifying quadrant at all -- a dominant
   population rhythm inside 30-80 Hz *and* a firing-rate modulation depth above
   the criterion-10 floor?
b. What do the two ends look like as a waveform, once the plateau is not
   removed by detrending?
c. How does the trade-off arise along the one control parameter, and where does
   the low-to-high bifurcation sit?

This is a diagnostic results figure for the archive, not a paper-ready Fig5A.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_tonic_fixed_point import (  # noqa: E402
    MODULATION_DEPTH_FLOOR,
    population_rate_modulation,
)

INK = "#252525"
BAND = "#DDE7EF"
QUAD = "#E6F2E1"
LOWC = "#3B6FB6"
HIGHC = "#C6431F"
TARGET_HZ = (30.0, 80.0)
HIGH_STATE_RATE_HZ = 120.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _late_state(npz_path, span_ms=1000.0):
    with np.load(npz_path) as artifact:
        dt = float(artifact["lfp_dt_ms"])
        rate = np.asarray(artifact["rate_E_hz"], float)
        active = np.asarray(artifact["active_neuron_fraction_20ms"], float)
    n = int(round(span_ms / dt))
    if len(rate) < n:
        return None
    got = population_rate_modulation(rate[-n:], dt_ms=dt)
    got["median_active_fraction"] = float(np.median(active[-max(1, len(active) // 4):]))
    got["dt_ms"] = dt
    return got


def _onset_state(npz_path, onset_ms, settle_ms=300.0, span_ms=1000.0):
    with np.load(npz_path) as artifact:
        dt = float(artifact["lfp_dt_ms"])
        rate = np.asarray(artifact["rate_E_hz"], float)
    time = np.arange(len(rate), dtype=float) * dt
    selected = ((time >= float(onset_ms) + settle_ms)
                & (time < float(onset_ms) + settle_ms + span_ms))
    if int(np.sum(selected)) < 64:
        return None
    got = population_rate_modulation(rate[selected], dt_ms=dt)
    got["dt_ms"] = dt
    return got


def collect(paths_config):
    """Collect (family, label, frequency, depth, mean rate, q) for every state."""
    rows = []
    for npz_path, q_value in paths_config["frozen_q_axis"]:
        state = _late_state(ROOT / npz_path)
        if state is None:
            continue
        rows.append({"family": "frozen q axis", "q": q_value,
                     "dominant_hz": state["dominant_hz"],
                     "depth": state["modulation_depth"],
                     "mean_rate_hz": state["mean_rate_hz"]})
    for pattern in paths_config["frozen_gk_globs"]:
        for path in sorted(glob.glob(str(ROOT / pattern))):
            json_path = path.replace(".npz", ".json")
            if not os.path.exists(json_path):
                continue
            config = (json.loads(Path(json_path).read_text()).get(
                "hybrid_config") or {})
            try:
                state = _late_state(path)
            except ValueError:
                continue
            if state is None:
                continue
            rows.append({"family": "frozen q + adaptation", "q": config.get("q_init"),
                         "dominant_hz": state["dominant_hz"],
                         "depth": state["modulation_depth"],
                         "mean_rate_hz": state["mean_rate_hz"]})
    for pattern in paths_config["dynamic_globs"]:
        for path in sorted(glob.glob(str(ROOT / pattern), recursive=True)):
            json_path = path.replace(".npz", ".json")
            if not os.path.exists(json_path):
                continue
            payload = json.loads(Path(json_path).read_text())
            onset = payload.get("scientific_onset_ms")
            if onset is None:
                continue
            try:
                state = _onset_state(path, float(onset))
            except ValueError:
                continue
            if state is None:
                continue
            family = ("dynamic trajectory (this round)"
                      if payload.get("status") == "SPATIAL_ZM_OU_TRANSITION_COMPLETE"
                      else "dynamic trajectory (archived)")
            rows.append({"family": family,
                         "q": (payload.get("hybrid_config") or {}).get("q_min"),
                         "dominant_hz": state["dominant_hz"],
                         "depth": state["modulation_depth"],
                         "mean_rate_hz": state["mean_rate_hz"]})
    return rows


MARKERS = {
    "frozen q axis": ("o", 30),
    "frozen q + adaptation": ("s", 26),
    "dynamic trajectory (archived)": ("^", 26),
    "dynamic trajectory (this round)": ("D", 40),
}


def render(rows, waveforms, q_axis):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.7,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(7.2, 5.6))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.88),
                            width_ratios=(1.20, 1.0), hspace=0.46, wspace=0.42,
                            left=0.095, right=0.905, bottom=0.085, top=0.855)
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    # --- a: does anything reach the qualifying quadrant? ---------------------
    # Only the quadrant is shaded, so no other region can read as qualifying.
    ax_a.add_patch(plt.Rectangle(
        (TARGET_HZ[0], MODULATION_DEPTH_FLOOR),
        TARGET_HZ[1] - TARGET_HZ[0], 1.06 - MODULATION_DEPTH_FLOOR,
        facecolor=QUAD, edgecolor="#4E8C3A", lw=1.0, zorder=1))
    ax_a.axhline(MODULATION_DEPTH_FLOOR, color="#4E8C3A", lw=0.7, ls="--",
                 zorder=2)
    for family, (marker, size) in MARKERS.items():
        subset = [row for row in rows if row["family"] == family]
        if not subset:
            continue
        high = np.array([row["mean_rate_hz"] >= HIGH_STATE_RATE_HZ
                         for row in subset])
        x = np.array([row["dominant_hz"] for row in subset])
        y = np.array([row["depth"] for row in subset])
        if np.any(high):
            ax_a.scatter(x[high], y[high], marker=marker, s=size,
                         facecolor=HIGHC, edgecolor="none", alpha=0.80,
                         zorder=5)
        if np.any(~high):
            ax_a.scatter(x[~high], y[~high], marker=marker, s=size,
                         facecolor="none", edgecolor=LOWC, linewidths=0.9,
                         zorder=4)
    ax_a.set_xlim(16.0, 55.0)
    ax_a.set_ylim(-0.02, 1.06)
    ax_a.set_xlabel("dominant population-rate frequency (Hz)")
    ax_a.set_ylabel("firing modulation depth\n(cycle peak-to-trough / mean)")
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.text(0.5 * sum(TARGET_HZ) - 2.0, 1.00,
              "qualifying region:\nfilled markers only",
              ha="center", va="top", fontsize=6.6, color="#3C6B2C", zorder=6)
    shape_handles = [plt.Line2D([], [], marker=MARKERS[name][0], ls="none",
                                markerfacecolor="0.45", markeredgecolor="0.45",
                                color="0.45", markersize=4.2, label=name)
                     for name in MARKERS if any(r["family"] == name for r in rows)]
    fill_handles = [
        plt.Line2D([], [], marker="o", ls="none", markerfacecolor=HIGHC,
                   markeredgecolor="none", markersize=4.6,
                   label="high-activity state (>=120 Hz)"),
        plt.Line2D([], [], marker="o", ls="none", markerfacecolor="none",
                   markeredgecolor=LOWC, markersize=4.6,
                   label="low-activity state (<120 Hz)")]
    ax_a.legend(handles=shape_handles + fill_handles, loc="upper left",
                frameon=False, fontsize=6.2, handletextpad=0.35,
                borderpad=0.15, labelspacing=0.26)

    # --- b: the same two states as waveforms --------------------------------
    for label, profile, mean_rate, colour, style in waveforms:
        phase = np.linspace(-np.pi, np.pi, len(profile), endpoint=False)
        ax_b.plot(np.degrees(phase), np.asarray(profile) / mean_rate,
                  color=colour, lw=1.4, ls=style, label=label)
    ax_b.axhline(1.0, color="0.7", lw=0.6, ls=":")
    ax_b.set_xlim(-180, 180)
    ax_b.set_xticks([-180, -90, 0, 90, 180])
    ax_b.set_xlabel("phase within one cycle (deg)")
    ax_b.set_ylabel("population rate / its own mean")
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.legend(loc="lower left", frameon=False, fontsize=6.0,
                handlelength=1.5, labelspacing=0.22, borderpad=0.1,
                handletextpad=0.4)

    # --- c: the trade-off along the control parameter ------------------------
    q = np.array([item["q"] for item in q_axis], float)
    order = np.argsort(-q)
    q = q[order]
    rate = np.array([item["mean_rate_hz"] for item in q_axis], float)[order]
    depth = np.array([item["depth"] for item in q_axis], float)[order]
    in_band = np.array([TARGET_HZ[0] <= item["dominant_hz"] <= TARGET_HZ[1]
                        for item in q_axis], bool)[order]
    rate_ax = ax_c.twinx()
    rate_ax.plot(q, rate, color=LOWC, lw=1.0, ls="--", zorder=2)
    rate_ax.axhline(HIGH_STATE_RATE_HZ, color=LOWC, lw=0.7, ls=":", zorder=1)
    rate_ax.set_ylabel("population rate (Hz)", color=LOWC, labelpad=2.0)
    rate_ax.tick_params(axis="y", colors=LOWC, pad=1.5)
    rate_ax.spines[["top"]].set_visible(False)
    ax_c.plot(q, depth, color=INK, lw=1.1, zorder=3)
    ax_c.scatter(q[in_band], depth[in_band], s=32, facecolor=HIGHC,
                 edgecolor="none", zorder=4, label="rhythm in 30-80 Hz")
    ax_c.scatter(q[~in_band], depth[~in_band], s=28, facecolor="white",
                 edgecolor=INK, lw=0.8, zorder=4, label="rhythm below 30 Hz")
    ax_c.axhline(MODULATION_DEPTH_FLOOR, color="#4E8C3A", lw=0.9, ls="--",
                 zorder=2)
    ax_c.set_xlim(q.max() + 0.015, q.min() - 0.015)
    ax_c.set_ylim(0, max(0.72, float(depth.max()) * 1.20))
    ax_c.set_xlabel("remaining inhibitory efficacy $q$")
    ax_c.set_ylabel("firing modulation depth")
    ax_c.spines[["top"]].set_visible(False)
    ax_c.set_zorder(rate_ax.get_zorder() + 1)
    ax_c.patch.set_visible(False)
    ax_c.text(q.max() + 0.008, MODULATION_DEPTH_FLOOR + 0.015,
              "criterion-10 floor", fontsize=6.2, color="#3C6B2C",
              ha="left", va="bottom")
    ax_c.legend(loc="upper right", frameon=False, fontsize=6.2,
                handletextpad=0.3, labelspacing=0.25, borderpad=0.15)

    for axis, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c")):
        axis.text(-0.075, 1.12, letter, transform=axis.transAxes,
                  fontsize=11, fontweight="bold", va="top", ha="right")
    fig.text(0.095, 0.985,
             "A 30-80 Hz rhythm and a deeply modulated one",
             fontsize=9.6, fontweight="bold", ha="left", va="top")
    fig.text(0.095, 0.945,
             "are mutually exclusive on this frozen substrate",
             fontsize=9.6, fontweight="bold", ha="left", va="top")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=(
        "results/topic4_sef_hfo/data_driven_zm_ictal_transition/"
        "spatial_zm_ou/figures"))
    args = parser.parse_args()
    locator = ("results/topic4_sef_hfo/data_driven_zm_ictal_transition/"
               "spatial_zm_ou/stage_b_q_locator")
    atlas = ("results/topic4_sef_hfo/data_driven_zm_ictal_transition/"
             "spatial_zqim_hybrid")
    frozen_q_axis = [(f"{atlas}/frozen_q_atlas/seed1801_q1000.npz", 1.000)]
    frozen_q_axis += [(f"{locator}/seed1801_q{int(round(q * 1000)):04d}.npz", q)
                      for q in (0.95, 0.925, 0.90, 0.875, 0.85, 0.825, 0.80, 0.775)]
    frozen_q_axis += [(f"{atlas}/frozen_q_atlas/seed1801_q0750.npz", 0.750),
                      (f"{atlas}/frozen_q_atlas/seed1801_q0700.npz", 0.700),
                      (f"{atlas}/frozen_q_atlas/seed1801_q0600.npz", 0.600)]
    paths_config = {
        "frozen_q_axis": frozen_q_axis,
        "frozen_gk_globs": [f"{atlas}/frozen_q*gk*atlas/*.npz"],
        "dynamic_globs": [
            f"{atlas}/discovery/**/*.npz", f"{atlas}/dynamic_*/*.npz",
            ("results/topic4_sef_hfo/data_driven_zm_ictal_transition/"
             "spatial_zm_ou/stage_b_dynamic/*.npz")],
    }
    rows = collect(paths_config)
    q_axis = [row for row in rows if row["family"] == "frozen q axis"]

    low = _late_state(ROOT / frozen_q_axis[0][0])
    high = _late_state(ROOT / f"{locator}/seed1801_q0775.npz")
    waveforms = [
        (f"interictal low state · {low['dominant_hz']:.0f} Hz · "
         f"depth {low['modulation_depth']:.2f}",
         low["cycle_profile_hz"], low["mean_rate_hz"], LOWC, "-"),
        (f"best high state · {high['dominant_hz']:.0f} Hz · "
         f"depth {high['modulation_depth']:.2f}",
         high["cycle_profile_hz"], high["mean_rate_hz"], HIGHC, "-"),
    ]
    fig = render(rows, waveforms, q_axis)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "spatial_zm_frequency_vs_modulation_depth"
    outputs = {}
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, facecolor="white")
        outputs[suffix] = path
    plt.close(fig)
    qualifying = [row for row in rows
                  if TARGET_HZ[0] <= row["dominant_hz"] <= TARGET_HZ[1]
                  and row["depth"] >= MODULATION_DEPTH_FLOOR
                  and row["mean_rate_hz"] >= HIGH_STATE_RATE_HZ]
    metadata = {
        "status": "SPATIAL_ZM_FREQUENCY_DEPTH_FIGURE_RENDERED",
        "figure_role": "archive results diagnostic, not a paper-ready Fig5A",
        "n_states": len(rows),
        "n_states_by_family": {name: sum(row["family"] == name for row in rows)
                               for name in MARKERS},
        "criterion10_depth_floor": MODULATION_DEPTH_FLOOR,
        "target_hz": list(TARGET_HZ),
        "high_state_rate_floor_hz": HIGH_STATE_RATE_HZ,
        "n_states_in_qualifying_quadrant": len(qualifying),
        "qualifying_states": qualifying,
        "max_depth_among_in_band_high_states": max(
            [row["depth"] for row in rows
             if TARGET_HZ[0] <= row["dominant_hz"] <= TARGET_HZ[1]
             and row["mean_rate_hz"] >= HIGH_STATE_RATE_HZ] or [None]),
        "records": rows,
        "outputs": {suffix: {"path": str(path.relative_to(ROOT)),
                             "sha256": _sha256(path)}
                    for suffix, path in outputs.items()},
        "claim_boundary": (
            "model-state morphology on one frozen patient-derived scaffold; "
            "not clinical seizure reproduction or mechanism identification"),
    }
    (out_dir / "spatial_zm_frequency_vs_modulation_depth_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in metadata.items()
                      if key not in {"records", "qualifying_states"}},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
