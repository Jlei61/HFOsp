#!/usr/bin/env python3
"""Render the full-SNN fast-inhibition probe as a diagnostic, not Fig. 5."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_dual_core_oscillation_phase import contact_oscillation_assay


DEFAULT_JSON = Path(
    "/data/hfosp_topic4_fig45_artifacts/fig5/"
    "data_driven_dual_core_spatial_z/oscillatory_snn_probe/"
    "rev21_si_0p7_sm_0p5_t2542_d2641_tauGABA8.json")
ONSET = "#C7254E"
RATE = "#333333"
TRACE = "#E87942"
ZCORE = "#2F5D95"
ZCORE_B = "#76519B"
ZSUR = "#86A8C5"
ADAPT = "#D76B38"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    try:
        Path(temporary).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _label(axis: plt.Axes, text: str, *, x=-0.10, y=1.08) -> None:
    axis.text(x, y, text, transform=axis.transAxes, fontsize=15,
              fontweight="bold", ha="left", va="top")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()
    result_path = args.result.resolve()
    meta = json.loads(result_path.read_text())
    arrays_path = Path(meta["arrays"]["path"])
    data = np.load(arrays_path)
    time_ms = np.asarray(data["time_ms"], float)
    time_s = time_ms / 1000.0
    onset_value = meta["operational_runaway_onset_ms"]
    onset_ms = (None if onset_value is None else float(onset_value))
    if onset_ms is None:
        raise RuntimeError(
            "this diagnostic requires an operational runaway onset; the probe has none")
    onset_s = onset_ms / 1000.0
    rate = np.asarray(data["population_rate_E_hz_smooth_2ms"], float)
    lfp = np.asarray(data["virtual_seeg"], float)
    dt_ms = float(np.median(np.diff(time_ms)))
    fs = 1000.0 / dt_ms
    filtered = sosfiltfilt(
        butter(4, (30.0, 80.0), btype="bandpass", fs=fs, output="sos"),
        lfp, axis=0)
    pre = time_ms < onset_ms
    scale = float(np.percentile(np.abs(filtered[pre]), 99.0))
    filtered = filtered / max(scale, 1e-12)
    if "contact_names" in data.files:
        contact_names = np.asarray(data["contact_names"]).astype(str)
    else:
        contact_names = np.asarray([
            "SCL9", "ICL11", "SCL8", "ICL10", "SCL7", "ICL9", "ICL8",
            "SCL6", "ICL7", "ICL6", "ICL5", "ICL4", "ICL3", "ICL2", "ICL1",
        ])
    if contact_names.size != lfp.shape[1]:
        raise RuntimeError("contact labels do not align with virtual SEEG columns")
    baseline_window = (time_ms >= 500.0) & (time_ms < 1000.0)
    early_window = ((time_ms >= onset_ms + 100.0)
                    & (time_ms < min(time_ms[-1], onset_ms + 1100.0)))
    terminal_window = time_ms >= time_ms[-1] - 1000.0
    early_contact_state = contact_oscillation_assay(
        lfp[baseline_window], lfp[early_window], dt_ms=dt_ms)
    contact_state = contact_oscillation_assay(
        lfp[baseline_window], lfp[terminal_window], dt_ms=dt_ms)
    passes = np.asarray(contact_state["persistent_contact_mask"], bool)
    shaft_summary = {}
    for shaft in ("ICL", "SCL"):
        selected_shaft = np.char.startswith(contact_names, shaft)
        shaft_summary[shaft] = {
            "n_contacts": int(np.sum(selected_shaft)),
            "n_passing": int(np.sum(passes[selected_shaft])),
        }
    coverage_pass = bool(
        np.mean(passes) >= 0.8
        and all(item["n_passing"] >= 0.5 * item["n_contacts"]
                for item in shaft_summary.values()))
    peak_hz = float(meta["state_assay"]["whole_tail"]["dominant_hz"])

    fig = plt.figure(figsize=(13.0, 8.0))
    outer = fig.add_gridspec(2, 2, left=0.065, right=0.965, bottom=0.08,
                             top=0.92, width_ratios=[1.5, 1.0],
                             height_ratios=[1.15, 1.0], wspace=0.23, hspace=0.31)
    left = outer[0, 0].subgridspec(2, 1, height_ratios=[1, 3], hspace=0.05)
    ax_rate = fig.add_subplot(left[0])
    ax_trace = fig.add_subplot(left[1], sharex=ax_rate)
    slow = outer[1, 0].subgridspec(2, 1, hspace=0.07)
    ax_z = fig.add_subplot(slow[0])
    ax_m = fig.add_subplot(slow[1], sharex=ax_z)
    ax_zoom = fig.add_subplot(outer[0, 1])
    spatial = outer[1, 1].subgridspec(1, 3, wspace=0.08)
    map_axes = [fig.add_subplot(spatial[index]) for index in range(3)]
    figure_title = (
        f"Full-SNN candidate: spatially widespread {peak_hz:.0f}-Hz readout on global runaway"
        if coverage_pass else
        "Full-SNN fast-inhibition probe: high-frequency ripple remains axis-restricted")
    fig.suptitle(
        figure_title,
        fontsize=13, fontweight="bold", y=0.975)

    for axis in (ax_rate, ax_trace, ax_z, ax_m):
        axis.axvline(onset_s, color=ONSET, lw=1.0, ls="--")
        axis.axvspan(onset_s, time_s[-1], color=ONSET, alpha=0.055, lw=0)
    ax_rate.plot(time_s, rate, color=RATE, lw=0.75)
    ax_rate.axhline(120.0, color=ONSET, lw=0.7, ls=":")
    ax_rate.set_ylabel("E rate\n(Hz)", fontsize=8)
    ax_rate.set_ylim(0, max(380, float(np.percentile(rate, 99.9)) * 1.05))
    ax_rate.set_title("A  Continuous OU + Z/M trajectory", fontsize=10,
                      fontweight="bold", loc="left")
    ax_rate.tick_params(axis="x", labelbottom=False)

    selected = np.arange(lfp.shape[1])
    offsets = np.arange(len(selected), dtype=float)
    for row, index in enumerate(selected):
        ax_trace.plot(time_s, 0.42 * filtered[:, index] + offsets[row],
                      color=(TRACE if contact_names[index].startswith("ICL")
                             else "#2EA3B0"), lw=0.48,
                      rasterized=True)
    ax_trace.set_yticks(offsets)
    ax_trace.set_yticklabels(contact_names[selected], fontsize=6)
    ax_trace.set_ylabel("virtual SEEG\n30–80 Hz", fontsize=8)
    ax_trace.set_xlabel("time in one uninterrupted SNN trajectory (s)")
    ax_trace.set_xlim(time_s[0], time_s[-1])

    slow_time = np.asarray(data["slow_time_ms"], float) / 1000.0
    has_regions = "slow_region_core_a_z_mean" in data.files
    if has_regions:
        region_time = np.asarray(data["slow_region_time_ms"], float) / 1000.0
        ax_z.plot(region_time, data["slow_region_core_a_z_mean"],
                  color=ZCORE, lw=1.0, label="core A")
        ax_z.plot(region_time, data["slow_region_core_b_z_mean"],
                  color=ZCORE_B, lw=0.95, label="core B")
        ax_z.plot(region_time, data["slow_region_surround_z_mean"],
                  color=ZSUR, lw=0.9, label="surround")
    else:
        ax_z.plot(slow_time, data["slow_z_core_mean"], color=ZCORE, lw=1.0,
                  label="core union")
        ax_z.plot(slow_time, data["slow_z_surround_mean"], color=ZSUR, lw=0.9,
                  label="surround")
    ax_z.set_ylabel("inhibitory\nefficacy Z", fontsize=8)
    ax_z.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    ax_z.tick_params(axis="x", labelbottom=False)
    eta_m = 0.003725797177793549
    if has_regions:
        ax_m.plot(
            region_time, eta_m * np.asarray(data["slow_region_core_a_m_mean"], float),
            color=ADAPT, lw=0.95, label="core A")
        ax_m.plot(
            region_time, eta_m * np.asarray(data["slow_region_core_b_m_mean"], float),
            color=ZCORE_B, lw=0.9, label="core B")
        ax_m.plot(
            region_time, eta_m * np.asarray(data["slow_region_surround_m_mean"], float),
            color="#B6A38B", lw=0.8, label="surround")
        ax_m.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper left")
    else:
        adaptation = eta_m * np.asarray(data["slow_m_core_mean"], float)
        ax_m.plot(slow_time, adaptation, color=ADAPT, lw=0.95)
    ax_m.set_ylabel(r"regional $\eta_m m$", fontsize=8)
    ax_m.set_xlabel("time on the same axis (s)")
    ax_m.set_xlim(time_s[0], time_s[-1])
    _label(ax_z, "B", x=-0.10, y=1.10)

    zoom = time_ms >= time_ms[-1] - 1000.0
    ax_zoom.plot(time_ms[zoom] - time_ms[zoom][0], rate[zoom], color=TRACE,
                 lw=0.85)
    state = meta["state_assay"]
    ax_zoom.text(
        0.03, 0.97,
        f"mean = {state['whole_tail']['mean_rate_hz']:.0f} Hz\n"
        f"peak = {state['whole_tail']['dominant_hz']:.0f} Hz\n"
        f"modulation depth = {state['whole_tail']['modulation_depth']:.3f}\n"
        f"passing windows = {state['passing_windows']}/4",
        transform=ax_zoom.transAxes, ha="left", va="top", fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.88,
              "boxstyle": "round,pad=0.35"})
    ax_zoom.text(
        0.97, 0.97,
        f"persistent contact pass: {contact_state['n_persistent_contacts']}/{lfp.shape[1]}\n"
        f"ICL {shaft_summary['ICL']['n_passing']}/{shaft_summary['ICL']['n_contacts']}  |  "
        f"SCL {shaft_summary['SCL']['n_passing']}/{shaft_summary['SCL']['n_contacts']}\n"
        f"early-recruitment pass: {early_contact_state['n_persistent_contacts']}/{lfp.shape[1]}",
        transform=ax_zoom.transAxes, ha="right", va="top", fontsize=8,
        bbox={"facecolor": "white", "edgecolor": ONSET, "alpha": 0.88,
              "boxstyle": "round,pad=0.35"})
    ax_zoom.set_xlabel("last 1 s (ms)")
    ax_zoom.set_ylabel("2-ms-smoothed E rate (Hz)")
    ax_zoom.set_title(
        ("C  Global high rate with widespread contact rhythm"
         if coverage_pass else
         "C  Global high rate, shallow and spatially incomplete rhythm"),
                      fontsize=10, fontweight="bold", loc="left")
    ax_zoom.set_xlim(0, 1000)

    frames = np.asarray(data["spatial_spike_count_20ms_1mm"], float)
    frame_time = np.asarray(data["spatial_frame_time_ms"], float)
    windows = [(500.0, 1000.0), (onset_ms - 500.0, onset_ms),
               (onset_ms + 500.0, onset_ms + 1000.0)]
    maps = []
    for lo, hi in windows:
        keep = (frame_time >= lo) & (frame_time < hi)
        maps.append(np.log10(1.0 + np.mean(frames[keep], axis=0)))
    vmax = float(np.percentile(np.concatenate([item.ravel() for item in maps]), 99.5))
    titles = ("baseline", "pre-onset", "late high state")
    image = None
    for index, (axis, field, title) in enumerate(zip(map_axes, maps, titles)):
        image = axis.imshow(field, origin="lower", extent=(0, 20, 0, 20),
                            cmap="magma", vmin=0, vmax=vmax,
                            interpolation="nearest")
        axis.set_title(title, fontsize=8, fontweight="bold")
        axis.set_xticks([0, 10, 20])
        axis.set_yticks([0, 10, 20])
        axis.set_xlabel("x (mm)", fontsize=7)
        if index == 0:
            axis.set_ylabel("y (mm)", fontsize=7)
            _label(axis, "D", x=-0.35, y=1.14)
        else:
            axis.set_yticklabels([])
        axis.tick_params(labelsize=6)
    colorbar = fig.colorbar(image, ax=map_axes, fraction=0.035, pad=0.025)
    colorbar.set_label(r"log$_{10}$(1 + spikes / 20 ms / bin)", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)

    for axis in (ax_rate, ax_trace, ax_z, ax_m, ax_zoom):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7)
    out_dir = result_path.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    tau_label = f"{float(meta['tau_d_GABA_ms']):g}".replace(".", "p")
    candidate_label = str(meta["candidate_id"]).replace(".", "p")
    if candidate_label == "rev21_si_0p7_sm_0p5":
        filename = f"dualcore-fast-gaba-full-snn-diagnostic-tauGABA{tau_label}"
    else:
        filename = (
            f"dualcore-{candidate_label}-full-snn-diagnostic-"
            f"tauGABA{tau_label}")
    stem = out_dir / filename
    outputs = {}
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        path = stem.with_suffix("." + suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs[suffix] = {"path": str(path), "sha256": _sha256(path)}
    plt.close(fig)
    audit = {
        "status": (
            "FULL_SNN_GLOBAL_RATE_WIDESPREAD_CONTACT_OSCILLATION"
            if coverage_pass else
            "FULL_SNN_GLOBAL_RATE_AXIS_RESTRICTED_OSCILLATION"),
        "source": {"path": str(result_path), "sha256": _sha256(result_path)},
        "arrays": {"path": str(arrays_path), "sha256": _sha256(arrays_path)},
        "outputs": outputs,
        "panel_semantics": {
            "A": "one uninterrupted OU-driven Z/M-on 40,000-cell SNN trajectory",
            "B": "Z and adaptation on the identical time axis",
            "C": "last-1-s population rate and preregistered state metrics",
            "D": "20-ms spike-count maps from exact windows of the same run",
        },
        "contact_oscillation_assay": contact_state,
        "early_recruitment_contact_oscillation_assay": early_contact_state,
        "shaft_summary": shaft_summary,
        "contact_spatial_coverage_pass": coverage_pass,
        "claim_boundary": (
            ("Development-only panel-A candidate under the author's relaxed "
             "tonic-versus-deep-modulation semantics. It has global regional "
             "recruitment and widespread persistent terminal 30-80-Hz contact "
             "readout; early recruitment is reported separately. It remains "
             "a single-seed retuned-GABA sensitivity, not rev21 confirmation."
             if coverage_pass else
             "Diagnostic only. Faster GABA raises the ICL-axis ripple into the "
             "target band. Population modulation is reported descriptively under "
             "the author's relaxed tonic-versus-deep criterion, but the SCL "
             "fine-window persistence gate fails; this result cannot yet supply "
             "paper Figure 5A.")),
    }
    _atomic_json(stem.with_suffix(".metadata.json"), audit)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
