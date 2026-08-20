#!/usr/bin/env python3
"""Render the frozen target-informed Z/M bridge candidate and movie."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_fig5_target_informed_bridge import exact_contact_reorder  # noqa: E402


ORANGE = "#E97932"
CYAN = "#2E9DA5"
RED = "#B5222E"
BLUE = "#2C6CA3"
INK = "#252525"
LIGHT = "#E8E8E8"


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _display_record(confirmation):
    eligible = [row for row in confirmation["records"]
                if row["status"] == "BRIDGE_EVALUABLE"]
    if not eligible:
        raise RuntimeError("confirmation has no bridge-evaluable trajectory")
    median = float(np.median([row["J_bridge_without_time"] for row in eligible]))
    return min(eligible, key=lambda row: (
        abs(row["J_bridge_without_time"] - median), row["seed"]))


def _filter(trace, dt_ms, band=(30.0, 80.0)):
    fs = 1000.0 / float(dt_ms)
    sos = butter(3, band, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(trace, float), axis=0)


def _montage_xy(candidate_names):
    source = ROOT / (
        "results/topic4_sef_hfo/data_driven_zm_ictal_transition/workers/"
        "joint_04_control_seed_1801_zmoff.npz")
    with np.load(source, allow_pickle=False) as data:
        names = data["contact_names"].astype(str)
        xy = np.asarray(data["contact_xy_mm"], float)
    return exact_contact_reorder(xy.T, names, candidate_names).T


def _stacked_readout(ax, trace, dt_ms, names, start_ms, stop_ms, onset_ms,
                     read_window):
    filtered = _filter(trace, dt_ms)
    lo = max(0, int(round(start_ms / dt_ms)))
    hi = min(len(filtered), int(round(stop_ms / dt_ms)))
    segment = filtered[lo:hi]
    time_ms = np.arange(lo, hi) * dt_ms - onset_ms
    scale = float(np.quantile(np.abs(segment), 0.995))
    scale = max(scale, 1e-12)
    offsets = np.arange(len(names))[::-1] * 2.6
    for index, name in enumerate(names):
        color = CYAN if name.startswith("SCL") else ORANGE
        ax.plot(time_ms, segment[:, index] / scale + offsets[index],
                lw=0.65, color=color)
    ax.axvline(0, color=RED, lw=0.9, ls="--")
    ax.axvspan(read_window[0] - onset_ms, read_window[1] - onset_ms,
               color="#D8E5F1", alpha=0.75, lw=0)
    ax.set_yticks(offsets)
    ax.set_yticklabels(names, fontsize=5.8)
    ax.set_xlim(time_ms[0], time_ms[-1])
    ax.set_xlabel("Time from model transition (ms)")
    ax.set_ylabel("Virtual contacts")
    ax.spines[["top", "right"]].set_visible(False)
    return scale


def _main_figure(out_dir, candidate_npz, candidate_json, record, confirmation,
                 target, target_npz, null_payload):
    with np.load(candidate_npz, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    names = arrays["contact_names"].astype(str)
    onset = float(candidate_json["operational_onset_ms"]) - 100.0
    read = record["readout_window"]
    model = np.asarray(record["model_early_robust_z"], float)
    patient = np.asarray(target["early"]["median"], float)
    qlo = np.asarray(target["early"]["q025"], float)
    qhi = np.asarray(target["early"]["q975"], float)
    target_names = target_npz["contact_names"].astype(str)
    shafts = target_npz["shaft_ids"].astype(str)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.0,
        "axes.linewidth": 0.7, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(7.25, 6.45), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.86),
                            width_ratios=(1.55, 0.95, 1.0))
    ax_a = fig.add_subplot(grid[0, :2])
    scale = _stacked_readout(
        ax_a, arrays["lfp_trace"], float(arrays["lfp_dt_ms"]), names,
        max(0.0, onset - 600.0), min(len(arrays["lfp_trace"]) * float(arrays["lfp_dt_ms"]),
                                      onset + 1400.0), onset,
        (float(read["start_ms"]), float(read["stop_ms"])))
    ax_a.set_title("Data-driven Z/M transition", loc="left", fontweight="bold")

    ax_b = fig.add_subplot(grid[0, 2])
    z = arrays.get("slow_z_core_mean", arrays.get("slow_z_mean"))
    m = arrays.get("slow_m_core_mean", arrays.get("slow_m_mean"))
    slow_t = arrays["slow_time_ms"]
    eta = float(candidate_json["parameters"]["eta_m"])
    use = np.isfinite(z) & np.isfinite(m)
    color = slow_t[use] - onset
    ax_b.plot(1.0 - z[use], eta * m[use], color="#CCD3DB", lw=0.6)
    scatter = ax_b.scatter(1.0 - z[use], eta * m[use], c=color,
                           cmap="coolwarm", s=4, linewidths=0)
    ax_b.set_xlabel("Disinhibition 1 - z")
    ax_b.set_ylabel("M current (a.u.)")
    ax_b.set_title("Projected Z/M trajectory", fontweight="bold")
    ax_b.spines[["top", "right"]].set_visible(False)
    cb = fig.colorbar(scatter, ax=ax_b, fraction=0.05, pad=0.03)
    cb.set_label("Time from transition (ms)")

    ax_c = fig.add_subplot(grid[1, 0])
    times = arrays["spatial_frame_time_ms"]
    frame_index = int(np.argmin(np.abs(times - 0.5 * (
        float(read["start_ms"]) + float(read["stop_ms"])))))
    image = ax_c.imshow(
        arrays["spatial_spike_count_20ms"][frame_index], origin="lower",
        extent=(0, 20, 0, 20), cmap="magma", interpolation="bilinear",
        aspect="equal")
    xy = _montage_xy(names)
    for shaft, color_value in (("ICL", ORANGE), ("SCL", CYAN)):
        use_shaft = np.char.startswith(names, shaft)
        ax_c.plot(xy[use_shaft, 0], xy[use_shaft, 1], "o-", ms=2.8,
                  lw=0.7, color=color_value, mec="white", mew=0.3)
    ax_c.set_xlabel("Sheet x (mm)")
    ax_c.set_ylabel("Sheet y (mm)")
    ax_c.set_title("Early high-state activity", fontweight="bold")
    cb = fig.colorbar(image, ax=ax_c, fraction=0.046, pad=0.03)
    cb.set_label("Spikes / 20 ms")

    ax_d = fig.add_subplot(grid[1, 1])
    x = np.arange(len(target_names))
    ax_d.fill_between(x, qlo, qhi, color="#B9CFE2", alpha=0.5, lw=0)
    ax_d.plot(x, patient, color=BLUE, lw=1.2, marker="o", ms=2.4,
              label="Patient")
    ax_d.plot(x, model, color=RED, lw=1.2, marker="o", ms=2.4,
              label="Model")
    ax_d.axvline(3.5, color="#AAAAAA", lw=0.6)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(target_names, rotation=90, fontsize=5.5)
    ax_d.set_ylabel("10-150 Hz energy (robust-z)")
    ax_d.set_title("Contact energy gradient", fontweight="bold")
    ax_d.legend(frameon=False, fontsize=6, ncol=2)
    ax_d.spines[["top", "right"]].set_visible(False)

    ax_e = fig.add_subplot(grid[1, 2])
    colors = [CYAN if shaft == "SCL" else ORANGE for shaft in shafts]
    ax_e.scatter(patient, model, c=colors, s=20, edgecolor="white", linewidth=0.4)
    lo = min(float(np.min(patient)), float(np.min(model)))
    hi = max(float(np.max(patient)), float(np.max(model)))
    ax_e.plot([lo, hi], [lo, hi], color="#999999", lw=0.7, ls="--")
    rho = float(spearmanr(patient, model).statistic)
    ax_e.text(0.04, 0.96, f"rho = {rho:.2f}", transform=ax_e.transAxes,
              ha="left", va="top")
    ax_e.set_xlabel("Patient target")
    ax_e.set_ylabel("Model")
    ax_e.set_title("Frozen confirmation", fontweight="bold")
    ax_e.spines[["top", "right"]].set_visible(False)

    for label, ax in zip("ABCDE", (ax_a, ax_b, ax_c, ax_d, ax_e)):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom")

    stem = out_dir / "fig5-target-informed-zm-bridge"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"display_seed": int(record["seed"]), "readout_window_ms": read,
            "readout_scale_a_u": scale, "early_spearman": rho,
            "spatial_frame_time_ms": float(times[frame_index]),
            "selection_null": null_payload}


def _movie(out_dir, candidate_npz, candidate_json, record):
    with np.load(candidate_npz, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    names = arrays["contact_names"].astype(str)
    onset = float(candidate_json["operational_onset_ms"]) - 100.0
    frames_t = arrays["spatial_frame_time_ms"]
    use = np.flatnonzero((frames_t >= onset - 800.0) & (frames_t <= onset + 1200.0))
    if not len(use):
        raise RuntimeError("movie interval has no spatial frames")
    trace = _filter(arrays["lfp_trace"], float(arrays["lfp_dt_ms"]))
    dt = float(arrays["lfp_dt_ms"])
    lo = max(0, int(round((onset - 800.0) / dt)))
    hi = min(len(trace), int(round((onset + 1200.0) / dt)))
    trace_t = np.arange(lo, hi) * dt - onset
    scale = max(float(np.quantile(np.abs(trace[lo:hi]), 0.995)), 1e-12)
    offsets = np.arange(len(names))[::-1] * 2.6
    z = arrays.get("slow_z_core_mean", arrays.get("slow_z_mean"))
    m = arrays.get("slow_m_core_mean", arrays.get("slow_m_mean"))
    slow_t = arrays["slow_time_ms"]
    eta = float(candidate_json["parameters"]["eta_m"])

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.0),
                             gridspec_kw={"width_ratios": [0.85, 1.0, 1.8]},
                             constrained_layout=True)
    axes[0].plot(1-z, eta*m, color="#CBD2D9", lw=0.7)
    dot, = axes[0].plot([], [], "o", color=RED, ms=4)
    axes[0].set(xlabel="1 - z", ylabel="M current", title="Z/M state")
    vmax = float(np.quantile(arrays["spatial_spike_count_20ms"][use], 0.995))
    im = axes[1].imshow(arrays["spatial_spike_count_20ms"][use[0]], origin="lower",
                        extent=(0,20,0,20), cmap="magma", vmin=0, vmax=max(vmax,1))
    axes[1].set(xlabel="Sheet x (mm)", ylabel="Sheet y (mm)", title="Population activity")
    for index, name in enumerate(names):
        color = CYAN if name.startswith("SCL") else ORANGE
        axes[2].plot(trace_t, trace[lo:hi,index]/scale+offsets[index],
                     color=color, lw=0.55)
    cursor = axes[2].axvline(frames_t[use[0]]-onset, color=RED, lw=0.9)
    axes[2].set(yticks=offsets, yticklabels=names, xlabel="Time from transition (ms)",
                title="Virtual-contact readout", xlim=(trace_t[0],trace_t[-1]))
    axes[2].tick_params(axis="y", labelsize=5.5)
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
    def update(frame_index):
        index = use[frame_index]
        time_ms = float(frames_t[index])
        slow_index = int(np.argmin(np.abs(slow_t-time_ms)))
        dot.set_data([1-z[slow_index]],[eta*m[slow_index]])
        im.set_data(arrays["spatial_spike_count_20ms"][index])
        cursor.set_xdata([time_ms-onset,time_ms-onset])
        return dot, im, cursor
    movie = animation.FuncAnimation(fig, update, frames=len(use), interval=80,
                                    blit=False)
    movie.save(out_dir / "fig5-target-informed-zm-bridge.gif",
               writer=animation.PillowWriter(fps=12.5), dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/topic4_data_driven_zm_target_informed_bridge_v1.json")
    args = parser.parse_args()
    config = _load(ROOT / args.config)
    results = ROOT / config["output_root"]
    out = ROOT / config["paper_root"]
    out.mkdir(parents=True, exist_ok=True)
    confirmation = _load(results / "confirmation_results.json")
    if confirmation["status"] != "FROZEN_CONFIRMATION_PASS":
        raise RuntimeError("a failed confirmation cannot be rendered as final Fig.5")
    record = _display_record(confirmation)
    candidate_npz = ROOT / record["source_npz"]
    candidate_json = ROOT / record["source_json"]
    candidate_payload = _load(candidate_json)
    target = _load(results / "clinical_target.json")["summaries"]["sensitivity_10_150"]
    target_npz = np.load(results / "clinical_target_vectors.npz", allow_pickle=False)
    null_payload = _load(results / "selection_aware_null.json")
    null_summary = [{key: value for key, value in row.items()
                     if key != "null_minimum_J"}
                    for row in null_payload["nulls"]]
    metadata = _main_figure(
        out, candidate_npz, candidate_payload, record, confirmation,
        target, target_npz, null_summary)
    _movie(out, candidate_npz, candidate_payload, record)
    metadata.update({
        "status": "FIG5_TARGET_INFORMED_ZM_BRIDGE_RENDERED",
        "candidate_id": confirmation["candidate_id"],
        "source_npz": record["source_npz"],
        "source_json": record["source_json"],
        "claim_boundary": "development-only target-informed model bridge",
    })
    (out / "fig5-target-informed-zm-bridge-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "### fig5-target-informed-zm-bridge.png\n"
        "展示冻结 Z/M 工作点从低活动段进入广泛高频模型状态，并比较状态定义读出窗的 "
        "10-150 Hz 触点能量与 E1146 的 24 次发作目标。阴影为患者发作间不确定性；模型读出不使用临床单位。\n\n"
        "**关注点**：转变后的全局招募、Z/M 轨迹，以及 ICL/SCL 两杆的能量梯度是否同时对齐。\n\n"
        "### fig5-target-informed-zm-bridge.gif\n"
        "同一条 confirmation 轨迹的 Z/M 状态、二维群体活动和 15 触点连续读出。红线为模型内部定义的转变时刻。\n\n"
        "**关注点**：高态必须表现为广泛且持续的兴奋，而不是仅有高事件率的间期 burst。\n",
        encoding="utf-8")
    print(json.dumps({"status": metadata["status"], "out": str(out)}))


if __name__ == "__main__":
    main()
