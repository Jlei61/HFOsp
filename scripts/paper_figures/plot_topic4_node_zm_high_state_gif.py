"""Render the Node-field Z/M transition with neuron-level activity frames."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, sosfiltfilt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORD = ROOT / (
    "results/topic4_sef_hfo/data_driven_zm_ictal_transition/"
    "node_zm_manual_boundary/default_seed1801_visual"
)
MONTAGE_SOURCE = Path("/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/") / (
    "data_driven_local_connectivity_rev11_nlc/frozen_substrate_confirmation/"
    "workers/node_baseline_seed_1569.npz"
)
DIRECT_METADATA = Path("/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/") / (
    "data_driven_local_connectivity_rev11_nlc/node_baseline_visual_acceptance/"
    "figures/fig4a_nlc_direct_readout_metadata.json"
)
DEFAULT_OUTPUT = ROOT / (
    "results/topic4_sef_hfo/data_driven_zm_ictal_transition/"
    "node_zm_manual_boundary/figures/node_field_zm_high_state_seed1801.gif"
)

SHAFT_COLORS = {"ICL": "#E67E22", "SCL": "#159EAE"}
ONSET_COLOR = "#D62745"
STATE_COLOR = "#F7E9ED"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bandpass(values: np.ndarray, dt_ms: float) -> np.ndarray:
    fs_hz = 1000.0 / float(dt_ms)
    sos = butter(3, (30.0, 80.0), btype="bandpass", fs=fs_hz, output="sos")
    return sosfiltfilt(sos, np.asarray(values, float), axis=0)


def _contact_order(names: np.ndarray, direct: dict) -> np.ndarray:
    accepted = direct["direct_readout"]["contact_order"]
    return np.asarray([int(np.flatnonzero(names == name)[0]) for name in accepted])


def render(record: Path, output: Path) -> dict:
    meta_path = record.with_suffix(".json")
    npz_path = record.with_suffix(".npz")
    meta = json.loads(meta_path.read_text())
    if meta.get("operational_onset_ms") is None:
        raise RuntimeError("record did not enter the operational high state")
    morphology = meta.get("runaway_morphology")
    if not isinstance(morphology, dict):
        raise RuntimeError("record has no high-state morphology audit")
    onset_ms = float(morphology["scientific_onset_ms"])

    with np.load(npz_path, allow_pickle=False) as loaded:
        frame_time = np.asarray(loaded["spatial_frame_time_ms"], float)
        fields = np.asarray(loaded["spatial_spike_count_20ms"], float)
        lfp = np.asarray(loaded["lfp_trace"], float)
        lfp_dt = float(loaded["lfp_dt_ms"])
        names = np.asarray(loaded["contact_names"]).astype(str)
        shafts = np.asarray(loaded["shaft_ids"]).astype(str)
    with np.load(MONTAGE_SOURCE, allow_pickle=False) as loaded:
        montage_names = np.asarray(loaded["contact_names"]).astype(str)
        montage_xy = np.asarray(loaded["contact_xy_mm"], float)
    contact_xy = np.asarray([
        montage_xy[int(np.flatnonzero(montage_names == name)[0])]
        for name in names
    ])
    direct = json.loads(DIRECT_METADATA.read_text())
    order = _contact_order(names, direct)

    start_ms = max(0.0, onset_ms - 600.0)
    stop_ms = min(float(frame_time[-1]), onset_ms + 1400.0)
    frame_mask = (frame_time >= start_ms) & (frame_time <= stop_ms)
    shown_frames = np.asarray([
        gaussian_filter(frame, sigma=0.5) for frame in fields[frame_mask]
    ])
    shown_frame_time = frame_time[frame_mask] - start_ms
    positive = shown_frames[shown_frames > 0]
    vmax = max(1.0, float(np.percentile(positive, 98.0)))

    filtered = _bandpass(lfp, lfp_dt)
    lfp_time = np.arange(filtered.shape[0]) * lfp_dt
    trace_mask = (lfp_time >= start_ms) & (lfp_time <= stop_ms)
    traces = filtered[trace_mask][:, order]
    trace_time = lfp_time[trace_mask] - start_ms
    scale = float(np.percentile(np.abs(traces), 99.5))
    traces = traces / max(scale, 1e-12)
    names, shafts, contact_xy = names[order], shafts[order], contact_xy[order]

    fig = plt.figure(figsize=(12.8, 5.0), facecolor="white")
    grid = fig.add_gridspec(
        1, 2, width_ratios=(0.92, 1.72), left=0.055, right=0.985,
        bottom=0.12, top=0.89, wspace=0.18,
    )
    ax_field = fig.add_subplot(grid[0, 0])
    ax_trace = fig.add_subplot(grid[0, 1])
    image = ax_field.imshow(
        shown_frames[0], origin="lower", extent=(0, 20, 0, 20),
        cmap="viridis", vmin=0.0, vmax=vmax, interpolation="bilinear",
    )
    for shaft in ("ICL", "SCL"):
        mask = shafts == shaft
        ax_field.scatter(
            contact_xy[mask, 0], contact_xy[mask, 1], s=34,
            c=SHAFT_COLORS[shaft], edgecolors="white", linewidths=0.8,
            zorder=4,
        )
    for name, (x, y) in zip(names, contact_xy):
        ax_field.text(x + 0.22, y + 0.18, name, fontsize=6.1, color="white")
    ax_field.set(
        xlim=(0, 20), ylim=(0, 20), xlabel="x (mm)", ylabel="y (mm)",
        title="2D SNN activity",
    )
    ax_field.set_aspect("equal")
    ax_field.title.set_fontweight("bold")
    ax_field.spines[["top", "right"]].set_visible(False)
    colorbar = fig.colorbar(image, ax=ax_field, fraction=0.046, pad=0.025)
    colorbar.set_label("E spikes / 20 ms bin", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    time_text = ax_field.text(
        0.02, 0.02, "", transform=ax_field.transAxes, fontsize=8.5,
        color="white", weight="bold", ha="left", va="bottom",
    )

    onset_relative = onset_ms - start_ms
    ax_trace.axvspan(onset_relative, stop_ms - start_ms,
                    color=STATE_COLOR, lw=0, zorder=0)
    ax_trace.axvline(onset_relative, color=ONSET_COLOR, lw=1.0, ls="--")
    offsets = np.arange(len(names))[::-1] * 1.55
    for row, offset in enumerate(offsets):
        ax_trace.plot(
            trace_time, 0.62 * traces[:, row] + offset,
            color=SHAFT_COLORS[shafts[row]], lw=1.0,
        )
    cursor = ax_trace.axvline(0.0, color="#202020", lw=1.1)
    ax_trace.set_yticks(offsets, names, fontsize=7.2)
    ax_trace.tick_params(axis="y", length=0, pad=4)
    ax_trace.set(
        xlim=(trace_time[0], trace_time[-1]),
        ylim=(-1.0, offsets[0] + 1.0),
        xlabel="Time in displayed window (ms)",
        ylabel="30-80 Hz virtual-contact activity",
    )
    ax_trace.spines[["top", "right", "left"]].set_visible(False)
    ax_trace.legend(
        handles=[Patch(facecolor=STATE_COLOR, edgecolor="none",
                       label="operational high state")],
        loc="upper right", frameon=False, fontsize=8,
    )
    fig.suptitle(
        "Frozen Node field + Z/M | learned EE and E-to-I redistribution off",
        x=0.055, ha="left", fontsize=10.5, fontweight="bold",
    )

    def update(frame_index: int):
        relative_time = float(shown_frame_time[frame_index])
        image.set_data(shown_frames[frame_index])
        cursor.set_xdata([relative_time, relative_time])
        time_text.set_text(f"t = {relative_time:.0f} ms")
        return image, cursor, time_text

    output.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(
        fig, update, frames=len(shown_frames), interval=80, blit=False,
    )
    animation.save(output, writer=PillowWriter(fps=12.5), dpi=115)
    plt.close(fig)

    metadata = {
        "status": "NODE_FIELD_ZM_HIGH_STATE_VISUAL_REPLAY",
        "candidate": "Node field; EE/E-to-I redistribution off",
        "seed": int(meta["seed"]),
        "zm_parameters": meta["parameters"],
        "scientific_onset_ms": onset_ms,
        "display_window_ms": [start_ms, stop_ms],
        "activity_field": "20 ms E-spike counts, 40x40 grid, sigma=0.5",
        "activity_vmax_q98": vmax,
        "right_panel": "30-80 Hz model-current proxy from the same run",
        "morphology_verdict": meta["verdict"],
        "not_clinical_seeg": True,
        "source_json": str(meta_path),
        "source_json_sha256": _sha256(meta_path),
        "source_npz": str(npz_path),
        "source_npz_sha256": _sha256(npz_path),
        "output_gif": str(output),
        "output_gif_sha256": _sha256(output),
    }
    metadata_path = output.with_name(output.stem + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    (output.parent / "README.md").write_text(
        f"""### {output.name}

固定已经验收的 data-driven Node field，关闭 learned EE 和 E-to-I 重分配，仅打开 Z/M。左侧为每 20 ms 的 E 神经元放电密度，右侧为同一条连续轨迹的 15 触点 30--80 Hz model-current proxy；红虚线是 operational transition，浅色背景是其后的高活动段。

**关注点**：进入后是否持续出现全场广泛招募和稳定振荡；这是一项模型状态可视检查，不是临床 SEEG 或患者发作判定。
"""
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(render(args.record.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
