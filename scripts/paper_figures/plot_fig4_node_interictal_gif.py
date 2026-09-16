"""Render a zero-simulation visual check for the frozen Node-only substrate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch
from scipy.signal import butter, sosfiltfilt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = ROOT / (
    "results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc/"
    "frozen_substrate_confirmation/workers/node_baseline_seed_1569.npz"
)
DEFAULT_DIRECT_METADATA = ROOT / (
    "results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc/"
    "node_baseline_visual_acceptance/figures/"
    "fig4a_nlc_direct_readout_metadata.json"
)
DEFAULT_OUTPUT = ROOT / (
    "results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc/"
    "node_baseline_visual_acceptance/figures/"
    "node_baseline_seed1569_interictal_readout.gif"
)

SHAFT_COLORS = {"ICL": "#E67E22", "SCL": "#159EAE"}
TA_COLOR = "#C43C39"
TB_COLOR = "#277DA1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bandpass(values: np.ndarray, dt_ms: float) -> np.ndarray:
    fs_hz = 1000.0 / dt_ms
    sos = butter(3, [30.0, 80.0], btype="bandpass", fs=fs_hz, output="sos")
    return sosfiltfilt(sos, values, axis=1)


def _field_image(positions: np.ndarray, h: np.ndarray, bins: int = 100):
    weighted, x_edges, y_edges = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=bins,
        range=((0.0, 20.0), (0.0, 20.0)), weights=h,
    )
    counts, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=(x_edges, y_edges),
    )
    mean = np.divide(weighted, counts, out=np.zeros_like(weighted), where=counts > 0)
    return mean.T


def render(npz_path: Path, direct_metadata_path: Path, output_path: Path):
    direct = json.loads(direct_metadata_path.read_text())
    window_start, window_stop = direct["direct_readout"]["display_window_ms"]
    seed = int(direct["direct_readout"]["seed"])
    if seed != 1569:
        raise ValueError("default visual contract expects seed 1569")

    with np.load(npz_path, allow_pickle=False) as loaded:
        names = np.asarray(loaded["contact_names"]).astype(str)
        shafts = np.asarray(loaded["shaft_ids"]).astype(str)
        contact_xy = np.asarray(loaded["contact_xy_mm"], float)
        positions = np.asarray(loaded["positions_E"], float)
        h = np.asarray(loaded["h"], float)
        envelope = np.asarray(loaded["contact_envelope"], float)
        dt_ms = float(loaded["contact_envelope_dt_ms"])
        event_on = np.asarray(loaded["event_t_on_ms"], float)
        event_off = np.asarray(loaded["event_t_off_ms"], float)

    order_names = direct["direct_readout"]["contact_order"]
    order = np.asarray([int(np.flatnonzero(names == name)[0]) for name in order_names])
    traces = _bandpass(envelope, dt_ms)[order]
    names, shafts, contact_xy = names[order], shafts[order], contact_xy[order]
    time_ms = np.arange(envelope.shape[1], dtype=float) * dt_ms
    selected = (time_ms >= window_start) & (time_ms <= window_stop)
    rel_time = time_ms[selected] - window_start
    traces = traces[:, selected]
    scale = float(np.quantile(np.abs(traces), 0.995))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("invalid contact scale")
    traces = traces / scale

    event_indices = np.flatnonzero(
        (event_on >= window_start) & (event_on <= window_stop)
    )
    if len(event_indices) < 2:
        raise RuntimeError("visual window no longer contains the expected event pair")
    event_indices = event_indices[[0, -1]]
    event_spans = [
        (event_on[index] - window_start, event_off[index] - window_start)
        for index in event_indices
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.8, 5.0), facecolor="white")
    grid = fig.add_gridspec(
        1, 2, width_ratios=(0.92, 1.72), left=0.055, right=0.985,
        bottom=0.12, top=0.91, wspace=0.18,
    )
    ax_map = fig.add_subplot(grid[0, 0])
    ax_trace = fig.add_subplot(grid[0, 1])

    field = _field_image(positions, h)
    ax_map.imshow(
        field, origin="lower", extent=(0, 20, 0, 20), cmap="plasma",
        vmin=0.0, vmax=max(0.85, float(np.nanmax(field))),
        interpolation="bilinear", alpha=0.92,
    )
    base_colors = [SHAFT_COLORS[shaft] for shaft in shafts]
    ax_map.scatter(
        contact_xy[:, 0], contact_xy[:, 1], s=30, c=base_colors,
        edgecolors="white", linewidths=0.8, zorder=4,
    )
    dynamic = ax_map.scatter(
        contact_xy[:, 0], contact_xy[:, 1], s=np.full(len(names), 42.0),
        facecolors="none", edgecolors="#111111", linewidths=1.1, zorder=5,
    )
    for name, (x, y) in zip(names, contact_xy):
        ax_map.text(x + 0.22, y + 0.18, name, fontsize=6.1, color="#243238")
    ax_map.set(xlim=(0, 20), ylim=(0, 20), xlabel="x (mm)", ylabel="y (mm)")
    ax_map.set_title("Continuous Node field", fontsize=11, weight="bold")
    ax_map.set_aspect("equal")
    ax_map.spines[["top", "right"]].set_visible(False)

    spacing = 1.55
    offsets = np.arange(len(names))[::-1] * spacing
    for row, offset in enumerate(offsets):
        ax_trace.plot(rel_time, traces[row] * 0.62 + offset,
                      color=base_colors[row], lw=1.05)
    for (start, stop), color, label in zip(
        event_spans, (TA_COLOR, TB_COLOR), ("MTA", "MTB"),
    ):
        ax_trace.axvspan(start, stop, color=color, alpha=0.10, lw=0)
    cursor = ax_trace.axvline(rel_time[0], color="#202020", lw=1.1, alpha=0.85)
    ax_trace.set_yticks(offsets, names, fontsize=7.2)
    ax_trace.tick_params(axis="y", length=0, pad=4)
    ax_trace.set_xlim(rel_time[0], rel_time[-1])
    ax_trace.set_ylim(-1.0, offsets[0] + 1.0)
    ax_trace.set_xlabel("Time in displayed window (ms)")
    ax_trace.set_ylabel("30-80 Hz virtual-contact activity")
    ax_trace.spines[["top", "right", "left"]].set_visible(False)
    ax_trace.legend(
        handles=[Patch(facecolor=TA_COLOR, alpha=0.14, label="MTA event"),
                 Patch(facecolor=TB_COLOR, alpha=0.14, label="MTB event")],
        loc="upper right", frameon=False, ncol=2, fontsize=8,
    )
    time_text = ax_map.text(
        0.02, 0.02, "", transform=ax_map.transAxes, fontsize=8.5,
        color="#243238", ha="left", va="bottom",
    )

    frame_indices = np.arange(0, len(rel_time), max(1, int(round(10.0 / dt_ms))))

    def update(frame_number):
        index = int(frame_indices[frame_number])
        amplitude = np.clip(np.abs(traces[:, index]), 0.0, 1.5)
        dynamic.set_sizes(42.0 + 220.0 * amplitude)
        dynamic.set_linewidths(0.8 + 1.8 * amplitude)
        cursor.set_xdata([rel_time[index], rel_time[index]])
        time_text.set_text(f"t = {rel_time[index]:.0f} ms")
        return dynamic, cursor, time_text

    animation = FuncAnimation(
        fig, update, frames=len(frame_indices), interval=80, blit=False,
    )
    animation.save(output_path, writer=PillowWriter(fps=12.5), dpi=115)
    plt.close(fig)

    metadata = {
        "status": "NODE_ONLY_INTERICTAL_VISUAL_CHECK",
        "candidate_id": "node_baseline",
        "network_seed": seed,
        "simulation_rerun": False,
        "source_npz": str(npz_path),
        "source_npz_sha256": _sha256(npz_path),
        "source_direct_metadata": str(direct_metadata_path),
        "source_direct_metadata_sha256": _sha256(direct_metadata_path),
        "display_window_ms": [window_start, window_stop],
        "event_spans_relative_ms": event_spans,
        "signal": "30-80 Hz bandpass of stored virtual-contact firing-density envelope",
        "left_panel": "static h field; marker size is instantaneous contact amplitude",
        "not_saved_neuron_activity": True,
        "not_clinical_seeg": True,
        "playback_fps": 12.5,
        "biological_frame_step_ms": 10.0,
        "files": {"gif": str(output_path), "gif_sha256": _sha256(output_path)},
    }
    metadata_path = output_path.with_name(output_path.stem + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))

    readme_path = output_path.parent / "README.md"
    with readme_path.open("a") as handle:
        handle.write(f"""

### {output_path.name}

这条 GIF 使用 `node_baseline` seed 1569 已保存的同网络窗口，不重跑仿真。左侧为冻结连续 Node field，触点外圈大小表示瞬时 30--80 Hz virtual-contact firing-density envelope；右侧为同一窗口的 15 触点连续读出，游标依次经过一例 MTA 和一例 MTB。

**关注点**：这是触点 readout 动画，不是逐神经元活动场，也不是临床 SEEG；用于作者目视判断纯 Node 底物是否呈现可接受的双模式间期传播。
""")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--direct-metadata", type=Path, default=DEFAULT_DIRECT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(render(args.npz, args.direct_metadata, args.output), indent=2))


if __name__ == "__main__":
    main()
