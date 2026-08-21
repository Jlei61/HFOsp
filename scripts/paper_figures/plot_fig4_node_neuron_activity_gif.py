"""Replay the accepted Node-only interictal window with neuron activity frames.

This is a visual replay of a frozen rev11-NLC artifact, not a new fit. The
simulation must match the stored 20 s worker prefix exactly before the GIF is
written. The right panel is held identical to the accepted seed-1569 contact
readout; only the left panel changes from the static node field to activity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src" / "snn_engine"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.snn_engine.slow_field import firing_rate_field
from src.topic4_zm_ictal_transition import (
    build_substrate,
    load_round_config,
    make_external_drive,
)


CONFIG = ROOT / "config/topic4_data_driven_zm_ictal_transition_v1.json"
REFERENCE = ROOT / (
    "../.."  # resolved below so the accepted artifact may remain in main results
)
REFERENCE = Path("/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/") / (
    "data_driven_local_connectivity_rev11_nlc/frozen_substrate_confirmation/"
    "workers/node_baseline_seed_1569.npz"
)
DIRECT_METADATA = Path("/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/") / (
    "data_driven_local_connectivity_rev11_nlc/node_baseline_visual_acceptance/"
    "figures/fig4a_nlc_direct_readout_metadata.json"
)
OUTPUT = ROOT / (
    "results/topic4_sef_hfo/data_driven_zm_ictal_transition/"
    "node_field_activity_visual_acceptance/figures/"
    "node_baseline_seed1569_neuron_activity.gif"
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


def _active_fraction(spikes, substrate):
    cmrun = substrate.extras["cmrun"]
    return cmrun.active_fraction(
        spikes, float(substrate.engine["dt"]), cmrun.BIN_MS,
    )


def _parity(reference_path, active, active_dt):
    with np.load(reference_path, allow_pickle=False) as stored:
        expected_active = np.asarray(stored["active_fraction"])[:len(active)]
        expected_active_dt = float(stored["active_fraction_bin_ms"])
    stored_active = np.asarray(active).astype(expected_active.dtype)
    return {
        "storage_dtype_contract": {"active_fraction": str(expected_active.dtype)},
        "active_fraction_exact": bool(np.array_equal(stored_active, expected_active)),
        "active_fraction_dt_exact": bool(active_dt == expected_active_dt),
        "active_fraction_precast_max_abs_error": float(np.max(
            np.abs(np.asarray(active, float) - expected_active.astype(float))
        )),
        "right_panel_contract": "read frozen contact envelope directly; do not recompute from a truncated replay",
    }


def render(config_path: Path, reference_path: Path,
           direct_metadata_path: Path, output_path: Path):
    direct = json.loads(direct_metadata_path.read_text())
    window_start, window_stop = direct["direct_readout"]["display_window_ms"]
    seed = int(direct["direct_readout"]["seed"])
    if seed != 1569:
        raise ValueError("accepted broad-participation visual contract expects seed 1569")

    config = load_round_config(config_path)
    cache_dir = ROOT / config["output_root"] / "network_cache"
    substrate = build_substrate(
        config, "node_baseline", seed, cache_dir=cache_dir,
        ee_dose=0.0, etoi_dose=0.0,
    )
    replay_context_ms = 0.0
    duration_ms = float(window_stop)
    substrate.params.T = duration_ms
    substrate.net["rng"] = np.random.default_rng(seed)

    from kick_probe import simulate_kick

    drive = make_external_drive(substrate, config["spatial_ou"], seed)
    result = simulate_kick(
        substrate.params, substrate.net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=substrate.vtheta, slow=None,
        early_stop_runaway=False, external_e_rate_drive=drive,
    )
    spikes = np.asarray(result["E_spk_bool"], bool)
    active, active_dt = _active_fraction(spikes, substrate)
    parity = _parity(reference_path, active, active_dt)
    if not all(parity[key] for key in (
        "active_fraction_exact", "active_fraction_dt_exact",
    )):
        raise RuntimeError(f"short replay does not match frozen artifact: {parity}")

    dt = float(substrate.engine["dt"])
    frame_step_ms = 10.0
    activity_window_ms = 10.0
    frame_steps = np.arange(
        int(round((window_start + 20.0) / dt)),
        int(round(window_stop / dt)),
        int(round(frame_step_ms / dt)), dtype=int,
    )
    fields = []
    positive = []
    window_steps = int(round(activity_window_ms / dt))
    for step in frame_steps:
        fired = spikes[max(0, step - window_steps):step + 1].any(axis=0)
        field = firing_rate_field(
            fired, substrate.positions_e, float(substrate.engine["L"]),
            50, sigma=0.5,
        )
        fields.append(field)
        if np.any(field > 0):
            positive.append(field[field > 0])
    vmax = max(1.0, float(np.percentile(np.concatenate(positive), 98)))

    with np.load(reference_path, allow_pickle=False) as stored:
        names = np.asarray(stored["contact_names"]).astype(str)
        shafts = np.asarray(stored["shaft_ids"]).astype(str)
        contact_xy = np.asarray(stored["contact_xy_mm"], float)
        stored_envelope = np.asarray(stored["contact_envelope"], float)
        stored_envelope_dt = float(stored["contact_envelope_dt_ms"])
        event_on = np.asarray(stored["event_t_on_ms"], float)
        event_off = np.asarray(stored["event_t_off_ms"], float)
    order_names = direct["direct_readout"]["contact_order"]
    order = np.asarray([int(np.flatnonzero(names == name)[0]) for name in order_names])
    traces = _bandpass(stored_envelope, stored_envelope_dt)[order]
    names, shafts, contact_xy = names[order], shafts[order], contact_xy[order]
    full_trace_time = np.arange(stored_envelope.shape[1]) * stored_envelope_dt
    selected = (full_trace_time >= window_start) & (full_trace_time <= window_stop)
    traces = traces[:, selected]
    trace_time = full_trace_time[selected] - window_start
    scale = float(np.quantile(np.abs(traces), 0.995))
    traces = traces / max(scale, 1e-12)

    event_indices = np.flatnonzero(
        (event_on >= window_start) & (event_on <= window_stop)
    )
    if len(event_indices) < 2:
        raise RuntimeError("accepted visual window no longer contains its event pair")
    event_indices = event_indices[[0, -1]]
    event_spans = [
        (event_on[index] - window_start, event_off[index] - window_start)
        for index in event_indices
    ]

    fig = plt.figure(figsize=(12.8, 5.0), facecolor="white")
    grid = fig.add_gridspec(
        1, 2, width_ratios=(0.92, 1.72), left=0.055, right=0.985,
        bottom=0.12, top=0.91, wspace=0.18,
    )
    ax_field = fig.add_subplot(grid[0, 0])
    ax_trace = fig.add_subplot(grid[0, 1])
    image = ax_field.imshow(
        fields[0], origin="lower", extent=(0, 20, 0, 20), cmap="viridis",
        vmin=0.0, vmax=vmax, interpolation="bilinear",
    )
    for shaft in ("ICL", "SCL"):
        mask = shafts == shaft
        ax_field.scatter(
            contact_xy[mask, 0], contact_xy[mask, 1],
            s=34, c=SHAFT_COLORS[shaft], edgecolors="white", linewidths=0.8,
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
    colorbar.set_label("active E neurons", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    time_label = ax_field.text(
        0.02, 0.02, "", transform=ax_field.transAxes, fontsize=8.5,
        color="white", weight="bold", ha="left", va="bottom",
    )

    offsets = np.arange(len(names))[::-1] * 1.55
    colors = [SHAFT_COLORS[shaft] for shaft in shafts]
    for row, offset in enumerate(offsets):
        ax_trace.plot(trace_time, traces[row] * 0.62 + offset,
                      color=colors[row], lw=1.05)
    for (start, stop), color in zip(event_spans, (TA_COLOR, TB_COLOR)):
        ax_trace.axvspan(start, stop, color=color, alpha=0.11, lw=0)
    cursor = ax_trace.axvline(0.0, color="#202020", lw=1.1)
    ax_trace.set_yticks(offsets, names, fontsize=7.2)
    ax_trace.tick_params(axis="y", length=0, pad=4)
    ax_trace.set(
        xlim=(trace_time[0], trace_time[-1]), ylim=(-1.0, offsets[0] + 1.0),
        xlabel="Time in displayed window (ms)",
        ylabel="30-80 Hz virtual-contact activity",
    )
    ax_trace.spines[["top", "right", "left"]].set_visible(False)
    ax_trace.legend(
        handles=[Patch(facecolor=TB_COLOR, alpha=0.15, label="MTB event"),
                 Patch(facecolor=TA_COLOR, alpha=0.15, label="MTA event")],
        loc="upper right", frameon=False, ncol=2, fontsize=8,
    )

    def update(frame_index):
        step = int(frame_steps[frame_index])
        absolute_time_ms = step * dt
        relative_time_ms = absolute_time_ms - window_start
        image.set_data(fields[frame_index])
        cursor.set_xdata([relative_time_ms, relative_time_ms])
        time_label.set_text(f"t = {relative_time_ms:.0f} ms")
        return image, cursor, time_label

    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(
        fig, update, frames=len(frame_steps), interval=80, blit=False,
    )
    animation.save(output_path, writer=PillowWriter(fps=12.5), dpi=115)
    plt.close(fig)

    metadata = {
        "status": "NODE_ONLY_NEURON_ACTIVITY_VISUAL_REPLAY",
        "candidate_id": "node_baseline",
        "network_seed": seed,
        "zm": "off",
        "learned_edge_redistribution": "off",
        "display_window_ms": [window_start, window_stop],
        "simulation_duration_ms": duration_ms,
        "replay_context_ms_not_displayed": replay_context_ms,
        "activity_window_ms": activity_window_ms,
        "biological_frame_step_ms": frame_step_ms,
        "activity_field": "last-window distinct E neurons, 50x50 grid, sigma=0.5",
        "activity_colormap": "viridis",
        "activity_vmax_q98_positive_across_frames": vmax,
        "parity_vs_frozen_20s_artifact": parity,
        "reference_npz": str(reference_path),
        "reference_npz_sha256": _sha256(reference_path),
        "source_direct_metadata": str(direct_metadata_path),
        "source_direct_metadata_sha256": _sha256(direct_metadata_path),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "event_spans_relative_ms": {
            "MTA": list(event_spans[0]), "MTB": list(event_spans[1]),
        },
        "not_clinical_seeg": True,
        "output_gif": str(output_path),
        "output_gif_sha256": _sha256(output_path),
    }
    metadata_path = output_path.with_name(output_path.stem + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    (output_path.parent / "README.md").write_text(f"""### {output_path.name}

这条 GIF 保留已验收的纯 Node seed 1569 右侧读出窗口、触点顺序和 MTA/MTB 事件不变，只将左侧替换为 qI/GK 图同语法的逐神经元活动场：每帧统计前 {activity_window_ms:g} ms 内至少放电一次的 E 神经元，经 50 x 50 网格和固定 `sigma=0.5` 平滑后用 `viridis` 显示。

**关注点**：重放的 active-fraction 逐值等于冻结 20 s artifact 的对应前缀；右侧直接读取上一张验收图的冻结 contact envelope，避免截短重放在滤波边界产生差异。本图只增加 SNN sheet 内的起始位置与扩展顺序，不是临床 SEEG。
""")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--direct-metadata", type=Path, default=DIRECT_METADATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(render(
        args.config.resolve(), args.reference.resolve(),
        args.direct_metadata.resolve(), args.output.resolve(),
    ), indent=2))


if __name__ == "__main__":
    main()
