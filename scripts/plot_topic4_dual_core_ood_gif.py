#!/usr/bin/env python3
"""Render an algorithmically selected Fig.2C-style dual-core Node movie."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_d6_natural_kmeans import normalize_event_ranks  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_dual_core_ood_node_pathways.json"
MODE_COLORS = ("#c43c39", "#277da1")
SHAFT_COLORS = {"ICL": "#f07c3e", "SCL": "#2aa6b5"}


def select_representative_pair(
    candidate_summary: dict, *, minimum_icl: int = 6, minimum_scl: int = 2,
) -> dict:
    """Select one network and its closest supported event in each frozen mode."""
    choices = []
    for row in candidate_summary["per_network"]:
        events = [
            event for event in row["events"]
            if event["returned"] and event["in_support"]
            and event["ICL_recruited"] >= minimum_icl
            and event["SCL_recruited"] >= minimum_scl
        ]
        by_mode = {
            mode: sorted(
                (event for event in events if event["mode"] == mode),
                key=lambda event: (
                    event["normalized_support_distance"], event["event_index"],
                ),
            )
            for mode in (0, 1)
        }
        if not all(by_mode.values()):
            continue
        choices.append((
            -min(len(by_mode[0]), len(by_mode[1])),
            float(row["ood_all_returned"]), int(row["seed"]), row, by_mode,
        ))
    if not choices:
        raise RuntimeError(
            "no confirmation network contains display-eligible events in both modes"
        )
    _, _, seed, row, by_mode = sorted(choices, key=lambda item: item[:3])[0]
    return {
        "seed": seed,
        "candidate_id": row["candidate_id"],
        "network_ood_all_returned": row["ood_all_returned"],
        "events": {str(mode): by_mode[mode][0] for mode in (0, 1)},
        "worker_json": row["worker_json"],
        "worker_npz": row["worker_npz"],
    }


def _patient_profiles(target_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(target_path, allow_pickle=False) as loaded:
        names = np.asarray(loaded["contact_names"]).astype(str)
        ranks = np.asarray(loaded["patient_train_ranks"], float)
        labels = np.asarray(loaded["patient_train_old_labels"], int)
    normalized = normalize_event_ranks(ranks)
    profiles = np.asarray([
        np.nanmean(normalized[labels == mode], axis=0) for mode in (0, 1)
    ])
    return names, profiles, np.bincount(labels, minlength=2)


def _event_grid(loaded, event_index: int) -> tuple[np.ndarray, np.ndarray]:
    start = int(loaded["activity_grid_event_start"][event_index])
    count = int(loaded["activity_grid_event_count"][event_index])
    stop = start + count
    return (
        np.asarray(loaded["activity_grid"][start:stop], float),
        np.asarray(loaded["activity_grid_time_ms"][start:stop], float),
    )


def _window_envelope(
    envelope: np.ndarray, dt_ms: float, start_ms: float, stop_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(envelope, float)
    if values.ndim != 2:
        raise ValueError("contact envelope must be two-dimensional")
    if values.shape[0] == 15 and values.shape[1] != 15:
        values = values.T
    left = max(0, int(np.floor(start_ms / dt_ms)))
    right = min(len(values), int(np.ceil(stop_ms / dt_ms)))
    time_ms = np.arange(left, right, dtype=float) * dt_ms
    return time_ms, values[left:right]


def ordered_shaft_segments(order: np.ndarray, shaft_ids: np.ndarray) -> list[np.ndarray]:
    """Split a temporal contact order into within-shaft paths."""
    order = np.asarray(order, int)
    shaft_ids = np.asarray(shaft_ids).astype(str)
    return [
        np.asarray([index for index in order if shaft_ids[index] == shaft], int)
        for shaft in ("ICL", "SCL")
        if np.sum(shaft_ids[order] == shaft) >= 2
    ]


def _plot_shaft_paths(axis, contact_xy, order, shaft_ids, **kwargs) -> None:
    label = kwargs.pop("label", None)
    for segment_index, segment in enumerate(ordered_shaft_segments(order, shaft_ids)):
        axis.plot(
            contact_xy[segment, 0], contact_xy[segment, 1],
            label=label if segment_index == 0 else None, **kwargs,
        )


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def render(config_path: Path, output_dir: Path, *, fps: float = 8.0) -> dict:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"] / "confirmation"
    aggregate_path = root / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text())
    if aggregate.get("status") != "DUAL_CORE_OOD_PHASE_COMPLETE":
        raise RuntimeError("confirmation aggregate is not complete")
    selected = select_representative_pair(aggregate["ranking"][0])
    json_path = ROOT / selected["worker_json"]
    npz_path = ROOT / selected["worker_npz"]
    worker = json.loads(json_path.read_text())
    target_path = ROOT / config["inputs"]["shaft_aware_target_npz"]["path"]
    patient_names, patient_profiles, patient_counts = _patient_profiles(target_path)
    with np.load(npz_path, allow_pickle=False) as loaded:
        contact_names = np.asarray(loaded["contact_names"]).astype(str)
        if not np.array_equal(contact_names, patient_names):
            raise RuntimeError("model and patient contact order differ")
        shaft_ids = np.asarray(loaded["shaft_ids"]).astype(str)
        contact_xy = np.asarray(loaded["contact_xy_mm"], float)
        onsets = np.asarray(loaded["onsets"], float)
        envelope = np.asarray(loaded["contact_envelope"], float)
        envelope_dt = float(loaded["contact_envelope_dt_ms"])
        event_payload = []
        for mode in (0, 1):
            event = selected["events"][str(mode)]
            event_index = int(event["event_index"])
            grid, grid_time = _event_grid(loaded, event_index)
            timing = worker["events"][event_index]
            onset_ms = float(timing["t_on_ms"])
            stop_ms = float(timing["t_off_ms"])
            trace_time, trace = _window_envelope(
                envelope, envelope_dt, onset_ms - 20.0, stop_ms + 40.0,
            )
            event_payload.append({
                "mode": mode, "event_index": event_index,
                "grid": grid, "grid_relative_ms": grid_time - onset_ms,
                "onsets": onsets[event_index],
                "duration_ms": stop_ms - onset_ms,
                "trace_time_ms": trace_time - onset_ms,
                "trace": trace,
                "support_distance": event["normalized_support_distance"],
            })
    relative_stop = max(row["grid_relative_ms"].max() for row in event_payload)
    relative_time = np.arange(-20.0, relative_stop + 2.5, 5.0)
    positive = np.concatenate([
        row["grid"].ravel()[row["grid"].ravel() > 0] for row in event_payload
    ])
    vmax = float(np.quantile(positive, 0.995)) if len(positive) else 1.0
    vmax = max(vmax, 1.0)
    _style()
    fig = plt.figure(figsize=(7.2, 5.1))
    layout = fig.add_gridspec(2, 2, height_ratios=(1.08, 1.0), hspace=0.28, wspace=0.22)
    field_axes = [fig.add_subplot(layout[0, mode]) for mode in (0, 1)]
    trace_axes = [fig.add_subplot(layout[1, mode]) for mode in (0, 1)]
    images, cursors = [], []
    for mode, (field_axis, trace_axis, row) in enumerate(zip(
        field_axes, trace_axes, event_payload,
    )):
        image = field_axis.imshow(
            np.zeros_like(row["grid"][0]), origin="lower", extent=(0, 20, 0, 20),
            cmap="viridis", vmin=0.0, vmax=vmax, interpolation="bilinear",
            aspect="equal",
        )
        images.append(image)
        patient_order = np.argsort(patient_profiles[mode], kind="stable")
        finite = np.flatnonzero(np.isfinite(row["onsets"]))
        model_order = finite[np.argsort(row["onsets"][finite], kind="stable")]
        _plot_shaft_paths(
            field_axis, contact_xy, patient_order, shaft_ids,
            color="white", ls="--", lw=0.9, alpha=0.82,
            label="patient prototype (within shaft)",
        )
        _plot_shaft_paths(
            field_axis, contact_xy, model_order, shaft_ids,
            color=MODE_COLORS[mode], lw=1.35, alpha=0.95,
            label="model event (within shaft)",
        )
        field_axis.scatter(
            contact_xy[:, 0], contact_xy[:, 1], s=17, facecolor="white",
            edgecolor="#333333", linewidth=0.45, zorder=4,
        )
        field_axis.set(
            xlim=(0, 20), ylim=(0, 20), xlabel="sheet x (mm)",
            ylabel="sheet y (mm)" if mode == 0 else "",
        )
        field_axis.set_title(
            f"Mode {mode + 1}  |  support distance {row['support_distance']:.2f}",
            loc="left", color=MODE_COLORS[mode], fontweight="bold",
        )
        field_axis.legend(
            loc="upper right", frameon=False, fontsize=5.8, handlelength=1.4,
        )
        local = np.asarray(row["trace"], float)
        scale = np.nanpercentile(np.abs(local), 99, axis=0)
        scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
        normalized = local / scale[None, :]
        offsets = np.arange(len(contact_names))[::-1] * 1.55
        for index, name in enumerate(contact_names):
            trace_axis.plot(
                row["trace_time_ms"], normalized[:, index] + offsets[index],
                color=SHAFT_COLORS.get(shaft_ids[index], "#555555"), lw=0.65,
            )
        trace_axis.axvspan(
            0.0, row["duration_ms"], color=MODE_COLORS[mode], alpha=0.08, lw=0,
        )
        cursor = trace_axis.axvline(relative_time[0], color="#202020", lw=0.85)
        cursors.append(cursor)
        trace_axis.set_yticks(offsets)
        trace_axis.set_yticklabels(contact_names)
        trace_axis.set_xlim(relative_time[0], relative_time[-1])
        trace_axis.set_ylim(-1.2, offsets[0] + 1.2)
        trace_axis.set_xlabel("time from model event onset (ms)")
        trace_axis.set_ylabel("virtual-contact activity" if mode == 0 else "")
        trace_axis.tick_params(axis="y", length=0, pad=1.5)
    time_label = fig.text(
        0.5, 0.965, "", ha="center", va="top", fontsize=8.0, fontweight="bold",
    )
    fig.suptitle(
        "Fitted dual-core Node: model activity versus frozen patient mode order",
        y=0.995, fontsize=9.0, fontweight="bold",
    )
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.075, top=0.925)

    def update(frame_index: int):
        query = relative_time[frame_index]
        for image, cursor, row in zip(images, cursors, event_payload):
            nearest = int(np.argmin(np.abs(row["grid_relative_ms"] - query)))
            if query < row["grid_relative_ms"][0] - 2.5 or query > row[
                "grid_relative_ms"
            ][-1] + 2.5:
                image.set_data(np.zeros_like(row["grid"][0]))
            else:
                image.set_data(row["grid"][nearest])
            cursor.set_xdata([query, query])
        time_label.set_text(f"{query:+.0f} ms")
        return [*images, *cursors, time_label]

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "dual_core_node_fig2c_mode_check"
    animation = FuncAnimation(
        fig, update, frames=len(relative_time), interval=1000.0 / fps, blit=False,
    )
    animation.save(stem.with_suffix(".gif"), writer=PillowWriter(fps=fps), dpi=120)
    frame_indices = {
        "first": 0, "middle": len(relative_time) // 2,
        "last": len(relative_time) - 1,
    }
    for name, index in frame_indices.items():
        update(index)
        fig.savefig(
            output_dir / f"dual_core_node_fig2c_mode_check_{name}.png",
            dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03,
        )
    plt.close(fig)
    metadata = {
        "status": "DUAL_CORE_NODE_FIG2C_MODE_CHECK_RENDERED",
        "candidate_id": selected["candidate_id"],
        "seed": selected["seed"],
        "network_ood_all_returned": selected["network_ood_all_returned"],
        "selected_events": selected["events"],
        "patient_mode_event_counts": patient_counts.tolist(),
        "selection_rule": (
            "same confirmation network; both events returned and within frozen patient "
            "support; >=6 ICL and >=2 SCL contacts; minimum normalized support distance"
        ),
        "activity_readout": "5 ms E-neuron spike-count grid; viridis",
        "trace_readout": "virtual-contact model-current envelope; normalized per contact for display",
        "patient_reference": "mean normalized rank profile from frozen patient training modes",
        "not_clinical_seeg": True,
        "aggregate": str(aggregate_path.relative_to(ROOT)),
        "worker_json": selected["worker_json"],
        "worker_npz": selected["worker_npz"],
        "outputs": [
            str(stem.with_suffix(".gif").relative_to(ROOT)),
            *[
                str((output_dir / f"dual_core_node_fig2c_mode_check_{name}.png").relative_to(ROOT))
                for name in frame_indices
            ],
        ],
    }
    metadata_path = output_dir / "dual_core_node_fig2c_mode_check_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    readme = output_dir / "README.md"
    existing = readme.read_text() if readme.exists() else ""
    entry = (
        "### dual_core_node_fig2c_mode_check.gif\n"
        "严格双 core Node 候选在同一 confirmation 网络中的两类代表事件。上排是 "
        "5 ms 兴奋神经元活动场；实线和白色虚线分别在每根杆内连接模型事件与患者冻结原型的"
        "触点招募顺序，避免把跨杆跳转误画成空间传播边；"
        "下排为同步虚拟触点读出。该图只用于检查 Fig.2C 双向传播形态，不能替代 OOD 统计。\n\n"
        "**关注点**：两类事件是否都跨杆招募、传播方向是否与患者原型一致，以及是否存在局部点亮但"
        "未形成完整传播的情况。\n"
    )
    if "### dual_core_node_fig2c_mode_check.gif" not in existing:
        readme.write_text(existing + ("\n" if existing else "") + entry)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fps", type=float, default=8.0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output_dir or (
        ROOT / config["output_root"] / "confirmation/figures"
    )
    metadata = render(args.config, output.resolve(), fps=float(args.fps))
    print(json.dumps({
        "status": metadata["status"], "seed": metadata["seed"],
        "outputs": metadata["outputs"],
    }, indent=2))


if __name__ == "__main__":
    main()
