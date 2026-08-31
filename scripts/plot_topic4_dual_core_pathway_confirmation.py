#!/usr/bin/env python3
"""Plot paired confirmation and the strongest rare native-cycle example."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_dual_core_carrier import event_window_indices  # noqa: E402


DEFAULT_AGGREGATE = ROOT / (
    "results/topic4_sef_hfo/data_driven_dual_core_ood/pathway_refit/"
    "confirmation/aggregate.json"
)


def strongest_regular_event(candidate_summary: dict) -> dict | None:
    choices = []
    for network in candidate_summary["per_network"]:
        carrier = network.get("carrier")
        if carrier is None:
            continue
        for event in carrier["per_event"]:
            for core_index, core in enumerate(event["core_metrics"]):
                if not core["regular_three_cycle_burst"]:
                    continue
                choices.append((
                    -float(core["event_peak_value"]),
                    -float(core.get("power_30_80_over_5_30") or 0.0),
                    -int(core["raw_peak_count"]),
                    int(network["seed"]), int(event["event_index"]),
                    core_index, network, core,
                ))
    if not choices:
        return None
    _, _, _, seed, event_index, core_index, network, metrics = sorted(choices)[0]
    return {
        "seed": seed,
        "event_index": event_index,
        "core_index": core_index,
        "worker_npz": network["worker_npz"],
        "metrics": metrics,
    }


def _paired_values(summary: dict, key) -> tuple[np.ndarray, np.ndarray]:
    rows = sorted(summary["per_network"], key=lambda row: row["seed"])
    seeds = np.asarray([row["seed"] for row in rows], int)
    values = np.asarray([key(row) for row in rows], float)
    return seeds, values


def _paired_axis(ax, node, work, ylabel, *, patient=None, percent=False):
    x = np.array([0.0, 1.0])
    for left, right in zip(node, work):
        ax.plot(x, [left, right], color="0.78", lw=0.65, zorder=1)
    ax.scatter(np.zeros(len(node)), node, color="#777777", s=16, zorder=2)
    ax.scatter(np.ones(len(work)), work, color="#16817A", s=16, zorder=2)
    means = [np.mean(node), np.mean(work)]
    ax.plot(x, means, color="#222222", lw=1.2, marker="o", ms=3.2, zorder=3)
    if patient is not None:
        ax.axhline(patient, color="#C43C39", lw=0.9, ls="--")
    ax.set_xticks(x, ["Node", "+EE/E→I"])
    ax.set_ylabel(ylabel)
    if percent:
        upper = max(float(np.max(node, initial=0.0)), float(np.max(work, initial=0.0)))
        ax.set_ylim(0.0, max(1.0, 1.15 * upper))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.5)


def plot(aggregate_path: Path, output_stem: Path) -> dict:
    aggregate = json.loads(aggregate_path.read_text())
    if aggregate.get("status") != "DUAL_CORE_PATHWAY_REFIT_CONFIRMATION_AGGREGATED":
        raise RuntimeError("paired pathway confirmation is not aggregated")
    lookup = {row["candidate_id"]: row for row in aggregate["summaries"]}
    node = lookup["gee000_getoi000"]
    work = lookup[aggregate["frozen_work_point"]]
    example = strongest_regular_event(work)
    if example is None:
        raise RuntimeError("no regular-cycle event exists for diagnostic plotting")

    _, node_ood = _paired_values(node, lambda row: row["ood_all_returned"])
    _, work_ood = _paired_values(work, lambda row: row["ood_all_returned"])
    _, node_mode = _paired_values(node, lambda row: row["mode_2_fraction"])
    _, work_mode = _paired_values(work, lambda row: row["mode_2_fraction"])
    _, node_k = _paired_values(
        node, lambda row: row["natural_kmeans"]["direction_balanced_alignment"],
    )
    _, work_k = _paired_values(
        work, lambda row: row["natural_kmeans"]["direction_balanced_alignment"],
    )
    _, node_cycle = _paired_values(
        node, lambda row: row["carrier"]["native_population_three_cycle_event_fraction"],
    )
    _, work_cycle = _paired_values(
        work, lambda row: row["carrier"]["native_population_three_cycle_event_fraction"],
    )

    npz_path = ROOT / example["worker_npz"]
    with np.load(npz_path, allow_pickle=False) as loaded:
        event_on = np.asarray(loaded["event_t_on_ms"], float)
        time_ms = np.asarray(loaded["carrier_time_ms"], float)
        bin_ms = float(loaded["carrier_bin_ms"])
        indices, complete = event_window_indices(
            event_on, trace_length=len(time_ms), bin_ms=bin_ms,
        )
        if not complete[example["event_index"]]:
            raise RuntimeError("selected carrier example has an incomplete window")
        index = indices[example["event_index"]]
        e_rate = np.asarray(loaded["carrier_E_rate_hz"], float)[index, :2]
        current = np.asarray(loaded["carrier_current_activity"], float)[index, :2]
    offset = np.arange(len(index), dtype=float) * bin_ms - 64.0

    fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05])
    axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(3)]
    _paired_axis(axes[0], node_ood, work_ood, "OOD (lower is better)")
    _paired_axis(
        axes[1], node_mode, work_mode, "Mode 2 fraction",
        patient=aggregate["patient_reference"]["mode_2_fraction"],
    )
    _paired_axis(axes[2], node_k, work_k, "Natural KMeans")
    _paired_axis(
        axes[3], 100.0 * node_cycle, 100.0 * work_cycle,
        "Population ≥3-cycle events (%)", percent=True,
    )
    axes[3].text(
        0.5, 0.5, "0% in both arms", transform=axes[3].transAxes,
        ha="center", va="center", fontsize=7.2, color="0.3",
    )
    selected_core = example["core_index"]
    core_color = ["#D1495B", "#177E89"][selected_core]
    axes[4].plot(offset, e_rate[:, selected_core], color=core_color, lw=1.0)
    axes[5].plot(offset, current[:, selected_core], color=core_color, lw=1.0)
    for ax in axes[4:]:
        ax.axvline(0.0, color="0.25", lw=0.7, ls="--")
        ax.set_xlim(-10.0, 80.0)
        ax.set_xlabel("Time from event onset (ms)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6.5)
    axes[4].set_ylabel("Raw E population rate (Hz)")
    axes[4].axhline(50.0, color="0.4", lw=0.7, ls=":")
    axes[4].set_title(
        f"Timing-only candidate: core {selected_core + 1}",
        fontsize=7.2, weight="bold",
    )
    axes[5].set_ylabel("Current proxy (a.u.)")
    axes[5].set_title("Same event, unfiltered current", fontsize=7.2, weight="bold")
    fig.canvas.draw()
    for index_panel, ax in enumerate(axes):
        position = ax.get_position()
        fig.text(position.x0 - 0.065, position.y1 + 0.02,
                 chr(ord("A") + index_panel), fontsize=10.5,
                 weight="bold", va="bottom", ha="right")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "status": "DUAL_CORE_PATHWAY_CONFIRMATION_PLOTTED",
        "aggregate": str(aggregate_path.relative_to(ROOT)),
        "work_point": aggregate["frozen_work_point"],
        "rare_cycle_example": example,
        "claim_boundary": (
            "the example is the strongest registered diagnostic among rare regular-"
            "cycle events; a separate population-amplitude criterion rejects it, "
            "so it does not establish an HFO carrier"
        ),
        "outputs": [
            str(output_stem.with_suffix(".png").relative_to(ROOT)),
            str(output_stem.with_suffix(".pdf").relative_to(ROOT)),
        ],
    }
    output_stem.with_name(output_stem.name + "_metadata").with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    readme = output_stem.parent / "README.md"
    existing = readme.read_text() if readme.exists() else ""
    entry = (
        "\n### dual_core_pathway_confirmation.png\n\n"
        "A--D 以 network seed 为独立单位配对比较冻结 Node 与重新标定通路。E/F 仅展示候选中"
        "峰值最大的 timing-only 周期候选；虚线是每 1 ms 至少 5% core 神经元同步的群体阈值。"
        "旧判据记录到约 2% 候选，但没有事件满足三次群体峰，因此不构成原生 HFO carrier。\n\n"
        "**关注点**：OOD 和 Mode 2 占比是否稳定改善，以及 timing-only 峰为何不能算群体载波。\n"
    )
    header = "### dual_core_pathway_confirmation.png"
    if header in existing:
        prefix = existing.split(header, 1)[0].rstrip()
        suffix = existing.split(header, 1)[1]
        next_header = suffix.find("\n### ")
        trailing = "" if next_header < 0 else suffix[next_header:]
        readme.write_text(prefix + entry + trailing)
    else:
        readme.write_text(existing.rstrip() + entry)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--output-stem", type=Path)
    args = parser.parse_args()
    output = args.output_stem or (
        args.aggregate.parent / "figures/dual_core_pathway_confirmation"
    )
    metadata = plot(args.aggregate.resolve(), output.resolve())
    print(json.dumps({
        "status": metadata["status"], "outputs": metadata["outputs"],
    }, indent=2))


if __name__ == "__main__":
    main()
