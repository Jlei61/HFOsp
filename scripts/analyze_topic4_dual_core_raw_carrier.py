#!/usr/bin/env python3
"""Analyze and plot the frozen dual-core raw carrier canary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.topic4_dual_core_carrier import (
    baseline_mask_from_events,
    event_window_indices,
    raw_population_burst_summary,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = ROOT / (
    "results/topic4_sef_hfo/data_driven_dual_core_ood/carrier_canary"
)
DEFAULT_FINAL = ROOT / (
    "results/topic4_sef_hfo/data_driven_dual_core_ood/final_analysis.json"
)
DEFAULT_GIF_META = ROOT / (
    "results/topic4_sef_hfo/data_driven_dual_core_ood/confirmation/figures/"
    "dual_core_node_fig2c_mode_check_metadata.json"
)


def _event_records(final_analysis: dict, seed: int) -> list[dict]:
    return list(final_analysis["node_confirmation"]["per_network"][str(seed)]["events"])


def _representative_indices(metadata: dict) -> dict[int, int]:
    return {
        int(mode): int(record["event_index"])
        for mode, record in metadata["selected_events"].items()
    }


def analyze(npz_path: Path, final_path: Path, gif_meta_path: Path) -> dict:
    final = json.loads(final_path.read_text())
    gif_meta = json.loads(gif_meta_path.read_text())
    seed = int(gif_meta["seed"])
    events = _event_records(final, seed)
    with np.load(npz_path, allow_pickle=False) as loaded:
        if not bool(loaded["carrier_readout_enabled"]):
            raise RuntimeError("worker artifact does not contain raw carrier readout")
        time_ms = np.asarray(loaded["carrier_time_ms"], float)
        region_names = np.asarray(loaded["carrier_region_names"]).astype(str)
        e_rate = np.asarray(loaded["carrier_E_rate_hz"], float)
        i_rate = np.asarray(loaded["carrier_I_rate_hz"], float)
        current = np.asarray(loaded["carrier_current_activity"], float)
        current_names = np.asarray(loaded["carrier_current_site_names"]).astype(str)
        event_on = np.asarray(loaded["event_t_on_ms"], float)
        event_off = np.asarray(loaded["event_t_off_ms"], float)
        event_returned = np.asarray(loaded["event_returned"], bool)
        bin_ms = float(loaded["carrier_bin_ms"])
    if len(events) != len(event_on):
        raise RuntimeError("final analysis and carrier event inventories disagree")
    if not np.array_equal(
        event_returned, np.asarray([row["returned"] for row in events], bool),
    ):
        raise RuntimeError("carrier event-return flags drifted")
    event_intervals = [
        {"t_on_ms": on, "t_off_ms": off}
        for on, off in zip(event_on, event_off)
    ]
    baseline_mask = baseline_mask_from_events(time_ms, event_intervals)
    indices, complete = event_window_indices(
        event_on, trace_length=len(time_ms), bin_ms=bin_ms,
    )
    offset_ms = (np.arange(indices.shape[1]) * bin_ms) - 64.0
    region_lookup = {name: index for index, name in enumerate(region_names)}
    current_lookup = {name: index for index, name in enumerate(current_names)}
    core_regions = [region_lookup["core_1"], region_lookup["core_2"]]
    core_currents = [
        current_lookup["core_1_center"], current_lookup["core_2_center"],
    ]

    scored_events = []
    for event_index, record in enumerate(events):
        if not (
            complete[event_index] and bool(record["returned"])
            and bool(record["in_support"])
        ):
            continue
        regional = []
        current_sites = []
        for core_number, (region_index, current_index) in enumerate(
            zip(core_regions, core_currents), start=1,
        ):
            regional.append({
                "core": core_number,
                **raw_population_burst_summary(
                    e_rate[indices[event_index], region_index],
                    bin_ms=bin_ms,
                    baseline_values=e_rate[baseline_mask, region_index],
                ),
            })
            current_sites.append({
                "core": core_number,
                **raw_population_burst_summary(
                    current[indices[event_index], current_index],
                    bin_ms=bin_ms,
                    baseline_values=current[baseline_mask, current_index],
                ),
            })
        scored_events.append({
            "event_index": event_index,
            "mode": int(record["mode"]),
            "normalized_support_distance": float(
                record["normalized_support_distance"]
            ),
            "E_population": regional,
            "current_proxy": current_sites,
        })

    summaries = {}
    for readout in ("E_population", "current_proxy"):
        summaries[readout] = {}
        for mode in (0, 1):
            rows = [row for row in scored_events if row["mode"] == mode]
            summaries[readout][str(mode)] = {}
            for core in (1, 2):
                values = [
                    next(value for value in row[readout] if value["core"] == core)
                    for row in rows
                ]
                summaries[readout][str(mode)][f"core_{core}"] = {
                    "n_events": len(values),
                    "regular_three_cycle_fraction": (
                        float(np.mean([
                            value["regular_three_cycle_burst"] for value in values
                        ])) if values else None
                    ),
                    "median_raw_peak_count": (
                        float(np.median([
                            value["raw_peak_count"] for value in values
                        ])) if values else None
                    ),
                    "median_peak_hz": (
                        float(np.median([
                            value["peak_hz"] for value in values
                            if value["peak_hz"] is not None
                        ])) if values else None
                    ),
                }
    representatives = _representative_indices(gif_meta)
    return {
        "status": "DUAL_CORE_RAW_CARRIER_CANARY_ANALYZED",
        "seed": seed,
        "bin_ms": bin_ms,
        "temporal_smoothing_ms": 0.0,
        "readout_boundary": (
            "model E/I rates and current proxy; not clinical SEEG or patient HFO"
        ),
        "region_names": region_names.tolist(),
        "current_site_names": current_names.tolist(),
        "n_baseline_bins": int(np.sum(baseline_mask)),
        "n_supported_returned_complete_events": len(scored_events),
        "representative_event_index_by_mode": {
            str(key): value for key, value in representatives.items()
        },
        "summary": summaries,
        "events": scored_events,
        "plot_arrays": {
            "offset_ms": offset_ms,
            "representative_E_rate": {
                str(mode): e_rate[indices[event_index]][:, core_regions]
                for mode, event_index in representatives.items()
            },
            "representative_I_rate": {
                str(mode): i_rate[indices[event_index]][:, core_regions]
                for mode, event_index in representatives.items()
            },
            "representative_current": {
                str(mode): current[indices[event_index]][:, core_currents]
                for mode, event_index in representatives.items()
            },
        },
    }


def _event_metric(payload: dict, mode: int, readout: str, core: int, key: str):
    values = []
    for event in payload["events"]:
        if event["mode"] != mode:
            continue
        record = next(row for row in event[readout] if row["core"] == core)
        values.append(record[key])
    return np.asarray(values, float)


def plot(payload: dict, output: Path) -> None:
    arrays = payload.pop("plot_arrays")
    offset = np.asarray(arrays["offset_ms"], float)
    colors = ["#D1495B", "#177E89"]
    fig = plt.figure(figsize=(7.2, 5.1), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    for mode in (0, 1):
        ax = fig.add_subplot(grid[0, mode])
        rates = np.asarray(arrays["representative_E_rate"][str(mode)], float)
        for core in (0, 1):
            ax.plot(
                offset, rates[:, core], color=colors[core], lw=1.2,
                label=f"core {core + 1}",
            )
        ax.axvline(0.0, color="0.25", lw=0.8, ls="--")
        ax.set_title(f"Mode {'A' if mode == 0 else 'B'}", fontsize=9, weight="bold")
        ax.set_xlabel("Time from event onset (ms)")
        if mode == 0:
            ax.set_ylabel("Raw E population rate (Hz)")
            ax.legend(frameon=False, fontsize=7, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)

    ax = fig.add_subplot(grid[1, 0])
    x_positions, values, point_colors = [], [], []
    position = 0
    labels = []
    for mode in (0, 1):
        for core in (1, 2):
            rows = _event_metric(payload, mode, "E_population", core, "raw_peak_count")
            x_positions.extend([position] * len(rows))
            values.extend(rows.tolist())
            point_colors.extend([colors[core - 1]] * len(rows))
            labels.append(f"{'A' if mode == 0 else 'B'}\nC{core}")
            position += 1
        position += 0.35
    ax.scatter(x_positions, values, c=point_colors, s=10, alpha=0.55, linewidths=0)
    ax.set_xticks([0, 1, 2.35, 3.35], labels)
    ax.set_ylabel("Raw population peaks per event")
    ax.set_title("All supported events", fontsize=9, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)

    ax = fig.add_subplot(grid[1, 1])
    fractions = []
    bar_colors = []
    labels = []
    for mode in (0, 1):
        for core in (1, 2):
            row = payload["summary"]["E_population"][str(mode)][f"core_{core}"]
            fractions.append(row["regular_three_cycle_fraction"])
            bar_colors.append(colors[core - 1])
            labels.append(f"{'A' if mode == 0 else 'B'}\nC{core}")
    ax.bar([0, 1, 2.35, 3.35], fractions, width=0.75, color=bar_colors, alpha=0.88)
    ax.set_xticks([0, 1, 2.35, 3.35], labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction with >=3 regular raw cycles")
    ax.set_title("Native-cycle diagnostic", fontsize=9, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)

    fig.text(0.01, 0.99, "A", va="top", fontsize=11, weight="bold")
    fig.text(0.01, 0.49, "B", va="top", fontsize=11, weight="bold")
    fig.text(0.50, 0.49, "C", va="top", fontsize=11, weight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker-npz", type=Path,
        default=DEFAULT_RESULT_ROOT / "dualcore_s39_seed_2430.npz",
    )
    parser.add_argument("--final-analysis", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--gif-metadata", type=Path, default=DEFAULT_GIF_META)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULT_ROOT)
    args = parser.parse_args()
    payload = analyze(
        args.worker_npz.resolve(), args.final_analysis.resolve(),
        args.gif_metadata.resolve(),
    )
    plot_payload = dict(payload)
    plot_payload["plot_arrays"] = payload["plot_arrays"]
    figure_stem = args.output_root / "figures/dual_core_raw_carrier_canary"
    plot(plot_payload, figure_stem)
    payload.pop("plot_arrays")
    payload["outputs"] = [
        str(figure_stem.with_suffix(".png")),
        str(figure_stem.with_suffix(".pdf")),
    ]
    output_json = args.output_root / "raw_carrier_analysis.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
    readme = args.output_root / "figures/README.md"
    readme.write_text(
        "### dual_core_raw_carrier_canary.png\n\n"
        "A 显示与 Fig.4 GIF 相同的两个代表事件，但使用 1 ms、未经时间平滑的 core 群体放电率。"
        "B 汇总所有落在患者支持内且完整返回的事件中，原始群体峰的数量。"
        "C 报告满足至少三个近似规则原始周期的事件比例；它不使用带通后的振铃作为周期证据。\n\n"
        "**关注点**：若 A 仍只有单个宽峰且 C 接近零，当前工作点支持传播顺序，但不支持局部原生高频爆发。\n"
    )
    print(json.dumps({
        "status": payload["status"], "output": str(output_json),
        "figure": str(figure_stem.with_suffix(".png")),
    }, indent=2))


if __name__ == "__main__":
    main()
