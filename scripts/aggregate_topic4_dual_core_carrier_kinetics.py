#!/usr/bin/env python3
"""Aggregate carrier kinetics while protecting the frozen interictal repertoire."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_topic4_dual_core_ood_phase import _score_worker  # noqa: E402
from src.topic4_d6_natural_kmeans import natural_kmeans  # noqa: E402
from src.topic4_dual_core_carrier import (  # noqa: E402
    baseline_mask_from_events,
    event_window_indices,
    raw_population_burst_summary,
)
from src.topic4_dual_core_ood import load_embedding  # noqa: E402
from src.topic4_shaft_aware import contract_groups  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_dual_core_carrier_kinetics.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _finite_summary(values, reducer=np.mean):
    array = np.asarray([
        np.nan if value is None else float(value) for value in values
    ], float)
    finite = array[np.isfinite(array)]
    return float(reducer(finite)) if len(finite) else None


def _carrier_metrics(npz_path: Path, events: list[dict]) -> dict:
    with np.load(npz_path, allow_pickle=False) as loaded:
        time_ms = np.asarray(loaded["carrier_time_ms"], float)
        e_rate = np.asarray(loaded["carrier_E_rate_hz"], float)
        region_names = np.asarray(loaded["carrier_region_names"]).astype(str)
        ranks = np.asarray(loaded["ranks"], float)
        event_on = np.asarray(loaded["event_t_on_ms"], float)
        event_off = np.asarray(loaded["event_t_off_ms"], float)
        bin_ms = float(loaded["carrier_bin_ms"])
    baseline = baseline_mask_from_events(
        time_ms, [
            {"t_on_ms": on, "t_off_ms": off}
            for on, off in zip(event_on, event_off)
        ],
    )
    indices, complete = event_window_indices(
        event_on, trace_length=len(time_ms), bin_ms=bin_ms,
    )
    lookup = {name: index for index, name in enumerate(region_names)}
    core_indices = [lookup["core_1"], lookup["core_2"]]
    support = np.asarray([row["in_support"] for row in events], bool)
    returned = np.asarray([row["returned"] for row in events], bool)
    labels = np.asarray([row["mode"] for row in events], int)
    selected = support & returned & complete
    per_event = []
    for event_index in np.flatnonzero(selected):
        core_rows = [
            raw_population_burst_summary(
                e_rate[indices[event_index], core], bin_ms=bin_ms,
                baseline_values=e_rate[baseline, core],
            )
            for core in core_indices
        ]
        per_event.append({
            "event_index": int(event_index), "mode": int(labels[event_index]),
            "any_core_regular_three_cycle": bool(any(
                row["regular_three_cycle_burst"] for row in core_rows
            )),
            "core_metrics": core_rows,
        })
    peak_values = [
        row["peak_hz"] for event in per_event for row in event["core_metrics"]
        if row["peak_hz"] is not None
    ]
    intervals = [
        row["raw_peak_interval_frequency_hz"]
        for event in per_event for row in event["core_metrics"]
        if row["raw_peak_interval_frequency_hz"] is not None
    ]
    return {
        "n_supported_complete_events": len(per_event),
        "native_three_cycle_event_fraction": (
            float(np.mean([
                row["any_core_regular_three_cycle"] for row in per_event
            ])) if per_event else None
        ),
        "median_fourier_peak_hz": float(np.median(peak_values)) if peak_values else None,
        "median_raw_peak_interval_hz": float(np.median(intervals)) if intervals else None,
        "per_event": per_event,
        "ranks": ranks,
        "selected": selected,
        "labels": labels,
    }


def _heatmap(ax, matrix, xlabels, ylabels, title, *, fmt=".2f", cmap="viridis"):
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_xlabel("AMPA decay (ms)")
    ax.set_ylabel("GABA decay (ms)")
    ax.set_title(title, fontsize=9, weight="bold")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column, row, "NA" if not np.isfinite(value) else format(value, fmt),
                ha="center", va="center", fontsize=6.5,
                color="white" if np.isfinite(value) and value > np.nanmedian(matrix) else "black",
            )
    return image


def aggregate(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    for record in config["inputs"].values():
        path = ROOT / record["path"]
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen input changed: {path}")
    output_root = ROOT / config["output_root"]
    worker_config = json.loads((ROOT / config["inputs"]["worker_config"]["path"]).read_text())
    contract = json.loads((ROOT / config["inputs"]["contact_contract"]["path"]).read_text())
    embedding = load_embedding(str(ROOT / config["inputs"]["patient_target"]["path"]))
    manifest = json.loads((ROOT / config["inputs"]["confirmation_manifest"]["path"]).read_text())
    classifier = manifest["direction_classifier"]
    groups = contract_groups(contract)
    rows = []
    for candidate in config["candidates"]:
        for seed in config["simulation"]["network_seeds"]:
            stem = output_root / "workers" / f"{candidate['candidate_id']}_seed_{seed}"
            worker_json = stem.with_suffix(".json")
            worker_npz = stem.with_suffix(".npz")
            payload = json.loads(worker_json.read_text())
            if payload.get("status") != "REV10R_EDGE_FLOW_WORKER_COMPLETE":
                raise RuntimeError(f"incomplete worker: {worker_json}")
            if payload["arrays"]["sha256"] != _sha256(worker_npz):
                raise RuntimeError(f"worker hash changed: {worker_npz}")
            score, events = _score_worker(
                worker_npz, contract, embedding, classifier,
            )
            carrier = _carrier_metrics(worker_npz, events)
            support = carrier.pop("selected")
            labels = carrier.pop("labels")
            ranks = carrier.pop("ranks")
            joint = np.zeros(len(events), bool)
            with np.load(worker_npz, allow_pickle=False) as loaded:
                onsets = np.asarray(loaded["onsets"], float)
            joint = (
                np.isfinite(onsets[:, groups["ICL"]]).any(axis=1)
                & np.isfinite(onsets[:, groups["SCL"]]).any(axis=1)
            )
            natural = natural_kmeans(
                ranks[support & joint], labels[support & joint],
                random_state=int(seed),
            )
            rows.append({
                **candidate, "seed": int(seed), **score,
                "natural_kmeans": natural, "carrier": carrier,
                "worker_json": str(worker_json.relative_to(ROOT)),
                "worker_npz": str(worker_npz.relative_to(ROOT)),
            })
    summaries = []
    for candidate in config["candidates"]:
        selected = [
            row for row in rows if row["candidate_id"] == candidate["candidate_id"]
        ]
        kmeans_values = [
            row["natural_kmeans"].get("direction_balanced_alignment")
            for row in selected
        ]
        carrier_values = [
            row["carrier"]["native_three_cycle_event_fraction"]
            for row in selected
        ]
        summaries.append({
            **candidate,
            "n_networks": len(selected),
            "networks_with_both_modes": int(sum(
                row["both_modes_in_support"] for row in selected
            )),
            "n_networks_kmeans_evaluable": int(sum(
                value is not None and np.isfinite(float(value))
                for value in kmeans_values
            )),
            "n_networks_carrier_evaluable": int(sum(
                value is not None and np.isfinite(float(value))
                for value in carrier_values
            )),
            "equal_network_ood_all_returned": float(np.mean([
                row["ood_all_returned"] for row in selected
            ])),
            "equal_network_returned_events": float(np.mean([
                row["n_returned"] for row in selected
            ])),
            "natural_kmeans_alignment": _finite_summary(kmeans_values),
            "native_three_cycle_event_fraction": _finite_summary(carrier_values),
            "median_fourier_peak_hz": _finite_summary([
                row["carrier"]["median_fourier_peak_hz"] for row in selected
            ], reducer=np.median),
            "per_network": selected,
        })
    output = {
        "status": "DUAL_CORE_CARRIER_KINETICS_AGGREGATED",
        "scientific_role": config["scientific_role"],
        "frozen_substrate": config["frozen_substrate"],
        "n_candidates": len(summaries),
        "n_networks_per_candidate": len(config["simulation"]["network_seeds"]),
        "summaries": summaries,
        "exploratory_verdict": (
            "NATIVE_LOCAL_MULTICYCLE_CARRIER_NOT_RECOVERED"
            if not any(
                (row["native_three_cycle_event_fraction"] or 0.0) > 0.0
                for row in summaries
            ) else "NATIVE_LOCAL_MULTICYCLE_CARRIER_OBSERVED"
        ),
        "claim_boundary": config["claim_boundary"],
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha256(config_path)},
    }
    aggregate_path = output_root / "aggregate.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(_jsonable(output), indent=2) + "\n")

    ampa = sorted({float(row["tau_d_AMPA_ms"]) for row in summaries})
    gaba = sorted({float(row["tau_d_GABA_ms"]) for row in summaries})
    metrics = [
        ("equal_network_ood_all_returned", "OOD (lower is better)", "magma_r"),
        ("natural_kmeans_alignment", "Natural KMeans alignment", "viridis"),
        ("native_three_cycle_event_fraction", "Native >=3-cycle events", "viridis"),
        ("equal_network_returned_events", "Returned events / network", "viridis"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.4), constrained_layout=True)
    for panel, (ax, (key, title, cmap)) in enumerate(zip(axes.flat, metrics)):
        matrix = np.full((len(gaba), len(ampa)), np.nan)
        for row in summaries:
            y = gaba.index(float(row["tau_d_GABA_ms"]))
            x = ampa.index(float(row["tau_d_AMPA_ms"]))
            matrix[y, x] = row[key]
        image = _heatmap(ax, matrix, ampa, gaba, title, cmap=cmap)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        ax.text(-0.18, 1.08, chr(ord("A") + panel), transform=ax.transAxes,
                fontsize=11, weight="bold", va="top")
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_root / "dual_core_carrier_kinetics.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_root / "dual_core_carrier_kinetics.pdf", bbox_inches="tight")
    plt.close(fig)
    (figure_root / "README.md").write_text(
        "### dual_core_carrier_kinetics.png\n\n"
        "在冻结双 core Node、空间 OU、拓扑和零连接重分配下，仅比较 AMPA/GABA 衰减时间。"
        "A/B 检查患者支持和自然双模式是否保留，C 检查原始 1 ms 群体率是否出现至少三个规则周期，D 报告返回事件产率。\n\n"
        "**关注点**：六个组合均未产生原生三周期振荡；NA 表示该组合的支持内事件不足以评价自然 KMeans，"
        "不是零分。频率或周期改善只有在 OOD、KMeans 和事件产率没有明显恶化时才可继续。\n"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = aggregate(args.config.resolve())
    print(json.dumps({
        "status": result["status"], "n_candidates": result["n_candidates"],
    }, indent=2))


if __name__ == "__main__":
    main()
