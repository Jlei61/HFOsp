#!/usr/bin/env python3
"""Aggregate the frozen dual-core EE/E-to-I expression surface."""
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
from scripts.aggregate_topic4_dual_core_carrier_kinetics import (  # noqa: E402
    _carrier_metrics,
)
from src.topic4_d6_natural_kmeans import natural_kmeans  # noqa: E402
from src.topic4_dual_core_ood import load_embedding  # noqa: E402
from src.topic4_shaft_aware import contract_groups  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_dual_core_pathway_refit.json"


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


def _finite_mean(values, default=None):
    array = np.asarray([
        np.nan if value is None else float(value) for value in values
    ], float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if len(finite) else default


def _event_spans_ms(onsets: np.ndarray) -> np.ndarray:
    spans = np.full(len(onsets), np.nan, float)
    for index, row in enumerate(np.asarray(onsets, float)):
        finite = row[np.isfinite(row)]
        if len(finite) >= 2:
            spans[index] = float(np.max(finite) - np.min(finite))
    return spans


def pathway_objective(components: dict, weights: dict) -> float:
    return float(
        components["ood"]
        + float(weights["mode_fraction"]) * components["mode_fraction"]
        + float(weights["kmeans"]) * components["kmeans"]
        + float(weights["event_yield"]) * components["event_yield"]
        + float(weights["absolute_timing"]) * components["absolute_timing"]
    )


def nondominated_mask(matrix: np.ndarray) -> np.ndarray:
    """Return the minimization Pareto set, retaining exact ties."""
    values = np.asarray(matrix, float)
    output = np.ones(len(values), bool)
    for index, row in enumerate(values):
        dominates = np.all(values <= row, axis=1) & np.any(values < row, axis=1)
        dominates[index] = False
        output[index] = not bool(np.any(dominates))
    return output


def _patient_timing(target_path: Path) -> tuple[np.ndarray, float]:
    with np.load(target_path, allow_pickle=False) as loaded:
        onsets = np.asarray(loaded["patient_train_onsets"], float) * 1000.0
        labels = np.asarray(loaded["patient_train_old_labels"], int)
    spans = _event_spans_ms(onsets)
    medians = np.asarray([
        np.nanmedian(spans[labels == mode]) for mode in (0, 1)
    ], float)
    counts = np.bincount(labels, minlength=2)
    return medians, float(counts[1] / counts.sum())


def _network_row(
    npz_path: Path, *, seed: int, contract: dict, embedding: dict,
    classifier: dict, groups: dict,
) -> dict:
    score, events = _score_worker(npz_path, contract, embedding, classifier)
    labels = np.asarray([row["mode"] for row in events], int)
    support = np.asarray([row["in_support"] for row in events], bool)
    with np.load(npz_path, allow_pickle=False) as loaded:
        onsets = np.asarray(loaded["onsets"], float)
        ranks = np.asarray(loaded["ranks"], float)
    joint = (
        np.isfinite(onsets[:, groups["ICL"]]).any(axis=1)
        & np.isfinite(onsets[:, groups["SCL"]]).any(axis=1)
    )
    natural = natural_kmeans(
        ranks[support & joint], labels[support & joint],
        random_state=int(seed),
    )
    counts = np.bincount(labels[support], minlength=2)
    spans = _event_spans_ms(onsets)
    timing = [
        float(np.nanmedian(spans[support & (labels == mode)]))
        if np.any(np.isfinite(spans[support & (labels == mode)])) else None
        for mode in (0, 1)
    ]
    carrier = None
    with np.load(npz_path, allow_pickle=False) as loaded:
        if "carrier_time_ms" in loaded.files:
            carrier = _carrier_metrics(npz_path, events)
            carrier.pop("selected", None)
            carrier.pop("labels", None)
            carrier.pop("ranks", None)
    return {
        **score,
        "mode_counts_in_support": counts,
        "mode_2_fraction": float(counts[1] / counts.sum()) if counts.sum() else 0.0,
        "natural_kmeans": {
            key: value for key, value in natural.items()
            if key not in {"valid_event_mask", "cluster_labels"}
        },
        "median_recruitment_span_ms_by_mode": timing,
        "carrier": carrier,
    }


def _heatmap(ax, matrix, xlabels, ylabels, title, *, fmt=".2f", cmap="viridis"):
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_xlabel(r"$g_{EE}$")
    ax.set_ylabel(r"$g_{E\to I}$")
    ax.set_title(title, fontsize=8.5, weight="bold")
    finite = matrix[np.isfinite(matrix)]
    midpoint = float(np.median(finite)) if len(finite) else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column, row,
                "NA" if not np.isfinite(value) else format(value, fmt),
                ha="center", va="center", fontsize=5.8,
                color="white" if np.isfinite(value) and value > midpoint else "black",
            )
    return image


def aggregate(config_path: Path, phase: str = "screen") -> dict:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    for record in config["inputs"].values():
        path = ROOT / record["path"]
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen input changed: {path}")
    output_root = ROOT / config["output_root"]
    if phase in {"selection", "confirmation"}:
        output_root = output_root / phase
    elif phase != "screen":
        raise ValueError(f"unsupported pathway refit phase: {phase}")
    manifest_path = output_root / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "REV16_DUAL_CORE_OOD_PHASE_FROZEN"
        or manifest.get("config", {}).get("sha256") != _sha256(config_path)
    ):
        raise RuntimeError("pathway-refit manifest is stale")

    contract = json.loads(
        (ROOT / config["inputs"]["contact_contract"]["path"]).read_text()
    )
    target_path = ROOT / config["inputs"]["shaft_aware_target_npz"]["path"]
    embedding = load_embedding(str(target_path))
    groups = contract_groups(contract)
    classifier = manifest["direction_classifier"]
    patient_timing, empirical_mode_2 = _patient_timing(target_path)
    target_mode_2 = float(config["pathway_refit"]["patient_mode_2_fraction"])
    if not np.isclose(target_mode_2, empirical_mode_2, atol=5e-4):
        raise RuntimeError("registered patient mode fraction changed")

    per_network = []
    for candidate in manifest["candidate_set"]["candidates"]:
        for seed in manifest["fixed_contract"]["network_seeds"]:
            stem = output_root / "workers" / f"{candidate['candidate_id']}_seed_{seed}"
            worker_json = stem.with_suffix(".json")
            worker_npz = stem.with_suffix(".npz")
            payload = json.loads(worker_json.read_text())
            if payload.get("status") != "REV10R_EDGE_FLOW_WORKER_COMPLETE":
                raise RuntimeError(f"worker incomplete: {worker_json}")
            if payload["arrays"]["sha256"] != _sha256(worker_npz):
                raise RuntimeError(f"worker hash changed: {worker_npz}")
            per_network.append({
                "candidate_id": candidate["candidate_id"],
                "g_EE": candidate["search_coordinates"]["g_EE"],
                "g_EtoI": candidate["search_coordinates"]["g_EtoI"],
                "seed": int(seed),
                **_network_row(
                    worker_npz, seed=int(seed), contract=contract,
                    embedding=embedding, classifier=classifier, groups=groups,
                ),
                "worker_json": str(worker_json.relative_to(ROOT)),
                "worker_npz": str(worker_npz.relative_to(ROOT)),
            })

    node_id = "gee000_getoi000"
    node_rows = {
        row["seed"]: row for row in per_network if row["candidate_id"] == node_id
    }
    if len(node_rows) != len(manifest["fixed_contract"]["network_seeds"]):
        raise RuntimeError("zero-dose Node reference is incomplete")
    refit = config["pathway_refit"]
    summaries = []
    for candidate in manifest["candidate_set"]["candidates"]:
        rows = [
            row for row in per_network
            if row["candidate_id"] == candidate["candidate_id"]
        ]
        kmeans_values = [
            row["natural_kmeans"].get("direction_balanced_alignment")
            for row in rows
        ]
        kmeans = _finite_mean(
            kmeans_values, default=float(refit["missing_kmeans_alignment"]),
        )
        mode_2 = float(np.mean([row["mode_2_fraction"] for row in rows]))
        returned = float(np.mean([row["n_returned"] for row in rows]))
        paired_log_yield = float(np.mean([
            abs(np.log(
                (row["n_returned"] + 0.5)
                / (node_rows[row["seed"]]["n_returned"] + 0.5)
            )) for row in rows
        ]))
        model_timing = []
        for mode in (0, 1):
            model_timing.append(_finite_mean([
                row["median_recruitment_span_ms_by_mode"][mode] for row in rows
            ]))
        timing_penalty = (
            float(np.mean(np.abs(np.log(
                np.asarray(model_timing, float) / patient_timing
            )))) if all(value is not None and value > 0 for value in model_timing)
            else float(refit["missing_timing_penalty"])
        )
        components = {
            "ood": float(np.mean([row["ood_all_returned"] for row in rows])),
            "mode_fraction": abs(mode_2 - target_mode_2),
            "kmeans": 1.0 - float(kmeans),
            "event_yield": paired_log_yield,
            "absolute_timing": timing_penalty,
        }
        summaries.append({
            "candidate_id": candidate["candidate_id"],
            "g_EE": candidate["search_coordinates"]["g_EE"],
            "g_EtoI": candidate["search_coordinates"]["g_EtoI"],
            "n_networks": len(rows),
            "networks_with_both_modes": int(sum(
                row["both_modes_in_support"] for row in rows
            )),
            "n_networks_kmeans_evaluable": int(sum(
                value is not None and np.isfinite(float(value))
                for value in kmeans_values
            )),
            "equal_network_ood_all_returned": components["ood"],
            "equal_network_mode_2_fraction": mode_2,
            "natural_kmeans_alignment": float(kmeans),
            "equal_network_returned_events": returned,
            "median_recruitment_span_ms_by_mode": model_timing,
            "objective_components": components,
            "J_interictal": pathway_objective(
                components, refit["objective_weights"],
            ),
            "n_networks_carrier_evaluable": int(sum(
                row["carrier"] is not None
                and row["carrier"]["native_three_cycle_event_fraction"] is not None
                for row in rows
            )),
            "native_three_cycle_event_fraction": _finite_mean([
                None if row["carrier"] is None else
                row["carrier"]["native_three_cycle_event_fraction"]
                for row in rows
            ]),
            "per_network": rows,
        })
    component_matrix = np.asarray([
        list(row["objective_components"].values()) for row in summaries
    ], float)
    pareto = nondominated_mask(component_matrix)
    for row, is_pareto in zip(summaries, pareto):
        row["pareto_nondominated"] = bool(is_pareto)
    ranking = sorted(summaries, key=lambda row: (row["J_interictal"], row["candidate_id"]))
    registered_selection = list(
        manifest["fixed_contract"].get("selection_candidate_ids", [])
    )
    if phase in {"selection", "confirmation"}:
        selectable = [
            row for row in ranking if row["candidate_id"] in registered_selection
        ]
        if len(selectable) != 4:
            raise RuntimeError("selection manifest candidates are incomplete")
        shortlist = [row["candidate_id"] for row in selectable]
        frozen_work_point = shortlist[0]
    else:
        shortlist = [row["candidate_id"] for row in ranking[:4]]
        frozen_work_point = None
    output = {
        "status": f"DUAL_CORE_PATHWAY_REFIT_{phase.upper()}_AGGREGATED",
        "scientific_role": config["scientific_role"],
        "patient_reference": {
            "mode_2_fraction": target_mode_2,
            "median_recruitment_span_ms_by_mode": patient_timing,
        },
        "n_candidates": len(summaries),
        "n_networks_per_candidate": len(manifest["fixed_contract"]["network_seeds"]),
        "summaries": summaries,
        "ranking": [row["candidate_id"] for row in ranking],
        "selection_shortlist": shortlist,
        "frozen_work_point": frozen_work_point,
        "claim_boundary": config["claim_boundary"],
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha256(config_path)},
        "manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": _sha256(manifest_path)},
    }
    aggregate_path = output_root / "aggregate.json"
    aggregate_path.write_text(json.dumps(_jsonable(output), indent=2) + "\n")

    g_ee = list(map(float, refit["g_EE"]))
    g_etoi = list(map(float, refit["g_EtoI"]))
    metrics = [
        ("equal_network_ood_all_returned", "OOD", "magma_r"),
        ("equal_network_mode_2_fraction", "Mode 2 fraction", "viridis"),
        ("natural_kmeans_alignment", "Natural KMeans", "viridis"),
        ("equal_network_returned_events", "Returned events", "viridis"),
        ("timing", "Absolute timing error", "magma_r"),
        ("J_interictal", r"$J_{interictal}$", "magma_r"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.0), constrained_layout=True)
    for panel, (ax, (key, title, cmap)) in enumerate(zip(axes.flat, metrics)):
        matrix = np.full((len(g_etoi), len(g_ee)), np.nan)
        for row in summaries:
            x = g_ee.index(float(row["g_EE"]))
            y = g_etoi.index(float(row["g_EtoI"]))
            matrix[y, x] = (
                row["objective_components"]["absolute_timing"]
                if key == "timing" else row[key]
            )
        image = _heatmap(ax, matrix, g_ee, g_etoi, title, cmap=cmap)
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
        ax.text(-0.2, 1.09, chr(ord("A") + panel), transform=ax.transAxes,
                fontsize=10.5, weight="bold", va="top")
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    figure_stem = f"dual_core_pathway_refit_{phase}"
    fig.savefig(figure_root / f"{figure_stem}.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(figure_root / f"{figure_stem}.pdf",
                bbox_inches="tight")
    plt.close(fig)
    (figure_root / "README.md").write_text(
        f"### {figure_stem}.png\n\n"
        "冻结双 core Node 后，分别缩放既有 EE 与 E→I coefficient row。A--E 展示患者支持、"
        "模式占用、自然双簇、事件产率和绝对招募时间；F 是预注册连续开发分数。\n\n"
        "**关注点**：低 OOD 不能靠压低事件产率或破坏自然 KMeans 获得；本图不评价原生高频载波。\n"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase", choices=["screen", "selection", "confirmation"],
        default="screen",
    )
    args = parser.parse_args()
    output = aggregate(args.config, args.phase)
    print(json.dumps({
        "status": output["status"],
        "shortlist": output["selection_shortlist"],
    }, indent=2))


if __name__ == "__main__":
    main()
