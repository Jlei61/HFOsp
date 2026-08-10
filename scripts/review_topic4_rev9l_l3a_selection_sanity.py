"""Review the frozen L3a route candidate on selection networks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.aggregate_topic4_rev9l_component_pair_phase1 import (  # noqa: E402
    _candidate_summary,
    _load_workers,
    _patient,
)
from scripts.review_topic4_rev9l_l3_network_surrogate import (  # noqa: E402
    _provenance,
    _sha256,
    intended_source_distances,
)
from src.topic4_component_pair_search import DESCRIPTOR_NAMES, score_candidate  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402


DEFAULT_CONFIG = "config/topic4_rev9l_l3a_selection_sanity.json"


def _corrected_score(row, floor, base):
    source_for = {
        "A": base["primary_mapping"]["mode_A_source"],
        "B": base["primary_mapping"]["mode_B_source"],
    }
    readable = {
        mode: row["geometry"][source]["curve_usable_fraction"]
        for mode, source in source_for.items()
    }
    ood = {
        mode: row["geometry"][source]["ood_fraction"]
        for mode, source in source_for.items()
    }
    objective = base["objective"]
    return score_candidate(
        row["mode_descriptors"], floor["floor"], readable, ood,
        readable_weight=objective["readable_fraction_penalty_weight"],
        tau=objective["weakest_mode_lse_tau"],
        ood_weight=objective["ood_weight"])


def _route_row(path, source_for):
    with np.load(path, allow_pickle=False) as loaded:
        return intended_source_distances(
            loaded["source_ids"], loaded["assigned_distance_to_A_B"],
            source_for)


def _plot(payload, floor, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = payload["selection_network_seeds"]
    route = payload["route_surrogate_by_network"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), constrained_layout=True)
    x = np.arange(len(seeds))
    axes[0].plot(x, [route[str(seed)]["scalar"]["weak"] for seed in seeds],
                 "o-", color="0.35", label="scalar")
    axes[0].plot(x, [route[str(seed)]["candidate"]["weak"] for seed in seeds],
                 "D-", color="#d1495b", label=payload["candidate_id"])
    axes[0].set_xticks(x, seeds)
    axes[0].set_ylabel("single-event weak centroid distance")
    axes[0].set_title("A  Out-of-fit route sanity", loc="left", weight="bold")
    axes[0].legend(frameon=False, fontsize=8)

    baseline = payload["corrected_full_objective"]["scalar"]
    selected = payload["corrected_full_objective"]["candidate"]
    positions = np.arange(2)
    axes[1].bar(positions - 0.18,
                [baseline["mode_scores"][mode] for mode in ("A", "B")],
                0.36, color="0.65", label="scalar")
    axes[1].bar(positions + 0.18,
                [selected["mode_scores"][mode] for mode in ("A", "B")],
                0.36, color=["#d1495b", "#277da1"], label=payload["candidate_id"])
    axes[1].set_xticks(positions, ["mode A", "mode B"])
    axes[1].set_ylabel("floor-normalized mode score")
    axes[1].set_title("B  Full weakest-mode readout", loc="left", weight="bold")
    axes[1].legend(frameon=False, fontsize=8)

    mode = "A"
    x = np.arange(len(DESCRIPTOR_NAMES))
    baseline_raw = payload["mode_descriptors"]["scalar"][mode]
    candidate_raw = payload["mode_descriptors"]["candidate"][mode]
    floor_mode = floor["floor"]["modes"][mode]
    axes[2].plot(x, [baseline_raw[key] / floor_mode[key]["median"]
                     for key in DESCRIPTOR_NAMES], "o-", color="0.35",
                 label="scalar")
    axes[2].plot(x, [candidate_raw[key] / floor_mode[key]["median"]
                     for key in DESCRIPTOR_NAMES], "D-", color="#d1495b",
                 label=payload["candidate_id"])
    axes[2].axhline(1.0, color="0.6", ls="--", lw=0.8)
    axes[2].set_xticks(x, ["recruit", "precedence", "profile", "event cloud"],
                       rotation=25, ha="right")
    axes[2].set_ylabel("mode A distance / patient floor median")
    axes[2].set_title("C  Mode A remains the test", loc="left", weight="bold")
    axes[2].legend(frameon=False, fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"rev9l_l3a_selection_sanity.{suffix}", dpi=300)
    plt.close(fig)
    (output_dir / "README.md").write_text(
        "### rev9l_l3a_selection_sanity.png\n"
        "A 比较冻结 L3a 候选与 scalar edge 在三张 selection 网络上的单事件 route distance；B 使用计数匹配的 n=3 patient-training floor 比较完整 A/B mode score；C 展开 mode A 四项绝对距离。该图不读取 patient held-out，也不使用 confirmation seeds。\n\n"
        "**关注点**：fit 中的 route-surrogate 改善能否在新网络保留，并且是否同步改善完整 mode-A 分布读出。\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    sanity_path = Path(args.config)
    sanity = json.loads(sanity_path.read_text())
    for name, record in sanity["inputs"].items():
        if _sha256(record["path"]) != record["sha256"]:
            raise RuntimeError(f"L3a selection input changed: {name}")
    base_path = Path(sanity["inputs"]["component_pair_config"]["path"])
    base = json.loads(base_path.read_text())
    l3a = json.loads(Path(sanity["inputs"]["l3a_fit_surrogate"]["path"]).read_text())
    floor = json.loads(Path(sanity["inputs"]["selection_floor_n3"]["path"]).read_text())
    selection = json.loads(Path(sanity["inputs"]["selection_summary"]["path"]).read_text())
    l2_review = json.loads(Path(sanity["inputs"]["l2_scientific_review"]["path"]).read_text())
    candidate = sanity["candidate"]
    if (candidate["candidate_id"]
            != l3a["shared_tie_break_candidate"]["candidate_id"]
            or floor["n_events_per_mode_per_draw"] != 3
            or floor.get("patient_heldout_scores_computed") is not False):
        raise RuntimeError("L3a selection freeze or floor is invalid")

    stage = sanity["output_stage"]
    workers, contact_names, worker_inputs = _load_workers(
        base, base_path, args.expected_commit,
        candidates=[candidate], stage=stage,
        seeds=sanity["network_seeds"])
    reference, patient, patient_labels, prototypes = _patient(base, contact_names)
    candidate_row = _candidate_summary(
        workers[candidate["candidate_id"]], base, reference, patient,
        patient_labels, prototypes)
    corrected = _corrected_score(candidate_row, floor, base)
    baseline_score = l2_review["corrected_objective"]["baseline"]
    baseline_id = "sobol_000"
    source_for = {
        "A": base["primary_mapping"]["mode_A_source"],
        "B": base["primary_mapping"]["mode_B_source"],
    }
    selection_worker = {
        (row["candidate_id"], int(row["seed"])): row
        for row in selection["worker_inputs"]
    }
    route = {}
    for seed in sanity["network_seeds"]:
        candidate_npz = Path(base["output_root"]) / stage / "workers" / (
            f"{candidate['candidate_id']}_seed{seed}.npz")
        baseline_record = selection_worker[(baseline_id, int(seed))]["npz"]
        if _sha256(baseline_record["path"]) != baseline_record["sha256"]:
            raise RuntimeError("selection scalar worker hash changed")
        route[str(seed)] = {
            "scalar": _route_row(baseline_record["path"], source_for),
            "candidate": _route_row(candidate_npz, source_for),
        }
    improvement = float(baseline_score["objective"] - corrected["objective"])
    descriptor_modes = {
        "candidate": candidate_row["mode_descriptors"]["modes"],
        "scalar": {
            mode: {
                name: baseline_score["standardized_descriptors"][mode][name]["raw"]
                for name in DESCRIPTOR_NAMES
            }
            for mode in ("A", "B")
        },
    }
    payload = {
        "status": "L3A_SELECTION_SANITY_COMPLETE",
        "scientific_role": (
            "out-of-fit single-event route surrogate sanity plus full n=3 "
            "training-only mode readout"),
        "candidate_id": candidate["candidate_id"],
        "gamma": candidate["gamma"],
        "selection_network_seeds": list(map(int, sanity["network_seeds"])),
        "route_surrogate_by_network": route,
        "n_networks_route_weak_improved_vs_scalar": int(sum(
            rows["candidate"]["weak"] < rows["scalar"]["weak"] - 1e-12
            for rows in route.values())),
        "corrected_full_objective": {
            "candidate": corrected,
            "scalar": baseline_score,
            "scalar_minus_candidate": improvement,
            "relative_improvement": float(improvement / baseline_score["objective"]),
        },
        "mode_descriptors": descriptor_modes,
        "shared_distribution_capacity_status": (
            "NOT_ESTABLISHED_THREE_NETWORKS_ONE_EVENT_PER_SOURCE"),
        "patient_heldout_scores_computed": False,
        "inputs": {
            "sanity_config": {
                "path": str(sanity_path), "sha256": _sha256(sanity_path)},
            "worker_inputs": worker_inputs,
        },
        "provenance": _provenance(args.expected_commit),
    }
    output_dir = Path(base["output_root"]) / stage
    output_path = output_dir / "l3a_selection_sanity.json"
    atomic_write_json(payload, output_path)
    _plot(payload, floor, output_dir / "figures")

    decision_path = Path(base["output_root"]).parent / "decision.json"
    decision = json.loads(decision_path.read_text())
    decision["status"] = payload["status"]
    decision["network_realization"]["selection_sanity"] = {
        "candidate_id": candidate["candidate_id"],
        "route_networks_improved": payload[
            "n_networks_route_weak_improved_vs_scalar"],
        "route_networks_total": len(sanity["network_seeds"]),
        "full_objective_improvement_vs_scalar": improvement,
        "formal_capacity_status": payload["shared_distribution_capacity_status"],
    }
    decision["patient_heldout_scores_computed"] = False
    decision["l3a_selection_sanity_provenance"] = payload["provenance"]
    atomic_write_json(decision, decision_path)
    print(json.dumps({
        "status": payload["status"],
        "candidate_id": candidate["candidate_id"],
        "route_networks_improved": payload[
            "n_networks_route_weak_improved_vs_scalar"],
        "route_networks_total": len(sanity["network_seeds"]),
        "full_objective_improvement_vs_scalar": improvement,
        "relative_improvement": payload[
            "corrected_full_objective"]["relative_improvement"],
        "patient_heldout_scores_computed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
