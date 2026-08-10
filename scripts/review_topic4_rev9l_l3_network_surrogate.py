"""Audit per-network route-shape capacity from existing rev9-L L2 events."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "config/topic4_rev9l_component_pair_edge.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _provenance(expected_commit):
    paths = set()
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py":
            continue
        try:
            paths.add(str(path.relative_to(ROOT)))
        except ValueError:
            continue
    paths.add(str(Path(__file__).resolve().relative_to(ROOT)))
    paths = sorted(paths)
    expected = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True).strip()
    current = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT,
        text=True).strip()
    hashes = {path: _sha256(ROOT / path) for path in paths}
    expected_hashes = {
        path: hashlib.sha256(subprocess.check_output(
            ["git", "show", f"{expected}:{path}"], cwd=ROOT)).hexdigest()
        for path in paths
    }
    if (dirty or current != expected
            or any(hashes[path] != expected_hashes[path] for path in paths)):
        raise RuntimeError("L3a surrogate producer differs from expected commit")
    return {
        "git_commit": current,
        "expected_git_commit": expected,
        "runtime_modules_dirty": False,
        "runtime_module_sha256": hashes,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def intended_source_distances(source_ids, distance_matrix, source_for):
    source_ids = np.asarray(source_ids).astype(str)
    distances = np.asarray(distance_matrix, float)
    if distances.shape != (len(source_ids), 2):
        raise ValueError("source-mode distance matrix must have two patient modes")
    output = {}
    for mode, mode_index in (("A", 0), ("B", 1)):
        selected = np.flatnonzero(source_ids == source_for[mode])
        if len(selected) != 1:
            raise ValueError(f"intended source for mode {mode} is not unique")
        value = float(distances[selected[0], mode_index])
        if not np.isfinite(value):
            raise ValueError(f"non-finite intended-source distance for mode {mode}")
        output[mode] = value
    output["weak"] = max(output["A"], output["B"])
    return output


def oracle_summary(values, full_objective, *, tolerance=1e-12):
    candidates = sorted(values)
    seeds = sorted(next(iter(values.values())))
    per_network = []
    for seed in seeds:
        minimum = min(values[candidate][seed]["weak"] for candidate in candidates)
        tied = [
            candidate for candidate in candidates
            if abs(values[candidate][seed]["weak"] - minimum) <= tolerance
        ]
        selected = min(tied, key=lambda candidate: (
            full_objective[candidate], candidate))
        per_network.append({
            "seed": int(seed),
            "minimum_weak_distance": float(minimum),
            "tied_candidate_ids": tied,
            "tie_break_candidate_id": selected,
        })
    candidate_medians = {
        candidate: float(np.median([
            values[candidate][seed]["weak"] for seed in seeds
        ]))
        for candidate in candidates
    }
    shared_minimum = min(candidate_medians.values())
    shared_ties = [
        candidate for candidate in candidates
        if abs(candidate_medians[candidate] - shared_minimum) <= tolerance
    ]
    shared_selected = min(shared_ties, key=lambda candidate: (
        full_objective[candidate], candidate))
    per_network_capacity = float(np.median([
        row["minimum_weak_distance"] for row in per_network
    ]))
    return {
        "per_network": per_network,
        "C_per_net_1": per_network_capacity,
        "candidate_median_weak_distance": candidate_medians,
        "shared": {
            "C_shared_1": float(shared_minimum),
            "tied_candidate_ids": shared_ties,
            "n_tied_candidates": len(shared_ties),
            "tie_break_rule": (
                "lowest frozen L2 full fit objective, then candidate id"),
            "tie_break_candidate_id": shared_selected,
        },
        "Delta_network_1": float(shared_minimum - per_network_capacity),
    }


def _plot(payload, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    values = payload["candidate_network_distances"]
    seeds = payload["fit_network_seeds"]
    oracle = payload["single_event_oracle_surrogate"]
    baseline_id = payload["scalar_baseline_id"]
    shared_id = oracle["shared"]["tie_break_candidate_id"]
    l2_id = payload["l2_selection_candidate_id"]
    per_network = {
        row["seed"]: row for row in oracle["per_network"]
    }

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.3), constrained_layout=True)
    x = np.arange(len(seeds))
    axes[0].plot(x, [values[baseline_id][str(seed)]["weak"] for seed in seeds],
                 "o-", color="0.35", label="scalar")
    axes[0].plot(x, [per_network[seed]["minimum_weak_distance"] for seed in seeds],
                 "D-", color="#d1495b", label="per-network best")
    axes[0].plot(x, [values[shared_id][str(seed)]["weak"] for seed in seeds],
                 "s-", color="#277da1", label=f"shared tie-break {shared_id[-3:]}")
    axes[0].set_xticks(x, seeds, rotation=30)
    axes[0].set_ylabel("intended-source weak centroid distance")
    axes[0].set_title("A  Per-network route surrogate", loc="left", weight="bold")
    axes[0].legend(frameon=False, fontsize=7)

    rows = payload["candidate_summary"]
    scatter = axes[1].scatter(
        [row["median_weak_distance"] for row in rows],
        [row["full_fit_objective"] for row in rows],
        c=[row["n_networks_improved_vs_scalar"] for row in rows],
        cmap="viridis", vmin=0, vmax=len(seeds), s=30)
    by_id = {row["candidate_id"]: row for row in rows}
    for candidate_id, marker, color in (
            (baseline_id, "o", "0.2"), (l2_id, "D", "#d1495b"),
            (shared_id, "*", "#277da1")):
        row = by_id[candidate_id]
        axes[1].scatter(row["median_weak_distance"], row["full_fit_objective"],
                        marker=marker, color=color, s=110, edgecolor="white",
                        linewidth=0.7)
        axes[1].annotate(candidate_id[-3:],
                         (row["median_weak_distance"], row["full_fit_objective"]),
                         xytext=(3, 3), textcoords="offset points", fontsize=7)
    axes[1].set_xlabel("median single-event weak distance")
    axes[1].set_ylabel("L2 full fit objective")
    axes[1].set_title("B  Surrogate versus full objective", loc="left", weight="bold")
    fig.colorbar(scatter, ax=axes[1], shrink=0.78,
                 label="networks improved vs scalar")

    shown = list(dict.fromkeys(
        [baseline_id, l2_id, shared_id]
        + [per_network[seed]["tie_break_candidate_id"] for seed in seeds]))
    matrix = np.asarray([
        [values[candidate_id][str(seed)]["weak"] for seed in seeds]
        for candidate_id in shown
    ])
    image = axes[2].imshow(matrix, cmap="magma", aspect="auto")
    axes[2].set_yticks(range(len(shown)), [row[-3:] for row in shown])
    axes[2].set_xticks(range(len(seeds)), seeds, rotation=30)
    axes[2].set_xlabel("network seed")
    axes[2].set_ylabel("candidate")
    axes[2].set_title("C  Quantized route responses", loc="left", weight="bold")
    fig.colorbar(image, ax=axes[2], shrink=0.78, label="weak distance")

    gamma = np.asarray([by_id[candidate_id]["gamma"] for candidate_id in shown])
    scale = max(float(np.max(np.abs(gamma))), 1e-12)
    image = axes[3].imshow(
        gamma, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
    axes[3].set_yticks(range(len(shown)), [row[-3:] for row in shown])
    axes[3].set_xticks(
        range(6), ("C1<-C1", "C1<-C2", "C2<-C1", "C2<-C2",
                   "BG<-C1", "BG<-C2"), rotation=30, ha="right", fontsize=7)
    axes[3].set_title("D  Candidate residuals", loc="left", weight="bold")
    fig.colorbar(image, ax=axes[3], shrink=0.78, label="gamma")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"rev9l_l3a_network_surrogate.{suffix}", dpi=300)
    plt.close(fig)
    (output_dir / "README.md").write_text(
        "### rev9l_l3a_network_surrogate.png\n"
        "A 比较 scalar、逐网络最优和 shared 并列集合二级候选的单事件 "
        "intended-source 距离；B 显示该 surrogate 与完整 L2 objective 并不等价；"
        "C 展示不同网络上的离散 route response；D 给出所示候选的六个 edge "
        "residual。该图只重放 fit workers，不读取 patient held-out。\n\n"
        "**关注点**：每张网络能否被某个候选移动、同一候选能否跨网络稳定，以及大量并列是否阻止唯一参数解释。\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    root = Path(config["output_root"])
    fit_path = root / "sobol_fit/sobol_fit_summary.json"
    review_path = root / "scientific_review/l2_scientific_review.json"
    fit = json.loads(fit_path.read_text())
    review = json.loads(review_path.read_text())
    if (fit["status"] != "REV9L_L2_SOBOL_FIT_COMPLETE"
            or review["status"]
            != "L2_COMPONENT_PAIR_SEARCH_NO_SHARED_MODE_A_RESTORATION"):
        raise RuntimeError("L3a requires completed L2 fit and scientific review")
    if (fit.get("patient_heldout_scores_computed") is not False
            or review.get("patient_heldout_scores_computed") is not False):
        raise RuntimeError("L3a cannot consume patient held-out scores")

    seeds = list(map(int, fit["network_seeds"]))
    source_for = {
        "A": config["primary_mapping"]["mode_A_source"],
        "B": config["primary_mapping"]["mode_B_source"],
    }
    fit_by = {row["candidate_id"]: row for row in fit["candidates"]}
    eligible = sorted(
        candidate_id for candidate_id, row in fit_by.items()
        if row["score"].get("eligible"))
    worker_by = {
        (row["candidate_id"], int(row["seed"])): row
        for row in fit["worker_inputs"]
    }
    values = {}
    manifest = hashlib.sha256()
    for candidate_id in eligible:
        values[candidate_id] = {}
        for seed in seeds:
            record = worker_by[candidate_id, seed]["npz"]
            path = Path(record["path"])
            digest = _sha256(path)
            if digest != record["sha256"]:
                raise RuntimeError(f"L3a worker hash changed: {path}")
            manifest.update(f"{path}:{digest}\n".encode())
            with np.load(path, allow_pickle=False) as loaded:
                row = intended_source_distances(
                    loaded["source_ids"], loaded["assigned_distance_to_A_B"],
                    source_for)
                source_ids = np.asarray(loaded["source_ids"]).astype(str)
                ood = np.asarray(loaded["assigned_ood"], bool)
                row["classifier_ood"] = {
                    mode: bool(ood[np.flatnonzero(
                        source_ids == source_for[mode])[0]])
                    for mode in ("A", "B")
                }
            values[candidate_id][seed] = row

    full_objective = {
        candidate_id: float(fit_by[candidate_id]["score"]["objective"])
        for candidate_id in eligible
    }
    oracle = oracle_summary(values, full_objective)
    baseline_id = "sobol_000"
    baseline = values[baseline_id]
    candidates = []
    for candidate_id in eligible:
        weak = [values[candidate_id][seed]["weak"] for seed in seeds]
        candidates.append({
            "candidate_id": candidate_id,
            "gamma": fit_by[candidate_id]["gamma"],
            "full_fit_objective": full_objective[candidate_id],
            "median_A_distance": float(np.median([
                values[candidate_id][seed]["A"] for seed in seeds])),
            "median_B_distance": float(np.median([
                values[candidate_id][seed]["B"] for seed in seeds])),
            "median_weak_distance": float(np.median(weak)),
            "n_networks_improved_vs_scalar": int(sum(
                values[candidate_id][seed]["weak"]
                < baseline[seed]["weak"] - 1e-12 for seed in seeds)),
            "n_source_events_classifier_ood": int(sum(
                values[candidate_id][seed]["classifier_ood"][mode]
                for seed in seeds for mode in ("A", "B"))),
        })
    shared_id = oracle["shared"]["tie_break_candidate_id"]
    selection_ids = review["corrected_selection_rank"]
    payload = {
        "status": "L3A_SINGLE_EVENT_ROUTE_SURROGATE_COMPLETE",
        "scientific_role": (
            "zero-simulation single-event route-shape diagnostic; not a "
            "distribution-level capacity oracle"),
        "safe_claim": (
            "the finite fit library contains per-network route-shape responses "
            "closer to the intended patient centroid than scalar edge, but the "
            "shared optimum is highly tied and does not establish shared "
            "distribution-level mode capacity"),
        "metric": (
            "max intended-source Euclidean distance to frozen patient-training "
            "mode centroid in the frozen PCA space"),
        "fit_network_seeds": seeds,
        "n_eligible_candidates": len(eligible),
        "scalar_baseline_id": baseline_id,
        "l2_selection_candidate_id": review["corrected_selected_candidate_id"],
        "single_event_oracle_surrogate": oracle,
        "shared_tie_break_candidate": {
            "candidate_id": shared_id,
            "gamma": fit_by[shared_id]["gamma"],
            "already_selection_evaluated": shared_id in selection_ids,
        },
        "candidate_summary": candidates,
        "candidate_network_distances": {
            candidate_id: {str(seed): row for seed, row in rows.items()}
            for candidate_id, rows in values.items()
        },
        "formal_capacity_status": (
            "NOT_ESTABLISHED_ONE_EVENT_PER_SOURCE_NETWORK_NO_DISTRIBUTION"),
        "optimizer_status": "NOT_ATTRIBUTABLE_NO_KNOWN_GOOD_SHARED_SOLUTION",
        "next_recommendation": (
            "RUN_FROZEN_SHARED_SURROGATE_TIE_BREAK_ON_SELECTION_NETWORKS"
            if shared_id not in selection_ids
            else "DESIGN_REPEATED_EVENT_L3B_ORACLE"),
        "patient_heldout_scores_computed": False,
        "inputs": {
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "fit": {"path": str(fit_path), "sha256": _sha256(fit_path)},
            "l2_review": {
                "path": str(review_path), "sha256": _sha256(review_path)},
            "verified_worker_npz_count": len(eligible) * len(seeds),
            "worker_npz_manifest_sha256": manifest.hexdigest(),
        },
        "provenance": _provenance(args.expected_commit),
    }
    output_dir = root / "network_oracle_surrogate"
    output_path = output_dir / "l3a_network_surrogate.json"
    atomic_write_json(payload, output_path)
    _plot(payload, output_dir / "figures")

    decision_path = root.parent / "decision.json"
    decision = json.loads(decision_path.read_text())
    decision["status"] = payload["status"]
    decision["network_realization"] = {
        "status": "SINGLE_EVENT_SURROGATE_SHARED_GAP_OBSERVED",
        "C_per_net_1": oracle["C_per_net_1"],
        "C_shared_1": oracle["shared"]["C_shared_1"],
        "Delta_network_1": oracle["Delta_network_1"],
        "shared_tied_candidate_count": oracle["shared"]["n_tied_candidates"],
        "shared_tie_break_candidate_id": shared_id,
        "formal_per_network_oracle_status": payload["formal_capacity_status"],
        "claim_boundary": (
            "single forced events diagnose route shape but not mode distributions"),
    }
    decision["optimizer"] = {
        "status": payload["optimizer_status"],
        "reason": "L3a is not a known-good distribution-level shared solution",
    }
    decision["patient_heldout_scores_computed"] = False
    decision["l3a_network_surrogate_provenance"] = payload["provenance"]
    atomic_write_json(decision, decision_path)
    print(json.dumps({
        "status": payload["status"],
        "C_per_net_1": oracle["C_per_net_1"],
        "C_shared_1": oracle["shared"]["C_shared_1"],
        "Delta_network_1": oracle["Delta_network_1"],
        "shared_ties": oracle["shared"]["n_tied_candidates"],
        "shared_tie_break_candidate": shared_id,
        "already_selection_evaluated": shared_id in selection_ids,
        "patient_heldout_scores_computed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
