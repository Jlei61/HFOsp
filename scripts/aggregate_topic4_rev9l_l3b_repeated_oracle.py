"""Aggregate repeated dynamics events into rev9-L per-network oracles."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.aggregate_topic4_rev9l_component_pair_phase1 import (  # noqa: E402
    _candidate_summary,
    _patient,
)
from scripts.review_topic4_rev9l_l3_network_surrogate import (  # noqa: E402
    _provenance,
    _sha256,
)
from src.topic4_component_pair_search import score_candidate  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_repeated_network_oracle import summarize_network_oracles  # noqa: E402


DEFAULT_CONFIG = "config/topic4_rev9l_l3b_repeated_oracle.json"


def _load_group(base, base_path, config, candidate, network_seed, expected_commit):
    root = Path(base["output_root"]) / config["output_stage"]
    base_sha = _sha256(base_path)
    rows, arrays, diagnostics, inputs = [], [], [], []
    contact_names = None
    network_contract = None
    for dynamics_seed in config["dynamics_seeds"]:
        stem = root / "workers" / (
            f"{candidate['candidate_id']}_net{network_seed}_dyn{dynamics_seed}")
        json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
        payload = json.loads(json_path.read_text())
        provenance = payload["provenance"]
        if (payload["status"] != "REV9L_FORCED_SOURCE_WORKER_COMPLETE"
                or payload["scientific_role"] != base["scientific_role"]
                or payload["edge_family"]
                != "component_pair_residual_target_normalized"
                or payload["candidate_id"] != candidate["candidate_id"]
                or int(payload["network_seed"]) != int(network_seed)
                or int(payload["dynamics_seed"]) != int(dynamics_seed)
                or not np.allclose(payload["component_pair_gamma"], candidate["gamma"])
                or payload["config"]["sha256"] != base_sha
                or provenance.get("expected_git_commit") != expected_commit
                or provenance.get("runtime_modules_dirty")
                or not provenance.get("runtime_modules_match_expected_commit")
                or payload["arrays"]["sha256"] != _sha256(npz_path)):
            raise RuntimeError(f"invalid L3b worker: {json_path}")
        contract = {
            "node_hashes": payload["network"]["node_hashes"],
            "edge_diagnostics": payload["network"]["edge_diagnostics"],
            "n_E": payload["network"]["n_E"],
            "n_I": payload["network"]["n_I"],
        }
        contract = hashlib.sha256(json.dumps(
            contract, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if network_contract is None:
            network_contract = contract
        elif contract != network_contract:
            raise RuntimeError(
                f"network or substrate changed across dynamics repeats: {json_path}")
        with np.load(npz_path, allow_pickle=False) as loaded:
            names = np.asarray(loaded["contact_names"]).astype(str)
            if contact_names is None:
                contact_names = names
            elif not np.array_equal(contact_names, names):
                raise RuntimeError("L3b contact order changed")
            record = {key: np.asarray(loaded[key]) for key in (
                "source_ids", "rank_curves", "contact_ranks", "assigned_ood",
                "assigned_distance_to_A_B",
            )}
        for index, run in enumerate(payload["runs"]):
            if str(record["source_ids"][index]) != run["source_id"]:
                raise RuntimeError("L3b JSON/NPZ source order changed")
            rows.append({
                "seed": int(network_seed), "dynamics_seed": int(dynamics_seed),
                **run,
            })
            arrays.append({
                key: value[index] for key, value in record.items()
                if key != "source_ids"
            })
        diagnostics.append(payload["network"]["edge_diagnostics"])
        inputs.append({
            "candidate_id": candidate["candidate_id"],
            "network_seed": int(network_seed),
            "dynamics_seed": int(dynamics_seed),
            "json": {"path": str(json_path), "sha256": _sha256(json_path)},
            "npz": {"path": str(npz_path), "sha256": _sha256(npz_path)},
        })
    return {
        "gamma": candidate["gamma"], "rows": rows, "arrays": arrays,
        "edge_diagnostics": diagnostics,
    }, contact_names, inputs


def _score(row, floors, base, *, failure_objective, minimum_usable=2):
    if row["mode_descriptors"] is None:
        return {
            "objective": float(failure_objective),
            "mode_scores": {
                "A": float(failure_objective),
                "B": float(failure_objective),
            },
            "weak_mode": None,
            "readout_failure": True,
            "failure_reason": "one or both modes have no usable repeated event",
        }
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
    counts = {
        mode: int(row["geometry"][source]["n_curves_usable"])
        for mode, source in source_for.items()
    }
    if any(count < int(minimum_usable) or count not in floors
           for count in counts.values()):
        return {
            "objective": float(failure_objective),
            "mode_scores": {
                "A": float(failure_objective),
                "B": float(failure_objective),
            },
            "weak_mode": None,
            "readout_failure": True,
            "failure_reason": (
                "one or both modes lack a supported count-matched patient floor"),
            "matched_floor_event_count_by_mode": counts,
        }
    matched_floor = {
        "modes": {
            mode: floors[counts[mode]]["floor"]["modes"][mode]
            for mode in ("A", "B")
        }
    }
    objective = base["objective"]
    result = score_candidate(
        row["mode_descriptors"], matched_floor, readable, ood,
        readable_weight=objective["readable_fraction_penalty_weight"],
        tau=objective["weakest_mode_lse_tau"],
        ood_weight=objective["ood_weight"])
    if not np.isfinite(result["objective"]):
        return {
            "objective": float(failure_objective),
            "mode_scores": {
                "A": float(failure_objective),
                "B": float(failure_objective),
            },
            "weak_mode": None,
            "readout_failure": True,
            "failure_reason": "non-finite repeated-event descriptor",
        }
    result["readout_failure"] = False
    result["matched_floor_event_count_by_mode"] = counts
    result["floor_policy"] = (
        "mode-specific actual-readable-count patient-training floor")
    return result


def _plot(payload, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = payload["network_seeds"]
    oracle = payload["oracle"]
    values = payload["objective_by_candidate_network"]
    baseline_id = "sobol_000"
    shared_id = oracle["shared"]["selected_candidate_id"]
    per_network = {row["network_seed"]: row for row in oracle["per_network"]}
    summaries = {row["candidate_id"]: row for row in payload["candidate_summary"]}
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.3), constrained_layout=True)

    x = np.arange(len(seeds))
    axes[0].plot(x, [values[baseline_id][str(seed)] for seed in seeds],
                 "o-", color="0.35", label="scalar")
    axes[0].plot(x, [per_network[seed]["minimum_objective"] for seed in seeds],
                 "D-", color="#d1495b", label="per-network oracle")
    axes[0].plot(x, [values[shared_id][str(seed)] for seed in seeds],
                 "s-", color="#277da1", label=f"shared {shared_id[-3:]}")
    axes[0].set_xticks(x, seeds, rotation=30)
    axes[0].set_ylabel("count-matched weakest-mode objective")
    axes[0].set_title("A  Repeated-event capacity", loc="left", weight="bold")
    axes[0].legend(frameon=False, fontsize=7)

    rows = payload["candidate_summary"]
    scatter = axes[1].scatter(
        [row["median_mode_A_score"] for row in rows],
        [row["median_mode_B_score"] for row in rows],
        c=[row["median_objective"] for row in rows], cmap="magma", s=30)
    for candidate_id, marker, color in (
            (baseline_id, "o", "0.25"), (shared_id, "*", "#277da1")):
        row = summaries[candidate_id]
        axes[1].scatter(row["median_mode_A_score"], row["median_mode_B_score"],
                        marker=marker, color=color, s=120, edgecolor="white")
        axes[1].annotate(candidate_id[-3:],
                         (row["median_mode_A_score"], row["median_mode_B_score"]),
                         xytext=(3, 3), textcoords="offset points", fontsize=7)
    axes[1].set_xlabel("median mode A score")
    axes[1].set_ylabel("median mode B score")
    axes[1].set_title("B  Weak-mode landscape", loc="left", weight="bold")
    fig.colorbar(scatter, ax=axes[1], shrink=0.78, label="median objective")

    ranked = sorted(rows, key=lambda row: row["median_objective"])
    shown = list(dict.fromkeys(
        [baseline_id, shared_id]
        + [row["representative_candidate_id"] for row in oracle["per_network"]]
        + [row["candidate_id"] for row in ranked[:4]]))
    matrix = np.asarray([
        [values[candidate_id][str(seed)] for seed in seeds]
        for candidate_id in shown
    ])
    image = axes[2].imshow(matrix, cmap="viridis", aspect="auto")
    axes[2].set_yticks(range(len(shown)), [row[-3:] for row in shown])
    axes[2].set_xticks(range(len(seeds)), seeds, rotation=30)
    axes[2].set_title("C  Candidate x network", loc="left", weight="bold")
    fig.colorbar(image, ax=axes[2], shrink=0.78, label="objective")

    gamma = np.asarray([summaries[candidate_id]["gamma"] for candidate_id in shown])
    scale = max(float(np.max(np.abs(gamma))), 1e-12)
    image = axes[3].imshow(
        gamma, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
    axes[3].set_yticks(range(len(shown)), [row[-3:] for row in shown])
    axes[3].set_xticks(
        range(6), ("C1<-C1", "C1<-C2", "C2<-C1", "C2<-C2",
                   "BG<-C1", "BG<-C2"), rotation=30, ha="right", fontsize=7)
    axes[3].set_title("D  Edge residuals", loc="left", weight="bold")
    fig.colorbar(image, ax=axes[3], shrink=0.78, label="gamma")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"rev9l_l3b_repeated_network_oracle.{suffix}", dpi=300)
    plt.close(fig)
    (output_dir / "README.md").write_text(
        "### rev9l_l3b_repeated_network_oracle.png\n"
        "A 用每张固定网络上的三个独立 dynamics repeats 比较 scalar、逐网络 oracle 和 shared candidate；B 展示完整 A/B mode score；C 显示候选跨网络 objective；D 给出对应 edge residual。全部指标只使用 patient training，并按每个 mode 实际可读的 2 或 3 个事件匹配 floor。\n\n"
        "**关注点**：逐网络最优是否显著优于同一 shared 参数，以及 shared 改善是否同时保护 mode A 和 mode B。\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    for name, record in config["inputs"].items():
        if _sha256(record["path"]) != record["sha256"]:
            raise RuntimeError(f"L3b input changed: {name}")
    base_path = Path(config["inputs"]["component_pair_config"]["path"])
    base = json.loads(base_path.read_text())
    fit = json.loads(Path(config["inputs"]["l2_fit_summary"]["path"]).read_text())
    floors = {}
    for count in config["patient_floor_event_counts"]:
        floor = json.loads(Path(
            config["inputs"][f"patient_floor_n{count}"]["path"]).read_text())
        if (floor["n_events_per_mode_per_draw"] != int(count)
                or floor.get("patient_heldout_scores_computed") is not False):
            raise RuntimeError(f"L3b patient floor n={count} is invalid")
        floors[int(count)] = floor
    candidates = [
        {"candidate_id": row["candidate_id"], "gamma": row["gamma"]}
        for row in fit["candidates"] if row["score"].get("eligible")
    ]
    worker_expected = subprocess.check_output(
        ["git", "rev-parse", config["worker_expected_commit"]],
        cwd=Path(__file__).resolve().parents[1], text=True).strip()
    patient_state = None
    rows, objective, worker_inputs = [], {}, []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        objective[candidate_id] = {}
        for network_seed in config["network_seeds"]:
            data, contact_names, inputs = _load_group(
                base, base_path, config, candidate, network_seed,
                worker_expected)
            worker_inputs.extend(inputs)
            if patient_state is None:
                patient_state = _patient(base, contact_names)
            reference, patient, patient_labels, prototypes = patient_state
            row = _candidate_summary(
                data, base, reference, patient, patient_labels, prototypes)
            score = _score(
                row, floors, base,
                failure_objective=float(config["readout_failure_objective"]),
                minimum_usable=int(config["minimum_usable_events_per_mode"]))
            objective[candidate_id][int(network_seed)] = score["objective"]
            rows.append({
                "candidate_id": candidate_id,
                "network_seed": int(network_seed),
                "score": score,
                "mode_descriptors": row["mode_descriptors"],
                "geometry": row["geometry"],
                "structural": row["structural"],
                "n_runaway": row["n_runaway"],
                "n_pretrigger_mismatch": row["n_pretrigger_mismatch"],
            })
    oracle = summarize_network_oracles(objective)
    by_candidate = {}
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        selected = [row for row in rows if row["candidate_id"] == candidate_id]
        by_candidate[candidate_id] = {
            "candidate_id": candidate_id,
            "gamma": candidate["gamma"],
            "median_objective": float(np.median([
                row["score"]["objective"] for row in selected])),
            "mean_objective": float(np.mean([
                row["score"]["objective"] for row in selected])),
            "median_mode_A_score": float(np.median([
                row["score"]["mode_scores"]["A"] for row in selected])),
            "median_mode_B_score": float(np.median([
                row["score"]["mode_scores"]["B"] for row in selected])),
            "n_readout_failure_networks": int(sum(
                row["score"]["readout_failure"] for row in selected)),
            "n_runaway": int(sum(row["n_runaway"] for row in selected)),
            "n_pretrigger_mismatch": int(sum(
                row["n_pretrigger_mismatch"] for row in selected)),
        }
    payload = {
        "status": "L3B_REPEATED_FIT_ORACLE_COMPLETE",
        "scientific_role": (
            "finite-library repeated-event fit-network capacity audit; no "
            "selection, confirmation, or patient held-out"),
        "network_seeds": list(map(int, config["network_seeds"])),
        "dynamics_seeds": list(map(int, config["dynamics_seeds"])),
        "events_per_mode_per_network": len(config["dynamics_seeds"]),
        "floor_policy": config["floor_policy"],
        "patient_floor_event_counts": list(map(
            int, config["patient_floor_event_counts"])),
        "n_candidates": len(candidates),
        "n_workers": len(worker_inputs),
        "oracle": oracle,
        "candidate_summary": list(by_candidate.values()),
        "candidate_network_rows": rows,
        "objective_by_candidate_network": {
            candidate: {str(seed): value for seed, value in values.items()}
            for candidate, values in objective.items()
        },
        "runaway_is_reported_not_hidden": True,
        "formal_scope": (
            "three attempted dynamics repeats per source/network with mode-wise "
            "count-matched training floors are a minimum repeated-event "
            "exploratory oracle, not a high-precision capacity ceiling"),
        "optimizer_status": "NOT_TESTED_PENDING_SHARED_SELECTION_SANITY",
        "patient_heldout_scores_computed": False,
        "inputs": {
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "worker_inputs": worker_inputs,
            "worker_expected_commit": worker_expected,
        },
        "provenance": _provenance(args.expected_commit),
    }
    output_dir = Path(base["output_root"]) / config["output_stage"]
    output_path = output_dir / "l3b_repeated_fit_oracle.json"
    atomic_write_json(payload, output_path)
    _plot(payload, output_dir / "figures")
    decision_path = Path(base["output_root"]).parent / "decision.json"
    decision = json.loads(decision_path.read_text())
    decision["status"] = payload["status"]
    decision["network_realization"] = {
        "status": "REPEATED_EVENT_FIT_ORACLE_QUANTIFIED_SELECTION_NOT_RUN",
        "C_per_net": oracle["C_per_net"],
        "C_shared": oracle["shared"]["C_shared"],
        "Delta_network": oracle["Delta_network"],
        "shared_candidate_id": oracle["shared"]["selected_candidate_id"],
        "shared_tied_candidate_count": oracle["shared"]["n_tied_candidates"],
        "scope": payload["formal_scope"],
    }
    decision["optimizer"] = {
        "status": payload["optimizer_status"],
        "reason": "shared candidate requires out-of-fit selection sanity first",
    }
    decision["patient_heldout_scores_computed"] = False
    decision["l3b_repeated_oracle_provenance"] = payload["provenance"]
    atomic_write_json(decision, decision_path)
    print(json.dumps({
        "status": payload["status"],
        "C_per_net": oracle["C_per_net"],
        "C_shared": oracle["shared"]["C_shared"],
        "Delta_network": oracle["Delta_network"],
        "shared_candidate_id": oracle["shared"]["selected_candidate_id"],
        "n_workers": len(worker_inputs),
        "patient_heldout_scores_computed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
