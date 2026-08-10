"""Aggregate rev9-L L2 Sobol fit or selection-network confirmation."""
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
    _provenance,
    _sha256,
)
from src.topic4_component_pair_search import (  # noqa: E402
    score_candidate,
    selection_candidates_with_baseline,
    sobol_candidates,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "config/topic4_rev9l_component_pair_edge.json"


def _load_floor(config, config_path):
    path = Path(config["objective"]["floor_output"])
    payload = json.loads(path.read_text())
    if (payload["status"] != "REV9L_L2_PATIENT_TRAINING_FLOOR_COMPLETE"
            or payload.get("patient_heldout_scores_computed") is not False
            or payload["config"]["sha256"] != _sha256(config_path)
            or _sha256(payload["samples"]["path"]) != payload["samples"]["sha256"]):
        raise RuntimeError("L2 patient-training floor provenance mismatch")
    return payload, {"path": str(path), "sha256": _sha256(path)}


def _fit_candidates(config):
    return sobol_candidates(
        config["sobol_search"], config["component_pair_family"]["gamma_bounds"])


def _selection_candidates(config):
    path = Path(config["output_root"]) / "sobol_fit" / "sobol_fit_summary.json"
    summary = json.loads(path.read_text())
    if summary["status"] != "REV9L_L2_SOBOL_FIT_COMPLETE":
        raise RuntimeError("Sobol fit is not complete")
    return selection_candidates_with_baseline(
        summary["top_for_selection"],
        dimension=int(config["sobol_search"]["dimension"])), {
        "path": str(path), "sha256": _sha256(path),
    }


def _score(row, config, floor):
    source_for = {
        "A": config["primary_mapping"]["mode_A_source"],
        "B": config["primary_mapping"]["mode_B_source"],
    }
    if row["mode_descriptors"] is None:
        return {
            "objective": None, "eligible": False,
            "reason": "mode descriptors unavailable",
        }
    readable = {
        mode: row["geometry"][source]["curve_usable_fraction"]
        for mode, source in source_for.items()
    }
    ood = {
        mode: row["geometry"][source]["ood_fraction"]
        for mode, source in source_for.items()
    }
    objective = config["objective"]
    result = score_candidate(
        row["mode_descriptors"], floor["floor"], readable, ood,
        readable_weight=objective["readable_fraction_penalty_weight"],
        tau=objective["weakest_mode_lse_tau"],
        ood_weight=objective["ood_weight"])
    result["eligible"] = bool(
        np.isfinite(result["objective"])
        and row["structural"]["ratio_within_0p25_4"]
        and row["n_runaway"] == 0)
    result["structural_admissible"] = bool(
        row["structural"]["ratio_within_0p25_4"])
    return result


def _pareto(rows):
    eligible = [row for row in rows if row["score"].get("eligible")]
    front = []
    for row in eligible:
        left = (row["score"]["objective"],
                row["structural"]["residual_kl_all_targets"]["median"])
        dominated = False
        for other in eligible:
            if other is row:
                continue
            right = (other["score"]["objective"],
                     other["structural"]["residual_kl_all_targets"]["median"])
            if right[0] <= left[0] and right[1] <= left[1] and right != left:
                dominated = True
                break
        if not dominated:
            front.append(row["candidate_id"])
    return front


def _plot(rows, config, stage, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible = [row for row in rows if row["score"].get("eligible")]
    rejected = [row for row in rows if not row["score"].get("eligible")]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), constrained_layout=True)
    for subset, color, marker, label in (
            (eligible, "#277da1", "o", "admissible"),
            (rejected, "0.72", "x", "excluded from ranking")):
        axes[0].scatter(
            [row["score"].get("mode_scores", {}).get("A", np.nan) for row in subset],
            [row["score"].get("mode_scores", {}).get("B", np.nan) for row in subset],
            c=color, marker=marker, s=28, label=label)
    ranked = sorted(eligible, key=lambda row: row["score"]["objective"])
    if ranked:
        best = ranked[0]
        axes[0].scatter(
            best["score"]["mode_scores"]["A"],
            best["score"]["mode_scores"]["B"], marker="*", s=160,
            color="#d1495b", edgecolor="white", linewidth=0.8, label="best")
    axes[0].set_xlabel("floor-normalized mode A loss")
    axes[0].set_ylabel("floor-normalized mode B loss")
    axes[0].set_title("A  Weakest-mode landscape", loc="left", weight="bold")
    axes[0].legend(frameon=False, fontsize=7)

    if eligible:
        axes[1].scatter(
            [row["structural"]["residual_kl_all_targets"]["median"] for row in eligible],
            [row["score"]["objective"] for row in eligible],
            c=[row["score"]["mode_scores"]["A"] for row in eligible],
            cmap="magma", s=32)
    axes[1].set_xlabel("median target KL vs scalar edge")
    axes[1].set_ylabel("weakest-mode objective")
    axes[1].set_title("B  Fit-distortion trade-off", loc="left", weight="bold")

    shown = ranked[:min(8, len(ranked))]
    gamma = np.asarray([row["gamma"] for row in shown], float)
    if len(gamma):
        scale = max(np.max(np.abs(gamma)), 1e-12)
        image = axes[2].imshow(gamma, cmap="RdBu_r", vmin=-scale, vmax=scale,
                               aspect="auto")
        axes[2].set_yticks(range(len(shown)), [row["candidate_id"] for row in shown],
                           fontsize=7)
        axes[2].set_xticks(range(6), [value.replace("gamma_", "").replace("_from_", "<-\n")
                                     for value in config["component_pair_family"]["gamma_order"]],
                           fontsize=7)
        fig.colorbar(image, ax=axes[2], shrink=0.75, label="gamma")
    axes[2].set_title("C  Best residual structures", loc="left", weight="bold")
    stem = f"rev9l_l2_component_pair_{stage}"
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300)
    plt.close(fig)
    (output_dir / "README.md").write_text(
        f"### {stem}.png\n"
        "左图显示 floor-normalized A/B 模式损失，星号是当前阶段最低 weakest-mode objective；中图把拟合改善与相对 scalar edge 的 target KL 扰动并列；右图显示排名最前候选的六个 residual 系数。灰色候选仅因结构范围、非有限 readout 或 runaway 不进入排序，仍保留为探索结果。\n\n"
        "**关注点**：mode A 是否在不牺牲 mode B 的条件下改善，以及改善是否需要超出 0.25-4 倍 edge ratio 的连接畸变。\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--stage", required=True, choices=("sobol_fit", "selection_confirmation"))
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    for name, record in config["inputs"].items():
        if _sha256(record["path"]) != record["sha256"]:
            raise RuntimeError(f"L2 input hash changed: {name}")
    floor, floor_input = _load_floor(config, config_path)
    fit_input = None
    if args.stage == "sobol_fit":
        candidates = _fit_candidates(config)
        seeds = config["network_seeds"]["fit"]
    else:
        candidates, fit_input = _selection_candidates(config)
        seeds = config["network_seeds"]["selection"]
    workers, contact_names, worker_inputs = _load_workers(
        config, config_path, args.expected_commit, candidates=candidates,
        stage=args.stage, seeds=seeds)
    reference, patient, patient_labels, prototypes = _patient(config, contact_names)
    rows = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        row = _candidate_summary(
            workers[candidate_id], config, reference, patient,
            patient_labels, prototypes)
        row["candidate_id"] = candidate_id
        row["score"] = _score(row, config, floor)
        rows.append(row)
    ranked = sorted(
        [row for row in rows if row["score"].get("eligible")],
        key=lambda row: row["score"]["objective"])
    top_k = int(config["sobol_search"]["top_k_for_selection_seeds"])
    status = ("REV9L_L2_SOBOL_FIT_COMPLETE" if args.stage == "sobol_fit"
              else "REV9L_L2_SELECTION_CONFIRMATION_COMPLETE")
    payload = {
        "status": status,
        "scientific_role": (
            "patient-training exploratory component-pair edge oracle; no held-out"),
        "stage": args.stage,
        "n_candidates": len(rows),
        "n_eligible": len(ranked),
        "network_seeds": list(map(int, seeds)),
        "candidates": rows,
        "ranked_candidate_ids": [row["candidate_id"] for row in ranked],
        "pareto_candidate_ids": _pareto(rows),
        "best_candidate": None if not ranked else ranked[0],
        "top_for_selection": ([
            {"candidate_id": row["candidate_id"], "gamma": row["gamma"]}
            for row in ranked[:top_k]
        ] if args.stage == "sobol_fit" else []),
        "patient_heldout_scores_computed": False,
        "floor_input": floor_input,
        "fit_summary_input": fit_input,
        "worker_inputs": worker_inputs,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "provenance": _provenance(args.expected_commit),
    }
    output_dir = Path(config["output_root"]) / args.stage
    output_path = output_dir / f"{args.stage}_summary.json"
    atomic_write_json(payload, output_path)
    _plot(rows, config, args.stage, output_dir / "figures")
    decision_path = Path(config["output_root"]).parent / "decision.json"
    decision = json.loads(decision_path.read_text())
    decision["status"] = status
    decision["propagation_family"][args.stage] = {
        "status": status,
        "summary_path": str(output_path),
        "n_eligible": len(ranked),
        "best_candidate_id": None if not ranked else ranked[0]["candidate_id"],
        "best_objective": None if not ranked else ranked[0]["score"]["objective"],
    }
    decision["patient_heldout_scores_computed"] = False
    atomic_write_json(decision, decision_path)
    print(json.dumps({
        "status": status, "n_candidates": len(rows), "n_eligible": len(ranked),
        "best_candidate_id": None if not ranked else ranked[0]["candidate_id"],
        "best_objective": None if not ranked else ranked[0]["score"]["objective"],
        "patient_heldout_scores_computed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
