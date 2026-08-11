"""Scientifically review the rev9-L repeated-event finite-library oracle."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.review_topic4_rev9l_l3_network_surrogate import (  # noqa: E402
    _provenance,
    _sha256,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_repeated_network_oracle import (  # noqa: E402
    review_repeated_capacity,
)


DEFAULT_CONFIG = "config/topic4_rev9l_l3b_review.json"


def _plot(result, payload, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = result["network_seeds"]
    review = result["capacity_review"]
    oracle = payload["oracle"]
    values = payload["objective_by_candidate_network"]
    baseline_id = review["baseline_candidate_id"]
    shared_id = review["shared_candidate_id"]
    per_network = {row["network_seed"]: row for row in oracle["per_network"]}
    x = np.arange(len(seeds))
    fig, axes = plt.subplots(1, 4, figsize=(16.2, 4.4), constrained_layout=True)

    axes[0].plot(x, [values[baseline_id][str(seed)] for seed in seeds],
                 "o-", color="0.35", label="scalar")
    axes[0].plot(x, [per_network[seed]["minimum_objective"] for seed in seeds],
                 "D-", color="#d1495b", label="per-network oracle")
    axes[0].plot(x, [values[shared_id][str(seed)] for seed in seeds],
                 "s-", color="#277da1", label=f"shared {shared_id[-3:]}")
    axes[0].set_xticks(x, seeds, rotation=30)
    axes[0].set_ylabel("count-matched objective")
    axes[0].set_title("A  Full forced objective", loc="left", weight="bold")
    axes[0].legend(frameon=False, fontsize=7)

    per_gain = [row["objective_gain_vs_scalar"]
                for row in review["per_network_oracle_gain_vs_scalar"]]
    shared_gain = [review["shared_gain_vs_scalar_by_network"][str(seed)]
                   for seed in seeds]
    width = 0.36
    axes[1].bar(x - width / 2, per_gain, width, color="#d1495b",
                label="per-network oracle")
    axes[1].bar(x + width / 2, shared_gain, width, color="#277da1",
                label="shared candidate")
    axes[1].axhline(0.0, color="0.25", linewidth=0.8)
    axes[1].set_xticks(x, seeds, rotation=30)
    axes[1].set_ylabel("scalar minus candidate")
    axes[1].set_title("B  Paired improvement", loc="left", weight="bold")
    axes[1].legend(frameon=False, fontsize=7)

    metric_labels = {
        "recruitment_mean_absolute_error": "recruitment",
        "precedence_mean_absolute_error": "precedence",
        "mean_rank_profile_absolute_error": "rank profile",
        "event_distribution_sliced_wasserstein": "event cloud",
    }
    review_rows = review["per_network_oracle_gain_vs_scalar"]
    for name, label in metric_labels.items():
        axes[2].plot(
            x, [row["mode_A_descriptors"][name]["raw_over_q95"]
                for row in review_rows], marker="o", label=label)
    axes[2].axhline(1.0, color="0.25", linestyle="--", linewidth=0.9,
                   label="patient q95")
    axes[2].set_xticks(x, seeds, rotation=30)
    axes[2].set_ylabel("mode A error / patient q95")
    axes[2].set_title("C  Mode A remains outside floor", loc="left", weight="bold")
    axes[2].legend(frameon=False, fontsize=6, ncol=2)

    shown = [review["shared_candidate_id"]] + [
        row["candidate_id"] for row in review_rows]
    summaries = {row["candidate_id"]: row for row in payload["candidate_summary"]}
    gamma = np.asarray([summaries[candidate_id]["gamma"] for candidate_id in shown])
    scale = max(float(np.max(np.abs(gamma))), 1e-12)
    image = axes[3].imshow(
        gamma, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
    axes[3].set_yticks(range(len(shown)),
                       ["shared"] + [str(seed) for seed in seeds])
    axes[3].set_xticks(
        range(6), ("C1<-C1", "C1<-C2", "C2<-C1", "C2<-C2",
                   "BG<-C1", "BG<-C2"), rotation=30, ha="right", fontsize=7)
    axes[3].set_title("D  Different network optima", loc="left", weight="bold")
    fig.colorbar(image, ax=axes[3], shrink=0.78, label="gamma")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"rev9l_l3b_scientific_review.{suffix}", dpi=300)
    plt.close(fig)
    (output_dir / "README.md").write_text(
        "### rev9l_l3b_scientific_review.png\n"
        "A 比较 scalar、逐网络 oracle 与冻结 shared 候选的完整 forced objective；B 改写为同网络 paired gain；C 将每张网络 oracle 的 mode A 四项误差直接除以 patient-training q95；D 显示 shared 与六个逐网络最优的 edge residual。所有患者量只来自训练记录。\n\n"
        "**关注点**：逐网络小幅改善是否对应 mode A 真正回到患者训练变异范围，以及同一 shared 参数是否在多数网络方向一致。\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    if config.get("patient_heldout_permitted") is not False:
        raise RuntimeError("L3b review must remain patient-training-only")
    for name, record in config["inputs"].items():
        if _sha256(record["path"]) != record["sha256"]:
            raise RuntimeError(f"L3b review input changed: {name}")
    payload = json.loads(Path(config["inputs"]["l3b_result"]["path"]).read_text())
    floors = {
        count: json.loads(Path(
            config["inputs"][f"patient_floor_n{count}"]["path"]).read_text())
        for count in (2, 3)
    }
    review = review_repeated_capacity(
        payload, floors, baseline_id=config["baseline_candidate_id"])
    selection_run = bool(review["shared_forced_capacity_supported"])
    result = {
        "status": "REV9L_L3B_SCIENTIFIC_REVIEW_COMPLETE",
        "scientific_role": config["scientific_role"],
        "capacity_review": review,
        "network_seeds": list(map(int, payload["network_seeds"])),
        "selection_sanity_recommended": selection_run,
        "optimizer_recommended": selection_run,
        "spontaneous_confirmation_recommended": selection_run,
        "decision": (
            "STOP_NO_SHARED_FORCED_CAPACITY" if not selection_run
            else "PROCEED_TO_FROZEN_SHARED_SELECTION_SANITY"),
        "claim_boundary": (
            "the bounded finite static-edge library did not restore patient mode A; "
            "this is not a claim that patient modes are generally unlearnable"),
        "patient_heldout_scores_computed": False,
        "inputs": config["inputs"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "provenance": _provenance(args.expected_commit),
    }
    output_path = Path(config["output"])
    atomic_write_json(result, output_path)
    _plot(result, payload, output_path.parent / "figures")

    decision_path = Path(
        "results/topic4_sef_hfo/data_driven_core_field_rev9_learnability/decision.json")
    decision = json.loads(decision_path.read_text())
    decision["status"] = "REV9L_DEVELOPMENT_AUDIT_COMPLETE"
    decision["propagation_family"]["repeated_oracle"] = {
        "status": review["status"],
        "per_network_oracle_median_gain": review[
            "per_network_oracle_median_gain"],
        "mode_A_networks_with_all_descriptors_within_patient_q95": review[
            "n_networks_with_mode_A_all_descriptors_within_patient_q95"],
        "mode_A_networks_total": len(result["network_seeds"]),
        "claim_boundary": result["claim_boundary"],
    }
    decision["network_realization"] = {
        "status": "PER_NETWORK_SMALL_GAINS_NO_SHARED_MODE_A_REALIZABILITY",
        "C_per_net": payload["oracle"]["C_per_net"],
        "C_shared": payload["oracle"]["shared"]["C_shared"],
        "Delta_network": payload["oracle"]["Delta_network"],
        "shared_candidate_id": review["shared_candidate_id"],
        "shared_n_networks_improved": review["shared_n_networks_improved"],
        "shared_n_networks_total": review["shared_n_networks_total"],
        "shared_mean_gain_vs_scalar": review["shared_mean_gain_vs_scalar"],
        "per_network_oracle_improved_all_networks": review[
            "per_network_oracle_improved_all_networks"],
        "per_network_oracle_median_gain": review[
            "per_network_oracle_median_gain"],
        "shared_forced_capacity_supported": False,
        "scope": payload["formal_scope"],
    }
    decision["optimizer"] = {
        "status": "NOT_TESTED_NO_KNOWN_GOOD_SHARED_SOLUTION",
        "reason": (
            "optimizer attribution requires a known-good full shared solution; "
            "the finite repeated-event oracle found none"),
    }
    decision["spontaneous_confirmation"] = {
        "status": "NOT_RUN_FORCED_SHARED_CAPACITY_NOT_ESTABLISHED",
        "patient_blind_opened": False,
    }
    decision["final_claim_boundary"] = result["claim_boundary"]
    decision["patient_heldout_scores_computed"] = False
    decision["l3b_scientific_review_provenance"] = result["provenance"]
    atomic_write_json(decision, decision_path)
    print(json.dumps({
        "status": result["status"], "decision": result["decision"],
        "shared_candidate_id": review["shared_candidate_id"],
        "shared_n_networks_improved": review["shared_n_networks_improved"],
        "mode_A_networks_within_q95": review[
            "n_networks_with_mode_A_all_descriptors_within_patient_q95"],
        "patient_heldout_scores_computed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
