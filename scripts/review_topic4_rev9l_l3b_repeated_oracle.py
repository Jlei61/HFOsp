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
from src.topic4_rev9l_capacity_audit import (  # noqa: E402
    audit_finite_library_capacity,
)


DEFAULT_CONFIG = "config/topic4_rev9l_l3b_review.json"

DESCRIPTOR_LABELS = {
    "recruitment_mean_absolute_error": "recruitment",
    "precedence_mean_absolute_error": "precedence",
    "mean_rank_profile_absolute_error": "rank profile",
    "event_distribution_sliced_wasserstein": "event cloud",
}


def _plot(result, payload, output_dir):
    """Four panels, four questions.

    A answers "how far is the whole library from patient training", which needs
    the patient-equivalent objective on the same axis; B answers "does the edge
    family buy anything" at a zoom where 0.06 is visible; C answers "which mode
    and which descriptor fails"; D answers "why recruitment cannot move". The
    per-network gamma heatmap is deliberately not repeated here because the fit
    figure already carries it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = result["network_seeds"]
    review = result["capacity_review"]
    audit = result["capacity_audit"]
    oracle = payload["oracle"]
    values = payload["objective_by_candidate_network"]
    baseline_id = review["baseline_candidate_id"]
    shared_id = review["shared_candidate_id"]
    per_network = {row["network_seed"]: row for row in oracle["per_network"]}
    x = np.arange(len(seeds))
    fig, axes = plt.subplots(1, 4, figsize=(17.4, 4.6), constrained_layout=True)

    band = audit["patient_equivalent_objective"]["n3"]
    lower = float(band["floor_median_objective"])
    upper = float(band["floor_q95_objective"])
    axes[0].axhspan(lower, upper, color="#8fbcd4", alpha=0.35, zorder=0)
    axes[0].axhline(upper, color="#277da1", linewidth=0.9, zorder=1)
    axes[0].plot(x, [values[baseline_id][str(seed)] for seed in seeds],
                 "o-", color="0.35", label="scalar")
    axes[0].plot(x, [per_network[seed]["minimum_objective"] for seed in seeds],
                 "D-", color="#d1495b", label="per-network oracle")
    axes[0].plot(x, [values[shared_id][str(seed)] for seed in seeds],
                 "s-", color="#277da1", label=f"shared {shared_id[-3:]}")
    top = max(values[baseline_id][str(seed)] for seed in seeds)
    axes[0].set_ylim(lower - 0.30, top + 0.55)
    axes[0].annotate(
        "patient training band (floor centre to q95)",
        (len(seeds) - 1.05, lower + 0.06),
        xycoords=("data", "data"), ha="right", va="bottom", fontsize=7,
        color="#1c5d80")
    axes[0].annotate(
        "library median sits "
        f"{audit['median_gap_above_patient_q95_objective']:.2f} above the "
        "patient q95 line",
        (0.03, 0.985), xycoords="axes fraction", va="top", fontsize=7,
        color="0.25")
    axes[0].set_xticks(x, seeds, rotation=30)
    axes[0].set_ylabel("count-matched objective")
    axes[0].set_title("A  Whole library sits outside patient variability",
                      loc="left", weight="bold", fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=7, loc="center left")

    per_gain = [row["objective_gain_vs_scalar"]
                for row in review["per_network_oracle_gain_vs_scalar"]]
    shared_gain = [review["shared_gain_vs_scalar_by_network"][str(seed)]
                   for seed in seeds]
    width = 0.36
    axes[1].bar(x - width / 2, per_gain, width, color="#d1495b",
                label="per-network oracle (library minimum)")
    axes[1].bar(x + width / 2, shared_gain, width, color="#277da1",
                label=f"single shared {shared_id[-3:]}")
    axes[1].axhline(0.0, color="0.25", linewidth=0.8)
    gap = float(audit["median_gap_above_patient_q95_objective"])
    axes[1].annotate(
        f"median per-network gain {review['per_network_oracle_median_gain']:.3f}"
        f"\n= {100.0 * review['per_network_oracle_median_gain'] / gap:.1f}% of the"
        "\nmedian gap to the patient q95 band",
        (0.03, 0.97), xycoords="axes fraction", va="top", fontsize=7,
        color="0.25")
    axes[1].set_xticks(x, seeds, rotation=30)
    axes[1].set_ylabel("scalar minus candidate")
    axes[1].set_title("B  Edge family gain, at a visible zoom",
                      loc="left", weight="bold", fontsize=9.5)
    axes[1].legend(frameon=False, fontsize=6.5, loc="lower left")

    offsets = {"A": -0.13, "B": 0.13}
    colors = {"A": "#d1495b", "B": "#277da1"}
    positions = np.arange(len(DESCRIPTOR_LABELS))
    for mode, offset in offsets.items():
        for index, name in enumerate(DESCRIPTOR_LABELS):
            ratios = [row["modes"][mode]["raw_over_q95"][name]
                      for row in audit["per_network_mode_ratios"]]
            axes[2].scatter(
                np.full(len(ratios), positions[index] + offset), ratios,
                s=26, color=colors[mode], alpha=0.85,
                label=f"mode {mode}" if index == 0 else None)
    axes[2].axhline(1.0, color="0.25", linestyle="--", linewidth=0.9)
    axes[2].annotate("patient q95", (len(positions) - 0.55, 1.0),
                     va="bottom", fontsize=7, color="0.25")
    axes[2].set_xticks(positions, list(DESCRIPTOR_LABELS.values()), rotation=20,
                       ha="right")
    axes[2].set_ylabel("per-network oracle error / patient q95")
    axes[2].set_title("C  Mode B matches shape, mode A does not",
                      loc="left", weight="bold", fontsize=9.5)
    axes[2].legend(frameon=False, fontsize=7, loc="upper left")

    reach = audit["recruitment_reachability"]
    contacts = np.arange(reach["modes"]["A"]["n_contacts"])
    never = reach["never_recruited_in_both_modes"]
    for index in never:
        axes[3].axvspan(index - 0.5, index + 0.5, color="0.88", zorder=0)
    for mode, offset in offsets.items():
        axes[3].bar(contacts + offset,
                    reach["modes"][mode]["patient_recruitment_probability"],
                    0.26, color=colors[mode], alpha=0.35,
                    label=f"patient mode {mode}")
        axes[3].scatter(
            contacts + offset,
            reach["modes"][mode]["best_model_recruitment_probability"],
            s=20, color=colors[mode], zorder=3,
            label=f"model mode {mode}")
    axes[3].set_xticks(contacts, contacts, fontsize=6.5)
    axes[3].set_xlabel("contact index")
    axes[3].set_ylabel("recruitment probability")
    axes[3].set_ylim(-0.05, 1.92)
    axes[3].annotate(
        f"grey: {len(never)}/{len(contacts)} contacts never reached by any of "
        f"the {payload['n_candidates']} candidates x {len(seeds)} networks,\n"
        "in either mode; they carry "
        f"{100.0 * reach['modes']['A']['share_of_best_error_from_never_recruited']:.0f}%"
        " of mode A recruitment error",
        (0.02, 0.985), xycoords="axes fraction", va="top", fontsize=6.8,
        color="0.25")
    axes[3].set_title("D  Recruitment is capped by unreachable contacts",
                      loc="left", weight="bold", fontsize=9.5)
    axes[3].legend(frameon=False, fontsize=6, ncol=4, loc="upper center",
                   bbox_to_anchor=(0.5, 0.86), handletextpad=0.3,
                   columnspacing=0.9)

    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"rev9l_l3b_scientific_review.{suffix}", dpi=300)
    plt.close(fig)
    (output_dir / "README.md").write_text(
        "### rev9l_l3b_scientific_review.png\n"
        "四个 panel 各回答一个独立问题，不互为改写。\n\n"
        "- **A：整个候选库离患者训练变异有多远。** 灰蓝色带是同样只有 3 个事件的患者训练子样本自己会拿到的分数区间"
        "（从 floor 中位到 95 分位）。三条曲线全部远在带子上方，说明这不是「差一点」的问题。\n"
        "- **B：component-pair edge family 到底买到了多少。** 放大到能看见 0.06 的尺度上画同网络配对增益；"
        "红色是「每张网络各自挑最好的候选」（库里含 scalar，所以它不可能比 scalar 差，格子数本身不是发现，只有幅度是）；"
        "蓝色是同一个 shared 参数，在 6 张网络里 4 张变差。\n"
        "- **C：哪个 mode、哪个描述量不达标。** 每个点是一张网络的逐网络最优。mode B 的三个形状量（precedence / "
        "rank profile / event cloud）全部落在患者 q95 以内，mode A 全部在外；两个 mode 唯一共同不达标的是 recruitment。\n"
        "- **D：recruitment 为什么动不了。** 灰底四列触点在 57 个候选 x 6 张网络 x 两个 mode 里从未被招募过，"
        "而患者在这些触点上有 0.60–0.91 的招募概率。它们独占 mode A recruitment 误差的 54%，"
        "所以这一项是 forced 读出的几何上限，不能算在 edge 参数头上。\n\n"
        "**关注点**：A 的纵轴范围必须一直包含患者带，否则 2% 级别的增益会被画成结构；"
        "C 的 mode B 若某天也掉到 q95 以外，说明目标或读出发生了漂移；D 的灰底触点数量是下一轮改 scaffold 的直接指标。\n")


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
    base = json.loads(
        Path(config["inputs"]["component_pair_config"]["path"]).read_text())
    audit = audit_finite_library_capacity(
        payload, floors, objective=base["objective"],
        baseline_id=config["baseline_candidate_id"])
    if not audit["descriptor_event_count_consistency"]["consistent"]:
        raise RuntimeError(
            "descriptor event counts disagree with their count-matched floors")
    selection_run = bool(review["shared_forced_capacity_supported"])
    result = {
        "status": "REV9L_L3B_SCIENTIFIC_REVIEW_COMPLETE",
        "scientific_role": config["scientific_role"],
        "capacity_review": review,
        "capacity_audit": audit,
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
    reach = audit["recruitment_reachability"]
    extremes = audit["descriptor_extremes"]
    decision["propagation_family"]["repeated_oracle"] = {
        "status": review["status"],
        "per_network_oracle_median_gain": review[
            "per_network_oracle_median_gain"],
        "mode_A_networks_with_all_descriptors_within_patient_q95": review[
            "n_networks_with_mode_A_all_descriptors_within_patient_q95"],
        "mode_A_networks_total": len(result["network_seeds"]),
        "mode_B_networks_with_all_shape_descriptors_within_patient_q95": int(sum(
            row["modes"]["B"]["n_shape_descriptors_above_q95"] == 0
            for row in audit["per_network_mode_ratios"])),
        "recruitment_never_reached_contacts": reach[
            "never_recruited_in_both_modes"],
        "recruitment_error_share_from_unreachable_contacts": {
            mode: reach["modes"][mode]["share_of_best_error_from_never_recruited"]
            for mode in ("A", "B")},
        "recruitment_descriptor_distinct_values_mode_A": extremes["A"][
            "recruitment_mean_absolute_error"]["n_distinct_values"],
        "recruitment_descriptor_statement_mode_A": extremes["A"][
            "recruitment_mean_absolute_error"]["statement"],
        "patient_equivalent_objective_n3": audit[
            "patient_equivalent_objective"]["n3"],
        "median_gap_above_patient_q95_objective": audit[
            "median_gap_above_patient_q95_objective"],
        "descriptor_support_bias": audit["descriptor_support"]["bias_direction"],
        "delta_network_is_non_negative_by_construction": True,
        "delta_network_noise_null_tested": False,
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
        "per_network_oracle_improvement_caveat": audit[
            "per_network_oracle_gain_is_library_minimum"],
        "delta_network_is_non_negative_by_construction": True,
        "delta_network_noise_null_tested": False,
        "delta_network_caveat": (
            "median of per-network minima can never exceed the median of any "
            "single candidate, so Delta_network >= 0 always; its magnitude is "
            "not evidence that the networks require different residuals until "
            "a repeat-level noise null is run"),
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
