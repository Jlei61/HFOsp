#!/usr/bin/env python3
"""Adjudicate ECoG graph-learning and post-training edge-use claims separately."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("results/topic5_ecog_physical_neighborhood_rnn_v0_1")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    graph = {row["subject"]: row for row in rows(ROOT / "summary/PATIENT_RESULTS.csv")}
    extended = {
        row["subject"]: row
        for row in rows(ROOT / "summary/HELDOUT_EXTENDED_PATIENT_RESULTS.csv")
    }
    symmetric = {
        row["subject"]: row for row in rows(ROOT / "summary/PATCH_PATIENT_RESULTS.csv")
        if int(row["patch_side"]) == 2
    }
    inbound = {
        row["subject"]: row
        for row in rows(ROOT / "summary_inbound/INBOUND_ENTRY_PATIENT_RESULTS.csv")
    }
    eight = {
        row["subject"]: row
        for row in rows(ROOT / "summary/EIGHT_NEIGHBOUR_PATIENT_RESULTS.csv")
    }
    one_step = {
        row["subject"]: row
        for row in rows(ROOT / "summary/ONE_MICROSTEP_PATIENT_RESULTS.csv")
    }

    patients = []
    for subject in ("958", "1084"):
        training_effect = float(graph[subject]["true_minus_wrong_grid_nll_median"])
        training_p = float(graph[subject]["true_vs_wrong_grid_exact_p_lower"])
        physical_helps = training_effect < 0 and training_p <= 0.05

        immediate_effect = float(
            extended[subject][
                "true_minus_wrong_grid_distance_up_down_left_right_nll_median_graph"
            ]
        )
        immediate_p_raw = float(
            extended[subject][
                "true_vs_wrong_grid_distance_up_down_left_right_nll_exact_p_one_sided"
            ]
        )
        immediate_p_holm = float(
            extended[subject][
                "true_vs_wrong_grid_distance_up_down_left_right_nll_holm_across_four_distance_bins"
            ]
        )
        immediate_directional = immediate_effect < 0 and immediate_p_raw <= 0.05
        immediate_strict = immediate_effect < 0 and immediate_p_holm <= 0.05

        symmetric_effect = float(
            symmetric[subject]["full_attenuation_difference_in_difference_median_patch"]
        )
        symmetric_p = float(symmetric[subject]["stratified_randomization_p_one_sided"])
        symmetric_supported = (
            symmetric_effect > 0
            and symmetric_p <= 0.05
            and as_bool(symmetric[subject]["patient_median_dose_curve_monotonic"])
        )

        inbound_effect = float(inbound[subject]["entry_damage_contrast_dose_0_median_patch"])
        inbound_p = float(inbound[subject]["stratified_randomization_p_one_sided"])
        inbound_supported = (
            inbound_effect > 0
            and inbound_p <= 0.05
            and as_bool(inbound[subject]["patient_dose_curve_monotonic"])
        )

        eight_effect = float(eight[subject]["true_minus_wrong_nll_median_graph"])
        eight_p = float(eight[subject]["exact_one_sided_p"])
        eight_supports = eight_effect < 0 and eight_p <= 0.05
        one_step_effect = float(one_step[subject]["true_minus_wrong_nll_median_graph"])
        one_step_p = float(one_step[subject]["exact_one_sided_p"])
        one_step_supports = one_step_effect < 0 and one_step_p <= 0.05

        if physical_helps and inbound_supported:
            interpretation = (
                "真实物理近邻既改善从头学习，也在修复性首次进入检验中显示在线必要性。"
            )
        elif physical_helps:
            interpretation = (
                "真实物理近邻改善从头学习，但训练后的首次进入预测未显示对这些局部入边的选择性依赖。"
            )
        elif immediate_directional:
            interpretation = (
                "整体预测没有通过严格图零模型，但真实网格对紧邻下一触点有方向一致的局部优势；"
                "该优势未建立训练后局部入边必要性。"
            )
        else:
            interpretation = (
                "未发现真实物理近邻相对位置打乱图的严格整体学习优势，也未发现训练后局部入边必要性。"
            )
        patients.append({
            "subject": subject,
            "role_for_training": "primary" if subject == "958" else "pre_specified_replication",
            "role_for_post_unblinding_inbound_repair": (
                "independent_confirmation" if subject == "958" else "development"
            ),
            "overall_learning": {
                "supported": physical_helps,
                "true_minus_wrong_grid_nll": training_effect,
                "exact_graph_rank_p_one_sided": training_p,
            },
            "immediate_physical_neighbour_transitions": {
                "directional_secondary": immediate_directional,
                "strict_after_four_distance_bin_holm": immediate_strict,
                "true_minus_wrong_grid_nll": immediate_effect,
                "raw_graph_rank_p_one_sided": immediate_p_raw,
                "holm_p": immediate_p_holm,
            },
            "pre_registered_symmetric_isolation": {
                "supported": symmetric_supported,
                "effect": symmetric_effect,
                "p_one_sided": symmetric_p,
                "interpretation_limit": (
                    "This intervention cuts both incoming and outgoing edges and cannot isolate first-entry necessity."
                ),
            },
            "post_unblinding_direct_inbound_entry_repair": {
                "supported": inbound_supported,
                "effect": inbound_effect,
                "p_one_sided": inbound_p,
            },
            "eight_neighbour_sensitivity": {
                "supported": eight_supports,
                "true_minus_wrong_grid_nll": eight_effect,
                "exact_graph_rank_p_one_sided": eight_p,
            },
            "one_internal_update_sensitivity": {
                "supported": one_step_supports,
                "true_minus_wrong_grid_nll": one_step_effect,
                "exact_graph_rank_p_one_sided": one_step_p,
            },
            "interpretation": interpretation,
        })

    payload = {
        "schema": "topic5_ecog_claim_adjudication_v0.2",
        "patients": patients,
        "replication": {
            "overall_learning_supported_in_both": all(
                row["overall_learning"]["supported"] for row in patients
            ),
            "direct_inbound_entry_supported_in_both": all(
                row["post_unblinding_direct_inbound_entry_repair"]["supported"]
                for row in patients
            ),
        },
        "claim_boundary": (
            "E958 and E1084 are two separately reported ECoG patients, not a cohort estimate. "
            "All graph masks were fixed before each from-scratch training run. Only the lesion "
            "experiments changed weights after training. The graph edges are computational "
            "constraints between physical ECoG neighbours, not white-matter anatomy. The inbound "
            "repair was frozen only after the original symmetric estimand was found to mix incoming "
            "and outgoing effects, so it is labelled post-unblinding and cannot replace the original primary."
        ),
    }
    output = ROOT / "FINAL_CLAIM_ADJUDICATION.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
