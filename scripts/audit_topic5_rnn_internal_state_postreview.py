#!/usr/bin/env python3
"""Post-review scientific acceptance audit for Topic 5 RNN v0.1."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_rnn_internal_state_reduction"
FIGURE = (
    ROOT
    / "results/paper-ready-figure/"
    "fig6_rnn_internal_state_reduction/figures/"
    "fig6_rnn_internal_state_reduction_metadata.json"
)


def paired_greater(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    p = (
        1.0
        if not len(values) or np.allclose(values, 0.0)
        else float(wilcoxon(values, alternative="greater").pvalue)
    )
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "n_positive": int(np.count_nonzero(values > 0)),
        "wilcoxon_greater_p": p,
    }


def main() -> None:
    reproduction = json.loads((BASE / "REPRODUCTION_AUDIT.json").read_text())
    if reproduction["status"] != "PASS":
        raise RuntimeError("reproduction audit is not PASS")

    order = pd.read_csv(BASE / "interictal_order_perturbation_metrics.csv")
    order = order.loc[
        (order.order_perturbation == "shuffle")
        & (order.prefix_bin == "all")
        & (order.metric == "nll_loss")
    ]
    order = (
        order.groupby(["subject", "control"], as_index=False).value.median()
        .pivot(index="subject", columns="control", values="value")
        .dropna()
    )
    order_difference = paired_greater(
        order.full_history_gru.to_numpy(float)
        - order.rank_shuffle_gru.to_numpy(float)
    )

    fixed = pd.read_csv(BASE / "early_ictal_fixed_readback_patient_metrics.csv")
    fixed = fixed.loc[
        (fixed.seizure_split == "all") & (fixed.field == "participation")
    ].copy()
    if fixed.subject.nunique() != 16:
        raise RuntimeError("fixed-readout patient denominator drifted")
    if "signed_rho" not in fixed:
        raise RuntimeError("signed correlation was not retained")
    wide_margin = fixed.pivot(
        index="subject", columns="model", values="all_contact_margin"
    )
    fixed_comparisons = {}
    for control in (
        "static_contact_hazard",
        "unordered_prefix",
        "last_set_first_order",
        "rank_shuffle_gru",
    ):
        fixed_comparisons[f"full_minus_{control}"] = paired_greater(
            (
                wide_margin.full_history_gru
                - wide_margin[control]
            ).dropna().to_numpy(float)
        )

    full = fixed.loc[fixed.model == "full_history_gru"]
    static_correspondence = {
        "n": int(len(full)),
        "median_absolute_rho": float(full.absolute_rho.median()),
        "median_all_contact_margin": float(full.all_contact_margin.median()),
        "n_positive_all_contact_margin": int(
            np.count_nonzero(full.all_contact_margin > 0)
        ),
        "signed_rho_retained_but_signed_null_not_yet_implemented": True,
    }

    internal = pd.read_csv(BASE / "early_ictal_internal_full_vs_rank_shuffle.csv")
    internal = internal.loc[
        (internal.field == "probability_contrast_residual_participation")
        & (internal.direction_type == "pca")
        & internal.metric.isin(("all_contact_margin", "within_shaft_margin"))
    ]
    exploratory_internal = {
        f"{row.metric}_pc{int(row.direction_index)}": {
            "n": int(row.n),
            "median": float(row["median"]),
            "n_positive": int(row.n_positive),
            "fdr_q": float(row.direction_family_bh_fdr_q),
        }
        for _, row in internal.iterrows()
    }

    figure = json.loads(FIGURE.read_text())
    if figure["status"] != "supplementary_exploratory_candidate":
        raise RuntimeError("figure status was not downgraded after review")

    result = {
        "contract": "topic5_rnn_internal_state_v0_1_postreview_acceptance",
        "status": "PASS_WITH_SCIENTIFIC_REFRAMING",
        "execution_integrity": reproduction,
        "scientific_acceptance": {
            "interictal_order_sensitivity": {
                "status": "SUPPORTED",
                "matched_order_shuffle_full_minus_rank_shuffle": (
                    order_difference
                ),
            },
            "static_contact_correspondence": {
                "status": "SUPPORTED_WITHIN_REUSED_DATASET",
                "fixed_participation": static_correspondence,
            },
            "gru_specific_static_increment": {
                "status": "NOT_ESTABLISHED",
                "fixed_participation_all_contact_margin": fixed_comparisons,
            },
            "ordered_state_early_ictal_readback": {
                "status": "EXPLORATORY_TARGET_REUSED",
                "participation_residualized_pc_fields": exploratory_internal,
                "independent_confirmation": False,
            },
            "physical_axis_or_dynamic_seizure_mechanism": {
                "status": "NOT_SUPPORTED_BY_CURRENT_MODEL_FAMILY",
                "interpretation": (
                    "This does not deny a patient pathological axis; it only "
                    "prevents attributing the static bridge to the current "
                    "axis/history/source parameterization."
                ),
            },
        },
        "review_items": {
            "per_seizure_five_field_oracle_primary": "RESOLVED_FIXED_FIELDS_USED",
            "signed_primary_and_signed_null": "OPEN_NEXT_CONTRACT",
            "shaft_preserving_nulls": "PARTIAL_GENERIC_WITHIN_SHAFT_ONLY",
            "geometry_smooth_null": "OPEN_NEXT_CONTRACT",
            "regularized_nonrecurrent_baselines": "OPEN_NEXT_CONTRACT",
            "free_vs_teacher_forced_field": "OPEN_NEXT_CONTRACT",
            "static_contact_confound_control": "OPEN_NEXT_CONTRACT",
            "off_manifold_pca_perturbation": (
                "DOWNGRADED_NOT_USED_AS_PRIMARY_EVIDENCE"
            ),
            "target_reuse_disclosure": "RESOLVED_EXPLICIT",
        },
        "next_contract": (
            "interictal_early_ictal_static_scaffold_fixed_readout_validation_v0_1"
        ),
        "figure_status": figure["status"],
    }
    path = BASE / "POSTREVIEW_ACCEPTANCE.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["scientific_acceptance"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
