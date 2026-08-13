#!/usr/bin/env python3
"""Adjudicate the full-tissue LBSS claims without a minimum-of-controls bias.

The v0.2 closeout additionally reported a patient-wise minimum of three L3
contrasts.  Such a minimum is negative under an exchangeable null and is not a
valid zero-centred test.  This v0.3 summary keeps every prespecified paired
contrast explicit and applies Holm correction within each scientific family.
It only reads already frozen patient tables and never trains or scores a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.summarize_topic5_lbss_claims_v0_2 import (
    L0,
    L1,
    L2,
    L3,
    SHUFFLE,
    attenuation_damage_auc,
    add_family_holm,
    paired,
)


REFS = (L0, L1, L2)


def _condition_wide(frame: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    use = frame[
        frame.primary.astype(bool)
        & frame.endpoint.eq(endpoint)
        & frame.family.eq("intact")
    ].copy()
    use["margin"] = use["all_contact_margin"].astype(float)
    return use.pivot(index="subject", columns="arm", values="margin")


def summarize(out: Path) -> dict:
    required = (
        "INTERICTAL_ANALYSIS_COMPLETE.json",
        "PATHWAY_ANALYSIS_COMPLETE.json",
        "ATTENUATION_COMPLETE.json",
        "EARLY_ICTAL_SCORING_COMPLETE.json",
    )
    missing = [name for name in required if not (out / name).exists()]
    if missing:
        raise RuntimeError(f"claim adjudication requires {missing}")

    interictal = pd.read_csv(out / "interictal_per_patient.csv")
    ip = interictal.pivot(index="subject", columns="arm")
    eligible = np.ones(len(ip.index), dtype=bool)
    for arm in (L0, L1, L2, L3, SHUFFLE):
        eligible &= ip["distal_n"][arm].to_numpy(float) >= 20
    subjects = ip.index[eligible]

    claim_a = {
        "L0_vs_no_recurrence_all": paired(
            ip.loc[subjects, "no_rec_contact_nll"][L0]
            - ip.loc[subjects, "test_contact_nll"][L0]
        )
    }
    claim_b = add_family_holm({
        f"L3_vs_{arm}_distal": paired(
            ip.loc[subjects, "distal_contact_nll"][arm]
            - ip.loc[subjects, "distal_contact_nll"][L3]
        )
        for arm in REFS
    })

    pathway = pd.read_csv(
        out / "pathway_analysis/true_vs_shuffle_patient_patterns.csv"
    ).set_index("subject")
    common = subjects.intersection(pathway.index)
    # Endpoint-density and effective-influence patterns are distinct,
    # prespecified observables.  Do not take a patient-wise minimum: even when
    # both inputs are individually zero-centred, their minimum is not, and it
    # would silently turn this into a conjunction test with a shifted null.
    endpoint_pattern = pathway.loc[
        common, "endpoint_dissimilarity_beyond_proposal"
    ]
    effective_pattern = pathway.loc[
        common, "effective_dissimilarity_beyond_proposal"
    ]
    true_shuffle = (
        ip.loc[common, "distal_contact_nll"][SHUFFLE]
        - ip.loc[common, "distal_contact_nll"][L3]
    )
    attenuation = pd.read_csv(out / "attenuation/attenuation_patient_auc.csv")
    local_ok = attenuation[
        attenuation.target.eq("L3_MATCHED_LOCAL")
        & attenuation.inferential_eligible.astype(bool)
    ].subject
    aw = attenuation[attenuation.subject.isin(local_ok)].pivot(
        index="subject", columns="target", values="auc_distal_selectivity"
    )
    common_c = common.intersection(aw.index)
    double_dissociation = (
        aw.loc[common_c, "L3_ADDED"] - aw.loc[common_c, "L3_MATCHED_LOCAL"]
    )
    claim_c = add_family_holm({
        "endpoint_pattern_difference_beyond_proposal": paired(endpoint_pattern),
        "effective_pattern_difference_beyond_proposal": paired(effective_pattern),
        "true_order_vs_shuffle_distal": paired(true_shuffle),
        "selected_nonlocal_vs_matched_local_attenuation_dd": paired(
            double_dissociation
        ),
    })

    early = pd.read_csv(out / "early_ictal/early_ictal_per_patient_condition.csv")
    canonical = _condition_wide(early, "canonical_full")
    seed_removed = _condition_wide(early, "seed_removed")
    claim_d_values: dict[str, pd.Series] = {
        "D1_L3_canonical_full_margin_gt_zero": canonical[L3],
    }
    for arm in REFS:
        claim_d_values[f"D2_L3_vs_{arm}_seed_removed"] = (
            seed_removed[L3] - seed_removed[arm]
        )

    early_auc = attenuation_damage_auc(
        early[early.primary.astype(bool)], "seed_removed"
    )
    early_auc = early_auc[early_auc.subject.isin(set(local_ok))]
    eaw = early_auc.pivot(index="subject", columns="target", values="damage_auc")
    for target in ("L1_ADDED", "L2_ADDED", "L3_MATCHED_LOCAL"):
        claim_d_values[f"D2_L3_ADDED_vs_{target}_attenuation_auc"] = (
            eaw["L3_ADDED"] - eaw[target]
        )
    claim_d = add_family_holm({
        name: paired(values) for name, values in claim_d_values.items()
    })

    return {
        "contract": "topic5_lbss_claim_adjudication_v0_3",
        "direction": "positive values support the named claim",
        "minimum_of_controls_used_for_inference": False,
        "minimum_heldout_transitions_per_distance_bin": 20,
        "n_interictal_patients": int(interictal.subject.nunique()),
        "n_distance_eligible_patients": int(len(subjects)),
        "n_primary_early_ictal_patients": int(
            early.loc[early.primary.astype(bool), "subject"].nunique()
        ),
        "claim_A": claim_a,
        "claim_B_holm_family": claim_b,
        "claim_C_holm_family": claim_c,
        "claim_D_holm_family": claim_d,
        "claim_logic": {
            "B": "three explicit L3-minus-control distal contrasts",
            "C": "separate endpoint/effective patterns, distal gain and attenuation double dissociation",
            "D1": "canonical full L3 field versus synchronized all-contact null",
            "D2": "three explicit seed-removed contrasts and three explicit attenuation-AUC contrasts",
        },
        "hard_global_gate": False,
        "target_values_read": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/topic5_lbss_full_tissue_rnn_v0_3"),
    )
    args = parser.parse_args()
    out = args.out_root.resolve()
    result = summarize(out)
    (out / "LBSS_CLAIM_SUMMARY_V0_3.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (out / "LBSS_CLAIM_ADJUDICATION_V0_3_COMPLETE.json").write_text(
        json.dumps({
            "status": "PASS",
            "contract": result["contract"],
            "target_values_read": True,
            "minimum_of_controls_used_for_inference": False,
        }, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
