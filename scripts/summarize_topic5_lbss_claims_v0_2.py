#!/usr/bin/env python3
"""Patient-first adjudication of the four prespecified LBSS claims.

This script does not fit or regenerate a model or field.  It combines the
already frozen interictal, pathway, attenuation, and early-ictal patient tables
and applies one Holm correction within each scientific claim family.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


L0 = "L0_LOCAL_ONLY"
L1 = "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL"
L2 = "L2_LOCAL_PLUS_RANDOM_LR"
L3 = "L3_LOCAL_PLUS_LEARNED_LR"
SHUFFLE = "C_L3_ORDER_SHUFFLED"
REFS = (L0, L1, L2)
TARGET_ARM = {
    "L1_ADDED": L1,
    "L2_ADDED": L2,
    "L3_ADDED": L3,
    "L3_MATCHED_LOCAL": L3,
}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda key: pvalues[key])
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for index, key in enumerate(ordered):
        value = min(1.0, (m - index) * float(pvalues[key]))
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def paired(values: pd.Series | np.ndarray, tolerance: float = 1e-9) -> dict:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    tied = np.abs(x) <= tolerance
    nonzero = x[~tied]
    p = 1.0 if len(nonzero) == 0 else float(
        wilcoxon(nonzero, alternative="two-sided", method="auto").pvalue
    )
    if len(x):
        rng = np.random.default_rng(20260811 + len(x))
        draws = np.median(rng.choice(x, size=(5000, len(x)), replace=True), axis=1)
        ci = np.quantile(draws, [0.025, 0.975]).tolist()
    else:
        ci = [float("nan"), float("nan")]
    return {
        "n": int(len(x)),
        "median": float(np.median(x)) if len(x) else float("nan"),
        "bootstrap_95ci": ci,
        "n_positive": int((x > tolerance).sum()),
        "n_negative": int((x < -tolerance).sum()),
        "n_tied": int(tied.sum()),
        "wilcoxon_p_two_sided": p,
    }


def add_family_holm(stats: dict[str, dict]) -> dict[str, dict]:
    adjusted = holm({key: value["wilcoxon_p_two_sided"] for key, value in stats.items()})
    for key, value in adjusted.items():
        stats[key]["holm_q_within_claim"] = value
    return stats


def attenuation_damage_auc(patient: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    use = patient[(patient.primary) & (patient.endpoint == endpoint)]
    lookup = {(row.subject, row.condition): row for row in use.itertuples()}
    rows = []
    for subject in sorted(use.subject.unique()):
        for target, arm in TARGET_ARM.items():
            base = lookup.get((subject, f"INTACT|{arm}"))
            if base is None:
                continue
            x, damage = [0.0], [0.0]
            for alpha in (0.25, 0.50, 0.75, 1.00):
                item = lookup.get((subject, f"ATTEN|{target}|{alpha:.2f}"))
                if item is not None:
                    x.append(alpha)
                    damage.append(float(base.all_contact_margin - item.all_contact_margin))
            if len(x) == 5:
                rows.append({
                    "subject": subject,
                    "target": target,
                    "damage_auc": float(np.trapz(damage, x)),
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("results/topic5_lbss_rnn_v0_2"))
    args = parser.parse_args()
    out = args.out_root.resolve()
    required = (
        "INTERICTAL_ANALYSIS_COMPLETE.json", "PATHWAY_ANALYSIS_COMPLETE.json",
        "ATTENUATION_COMPLETE.json", "EARLY_ICTAL_SCORING_COMPLETE.json",
    )
    for name in required:
        if not (out / name).exists():
            raise RuntimeError(f"claim adjudication requires {name}")

    interictal = pd.read_csv(out / "interictal_per_patient.csv")
    ip = interictal.pivot(index="subject", columns="arm")
    distal_eligible = np.ones(len(ip.index), dtype=bool)
    for arm in (L0, L1, L2, L3, SHUFFLE):
        distal_eligible &= ip["distal_n"][arm].to_numpy(float) >= 20
    eligible_subjects = ip.index[distal_eligible]

    claim_a = {
        "L0_vs_no_recurrence_all": paired(
            ip.loc[eligible_subjects, "no_rec_contact_nll"][L0]
            - ip.loc[eligible_subjects, "test_contact_nll"][L0]
        )
    }
    claim_b_values = {
        f"L3_vs_{arm}_distal": (
            ip.loc[eligible_subjects, "distal_contact_nll"][arm]
            - ip.loc[eligible_subjects, "distal_contact_nll"][L3]
        ) for arm in REFS
    }
    claim_b = add_family_holm({name: paired(value) for name, value in claim_b_values.items()})

    pathway = pd.read_csv(out / "pathway_analysis" / "true_vs_shuffle_patient_patterns.csv").set_index("subject")
    common = eligible_subjects.intersection(pathway.index)
    pattern_min = np.minimum(
        pathway.loc[common, "endpoint_dissimilarity_beyond_proposal"],
        pathway.loc[common, "effective_dissimilarity_beyond_proposal"],
    )
    true_shuffle_distal = (
        ip.loc[common, "distal_contact_nll"][SHUFFLE]
        - ip.loc[common, "distal_contact_nll"][L3]
    )
    attenuation = pd.read_csv(out / "attenuation" / "attenuation_patient_auc.csv")
    local_ok = attenuation[
        (attenuation.target == "L3_MATCHED_LOCAL") & (attenuation.inferential_eligible == True)  # noqa: E712
    ].subject
    attenuation = attenuation[attenuation.subject.isin(local_ok)]
    aw = attenuation.pivot(index="subject", columns="target", values="auc_distal_selectivity")
    common_c = common.intersection(aw.index)
    dd = aw.loc[common_c, "L3_ADDED"] - aw.loc[common_c, "L3_MATCHED_LOCAL"]
    claim_c = add_family_holm({
        "coarse_pattern_difference_beyond_proposal": paired(pattern_min),
        "true_order_vs_shuffle_distal": paired(true_shuffle_distal),
        "selected_nonlocal_vs_matched_local_attenuation_dd": paired(dd),
    })

    early = pd.read_csv(out / "early_ictal" / "early_ictal_per_patient_condition.csv")
    primary = early[early.primary.astype(bool)]
    canonical = primary[primary.endpoint == "canonical_full"]
    canonical_lookup = {(row.subject, row.condition): row for row in canonical.itertuples()}
    d1 = pd.Series({
        subject: canonical_lookup[(subject, f"INTACT|{L3}")].all_contact_margin
        for subject in sorted(canonical.subject.unique())
        if (subject, f"INTACT|{L3}") in canonical_lookup
    })
    seed = primary[primary.endpoint == "seed_removed"]
    seed_lookup = {(row.subject, row.condition): row for row in seed.itertuples()}
    d2_seed = {}
    for subject in sorted(seed.subject.unique()):
        l3 = seed_lookup.get((subject, f"INTACT|{L3}"))
        refs = [seed_lookup.get((subject, f"INTACT|{arm}")) for arm in REFS]
        if l3 is not None and all(value is not None for value in refs):
            d2_seed[subject] = min(float(l3.all_contact_margin - value.all_contact_margin) for value in refs)
    early_auc = attenuation_damage_auc(primary, "seed_removed")
    early_auc = early_auc[early_auc.subject.isin(set(local_ok))]
    eaw = early_auc.pivot(index="subject", columns="target", values="damage_auc")
    d2_attenuation = pd.Series({
        subject: min(
            float(eaw.loc[subject, "L3_ADDED"] - eaw.loc[subject, other])
            for other in ("L1_ADDED", "L2_ADDED", "L3_MATCHED_LOCAL")
        ) for subject in eaw.index
        if all(name in eaw.columns and np.isfinite(eaw.loc[subject, name])
               for name in ("L3_ADDED", "L1_ADDED", "L2_ADDED", "L3_MATCHED_LOCAL"))
    })
    claim_d = add_family_holm({
        "D1_L3_canonical_full_margin_gt_zero": paired(d1),
        "D2_L3_seed_removed_better_than_all_controls": paired(pd.Series(d2_seed)),
        "D2_L3_attenuation_damage_auc_better_than_all_controls": paired(d2_attenuation),
    })

    result = {
        "contract": "topic5_lbss_claim_adjudication_v0_2",
        "direction": "positive values support the named claim",
        "minimum_heldout_transitions_per_distance_bin": 20,
        "n_interictal_patients": int(interictal.subject.nunique()),
        "n_distance_eligible_patients": int(len(eligible_subjects)),
        "n_primary_early_ictal_patients": int(primary.subject.nunique()),
        "claim_A": claim_a,
        "claim_B_holm_family": claim_b,
        "claim_C_holm_family": claim_c,
        "claim_D_holm_family": claim_d,
        "claim_logic": {
            "B": "L3 must beat L0, L1 and L2 on heldout distal transitions",
            "C_pattern": "minimum of endpoint and effective-influence dissimilarity after subtracting proposal-exposure dissimilarity",
            "C_attenuation": "L3 selected-nonlocal distal-selectivity AUC minus matched-local AUC",
            "D1": "canonical full L3 field versus synchronized all-contact null",
            "D2_seed_removed": "patient-wise minimum L3 improvement over L0, L1 and L2",
            "D2_attenuation": "patient-wise minimum L3 attenuation-damage AUC advantage over extra-local, random-nonlocal and matched-local controls",
        },
        "hard_global_gate": False,
        "target_values_read": True,
    }
    (out / "LBSS_CLAIM_SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "LBSS_CLAIM_ADJUDICATION_COMPLETE.json").write_text(json.dumps({
        "status": "PASS", "target_values_read": True,
        "n_claim_families": 4, "global_hard_gate": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
