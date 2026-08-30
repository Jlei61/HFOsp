#!/usr/bin/env python3
"""Patient-first aggregation of H2b v0.2 probe results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch as _torch  # noqa: F401
import pandas as pd
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    PRIMARY_LEAD_MINUTES,
    SUPPORT_TIERS,
    V0_2_RESULT_ROOT,
    atomic_csv,
    atomic_json,
    sha256_file,
    utc_now,
)


WRONG_TIME_EFFECT = "correct_minus_wrong_time_conditional_log_loss"
EFFECTS = (
    "state_minus_observation_conditional_log_loss",
    "persistent_minus_memoryless_conditional_log_loss",
    WRONG_TIME_EFFECT,
)
# Each effect is summarised under the tier of the table it was actually
# estimated on.  The wrong-time comparison is fitted on a strictly smaller
# donor-valid seizure population, so reporting it under the primary tier would
# re-hide the very downgrade this alignment was fixed to expose (v0.2 §4).
EFFECT_TIER_COLUMN = {
    EFFECTS[0]: "evaluation_tier",
    EFFECTS[1]: "evaluation_tier",
    WRONG_TIME_EFFECT: "wrong_time_evaluation_tier",
}
ALIGNMENT_KEY = ["patient_id", "lead_minutes"]


def _records(frame: pd.DataFrame) -> list[dict]:
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def _require_unique(frame: pd.DataFrame, label: str) -> None:
    """Prove one row per patient/lead before aligning two tables by that key."""
    missing = [name for name in ALIGNMENT_KEY if name not in frame]
    if missing:
        raise ValueError(f"{label} table lacks alignment columns {missing}")
    duplicated = frame.duplicated(ALIGNMENT_KEY, keep=False)
    if bool(duplicated.any()):
        offenders = (
            frame.loc[duplicated, ALIGNMENT_KEY]
            .drop_duplicates().to_dict(orient="records")
        )
        raise ValueError(
            f"{label} table is not unique per patient/lead; cannot align: {offenders}"
        )


def _tier_rank(value) -> int:
    text = str(value)
    return SUPPORT_TIERS.index(text) if text in SUPPORT_TIERS else len(SUPPORT_TIERS)


def _summary_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    groupings = [("all_checkpoint_available", frame)]
    if "h1_stable_subject" in frame:
        groupings.append(("h1_stable_stratum", frame[frame.h1_stable_subject]))
        groupings.append(("h1_unstable_stratum", frame[~frame.h1_stable_subject]))
    for stratum, subset in groupings:
        for effect in EFFECTS:
            if effect not in subset:
                continue
            tier_column = EFFECT_TIER_COLUMN[effect]
            if tier_column not in subset:
                continue
            scoped = subset[subset[tier_column].notna() & subset[effect].notna()]
            if scoped.empty:
                continue
            for (tier, lead), group in scoped.groupby(
                    [tier_column, "lead_minutes"], sort=True):
                values = group[effect].dropna().to_numpy(dtype=float)
                if not len(values):
                    continue
                favourable = int(np.sum(values < 0))
                rows.append({
                    "stratum": stratum,
                    "evaluation_tier": str(tier),
                    "tier_column": tier_column,
                    "lead_minutes": int(lead),
                    "primary_lead": int(lead) == int(PRIMARY_LEAD_MINUTES),
                    "effect": effect,
                    "n_patients": int(len(values)),
                    "n_favourable": favourable,
                    "patient_median_effect": float(np.median(values)),
                    "patient_q25": float(np.quantile(values, 0.25)),
                    "patient_q75": float(np.quantile(values, 0.75)),
                    "two_sided_exact_sign_p": float(
                        binomtest(favourable, len(values), 0.5).pvalue
                    ),
                    "favourable_direction": "negative",
                    "development_only": True,
                })
    return sorted(
        rows,
        key=lambda row: (
            row["stratum"], row["effect"], row["lead_minutes"],
            row["evaluation_tier"],
        ),
    )


def _wrong_time_evidence(merged: pd.DataFrame, *, table_present: bool) -> dict:
    """State plainly whether correct-vs-wrong-time evidence exists, and at which tier."""
    primary_lead = merged[merged["lead_minutes"].astype(int)
                          == int(PRIMARY_LEAD_MINUTES)]
    estimated = primary_lead[primary_lead[WRONG_TIME_EFFECT].notna()]
    at_primary_tier = estimated[
        estimated["wrong_time_evaluation_tier"].astype(str) == "primary_chronological"
    ]
    downgraded = merged[merged["wrong_time_tier_downgraded"].fillna(False).astype(bool)]
    tiers = sorted(
        merged.loc[merged["wrong_time_evaluation_tier"].notna(),
                   "wrong_time_evaluation_tier"].astype(str).unique().tolist()
    )
    return {
        "wrong_time_table_present": bool(table_present),
        "alignment_key": list(ALIGNMENT_KEY),
        "aligned_on_evaluation_tier": False,
        "n_patients_with_wrong_time_effect_at_primary_lead": int(
            estimated["patient_id"].nunique()
        ),
        "primary_chronological_wrong_time_evidence_exists": bool(
            not at_primary_tier.empty
        ),
        "n_patients_primary_chronological_wrong_time": int(
            at_primary_tier["patient_id"].nunique()
        ),
        "n_patients_wrong_time_tier_downgraded": int(
            downgraded["patient_id"].nunique()
        ),
        "wrong_time_tiers_present": tiers,
        "note": (
            "the wrong-time comparison is estimated on the donor-valid seizure "
            "subset, so it carries its own support tier; a downgrade relative to "
            "the primary table is recorded, never dropped"
        ),
    }


def run(root: Path) -> dict:
    root = root.resolve()
    primary_path = root / "fits/primary/patient_median_probe_metrics.csv"
    wrong_path = root / "fits/matched_wrong_time/patient_median_probe_metrics.csv"
    inventory_path = root / "manifests/r1_7_checkpoint_inventory.json"
    primary = pd.read_csv(primary_path)
    _require_unique(primary, "primary")
    wrong_present = wrong_path.is_file()
    if wrong_present:
        wrong = pd.read_csv(wrong_path)
        _require_unique(wrong, "wrong-time")
        keep = [*ALIGNMENT_KEY, "evaluation_tier", WRONG_TIME_EFFECT]
        wrong = wrong[[name for name in keep if name in wrong]].copy()
        # The wrong-time table keeps its own tier under its own name.  Aligning on
        # evaluation_tier would drop every patient whose donor-valid subset fell a
        # tier below its primary table -- in this cohort that is the only patient
        # with a primary chronological split.
        wrong = wrong.rename(
            columns={"evaluation_tier": "wrong_time_evaluation_tier"}
        )
        merged = primary.merge(
            wrong, on=ALIGNMENT_KEY, how="left", validate="one_to_one",
        )
    else:
        merged = primary.copy()
        merged[WRONG_TIME_EFFECT] = np.nan
        merged["wrong_time_evaluation_tier"] = pd.Series(
            [None] * len(merged), dtype=object,
        )
    if WRONG_TIME_EFFECT not in merged:
        merged[WRONG_TIME_EFFECT] = np.nan
    if "wrong_time_evaluation_tier" not in merged:
        merged["wrong_time_evaluation_tier"] = pd.Series(
            [None] * len(merged), dtype=object,
        )
    merged["wrong_time_tier_downgraded"] = [
        bool(pd.notna(wrong_tier)
             and _tier_rank(wrong_tier) > _tier_rank(primary_tier))
        for primary_tier, wrong_tier in zip(
            merged["evaluation_tier"], merged["wrong_time_evaluation_tier"],
        )
    ]
    inventory = json.loads(inventory_path.read_text())
    stable = set(map(str, inventory.get("h1_stable_subjects") or []))
    merged["h1_stable_subject"] = merged["patient_id"].astype(str).isin(stable)
    merged["h1_is_stratification_not_h2b_gate"] = True
    merged["development_only"] = True
    merged["formal_test_partition_opened"] = False
    merged["sealed_opened"] = False

    support_counts = {}
    for path in (root / "risk_sets").glob("*/input_manifest.json"):
        value = json.loads(path.read_text())
        support_counts[str(value["subject"])] = value
    merged["n_primary_eligible_seizures"] = merged["patient_id"].map(
        lambda subject: support_counts.get(str(subject), {}).get(
            "n_primary_eligible_seizures"
        )
    )
    merged["final_support_tier"] = merged["patient_id"].map(
        lambda subject: support_counts.get(str(subject), {}).get("support_tier")
    )
    if not (
        merged["final_support_tier"].fillna(merged["evaluation_tier"])
        == merged["evaluation_tier"]
    ).all():
        raise ValueError("probe tier disagrees with final raw-reader support tier")

    per_patient_path = root / "reports/per_patient_lead_results.csv"
    atomic_csv(per_patient_path, _records(merged), fieldnames=list(merged.columns))
    summary = pd.DataFrame(_summary_rows(merged))
    summary_path = root / "reports/cohort_patient_first_summary.csv"
    atomic_csv(summary_path, _records(summary), fieldnames=list(summary.columns))

    primary_rows = summary[
        (summary["stratum"] == "all_checkpoint_available")
        & (summary["lead_minutes"] == int(PRIMARY_LEAD_MINUTES))
    ].to_dict(orient="records") if not summary.empty else []
    payload = {
        "status": "COMPLETE",
        "revision": "h2b_cross_task_patient_first_aggregation_v0_2",
        "created_utc": utc_now(),
        "n_patients_with_any_probe_row": int(merged.patient_id.nunique()),
        "n_patients_primary_tier": int(
            merged.loc[
                merged.evaluation_tier == "primary_chronological", "patient_id"
            ].nunique()
        ),
        "primary_30min_evidence_vector": primary_rows,
        "effect_definitions": {
            EFFECTS[0]: "B_state minus B_observation; negative favours persistent state",
            EFFECTS[1]: "persistent state minus memoryless current-window code; negative favours carry",
            EFFECTS[2]: "correct-time state minus matched wrong-time state; negative favours time specificity",
        },
        "correct_vs_wrong_time_evidence": _wrong_time_evidence(
            merged, table_present=wrong_present,
        ),
        "effect_tier_columns": dict(EFFECT_TIER_COLUMN),
        "patient_first": True,
        "seed_aggregation": "median_within_patient_before_cohort_summary",
        "h1_stability_used_as_gate": False,
        "claim_boundary": (
            "development cross-task prediction; not seizure causality, cohort confirmation, "
            "or a mechanism claim"
        ),
        "inputs": {
            str(primary_path): sha256_file(primary_path),
            str(wrong_path): sha256_file(wrong_path) if wrong_path.is_file() else None,
            str(inventory_path): sha256_file(inventory_path),
        },
        "outputs": {
            str(per_patient_path): sha256_file(per_patient_path),
            str(summary_path): sha256_file(summary_path),
        },
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "h3_or_t2_run": False,
    }
    atomic_json(root / "reports/cohort_patient_first_summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=V0_2_RESULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.result_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
