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
    V0_2_RESULT_ROOT,
    atomic_csv,
    atomic_json,
    sha256_file,
    utc_now,
)


EFFECTS = (
    "state_minus_observation_conditional_log_loss",
    "persistent_minus_memoryless_conditional_log_loss",
    "correct_minus_wrong_time_conditional_log_loss",
)


def _records(frame: pd.DataFrame) -> list[dict]:
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def _summary_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    groupings = [("all_checkpoint_available", frame)]
    if "h1_stable_subject" in frame:
        groupings.append(("h1_stable_stratum", frame[frame.h1_stable_subject]))
        groupings.append(("h1_unstable_stratum", frame[~frame.h1_stable_subject]))
    for stratum, subset in groupings:
        for (tier, lead), group in subset.groupby(
                ["evaluation_tier", "lead_minutes"], sort=True):
            for effect in EFFECTS:
                if effect not in group:
                    continue
                values = group[effect].dropna().to_numpy(dtype=float)
                if not len(values):
                    continue
                favourable = int(np.sum(values < 0))
                rows.append({
                    "stratum": stratum,
                    "evaluation_tier": str(tier),
                    "lead_minutes": int(lead),
                    "primary_lead": int(lead) == 30,
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
    return rows


def run(root: Path) -> dict:
    root = root.resolve()
    primary_path = root / "fits/primary/patient_median_probe_metrics.csv"
    wrong_path = root / "fits/matched_wrong_time/patient_median_probe_metrics.csv"
    inventory_path = root / "manifests/r1_7_checkpoint_inventory.json"
    primary = pd.read_csv(primary_path)
    if wrong_path.is_file():
        wrong = pd.read_csv(wrong_path)
        keep = [
            "patient_id", "lead_minutes", "evaluation_tier",
            "correct_minus_wrong_time_conditional_log_loss",
        ]
        wrong = wrong[[name for name in keep if name in wrong]].copy()
        merged = primary.merge(
            wrong, on=["patient_id", "lead_minutes", "evaluation_tier"],
            how="left", validate="one_to_one",
        )
    else:
        merged = primary.copy()
        merged["correct_minus_wrong_time_conditional_log_loss"] = np.nan
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
        & (summary["lead_minutes"] == 30)
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
