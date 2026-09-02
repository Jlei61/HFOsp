#!/usr/bin/env python3
"""Post-hoc sign-semantics audit for the preregistered progress response.

The preregistered output axis is the centered mean rank field, which assigns
larger values to earlier contacts.  The positive hidden tangent, however, is
oriented toward increasing event phase.  This audit never rewrites the
preregistered C3 result.  It reports two explicitly post-hoc diagnostics:

* laterness-oriented selectivity: ``-R_prog<-prog - |R_field<-prog|``;
* sign-invariant selectivity: ``|R_prog<-prog| - |R_field<-prog|``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import atomic_write_csv, atomic_write_json  # noqa: E402
from scripts.summarize_topic5_latent_geometry_v0_2 import one_sided_summary  # noqa: E402


OUT = ROOT / "results" / "topic5_latent_propagation_landscape_v0_2"
SOURCE = OUT / "axis_perturbation" / "responses" / "C3_CELL_PHASE_RESPONSE.csv"
DESTINATION = OUT / "axis_perturbation" / "responses"
C5_SOURCE = OUT / "spatial_control_field" / "data_alignment" / "C5_PATIENT_EFFECTS.csv"
REAL_ARMS = ("L0", "L1", "L2m", "L3")


def aggregate(values: pd.DataFrame, canonical_only: bool) -> pd.DataFrame:
    selected = values[values["canonical_ab"]].copy() if canonical_only else values.copy()
    metrics = [
        "D_progress_preregistered",
        "D_progress_laterness_posthoc",
        "D_progress_magnitude_posthoc",
        "R_progress_from_progress",
        "R_field_from_progress",
    ]
    seed = selected.groupby(
        ["patient", "fit_id", "geometry_view", "public_arm", "seed"], as_index=False
    )[metrics].mean()
    fit = seed.groupby(
        ["patient", "fit_id", "geometry_view", "public_arm"], as_index=False
    )[metrics].median()
    arm = fit.groupby(["patient", "public_arm"], as_index=False)[metrics].median()
    rows: list[dict[str, object]] = []
    tier = "canonical_ab_shared" if canonical_only else "generic_all_identifiable"
    for patient, group in arm.groupby("patient"):
        indexed = group.set_index("public_arm")
        if not set(REAL_ARMS).issubset(indexed.index):
            continue
        real = indexed.loc[list(REAL_ARMS), metrics]
        rows.append({
            "tier": tier,
            "patient": str(patient),
            **{metric: float(real[metric].median()) for metric in metrics},
            "target_values_read": False,
            "analysis_role": "POSTHOC_SIGN_SEMANTICS_ONLY",
        })
    return pd.DataFrame(rows)


def main() -> None:
    source = pd.read_csv(SOURCE)
    if source["target_values_read"].astype(bool).any():
        raise RuntimeError("C3 source unexpectedly reports target access")
    source["D_progress_preregistered"] = source["D_progress"]
    source["D_progress_laterness_posthoc"] = (
        -source["R_progress_from_progress"] - source["R_field_from_progress"].abs()
    )
    source["D_progress_magnitude_posthoc"] = (
        source["R_progress_from_progress"].abs() - source["R_field_from_progress"].abs()
    )
    patient = pd.concat(
        [aggregate(source, False), aggregate(source, True)], ignore_index=True
    )
    atomic_write_csv(DESTINATION / "PROGRESS_SIGN_SEMANTICS_PATIENT_EFFECTS.csv", patient)

    tiers: dict[str, object] = {}
    for tier, group in patient.groupby("tier", sort=False):
        tiers[str(tier)] = {
            metric: one_sided_summary(group[metric].to_numpy(float), seed=7301 + index)
            for index, metric in enumerate((
                "D_progress_preregistered",
                "D_progress_laterness_posthoc",
                "D_progress_magnitude_posthoc",
                "R_progress_from_progress",
            ))
        }
    c5_source = pd.read_csv(C5_SOURCE)
    c5_tiers: dict[str, object] = {}
    for tier, group in c5_source.groupby("tier", sort=False):
        c5_tiers[str(tier)] = {
            "preregistered_earlyness_spatial_margin": one_sided_summary(
                group["progress_spatial_null_margin"].to_numpy(float), seed=7351
            ),
            "laterness_spatial_margin_posthoc": one_sided_summary(
                -group["progress_spatial_null_margin"].to_numpy(float), seed=7352
            ),
            "preregistered_earlyness_identity_margin": one_sided_summary(
                group["progress_identity_margin"].to_numpy(float), seed=7353
            ),
            "laterness_identity_margin_posthoc": one_sided_summary(
                -group["progress_identity_margin"].to_numpy(float), seed=7354
            ),
        }
    payload = {
        "contract": "topic5_progress_response_sign_semantics_audit_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_POSTHOC_DIAGNOSTIC",
        "source": str(SOURCE.relative_to(ROOT)),
        "semantic_facts": {
            "rank_field_definition": "1_minus_normalized_rank; larger means earlier",
            "positive_hidden_tangent": "gamma(s+delta)-gamma(s); increasing event phase",
            "sign_ambiguity": (
                "A phase-advancing hidden perturbation may favor later contacts and therefore "
                "project negatively onto the preregistered static earlyness field."
            ),
        },
        "primary_integrity": {
            "preregistered_D_progress_changed": False,
            "C3_status_changed": False,
            "C5_status_changed": False,
            "diagnostics_may_rescue_primary": False,
            "target_values_read": False,
        },
        "tiers": tiers,
        "C5_progress_orientation_sensitivity": {
            "status": "POSTHOC_TARGET_FREE_DIAGNOSTIC",
            "algebra": "Flipping the response sign flips observed, null, cross-patient, and therefore margin signs exactly.",
            "tiers": c5_tiers,
            "C5_status_changed": False,
        },
        "interpretation_boundary": (
            "Laterness and magnitude results diagnose an orientation mismatch only; they are "
            "post-hoc and cannot support the preregistered C3 progress-control claim."
        ),
    }
    atomic_write_json(DESTINATION / "PROGRESS_SIGN_SEMANTICS_AUDIT.json", payload)
    print(DESTINATION / "PROGRESS_SIGN_SEMANTICS_AUDIT.json")


if __name__ == "__main__":
    main()
