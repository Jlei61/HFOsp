#!/usr/bin/env python3
"""Independent contract audit for heldout C5 data-field alignment."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import atomic_write_json, sha256_file  # noqa: E402
from scripts.freeze_topic5_cross_patient_geometry_mappings_v0_2 import MAPPING, SPATIAL  # noqa: E402


DATA = SPATIAL / "data_alignment"


def main() -> None:
    summary = json.loads((DATA / "DATA_ALIGNMENT_SUMMARY.json").read_text())
    seal = json.loads((DATA / "DATA_ALIGNMENT_FREEZE_SEAL.json").read_text())
    quality = pd.read_csv(DATA / "DATA_FIELD_QUALITY.csv")
    within = pd.read_csv(DATA / "WITHIN_PATIENT_ALIGNMENT.csv")
    nulls = pd.read_csv(DATA / "SPATIAL_NULL_ALIGNMENT.csv")
    identity = pd.read_csv(DATA / "CROSS_PATIENT_IDENTITY_ALIGNMENT.csv")
    patients = pd.read_csv(DATA / "C5_PATIENT_EFFECTS.csv")
    mapping_audit = json.loads((MAPPING / "GEOMETRY_REGISTRATION_AUDIT.json").read_text())
    failures = []
    for name, digest in seal.get("files", {}).items():
        if not (DATA / name).is_file() or sha256_file(DATA / name) != digest: failures.append(f"hash:{name}")
    if mapping_audit.get("status") != "PASS" or mapping_audit.get("field_values_read") is not False: failures.append("mapping")
    if summary.get("status") != "COMPLETE" or summary.get("target_values_read") is not False: failures.append("summary")
    if len(quality) != 42: failures.append("quality_denominator")
    if int((quality.status == "DATA_FIELD_NOT_IDENTIFIABLE").sum()) != 4: failures.append("not_identifiable_denominator")
    excluded = set(quality.loc[quality.status.eq("DATA_FIELD_NOT_IDENTIFIABLE"), "fit_id"])
    if set(nulls.fit_id) & excluded or set(identity.fit_id) & excluded: failures.append("ineligible_fit_scored")
    if set(nulls.n_draws.unique()) != {4096}: failures.append("null_draws")
    if set(nulls.null_family.unique()) != {
        "ALL_CONTACT_SYNCHRONIZED", "WITHIN_SHAFT", "DISTANCE_BIN_LOCAL", "GRAPH_SPECTRAL_AUTOCORRELATION"
    }: failures.append("null_families")
    if int(identity.n_cross_patients.min()) < 12: failures.append("identity_denominator")
    expected_tiers = {"generic_all_identifiable": 26, "canonical_ab_shared": 14}
    for tier, n in expected_tiers.items():
        part = patients[patients.tier.eq(tier)]
        if part.patient.nunique() != n: failures.append(f"patient_tier:{tier}")
        if summary["tiers"][tier]["n_patients"] != n: failures.append(f"summary_tier:{tier}")
    if not within.target_values_read.eq(False).all() or not nulls.target_values_read.eq(False).all() or not identity.target_values_read.eq(False).all():
        failures.append("target_leak")
    with np.load(DATA / "FINITE_TIME_RESPONSE_FIELDS.npz", allow_pickle=False) as source:
        if len(source["response"]) != 169290: failures.append("finite_field_rows")
        response_values = np.asarray(source["response"], float)
        finite_field_values = int(np.isfinite(response_values).sum())
        nonfinite_field_values = int((~np.isfinite(response_values)).sum())
        # Unsupported reference/axis cells remain NaN by contract; they are not
        # imputed or converted to zero before fit-level C5 eligibility is checked.
        if finite_field_values == 0: failures.append("no_finite_field_values")
    payload = {
        "contract": "topic5_data_alignment_audit_v0_2", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures else "FAIL", "failure_count": len(failures), "failures": failures,
        "n_fits": 42, "n_data_field_not_identifiable": len(excluded),
        "within_rows": len(within), "spatial_null_rows": len(nulls), "identity_rows": len(identity),
        "finite_field_values": finite_field_values, "nonfinite_unsupported_field_values": nonfinite_field_values,
        "generic_patients": 26, "canonical_patients": 14,
        "sign_policy": "TRAIN_FROZEN_NO_HELDOUT_FLIP", "target_values_read": False,
    }
    atomic_write_json(DATA / "DATA_ALIGNMENT_AUDIT.json", payload)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()
