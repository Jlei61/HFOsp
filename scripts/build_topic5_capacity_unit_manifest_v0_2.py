#!/usr/bin/env python3
"""Freeze the SEEG training matrix before a single ordered model is trained.

The manifest is the contract: every unit that will ever be run is enumerated
here, with its eligibility already decided from geometry and basis dimension.
Nothing may be added or removed later because of a result.

Two places expand the plan's illustrative per-patient counts, both decided here
and for the same reason — a "patient-median angle-rotated null" is only a median
if more than one angle null exists in that family:

* Core 2 runs the direct family against all four angle nulls (plan §F1 sketched
  one), 17 ordered units per patient instead of 14;
* the time-proxy block runs four angle nulls, 7 units per patient instead of 4.

The learning-curve block shrinks for the mirror-image reason: the geometry,
shaft and free arms do not have a training-fraction-dependent basis, so the
end-to-end and fixed-basis curves share those units instead of training
byte-identical duplicates (26 unique units per patient instead of 32).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_structural_identifiability_v0_2 import (  # noqa: E402
    ANGLE_SUBSET_2,
    ANGLE_SUBSET_4,
    N_ANGLE_NULLS,
    N_IDENTITY_NULLS,
    N_REWIRE_NULLS,
    load_basis_bundle,
)

RESULT_ROOT = ROOT / "results/topic5_capacity_constrained_history_motif_v0_2"
BASIS_ROOT = RESULT_ROOT / "basis"
DIRECT = "DIRECT_HORIZON_UPPER_BOUND"
AUTO = "AUTONOMOUS_SHARED_OPERATOR"
BAG = "ORDERLESS_BAG"
CORE_SEEDS = (0, 1, 2)


def unit_row(**fields) -> dict:
    key = "|".join(
        str(fields[name]) for name in (
            "patient", "prefix_len", "baseline_level", "structure", "null_id",
            "basis_fraction", "data_fraction", "rank", "family", "f_form", "seed", "time_head",
        )
    )
    fields["unit_id"] = key
    fields["unit_hash"] = hashlib.sha256(key.encode()).hexdigest()[:16]
    fields["output_dir"] = f"units/{fields['patient']}/{fields['unit_hash']}"
    return fields


def expand_patient(patient: str, available: set[str], aligned_max_rank: int,
                   angle_eligible: bool) -> list[dict]:
    rows: list[dict] = []

    def add(block: str, structure: str, null_id: str, family: str, *, rank: int = 4,
            data_fraction: int = 100, basis_fraction: int = 100, level: str = "U_FULL_SET",
            seed: int = 0, f_form: str = "FULL", prefix_len: int = 3, time_head: bool = False) -> None:
        if structure == "H1_FREE_LOW_RANK":
            basis_key = ""
            eligible, reason = True, ""
        else:
            kind = {
                "H1_GEOMETRY_LAYOUT": "GEOMETRY_LAYOUT",
                "H1_SHAFT_GRADIENT": "SHAFT_GRADIENT",
                "H1_PATIENT_ALIGNED": "PATIENT_ALIGNED",
                "H1_ALIGNED_ORDERLESS_BAG": "PATIENT_ALIGNED",
                "H1_ANGLE_ROTATED_AXIS": "ANGLE_ROTATED_AXIS",
                "H1_IDENTITY_PERMUTED": "IDENTITY_PERMUTED",
                "H1_LOCALITY_REWIRED": "LOCALITY_REWIRED",
            }[structure]
            basis_key = f"{kind}|{null_id}|f{basis_fraction}|r{rank}"
            eligible = basis_key in available
            reason = "" if eligible else (
                "ANGLE_NULL_INELIGIBLE" if kind == "ANGLE_ROTATED_AXIS" and not angle_eligible
                else "RANK_EXCEEDS_BASIS_DIMENSION" if rank > aligned_max_rank
                else "BASIS_NOT_BUILT"
            )
        rows.append(unit_row(
            block=block, patient=patient, prefix_len=prefix_len, baseline_level=level,
            structure=structure, null_id=null_id, basis_fraction=basis_fraction,
            data_fraction=data_fraction, rank=rank, family=family, f_form=f_form, seed=seed,
            time_head=time_head, basis_key=basis_key, eligible=eligible, ineligible_reason=reason,
            split_scored="2",
        ))

    # ---- Core 1: full null family, r=4, 100% data, strong unordered bypass --
    for family in (DIRECT, AUTO):
        for structure in ("H1_GEOMETRY_LAYOUT", "H1_PATIENT_ALIGNED", "H1_FREE_LOW_RANK"):
            for seed in CORE_SEEDS:
                add("CORE1", structure, "observed", family, seed=seed)
        add("CORE1", "H1_SHAFT_GRADIENT", "observed", family)
        for index in range(N_ANGLE_NULLS):
            add("CORE1", "H1_ANGLE_ROTATED_AXIS", f"angle{index}", family)
        for index in range(N_IDENTITY_NULLS):
            add("CORE1", "H1_IDENTITY_PERMUTED", f"permute{index}", family)
        for index in range(N_REWIRE_NULLS):
            add("CORE1", "H1_LOCALITY_REWIRED", f"rewire{index}", family)
    add("CORE1", "H1_ALIGNED_ORDERLESS_BAG", "observed", BAG)

    # ---- Core 2: same comparison with the weak unordered bypass ------------
    for structure in ("H1_GEOMETRY_LAYOUT", "H1_SHAFT_GRADIENT"):
        add("CORE2", structure, "observed", AUTO, level="U_MINIMAL")
    for structure in ("H1_PATIENT_ALIGNED", "H1_FREE_LOW_RANK"):
        for seed in (0, 1):
            add("CORE2", structure, "observed", AUTO, level="U_MINIMAL", seed=seed)
    for index in ANGLE_SUBSET_4:
        add("CORE2", "H1_ANGLE_ROTATED_AXIS", f"angle{index}", AUTO, level="U_MINIMAL")
    add("CORE2", "H1_ALIGNED_ORDERLESS_BAG", "observed", BAG, level="U_MINIMAL")
    for structure in ("H1_PATIENT_ALIGNED", "H1_FREE_LOW_RANK"):
        add("CORE2", structure, "observed", DIRECT, level="U_MINIMAL")
    for index in ANGLE_SUBSET_4:
        add("CORE2", "H1_ANGLE_ROTATED_AXIS", f"angle{index}", DIRECT, level="U_MINIMAL")

    # ---- capacity curve ----------------------------------------------------
    for rank in (1, 2, 8):
        for structure in ("H1_GEOMETRY_LAYOUT", "H1_SHAFT_GRADIENT", "H1_PATIENT_ALIGNED",
                          "H1_FREE_LOW_RANK"):
            add("CAPACITY", structure, "observed", AUTO, rank=rank)
        for index in ANGLE_SUBSET_2:
            add("CAPACITY", "H1_ANGLE_ROTATED_AXIS", f"angle{index}", AUTO, rank=rank)

    # ---- learning curves ---------------------------------------------------
    for fraction in (25, 50):
        for structure in ("H1_GEOMETRY_LAYOUT", "H1_SHAFT_GRADIENT", "H1_FREE_LOW_RANK"):
            add("LEARNING", structure, "observed", AUTO, data_fraction=fraction)
        for basis_fraction in (fraction, 100):
            add("LEARNING", "H1_PATIENT_ALIGNED", "observed", AUTO,
                data_fraction=fraction, basis_fraction=basis_fraction)
            for index in ANGLE_SUBSET_4:
                add("LEARNING", "H1_ANGLE_ROTATED_AXIS", f"angle{index}", AUTO,
                    data_fraction=fraction, basis_fraction=basis_fraction)

    # ---- spectral-centroid latency proxy -----------------------------------
    for structure in ("H1_GEOMETRY_LAYOUT", "H1_PATIENT_ALIGNED", "H1_FREE_LOW_RANK"):
        add("TIME_PROXY", structure, "observed", AUTO, time_head=True)
    for index in ANGLE_SUBSET_4:
        add("TIME_PROXY", "H1_ANGLE_ROTATED_AXIS", f"angle{index}", AUTO, time_head=True)

    # ---- transition-form sensitivity (spec §4.10) --------------------------
    for f_form in ("DIAGONAL_ONLY", "BANDWIDTH_1", "STABLE_NORMAL", "LOW_DIMENSIONAL_TANH"):
        add("F_FORM_SENSITIVITY", "H1_PATIENT_ALIGNED", "observed", AUTO, f_form=f_form)
        for index in ANGLE_SUBSET_2:
            add("F_FORM_SENSITIVITY", "H1_ANGLE_ROTATED_AXIS", f"angle{index}", AUTO, f_form=f_form)

    # ---- harder prefix sensitivity (spec §5.1) -----------------------------
    for structure in ("H1_PATIENT_ALIGNED", "H1_FREE_LOW_RANK"):
        add("PREFIX2_SENSITIVITY", structure, "observed", AUTO, prefix_len=2)
    for index in ANGLE_SUBSET_4:
        add("PREFIX2_SENSITIVITY", "H1_ANGLE_ROTATED_AXIS", f"angle{index}", AUTO, prefix_len=2)

    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RESULT_ROOT))
    arguments = parser.parse_args()

    eligibility = pd.read_csv(BASIS_ROOT / "BASIS_ELIGIBILITY.csv").set_index("patient")
    rows: list[dict] = []
    for patient in eligibility.index:
        _, index = load_basis_bundle(BASIS_ROOT / "per_patient" / f"{patient}.npz")
        available = {entry["key"] for entry in index}
        rows.extend(expand_patient(
            patient, available,
            int(eligibility.loc[patient, "aligned_family_max_rank"]),
            bool(eligibility.loc[patient, "angle_null_eligible"]),
        ))

    table = pd.DataFrame(rows)
    if table["unit_id"].duplicated().any():
        duplicated = table.loc[table["unit_id"].duplicated(), "unit_id"].head().tolist()
        raise SystemExit(f"duplicate unit ids in manifest: {duplicated}")
    out = Path(arguments.out)
    table.to_csv(out / "MASTER_UNIT_MANIFEST.csv", index=False)
    summary = {
        "contract": "topic5_capacity_constrained_history_motif_v0_2_master_unit_manifest",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "n_patients": int(table["patient"].nunique()),
        "n_units_total": int(len(table)),
        "n_units_eligible": int(table["eligible"].sum()),
        "n_units_ineligible": int((~table["eligible"]).sum()),
        "ineligible_reasons": table.loc[~table["eligible"], "ineligible_reason"]
        .value_counts().to_dict(),
        "per_block": table.groupby("block")["eligible"].agg(["size", "sum"]).rename(
            columns={"size": "planned", "sum": "eligible"}).to_dict("index"),
        "per_family": table.loc[table["eligible"]].groupby("family").size().to_dict(),
        "plan_deviations": [
            "CORE2 direct family runs all four angle nulls (plan §F1 sketched one) so the "
            "patient-median angle null is a median in both families",
            "TIME_PROXY runs four angle nulls for the same reason",
            "LEARNING shares the geometry/shaft/free units between the end-to-end and "
            "fixed-basis curves because those bases do not depend on the training fraction",
            "spec §4.10 transition-form sensitivities and spec §5.1 prefix=2 sensitivity are "
            "enumerated here so they are frozen with the rest of the matrix",
        ],
        "angle_subsets": {"four": list(ANGLE_SUBSET_4), "two": list(ANGLE_SUBSET_2)},
    }
    (out / "MASTER_UNIT_MANIFEST.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"units planned : {len(table)}")
    print(f"units eligible: {int(table['eligible'].sum())}")
    print(f"ineligible    : {summary['ineligible_reasons']}")
    print(table.groupby("block")["eligible"].agg(["size", "sum"]).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
