#!/usr/bin/env python3
"""Freeze patient inputs and the observation-independent search basis."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# lstsq is not bit-stable across BLAS thread counts, and this basis is frozen.
for _name in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
    os.environ.setdefault(f"{_name}_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_rev9l_forced_source_worker import _runtime_provenance  # noqa: E402
from src.topic4_patient_specific_field_cohort import (  # noqa: E402
    atomic_json,
    load_config,
    load_subject_contract,
    projected_field_basis,
    sha256,
    verify_inputs,
)


DEFAULT_CONFIG = ROOT / "config/topic4_patient_specific_field_connectivity_cohort_v2.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    verify_inputs(config, code_root=ROOT)
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("cannot freeze a dirty config")
    provenance = _runtime_provenance(args.expected_commit)
    if provenance["runtime_modules_dirty"] or not provenance["runtime_modules_match_expected_commit"]:
        raise RuntimeError("freezer modules differ from expected commit")

    target_audit_path = Path(config["source_workspace"]) / config["inputs"]["target_audit"]["path"]
    target_audit = json.loads(target_audit_path.read_text())
    rows = []
    for source_row in target_audit["subjects"]:
        subject_id = source_row["subject_id"]
        contract = load_subject_contract(config, subject_id)
        real = contract["real_coords_sheet"]
        rows.append({
            "subject_id": subject_id,
            "patient_target_eligible": bool(source_row["patient_target_eligible"]),
            "real_geometry_eligible": real is not None,
            "n_contacts": len(contract["contact_order"]),
            "n_shafts": len({name.rstrip("0123456789") for name in contract["contact_order"]}),
            "development_source": subject_id == config["cohort"]["development_source_subject"],
            "input_hashes": contract["hashes"],
            "real_geometry_span_mm": None if real is None else (
                np.ptp(np.asarray(real, float), axis=0).tolist()
            ),
        })
    if len(rows) != int(config["cohort"]["target_denominator"]):
        raise RuntimeError("target denominator changed")
    eligible = [row for row in rows if row["real_geometry_eligible"]]
    if len(eligible) != int(config["cohort"]["real_geometry_denominator"]):
        raise RuntimeError("real-geometry denominator changed")

    basis = projected_field_basis(config)
    if int(basis["direction_count"]) + int(
        config["local_connectivity"]["coefficient_count"]
    ) != int(config["search"]["dimension"]):
        raise RuntimeError("search dimension does not match field plus edge basis")
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    atomic_json({
        "status": "PATIENT_SPECIFIC_DATA_MANIFEST_FROZEN",
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256(config_path)},
        "n_targets": len(rows), "n_real_geometry": len(eligible),
        "n_primary_nondevelopment": sum(
            row["real_geometry_eligible"] and not row["development_source"] for row in rows
        ),
        "subjects": rows, "provenance": provenance,
    }, output / "DATA_MANIFEST.json")
    atomic_json({
        "status": "REAL_GEOMETRY_ELIGIBILITY_FROZEN",
        "eligible_subjects": [row["subject_id"] for row in eligible],
        "not_evaluable_subjects": [
            row["subject_id"] for row in rows if not row["real_geometry_eligible"]
        ],
        "canonical_substitution_forbidden": True,
    }, output / "GEOMETRY_ELIGIBILITY.json")
    np.savez_compressed(
        output / "SEARCH_BASIS.npz",
        directions=np.asarray(basis["directions"], np.float64),
        wavevectors_per_mm=np.asarray(basis["wavevectors_per_mm"], np.float64),
    )
    atomic_json({
        "status": "OBSERVATION_INDEPENDENT_SEARCH_BASIS_FROZEN",
        "representation": "uniform_whole_sheet_fourier_projected_to_cubic_bspline",
        "direction_count": basis["direction_count"],
        "direction_sha256": basis["direction_sha256"],
        "maximum_projection_rmse": basis["maximum_projection_rmse"],
        "uses_contact_geometry": basis["uses_contact_geometry"],
        "component_count": None, "peak_count_constraint": None,
        "basis_npz": str(output / "SEARCH_BASIS.npz"),
        "basis_npz_sha256": sha256(output / "SEARCH_BASIS.npz"),
    }, output / "SEARCH_BASIS.json")
    print(json.dumps({
        "status": "FROZEN", "targets": len(rows), "real_geometry": len(eligible),
        "basis_directions": basis["direction_count"],
    }))


if __name__ == "__main__":
    main()
