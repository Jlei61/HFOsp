#!/usr/bin/env python3
"""Phase C1 + C2: frozen spatial bases and the matched null family (SEEG).

Runs before any ordered model is trained, and before any model result exists, so
no basis or null can be selected on performance.  Everything it writes is
immutable input to Phase F.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_strict_history_data_v0_2 import (  # noqa: E402
    DATA_FRACTIONS,
    PRIMARY_PREFIX_LEN,
    load_sample_set,
    load_seeg_patient,
)
from src.topic5_structural_identifiability_v0_2 import (  # noqa: E402
    ANGLE_GRID_RAD,
    ANGLE_NULL_INELIGIBLE,
    ANGLE_SUBSET_4,
    IDENTITY_SEED,
    N_IDENTITY_NULLS,
    N_REWIRE_NULLS,
    RANKS,
    REWIRE_SEED,
    BasisRecord,
    aligned_dictionary,
    estimate_axis_2d,
    geometry_basis,
    identity_permutation,
    isotropic_kernel,
    local_graph,
    local_kernel_sigma,
    orthonormal_truncation,
    principal_angles,
    rewire_support,
    rotate_axis,
    save_basis_bundle,
    shaft_basis,
    shaft_indicator_matrix,
)

FRAME_ROOT = ROOT / "results/topic5_dynamical_motif_rnn_v0_1/frame_cache/GEOMETRY_ONLY_PCA2"
RESULT_ROOT = ROOT / "results/topic5_capacity_constrained_history_motif_v0_2"
BASIS_ROOT = RESULT_ROOT / "basis"


def build_for_patient(name: str) -> dict:
    patient = load_seeg_patient(FRAME_ROOT, name)
    samples = load_sample_set(RESULT_ROOT / "sample_cache" / f"prefix{PRIMARY_PREFIX_LEN}" / f"{name}.npz")
    coords_3d = patient.coords_3d_mm
    coords_2d = patient.contacts_xy_mm
    shafts = patient.shafts
    n_contacts = patient.n_contacts
    n_shafts = len(set(shafts))
    sigma = local_kernel_sigma(coords_3d)
    support = local_graph(coords_3d)
    kernel = isotropic_kernel(coords_3d, sigma, support)

    geometry_max = min(max(RANKS), n_contacts)
    aligned_max = min(max(RANKS), max(1, n_contacts - n_shafts))
    records: list[BasisRecord] = []
    manifest_rows: list[dict] = []
    null_rows: list[dict] = []

    def register(kind: str, null_id: str, fraction: int, full: np.ndarray,
                 singular: np.ndarray, metadata: dict, max_rank: int) -> None:
        for rank in RANKS:
            if rank > max_rank:
                manifest_rows.append({
                    "patient": name, "kind": kind, "null_id": null_id, "fraction": fraction,
                    "rank": rank, "eligible": False,
                    "ineligible_reason": "RANK_EXCEEDS_BASIS_DIMENSION",
                    "basis_dimension_limit": max_rank, **metadata,
                })
                continue
            record = BasisRecord(
                patient=name, kind=kind, null_id=f"{null_id}|f{fraction}", rank=rank,
                basis=np.ascontiguousarray(full[:, :rank]),
                singular_values=singular[:rank], metadata={"fraction": fraction, **metadata},
            )
            records.append(record)
            manifest_rows.append({
                "patient": name, "kind": kind, "null_id": null_id, "fraction": fraction,
                "rank": rank, "eligible": True, "ineligible_reason": "",
                "basis_dimension_limit": max_rank,
                "orthogonality_error": record.orthogonality_error(),
                "shaft_projection_norm": record.shaft_projection(shafts),
                "singular_value_first": float(singular[0]),
                "singular_value_last": float(singular[rank - 1]),
                "sha256": record.hash(), **metadata,
            })

    # -- event-free bases ---------------------------------------------------
    geometry_full, geometry_sv = geometry_basis(kernel, geometry_max)
    register("GEOMETRY_LAYOUT", "observed", 100, geometry_full, geometry_sv,
             {"reads_events": False}, geometry_max)
    shaft_full, shaft_sv = shaft_basis(shafts, coords_3d, min(max(RANKS), n_contacts))
    register("SHAFT_GRADIENT", "observed", 100, shaft_full, shaft_sv,
             {"reads_events": False}, min(max(RANKS), shaft_full.shape[1]))

    # -- patient-aligned axis per training fraction -------------------------
    geometry_class = patient.provenance["geometry_class"]
    angle_eligible = geometry_class == "TWO_DIMENSIONAL"
    axis_by_fraction: dict[int, np.ndarray] = {}
    axis_stats: dict[int, dict] = {}
    for fraction in DATA_FRACTIONS:
        rows = np.flatnonzero(samples.fraction_mask(fraction))
        start = np.asarray(samples.start_set[rows], dtype=np.float64)
        start_centroid = (start @ coords_2d) / np.maximum(start.sum(axis=1, keepdims=True), 1.0)
        displacement = np.asarray(samples.late_field_centroid[rows], dtype=np.float64) - start_centroid
        axis, stats = estimate_axis_2d(displacement)
        axis_by_fraction[fraction] = axis
        axis_stats[fraction] = stats
        dictionary = aligned_dictionary(kernel, coords_3d, coords_2d, axis, shafts)
        full, singular = orthonormal_truncation(dictionary, aligned_max)
        register("PATIENT_ALIGNED", "observed", fraction, full, singular,
                 {"reads_events": True, "axis_x": float(axis[0]), "axis_y": float(axis[1]),
                  "axis_anisotropy_ratio": stats["anisotropy_ratio"],
                  "n_basis_events": stats["n_displacements"]}, aligned_max)

        indices = range(len(ANGLE_GRID_RAD)) if fraction == 100 else ANGLE_SUBSET_4
        for index in indices:
            angle = ANGLE_GRID_RAD[index]
            if not angle_eligible:
                manifest_rows.append({
                    "patient": name, "kind": "ANGLE_ROTATED_AXIS", "null_id": f"angle{index}",
                    "fraction": fraction, "rank": -1, "eligible": False,
                    "ineligible_reason": ANGLE_NULL_INELIGIBLE,
                    "basis_dimension_limit": aligned_max, "geometry_class": geometry_class,
                })
                continue
            rotated = rotate_axis(axis, angle)
            rotated_dictionary = aligned_dictionary(kernel, coords_3d, coords_2d, rotated, shafts)
            rotated_full, rotated_sv = orthonormal_truncation(rotated_dictionary, aligned_max)
            register("ANGLE_ROTATED_AXIS", f"angle{index}", fraction, rotated_full, rotated_sv,
                     {"reads_events": True, "rotation_rad": float(angle),
                      "axis_x": float(rotated[0]), "axis_y": float(rotated[1])}, aligned_max)
            null_rows.append({
                "patient": name, "kind": "ANGLE_ROTATED_AXIS", "null_id": f"angle{index}",
                "fraction": fraction, "rotation_rad": float(angle),
                "singular_value_ratio_to_aligned": float(rotated_sv[0] / singular[0]),
                "kernel_identical": True, "anisotropy_strength_identical": True,
                "contact_identity_overlap": 1.0, "unmatched": "",
            })

    # -- identity permuted --------------------------------------------------
    aligned_dict_100 = aligned_dictionary(kernel, coords_3d, coords_2d, axis_by_fraction[100], shafts)
    aligned_full_100, aligned_sv_100 = orthonormal_truncation(aligned_dict_100, aligned_max)
    for index in range(N_IDENTITY_NULLS):
        permutation = identity_permutation(coords_3d, shafts, support, IDENTITY_SEED + 101 * index)
        permuted = aligned_full_100[permutation]
        register("IDENTITY_PERMUTED", f"permute{index}", 100, permuted, aligned_sv_100,
                 {"reads_events": True, "moved_contacts": int((permutation != np.arange(n_contacts)).sum())},
                 aligned_max)
        null_rows.append({
            "patient": name, "kind": "IDENTITY_PERMUTED", "null_id": f"permute{index}",
            "fraction": 100, "rotation_rad": float("nan"),
            "singular_value_ratio_to_aligned": 1.0, "kernel_identical": True,
            "anisotropy_strength_identical": True,
            "contact_identity_overlap": float((permutation == np.arange(n_contacts)).mean()),
            "unmatched": "",
        })

    # -- locality rewired ---------------------------------------------------
    for index in range(N_REWIRE_NULLS):
        rewired_support, report = rewire_support(coords_3d, shafts, support, REWIRE_SEED + 211 * index)
        rewired_kernel = isotropic_kernel(coords_3d, sigma, rewired_support)
        rewired_dictionary = aligned_dictionary(
            rewired_kernel, coords_3d, coords_2d, axis_by_fraction[100], shafts
        )
        rewired_full, rewired_sv = orthonormal_truncation(rewired_dictionary, aligned_max)
        register("LOCALITY_REWIRED", f"rewire{index}", 100, rewired_full, rewired_sv,
                 {"reads_events": True, "accepted_swaps": report["accepted_swaps"],
                  "rewire_degenerate": "REWIRE_DEGENERATE" in report["unmatched"]}, aligned_max)
        null_rows.append({
            "patient": name, "kind": "LOCALITY_REWIRED", "null_id": f"rewire{index}",
            "fraction": 100, "rotation_rad": float("nan"),
            "singular_value_ratio_to_aligned": float(rewired_sv[0] / aligned_sv_100[0]),
            "kernel_identical": False, "anisotropy_strength_identical": True,
            "contact_identity_overlap": report["contact_identity_overlap"],
            "unmatched": ",".join(report["unmatched"]),
            **{f"observed_{key}": value for key, value in report["observed"].items()},
            **{f"rewired_{key}": value for key, value in report["rewired"].items()},
        })

    bundle_path = BASIS_ROOT / "per_patient" / f"{name}.npz"
    bundle_hash = save_basis_bundle(records, bundle_path)

    # -- principal angles against the aligned basis -------------------------
    angle_rows = []
    reference_rank = 4 if aligned_max >= 4 else aligned_max
    reference = aligned_full_100[:, :reference_rank]
    comparisons = {
        "GEOMETRY_LAYOUT": geometry_full[:, :min(reference_rank, geometry_max)],
        "SHAFT_GRADIENT": shaft_full[:, :min(reference_rank, shaft_full.shape[1])],
        "IDENTITY_PERMUTED": aligned_full_100[
            identity_permutation(coords_3d, shafts, support, IDENTITY_SEED)][:, :reference_rank],
    }
    if angle_eligible:
        for index in ANGLE_SUBSET_4:
            rotated = rotate_axis(axis_by_fraction[100], ANGLE_GRID_RAD[index])
            rotated_dictionary = aligned_dictionary(kernel, coords_3d, coords_2d, rotated, shafts)
            comparisons[f"ANGLE_ROTATED_AXIS|angle{index}"] = orthonormal_truncation(
                rotated_dictionary, aligned_max
            )[0][:, :reference_rank]
    for label, other in comparisons.items():
        angles = principal_angles(reference, other)
        angle_rows.append({
            "patient": name, "reference": "PATIENT_ALIGNED", "comparison": label,
            "rank": reference_rank, "n_angles": len(angles),
            "min_angle_rad": float(np.min(angles)), "max_angle_rad": float(np.max(angles)),
            "mean_angle_rad": float(np.mean(angles)),
            "subspace_overlap": float(np.mean(np.cos(angles) ** 2)),
        })

    return {
        "patient": name,
        "manifest": manifest_rows,
        "nulls": null_rows,
        "angles": angle_rows,
        "summary": {
            "patient": name,
            "n_contacts": n_contacts,
            "n_shafts": n_shafts,
            "sigma_mm": sigma,
            "local_graph_edges": int(support.sum() // 2),
            "geometry_class": geometry_class,
            "angle_null_eligible": bool(angle_eligible),
            "aligned_family_max_rank": int(aligned_max),
            "geometry_family_max_rank": int(geometry_max),
            "bundle_sha256": bundle_hash,
            "axis_by_fraction": {
                str(fraction): {
                    "axis": [float(v) for v in axis_by_fraction[fraction]],
                    **axis_stats[fraction],
                } for fraction in DATA_FRACTIONS
            },
            "shaft_indicator_rank": int(np.linalg.matrix_rank(shaft_indicator_matrix(shafts))),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=14)
    arguments = parser.parse_args()
    BASIS_ROOT.mkdir(parents=True, exist_ok=True)
    names = sorted(path.name for path in FRAME_ROOT.iterdir() if path.is_dir())
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        results = list(pool.map(build_for_patient, names))

    pd.DataFrame([row for entry in results for row in entry["manifest"]]).to_csv(
        BASIS_ROOT / "STRUCTURE_BASIS_MANIFEST.csv", index=False
    )
    pd.DataFrame([row for entry in results for row in entry["angles"]]).to_csv(
        BASIS_ROOT / "BASIS_PRINCIPAL_ANGLES.csv", index=False
    )
    null_audit = {
        "contract": "topic5_capacity_constrained_history_motif_v0_2_null_match_audit",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "angle_grid_rad": list(ANGLE_GRID_RAD),
        "angle_subset_for_peripheral_blocks": list(ANGLE_SUBSET_4),
        "n_identity_nulls": N_IDENTITY_NULLS,
        "n_rewire_nulls": N_REWIRE_NULLS,
        "per_patient": {entry["patient"]: entry["summary"] for entry in results},
        "null_rows": [row for entry in results for row in entry["nulls"]],
    }
    (BASIS_ROOT / "NULL_MATCH_AUDIT.json").write_text(json.dumps(null_audit, indent=2) + "\n")
    summary = pd.DataFrame([entry["summary"] for entry in results])
    summary.drop(columns=["axis_by_fraction"]).to_csv(BASIS_ROOT / "BASIS_ELIGIBILITY.csv", index=False)

    print(f"patients: {len(results)}")
    print(f"  angle-null eligible : {int(summary['angle_null_eligible'].sum())}/{len(summary)}")
    for rank in RANKS:
        eligible = int((summary["aligned_family_max_rank"] >= rank).sum())
        print(f"  aligned family r={rank}: {eligible}/{len(summary)} patients eligible")
    degenerate = [row for entry in results for row in entry["nulls"]
                  if "REWIRE_DEGENERATE" in str(row.get("unmatched", ""))]
    print(f"  degenerate rewire nulls: {len(degenerate)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
