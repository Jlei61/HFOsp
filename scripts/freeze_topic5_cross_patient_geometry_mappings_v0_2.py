#!/usr/bin/env python3
"""Freeze field-blind cross-patient transports in normalized propagation-axis space."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.propagation_skeleton_geometry import parse_shaft  # noqa: E402
from src.topic5_latent_landscape_v0_2 import (  # noqa: E402
    atomic_write_csv, atomic_write_json, canonical_json_sha256, sha256_file,
)
from scripts.run_topic5_latent_pass1_v0_2 import OUT, PARENT  # noqa: E402


SPATIAL = OUT / "spatial_control_field"
MAPPING = SPATIAL / "cross_patient_geometry_mapping"
MAPPING_REVISION = "C5_GEOMETRY_R0_FROZEN_PROPAGATION_AXIS_KERNEL"
MIN_BANDWIDTH = 0.05


def stable_id(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()[:20]


def median_spacing(values: np.ndarray) -> float:
    unique = np.unique(np.asarray(values, float))
    if len(unique) < 2: return MIN_BANDWIDTH
    gaps = np.diff(np.sort(unique)); gaps = gaps[gaps > 1e-8]
    return float(np.median(gaps)) if len(gaps) else MIN_BANDWIDTH


def geometry_record(row: pd.Series) -> dict[str, object]:
    cache = PARENT / "cache" / str(row.fit_id)
    provenance = json.loads((cache / "provenance.json").read_text())
    with np.load(cache / "plane.npz", allow_pickle=False) as source:
        xy = np.asarray(source["contacts_xy_mm"], float)
        scale = float(np.asarray(source["scale_mm"]).ravel()[0])
    if xy.shape != (int(row.n_contacts), 2) or not np.isfinite(xy).all() or scale <= 0:
        raise RuntimeError(f"invalid frozen plane: {row.fit_id}")
    names = [str(value) for value in provenance["joint_contacts"]]
    shafts = [str(parse_shaft(name)[0]) for name in names]
    return {
        "patient": str(row.patient), "fit_id": str(row.fit_id),
        "geometry_view": str(row.geometry_view), "contact_names": names,
        "shaft_labels": shafts, "normalized_axis": (xy[:, 0] / scale),
        "normalized_transverse": (xy[:, 1] / scale), "scale_mm": scale,
        "plane_sha256": sha256_file(cache / "plane.npz"),
        "provenance_sha256": sha256_file(cache / "provenance.json"),
    }


def main() -> None:
    manifest = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv").drop_duplicates("fit_id")
    eligibility = pd.read_csv(OUT / "MODE_AXIS_ELIGIBILITY.csv")[["fit_id", "canonical_ab"]]
    manifest = manifest.merge(eligibility, on="fit_id", validate="one_to_one")
    records = {str(row.fit_id): geometry_record(row) for _, row in manifest.iterrows()}
    MAPPING.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for target_fit, target in records.items():
        tx = np.asarray(target["normalized_axis"], float)
        for source_fit, source in records.items():
            if source["patient"] == target["patient"]: continue
            sx = np.asarray(source["normalized_axis"], float)
            bandwidth = max(median_spacing(tx), median_spacing(sx), MIN_BANDWIDTH)
            squared = (tx[:, None] - sx[None, :]) ** 2
            weights = np.exp(-squared / (2.0 * bandwidth ** 2))
            weights /= weights.sum(axis=1, keepdims=True)
            mapping_id = stable_id(target_fit, source_fit, MAPPING_REVISION)
            path = MAPPING / "pairs" / f"{mapping_id}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream, weights=weights.astype(np.float32),
                    target_axis=tx.astype(np.float32), source_axis=sx.astype(np.float32),
                    target_fit_id=np.asarray([target_fit]), source_fit_id=np.asarray([source_fit]),
                )
            temporary.replace(path)
            effective = 1.0 / np.sum(weights ** 2, axis=1)
            rows.append({
                "mapping_id": mapping_id, "target_patient": target["patient"],
                "target_fit_id": target_fit, "target_geometry_view": target["geometry_view"],
                "source_patient": source["patient"], "source_fit_id": source_fit,
                "source_geometry_view": source["geometry_view"],
                "n_target_contacts": len(tx), "n_source_contacts": len(sx),
                "bandwidth_normalized_axis": bandwidth,
                "minimum_effective_source_contacts": float(effective.min()),
                "median_effective_source_contacts": float(np.median(effective)),
                "mapping_path": str(path.relative_to(ROOT)), "mapping_sha256": sha256_file(path),
                "target_plane_sha256": target["plane_sha256"],
                "source_plane_sha256": source["plane_sha256"],
                "field_values_read": False, "target_values_read": False,
            })
    frame = pd.DataFrame(rows)
    atomic_write_csv(MAPPING / "CROSS_PATIENT_GEOMETRY_MAPPING_MANIFEST.csv", frame)
    failure = []
    for item in frame.itertuples(index=False):
        path = ROOT / item.mapping_path
        with np.load(path, allow_pickle=False) as source:
            weights = np.asarray(source["weights"], float)
        reasons = []
        if weights.shape != (item.n_target_contacts, item.n_source_contacts): reasons.append("shape")
        if not np.isfinite(weights).all() or np.any(weights < 0): reasons.append("weights")
        if not np.allclose(weights.sum(axis=1), 1.0, atol=2e-6): reasons.append("row_sum")
        if sha256_file(path) != item.mapping_sha256: reasons.append("hash")
        if reasons: failure.append({"mapping_id": item.mapping_id, "reasons": reasons})
    payload = {
        "contract": "topic5_cross_patient_geometry_mapping_freeze_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mapping_revision": MAPPING_REVISION,
        "status": "PASS" if not failure and len(records) == 42 else "FAIL",
        "n_fits": len(records), "n_patients": int(manifest.patient.nunique()),
        "n_pair_mappings": int(len(frame)), "failure_count": len(failure),
        "failures_first20": failure[:20],
        "registration_contract": {
            "coordinate": "FROZEN_PLANE_PROPAGATION_AXIS_DIVIDED_BY_FROZEN_SCALE_MM",
            "orientation": "TRAIN_FROZEN_SOURCE_TO_SINK_SIGN",
            "transport": "ONE_DIMENSIONAL_GAUSSIAN_NADARAYA_WATSON",
            "bandwidth": "MAX_SOURCE_TARGET_MEDIAN_AXIS_SPACING_AND_0P05",
            "transverse_coordinate": "NOT_USED_BECAUSE_SVD_SIGN_HAS_NO_CROSS_PATIENT_ANCHOR",
            "anatomical_claim_boundary": "NORMALIZED_PROPAGATION_AXIS_IDENTITY_NULL_NOT_WHOLE_BRAIN_REGISTRATION",
        },
        "manifest_sha256": sha256_file(MAPPING / "CROSS_PATIENT_GEOMETRY_MAPPING_MANIFEST.csv"),
        "geometry_contract_sha256": canonical_json_sha256({
            fit: {key: value for key, value in record.items() if key not in {"normalized_axis", "normalized_transverse"}}
            for fit, record in records.items()
        }),
        "field_values_read": False, "target_values_read": False,
    }
    atomic_write_json(MAPPING / "GEOMETRY_REGISTRATION_AUDIT.json", payload)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
