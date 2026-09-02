#!/usr/bin/env python3
"""Audit and aggregate Topic 5.2 Gaussian tissue-patch response fields."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import atomic_write_csv, atomic_write_json, sha256_file  # noqa: E402
from scripts.freeze_topic5_cross_patient_geometry_mappings_v0_2 import SPATIAL  # noqa: E402
from scripts.freeze_topic5_latent_reference_states_v0_2 import reference_dir  # noqa: E402
from scripts.freeze_topic5_spatial_patch_contract_v0_2 import PATCH, patch_dir  # noqa: E402
from scripts.run_topic5_latent_pass1_v0_2 import OUT  # noqa: E402
from scripts.run_topic5_spatial_patch_response_v0_2 import (  # noqa: E402
    AXIS_NAMES, PATCH_RESPONSE_REVISION, RESPONSE, response_dir,
)


REAL_ARMS = ("L0", "L1", "L2m", "L3")
PRIMARY_DOSE_INDEX = 1


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream: np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, float), np.asarray(right, float); use = np.isfinite(a) & np.isfinite(b)
    if int(use.sum()) < 4: return float("nan")
    a, b = a[use] - a[use].mean(), b[use] - b[use].mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else float("nan")


def audit(manifest: pd.DataFrame) -> dict[str, object]:
    failures = []; eligible = finite = 0
    for item in manifest.itertuples(index=False):
        row = pd.Series(item._asdict()); target = response_dir(row); reasons = []
        try:
            metrics = json.loads((target / "metrics.json").read_text()); done = json.loads((target / "DONE.json").read_text())
            if metrics.get("status") != "PASS" or metrics.get("patch_response_revision") != PATCH_RESPONSE_REVISION: reasons.append("status_revision")
            if not metrics.get("model_hash_unchanged") or not metrics.get("decoder_hash_unchanged"): reasons.append("parameter_hash")
            if metrics.get("target_values_read") is not False: reasons.append("target_leak")
            if done.get("response_sha256") != sha256_file(target / "patch_response.npz"): reasons.append("response_hash")
            if metrics.get("reference_contract_sha256") != sha256_file(reference_dir(row) / "reference_contract.npz"): reasons.append("reference_hash")
            if metrics.get("patch_contract_sha256") != sha256_file(patch_dir(row) / "patch_contract.npz"): reasons.append("patch_hash")
            with np.load(target / "patch_response.npz", allow_pickle=False) as source:
                q = {name: np.asarray(source[name]) for name in source.files}
            expected = (3, int(item.n_nodes), 3, 4, 2)
            if q["mean_scores"].shape != expected: reasons.append("score_shape")
            if q["valid_counts"].shape != expected[:-1]: reasons.append("count_shape")
            count = q["valid_counts"] > 0
            if not np.isfinite(q["mean_scores"][np.broadcast_to(count[..., None], expected)]).all(): reasons.append("nonfinite_valid")
            if np.isfinite(q["mean_scores"][~np.broadcast_to(count[..., None], expected)]).any(): reasons.append("finite_invalid")
            eligible += int(metrics["eligible_state_center_dose_pairs"]); finite += int(metrics["finite_state_center_dose_tau"])
        except Exception as error: reasons.append(f"{type(error).__name__}:{error}")
        if reasons: failures.append({"cell_key": f"{item.fit_id}/{item.public_arm}/seed{item.seed}", "reasons": sorted(set(reasons))})
    payload = {
        "contract": "topic5_spatial_patch_response_audit_v0_2", "created_utc": datetime.now(timezone.utc).isoformat(),
        "patch_response_revision": PATCH_RESPONSE_REVISION,
        "status": "PASS" if len(manifest) == 630 and not failures else "FAIL", "audited_cells": int(len(manifest)),
        "eligible_state_center_dose_pairs": eligible, "finite_state_center_dose_tau": finite,
        "failure_count": len(failures), "failures_first20": failures[:20],
        "n0_enforcement": "ONLY_BOTH_SIGN_NODE_KNN_MANIFOLD_SUPPORTED_BRANCHES_EXECUTED",
        "target_values_read": False,
    }
    atomic_write_json(RESPONSE / "PATCH_RESPONSE_AUDIT.json", payload); return payload


def main() -> None:
    manifest = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv")
    freeze_audit = json.loads((PATCH / "PATCH_FREEZE_AUDIT.json").read_text())
    if freeze_audit.get("status") != "PASS": raise RuntimeError("patch freeze audit must pass")
    response_audit = audit(manifest)
    if response_audit["status"] != "PASS":
        print(json.dumps(response_audit, indent=2)); raise SystemExit(1)
    rows = []; map_cache = {}
    for item in manifest.itertuples(index=False):
        row = pd.Series(item._asdict())
        with np.load(response_dir(row) / "patch_response.npz", allow_pickle=False) as source:
            scores = np.asarray(source["mean_scores"], float)
            sign = np.asarray(source["positive_sign_fraction"], float)
            nodes = np.asarray(source["node_xy_mm"], float)
        for dose_index in range(3):
            future = np.nanmean(scores[:, :, dose_index, 1:4], axis=(0, 2))
            immediate = np.nanmean(scores[:, :, dose_index, 0], axis=0)
            sign_future = np.nanmean(sign[:, :, dose_index, 1:4], axis=(0, 2))
            for center in range(len(nodes)):
                for axis_index, axis in enumerate(AXIS_NAMES):
                    rows.append({
                        "patient": item.patient, "fit_id": item.fit_id, "geometry_view": item.geometry_view,
                        "public_arm": item.public_arm, "seed": int(item.seed), "dose_index": dose_index,
                        "dose": [0.25, 0.5, 1.0][dose_index], "node_index": center,
                        "node_x_mm": nodes[center, 0], "node_y_mm": nodes[center, 1], "axis": axis,
                        "future_response": future[center, axis_index],
                        "immediate_response": immediate[center, axis_index],
                        "sign_agreement_fraction": max(sign_future[center, axis_index], 1.0 - sign_future[center, axis_index])
                        if np.isfinite(sign_future[center, axis_index]) else np.nan,
                        "target_values_read": False,
                    })
    cell = pd.DataFrame(rows)
    primary = cell[cell.dose_index.eq(PRIMARY_DOSE_INDEX)].copy()
    seed = primary.groupby(
        ["patient", "fit_id", "geometry_view", "public_arm", "axis", "node_index", "node_x_mm", "node_y_mm"], as_index=False
    )[["future_response", "immediate_response", "sign_agreement_fraction"]].median()
    fit_rows = []; consistency_rows = []
    for (patient, fit_id, axis), group in seed.groupby(["patient", "fit_id", "axis"]):
        vectors = {arm: part.sort_values("node_index").future_response.to_numpy(float) for arm, part in group.groupby("public_arm")}
        if not set((*REAL_ARMS, "C-suffix")).issubset(vectors): continue
        real_pairs = [cosine(vectors[REAL_ARMS[i]], vectors[REAL_ARMS[j]]) for i in range(4) for j in range(i + 1, 4)]
        control_pairs = [cosine(vectors[arm], vectors["C-suffix"]) for arm in REAL_ARMS]
        real = group[group.public_arm.isin(REAL_ARMS)].groupby(
            ["node_index", "node_x_mm", "node_y_mm"], as_index=False
        )[["future_response", "immediate_response", "sign_agreement_fraction"]].median()
        for item in real.itertuples(index=False):
            fit_rows.append({
                "patient": patient, "fit_id": fit_id, "axis": axis, "node_index": item.node_index,
                "node_x_mm": item.node_x_mm, "node_y_mm": item.node_y_mm,
                "future_response": item.future_response, "immediate_response": item.immediate_response,
                "sign_agreement_fraction": item.sign_agreement_fraction,
            })
        consistency_rows.append({
            "patient": patient, "fit_id": fit_id, "axis": axis,
            "real_arm_pair_cosine": float(np.nanmedian(real_pairs)),
            "real_arm_to_C_suffix_cosine": float(np.nanmedian(control_pairs)),
            "topology_margin": float(np.nanmedian(real_pairs) - np.nanmedian(control_pairs)),
            "median_sign_agreement_fraction": float(np.nanmedian(real.sign_agreement_fraction)),
            "future_map_norm": float(np.linalg.norm(real.future_response)),
            "immediate_future_cosine": cosine(real.immediate_response.to_numpy(float), real.future_response.to_numpy(float)),
        })
    fit_fields = pd.DataFrame(fit_rows); consistency = pd.DataFrame(consistency_rows)

    dose_rows = []
    dose_seed = cell.groupby(
        ["patient", "fit_id", "public_arm", "axis", "dose_index", "node_index"], as_index=False
    ).future_response.median()
    for (patient, fit_id, arm, axis), group in dose_seed.groupby(["patient", "fit_id", "public_arm", "axis"]):
        vectors = {dose: part.sort_values("node_index").future_response.to_numpy(float) for dose, part in group.groupby("dose_index")}
        if len(vectors) == 3:
            dose_rows.append({"patient": patient, "fit_id": fit_id, "public_arm": arm, "axis": axis,
                              "dose_025_05_cosine": cosine(vectors[0], vectors[1]),
                              "dose_05_10_cosine": cosine(vectors[1], vectors[2])})
    dose_consistency = pd.DataFrame(dose_rows)

    write_npz(RESPONSE / "SPATIAL_PATCH_CONTROL_FIELDS.npz", {
        column: fit_fields[column].to_numpy() for column in fit_fields.columns
    })
    atomic_write_csv(RESPONSE / "PATCH_CELL_NODE_FIELDS.csv", cell)
    atomic_write_csv(RESPONSE / "PATCH_FIT_CONTROL_FIELDS.csv", fit_fields)
    atomic_write_csv(RESPONSE / "PATCH_TOPOLOGY_CONSISTENCY.csv", consistency)
    atomic_write_csv(RESPONSE / "PATCH_DOSE_CONSISTENCY.csv", dose_consistency)
    patient = consistency.groupby(["patient", "axis"], as_index=False)[
        ["real_arm_pair_cosine", "real_arm_to_C_suffix_cosine", "topology_margin", "median_sign_agreement_fraction", "future_map_norm", "immediate_future_cosine"]
    ].median()
    summary = {
        "contract": "topic5_spatial_patch_control_field_summary_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "SPATIAL_CONTROL_FIELD_COMPLETE",
        "n_cells": 630, "n_fits": int(fit_fields.fit_id.nunique()), "n_patients": int(fit_fields.patient.nunique()),
        "primary": "DOSE_0P5_MEAN_PHASES_FUTURE_TAU1_TO3_MEDIAN_SEED_REAL_ARM",
        "axis_summaries": {
            axis: {
                "median_real_arm_pair_cosine": float(np.nanmedian(patient.loc[patient.axis.eq(axis), "real_arm_pair_cosine"])),
                "median_topology_margin_vs_C_suffix": float(np.nanmedian(patient.loc[patient.axis.eq(axis), "topology_margin"])),
                "median_sign_agreement_fraction": float(np.nanmedian(patient.loc[patient.axis.eq(axis), "median_sign_agreement_fraction"])),
                "median_future_map_norm": float(np.nanmedian(patient.loc[patient.axis.eq(axis), "future_map_norm"])),
                "median_immediate_future_cosine": float(np.nanmedian(patient.loc[patient.axis.eq(axis), "immediate_future_cosine"])),
            } for axis in AXIS_NAMES
        },
        "dose_consistency": {
            "median_025_05_cosine": float(np.nanmedian(dose_consistency.dose_025_05_cosine)),
            "median_05_10_cosine": float(np.nanmedian(dose_consistency.dose_05_10_cosine)),
        },
        "response_audit": response_audit,
        "claim_boundary": "MODEL_INTERNAL_TISSUE_NODE_SUSCEPTIBILITY_NOT_STIMULATION_MAP",
        "target_values_read": False,
    }
    atomic_write_json(RESPONSE / "SPATIAL_PATCH_CONTROL_SUMMARY.json", summary)
    atomic_write_json(RESPONSE / "SPATIAL_PATCH_FREEZE_SEAL.json", {
        "sealed": True, "status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
        "fields_sha256": sha256_file(RESPONSE / "SPATIAL_PATCH_CONTROL_FIELDS.npz"),
        "summary_sha256": sha256_file(RESPONSE / "SPATIAL_PATCH_CONTROL_SUMMARY.json"),
        "target_values_read": False,
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
