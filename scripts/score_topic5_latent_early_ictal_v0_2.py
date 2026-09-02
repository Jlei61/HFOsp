#!/usr/bin/env python3
"""Locked exploratory C7 scoring of frozen RNN fields against early-ictal energy."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import atomic_write_csv, atomic_write_json, sha256_file  # noqa: E402
from scripts.freeze_topic5_cross_patient_geometry_mappings_v0_2 import MAPPING, SPATIAL  # noqa: E402
from scripts.run_topic5_latent_pass1_v0_2 import OUT, PARENT  # noqa: E402
from scripts.score_topic5_multiscale_early_ictal_v0_5 import (  # noqa: E402
    signed_spearman_permutations, signed_spearman_target_matrix,
    spectral_surrogates, variogram_surrogates,
)
from scripts.summarize_topic5_latent_geometry_v0_2 import one_sided_summary  # noqa: E402


EARLY = OUT / "early_ictal_exploratory"
SOURCE_EARLY = PARENT / "early_ictal"
AUTHORIZATION = OUT / "EARLY_ICTAL_UNLOCK_AUTHORIZATION.json"
SCORER_REVISION = "C7_R0_FIXED_ALL_PHASE_TAU_RESPONSE_NO_ORACLE"
AXES = ("PROGRESS", "FIELD")


def signed_spearman(left: np.ndarray, right: np.ndarray) -> float:
    use = np.isfinite(left) & np.isfinite(right)
    if int(use.sum()) < 4 or np.ptp(left[use]) <= 1e-12 or np.ptp(right[use]) <= 1e-12: return float("nan")
    return float(spearmanr(left[use], right[use]).statistic)


def load_predictions() -> dict[tuple[str, str], tuple[list[str], np.ndarray]]:
    fields = pd.read_csv(SPATIAL / "data_alignment/PRIMARY_FIT_RESPONSE_FIELDS.csv")
    result = {}
    for (patient, axis), group in fields.groupby(["patient", "perturbation_axis"]):
        names_by_fit = [part.sort_values("contact_index").contact_name.astype(str).tolist() for _, part in group.groupby("fit_id")]
        if any(names != names_by_fit[0] for names in names_by_fit[1:]):
            raise RuntimeError(f"fit contact order mismatch: {patient}/{axis}")
        matrix = np.stack([
            part.sort_values("contact_index").response.to_numpy(float) for _, part in group.groupby("fit_id")
        ])
        result[(patient, axis)] = (names_by_fit[0], np.nanmedian(matrix, axis=0))
    return result


def mapped(weights: np.ndarray, source: np.ndarray) -> np.ndarray:
    valid = np.isfinite(source)
    usable = weights[:, valid]; usable /= np.maximum(usable.sum(axis=1, keepdims=True), 1e-12)
    return usable @ source[valid]


def main() -> None:
    authorization = json.loads(AUTHORIZATION.read_text())
    if not authorization.get("authorized") or authorization.get("scorer_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("C7 scorer is not frozen/authorized")
    for relative, digest in authorization["target_free_hashes"].items():
        if sha256_file(ROOT / relative) != digest: raise RuntimeError(f"target-free artifact drift: {relative}")
    target_manifest_path = SOURCE_EARLY / "EARLY_ICTAL_TARGET_MANIFEST.csv"
    null_manifest_path = PARENT / "NULL_INDEX_MAP_MANIFEST.csv"
    routing_path = PARENT / "EARLY_ICTAL_ROUTING_METADATA.csv"
    if sha256_file(target_manifest_path) != authorization["target_manifest_sha256"]: raise RuntimeError("target manifest drift")
    if sha256_file(null_manifest_path) != authorization["null_manifest_sha256"]: raise RuntimeError("null manifest drift")
    if sha256_file(routing_path) != authorization["routing_sha256"]: raise RuntimeError("routing drift")
    target_manifest = pd.read_csv(target_manifest_path); null_manifest = pd.read_csv(null_manifest_path)
    routing = pd.read_csv(routing_path)
    if target_manifest.subject.nunique() != 17 or len(routing) != 167: raise RuntimeError("locked C7 denominator drift")
    for item in target_manifest.itertuples(index=False):
        if sha256_file(Path(item.path)) != item.sha256: raise RuntimeError(f"target payload hash drift: {item.subject}")
    for item in null_manifest.itertuples(index=False):
        if sha256_file(Path(item.path)) != item.sha256: raise RuntimeError(f"null payload hash drift: {item.subject}/{item.seizure_idx}")
    EARLY.mkdir(parents=True, exist_ok=True)
    atomic_write_json(EARLY / "TARGET_UNLOCK_RECORD.json", {
        "contract": "topic5_latent_C7_target_unlock_v0_2", "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_sha256": sha256_file(AUTHORIZATION), "project_history_target_previously_viewed": True,
        "training_or_model_selection_after_unlock": False,
        "target": "clinical onset 0-10 s, 1-150 Hz broadband energy", "target_values_read": True,
    })

    predictions = load_predictions()
    seizure_rows = []; patient_nulls: dict[tuple[str, str, str], list[np.ndarray]] = {}
    patient_targets = {}; contact_lookup = {}
    null_index = null_manifest.set_index(["subject", "seizure_idx"])
    for target_row in target_manifest.itertuples(index=False):
        with np.load(target_row.path, allow_pickle=False) as source:
            contacts = source["contacts"].astype(str).tolist()
            all_targets = np.asarray(source["all_seizure_broadband_energy"], float)
            median_target = np.asarray(source["median_broadband_energy"], float)
        patient_targets[target_row.subject] = median_target; contact_lookup[target_row.subject] = contacts
        subject_routing = routing[routing.subject.eq(target_row.subject)].reset_index(drop=True)
        if len(subject_routing) != len(all_targets): raise RuntimeError(f"seizure target order mismatch: {target_row.subject}")
        for seizure_position, event in subject_routing.iterrows():
            target = all_targets[seizure_position]
            null_row = null_index.loc[(target_row.subject, int(event.seizure_idx))]
            with np.load(null_row.path, allow_pickle=False) as source:
                if source["contacts"].astype(str).tolist() != contacts: raise RuntimeError("null contact order drift")
                permutations_all = np.asarray(source["all_contact"], int)
                permutations_shaft = np.asarray(source["within_shaft"], int)
                permutations_distance = np.asarray(source["distance_bin"], int)
                spectral_basis = np.asarray(source["spectral_eigenvectors"], float)
                spectral_signs = np.asarray(source["spectral_signs"], float)
                variogram_normals = np.asarray(source["variogram_normals"], float)
                xy = np.asarray(source["contact_xy_mm"], float)
            spectral_target = spectral_surrogates(target, spectral_basis, spectral_signs) if len(spectral_signs) else np.empty((0, len(target)))
            variogram_target = variogram_surrogates(target, xy, variogram_normals)[0] if len(variogram_normals) else np.empty((0, len(target)))
            for axis in AXES:
                prediction_contacts, prediction = predictions[(target_row.subject, axis)]
                if prediction_contacts != contacts: raise RuntimeError(f"prediction/target contacts drift: {target_row.subject}/{axis}")
                observed, all_null = signed_spearman_permutations(prediction, target, permutations_all)
                record = {
                    "subject": target_row.subject, "seizure_idx": int(event.seizure_idx), "axis": axis,
                    "n_contacts": len(target), "observed": observed,
                    "all_contact_null_median": float(np.nanmedian(all_null)),
                    "all_contact_margin": float(observed - np.nanmedian(all_null)),
                    "within_shaft_margin": np.nan, "distance_bin_margin": np.nan,
                    "spectral_margin": np.nan, "variogram_margin": np.nan,
                }
                patient_nulls.setdefault((target_row.subject, axis, "ALL"), []).append(all_null)
                if len(permutations_shaft):
                    _, values = signed_spearman_permutations(prediction, target, permutations_shaft)
                    record["within_shaft_margin"] = observed - float(np.nanmedian(values))
                if len(permutations_distance):
                    _, values = signed_spearman_permutations(prediction, target, permutations_distance)
                    record["distance_bin_margin"] = observed - float(np.nanmedian(values))
                if len(spectral_target):
                    values = signed_spearman_target_matrix(prediction, spectral_target)
                    record["spectral_margin"] = observed - float(np.nanmedian(values))
                if len(variogram_target):
                    values = signed_spearman_target_matrix(prediction, variogram_target)
                    record["variogram_margin"] = observed - float(np.nanmedian(values))
                seizure_rows.append(record)
    seizure = pd.DataFrame(seizure_rows)
    patient_rows = []
    for (subject, axis), group in seizure.groupby(["subject", "axis"]):
        observed = float(np.nanmedian(group.observed)); matrices = patient_nulls[(subject, axis, "ALL")]
        null = np.nanmedian(np.stack(matrices), axis=0)
        patient_rows.append({
            "subject": subject, "axis": axis, "n_seizures": len(group), "observed": observed,
            "all_contact_null_median": float(np.nanmedian(null)),
            "all_contact_margin": float(observed - np.nanmedian(null)),
            "within_shaft_margin": float(np.nanmedian(group.within_shaft_margin)),
            "distance_bin_margin": float(np.nanmedian(group.distance_bin_margin)),
            "spectral_margin": float(np.nanmedian(group.spectral_margin)),
            "variogram_margin": float(np.nanmedian(group.variogram_margin)),
        })
    patient = pd.DataFrame(patient_rows)

    # Frozen propagation-axis geometry identity sensitivity on patient-median targets.
    mapping = pd.read_csv(MAPPING / "CROSS_PATIENT_GEOMETRY_MAPPING_MANIFEST.csv")
    fit_fields = pd.read_csv(SPATIAL / "data_alignment/PRIMARY_FIT_RESPONSE_FIELDS.csv")
    identity_rows = []
    early_subjects = set(patient_targets)
    for (fit_id, axis), group in fit_fields.groupby(["fit_id", "perturbation_axis"]):
        subject = str(group.patient.iloc[0])
        if subject not in early_subjects: continue
        response = group.sort_values("contact_index").response.to_numpy(float)
        same = signed_spearman(response, patient_targets[subject])
        candidates = mapping[mapping.target_fit_id.eq(fit_id) & mapping.source_patient.isin(early_subjects)]
        cross = []
        for item in candidates.itertuples(index=False):
            source = patient_targets[item.source_patient]
            with np.load(ROOT / item.mapping_path, allow_pickle=False) as payload: weights = np.asarray(payload["weights"], float)
            score = signed_spearman(response, mapped(weights, source)); cross.append({"source_patient": item.source_patient, "score": score})
        cross = pd.DataFrame(cross).groupby("source_patient", as_index=False).score.median()
        identity_rows.append({"subject": subject, "fit_id": fit_id, "axis": axis, "same_patient": same,
                              "cross_patient_median": float(np.nanmedian(cross.score)),
                              "identity_margin": float(same - np.nanmedian(cross.score)),
                              "n_cross_patients": int(cross.score.notna().sum())})
    identity = pd.DataFrame(identity_rows)
    identity_patient = identity.groupby(["subject", "axis"], as_index=False).identity_margin.median()
    patient = patient.merge(identity_patient, on=["subject", "axis"], how="left", validate="one_to_one")
    summaries = {}
    for axis in AXES:
        part = patient[patient.axis.eq(axis)]
        summaries[axis] = {
            "n_patients": int(part.subject.nunique()),
            "all_contact_margin": one_sided_summary(part.all_contact_margin.to_numpy(float), 520700 + len(summaries)),
            "identity_margin": one_sided_summary(part.identity_margin.to_numpy(float), 520710 + len(summaries)),
            "spatial_sensitivities": {
                name: {"n": int(np.isfinite(part[name]).sum()), "median": float(np.nanmedian(part[name]))}
                for name in ["within_shaft_margin", "distance_bin_margin", "spectral_margin", "variogram_margin"]
            },
        }
        if axis == "PROGRESS":
            summaries[axis]["laterness_orientation_sensitivity"] = {
                "role": "TARGET_FREE_FROZEN_SIGN_SEMANTICS_SENSITIVITY_PRIMARY_UNCHANGED",
                "all_contact_margin": one_sided_summary(
                    -part.all_contact_margin.to_numpy(float), 520720
                ),
                "identity_margin": one_sided_summary(
                    -part.identity_margin.to_numpy(float), 520721
                ),
            }
    atomic_write_csv(EARLY / "EARLY_ICTAL_PER_SEIZURE.csv", seizure)
    atomic_write_csv(EARLY / "EARLY_ICTAL_PER_PATIENT.csv", patient)
    atomic_write_csv(EARLY / "EARLY_ICTAL_IDENTITY.csv", identity)
    summary = {
        "contract": "topic5_latent_early_ictal_C7_v0_2", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scorer_revision": SCORER_REVISION, "status": "CROSS_STATE_EXPLORATORY_COMPLETE",
        "target": "clinical onset 0-10 s, 1-150 Hz broadband energy", "patients": 17, "seizures": 167,
        "prediction": "FIXED_MEAN_PHASES_TAU1_TO3_MEDIAN_SEED_REAL_ARM_NO_ORACLE",
        "axes": summaries,
        "claim_boundary": "LOCKED_INTERNAL_EXPLORATORY; TARGET PREVIOUSLY VIEWED; NOT CONFIRMATORY",
        "training_or_model_selection_after_unlock": False, "target_values_read": True,
    }
    atomic_write_json(EARLY / "EARLY_ICTAL_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
