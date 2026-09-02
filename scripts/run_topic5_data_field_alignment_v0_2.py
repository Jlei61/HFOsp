#!/usr/bin/env python3
"""Freeze finite-time fields and adjudicate heldout interictal C5 alignment."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.propagation_skeleton_geometry import parse_shaft  # noqa: E402
from src.topic5_latent_landscape_v0_2 import (  # noqa: E402
    atomic_write_csv, atomic_write_json, rank_matrix_to_event_fields, sha256_file,
)
from src.topic5_latent_perturbation_v0_2 import centered_unit_field, stable_seed  # noqa: E402
from scripts.freeze_topic5_cross_patient_geometry_mappings_v0_2 import MAPPING, SPATIAL  # noqa: E402
from scripts.run_topic5_latent_pass1_v0_2 import OUT, PARENT  # noqa: E402
from scripts.run_topic5_axis_perturbations_v0_2 import PERTURB  # noqa: E402
from scripts.summarize_topic5_latent_geometry_v0_2 import holm_adjust, one_sided_summary  # noqa: E402


DATA = SPATIAL / "data_alignment"
ALIGNMENT_REVISION = "C5_DATA_R0_HELDOUT_PREFIX_ASSIGNED_START_REMOVED"
N_NULL = 4096
REAL_ARMS = ("L0", "L1", "L2m", "L3")
AXES = ("PROGRESS", "FIELD")
VARIANTS = ("START_REMOVED", "FULL")
PRIMARY_VARIANT = "START_REMOVED"
RELIABILITY_MIN_MODE_EVENTS = 20
RELIABILITY_MIN_SPLIT_SPEARMAN = 0.20


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream: np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def centered_valid(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    raw = np.asarray(value, float)
    valid = np.isfinite(raw)
    if int(valid.sum()) < 4: return np.full_like(raw, np.nan), valid, False
    result = np.full_like(raw, np.nan)
    centered = raw[valid] - float(np.mean(raw[valid]))
    norm = float(np.linalg.norm(centered))
    if not np.isfinite(norm) or norm <= 1e-10: return result, valid, False
    result[valid] = centered / norm
    return result, valid, True


def signed_scores(response: np.ndarray, data: np.ndarray) -> tuple[float, float]:
    use = np.isfinite(response) & np.isfinite(data)
    if int(use.sum()) < 4: return float("nan"), float("nan")
    a, b = response[use], data[use]
    spearman = float(spearmanr(a, b).statistic) if np.ptp(a) > 1e-12 and np.ptp(b) > 1e-12 else float("nan")
    a = a - a.mean(); b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.dot(a, b) / denominator) if denominator > 1e-12 else float("nan")
    return spearman, cosine


def correlation_many(response: np.ndarray, surrogates: np.ndarray) -> np.ndarray:
    """Spearman correlations, with a fixed response and many finite surrogate rows."""
    response_rank = rankdata(np.asarray(response, float)); response_rank -= response_rank.mean()
    response_rank /= max(float(np.linalg.norm(response_rank)), 1e-12)
    ranks = np.apply_along_axis(rankdata, 1, np.asarray(surrogates, float))
    ranks -= ranks.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(ranks, axis=1, keepdims=True)
    return (ranks / np.maximum(norm, 1e-12)) @ response_rank


def permutation_indices(groups: list[np.ndarray], n_contacts: int, seed: int) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed); base = np.arange(n_contacts)
    output = np.tile(base, (N_NULL, 1)); movable = int(sum(len(group) for group in groups if len(group) > 1))
    for draw in range(N_NULL):
        for group in groups:
            if len(group) > 1: output[draw, group] = rng.permutation(group)
    return output, movable


def null_contract(names: list[str], xy: np.ndarray, fit_id: str) -> dict[str, object]:
    n = len(names); all_groups = [np.arange(n)]
    shaft_map: dict[str, list[int]] = {}
    for index, name in enumerate(names): shaft_map.setdefault(str(parse_shaft(name)[0]), []).append(index)
    shaft_groups = [np.asarray(values, int) for _, values in sorted(shaft_map.items())]
    axis = np.asarray(xy, float)[:, 0]
    order = np.argsort(axis); distance_groups = [part for part in np.array_split(order, min(4, max(1, n // 2))) if len(part)]
    all_perm, all_movable = permutation_indices(all_groups, n, stable_seed(fit_id, "all_contact"))
    shaft_perm, shaft_movable = permutation_indices(shaft_groups, n, stable_seed(fit_id, "shaft"))
    distance_perm, distance_movable = permutation_indices(distance_groups, n, stable_seed(fit_id, "distance_bins"))
    distance = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    positive = distance[distance > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    weight = np.exp(-(distance ** 2) / max(2.0 * bandwidth ** 2, 1e-12)); np.fill_diagonal(weight, 0.0)
    degree = np.diag(weight.sum(axis=1)); laplacian = degree - weight
    _, eigenvectors = np.linalg.eigh(laplacian)
    return {
        "all_permutations": all_perm, "shaft_permutations": shaft_perm,
        "distance_permutations": distance_perm, "spectral_basis": eigenvectors,
        "all_movable": all_movable, "shaft_movable": shaft_movable,
        "distance_movable": distance_movable, "n_shafts": len(shaft_groups),
        "n_distance_bins": len(distance_groups),
    }


def spectral_surrogates(field: np.ndarray, basis: np.ndarray, seed: int) -> np.ndarray:
    value = np.asarray(field, float); centered = value - value.mean()
    coefficients = basis.T @ centered
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(N_NULL, len(value)))
    signs[:, 0] = 1.0
    return (signs * coefficients[None, :]) @ basis.T


def half_reliability(fields: np.ndarray, mode: np.ndarray, use: np.ndarray, positive: int, negative: int) -> dict[str, float]:
    estimates: list[dict[str, np.ndarray]] = []
    indices_by_mode = {label: np.flatnonzero(use & (mode == label)) for label in (positive, negative)}
    for half in (0, 1):
        means = {}
        for label, indices in indices_by_mode.items():
            selected = indices[half::2]
            with np.errstate(invalid="ignore"): means[label] = np.nanmean(fields[selected], axis=0)
        estimates.append({
            "PROGRESS": (means[positive] + means[negative]) / 2.0,
            "FIELD": means[positive] - means[negative],
        })
    output = {}
    for axis in AXES:
        left, _, left_ok = centered_valid(estimates[0][axis]); right, _, right_ok = centered_valid(estimates[1][axis])
        output[axis] = signed_scores(left, right)[0] if left_ok and right_ok else float("nan")
    return output


def build_data_fields(manifest: pd.DataFrame, eligibility: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, quality = [], []
    indexed = eligibility.set_index("fit_id")
    for item in manifest.drop_duplicates("fit_id").itertuples(index=False):
        cache = PARENT / "cache" / item.fit_id
        with np.load(cache / "events.npz", allow_pickle=False) as source:
            ranks = np.asarray(source["ranks"]); split = np.asarray(source["split"]); mode = np.asarray(source["mode"])
        full, removed = rank_matrix_to_event_fields(ranks)
        contract = indexed.loc[item.fit_id]
        positive, negative = int(contract.positive_mode), int(contract.negative_mode)
        test = split == 2; counts = {label: int(np.sum(test & (mode == label))) for label in (positive, negative)}
        contact_names = json.loads((cache / "provenance.json").read_text())["joint_contacts"]
        reliability = half_reliability(removed, mode, test, positive, negative)
        reliability_values = np.asarray(list(reliability.values()), float)
        low = (
            min(counts.values()) < RELIABILITY_MIN_MODE_EVENTS
            or not np.isfinite(reliability_values).all()
            or float(np.min(reliability_values)) < RELIABILITY_MIN_SPLIT_SPEARMAN
        )
        fit_ok = True
        for variant, event_fields in zip(VARIANTS, (removed, full)):
            means = {}
            for label in (positive, negative):
                with np.errstate(invalid="ignore"): means[label] = np.nanmean(event_fields[test & (mode == label)], axis=0)
            raw = {"PROGRESS": (means[positive] + means[negative]) / 2.0, "FIELD": means[positive] - means[negative]}
            for axis in AXES:
                vector, valid, ok = centered_valid(raw[axis]); fit_ok &= ok
                for contact, name, value, legal in zip(range(len(vector)), contact_names, vector, valid):
                    rows.append({
                        "patient": item.patient, "fit_id": item.fit_id, "geometry_view": item.geometry_view,
                        "canonical_ab": bool(contract.canonical_ab), "variant": variant, "axis": axis,
                        "contact_index": contact, "contact_name": name,
                        "field_value": float(value) if np.isfinite(value) else np.nan, "valid_contact": bool(legal),
                        "target_values_read": False,
                    })
        quality.append({
            "patient": item.patient, "fit_id": item.fit_id, "geometry_view": item.geometry_view,
            "canonical_ab": bool(contract.canonical_ab), "positive_mode": positive, "negative_mode": negative,
            "positive_label": contract.positive_label, "negative_label": contract.negative_label,
            "n_test_positive": counts[positive], "n_test_negative": counts[negative],
            "progress_split_half_spearman": reliability["PROGRESS"],
            "field_split_half_spearman": reliability["FIELD"],
            "status": "DATA_FIELD_NOT_IDENTIFIABLE" if not fit_ok else ("DATA_FIELD_LOW_RELIABILITY" if low else "DATA_FIELD_IDENTIFIABLE"),
            "grouping_contract": "HELDOUT_PREFIX_ONLY_ASSIGNMENT_TO_TRAIN_FROZEN_MODES",
            "target_values_read": False,
        })
    return pd.DataFrame(rows), pd.DataFrame(quality)


def response_tables(fields: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fields = fields[fields.public_arm.isin(REAL_ARMS)].copy()
    cell = fields.groupby(
        ["patient", "fit_id", "geometry_view", "public_arm", "seed", "canonical_ab", "perturbation_axis", "contact_index", "contact_name"],
        as_index=False,
    ).response.mean()
    seed = cell.groupby(
        ["patient", "fit_id", "geometry_view", "public_arm", "canonical_ab", "perturbation_axis", "contact_index", "contact_name"],
        as_index=False,
    ).response.median()
    fit = seed.groupby(
        ["patient", "fit_id", "geometry_view", "canonical_ab", "perturbation_axis", "contact_index", "contact_name"],
        as_index=False,
    ).response.median()
    return cell, fit


def mapped_field(mapping_row: pd.Series, source_vector: np.ndarray) -> np.ndarray:
    with np.load(ROOT / str(mapping_row.mapping_path), allow_pickle=False) as source:
        weights = np.asarray(source["weights"], float)
    valid = np.isfinite(source_vector)
    if int(valid.sum()) < 4:
        return np.full(weights.shape[0], np.nan)
    usable = weights[:, valid]
    usable /= np.maximum(usable.sum(axis=1, keepdims=True), 1e-12)
    result = usable @ np.asarray(source_vector, float)[valid]
    result, _, ok = centered_valid(result)
    return result if ok else np.full(weights.shape[0], np.nan)


def main() -> None:
    if json.loads((PERTURB / "PERTURBATION_AUDIT.json").read_text()).get("status") != "PASS":
        raise RuntimeError("perturbation audit must pass")
    registration = json.loads((MAPPING / "GEOMETRY_REGISTRATION_AUDIT.json").read_text())
    if registration.get("status") != "PASS" or registration.get("field_values_read") is not False:
        raise RuntimeError("field-blind geometry registration must be frozen first")
    manifest = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv")
    eligibility = pd.read_csv(OUT / "MODE_AXIS_ELIGIBILITY.csv")
    functional = pd.read_csv(PERTURB / "FUNCTIONAL_RESPONSE_FIELDS.csv")
    cell_response, fit_response = response_tables(functional)
    data_fields, quality = build_data_fields(manifest, eligibility)
    DATA.mkdir(parents=True, exist_ok=True)

    # Required finite-time artifact: source rows retain phase/tau; primary fit rows lock aggregation.
    write_npz(DATA / "FINITE_TIME_RESPONSE_FIELDS.npz", {
        column: functional[column].to_numpy() for column in functional.columns
    })
    atomic_write_json(DATA / "PERTURBATION_RESPONSE_MATRIX.json", {
        "contract": "topic5_perturbation_response_matrix_freeze_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256_file(PERTURB / "PERTURBATION_CLAIM_SUMMARY.json"),
        "matrix_source": "C3_CELL_PHASE_RESPONSE.csv",
        "aggregation_for_C5": "MEAN_PHASE_TAU_THEN_MEDIAN_SEED_AND_REAL_ARM",
        "axis_sign": "TRAIN_FROZEN_NO_HELDOUT_SIGN_FLIP", "target_values_read": False,
    })
    atomic_write_csv(DATA / "HELDOUT_DATA_FIELDS.csv", data_fields)
    atomic_write_csv(DATA / "DATA_FIELD_QUALITY.csv", quality)
    atomic_write_csv(DATA / "PRIMARY_FIT_RESPONSE_FIELDS.csv", fit_response)

    data_lookup = {}
    for (fit_id, variant, axis), group in data_fields.groupby(["fit_id", "variant", "axis"]):
        data_lookup[(fit_id, variant, axis)] = group.sort_values("contact_index").field_value.to_numpy(float)
    response_lookup = {}
    for (fit_id, axis), group in fit_response.groupby(["fit_id", "perturbation_axis"]):
        response_lookup[(fit_id, axis)] = group.sort_values("contact_index").response.to_numpy(float)
    mapping = pd.read_csv(MAPPING / "CROSS_PATIENT_GEOMETRY_MAPPING_MANIFEST.csv")
    eligibility_index = eligibility.set_index("fit_id")
    quality_index = quality.set_index("fit_id")
    fit_manifest = manifest.drop_duplicates("fit_id").set_index("fit_id")
    within_rows, null_rows, identity_rows = [], [], []
    for fit_id in sorted(fit_manifest.index):
        item = fit_manifest.loc[fit_id]; cache = PARENT / "cache" / fit_id
        if quality_index.loc[fit_id].status == "DATA_FIELD_NOT_IDENTIFIABLE":
            continue
        provenance = json.loads((cache / "provenance.json").read_text())
        names = [str(value) for value in provenance["joint_contacts"]]
        with np.load(cache / "plane.npz", allow_pickle=False) as source: xy = np.asarray(source["contacts_xy_mm"], float)
        nulls = null_contract(names, xy, fit_id)
        for variant in VARIANTS:
            spectral_cache = {}
            for axis in AXES:
                response = response_lookup[(fit_id, axis)]
                data = data_lookup[(fit_id, variant, axis)]
                valid = np.isfinite(response) & np.isfinite(data)
                observed_spearman, observed_cosine = signed_scores(response, data)
                within_rows.append({
                    "patient": item.patient, "fit_id": fit_id, "geometry_view": item.geometry_view,
                    "canonical_ab": bool(eligibility_index.loc[fit_id].canonical_ab), "variant": variant,
                    "axis": axis, "spearman": observed_spearman, "cosine": observed_cosine,
                    "n_valid_contacts": int(valid.sum()), "target_values_read": False,
                })
                if int(valid.sum()) < 4 or not np.isfinite(observed_spearman):
                    # Keep the explicit within-fit non-identifiability row, but do not
                    # manufacture spatial or identity nulls for a degenerate RNN field.
                    continue
                if not valid.all():
                    # Null indices must operate on the same common-contact support.
                    use = np.flatnonzero(valid); response_null = response[use]; data_null = data[use]
                    sub_names = [names[index] for index in use]; sub_xy = xy[use]
                    local_nulls = null_contract(sub_names, sub_xy, f"{fit_id}/{variant}/{axis}")
                else:
                    response_null, data_null, local_nulls = response, data, nulls
                families = {
                    "ALL_CONTACT_SYNCHRONIZED": data_null[local_nulls["all_permutations"]],
                    "WITHIN_SHAFT": data_null[local_nulls["shaft_permutations"]],
                    "DISTANCE_BIN_LOCAL": data_null[local_nulls["distance_permutations"]],
                    "GRAPH_SPECTRAL_AUTOCORRELATION": spectral_surrogates(
                        data_null, local_nulls["spectral_basis"], stable_seed(fit_id, variant, "spectral", axis)
                    ),
                }
                for family, surrogate in families.items():
                    distribution = correlation_many(response_null, surrogate)
                    eligible_family = family == "ALL_CONTACT_SYNCHRONIZED" or (
                        family == "WITHIN_SHAFT" and local_nulls["shaft_movable"] >= 4
                    ) or (family == "DISTANCE_BIN_LOCAL" and local_nulls["distance_movable"] >= 4) or family.startswith("GRAPH")
                    null_rows.append({
                        "patient": item.patient, "fit_id": fit_id, "geometry_view": item.geometry_view,
                        "canonical_ab": bool(eligibility_index.loc[fit_id].canonical_ab), "variant": variant,
                        "axis": axis, "null_family": family, "eligible": bool(eligible_family),
                        "observed_spearman": observed_spearman,
                        "null_median": float(np.median(distribution)),
                        "null_margin": float(observed_spearman - np.median(distribution)),
                        "p_one_sided_spatial": float((1 + np.sum(distribution >= observed_spearman)) / (N_NULL + 1)),
                        "n_draws": N_NULL, "shaft_movable": local_nulls["shaft_movable"],
                        "distance_movable": local_nulls["distance_movable"], "target_values_read": False,
                    })
                if variant == PRIMARY_VARIANT:
                    candidates = mapping[mapping.target_fit_id.eq(fit_id)].copy()
                    target_canonical = bool(eligibility_index.loc[fit_id].canonical_ab)
                    if target_canonical:
                        candidates = candidates[candidates.source_fit_id.map(lambda value: bool(eligibility_index.loc[value].canonical_ab))]
                    candidates = candidates[
                        candidates.source_fit_id.map(
                            lambda value: quality_index.loc[value].status != "DATA_FIELD_NOT_IDENTIFIABLE"
                        )
                    ]
                    cross = []
                    for candidate in candidates.itertuples(index=False):
                        source_vector = data_lookup[(candidate.source_fit_id, variant, axis)]
                        transported = mapped_field(pd.Series(candidate._asdict()), source_vector)
                        score = signed_scores(response, transported)[0]
                        cross.append({"source_patient": candidate.source_patient, "score": score})
                    cross_frame = pd.DataFrame(cross).groupby("source_patient", as_index=False).score.median()
                    cross_values = cross_frame.score.to_numpy(float)
                    identity_rows.append({
                        "patient": item.patient, "fit_id": fit_id, "geometry_view": item.geometry_view,
                        "canonical_ab": target_canonical, "axis": axis,
                        "same_patient_spearman": observed_spearman,
                        "cross_patient_median_spearman": float(np.nanmedian(cross_values)),
                        "identity_margin": float(observed_spearman - np.nanmedian(cross_values)),
                        "n_cross_patients": int(np.isfinite(cross_values).sum()),
                        "mapping_contract": registration["registration_contract"]["transport"],
                        "target_values_read": False,
                    })

    within = pd.DataFrame(within_rows); spatial_null = pd.DataFrame(null_rows); identity = pd.DataFrame(identity_rows)

    # Cell-level sign stability relative to the frozen heldout data field.
    stability_rows = []
    for (fit_id, axis), group in cell_response.groupby(["fit_id", "perturbation_axis"]):
        data = data_lookup[(fit_id, PRIMARY_VARIANT, axis)]
        scores = []
        for _, part in group.groupby(["public_arm", "seed"]):
            vector = part.sort_values("contact_index").response.to_numpy(float)
            scores.append(signed_scores(vector, data)[0])
        stability_rows.append({
            "patient": fit_manifest.loc[fit_id].patient, "fit_id": fit_id, "axis": axis,
            "median_cell_spearman": float(np.nanmedian(scores)),
            "positive_cell_fraction": float(np.mean(np.asarray(scores) > 0)), "n_cells": len(scores),
        })
    stability = pd.DataFrame(stability_rows)

    primary_null = spatial_null[
        spatial_null.variant.eq(PRIMARY_VARIANT) & spatial_null.null_family.eq("ALL_CONTACT_SYNCHRONIZED")
    ]
    patient_frames, summaries = [], {}
    for canonical, tier in ((False, "generic_all_identifiable"), (True, "canonical_ab_shared")):
        null_part = primary_null[primary_null.canonical_ab].copy() if canonical else primary_null.copy()
        identity_part = identity[identity.canonical_ab].copy() if canonical else identity.copy()
        stability_part = stability[stability.fit_id.isin(identity_part.fit_id.unique())]
        null_patient = null_part.groupby(["patient", "axis"], as_index=False).null_margin.median()
        identity_patient = identity_part.groupby(["patient", "axis"], as_index=False).identity_margin.median()
        stability_patient = stability_part.groupby(["patient", "axis"], as_index=False)[["median_cell_spearman", "positive_cell_fraction"]].median()
        pivot_null = null_patient.pivot(index="patient", columns="axis", values="null_margin")
        pivot_identity = identity_patient.pivot(index="patient", columns="axis", values="identity_margin")
        rows = []
        for patient in sorted(set(pivot_null.index) & set(pivot_identity.index)):
            row = {"tier": tier, "patient": patient}
            for axis in AXES:
                row[f"{axis.lower()}_spatial_null_margin"] = float(pivot_null.loc[patient, axis])
                row[f"{axis.lower()}_identity_margin"] = float(pivot_identity.loc[patient, axis])
                entry = stability_patient[(stability_patient.patient == patient) & (stability_patient.axis == axis)]
                row[f"{axis.lower()}_median_cell_spearman"] = float(entry.median_cell_spearman.iloc[0])
                row[f"{axis.lower()}_positive_cell_fraction"] = float(entry.positive_cell_fraction.iloc[0])
            rows.append(row)
        frame = pd.DataFrame(rows); patient_frames.append(frame)
        endpoint_names = [f"{axis.lower()}_{kind}_margin" for axis in AXES for kind in ("spatial_null", "identity")]
        endpoints = {
            name: one_sided_summary(frame[name].to_numpy(float), stable_seed("C5", tier, name))
            for name in endpoint_names
        }
        adjusted = holm_adjust({name: values["p_one_sided"] for name, values in endpoints.items()})
        for name, values in endpoints.items():
            values["p_holm_C5_family"] = adjusted[name]
            values["status"] = "SUPPORTED" if values["median"] > 0 and adjusted[name] < 0.05 else "UNSUPPORTED"
        stable = all(float(np.nanmedian(frame[f"{axis.lower()}_positive_cell_fraction"])) > 0.5 for axis in AXES)
        summaries[tier] = {
            "n_patients": int(frame.patient.nunique()), "endpoints": endpoints,
            "cross_seed_arm_stability": {
                axis: {
                    "median_cell_alignment": float(np.nanmedian(frame[f"{axis.lower()}_median_cell_spearman"])),
                    "median_positive_cell_fraction": float(np.nanmedian(frame[f"{axis.lower()}_positive_cell_fraction"])),
                } for axis in AXES
            },
            "C5_status": "SUPPORTED" if all(v["status"] == "SUPPORTED" for v in endpoints.values()) and stable else "UNSUPPORTED",
        }

    atomic_write_csv(DATA / "WITHIN_PATIENT_ALIGNMENT.csv", within)
    atomic_write_csv(DATA / "SPATIAL_NULL_ALIGNMENT.csv", spatial_null)
    atomic_write_csv(DATA / "CROSS_PATIENT_IDENTITY_ALIGNMENT.csv", identity)
    atomic_write_csv(DATA / "CELL_RESPONSE_ALIGNMENT_STABILITY.csv", stability)
    atomic_write_csv(DATA / "C5_PATIENT_EFFECTS.csv", pd.concat(patient_frames, ignore_index=True))
    payload = {
        "contract": "topic5_data_field_alignment_C5_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "alignment_revision": ALIGNMENT_REVISION, "status": "COMPLETE",
        "tiers": summaries, "primary_variant": PRIMARY_VARIANT,
        "response_aggregation": "EVENT_FIRST_EXISTING_THEN_EQUAL_PHASE_TAU_MEAN_MEDIAN_SEED_REAL_ARM",
        "heldout_grouping": "PREFIX_ONLY_ASSIGNMENT_TO_TRAIN_FROZEN_MODES; SUFFIX FIELD VALUES HELD OUT",
        "identity_registration": registration["registration_contract"],
        "spatial_nulls": ["ALL_CONTACT_SYNCHRONIZED", "WITHIN_SHAFT", "DISTANCE_BIN_LOCAL", "GRAPH_SPECTRAL_AUTOCORRELATION"],
        "n_null_draws": N_NULL,
        "data_field_quality": quality.status.value_counts().to_dict(),
        "claim_boundary": "AXIS_NORMALIZED_IDENTITY; NOT WHOLE_BRAIN_ANATOMICAL_REGISTRATION",
        "target_values_read": False,
    }
    atomic_write_json(DATA / "DATA_ALIGNMENT_SUMMARY.json", payload)
    seal = {
        "contract": "topic5_data_alignment_freeze_seal_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS",
        "files": {name: sha256_file(DATA / name) for name in [
            "FINITE_TIME_RESPONSE_FIELDS.npz", "PERTURBATION_RESPONSE_MATRIX.json",
            "HELDOUT_DATA_FIELDS.csv", "DATA_FIELD_QUALITY.csv", "PRIMARY_FIT_RESPONSE_FIELDS.csv",
            "WITHIN_PATIENT_ALIGNMENT.csv", "SPATIAL_NULL_ALIGNMENT.csv",
            "CROSS_PATIENT_IDENTITY_ALIGNMENT.csv", "DATA_ALIGNMENT_SUMMARY.json",
        ]},
        "geometry_mapping_manifest_sha256": sha256_file(MAPPING / "CROSS_PATIENT_GEOMETRY_MAPPING_MANIFEST.csv"),
        "target_values_read": False,
    }
    atomic_write_json(DATA / "DATA_ALIGNMENT_FREEZE_SEAL.json", seal)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
