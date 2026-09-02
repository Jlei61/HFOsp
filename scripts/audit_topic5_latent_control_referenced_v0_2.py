#!/usr/bin/env python3
"""Recover the preregistered control-referenced statistics for Topic 5.2 v0.2.

Three spec clauses were computed during execution but never aggregated into a
summary, so the closeout reported effects against a bare zero reference:

* spec 5.7 -- tangent transport and transverse contraction must be compared
  against control directions, and `C_perp` must be shown to be *below controls*,
  not only below one.  `C2_PATIENT_EFFECTS.csv` already carries the frozen
  `*_real_minus_C_suffix` columns; this script aggregates them.
* spec 9.2 -- C5 must report shaft / distance / autocorrelation-preserving
  spatial nulls.  `SPATIAL_NULL_ALIGNMENT.csv` already carries all four null
  families per fit; only the synchronized all-contact family reached the
  patient-level summary.
* spec 9.3 -- the cross-patient identity null transports the *other* patient's
  field through a one-dimensional kernel while the same-patient arm stays raw.
  This script rebuilds the frozen kernel with the identical per-pair bandwidth
  and applies it to the patient's own field, giving a smoothing-matched margin.

Nothing here re-runs a model, re-selects an event, or edits a sealed producer
output.  It is additive reporting on frozen artifacts.
"""
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
from src.topic5_latent_perturbation_v0_2 import stable_seed  # noqa: E402
from scripts.freeze_topic5_cross_patient_geometry_mappings_v0_2 import MAPPING  # noqa: E402
from scripts.run_topic5_data_field_alignment_v0_2 import (  # noqa: E402
    DATA, PRIMARY_VARIANT, centered_valid, signed_scores,
)
from scripts.run_topic5_latent_pass1_v0_2 import OUT, PARENT  # noqa: E402
from scripts.summarize_topic5_latent_transport_v0_2 import (  # noqa: E402
    PRIMARY_ENDPOINTS as C2_PRIMARY_ENDPOINTS, TRANSPORT,
)
from scripts.summarize_topic5_latent_geometry_v0_2 import holm_adjust, one_sided_summary  # noqa: E402


ADDENDUM_REVISION = "V0_2_CONTROL_REFERENCED_ADDENDUM_R0"
NULL_FAMILIES = (
    "ALL_CONTACT_SYNCHRONIZED", "WITHIN_SHAFT",
    "DISTANCE_BIN_LOCAL", "GRAPH_SPECTRAL_AUTOCORRELATION",
)
PREREGISTERED_PRIMARY_FAMILY = "ALL_CONTACT_SYNCHRONIZED"
TIERS = (("generic_all_identifiable", False), ("canonical_ab_shared", True))


def summarize(values: np.ndarray, *, label: str) -> dict[str, object]:
    return one_sided_summary(np.asarray(values, float), stable_seed(ADDENDUM_REVISION, label))


def transport_control_reference() -> dict[str, object]:
    """Spec 5.7: transport and contraction relative to the C-suffix control arm."""
    effects = pd.read_csv(TRANSPORT / "C2_PATIENT_EFFECTS.csv")
    output: dict[str, object] = {}
    for tier, _ in TIERS:
        group = effects[effects["tier"].eq(tier)]
        endpoints = {
            name: summarize(
                group[f"{name}_real_minus_C_suffix"].to_numpy(float),
                label=f"C2_control/{tier}/{name}",
            )
            for name in C2_PRIMARY_ENDPOINTS
        }
        adjusted = holm_adjust({name: value["p_one_sided"] for name, value in endpoints.items()})
        for name, value in endpoints.items():
            value["p_holm_control_family"] = adjusted[name]
            value["absolute_median_vs_zero"] = float(np.nanmedian(group[name]))
            value["status_vs_control"] = (
                "SUPPORTED" if value["median"] > 0 and adjusted[name] < 0.05 else "UNSUPPORTED"
            )
        output[tier] = {"n_patients": int(group["patient"].nunique()), "endpoints": endpoints}
    return output


def spatial_null_family_sensitivity() -> tuple[pd.DataFrame, dict[str, object]]:
    """Spec 9.2: all four spatial null families at the patient level."""
    nulls = pd.read_csv(DATA / "SPATIAL_NULL_ALIGNMENT.csv")
    nulls = nulls[nulls["variant"].eq(PRIMARY_VARIANT) & nulls["eligible"]]
    reference = pd.read_csv(DATA / "C5_PATIENT_EFFECTS.csv")
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    for tier, canonical_only in TIERS:
        cohort = set(reference[reference["tier"].eq(tier)]["patient"])
        part = nulls[nulls["canonical_ab"]] if canonical_only else nulls
        tier_summary: dict[str, object] = {}
        for family in NULL_FAMILIES:
            family_rows = part[part["null_family"].eq(family)]
            for axis in ("PROGRESS", "FIELD"):
                patient = family_rows[family_rows["axis"].eq(axis)].groupby(
                    "patient"
                )["null_margin"].median()
                patient = patient[patient.index.isin(cohort)].sort_index()
                preregistered = patient.to_numpy(float)
                entry = {
                    "preregistered_orientation": summarize(
                        preregistered, label=f"C5_null/{tier}/{family}/{axis}"
                    ),
                }
                if axis == "PROGRESS":
                    entry["laterness_posthoc_exact_mirror"] = summarize(
                        -preregistered, label=f"C5_null_lat/{tier}/{family}/{axis}"
                    )
                tier_summary[f"{family}/{axis}"] = entry
                for name, value in zip(patient.index, preregistered):
                    rows.append({
                        "tier": tier, "patient": name, "null_family": family, "axis": axis,
                        "preregistered_margin": float(value),
                        "laterness_margin_posthoc": float(-value) if axis == "PROGRESS" else np.nan,
                        "is_preregistered_primary_family": family == PREREGISTERED_PRIMARY_FAMILY,
                    })
        summary[tier] = tier_summary
    return pd.DataFrame(rows), summary


def _normalized_axis(fit_id: str) -> np.ndarray:
    with np.load(PARENT / "cache" / fit_id / "plane.npz", allow_pickle=False) as source:
        xy = np.asarray(source["contacts_xy_mm"], float)
        scale = float(np.asarray(source["scale_mm"]).ravel()[0])
    return xy[:, 0] / scale


def _kernel_transport(target_axis: np.ndarray, source_axis: np.ndarray, bandwidth: float) -> np.ndarray:
    squared = (target_axis[:, None] - source_axis[None, :]) ** 2
    weights = np.exp(-squared / (2.0 * bandwidth ** 2))
    return weights / weights.sum(axis=1, keepdims=True)


def _apply(weights: np.ndarray, field: np.ndarray) -> np.ndarray:
    valid = np.isfinite(field)
    if int(valid.sum()) < 4:
        return np.full(weights.shape[0], np.nan)
    usable = weights[:, valid]
    usable = usable / np.maximum(usable.sum(axis=1, keepdims=True), 1e-12)
    result, _, ok = centered_valid(usable @ np.asarray(field, float)[valid])
    return result if ok else np.full(weights.shape[0], np.nan)


def smoothing_matched_identity() -> tuple[pd.DataFrame, dict[str, object]]:
    """Spec 9.3: give the same-patient arm the same kernel the null arm gets."""
    fields = pd.read_csv(DATA / "HELDOUT_DATA_FIELDS.csv")
    fields = fields[fields["variant"].eq(PRIMARY_VARIANT)]
    response = pd.read_csv(DATA / "PRIMARY_FIT_RESPONSE_FIELDS.csv")
    quality = pd.read_csv(DATA / "DATA_FIELD_QUALITY.csv").set_index("fit_id")
    eligibility = pd.read_csv(OUT / "MODE_AXIS_ELIGIBILITY.csv").set_index("fit_id")
    mapping = pd.read_csv(MAPPING / "CROSS_PATIENT_GEOMETRY_MAPPING_MANIFEST.csv")
    data_lookup = {
        key: group.sort_values("contact_index")["field_value"].to_numpy(float)
        for key, group in fields.groupby(["fit_id", "axis"])
    }
    response_lookup = {
        key: group.sort_values("contact_index")["response"].to_numpy(float)
        for key, group in response.groupby(["fit_id", "perturbation_axis"])
    }
    axis_lookup = {fit_id: _normalized_axis(fit_id) for fit_id in sorted(set(fields["fit_id"]))}
    identity = pd.read_csv(DATA / "CROSS_PATIENT_IDENTITY_ALIGNMENT.csv")
    rows: list[dict[str, object]] = []
    for item in identity.itertuples(index=False):
        fit_id, axis = str(item.fit_id), str(item.axis)
        target_axis = axis_lookup[fit_id]
        own_field = data_lookup[(fit_id, axis)]
        own_response = response_lookup[(fit_id, axis)]
        candidates = mapping[mapping["target_fit_id"].eq(fit_id)]
        if bool(item.canonical_ab):
            candidates = candidates[candidates["source_fit_id"].map(
                lambda value: bool(eligibility.loc[value, "canonical_ab"])
            )]
        candidates = candidates[candidates["source_fit_id"].map(
            lambda value: quality.loc[value, "status"] != "DATA_FIELD_NOT_IDENTIFIABLE"
        )]
        matched, unmatched = [], []
        for candidate in candidates.itertuples(index=False):
            bandwidth = float(candidate.bandwidth_normalized_axis)
            self_weights = _kernel_transport(target_axis, target_axis, bandwidth)
            matched.append({
                "source_patient": candidate.source_patient,
                "score": signed_scores(own_response, _apply(self_weights, own_field))[0],
            })
            source_weights = _kernel_transport(
                target_axis, axis_lookup[str(candidate.source_fit_id)], bandwidth
            )
            unmatched.append({
                "source_patient": candidate.source_patient,
                "score": signed_scores(
                    own_response, _apply(source_weights, data_lookup[(str(candidate.source_fit_id), axis)])
                )[0],
            })
        matched_self = float(np.nanmedian(
            pd.DataFrame(matched).groupby("source_patient").score.median().to_numpy(float)
        ))
        cross = float(np.nanmedian(
            pd.DataFrame(unmatched).groupby("source_patient").score.median().to_numpy(float)
        ))
        rows.append({
            "patient": str(item.patient), "fit_id": fit_id,
            "canonical_ab": bool(item.canonical_ab), "axis": axis,
            "raw_same_patient_spearman": float(item.same_patient_spearman),
            "smoothing_matched_same_patient_spearman": matched_self,
            "cross_patient_median_spearman": cross,
            "frozen_identity_margin": float(item.identity_margin),
            "smoothing_matched_identity_margin": matched_self - cross,
        })
    frame = pd.DataFrame(rows)
    summary: dict[str, object] = {}
    reference = pd.read_csv(DATA / "C5_PATIENT_EFFECTS.csv")
    for tier, canonical_only in TIERS:
        cohort = set(reference[reference["tier"].eq(tier)]["patient"])
        part = frame[frame["canonical_ab"]] if canonical_only else frame
        tier_summary: dict[str, object] = {}
        for axis in ("PROGRESS", "FIELD"):
            patient = part[part["axis"].eq(axis)].groupby("patient")[
                ["frozen_identity_margin", "smoothing_matched_identity_margin"]
            ].median()
            patient = patient[patient.index.isin(cohort)].sort_index()
            entry = {
                "frozen_identity_margin": summarize(
                    patient["frozen_identity_margin"].to_numpy(float),
                    label=f"C5_identity_frozen/{tier}/{axis}",
                ),
                "smoothing_matched_identity_margin": summarize(
                    patient["smoothing_matched_identity_margin"].to_numpy(float),
                    label=f"C5_identity_matched/{tier}/{axis}",
                ),
            }
            if axis == "PROGRESS":
                entry["smoothing_matched_laterness_posthoc"] = summarize(
                    -patient["smoothing_matched_identity_margin"].to_numpy(float),
                    label=f"C5_identity_matched_lat/{tier}/{axis}",
                )
            tier_summary[axis] = entry
        summary[tier] = tier_summary
    return frame, summary


def topology_per_axis() -> dict[str, object]:
    """Spec 8 C4: keep the absolute similarities next to the control margin."""
    topology = pd.read_csv(OUT / "axis_perturbation" / "responses" / "C4_TOPOLOGY_FIELD_EFFECTS.csv")
    output: dict[str, object] = {}
    for axis in ("PROGRESS", "FIELD"):
        part = topology[topology["perturbation_axis"].eq(axis)]
        margin = summarize(part["topology_convergence_margin"].to_numpy(float), label=f"C4/{axis}")
        low, high = margin["ci95_median"]
        output[axis] = {
            "median_real_arm_pair_cosine": float(part["real_arm_pair_cosine"].median()),
            "median_real_arm_to_order_shuffled_cosine": float(part["real_arm_to_C_suffix_cosine"].median()),
            "topology_convergence_margin": margin,
            "ci95_median_excludes_zero": bool(low > 0.0),
        }
    return output


def sign_semantics_completeness() -> dict[str, object]:
    """Report both tiers of the post-hoc laterness reorientation, not just the better one."""
    audit = json.loads(
        (OUT / "axis_perturbation" / "responses" / "PROGRESS_SIGN_SEMANTICS_AUDIT.json").read_text()
    )
    output: dict[str, object] = {}
    for tier, _ in TIERS:
        entry = audit["tiers"][tier]["D_progress_laterness_posthoc"]
        low, high = entry["ci95_median"]
        output[tier] = {
            "median": entry["median"], "positive": entry["positive"],
            "n_patients": entry["n_patients"], "p_one_sided": entry["p_one_sided"],
            "ci95_median": entry["ci95_median"],
            "ci95_median_excludes_zero": bool(low > 0.0),
        }
    return output


def main() -> None:
    control = transport_control_reference()
    null_frame, null_summary = spatial_null_family_sensitivity()
    identity_frame, identity_summary = smoothing_matched_identity()
    atomic_write_csv(OUT / "C5_SPATIAL_NULL_FAMILY_PATIENT_EFFECTS.csv", null_frame)
    atomic_write_csv(OUT / "C5_SMOOTHING_MATCHED_IDENTITY.csv", identity_frame)
    payload = {
        "contract": "topic5_latent_landscape_control_referenced_addendum_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "addendum_revision": ADDENDUM_REVISION,
        "status": "COMPLETE",
        "role": (
            "ADDITIVE_REPORTING_ON_FROZEN_ARTIFACTS; NO_MODEL_REPLAY; "
            "NO_PREREGISTERED_PRIMARY_IS_CHANGED"
        ),
        "C2_transport_vs_order_shuffled_control": control,
        "C4_topology_per_axis": topology_per_axis(),
        "C5_spatial_null_families": {
            "preregistered_primary_family": PREREGISTERED_PRIMARY_FAMILY,
            "note": (
                "The preregistered primary family destroys all spatial structure, so its null "
                "median is ~0 and the reported margin is numerically the raw signed Spearman. "
                "The shaft- and distance-preserving families retain part of the alignment inside "
                "the null and therefore give the spatially controlled effect size."
            ),
            "tiers": null_summary,
        },
        "C5_identity_smoothing_match": {
            "issue": (
                "The frozen identity null transports every cross-patient field through a "
                "one-dimensional Nadaraya-Watson kernel on the normalized propagation axis while "
                "the same-patient arm keeps its raw contact resolution."
            ),
            "control": "SAME_KERNEL_AND_PER_PAIR_BANDWIDTH_APPLIED_TO_THE_PATIENTS_OWN_FIELD",
            "tiers": identity_summary,
        },
        "sign_semantics_both_tiers": sign_semantics_completeness(),
        "target_values_read": False,
    }
    atomic_write_json(OUT / "CONTROL_REFERENCED_ADDENDUM.json", payload)
    payload_out = dict(payload)
    payload_out["artifact_hashes"] = {
        name: sha256_file(OUT / name)
        for name in (
            "C5_SPATIAL_NULL_FAMILY_PATIENT_EFFECTS.csv",
            "C5_SMOOTHING_MATCHED_IDENTITY.csv",
        )
    }
    atomic_write_json(OUT / "CONTROL_REFERENCED_ADDENDUM.json", payload_out)
    print(json.dumps(payload_out, indent=2))


if __name__ == "__main__":
    main()
