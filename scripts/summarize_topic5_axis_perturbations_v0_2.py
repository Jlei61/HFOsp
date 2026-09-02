#!/usr/bin/env python3
"""Adjudicate Topic 5.2 C3 axis selectivity and C4 topology convergence."""
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

from src.topic5_latent_landscape_v0_2 import atomic_write_csv, atomic_write_json  # noqa: E402
from scripts.run_topic5_axis_perturbations_v0_2 import (  # noqa: E402
    CONTROL_NAMES, PERTURB, PERTURB_REVISION, response_dir,
)
from scripts.run_topic5_latent_pass1_v0_2 import ANALYSIS_REVISION, OUT, PARENT  # noqa: E402
from scripts.summarize_topic5_latent_geometry_v0_2 import holm_adjust, one_sided_summary  # noqa: E402


REAL_ARMS = ("L0", "L1", "L2m", "L3")
PRIMARY_DOSE_INDEX = 1
FUTURE_TAU = (1, 2, 3)


def finite_mean(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    numerator = np.where(finite, array, 0.0).sum(axis=axis)
    denominator = finite.sum(axis=axis)
    return np.divide(
        numerator, denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def finite_median(values: object) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else float("nan")


def gain_adjusted_mean(effect: np.ndarray, gain: np.ndarray) -> float:
    effect = np.asarray(effect, dtype=float)
    gain = np.asarray(gain, dtype=float)
    use = np.isfinite(effect) & np.isfinite(gain)
    if int(use.sum()) < 5:
        return float("nan")
    x, y = gain[use], effect[use]
    centered = x - float(np.median(x))
    denominator = float(np.dot(centered, centered))
    slope = float(np.dot(centered, y - y.mean()) / denominator) if denominator > 1e-12 else 0.0
    adjusted = y - slope * centered
    return float(np.mean(adjusted))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, float), np.asarray(right, float)
    use = np.isfinite(a) & np.isfinite(b)
    if int(use.sum()) < 2:
        return float("nan")
    a, b = a[use] - a[use].mean(), b[use] - b[use].mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else float("nan")


def cell_tables(row: pd.Series) -> tuple[
    list[dict[str, object]], list[dict[str, object]], list[dict[str, object]],
    list[dict[str, object]],
]:
    target = response_dir(row)
    with np.load(target / "axis_responses.npz", allow_pickle=False) as source:
        axis = {key: np.asarray(source[key]) for key in source.files}
    with np.load(target / "control_responses.npz", allow_pickle=False) as source:
        controls = {key: np.asarray(source[key]) for key in source.files}
    with np.load(target / "chord_responses.npz", allow_pickle=False) as source:
        chords = {key: np.asarray(source[key]) for key in source.files}
    eligibility = pd.read_csv(OUT / "MODE_AXIS_ELIGIBILITY.csv").set_index("fit_id").loc[row.fit_id]
    canonical = bool(eligibility["canonical_ab"])
    phase_target = axis["phase_target"].astype(float)
    phase_rows: list[dict[str, object]] = []
    field_rows: list[dict[str, object]] = []
    chord_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    provenance = json.loads((PARENT / "cache" / str(row.fit_id) / "provenance.json").read_text())
    contact_names = [str(value) for value in provenance["joint_contacts"]]

    primary = axis["open_scores"][:, :, PRIMARY_DOSE_INDEX]
    primary_future = finite_mean(primary[:, :, FUTURE_TAU, :], axis=2)
    closed_future = finite_mean(
        axis["closed_scores"][:, :, PRIMARY_DOSE_INDEX, FUTURE_TAU, :], axis=2
    )
    immediate_gain = axis["open_logit_response_norm"][:, :, PRIMARY_DOSE_INDEX, 0]
    for phase in sorted(np.unique(phase_target)):
        use = phase_target == phase
        matrix = finite_mean(primary_future[use], axis=0)
        closed_matrix = finite_mean(closed_future[use], axis=0)
        terminal_matrix = finite_mean(
            axis["terminal_scores"][use, :, PRIMARY_DOSE_INDEX], axis=0
        )
        progress_event = primary_future[use, 0, 0] - np.abs(primary_future[use, 0, 1])
        field_event = primary_future[use, 1, 1] - np.abs(primary_future[use, 1, 0])
        control_future = finite_mean(
            controls["open_scores"][use][:, :, FUTURE_TAU, :], axis=2
        )
        local_normal_abs = finite_mean(np.abs(control_future[:, :8]), axis=1)
        for control_index, control_name in enumerate(CONTROL_NAMES):
            response = finite_mean(control_future[:, control_index], axis=0)
            control_rows.append({
                "patient": str(row.patient), "fit_id": str(row.fit_id),
                "geometry_view": str(row.geometry_view), "public_arm": str(row.public_arm),
                "seed": int(row.seed), "canonical_ab": canonical,
                "phase_target": float(phase), "control_family": control_name,
                "S_progress": float(response[0]), "S_field": float(response[1]),
                "response_norm": float(np.linalg.norm(response)),
                "target_values_read": False,
            })
        phase_rows.append({
            "patient": str(row.patient), "fit_id": str(row.fit_id),
            "geometry_view": str(row.geometry_view), "public_arm": str(row.public_arm),
            "seed": int(row.seed), "canonical_ab": canonical, "phase_target": float(phase),
            "R_progress_from_progress": float(matrix[0, 0]),
            "R_field_from_progress": float(matrix[0, 1]),
            "R_progress_from_field": float(matrix[1, 0]),
            "R_field_from_field": float(matrix[1, 1]),
            "D_progress": float(matrix[0, 0] - abs(matrix[0, 1])),
            "D_field": float(matrix[1, 1] - abs(matrix[1, 0])),
            "D_progress_gain_adjusted": gain_adjusted_mean(progress_event, immediate_gain[use, 0]),
            "D_field_gain_adjusted": gain_adjusted_mean(field_event, immediate_gain[use, 1]),
            "closed_D_progress": float(closed_matrix[0, 0] - abs(closed_matrix[0, 1])),
            "closed_D_field": float(closed_matrix[1, 1] - abs(closed_matrix[1, 0])),
            "terminal_D_progress": float(terminal_matrix[0, 0] - abs(terminal_matrix[0, 1])),
            "terminal_D_field": float(terminal_matrix[1, 1] - abs(terminal_matrix[1, 0])),
            "progress_diagonal_minus_local_normal_abs": float(
                matrix[0, 0] - finite_mean(local_normal_abs[:, 0], axis=0)
            ),
            "field_diagonal_minus_local_normal_abs": float(
                matrix[1, 1] - finite_mean(local_normal_abs[:, 1], axis=0)
            ),
            "n_progress_states": int(np.isfinite(primary_future[use, 0, 0]).sum()),
            "n_field_states": int(np.isfinite(primary_future[use, 1, 1]).sum()),
            "target_values_read": False,
        })
        for axis_index, axis_name in enumerate(("PROGRESS", "FIELD")):
            for tau in FUTURE_TAU:
                contact = finite_mean(
                    axis["open_contact_response"][use, axis_index, PRIMARY_DOSE_INDEX, tau],
                    axis=0,
                )
                for contact_index, value in enumerate(contact):
                    field_rows.append({
                        "patient": str(row.patient), "fit_id": str(row.fit_id),
                        "geometry_view": str(row.geometry_view), "public_arm": str(row.public_arm),
                        "seed": int(row.seed), "canonical_ab": canonical,
                        "phase_target": float(phase), "tau": tau,
                        "perturbation_axis": axis_name, "contact_index": contact_index,
                        "contact_name": contact_names[contact_index], "response": float(value),
                        "target_values_read": False,
                    })

    if len(chords["family"]):
        source_phase = phase_target[chords["reference_index"].astype(int)]
        chord_future = finite_mean(
            chords["open_scores"][:, PRIMARY_DOSE_INDEX, FUTURE_TAU], axis=1
        )
        chord_closed = finite_mean(
            chords["closed_scores"][:, PRIMARY_DOSE_INDEX, FUTURE_TAU], axis=1
        )
        orientation = np.sign(chords["u_difference"].astype(float))
        for phase in sorted(np.unique(source_phase)):
            for family in ("HIGH_U", "SMALL_U"):
                use = (source_phase == phase) & (chords["family"] == family)
                oriented = chord_future[use, 1] * orientation[use]
                oriented_closed = chord_closed[use, 1] * orientation[use]
                oriented_terminal = (
                    chords["terminal_scores"][use, PRIMARY_DOSE_INDEX, 1] * orientation[use]
                )
                chord_rows.append({
                    "patient": str(row.patient), "fit_id": str(row.fit_id),
                    "geometry_view": str(row.geometry_view), "public_arm": str(row.public_arm),
                    "seed": int(row.seed), "canonical_ab": canonical,
                    "phase_target": float(phase), "family": family,
                    "oriented_open_field_response": float(finite_mean(oriented, axis=0)),
                    "oriented_closed_field_response": float(finite_mean(oriented_closed, axis=0)),
                    "oriented_terminal_field_response": float(finite_mean(oriented_terminal, axis=0)),
                    "n_pairs": int(np.isfinite(oriented).sum()), "target_values_read": False,
                })
    return phase_rows, field_rows, chord_rows, control_rows


def aggregate_patient_effects(cell_phase: pd.DataFrame, canonical_only: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = cell_phase[cell_phase["canonical_ab"]].copy() if canonical_only else cell_phase.copy()
    metrics = [
        "D_progress", "D_field", "D_progress_gain_adjusted", "D_field_gain_adjusted",
        "closed_D_progress", "closed_D_field", "terminal_D_progress", "terminal_D_field",
        "progress_diagonal_minus_local_normal_abs", "field_diagonal_minus_local_normal_abs",
    ]
    seed = selected.groupby(
        ["patient", "fit_id", "geometry_view", "public_arm", "seed"], as_index=False
    )[metrics].mean()
    fit = seed.groupby(
        ["patient", "fit_id", "geometry_view", "public_arm"], as_index=False
    )[metrics].median()
    arm = fit.groupby(["patient", "public_arm"], as_index=False)[metrics].median()
    patient_rows, arm_rows = [], []
    tier = "canonical_ab_shared" if canonical_only else "generic_all_identifiable"
    for patient, group in arm.groupby("patient"):
        indexed = group.set_index("public_arm")
        real = indexed.loc[list(REAL_ARMS), metrics]
        row = {"tier": tier, "patient": patient}
        for metric in metrics:
            row[metric] = float(real[metric].median())
            row[f"{metric}_real_minus_C_suffix"] = float(
                real[metric].median() - indexed.loc["C-suffix", metric]
            )
        row["leave_one_arm_out_min_D_progress"] = float(min(
            real.drop(index=arm_name)["D_progress"].median() for arm_name in REAL_ARMS
        ))
        row["leave_one_arm_out_min_D_field"] = float(min(
            real.drop(index=arm_name)["D_field"].median() for arm_name in REAL_ARMS
        ))
        patient_rows.append(row)
        for arm_name in (*REAL_ARMS, "C-suffix"):
            arm_rows.append({
                "tier": tier, "patient": patient, "public_arm": arm_name,
                **{metric: float(indexed.loc[arm_name, metric]) for metric in metrics},
            })
    return pd.DataFrame(patient_rows), pd.DataFrame(arm_rows)


def topology_field_effects(fields: pd.DataFrame) -> pd.DataFrame:
    seed = fields.groupby(
        ["patient", "fit_id", "geometry_view", "public_arm", "seed", "perturbation_axis", "contact_index"],
        as_index=False,
    )["response"].mean()
    fit = seed.groupby(
        ["patient", "fit_id", "geometry_view", "public_arm", "perturbation_axis", "contact_index"],
        as_index=False,
    )["response"].median()
    rows = []
    for (patient, fit_id, axis_name), group in fit.groupby(["patient", "fit_id", "perturbation_axis"]):
        vectors = {
            arm: part.sort_values("contact_index")["response"].to_numpy(float)
            for arm, part in group.groupby("public_arm")
        }
        if not set((*REAL_ARMS, "C-suffix")).issubset(vectors):
            continue
        real_pair = [
            cosine(vectors[REAL_ARMS[left]], vectors[REAL_ARMS[right]])
            for left in range(len(REAL_ARMS)) for right in range(left + 1, len(REAL_ARMS))
        ]
        real_control = [cosine(vectors[arm], vectors["C-suffix"]) for arm in REAL_ARMS]
        rows.append({
            "patient": patient, "fit_id": fit_id, "perturbation_axis": axis_name,
            "real_arm_pair_cosine": finite_median(real_pair),
            "real_arm_to_C_suffix_cosine": finite_median(real_control),
            "topology_convergence_margin": finite_median(real_pair) - finite_median(real_control),
        })
    frame = pd.DataFrame(rows)
    return frame.groupby(["patient", "perturbation_axis"], as_index=False)[
        ["real_arm_pair_cosine", "real_arm_to_C_suffix_cosine", "topology_convergence_margin"]
    ].median()


def main() -> None:
    audit = json.loads((PERTURB / "PERTURBATION_AUDIT.json").read_text())
    if audit.get("status") != "PASS" or audit.get("perturbation_revision") != PERTURB_REVISION:
        raise RuntimeError("Pass 2 response audit must pass before C3/C4 aggregation")
    manifest = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv")
    phase_rows, field_rows, chord_rows, control_rows = [], [], [], []
    for _, row in manifest.iterrows():
        phase, fields, chords, controls = cell_tables(row)
        phase_rows.extend(phase)
        field_rows.extend(fields)
        chord_rows.extend(chords)
        control_rows.extend(controls)
    cell_phase = pd.DataFrame(phase_rows)
    fields = pd.DataFrame(field_rows)
    chord_frame = pd.DataFrame(chord_rows)
    control_frame = pd.DataFrame(control_rows)
    patient_parts, arm_parts = [], []
    for canonical_only in (False, True):
        patient, arm = aggregate_patient_effects(cell_phase, canonical_only)
        patient_parts.append(patient); arm_parts.append(arm)
    patients = pd.concat(patient_parts, ignore_index=True)
    arms = pd.concat(arm_parts, ignore_index=True)
    topology = topology_field_effects(fields)

    tiers: dict[str, object] = {}
    for tier, group in patients.groupby("tier", sort=False):
        primary = {
            endpoint: one_sided_summary(
                group[endpoint].to_numpy(float),
                int.from_bytes(hashlib.sha256(f"C3/{tier}/{endpoint}".encode()).digest()[:8], "little"),
            )
            for endpoint in ("D_progress", "D_field")
        }
        adjusted = holm_adjust({key: value["p_one_sided"] for key, value in primary.items()})
        for endpoint in primary:
            primary[endpoint]["p_holm"] = adjusted[endpoint]
            adjusted_endpoint = f"{endpoint}_gain_adjusted"
            adjusted_values = group[adjusted_endpoint].to_numpy(float)
            primary[endpoint]["gain_adjusted_median"] = float(np.nanmedian(adjusted_values))
            closed_endpoint = f"closed_{endpoint}"
            primary[endpoint]["closed_loop_median"] = float(np.nanmedian(group[closed_endpoint]))
            primary[endpoint]["status"] = (
                "SUPPORTED" if (
                    primary[endpoint]["median"] > 0 and adjusted[endpoint] < 0.05
                    and np.nanmedian(adjusted_values) > 0
                    and np.nanmedian(group[closed_endpoint]) > 0
                ) else "UNSUPPORTED"
            )
        tiers[tier] = {
            "n_patients": int(group["patient"].nunique()),
            "co_primary": primary,
            "C3_progress_status": primary["D_progress"]["status"],
            "C3_field_status": primary["D_field"]["status"],
            "C3_joint_status": (
                "SUPPORTED" if all(value["status"] == "SUPPORTED" for value in primary.values())
                else "UNSUPPORTED"
            ),
            "local_normal_sensitivity": {
                "progress_median": float(np.nanmedian(group["progress_diagonal_minus_local_normal_abs"])),
                "field_median": float(np.nanmedian(group["field_diagonal_minus_local_normal_abs"])),
            },
            "leave_one_arm_out": {
                "progress_patient_median_min": float(np.nanmedian(group["leave_one_arm_out_min_D_progress"])),
                "field_patient_median_min": float(np.nanmedian(group["leave_one_arm_out_min_D_field"])),
            },
        }

    topology_summary = {}
    topology_endpoints = {}
    for axis_name in ("PROGRESS", "FIELD"):
        part = topology[topology["perturbation_axis"].eq(axis_name)]
        topology_endpoints[axis_name] = one_sided_summary(
            part["topology_convergence_margin"].to_numpy(float),
            int.from_bytes(hashlib.sha256(f"C4/{axis_name}".encode()).digest()[:8], "little"),
        )
    topology_adjusted = holm_adjust({key: value["p_one_sided"] for key, value in topology_endpoints.items()})
    for key, value in topology_endpoints.items():
        value["p_holm"] = topology_adjusted[key]
        value["status"] = "SUPPORTED" if value["median"] > 0 and value["p_holm"] < 0.05 else "UNSUPPORTED"
    topology_summary = {
        "endpoints": topology_endpoints,
        "patient_median_real_arm_pair_cosine": {
            axis_name: finite_median(
                topology.loc[topology["perturbation_axis"].eq(axis_name), "real_arm_pair_cosine"]
            ) for axis_name in ("PROGRESS", "FIELD")
        },
        "patient_median_real_arm_to_C_suffix_cosine": {
            axis_name: finite_median(
                topology.loc[topology["perturbation_axis"].eq(axis_name), "real_arm_to_C_suffix_cosine"]
            ) for axis_name in ("PROGRESS", "FIELD")
        },
        "C4_status": "SUPPORTED" if all(value["status"] == "SUPPORTED" for value in topology_endpoints.values()) else "UNSUPPORTED",
    }

    chord_summary = {}
    chord_patient_contrasts = pd.DataFrame()
    if len(chord_frame):
        collapsed = chord_frame.groupby(
            ["patient", "fit_id", "public_arm", "seed", "family"], as_index=False
        )[["oriented_open_field_response", "oriented_closed_field_response", "oriented_terminal_field_response"]].mean()
        collapsed = collapsed.groupby(["patient", "family"], as_index=False)[
            ["oriented_open_field_response", "oriented_closed_field_response", "oriented_terminal_field_response"]
        ].median()
        for family, part in collapsed.groupby("family"):
            chord_summary[family] = {
                endpoint: one_sided_summary(
                    part[endpoint].to_numpy(float),
                    int.from_bytes(hashlib.sha256(f"CHORD/{family}/{endpoint}".encode()).digest()[:8], "little"),
                )
                for endpoint in (
                    "oriented_open_field_response", "oriented_closed_field_response",
                    "oriented_terminal_field_response",
                )
            }
        pivot = collapsed.pivot(index="patient", columns="family")
        endpoints = (
            "oriented_open_field_response", "oriented_closed_field_response",
            "oriented_terminal_field_response",
        )
        contrast_rows = []
        for patient in pivot.index:
            if not all((endpoint, family) in pivot.columns for endpoint in endpoints for family in ("HIGH_U", "SMALL_U")):
                continue
            contrast_rows.append({
                "patient": patient,
                **{
                    f"{endpoint}_high_minus_small": float(
                        pivot.loc[patient, (endpoint, "HIGH_U")]
                        - pivot.loc[patient, (endpoint, "SMALL_U")]
                    ) for endpoint in endpoints
                },
            })
        chord_patient_contrasts = pd.DataFrame(contrast_rows)
        if len(chord_patient_contrasts):
            chord_summary["HIGH_MINUS_SMALL"] = {
                endpoint: one_sided_summary(
                    chord_patient_contrasts[f"{endpoint}_high_minus_small"].to_numpy(float),
                    int.from_bytes(hashlib.sha256(
                        f"CHORD/HIGH_MINUS_SMALL/{endpoint}".encode()
                    ).digest()[:8], "little"),
                ) for endpoint in endpoints
            }

    payload = {
        "contract": "topic5_axis_perturbation_C3_C4_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_revision": ANALYSIS_REVISION, "perturbation_revision": PERTURB_REVISION,
        "status": "COMPLETE", "C3_tiers": tiers, "C4_topology": topology_summary,
        "empirical_chords": chord_summary,
        "primary_response": "OPEN_LOOP_TAU_1_TO_3_CONTINUOUS_PRE_MASK_CONTACT_LOGITS",
        "closed_loop_role": "DIRECTIONAL_CONSISTENCY_AND_TERMINAL_SECONDARY",
        "aggregation_order": ["tau", "event", "phase", "seed", "fit", "arm", "patient"],
        "target_values_read": False,
    }
    atomic_write_csv(PERTURB / "C3_CELL_PHASE_RESPONSE.csv", cell_phase)
    atomic_write_csv(PERTURB / "C3_PATIENT_EFFECTS.csv", patients)
    atomic_write_csv(PERTURB / "C3_PATIENT_ARM_EFFECTS.csv", arms)
    atomic_write_csv(PERTURB / "FUNCTIONAL_RESPONSE_FIELDS.csv", fields)
    atomic_write_csv(PERTURB / "EMPIRICAL_CHORD_CELL_PHASE.csv", chord_frame)
    atomic_write_csv(PERTURB / "EMPIRICAL_CHORD_PATIENT_CONTRASTS.csv", chord_patient_contrasts)
    atomic_write_csv(PERTURB / "CONTROL_FAMILY_CELL_PHASE.csv", control_frame)
    atomic_write_csv(PERTURB / "C4_TOPOLOGY_FIELD_EFFECTS.csv", topology)
    atomic_write_json(PERTURB / "PERTURBATION_CLAIM_SUMMARY.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
