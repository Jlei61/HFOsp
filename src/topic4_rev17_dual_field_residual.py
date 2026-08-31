"""Continuous residual coordinates around the accepted dual-field Node anchor."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

import numpy as np

from src.topic4_node_field_search import cosine_sheet_residuals, residual_candidate


FORMULA = (
    "delta_vtheta_i=-h_mean_i*mu_mean"
    "-h_dispersion_i*(d_i-mu_dispersion)"
)


def mapping_sha256(mean_hash: str, dispersion_hash: str) -> str:
    payload = {
        "formula": FORMULA,
        "mean_field_sha256": str(mean_hash),
        "dispersion_field_sha256": str(dispersion_hash),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def accepted_anchor(exact_reconstruction: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the accepted mean and dispersion fields without changing semantics."""
    required = {"node_field", "node_dispersion_field", "node_mapping"}
    if not required.issubset(exact_reconstruction):
        raise ValueError("exact reconstruction lacks the accepted dual Node mapping")
    source_mapping = exact_reconstruction["node_mapping"]
    if source_mapping.get("mapping_type") != "dual_continuous_mean_dispersion":
        raise ValueError("accepted Node anchor is not a dual continuous mapping")
    mean_field = copy.deepcopy(exact_reconstruction["node_field"])
    dispersion_field = copy.deepcopy(exact_reconstruction["node_dispersion_field"])
    for name, field in (("mean", mean_field), ("dispersion", dispersion_field)):
        if field.get("field_type") != "spline_continuous":
            raise ValueError(f"accepted {name} field is not spline_continuous")
        values = np.asarray(field.get("coefficients"), dtype=np.float64)
        expected = (int(field["n_basis"]), int(field["n_basis"]))
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"accepted {name} field coefficients are invalid")
    mapping = {
        "mapping_type": "dual_continuous_mean_dispersion",
        "signed_depth_shrinkage": 1.0,
        "node_gain": 1.0,
        "mapping_sha256": mapping_sha256(
            mean_field["field_sha256"], dispersion_field["field_sha256"],
        ),
    }
    return {
        "candidate_id": "exact_dual_anchor",
        "role": "accepted_dual_continuous_node_anchor_not_selectable",
        "selection_eligible": False,
        "source_candidate_ids": copy.deepcopy(
            exact_reconstruction.get("source_candidate_ids", {})
        ),
        "node_field": mean_field,
        "node_dispersion_field": dispersion_field,
        "node_mapping": mapping,
        "residual_coordinates": None,
        "pathways": {
            "learned_E_to_E_redistribution": "off",
            "learned_E_to_I_redistribution": "off",
            "Z_M": "off",
        },
    }


def build_residual_atlas(
    exact_reconstruction: Mapping[str, Any], *, maximum_frequency: int,
    amplitude: float, target_n_basis: int = 18, degree: int = 3,
    sheet_mm: float = 20.0, projection_grid_per_axis: int = 61,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Perturb one dual-field channel at a time around an exact zero residual."""
    amplitude = float(amplitude)
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("residual amplitude must be finite and positive")
    anchor = accepted_anchor(exact_reconstruction)
    mean_anchor = anchor["node_field"]
    dispersion_anchor = anchor["node_dispersion_field"]
    for field in (mean_anchor, dispersion_anchor):
        if int(field["n_basis"]) != int(target_n_basis) or int(field["degree"]) != int(degree):
            raise ValueError("residual basis and accepted spline basis do not align")
    basis = cosine_sheet_residuals(
        maximum_frequency=int(maximum_frequency),
        target_n_basis=int(target_n_basis), degree=int(degree),
        sheet_mm=float(sheet_mm),
        projection_grid_per_axis=int(projection_grid_per_axis),
    )
    candidates = [anchor]
    audit_rows = []
    for row in basis["rows"]:
        mode = int(row["mode_index"])
        residual = np.asarray(row["coefficients"], dtype=np.float64)
        for channel in ("mean", "dispersion"):
            for sign in (-1, 1):
                token = "m" if sign < 0 else "p"
                candidate_id = f"{channel}_f{mode:02d}_{token}_a{round(100 * amplitude):02d}"
                mean_field = copy.deepcopy(mean_anchor)
                dispersion_field = copy.deepcopy(dispersion_anchor)
                source = mean_anchor if channel == "mean" else dispersion_anchor
                perturbed = residual_candidate(
                    source, residual, amplitude=sign * amplitude,
                    candidate_id=f"{candidate_id}_{channel}",
                    residual_index=mode,
                    coarse_n_basis=int(maximum_frequency) + 1,
                )
                perturbed["role"] = f"dual_node_{channel}_low_frequency_residual"
                perturbed["residual_coordinates"].update({
                    "basis_family": "uniform_sheet_cosine",
                    "channel": channel,
                    "kx": int(row["kx"]), "ky": int(row["ky"]),
                    "orientation": int(sign),
                    "observation_coordinates_used": False,
                    "manual_geometry_used": False,
                })
                if channel == "mean":
                    mean_field = perturbed
                else:
                    dispersion_field = perturbed
                mapping_hash = mapping_sha256(
                    mean_field["field_sha256"], dispersion_field["field_sha256"],
                )
                candidate = {
                    "candidate_id": candidate_id,
                    "role": "dual_continuous_node_local_response_coordinate",
                    "selection_eligible": True,
                    "source_candidate_ids": copy.deepcopy(
                        anchor["source_candidate_ids"]
                    ),
                    "node_field": mean_field,
                    "node_dispersion_field": dispersion_field,
                    "node_mapping": {
                        "mapping_type": "dual_continuous_mean_dispersion",
                        "signed_depth_shrinkage": 1.0,
                        "node_gain": 1.0,
                        "mapping_sha256": mapping_hash,
                    },
                    "residual_coordinates": {
                        "channel": channel, "mode_index": mode,
                        "kx": int(row["kx"]), "ky": int(row["ky"]),
                        "orientation": int(sign),
                        "signed_log_surface_rms": float(sign * amplitude),
                        "zero_is_exact_dual_anchor": True,
                    },
                    "pathways": copy.deepcopy(anchor["pathways"]),
                }
                candidates.append(candidate)
                audit_rows.append({
                    "candidate_id": candidate_id,
                    **candidate["residual_coordinates"],
                    "mean_field_sha256": mean_field["field_sha256"],
                    "dispersion_field_sha256": dispersion_field["field_sha256"],
                    "mapping_sha256": mapping_hash,
                })
    expected = 1 + 4 * int(basis["n_modes"])
    if len(candidates) != expected:
        raise RuntimeError("dual-field residual atlas size changed")
    if len({row["node_mapping"]["mapping_sha256"] for row in candidates}) != len(candidates):
        raise RuntimeError("dual-field residual atlas contains duplicate mappings")
    return candidates, {
        "formula": FORMULA,
        "maximum_frequency": int(maximum_frequency),
        "amplitude": amplitude,
        "basis_mode_count": int(basis["n_modes"]),
        "candidate_count": len(candidates),
        "maximum_absolute_gram_error": float(basis["maximum_absolute_gram_error"]),
        "observation_coordinates_used": False,
        "zero_residual_exact": True,
        "coordinates": audit_rows,
    }
