#!/usr/bin/env python
"""Aggregate two-seed projected modal/operator diagnostics."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.topic4_zm_modal_operator as MO  # noqa: E402


OUT = os.path.join(_ROOT, "results", "topic4_sef_hfo", "zm_branch_decision")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _primary_row(manifest):
    horizons = [float(value) for value in manifest["horizons_ms"]]
    energy = float(min(manifest["total_energy_mv2"]))
    horizon = float(horizons[0])
    key = f"h{horizon:g}_E{energy:g}".replace(".", "p")
    summary = (manifest.get("operator_summaries") or {}).get(key)
    if summary is None:
        return None
    path = os.path.join(_ROOT, manifest["operator_path"])
    with np.load(path, allow_pickle=False) as arrays:
        operator = np.asarray(arrays[f"{key}_operator"], float)
        right = np.asarray(arrays[f"{key}_leading_right_mode"])
        optimal = np.asarray(arrays[f"{key}_optimal_input_mode"])
    order = list(manifest["input_order"])
    axial = [order.index("axial_E"), order.index("axial_I")]
    transverse = [order.index("transverse_E"), order.index("transverse_I")]
    axial_basis = np.eye(len(order))[axial]
    pathology_axis = np.eye(len(order))[order.index("axial_E")]
    axial_gain = float(np.mean(np.linalg.norm(operator[:, axial], axis=0)))
    transverse_gain = float(
        np.mean(np.linalg.norm(operator[:, transverse], axis=0))
    )
    return {
        "seed": int(manifest["seed"]),
        "carrier_type": manifest["carrier_type"],
        "operator_tool": manifest["operator_tool"],
        "horizon_ms": horizon,
        "total_energy_mv2": energy,
        "spectral_radius": float(summary["spectral_radius"]),
        "spectral_abscissa_per_ms": float(
            summary["spectral_abscissa_per_ms"]
        ),
        "finite_time_gain": float(summary["finite_time_gain"]),
        "heldout_median_relative_error": float(
            summary["heldout_median_relative_error"]
        ),
        "heldout_pass": bool(summary["heldout_pass"]),
        "right_mode_angle_to_E_pathology_axis_deg": (
            MO.mode_axis_angle_deg(right, pathology_axis)
        ),
        "right_mode_angle_to_axial_EI_subspace_deg": (
            MO.mode_subspace_angle_deg(right, axial_basis)
        ),
        "optimal_input_angle_to_axial_EI_subspace_deg": (
            MO.mode_subspace_angle_deg(optimal, axial_basis)
        ),
        "axial_column_gain": axial_gain,
        "transverse_column_gain": transverse_gain,
        "axial_minus_transverse_gain": axial_gain - transverse_gain,
    }


def main():
    paths = sorted(
        glob.glob(os.path.join(OUT, "modal_operator", "seed*", "modal_probes.json"))
    )
    manifests = [json.load(open(path)) for path in paths]
    complete = [manifest for manifest in manifests if manifest.get("complete")]
    rows = []
    for manifest in complete:
        row = _primary_row(manifest)
        if row is not None:
            rows.append(row)
    types = sorted({row["carrier_type"] for row in rows})
    if len(rows) < 2:
        status = "insufficient_seeds"
    elif len(types) != 1:
        status = "carrier_type_disagreement"
    elif all(manifest.get("scientific_pass") for manifest in complete):
        status = "replicated_validated_operator"
    else:
        status = "replicated_but_nonlinear_or_unpredicted"
    numeric = [
        "spectral_radius",
        "spectral_abscissa_per_ms",
        "finite_time_gain",
        "heldout_median_relative_error",
        "right_mode_angle_to_E_pathology_axis_deg",
        "right_mode_angle_to_axial_EI_subspace_deg",
        "optimal_input_angle_to_axial_EI_subspace_deg",
        "axial_minus_transverse_gain",
    ]
    medians = {
        key: float(np.median([row[key] for row in rows]))
        for key in numeric
    } if rows else {}
    output = {
        "status": status,
        "n_complete_seeds": len(complete),
        "seeds": sorted(row["seed"] for row in rows),
        "carrier_types": types,
        "rows": rows,
        "median": medians,
        "inputs": [
            {
                "path": os.path.relpath(path, _ROOT),
                "sha256": _sha256(path),
            }
            for path in paths
        ],
        "modal_operator_version": MO.MODAL_OPERATOR_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "claim_boundary": (
            "projected local E/I-voltage operator and finite-time gain; "
            "explanatory only, not carrier identity or lifecycle evidence"
        ),
    }
    path = os.path.join(OUT, "modal_operator", "modal_operator_summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(output, handle, indent=2)
    os.replace(tmp, path)
    print(f"[modal-summary] status={status} seeds={output['seeds']} -> {path}")


if __name__ == "__main__":
    main()
