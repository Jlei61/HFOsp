#!/usr/bin/env python3
"""How much of the "patient-trained axis" is just the shape of the implantation?

The aligned basis is built from an axis estimated as the principal eigenvector of
the split-0 start-to-late-field displacement outer product.  That statistic
measures the direction of largest displacement *spread*, which for an elongated
contact cloud is the cloud's own long axis whether or not anything propagates
along it.  This diagnostic states, per patient, how close the trained axis lands
to the contact-cloud principal axis and to the dominant shaft axis, next to how
elongated that patient's implantation is.

It is a description of the estimator, not a correction to it: nothing downstream
is re-weighted or excluded on the basis of this file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/topic5_capacity_constrained_history_motif_v0_2"
FRAME_ROOT = ROOT / "results/topic5_dynamical_motif_rnn_v0_1/frame_cache/GEOMETRY_ONLY_PCA2"


def undirected_gap(first: float | None, second: float | None) -> float:
    if first is None or second is None:
        return float("nan")
    gap = abs(first - second) % np.pi
    return float(min(gap, np.pi - gap))


def main() -> int:
    audit = json.loads((RESULT_ROOT / "basis" / "NULL_MATCH_AUDIT.json").read_text())
    rows = []
    for patient, entry in audit["per_patient"].items():
        provenance = json.loads((FRAME_ROOT / patient / "provenance.json").read_text())
        layout = provenance["layout_axes_in_frame"]
        singular = provenance["singular_values_3d"]
        for fraction, axis_entry in entry["axis_by_fraction"].items():
            axis = np.asarray(axis_entry["axis"], dtype=float)
            theta = float(np.arctan2(axis[1], axis[0]) % np.pi)
            rows.append({
                "patient": patient,
                "basis_fraction": int(fraction),
                "n_contacts": entry["n_contacts"],
                "n_shafts": entry["n_shafts"],
                "geometry_class": entry["geometry_class"],
                "cloud_aspect_2d": float(singular[0] / max(singular[1], 1e-9)),
                "trained_axis_theta_rad": theta,
                "axis_anisotropy_ratio": axis_entry["anisotropy_ratio"],
                "n_basis_events": axis_entry["n_displacements"],
                "gap_to_contact_cloud_axis_deg": np.degrees(undirected_gap(
                    theta, layout["contact_cloud_pca1"]["theta_rad"]
                    if layout["contact_cloud_pca1"]["estimable"] else None)),
                "gap_to_dominant_shaft_axis_deg": np.degrees(undirected_gap(
                    theta, layout["dominant_shaft"]["theta_rad"]
                    if layout["dominant_shaft"]["estimable"] else None)),
            })
    table = pd.DataFrame(rows)
    table.to_csv(RESULT_ROOT / "PER_PATIENT_AXIS_VS_IMPLANTATION.csv", index=False)

    full = table[table["basis_fraction"] == 100]
    cloud = full["gap_to_contact_cloud_axis_deg"].dropna()
    shaft = full["gap_to_dominant_shaft_axis_deg"].dropna()
    stability = full.merge(
        table[table["basis_fraction"] == 25][["patient", "trained_axis_theta_rad"]],
        on="patient", suffixes=("", "_25"))
    stability["gap_100_vs_25_deg"] = [
        np.degrees(undirected_gap(a, b)) for a, b in
        zip(stability["trained_axis_theta_rad"], stability["trained_axis_theta_rad_25"])]

    summary = {
        "contract": "topic5_capacity_constrained_history_motif_v0_2_axis_vs_implantation",
        "what_this_measures": "the undirected angle between the axis the aligned basis was "
                              "built from and two purely geometric axes of the same "
                              "implantation; 0 deg means the trained axis carries nothing "
                              "beyond where the electrodes were placed",
        "n_patients": int(len(full)),
        "gap_to_contact_cloud_axis_deg": {
            "median": float(cloud.median()), "min": float(cloud.min()), "max": float(cloud.max()),
            "n_within_20_deg": int((cloud < 20).sum()), "n": int(len(cloud))},
        "gap_to_dominant_shaft_axis_deg": {
            "median": float(shaft.median()), "min": float(shaft.min()), "max": float(shaft.max()),
            "n_within_20_deg": int((shaft < 20).sum()), "n": int(len(shaft))},
        "spearman_aspect_vs_gap_to_cloud": float(
            pd.Series(full["cloud_aspect_2d"]).corr(full["gap_to_contact_cloud_axis_deg"],
                                                    method="spearman")),
        "axis_stability_100_vs_25_percent_deg": {
            "median": float(stability["gap_100_vs_25_deg"].median()),
            "max": float(stability["gap_100_vs_25_deg"].max())},
        "reading": "a small gap does not by itself invalidate the aligned arm — the "
                   "direction-rotated null is still matched on kernel, anisotropy strength, "
                   "rank and parameter count — but it does mean the aligned-versus-rotated "
                   "contrast is largely a contrast between the implantation's long axis and "
                   "a rotation of it, and it must be reported that way",
    }
    (RESULT_ROOT / "AXIS_VS_IMPLANTATION_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
