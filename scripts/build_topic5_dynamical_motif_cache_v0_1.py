#!/usr/bin/env python3
"""Phase A for Topic 5.2 dynamical motif RNN v0.1-r2.

Builds the two frames the design experiment compares and writes the minimal
integrity audits.  Nothing here reads TA/TB template values, event suffixes,
seizure energy or any model output.

* ``PARENT_FROZEN_FRAME``  reuses the v0.5 multiscale cache bit for bit.
* ``GEOMETRY_ONLY_PCA2``   re-derives the plane from frozen 3-D contact
  coordinates alone (PCA2, singular-value order, sign from the largest
  absolute 3-D loading) and rebuilds mesh / ``H`` / distances / local support
  with the frozen v0.5 rules.

Dual-view patients (``own_a``/``own_b``) share one event set and one contact
set, so the geometry frame collapses them into a single patient-level fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CANONICAL_ROOT = ROOT.parents[1] if (ROOT.parents[1] / "results").exists() else ROOT

from src.topic5_lbss_rnn_v0_2 import build_pool_contract, strong_component_audit  # noqa: E402
from src.topic5_virtual_seeg_operator import (  # noqa: E402
    kernel_sigma_mm,
    resolve_full_tissue_layout,
)

PARENT_ROOT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
PARENT_CACHE = PARENT_ROOT / "cache"
DATASET_ROOT = CANONICAL_ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4/per_subject"
FIELD_ROOT = CANONICAL_ROOT / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
OUT_ROOT = ROOT / "results/topic5_dynamical_motif_rnn_v0_1"

# Frozen from the v0.5 producer; changing any of these makes a different mesh.
NODE_SEED = 20260812
SIGMA_FLOOR_MM = 0.0
ONE_D_RATIO = 0.05


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value)).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonable) + "\n")
    temporary.replace(path)


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")


def geometry_only_basis(coords_3d: np.ndarray) -> dict:
    """PCA2 of the frozen contact cloud with a deterministic, target-free sign.

    Component order follows singular values.  The sign of each component is
    fixed by the coordinate with the largest absolute loading, so no rank,
    template, suffix, seizure value or model output touches the frame.
    """
    points = np.asarray(coords_3d, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("geometry frame needs at least three 3-D contacts")
    origin = points.mean(axis=0)
    centred = points - origin
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    basis = []
    for component in range(2):
        vector = np.asarray(vt[component], dtype=float)
        dominant = int(np.argmax(np.abs(vector)))
        if vector[dominant] < 0:
            vector = -vector
        basis.append(vector / np.linalg.norm(vector))
    u, w = basis
    # Re-orthogonalise defensively; SVD already gives orthonormal rows.
    w = w - (w @ u) * u
    w = w / np.linalg.norm(w)
    ratio_21 = float(singular[1] / singular[0]) if singular[0] > 0 else 0.0
    ratio_31 = float(singular[2] / singular[0]) if singular[0] > 0 else 0.0
    return {
        "origin": origin,
        "u": u,
        "w": w,
        "singular_values": singular,
        "ratio_second_to_first": ratio_21,
        "ratio_third_to_first": ratio_31,
        "geometry_class": (
            "TWO_DIMENSIONAL" if ratio_21 >= ONE_D_RATIO else "DEGENERATE_ONE_DIMENSIONAL"
        ),
    }


def build_geometry_frame(contacts_xy_mm: np.ndarray) -> dict:
    """Frozen v0.5 mesh rule applied to an arbitrary 2-D contact layout."""
    contacts = np.asarray(contacts_xy_mm, dtype=np.float32).astype(float)
    sigma = float(np.float32(kernel_sigma_mm(contacts, floor_mm=SIGMA_FLOOR_MM)))
    layout = resolve_full_tissue_layout(contacts, sigma, seed=NODE_SEED)
    nodes = np.asarray(layout.nodes_xy, dtype=float)
    H = np.asarray(layout.H, dtype=float)
    distance = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=-1)
    pools = build_pool_contract(distance)
    supported = H.sum(axis=0) > 1e-12
    graph = strong_component_audit(pools.local_mask, supported=supported)
    return {
        "contacts_xy_mm": contacts,
        "nodes_xy_mm": nodes,
        "H": H,
        "D_mm": distance,
        "sigma_mm": sigma,
        "pools": pools,
        "graph": graph,
        "n_zero_h_nodes": int(layout.n_zero_h_nodes),
        "zero_h_fraction": float(layout.zero_h_fraction),
    }


def dominant_shaft_axis(coords_3d: np.ndarray, shafts: list[str]) -> tuple[np.ndarray, str, int]:
    """Longest shaft by effective contact count; ties break on shaft name."""
    labels = [str(value) for value in shafts]
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    best = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    members = np.asarray([index for index, label in enumerate(labels) if label == best], dtype=int)
    if members.size < 2:
        return np.full(3, np.nan), best, int(members.size)
    points = np.asarray(coords_3d, float)[members]
    centred = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    vector = np.asarray(vt[0], dtype=float)
    dominant = int(np.argmax(np.abs(vector)))
    if vector[dominant] < 0:
        vector = -vector
    return vector / np.linalg.norm(vector), best, int(members.size)


def load_patient_inputs(subject: str, fit_ids: list[str]) -> dict:
    """Frozen contacts, 3-D coordinates and event arrays shared by every view."""
    reference = fit_ids[0]
    raw = np.load(PARENT_CACHE / reference / "events_raw.npz", allow_pickle=True)
    contact_names = [str(value) for value in raw["contact_names"]]
    dataset = np.load(DATASET_ROOT / f"{subject}.npz", allow_pickle=False)
    dataset_names = [str(value) for value in dataset["contact_names"]]
    columns = [dataset_names.index(name) for name in contact_names]
    coords_3d = np.asarray(dataset["contact_coords"], dtype=float)[columns]
    field = json.loads((FIELD_ROOT / f"{subject}.json").read_text())
    field_names = [str(value) for value in field["names"]]
    field_columns = [field_names.index(name) for name in contact_names]
    field_coords = np.asarray(field["coords"], dtype=float)[field_columns]
    if not np.allclose(coords_3d, field_coords, atol=1e-3):
        raise RuntimeError(f"{subject}: dataset and template-field 3-D coordinates disagree")
    shafts = [str(field["shafts"][index]) for index in field_columns]
    events = np.load(PARENT_CACHE / reference / "events.npz", allow_pickle=True)
    modes = np.load(PARENT_CACHE / reference / "train_only_modes.npz", allow_pickle=False)
    return {
        "contact_names": contact_names,
        "coords_3d": coords_3d,
        "shafts": shafts,
        "ranks": np.asarray(raw["ranks"]),
        "split": np.asarray(events["split"]),
        "event_group_count": np.asarray(raw["event_group_count"]),
        "event_abs_time": np.asarray(raw["event_abs_time"]),
        "event_source_index": np.asarray(raw["event_source_index"]),
        "event_lag_raw": np.asarray(raw["event_lag_raw"]),
        "parent_event_split": np.asarray(dataset["event_split"]),
        "train_only_modes": {key: np.asarray(modes[key]) for key in modes.files},
        "prefix_posterior": np.asarray(events["prefix_posterior"]),
        "full_train_mode": np.asarray(events["full_train_mode"]),
        "prefix_mode": np.asarray(events["mode"]),
        "events_raw_sha256": sha256_file(PARENT_CACHE / reference / "events_raw.npz"),
        "events_sha256": sha256_file(PARENT_CACHE / reference / "events.npz"),
    }


def view_consistency(subject: str, fit_ids: list[str]) -> dict:
    """Dual views must be two planes on one event set, not two datasets."""
    digests = {"events_raw": set(), "events": set(), "contacts": set(), "n_events": set()}
    for fit_id in fit_ids:
        digests["events_raw"].add(sha256_file(PARENT_CACHE / fit_id / "events_raw.npz"))
        digests["events"].add(sha256_file(PARENT_CACHE / fit_id / "events.npz"))
        raw = np.load(PARENT_CACHE / fit_id / "events_raw.npz", allow_pickle=True)
        digests["contacts"].add(tuple(str(value) for value in raw["contact_names"]))
        digests["n_events"].add(int(raw["ranks"].shape[0]))
    return {
        "subject": subject,
        "views": fit_ids,
        "n_views": len(fit_ids),
        "identical_events_raw": len(digests["events_raw"]) == 1,
        "identical_events": len(digests["events"]) == 1,
        "identical_contacts": len(digests["contacts"]) == 1,
        "identical_n_events": len(digests["n_events"]) == 1,
    }


def split_audit(inputs: dict) -> dict:
    """``split == -1`` must be exactly the parent held-out events."""
    split = inputs["split"]
    parent_heldout = inputs["parent_event_split"] == 1
    rank_ineligible = inputs["event_group_count"] < 2
    model_unseen = split == -1
    return {
        "n_events": int(split.size),
        "n_train": int((split == 0).sum()),
        "n_calibration": int((split == 1).sum()),
        "n_development_test": int((split == 2).sum()),
        "n_model_unseen": int(model_unseen.sum()),
        "n_parent_heldout": int(parent_heldout.sum()),
        "n_rank_ineligible": int(rank_ineligible.sum()),
        "model_unseen_equals_parent_heldout": bool(np.array_equal(model_unseen, parent_heldout)),
        "model_unseen_disjoint_from_train": bool(not np.any(model_unseen & (split == 0))),
        "parent_event_split_sha256": sha256_array(inputs["parent_event_split"]),
        "split_sha256": sha256_array(split),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    out = args.out_root
    (out / "frame_cache").mkdir(parents=True, exist_ok=True)

    census = pd.read_csv(PARENT_ROOT / "FULL_PARENT_FIT_CENSUS.csv")
    by_subject: dict[str, list[str]] = {}
    for _, row in census.iterrows():
        by_subject.setdefault(str(row.subject), []).append(str(row.fit_id))

    view_rows, split_rows, geometry_rows, errors = [], [], [], []
    for subject in sorted(by_subject):
        fit_ids = sorted(by_subject[subject])
        consistency = view_consistency(subject, fit_ids)
        view_rows.append(consistency)
        if not all(
            consistency[key]
            for key in ("identical_events_raw", "identical_events", "identical_contacts")
        ):
            errors.append(f"{subject}: dual views do not share one event/contact set")
            continue
        inputs = load_patient_inputs(subject, fit_ids)
        audit = split_audit(inputs)
        audit["subject"] = subject
        split_rows.append(audit)
        if not audit["model_unseen_equals_parent_heldout"]:
            errors.append(f"{subject}: split -1 is not exactly the parent held-out set")

        basis = geometry_only_basis(inputs["coords_3d"])
        contacts_xy = (inputs["coords_3d"] - basis["origin"]) @ np.stack([basis["u"], basis["w"]], axis=1)
        frame = build_geometry_frame(contacts_xy)
        shaft_axis, shaft_name, shaft_n = dominant_shaft_axis(inputs["coords_3d"], inputs["shafts"])
        layout_axes = {
            "contact_cloud_pca1_3d": basis["u"],
            "dominant_shaft_3d": shaft_axis,
            "dominant_shaft_name": shaft_name,
            "dominant_shaft_n_contacts": shaft_n,
        }
        projected = {}
        for name, vector in (("contact_cloud_pca1", basis["u"]), ("dominant_shaft", shaft_axis)):
            if not np.all(np.isfinite(vector)):
                projected[name] = {"estimable": False, "theta_rad": None, "in_plane_norm": None}
                continue
            planar = np.array([vector @ basis["u"], vector @ basis["w"]], dtype=float)
            norm = float(np.linalg.norm(planar))
            projected[name] = {
                "estimable": bool(norm > 0.2),
                "theta_rad": float(np.arctan2(planar[1], planar[0]) % np.pi) if norm > 1e-9 else None,
                "in_plane_norm": norm,
            }

        target = out / "frame_cache" / "GEOMETRY_ONLY_PCA2" / subject
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target / "plane.npz",
            contacts_xy_mm=frame["contacts_xy_mm"].astype(np.float32),
            nodes_xy_mm=frame["nodes_xy_mm"].astype(np.float32),
            H=frame["H"].astype(np.float32),
            D_mm=frame["D_mm"].astype(np.float32),
            sigma_mm=np.asarray([frame["sigma_mm"]], np.float32),
            local_mask=frame["pools"].local_mask.astype(np.uint8),
            coords_3d_mm=inputs["coords_3d"].astype(np.float32),
            basis_u=basis["u"].astype(np.float64),
            basis_w=basis["w"].astype(np.float64),
            basis_origin=basis["origin"].astype(np.float64),
            layout_pca1_3d=basis["u"].astype(np.float64),
            layout_shaft_3d=np.asarray(shaft_axis, dtype=np.float64),
            frame_version=np.asarray(["GEOMETRY_ONLY_PCA2_V0_1"]),
        )
        np.savez_compressed(
            target / "events.npz",
            ranks=inputs["ranks"].astype(np.int16),
            split=inputs["split"].astype(np.int8),
            event_group_count=inputs["event_group_count"].astype(np.int16),
            event_lag_raw=inputs["event_lag_raw"].astype(np.float32),
            event_abs_time=inputs["event_abs_time"].astype(np.float64),
            event_source_index=inputs["event_source_index"].astype(np.int64),
            prefix_posterior=inputs["prefix_posterior"].astype(np.float32),
            full_train_mode=inputs["full_train_mode"].astype(np.int8),
            prefix_mode=inputs["prefix_mode"].astype(np.int8),
            contact_names=np.asarray(inputs["contact_names"], dtype=str),
            shafts=np.asarray(inputs["shafts"], dtype=str),
        )
        write_json(target / "provenance.json", {
            "contract": "topic5_dynamical_motif_geometry_frame_v0_1",
            "frame": "GEOMETRY_ONLY_PCA2",
            "subject": subject,
            "parent_views": fit_ids,
            "n_contacts": len(inputs["contact_names"]),
            "contact_names": inputs["contact_names"],
            "shafts": inputs["shafts"],
            "n_events": int(inputs["ranks"].shape[0]),
            "n_nodes": int(frame["nodes_xy_mm"].shape[0]),
            "n_zero_h_nodes": frame["n_zero_h_nodes"],
            "zero_h_fraction": frame["zero_h_fraction"],
            "sigma_mm": frame["sigma_mm"],
            "local_edges": int(frame["pools"].local_mask.sum()),
            "r_local_mm": float(frame["pools"].r_local_mm),
            "k_neighbors": int(frame["pools"].k_neighbors),
            "graph": frame["graph"],
            "geometry_class": basis["geometry_class"],
            "singular_values_3d": basis["singular_values"],
            "ratio_second_to_first": basis["ratio_second_to_first"],
            "ratio_third_to_first": basis["ratio_third_to_first"],
            "layout_axes_3d": layout_axes,
            "layout_axes_in_frame": projected,
            "split_audit": audit,
            "events_raw_sha256": inputs["events_raw_sha256"],
            "events_sha256": inputs["events_sha256"],
            "node_seed": NODE_SEED,
            "sigma_floor_mm": SIGMA_FLOOR_MM,
            "target_values_read": False,
        })
        np.savez_compressed(target / "train_only_modes.npz", **inputs["train_only_modes"])

        parent_rows = census[census.subject == subject]
        geometry_rows.append({
            "subject": subject,
            "n_parent_views": len(fit_ids),
            "n_contacts": len(inputs["contact_names"]),
            "n_events": int(inputs["ranks"].shape[0]),
            "n_train": audit["n_train"],
            "n_calibration": audit["n_calibration"],
            "n_development_test": audit["n_development_test"],
            "n_model_unseen": audit["n_model_unseen"],
            "n_nodes": int(frame["nodes_xy_mm"].shape[0]),
            "parent_n_nodes_min": int(parent_rows.n_nodes.min()),
            "parent_n_nodes_max": int(parent_rows.n_nodes.max()),
            "n_zero_h_nodes": frame["n_zero_h_nodes"],
            "zero_h_fraction": frame["zero_h_fraction"],
            "parent_zero_h_fraction_min": float(parent_rows.zero_h_fraction.min()),
            "parent_zero_h_fraction_max": float(parent_rows.zero_h_fraction.max()),
            "local_edges": int(frame["pools"].local_mask.sum()),
            "r_local_mm": float(frame["pools"].r_local_mm),
            "sigma_mm": frame["sigma_mm"],
            "geometry_class": basis["geometry_class"],
            "ratio_second_to_first": basis["ratio_second_to_first"],
            "parent_geometry_class": str(parent_rows.geometry_class.iloc[0]),
            "strongly_connected": frame["graph"]["all_nodes_one_strong_component"],
            "contact_supported_reachability": frame["graph"][
                "contact_supported_pairwise_reachability"
            ],
            "layout_pca1_theta_rad": projected["contact_cloud_pca1"]["theta_rad"],
            "layout_shaft_theta_rad": projected["dominant_shaft"]["theta_rad"],
            "layout_shaft_in_plane_norm": projected["dominant_shaft"]["in_plane_norm"],
            "dominant_shaft": shaft_name,
            "dominant_shaft_n_contacts": shaft_n,
            "valid": bool(
                frame["graph"]["all_nodes_one_strong_component"]
                and frame["graph"]["contact_supported_pairwise_reachability"] == 1.0
                and frame["n_zero_h_nodes"] >= 1
            ),
        })
        print(f"[geometry] {subject}: {geometry_rows[-1]['n_nodes']} nodes, "
              f"{geometry_rows[-1]['local_edges']} local edges, "
              f"{basis['geometry_class']}", flush=True)

    pd.DataFrame(geometry_rows).to_csv(out / "GEOMETRY_ONLY_FIT_CENSUS.csv", index=False)
    pd.DataFrame(view_rows).to_csv(out / "PARENT_VIEW_CENSUS.csv", index=False)
    pd.DataFrame(split_rows).to_csv(out / "SPLIT_PROVENANCE_PER_PATIENT.csv", index=False)
    write_json(out / "SPLIT_PROVENANCE_AUDIT.json", {
        "contract": "topic5_dynamical_motif_split_audit_v0_1",
        "n_patients": len(split_rows),
        "all_model_unseen_equal_parent_heldout": bool(
            all(row["model_unseen_equals_parent_heldout"] for row in split_rows)
        ),
        "total_train": int(sum(row["n_train"] for row in split_rows)),
        "total_calibration": int(sum(row["n_calibration"] for row in split_rows)),
        "total_development_test": int(sum(row["n_development_test"] for row in split_rows)),
        "total_model_unseen": int(sum(row["n_model_unseen"] for row in split_rows)),
        "total_rank_ineligible": int(sum(row["n_rank_ineligible"] for row in split_rows)),
        "per_patient": split_rows,
    })
    write_json(out / "FRAME_FEASIBILITY.json", {
        "contract": "topic5_dynamical_motif_frame_feasibility_v0_1",
        "frames": ["PARENT_FROZEN_FRAME", "GEOMETRY_ONLY_PCA2"],
        "parent_fits": int(len(census)),
        "geometry_fits": len(geometry_rows),
        "geometry_all_valid": bool(all(row["valid"] for row in geometry_rows)),
        "geometry_one_dimensional": [
            row["subject"] for row in geometry_rows
            if row["geometry_class"] == "DEGENERATE_ONE_DIMENSIONAL"
        ],
        "errors": errors,
        "node_seed": NODE_SEED,
        "sigma_floor_mm": SIGMA_FLOOR_MM,
        "one_dimensional_ratio_threshold": ONE_D_RATIO,
    })
    print(f"\n[done] {len(geometry_rows)} geometry fits, {len(errors)} errors")
    for message in errors:
        print("  ERROR:", message)


if __name__ == "__main__":
    main()
