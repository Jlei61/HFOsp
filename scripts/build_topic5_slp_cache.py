"""Milestone C: build the per-patient spatial cache for the SLP-RNN.

One directory per patient holding everything a run needs and nothing it does
not: the contact plane in the frozen intersection order, the latent node
positions, the observation operator, and the rank events already split
chronologically inside train80.

Ictal targets, A/B labels, SOZ and SNN parameters never enter this cache.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.topic5_shared_propagation_field import load_subject_rank_events, sha256_file
from src.topic5_virtual_seeg_operator import (
    MIN_NODES_PER_CONTACT,
    kernel_sigma_mm,
    nearest_node,
    resolve_node_count,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
OUT_ROOT = ROOT / "results/topic5_spatial_latent_propagation_rnn_v0_1"
NODE_SEED = 20260806

# An event needs at least one transition to teach anything about propagation.
MIN_RANKS_PER_EVENT = 2


def densify_ranks(group_ids: np.ndarray) -> np.ndarray:
    """Re-number ranks to 0..T-1 per event after contacts have been dropped.

    Dropping a contact can empty a whole rank set.  Leaving the gap would make
    the model predict an empty set at that step, which is not a rank set of the
    patient's event -- it is an artefact of the montage intersection.
    """
    out = np.full(group_ids.shape, -1, dtype=np.int16)
    for e, row in enumerate(group_ids):
        present = np.unique(row[row >= 0])
        remap = {int(old): new for new, old in enumerate(present)}
        for c, value in enumerate(row):
            if value >= 0:
                out[e, c] = remap[int(value)]
    return out


def build_patient(subject: str, manifest: Dict[str, Any], out_root: Path) -> Dict[str, Any]:
    tree = manifest["primary_geometry_tree"]
    entry = next(e for e in manifest["subjects"] if e["subject"] == subject)
    geometry = entry["geometry"][tree]
    joint = list(geometry["joint_contacts"])

    record = load_subject_rank_events(DATASET_DIR, subject)
    event_names = [str(n) for n in record.contact_names]
    columns = np.array([event_names.index(name) for name in joint], int)

    plane_payload = json.loads((ROOT / geometry["path"]).read_text())
    by_name = {str(c["name"]): c for c in plane_payload["channels"]}
    xy = np.array(
        [[by_name[n]["along_axis_mm"], by_name[n]["signed_transverse_mm"]] for n in joint],
        float,
    )
    xyz_raw = [by_name[n].get("coord_mm") for n in joint]
    xyz = np.array(xyz_raw, float) if all(v is not None for v in xyz_raw) else None
    is_soz = np.array([bool(by_name[n]["is_soz"]) for n in joint], bool)

    ranks = densify_ranks(record.group_ids[:, columns])
    n_ranks = np.array([len(np.unique(r[r >= 0])) for r in ranks], int)
    keep = n_ranks >= MIN_RANKS_PER_EVENT

    train, validation, test = record.development_split(0.15, 0.15)
    split = np.full(len(ranks), -1, np.int8)
    split[train] = 0
    split[validation] = 1
    split[test] = 2
    split[~keep] = -1  # dropped events belong to no partition

    sigma = kernel_sigma_mm(xy)
    n_nodes, nodes, H, nominal_nodes = resolve_node_count(xy, sigma, seed=NODE_SEED)
    anchors = nearest_node(xy, nodes)

    out_dir = out_root / "cache" / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "plane_coordinates.npz",
        contact_names=np.array(joint, dtype=object),
        xy_mm=xy,
        xyz_mm=xyz if xyz is not None else np.zeros((0, 3)),
        xyz_available=np.array([xyz is not None]),
        is_soz=is_soz,
        sigma_mm=np.array([sigma]),
    )
    np.savez_compressed(
        out_dir / "latent_nodes.npz",
        nodes_xy=nodes,
        seed=np.array([NODE_SEED]),
        contact_anchor_node=anchors,
    )
    np.savez_compressed(out_dir / "seeg_operator.npz", H=H, sigma_mm=np.array([sigma]))
    np.savez_compressed(
        out_dir / "events.npz",
        group_ids=ranks,
        split=split,
        n_ranks=n_ranks,
    )

    provenance = {
        "subject": subject,
        "dataset": record.dataset,
        "geometry_tree": tree,
        "geometry_path": geometry["path"],
        "geometry_sha256": geometry["sha256"],
        "event_input_sha256": record.input_sha256,
        "target_values_read": bool(record.target_values_read),
        "geometry_status": manifest["geometry_status"],
        "contact_names": joint,
        "n_contacts": len(joint),
        "n_latent_nodes": int(n_nodes),
        "n_latent_nodes_spec_nominal": int(nominal_nodes),
        "node_count_raised_for_locality": bool(n_nodes > nominal_nodes),
        "min_nodes_per_contact_required": MIN_NODES_PER_CONTACT,
        "node_seed": NODE_SEED,
        "sigma_mm": float(sigma),
        "n_events_total": int(len(ranks)),
        "n_events_dropped_below_min_ranks": int(np.sum(~keep)),
        "n_train": int(np.sum(split == 0)),
        "n_validation": int(np.sum(split == 1)),
        "n_test": int(np.sum(split == 2)),
        "split_rule": "development_split(0.15, 0.15) inside train80; old heldout20 burned",
        "H_row_sum_max_error": float(np.abs(H.sum(axis=1) - 1.0).max()),
        "H_nonzero_fraction": float(np.mean(H > 0)),
        "min_nodes_seen_per_contact": int((H > 0).sum(axis=1).min()),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=1))
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()

    manifest = json.loads((args.out_root / "INPUT_MANIFEST.json").read_text())
    subjects = args.subjects or manifest["frozen_cohort"]["primary"]

    rows = []
    for subject in subjects:
        rows.append(build_patient(subject, manifest, args.out_root))
        p = rows[-1]
        print(
            f"{subject:26s} C={p['n_contacts']:3d} M={p['n_latent_nodes']:3d} "
            f"sigma={p['sigma_mm']:5.2f}mm nodes/contact>={p['min_nodes_seen_per_contact']:3d} "
            f"events {p['n_train']}/{p['n_validation']}/{p['n_test']} "
            f"dropped={p['n_events_dropped_below_min_ranks']}"
        )

    summary = {
        "contract": "topic5_slp_cache_v0_1",
        "node_seed": NODE_SEED,
        "min_ranks_per_event": MIN_RANKS_PER_EVENT,
        "n_patients": len(rows),
        "patients": rows,
    }
    (args.out_root / "cache" / "CACHE_SUMMARY.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {(args.out_root / 'cache' / 'CACHE_SUMMARY.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
