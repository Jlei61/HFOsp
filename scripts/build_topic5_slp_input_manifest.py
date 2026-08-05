"""Milestone A input manifest for the Topic 5 spatial latent propagation RNN.

The manifest is the provenance gate for every later run.  It answers three
questions the pasted plan assumed rather than checked:

1. which patients have BOTH a frozen rank-event record and a physical contact
   plane, under exact-name alignment (no fuzzy join);
2. where each patient's propagation axis came from, so a retrospective plane is
   never silently reported as a predictive one;
3. how much of each patient's contact spread survives the 2D projection.

Nothing here trains or reads an ictal target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.topic5_shared_propagation_field import load_subject_rank_events, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
GEOMETRY_TREES = {
    "narrow": ROOT / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects",
    "broad": ROOT / "results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects",
}
OUT_DIR = ROOT / "results/topic5_spatial_latent_propagation_rnn_v0_1"

# A patient enters the formal cohort only with enough jointly-resolved contacts
# to define a 2D field and a leave-one-contact-out test that still leaves a
# field behind.  Below this the observation operator has no interior.
MIN_JOINT_CONTACTS = 8


def _geometry_record(tree: Path, subject: str, template: str) -> Dict[str, Any] | None:
    path = tree / f"{subject}_t_{template}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if "channels" not in payload:
        # status-only stub: the plane was never solved for this patient.
        return {"path": path, "status": payload.get("status", "STUB"), "channels": None}
    return {"path": path, "status": "SOLVED", "payload": payload}


def _plane_contacts(payload: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for channel in payload["channels"]:
        coord = channel.get("coord_mm")
        out[str(channel["name"])] = {
            "along_axis_mm": float(channel["along_axis_mm"]),
            "signed_transverse_mm": float(channel["signed_transverse_mm"]),
            "x_norm": float(channel["x_norm"]),
            "y_norm": float(channel["y_norm"]),
            "support": float(channel["support"]),
            "is_soz": bool(channel["is_soz"]),
            "coord_mm": [float(v) for v in coord] if coord else None,
        }
    return out


def audit_subject(subject: str) -> Dict[str, Any]:
    record = load_subject_rank_events(DATASET_DIR, subject)
    event_contacts = [str(name) for name in record.contact_names]
    train, validation, test = record.development_split(0.15, 0.15)

    entry: Dict[str, Any] = {
        "subject": subject,
        "dataset": record.dataset,
        "input_sha256": record.input_sha256,
        "target_values_read": bool(record.target_values_read),
        "n_events": int(record.group_ids.shape[0]),
        "n_event_contacts": len(event_contacts),
        "event_contacts": event_contacts,
        "n_train80": int(len(record.train80_indices)),
        "n_old_heldout20_burned": int(len(record.old_heldout20_indices)),
        "development_split": {
            "n_train": int(len(train)),
            "n_validation": int(len(validation)),
            "n_test": int(len(test)),
            "rule": "chronological development_split(0.15, 0.15) inside train80",
            "old_heldout20_status": "BURNED_BY_EARLIER_RNN_DEVELOPMENT_NOT_USED_HERE",
        },
        "geometry": {},
    }

    for tree_name, tree in GEOMETRY_TREES.items():
        found = _geometry_record(tree, subject, "a")
        if found is None:
            entry["geometry"][tree_name] = {"status": "ABSENT"}
            continue
        if found.get("status") != "SOLVED":
            entry["geometry"][tree_name] = {"status": found["status"]}
            continue
        payload = found["payload"]
        plane = _plane_contacts(payload)
        joint = [name for name in event_contacts if name in plane]
        flags = payload.get("flags", {})
        pc1 = payload.get("transverse_pc1_variance_explained")
        entry["geometry"][tree_name] = {
            "status": "SOLVED",
            "path": str(found["path"].relative_to(ROOT)),
            "sha256": sha256_file(found["path"]),
            "n_plane_contacts": len(plane),
            "n_joint_contacts": len(joint),
            "joint_contacts": joint,
            "event_contacts_without_geometry": [
                n for n in event_contacts if n not in plane
            ],
            "axis_length_mm": float(payload["axis_length_mm"]),
            "transverse_pc1_variance_explained": (
                float(pc1) if pc1 is not None else None
            ),
            "projection_loss_fraction": (
                float(1.0 - pc1) if pc1 is not None else None
            ),
            "flags": flags,
            "sampling_geometry": payload.get("sampling_geometry"),
            "coord_mm_available": all(
                plane[n]["coord_mm"] is not None for n in joint
            ) if joint else False,
            "axis_provenance": "FULL_RECORD_AXIS",
            "eligible": bool(len(joint) >= MIN_JOINT_CONTACTS),
        }
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DIR / "INPUT_MANIFEST.json")
    args = parser.parse_args()

    manifest_path = DATASET_DIR / "dataset_manifest.json"
    dataset_manifest = json.loads(manifest_path.read_text())
    subjects = sorted(
        p.stem for p in (DATASET_DIR / "per_subject").glob("*.npz")
    )

    entries: List[Dict[str, Any]] = []
    for subject in subjects:
        entries.append(audit_subject(subject))

    cohorts: Dict[str, Any] = {}
    for tree_name in GEOMETRY_TREES:
        eligible = [
            e["subject"] for e in entries
            if e["geometry"].get(tree_name, {}).get("eligible")
        ]
        planar = [
            e["subject"] for e in entries
            if e["geometry"].get(tree_name, {}).get("eligible")
            and not e["geometry"][tree_name]["flags"].get("poor_planarity")
            and not e["geometry"][tree_name]["flags"].get("one_dimensional_sampling")
        ]
        joint_counts = [
            e["geometry"][tree_name]["n_joint_contacts"]
            for e in entries if e["geometry"].get(tree_name, {}).get("eligible")
        ]
        cohorts[tree_name] = {
            "n_eligible": len(eligible),
            "eligible_subjects": eligible,
            "n_planar_subset": len(planar),
            "planar_subjects": planar,
            "joint_contact_min": int(min(joint_counts)) if joint_counts else 0,
            "joint_contact_median": float(np.median(joint_counts)) if joint_counts else 0.0,
            "joint_contact_max": int(max(joint_counts)) if joint_counts else 0,
        }

    # One montage tree for the whole cohort.  Mixing trees inside a cohort
    # breaks the montage-consistency clause that src/sef_hfo_subject_placement.py
    # already enforces for the SNN placement.  `narrow` is chosen because it
    # strictly contains `broad` after exact-name intersection and has the higher
    # transverse PC1 on most patients.
    primary_tree = "narrow"
    primary = cohorts[primary_tree]
    by_subject = {e["subject"]: e for e in entries}
    well_sampled = [
        s for s in primary["eligible_subjects"]
        if by_subject[s]["n_events"] >= 2000
    ]

    payload = {
        "contract": "topic5_spatial_latent_propagation_rnn_v0_1_input_manifest",
        "primary_geometry_tree": primary_tree,
        "primary_geometry_tree_reason": (
            "broad is a strict subset of narrow after exact-name intersection "
            "(15 of 21), and narrow carries the higher transverse PC1 on most "
            "patients. One tree per cohort keeps montage consistency."
        ),
        "frozen_cohort": {
            "primary": primary["eligible_subjects"],
            "n_primary": primary["n_eligible"],
            "strata": {
                "planar": {
                    "subjects": primary["planar_subjects"],
                    "n": primary["n_planar_subset"],
                    "rule": "transverse PC1 >= 0.80 and not 1D sampling",
                },
                "well_sampled": {
                    "subjects": well_sampled,
                    "n": len(well_sampled),
                    "rule": "n_events >= 2000",
                    "reason": (
                        "a support stratum is pre-registered because an earlier "
                        "Topic 5 RNN comparison changed sign once low-support "
                        "patients were removed"
                    ),
                },
            },
        },
        "geometry_status": "RETROSPECTIVE_GEOMETRY_PILOT",
        "geometry_status_reason": (
            "Every solved contact plane was estimated from the full recording, not "
            "from train events only. Predictions on later events therefore sit on a "
            "retrospective substrate and must not be reported as prospective "
            "predictive geometry."
        ),
        "min_joint_contacts": MIN_JOINT_CONTACTS,
        "dataset": {
            "dir": str(DATASET_DIR.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "target_values_read": bool(dataset_manifest.get("target_values_read", True)),
            "ab_or_kmeans_labels_read": bool(
                dataset_manifest.get("ab_or_kmeans_labels_read", True)
            ),
            "n_subjects": len(subjects),
        },
        "cohorts": cohorts,
        "accepted_deviations_from_pasted_plan": [
            {
                "id": "D1",
                "claim_in_plan": "31 primary patients",
                "measured": "see cohorts.*.n_eligible; the plan's 31 is the "
                            "coordinate-free cohort, not the physical-coordinate one",
                "user_decision": "accepted 2026-08-06; run on the shared-axis cohort",
            },
            {
                "id": "D2",
                "claim_in_plan": "patient-specific 2D propagation plane",
                "measured": "over half of solved planes carry poor_planarity "
                            "(transverse PC1 < 0.80); see projection_loss_fraction",
                "user_decision": "accepted 2026-08-06; 2D stays primary, 3D node "
                                 "placement is a pre-registered sensitivity arm",
            },
            {
                "id": "D3",
                "claim_in_plan": "contact-node graph RNN as stage 1",
                "measured": "overlaps the persistent path-mode graph RNN frozen on "
                            "2026-07-28 (do_not_tune list)",
                "user_decision": "accepted 2026-08-06; retained as a baseline arm, "
                                 "not as a novel claim",
            },
        ],
        "subjects": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=False))
    for tree_name, summary in cohorts.items():
        print(
            f"{tree_name:7s} eligible={summary['n_eligible']:3d} "
            f"planar_subset={summary['n_planar_subset']:3d} "
            f"joint_contacts min/med/max="
            f"{summary['joint_contact_min']}/{summary['joint_contact_median']:.0f}/"
            f"{summary['joint_contact_max']}"
        )
    print(f"wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
