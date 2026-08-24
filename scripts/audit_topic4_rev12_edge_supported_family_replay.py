#!/usr/bin/env python3
"""Compare the edge-supported exact replay with its causal-root source fit."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_core_field_runner import atomic_write_json


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_OLD = (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev12/"
    "node_stage_q_causal_root_field_fit"
)
DEFAULT_NEW = (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev12/"
    "node_stage_r_edge_supported_family_replay"
)
DYNAMICS_ARRAYS = (
    "active_fraction", "sheet_activity_counts", "contact_envelope",
    "contact_names", "contact_xy_mm",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> tuple[dict, dict[str, dict]]:
    payload = json.loads(path.read_text())
    return payload, {row["candidate_id"]: row for row in payload["rows"]}


def _primary(row: dict) -> dict:
    found = [item for item in row["event_sensitivity_rows"] if item["primary"]]
    if len(found) != 1:
        raise RuntimeError("candidate does not have exactly one primary event variant")
    return found[0]


def _array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if np.issubdtype(left.dtype, np.number):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", default=DEFAULT_OLD)
    parser.add_argument("--new-root", default=DEFAULT_NEW)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    old_root = artifact_root / args.old_root
    new_root = artifact_root / args.new_root
    old_path = old_root / "aggregate/fit_cascade_summary.json"
    new_path = new_root / "aggregate/fit_cascade_summary.json"
    old_payload, old = _rows(old_path)
    new_payload, new = _rows(new_path)
    if set(new) - set(old):
        raise RuntimeError("replay contains a field absent from the source fit")

    comparisons = []
    all_dynamics_equal = True
    k2_fragility = []
    for candidate_id, new_row in new.items():
        old_row = old[candidate_id]
        per_seed = []
        for seed in new_payload["requested_seeds"]:
            stem = f"{candidate_id}_seed_{seed}.npz"
            old_npz = old_root / "workers" / stem
            new_npz = new_root / "workers" / stem
            with np.load(old_npz, allow_pickle=False) as before, np.load(
                    new_npz, allow_pickle=False) as after:
                parity = {
                    name: _array_equal(before[name], after[name])
                    for name in DYNAMICS_ARRAYS
                }
            all_dynamics_equal = all_dynamics_equal and all(parity.values())
            per_seed.append({"seed": int(seed), "dynamics_array_parity": parity})

        old_primary, new_primary = _primary(old_row), _primary(new_row)
        old_selection = old_row["selection_objective"]
        new_selection = new_row["selection_objective"]
        old_counts = {
            int(row["seed"]): int(row["n_cascade_events"])
            for row in old_row["per_seed"]
        }
        new_counts = {
            int(row["seed"]): int(row["n_cascade_events"])
            for row in new_row["per_seed"]
        }
        maximum_event_delta = max(
            abs(new_counts[seed] - old_counts[seed]) for seed in old_counts
        )
        k2_delta = float(
            new_selection["k2_support"] - old_selection["k2_support"]
        )
        alignment_delta = float(
            new_primary["kmeans_balanced_alignment"]
            - old_primary["kmeans_balanced_alignment"]
        )
        fragile = bool(
            abs(k2_delta) >= 0.25
            and abs(alignment_delta) <= 0.10
            and maximum_event_delta <= 1
        )
        if fragile:
            k2_fragility.append(candidate_id)
        comparisons.append({
            "candidate_id": candidate_id,
            "dynamics": per_seed,
            "maximum_primary_event_count_delta": int(maximum_event_delta),
            "old_primary": {
                "matched_patient_loss": old_primary["matched_patient_loss"],
                "kmeans_balanced_alignment": old_primary[
                    "kmeans_balanced_alignment"
                ],
                "k2_support": old_primary["k2_support"],
            },
            "new_primary": {
                "matched_patient_loss": new_primary["matched_patient_loss"],
                "kmeans_balanced_alignment": new_primary[
                    "kmeans_balanced_alignment"
                ],
                "k2_support": new_primary["k2_support"],
            },
            "old_robust_objective": old_selection["objective"],
            "new_robust_objective": new_selection["objective"],
            "robust_k2_support_delta": k2_delta,
            "primary_kmeans_alignment_delta": alignment_delta,
            "single_event_k2_fragility": fragile,
        })

    old_order = [
        row["candidate_id"] for row in old_payload["rows"]
        if row["candidate_id"] in new
    ]
    new_order = [row["candidate_id"] for row in new_payload["rows"]]
    same_leader = bool(old_order and new_order and old_order[0] == new_order[0])
    verdict = {
        "dynamics_parity": "PASS" if all_dynamics_equal else "FAIL",
        "leading_candidate_stable": same_leader,
        "candidate_order_stable": old_order == new_order,
        "k2_auxiliary_fragility": (
            "CONFIRMED" if k2_fragility else "NOT_DETECTED"
        ),
        "k2_fragile_candidates": k2_fragility,
        "stage_q_geometry_library": "RETAIN",
        "stage_q_objective_ranking": "INVALIDATE",
        "next_action": (
            "replay all 54 Stage-Q fields under the edge-supported event unit; "
            "keep KMeans-patient direction alignment but set the unstable "
            "diagonal-GMM K2 auxiliary weight to zero"
        ),
    }
    if not all_dynamics_equal:
        verdict["next_action"] = "stop: exact replay changed SNN dynamics"
    output = {
        "schema_id": "topic4_rev12_edge_supported_family_replay_audit_v1",
        "scientific_role": "fit_only_event_identity_and_objective_stability_audit",
        "old_summary": str(old_path.relative_to(artifact_root)),
        "old_summary_sha256": _sha256(old_path),
        "new_summary": str(new_path.relative_to(artifact_root)),
        "new_summary_sha256": _sha256(new_path),
        "old_order_within_replayed_fields": old_order,
        "new_order": new_order,
        "comparisons": comparisons,
        "verdict": verdict,
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            ).strip(),
        },
    }
    output_path = args.output or new_root / "aggregate/replay_audit.json"
    atomic_write_json(output, output_path)
    print(json.dumps(verdict, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
