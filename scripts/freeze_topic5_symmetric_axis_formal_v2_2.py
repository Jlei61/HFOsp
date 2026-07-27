#!/usr/bin/env python3
"""Freeze formal cohort and execution manifests after development PASS."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_symmetric_axis_propagation_state_v2_2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    development_path = BASE / "development/DEVELOPMENT_LOCK.json"
    physical_path = BASE / "input_audit/physical_axis_formal_cohort.json"
    sequence_path = BASE / "input_audit/all_subject_sequence_cohort.json"
    target_path = BASE / "target_audit/TARGET_VALUES_SEALED.json"
    development = json.loads(development_path.read_text(encoding="utf-8"))
    physical = json.loads(physical_path.read_text(encoding="utf-8"))
    sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    target = json.loads(target_path.read_text(encoding="utf-8"))
    if development["status"] != "pass" or len(physical["subjects"]) != 22:
        raise SystemExit("development/physical cohort is not eligible for formal freeze")
    if len(sequence["subjects"]) != 31:
        raise SystemExit("sequence cohort is not 31")
    if target.get("energy_values_read") or target.get("recruitment_values_read"):
        raise SystemExit("target-value seal was violated")
    formal = BASE / "formal"
    atomic_json(
        formal / "PHYSICAL_AXIS_FORMAL_LOCK.json",
        {
            "contract": "topic5_symmetric_axis_propagation_state_rnn",
            "version": "2.2",
            "status": "locked",
            "subjects": physical["subjects"],
            "n_folds": 22,
            "seeds": [17, 29, 43],
            "selected_objective": development["selected_objective"],
            "H_train": development["H_train"],
            "loso_shared_training_patients_per_fold": 21,
            "heldout_patient_scaffold_fit_partition": "train80",
            "evaluation_partition": "heldout20",
            "primary_comparison": "symmetric_axis_full_vs_local_isotropic",
            "later_controls": [
                "random_axis_256",
                "shaft_preserving_coordinate_permutation_256",
                "pca1_axis",
                "two_direction_operator",
            ],
            "null_seed": 20260726,
            "target_values_sealed": True,
            "hashes": {
                "development_lock": sha256(development_path),
                "physical_cohort": sha256(physical_path),
                "target_seal": sha256(target_path),
            },
        },
    )
    atomic_json(
        formal / "ALL_SUBJECT_SEQUENCE_LOCK.json",
        {
            "contract": "topic5_symmetric_axis_propagation_state_rnn",
            "version": "2.2",
            "status": "locked",
            "subjects": sequence["subjects"],
            "n_subjects": 31,
            "geometry_incomplete_subjects": sequence[
                "geometry_incomplete_subjects"
            ],
            "geometry_incomplete_axis_fallback": False,
            "allowed_models": ["node_bias_no_history", "empirical_first_order_markov"],
            "physical_axis_claim_allowed": False,
            "target_values_sealed": True,
            "hashes": {
                "development_lock": sha256(development_path),
                "sequence_cohort": sha256(sequence_path),
                "target_seal": sha256(target_path),
            },
        },
    )
    print("formal locks written: physical=22, sequence=31, target sealed")


if __name__ == "__main__":
    main()
