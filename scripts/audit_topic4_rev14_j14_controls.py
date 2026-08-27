#!/usr/bin/env python3
"""Zero-simulation anti-cheating controls for the frozen rev14 J14 objective."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rescore_topic4_rev13_exact_off_static_node import (  # noqa: E402
    load_patient_training_target,
)
from src.topic4_node_dualmode import (  # noqa: E402
    calibrate_component_scales,
    fixed_projection_matrix,
)
from src.topic4_rev14_static_node_objective import rev14_objective  # noqa: E402


DEFAULT_ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev14_static_node_causal_family_diagnostic.json"
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev14/"
    "j14_positive_negative_controls/j14_controls.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _score_once(ranks: np.ndarray, probability_b: np.ndarray, *, patient: dict,
                projections: np.ndarray, calibration: dict, formal: dict,
                objective_seed: int,
                ood_mask: np.ndarray | None = None) -> dict:
    finite_contacts = np.sum(np.isfinite(ranks), axis=1) if len(ranks) else np.empty(0)
    ood = (
        np.zeros(len(ranks), dtype=bool)
        if ood_mask is None else np.asarray(ood_mask, dtype=bool)
    )
    if ood.shape != (len(ranks),):
        raise ValueError("OOD mask does not align with control events")
    evidence = (finite_contacts >= 3) & ~ood
    result = rev14_objective(
        ranks, probability_b,
        patient["all_ranks"], patient["all_labels"], patient["all_blocks"],
        patient["contact_names"], projections=projections,
        calibration=calibration, returned_families=len(ranks),
        contact_evaluable_families=len(ranks), overlap_excluded_families=0,
        less_than_three_contact_families=int(np.sum(finite_contacts < 3)),
        mode_evidence_mask=evidence,
        sample_size=int(formal["sample_size_per_side"]),
        draws=int(formal["draws_per_network"]), seed=int(objective_seed),
        tau=float(formal["tau"]),
    )
    return {
        "objective": float(result["objective"]),
        "weakest_mode_lse": float(result["weakest_mode_lse"]),
        "mode_means": [
            float(result["modes"][str(mode)]["mean"]) for mode in (0, 1)
        ],
        "effective_events": [
            float(result["modes"][str(mode)]["effective_events"])
            for mode in (0, 1)
        ],
        "occupancy_js": float(result["occupancy_js"]),
        "ambiguity": float(result["ambiguity"]),
        "contrast_loss": float(result["contrast"]["loss"]),
        "support_loss": float(result["support_loss"]),
        "n_events": int(len(ranks)),
        "objective_seed": int(objective_seed),
    }


def _score(ranks: np.ndarray, probability_b: np.ndarray, *, patient: dict,
           projections: np.ndarray, calibration: dict, formal: dict,
           objective_seeds: tuple[int, ...],
           ood_mask: np.ndarray | None = None) -> dict:
    records = [
        _score_once(
            ranks, probability_b, patient=patient, projections=projections,
            calibration=calibration, formal=formal, objective_seed=seed,
            ood_mask=ood_mask,
        )
        for seed in objective_seeds
    ]
    aggregate = {
        key: float(np.mean([record[key] for record in records]))
        for key in (
            "objective", "weakest_mode_lse", "occupancy_js", "ambiguity",
            "contrast_loss", "support_loss",
        )
    }
    aggregate.update({
        "mode_means": np.mean(
            np.asarray([record["mode_means"] for record in records]), axis=0,
        ).tolist(),
        "effective_events": np.mean(
            np.asarray([record["effective_events"] for record in records]), axis=0,
        ).tolist(),
        "n_events": int(len(ranks)),
        "objective_seeds": [int(seed) for seed in objective_seeds],
        "per_objective_seed": records,
    })
    return aggregate


def _objective_seeds(seed: int) -> tuple[int, int, int]:
    return int(seed), int(seed) + 104729, int(seed) + 209759


def _strict_per_seed_ordering(
        controls: dict, *, baseline_name: str = "block_disjoint_patient_self",
) -> dict[str, bool]:
    if baseline_name not in controls:
        raise ValueError(f"missing baseline control: {baseline_name}")

    def by_seed(record: dict) -> dict[int, float]:
        entries = record.get("per_objective_seed", [])
        mapped = {
            int(entry["objective_seed"]): float(entry["objective"])
            for entry in entries
        }
        if len(mapped) != len(entries):
            raise ValueError("duplicate objective seed in J14 control record")
        return mapped

    baseline = by_seed(controls[baseline_name])
    if len(baseline) != 3:
        raise ValueError("J14 controls require exactly three objective seeds")
    ordering = {}
    for name, record in controls.items():
        if name == baseline_name:
            continue
        candidate = by_seed(record)
        if set(candidate) != set(baseline):
            raise ValueError(f"objective seeds do not align for control {name}")
        ordering[name] = all(
            candidate[seed] > baseline[seed] for seed in sorted(baseline)
        )
    return ordering


def _block_disjoint_control_sets(patient: dict, *, seed: int) -> tuple[dict, dict]:
    ranks = np.asarray(patient["all_ranks"], dtype=np.float64)
    labels = np.asarray(patient["all_labels"], dtype=np.int8)
    blocks = np.asarray(patient["all_blocks"])
    unique_blocks = np.unique(blocks)
    model_blocks = unique_blocks[::2]
    reference_blocks = unique_blocks[1::2]
    if set(model_blocks.tolist()).intersection(reference_blocks.tolist()):
        raise RuntimeError("control model/reference blocks overlap")
    rng = np.random.default_rng(int(seed))
    model_indices = []
    for mode in (0, 1):
        for block in model_blocks:
            available = np.flatnonzero((labels == mode) & (blocks == block))
            if len(available):
                model_indices.append(int(rng.choice(available)))
    model_indices = np.asarray(sorted(model_indices), dtype=np.int64)
    reference_mask = np.isin(blocks, reference_blocks)
    for mode in (0, 1):
        eligible = [
            block for block in reference_blocks
            if np.sum((labels == mode) & (blocks == block)) >= 6
        ]
        if not eligible:
            raise RuntimeError(f"reference split has no eligible block for mode {mode}")
    model = {
        "ranks": ranks[model_indices],
        "labels": labels[model_indices],
        "indices": model_indices,
        "blocks": blocks[model_indices],
    }
    reference = {
        **patient,
        "all_ranks": ranks[reference_mask],
        "all_labels": labels[reference_mask],
        "all_blocks": blocks[reference_mask],
        "reference_training_rows": np.flatnonzero(reference_mask),
    }
    audit = {
        "split_seed": int(seed),
        "model_blocks": model_blocks.tolist(),
        "reference_blocks": reference_blocks.tolist(),
        "block_disjoint": True,
        "model_events_per_mode": np.bincount(model["labels"], minlength=2).tolist(),
        "reference_events_per_mode": np.bincount(
            reference["all_labels"], minlength=2,
        ).tolist(),
    }
    return {**model, "audit": audit}, reference


def produce(config_path: Path, artifact_root: Path, output: Path) -> dict:
    config_path = config_path.resolve()
    artifact_root = artifact_root.resolve()
    config = json.loads(config_path.read_text())
    target_record = config["inputs"]["patient_training_target"]
    target_path = artifact_root / target_record["path"]
    if _sha256(target_path) != target_record["sha256"]:
        raise RuntimeError("patient training target hash changed")
    legacy = config["soft_objective"]
    formal = config["formal_objective"]
    patient = load_patient_training_target(
        target_path, events_per_mode=int(legacy["patient_reference_events_per_mode"]),
        seed=int(legacy["projection_seed"]),
    )
    projections = fixed_projection_matrix(
        2 * len(patient["contact_names"]),
        n_directions=int(legacy["projection_count"]),
        seed=int(legacy["projection_seed"]),
    )
    calibration = calibrate_component_scales(
        patient["all_ranks"], patient["all_labels"], patient["all_blocks"],
        patient["contact_names"], projections,
        sample_size=int(legacy["calibration_sample_size"]),
        draws=int(legacy["calibration_draws"]),
        seed=int(legacy["calibration_seed"]),
    )

    model, reference = _block_disjoint_control_sets(
        patient, seed=int(formal["seed"]),
    )
    balanced_ranks = np.asarray(model["ranks"], dtype=np.float64)
    balanced_labels = np.asarray(model["labels"], dtype=np.int8)
    scl = np.asarray([
        str(name).startswith("SCL") for name in patient["contact_names"]
    ])
    censored = balanced_ranks.copy()
    censored[:, scl] = np.nan
    mode_zero = balanced_ranks[balanced_labels == 0]
    repeated = np.repeat(balanced_ranks[[0]], 12, axis=0)
    objective_seeds = _objective_seeds(int(formal["seed"]))

    score_kwargs = {
        "patient": reference, "projections": projections,
        "calibration": calibration, "formal": formal,
        "objective_seeds": objective_seeds,
    }

    controls = {
        "block_disjoint_patient_self": _score(
            balanced_ranks, balanced_labels.astype(float), **score_kwargs,
        ),
        "scl_censored": _score(
            censored, balanced_labels.astype(float), **score_kwargs,
        ),
        "single_mode": _score(
            mode_zero, np.zeros(len(mode_zero)), **score_kwargs,
        ),
        "fully_ambiguous": _score(
            balanced_ranks, np.full(len(balanced_ranks), 0.5), **score_kwargs,
        ),
        "repeated_single_event": _score(
            repeated, np.r_[np.zeros(6), np.ones(6)], **score_kwargs,
        ),
        "zero_events": _score(
            np.empty((0, balanced_ranks.shape[1])), np.empty(0), **score_kwargs,
        ),
    }
    ordering = _strict_per_seed_ordering(controls)
    runtime_paths = [
        Path(__file__).resolve(), config_path,
        ROOT / "scripts/rescore_topic4_rev13_exact_off_static_node.py",
        ROOT / "src/topic4_rev14_static_node_objective.py",
        ROOT / "src/topic4_node_dualmode.py",
    ]
    relative = [str(path.relative_to(ROOT)) for path in runtime_paths]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *relative],
        cwd=ROOT, text=True,
    ).strip()
    payload = {
        "schema_id": "topic4_rev14_j14_controls_v1",
        "status": (
            "REV14_J14_CONTROLS_PASS" if all(ordering.values())
            else "REV14_J14_CONTROLS_FAIL"
        ),
        "objective_schema_id": formal["schema_id"],
        "patient_heldout_loaded": False,
        "patient_heldout_used": False,
        "controls": controls,
        "all_degenerate_controls_worse_than_patient_self": all(ordering.values()),
        "ordering_checks": ordering,
        "required_ordering": "every degenerate control > block-disjoint patient self at every frozen objective seed",
        "block_disjoint_control_split": model["audit"],
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            ).strip(),
            "runtime_paths_dirty": bool(dirty),
            "runtime_dirty_porcelain": dirty.splitlines(),
            "runtime_path_sha256": {
                str(path): _sha256(path) for path in runtime_paths
            },
            "patient_training_target": {
                "path": str(target_path), "sha256": _sha256(target_path),
            },
        },
        "claim_boundary": (
            "Zero-simulation patient-training anti-cheating audit only; no "
            "candidate selection, held-out read, SNN run, connectivity or Z/M claim."
        ),
    }
    _atomic_json(output.resolve(), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = produce(args.config, args.artifact_root, args.output)
    print(json.dumps({
        "status": payload["status"],
        "objectives": {
            key: value["objective"] for key, value in payload["controls"].items()
        },
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
