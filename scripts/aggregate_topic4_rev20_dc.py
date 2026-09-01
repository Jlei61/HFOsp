#!/usr/bin/env python3
"""Aggregate one rev20-DC phase under separated endpoint semantics."""
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

from src.topic4_rev20_dual_core_endpoint import (  # noqa: E402
    embedding_from_training_arrays, matched_patient_floor,
    score_returned_families,
)
from src.topic4_shaft_aware import contract_groups  # noqa: E402


WORKER_STATUS = "REV12ND_NODE_WORKER_COMPLETE"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, path: str) -> Path:
    local = ROOT / path
    return local if local.exists() else artifact_root / path


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]).copy() for key in loaded.files}


def _analysis_provenance(config_path: Path) -> dict:
    paths = [
        Path(__file__).resolve(), config_path.resolve(),
        ROOT / "src/topic4_rev20_dual_core_endpoint.py",
        ROOT / "src/topic4_shaft_aware.py",
        ROOT / "src/topic4_shaft_aware_direction.py",
        ROOT / "src/topic4_d6_natural_kmeans.py",
    ]
    relative = [str(path.relative_to(ROOT)) for path in paths]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *relative],
        cwd=ROOT, text=True,
    ).strip()
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "runtime_paths_dirty": bool(dirty),
        "runtime_dirty_porcelain": dirty.splitlines(),
        "runtime_path_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in paths
        },
    }


def _phase_contract(config: dict, manifest: dict, phase: str,
                    output_root: Path) -> tuple[list[dict], list[int]]:
    if phase == "canary":
        candidates = [row for row in manifest["candidates"] if row["is_reference"]]
        seeds = config["search"]["canary_network_seeds"]
    elif phase == "screen":
        candidates = manifest["candidates"]
        seeds = config["search"]["fit_network_seeds"]
    elif phase == "confirmation":
        selected = json.loads((output_root / "selected_candidates.json").read_text())
        identifiers = set(selected["candidate_ids"])
        candidates = [
            row for row in manifest["candidates"]
            if row["candidate_id"] in identifiers
        ]
        if {row["candidate_id"] for row in candidates} != identifiers:
            raise RuntimeError("confirmation selection lies outside manifest")
        seeds = config["search"]["confirmation_network_seeds"]
    else:
        raise ValueError(phase)
    return candidates, [int(seed) for seed in seeds]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase", choices=("canary", "screen", "confirmation"),
                        required=True)
    parser.add_argument(
        "--artifact-root", type=Path,
        default=Path("/home/honglab/leijiaxin/HFOsp"),
    )
    parser.add_argument("--open-heldout", action="store_true")
    args = parser.parse_args()
    if args.open_heldout and args.phase != "confirmation":
        raise RuntimeError("held-out endpoint opens only in confirmation")
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    analysis_provenance = _analysis_provenance(config_path)
    if analysis_provenance["runtime_paths_dirty"]:
        raise RuntimeError("aggregate runtime paths are dirty")
    output_root = artifact_root / config["output_root"]
    manifest_path = artifact_root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("candidate manifest is stale")
    candidates, seeds = _phase_contract(
        config, manifest, args.phase, output_root,
    )

    training_path = _resolve(
        artifact_root, config["inputs"]["patient_training_target"]["path"],
    )
    training = _load_npz(training_path)
    support_config_path = _resolve(
        artifact_root, config["inputs"]["patient_support_config"]["path"],
    )
    support_config = json.loads(support_config_path.read_text())
    support_manifest_path = _resolve(
        artifact_root,
        support_config["inputs"]["old_ab_train_only_classifier"]["path"],
    )
    if _sha256(support_manifest_path) != support_config["inputs"][
            "old_ab_train_only_classifier"]["sha256"]:
        raise RuntimeError("direction classifier source changed")
    support_manifest = json.loads(support_manifest_path.read_text())
    classifier = support_manifest["direction_classifier"]
    contract_path = _resolve(
        artifact_root, support_config["inputs"]["contact_contract"]["path"],
    )
    if _sha256(contract_path) != support_config["inputs"]["contact_contract"][
            "sha256"]:
        raise RuntimeError("contact contract changed")
    contract = json.loads(contract_path.read_text())
    if list(training["contact_names"]) != [
            row["contact_name"] for row in contract["contacts"]]:
        raise RuntimeError("patient target and contact contract orders differ")

    rows, floor_cache = [], {}
    worker_root = output_root / args.phase / "workers"
    for candidate in candidates:
        for seed in seeds:
            stem = worker_root / f"{candidate['candidate_id']}_seed_{seed}"
            json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
            payload = json.loads(json_path.read_text())
            if payload.get("status") != WORKER_STATUS:
                raise RuntimeError(f"worker incomplete: {json_path}")
            if payload.get("arrays", {}).get("sha256") != _sha256(npz_path):
                raise RuntimeError(f"worker array hash changed: {npz_path}")
            provenance = payload.get("provenance", {})
            if (provenance.get("git_commit") != manifest["git_commit"]
                    or provenance.get("expected_git_commit") != manifest["git_commit"]
                    or provenance.get("runtime_modules_dirty")
                    or not provenance.get("runtime_modules_match_expected_commit")):
                raise RuntimeError(f"worker provenance is not frozen: {json_path}")
            if payload.get("event_unit", {}).get("name") != config["event_unit"]["name"]:
                raise RuntimeError("formal event unit changed")
            if not payload.get("event_unit", {}).get("all_detector_fragments_represented"):
                raise RuntimeError("causal-family partition dropped a detector fragment")
            if payload.get("mechanism_freeze", {}).get("Z_M") != "off":
                raise RuntimeError("Z/M was active in an interictal mechanism run")
            arrays = _load_npz(npz_path)
            scored = score_returned_families(
                arrays["onsets"], arrays["ranks"], arrays["event_returned"],
                contract=contract, training_arrays=training,
                classifier=classifier,
                kmeans_seed=int(config["validation"]["natural_kmeans_seed"]),
            )
            n_returned = scored["validation_diagnostic"]["n_returned_families"]
            if n_returned not in floor_cache:
                floor_cache[n_returned] = matched_patient_floor(
                    training["patient_train_onsets"], n_returned,
                    groups=contract_groups(contract),
                    embedding=embedding_from_training_arrays(training),
                    draws=int(config["complete_distribution"]["floor_draws"]),
                    seed=int(config["complete_distribution"]["floor_seed"]) + n_returned,
                )
            rows.append({
                "candidate_id": candidate["candidate_id"],
                "family": candidate["family"],
                "level": candidate["level"],
                "seed": seed,
                "runaway_early_stop_ms": payload["simulation"].get(
                    "runaway_early_stop_ms"
                ),
                "raw_detector_fragment_count": payload["event_unit"][
                    "raw_detector_fragment_count"
                ],
                "causal_family_count": payload["event_unit"]["episode_count"],
                "compound_detector_fragment_count": payload["event_unit"].get(
                    "n_compound_detector_fragments", 0
                ),
                "training_patient_floor": floor_cache[n_returned],
                **scored,
            })

    aggregate = {
        "schema_id": "topic4_rev20_dc_phase_aggregate_v1",
        "status": f"REV20_DC_{args.phase.upper()}_AGGREGATE_COMPLETE",
        "phase": args.phase,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "git_commit": manifest["git_commit"],
        "analysis_provenance": analysis_provenance,
        "candidate_count": len(candidates),
        "network_count": len(seeds),
        "run_count": len(rows),
        "selection_endpoint": (
            "unconditional complete-event distribution on patient training only"
        ),
        "validation_endpoint_status": (
            "TRAINING_ONLY_DIAGNOSTIC" if args.phase != "confirmation"
            else "FROZEN_VALIDATION_OPENED"
        ),
        "heldout_opened": bool(args.open_heldout),
        "per_network": rows,
        "claim_boundary": config["claim_boundary"],
    }
    output = output_root / args.phase / "aggregate.json"
    _atomic_json(output, _json_safe(aggregate))
    print(json.dumps({
        "status": aggregate["status"], "run_count": len(rows),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
