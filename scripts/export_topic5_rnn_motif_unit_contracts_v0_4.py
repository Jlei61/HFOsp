#!/usr/bin/env python3
"""Export explicit per-unit config and input-hash contracts for v0.4.

The immutable training runner stored this information inside ``metrics.json``.
The locked specification also requires standalone ``config.json`` and
``input_hashes.json`` beside every training unit.  This closeout exporter is a
lossless, idempotent split of already-frozen metadata; it never trains a model
or changes a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def atomic_same_or_write(path: Path, payload: Any) -> str:
    text = canonical(payload)
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"refusing to overwrite a different frozen contract: {path}")
        return "existing_identical"
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    temporary.replace(path)
    return "created"


def build_contracts(out_root: Path, metrics_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = json.loads(metrics_path.read_text())
    fit_id = str(metrics["fit_id"])
    cache = out_root / "cache" / fit_id
    inputs = {
        name: {"path": str(cache / name), "sha256": sha256(cache / name)}
        for name in ("plane.npz", "events.npz", "provenance.json")
    }
    manifest = out_root / "INPUT_MANIFEST.json"
    manifest_sha = sha256(manifest)
    recorded_manifest_sha = str(metrics["producer_hashes"]["input_manifest"])
    if manifest_sha != recorded_manifest_sha:
        raise RuntimeError(
            f"input manifest differs from the value frozen during training: {metrics_path}"
        )
    identity = {
        "fit_id": fit_id,
        "subject": str(metrics["subject"]),
        "fit_scope": str(metrics["fit_scope"]),
        "model_id": str(metrics["model_id"]),
        "arm": str(metrics["arm"]),
        "cell": str(metrics["cell"]),
        "seed": int(metrics["seed"]),
    }
    config = {
        "contract": "topic5_rnn_motif_unit_config_export_v0_4",
        **identity,
        "shuffled_targets": bool(metrics["shuffled_targets"]),
        "shuffle_mode": str(metrics["shuffle_mode"]),
        "training_config": metrics["config"],
        "rollout_decoder_training": metrics["rollout_decoder"],
        "config_sha256_recorded_by_trainer": str(metrics["config_sha256"]),
        "source_metrics_sha256": sha256(metrics_path),
    }
    input_hashes = {
        "contract": "topic5_rnn_motif_unit_input_hashes_v0_4",
        **identity,
        "input_manifest": {"path": str(manifest), "sha256": manifest_sha},
        "fit_cache": inputs,
        "producer_hashes_recorded_by_trainer": metrics["producer_hashes"],
        "source_metrics_sha256": sha256(metrics_path),
    }
    return config, input_hashes


def export_preflight_inventory(out_root: Path) -> dict[str, Any]:
    """Index the immutable preflight evidence under the plan's named artifact."""
    manifest_path = out_root / "INPUT_MANIFEST.json"
    preflight_path = out_root / "PRE_FLIGHT_AUDIT.json"
    metadata_path = out_root / "EARLY_ICTAL_METADATA_INVENTORY.json"
    manifest = json.loads(manifest_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    if preflight.get("target_values_read") is not False:
        raise RuntimeError("preflight source does not preserve the target seal")
    if metadata.get("target_values_read") is not False:
        raise RuntimeError("metadata source does not preserve the target seal")
    payload = {
        "contract": "topic5_rnn_motif_preflight_inventory_compatibility_v0_4",
        "artifact_role": (
            "post-run index of immutable target-free preflight sources; not a re-run"
        ),
        "source_artifacts": {
            "input_manifest": {
                "path": str(manifest_path), "sha256": sha256(manifest_path),
            },
            "preflight_audit": {
                "path": str(preflight_path), "sha256": sha256(preflight_path),
            },
            "early_ictal_metadata_inventory": {
                "path": str(metadata_path), "sha256": sha256(metadata_path),
            },
        },
        "n_patients": int(preflight["n_patients"]),
        "n_fits": int(preflight["n_fits"]),
        "n_training_units": int(preflight["n_training_units"]),
        "geometry_status": preflight["geometry_status"],
        "shared_and_noncollinear_fit_inventory": manifest.get("fits", []),
        "expected_primary_n": metadata.get("expected_primary_n"),
        "actual_primary_join_known_before_unseal": metadata.get("actual_primary_join", []),
        "target_values_read_by_source_preflight": False,
        "target_values_deserialized_by_this_exporter": False,
        "created_after_target_unseal": (out_root / "target_access_audit.json").exists(),
    }
    atomic_same_or_write(out_root / "PREFLIGHT_INVENTORY.json", payload)
    return payload


def export(out_root: Path) -> dict[str, Any]:
    metrics_paths = sorted((out_root / "per_subject").glob("*/*__*/seed*/metrics.json"))
    if not metrics_paths:
        raise RuntimeError(f"no v0.4 metrics found under {out_root}")
    formal = 0
    for metrics_path in metrics_paths:
        if not metrics_path.parents[1].name.startswith("SMOKE_"):
            formal += 1
        config, input_hashes = build_contracts(out_root, metrics_path)
        atomic_same_or_write(metrics_path.parent / "config.json", config)
        atomic_same_or_write(metrics_path.parent / "input_hashes.json", input_hashes)
    audit = {
        "contract": "topic5_rnn_motif_unit_contract_export_audit_v0_4",
        "n_all_training_units": len(metrics_paths),
        "n_formal_training_units": formal,
        "n_smoke_training_units": len(metrics_paths) - formal,
        "n_config_contracts": len(metrics_paths),
        "n_input_hash_contracts": len(metrics_paths),
        "checkpoint_or_metric_values_changed": False,
        "export_semantics": "lossless split of frozen metrics plus hashes of frozen fit-cache inputs",
    }
    # The locked plan named a standalone PREFLIGHT_INVENTORY.json, while the
    # immutable execution stored the same evidence across INPUT_MANIFEST and
    # PRE_FLIGHT_AUDIT.  Materialize a transparent compatibility index without
    # re-running preflight or touching any target array after unseal.
    preflight_sources = (
        out_root / "INPUT_MANIFEST.json",
        out_root / "PRE_FLIGHT_AUDIT.json",
        out_root / "EARLY_ICTAL_METADATA_INVENTORY.json",
    )
    if all(path.exists() for path in preflight_sources):
        export_preflight_inventory(out_root)
        audit["preflight_inventory_export"] = "created_or_existing_identical"
    else:
        audit["preflight_inventory_export"] = "not_available_in_unit_only_fixture"
    atomic_same_or_write(out_root / "UNIT_CONTRACT_EXPORT_AUDIT.json", audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.out_root.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
