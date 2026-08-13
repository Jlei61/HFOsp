#!/usr/bin/env python3
"""Materialize an isolated primary artifact root for a confirmed spatial config.

The target-free search writes its full-cohort units below the search directory,
not below ``per_fit``.  The established postprocess chain deliberately consumes
only ``<out>/per_fit`` and ``<out>/cache``.  This helper joins those two contracts
without replacing or mutating the original v0.3 results: it creates a sibling
artifact root whose ``per_fit`` and ``cache`` entries are immutable symlinks to
the confirmed units and the frozen full-tissue cache.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


FORBIDDEN_TARGET_MARKERS = (
    "TARGET_UNSEAL_AUTHORIZATION.json",
    "TARGET_ACCESS_AUDIT.json",
    "EARLY_ICTAL_SCORING_COMPLETE.json",
)

LINKED_METADATA = (
    "INPUT_CACHE_MANIFEST.json",
    "FULL_TISSUE_CACHE_COMPLETE.json",
    "RUN_CONTRACT.json",
    "LATENT_DOMAIN_AUDIT.csv",
    "EARLY_ICTAL_METADATA_INVENTORY.csv",
    "EARLY_ICTAL_METADATA_INVENTORY.json",
    "EARLY_ICTAL_METADATA_AUDIT_COMPLETE.json",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def assert_target_sealed(root: Path) -> None:
    present = [name for name in FORBIDDEN_TARGET_MARKERS if (root / name).exists()]
    if present:
        raise RuntimeError(f"selected-root preparation requires sealed target: {root}: {present}")


def checked_symlink(destination: Path, source: Path) -> None:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.is_symlink():
        if destination.resolve() != source:
            raise RuntimeError(f"existing symlink points elsewhere: {destination}")
        return
    if destination.exists():
        raise FileExistsError(destination)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def validate_units(unit_root: Path, expected_overrides: dict) -> dict:
    metrics = sorted(unit_root.glob("*/*/seed*/metrics.json"))
    done = sorted(unit_root.glob("*/*/seed*/DONE.json"))
    if len(metrics) != 465 or len(done) != 465:
        raise RuntimeError(
            f"selected full-cohort root must contain 465 metrics/DONE files; "
            f"observed metrics={len(metrics)}, done={len(done)}"
        )
    for path in metrics:
        payload = json.loads(path.read_text())
        if payload.get("target_values_read") is not False:
            raise RuntimeError(f"target contamination in selected unit: {path}")
        if not payload.get("best_checkpoint_eligible") or payload.get("hit_ceiling"):
            raise RuntimeError(f"ineligible selected unit: {path}")
        observed = payload.get("config", {})
        for key, expected in expected_overrides.items():
            value = observed.get(key)
            if value is None or abs(float(value) - float(expected)) > 1e-12:
                raise RuntimeError(
                    f"selected unit config mismatch for {key}: {path}: "
                    f"{value} != {expected}"
                )
    for path in done:
        payload = json.loads(path.read_text())
        if not payload.get("ok") or not payload.get("converged"):
            raise RuntimeError(f"incomplete selected unit: {path}")
    return {
        "n_metrics": len(metrics),
        "n_done": len(done),
        "metrics_listing_sha256": hashlib.sha256(
            "\n".join(f"{path.relative_to(unit_root)}|{sha256(path)}" for path in metrics).encode()
        ).hexdigest(),
    }


def prepare(source_out: Path, selected_root: Path, search_name: str) -> dict:
    source_out = source_out.resolve()
    selected_root = selected_root.resolve()
    assert_target_sealed(source_out)
    decision_path = source_out / search_name / "SPATIAL_MODEL_DECISION.json"
    formal_path = source_out / search_name / "FORMAL_SELECTED_DECISION.json"
    decision = json.loads(decision_path.read_text())
    formal = json.loads(formal_path.read_text())
    config_id = decision.get("selected_config_id")
    if not config_id or formal.get("config_id") != config_id:
        raise RuntimeError("selected configuration is absent or inconsistent")
    if formal.get("verdict") != "FULL_COHORT_SELECTIVE_NONLOCAL_CONFIRMED":
        raise RuntimeError("selected configuration did not pass full-cohort confirmation")
    if decision.get("target_values_read") is not False or formal.get("target_values_read") is not False:
        raise RuntimeError("spatial selection is not target-free")

    selected_config_path = source_out / search_name / "configs" / f"{config_id}.json"
    selected_config = json.loads(selected_config_path.read_text())
    unit_root = source_out / search_name / "units" / "formal_selected" / config_id
    unit_audit = validate_units(unit_root, selected_config)
    marker_path = selected_root / "SELECTED_PRIMARY_ROOT.json"
    if selected_root.exists():
        if not marker_path.exists():
            raise FileExistsError(f"refusing to reuse unmarked selected root: {selected_root}")
        existing = json.loads(marker_path.read_text())
        if existing.get("config_id") != config_id or Path(existing["unit_root"]).resolve() != unit_root.resolve():
            raise RuntimeError("existing selected root was created for another configuration")
        assert_target_sealed(selected_root)
        return existing

    selected_root.mkdir(parents=True)
    checked_symlink(selected_root / "per_fit", unit_root)
    checked_symlink(selected_root / "cache", source_out / "cache")
    linked_hashes = {}
    for name in LINKED_METADATA:
        source = source_out / name
        checked_symlink(selected_root / name, source)
        linked_hashes[name] = sha256(source)
    atomic_json(selected_root / "SELECTED_SPATIAL_CONFIG.json", {
        "contract": "topic5_lbss_selected_spatial_config_v0_4",
        "config_id": config_id,
        "overrides": selected_config,
        "source_config": str(selected_config_path.resolve()),
        "source_config_sha256": sha256(selected_config_path),
        "all_465_units_match_overrides": True,
        "target_values_read": False,
    })

    formal_marker = {
        "contract": "topic5_lbss_selected_spatial_formal_training_v0_4",
        "complete": 465,
        "unresolved": 0,
        "config_id": config_id,
        "unit_root": str(unit_root.resolve()),
        "source_formal_decision": str(formal_path.resolve()),
        "source_formal_decision_sha256": sha256(formal_path),
        "target_values_read": False,
        "created_at": now(),
    }
    atomic_json(selected_root / "FORMAL_TRAINING_COMPLETE.json", formal_marker)
    payload = {
        "contract": "topic5_lbss_selected_primary_artifact_root_v0_4",
        "status": "READY_FOR_PRETARGET_POSTPROCESS",
        "config_id": config_id,
        "source_out_root": str(source_out),
        "selected_root": str(selected_root),
        "unit_root": str(unit_root.resolve()),
        "cache_root": str((source_out / "cache").resolve()),
        "spatial_model_decision_sha256": sha256(decision_path),
        "formal_selected_decision_sha256": sha256(formal_path),
        "linked_metadata_sha256": linked_hashes,
        "unit_audit": unit_audit,
        "selected_config": selected_config,
        "selected_config_sha256": sha256(selected_root / "SELECTED_SPATIAL_CONFIG.json"),
        "target_values_read": False,
        "created_at": now(),
    }
    atomic_json(marker_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-out-root", type=Path, required=True)
    parser.add_argument("--selected-root", type=Path, required=True)
    parser.add_argument("--search-name", default="development_spatial_search_v0_4")
    args = parser.parse_args()
    print(json.dumps(
        prepare(args.source_out_root, args.selected_root, args.search_name),
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
