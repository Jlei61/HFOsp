#!/usr/bin/env python3
"""Audit the target-free Stage-F attenuation cache handoff.

The original executor and the exact-equivalent deduplicated producer write the
same atomic unit-cache schema.  This audit records which producer supplied each
of the 504 expected unit-targets and proves that aggregation was handed a
complete, target-free, non-overlapping cache set.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    cache_root = OUT / "attenuation/unit_cache"
    cache_paths = sorted(cache_root.glob("*/*/seed*.json.gz"))
    parity_path = OUT / "ATTENUATION_HOTFILL_EXACT_PARITY.json"
    completion_path = OUT / "ATTENUATION_HOTFILL_COMPLETE.json"
    parity = json.loads(parity_path.read_text())
    completion = json.loads(completion_path.read_text())

    identities: list[tuple[str, str, int]] = []
    inventory: list[dict] = []
    target_flags: list[bool] = []
    deduplicated = 0
    original = 0
    for path in cache_paths:
        with gzip.open(path, "rt") as stream:
            payload = json.load(stream)
        relative = path.relative_to(cache_root)
        fit_id, target, seed_name = relative.parts
        seed = int(seed_name.removeprefix("seed").removesuffix(".json.gz"))
        identities.append((fit_id, target, seed))
        is_deduplicated = "rollout_dedup_contract" in payload
        deduplicated += int(is_deduplicated)
        original += int(not is_deduplicated)
        target_flags.append(payload.get("target_values_read") is False)
        inventory.append({
            "fit_id": fit_id,
            "target": target,
            "seed": seed,
            "producer_class": "EXACT_DEDUPLICATED" if is_deduplicated else "ORIGINAL_EXECUTOR",
            "cache_sha256": sha256_file(path),
            "metrics_sha256": payload.get("metrics_sha256"),
        })

    partials = sorted(str(path.relative_to(OUT)) for path in cache_root.rglob("*.tmp"))
    active_marker = (OUT / "ATTENUATION_HOTFILL_ACTIVE.json").exists()
    authorization_exists = (OUT / "TARGET_UNSEAL_AUTHORIZATION.json").exists()
    expected_targets = {"L1_ADDED", "L2M_ADDED", "L3_ADDED", "L3_MATCHED_LOCAL"}
    fit_ids = {identity[0] for identity in identities}
    seeds = {identity[2] for identity in identities}
    targets = {identity[1] for identity in identities}
    unique = len(set(identities)) == len(identities)
    complete_grid = all(
        (fit_id, target, seed) in set(identities)
        for fit_id in fit_ids
        for target in expected_targets
        for seed in range(3)
    )
    parity_pass = bool(
        parity.get("status") == "PASS_TARGET_FREE"
        and parity.get("target_values_read") is False
        and int(parity.get("events", -1)) == 1492
        and int(parity.get("mismatches", -1)) == 0
    )
    completion_pass = bool(
        completion.get("status") == "PASS_TARGET_FREE"
        and completion.get("target_values_read") is False
        and int(completion.get("hotfilled", -1)) == deduplicated
        and completion.get("cumulative_provenance_normalization") is True
    )
    passed = bool(
        len(cache_paths) == 504
        and len(fit_ids) == 42
        and targets == expected_targets
        and seeds == {0, 1, 2}
        and unique
        and complete_grid
        and all(target_flags)
        and deduplicated == 425
        and original == 79
        and not partials
        and not active_marker
        and not authorization_exists
        and parity_pass
        and completion_pass
    )
    payload = {
        "contract": "topic5_stage_f_cache_handoff_audit_v0_5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_TARGET_FREE" if passed else "FAIL_CLOSED",
        "target_values_read": False,
        "target_authorization_absent_at_audit": not authorization_exists,
        "cache_units": len(cache_paths),
        "fits": len(fit_ids),
        "targets": sorted(targets),
        "seeds": sorted(seeds),
        "deduplicated_units": deduplicated,
        "original_executor_units": original,
        "identity_unique": unique,
        "complete_fit_target_seed_grid": complete_grid,
        "all_cache_target_flags_false": all(target_flags),
        "partial_cache_files": partials,
        "hotfill_active_marker_absent": not active_marker,
        "exact_parity": {
            "status": parity.get("status"),
            "events": parity.get("events"),
            "unique_starts": parity.get("unique_starts"),
            "mismatches": parity.get("mismatches"),
            "sha256": sha256_file(parity_path),
        },
        "hotfill_completion_sha256": sha256_file(completion_path),
        "cache_inventory": inventory,
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(OUT / "STAGE_F_CACHE_HANDOFF_AUDIT.json", payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "cache_inventory"}, indent=2))
    if not passed:
        raise RuntimeError("Stage-F cache handoff audit failed")


if __name__ == "__main__":
    main()
