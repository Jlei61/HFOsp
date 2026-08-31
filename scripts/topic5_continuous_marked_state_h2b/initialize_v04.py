#!/usr/bin/env python3
"""Freeze the v0.4 contract and inventory exact v0.3 audited source cells."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    CANONICAL_V0_3_RESULT_ROOT,
    V0_4_RESULT_ROOT,
    atomic_json,
    sha256_file,
    utc_now,
)


CONFIG = REPO / "config/topic5_continuous_marked_state_h2b_v0_4.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def initialise(*, v03_root: Path, result_root: Path) -> dict:
    contract = _json(CONFIG)
    audit_path = v03_root / "reports/machine_audit.json"
    if sha256_file(audit_path) != contract["source"]["v0_3_machine_audit_sha256"]:
        raise ValueError("v0.3 final machine audit SHA256 drift")
    audit = _json(audit_path)
    if audit.get("all_checks_pass") is not True:
        raise ValueError("v0.3 source audit did not pass engineering checks")
    if audit.get("formal_test_partition_opened") or audit.get("sealed_opened"):
        raise ValueError("v0.3 source opened a forbidden partition")

    frozen_path = result_root / "analysis_contract.json"
    if frozen_path.is_file() and _json(frozen_path) != contract:
        previous = _json(frozen_path)
        allowed = {
            (
                "h2b_v0_4_heterogeneous_seizure_entry_routes_v1",
                "h2b_v0_4_heterogeneous_seizure_entry_routes_v2",
            ),
            (
                "h2b_v0_4_heterogeneous_seizure_entry_routes_v2",
                "h2b_v0_4_heterogeneous_seizure_entry_routes_v3",
            ),
            (
                "h2b_v0_4_heterogeneous_seizure_entry_routes_v3",
                "h2b_v0_4_heterogeneous_seizure_entry_routes_v4",
            ),
            (
                "h2b_v0_4_heterogeneous_seizure_entry_routes_v4",
                "h2b_v0_4_heterogeneous_seizure_entry_routes_v5",
            ),
            (
                "h2b_v0_4_heterogeneous_seizure_entry_routes_v5",
                "h2b_v0_4_heterogeneous_seizure_entry_routes_v6",
            ),
            (
                "h2b_v0_4_heterogeneous_seizure_entry_routes_v6",
                "h2b_v0_4_heterogeneous_seizure_entry_routes_v7",
            ),
            (
                "h2b_v0_4_heterogeneous_seizure_entry_routes_v7",
                "h2b_v0_4_heterogeneous_seizure_entry_routes_v8",
            ),
            (
                "h2b_v0_4_heterogeneous_seizure_entry_routes_v8",
                "h2b_v0_4_heterogeneous_seizure_entry_routes_v9",
            ),
            (
                "h2b_v0_4_heterogeneous_seizure_entry_routes_v9",
                "h2b_v0_4_heterogeneous_seizure_entry_routes_v10",
            ),
        }
        transition = (
            previous.get("schema_revision"), contract.get("schema_revision")
        )
        if transition not in allowed:
            raise ValueError("an incompatible v0.4 analysis contract already exists")
        atomic_json(
            result_root / f"manifests/analysis_contract_{transition[0].rsplit('_', 1)[-1]}_superseded.json",
            previous,
        )
    atomic_json(frozen_path, contract)

    audited_cache = audit["full_grid_cache_sha256"]
    audited_instrument = audit["instrument_manifest_sha256"]
    audited_checkpoint = audit["checkpoint_sha256"]
    manifest_paths = sorted(
        Path(path) for path in audited_cache if path.endswith("/states.manifest.json")
    )
    cells = []
    for old_manifest_path in manifest_paths:
        # The audit contains canonical absolute paths.  Re-anchor only the
        # v0.3 prefix so an explicitly supplied read-only mirror can be used.
        relative = old_manifest_path.relative_to(CANONICAL_V0_3_RESULT_ROOT)
        manifest_path = v03_root / relative
        expected_manifest_sha = audited_cache[str(old_manifest_path)]
        if sha256_file(manifest_path) != expected_manifest_sha:
            raise ValueError(f"state manifest SHA256 drift: {manifest_path}")
        manifest = _json(manifest_path)
        subject = str(manifest["checkpoint_subject"])
        seed = int(manifest["checkpoint_seed"])
        cache_path = manifest_path.with_name("states.npz")
        old_cache_path = old_manifest_path.with_name("states.npz")
        if sha256_file(cache_path) != audited_cache[str(old_cache_path)]:
            raise ValueError(f"state cache SHA256 drift: {cache_path}")
        if manifest.get("cache_sha256") != sha256_file(cache_path):
            raise ValueError(f"state cache self-hash drift: {cache_path}")
        if manifest.get("all_parameters_frozen") is not True:
            raise ValueError(f"state cache not frozen: {cache_path}")
        if manifest.get("seizure_gradient_path") is not False:
            raise ValueError(f"seizure gradient path present: {cache_path}")
        if manifest.get("max_source_time_le_anchor") is not True:
            raise ValueError(f"causal anchor receipt missing: {cache_path}")
        if manifest.get("gap_reset") is not True:
            raise ValueError(f"gap reset receipt missing: {cache_path}")

        instrument_path = (
            v03_root / "instrument/by_cell" / subject / f"seed_{seed}"
            / "instrument_manifest.json"
        )
        old_instrument = (
            CANONICAL_V0_3_RESULT_ROOT / "instrument/by_cell" / subject
            / f"seed_{seed}" / "instrument_manifest.json"
        )
        if str(old_instrument) not in audited_instrument:
            raise ValueError(f"instrument absent from v0.3 audit: {old_instrument}")
        if sha256_file(instrument_path) != audited_instrument[str(old_instrument)]:
            raise ValueError(f"instrument SHA256 drift: {instrument_path}")
        instrument = _json(instrument_path)
        checkpoint = Path(instrument["source"]["checkpoint"]["checkpoint"])
        if str(checkpoint) not in audited_checkpoint:
            raise ValueError(f"checkpoint absent from v0.3 audit: {checkpoint}")
        if sha256_file(checkpoint) != audited_checkpoint[str(checkpoint)]:
            raise ValueError(f"checkpoint SHA256 drift: {checkpoint}")
        cells.append({
            "subject": subject,
            "seed": seed,
            "state_manifest": str(manifest_path),
            "state_manifest_sha256": expected_manifest_sha,
            "state_cache": str(cache_path),
            "state_cache_sha256": audited_cache[str(old_cache_path)],
            "instrument_manifest": str(instrument_path),
            "instrument_manifest_sha256": audited_instrument[str(old_instrument)],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": audited_checkpoint[str(checkpoint)],
            "state_stratum_is_nonblocking": True,
        })
    if len(cells) != 46:
        raise ValueError(f"expected 46 audited full-grid cells, found {len(cells)}")
    payload = {
        "status": "COMPLETE",
        "revision": "h2b_v0_4_exact_source_inventory_v1",
        "created_utc": utc_now(),
        "n_cells": len(cells),
        "n_subjects": len({row["subject"] for row in cells}),
        "cells": cells,
        "v0_3_machine_audit": str(audit_path),
        "v0_3_machine_audit_sha256": sha256_file(audit_path),
        "analysis_contract": str(frozen_path),
        "analysis_contract_source": str(CONFIG),
        "analysis_contract_source_sha256": sha256_file(CONFIG),
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }
    atomic_json(result_root / "manifests/source_cells.json", payload)
    atomic_json(result_root / "CURRENT_HANDOFF.json", {
        "status": "INITIALISED",
        "created_utc": utc_now(),
        "next": "run semi-synthetic assay and 46-cell development queue",
        "n_source_cells": len(cells),
    })
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-3-root", type=Path, default=CANONICAL_V0_3_RESULT_ROOT)
    parser.add_argument("--result-root", type=Path, default=V0_4_RESULT_ROOT)
    args = parser.parse_args()
    result = initialise(
        v03_root=args.v0_3_root.resolve(), result_root=args.result_root.resolve(),
    )
    print(result["status"], result["n_cells"], result["n_subjects"])


if __name__ == "__main__":
    main()
