#!/usr/bin/env python3
"""Fail-closed machine audit for the complete R1.7A / T2-R2.0 goal."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.r1_7 import R1_7A_REVISION
from src.topic5_continuous_marked_state_r1.r1_7_t2 import R1_7_T2_REVISION


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=contract.RESULT_ROOT / "r1_7a")
    args = parser.parse_args()
    inventory_path = args.root / "manifests/cohort_inventory.json"
    r1_path = args.root / "reports/r1_7a_summary.json"
    t2_path = args.root / "reports/t2_r2_summary.json"
    inventory, r1, t2 = load(inventory_path), load(r1_path), load(t2_path)
    subjects = inventory["selected_subjects"]
    fits = []
    r1_source_payloads = set()
    for subject in subjects:
        for seed in range(5):
            path = args.root / "fits" / subject / f"seed_{seed}/result.json"
            value = load(path)
            support = value["d_state"]["support"]
            checks = {
                "status": value.get("status") == "COMPLETE",
                "revision": value.get("revision") == R1_7A_REVISION,
                "subject_seed": value.get("subject") == subject and value.get("seed") == seed,
                "layers_touch_not_overlap": support["state_stop"] == support["mechanism_start"],
                "recorded_support_conserved": abs(
                    support["state_recorded_seconds"]
                    + support["mechanism_recorded_seconds"]
                    - support["total_recorded_seconds"]
                ) < 1e-6,
                "state_fraction": abs(
                    support["state_recorded_seconds"]
                    / support["total_recorded_seconds"] - .60
                ) < 1e-9,
                "d_mechanism_not_scored": value.get("d_mechanism_scored_here") is False,
                "selection_safe": value.get("development_validation_used_for_selection") is False,
                "formal_closed": value.get("formal_test_partition_opened") is False,
                "sealed_closed": value.get("sealed_opened") is False,
                "checkpoint_hash": contract.sha256_file(value["checkpoint"]) == value["checkpoint_sha256"],
            }
            if not all(checks.values()):
                raise ValueError(f"R1.7A audit failed {path}: {checks}")
            r1_source_payloads.add(json.dumps(value["source_hashes"], sort_keys=True))
            fits.append({"path": str(path), "sha256": contract.sha256_file(path),
                         "stable": value["stable_checkpoint"], "checks": checks})
    t2_rows = []
    t2_source_payloads = set()
    for path in sorted((args.root / "t2_r2").glob("*/*/result.json")):
        value = load(path)
        checks = {
            "status": value.get("status") == "COMPLETE",
            "revision": value.get("revision") == R1_7_T2_REVISION,
            "formal_closed": value.get("formal_test_partition_opened") is False,
            "sealed_closed": value.get("sealed_opened") is False,
        }
        if value.get("analysis_status") == "ESTIMATED":
            checks.update({
                "n100_only": value.get("scale_events") == 100,
                "no_free_intercept": value.get("free_exposure_intercept_present") is False,
                "d_state_excluded": value["design"].get("d_state_validation_events_excluded") is True,
            })
        if not all(checks.values()):
            raise ValueError(f"T2 audit failed {path}: {checks}")
        t2_source_payloads.add(json.dumps(value["source_hashes"], sort_keys=True))
        t2_rows.append({"path": str(path), "sha256": contract.sha256_file(path),
                        "analysis_status": value.get("analysis_status"), "checks": checks})
    if len(fits) != 50:
        raise ValueError(f"expected 50 R1.7A fits, found {len(fits)}")
    if len(r1_source_payloads) != 1:
        raise ValueError(f"R1.7A cells used {len(r1_source_payloads)} source payloads")
    if t2_rows and len(t2_source_payloads) != 1:
        raise ValueError(f"T2 cells used {len(t2_source_payloads)} source payloads")
    audit = {
        "status": "COMPLETE", "revision": "r1_7a_r2_goal_machine_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inventory": str(inventory_path), "inventory_sha256": contract.sha256_file(inventory_path),
        "r1_summary": str(r1_path), "r1_summary_sha256": contract.sha256_file(r1_path),
        "t2_summary": str(t2_path), "t2_summary_sha256": contract.sha256_file(t2_path),
        "n_subjects": len(subjects), "n_r1_fits": len(fits),
        "n_t2_cells": len(t2_rows), "r1_fits": fits, "t2_cells": t2_rows,
        "r1_source_payloads": [json.loads(value) for value in r1_source_payloads],
        "t2_source_payloads": [json.loads(value) for value in t2_source_payloads],
        "r1_5_retired": True, "n_ge_1000_runs": 0,
        "six_hour_boxcar_runs": 0, "physical_clock_runs": 0,
        "seizure_probe_opened": False, "paper_ready_figures_modified": False,
        "formal_test_partition_opened": False, "sealed_opened": False,
    }
    contract.atomic_json(args.root / "reports/machine_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
