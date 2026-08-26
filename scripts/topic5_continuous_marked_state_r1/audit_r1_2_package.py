#!/usr/bin/env python3
"""Fail-closed audit for the six-subject full-anchor R1.2 package."""
from __future__ import annotations

import json
from pathlib import Path

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.r1_2 import (
    CACHE_REVISION, R1_2_REVISION, load_full_design,
)


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if value.get("status") != "COMPLETE":
        raise ValueError(f"incomplete artifact: {path}")
    if value.get("sealed_opened") is not False:
        raise ValueError(f"sealed flag is not false: {path}")
    return value


def main() -> None:
    root = contract.RESULT_ROOT / "r1_2"
    denominator = _load(
        root / "manifests/R1_2_ADMISSIBLE_DENOMINATORS.json"
    )
    if denominator.get("r1_2_revision") != R1_2_REVISION:
        raise ValueError("R1.2 denominator revision mismatch")
    expected = {
        row["subject"]: (row["train_anchors"], row["validation_anchors"])
        for row in denominator["rows"]
    }
    checks = {"subjects": {}}
    for subject in contract.PILOT_SUBJECTS:
        baseline = _load(
            root / "baselines" / subject / "seed_0/result.json"
        )
        bridge = _load(
            root / "bridge_e1" / subject / "seed_0/result.json"
        )
        if baseline.get("r1_2_revision") != R1_2_REVISION:
            raise ValueError(f"{subject}: baseline R1.2 revision mismatch")
        if bridge.get("r1_2_revision") != R1_2_REVISION:
            raise ValueError(f"{subject}: Bridge R1.2 revision mismatch")
        cache_path = root / "cache" / subject / "manifest.json"
        cache = _load(cache_path)
        if cache.get("cache_revision") != CACHE_REVISION:
            raise ValueError(f"{subject}: cache revision mismatch")
        if not cache.get("observer_frozen") or not cache.get("full_recorded_support"):
            raise ValueError(f"{subject}: cache scientific flags missing")
        if (cache["n_train_anchors"], cache["n_validation_anchors"]) != expected[subject]:
            raise ValueError(f"{subject}: full-anchor denominator mismatch")
        if cache.get("n_unreadable_anchors") != 0:
            raise ValueError(f"{subject}: unreadable anchor survived frozen denominator")
        for key in (
            "design", "explicit_embedding", "explicit_raw_embedding",
            "bridge_checkpoint", "baseline_checkpoint", "coverage",
        ):
            path = Path(cache[key])
            if contract.sha256_file(path) != cache[f"{key}_sha256"]:
                raise ValueError(f"{subject}: {key} hash mismatch")
        design = load_full_design(Path(cache["design"]))
        if cache["n_train_anchors"] != int((design.anchor_split == 0).sum()):
            raise ValueError(f"{subject}: TRAIN anchor denominator mismatch")
        if cache["n_validation_anchors"] != int((design.anchor_split == 1).sum()):
            raise ValueError(f"{subject}: validation anchor denominator mismatch")
        if cache["n_train_events_full_recorded_support"] != int(
            (design.event_split == 0).sum()
        ):
            raise ValueError(f"{subject}: TRAIN event denominator mismatch")
        if cache["n_validation_events_full_recorded_support"] != int(
            (design.event_split == 1).sum()
        ):
            raise ValueError(f"{subject}: validation event denominator mismatch")
        arms = {}
        for arm in ("explicit", "explicit_raw"):
            result_path = root / "t1_full" / subject / f"{arm}_d8_seed_0/result.json"
            result = _load(result_path)
            if result.get("r1_2_revision") != R1_2_REVISION:
                raise ValueError(f"{subject}/{arm}: R1.2 revision mismatch")
            if not result.get("observer_frozen") or not result.get("full_recorded_support"):
                raise ValueError(f"{subject}/{arm}: scientific flags missing")
            if contract.sha256_file(result["checkpoint"]) != result["checkpoint_sha256"]:
                raise ValueError(f"{subject}/{arm}: checkpoint hash mismatch")
            if max(result["initial_parity_abs"].values()) > 1e-6:
                raise ValueError(f"{subject}/{arm}: initial parity failed")
            arms[arm] = {
                "selected_epochs": result["selected_epochs"],
                "validation_events": result["final_validation"]["filtered"]["n_events"],
                "matched_events": result["wrong_time_match"]["matched_support_events"],
            }
        checks["subjects"][subject] = {
            "bridge_selected_epochs": bridge["selected_epochs"],
            "anchors": expected[subject],
            "events": (
                cache["n_train_events_full_recorded_support"],
                cache["n_validation_events_full_recorded_support"],
            ),
            "arms": arms,
        }
    for path in (
        root / "reports/r1_2_summary.json",
        root / "reports/r1_2_patient_first.csv",
        root / "reports/plain_report_2026-08-25.md",
        root / "reports/technical_report_2026-08-25.md",
    ):
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    source_hashes = {}
    for base in (
        contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1",
        contract.REPO_ROOT / "scripts/topic5_continuous_marked_state_r1",
        contract.REPO_ROOT / "tests/topic5_continuous_marked_state_r1",
    ):
        for path in sorted(base.glob("*.py")):
            source_hashes[str(path.relative_to(contract.REPO_ROOT))] = contract.sha256_file(path)
    output = {
        "status": "PASS",
        "contract": contract.REVISION,
        "r1_2_revision": R1_2_REVISION,
        "checks": checks,
        "source_hashes": source_hashes,
        "claim_boundary": (
            "engineering/provenance closure of a six-subject development pilot; "
            "PASS is not acceptance of H1, H2a-state, H3, or a cohort claim"
        ),
        "sealed_opened": False,
    }
    contract.atomic_json(root / "manifests/FINAL_PACKAGE_AUDIT.json", output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
