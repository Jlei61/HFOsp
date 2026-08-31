from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.topic5_continuous_marked_state_h2b.v03_attrition import (
    build_attrition_payload,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> None:
    cache = root / "state_cache/s1/seed_0/states.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"frozen-state")
    _write_json(cache.with_suffix(".manifest.json"), {
        "status": "COMPLETE",
        "cache_sha256": _sha(cache),
        "checkpoint_sha256": "checkpoint-0",
        "checkpoint_subject": "s1",
        "checkpoint_seed": 0,
    })
    _write_json(root / "manifests/r1_7_checkpoint_inventory.json", {
        "status": "COMPLETE",
        "n_cells": 2,
        "n_checkpoint_available_cells": 1,
        "entries": [
            {
                "subject": "s1", "dataset": "d", "seed": 0,
                "checkpoint_available": True, "checkpoint_sha256": "checkpoint-0",
                "analysis_status": "SCORED", "h1_stable_subject": True,
            },
            {
                "subject": "s1", "dataset": "d", "seed": 1,
                "checkpoint_available": False, "checkpoint_sha256": None,
                "analysis_status": "NONFINITE_GRADIENT", "h1_stable_subject": True,
            },
        ],
    })
    _write_json(root / "manifests/support_census.json", {
        "status": "COMPLETE",
        "patient_rows": [{
            "subject": "s1", "n_seizures_in_frozen_inventory": 3,
            "primary_complete_coverage_seizures": 3,
            "coverage_available": True, "upstream_design_available": True,
            "raw_inference_cache_available": True,
        }],
    })
    _write_json(root / "reports/machine_audit.json", {
        "status": "PASS_COMPLETE", "formal_test_partition_opened": False,
        "sealed_opened": False,
        "details": {"n_state_cache_cells": 1, "n_subjects_with_input_manifest": 1},
    })
    _write_json(
        root / "fits/by_subject/s1/primary/risk_probe_machine_audit.json",
        {
            "status": "COMPLETE", "execution_status": "COMPLETE",
            "scientific_estimability": "ESTIMABLE",
            "time_label_permutation": {"status": "COMPLETE"},
            "sensitive_loss_value_that_must_not_be_copied": -99.0,
        },
    )


def test_v03_attrition_tracks_cells_without_reading_outcome_values(tmp_path) -> None:
    _fixture(tmp_path)
    payload = build_attrition_payload(tmp_path)
    assert payload["funnel"] == {
        "total_r1_7b_cells": 2,
        "checkpoint_available_cells": 1,
        "state_cache_cells": 1,
        "state_cache_subjects": 1,
        "subjects_with_input_manifest": 1,
        "probe_tasks": 1,
    }
    assert payload["outcome_values_read"] is False
    assert payload["attrition_reason_counts"] == {
        "checkpoint_unavailable:NONFINITE_GRADIENT": 1,
        "state_cache_available": 1,
    }
    assert "sensitive_loss_value_that_must_not_be_copied" not in json.dumps(payload)


def test_v03_attrition_fails_on_cache_hash_drift(tmp_path) -> None:
    _fixture(tmp_path)
    cache = tmp_path / "state_cache/s1/seed_0/states.npz"
    cache.write_bytes(b"drift")
    with pytest.raises(ValueError, match="state cache hash drift"):
        build_attrition_payload(tmp_path)
