"""Tests for the Phase-C analysis-only runaway-scope amendment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.write_topic4_zm_phasec_analysis_amendment as W  # noqa: E402


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_fixture(tmp_path, *, include_all_sheet):
    code = tmp_path / "code"
    result = code / "results"
    part_dir = result / "parts/c1_base/dt/seed1/primary/cell/phase/noise"
    part_dir.mkdir(parents=True)
    for relative in W.ORIGINAL_ANALYSIS_FILES:
        path = code / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    corrected = code / "scripts/corrected.py"
    corrected.parent.mkdir(parents=True, exist_ok=True)
    corrected.write_text("corrected")
    writer = code / "scripts/writer.py"
    writer.write_text("writer")

    arrays = {
        "source_rate_hz": np.full(16, 440.0, np.float32),
        "bin_ms": np.asarray(2.0),
    }
    if include_all_sheet:
        arrays.update({
            "carrier_gate_r_all_hz": np.full(4, 150.0, np.float32),
            "carrier_gate_bin_ms": np.asarray(25.0),
        })
    obs = part_dir / "observables.npz"
    np.savez_compressed(obs, **arrays)
    part = {
        "status": "complete",
        "observables_path": str(obs),
        "observables_sha256": _sha(obs),
    }
    (part_dir / "phenotype.json").write_text(json.dumps(part))

    original = {
        relative: _sha(code / relative)
        for relative in W.ORIGINAL_ANALYSIS_FILES
    }
    manifest_body = {
        "provenance": {"producer_file_sha256": original},
    }
    manifest = {
        **manifest_body,
        "manifest_sha256": W._canonical_sha(manifest_body),
    }
    manifest_path = result / "phasec_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    summary = {
        "schema": "zm_phasec1_coordinator_v1_2026-07-28",
        "phase": "base",
        "phasec_manifest_sha256": manifest["manifest_sha256"],
        "n_expected_simulations": 1,
        "n_pending_after_stop": 0,
        "n_failures": 0,
    }
    coordinator = result / "coordinator.json"
    coordinator.write_text(json.dumps(summary))
    return {
        "code": code,
        "result": result,
        "manifest": manifest_path,
        "coordinator": coordinator,
        "corrected": {"scripts/corrected.py": _sha(corrected)},
        "writer": {"scripts/writer.py": _sha(writer)},
    }


def test_amendment_fails_closed_without_all_sheet_trace(tmp_path):
    fixture = _write_fixture(tmp_path, include_all_sheet=False)
    with pytest.raises(ValueError, match="missing:carrier_gate_r_all_hz"):
        W.build_payload(
            result_root=fixture["result"],
            manifest_path=fixture["manifest"],
            coordinator_summary_path=fixture["coordinator"],
            corrected_analysis_producers=fixture["corrected"],
            amendment_producers=fixture["writer"],
            analysis_git_sha="a" * 40,
            created_at="2026-07-31T00:00:00+00:00",
            code_root=fixture["code"],
        )


def test_amendment_binds_original_corrected_and_raw_hashes(tmp_path):
    fixture = _write_fixture(tmp_path, include_all_sheet=True)
    payload = W.build_payload(
        result_root=fixture["result"],
        manifest_path=fixture["manifest"],
        coordinator_summary_path=fixture["coordinator"],
        corrected_analysis_producers=fixture["corrected"],
        amendment_producers=fixture["writer"],
        analysis_git_sha="a" * 40,
        created_at="2026-07-31T00:00:00+00:00",
        code_root=fixture["code"],
    )
    body = {
        key: value for key, value in payload.items()
        if key != "amendment_sha256"
    }
    assert payload["amendment_sha256"] == W._canonical_sha(body)
    assert payload["threshold_changed"] is False
    assert payload["raw_snn_parts_reused"] is True
    assert payload["raw_observable_audit"]["validated_part_count"] == 1
    assert payload["corrected_analysis_producer_file_sha256"] == (
        fixture["corrected"]
    )
    assert payload["amendment_producer_file_sha256"] == fixture["writer"]
