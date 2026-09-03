import json

import pytest

from scripts.plot_topic4_zm_phase_diagram import render


def _pair(q, arm_pair):
    low_label, high_label, pair_label = arm_pair
    return {
        "q_clamp": q,
        "eta_m": 0.02,
        "noise_seed": 9101,
        "low_start_label": low_label,
        "high_start_label": high_label,
        "pair_label": pair_label,
        "future_noise_sha256": "same",
        "low_median_rate_hz": 55.0,
        "high_median_rate_hz": 390.0,
        "low_active_fraction": 0.2,
        "high_active_fraction": 1.0,
        "low_sheet_fraction": 0.2,
        "high_sheet_fraction": 1.0,
        "low_json": "/low.json",
        "high_json": "/high.json",
    }


def test_renderer_writes_all_formats_and_preserves_claim_boundary(tmp_path):
    payload = {
        "status": "SPATIAL_ZM_PHASE_DIAGRAM_AGGREGATED",
        "phase_config_sha256": "frozen",
        "pairs": [
            _pair(0.82, ("LOW", "LOW", "LOW_MONOSTABLE_CANDIDATE")),
            _pair(0.79, ("LOW", "TONIC_HIGH", "BISTABLE_CANDIDATE")),
        ],
    }
    stem = tmp_path / "phase"
    metadata = render(payload, stem)
    for extension in ("png", "pdf", "svg"):
        assert stem.with_suffix(f".{extension}").is_file()
        assert metadata["outputs"][extension]["sha256"]
    assert "no unstable analytic branch" in metadata["claim_boundary"]
    assert metadata["outcome_audit"][1]["D"] == pytest.approx(0.21)
    json.dumps(metadata)


def test_renderer_marks_unsampled_cross_grid_cells_as_missing(tmp_path):
    q_only = _pair(0.825, ("LOW", "INTERMEDIATE", "MIXED_OR_UNRESOLVED"))
    q_only["eta_m"] = 0.0
    payload = {
        "status": "SPATIAL_ZM_PHASE_DIAGRAM_AGGREGATED",
        "phase_config_sha256": "frozen",
        "pairs": [
            q_only,
            _pair(0.79, ("INTERMEDIATE", "TONIC_HIGH",
                         "MIXED_OR_UNRESOLVED")),
        ],
    }
    metadata = render(payload, tmp_path / "sparse_phase")
    assert metadata["n_missing_phase_cells"] == 2
    missing = [row for row in metadata["outcome_audit"]
               if row.get("coverage") == "missing"]
    assert len(missing) == 2
