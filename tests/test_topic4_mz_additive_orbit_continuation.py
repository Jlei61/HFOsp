from pathlib import Path

import numpy as np
import pytest

from src.topic4_mz_additive_orbit_continuation import (
    integrate_additive_return,
    predict_section_state,
    shoot_additive_cycle,
)
from src.topic4_spatial_slowfast_stage0c import PoolParameters
from src.topic4_spatial_slowfast_stage0c_transfer import ExtendedSiegertTransfer
from src.topic4_spatial_slowfast_stage0e import (
    SectionDefinition,
    integrate_to_returns_batch,
)


ROOT = Path(__file__).resolve().parents[1]
TRANSFER_PATH = (
    ROOT
    / "results/topic4_sef_hfo/spatial_slowfast_topology/"
    "stage0c_transfer_support_audit_v1_1/extended_transfer_extra_fine.npz"
)
SHOOTING_PATH = (
    ROOT
    / "results/topic4_sef_hfo/spatial_slowfast_topology/"
    "stage0e_poincare_floquet_audit/per_point/z_0p85_alpha_15/shooting_iterates.npz"
)
REPORT_PATH = (
    ROOT
    / "results/topic4_sef_hfo/spatial_slowfast_topology/"
    "stage0f_smooth_transfer_variational_certificate_v1_1/"
    "stage0f_v1_1_variational_summary.json"
)


def _locked_inputs():
    with np.load(TRANSFER_PATH, allow_pickle=False) as payload:
        transfer = ExtendedSiegertTransfer(
            payload["mu_axis"],
            payload["sigma_axis"],
            payload["log_integral_table"],
            name="extra_fine",
        )
    with np.load(SHOOTING_PATH, allow_pickle=False) as payload:
        fixed = np.asarray(payload["base_state"][-1], dtype=float)
    import json

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    scales = np.asarray(report["parameter_points"][0]["state_scales"], dtype=float)
    return transfer, fixed, scales


@pytest.mark.skipif(
    not all(path.is_file() for path in (TRANSFER_PATH, SHOOTING_PATH, REPORT_PATH)),
    reason="requires locked Stage0E/F orbit artifacts",
)
def test_zero_additive_return_is_stage0e_map_parity():
    transfer, fixed, _ = _locked_inputs()
    params = PoolParameters(0.85, 15.0, 1.1, 1.0)
    section = SectionDefinition(max_return_ms=1200.0)
    expected = integrate_to_returns_batch(
        fixed[None, :], [params], transfer, dt_ms=0.125, n_returns=1, section=section
    )
    observed = integrate_additive_return(
        fixed, params, transfer, 0.0, dt_ms=0.125, section=section
    )
    assert expected["valid"][0]
    assert observed.valid
    assert abs(observed.return_time_ms - expected["return_time_ms"][0, 0]) < 1e-12
    np.testing.assert_array_equal(observed.crossing_state, expected["return_state"][0, 0])


@pytest.mark.skipif(
    not all(path.is_file() for path in (TRANSFER_PATH, SHOOTING_PATH, REPORT_PATH)),
    reason="requires locked Stage0E/F orbit artifacts",
)
def test_additive_shooting_tracks_cycle_beyond_old_1200ms_window():
    transfer, fixed, scales = _locked_inputs()
    params = PoolParameters(0.85, 15.0, 1.1, 1.0)
    result = shoot_additive_cycle(
        fixed,
        params,
        transfer,
        0.31,
        dt_ms=0.125,
        section=SectionDefinition(max_return_ms=4000.0),
        scales=scales,
        max_iterations=20,
    )
    assert result["accepted"]
    assert 1200.0 < result["validated_period_ms"] < 2500.0


def test_secant_predictor_preserves_locked_section():
    section = SectionDefinition(max_return_ms=4000.0)
    older = np.linspace(0.01, 0.18, 9)
    current = older + 0.01
    predicted = predict_section_state(current, older, 0.3, 0.2, 0.1, section)
    np.testing.assert_allclose(predicted[:8], (current + (current - older))[:8])
    assert predicted[8] == section.level

