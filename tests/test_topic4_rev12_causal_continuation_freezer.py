import json
from pathlib import Path

import numpy as np

from scripts.freeze_topic4_rev12_causal_continuation import (
    METRICS,
    finite_difference_slopes,
    manual_smooth_spline_control,
    metric_scales,
    proposal_weights,
)


ROOT = Path(__file__).resolve().parents[1]


def _row(values):
    return {"selection_objective": {
        "matched_patient_loss": values[0],
        "kmeans_direction_loss": values[1],
        "ood_fraction": values[2],
        "compound_fraction": values[3],
        "causal_direction_loss": values[4],
    }}


def test_finite_difference_and_downhill_proposal_have_declared_sign():
    by_id = {}
    gradient = np.asarray([1.0, -2.0, 0.5, -0.25])
    for direction in range(4):
        for amplitude_index, amplitude in enumerate((0.08, 0.16)):
            for sign, suffix in ((-1.0, "m"), (1.0, "p")):
                values = np.full(5, sign * amplitude * gradient[direction])
                by_id[
                    f"stage_i_a00_d{direction:02d}_s{amplitude_index:02d}_{suffix}"
                ] = _row(values)
    finite = finite_difference_slopes(
        by_id, anchor_index=0, amplitudes=[0.08, 0.16], n_directions=4,
    )
    for metric in METRICS:
        assert np.allclose(finite["slopes"][metric], gradient)
    direction = proposal_weights(
        finite["slopes"], {metric: 1.0 for metric in METRICS},
        {metric: 1.0 for metric in METRICS},
    )
    assert float(direction @ gradient) < 0.0


def test_metric_scales_are_positive_for_constant_endpoints():
    scales = metric_scales([_row(np.ones(5)), _row(np.ones(5))])
    assert set(scales) == set(METRICS)
    assert all(value > 0.0 for value in scales.values())


def test_manual_control_is_continuous_accurate_and_not_selectable():
    stage = {"engine": {"L": 20.0}}
    placement = {
        "center": np.asarray([10.0, 10.0]),
        "axis_unit_vec": np.asarray([1.0, 0.0]),
        "source_centroid": np.asarray([4.0, 10.0]),
        "sink_centroid": np.asarray([16.0, 10.0]),
    }
    field = manual_smooth_spline_control(
        stage, placement, n_basis=18, degree=3, grid_per_axis=41,
    )
    assert field["field_type"] == "spline_continuous"
    assert field["component_count"] is None
    assert field["residual_coordinates"]["selection_eligible"] is False
    assert field["capacity_control_audit"]["latent_surface_correlation"] >= 0.995


def test_continuation_config_keeps_slow_and_connectivity_mechanisms_closed():
    config = json.loads(
        (ROOT / "config/topic4_rev12_nd_causal_continuation.json").read_text()
    )
    assert config["scientific_role"] == (
        "development_only_causal_continuation_and_capacity"
    )
    assert config["continuation_design"]["expected_candidate_count"] == 19
    assert config["cascade_objective"]["k2_support_weight"] == 0.0
    assert "selection_network_seeds" not in config["search"]
    assert "confirmation_network_seeds" not in config["search"]
    assert "EE, E-to-I and Z/M remain off" in config["claim_boundary"]
