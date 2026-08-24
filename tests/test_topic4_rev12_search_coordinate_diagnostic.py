import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.diagnose_topic4_rev12_search_coordinates import (
    advisory_status,
    agreement_summary,
    project_coordinates,
    scale_predictivity,
)


ROOT = Path(__file__).resolve().parents[1]


def test_agreement_summary_reports_sign_and_rank_predictivity():
    result = agreement_summary(
        np.asarray([-2.0, -1.0, 1.0, 3.0]),
        np.asarray([-4.0, -2.0, 2.0, 6.0]),
    )
    assert result["sign_agreement_fraction"] == 1.0
    assert result["spearman"] == 1.0
    assert np.isclose(result["cosine"], 1.0)


def test_scale_predictivity_pairs_amplitudes_within_anchor_direction():
    records = []
    for anchor in range(2):
        for direction in range(2):
            for scale, amplitude in enumerate((0.08, 0.16)):
                records.append({
                    "anchor": anchor, "direction": direction,
                    "scale": scale, "amplitude": amplitude,
                    "slopes": {
                        endpoint: float(anchor + direction + amplitude)
                        for endpoint in (
                            "objective_utility", "patient_utility", "kmeans_utility",
                            "ood_utility", "compound_utility", "direction_utility",
                        )
                    },
                })
    result = scale_predictivity(records)
    assert result["n_anchor_direction_pairs"] == 4
    assert result["endpoints"]["direction_utility"]["sign_agreement_fraction"] == 1.0
    assert result["endpoints"]["direction_utility"]["spearman"] == 1.0


def test_surface_projection_recovers_coordinates_and_residual_fraction():
    vectors = np.asarray([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    target = vectors @ np.asarray([2.0, -0.5])
    result = project_coordinates(target, vectors)
    assert np.allclose(result["coordinates"], [2.0, -0.5])
    assert result["relative_residual_norm"] < 1e-12
    assert np.isclose(result["projected_energy_fraction"], 1.0)


def test_advisory_requires_scale_seed_and_out_of_surface_prediction():
    endpoint = {
        "objective_utility": {"sign_agreement_fraction": 0.8, "spearman": 0.8},
        "direction_utility": {"sign_agreement_fraction": 0.8, "spearman": 0.8},
    }
    result = advisory_status(
        {"endpoints": endpoint},
        {"endpoints": endpoint},
        {"endpoints": endpoint},
        {
            "scale_sign_agreement_min": 0.75,
            "stage_t_spearman_min": 0.5,
            "seed_sign_agreement_min": 0.67,
            "required_endpoints": ["objective_utility", "direction_utility"],
        },
    )
    assert result["status"] == "LOCAL_FOUR_DIRECTION_GRADIENT_ACTIONABLE"
    failed = {key: dict(value) for key, value in endpoint.items()}
    failed["direction_utility"] = {
        "sign_agreement_fraction": 0.2, "spearman": 0.8,
    }
    result = advisory_status(
        {"endpoints": endpoint},
        {"endpoints": failed},
        {"endpoints": endpoint},
        result["thresholds"],
    )
    assert result["status"] == "LOCAL_FOUR_DIRECTION_GRADIENT_NOT_ACTIONABLE"
    assert "direction_utility:seed_sign" in result["failed_advisory_checks"]


def test_search_coordinate_script_is_directly_invocable():
    result = subprocess.run(
        [sys.executable, "scripts/diagnose_topic4_rev12_search_coordinates.py", "--help"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
