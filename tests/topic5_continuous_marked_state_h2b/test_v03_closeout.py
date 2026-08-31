from __future__ import annotations

from scripts.topic5_continuous_marked_state_h2b.build_v03_closeout import (
    _direction_fraction,
    _phenotype_direction,
)


def test_direction_fraction_uses_patients_not_seeds() -> None:
    summary = {
        "cohort_direction": {
            "T": {"favourable": 3, "total": 5},
        },
    }
    assert _direction_fraction(summary, "T") == 0.6
    assert _direction_fraction(summary, "missing") is None


def test_phenotype_direction_uses_finite_observed_target_rows() -> None:
    summary = {
        "patient_rows": [
            {
                "target_name": "ied_ictal_reuse_observed",
                "state_minus_observation_loss": -0.1,
                "evaluation_tier": "primary_chronological",
            },
            {
                "target_name": "ied_ictal_reuse_observed",
                "state_minus_observation_loss": 0.2,
                "evaluation_tier": "sensitivity_loso",
            },
            {
                "target_name": "ied_ictal_reuse_observed",
                "state_minus_observation_loss": float("nan"),
                "evaluation_tier": "sensitivity_loso",
            },
            {
                "target_name": "ied_ictal_reuse_margin",
                "state_minus_observation_loss": -1.0,
                "evaluation_tier": "primary_chronological",
            },
            {
                "target_name": "ied_ictal_reuse_observed",
                "state_minus_observation_loss": -1.0,
                "evaluation_tier": "descriptive_case_series",
            },
        ],
    }
    assert _phenotype_direction(summary) == (1, 2)
