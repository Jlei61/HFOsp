import numpy as np

from scripts.analyze_topic4_rev12_joint_objective_conflict import _ridge_loo


def test_ridge_loo_recovers_predictable_directional_slopes():
    rng = np.random.default_rng(1)
    design = rng.normal(size=(40, 5))
    truth = np.asarray([0.5, -0.2, 0.1, 0.0, 0.3])
    values = design @ truth
    result = _ridge_loo(design, values, [1e-6, 1e-3, 0.1])
    assert result["loo_pearson"] > 0.99
    assert result["loo_sign_agreement_fraction"] > 0.95


def test_ridge_loo_reports_unpredictable_noise_as_weak():
    rng = np.random.default_rng(2)
    design = rng.normal(size=(24, 6))
    values = rng.normal(size=24)
    result = _ridge_loo(design, values, [0.01, 0.1, 1.0, 10.0])
    assert result["loo_pearson"] < 0.5
