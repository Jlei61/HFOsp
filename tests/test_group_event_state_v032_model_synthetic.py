"""Task 8: synthetic residual-positive and H-only-null assays (S1-S4)."""

from __future__ import annotations

import numpy as np
import torch

from src.topic5_group_event_state.v032_model.config import ModelConfig
from src.topic5_group_event_state.v032_model.synthetic import (
    NULL_MAX_FALSE_POSITIVE_REPLICATES,
    apply_synthetic_targets,
    hidden_component_r2_against_h,
    judge_synthetic,
    make_synthetic_targets,
    run_synthetic_assay,
)
from tests.test_group_event_state_v032_model_toyutil import make_toy_bundle

CPU = torch.device("cpu")


def test_s1_s3_generator_is_reproducible_hidden_from_h_and_null_has_no_effect():
    bundle, _ = make_toy_bundle(seed=20, planted_beta=0.0)
    pos = make_synthetic_targets(bundle, horizon=1800.0, beta=0.35, dispersion_r=5.0,
                                 generator_seed=1, noise_seed=2)
    again = make_synthetic_targets(bundle, horizon=1800.0, beta=0.35, dispersion_r=5.0,
                                   generator_seed=1, noise_seed=2)
    other_noise = make_synthetic_targets(bundle, horizon=1800.0, beta=0.35, dispersion_r=5.0,
                                         generator_seed=1, noise_seed=3)
    assert np.array_equal(pos.counts, again.counts)                                  # S3
    assert not np.array_equal(pos.counts, other_noise.counts)
    assert np.allclose(pos.log_mu_true, bundle.log_mu_h(1800.0) + 0.35 * pos.z)
    train = bundle.anchor_mask("state_train", 1800.0)
    assert abs(pos.z[train].mean()) < 1e-6 and abs(pos.z[train].std() - 1.0) < 1e-6
    assert hidden_component_r2_against_h(bundle, pos) < 0.5                          # S1
    null = make_synthetic_targets(bundle, horizon=1800.0, beta=0.0, dispersion_r=5.0,
                                  generator_seed=1, noise_seed=2)
    assert np.allclose(null.log_mu_true, bundle.log_mu_h(1800.0))                    # S2
    replaced = apply_synthetic_targets(bundle, pos)
    h_i = bundle.horizon_index(1800.0)
    assert np.array_equal(replaced.counts[:, h_i], pos.counts)
    assert np.array_equal(replaced.counts[:, 0], bundle.counts[:, 0])
    assert replaced is not bundle and replaced.fingerprint["synthetic"]["beta"] == 0.35


def test_s4_judgement_thresholds_are_echoed_and_applied():
    def fake(gain_low, shifted_mean, gain_mean):
        return {"dev_test": {"contrasts": {
            "h_minus_correct": {"ci_low": gain_low, "mean": gain_mean},
            "shifted_minus_correct": {"mean": shifted_mean},
        }}}

    good = judge_synthetic([fake(0.02, 0.01, 0.1), fake(0.03, 0.02, 0.1), fake(-0.01, 0.0, 0.0)], "positive")
    assert good["pass"] is True and "criteria" in good
    bad = judge_synthetic([fake(0.02, -0.01, 0.1), fake(-0.03, 0.02, 0.1), fake(-0.01, 0.0, 0.0)], "positive")
    assert bad["pass"] is False
    nulls = [fake(-0.01, 0.0, 0.001)] * 5 + [fake(0.01, 0.0, 0.02)]
    assert judge_synthetic(nulls, "null")["pass"] is True
    assert judge_synthetic(nulls + [fake(0.01, 0.0, 0.02)], "null")["n_false_positive_replicates"] == 2
    assert judge_synthetic(nulls + [fake(0.01, 0.0, 0.02)], "null")["pass"] is False
    assert NULL_MAX_FALSE_POSITIVE_REPLICATES == 1


def test_assay_runs_end_to_end_on_toy_and_recovers_positive(tmp_path):
    bundle, _ = make_toy_bundle(seed=21, planted_beta=0.0)
    cfg = ModelConfig(max_steps=200, min_steps=40, validate_every=10, patience=6, alpha_freeze_steps=10)
    out = run_synthetic_assay(bundle, cfg, kind="positive", replicate=0, seed=3, device=CPU,
                              out_dir=tmp_path, beta=0.8, dispersion_r=8.0)
    assert out["kind"] == "positive" and out["status"] == "complete"
    assert out["dev_test"]["contrasts"]["h_minus_correct"]["mean"] > 0
    assert out["dev_test"]["contrasts"]["shifted_minus_correct"]["mean"] > 0
    assert out["h_source"] == "provisional_local" and out["synthetic"]["beta"] == 0.8
    assert (tmp_path / "assay.json").exists()
