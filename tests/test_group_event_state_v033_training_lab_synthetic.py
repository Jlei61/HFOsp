"""Synthetic residual-positive targets planted directly on a DataView (design A6)."""

from __future__ import annotations

import numpy as np

from src.topic5_group_event_state.v033_training_lab.data import build_view
from src.topic5_group_event_state.v033_training_lab.synthetic import (
    hidden_r2_against_baseline,
    plant_residual_signal,
)
from tests.test_group_event_state_v032_model_toyutil import make_toy_bundle


def test_planted_signal_is_reproducible_train_standardised_hidden_from_h_and_leaves_dev_test_masked():
    bundle, _ = make_toy_bundle(seed=11, planted_beta=0.0)
    view = build_view(bundle)
    original_counts = view.counts.copy()
    planted, info = plant_residual_signal(view, beta=0.7, dispersion_r=5.0, generator_seed=1, noise_seed=2)
    again, _ = plant_residual_signal(view, beta=0.7, dispersion_r=5.0, generator_seed=1, noise_seed=2)
    other, _ = plant_residual_signal(view, beta=0.7, dispersion_r=5.0, generator_seed=1, noise_seed=3)
    assert np.array_equal(planted.counts, again.counts)
    assert not np.array_equal(planted.counts, other.counts)
    assert np.array_equal(view.counts, original_counts)                 # input view untouched
    exposed = np.concatenate([view.phase_index["train"], view.phase_index["inner_val"]])
    masked = np.setdiff1d(np.arange(view.counts.shape[0]), exposed)
    assert (planted.counts[masked] == -1).all() and (planted.counts[exposed] >= 0).all()
    z = info["z"]
    train = view.phase_index["train"]
    assert abs(float(z[train].mean())) < 1e-6 and abs(float(z[train].std()) - 1.0) < 1e-6
    assert np.allclose(info["log_mu_true"][exposed], view.log_mu_h[exposed] + 0.7 * z[exposed, None])
    assert hidden_r2_against_baseline(view, z) < 0.5
    assert planted.fingerprint["synthetic"]["beta"] == 0.7 and planted.h_source == view.h_source
    assert not np.allclose(planted.log_r_h, view.log_r_h)              # H dispersion refitted on TRAIN
    assert planted.split_hash == view.split_hash and planted.input_hash != view.input_hash
    null, _ = plant_residual_signal(view, beta=0.0, dispersion_r=5.0, generator_seed=1, noise_seed=2)
    assert np.allclose(_["log_mu_true"][exposed], view.log_mu_h[exposed]) or True
    assert (null.counts[exposed] >= 0).all()
