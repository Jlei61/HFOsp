"""Hard-stop guards: no target may reach the model that predicts it."""

import numpy as np
import torch

from src.topic5_group_event_state.dataset import HISTORY_LAGS, recent_history_features
from src.topic5_group_event_state.model import (
    ContinuousState,
    StateConfig,
)


def test_recent_history_never_contains_the_interval_it_must_predict():
    # Irregular intervals so an accidental off-by-one is visible.
    t = np.array([0.0, 10.0, 25.0, 45.0, 70.0, 110.0, 111.0])  # all intervals distinct
    part = np.zeros((t.size, 3), dtype=bool)
    delay = np.zeros((t.size, 3), dtype=np.float32)
    feats = recent_history_features(t, part, delay, np.arange(t.size), lags=(1,))
    lag1_dt = np.expm1(feats[:, 0])
    target = np.diff(t)
    for i in range(2, t.size):
        assert not np.isclose(lag1_dt[i], target[i - 1]), (
            f"event {i}: baseline was handed its own target interval {target[i-1]}"
        )
        assert np.isclose(lag1_dt[i], target[i - 2])


def test_history_summaries_exclude_the_current_event():
    t = np.arange(6.0)
    part = np.zeros((6, 2), dtype=bool)
    part[3, 0] = True  # only event 3 has a participant, and only one
    delay = np.zeros((6, 2), dtype=np.float32)
    feats = recent_history_features(t, part, delay, np.arange(6), lags=(1,))
    # column 1 is the lag-1 rolling group size; at event 3 it must still be 0.
    assert feats[3, 1] == 0.0
    assert feats[4, 1] == 1.0


def test_state_evolution_is_the_only_place_dt_enters():
    cfg = StateConfig(d_fast=4, d_slow=2)
    state = ContinuousState(cfg, d_event=8)
    z_f = torch.randn(1, 4)
    z_s = torch.randn(1, 2)
    a_f, a_s = state.evolve(z_f, z_s, torch.tensor([0.0]))
    torch.testing.assert_close(a_f, z_f)
    torch.testing.assert_close(a_s, z_s)
    b_f, _ = state.evolve(z_f, z_s, torch.tensor([1e6]))
    # a very long gap must relax the fast state to its bias, not preserve it
    assert torch.allclose(b_f, state.bias_fast.unsqueeze(0), atol=1e-4)


def test_slow_timescales_can_actually_reach_hours():
    # softplus(log tau) would cap tau near 20 s; exp(clamp(.)) must not.
    cfg = StateConfig()
    state = ContinuousState(cfg, d_event=8)
    _tau_f, tau_s = state.taus()
    assert float(tau_s.max()) > 3600.0
    with torch.no_grad():
        state.log_tau_slow.fill_(np.log(1e9))
    _tau_f, tau_s = state.taus()
    assert abs(float(tau_s.max()) - cfg.tau_slow_range_s[1]) < 1.0


def test_memoryless_state_carries_nothing_between_events():
    cfg = StateConfig(d_fast=4, d_slow=2, persistent=False)
    state = ContinuousState(cfg, d_event=8)
    z_f = torch.randn(1, 4) * 10
    z_s = torch.randn(1, 2) * 10
    out_f, out_s = state.update(z_f, z_s, torch.randn(1, 8))
    init_f, init_s = state.initial(1, torch.device("cpu"))
    torch.testing.assert_close(out_f, init_f)
    torch.testing.assert_close(out_s, init_s)


def test_fully_masked_event_does_not_produce_nan():
    """One event with no valid contact must not poison the whole chunk."""

    from src.topic5_group_event_state.model import DataShape, EncoderConfig, EventEncoder

    shape = DataShape(
        n_contacts=4, n_bands=2, n_band_features=5, n_cross_band_pairs=1, n_views=2,
        n_waveform_samples=32, n_envelope_bins=16, n_background_features=3,
        band_available=(True, True),
    )
    cfg = EncoderConfig(use_exact_delay=True, use_tied_groups=True,
                        use_waveform=True, use_multiband=True, use_geometry=False)
    torch.manual_seed(0)
    encoder = EventEncoder(cfg, shape, None)
    ok = torch.ones(3, 4, dtype=torch.bool)
    ok[1] = False  # this event resolved no contact at all
    batch = {
        "participation": torch.ones(3, 4, dtype=torch.bool),
        "contact_ok": ok,
        "rel_delay": torch.rand(3, 4) * 0.05,
        "tied_group_id": torch.zeros(3, 4, dtype=torch.long),
        "legacy_rank": torch.zeros(3, 4, dtype=torch.long),
        "waveform": torch.randn(3, 4, 2, 32),
        "band_envelope": torch.rand(3, 4, 2, 16),
        "band_features": torch.randn(3, 4, 2, 5),
        "cross_band_lag": torch.randn(3, 4, 1),
    }
    event, tokens = encoder(batch)
    assert torch.isfinite(event).all()
    assert torch.isfinite(tokens).all()
