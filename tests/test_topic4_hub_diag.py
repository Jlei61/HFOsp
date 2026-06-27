"""TDD tests for src.topic4_hub_diag.hub_global_recruitment.

Synthetic spike arrays only. We test the relay-timing diagnostics that
distinguish (a) corridor event dying at the hub (interictal) from
(b) hub-crossing broadcast to the global region (seizure-like).
"""

import numpy as np
import pytest

from src.topic4_hub_diag import hub_global_recruitment


def _make_record(n_timesteps, n_E):
    """All-silent boolean spike record."""
    return np.zeros((n_timesteps, n_E), dtype=bool)


def test_relay_timing_corridor_then_hub_then_global():
    """Corridor fires early (t=2), hub middle (t=6), global late (t=10).

    The relay direction must be reflected: hub after corridor (>0) and
    global after hub (>0), with exact ms = dt * (index difference).
    """
    n_timesteps, n_E = 20, 12
    spk = _make_record(n_timesteps, n_E)

    corridor_idx = np.array([0, 1, 2, 3, 4, 5])  # hub is a subset of corridor
    hub_idx = np.array([4, 5])
    global_idx = np.array([8, 9, 10])

    dt = 0.5  # ms per timestep

    # corridor (non-hub portion) onset at t=2
    spk[2, 0] = True
    spk[2, 1] = True
    # hub onset at t=6
    spk[6, 4] = True
    spk[6, 5] = True
    # global onset at t=10
    spk[10, 8] = True
    spk[10, 9] = True
    spk[10, 10] = True

    out = hub_global_recruitment(spk, hub_idx, global_idx, corridor_idx, dt)

    assert out["corridor_onset_ms"] == pytest.approx(2 * dt)
    assert out["hub_onset_ms"] == pytest.approx(6 * dt)
    assert out["global_onset_ms"] == pytest.approx(10 * dt)

    # relay direction positive
    assert out["global_first_spike_after_hub_ms"] > 0
    assert out["hub_after_corridor_ms"] > 0

    # exact ms differences
    assert out["global_first_spike_after_hub_ms"] == pytest.approx((10 - 6) * dt)
    assert out["hub_after_corridor_ms"] == pytest.approx((6 - 2) * dt)


def test_interictal_global_never_fires():
    """Interictal-like: corridor + hub fire, global cells NEVER fire."""
    n_timesteps, n_E = 20, 12
    spk = _make_record(n_timesteps, n_E)

    corridor_idx = np.array([0, 1, 2, 3, 4, 5])
    hub_idx = np.array([4, 5])
    global_idx = np.array([8, 9, 10])

    dt = 1.0

    spk[3, 0] = True   # corridor onset
    spk[7, 4] = True   # hub onset
    # global never fires

    out = hub_global_recruitment(spk, hub_idx, global_idx, corridor_idx, dt)

    assert out["global_E_spike_fraction"] == 0.0
    assert np.isnan(out["global_onset_ms"])
    assert np.isnan(out["global_first_spike_after_hub_ms"])
    # hub_after_corridor should still be a finite positive number
    assert out["hub_after_corridor_ms"] == pytest.approx((7 - 3) * dt)


def test_fractions_partial_hub_recruitment():
    """2 of 4 hub cells fire => hub_recruited_fraction == 0.5."""
    n_timesteps, n_E = 10, 10
    spk = _make_record(n_timesteps, n_E)

    corridor_idx = np.array([0, 1, 2, 3, 4, 5])
    hub_idx = np.array([2, 3, 4, 5])  # 4 hub cells
    global_idx = np.array([8, 9])

    dt = 2.0

    # 2 of the 4 hub cells fire
    spk[5, 2] = True
    spk[6, 3] = True
    # leave hub cols 4, 5 silent

    out = hub_global_recruitment(spk, hub_idx, global_idx, corridor_idx, dt)

    assert out["hub_recruited_fraction"] == 0.5


def test_corridor_and_global_fractions():
    """Sanity-check corridor and global fractions independently."""
    n_timesteps, n_E = 10, 10
    spk = _make_record(n_timesteps, n_E)

    corridor_idx = np.array([0, 1, 2, 3])  # 4 corridor cells
    hub_idx = np.array([2, 3])
    global_idx = np.array([8, 9])          # 2 global cells

    dt = 1.0

    spk[1, 0] = True  # 1 of 4 corridor cells
    spk[4, 2] = True  # another corridor (also hub) cell -> 2 of 4 corridor
    spk[8, 8] = True  # 1 of 2 global cells

    out = hub_global_recruitment(spk, hub_idx, global_idx, corridor_idx, dt)

    assert out["corridor_spike_fraction"] == pytest.approx(2 / 4)
    assert out["global_E_spike_fraction"] == pytest.approx(1 / 2)
    assert out["hub_recruited_fraction"] == pytest.approx(1 / 2)


def test_all_silent_record():
    """All-silent record => 0.0 fractions and nan onsets, no raise."""
    n_timesteps, n_E = 15, 8
    spk = _make_record(n_timesteps, n_E)

    corridor_idx = np.array([0, 1, 2])
    hub_idx = np.array([1, 2])
    global_idx = np.array([6, 7])

    dt = 0.25

    out = hub_global_recruitment(spk, hub_idx, global_idx, corridor_idx, dt)

    assert out["hub_recruited_fraction"] == 0.0
    assert out["global_E_spike_fraction"] == 0.0
    assert out["corridor_spike_fraction"] == 0.0
    assert np.isnan(out["corridor_onset_ms"])
    assert np.isnan(out["hub_onset_ms"])
    assert np.isnan(out["global_onset_ms"])
    assert np.isnan(out["global_first_spike_after_hub_ms"])
    assert np.isnan(out["hub_after_corridor_ms"])


def test_empty_region_guard():
    """Empty region index arrays => nan onset / 0.0 fraction, no raise."""
    n_timesteps, n_E = 10, 6
    spk = _make_record(n_timesteps, n_E)
    spk[2, 0] = True
    spk[3, 1] = True

    corridor_idx = np.array([0, 1])
    hub_idx = np.array([], dtype=int)       # empty hub
    global_idx = np.array([], dtype=int)    # empty global

    dt = 1.0

    out = hub_global_recruitment(spk, hub_idx, global_idx, corridor_idx, dt)

    assert out["hub_recruited_fraction"] == 0.0
    assert out["global_E_spike_fraction"] == 0.0
    assert np.isnan(out["hub_onset_ms"])
    assert np.isnan(out["global_onset_ms"])
    # corridor present and fired
    assert out["corridor_onset_ms"] == pytest.approx(2 * dt)
    assert out["corridor_spike_fraction"] == 1.0
    # differences involving nan onsets are nan
    assert np.isnan(out["global_first_spike_after_hub_ms"])
    assert np.isnan(out["hub_after_corridor_ms"])


def test_return_values_are_floats():
    """All returned values must be Python/numpy floats (or nan)."""
    n_timesteps, n_E = 10, 8
    spk = _make_record(n_timesteps, n_E)
    spk[1, 0] = True
    spk[3, 4] = True
    spk[5, 6] = True

    corridor_idx = np.array([0, 1, 2, 3, 4])
    hub_idx = np.array([3, 4])
    global_idx = np.array([6, 7])

    out = hub_global_recruitment(spk, hub_idx, global_idx, corridor_idx, 1.0)

    expected_keys = {
        "hub_recruited_fraction",
        "global_E_spike_fraction",
        "corridor_spike_fraction",
        "corridor_onset_ms",
        "hub_onset_ms",
        "global_onset_ms",
        "global_first_spike_after_hub_ms",
        "hub_after_corridor_ms",
    }
    assert set(out.keys()) == expected_keys
    for k, v in out.items():
        assert isinstance(v, float), f"{k} is not a float: {type(v)}"
