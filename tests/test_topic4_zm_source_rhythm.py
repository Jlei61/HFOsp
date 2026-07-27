import numpy as np

from src.topic4_zm_source_rhythm import (
    bin_spikes_to_grid,
    characterize_source_rhythm,
    source_rhythm_authorized,
)


def test_bin_spikes_to_grid_preserves_population_rate_and_ei_shapes():
    dt_ms = 1.0
    E = np.zeros((20, 4), dtype=bool)
    I = np.zeros((20, 2), dtype=bool)
    E[[0, 2, 10, 12], [0, 1, 2, 3]] = True
    I[[1, 11], [0, 1]] = True
    posE = np.array([[0.1, 0.1], [0.2, 0.1], [0.7, 0.7], [0.8, 0.7]])
    posI = np.array([[0.1, 0.2], [0.8, 0.8]])

    out = bin_spikes_to_grid(
        E, I, posE, posI, L=1.0, dt_ms=dt_ms, bin_ms=10.0, n_grid=2
    )

    assert out["E_rate_grid"].shape == (2, 2, 2)
    assert out["I_rate_grid"].shape == (2, 2, 2)
    assert np.allclose(out["global_E_rate_hz"], [50.0, 50.0])
    assert np.allclose(out["global_I_rate_hz"], [50.0, 50.0])


def test_source_rhythm_gate_is_fail_closed_until_native_confirmation_passes():
    assert not source_rhythm_authorized({})
    assert not source_rhythm_authorized({
        "confirmation": {"status": "pending"},
        "layers": {"source_space_carrier": "provisional_carrier_window"},
    })
    assert not source_rhythm_authorized({
        "confirmation": {"status": "passed"},
        "layers": {"source_space_carrier": "provisional_carrier_window"},
    })
    assert source_rhythm_authorized({
        "confirmation": {"status": "passed"},
        "layers": {"source_space_carrier": "source_space_carrier"},
    })


def _periodic_field(phases, *, f0=47.0, duration_s=4.0, bin_ms=2.0):
    t = np.arange(int(duration_s * 1000 / bin_ms)) * bin_ms * 1e-3
    phase = np.asarray(phases).reshape(1, 4, 4)
    E = 80.0 + 35.0 * np.sin(2 * np.pi * f0 * t[:, None, None] + phase)
    I = 50.0 + 20.0 * np.sin(
        2 * np.pi * f0 * t[:, None, None] + phase + 0.7
    )
    return E, I


def test_phase_staggered_local_oscillations_are_not_called_fixed_or_asynchronous():
    phases = np.linspace(0, 2 * np.pi, 16, endpoint=False).reshape(4, 4)
    E, I = _periodic_field(phases)
    out = characterize_source_rhythm(E, I, bin_ms=2.0)

    assert out["source_temporal_class"] == "phase_staggered_periodic_candidate"
    assert 45.0 < out["dominant_frequency_median_hz"] < 49.0
    assert out["local_frequency_agreement"] > 0.9
    assert out["local_phase_locking"] > 0.9
    assert out["global_modulation_fraction"] < 0.1


def test_aligned_field_is_global_periodic_candidate():
    E, I = _periodic_field(np.zeros((4, 4)))
    out = characterize_source_rhythm(E, I, bin_ms=2.0)

    assert out["source_temporal_class"] == "global_periodic_candidate"
    assert out["global_modulation_fraction"] > 0.5


def test_independent_fluctuations_are_not_periodic_candidate():
    rng = np.random.default_rng(8)
    E = np.clip(80.0 + 20.0 * rng.standard_normal((2000, 4, 4)), 0, None)
    I = np.clip(50.0 + 15.0 * rng.standard_normal((2000, 4, 4)), 0, None)
    out = characterize_source_rhythm(E, I, bin_ms=2.0)

    assert out["source_temporal_class"] == "asynchronous_or_irregular_candidate"
    assert out["local_peak_fraction_median"] < 0.25
