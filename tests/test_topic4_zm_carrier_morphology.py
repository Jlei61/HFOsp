import numpy as np

from src.topic4_zm_carrier_morphology import characterize_confirmation


def test_coarse_rate_cannot_turn_a_narrowband_carrier_into_a_fixed_point():
    fs = 4000.0
    t = np.arange(int(4 * fs)) / fs
    phases = np.linspace(0.0, 1.2, 8)
    lfp = np.column_stack([
        np.sin(2 * np.pi * 47.0 * t + ph)
        + 0.08 * np.random.default_rng(i).standard_normal(t.size)
        for i, ph in enumerate(phases)
    ])
    rate_25ms = np.full(160, 145.0)
    kymo = np.tile(np.linspace(0.8, 1.2, 24)[:, None], (1, rate_25ms.size))

    out = characterize_confirmation(
        rate_25ms, lfp, fs, burn_in_ms=250.0, kymo_axial=kymo, bin_ms=25.0
    )

    assert out["coarse_rate_label"] == "tonic_at_25ms"
    assert out["readout_temporal_class"] == "narrowband_readout_candidate"
    assert 45.0 < out["dominant_frequency_median_hz"] < 49.0
    assert out["dominant_frequency_agreement"] >= 0.75
    assert "fixed" not in out["coarse_rate_label"]


def test_independent_broadband_noise_is_not_called_narrowband():
    fs = 2000.0
    rng = np.random.default_rng(4)
    lfp = rng.standard_normal((int(5 * fs), 10))
    rate_25ms = 80.0 + rng.standard_normal(200)
    kymo = np.abs(rng.standard_normal((24, rate_25ms.size)))

    out = characterize_confirmation(
        rate_25ms, lfp, fs, burn_in_ms=250.0, kymo_axial=kymo, bin_ms=25.0
    )

    assert out["readout_temporal_class"] == "broadband_or_asynchronous_readout"
    assert out["spectral_entropy_median"] > 0.75
    assert 0.0 <= out["dominant_frequency_agreement"] <= 1.0


def test_characterization_rejects_mismatched_time_axes():
    with np.testing.assert_raises(ValueError):
        characterize_confirmation(
            np.ones(10), np.ones((100, 3)), 1000.0,
            burn_in_ms=0.0, kymo_axial=np.ones((24, 9)), bin_ms=25.0,
        )
