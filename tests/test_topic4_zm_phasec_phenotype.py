"""Synthetic contracts for the Phase-C run-level phenotype taxonomy."""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.topic4_zm_phasec_phenotype as P  # noqa: E402


BIN_MS = 2.0
N = 1500


def _grids(source, n=8):
    source = np.asarray(source, float)
    # A localized but persistent focus; source_rate_hz is intentionally passed
    # separately so dilution by inactive space cannot change temporal labels.
    E = np.zeros((source.size, n, n), np.float32)
    I = np.zeros_like(E)
    E[:, 2:6, 2:6] = source[:, None, None]
    I[:, 2:6, 2:6] = 0.9 * source[:, None, None]
    area = np.full(source.size, 16 / (n * n))
    return E, I, area


def _classify(source, **kwargs):
    E, I, area = _grids(source)
    kymograph = np.zeros((len(source), 12), float)
    kymograph[:, 2:5] = np.asarray(source)[:, None]
    kymograph[:, 8:11] = np.asarray(source)[:, None]
    return P.classify_phasec_run(
        E,
        I,
        bin_ms=BIN_MS,
        source_rate_hz=source,
        active_area_fraction=area,
        kymograph=kymograph,
        axis_positions=np.linspace(-6, 6, 12),
        readout_kernel_width_mm=0.278,
        **kwargs,
    )


def test_tonic_noise_cannot_be_promoted_to_non_tonic():
    rng = np.random.default_rng(2)
    source = 80.0 + rng.normal(0, 1.0, N)
    out = _classify(source)
    assert out["bounded_gate_pass"] is True
    assert out["phenotype"] == "tonic_non_AI"
    assert out["temporal_diagnostics"]["global_modulation_fraction"] < 0.2


def test_smooth_fast_periodic_carrier_requires_ten_regular_cycles():
    t = np.arange(N) * BIN_MS * 1e-3
    source = 70.0 + 25.0 * np.sin(2 * np.pi * 12.0 * t)
    out = _classify(source)
    assert out["phenotype"] == "periodic_non_tonic_carrier"
    assert out["temporal_diagnostics"]["periodic"]["n_cycles"] >= 10
    assert out["temporal_diagnostics"]["periodic"]["period_cv"] <= 0.2
    assert out["temporal_diagnostics"]["periodic"][
        "source_phase_signature"
    ]["status"] == "ok"


def test_periodic_harmonic_rest_resets_are_not_a_sustained_carrier():
    t = np.arange(N) * BIN_MS * 1e-3
    source = 70.0 + 25.0 * np.sin(2 * np.pi * 12.0 * t)
    rest = np.zeros(N, bool)
    cycle = int(round((1000.0 / 12.0) / BIN_MS))
    rest[np.arange(cycle // 2, N, cycle)] = True
    out = _classify(source, rest_mask=rest)
    assert out["bounded_gate"]["longest_rest_dwell_ms"] < 100.0
    assert out["temporal_diagnostics"]["periodic"][
        "rest_reset_fraction"
    ] > P.DEFAULTS["maximum_periodic_rest_reset_fraction"]
    assert out["phenotype"] != "periodic_non_tonic_carrier"


def test_slow_bursting_with_shallow_troughs_is_clonic_not_hfo_train():
    t = np.arange(N) * BIN_MS * 1e-3
    phase = np.sin(2 * np.pi * 3.0 * t)
    source = 25.0 + 90.0 * np.maximum(phase, 0.0) ** 4
    out = _classify(source)
    assert out["bounded_gate"]["active_occupancy"] >= 0.8
    assert out["bounded_gate"]["longest_rest_dwell_ms"] < 100
    assert out["phenotype"] == "clonic_or_bursting_carrier"
    assert out["temporal_diagnostics"]["clonic"]["n_cycles"] >= 5


def test_separated_short_events_are_hfo_train_not_clonic_carrier():
    source = np.zeros(N)
    width = int(40 / BIN_MS)
    spacing = int(250 / BIN_MS)
    for start in range(30, N - width, spacing):
        source[start:start + width] = 130.0
    out = _classify(source)
    assert out["bounded_gate_pass"] is False
    assert out["phenotype"] == "hfo_like_relaxation_train"
    assert out["bounded_gate"]["active_occupancy"] < 0.8
    assert out["bounded_gate"]["longest_rest_dwell_ms"] >= 100


def test_rest_runaway_and_saturation_preempt_bounded_labels():
    assert _classify(np.zeros(N))["phenotype"] == "rest_or_silence"

    ramp = np.linspace(20.0, 330.0, N)
    assert _classify(ramp)["phenotype"] == "runaway"

    source = np.full(N, 420.0)
    E, I, _ = _grids(source)
    whole_sheet = np.ones(N)
    out = P.classify_phasec_run(
        E,
        I,
        bin_ms=BIN_MS,
        source_rate_hz=source,
        active_area_fraction=whole_sheet,
    )
    assert out["phenotype"] == "runaway"

    # Core ceiling occupancy alone is not a refractory diagnosis.
    out = _classify(
        np.full(N, 80.0),
        saturation_fraction=0.70,
        refractory_fraction=0.10,
    )
    assert out["phenotype"] == "tonic_non_AI"
    out = _classify(
        np.full(N, 80.0),
        saturation_fraction=0.70,
        refractory_fraction=0.90,
    )
    assert out["phenotype"] == "refractory_saturated"


def test_bounded_but_drifting_candidate_fails_stationarity_gate():
    source = np.linspace(40.0, 130.0, N)
    out = _classify(source)
    assert out["phenotype"] == "probabilistically_indeterminate"
    assert out["temporal_diagnostics"]["stationarity_ok"] is False


def _traveling_kymograph(n_t=180, n_x=12):
    K = np.zeros((n_t, n_x), float)
    for j in range(n_x):
        onset = 20 + 5 * j
        K[onset:onset + 30, j] = 1.0
    return K, np.linspace(-6, 6, n_x)


def test_ordered_first_passage_is_a_spatial_relay_modifier():
    K, pos = _traveling_kymograph()
    out = P.spatial_relay_modifier(
        K, pos, bin_ms=BIN_MS, n_perm=199, rng_seed=4
    )
    assert out["is_spatial_relay"] is True
    assert out["status"] == "relay"
    assert abs(out["axial_first_passage_rho"]) > 0.9
    assert out["permutation_p"] <= 0.05


def test_simultaneous_whole_field_flash_is_never_a_relay():
    K = np.zeros((180, 12), float)
    K[40:70, :] = 1.0
    out = P.spatial_relay_modifier(
        K, np.linspace(-6, 6, 12), bin_ms=BIN_MS, n_perm=99
    )
    assert out["is_spatial_relay"] is False
    assert out["status"] == "no_relay"
    assert "simultaneous" in out["reason"] or "flash" in out["reason"]


def test_missing_or_malformed_relay_inputs_fail_closed():
    source = np.full(N, 75.0)
    E, I, area = _grids(source)
    out = P.classify_phasec_run(
        E,
        I,
        bin_ms=BIN_MS,
        source_rate_hz=source,
        active_area_fraction=area,
    )
    assert out["phenotype"] == "probabilistically_indeterminate"
    assert out["temporal_diagnostics"]["spatial_extent"]["pass"] is False
    assert out["spatial_relay"]["status"] == "not_tested"
    assert out["spatial_relay"]["is_spatial_relay"] is False

    bad = P.spatial_relay_modifier(
        np.zeros((3, 2)), np.arange(2), bin_ms=BIN_MS
    )
    assert bad["status"] == "indeterminate"
    assert bad["is_spatial_relay"] is False


def test_two_zone_extent_is_separate_from_ordered_relay():
    source = np.full(N, 75.0)
    E, I, area = _grids(source)
    K = np.zeros((N, 12), float)
    K[:, 1:4] = 75.0
    K[:, 8:11] = 75.0
    out = P.classify_phasec_run(
        E,
        I,
        bin_ms=BIN_MS,
        source_rate_hz=source,
        active_area_fraction=area,
        kymograph=K,
        axis_positions=np.linspace(-6, 6, 12),
        readout_kernel_width_mm=0.278,
    )
    assert out["phenotype"] == "tonic_non_AI"
    assert out["temporal_diagnostics"]["spatial_extent"]["pass"] is True
    assert out["spatial_relay"]["is_spatial_relay"] is False
