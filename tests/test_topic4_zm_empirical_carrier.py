"""Task 5 (spec rev3.1 §4.2/§4.3): the observation gate must separate a real broadband carrier from
the two impostors -- a sharp harmonic pulse train and a stationary global oscillator.

These are the falsifiable claims, so they are tested on synthetic fixtures where ground truth is
known. The empirical INTERVALS come from real E1146 references and are locked separately; without a
sufficient real sample the lock refuses to grade a model readout at all.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.topic4_zm_empirical_carrier as EC  # noqa: E402

FS = 2000.0
DUR = 6.0


@pytest.fixture(scope="module")
def batteries():
    out = {}
    for name, x in (("pulse", EC.synth_pulse_train(FS, DUR)),
                    ("oscillator", EC.synth_global_oscillator(FS, DUR)),
                    ("broadband", EC.synth_broadband_carrier(FS, DUR))):
        out[name] = EC.metric_battery(x, FS)
    return out


def test_pulse_train_is_caught_by_the_harmonic_comb(batteries):
    """A pulse train has huge cross-band power; only the comb statistic exposes it as harmonic."""
    assert batteries["pulse"]["harmonic_comb"] > batteries["broadband"]["harmonic_comb"] * 1.5
    assert batteries["pulse"]["spectral_entropy"] < batteries["broadband"]["spectral_entropy"]


def test_stationary_global_oscillator_is_caught_by_coherence_and_drift(batteries):
    """A fixed ~5 Hz whole-field rhythm can match occupancy and energy; it must still fail."""
    osc, bb = batteries["oscillator"], batteries["broadband"]
    assert osc["phase_coherence"] > 0.7, "the synthetic oscillator must be whole-field coherent"
    assert osc["phase_coherence"] > bb["phase_coherence"] + 0.2
    assert osc["inst_freq_drift_hz"] < bb["inst_freq_drift_hz"]


def test_broadband_carrier_passes_the_gate_built_from_those_nulls(batteries):
    """The gate is only useful if the thing it is meant to accept actually passes it."""
    ictal = [EC.metric_battery(EC.synth_broadband_carrier(FS, DUR, seed=s), FS) for s in range(4)]
    inter = [EC.metric_battery(0.25 * EC.synth_broadband_carrier(FS, DUR, seed=100 + s), FS)
             for s in range(4)]
    pulse = [EC.metric_battery(EC.synth_pulse_train(FS, DUR, seed=s), FS) for s in range(3)]
    osc = [EC.metric_battery(EC.synth_global_oscillator(FS, DUR, seed=s), FS) for s in range(3)]
    lock = EC.build_lock(ictal, inter, pulse, osc)
    assert lock["sufficient_reference_sample"]
    good = EC.evaluate_against_lock(EC.metric_battery(
        EC.synth_broadband_carrier(FS, DUR, seed=999), FS), lock)
    bad_pulse = EC.evaluate_against_lock(EC.metric_battery(
        EC.synth_pulse_train(FS, DUR, seed=999), FS), lock)
    bad_osc = EC.evaluate_against_lock(EC.metric_battery(
        EC.synth_global_oscillator(FS, DUR, seed=999), FS), lock)
    assert bad_pulse["verdict"] == "fails_observation_gate", bad_pulse["per_metric"]
    assert bad_osc["verdict"] == "fails_observation_gate", bad_osc["per_metric"]
    assert good["n_failed"] < bad_pulse["n_failed"], (good["per_metric"], bad_pulse["per_metric"])


def test_lock_refuses_to_grade_without_enough_real_references():
    lock = EC.build_lock([], [], [], [])
    assert not lock["sufficient_reference_sample"]
    out = EC.evaluate_against_lock(dict(duration_ms=5000.0), lock)
    assert out["verdict"] == "observation_layer_blocked"


def test_missing_or_nan_model_metric_fails_closed():
    ictal = [EC.metric_battery(EC.synth_broadband_carrier(FS, DUR, seed=s), FS) for s in range(4)]
    inter = [EC.metric_battery(0.25 * EC.synth_broadband_carrier(FS, DUR, seed=50 + s), FS)
             for s in range(4)]
    lock = EC.build_lock(ictal, inter, [], [])
    out = EC.evaluate_against_lock({}, lock)
    assert out["verdict"] == "fails_observation_gate"
    assert all(not p["passed"] and "fail-closed" in p["why"] for p in out["per_metric"].values())


def test_adjacent_contacts_from_one_hotspot_count_as_one_independent_contact():
    coords = np.array([[0.0, 0.0], [0.05, 0.0], [0.10, 0.0], [5.0, 0.0]])
    assert EC.independent_contacts([0, 1, 2], coords, kernel_width=0.278) == 1
    assert EC.independent_contacts([0, 1, 2, 3], coords, kernel_width=0.278) == 2
    assert EC.independent_contacts([], coords, kernel_width=0.278) == 0


def test_historical_literals_are_not_the_primary_gate():
    """Historic occupancy>=0.8 / duration>=2 s stay diagnostic comparators (spec §4.2): the lock
    must derive its own bounds from the reference classes."""
    ictal = [EC.metric_battery(EC.synth_broadband_carrier(FS, DUR, seed=s), FS) for s in range(4)]
    inter = [EC.metric_battery(0.25 * EC.synth_broadband_carrier(FS, DUR, seed=70 + s), FS)
             for s in range(4)]
    lock = EC.build_lock(ictal, inter, [], [])
    occ = lock["thresholds"]["occupancy"]
    assert "null_bound" in occ and occ["null_bound"] != 0.8
    assert lock["thresholds"]["duration_ms"].get("ictal_lo") is not None


def test_metrics_absent_from_the_observation_layer_are_declared_not_silently_dropped():
    lock = EC.build_lock([], [], [], [])
    assert set(lock["metrics_not_in_observation_layer"]) == \
        set(EC.GATE_METRICS_NOT_IN_OBSERVATION)
    assert "wavefront_velocity_variability" in lock["metrics_not_in_observation_layer"]


def test_real_reference_resolver_is_honest_about_availability():
    """Either it resolves real E1146 windows with provenance, or it returns nothing -- it must never
    fabricate a window."""
    rows = EC._block_index()
    if not rows:
        pytest.skip("E1146 raw tree not mounted here")
    assert all(r["fs"] > 300.0 for r in rows), "Nyquist must cover the 150 Hz band"
    fake_seizure = [dict(eeg_onset_epoch=rows[0]["start_epoch"] + 100.0, seizure_id="X")]
    w = EC.resolve_early_ictal_windows(fake_seizure)
    assert len(w) == 1 and w[0]["crop_start_sec"] >= 0
    assert EC.resolve_early_ictal_windows([dict(eeg_onset_epoch=0.0, seizure_id="Y")]) == []
