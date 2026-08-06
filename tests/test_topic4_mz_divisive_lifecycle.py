"""Pure lifecycle-classifier, resource-gate, and import-safe runner tests."""
import os
import subprocess
import sys

import numpy as np
import pytest

from src.topic4_mz_divisive_lifecycle import (
    LifecycleThresholds,
    analyze_lifecycle,
    audit_lifecycle_against_reference,
    safe_worker_count,
)


DT = 5.0


def _trace(parts):
    return np.concatenate([np.full(int(ms / DT), value, float) for ms, value in parts])


def test_classifier_separates_flat_bounded_plateau_from_bursting():
    plateau = _trace([(1000, 1.0), (3000, 45.0)])
    t = np.arange(0, 3000, DT) / 1000.0
    burst = np.r_[np.full(int(1000 / DT), 1.0), 42.0 + 30.0 * np.sin(2 * np.pi * 4.0 * t)]
    a = analyze_lifecycle(plateau, DT, baseline_rate_hz=1.0)
    b = analyze_lifecycle(burst, DT, baseline_rate_hz=1.0)
    assert a["phenotype"] == "bounded_plateau"
    assert b["phenotype"] == "bounded_bursting"
    assert b["burst_peak_count"] >= 4


def test_classifier_detects_termination_and_return():
    t = np.arange(0, 2500, DT) / 1000.0
    rate = np.r_[
        np.full(int(1000 / DT), 1.0),
        45.0 + 25.0 * np.sin(2 * np.pi * 3.0 * t),
        np.full(int(2500 / DT), 1.0),
    ]
    out = analyze_lifecycle(rate, DT, baseline_rate_hz=1.0)
    assert out["phenotype"] == "terminate_bursting"
    assert out["returned_to_baseline"] is True
    assert out["offset_ms"] is not None


def _slowoff_reference(duration_ms=6000):
    n = int(duration_ms / DT)
    rate = np.full(n, 0.025)
    for start_ms in range(400, duration_ms, 400):
        i0 = int(start_ms / DT)
        rate[i0 : i0 + int(40 / DT)] = 40.0
    return rate


def test_strict_audit_recognizes_short_burst_long_gap_train_as_macrostate():
    reference = _slowoff_reference()
    low = np.full(int(1000 / DT), 0.025)
    one_cycle = np.r_[np.full(int(100 / DT), 60.0), np.zeros(int(150 / DT))]
    train = np.tile(one_cycle, 20)  # 4 Hz, 5 s
    out = audit_lifecycle_against_reference(
        np.r_[low, train],
        DT,
        reference_rate_hz=reference,
        reference_dt_ms=DT,
    )
    assert out["strict_phenotype"] == "bounded_recruited_bursting"
    assert out["rhythmic_bursting"] is True
    assert out["burst_peak_hz"] == pytest.approx(4.0, rel=0.05)


def test_strict_audit_does_not_call_noisy_plateau_bursting():
    rng = np.random.default_rng(12)
    reference = _slowoff_reference()
    noisy = np.clip(rng.normal(40.0, 15.0, int(5000 / DT)), 0.0, None)
    out = audit_lifecycle_against_reference(
        np.r_[np.full(int(1000 / DT), 0.025), noisy],
        DT,
        reference_rate_hz=reference,
        reference_dt_ms=DT,
    )
    assert out["strict_phenotype"] == "bounded_recruited_nonrhythmic"
    assert out["burst_peak_power_ratio"] < 0.10


def test_strict_return_rejects_flat_four_hz_tail_but_accepts_returning_event_train():
    reference = _slowoff_reference()
    high = np.full(int(2000 / DT), 45.0)
    flat = np.full(int(3000 / DT), 4.0)
    rejected = audit_lifecycle_against_reference(
        np.r_[np.full(int(1000 / DT), 0.025), high, flat],
        DT,
        reference_rate_hz=reference,
        reference_dt_ms=DT,
    )
    assert rejected["returned_to_same_seed_slowoff"] is False
    assert rejected["late_returning_event_count"] == 0

    tail = _slowoff_reference(3000)
    accepted = audit_lifecycle_against_reference(
        np.r_[np.full(int(1000 / DT), 0.025), high, tail],
        DT,
        reference_rate_hz=reference,
        reference_dt_ms=DT,
    )
    assert accepted["returned_to_same_seed_slowoff"] is True
    assert accepted["late_returning_event_count"] >= 1


def test_classifier_runaway_verdict_has_priority():
    rate = _trace([(1000, 1.0), (1000, 150.0)])
    out = analyze_lifecycle(rate, DT, baseline_rate_hz=1.0, runaway_ms=1200.0)
    assert out["phenotype"] == "runaway"
    assert out["runaway_ms"] == 1200.0


def test_short_interictal_transients_do_not_count_as_recruited_state():
    rate = _trace([(1000, 1.0), (100, 60.0), (1000, 1.0), (100, 55.0), (1000, 1.0)])
    out = analyze_lifecycle(rate, DT, baseline_rate_hz=1.0)
    assert out["phenotype"] == "interictal_like"


def test_safe_worker_count_preserves_reserve_and_caps():
    assert safe_worker_count(20, 30, 180.0, 6.5, reserve_gib=96.0, hard_cap=12) == 10
    assert safe_worker_count(20, 30, 100.0, 6.5, reserve_gib=96.0, hard_cap=12) == 0
    assert safe_worker_count(3, 2, 180.0, 6.5, reserve_gib=96.0, hard_cap=12) == 2
    with pytest.raises(ValueError):
        safe_worker_count(2, 3, 180.0, 0.0)


def test_threshold_validation_rejects_impossible_contract():
    with pytest.raises(ValueError, match="burst_band_hz"):
        LifecycleThresholds(burst_band_hz=(20.0, 0.5)).validate()


def test_runner_requires_explicit_confirmation_and_creates_no_result(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(root, "scripts", "run_topic4_mz_divisive_lifecycle.py")
    proc = subprocess.run(
        [sys.executable, script, "containment"], capture_output=True, text=True, cwd=root
    )
    assert proc.returncode == 2
    assert "pass --confirm-run" in proc.stderr
