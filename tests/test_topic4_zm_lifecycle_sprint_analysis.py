import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_topic4_zm_lifecycle_sprint.py"
SPEC = importlib.util.spec_from_file_location("topic4_zm_lifecycle_sprint_analysis", SCRIPT)
A = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(A)


def test_event_free_baseline_rejects_initial_high_rate_event():
    core = np.r_[np.zeros(20), np.full(8, 80.0), np.zeros(40)]
    mask = A.event_free_baseline_bins(core)
    assert mask[:20].all()
    assert not mask[20:28].any()


def test_baseline_referenced_intensity_distinguishes_sustained_gain():
    rms = np.ones((40, 2))
    rms[20:] = 10.0
    baseline = np.zeros(40, bool)
    baseline[:20] = True
    got = A.baseline_referenced_intensity(rms, baseline, slice(20, 40))
    assert got["median_gain_db_across_contacts"] == pytest.approx(10.0)
    assert got["occupancy_above_6db"] == 1.0
    assert got["normalized_integrated_energy_per_s"] == 10.0


def test_contact_rms_uses_event_free_baseline_mean():
    raw = np.zeros((400, 1))
    raw[200:] = 4.0
    baseline = np.array([True, True, False, False])
    rms, status = A.contact_rms_from_baseline(raw, 1000.0, baseline, bin_ms=100.0)
    assert status == "insufficient_event_free_baseline"
    baseline[:] = True
    rms, status = A.contact_rms_from_baseline(raw, 1000.0, baseline, bin_ms=100.0)
    assert status == "ok" and rms.shape == (4, 1)
