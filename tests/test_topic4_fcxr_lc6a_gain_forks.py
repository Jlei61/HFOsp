import importlib.util
import json
from pathlib import Path

import numpy as np

from src.topic4_fcxr_lc5 import SparseSpikeStream
from src.topic4_fcxr_lc6_gain import (
    active_area_mm2, binned_global_rate, paired_gain_readout, relaxation_readout,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_topic4_fcxr_lc6a_gain_forks.py"
SPEC = importlib.util.spec_from_file_location("lc6a_gain_forks", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_gain_prelock_keeps_response_independent_of_boundedness():
    lock = RUNNER._prelock()
    assert lock["selection"]["maximum_conditions"] == 2
    assert lock["selection"]["checkpoint_names"] == ["onset_plus_2s", "onset_plus_6s"]
    assert lock["interpretation"]["gain_threshold_is_a_carrier_gate"] is False
    assert lock["interpretation"]["termination_tested"] is False


def test_global_rate_and_area_use_fixed_complete_bins():
    rate = binned_global_rate(
        np.array([0, 1, 10]), n_steps=20, n_cells=10, dt_ms=1., bin_ms=10.,
    )
    np.testing.assert_allclose(rate, [20., 10.])
    maps = np.array([[11., 0., 12., np.nan], [0., 0., 15., 0.]])
    area = active_area_mm2(maps, np.ones(4), rate_threshold_hz=10., sheet_size_mm=2.)
    np.testing.assert_allclose(area, [2., 1.])


def test_relaxation_requires_both_rate_and_area_to_stay_low():
    rate = np.r_[np.ones(10), np.zeros(30)]
    area = np.array([1., 1., 0., 0.])
    result = relaxation_readout(
        rate, area, pulse_ms=50., rate_bin_ms=10., area_bin_ms=100.,
        fraction=.1, hold_ms=200.,
    )
    assert result["relaxation_ms_after_pulse"] == 150.
    assert result["right_censored"] is False


def test_paired_readout_uses_registered_500ms_window_and_returns_continuous_fields():
    sham = np.zeros(100)
    probe = np.r_[np.ones(50), np.zeros(50)]
    area0 = np.zeros(10)
    area1 = np.r_[np.ones(5), np.zeros(5)]
    result = paired_gain_readout(
        sham, probe, area0, area1, pulse_l2_current=2., pulse_ms=50.,
        susceptibility_window_ms=500., rate_bin_ms=10., area_bin_ms=100.,
        relaxation_fraction=.1, relaxation_hold_ms=200.,
    )
    assert result["global_rate_l1_response_hz_s"] == .5
    assert result["susceptibility_hz_s_per_l2_current_s"] == 5.
    assert result["active_area_l1_deviation_mm2_s"] == .5


def test_checkpoint_load_excludes_registered_saturation():
    # One spike per cell in the final 1 s gives 1 Hz and no refractory tail.
    stream = SparseSpikeStream(
        steps=np.arange(10, dtype=np.int64) + 1000,
        cells=np.arange(10, dtype=np.int64), n_steps=2000, n_cells=10,
    )
    old = RUNNER.NAT.U2.DT_MS
    RUNNER.NAT.U2.DT_MS = 1.
    try:
        got = RUNNER._previous_second_checkpoint_load(stream, 2000., tau_ref_ms=2.)
    finally:
        RUNNER.NAT.U2.DT_MS = old
    assert got == {"global_rate_hz": 1.0, "near_refractory_fraction": 0.0}


def test_autopilot_runs_gain_only_after_fixed_phenotype_map():
    source = (ROOT / "scripts/run_topic4_fcxr_lc6a_dynamics_autopilot.sh").read_text()
    assert source.index("aggregate_topic4_fcxr_lc6a_phenotypes.py") < source.index(
        "scripts/run_topic4_fcxr_lc6a_gain_forks.py lock"
    )
    assert "run_pool gain" in source
    assert "scripts/run_topic4_fcxr_lc6a_gain_forks.py finalize" in source
