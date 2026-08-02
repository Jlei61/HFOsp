import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_topic4_zm_lifecycle_m_panel.py"
SPEC = importlib.util.spec_from_file_location("topic4_zm_lifecycle_m_analysis", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def _row(onset=500.0, offset=None):
    return {"episode": {"onset_ms": onset, "offset_ms": offset}}


def test_m_effect_requires_durable_offset_and_paired_gm0_difference():
    censored = _row(offset=None)
    exited = _row(offset=3500.0)
    got = M.paired_m_effect(exited, censored)
    assert got["causal_exit_candidate"] is True
    assert got["status"] == "offset_vs_censored_gM0"
    assert M.paired_m_effect(censored, censored)["causal_exit_candidate"] is False


def test_m_effect_rejects_prevention_and_small_duration_shift():
    assert M.paired_m_effect(_row(onset=None), _row(offset=None))["status"] == "prevention_or_no_onset"
    baseline = _row(onset=500.0, offset=4000.0)
    small = _row(onset=500.0, offset=3500.0)
    assert M.paired_m_effect(small, baseline)["causal_exit_candidate"] is False
    large = _row(onset=500.0, offset=2500.0)
    assert M.paired_m_effect(large, baseline)["causal_exit_candidate"] is True


def test_slow_trace_metrics_record_offset_and_z_recovery(tmp_path):
    np.savez(
        tmp_path / "traces.npz",
        trace_m_core_mean=np.asarray([1.0, 2.0, 3.0, 2.0]),
        trace_z_core_mean=np.asarray([0.8, 0.5, 0.4, 0.7]),
        trace_S_G=np.asarray([0.1, 0.4, 0.2, 0.1]),
        trace_phi_core_mean=np.asarray([0.0, 0.2, 0.1, 0.0]),
        trace_i2e_resource_mean=np.asarray([1.0, 0.8, 0.7, 0.9]),
        trace_i_adaptation_mean=np.asarray([0.0, 0.3, 0.2, 0.0]),
        fine_time_ms=np.asarray([0.0, 1.0, 2.0, 3.0]),
        fine_core_rate_hz=np.asarray([0.0, 10.0, 20.0, 30.0]),
        fine_all_e_rate_hz=np.asarray([0.0, 2.0, 4.0, 6.0]),
    )
    got = M._trace_metrics(tmp_path, {"onset_ms": 2.0, "offset_ms": 2.0})
    assert got["m_peak"] == 3.0
    assert got["z_core_at_offset"] == 0.4
    assert got["z_core_post_offset_recovery"] == pytest.approx(0.3)
    assert got["S_G_maximum"] == 0.4
    assert got["post_onset_core_mean_hz"] == 25.0
    assert got["post_onset_all_E_mean_hz"] == 5.0


def test_paired_m_continuous_response_preserves_censored_state_effects():
    baseline = {
        "core_mean_hz": 100.0,
        "all_E_mean_hz": 40.0,
        "median_energy_gain_db": 20.0,
        "energy_occupancy_6db": 0.8,
        "post_entry_core_cv": 0.2,
        "spatial_effective_rank": 1.1,
        "common_mode_pc1_fraction": 0.98,
        "slow_trace": {"z_core_final": 0.1, "z_core_minimum": 0.05, "m_peak": 100.0},
    }
    treated = {
        **baseline,
        "core_mean_hz": 60.0,
        "all_E_mean_hz": 20.0,
        "median_energy_gain_db": 15.0,
        "slow_trace": {"z_core_final": 0.3, "z_core_minimum": 0.1, "m_peak": 95.0},
    }
    got = M.paired_m_continuous_response(treated, baseline)
    assert got["status"] == "paired"
    assert got["delta_core_mean_hz"] == -40.0
    assert got["ratio_core_mean_hz"] == pytest.approx(0.6)
    assert got["delta_median_energy_gain_db"] == -5.0
    assert got["delta_z_core_final"] == pytest.approx(0.2)
