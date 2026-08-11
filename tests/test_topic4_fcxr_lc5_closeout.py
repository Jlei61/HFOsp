"""FCXR-LC5 closeout contracts: resource guard, refractory adjudication, scale admissibility."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.topic4_fcxr_lc5 import (
    admissible_target_activation,
    lock_load_scales,
    refractory_ceiling_report,
    resource_stop_reason,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_topic4_fcxr_lc5.py"
_spec = importlib.util.spec_from_file_location("lc5_runner", _SCRIPT)
LC5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LC5)


# --- design §12 swap guard -------------------------------------------------------------------


def _guard(swap_delta, *, rss_delta_gib=0.0):
    return resource_stop_reason(
        swap_used_mib=705.0 + swap_delta,
        swap_baseline_mib=705.0,
        self_rss_gib=6.79 + rss_delta_gib,
        self_rss_baseline_gib=6.79,
    )


def test_swap_guard_is_quiet_below_the_hold_threshold():
    assert _guard(255.9)["action"] is None


def test_swap_guard_holds_new_submissions_at_256_mib():
    assert _guard(256.0)["action"] == "HOLD_NEW_SUBMISSIONS"
    assert _guard(511.9)["action"] == "HOLD_NEW_SUBMISSIONS"


def test_swap_guard_terminates_the_worker_at_512_mib():
    assert _guard(512.0)["action"] == "TERMINATE_NEWEST_WORKER"


def test_swap_guard_would_have_fired_on_the_observed_u1a_growth():
    """705.0 -> 2001.5 MiB is the peak actually logged by the U1a capture."""

    got = _guard(2001.50390625 - 705.01171875)
    assert got["action"] == "TERMINATE_NEWEST_WORKER"
    assert got["swap_delta_mib"] == pytest.approx(1296.492, abs=1e-3)


def test_swap_guard_reports_own_resident_growth_separately_from_the_verdict():
    """Attribution must stay visible: the same swap delta stops regardless of who caused it."""

    sibling_driven = _guard(600.0, rss_delta_gib=0.0)
    self_driven = _guard(600.0, rss_delta_gib=0.6)
    assert sibling_driven["action"] == self_driven["action"] == "TERMINATE_NEWEST_WORKER"
    assert sibling_driven["self_rss_delta_mib"] == pytest.approx(0.0)
    assert self_driven["self_rss_delta_mib"] == pytest.approx(614.4)


def test_swap_guard_rejects_an_inverted_threshold_pair():
    with pytest.raises(ValueError, match="kill_delta_mib"):
        resource_stop_reason(
            swap_used_mib=1.0, swap_baseline_mib=0.0,
            self_rss_gib=1.0, self_rss_baseline_gib=1.0,
            hold_delta_mib=512.0, kill_delta_mib=256.0,
        )


def test_swap_guard_rejects_non_finite_readings():
    with pytest.raises(ValueError, match="swap readings"):
        resource_stop_reason(
            swap_used_mib=float("nan"), swap_baseline_mib=0.0,
            self_rss_gib=1.0, self_rss_baseline_gib=1.0,
        )


# --- refractory ceiling adjudication ---------------------------------------------------------


def test_refractory_report_uses_the_hard_single_cell_ceiling():
    got = refractory_ceiling_report(np.full(10, 250.0), tau_ref_ms=2.0)
    assert got["ceiling_hz"] == pytest.approx(500.0)
    assert got["mean_ceiling_ratio"] == pytest.approx(0.5)
    assert got["near_ceiling_fraction"] == pytest.approx(0.0)


def test_refractory_report_counts_cells_pinned_at_the_wall():
    rates = np.concatenate([np.full(368, 455.0), np.full(632, 100.0)])
    got = refractory_ceiling_report(rates, tau_ref_ms=2.0)
    assert got["near_ceiling_threshold_hz"] == pytest.approx(450.0)
    assert got["near_ceiling_fraction"] == pytest.approx(0.368)


def test_refractory_report_separates_a_bounded_carrier_from_a_saturated_source():
    """A 50 Hz population and a 448 Hz population must not read the same."""

    bounded = refractory_ceiling_report(np.full(1000, 50.0), tau_ref_ms=2.0)
    saturated = refractory_ceiling_report(np.full(1000, 448.0), tau_ref_ms=2.0)
    assert bounded["mean_ceiling_ratio"] == pytest.approx(0.10)
    assert saturated["mean_ceiling_ratio"] == pytest.approx(0.896)


def test_a_population_mean_just_under_the_wall_does_not_hide_pinned_cells():
    """The observed 20-22 s mean (447.4 Hz) sits just below 0.9*ceiling while a third of the
    individual cells are already at or above it -- the mean alone must not clear the source."""

    rates = np.concatenate([np.full(368, 455.0), np.full(632, 443.0)])
    got = refractory_ceiling_report(rates, tau_ref_ms=2.0)
    assert got["mean_hz"] < got["near_ceiling_threshold_hz"]
    assert got["near_ceiling_fraction"] == pytest.approx(0.368)


def test_registered_saturation_ceiling_is_the_stricter_of_the_two_lines():
    """The substrate's registered saturation line (0.5*1000/tau_ref_E = 250 Hz) catches a state
    that the hard 500 Hz refractory wall alone still lets through."""

    rates = np.full(1000, 447.4)
    got = refractory_ceiling_report(rates, tau_ref_ms=2.0, sat_ceiling_hz=250.0)
    assert got["near_ceiling_fraction"] == 0.0            # below 0.9 * 500 Hz
    assert got["above_sat_ceiling_fraction"] == 1.0       # but past the registered line
    assert got["mean_sat_ceiling_ratio"] == pytest.approx(1.7896)


def test_registered_saturation_ceiling_is_optional_and_validated():
    assert "registered_sat_ceiling_hz" not in refractory_ceiling_report(
        np.full(4, 10.0), tau_ref_ms=2.0
    )
    with pytest.raises(ValueError, match="sat_ceiling_hz"):
        refractory_ceiling_report(np.full(4, 10.0), tau_ref_ms=2.0, sat_ceiling_hz=0.0)


def test_refractory_report_rejects_a_negative_rate_field():
    with pytest.raises(ValueError, match="finite and non-negative"):
        refractory_ceiling_report(np.array([-1.0, 2.0]), tau_ref_ms=2.0)


# --- scale admissibility ---------------------------------------------------------------------


def test_admissible_target_is_bound_by_the_fastest_cell():
    rates = np.array([10.0, 53.0, 110.33333587646484])
    got = admissible_target_activation(rates, r_hi_ref_hz=53.0)
    assert got == pytest.approx(53.0 / 110.33333587646484)
    assert got < 0.5  # the pre-registered target is inadmissible on this field


def test_target_at_the_supremum_is_excluded_and_just_below_it_is_admissible():
    rates = np.array([10.0, 53.0, 110.33333587646484])
    sup = admissible_target_activation(rates, r_hi_ref_hz=53.0)
    at_sup = lock_load_scales(r_hi_ref_hz=53.0, per_cell_rate_hz=rates, target_activation=sup)
    below = lock_load_scales(
        r_hi_ref_hz=53.0, per_cell_rate_hz=rates, target_activation=sup * (1.0 - 1e-9)
    )
    assert at_sup["q_star_max"] >= 1.0 and at_sup["divergent_fraction"] > 0.0
    assert below["q_star_max"] < 1.0 and below["divergent_fraction"] == 0.0


def test_locked_target_is_inadmissible_on_the_captured_high_reference_field():
    """The five divergent cells are a gate failure, not a rounding artefact."""

    rates = np.full(32000, 53.0)
    rates[:5] = np.array([110.33, 108.67, 108.33, 107.67, 106.67])
    got = lock_load_scales(r_hi_ref_hz=53.0, per_cell_rate_hz=rates, target_activation=0.5)
    assert got["q_star_max"] > 1.0
    assert got["divergent_fraction"] > 0.0
    assert got["admissible"] is False


def test_q99_passing_alone_never_makes_the_field_admissible():
    """q99 and max are two gates, not interchangeable ones."""

    rates = np.full(1000, 53.0)
    rates[0] = 200.0
    got = lock_load_scales(r_hi_ref_hz=53.0, per_cell_rate_hz=rates, target_activation=0.5)
    assert got["q_star_q99"] < 0.90
    assert got["admissible"] is False


def test_admissible_target_rejects_a_silent_field():
    with pytest.raises(ValueError, match="at least one active cell"):
        admissible_target_activation(np.zeros(10), r_hi_ref_hz=53.0)


# --- the written STOP has to bind mechanically ------------------------------------------------


def test_stop_sentinel_blocks_a_forward_stage(tmp_path, monkeypatch):
    """A stage added later -- U2 above all -- must refuse to start while the stop file exists."""

    monkeypatch.setattr(LC5, "OUT", tmp_path)
    (tmp_path / LC5.SCALE_STOP).write_text("{}")
    with pytest.raises(SystemExit, match="blocked by"):
        LC5._assert_no_stop("u2")


def test_stop_sentinel_leaves_the_adjudication_stages_runnable(tmp_path, monkeypatch):
    """The stop must stay reproducible, so the read-only stages are exempt by name."""

    monkeypatch.setattr(LC5, "OUT", tmp_path)
    (tmp_path / LC5.SCALE_STOP).write_text("{}")
    for stage in LC5.STOP_EXEMPT_STAGES:
        LC5._assert_no_stop(stage)


def test_no_stop_file_blocks_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(LC5, "OUT", tmp_path)
    LC5._assert_no_stop("u2")


def test_guard_breach_writes_a_stop_record_and_actually_raises(tmp_path, monkeypatch):
    """The unit thresholds are not the contract on their own -- the breach must stop the worker."""

    monkeypatch.setattr(LC5, "OUT", tmp_path)
    monkeypatch.setattr(LC5, "_RESOURCE_BASELINE", {"swap_used_mib": 705.0})
    row = {
        "stage": "U1_CHUNK", "swap_used_mib": 2001.5, "mem_available_gib": 143.7,
        "sibling_topic4_python_count": 37,
        "guard": resource_stop_reason(
            swap_used_mib=2001.5, swap_baseline_mib=705.0,
            self_rss_gib=6.79, self_rss_baseline_gib=6.79,
        ),
    }
    with pytest.raises(SystemExit, match="RESOURCE_STOP"):
        LC5._enforce_resource_guard(row)
    written = json.loads((tmp_path / "RESOURCE_STOP.json").read_text())
    assert written["status"] == "RESOURCE_STOP"
    assert written["sibling_topic4_python_count"] == 37


def test_guard_is_silent_below_the_kill_line(tmp_path, monkeypatch):
    monkeypatch.setattr(LC5, "OUT", tmp_path)
    monkeypatch.setattr(LC5, "_RESOURCE_BASELINE", {"swap_used_mib": 705.0})
    LC5._enforce_resource_guard({
        "stage": "U1_CHUNK", "swap_used_mib": 1000.0, "mem_available_gib": 200.0,
        "sibling_topic4_python_count": 0,
        "guard": resource_stop_reason(
            swap_used_mib=1000.0, swap_baseline_mib=705.0,
            self_rss_gib=6.79, self_rss_baseline_gib=6.79,
        ),
    })
    assert not (tmp_path / "RESOURCE_STOP.json").exists()


def test_terminal_rows_opt_out_of_enforcement():
    """A stop written after the bundle is published would destroy a finished result."""

    import inspect
    src = inspect.getsource(LC5.stage_capture)
    for terminal in ('_append_resource("U1_DONE"', '_append_resource("U1_FAILED"'):
        line = next(ln for ln in src.splitlines() if terminal in ln)
        assert "enforce=False" in line


def test_every_declared_stage_is_wired_into_the_dispatcher():
    """STAGES is the argparse choice list; an unwired name would fail only at runtime."""

    for stage in LC5.STAGES:
        assert callable(getattr(LC5, f"stage_{stage}"))
    assert set(LC5.STOP_EXEMPT_STAGES) <= set(LC5.STAGES)


def test_registered_saturation_ceiling_comes_from_the_engine_not_a_literal():
    assert LC5.TAU_REF_E_MS == pytest.approx(2.0)
    assert LC5.SAT_CEILING_HZ == pytest.approx(250.0)


def test_analysis_windows_start_from_the_pre_locked_supports():
    """The first two windows are exactly the supports the design fixed before the capture ran."""

    got = LC5._analysis_windows(11000.0, 22000.0)
    assert got[0] == ("baseline", 7000.0, 11000.0)
    assert got[2] == ("high_reference", 12000.0, 15000.0)
    assert got[-1][2] == 22000.0
    assert all(hi > lo for _, lo, hi in got)
