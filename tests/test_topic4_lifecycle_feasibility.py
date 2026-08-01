"""Contract tests for pre-flight ictal-lifecycle design diagnostics.

The regression block at the bottom is the load-bearing part: each case replays the
locked constants of a mechanism round that actually ran and asserts the diagnostic
would have exposed the corresponding scale risk before the first timestep.  No
test is allowed to turn a risk flag into a mechanism non-existence claim.
"""
import math

import pytest

from src.topic4_lifecycle_feasibility import (
    CheckResult,
    brake_authority,
    operating_point_headroom,
    screen_mechanism,
    slow_variable_reversal,
    timescale_separation,
)


# --------------------------------------------------------------------- authority
def test_brake_authority_uses_the_refractory_capped_ceiling_not_an_observed_rate():
    """The ceiling must come from 1/tau_ref; an observed rate would understate it."""
    res = brake_authority(
        name="m",
        gain_mv_per_unit=0.001,
        tau_accum_ms=500.0,
        tau_ref_ms=2.0,
        v_th_mv=17.5,
        v_reset_mv=11.0,
    )
    assert res.detail["max_sustained_rate_hz"] == pytest.approx(500.0)
    assert res.detail["accumulator_ceiling"] == pytest.approx(250.0)
    assert res.detail["brake_ceiling_mv"] == pytest.approx(0.25)
    assert res.detail["reset_to_threshold_gap_mv"] == pytest.approx(6.5)
    assert res.detail["authority_ratio"] == pytest.approx(0.25 / 6.5)


def test_brake_authority_passes_when_the_ceiling_spans_the_gap():
    res = brake_authority(
        name="strong",
        gain_mv_per_unit=0.1,
        tau_accum_ms=500.0,
        tau_ref_ms=2.0,
        v_th_mv=17.5,
        v_reset_mv=11.0,
    )
    assert res.passed
    assert res.detail["authority_ratio"] > 1.0


def test_brake_authority_rejects_a_non_spiking_gap():
    with pytest.raises(ValueError, match="v_th_mv must exceed v_reset_mv"):
        brake_authority(
            name="bad",
            gain_mv_per_unit=1.0,
            tau_accum_ms=500.0,
            tau_ref_ms=2.0,
            v_th_mv=11.0,
            v_reset_mv=11.0,
        )


def test_brake_authority_rejects_non_positive_timescales():
    with pytest.raises(ValueError, match="tau_ref_ms"):
        brake_authority(
            name="bad",
            gain_mv_per_unit=1.0,
            tau_accum_ms=500.0,
            tau_ref_ms=0.0,
            v_th_mv=17.5,
            v_reset_mv=11.0,
        )


# ---------------------------------------------------------------------- reversal
def test_reversal_detects_a_latch_when_the_ictal_target_is_beyond_the_entry_point():
    """Z/M z: interictal 0.75, entry at 0.30, ictal target 0.0 -> never turns around."""
    res = slow_variable_reversal(
        name="z", u_interictal=0.75, u_entry=0.30, u_inf_ictal=0.0
    )
    assert not res.passed
    assert res.detail["entry_direction"] == -1.0
    assert res.detail["reversal_margin"] > 0.0
    assert "latch" in res.reason


def test_reversal_passes_when_the_ictal_target_sits_back_inside_the_interictal_basin():
    res = slow_variable_reversal(
        name="permittivity", u_interictal=0.75, u_entry=0.30, u_inf_ictal=0.60
    )
    assert res.passed
    assert res.detail["reversal_margin"] < 0.0


def test_reversal_handles_an_increasing_entry_direction():
    """Accumulating variables (extracellular K+) enter by rising, not falling."""
    latch = slow_variable_reversal(
        name="K_up", u_interictal=3.0, u_entry=8.0, u_inf_ictal=12.0
    )
    turns = slow_variable_reversal(
        name="K_up_pumped", u_interictal=3.0, u_entry=8.0, u_inf_ictal=5.0
    )
    assert latch.detail["entry_direction"] == 1.0
    assert not latch.passed
    assert turns.passed


def test_reversal_at_exactly_the_entry_point_is_a_latch_not_a_pass():
    """Marginal case: a target sitting exactly on the entry point never crosses back."""
    res = slow_variable_reversal(
        name="marginal", u_interictal=0.75, u_entry=0.30, u_inf_ictal=0.30
    )
    assert not res.passed


def test_reversal_requires_a_defined_entry_direction():
    with pytest.raises(ValueError, match="entry direction"):
        slow_variable_reversal(
            name="degenerate", u_interictal=0.5, u_entry=0.5, u_inf_ictal=0.1
        )


# ------------------------------------------------------------------- timescales
def test_timescale_reports_ratchet_and_hold_independently():
    res = timescale_separation(
        name="probe",
        tau_recover_ms=650.0,
        interictal_event_interval_ms=500.0,
        target_ictal_duration_ms=20000.0,
    )
    assert res.detail["accumulation_factor"] == pytest.approx(
        1.0 / (1.0 - math.exp(-500.0 / 650.0))
    )
    assert res.detail["hold_ratio"] == pytest.approx(650.0 / 20000.0)
    assert not res.detail["ratchet_ok"]
    assert not res.detail["hold_ok"]


def test_timescale_ratchet_alone_fails_the_check():
    """Long enough to hold the ictal state, but still ratcheting on the IED train."""
    res = timescale_separation(
        name="ratchet_only",
        tau_recover_ms=40000.0,
        interictal_event_interval_ms=500.0,
        target_ictal_duration_ms=20000.0,
    )
    assert res.detail["hold_ok"]
    assert not res.detail["ratchet_ok"]
    assert not res.passed


def test_timescale_hold_alone_fails_the_check():
    """Fast enough to reset between IEDs, far too fast to terminate a seizure."""
    res = timescale_separation(
        name="hold_only",
        tau_recover_ms=80.0,
        interictal_event_interval_ms=500.0,
        target_ictal_duration_ms=20000.0,
    )
    assert res.detail["ratchet_ok"]
    assert not res.detail["hold_ok"]
    assert not res.passed


def test_timescale_passes_only_when_both_bars_clear():
    res = timescale_separation(
        name="both",
        tau_recover_ms=30000.0,
        interictal_event_interval_ms=120000.0,
        target_ictal_duration_ms=20000.0,
    )
    assert res.passed
    assert res.detail["ratchet_ok"] and res.detail["hold_ok"]


# --------------------------------------------------------------------- headroom
def test_headroom_reports_interval_room_above_the_refractory_floor():
    res = operating_point_headroom(
        name="core", target_rate_hz=439.23, tau_ref_ms=2.0
    )
    assert res.detail["refractory_ceiling_hz"] == pytest.approx(500.0)
    assert res.detail["refractory_occupancy"] == pytest.approx(0.87846)
    assert res.detail["isi_headroom_ms"] == pytest.approx(1e3 / 439.23 - 2.0)
    assert not res.passed


def test_headroom_passes_at_a_rate_with_room_below_the_ceiling():
    res = operating_point_headroom(
        name="all_sheet", target_rate_hz=149.66, tau_ref_ms=2.0
    )
    assert res.passed
    assert res.detail["refractory_occupancy"] < 0.5


def test_headroom_rejects_a_degenerate_occupancy_bar():
    with pytest.raises(ValueError, match="max_occupancy must be < 1"):
        operating_point_headroom(
            name="bad", target_rate_hz=100.0, tau_ref_ms=2.0, max_occupancy=1.0
        )


# ---------------------------------------------------------------------- verdict
def _passing_check(name="ok"):
    return CheckResult(name=name, passed=True, reason="ok", detail={})


def test_screen_reports_risks_without_calling_the_mechanism_infeasible():
    verdict = screen_mechanism(
        "mixed",
        [
            _passing_check("a"),
            CheckResult(name="b", passed=False, reason="nope", detail={}),
            _passing_check("c"),
        ],
    )
    assert verdict["verdict"] == "diagnostic_risks_present"
    assert verdict["failed_checks"] == ["b"]
    assert verdict["n_failed"] == 1
    assert "not an infeasibility proof" in verdict["interpretation"]


def test_screen_never_claims_a_result_when_every_check_passes():
    """No heuristic flags is still not evidence for a lifecycle leg."""
    verdict = screen_mechanism("clean", [_passing_check("a"), _passing_check("b")])
    assert verdict["verdict"] == "no_diagnostic_flags"
    assert "not evidence" in verdict["interpretation"].lower()
    assert "diagnostic only" in verdict["claim_boundary"].lower()


def test_screen_rejects_an_empty_or_mistyped_check_list():
    with pytest.raises(ValueError, match="at least one check"):
        screen_mechanism("empty", [])
    with pytest.raises(TypeError, match="CheckResult"):
        screen_mechanism("wrong", [{"name": "a", "passed": True}])


# ================================================================= regressions
# Each case replays constants that were locked before a round that actually ran
# and asserts that the corresponding scale risk is visible before simulation.
def test_regression_zm_adaptation_brake_scale_is_flagged_before_simulating():
    """Z/M `m`: eta_m=0.001, tau_adp=500 ms, tau_ref=2 ms, core V_th=17.5, V_reset=11.

    Source: scripts/run_zm_snn_native_exit.py (TAU_ADP, ETA_M), src/snn_engine/params.py
    (tau_ref_E, V_reset), scripts/run_m4_phaseplane.py (CORE_MEAN).
    """
    check = brake_authority(
        name="m",
        gain_mv_per_unit=0.001,
        tau_accum_ms=500.0,
        tau_ref_ms=2.0,
        v_th_mv=17.5,
        v_reset_mv=11.0,
    )
    assert not check.passed
    assert check.detail["authority_ratio"] == pytest.approx(0.0385, abs=5e-4)
    verdict = screen_mechanism("zm_adaptation", [check])
    assert verdict["verdict"] == "diagnostic_risks_present"
    assert "not an infeasibility proof" in verdict["interpretation"]


def test_regression_zm_inhibitory_efficacy_is_a_latch():
    """Z/M `z`: z_inf = H(I_th_EI - I_I) is 0 for every elevated-inhibition state.

    Baseline z sits near 0.75 because I_th_EI is the q75 of the interictal E-cell
    inhibitory current; the observed entry crossing is z ~ 0.30 (lifecycle_seed1.json
    z_core_final for the bounded arms).
    """
    check = slow_variable_reversal(
        name="z", u_interictal=0.75, u_entry=0.30, u_inf_ictal=0.0
    )
    assert not check.passed
    assert screen_mechanism("zm_efficacy", [check])["verdict"] == "diagnostic_risks_present"


def test_regression_zm_divisive_pool_has_authority_but_no_hold():
    """Z/M `S_G`: alpha_G=16 gives 94% recurrent removal, but tau_S=80 ms cannot hold."""
    hold = timescale_separation(
        name="S_G",
        tau_recover_ms=80.0,
        interictal_event_interval_ms=500.0,
        target_ictal_duration_ms=20000.0,
    )
    assert hold.detail["ratchet_ok"], "S_G does reset between interictal events"
    assert not hold.detail["hold_ok"], "80 ms supplies little post-activity hold"
    verdict = screen_mechanism("zm_divisive_pool", [hold])
    assert verdict["verdict"] == "diagnostic_risks_present"
    assert "not inability to affect a driven state" in hold.reason


def test_regression_phase_c_carrier_search_had_no_refractory_headroom():
    """Phase C characterised a 435-443 Hz core against a 500 Hz refractory ceiling."""
    for rate in (435.55, 439.23, 442.58):
        check = operating_point_headroom(name="core", target_rate_hz=rate, tau_ref_ms=2.0)
        assert not check.passed
        assert check.detail["refractory_occupancy"] > 0.86
        assert check.detail["isi_headroom_ms"] < 0.31
        assert "carrier is not ruled out" in check.reason


def test_regression_fcxr_hyb1_potassium_ratchets_against_the_interictal_train():
    """FCXR-HYB1: tau_K=650 ms against interictal events every 400-600 ms."""
    for interval, floor in ((400.0, 2.1), (500.0, 1.8), (600.0, 1.6)):
        check = timescale_separation(
            name="delta_K",
            tau_recover_ms=650.0,
            interictal_event_interval_ms=interval,
            target_ictal_duration_ms=20000.0,
        )
        assert not check.detail["ratchet_ok"]
        assert check.detail["accumulation_factor"] > floor


def test_regression_all_four_diagnostics_report_risks_without_a_no_go():
    """The registered substrate carries risk flags, not an analytic non-existence proof."""
    verdict = screen_mechanism(
        "zm_registered_substrate",
        [
            brake_authority(
                name="m",
                gain_mv_per_unit=0.001,
                tau_accum_ms=500.0,
                tau_ref_ms=2.0,
                v_th_mv=17.5,
                v_reset_mv=11.0,
            ),
            slow_variable_reversal(
                name="z", u_interictal=0.75, u_entry=0.30, u_inf_ictal=0.0
            ),
            timescale_separation(
                name="S_G",
                tau_recover_ms=80.0,
                interictal_event_interval_ms=500.0,
                target_ictal_duration_ms=20000.0,
            ),
        ],
    )
    assert verdict["verdict"] == "diagnostic_risks_present"
    assert verdict["n_failed"] == 3
    assert "not an infeasibility proof" in verdict["interpretation"]
