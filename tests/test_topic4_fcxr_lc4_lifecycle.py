import numpy as np

from src.topic4_fcxr_lc4_lifecycle import (
    adjudicate_frozen_D,
    adjudicate_nominal,
    first_ictal_bout,
    refractory_ceiling_fraction,
)


BAND = dict(
    event_rate_lo=0.1,
    event_rate_hi=2.0,
    dur_lo_ms=8.0,
    dur_hi_ms=22.0,
    part_lo=0.04,
    part_hi=0.08,
)


def _events(times):
    return [dict(index=i + 1, t_on=float(t), t_off=float(t + 10), dur_ms=10.0,
                 peak_ext=0.05, returned=True) for i, t in enumerate(times)]


def _good_regimes():
    # 10 s pre -> 3 s bounded high -> 4 s protected/transition -> 8 s returning tail.
    return (["INTERICTAL"] * 10 + ["ICTAL"] * 3 + ["SILENT"] * 2
            + ["INTERICTAL"] * 10)


def _good_events():
    return _events([1000, 4000, 7000, 9000, 18000, 20000, 22000, 24000])


def test_first_bout_requires_registered_duration():
    assert first_ictal_bout(["ICTAL", "OTHER", "ICTAL"], 500.0) is None
    assert first_ictal_bout(["INTERICTAL", "ICTAL", "ICTAL"], 500.0) == (1, 2)


def test_refractory_fraction_reads_per_cell_bout_rate():
    x = np.zeros((1000, 4), bool)
    x[::2, 0] = True  # 500 Hz at dt=1 ms, at the 2-ms refractory ceiling
    x[::10, 1] = True
    assert refractory_ceiling_fraction(
        x, dt_ms=1.0, onset_ms=0, offset_ms=1000, tau_ref_ms=2.0) == 0.25


def test_nominal_good_path_is_only_eligible_not_complete():
    out = adjudicate_nominal(
        regimes=_good_regimes(), win_ms=1000.0, events=_good_events(), total_ms=25000.0,
        reference_band=BAND, numerical_safe=True, refractory_fraction=0.0,
        pre_rate_hz=4.0, postictal_rate_hz=0.2)
    assert out["passed"] is True
    assert out["verdict"] == "F2_NOMINAL_ELIGIBLE_FOR_FROZEN_D"
    assert out["offset_ms"] == 13000.0
    assert out["n_returning_before_onset"] == 4


def test_bout_at_record_end_is_not_an_offset():
    out = adjudicate_nominal(
        regimes=["INTERICTAL"] * 10 + ["ICTAL"] * 3,
        win_ms=1000.0, events=_events([1000, 4000, 7000]), total_ms=13000.0,
        reference_band=BAND, numerical_safe=True, refractory_fraction=0.0,
        pre_rate_hz=4.0, postictal_rate_hz=0.0)
    assert out["clauses"]["autonomous_offset"] is False
    assert out["offset_ms"] is None
    assert out["passed"] is False


def test_rapid_relapse_fails_even_if_late_tail_looks_good():
    r = _good_regimes()
    r[14] = "ICTAL"
    out = adjudicate_nominal(
        regimes=r, win_ms=1000.0, events=_good_events(), total_ms=25000.0,
        reference_band=BAND, numerical_safe=True, refractory_fraction=0.0,
        pre_rate_hz=4.0, postictal_rate_hz=0.2)
    assert out["clauses"]["no_rapid_relapse"] is False
    assert out["passed"] is False


def test_silent_tail_is_not_returning_interictal():
    r = _good_regimes()
    r[-8:] = ["SILENT"] * 8
    out = adjudicate_nominal(
        regimes=r, win_ms=1000.0, events=_good_events()[:4], total_ms=25000.0,
        reference_band=BAND, numerical_safe=True, refractory_fraction=0.0,
        pre_rate_hz=4.0, postictal_rate_hz=0.2)
    assert out["clauses"]["return_window_interictal"] is False
    assert out["clauses"]["returning_reference"] is False


def test_return_distribution_failure_cannot_be_hidden_by_mean_rate():
    ev = _good_events()
    ev[-1] = dict(ev[-1], dur_ms=100.0)
    out = adjudicate_nominal(
        regimes=_good_regimes(), win_ms=1000.0, events=ev, total_ms=25000.0,
        reference_band=BAND, numerical_safe=True, refractory_fraction=0.0,
        pre_rate_hz=4.0, postictal_rate_hz=0.2)
    assert out["return_window"]["reference"]["checks"]["duration"] is False
    assert out["passed"] is False


def test_frozen_D_is_required_for_complete_label():
    good = adjudicate_frozen_D(
        regimes=["INTERICTAL"] * 12, win_ms=1000.0,
        events=_events([2500, 5000, 7500, 10000]), total_ms=12000.0, burn_ms=2000.0,
        reference_band=BAND, numerical_safe=True, refractory_fraction=0.0)
    assert good["verdict"] == "LC4_CANDIDATE_COMPLETE_LIFECYCLE"
    bad = adjudicate_frozen_D(
        regimes=["INTERICTAL"] * 9 + ["ICTAL"] * 3, win_ms=1000.0,
        events=_events([2500, 5000, 7500, 10000]), total_ms=12000.0, burn_ms=2000.0,
        reference_band=BAND, numerical_safe=True, refractory_fraction=0.0)
    assert bad["passed"] is False


def test_frozen_D_dense_tail_is_not_a_low_stable_return():
    bad = adjudicate_frozen_D(
        regimes=["INTERICTAL"] * 4 + ["DENSE"] * 8, win_ms=1000.0,
        events=_events([2500, 5000, 7500, 10000]), total_ms=12000.0, burn_ms=2000.0,
        reference_band=BAND, numerical_safe=True, refractory_fraction=0.0)
    assert bad["clauses"]["low_regime_after_burn"] is False
    assert bad["passed"] is False
