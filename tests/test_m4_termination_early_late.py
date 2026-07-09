"""TDD for Task 6: retrigger_verdict 4-label split (reignite_bounded/attenuated/runaway/not_run).

The change is ONLY the return strings + docstring; the signature and math are unchanged.
Tests use the verified signature: retrigger_verdict(termination_class, post_af, baseline=, ref_peak=)
which computes amp = ref_peak - baseline internally.
"""
import numpy as np
import pytest
from src.sef_hfo_m4_termination import retrigger_verdict, run_cell_with_retrigger


def _post(peak, tail):
    """Build post-kick activity trace: leading peak impulse (bins 50-100), tail at the end."""
    a = np.full(400, 0.0)
    a[50:100] = peak
    a[-80:] = tail
    return a


def test_not_run_when_not_terminate_clean():
    """class != terminate_clean returns not_run before baseline/ref_peak are ever needed."""
    assert retrigger_verdict("persist", _post(0.6, 0.0)) == "not_run"


def test_attenuated_when_kick_fizzles():
    """Peak < baseline + 0.5*amp -> refractory (fizzle, was 'fail')."""
    v = retrigger_verdict("terminate_clean", _post(0.1, 0.0), baseline=0.0, ref_peak=1.0)
    assert v == "attenuated"


def test_runaway_when_tail_stays_high():
    """Tail >= baseline + 0.8*amp -> stayed high (runaway re-ignition, was 'fail')."""
    v = retrigger_verdict("terminate_clean", _post(0.9, 0.9), baseline=0.0, ref_peak=1.0)
    assert v == "runaway"


def test_reignite_bounded_when_fires_then_falls():
    """Rose > baseline+0.5*amp and tail fell < baseline+0.8*amp -> bounded re-event (was 'pass')."""
    v = retrigger_verdict("terminate_clean", _post(0.9, 0.0), baseline=0.0, ref_peak=1.0)
    assert v == "reignite_bounded"


# ==================================================================== Task 7: additive early retrigger window
def _clean_trace(n):
    """Deterministic terminate_clean trace of length n: quiet, plateau [100,400)@0.9, then quiet.
    Absolute positions -> _clean_trace(a)[:a] == _clean_trace(b)[:a], so pre-probe identity holds."""
    a = np.zeros(int(n)); a[100:400] = 0.9
    return a


def test_early_refractory_late_recovers():
    calls = {"n": 0}

    def run_fn(t_kick2, boost, min_T):
        if t_kick2 is None:
            return {"af": _clean_trace(1200), "baseline_af": 0.0}    # NOTE: no 'offset_ms' key
        calls["n"] += 1
        af = _clean_trace(int(min_T) + 10)                          # pass-2 must run to >= min_T
        i = int(round(t_kick2))
        if t_kick2 > 5000.0:                                        # late -> bounded re-event
            af[i:i + 60] = 0.9; af[i + 60:i + 200] = 0.0
        else:                                                       # early -> fizzle
            af[i:i + 60] = 0.05
        return {"af": af, "baseline_af": 0.0}

    out = run_cell_with_retrigger(run_fn, bin_ms=1.0, recovery_ms=5000.0, recovery_factor=2.0,
                                  probe_window_ms=3000.0, baseline_af=0.0, early_offset_ms=750.0)
    assert out["termination_class"] == "terminate_clean"
    assert out["offset_ms"] == pytest.approx(400.0)                 # P1-5: from classify, not run_fn
    assert out["retrigger_early"] == "attenuated"
    assert out["retrigger_probe"] == "reignite_bounded"            # late == the existing single probe
    assert calls["n"] == 2                                          # one early + one late pass-2


def test_early_offset_none_is_m4_2_parity():
    def run_fn(t_kick2, boost, min_T):
        if t_kick2 is None:
            return {"af": _clean_trace(1200), "baseline_af": 0.0}
        af = _clean_trace(int(min_T) + 10); i = int(round(t_kick2))
        af[i:i + 60] = 0.9; af[i + 60:i + 200] = 0.0
        return {"af": af, "baseline_af": 0.0}
    out = run_cell_with_retrigger(run_fn, bin_ms=1.0, recovery_ms=5000.0, recovery_factor=2.0,
                                  probe_window_ms=3000.0, baseline_af=0.0)   # early_offset_ms default None
    assert out["retrigger_probe"] == "reignite_bounded"
    assert "retrigger_early" not in out                            # no early probe when not requested


def test_pre_probe_identity_enforced():
    def bad_run_fn(t_kick2, boost, min_T):
        if t_kick2 is None:
            return {"af": _clean_trace(1200), "baseline_af": 0.0}
        af = _clean_trace(int(min_T) + 10); af[0] += 1.0           # corrupt the pre-probe prefix
        return {"af": af, "baseline_af": 0.0}
    with pytest.raises(RuntimeError, match="pre-probe identity"):
        run_cell_with_retrigger(bad_run_fn, bin_ms=1.0, baseline_af=0.0, early_offset_ms=750.0)
