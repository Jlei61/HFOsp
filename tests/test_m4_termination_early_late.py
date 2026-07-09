"""TDD for Task 6: retrigger_verdict 4-label split (reignite_bounded/attenuated/runaway/not_run).

The change is ONLY the return strings + docstring; the signature and math are unchanged.
Tests use the verified signature: retrigger_verdict(termination_class, post_af, baseline=, ref_peak=)
which computes amp = ref_peak - baseline internally.
"""
import numpy as np
import pytest
from src.sef_hfo_m4_termination import retrigger_verdict


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
