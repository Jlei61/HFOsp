"""TDD for M4-2 engine instrumentation (spec 2026-07-07 rev2, Task 1).

Two NON-behavioral hooks added to simulate_kick, OFF by default -> byte-identical to today:
  1. dump_ee_std_trace  -> x_dep depression summary trace (mean/min, + optional axis-mask mean).
                           Arm 0 (ee_std_u=0) emits CONSTANT 1.0 (availability un-depleted).
                           Recording point is AFTER the spike depletion (:371), NOT after recovery (:259).
  2. t_kick2/KICK_BOOST2 -> a second kick window (post-offset retrigger). t_kick2=None -> parity.
                           Pre-probe identity: for t < t_kick2 the trajectory is byte-identical to a
                           run without the second kick (makes retrigger_probe interpretable).
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from params import Params                          # noqa: E402
from connectivity import place_neurons             # noqa: E402
from connectivity_rot import build_connectivity_rot  # noqa: E402
from kick_probe import simulate_kick               # noqa: E402
from src.sef_hfo_m4_termination import (            # noqa: E402
    classify_termination, retrigger_verdict, run_cell_with_retrigger,
)

DT = 0.1


def _net(L=6.0, T=200.0, seed=1, density=100.0, nu=0.8):
    p = Params(L=L, density=density, T=T, dt=DT, nu_ext_ratio=nu, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0)
    return p, net


def _fresh(net, seed=1):
    net["rng"] = np.random.default_rng(seed)
    return net


# -------------------------------------------------- byte-parity of the new (default) params
def test_new_params_default_byte_identical():
    """simulate_kick with all new params at their defaults == today (no alloc/RNG/float change)."""
    p, net = _net()
    base = simulate_kick(p, _fresh(net), 3.0, slow=None, t_kick=50.0, r_kick=2.0)
    new = simulate_kick(p, _fresh(net), 3.0, slow=None, t_kick=50.0, r_kick=2.0,
                        dump_ee_std_trace=False, ee_std_trace_maskE=None,
                        t_kick2=None, KICK_BOOST2=0.0)
    assert np.array_equal(base["E_spk_bool"], new["E_spk_bool"])
    assert np.array_equal(base["rate_E"], new["rate_E"])


def test_trace_does_not_perturb_dynamics():
    """dump_ee_std_trace is read-only: with ee_std_u>0 it must not change spikes; it only adds outputs."""
    p, net = _net()
    a = simulate_kick(p, _fresh(net), 3.0, slow=None, t_kick=50.0, r_kick=2.0,
                      ee_std_u=0.2, ee_std_tau_ms=500.0)
    b = simulate_kick(p, _fresh(net), 3.0, slow=None, t_kick=50.0, r_kick=2.0,
                      ee_std_u=0.2, ee_std_tau_ms=500.0, dump_ee_std_trace=True)
    assert np.array_equal(a["E_spk_bool"], b["E_spk_bool"])
    assert "xdep_min" in b and "xdep_min" not in a


# -------------------------------------------------- Arm 0 constant-ones schema (P1-b)
def test_arm0_xdep_trace_constant_ones():
    """ee_std_u=0 with dump_ee_std_trace=True -> constant 1.0 trace (schema aligned with Arm 1)."""
    p, net = _net()
    r = simulate_kick(p, _fresh(net), 3.0, slow=None, t_kick=50.0, r_kick=2.0,
                      ee_std_u=0.0, dump_ee_std_trace=True)
    assert np.allclose(r["xdep_mean"], 1.0)
    assert np.allclose(r["xdep_min"], 1.0)


# -------------------------------------------------- pre-probe identity (P1-a)
def test_second_kick_prewindow_identity():
    """A run with a 2nd kick at t2 is byte-identical to a run without it for all t < t2
    (the t_kick2 branch is skipped for t<t2); t>=t2 must differ (the 2nd kick does something)."""
    p, net = _net(T=300.0)
    t2 = 150.0
    i2 = int(round(t2 / DT))
    a = simulate_kick(p, _fresh(net), 3.0, slow=None, t_kick=50.0, r_kick=2.0)                 # no 2nd kick
    b = simulate_kick(p, _fresh(net), 3.0, slow=None, t_kick=50.0, r_kick=2.0,
                      t_kick2=t2, KICK_BOOST2=3.0)                                              # 2nd kick at t2
    assert np.array_equal(a["E_spk_bool"][:i2], b["E_spk_bool"][:i2])          # pre-probe identity (承重)
    assert not np.array_equal(a["E_spk_bool"][i2:], b["E_spk_bool"][i2:])      # 2nd kick perturbs t>=t2 (not a no-op)


# -------------------------------------------------- trace recorded AFTER depletion (P1-c phase)
def test_xdep_trace_phase_post_depletion():
    """x_dep trace must be recorded AFTER the spike depletion (:371), not after recovery (:259):
    at the FIRST E-spike step the min availability already shows the depletion (== 1-u), not 1.0."""
    p, net = _net()
    u = 0.2
    r = simulate_kick(p, _fresh(net), 3.0, slow=None, t_kick=50.0, r_kick=2.0,
                      ee_std_u=u, ee_std_tau_ms=500.0, dump_ee_std_trace=True)
    espk = r["E_spk_bool"]
    xmin = r["xdep_min"]
    fired = np.where(espk.any(axis=1))[0]
    assert fired.size > 0
    t_spk = int(fired[0])                        # first step an E neuron fires
    assert xmin[t_spk] < 1.0                      # depletion reflected SAME step -> recorded post-:371
    assert np.isclose(xmin[t_spk], 1.0 - u)       # first spike: firers deplete from 1 -> (1-u)


# ==================================================================== Task 3: classify_termination
# Synthetic fixtures (hand-built activity traces, NO simulation) — thresholds must classify these
# correctly independent of sim data (avoids the "tune thresholds on the same real traces" circularity).
BIN = 5.0
BASE = 0.02


def _seg(*segs):
    """Build an activity trace: (value, n_bins) = flat; (v0, v1, n_bins) = linear ramp."""
    out = []
    for s in segs:
        if len(s) == 2:
            out.append(np.full(s[1], s[0]))
        else:
            out.append(np.linspace(s[0], s[1], s[2]))
    return np.concatenate(out)


def test_classify_terminate_clean():
    af = _seg((BASE, 40), (BASE, 0.5, 10), (0.5, 120), (0.5, BASE, 4), (BASE, 230))  # high plateau -> sharp offset
    cls, info = classify_termination(af, BIN, baseline=BASE)
    assert cls == "terminate_clean"
    assert info["offset_ms"] is not None


def test_classify_fade():
    af = _seg((BASE, 40), (BASE, 0.5, 3), (0.5, BASE, 200), (BASE, 160))             # monotone decline, no plateau
    assert classify_termination(af, BIN, baseline=BASE)[0] == "fade"


def test_classify_persist():
    af = _seg((BASE, 40), (BASE, 0.5, 10), (0.5, 350))                               # plateau to the end
    assert classify_termination(af, BIN, baseline=BASE)[0] == "persist"


def test_classify_suppress():
    af = _seg((BASE, 40), (BASE, 0.05, 5), (0.05, 20), (0.05, BASE, 5), (BASE, 330)) # never rises above a_min
    assert classify_termination(af, BIN, baseline=BASE)[0] == "suppress"


def test_classify_fragment():
    af = _seg((BASE, 40), *([(BASE, 15), (0.4, 8)] * 5), (BASE, 100))                # many short intermittent bursts
    assert classify_termination(af, BIN, baseline=BASE)[0] == "fragment"


def test_classify_rebound():
    af = _seg((BASE, 40), (BASE, 0.5, 10), (0.5, 100), (0.5, BASE, 4), (BASE, 60),   # clean event ...
              (BASE, 0.4, 5), (0.4, 30), (0.4, BASE, 4), (BASE, 60))                 # ... quiet gap, then re-ignition
    assert classify_termination(af, BIN, baseline=BASE)[0] == "rebound"


def test_retrigger_not_run_when_not_clean():
    assert retrigger_verdict("persist") == "not_run"
    assert retrigger_verdict("fade") == "not_run"


def test_retrigger_fail_on_fizzle():
    post = np.full(200, BASE)                                                        # post-kick stays quiet
    assert retrigger_verdict("terminate_clean", post_af=post, baseline=BASE, ref_peak=0.5) == "fail"


def test_retrigger_pass_on_bounded_reignition():
    post = _seg((BASE, 20), (BASE, 0.4, 5), (0.4, 40), (0.4, BASE, 5), (BASE, 130))  # re-igniting bounded event
    assert retrigger_verdict("terminate_clean", post_af=post, baseline=BASE, ref_peak=0.5) == "pass"


def test_retrigger_fail_on_runaway():
    post = _seg((BASE, 20), (BASE, 0.5, 5), (0.5, 175))                              # re-ignites but never comes down
    assert retrigger_verdict("terminate_clean", post_af=post, baseline=BASE, ref_peak=0.5) == "fail"


# ============================ review round 2 fixes ============================
def test_early_stop_trace_last_frame_written():
    """early_stop_runaway breaks mid-loop BEFORE the trace write; the break frame is still kept in the
    output (_stop_t=t+1), so its x_dep must be written — else it stays the init sentinel 0 and the
    (x_dep,q_I) diagnostic shows PHANTOM depletion to 0 at the runaway frame."""
    p, net = _net(T=400.0)
    r = simulate_kick(p, _fresh(net), 5.0, slow=None, t_kick=50.0, r_kick=2.0,
                      ee_std_u=0.2, ee_std_tau_ms=500.0, dump_ee_std_trace=True,
                      early_stop_runaway=True, es_thresh_hz=1.0, es_dur_ms=30.0)
    assert r["runaway_early_stop_ms"] is not None              # early-stop actually fired
    assert r["xdep_min"].shape[0] == r["E_spk_bool"].shape[0]  # trace truncated to the same length
    assert r["xdep_min"][-1] > 0.0                             # break frame written, NOT phantom 0


def test_classify_runaway_from_engine_verdict():
    """A run the engine flagged runaway (runaway_ms set) is 'runaway', NOT 'persist': the phase diagram
    must tell an unbounded runaway apart from a bounded persistent attractor (both fail to terminate)."""
    af = _seg((BASE, 40), (BASE, 0.9, 10), (0.9, 350))                    # high sustained -> 'persist' by shape
    assert classify_termination(af, BIN, baseline=BASE)[0] == "persist"
    assert classify_termination(af, BIN, baseline=BASE, runaway_ms=1234.0)[0] == "runaway"


def test_classify_raises_on_bad_input():
    with pytest.raises(ValueError):
        classify_termination(np.array([]), BIN, baseline=BASE)
    with pytest.raises(ValueError):
        classify_termination(np.array([0.1, np.nan, 0.2]), BIN, baseline=BASE)


def test_retrigger_raises_on_missing_refs():
    post = _seg((BASE, 20), (0.4, 40), (BASE, 40))
    with pytest.raises(ValueError):                                        # terminate_clean needs baseline+ref_peak
        retrigger_verdict("terminate_clean", post_af=post, baseline=None, ref_peak=None)


# ==================================================================== Task 2: two-pass retrigger orchestrator
# Pure logic with an INJECTED run_fn(t_kick2, kick_boost2) -> {'af','runaway_ms'} (no simulation):
# pass 1 classifies; pass 2 (only if terminate_clean) fires the second kick at offset+recovery and
# reads the retrigger verdict, after asserting pre-probe identity (pass-2 prefix == pass-1 for t<t_kick2).
_CLEAN = _seg((BASE, 40), (BASE, 0.5, 10), (0.5, 120), (0.5, BASE, 4), (BASE, 230))


def test_retrigger_orch_pass():
    def fake(t2, kb):
        if t2 is None:
            return {"af": _CLEAN, "runaway_ms": None}
        i2 = int(round(t2 / BIN))
        post = _seg((BASE, 30), (BASE, 0.4, 5), (0.4, 40), (0.4, BASE, 5), (BASE, 50))  # bounded re-ignition
        return {"af": np.concatenate([_CLEAN[:i2], post]), "runaway_ms": None}
    out = run_cell_with_retrigger(fake, BIN, recovery_ms=100.0, recovery_factor=1.0)
    assert out["termination_class"] == "terminate_clean"
    assert out["retrigger_probe"] == "pass"


def test_retrigger_orch_not_run_on_persist():
    persist = _seg((BASE, 40), (BASE, 0.5, 10), (0.5, 350))
    calls = []

    def fake(t2, kb):
        calls.append(t2)
        return {"af": persist, "runaway_ms": None}
    out = run_cell_with_retrigger(fake, BIN)
    assert out["termination_class"] == "persist"
    assert out["retrigger_probe"] == "not_run"
    assert calls == [None]                       # pass-2 must NOT run for a non-terminate_clean cell


def test_retrigger_orch_runaway_not_run():
    hi = _seg((BASE, 40), (BASE, 0.9, 10), (0.9, 350))
    out = run_cell_with_retrigger(lambda t2, kb: {"af": hi, "runaway_ms": 2000.0}, BIN)
    assert out["termination_class"] == "runaway"          # engine verdict wins
    assert out["retrigger_probe"] == "not_run"


def test_retrigger_orch_raises_on_identity_violation():
    def fake(t2, kb):
        if t2 is None:
            return {"af": _CLEAN, "runaway_ms": None}
        bad = _CLEAN.copy(); bad[:50] += 0.3     # corrupt the pre-probe prefix
        return {"af": bad, "runaway_ms": None}
    with pytest.raises(RuntimeError):
        run_cell_with_retrigger(fake, BIN, recovery_ms=100.0, recovery_factor=1.0)
