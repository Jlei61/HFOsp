"""TDD for the global-burst adaptation: a slow brake that only seizure-scale events can charge.

Why this shape.  The loop's last leg fails because the tissue settles into a smouldering
train it cannot leave: wear stalls at 0.089-0.092 (a fixed point -- 69 s is fourteen wear
time constants and it does not fall further), and a frozen field at that wear departs again
after 2.0 s.  Nothing that keys on how OFTEN events arrive can break this: the smoulder's
inter-event interval is 212-282 ms against 255-372 ms for the final gaps of the train that
produced entry, so the smoulder is DENSER than the pre-ictal train and any rate threshold
fires before entry rather than after it.

What does separate them is how much tissue each event recruits.  Measured, pooled over the
three no-kick trajectories and both 70 s tail arms: the pre-entry train peaks at 0.095 of the
array (34 events, max), while the smoulder's median is 0.178-0.281 and the discharge's is
0.390.  Across gates from 0.12 to 0.25 the pre-entry train crosses ZERO times while the
smoulder crosses 2-3.5 times a second, so the trigger has a two-fold window, not a knife edge.

Two design consequences are contracts here, not preferences:

* **charge fast, release slow.**  The brake has to still be holding while wear clears, and
  wear needs 3.2 s to fall from 0.089 to the lowest level that still departs, 7.5 s to reach
  the stable range.  The existing relay releases on 5000 ms -- exactly tau_z -- which is why
  it lets go just as the wear is clearing.  Release must dominate wear decay.
* **eta = 0 is sensor-only and membrane-identical.**  The cheap experiment is to watch
  whether the sensor separates entry from smoulder before the brake is allowed to act.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "snn_engine"))
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402

DT = 0.05
NE = 200
N = 250


def _mk(**kw):
    core = np.zeros(NE, bool)
    core[:10] = True
    return MZSlowVars(N, 18.0, MZSlowVarsConfig(**kw), NE=NE, core_mask_E=core)


def _labels():
    lab = np.zeros(N, dtype=int)
    lab[NE:] = 1
    return lab


def _drive(mz, *, fraction, ms, dt=DT, synchronous=False):
    """Fire `fraction` of E cells once per millisecond for `ms`, and step the slow vars.

    One spike per cell per millisecond is what the refractory period allows, so this is the
    quantity the detector reports as the active fraction over its 1 ms bin.  By default the
    cells are spread across the steps of each millisecond, which is what a real recruitment
    does; `synchronous=True` collapses them into a single step to probe the worst case a leaky
    sensor can see.
    """
    n_hot = int(round(fraction * NE))
    steps_per_ms = int(round(1.0 / dt))
    lab = _labels()
    phase = np.arange(n_hot) % steps_per_ms          # deterministic spread within the millisecond
    for k in range(int(round(ms / dt))):
        spk = np.zeros(N, bool)
        if n_hot:
            if synchronous:
                if k % steps_per_ms == 0:
                    spk[:n_hot] = True
            else:
                spk[:n_hot][phase == (k % steps_per_ms)] = True
        mz.step(spk, lab, dt)


# ---- off is off -------------------------------------------------------------------------

def test_off_allocates_no_state():
    mz = _mk(use_gba=False)
    assert getattr(mz, "gba_burst", None) is None
    assert getattr(mz, "gba_a", None) is None


def test_off_leaves_the_membrane_exactly_alone():
    off = _mk(use_gba=False)
    I_E = np.arange(N, dtype=float) + 1.0
    I_I = np.arange(N, dtype=float) * 0.5
    assert np.array_equal(off.apply_currents(I_E, I_I, labels=_labels()), I_E - I_I)


def _parity_pair(**cfg_kw):
    """A kick run with slow=None and one with the given config, on identical noise."""
    from params import Params
    from connectivity import place_neurons, build_connectivity
    from kick_probe import simulate_kick

    seed = 1
    p = Params(L=1.0, density=400.0, T=200.0, dt=0.1, seed=seed, nu_ext_ratio=1.0)
    rng = np.random.default_rng(seed)
    pos, labels, ne, ni = place_neurons(p, rng)
    net = build_connectivity(p, pos, labels, ne, ni, rng, verbose=False)
    n = ne + ni
    vth = np.full(n, 18.0)
    vth[:5] = 16.0
    center = np.array([p.L / 2, p.L / 2])

    def run(slow):
        net["rng"] = np.random.default_rng(seed)
        return simulate_kick(p, net, 5.0, slow=slow, kick_center=center, r_kick=0.3,
                             t_kick=50.0, V_th_per_neuron=vth, verbose=False)

    mz = MZSlowVars(n, 18.0, MZSlowVarsConfig(**cfg_kw), NE=ne,
                    core_mask_E=np.zeros(ne, bool))
    return run(None), run(mz), mz


def test_engine_byte_parity_when_off_equals_slow_none():
    base, off, _ = _parity_pair(use_gba=False)
    assert np.array_equal(base["rate_E"], off["rate_E"])
    assert np.array_equal(base["E_spk_bool"], off["E_spk_bool"])


def test_engine_byte_parity_in_sensor_only_mode():
    """eta = 0 must leave the tissue bit-identical while the sensor still moves."""
    base, sensed, mz = _parity_pair(use_gba=True, gba_gate=0.0, eta_gba=0.0,
                                    tau_gba_charge=50.0)
    assert np.array_equal(base["rate_E"], sensed["rate_E"])
    assert np.array_equal(base["E_spk_bool"], sensed["E_spk_bool"])
    assert mz.gba_a > 0.0, "sensor-only mode must still charge, or it observes nothing"


# ---- the sensor reads the same quantity the detector reports -----------------------------

def test_sensor_tracks_the_one_millisecond_active_fraction():
    mz = _mk(use_gba=True, gba_gate=1.0)           # gate above 1 so nothing charges
    _drive(mz, fraction=0.30, ms=40.0)
    assert mz.gba_burst == pytest.approx(0.30, rel=0.05)


def test_sensor_distinguishes_the_measured_interictal_and_smoulder_extents():
    """0.095 is the largest pre-entry event measured; 0.281 the smoulder's median."""
    quiet = _mk(use_gba=True, gba_gate=1.0)
    _drive(quiet, fraction=0.095, ms=40.0)
    loud = _mk(use_gba=True, gba_gate=1.0)
    _drive(loud, fraction=0.281, ms=40.0)
    assert quiet.gba_burst < 0.15 < loud.gba_burst


def test_sensor_decays_when_the_tissue_goes_quiet():
    mz = _mk(use_gba=True, gba_gate=1.0, tau_gba_sense=5.0)
    _drive(mz, fraction=0.30, ms=40.0)
    charged = mz.gba_burst
    lab = _labels()
    for _ in range(int(round(100.0 / DT))):        # 20 sensor time constants
        mz.step(np.zeros(N, bool), lab, DT)
    assert mz.gba_burst < 1e-6 * charged


def test_a_synchronous_burst_cannot_fake_a_larger_one():
    """The reason the window is wider than the bin it is normalised to.

    A leaky sum overshoots perfectly synchronous input.  If that overshoot were large, the
    largest interictal event measured (0.095 of the array) could read above a gate meant only
    for seizure-scale recruitment, and the brake would engage during the entry train.
    """
    spread = _mk(use_gba=True, gba_gate=1.0)
    _drive(spread, fraction=0.095, ms=60.0)
    burst = _mk(use_gba=True, gba_gate=1.0)
    _drive(burst, fraction=0.095, ms=60.0, synchronous=True)
    assert burst.gba_burst < 1.2 * spread.gba_burst
    assert burst.gba_burst < 0.15, "the worst-case interictal reading still clears the gate"


# ---- the slow variable charges only above the gate ----------------------------------------

def test_interictal_scale_recruitment_never_charges_it():
    mz = _mk(use_gba=True, gba_gate=0.15, tau_gba_charge=200.0)
    _drive(mz, fraction=0.095, ms=2000.0)          # the largest pre-entry event, sustained
    assert mz.gba_a == 0.0


def test_seizure_scale_recruitment_charges_it_toward_the_excess():
    mz = _mk(use_gba=True, gba_gate=0.15, tau_gba_charge=200.0)
    _drive(mz, fraction=0.39, ms=3000.0)           # the discharge's median recruitment
    assert mz.gba_a == pytest.approx(0.39 - 0.15, rel=0.10)


def test_it_charges_fast_and_releases_slow():
    """The measured requirement: still holding while wear clears, which takes 3.2-7.5 s."""
    mz = _mk(use_gba=True, gba_gate=0.15, tau_gba_charge=1000.0, tau_gba_release=30000.0)
    _drive(mz, fraction=0.39, ms=4000.0)
    charged = mz.gba_a
    assert charged > 0.5 * (0.39 - 0.15)
    lab = _labels()
    for _ in range(int(round(5000.0 / DT))):       # 5 s of silence, longer than wear needs
        mz.step(np.zeros(N, bool), lab, DT)
    assert mz.gba_a > 0.7 * charged, "released faster than wear clears"


def test_release_slower_than_charge_is_not_merely_the_default():
    fast = _mk(use_gba=True, gba_gate=0.15, tau_gba_charge=1000.0, tau_gba_release=1000.0)
    slow = _mk(use_gba=True, gba_gate=0.15, tau_gba_charge=1000.0, tau_gba_release=30000.0)
    lab = _labels()
    for mz in (fast, slow):
        _drive(mz, fraction=0.39, ms=4000.0)
        for _ in range(int(round(5000.0 / DT))):
            mz.step(np.zeros(N, bool), lab, DT)
    assert slow.gba_a > 3.0 * fast.gba_a


# ---- the actuator ------------------------------------------------------------------------

def test_sensor_only_mode_moves_the_state_but_not_the_membrane():
    """eta = 0 is the cheap first experiment: watch the sensor, leave the tissue untouched."""
    mz = _mk(use_gba=True, gba_gate=0.15, tau_gba_charge=200.0, eta_gba=0.0)
    _drive(mz, fraction=0.39, ms=2000.0)
    assert mz.gba_a > 0.0
    I_E = np.arange(N, dtype=float) + 1.0
    I_I = np.arange(N, dtype=float) * 0.5
    assert np.array_equal(mz.apply_currents(I_E, I_I, labels=_labels()), I_E - I_I)


def test_a_charged_brake_subtracts_current_from_E_cells_only():
    mz = _mk(use_gba=True, gba_gate=0.15, tau_gba_charge=200.0, eta_gba=2.0)
    _drive(mz, fraction=0.39, ms=2000.0)
    I_E = np.full(N, 10.0)
    I_I = np.zeros(N)
    out = mz.apply_currents(I_E, I_I, labels=_labels())
    assert np.all(out[:NE] < 10.0)
    assert np.array_equal(out[NE:], I_E[NE:])
    assert out[:NE] == pytest.approx(10.0 - 2.0 * mz.gba_a)


def test_an_uncharged_brake_takes_nothing():
    mz = _mk(use_gba=True, gba_gate=0.15, eta_gba=2.0)
    I_E = np.full(N, 10.0)
    I_I = np.zeros(N)
    assert np.array_equal(mz.apply_currents(I_E, I_I, labels=_labels()), I_E - I_I)


# ---- fail closed --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    dict(gba_gate=-0.1), dict(gba_gate=1.5),
    dict(tau_gba_sense=0.0), dict(tau_gba_charge=-1.0), dict(tau_gba_release=0.0),
    dict(eta_gba=-1.0),
])
def test_impossible_settings_are_refused(bad):
    with pytest.raises(ValueError):
        _mk(use_gba=True, **bad)


def test_the_knobs_are_inert_unless_the_brake_is_on():
    with pytest.raises(ValueError, match="use_gba"):
        _mk(use_gba=False, eta_gba=2.0)
