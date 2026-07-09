"""End-to-end smoke test for M4-3A (n->a load/shunt): Task 10, the final build task.

Two tests, both driving a REAL tiny E-I network end-to-end through
`simulate_kick`/`SpatialSlowField` (NOT the heavy `run_arm`/`PP.build_substrate`
E1146 pipeline the sweep runner uses -- this file builds its own small net,
mirroring the `_build_kicked_net`/`_slow_for_net` pattern already used by
tests/test_kick_probe_shunt.py, for speed):

1. `test_shunt_on_changes_dynamics` (the MORE IMPORTANT of the two): use_A=True
   with alpha_A=8.0, k_n=1.0 (enough to matter) produces a DIFFERENT E spike
   raster than the identical run with use_A=False. This proves the shunt
   genuinely couples end-to-end (load ODE ramp-up -> Hill readout -> conductance
   shunt -> membrane -> spikes), not just that the wiring exists in isolation
   (which test_kick_probe_shunt.py::test_shunt_engaged_suppresses_relative_to_off
   already checks, but with n_load/a_shunt PRE-SEEDED to bypass Task 4's ODE
   ramp-up -- this test lets it ramp up naturally from spiking).

2. `test_use_A_off_byte_parity_end_to_end` (off-parity gate; belt-and-suspenders,
   NOT the primary parity evidence -- that was already established by Tasks 4-5's
   own dedicated unit tests, test_slow_field_na.py + test_kick_probe_shunt.py, and
   by Task 5's 271-test full regression after re-bless): use_A=False (n/a config
   PRESENT -- alpha_A=8.0/k_n=1.0/tau_n=50.0 all non-trivial, but the master gate
   is off) must be bit-identical to a GOLDEN raster captured from the pre-M4-3A
   engine at commit 7ee73c6 (the docs commit immediately BEFORE Task 1).

   Method chosen (the brief's PREFERRED option, not the fallback): mirrors
   tests/test_a1c_feedback.py's FIXTURE["dyn16"] pattern (captured from a
   pre-edit commit via `git show <sha>:src/snn_engine/kick_probe.py`). Verified
   via `git diff --stat 7ee73c6 HEAD -- src/snn_engine/` that ONLY
   src/snn_engine/{kick_probe,slow_field}.py changed under src/snn_engine/ across
   ALL of Tasks 1-9 (params.py/connectivity.py/connectivity_rot.py/lfp.py, and
   src/sef_hfo_field.py/src/topic4_m3a_v2_2_sensors.py that slow_field.py itself
   imports, are byte-identical to 7ee73c6) -- so a golden built by running the
   OLD kick_probe.py/slow_field.py (fetched via `git show`) through the CURRENT,
   unchanged params/connectivity/connectivity_rot is a valid, independent
   "pre-M4-3A engine" reference, not an approximation.

   The golden fixture (tests/fixtures/m43a_pre_task4_use_a_off_golden.npz) was
   captured by a throwaway script (not committed -- same convention as the a1c
   fixture, which also has no committed capture script) that:
     1. wrote `git show 7ee73c6:src/snn_engine/kick_probe.py` and
        `git show 7ee73c6:src/snn_engine/slow_field.py` to a temp directory,
     2. ran a fresh `python` subprocess with that temp directory prepended to
        sys.path (ahead of the current src/snn_engine), so `import kick_probe` /
        `from slow_field import ...` resolved to the OLD files while
        `from params import Params` etc. fell through to the current (unchanged)
        src/snn_engine copies,
     3. built the IDENTICAL net/kick recipe as `_tiny_m4_run` below (n_grid=8,
        SpatialSlowFieldConfig with only n_grid=8 set -- the OLD dataclass has no
        use_A/alpha_A/k_n/tau_n fields at all, matching "config absent" exactly),
     4. dumped `res["E_spk_bool"]` and `slow.trace_qI_mean` to the npz above.
   Regenerable the same way if this test's net recipe ever changes; a fresh
   subprocess is required (not an in-process importlib swap) to avoid colliding
   with the CURRENT kick_probe/slow_field modules already cached in
   sys.modules under the same bare names elsewhere in a pytest session.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))
from params import Params  # noqa: E402
from connectivity import place_neurons  # noqa: E402
from connectivity_rot import build_connectivity_rot  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402
from slow_field import SpatialSlowField, SpatialSlowFieldConfig  # noqa: E402

DT = 0.1
GOLDEN_PATH = os.path.join(ROOT, "tests", "fixtures", "m43a_pre_task4_use_a_off_golden.npz")


def _tiny_m4_run(T=400.0, use_A=False, alpha_A=8.0, k_n=1.0, tau_n=50.0, seed_net=1, seed_rng=3, **cfgkw):
    """Smallest real M4-3A run: a tiny E-I net (L=6mm, density=100/mm^2 -- the same
    recipe as test_kick_probe_shunt.py::_build_kicked_net/_slow_for_net) driven by
    one KICK_BOOST-triggered burst, through the REAL SpatialSlowField.step
    ((spk, labels, dt) signature) -> simulate_kick pipeline (nothing mocked).

    Defaults (alpha_A=8.0, k_n=1.0, tau_n=50.0) are non-trivial M4-3A values
    regardless of use_A, so `_tiny_m4_run(use_A=False)` alone means "n/a config
    PRESENT, master gate off" (the scientifically-relevant parity case) rather
    than "every n/a knob left at zero" (a weaker, trivially-inert claim already
    covered by test_slow_field_na.py::test_shunt_off_by_default_is_byte_parity).
    tau_n=50ms (vs. the production default 20000ms, meant for real multi-second
    campaigns) is a TEST-SPEED choice only -- it does not change which code path
    runs, only how many simulated ms the load field needs to respond; T=400ms
    (kick at the default t_kick=150ms) leaves 250ms of post-kick evolution,
    several tau_n time constants, enough for a_shunt to visibly diverge.
    """
    p = Params(L=6.0, density=100.0, T=T, dt=DT, nu_ext_ratio=0.6, seed=seed_net)
    rng = np.random.default_rng(seed_net)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0)
    net["rng"] = np.random.default_rng(seed_rng)
    posE = net["pos"][net["labels"] == 0]
    posI = net["pos"][net["labels"] == 1]
    cfg = SpatialSlowFieldConfig(n_grid=8, use_A=use_A, alpha_A=alpha_A, k_n=k_n, tau_n=tau_n, **cfgkw)
    slow = SpatialSlowField(NE + NI, p.V_th, posE, posI, p.L, cfg=cfg)
    res = simulate_kick(p, net, KICK_BOOST=6.0, r_kick=1.5,
                        V_th_per_neuron=np.full(NE + NI, 16.5), slow=slow)
    return dict(spk=np.asarray(res["E_spk_bool"]), trace_qI_mean=np.asarray(slow.trace_qI_mean, dtype=float))


def test_shunt_on_changes_dynamics():
    """The more important test: the shunt actually couples (spec parity red-line's
    counterpart -- proves the mechanism is not just gated off correctly, but also
    genuinely ON when engaged)."""
    res_on = _tiny_m4_run(use_A=True, alpha_A=8.0, k_n=1.0)
    res_off = _tiny_m4_run(use_A=False)
    assert res_off["spk"].sum() > 0                               # sanity: baseline actually spikes
    assert not np.array_equal(res_on["spk"], res_off["spk"])      # shunt actually couples


def test_use_A_off_byte_parity_end_to_end():
    """Off-parity gate (belt-and-suspenders -- see module docstring for the golden
    capture method + why Tasks 4-5 are the primary parity evidence, not this test)."""
    res_off = _tiny_m4_run(use_A=False)           # n/a config present, gate off
    golden = np.load(GOLDEN_PATH)
    assert np.array_equal(res_off["spk"], golden["spk"])
    assert np.allclose(res_off["trace_qI_mean"], golden["trace_qI_mean"])
