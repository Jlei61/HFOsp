"""Task 2 gate (spec rev3.1 §3.1): the guarded checkpoint hook must be numerically invisible.

Two independent claims, both required before ANY scientific fork runs:

 1. PRE vs POST edit. `tests/fixtures/topic4_zm_preedit_parity.npz` was produced by the PRE-edit
    engine (kick_probe SHA 5faaedab...), so this is not the post-edit code grading itself.
 2. hook-enabled-but-idle. Passing a `ZMCheckpoint` that requests nothing must not add an RNG draw
    or change a single bit.

The historical BASELINE_SHA regressions (tests/test_snn_{shunting,gates}.py, test_zm_slow_field_
parity.py, test_a1c_feedback.py) stay green independently -- they are run alongside this file.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kick_probe import simulate_kick  # noqa: E402
from src.topic4_zm_checkpoint import ZMCheckpoint  # noqa: E402
import scripts.gen_topic4_zm_preedit_fixture as GEN  # noqa: E402

FIX = os.path.join(_ROOT, "tests", "fixtures", "topic4_zm_preedit_parity.npz")
PRE_EDIT_SHA = "5faaedab37ab6208888607f70dc387f378b1ec9e51bfc21e29f0c81e80aa8b99"


@pytest.fixture(scope="module")
def preedit():
    with np.load(FIX) as z:
        return {k: np.array(z[k]) for k in z.files}


def test_fixture_provenance_names_the_pre_edit_engine():
    meta = json.load(open(FIX.replace(".npz", ".json")))
    assert meta["pre_edit_kick_probe_sha256"] == PRE_EDIT_SHA, \
        "the parity fixture must come from the pre-edit engine, not a re-run of the edited one"


@pytest.mark.parametrize("case", ["A_plain", "B_zm_sg_lfp"])
def test_default_path_is_byte_identical_to_pre_edit_engine(preedit, case):
    fn = dict(GEN.build_cases())[case]
    res = fn()
    for key in ("E_spk_bool", "rate_E", "rate_I", "spk_inside", "spk_outside"):
        assert np.array_equal(res[key], preedit[f"{case}__{key}"]), f"{case}.{key} changed"
    if f"{case}__lfp_trace" in preedit:
        assert np.array_equal(res["lfp_trace"], preedit[f"{case}__lfp_trace"])


def test_idle_hook_adds_no_rng_draw_and_changes_nothing(preedit):
    """zm_ckpt supplied but requesting nothing == default path, including the final RNG state."""
    from params import Params
    from connectivity import place_neurons, build_connectivity

    p = Params(L=1.0, density=400.0, T=120.0, dt=0.1, seed=1, nu_ext_ratio=1.0)
    rng = np.random.default_rng(1)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity(p, pos, labels, NE, NI, rng, verbose=False)

    def run(ckpt):
        net["rng"] = np.random.default_rng(7)
        r = simulate_kick(p, net, 4.0, kick_center=np.array([p.L / 2, p.L / 2]), r_kick=0.3,
                          t_kick=40.0, verbose=False, zm_ckpt=ckpt)
        return r, net["rng"].bit_generator.state

    a, sa = run(None)
    b, sb = run(ZMCheckpoint())
    assert np.array_equal(a["E_spk_bool"], b["E_spk_bool"])
    assert np.array_equal(a["rate_E"], b["rate_E"]) and np.array_equal(a["rate_I"], b["rate_I"])
    assert json.dumps(sa, default=str) == json.dumps(sb, default=str), "hook consumed RNG draws"


def test_raster_is_sensitive_to_a_small_dynamical_change(preedit):
    """The byte-equality assertions above only mean something if the raster CAN change: a 0.05 mV
    threshold shift (0.3% of V_th) must move it. (The stronger mutation test -- resuming with the
    ring-slot phase off by one -- lives in test_topic4_zm_exact_resume.py.)"""
    from params import Params
    from connectivity import place_neurons, build_connectivity

    p = Params(L=1.0, density=400.0, T=120.0, dt=0.1, seed=1, nu_ext_ratio=1.0)
    rng = np.random.default_rng(1)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity(p, pos, labels, NE, NI, rng, verbose=False)

    def run(reset):
        net["rng"] = np.random.default_rng(7)
        return simulate_kick(p, net, 4.0, kick_center=np.array([p.L / 2, p.L / 2]), r_kick=0.3,
                             t_kick=40.0, verbose=False,
                             V_th_per_neuron=np.full(NE + NI, 18.0 + reset))

    base = run(0.0)
    mut = run(0.05)
    assert not np.array_equal(base["E_spk_bool"], mut["E_spk_bool"]), \
        "a 0.05 mV threshold shift left the raster unchanged: the parity test is not sensitive"


def test_raster_sha_of_fixture_cases_is_recorded():
    """Cheap human-readable anchor so a silent fixture regeneration is visible in the diff."""
    with np.load(FIX) as z:
        shas = {c: hashlib.sha1(np.array(z[f"{c}__E_spk_bool"]).tobytes()).hexdigest()[:16]
                for c in ("A_plain", "B_zm_sg_lfp")}
    assert shas == {"A_plain": "e58ef7ce66814ffe", "B_zm_sg_lfp": "fa3efb54a40d06d5"}, shas
