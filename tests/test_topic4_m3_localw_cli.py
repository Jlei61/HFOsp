"""M3 plan Task 3: W-coupled permissivity (h-coupling) wired into the runner CLI.

Three checks (plan Task 3 Step 1):
  1. --mu appears in the runner --help.
  2. Coexist smoke (SMALL/fast config): --mu 0.3 --h-source struct --h-scheme post
     --ee-std-u 0.2 runs end-to-end and the written config/provenance carries
     mu/h_source/h_scheme/h_control AND ee_std_u (M1 recovery is NOT displaced).
  3. Bit-parity: the runner's mu=0 V_th pre-transform is a NO-OP, so driving the
     EXACT anchor net/sim recipe (the same recipe the hub/degnorm parity test
     reproduces -- bare engine, flat-18 V_th, no kick) yields the anchored spike
     SHA == M3_BASE_SHA. We exercise the runner's mu=0 short-circuit on that net
     and assert the returned V_th is bit-identical and the SHA matches.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M3_BASE_SHA = "da5fc18c27d5340a"   # Task-0 anchor (L=6, density=100, T=300, dt=0.1,
#                                    drive=0.6, seed=1, theta_EE=45, AR=2, V_th=18, no kick)


def _load_runner():
    # the runner imports the engine via sys.path.insert("src/snn_engine"); needs CWD=ROOT
    old = os.getcwd()
    os.chdir(ROOT)
    try:
        spec = importlib.util.spec_from_file_location(
            "_sef_runner_uut", os.path.join(ROOT, "scripts", "run_sef_hfo_snn_cm_spontaneous_readout.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_sef_runner_uut"] = mod
        spec.loader.exec_module(mod)
    finally:
        os.chdir(old)
    return mod


def _anchor_net():
    """The exact net/sim recipe the hub parity test (tests/test_snn_hub_longrange.py)
    uses to reproduce M3_BASE_SHA: bare engine, flat-18 V_th, no kick."""
    sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))
    from params import Params
    from connectivity import place_neurons
    from connectivity_rot import build_connectivity_rot
    p = Params(L=6.0, density=100.0, T=300.0, dt=0.1, nu_ext_ratio=0.6, seed=1)
    rng = np.random.default_rng(1)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                 theta_EE=np.radians(45), AR=2.0)
    return p, net, pos, labels, NE, NI


# ---------------------------------------------------------------------------
# Check 1: --mu in --help
# ---------------------------------------------------------------------------
def test_mu_in_help():
    out = subprocess.run([sys.executable, "scripts/run_sef_hfo_snn_cm_spontaneous_readout.py", "--help"],
                         capture_output=True, text=True, timeout=120, cwd=ROOT).stdout
    for flag in ["--mu", "--delta-theta", "--h-source", "--h-scheme", "--h-control",
                 "--mu-impl", "--w-resp-cache", "--w-resp-calib-json",
                 "--allow-pilot-default-wresp"]:
        assert flag in out, f"{flag} missing from runner --help"


# ---------------------------------------------------------------------------
# Check 2: coexist smoke -- mu>0 runs end-to-end; provenance carries new params
#          AND ee_std_u (M1 recovery not displaced).
# ---------------------------------------------------------------------------
def test_coexist_smoke_provenance():
    with tempfile.TemporaryDirectory() as td:
        cmd = [sys.executable, "scripts/run_sef_hfo_snn_cm_spontaneous_readout.py",
               "--L", "8", "--T", "300", "--lesion", "oneend_neg",
               "--mu", "0.3", "--h-source", "struct", "--h-scheme", "post",
               "--ee-std-u", "0.2", "--ee-std-tau-ms", "200",
               "--out", td, "--tag", "smoke", "--seed", "1"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=ROOT)
        assert r.returncode == 0, f"runner failed:\nSTDOUT={r.stdout}\nSTDERR={r.stderr}"
        cfg = json.load(open(os.path.join(td, "readout_smoke.json")))["config"]
        assert cfg["mu"] == 0.3
        assert cfg["h_source"] == "struct"
        assert cfg["h_scheme"] == "post"
        assert cfg["h_control"] == "none"
        assert cfg["mu_impl"] == "threshold"
        assert "n_bins" in cfg
        # M1 recovery must survive alongside the M3 h-coupling
        assert cfg["ee_std_u"] == 0.2


# ---------------------------------------------------------------------------
# Check 3: mu=0 V_th pre-transform is a no-op -> anchor SHA preserved.
# ---------------------------------------------------------------------------
def test_mu0_vth_pretransform_is_noop_anchor_sha():
    runner = _load_runner()
    p, net, pos, labels, NE, NI = _anchor_net()
    posE = pos[:NE]
    vth0 = np.full(NE + NI, 18.0)

    # build an argparse-like namespace with mu=0 + the M3 h knobs at defaults
    import argparse
    a = argparse.Namespace(mu=0.0, delta_theta=3.0, h_source="struct", h_scheme="post",
                           h_control="none", mu_impl="threshold", w_resp_cache=None,
                           seed=1)
    # the runner's mu=0 short-circuit MUST return V_th untouched
    vth, prov = runner.apply_permissivity_vth_delta(
        vth0.copy(), net, NE, NI, posE, a,
        bins=None, p=p, V_th0=vth0, rng=np.random.default_rng(1))
    assert np.array_equal(vth, vth0), "mu=0 must not touch V_th_per_neuron (bit-parity)"

    # and the simulated spikes on that anchor recipe hash to M3_BASE_SHA
    sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))
    from kick_probe import simulate_kick
    net["rng"] = np.random.default_rng(1)
    res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=vth)
    sha = hashlib.sha1(res["E_spk_bool"].tobytes()).hexdigest()[:16]
    assert sha == M3_BASE_SHA, f"mu=0 anchor SHA {sha} != M3_BASE_SHA {M3_BASE_SHA}"


# ---------------------------------------------------------------------------
# Check 4 (review P1 2026-06-22): --mu>0 --h-source resp WITHOUT a measured-W
# source (no cache, no calibration JSON, not explicitly allowed) must FAIL CLOSED
# -- never silently fall back to pilot-default W kick/window in a canonical run.
# ---------------------------------------------------------------------------
def test_resp_mode_fails_closed_without_calibration():
    runner = _load_runner()
    p, net, pos, labels, NE, NI = _anchor_net()
    posE = pos[:NE]
    vth0 = np.full(NE + NI, 18.0)
    import argparse
    a = argparse.Namespace(mu=0.3, delta_theta=3.0, h_source="resp", h_scheme="post",
                           h_control="none", mu_impl="threshold", w_resp_cache=None,
                           w_resp_calib_json=None, allow_pilot_default_wresp=False, seed=1)
    with pytest.raises(RuntimeError, match="measured-W source"):
        runner.apply_permissivity_vth_delta(
            vth0.copy(), net, NE, NI, posE, a,
            bins=None, p=p, V_th0=vth0, rng=np.random.default_rng(1))
