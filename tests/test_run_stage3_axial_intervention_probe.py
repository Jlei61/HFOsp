"""Task 4 parity tests for the dynamic-Vth adapter + (Task 5) runner subprocess smokes.

The adapter must be a faithful copy of src/snn_engine/kick_probe.py::simulate_kick with the only
addition being a time-dependent per-neuron E-threshold. We verify bit-parity against the engine.
"""
import os
import sys
import json
import subprocess

import numpy as np

RUNNER = "scripts/run_stage3_axial_intervention_probe.py"
TINY = ["--L", "10", "--density", "20", "--T", "300", "--nc", "6", "--target-thickness", "2.5"]

ENG = os.path.join("src", "snn_engine")
sys.path.insert(0, ENG)
sys.path.insert(0, os.getcwd())

from params import Params                                   # noqa: E402
from connectivity import place_neurons                      # noqa: E402
from connectivity_rot import build_connectivity_rot         # noqa: E402
from kick_probe import simulate_kick                        # noqa: E402
from src.sef_hfo_axial_intervention import simulate_dynamic_vth   # noqa: E402


def _tiny_net(seed=1):
    p = Params(g=3.6, L=6.0, density=20.0, T=120.0, dt=0.1, nu_ext_ratio=0.6, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.deg2rad(45.0), AR=2.0)
    N = NE + NI
    is_E = np.zeros(N, bool); is_E[:NE] = True
    base_vth = np.full(N, 18.0)
    return p, net, base_vth, is_E, NE, N


def test_dynamic_adapter_no_intervention_matches_static_vth():
    p, net, base_vth, is_E, NE, N = _tiny_net()
    net["rng"] = np.random.default_rng(p.seed)
    res_kick = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=base_vth)
    net["rng"] = np.random.default_rng(p.seed)
    res_dyn = simulate_dynamic_vth(p, net, base_vth=base_vth, target_mask=None, is_E=is_E,
                                   on_ms=None, off_ms=None)
    assert np.array_equal(res_kick["E_spk_bool"], res_dyn["E_spk_bool"])
    assert np.array_equal(res_kick["rate_E"], res_dyn["rate_E"])


def test_dynamic_adapter_pre_intervention_parity():
    p, net, base_vth, is_E, NE, N = _tiny_net()
    target = np.zeros(N, bool); target[:20] = True
    on = 100.0
    net["rng"] = np.random.default_rng(p.seed)
    res_base = simulate_dynamic_vth(p, net, base_vth=base_vth, target_mask=None, is_E=is_E,
                                    on_ms=None, off_ms=None)
    net["rng"] = np.random.default_rng(p.seed)
    res_int = simulate_dynamic_vth(p, net, base_vth=base_vth, target_mask=target, is_E=is_E,
                                   on_ms=on, off_ms=140.0)
    s_on = int(round(on / p.dt))
    assert np.array_equal(res_base["E_spk_bool"][:s_on], res_int["E_spk_bool"][:s_on])
    assert np.array_equal(res_base["rate_E"][:s_on], res_int["rate_E"][:s_on])
    # after onset the traces are allowed to (and generally will) diverge
    assert res_int["intervention_active"][s_on:int(round(140.0 / p.dt))].all()


def test_dynamic_adapter_clamps_after_onset():
    p, net, base_vth, is_E, NE, N = _tiny_net()
    target = np.zeros(N, bool); target[:30] = True      # first 30 E cells
    on, off = 50.0, 90.0
    net["rng"] = np.random.default_rng(p.seed)
    res = simulate_dynamic_vth(p, net, base_vth=base_vth, target_mask=target, is_E=is_E,
                               on_ms=on, off_ms=off)
    dt = p.dt
    s0, s1 = int(round(on / dt)), int(round(off / dt))
    tgt_E = np.flatnonzero(target[:NE])
    assert res["E_spk_bool"][s0:s1][:, tgt_E].sum() == 0     # clamped E cells silent in window
    assert res["intervention_active"][s0:s1].all()
    assert not res["intervention_active"][:s0].any()
    assert not res["intervention_active"][s1:].any()


# ---------------- Task 5: runner subprocess smokes ----------------

def _run(args, cwd):
    return subprocess.run([sys.executable, RUNNER] + args, capture_output=True, text=True, cwd=cwd)


def test_runner_baseline_smoke_writes_schema(tmp_path):
    out = str(tmp_path)
    r = _run(TINY + ["--arm", "baseline", "--core-mean", "16.5", "--core-std", "1.0",
                     "--seed", "1", "--tag", "base", "--out", out], cwd=os.getcwd())
    assert r.returncode == 0, r.stderr
    data = json.load(open(os.path.join(out, "base.json")))
    assert data["arm"] == "baseline"
    for k in ("events", "n_returned", "baseline_eligibility"):
        assert k in data
    assert "eligible" in data["baseline_eligibility"]
    for e in data["events"]:
        for k in ("oracle_far_ratio", "oracle_reach_mm", "instr_far_ratio"):
            v = e.get(k)
            assert v is None or v >= 0.0


def test_runner_dynamic_on_axis_with_schedule_json(tmp_path):
    out = str(tmp_path)
    sched_path = os.path.join(out, "sched.json")
    json.dump({"on_ms": 50.0, "off_ms": 90.0}, open(sched_path, "w"))
    r = _run(TINY + ["--arm", "dynamic_on_axis", "--core-mean", "16.5", "--core-std", "1.0",
                     "--seed", "1", "--schedule-json", sched_path, "--tag", "dyn", "--out", out],
             cwd=os.getcwd())
    assert r.returncode == 0, r.stderr
    data = json.load(open(os.path.join(out, "dyn.json")))
    assert data["intervention_on"] == 50.0 and data["intervention_off"] == 90.0
    assert data["pre_intervention_parity"] is True       # guaranteed by construction


def test_runner_dynamic_on_axis_with_baseline_json(tmp_path):
    out = str(tmp_path)
    base_path = os.path.join(out, "synthetic_baseline.json")
    json.dump({"events": [
        {"event_id": 0, "core_source_raw": "neg", "source_onset": 40.0,
         "far_onset_time": 90.0, "oracle_far_ratio": 0.3, "t_on": 35.0},
    ]}, open(base_path, "w"))
    r = _run(TINY + ["--arm", "dynamic_on_axis", "--core-mean", "16.5", "--core-std", "1.0",
                     "--seed", "1", "--baseline-json", base_path, "--trigger-delay-ms", "8",
                     "--tag", "dynb", "--out", out], cwd=os.getcwd())
    assert r.returncode == 0, r.stderr
    data = json.load(open(os.path.join(out, "dynb.json")))
    assert data["selected_baseline_event"]["event_id"] == 0
    assert data["intervention_on"] == 48.0               # 40 + 8
    assert data["pre_intervention_parity"] is True


def test_runner_rejects_dynamic_without_schedule_or_baseline_event(tmp_path):
    out = str(tmp_path)
    # no foci (core_mean=18, std=0 == bare sheet) -> no eligible event, no schedule -> nonzero exit
    r = _run(["--L", "8", "--density", "15", "--T", "200", "--nc", "6", "--arm", "dynamic_on_axis",
              "--core-mean", "18.0", "--core-std", "0.0", "--seed", "1", "--tag", "rej", "--out", out],
             cwd=os.getcwd())
    assert r.returncode != 0
    assert "no eligible event" in r.stderr
