"""TDD for the SNN-native M4 exit RUNNER provenance + crash-safe resume (Phase 0, task brief §3).

The Stage-2 arms runner previously did `pool.map(...)` (blocking) then wrote ONE combined JSON at the
very end: an interrupt lost every arm, there was no resume, no manifest, and the JSON carried neither
base_sha nor engine_versions and dropped persist_onset_ms from cfg_effective. Phase 0 makes each arm's
output land the moment it finishes, records a run_manifest (pending/running/complete/error), skips
completed arms on --resume, and stamps full provenance. Contract clauses -> one test each:

  1. provenance          : _provenance() carries base_sha + engine_versions(guarded engine) + argv
  2. cfg_effective        : includes persist_onset_ms (+ tau_p/tau_p_down/eta_r) -- was silently dropped
  3. label distinguishes  : d-sweep label encodes onset + tau_down (not just tau_up/eta)
  4. per-arm write/load   : each arm -> its own JSON+NPZ, reloadable
  5. error rows not done  : an arm whose row has 'error' is NOT counted complete (re-runs on resume)
  6. manifest states      : complete / error / running / pending are all represented
  7. incremental + resume : orchestrator writes each arm as it completes; --resume skips completed arms
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_m4_snn_native_exit as R  # noqa: E402
import analyze_m4_snn_native_exit as A  # noqa: E402
from slow_field import SpatialSlowFieldConfig  # noqa: E402


def _cfg():
    return SpatialSlowFieldConfig(use_qI=True, k_q=0.10, use_SG=True, alpha_G=16.0,
                                  use_persist=True, tau_p=3000.0, tau_p_down=12000.0, eta_r=80.0,
                                  a50_p=0.3, sigma_p=1.5, p50_r=0.15, n_r=4.0, persist_onset_ms=2500.0)


# ---- Clause 1: provenance ------------------------------------------------------------------
def test_provenance_has_base_sha_and_engine_versions():
    prov = R._provenance()
    assert {"base_sha", "engine_versions", "argv"} <= set(prov)
    assert isinstance(prov["engine_versions"], dict) and prov["engine_versions"]
    assert any("kick_probe" in k for k in prov["engine_versions"])   # snapshots the real guarded engine


# ---- Clause 2: cfg_effective includes persist_onset_ms -------------------------------------
def test_cfg_effective_includes_persist_onset():
    ce = R._cfg_effective(_cfg())
    assert ce["persist_onset_ms"] == 2500.0
    assert ce["tau_p"] == 3000.0 and ce["tau_p_down"] == 12000.0 and ce["eta_r"] == 80.0


# ---- Clause 3: d-sweep label encodes onset + tau_down --------------------------------------
def test_d_sweep_label_encodes_onset():
    import types
    a = types.SimpleNamespace(tau_p=3000.0, theta_p=0.0, a50_p=0.3, sigma_p=1.5, p50_r=0.15, n_r=4.0,
                              tau_p_down=12000.0, persist_onset_ms=2500.0, T=25000.0,
                              d_sweep="3000:80", include_anchor=False, eta_r=15.0, clamp_val=0.8, arms="")
    (label, cfg, T, perturb) = R._build_arms(a)[0]
    assert cfg.persist_onset_ms == 2500.0                 # propagated, not dropped
    assert "on2500" in label and "dn12000" in label       # onset + tau_down both in the label
    # onset 0 -> not in label (backward-compatible naming)
    a.persist_onset_ms = 0.0
    (label0, _, _, _) = R._build_arms(a)[0]
    assert "on" not in label0.replace("eta", "")          # no onset token when onset==0


# ---- Clause 4: per-arm write + load --------------------------------------------------------
def test_write_and_load_arm_result(tmp_path):
    d = str(tmp_path)
    jp, npzp = R._write_arm_result(d, dict(label="D_test", verdict="no_runaway",
                                           termination_class="fragment"),
                                   dict(rate=np.arange(5, dtype=np.float32)))
    assert os.path.exists(jp) and os.path.exists(npzp)
    loaded = R._load_completed_arms(d)
    assert "D_test" in loaded and loaded["D_test"]["verdict"] == "no_runaway"


# ---- Clause 5: error rows are not "complete" -----------------------------------------------
def test_load_completed_skips_error_rows(tmp_path):
    d = str(tmp_path)
    R._write_arm_result(d, dict(label="ok", verdict="x"), {})
    R._write_arm_result(d, dict(label="bad", error="boom"), {})
    loaded = R._load_completed_arms(d)
    assert "ok" in loaded and "bad" not in loaded          # error -> re-run on resume, not skipped


# ---- Clause 6: manifest states -------------------------------------------------------------
def test_manifest_states():
    specs = [(nm, _cfg(), 1000.0, None) for nm in ("A", "B", "C", "D")]
    results = {"A": dict(label="A", verdict="x"), "B": dict(label="B", error="boom")}
    m = R._manifest_dict(specs, results, running={"C"}, provenance={"base_sha": "deadbee"}, meta={})
    st = {k: v["status"] for k, v in m["arms"].items()}
    assert st == dict(A="complete", B="error", C="running", D="pending")
    assert m["provenance"]["base_sha"] == "deadbee"
    assert m["arms"]["A"]["cfg_effective"]["persist_onset_ms"] == 2500.0   # cfg snapshot per arm


# ---- Clause 7: incremental write + resume --------------------------------------------------
def test_orchestrate_incremental_and_resume(tmp_path):
    d = str(tmp_path)
    calls = []

    def fake_run(spec):
        calls.append(spec[0])
        return dict(label=spec[0], verdict="no_runaway", termination_class="fragment"), \
            dict(rate=np.zeros(3, np.float32))

    specs = [("A", _cfg(), 1000.0, None), ("B", _cfg(), 1000.0, None)]
    rows = R._orchestrate_arms(specs, d, "t", 1, provenance={}, meta={}, workers=1,
                               run_one=fake_run, resume=False)
    assert {r["label"] for r in rows} == {"A", "B"}
    assert sorted(calls) == ["A", "B"]
    arm_dir = R._arm_dir(d, "t", 1)
    assert os.path.exists(os.path.join(arm_dir, "A.json"))          # landed incrementally
    assert os.path.exists(os.path.join(d, "run_manifest_t_seed1.json"))

    # resume with BOTH already complete -> no re-run, still returns all rows
    calls.clear()
    rows2 = R._orchestrate_arms(specs, d, "t", 1, provenance={}, meta={}, workers=1,
                                run_one=fake_run, resume=True)
    assert calls == []
    assert {r["label"] for r in rows2} == {"A", "B"}

    # drop B's per-arm output -> resume re-runs ONLY B
    os.remove(os.path.join(arm_dir, "B.json"))
    calls.clear()
    rows3 = R._orchestrate_arms(specs, d, "t", 1, provenance={}, meta={}, workers=1,
                                run_one=fake_run, resume=True)
    assert calls == ["B"]
    assert {r["label"] for r in rows3} == {"A", "B"}


# ---- Phase 2 frozen exit atlas: pure helpers -----------------------------------------------
def test_atlas_cfg_freezes_slow_coords():
    cfg = R._atlas_cfg(q_core=0.4, S_G=0.2, J_exit=8.0)
    assert cfg.k_q == 0.0 and cfg.q_init == 0.4                     # q_I FROZEN at q_core (no ODE)
    assert cfg.use_SG and cfg.clamp_SG == 0.2                       # S_G FROZEN (divisive containment)
    assert cfg.use_persist and cfg.clamp_persist == 1.0 and cfg.p50_r == 0.0 and cfg.eta_r == 8.0  # J_exit=eta_r*Phi(1)
    assert R._atlas_cfg(0.9, 0.0, 0.0).use_persist is False         # J_exit=0 -> no recovery current


def test_classify_atlas():
    assert R._classify_atlas(200.0, 0.1, 0.9, runaway=1200.0) == "runaway"
    assert R._classify_atlas(0.5, 0.1, 0.0, runaway=None) == "low"
    assert R._classify_atlas(40.0, 0.2, 0.5, runaway=None) == "bounded_high"
    assert R._classify_atlas(40.0, 0.9, 0.5, runaway=None) == "bounded_oscillatory"
    assert R._classify_atlas(20.0, 0.3, 0.0, runaway=None) == "fragment"


def test_build_atlas_cells_grid_and_ics():
    import types
    a = types.SimpleNamespace(q_core_grid="0.05,0.9", sg_grid="0.0,0.4", j_exit_grid="0.0", T=2500.0)
    cells = R._build_atlas_cells(a)
    assert len(cells) == 2 * 2 * 1 * 2                              # q x S_G x J_exit x {cold, warm}
    warm = next(c for c in cells if c[0].endswith("warm"))
    cold = next(c for c in cells if c[0].endswith("cold"))
    assert warm[3] is True and cold[3] is False                    # 4th tuple element = warm flag
    assert warm[1].q_init == 0.05 or warm[1].q_init == 0.9         # cfg carries a grid q_core


# ---- Phase 1 formed-state detector (data-driven t_form; NOT assumed 2500ms) ----------------
def _step_traces(dt=0.1, T=10000.0, t_step=3000.0):
    n = int(T / dt)
    t = np.arange(n) * dt
    rate = np.where(t < t_step, 2.0, 80.0)          # baseline -> bounded plateau
    SG = np.where(t < t_step, 0.0, 0.4)             # containment engages at formation
    qI = np.where(t < t_step, 1.0, 0.1)             # inhibitory resource depletes at formation
    nf = n // int(round(25.0 / dt))
    tf = np.arange(nf) * 25.0
    area = np.where(tf < t_step, 0.02, 0.30)        # spatial extent established at formation
    return rate, SG, qI, area


def test_formed_state_time_detects_step():
    rate, SG, qI, area = _step_traces(t_step=3000.0)
    res = A.formed_state_time(rate, SG, qI, area, dt=0.1, movie_bin_ms=25.0, window_ms=1500.0)
    assert res["t_form"] is not None
    assert 2800.0 <= res["t_form"] <= 3400.0        # ~3000ms up to smoothing + probe resolution


def test_formed_state_time_none_when_no_bounded_state():
    dt, n = 0.1, 50000
    rate = np.full(n, 2.0); SG = np.zeros(n); qI = np.ones(n)   # never leaves baseline
    area = np.full(n // 250, 0.02)
    res = A.formed_state_time(rate, SG, qI, area, dt=dt, movie_bin_ms=25.0)
    assert res["t_form"] is None                    # end-of-run is not a bounded state -> no formation
