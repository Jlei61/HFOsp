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
                              d_sweep="3000:80", include_anchor=False, eta_r=15.0, clamp_val=0.8, arms="", use_H=False, alpha_H=0.0, tau_H=6000.0)
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


def _step_traces_v2(dt=0.1, T=10000.0, t_step=3000.0):
    """Step traces incl. core/surround: baseline -> bounded M4 (core depletes more than surround, both recruit)."""
    rate, SG, qI, area = _step_traces(dt, T, t_step)
    n = int(T / dt); t = np.arange(n) * dt
    q_core = np.where(t < t_step, 1.0, 0.05)         # core q_I depletes to the floor
    q_surr = np.where(t < t_step, 1.0, 0.40)         # surround less depleted -> a gradient forms
    nf = n // int(round(25.0 / dt)); tf = np.arange(nf) * 25.0
    core_act = np.where(tf < t_step, 3.0, 90.0)      # core rate forms at onset
    surr_act = np.where(tf < t_step, 2.0, 40.0)      # surround recruited to a plateau
    return rate, SG, qI, area, core_act, surr_act, q_core, q_surr


def test_formed_state_uses_core_surround():
    rate, SG, qI, area, ca, sa, qc, qs = _step_traces_v2(t_step=3000.0)
    res = A.formed_state_time(rate, SG, qI, area, dt=0.1, movie_bin_ms=25.0, window_ms=1500.0,
                              core_activity=ca, surround_activity=sa, trace_q_core=qc, trace_q_surround=qs)
    assert res["used_core_surround"] is True
    assert res["t_form"] is not None and 2800.0 <= res["t_form"] <= 3400.0


def test_formed_state_rejects_when_core_never_forms():
    """Global rate high but CORE rate stays at baseline -> NOT a formed M4 core -> t_form None."""
    rate, SG, qI, area, ca, sa, qc, qs = _step_traces_v2(t_step=3000.0)
    ca[:] = 2.5                                       # core never elevates (only surround/global do)
    qc[:] = 1.0                                       # core q_I never depletes
    res = A.formed_state_time(rate, SG, qI, area, dt=0.1, movie_bin_ms=25.0,
                              core_activity=ca, surround_activity=sa, trace_q_core=qc, trace_q_surround=qs)
    assert res["t_form"] is None


def test_t_form_sensitivity_stable_on_clean_step():
    rate, SG, qI, area, ca, sa, qc, qs = _step_traces_v2(t_step=3000.0)
    sens = A.t_form_sensitivity(rate, SG, qI, area, dt=0.1, movie_bin_ms=25.0,
                                core_activity=ca, surround_activity=sa, trace_q_core=qc, trace_q_surround=qs)
    assert sens["stable"] is True                    # a clean step -> t_form stable across window+threshold
    assert sens["spread_ms"] is not None and sens["spread_ms"] <= 500.0
    assert all(v is not None for v in sens["t_form_by_variant"].values())


# ---- recovery matcher: post-offset events vs slow-off IEDs (review 07-22 P0) ------------------
def _feat(t_on, dur, peak, area, ratio, mode):
    return dict(t_on=t_on, dur=dur, peak=peak, area=area, core_surr_ratio=ratio, mode=np.asarray(mode, float))


def test_recovery_match_recovered_when_similar():
    m = np.ones((4, 4))
    base = [_feat(i * 300.0, 25.0, 40.0, 0.20, 1.5, m) for i in range(10)]
    post = [_feat(15000 + i * 320.0, 24.0, 42.0, 0.19, 1.55, m) for i in range(6)]   # matched distribution
    r = A.recovery_match(base, post)
    assert r["recovered"] is True
    assert {"duration", "iei", "peak_rate", "active_area", "core_surround_ratio", "spatial_mode"} <= set(r["per_metric"])


def test_recovery_match_rejects_fragment():
    m = np.ones((4, 4)); m2 = np.zeros((4, 4)); m2[0, 0] = 1.0          # wrong spatial mode
    base = [_feat(i * 300.0, 25.0, 40.0, 0.20, 1.5, m) for i in range(10)]
    post = [_feat(15000 + i * 4000.0, 200.0, 130.0, 0.80, 5.0, m2) for i in range(3)]  # long/sparse/broad/wrong
    r = A.recovery_match(base, post)
    assert r["recovered"] is False


def test_recovery_match_rejects_too_few():
    m = np.ones((4, 4))
    base = [_feat(i * 300.0, 25.0, 40.0, 0.20, 1.5, m) for i in range(10)]
    r = A.recovery_match(base, [_feat(15000.0, 25.0, 40.0, 0.20, 1.5, m)], min_post=3)
    assert r["recovered"] is False and "too few" in r["reason"]


def test_arm_event_features_extracts():
    dt = 0.1; n = 3000                                                  # 300 ms
    rate = np.full(n, 5.0); rate[1000:1500] = 60.0                      # event [100,150] ms peaks 60
    nf = n // 250
    movie = np.zeros((nf, 4, 4)); movie[4:6] = 0.5                      # frames 4-5 = [100,150) ms
    core = np.full(nf, 3.0); core[4:6] = 80.0
    surr = np.full(nf, 2.0); surr[4:6] = 40.0
    row = dict(events=[[100.0, 150.0]])
    npz = dict(rate=rate.astype("float32"), movie=movie.astype("float32"),
               core_activity=core.astype("float32"), surround_activity=surr.astype("float32"))
    feats = A.arm_event_features(row, npz, dt=0.1, movie_bin_ms=25.0, activity_bin_ms=25.0)
    assert len(feats) == 1
    assert feats[0]["dur"] == 50.0 and abs(feats[0]["peak"] - 60.0) < 1e-5
    assert feats[0]["core_surr_ratio"] > 1.5                            # 80/40 = 2


def test_verify_pre_onset_identity():
    n = 5000
    a = dict(rate=np.arange(n, dtype=float), trace_qI_mean=np.ones(n), trace_SG=np.zeros(n))
    c = dict(rate=np.arange(n, dtype=float), trace_qI_mean=np.ones(n), trace_SG=np.zeros(n))
    c["rate"] = c["rate"].copy(); c["rate"][3000:] += 1.0               # differ only AFTER onset (300 ms)
    assert A.verify_pre_onset_identity(a, c, onset_ms=300.0, dt=0.1)["pre_onset_identical"] is True
    c["rate"][100] += 1.0                                              # now differ BEFORE onset
    assert A.verify_pre_onset_identity(a, c, onset_ms=300.0, dt=0.1)["pre_onset_identical"] is False
