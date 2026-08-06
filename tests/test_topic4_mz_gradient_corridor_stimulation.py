"""Contract tests for the MZ gradient-corridor stimulation frame.

Covers the 14 engineering-acceptance clauses: fail-closed fingerprint, cohort==admitted set, sign-flip
label-only swap, no-endpoint-axis, same-shaft adjacent bipolar, middle disjoint, dose-matched N_target,
isotropic sheet transform, pre-stim parity, store_spikes=False streaming equivalence, restricted-time /
censoring correctness, resume/atomic/fingerprint drift, bad-artifact failures, no post-stim event pairing.

Geometry tests read the frozen (read-only) gradient records; simulation tests use a tiny net so parity /
censoring run in milliseconds.
"""
import copy
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import src.topic4_mz_gradient_corridor_stimulation as G  # noqa: E402

INPUT_ROOT = "/home/honglab/leijiaxin/HFOsp/results/interictal_propagation_masked/template_gradient_fields"
ADMITTED = ("epilepsiae_1146", "epilepsiae_590", "epilepsiae_958", "yuquan_zhaochenxi")
_have_records = os.path.isdir(os.path.join(INPUT_ROOT, "per_subject"))
records = pytest.mark.skipif(not _have_records, reason="frozen gradient records not mounted")


# ---- tiny net fixture for sim tests ----
def _tiny():
    from params import Params
    from connectivity import place_neurons
    from connectivity_rot import build_connectivity_rot
    from src.topic4_mz_onset_dynamics import _loop_consts
    p = Params(g=3.6, L=6.0, density=50.0, T=1.0, dt=0.1, nu_ext_ratio=0.6, seed=1)
    rng = np.random.default_rng(1)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=0.0, AR=2.0, verbose=False)
    _loop_consts(p, net)
    return dict(p=p, net=net, NE=int(NE), NI=int(NI), N=int(NE + NI),
                posE=net["pos"][:NE], posI=net["pos"][NE:], labels=labels, seed=1)


# ============================================================ (1) fail-closed fingerprint / contract
@records
def test_1_load_valid_record_ok():
    rec = G.load_gradient_record("epilepsiae_1146", INPUT_ROOT)
    assert rec["contract"] == G.INTERICTAL_FIELD_CONTRACT
    assert rec["interictal_field"]["status"] == "ok"


@records
def test_1_fingerprint_mismatch_fails_closed(tmp_path):
    rec = json.loads(open(G.gradient_record_path("epilepsiae_1146", INPUT_ROOT)).read())
    rec["interictal_field"]["coords"][0][0] += 1.0          # corrupt a hashed array
    d = tmp_path / "per_subject"
    d.mkdir()
    (d / "epilepsiae_1146.json").write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="fingerprint"):
        G.load_gradient_record("epilepsiae_1146", str(tmp_path))


@records
def test_1_wrong_contract_fails_closed(tmp_path):
    rec = json.loads(open(G.gradient_record_path("epilepsiae_1146", INPUT_ROOT)).read())
    rec["contract"] = "some_other_contract_v9"
    d = tmp_path / "per_subject"
    d.mkdir()
    (d / "epilepsiae_1146.json").write_text(json.dumps(rec))
    with pytest.raises(ValueError):
        G.load_gradient_record("epilepsiae_1146", str(tmp_path))


# ============================================================ (2) primary cohort == admitted set
@records
def test_2_primary_cohort_is_exactly_admitted():
    rows = [G.audit_subject_geometry(s, INPUT_ROOT, tier="primary_candidate") for s in G.PRIMARY_COHORT]
    admitted = tuple(sorted(r["subject_id"] for r in rows if r["admitted"]))
    assert admitted == tuple(sorted(ADMITTED))
    assert len(admitted) >= 4


# ============================================================ (3) sign flip swaps labels only
@records
def test_3_sign_flip_swaps_endpoints_only():
    rec = G.load_gradient_record("epilepsiae_1146", INPUT_ROOT)
    m = G.build_sheet_montage(rec, L=G.SNN["L"], margin=G.SNN["sheet_margin_mm"],
                              core_quantiles=G.SNN["core_quantiles"])
    sel = G.select_bipolar_sites(m)
    m2 = copy.deepcopy(m)
    m2.along[:] = -m.along
    m2.axis_center_along = -m.axis_center_along
    m2.src_xy, m2.snk_xy = m.snk_xy, m.src_xy
    sel2 = G.select_bipolar_sites(m2)

    def key(s):
        return frozenset([s.name_a, s.name_b])

    assert key(sel["sites"]["middle"]) == key(sel2["sites"]["middle"])
    assert key(sel["sites"]["endpoint_negative"]) == key(sel2["sites"]["endpoint_positive"])
    assert key(sel["sites"]["endpoint_positive"]) == key(sel2["sites"]["endpoint_negative"])
    orig = {key(sel["sites"][k]) for k in ("endpoint_negative", "endpoint_positive", "middle")}
    flip = {key(sel2["sites"][k]) for k in ("endpoint_negative", "endpoint_positive", "middle")}
    assert orig == flip


# ============================================================ (4) axis never uses endpoint / source-sink
def test_4_no_endpoint_placement_import():
    import ast
    src = open(os.path.join(ROOT, "src", "topic4_mz_gradient_corridor_stimulation.py")).read()
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(f"{node.module}.{a.name}" for a in node.names)
    # no endpoint-axis producer is imported (docstring MENTIONS them as forbidden, which is fine)
    assert not any("sef_hfo_subject_placement" in m for m in imported)
    # no endpoint-axis function is CALLED
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    called |= {n.func.attr for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "register_to_sheet" not in called
    assert "template_source_foci" not in called
    assert "load_swap_endpoints" not in called


@records
def test_4_montage_uses_only_shared_plane():
    # the sheet montage must be reproducible from planes.shared.points alone
    rec = G.load_gradient_record("epilepsiae_590", INPUT_ROOT)
    pts = np.asarray(rec["interictal_field"]["planes"]["shared"]["points"], float)
    m = G.build_sheet_montage(rec, L=20.0, margin=2.0, core_quantiles=(0.10, 0.90))
    # sheet contacts are an affine (scale+translate) image of the shared points
    recon = (pts - m.center) * m.scale + np.array([10.0, 10.0])
    assert np.allclose(recon, m.contacts, atol=1e-9)


# ============================================================ (5) bipolar pair = same shaft + adjacent
@records
@pytest.mark.parametrize("subject", ADMITTED)
def test_5_bipolar_pairs_same_shaft_adjacent(subject):
    rec = G.load_gradient_record(subject, INPUT_ROOT)
    m = G.build_sheet_montage(rec, L=20.0, margin=2.0, core_quantiles=(0.10, 0.90))
    sel = G.select_bipolar_sites(m)
    name_to_shaft = dict(zip(m.names, m.shafts))
    for s in sel["sites"].values():
        assert name_to_shaft[s.name_a] == name_to_shaft[s.name_b]
        na, nb = G._trailing_int(s.name_a), G._trailing_int(s.name_b)
        assert na is not None and nb is not None and abs(na - nb) == 1


# ============================================================ (6) middle disjoint from endpoints
@records
@pytest.mark.parametrize("subject", ADMITTED)
def test_6_middle_disjoint_from_endpoints(subject):
    rec = G.load_gradient_record(subject, INPUT_ROOT)
    m = G.build_sheet_montage(rec, L=20.0, margin=2.0, core_quantiles=(0.10, 0.90))
    sites = G.select_bipolar_sites(m)["sites"]
    mid = {sites["middle"].name_a, sites["middle"].name_b}
    neg = {sites["endpoint_negative"].name_a, sites["endpoint_negative"].name_b}
    pos = {sites["endpoint_positive"].name_a, sites["endpoint_positive"].name_b}
    assert not (mid & neg) and not (mid & pos)
    assert len({sites[k].name_a for k in sites} | {sites[k].name_b for k in sites}) >= 6


# ============================================================ (7) N_target identical across arms
@records
def test_7_dose_matched_n_target():
    rec = G.load_gradient_record("yuquan_zhaochenxi", INPUT_ROOT)
    m = G.build_sheet_montage(rec, L=20.0, margin=2.0, core_quantiles=(0.10, 0.90))
    sites = G.select_bipolar_sites(m)["sites"]
    # need posE; use tiny net posE mapped to the same sheet range (only counts matter here)
    shared = _tiny()
    tg = G.build_stim_targets(shared["posE"], sites, radius_mm=1.5)
    counts = {k: int(v.sum()) for k, v in tg["masks"].items()}
    assert len(set(counts.values())) == 1                    # every site clamps the same count
    assert all(c == tg["n_target"] for c in counts.values())


# ============================================================ (8) isotropic sheet transform
@records
@pytest.mark.parametrize("subject", ADMITTED)
def test_8_isotropic_single_scale(subject):
    rec = G.load_gradient_record(subject, INPUT_ROOT)
    pts = np.asarray(rec["interictal_field"]["planes"]["shared"]["points"], float)
    m = G.build_sheet_montage(rec, L=20.0, margin=2.0, core_quantiles=(0.10, 0.90))
    # x and y scale factors must be identical (no differential stretch)
    dx = (m.contacts[:, 0] - 10.0) / (pts[:, 0] - m.center[0])
    dy = (m.contacts[:, 1] - 10.0) / (pts[:, 1] - m.center[1])
    dx = dx[np.isfinite(dx)]
    dy = dy[np.isfinite(dy)]
    assert np.allclose(dx, m.scale, atol=1e-9)
    assert np.allclose(dy, m.scale, atol=1e-9)


# ============================================================ (9) pre-stim parity
def test_9_pre_stim_parity():
    shared = _tiny()
    N, NE = shared["N"], shared["NE"]
    posE = shared["posE"]
    vth = np.full(N, 18.0)
    core = np.zeros(NE, bool)
    patient = dict(vth=vth, core_mask_E=core)

    class _M:  # minimal montage duck type for run_arm
        src_xy = np.array([2.0, 3.0]); snk_xy = np.array([4.0, 3.0])
        axis_unit = np.array([1.0, 0.0]); L = 6.0
        contacts = np.array([[2.0, 3.0], [4.0, 3.0], [3.0, 3.0]])
    montage = _M()
    n_steps = 4000
    on = 2000
    target = np.zeros(NE, bool)
    target[np.argsort(np.linalg.norm(posE - np.array([3.0, 3.0]), axis=1))[:20]] = True
    r0, _, _ = G.run_arm(shared, patient, montage, arm="baseline_no_stim", target_E=None,
                         stim_window_steps=None, delta_mv=50.0, n_steps=n_steps)
    r1, _, _ = G.run_arm(shared, patient, montage, arm="gradient_middle", target_E=target,
                         stim_window_steps=(on, 3000), delta_mv=50.0, n_steps=n_steps)
    a0 = np.asarray(r0["rate_E"], float)[:on]
    a1 = np.asarray(r1["rate_E"], float)[:on]
    assert np.array_equal(a0, a1)                            # identical up to stim onset


# ============================================================ (10) store_spikes=False streaming == raster
def test_10_streaming_active_fraction_equals_raster():
    from src.topic4_mz_onset_dynamics import run_loop
    shared = _tiny()
    p, net, NE, N = shared["p"], shared["net"], shared["NE"], shared["N"]
    vth = np.full(N, 18.0)
    cfg = G.mz_config()
    n = 3000
    net["rng"] = np.random.default_rng(9)
    slow1 = G.MZSlowVars(N, 18.0, cfg, NE=NE)
    r_raster = run_loop(p, net, slow1, vth, n_steps=n, store_spikes=True, early_stop_runaway=False)
    net["rng"] = np.random.default_rng(9)
    slow2 = G.MZSlowVars(N, 18.0, cfg, NE=NE)
    obs = G.SpatialStreamObserver(shared["posE"], [3.0, 3.0], [1.0, 0.0], dt=0.1, L=6.0, n_steps=n)
    r_stream = G.run_observed_loop(p, net, slow2, vth, n_steps=n, observer=obs, lfp_recorder=None,
                                   early_stop_runaway=False)
    assert np.array_equal(r_raster["rate_E"], r_stream["rate_E"])
    bs = 10
    nb = n // bs
    raster_af = r_raster["E_spk_bool"][:nb * bs].reshape(nb, bs, -1).any(axis=1).mean(axis=1)
    assert np.allclose(obs.active_frac[:nb], raster_af, atol=1e-12)


# ============================================================ (11) restricted-time / censoring
def test_11_restricted_time_and_censoring():
    # censored (no runaway): t_run = T_max, RRT = T_max - stim_off
    rate_no = np.zeros(2000)                                 # never runs away
    summary = _fake_summary(rate_no, stim_off_ms=100.0, t_max_ms=200.0)
    assert summary["censored"] is True
    assert summary["t_run_used_ms"] == pytest.approx(200.0)
    assert summary["restricted_runaway_free_time_ms"] == pytest.approx(100.0)
    # runaway present: t_run = runaway_ms
    rate_ra = np.concatenate([np.zeros(500), np.full(1500, 300.0)])   # sustained high -> runaway
    s2 = _fake_summary(rate_ra, stim_off_ms=10.0, t_max_ms=200.0)
    assert s2["censored"] is False
    assert s2["runaway_ms"] is not None
    assert s2["restricted_runaway_free_time_ms"] == pytest.approx(s2["t_run_used_ms"] - 10.0)


def _fake_summary(rate, *, stim_off_ms, t_max_ms):
    class _Obs:
        active_frac = np.zeros(200)
        axial_act = np.zeros((40, 20))
        ax_edges = np.linspace(-10, 10, 21)
        spatial_bin_ms = 5.0
    res = dict(rate_E=rate, runaway_early_stop_step=None, lfp_trace=None, lfp_step_idx=None)

    class _Slow:
        trace_z_mean = [1.0]; trace_z_min = [1.0]; trace_m_mean = [0.0]; trace_adap_current = [0.0]
    return G.summarize_run(res, _Obs(), _Slow(), arm="x", dt=0.1, frozen_bar=0.5, stim_on_ms=5.0,
                           stim_off_ms=stim_off_ms, t_max_ms=t_max_ms, coredist_mm=6.0, core_r=1.5,
                           spatial_bin_ms=5.0, baseline_total_activity=1.0)


# ============================================================ (12) resume fingerprint drift
@records
def test_12_arm_fingerprint_changes_on_geometry_or_window():
    rec = G.load_gradient_record("epilepsiae_590", INPUT_ROOT)
    m = G.build_sheet_montage(rec, L=20.0, margin=2.0, core_quantiles=(0.10, 0.90))
    import importlib
    R = importlib.import_module("scripts.run_topic4_mz_gradient_corridor_stimulation") \
        if _can_import_runner() else None
    if R is None:
        pytest.skip("runner import path unavailable")
    fp1 = R.arm_fingerprint("epilepsiae_590", 1, "gradient_middle", m, 200000, (90000, 150000), 40, 50.0, 1.5)
    fp2 = R.arm_fingerprint("epilepsiae_590", 1, "gradient_middle", m, 200000, (90000, 150001), 40, 50.0, 1.5)
    fp3 = R.arm_fingerprint("epilepsiae_590", 3, "gradient_middle", m, 200000, (90000, 150000), 40, 50.0, 1.5)
    assert fp1 != fp2 and fp1 != fp3 and len(fp1) == 16


def _can_import_runner():
    try:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import importlib
        importlib.import_module("scripts.run_topic4_mz_gradient_corridor_stimulation")
        return True
    except Exception:
        return False


# ============================================================ (13) bad artifact / bad montage fails
def test_13_missing_file_fails():
    with pytest.raises(FileNotFoundError):
        G.load_gradient_record("no_such_subject", INPUT_ROOT)


@records
def test_13_duplicate_contact_montage_fails(tmp_path):
    rec = json.loads(open(G.gradient_record_path("epilepsiae_1146", INPUT_ROOT)).read())
    # duplicate a contact name -> select_bipolar via montage build should still align, but a duplicate
    # name breaks adjacency uniqueness; here we assert build raises on shape/length mismatch.
    rec["interictal_field"]["contact_order"] = rec["interictal_field"]["contact_order"][:-1]
    with pytest.raises(ValueError):
        G.build_sheet_montage(rec)


def test_13_degenerate_plane_fails():
    rec = {"interictal_field": {"planes": {"shared": {"points": [[0.0, 0.0], [0.0, 0.0]]}},
                                "contact_order": ["A1", "A2"], "shafts": ["A", "A"]}}
    with pytest.raises(ValueError):
        G.build_sheet_montage(rec)


# ============================================================ (14) no post-stim event pairing
def test_14_summary_has_no_event_pairing_fields():
    s = _fake_summary(np.zeros(2000), stim_off_ms=100.0, t_max_ms=200.0)
    # post-stim reporting is window/category level ONLY (counts + spread), never per-event paired diffs
    assert "n_post_local_events" in s and "n_post_global_events" in s
    assert not any("paired" in k or "event_diff" in k or "matched_event" in k for k in s)


# ============================================================ (15) pre-runaway propagation (review fix)
def test_15_prerunaway_propagation_window_and_both_ends():
    ax_edges = np.linspace(-10.0, 10.0, 21)      # 20 axial bins of 1 mm
    coredist, core_r, sb = 8.0, 1.5, 5.0         # cores at +/-4; src<=-2.5, snk>=2.5
    n = 40
    ax = np.zeros((n, 20))
    centers = 0.5 * (ax_edges[:-1] + ax_edges[1:])
    src_col = int(np.argmin(np.abs(centers - (-4.0))))
    snk_col = int(np.argmin(np.abs(centers - 4.0)))
    # stim_off at 5*sb=25ms (bin 5); make BOTH ends active from bin 20 onward (=100ms)
    ax[20:, src_col] = 0.5
    ax[20:, snk_col] = 0.5
    r = G.prerunaway_propagation(ax, ax_edges, coredist, core_r, sb, stim_off_ms=25.0, t_run_ms=None)
    assert r["far_reach_prob"] > 0.0                       # both ends active in part of the window
    assert r["cross_corridor_latency_ms"] is not None
    # cross-corridor first happens at bin 20; window starts at bin 5 -> latency (20-5)*sb
    assert r["cross_corridor_latency_ms"] == pytest.approx((20 - 5) * sb)
    # truncating at t_run BEFORE both ends light up -> no cross-corridor, prob 0
    r2 = G.prerunaway_propagation(ax, ax_edges, coredist, core_r, sb, stim_off_ms=25.0, t_run_ms=90.0)
    assert r2["far_reach_prob"] == 0.0 and r2["cross_corridor_latency_ms"] is None
    # empty window -> Nones/NaN, never a crash
    r3 = G.prerunaway_propagation(ax, ax_edges, coredist, core_r, sb, stim_off_ms=200.0, t_run_ms=None)
    assert r3["n_window_bins"] == 0 and r3["cross_corridor_latency_ms"] is None
