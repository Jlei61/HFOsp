"""Tests for src/topic5_event_resolved_alignment.py (Topic 5 event-resolved axis_bias secondary).

Contracts under test (spec §5): C1 positional alignment (3 raises), C2 cluster↔template map,
C3 per-class own plane, C4 per-event support + pinned sigma, C8 same reduction, C7 block-level
null, C10 stubs. Plus a real-data integration test on broad epilepsiae_1077.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import src.topic5_event_resolved_alignment as mod
from src.interictal_propagation import _legacy_hist_mean_rank
from src.propagation_contact_plane_readout import make_plane_grid, R_smooth_rank

REAL_1077 = Path("results/interictal_propagation_masked_broad/per_subject/epilepsiae_1077.json")


# --------------------------------------------------------------------------- synthetic C1 fixture
def _synthetic_ev(n_ch=8, n_ev=40, seed=0):
    """A deterministic synthetic loader payload: ranks/bools/names/blocks/times."""
    rng = np.random.default_rng(seed)
    names = [f"CH{i}" for i in range(n_ch)]
    bools = rng.random((n_ch, n_ev)) < 0.6
    # guarantee >=3 participating per event
    for e in range(n_ev):
        if bools[:, e].sum() < 3:
            bools[rng.choice(n_ch, 3, replace=False), e] = True
    ranks = np.zeros((n_ch, n_ev))
    for e in range(n_ev):
        p = np.where(bools[:, e])[0]
        order = rng.permutation(p.size)
        ranks[p, e] = order
        # phantom ints for non-participating (the bug C5 guards against)
        npp = np.where(~bools[:, e])[0]
        ranks[npp, e] = rng.integers(0, n_ch, npp.size)
    block_ids = np.repeat(np.arange(4), n_ev // 4)[:n_ev]
    return {"ranks": ranks, "bools": bools, "channel_names": names,
            "block_ids": block_ids, "event_abs_times": np.arange(n_ev, dtype=float),
            "lag_raw": ranks}


def _labels_json_for(ev, labels, *, tamper=None):
    """Build a labels JSON consistent with ev+labels (producer template via _legacy_hist_mean_rank)."""
    from src.interictal_propagation import _valid_event_indices
    ranks, bools = ev["ranks"], ev["bools"]
    valid = _valid_event_indices(bools, 3)
    lab = np.asarray(labels)[: valid.size]
    clusters = []
    for k in (0, 1):
        sel = valid[lab == k]
        templ = _legacy_hist_mean_rank(ranks[:, sel], bools[:, sel])
        tr = np.argsort(np.argsort(templ)).tolist()
        clusters.append({"cluster_id": k, "n_events": int(sel.size), "template_rank": tr})
    js = {"dataset": "synthetic", "subject": "s0",
          "channel_names": list(ev["channel_names"]),
          "adaptive_cluster": {"stable_k": 2, "chosen_k": 2,
                               "n_valid_events": int(valid.size),
                               "labels": lab.tolist(), "clusters": clusters}}
    if tamper == "channels":
        js["channel_names"] = ["X"] + js["channel_names"][1:]
    elif tamper == "count":
        js["adaptive_cluster"]["clusters"][0]["n_events"] += 1
        js["adaptive_cluster"]["clusters"][1]["n_events"] -= 1
    elif tamper == "template":
        js["adaptive_cluster"]["clusters"][0]["template_rank"] = \
            list(reversed(js["adaptive_cluster"]["clusters"][0]["template_rank"]))
    return js, valid, lab


@pytest.fixture
def patched_loader(tmp_path, monkeypatch):
    """Returns a factory that wires a synthetic ev + labels JSON into load_event_labels_ranks."""
    def _make(tamper=None, n_ev=40):
        ev = _synthetic_ev(n_ev=n_ev)
        # deterministic labels: first half class 0, rest class 1 (after valid filter)
        from src.interictal_propagation import _valid_event_indices
        nvalid = _valid_event_indices(ev["bools"], 3).size
        labels = np.array([0] * (nvalid // 2) + [1] * (nvalid - nvalid // 2))
        js, valid, lab = _labels_json_for(ev, labels, tamper=tamper)
        d = tmp_path / "labels"; d.mkdir(exist_ok=True)
        json.dump(js, open(d / "synthetic_s0.json", "w"))
        monkeypatch.setattr(mod, "load_subject_propagation_events", lambda _p: ev)
        return dict(labels_dir=str(d), ev=ev, valid=valid, labels=lab)
    return _make


# --------------------------------------------------------------------------- C1
def test_c1_happy_path_synthetic(patched_loader):
    cfg = patched_loader()
    b = mod.load_event_labels_ranks("synthetic", "s0", labels_dir=cfg["labels_dir"],
                                    lagpat_dir="/unused")
    assert b["labels"].size == cfg["valid"].size
    assert b["masked"].shape[1] == cfg["valid"].size
    # C5: masked is NaN-dropped for non-participating, finite for participating
    assert np.isnan(b["masked"][~b["bools"]]).all()
    assert np.isfinite(b["masked"][b["bools"]]).all()
    assert b["n_blocks"] >= 1


def test_c1_channel_mismatch_raises(patched_loader):
    cfg = patched_loader(tamper="channels")
    with pytest.raises(ValueError, match="channel_names mismatch"):
        mod.load_event_labels_ranks("synthetic", "s0", labels_dir=cfg["labels_dir"], lagpat_dir="/x")


def test_c1_count_mismatch_raises(patched_loader):
    cfg = patched_loader(tamper="count")
    with pytest.raises(ValueError, match="count"):
        mod.load_event_labels_ranks("synthetic", "s0", labels_dir=cfg["labels_dir"], lagpat_dir="/x")


def test_c1_template_mismatch_raises(patched_loader):
    cfg = patched_loader(tamper="template")
    with pytest.raises(ValueError, match="template"):
        mod.load_event_labels_ranks("synthetic", "s0", labels_dir=cfg["labels_dir"], lagpat_dir="/x")


# --------------------------------------------------------------------------- C2
def test_c2_clean_map():
    n = 10
    t_a = np.arange(n, dtype=float)
    t_b = (n - 1 - t_a)            # near-mirror (anti-correlated): the common case
    c0 = t_a + np.random.default_rng(0).normal(0, 0.3, n)
    c1 = t_b + np.random.default_rng(1).normal(0, 0.3, n)
    r = mod.map_clusters_to_templates(np.argsort(np.argsort(c0)), np.argsort(np.argsort(c1)),
                                      t_a, t_b, margin=0.30)
    assert not r["ambiguous"] and r["map"] == {0: "t_a", 1: "t_b"}


def test_c2_near_mirror_ambiguous():
    # both clusters ~ t_a (weak diagonal vs off-diagonal) -> ambiguous (non-bijection or low margin)
    n = 10
    t_a = np.arange(n, dtype=float); t_b = t_a + np.random.default_rng(2).normal(0, 0.2, n)
    c0 = t_a.copy(); c1 = t_a.copy()
    r = mod.map_clusters_to_templates(c0, c1, t_a, t_b, margin=0.30)
    assert r["ambiguous"]


def test_c2_signed_not_abs():
    # anti-correlated cluster must NOT map to the template it mirrors (signed corr)
    n = 12
    t_a = np.arange(n, dtype=float); t_b = (n - 1 - t_a)
    c0 = t_a.copy(); c1 = t_b.copy()
    r = mod.map_clusters_to_templates(c0, c1, t_a, t_b, margin=0.30)
    assert r["map"] == {0: "t_a", 1: "t_b"} and not r["ambiguous"]


# --------------------------------------------------------------------------- M field metric (C3/C4/C8)
def _plane(names, xs, ys):
    return {"channels": [{"name": n, "x_norm": float(x), "y_norm": float(y),
                          "typical_rank": 0.5, "support": 0.5}
                         for n, x, y in zip(names, xs, ys)]}


def _bundle_two_classes():
    """2 classes; class-0 events on A-channels, class-1 events on B-channels (disjoint names)."""
    X, Y = make_plane_grid()
    a_names = [f"A{i}" for i in range(6)]; b_names = [f"B{i}" for i in range(6)]
    names = a_names + b_names
    xs = np.linspace(0.1, 0.9, 6)
    plane_a = _plane(a_names, xs, np.zeros(6))
    plane_b = _plane(b_names, xs, np.zeros(6))
    n_ch = 12; n_ev = 8
    bools = np.zeros((n_ch, n_ev), bool); masked = np.full((n_ch, n_ev), np.nan)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    for e in range(n_ev):
        idx = range(0, 6) if labels[e] == 0 else range(6, 12)
        for r, j in enumerate(idx):
            bools[j, e] = True
            masked[j, e] = r / 5.0
    bundle = {"masked": masked, "bools": bools, "labels": labels,
              "channel_names": names, "block_ids": np.array([0, 0, 1, 1, 0, 0, 1, 1]),
              "event_abs_times": np.arange(n_ev, dtype=float), "valid_ev": np.arange(n_ev)}
    return bundle, plane_a, plane_b, X, Y


def test_m_uses_class_own_plane_C3():
    bundle, plane_a, plane_b, X, Y = _bundle_two_classes()
    # ictal fields on each plane (gradient matching the templates)
    ia = R_smooth_rank({"channels": [dict(c, typical_rank=c["x_norm"]) for c in plane_a["channels"]]},
                       X, Y, 0.2, 0.15)
    ib = R_smooth_rank({"channels": [dict(c, typical_rank=c["x_norm"]) for c in plane_b["channels"]]},
                       X, Y, 0.2, 0.15)
    out = mod.per_event_field_alignment(
        bundle, plane_by_label={0: plane_a, 1: plane_b},
        ictal_field_by_label={0: ia, 1: ib}, sigma_by_label={0: 0.2, 1: 0.2})
    # class-1 events use B-channels which are ONLY in plane_b -> they must resolve ok,
    # proving plane_by_label[1] (t_b) was used, not plane_a
    cls1 = [r for r in out["per_event"] if r["label"] == 1]
    assert all(r["status"] == "ok" for r in cls1)
    # and if we wrongly gave plane_a to class 1, they'd fail (too_few_plane_contacts)
    out_wrong = mod.per_event_field_alignment(
        bundle, plane_by_label={0: plane_a, 1: plane_a},
        ictal_field_by_label={0: ia, 1: ia}, sigma_by_label={0: 0.2, 1: 0.2})
    cls1w = [r for r in out_wrong["per_event"] if r["label"] == 1]
    assert all(r["status"] == "unresolved" for r in cls1w)
    # both-plane outputs exist (P0): each event carries align0 AND align1
    assert all(("align0" in r and "align1" in r) for r in out["per_event"])


def test_m_support_is_event_participation_C4():
    # plane channels have aggregate support=0.0; _event_field must force support=1.0 and still build
    bundle, plane_a, plane_b, X, Y = _bundle_two_classes()
    for c in plane_a["channels"]:
        c["support"] = 0.0
    ia = R_smooth_rank({"channels": [dict(c, typical_rank=c["x_norm"], support=1.0)
                                     for c in plane_a["channels"]]}, X, Y, 0.2, 0.15)
    out = mod.per_event_field_alignment(
        bundle, plane_by_label={0: plane_a, 1: plane_b},
        ictal_field_by_label={0: ia, 1: ia}, sigma_by_label={0: 0.2, 1: 0.2})
    cls0 = [r for r in out["per_event"] if r["label"] == 0]
    assert any(r["status"] == "ok" for r in cls0)   # built despite aggregate support 0


def test_m_overlap_gate_C8():
    bundle, plane_a, plane_b, X, Y = _bundle_two_classes()
    # ictal field placed far away (x≈1.5 corner) so no support overlap with events at x∈[0.1,0.9]
    far = {"channels": [{"name": f"A{i}", "x_norm": 1.5, "y_norm": 1.0,
                         "typical_rank": 0.5, "support": 1.0} for i in range(6)]}
    fa = R_smooth_rank(far, X, Y, 0.05, 0.15)
    out = mod.per_event_field_alignment(
        bundle, plane_by_label={0: plane_a, 1: plane_b},
        ictal_field_by_label={0: fa, 1: fa}, sigma_by_label={0: 0.05, 1: 0.05})
    cls0 = [r for r in out["per_event"] if r["label"] == 0]
    assert all(r["status"] == "unresolved" for r in cls0)


# --------------------------------------------------------------------------- M1d (headroom, no-sign)
def test_m1d_headroom_gate():
    masked = np.full((6, 10), np.nan); bools = np.zeros((6, 10), bool)
    bundle = {"masked": masked, "bools": bools, "labels": np.zeros(10, int),
              "channel_names": [f"C{i}" for i in range(6)], "valid_ev": np.arange(10)}
    out = mod.per_event_1d_alignment(bundle, {f"C{i}": 0.0 for i in range(6)})
    assert out["eligible"] is False           # n_ch=6 < min_part(5)+headroom(3)=8


def test_m1d_no_sign_and_runs():
    n_ch, n_ev = 10, 30
    rng = np.random.default_rng(3)
    bools = np.zeros((n_ch, n_ev), bool); masked = np.full((n_ch, n_ev), np.nan)
    for e in range(n_ev):
        k = rng.integers(5, 8)   # leaves headroom
        p = rng.choice(n_ch, k, replace=False)
        bools[p, e] = True; masked[p, e] = rng.permutation(k) / (k - 1)
    bundle = {"masked": masked, "bools": bools, "labels": rng.integers(0, 2, n_ev),
              "channel_names": [f"C{i}" for i in range(n_ch)], "valid_ev": np.arange(n_ev)}
    ict = {f"C{i}": float(i) for i in range(n_ch)}
    out = mod.per_event_1d_alignment(bundle, ict, n_perm=50, rng=rng)
    assert out["eligible"] is True and out["per_event"]
    assert all("sign" not in r for r in out["per_event"])           # §6: no replay sign side-channel
    assert all(0.0 <= r["null_p"] <= 1.0 and 0.0 <= r["align1d"] <= 1.0 for r in out["per_event"])


# --------------------------------------------------------------------------- R2 block null (C7)
def test_r2_block_level_keys_and_counts():
    rng = np.random.default_rng(4)
    n = 80
    block_ids = np.repeat(np.arange(8), 10)
    labels = rng.integers(0, 2, n)                 # MIXED within blocks (so within-block perm acts)
    align0 = np.clip(rng.normal(0.6, 0.15, n), 0, 1)   # each event scored on plane 0
    align1 = np.clip(rng.normal(0.5, 0.15, n), 0, 1)   # ... and on plane 1
    out = mod.class_separation_block_null(align0, align1, labels, block_ids, n_perm=200, rng=rng)
    assert out["status"] == "ok" and out["n_blocks"] == 8
    assert out["n_a"] + out["n_b"] == n            # event-level class sizes preserved (no drift)
    assert out["delta_median_null_p"] is not None and 0 <= out["delta_median_null_p"] <= 1
    assert out["size_matched_iqr_ratio"] is not None


def test_r2_observed_uses_own_plane():
    # P0: observed Δmedian uses each class's OWN-plane align (align0 for A, align1 for B)
    align0 = np.array([0.8, 0.8, 0.1, 0.1]); align1 = np.array([0.9, 0.9, 0.3, 0.3])
    labels = np.array([0, 0, 1, 1]); blk = np.array([0, 1, 2, 3])
    out = mod.class_separation_block_null(align0, align1, labels, blk, n_perm=10,
                                          rng=np.random.default_rng(0))
    assert abs(out["delta_median_obs"] - 0.5) < 1e-9     # 0.8 (A own) - 0.3 (B own)


def test_r2_requires_both_align_and_block_ids():
    # signature now (align0, align1, labels, block_ids) — missing args is a TypeError (C7)
    with pytest.raises(TypeError):
        mod.class_separation_block_null([0.1, 0.2], [0, 1])  # type: ignore


# --------------------------------------------------------------------------- diagnostics + stubs
def test_participation_diagnostics():
    bools = np.zeros((10, 6), bool)
    bools[:6, :3] = True; bools[:8, 3:] = True
    labels = np.array([0, 0, 0, 1, 1, 1]); blk = np.array([0, 0, 1, 0, 1, 1])
    d = mod.participation_diagnostics(bools, labels, blk)
    assert d["class_0"]["n_events"] == 3 and d["class_1"]["n_events"] == 3
    assert d["class_0"]["median_n_part"] == 6.0 and d["class_1"]["median_n_part"] == 8.0


def test_stage_b_c_stubs_raise():
    with pytest.raises(NotImplementedError):
        mod.stage_b_window_bias()
    with pytest.raises(NotImplementedError):
        mod.stage_c_sequential_effects()


# --------------------------------------------------------------- class-vs-template max-AB statistic
def test_field_from_values_support_override():
    X, Y = make_plane_grid()
    names = [f"C{i}" for i in range(6)]; xs = np.linspace(0.1, 0.9, 6)
    plane = {"channels": [{"name": n, "x_norm": float(x), "y_norm": 0.0, "typical_rank": 0.5,
                           "support": 0.0} for n, x in zip(names, xs)]}
    vals = {n: float(x) for n, x in zip(names, xs)}
    # plane aggregate support=0 -> default-support field cannot build
    assert mod.field_from_contact_values(plane, vals, sigma=0.2, X=X, Y=Y) is None
    # support override (e.g. class participation) lets it build
    F = mod.field_from_contact_values(plane, vals, support_by_name={n: 1.0 for n in names},
                                      sigma=0.2, X=X, Y=Y)
    assert F is not None and np.isfinite(F["T"]).any()


def test_maxab_picks_better_plane_and_pays_null():
    X, Y = make_plane_grid()
    names = [f"C{i}" for i in range(6)]; xs = np.linspace(0.1, 0.9, 6)
    plane = {"channels": [{"name": n, "x_norm": float(x), "y_norm": 0.0, "typical_rank": 0.5,
                           "support": 1.0} for n, x in zip(names, xs)]}
    from src.propagation_contact_plane_readout import R_smooth_rank as _R
    FA = _R({"channels": [dict(c, typical_rank=c["x_norm"]) for c in plane["channels"]]}, X, Y, 0.2, 0.15)
    FB = _R({"channels": [dict(c, typical_rank=0.5) for c in plane["channels"]]}, X, Y, 0.2, 0.15)  # constant
    target = [{n: float(x) for n, x in zip(names, xs)}]      # activation = x gradient (matches A)
    out = mod.maxab_alignment_vs_target(FA, FB, plane, plane, 0.2, 0.2, target,
                                        n_null=20, rng=np.random.default_rng(0), X=X, Y=Y)
    assert out["status"] == "ok"
    assert out["real_median_maxab"] > 0.8                    # max picks A (B constant -> nan)
    assert out["channel_null_p95"] is not None               # selection-cost null computed


def test_maxab_two_reps_shares_targets_and_matches_real():
    X, Y = make_plane_grid()
    names = [f"C{i}" for i in range(6)]; xs = np.linspace(0.1, 0.9, 6)
    plane = {"channels": [{"name": n, "x_norm": float(x), "y_norm": 0.0, "typical_rank": 0.5,
                           "support": 1.0} for n, x in zip(names, xs)]}
    from src.propagation_contact_plane_readout import R_smooth_rank as _R
    FA = _R({"channels": [dict(c, typical_rank=c["x_norm"]) for c in plane["channels"]]}, X, Y, 0.2, 0.15)
    FB = _R({"channels": [dict(c, typical_rank=1 - c["x_norm"]) for c in plane["channels"]]}, X, Y, 0.2, 0.15)
    target = [{n: float(x) for n, x in zip(names, xs)}]
    both = mod.maxab_two_reps_vs_target({"r1": (FA, FB), "r2": (FB, FA)}, plane, plane, 0.2, 0.2,
                                        target, n_null=10, rng=np.random.default_rng(0), X=X, Y=Y)
    assert set(both) == {"r1", "r2"} and both["r1"]["status"] == "ok"
    # real (rng-independent) matches the single-rep function for the same rep
    single = mod.maxab_alignment_vs_target(FA, FB, plane, plane, 0.2, 0.2, target,
                                           n_null=1, rng=np.random.default_rng(0), X=X, Y=Y)
    assert abs(both["r1"]["real_median_maxab"] - single["real_median_maxab"]) < 1e-9


# --------------------------------------------------------------------------- real-data integration
@pytest.mark.skipif(not REAL_1077.exists(), reason="broad 1077 labels not present")
def test_real_broad_1077_loads_and_aligns():
    b = mod.load_event_labels_ranks("epilepsiae", "1077")
    assert b["masked"].shape[1] == b["labels"].size == 2853
    assert set(np.unique(b["labels"]).tolist()) == {0, 1}
    assert b["n_blocks"] > 1
    # broad participation is dense (verified median 14) -> field metric viable
    npart = b["bools"].sum(axis=0)
    assert np.median(npart) >= 8
