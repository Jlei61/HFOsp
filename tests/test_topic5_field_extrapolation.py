import json
import numpy as np

from src.topic5_field_extrapolation import (
    broad_minus_narrow, ictal_zer_ranks,
    field_predict_at_points, predicted_interictal_order,
    compute_f_c, null_F, signed_spearman, radial_baseline_corr,
    compute_f_c_activation, null_F_activation,
    per_seizure_paired_median_abscorr, compute_c2_perchannel_energy,
    maxabscorr_series, field_null_p, quantile_bin_labels, _permute_within_labels,
)


def _chan(name, x, y, rank, support, along=None):
    c = {"name": name, "x_norm": x, "y_norm": y, "typical_rank": rank, "support": support}
    c["along_axis_mm"] = x if along is None else along
    return c


# ---- Task 1 ----
def test_broad_minus_narrow_exact_string():
    broad = ["TLA1", "TLA2", "TLB2", "TBA2"]
    narrow = ["TLA2", "TBA2"]
    assert broad_minus_narrow(broad, narrow) == ["TLA1", "TLB2"]


def test_broad_minus_narrow_no_basestrip():
    assert broad_minus_narrow(["TLA1", "TLA10"], ["TLA1"]) == ["TLA10"]


def test_ictal_zer_ranks_filters_low_valid_count(tmp_path):
    sid = "epilepsiae_TEST"
    d = {"per_er": {"broad_ER": {
        "r_sz": {"A1": 0.1, "A2": 0.9, "A3": 0.5},
        "r_sz_valid_count": {"A1": 5, "A2": 1, "A3": 4}}}}
    (tmp_path / f"{sid}.json").write_text(json.dumps(d))
    out = ictal_zer_ranks(sid, str(tmp_path), min_valid_count=3)
    assert set(out) == {"A1", "A3"}
    assert out["A1"] == 0.1


# ---- Task 2 ----
def test_field_predict_matches_nearby_high_support():
    chans = [_chan("C1", 0, 0, 0.2, 1.0), _chan("C2", 1.0, 0, 0.9, 0.05)]
    pred = field_predict_at_points(chans, np.array([[0.05, 0.0]]), sigma_xy=0.3)
    assert abs(pred[0] - 0.2) < 0.1


def test_loo_excludes_self():
    chans = [_chan("C1", 0, 0, 0.2, 1.0), _chan("SELF", 0.05, 0, 0.99, 1.0)]
    rec = {"channels": chans}
    out = predicted_interictal_order(rec, ["SELF"], loo=True, sigma_xy=0.3)
    assert abs(out["SELF"] - 0.2) < 0.15


# ---- Task 3 ----
def _rec_line():
    ch = [_chan(f"K{i}", i * 0.2, 0.0, i * 0.2, 1.0) for i in range(5)]
    ch += [_chan("H1", 0.35, 0.05, 0.99, 0.05), _chan("H2", 0.75, 0.05, 0.01, 0.05)]
    return {"channels": ch}


def test_F_uses_field_not_self():
    rec = _rec_line()
    ictal = {"H1": 0.30, "H2": 0.70}
    out = compute_f_c(rec, ["H1", "H2"], ictal, loo=True, sigma_xy=0.3)
    assert out["n_hidden"] == 2
    assert out["F"] > out["C"]


def test_null_returns_p_and_p95():
    rec = _rec_line()
    ictal = {"H1": 0.30, "H2": 0.70}
    nd = null_F(rec, ["H1", "H2"], ictal, n=200, seed=1, sigma_xy=0.3)
    assert "p_value" in nd and "p95" in nd


def test_signed_spearman_basic():
    assert signed_spearman([1, 2, 3], [1, 2, 3]) > 0.99
    assert signed_spearman([1, 2, 3], [3, 2, 1]) < -0.99


def test_radial_baseline_runs():
    rec = _rec_line()
    ictal = {"H1": 0.30, "H2": 0.70}
    r = radial_baseline_corr(rec, ["H1", "H2"], ictal)
    assert isinstance(r, float)


# ---- bb_auc activation basis ----
def _rec_line_big():
    ch = [_chan(f"K{i}", i * 0.15, 0.0, i * 0.15, 1.0) for i in range(4)]
    hid = [("H1", 0.2, 0.05, 0.9), ("H2", 0.4, 0.05, 0.1),
           ("H3", 0.6, 0.05, 0.8), ("H4", 0.8, 0.05, 0.2)]
    ch += [_chan(n, x, y, r, 0.1) for n, x, y, r in hid]
    return {"channels": ch}


def test_activation_F_per_seizure_median():
    rec = _rec_line_big()
    cache = ["H1", "H2", "H3", "H4"]
    sz = [np.array([1., 2., 3., 4.]), np.array([1.5, 2., 3., 3.5]), np.array([1., 2.5, 2.8, 4.])]
    out = compute_f_c_activation(rec, ["H1", "H2", "H3", "H4"], cache, sz, loo=True, sigma_xy=0.3)
    assert out["n_hidden"] == 4 and out["n_seizures_used"] == 3
    assert out["F"] > 0.5
    assert out["F"] > out["C"]


def test_activation_null_runs():
    rec = _rec_line_big()
    cache = ["H1", "H2", "H3", "H4"]
    sz = [np.array([1., 2., 3., 4.]), np.array([1.5, 2., 3., 3.5]), np.array([1., 2.5, 2.8, 4.])]
    nd = null_F_activation(rec, ["H1", "H2", "H3", "H4"], cache, sz, n=200, seed=1, sigma_xy=0.3)
    assert "p_value" in nd and 0.0 <= nd["p_value"] <= 1.0


def test_activation_max_ab_plumbing():
    rec = _rec_line_big()
    rec_b = {"channels": [dict(c, typical_rank=c["typical_rank"] * 0.9 + 0.05)
                          for c in rec["channels"]]}
    cache = ["H1", "H2", "H3", "H4"]
    sz = [np.array([1., 2., 3., 4.]), np.array([1.2, 2., 3., 3.8])]
    a = compute_f_c_activation(rec, ["H1", "H2", "H3", "H4"], cache, sz, sigma_xy=0.3)
    ab = compute_f_c_activation(rec, ["H1", "H2", "H3", "H4"], cache, sz, record_b=rec_b, sigma_xy=0.3)
    assert ab["n_templates"] == 2 and len(ab["predicted"]) == 2
    assert np.isfinite(ab["F"]) and ab["F"] >= a["F"] - 1e-9   # max(A,B) >= A


def test_core_only_field_uses_only_core():
    rec = _rec_line_big()
    core = {"K0", "K1", "K2", "K3"}     # narrow core (monotone rank ~ position)
    cache = ["H1", "H2", "H3", "H4"]
    sz = [np.array([1., 2., 3., 4.])]
    out = compute_f_c_activation(rec, ["H1", "H2", "H3", "H4"], cache, sz,
                                 sigma_xy=0.3, core_names=core)
    assert out["n_hidden"] == 4
    assert out["F"] > 0.8               # 场只用 core (干净单调) → 预测 hidden 与 ictal 对齐


def test_maxabscorr_series_len_and_align():
    ci = np.array([0, 1, 2, 3])
    x = [np.array([0.1, 0.2, 0.3, 0.4])]
    sz = [np.array([1., 2., 3., 4.]), np.array([4., 3., 2., 1.]), np.array([np.nan, np.nan, 1., 2.])]
    s = maxabscorr_series(x, ci, sz)
    assert len(s) == 3              # 与发作数对齐
    assert s[0] > 0.9 and s[1] > 0.9   # 单调对齐 |rho|~1
    assert not np.isfinite(s[2])    # <3 有限 → nan


def test_quantile_bin_labels_and_permute():
    lab = quantile_bin_labels(np.array([1., 2., 3., 4.]), n_bins=2)
    assert set(np.unique(lab)) <= {0, 1}
    rng = np.random.default_rng(0)
    pos = np.array([0, 1, 2, 3])
    labels = np.array([0, 0, 1, 1])
    perm = _permute_within_labels(pos, labels, rng)
    assert set(perm[:2]) == {0, 1} and set(perm[2:]) == {2, 3}   # 组内重排, 不跨组


def test_field_null_p_runs():
    rec = _rec_line_big()
    cache = ["H1", "H2", "H3", "H4"]
    ci = np.array([0, 1, 2, 3])
    sz = [np.array([1., 2., 3., 4.]), np.array([1.2, 2., 3., 3.8])]
    pred = [np.array([0.2, 0.4, 0.6, 0.8])]   # 与 sz 对齐 → F_obs 高
    F_obs = float(np.median([v for v in maxabscorr_series(pred, ci, sz) if np.isfinite(v)]))
    labels = [np.zeros(4, int), np.zeros(4, int)]   # channel null
    out = field_null_p(pred, ci, sz, F_obs, labels, n=200, seed=1)
    assert 0.0 <= out["p_value"] <= 1.0 and out["p_value"] < 0.2   # 对齐强 → 显著


def test_c2_perchannel_energy():
    rec = _rec_line_big()
    cache = ["H1", "H2", "H3", "H4"]
    # bact (interictal energy) anti-correlated with ictal energy -> low C2
    paired = [(np.array([4., 3., 2., 1.]), np.array([1., 2., 3., 4.])),
              (np.array([4., 3., 2., 1.]), np.array([1.2, 2., 3., 3.8]))]
    out = compute_c2_perchannel_energy(rec, ["H1", "H2", "H3", "H4"], cache, paired)
    assert out["n_hidden"] == 4 and out["n_seizures_used"] == 2
    assert out["C2"] > 0.9   # |corr| high (perfectly anti-correlated -> |rho|=1)
