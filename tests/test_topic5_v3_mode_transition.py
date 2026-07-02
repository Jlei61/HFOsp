import numpy as np

from scripts._topic5_v3_io import channel_is_valid
from src.topic5_v3_mode_transition import (
    axis_nonaxis_vectors,
    beta_axis,
    classify_contacts,
    geometry_sufficient,
    i1_range,
    load_v3_config,
    phase_bin_range,
    rank_forward,
    sliding_windows,
    subspace_projectors,
)


def test_channel_is_valid():
    assert channel_is_valid(np.array([np.nan, np.nan, np.nan, np.nan])) is False   # all-NaN
    assert channel_is_valid(np.array([1.0, 2.0])) is False                        # <3 finite samples
    assert channel_is_valid(np.array([3.0, 3.0, 3.0, 3.0])) is False              # flat/degenerate (std=0)
    assert channel_is_valid(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) is True          # >=3 finite, std>0


def test_v3_config_keys():
    c = load_v3_config()
    assert c["phases"]["I1_rel"] == [10.0, 30.0] and c["dynamics"]["finite_horizon_k"] == 3
    assert c["dynamics"]["demean_within_window"] is True and c["dynamics"]["mode_shift_normalization"] == "density"
    assert c["avalanche"]["flux_normalization"] == "source_mean"
    assert c["statistics"]["co_primary_correction"] == "holm" and c["cohorts"]["primary"] == "narrow"


def test_i1_short_seizure_fallback_is_usable():
    cfg = load_v3_config()
    lo, hi, ok = i1_range(0.0, 22.0, 22.0, cfg)            # 22 s seizure, offset=+22
    assert lo == 10.0 and hi == 20.0 and ok is True        # [+10, offset-2], one 10 s window
    lo2, hi2, ok2 = i1_range(0.0, 18.0, 18.0, cfg)         # offset-2=16 < lo+10 -> no window
    assert ok2 is False
    lo3, hi3, ok3 = i1_range(0.0, 205.0, 205.0, cfg)       # long -> primary [+10,+30]
    assert (lo3, hi3, ok3) == (10.0, 30.0, True)


def test_phase_bins_anchor_on_eeg_onset():
    cfg = load_v3_config(); relt = np.round(np.arange(-120, 60.001, 0.1), 3)
    p3 = phase_bin_range(relt, -3.75, 202.0, 205.0, "P3", cfg)
    assert relt[p3[0]] >= -3.75 - 30 - 1e-6 and relt[p3[1]-1] <= -3.75 - 10 + 1e-6
    p3j = phase_bin_range(relt, -3.75, 202.0, 205.0, "P3", cfg, onset_shift=10.0)
    assert relt[p3j[0]] >= -3.75 + 10 - 30 - 1e-6          # jitter shifts anchor


def test_sliding_windows_full_windows_only():
    cfg = load_v3_config()
    window_sec = cfg["phases"]["window_sec"]              # 10.0
    step_sec = cfg["phases"]["step_sec"]                  # 5.0
    relt = np.round(np.arange(-30, -9.999, 0.1), 3)        # 20 s span
    dt = float(np.median(np.diff(relt)))
    window_len_samples = int(round(window_sec / dt))

    full = sliding_windows(relt, 0, len(relt), window_sec, step_sec)
    assert len(full) == 3                                  # 20s / 5s step, full windows only
    for ws, we in full:
        assert abs((we - ws) - window_len_samples) <= 1    # no short tail

    one_window = sliding_windows(relt, 0, window_len_samples, window_sec, step_sec)
    assert len(one_window) == 1                            # exactly window_sec -> 1 window
    assert abs((one_window[0][1] - one_window[0][0]) - window_len_samples) <= 1

    short = sliding_windows(relt, 0, int(round(6.0 / dt)), window_sec, step_sec)
    assert len(short) == 0                                 # < window_sec -> 0 windows


def test_three_class_and_uniform_nonaxis_vector():
    cfg = load_v3_config(); thr = cfg["geometry"]["nonaxis_hfo_participation_max"]
    part = {"a0":.5,"a1":.5,"a2":.5,"a3":.5,"a4":.5,"n0":.0,"n1":.02,"n2":.0,"amb":.4}
    cl = classify_contacts(list(part), ["a0","a1","a2","a3","a4"], part, thr)
    assert cl["n_axis"] == 5 and cl["n_nonaxis"] == 3 and cl["n_ambiguous"] == 1   # 'amb' high part, no rank
    names = list(part)
    e_am, e_ag, e_nm = axis_nonaxis_vectors(names, {n:0. for n in names},
                                            ["a0","a1","a2","a3","a4"], ["n0","n1","n2"])
    assert np.isclose(np.linalg.norm(e_nm), 1.0) and abs(e_am @ e_nm) < 1e-9   # unit + orthogonal
    assert np.allclose(e_nm[[names.index(n) for n in ["n0","n1","n2"]]], e_nm[names.index("n0")])  # uniform, not part-weighted


def test_rank_forward_sign_convention():
    rf = rank_forward({"a": 1.0, "b": 2.0, "c": 3.0})
    assert rf["a"] == -1.0 and rf["b"] == 0.0 and rf["c"] == 1.0   # earliest->-1, middle->0, latest->+1

    tied = rank_forward({"a": 5.0, "b": 5.0, "c": 5.0})
    assert tied == {"a": 0.0, "b": 0.0, "c": 0.0}                  # all-tied -> all 0.0, no div-by-zero

    dropped = rank_forward({"a": 1.0, "b": float("nan"), "c": 3.0, "d": float("inf")})
    assert set(dropped) == {"a", "c"}                              # non-finite names excluded entirely
    assert dropped["a"] == -1.0 and dropped["c"] == 1.0            # rescale uses only the finite survivors


def test_beta_axis_signed_and_threshold():
    rf = {"a": -1.0, "b": -0.33, "c": 0.33, "d": 1.0}
    increasing = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}   # tracks rf order -> positive correlation
    decreasing = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}   # inverts rf order -> negative correlation

    b_up = beta_axis(increasing, rf)
    b_down = beta_axis(decreasing, rf)
    assert b_up > 0 and np.isclose(b_up, 1.0)               # monotonic increasing -> beta_axis ~= +1
    assert b_down < 0 and np.isclose(b_down, -1.0)          # monotonic decreasing -> beta_axis ~= -1

    few_pairs = beta_axis({"a": 1.0, "b": 2.0, "c": 3.0}, rf)                     # only 3 names in common
    assert np.isnan(few_pairs)
    few_finite = beta_axis({"a": 1.0, "b": 2.0, "c": 3.0, "d": float("nan")}, rf)  # 4th pair non-finite
    assert np.isnan(few_finite)


def test_subspace_projectors_excludes_ambiguous():
    names = ["a0", "a1", "n0", "n1", "amb"]                # 'amb' is in neither axis_names nor nonaxis_names
    P_A, P_N = subspace_projectors(names, ["a0", "a1"], ["n0", "n1"])

    assert P_A.shape == (5, 5) and P_N.shape == (5, 5)

    expected_P_A = np.diag([1.0, 1.0, 0.0, 0.0, 0.0])
    expected_P_N = np.diag([0.0, 0.0, 1.0, 1.0, 0.0])
    assert np.array_equal(P_A, expected_P_A)   # exact diagonal incl. all off-diagonal zeros
    assert np.array_equal(P_N, expected_P_N)

    amb_idx = names.index("amb")
    assert P_A[amb_idx, amb_idx] == 0.0 and P_N[amb_idx, amb_idx] == 0.0   # ambiguous is 0 in BOTH


def test_geometry_sufficient_and_gate():
    cfg = load_v3_config()
    assert cfg["geometry"]["min_n_axis"] == 5 and cfg["geometry"]["min_n_nonaxis"] == 3

    ok, reason = geometry_sufficient(5, 3, 1, cfg)
    assert ok is True and reason == "ok"

    ok, reason = geometry_sufficient(4, 3, 1, cfg)                     # axis shortfall
    assert ok is False and "n_axis" in reason and len(reason) > 0

    ok, reason = geometry_sufficient(5, 2, 1, cfg)                     # non-axis shortfall
    assert ok is False and "n_nonaxis" in reason and len(reason) > 0

    ok, reason = geometry_sufficient(5, 3, 0, cfg)                     # no shaft carries both classes
    assert ok is False and "shaft" in reason and len(reason) > 0
