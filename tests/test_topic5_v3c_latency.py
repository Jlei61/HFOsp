import numpy as np
from src.topic5_v3c_latency import first_crossing_latency, latency_seconds, encode_latency_for_rank
from src.topic5_v3c_latency import censoring_tallies, rank_diagnostics, threshold_stability, assay_valid
from src.topic5_v3_mode_transition import load_v3_config

RELT = np.round(np.arange(-5.0, 30.001, 0.1), 3)   # onset at 0.0


def test_first_crossing_finite():
    z = np.zeros_like(RELT); z[(RELT >= 5.0)] = 3.0            # crosses at +5s, sustained
    kind, sec = first_crossing_latency(z, RELT, 0.0, z_cross=2.0, window_sec=30.0, sustain_frames=3)
    assert kind == "finite" and abs(sec - 5.0) < 1e-6


def test_first_crossing_t0():
    z = np.full_like(RELT, 3.0)                                # already hot at onset
    kind, sec = first_crossing_latency(z, RELT, 0.0, z_cross=2.0, window_sec=30.0, sustain_frames=3)
    assert kind == "t0" and sec == 0.0


def test_first_crossing_censored_and_transient():
    z = np.zeros_like(RELT); z[(RELT >= 5.0) & (RELT < 5.15)] = 3.0   # 2 frames only -> not sustained
    kind, sec = first_crossing_latency(z, RELT, 0.0, z_cross=2.0, window_sec=30.0, sustain_frames=3)
    assert kind == "censored" and np.isnan(sec)


def test_encodings():
    assert latency_seconds("finite", 5.0) == 5.0 and latency_seconds("t0", 0.0) == 0.0
    assert np.isnan(latency_seconds("censored", float("nan")))
    assert encode_latency_for_rank("censored", float("nan"), window_sec=30.0) == 31.0
    assert encode_latency_for_rank("t0", 0.0, window_sec=30.0) == 0.0


def test_censoring_tallies():
    t = censoring_tallies(["finite", "finite", "t0", "censored"])
    assert t["finite_frac"] == 0.5 and t["t0_frac"] == 0.25 and t["cens_frac"] == 0.25


def test_rank_diagnostics_ties():
    d = rank_diagnostics(np.array([1.0, 1.0, 2.0, 3.0]))
    assert d["uniq_ranks"] == 3 and d["max_tie_block"] == 2


def test_threshold_stability_monotone():
    a = np.array([1.0, 2.0, 3.0, 4.0]); b = np.array([1.1, 2.2, 2.9, 4.5])
    assert threshold_stability(a, b) > 0.9


def test_assay_valid_gates():
    cfg = load_v3_config()
    good = {"finite_frac": 0.6, "t0_frac": 0.2, "uniq_ranks_med": 8, "thr_spearman": 0.8, "n_informative": 4}
    assert assay_valid(good, cfg) is True
    bad_t0 = {**good, "t0_frac": 0.56}                         # 1077-like
    assert assay_valid(bad_t0, cfg) is False
    bad_finite = {**good, "finite_frac": 0.37}
    assert assay_valid(bad_finite, cfg) is False
