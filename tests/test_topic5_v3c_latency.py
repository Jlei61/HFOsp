import numpy as np
from src.topic5_v3c_latency import first_crossing_latency, latency_seconds, encode_latency_for_rank

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
