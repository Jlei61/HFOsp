import numpy as np

from src.topic5_v3_mode_transition import (
    i1_range,
    load_v3_config,
    phase_bin_range,
    sliding_windows,
)


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
