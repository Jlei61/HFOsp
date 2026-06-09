import numpy as np
import pytest

from src.topic5_echo_gate import spearman_common


def test_spearman_common_identical_and_reverse():
    a = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert spearman_common(a, a, min_ch=8) == pytest.approx(1.0)
    assert spearman_common(a, a[::-1].copy(), min_ch=8) == pytest.approx(-1.0)


def test_spearman_common_too_few_returns_nan():
    a = np.array([0.0, 1.0, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan])
    b = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    # only 3 common finite -> below min_ch
    assert np.isnan(spearman_common(a, b, min_ch=8))


def test_spearman_common_phantom_channel_excluded():
    # template has NaN (masked phantom) at index 7 even though seizure has a value there.
    # Result must equal Spearman over indices 0..6 only (phantom never enters).
    templ = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan])
    seiz = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 99.0])
    from scipy.stats import spearmanr
    expected = spearmanr(seiz[:7], templ[:7]).statistic
    assert spearman_common(seiz, templ, min_ch=6) == pytest.approx(expected)


def test_spearman_common_shape_mismatch_raises():
    with pytest.raises(ValueError):
        spearman_common(np.zeros(8), np.zeros(7), min_ch=5)


# --- Task 2: echo_r_obs ---
from src.topic5_echo_gate import echo_r_obs


def test_echo_r_obs_takes_best_matching_template():
    base = np.arange(8, dtype=float)
    t0 = base.copy()
    seizure = base[::-1].copy()      # matches t0 reversed -> rho=-1
    t1 = seizure.copy()              # t1 matches the seizure exactly -> rho=1
    r = echo_r_obs(seizure, [t0, t1], min_ch=8)
    assert r == pytest.approx(1.0)   # best template (t1) wins


def test_echo_r_obs_single_template_k1():
    base = np.arange(8, dtype=float)
    assert echo_r_obs(base, [base.copy()], min_ch=8) == pytest.approx(1.0)


def test_echo_r_obs_all_insufficient_returns_nan():
    a = np.array([0.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    t = np.arange(8, dtype=float)
    assert np.isnan(echo_r_obs(a, [t], min_ch=8))


# --- Task 3: shuffle null modes + shaft_block capacity ---
from src.topic5_echo_gate import shuffle_null, shaft_block_capacity


def test_channel_shuffle_destroys_cross_shaft_order_but_within_shaft_preserves():
    # 2 shafts x 4 channels. Template = global ascending. Seizure = same global order.
    # within_shaft shuffle only scrambles inside each contiguous shaft block -> global
    # Spearman stays high; channel shuffle destroys cross-shaft order -> null lower.
    templ = np.arange(8, dtype=float)
    seizure = np.arange(8, dtype=float)
    shafts = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    rng = np.random.default_rng(0)
    null_chan = shuffle_null(seizure, [templ], B=500, rng=rng, null_mode="channel", min_ch=8)
    rng = np.random.default_rng(0)
    null_within = shuffle_null(seizure, [templ], B=500, rng=rng,
                               null_mode="within_shaft", blocks=shafts, min_ch=8)
    null_chan = null_chan[np.isfinite(null_chan)]
    null_within = null_within[np.isfinite(null_within)]
    assert np.nanmean(null_within) > np.nanmean(null_chan) + 0.2


def test_shaft_block_requires_blocks():
    with pytest.raises(ValueError):
        shuffle_null(np.arange(8.0), [np.arange(8.0)], B=10,
                     rng=np.random.default_rng(0), null_mode="within_shaft", min_ch=8)


def test_shaft_block_capacity_fail_closed_on_unequal_shafts():
    # sizes 4, 3, 2 -> no two shafts share a size -> nothing exchangeable.
    blocks = np.array(["A", "A", "A", "A", "B", "B", "B", "C", "C"])
    cap = shaft_block_capacity(blocks)
    assert cap["n_exchangeable_channels"] == 0
    assert cap["insufficient_block_exchange"] is True


def test_shaft_block_capacity_ok_when_two_equal_shafts():
    blocks = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])  # 2 shafts of size 4
    cap = shaft_block_capacity(blocks)
    assert cap["n_exchangeable_channels"] == 8
    assert cap["insufficient_block_exchange"] is False


# --- Task 4: compute_echo_strength (+ e_k_baddata real null draw) ---
from src.topic5_echo_gate import compute_echo_strength


def test_echo_strength_positive_for_matching_seizure():
    templ = np.arange(12, dtype=float)
    seizure = np.arange(12, dtype=float)          # perfect echo
    res = compute_echo_strength(seizure, [templ], B=1000,
                                rng=np.random.default_rng(1), min_ch=8)
    assert res["r_obs"] == pytest.approx(1.0)
    assert res["e_k"] > 3.0
    assert res["p_k"] < 0.01


def test_echo_strength_null_for_random_seizure():
    templ = np.arange(12, dtype=float)
    rng = np.random.default_rng(2)
    seizure = rng.permutation(12).astype(float)
    res = compute_echo_strength(seizure, [templ], B=1000, rng=rng, min_ch=8)
    assert abs(res["e_k"]) < 2.5
    assert 0.02 < res["p_k"] < 0.98


def test_echo_strength_insufficient_returns_nan_record():
    a = np.array([0.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    res = compute_echo_strength(a, [np.arange(8.0)], B=100,
                                rng=np.random.default_rng(3), min_ch=8)
    assert np.isnan(res["e_k"]) and res["n_null"] == 0


def test_echo_strength_baddata_field_is_centered_draw():
    templ = np.arange(12, dtype=float)
    seizure = np.arange(12, dtype=float)
    res = compute_echo_strength(seizure, [templ], B=2000,
                                rng=np.random.default_rng(4), min_ch=8)
    assert np.isfinite(res["e_k_baddata"])
    assert abs(res["e_k_baddata"]) < res["e_k"]
