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


def test_null_handles_none_and_int_mixed_blocks():
    # anchor_bins / unparseable shaft ids produce None mixed with int/str labels;
    # must not trip np.unique's int-vs-None sort. None-block channels stay put.
    templ = np.arange(10, dtype=float)
    seizure = np.arange(10, dtype=float)
    blocks = np.array([0, 0, 1, 1, None, None, 2, 2, None, 3], dtype=object)
    null = shuffle_null(seizure, [templ], B=50, rng=np.random.default_rng(0),
                        null_mode="anchor_matched", blocks=blocks, min_ch=8)
    assert np.all(np.isfinite(null))
    cap = shaft_block_capacity(np.array(["A", "A", None, "B", "B", None], dtype=object))
    assert cap["n_exchangeable_channels"] == 4   # A(2)+B(2); None ignored


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


# --- Task 5: LOO de-anchor + reliability ---
from src.topic5_echo_gate import loo_anchor, compute_deanchor_echo, anchor_reliability


def test_loo_anchor_excludes_current_seizure_no_leakage():
    M = np.array([
        [99.0, 99.0, 99.0, 99.0],   # seizure 0 extreme
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
    ])
    anc = loo_anchor(M)
    assert np.allclose(anc[0], [0.0, 1.0, 2.0, 3.0])     # seizure 0 NOT in its own anchor
    M2 = M.copy(); M2[0] = [-5.0, -5.0, -5.0, -5.0]
    assert np.allclose(loo_anchor(M2)[0], anc[0])        # changing sz0 doesn't change anc[0]


def test_loo_anchor_ignores_nan_in_other_seizures():
    M = np.array([
        [0.0, 1.0, 2.0, 3.0],
        [np.nan, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, np.nan],
    ])
    anc = loo_anchor(M)
    assert anc[0][0] == pytest.approx(0.0)               # mean of nan + 0.0 -> 0.0


def test_anchor_reliability_high_when_orders_agree():
    M = np.array([[0.0, 1, 2, 3, 4], [0.0, 1, 2, 3, 4], [0.0, 1, 2, 3, 4]])
    assert anchor_reliability(M) > 0.9


def test_deanchor_echo_uses_all_templates_not_first():
    # P1-A HARD: verify max-over-templates directly against echo_r_obs on the SAME
    # de-anchored deltas. An impl using only templates[0] would make r_obs equal the
    # t0-only value for every seizure -> the `any(differs)` assertion would fail.
    from src.topic5_echo_gate import loo_anchor, echo_r_obs
    rng = np.random.default_rng(5)
    seiz = np.vstack([rng.normal(0, 1, 12) for _ in range(4)])
    t0, t1 = rng.normal(0, 1, 12), rng.normal(0, 1, 12)
    recs = compute_deanchor_echo(seiz, [t0, t1], B=20, rng=np.random.default_rng(6), min_ch=8)
    anc = loo_anchor(seiz)
    differs = []
    for k, r in enumerate(recs):
        d = seiz[k] - anc[k]
        expected_both = echo_r_obs(d, [t0 - anc[k], t1 - anc[k]], min_ch=8)
        only_t0 = echo_r_obs(d, [t0 - anc[k]], min_ch=8)
        assert np.isclose(r["r_obs"], expected_both)       # uses BOTH de-anchored templates
        differs.append(not np.isclose(r["r_obs"], only_t0))
    assert any(differs)   # at least one seizure where max(t0,t1) != t0 alone


# --- Task 6: subject-level pooling + bad-data regression ---
from src.topic5_echo_gate import pool_echo_subject_level, bad_data_regression


def _records(per_subject_es):
    out = []
    for sid, evals in per_subject_es.items():
        for e in evals:
            out.append({"subject": sid, "e_k": e})
    return out


def test_pool_positive_when_subjects_consistently_positive():
    recs = _records({f"s{i}": [0.8, 1.1, 0.9] for i in range(12)})
    res = pool_echo_subject_level(recs)
    assert res["n_subjects"] == 12
    assert res["median_E_s"] > 0
    assert res["wilcoxon_p_onesided"] < 0.05


def test_pool_sanity_centered_zero_not_significant():
    rng = np.random.default_rng(7)
    recs = _records({f"s{i}": list(rng.normal(0, 1, 3)) for i in range(12)})
    res = pool_echo_subject_level(recs)
    assert res["wilcoxon_p_onesided"] > 0.10


def test_bad_data_regression_real_null_draw_flattens():
    templ = np.arange(12, dtype=float)
    records = []
    for i in range(12):
        rng = np.random.default_rng(100 + i)
        seiz = np.argsort(np.arange(12) + rng.normal(0, 0.6, 12)).astype(float)
        res = compute_echo_strength(seiz, [templ], B=1500, rng=rng, min_ch=8)
        res["subject"] = f"s{i}"
        records.append(res)
    primary = pool_echo_subject_level(
        [{"subject": r["subject"], "e_k": r["e_k"]} for r in records])
    bad = bad_data_regression(
        [{"subject": r["subject"], "e_k_baddata": r["e_k_baddata"]} for r in records])
    assert primary["wilcoxon_p_onesided"] < 0.05         # real echo survives
    assert bad["wilcoxon_p_onesided"] > 0.10             # fake (null-draw) obs flattens


# --- Task 7: compute_atlas_quality ---
from src.topic5_echo_gate import compute_atlas_quality


def test_atlas_quality_pass_for_clean_ranks():
    rank = np.arange(12, dtype=float)
    q = compute_atlas_quality(rank, tie_max=0.3, min_channels=8)
    assert q["atlas_quality_flag"] == "pass"
    assert q["rank_tie_fraction"] == pytest.approx(0.0)


def test_atlas_quality_fail_for_mostly_tied_ranks():
    rank = np.array([1.0] * 10 + [2.0, 3.0])
    q = compute_atlas_quality(rank, tie_max=0.3, min_channels=8)
    assert q["atlas_quality_flag"] == "fail"


def test_atlas_quality_fail_for_too_few_channels():
    rank = np.array([0.0, 1.0, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan])
    q = compute_atlas_quality(rank, tie_max=0.3, min_channels=8)
    assert q["atlas_quality_flag"] == "fail"


# --- runner-level review hardening (P0/P1) ---
def test_subject_atlas_quality_masks_to_joint_valid():
    import scripts.run_topic5_echo_gate as R
    # 8 channels but the last 2 are template-INVALID. They carry values that would prop up
    # quality; masking to joint_valid must exclude them -> only 6 ranked -> fail (<MIN_CH=8).
    seiz = np.array([[0.0, 1, 2, 3, 4, 5, 100, 101]])
    joint_valid = np.array([True] * 6 + [False, False])
    q = R.subject_atlas_quality(seiz, joint_valid)
    assert q["n_ranked_channels"] == 6          # invalid channels excluded
    assert q["atlas_quality_flag"] == "fail"


def test_verdict_no_standing_when_construct_pending():
    import scripts.run_topic5_echo_gate as R
    summary = {
        "construct_validity_status": "pending",
        "primary_channel_all": {"n_subjects": 12, "wilcoxon_p_onesided": 0.001,
                                "median_E_s": 0.9, "sign_p_onesided": 0.01, "boot_ci95": [0.1, 0.9]},
        "primary_within_shaft_all": {"wilcoxon_p_onesided": 0.001},
        "primary_anchor_matched_all": {"wilcoxon_p_onesided": 0.001},
        "negative_between_subject_epilepsiae": {"n_subjects": 8, "wilcoxon_p_onesided": 0.5},
        "bad_data_regression": {"wilcoxon_p_onesided": 0.5},
    }
    v = R._assign_verdict(summary)
    assert not v["label"].startswith("站住")     # construct pending forbids standing


def test_verdict_no_standing_without_sensitivities():
    import scripts.run_topic5_echo_gate as R
    summary = {
        "construct_validity_status": "pass",     # construct ok, but sensitivities missing
        "primary_channel_all": {"n_subjects": 12, "wilcoxon_p_onesided": 0.001,
                                "median_E_s": 0.9, "sign_p_onesided": float("nan"),
                                "boot_ci95": [float("nan"), float("nan")]},
        "primary_within_shaft_all": {"wilcoxon_p_onesided": 0.001},
        "primary_anchor_matched_all": {"wilcoxon_p_onesided": 0.001},
        "negative_between_subject_epilepsiae": {"n_subjects": 8, "wilcoxon_p_onesided": 0.5},
        "bad_data_regression": {"wilcoxon_p_onesided": 0.5},
    }
    v = R._assign_verdict(summary)
    assert not v["label"].startswith("站住")


# --- Task 7b: between_subject_control (Null D) ---
from src.topic5_echo_gate import between_subject_control


def test_between_subject_control_name_aligned_neutral_for_unrelated():
    rng = np.random.default_rng(11)
    this_channels = [f"X{i}" for i in range(12)]      # this subject's channel names
    seizure = np.arange(12, dtype=float)              # echoes its own (X-named) template
    # foreign templates: SAME channel names (anatomical labels shared across patients),
    # but UNRELATED rank order -> should be neutral.
    foreign = [(rng.permutation(12).astype(float).tolist(), list(this_channels))
               for _ in range(8)]
    res = between_subject_control(seizure, this_channels, foreign, B=800, rng=rng, min_ch=8)
    assert res["n_foreign_overlapping"] == 8
    assert abs(res["e_k"]) < 2.5
    assert 0.02 < res["p_k"] < 0.98


def test_between_subject_control_skips_nonoverlapping_names():
    # foreign templates on DIFFERENT channel names -> no overlap -> nan record.
    this_channels = [f"X{i}" for i in range(12)]
    seizure = np.arange(12, dtype=float)
    foreign = [(list(range(12)), [f"Y{i}" for i in range(12)])]
    res = between_subject_control(seizure, this_channels, foreign, B=200,
                                  rng=np.random.default_rng(0), min_ch=8)
    assert res["n_foreign_overlapping"] == 0
    assert np.isnan(res["e_k"])


def test_between_subject_control_name_align_not_positional():
    # Foreign template has the SAME values [0..11] but its channel NAMES are in REVERSED
    # order vs this subject. Name-alignment must place foreign rank by NAME (-> reversed
    # vs this subject -> rho=-1 with an ascending seizure). Positional truncation would
    # instead give rho=+1. So r_obs must be NEGATIVE, proving name-align (not positional).
    this_channels = [f"X{i}" for i in range(12)]
    seizure = np.arange(12, dtype=float)                      # ascending
    foreign_names = [f"X{i}" for i in range(11, -1, -1)]      # reversed name order
    foreign_vals = list(range(12))                            # 0..11 in that reversed order
    res = between_subject_control(seizure, this_channels, [(foreign_vals, foreign_names)],
                                  B=100, rng=np.random.default_rng(0), min_ch=8)
    assert res["n_foreign_overlapping"] == 1
    assert res["r_obs"] < -0.5    # name-aligned -> reversed -> negative (positional would be +1)


# --- Task 8: phantom-safe template contracts (1-D np.where vs 2-D rebuild) ---
from src.topic5_echo_gate import masked_template_rank_1d, rebuild_template_from_events


def test_masked_template_rank_1d_uses_valid_mask_not_helper():
    # 1-D ALREADY-AGGREGATED template rank + per-cluster valid_mask -> np.where.
    agg_rank = np.array([0.0, 1.0, 7.0, 2.0])        # idx2 carries a phantom value
    valid_mask = np.array([True, True, False, True]) # idx2 non-participating
    templ = masked_template_rank_1d(agg_rank, valid_mask)
    assert np.isnan(templ[2])                        # phantom excluded
    assert np.allclose(templ[[0, 1, 3]], agg_rank[[0, 1, 3]])


def test_masked_template_rank_1d_shape_mismatch_raises():
    with pytest.raises(ValueError):
        masked_template_rank_1d(np.zeros(4), np.ones(3, bool))


def test_rebuild_template_from_events_2d_uses_helper():
    # event-level (n_ch, n_ev) raw ranks + bools -> mask_phantom_ranks -> aggregate.
    raw = np.array([[0.0, 1.0], [1.0, 0.0], [9.0, 9.0], [2.0, 2.0]])   # (4 ch, 2 ev)
    bools = np.array([[True, True], [True, True], [False, False], [True, True]])
    templ = rebuild_template_from_events(raw, bools)
    assert np.isnan(templ[2])                        # ch2 never participates -> NaN
    assert np.all(np.isfinite(templ[[0, 1, 3]]))
