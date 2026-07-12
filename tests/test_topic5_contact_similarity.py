"""TDD tests for topic5_contact_similarity.kernel_smooth_at_contacts.

Task 1: grid-free Gaussian contact kernel proven equal to smooth_field.
Task 2: polarity-free maxAB similarity (raw + same-plane kernel, mirror-faithful).
"""
import numpy as np
import pytest
from scipy.stats import pearsonr
from src.topic5_contact_similarity import (
    kernel_smooth_at_contacts,
    _pearson_over_contacts,
    contact_corr,
    polarity_free_maxab,
    sequence_maxab,
)
from src.propagation_contact_plane_readout import smooth_field, make_plane_grid


def _toy_pts():
    # 3 contacts, irregular spacing on the plane
    pts = np.array([[0.0, 0.0], [0.4, 0.1], [1.0, -0.2]])
    vals = np.array([0.0, 1.0, 2.0])
    sup = np.ones(3)
    return pts, vals, sup


def test_kernel_matches_smooth_field_on_grid():
    """R2 kernel ≡ R3 field math: evaluating the contact kernel at the grid
    points reproduces smooth_field's grid (at finite, well-supported pixels)."""
    pts, vals, sup = _toy_pts()
    sigma = 0.3
    X, Y = make_plane_grid()
    grid_pts = np.column_stack([X.ravel(), Y.ravel()])   # (N,2): column_stack of two 1D arrays is correct
    record = {"channels": [{"x_norm": float(p[0]), "y_norm": float(p[1]),
                            "support": float(s), "typical_rank": float(v)}
                           for p, s, v in zip(pts, sup, vals)]}
    field = smooth_field(record, X, Y, sigma_xy=sigma, s_thresh=0.0)  # real sig: record-first, returns {"T","S"}
    T = field["T"]
    mine = kernel_smooth_at_contacts(vals, pts, grid_pts, sup, sigma).reshape(X.shape)
    m = np.isfinite(T) & np.isfinite(mine)
    assert m.sum() > 100
    assert np.allclose(T[m], mine[m], atol=1e-9)


def test_kernel_reduces_to_self_value_as_sigma_to_zero():
    """σ→0: each eval point (=its own contact) returns its own value."""
    pts, vals, sup = _toy_pts()
    out = kernel_smooth_at_contacts(vals, pts, pts, sup, sigma=1e-4)
    assert np.allclose(out, vals, atol=1e-6)


# ---------------------------------------------------------------------------
# Task 2: polarity-free maxAB similarity
# ---------------------------------------------------------------------------

def test_raw_mode_is_plain_abs_pearson():
    rng = np.random.default_rng(0)
    rank = rng.random(8); val = 2 * rank + rng.normal(0, 0.1, 8)
    pts = rng.random((8, 2)); sup = np.ones(8)
    got = contact_corr(rank, val, mode="raw", source_pts=pts, support=sup, sigma=0.3)
    # M1: check SIGNED value — the old abs(got)==abs(ref) passed even with a hidden abs()
    assert np.isclose(got, pearsonr(rank, val)[0])
    # M1: negative correlation case — a hidden abs() would return a positive value here
    got_neg = contact_corr(-rank, val, mode="raw", source_pts=pts, support=sup, sigma=0.3)
    assert got_neg < 0, f"raw mode must return signed corr; got {got_neg}"
    assert np.isclose(got_neg, pearsonr(-rank, val)[0])


def test_kernel_sigma_to_zero_equals_raw():
    rng = np.random.default_rng(1)
    rank = rng.random(10); val = rng.random(10)
    pts = rng.random((10, 2)); sup = np.ones(10)
    raw = contact_corr(rank, val, mode="raw", source_pts=pts, support=sup, sigma=0.3)
    ker = contact_corr(rank, val, mode="kernel", source_pts=pts, support=sup, sigma=1e-4)
    assert np.isclose(raw, ker, atol=1e-4)


def test_maxab_takes_better_template():
    rng = np.random.default_rng(2)
    val = rng.random(12)
    rank_a = rng.random(12)              # unrelated to val
    rank_b = val + rng.normal(0, 0.01, 12)  # strongly related
    pts = rng.random((12, 2)); sup = np.ones(12)
    mab = polarity_free_maxab(rank_a, rank_b, val, mode="raw",
                              source_pts=pts, support=sup, sigma=0.3)
    assert mab > 0.9   # picks template B


def test_maxab_sign_free_reverse_passes():
    """Sign-free: a perfectly reversed rank is a true positive (|corr|=1)."""
    rng = np.random.default_rng(3)
    val = rng.random(12); rank_a = -val   # reversed
    pts = rng.random((12, 2)); sup = np.ones(12)
    mab = polarity_free_maxab(rank_a, None, val, mode="raw",
                              source_pts=pts, support=sup, sigma=0.3)
    assert np.isclose(mab, 1.0, atol=1e-6)


def test_abs_mirror_max_then_abs_not_abs_then_max():
    """I1: _abs_mirror uses abs(max(c_id, c_mr)), NOT max(abs(c_id), abs(c_mr)).

    Fixture: 4 contacts whose y-mirror permutes contacts 0↔2 (same-x pairs
    straddle y=0) while contacts 1 and 3 are self-mapping.  With sigma=0.2 the
    kernel is nearly sharp.  value is designed so identity gives mild POSITIVE
    correlation while mirror gives stronger NEGATIVE correlation:
        c_id ≈ +0.39,  c_mr ≈ -0.80
    Then:
        abs(max(c_id, c_mr)) = abs(+0.39)            ≈ 0.39   [correct formula]
        max(abs(c_id), abs(c_mr)) = max(0.39, 0.80)  ≈ 0.80   [wrong formula]
    A regression that flips the formula (abs-before-max) would return 0.80,
    causing assertion (b) to fail.
    """
    pts = np.array([[0.0, 0.3], [1.0, 0.3], [0.0, -0.3], [2.0, -0.3]])
    rank  = np.array([1.0, 2.0, 3.0, 4.0])
    # Mirror maps: contact 0 ↔ contact 2 (same x, opposite y), 1 and 3 self-map.
    # value chosen so identity ≈ +0.39, mirror ≈ -0.80:
    value = np.array([1.0, 3.0, 4.0, 2.0])
    sup   = np.ones(4)
    sigma = 0.2

    c_id = contact_corr(rank, value, mode="kernel", source_pts=pts, support=sup,
                        sigma=sigma, mirror=False)
    c_mr = contact_corr(rank, value, mode="kernel", source_pts=pts, support=sup,
                        sigma=sigma, mirror=True)

    # (a) fixture is discriminating: opposite signs, different magnitudes
    assert c_id > 0 and c_mr < 0, (
        f"fixture failed: c_id={c_id:.4f}, c_mr={c_mr:.4f} — both must have opposite signs")
    assert not np.isclose(abs(max(c_id, c_mr)), max(abs(c_id), abs(c_mr))), (
        "fixture not discriminating: abs(max)==max(abs); geometry must produce unequal magnitudes")

    # (b) polarity_free_maxab returns abs(max(c_id, c_mr)), not max(abs(...))
    result         = polarity_free_maxab(rank, None, value, mode="kernel",
                                         source_pts=pts, support=sup, sigma=sigma)
    correct_answer = abs(max(c_id, c_mr))   # max-by-value → then abs
    wrong_answer   = max(abs(c_id), abs(c_mr))  # abs-first → then max (incorrect)

    assert np.isclose(result, correct_answer, atol=1e-9), (
        f"polarity_free_maxab={result:.4f}; expected abs(max)={correct_answer:.4f}")
    assert not np.isclose(result, wrong_answer, atol=1e-9), (
        f"result matches the wrong formula max(abs)={wrong_answer:.4f}; "
        "test would not catch a regression")


# ---------------------------------------------------------------------------
# Task 3: per-seizure → median-over-seizures null fold (pluggable statistic)
# ---------------------------------------------------------------------------

def test_fold_matches_p95_med():
    # draws[sz] = [B], obs[sz]; replicate np.nanmedian(draws, axis=0) then pct95
    from src.topic5_contact_similarity import fold_subject
    rng = np.random.default_rng(4)
    obs = [0.6, 0.7, 0.5]
    null = [list(rng.random(50)) for _ in range(3)]
    out = fold_subject(obs, null)
    expect_dist = np.nanmedian(np.asarray(null, float), axis=0)
    assert np.isclose(out["obs_subject"], np.median(obs))
    assert np.isclose(out["null_q"]["p95"], np.nanpercentile(expect_dist, 95))
    assert out["passed"] == bool(np.median(obs) > np.nanpercentile(expect_dist, 95))


def test_subject_null_recomputes_maxab_each_draw():
    """The null statistic must be the MAX-selected statistic, so a 2-template
    stat_fn yields a higher null upper tail than a single-template stat_fn."""
    from src.topic5_contact_similarity import subject_null
    rng = np.random.default_rng(5)
    names = [f"A{i}" for i in range(1, 7)] + [f"B{i}" for i in range(1, 7)]
    vals = {0: rng.random(12)}
    def stat_max(v):   # closure that internally takes max over 2 templates
        return max(abs(np.corrcoef(v, rng.random(12))[0, 1]),
                   abs(np.corrcoef(v, rng.random(12))[0, 1]))
    def stat_one(v):
        return abs(np.corrcoef(v, rng.random(12))[0, 1])
    r_max = subject_null(stat_max, vals, names, shuffle="channel", B=200, seed=1)
    r_one = subject_null(stat_one, vals, names, shuffle="channel", B=200, seed=1)
    assert r_max["null_q"]["p95"] >= r_one["null_q"]["p95"]


def test_within_shaft_never_crosses_shaft():
    from src.topic5_axis_alignment import within_shaft_shuffle
    from src.propagation_skeleton_geometry import parse_shaft
    rng = np.random.default_rng(6)
    names = ["A1", "A2", "A3", "B1", "B2", "B3"]
    vals = np.arange(6.0)
    out = within_shaft_shuffle(vals, names, rng)
    # multiset within each shaft preserved
    for sh in ("A", "B"):
        idx = [i for i, n in enumerate(names) if parse_shaft(n)[0] == sh]
        assert sorted(out[idx]) == sorted(vals[idx])


# ---------------------------------------------------------------------------
# Task 4: sequence-sanity (Spearman + Kendall)
# ---------------------------------------------------------------------------

def test_sequence_spearman_monotone():
    val = np.array([1.0, 2, 3, 4, 5, 6])
    rank_a = np.array([6.0, 5, 4, 3, 2, 1])   # reversed monotone -> |spearman|=1
    s = sequence_maxab(rank_a, None, val, method="spearman")
    assert np.isclose(s, 1.0, atol=1e-9)


def test_sequence_kendall_runs():
    rng = np.random.default_rng(7)
    val = rng.random(10); ra = rng.random(10); rb = val.copy()
    k = sequence_maxab(ra, rb, val, method="kendall")
    assert 0.9 < k <= 1.0   # template B identical -> tau ~ 1


def test_kernel_mirror_flips_eval_y_not_x():
    """I2: white-box pin — mirror=True negates eval-pt y, not x, not both.

    Manually build f_rank and f_val_mirror using kernel_smooth_at_contacts
    with pts_y_negated (column-1 negated) and verify contact_corr(mirror=True)
    returns exactly _pearson_over_contacts(f_rank, f_val_mirror).

    Also assert mirror=True != mirror=False for this y-asymmetric fixture
    (so the test is non-vacuous).
    """
    pts = np.array([[0.0, 0.3], [1.0, 0.3], [0.0, -0.3], [2.0, -0.3]])
    rank  = np.array([1.0, 2.0, 3.0, 4.0])
    value = np.array([1.0, 3.0, 4.0, 2.0])
    sup   = np.ones(4)
    sigma = 0.2

    # Manual reconstruction of what mirror=True should compute
    pts_y_negated = pts.copy()
    pts_y_negated[:, 1] = -pts_y_negated[:, 1]

    f_rank       = kernel_smooth_at_contacts(rank,  pts, pts,           sup, sigma)
    f_val_mirror = kernel_smooth_at_contacts(value, pts, pts_y_negated, sup, sigma)
    expected     = _pearson_over_contacts(f_rank, f_val_mirror)

    got_mirror   = contact_corr(rank, value, mode="kernel", source_pts=pts,
                                support=sup, sigma=sigma, mirror=True)
    got_identity = contact_corr(rank, value, mode="kernel", source_pts=pts,
                                support=sup, sigma=sigma, mirror=False)

    # White-box: mirror=True ≡ _pearson_over_contacts(f_rank, kernel(value, pts, pts_y_neg))
    assert np.isclose(got_mirror, expected, atol=1e-12), (
        f"mirror={got_mirror:.6f} != manual y-flip={expected:.6f}; "
        "implementation may negate x or both axes instead of y")

    # Sanity: fixture is y-asymmetric so mirror must differ from identity
    assert not np.isclose(got_mirror, got_identity, atol=1e-4), (
        "mirror=True equals mirror=False — fixture is degenerate or y-flip has no effect")


# ---------------------------------------------------------------------------
# R2b-3D sensitivity: generalize kernel_smooth_at_contacts to n-D (2D + 3D)
# ---------------------------------------------------------------------------

def test_kernel_3d_hand_weights():
    # 3 contacts in 3D; hand-compute the Nadaraya-Watson output at one eval point
    pts = np.array([[0., 0., 0.], [1., 0., 0.], [0., 0., 2.]]); vals = np.array([1., 2., 3.]); sup = np.ones(3); sigma = 1.0
    out = kernel_smooth_at_contacts(vals, pts, pts, sup, sigma)
    sig2 = 2.0 * sigma * sigma
    # eval at pts[0]: d2 to each = [0,1,4]; w=exp(-d2/sig2)
    w = np.exp(-np.array([0., 1., 4.]) / sig2); exp0 = (w * vals).sum() / w.sum()
    assert np.isclose(out[0], exp0)


def test_kernel_2d_regression_unchanged():
    # 2D path must be numerically identical to before generalization (protect the cross-check test)
    rng = np.random.default_rng(0); pts = rng.random((6, 2)); vals = rng.random(6); sup = np.ones(6)
    out = kernel_smooth_at_contacts(vals, pts, pts, sup, 0.3)
    # recompute with an explicit 2-col Euclidean reference
    ref = np.array([((np.exp(-(((pts - pts[i]) ** 2).sum(1)) / (2 * 0.3 ** 2)) * vals).sum()
                      / np.exp(-(((pts - pts[i]) ** 2).sum(1)) / (2 * 0.3 ** 2)).sum()) for i in range(6)])
    assert np.allclose(out, ref)


def test_kernel_nan_coords_excluded():
    # a source contact with NaN coord must not contribute to any weight
    pts = np.array([[0., 0.], [np.nan, np.nan], [1., 0.]]); vals = np.array([1., 9., 2.]); sup = np.ones(3)
    out = kernel_smooth_at_contacts(vals, pts, pts, sup, 0.5)
    assert np.isfinite(out[0]) and np.isfinite(out[2])  # value 9 (NaN-coord) must not leak in


def test_kernel_sigma_nonpositive_raises():
    pts = np.array([[0., 0.], [1., 0.]]); vals = np.array([1., 2.]); sup = np.ones(2)
    with pytest.raises((ValueError,)):
        kernel_smooth_at_contacts(vals, pts, pts, sup, 0.0)


def test_median_nn_spacing_hand_values():
    from src.topic5_contact_similarity import median_nn_spacing
    pts = np.array([[0., 0., 0.], [3., 0., 0.]])
    assert np.isclose(median_nn_spacing(pts), 3.0)
    pts_identical = np.array([[1., 1.], [1., 1.], [1., 1.]])
    assert median_nn_spacing(pts_identical) == 0.0
