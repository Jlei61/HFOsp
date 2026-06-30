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
