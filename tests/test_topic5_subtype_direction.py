"""TDD for Topic 5 C-line: subtype x activation-direction (pure functions).

The load-bearing test is `test_axial_collapses_180_but_polarity_does_not` — it is the
regression guard for the review's fatal Point 1 (axial u=(cos2θ,sin2θ) erases θ vs θ+π).
"""
import numpy as np
import pytest

from src.topic5_subtype_direction import (
    axial_distance,
    circular_distance,
    subtype_separation_stat,
    within_subject_perm_p,
    direction_clustering,
    coord_aspect_ratio,
    align_subtype_to_direction,
    oddeven_subtype_imbalance,
)

PI = np.pi


# --- axial_distance ∈ [0, π/2] -------------------------------------------------
def test_axial_distance_opposite_is_zero():
    assert axial_distance(0.0, PI) == pytest.approx(0.0, abs=1e-9)


def test_axial_distance_orthogonal_is_half_pi():
    assert axial_distance(0.0, PI / 2) == pytest.approx(PI / 2, abs=1e-9)


def test_axial_distance_symmetric_and_bounded():
    for a, b in [(0.3, 1.1), (2.0, 0.1), (0.0, 0.7)]:
        d = axial_distance(a, b)
        assert axial_distance(b, a) == pytest.approx(d, abs=1e-9)
        assert -1e-12 <= d <= PI / 2 + 1e-9


# --- circular_distance ∈ [0, π] ------------------------------------------------
def test_circular_distance_opposite_is_pi():
    assert circular_distance(0.0, PI) == pytest.approx(PI, abs=1e-9)


def test_circular_distance_orthogonal_is_half_pi():
    assert circular_distance(0.0, PI / 2) == pytest.approx(PI / 2, abs=1e-9)


def test_circular_distance_wraps_short_way():
    assert circular_distance(0.0, 2 * PI - 0.1) == pytest.approx(0.1, abs=1e-9)


def test_circular_distance_symmetric():
    assert circular_distance(0.3, 5.0) == pytest.approx(circular_distance(5.0, 0.3), abs=1e-9)


# --- subtype_separation_stat: THE Point-1 regression --------------------------
def test_axial_collapses_180_but_polarity_does_not():
    # subtype 0 points one way, subtype 1 the EXACT opposite end of the SAME axis.
    angles = np.array([0.3, 0.3, 0.3, 0.3 + PI, 0.3 + PI, 0.3 + PI])
    labels = np.array([0, 0, 0, 1, 1, 1])
    t_axis = subtype_separation_stat(angles, labels, mode="axis")
    t_pol = subtype_separation_stat(angles, labels, mode="pol")
    assert t_axis == pytest.approx(0.0, abs=1e-9)   # same axis → axial sep ~ 0
    assert t_pol == pytest.approx(PI, abs=1e-9)      # opposite ends → polarity sep ~ π


def test_orthogonal_subtypes_axial_sep_is_half_pi():
    angles = np.array([0.0, 0.0, 0.0, PI / 2, PI / 2, PI / 2])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert subtype_separation_stat(angles, labels, mode="axis") == pytest.approx(PI / 2, abs=1e-9)


def test_separation_stat_k3_size_weighted():
    # 3 subtypes; reduces to a finite weighted pairwise mean, bounded by π/2 (axis)
    angles = np.array([0.0, 0.0, PI / 2, PI / 2, PI / 4, PI / 4])
    labels = np.array([0, 0, 1, 1, 2, 2])
    t = subtype_separation_stat(angles, labels, mode="axis")
    assert 0.0 < t <= PI / 2 + 1e-9


# --- within_subject_perm_p ----------------------------------------------------
def test_perm_p_complete_separation_is_small():
    # two well-separated, noisy clusters (NOT on the 0 / π/2 axial fixed points), 5 per subtype
    # so a random label shuffle almost never reproduces the clean axis separation.
    angles = np.array([0.18, 0.22, 0.15, 0.25, 0.20,
                       1.35, 1.40, 1.32, 1.45, 1.38])
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    rng = np.random.default_rng(0)
    out = within_subject_perm_p(angles, labels, mode="axis", B=2000, rng=rng)
    assert out["eligibility"] == "ok"
    assert out["p"] < 0.05


def test_perm_p_no_signal_is_one():
    # all angles identical → T_obs=0 and every permutation also gives 0 → p == 1
    angles = np.full(6, 0.5)
    labels = np.array([0, 0, 0, 1, 1, 1])
    rng = np.random.default_rng(0)
    out = within_subject_perm_p(angles, labels, mode="axis", B=500, rng=rng)
    assert out["eligibility"] == "ok"
    assert out["p"] == pytest.approx(1.0, abs=1e-9)


def test_perm_p_single_subtype_is_case_series():
    angles = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0, 0, 0, 0])
    out = within_subject_perm_p(angles, labels, mode="axis", B=100, rng=np.random.default_rng(0))
    assert out["eligibility"] == "insufficient_subtypes"
    assert out["p"] is None


def test_perm_p_one_big_one_small_drops_to_insufficient():
    # subtype 1 has only 2 (< 3); after dropping it only subtype 0 remains → case-series
    angles = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    labels = np.array([0, 0, 0, 1, 1])
    out = within_subject_perm_p(angles, labels, mode="axis", B=100, rng=np.random.default_rng(0))
    assert out["eligibility"] == "insufficient_subtypes"
    assert out["dropped_subtypes"] == [1]
    assert out["p"] is None


def test_perm_p_drops_small_subtype_but_tests_the_rest():
    # sizes 5 / 3 / 2 → drop subtype 2 (size 2), test 0 vs 1 (both ≥3) → eligible
    angles = np.array([0.18, 0.22, 0.15, 0.25, 0.20,   # subtype 0
                       1.35, 1.40, 1.32,                # subtype 1
                       0.80, 0.85])                     # subtype 2 (dropped)
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
    out = within_subject_perm_p(angles, labels, mode="axis", B=2000, rng=np.random.default_rng(0))
    assert out["eligibility"] == "ok"
    assert out["dropped_subtypes"] == [2]
    assert out["k"] == 2                # only the two kept subtypes are tested
    assert out["p"] < 0.05             # 0 vs 1 are well separated


# --- direction_clustering -----------------------------------------------------
def test_direction_clustering_concentrated_is_clustered():
    angles = np.array([0.30, 0.31, 0.29, 0.30, 0.32])
    out = direction_clustering(angles, r_min=0.5)
    assert out["R_axial"] > 0.9
    assert out["clustered"] is True


def test_direction_clustering_uniform_is_not_clustered():
    angles = np.linspace(0.0, 2 * PI, 60, endpoint=False)
    out = direction_clustering(angles, r_min=0.5)
    assert out["R_axial"] < 0.2
    assert out["clustered"] is False


# --- coord_aspect_ratio -------------------------------------------------------
def test_coord_aspect_collinear_is_zero():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])   # perfectly collinear
    assert coord_aspect_ratio(x, y) == pytest.approx(0.0, abs=1e-9)


def test_coord_aspect_square_is_one():
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert coord_aspect_ratio(x, y) == pytest.approx(1.0, abs=1e-9)


# --- align_subtype_to_direction (join on seizure_id, fail-loud on mismatch) ----
def test_align_joins_on_seizure_id_and_drops_outliers_and_nan():
    seizure_id_to_subtype = {"S0": 1, "S2": 0, "S3": -1}   # S3 is outlier → dropped
    idx_to_seizure_id = {0: "S0", 1: "S1", 2: "S2", 3: "S3"}
    eligible_idxs = [0, 1, 2, 3]
    angles_by_idx = {0: 0.5, 1: 1.0, 2: float("nan"), 3: 2.0}  # idx2 NaN dropped; S1 has no subtype
    out = align_subtype_to_direction(seizure_id_to_subtype, idx_to_seizure_id, eligible_idxs, angles_by_idx)
    # only idx0 survives: S0 has subtype 1 and finite angle; S2 NaN; S3 outlier; S1 no label
    assert [r["seizure_id"] for r in out] == ["S0"]
    assert out[0]["subtype"] == 1
    assert out[0]["theta"] == pytest.approx(0.5)


def test_align_raises_on_namespace_mismatch():
    seizure_id_to_subtype = {"A": 0, "B": 1}     # z-ER namespace
    idx_to_seizure_id = {0: "1", 1: "2"}          # audit namespace — zero overlap
    with pytest.raises(ValueError):
        align_subtype_to_direction(seizure_id_to_subtype, idx_to_seizure_id, [0, 1], {0: 0.1, 1: 0.2})


# --- oddeven_subtype_imbalance (TV distance) ----------------------------------
def test_imbalance_balanced_is_zero():
    idx_to_subtype = {0: 0, 1: 1, 2: 0, 3: 1}
    even, odd = {0, 2}, {1, 3}   # even = {0,0}, odd = {1,1}? no — map: 0→0,2→0 ; 1→1,3→1
    # even half = subtypes {0,0}, odd half = {1,1} → fully imbalanced, not balanced.
    # Build a genuinely balanced split instead:
    idx_to_subtype = {0: 0, 1: 1, 2: 0, 3: 1}
    even, odd = {0, 1}, {2, 3}   # even = {0,1}, odd = {0,1} → balanced
    assert oddeven_subtype_imbalance(even, odd, idx_to_subtype) == pytest.approx(0.0, abs=1e-9)


def test_imbalance_fully_split_is_one():
    idx_to_subtype = {0: 0, 1: 0, 2: 1, 3: 1}
    even, odd = {0, 1}, {2, 3}   # even all subtype0, odd all subtype1
    assert oddeven_subtype_imbalance(even, odd, idx_to_subtype) == pytest.approx(1.0, abs=1e-9)


def test_imbalance_empty_half_is_nan():
    idx_to_subtype = {0: 0, 1: 1}
    assert np.isnan(oddeven_subtype_imbalance({0, 1}, set(), idx_to_subtype))
