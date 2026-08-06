import numpy as np

from scripts.paper_figures.plot_fig3f_ab_dominance_cohort import (
    _heatmap_pattern_metrics,
    _pretty,
)
from scripts.run_fig3f_ab_dominance_cohort import (
    _binomial_summary,
    _fixed_contrast,
    axis_present_fixed_mapping,
    circular_shift_polar_null,
    cohort_hierarchical_null,
    subject_paired_null,
)


def test_fixed_contrast_respects_frozen_contact_order():
    record = {
        "interictal_field": {
            "contact_order": ["A1", "A2", "B1", "B2"],
            "rank_a": [0, 1, 2, 3],
            "rank_b": [3, 2, 1, 0],
        }
    }
    result = _fixed_contrast(record, ["B2", "A1", "B1"])
    assert result["rho_AB"] == -1.0
    assert result["template_pair_tier"] == "reciprocal"
    full = 2 * np.array([-1.3416407865, 1.3416407865, -0.4472135955])
    assert np.allclose(result["D_AB"], full)
    assert result["D_AB"][1] > result["D_AB"][2] > result["D_AB"][0]


def test_fixed_mapping_gate_marks_low_dof_single_shaft():
    names = [f"A{i}" for i in range(1, 7)]
    e_a = np.linspace(1, -1, 6)
    e_b = -e_a
    values = np.tile(e_a, (4, 1))
    result = axis_present_fixed_mapping(
        values, names, e_a, e_b, np.random.default_rng(0), n_perm=99
    )
    assert result["testable"] is False
    assert result["low_dof"] is True


def test_cohort_binomial_uses_subjects_not_seizures():
    records = [
        {"primary": {"eligible": True, "subject_locked": True}},
        {"primary": {"eligible": True, "subject_locked": False}},
        {"primary": {"eligible": False, "subject_locked": True}},
    ]
    result = _binomial_summary(records)
    assert result["k_locked"] == 1
    assert result["n_eligible"] == 2
    assert np.isclose(result["one_sided_exact_binomial_p"], 0.0975)


def test_subject_paired_null_endpoints_match_plotted_delta():
    centers = np.arange(-115.0, 16.0, 2.0)
    base = np.zeros_like(centers)
    base[(centers >= -30) & (centers < 10)] = 0.8
    seizures = []
    for offset in (0.00, 0.03, -0.02):
        shifted = circular_shift_polar_null(
            base + offset, np.ones_like(base, bool), centers
        )
        assert shifted["status"] == "ok"
        seizures.append(shifted)
    result = subject_paired_null(seizures, n_perm=199, seed=4)
    assert result["n_valid_seizures"] == 3
    assert np.isclose(result["delta"], result["polar_near"] - result["polar_far"])
    assert result["delta"] > 0.7
    assert result["subject_locked"]


def test_cohort_hierarchical_null_can_use_sensitivity_key():
    records = [
        {"within_shaft_sensitivity": {"eligible": True, "delta": 0.4}},
        {"within_shaft_sensitivity": {"eligible": True, "delta": 0.2}},
        {"within_shaft_sensitivity": {"eligible": False, "delta": -1.0}},
    ]
    nulls = [np.linspace(-0.2, 0.1, 101), np.linspace(-0.1, 0.1, 101)]
    result = cohort_hierarchical_null(
        records, nulls, n_perm=499, seed=2, key="within_shaft_sensitivity"
    )
    assert result["n"] == 2
    assert np.isclose(result["median_delta"], 0.3)
    assert result["p_one_sided"] < 0.01


def test_heatmap_pattern_groups_red_blue_then_mixed():
    centers = np.arange(-115.0, 16.0, 2.0)
    near = (centers >= -30) & (centers < 10)
    red = np.zeros_like(centers)
    red[near] = 0.5
    blue = np.zeros_like(centers)
    blue[near] = -0.4
    mixed = np.zeros_like(centers)
    mixed[near] = np.tile([0.5, -0.5], near.sum() // 2)
    assert _heatmap_pattern_metrics(red, centers)["group_rank"] == 0
    assert _heatmap_pattern_metrics(blue, centers)["group_rank"] == 1
    assert _heatmap_pattern_metrics(mixed, centers)["group_rank"] == 2


def test_manuscript_labels_follow_supplementary_code_order():
    assert _pretty("epilepsiae_1096") == "E1"
    assert _pretty("epilepsiae_1146") == "E10"
    assert _pretty("epilepsiae_635") == "E20"
    assert _pretty("yuquan_zhangkexuan") == "Y1"
