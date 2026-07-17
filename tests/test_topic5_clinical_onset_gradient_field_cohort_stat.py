import numpy as np

from scripts.run_topic5_clinical_onset_gradient_field_cohort_stat import (
    event_ids_for_group,
    gamma30_onset_activation,
    readout_group_for_event,
    score_cached_activation,
)
from src.topic5_template_axis_field import make_field_scorer


def test_three_group_event_selectors_are_fixed_intersections():
    selector = {
        "strict_broadband": {1, 3, 8},
        "gamma_nonbroadband": {2, 7},
    }
    eligible = [1, 2, 3, 4]
    assert event_ids_for_group(
        eligible, selector, "all_phenotype_matched"
    ) == [1, 2, 3]
    assert event_ids_for_group(eligible, selector, "strict_broadband") == [1, 3]
    assert event_ids_for_group(eligible, selector, "gamma_nonbroadband") == [2]
    assert readout_group_for_event(
        "all_phenotype_matched", 1, selector
    ) == "strict_broadband"
    assert readout_group_for_event(
        "all_phenotype_matched", 2, selector
    ) == "gamma_nonbroadband"


def test_cached_activation_join_and_channel_null_are_seed_reproducible():
    names = ["A1", "A2", "A3", "B1", "B2", "B3"]
    points_a = np.column_stack([np.linspace(0, 1, 6), np.zeros(6)])
    points_b = np.column_stack([np.linspace(1, 0, 6), np.linspace(-0.2, 0.2, 6)])
    support = np.ones(6)
    scorers = {
        "own_a": make_field_scorer(
            np.linspace(1, -1, 6), points_a, support, 0.3
        ),
        "own_b": make_field_scorer(
            np.linspace(-1, 1, 6), points_b, support, 0.3
        ),
    }
    record = {"interictal_field": {"contact_order": names}}
    source_names = list(reversed(names))
    activation = np.asarray([0.2, 0.5, 0.9, -0.1, -0.4, -0.8])
    first = score_cached_activation(
        record, scorers, "own_maxab", source_names, activation,
        subject="synthetic", seizure_idx=2, band="gamma_30_80",
        n_perm=40, seed=13,
    )
    second = score_cached_activation(
        record, scorers, "own_maxab", source_names, activation,
        subject="synthetic", seizure_idx=2, band="gamma_30_80",
        n_perm=40, seed=13,
    )
    assert first["observed"] == second["observed"]
    np.testing.assert_allclose(first["null"], second["null"])
    assert first["null"].shape == (40,)
    assert first["n_finite_contacts"] == 6


def test_gamma30_activation_uses_only_clinical_onset_zero_to_ten_seconds():
    relt = np.array([-0.1, 0.0, 5.0, 10.0, 10.1])
    z = np.array([[100.0, 1.0, 3.0, 5.0, 200.0],
                  [100.0, 2.0, 4.0, 6.0, 200.0]])
    cache = {
        "gamma_LVFA__zt__4": z,
        "gamma_LVFA__relt__4": relt,
    }
    np.testing.assert_allclose(gamma30_onset_activation(cache, 4), [3.0, 4.0])
