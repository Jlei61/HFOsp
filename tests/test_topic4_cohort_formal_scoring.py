from __future__ import annotations

import numpy as np
import pytest

from src.topic4_cohort_formal_layout import build_subject_layout
from src.topic4_cohort_formal_scoring import (
    UNSCORABLE_LOSS,
    adjudicate,
    cohort_summary,
    confirm_subject,
    endpoint_statistic,
    permutation_p_value,
)
from src.topic4_data_driven_cohort import TargetConfig, build_crossfit_patient_target

NAMES = ["A1", "A2", "A3", "A4", "B1", "B2"]


def _patient_target():
    rng = np.random.default_rng(4)
    n_blocks, per_block = 10, 16
    ranks = np.empty((len(NAMES), n_blocks * per_block), float)
    for event in range(ranks.shape[1]):
        order = np.arange(len(NAMES))
        if event % 2:
            order = order[::-1]
        noisy = order + rng.normal(0.0, 0.01, len(NAMES))
        ranks[:, event] = np.argsort(np.argsort(noisy, kind="stable"), kind="stable")
    data = {
        "channel_names": NAMES,
        "ranks": ranks,
        "bools": np.ones_like(ranks, bool),
        "block_ids": np.repeat(np.arange(n_blocks), per_block),
    }
    pair = {
        "contact_order": NAMES,
        "rank_a": np.arange(len(NAMES), dtype=float),
        "rank_b": np.arange(len(NAMES), dtype=float)[::-1],
        "cluster_id_a": 0,
        "cluster_id_b": 1,
    }
    return build_crossfit_patient_target(
        data, pair,
        config=TargetConfig(
            minimum_participating_contacts=3, heldout_block_fraction=0.3,
            split_seed=8, kmeans_fit_max_events=1000, kmeans_n_init=10,
            kmeans_seed=9, stability_seeds=(9, 10, 11),
            stored_events_per_mode_per_split=100,
        ),
    )


def _kwargs(target):
    descriptors = target["heldout_descriptors"]
    return {
        "target": {
            "contact_order": NAMES,
            "profiles": target["heldout_profiles"],
            "recruitment": np.asarray([
                descriptors["TA"]["recruitment"], descriptors["TB"]["recruitment"],
            ]),
            "precedence": np.asarray([
                descriptors["TA"]["precedence"], descriptors["TB"]["precedence"],
            ]),
        },
        "contact_names": NAMES,
        "patient_centers": target["kmeans_centers"],
        "ood_threshold": target["train_distance_q95"],
        "minimum_contacts": 3,
        "minimum_events_per_mode": 3,
        "kmeans_seed": 20260815,
        "kmeans_n_init": 10,
    }


def _permutations():
    return build_subject_layout(
        "test_subject", NAMES, real_coords_sheet=None, n_permutations=64,
        base_seed=20260820, sheet_size_mm=20.0, margin_mm=2.0,
    )["arrays"]["within_shaft_null_permutations"]


def _matching_model(target, per_mode=20):
    return np.vstack([
        target["heldout_samples"]["TA"][:per_mode],
        target["heldout_samples"]["TB"][:per_mode],
    ])


def test_unscorable_readouts_are_charged_the_ceiling_not_dropped():
    assert endpoint_statistic({"status": "INSUFFICIENT_EVENTS"}) == UNSCORABLE_LOSS
    assert endpoint_statistic(
        {"status": "EVALUABLE", "weakest_mode_loss": 0.2}
    ) == pytest.approx(0.2)


def test_permutation_p_value_counts_the_observation_itself():
    assert permutation_p_value(0.1, np.asarray([0.2, 0.3, 0.4])) == pytest.approx(0.25)
    assert permutation_p_value(0.5, np.asarray([0.2, 0.3, 0.4])) == pytest.approx(1.0)


def test_a_model_that_matches_the_patient_beats_its_within_shaft_null():
    target = _patient_target()
    result = confirm_subject(
        _matching_model(target), permutations=_permutations(),
        minimum_events=3, minimum_seed_ami=0.5, **_kwargs(target),
    )
    assert result["status"] == "EVALUABLE"
    assert result["subject_endpoint_pass"] is True
    assert result["delta_null_median_minus_observed"] > 0.0
    assert result["permutation_p"] <= result["minimum_reachable_p"] + 1e-12
    assert result["natural_kmeans"]["same_network_k2"] is True


def test_a_contact_scrambled_model_does_not_beat_its_own_null():
    target = _patient_target()
    model = _matching_model(target)
    rng = np.random.default_rng(0)
    scrambled = model[:, rng.permutation(len(NAMES))]
    result = confirm_subject(
        scrambled, permutations=_permutations(),
        minimum_events=3, minimum_seed_ami=0.5, **_kwargs(target),
    )
    assert result["subject_endpoint_pass"] is False


def test_null_rejects_permutations_that_do_not_span_the_model_contacts():
    target = _patient_target()
    with pytest.raises(ValueError, match="do not span the model contacts"):
        confirm_subject(
            _matching_model(target), permutations=np.zeros((4, 3), int),
            minimum_events=3, minimum_seed_ami=0.5, **_kwargs(target),
        )


def _row(subject, delta, passed, events=50, seed=1681, k2=True):
    return {
        "subject_id": subject,
        "confirmation_seed": seed,
        "delta_null_median_minus_observed": delta,
        "subject_endpoint_pass": passed,
        "n_in_distribution_events": events,
        "natural_kmeans": {"same_network_k2": k2},
    }


def test_cohort_summary_uses_the_subject_as_the_unit_and_reports_robustness():
    rows = [
        _row(f"s{index}", 0.05 + 0.001 * index, True, events=40 + index,
             seed=1681 + index % 4)
        for index in range(20)
    ]
    rows += [_row(f"s{20 + index}", -0.01, False, events=200) for index in range(4)]
    per_seed = {
        str(seed): [True] * 20 + [False] * 4 for seed in (1681, 1682, 1683, 1684)
    }
    summary = cohort_summary(
        rows, [], pass_fraction_min=0.6, alpha=0.05, per_seed_pass=per_seed,
    )
    assert summary["n_subjects"] == 24
    assert summary["pass_fraction"] == pytest.approx(20 / 24)
    assert summary["pass_fraction_met"] is True
    assert summary["primary_test"]["wilcoxon_p"] < 0.05
    assert summary["primary_significant"] is True
    assert summary["robustness"]["without_top_event_quartile"]["n"] == 18
    assert set(summary["robustness"]["pass_by_confirmation_seed"]) == {
        "1681", "1682", "1683", "1684",
    }
    assert summary["robustness"]["cohort_survives_every_single_seed"] is True


def test_cohort_flags_when_one_confirmation_seed_carries_the_result():
    rows = [_row(f"s{index}", 0.05, True) for index in range(12)]
    per_seed = {
        "1681": [True] * 12, "1682": [False] * 12,
        "1683": [False] * 12, "1684": [False] * 12,
    }
    summary = cohort_summary(
        rows, [], pass_fraction_min=0.6, alpha=0.05, per_seed_pass=per_seed,
    )
    assert summary["robustness"]["worst_single_seed_pass_fraction"] == 0.0
    assert summary["robustness"]["cohort_survives_every_single_seed"] is False


def test_cohort_summary_flags_a_layout_direction_disagreement():
    canonical = [_row(f"s{index}", 0.05, True) for index in range(10)]
    real = [_row(f"s{index}", -0.05, False) for index in range(10)]
    summary = cohort_summary(canonical, real, pass_fraction_min=0.6, alpha=0.05)
    assert summary["sensitivity"]["directions_agree"] is False
    verdict = adjudicate(summary, same_network_k2_min=0.5)
    assert verdict["status"] == "OBSERVATION_LAYOUT_DEPENDENCE_UNRESOLVED"


def test_adjudication_reports_insufficient_support_before_any_other_gate():
    rows = [_row(f"s{index}", -0.05, False) for index in range(10)]
    summary = cohort_summary(rows, [], pass_fraction_min=0.6, alpha=0.05)
    verdict = adjudicate(summary, same_network_k2_min=0.5)
    assert verdict["status"] == "COHORT_MODEL_SUPPORT_INSUFFICIENT"
    assert len(verdict["reasons"]) == 2


def test_adjudication_lists_every_failed_gate_not_just_the_first():
    """Two gates can break at once; naming one hides the other."""
    canonical = [_row(f"s{index}", 0.05, True, seed=1681 + index % 4, k2=False)
                 for index in range(12)]
    real = [_row(f"s{index}", -0.05, False) for index in range(12)]
    summary = cohort_summary(canonical, real, pass_fraction_min=0.6, alpha=0.05)
    verdict = adjudicate(summary, same_network_k2_min=0.5)
    assert verdict["status"] == "SAME_NETWORK_K2_INSUFFICIENT"
    assert verdict["failed_gates"] == [
        "SAME_NETWORK_K2_INSUFFICIENT", "OBSERVATION_LAYOUT_DEPENDENCE_UNRESOLVED",
    ]
    assert len(verdict["all_reasons"]) == 2


def test_adjudication_reports_a_missing_same_network_k2_separately():
    rows = [_row(f"s{index}", 0.05, True, seed=1681 + index % 4, k2=False)
            for index in range(12)]
    summary = cohort_summary(rows, [], pass_fraction_min=0.6, alpha=0.05)
    verdict = adjudicate(summary, same_network_k2_min=0.5)
    assert verdict["status"] == "SAME_NETWORK_K2_INSUFFICIENT"
