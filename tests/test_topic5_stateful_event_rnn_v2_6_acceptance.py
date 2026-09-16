from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.accept_topic5_stateful_event_rnn_v2_6 import (
    RESULT_ROOT,
    SCIENTIFIC_ADJUDICATION,
    build_state,
    directional_summary,
    load_patient_frame,
    support_strata,
)


FROZEN_TEST_STATE = RESULT_ROOT / "STATEFUL_TEST_STATE.json"


def test_directional_summary_sign_convention_treats_negative_as_favorable():
    favorable = directional_summary([-0.3, -0.2, -0.1, -0.05], favorable="negative")
    unfavorable = directional_summary([0.3, 0.2, 0.1, 0.05], favorable="negative")
    assert favorable["n_favorable"] == 4
    assert unfavorable["n_favorable"] == 0
    assert favorable["wilcoxon_one_sided_p"] < 0.1
    assert unfavorable["wilcoxon_one_sided_p"] > 0.9
    assert favorable["median"] == pytest.approx(-0.15)


def test_directional_summary_reports_both_tails_for_null_contrasts():
    values = [0.4, 0.3, 0.2, -0.05]
    summary = directional_summary(values, favorable="negative")
    assert summary["wilcoxon_one_sided_p"] + summary["wilcoxon_opposite_tail_p"] > 1.0
    assert summary["wilcoxon_opposite_tail_p"] < summary["wilcoxon_one_sided_p"]


def test_support_strata_are_nested_and_include_the_full_cohort():
    support = np.asarray([1, 2, 12, 25, 400])
    strata = support_strata(support, thresholds=(1, 10, 20, 50))
    assert list(strata[1]) == [True] * 5
    assert int(np.sum(strata[10])) == 3
    assert int(np.sum(strata[20])) == 2
    for smaller, larger in zip((1, 10, 20), (10, 20, 50)):
        assert np.all(strata[larger] <= strata[smaller])


@pytest.mark.skipif(
    not FROZEN_TEST_STATE.exists(), reason="frozen v2.6 test state is not available"
)
def test_derived_layer_reproduces_the_frozen_primary_endpoint_exactly():
    frame = load_patient_frame(RESULT_ROOT)
    frozen = json.load(FROZEN_TEST_STATE.open())["trained_primary_propagation"]
    derived = directional_summary(
        frame["trained_rnn_minus_ewma_propagation"], favorable="negative"
    )
    assert derived["n"] == frozen["n"]
    assert derived["median"] == pytest.approx(
        frozen["median_rnn_minus_ewma"], abs=0.0, rel=0.0
    )
    assert derived["n_favorable"] == frozen["n_rnn_better"]
    assert derived["wilcoxon_one_sided_p"] == pytest.approx(
        frozen["wilcoxon_one_sided_less_p"], abs=0.0, rel=0.0
    )
    assert derived["bootstrap_median_ci95"] == pytest.approx(
        frozen["bootstrap_median_ci95"], abs=0.0, rel=0.0
    )


@pytest.mark.skipif(
    not FROZEN_TEST_STATE.exists(), reason="frozen v2.6 test state is not available"
)
def test_scientific_adjudication_separates_state_tracking_from_state_shaping():
    state, _ = build_state(RESULT_ROOT)
    adjudication = state["scientific_adjudication"]
    assert adjudication["status"] == SCIENTIFIC_ADJUDICATION
    assert "trained_state_uses_recent_event_history" in adjudication["established"]
    assert "event_innovation_predicts_state_update" in adjudication["not_established"]
    assert "activity_dependent_network_shaping" in adjudication["not_established"]
