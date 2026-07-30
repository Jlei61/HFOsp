import pandas as pd
import pytest

from scripts.select_topic5_stagea_hidden_size import select_smallest_one_se
from scripts.summarize_topic5_interictal_operator_stage_a import (
    _coverage_status,
)


def test_one_se_prefers_smaller_size_when_it_is_within_threshold():
    summary = pd.DataFrame(
        {
            "hidden_size": [32, 64],
            "mean_inner_validation_loss": [1.02, 1.00],
            "se_inner_validation_loss": [0.01, 0.03],
        }
    )
    selected, best, threshold = select_smallest_one_se(summary)
    assert selected == 32
    assert best == 64
    assert threshold == pytest.approx(1.03)


def test_one_se_keeps_best_when_smaller_model_is_outside_threshold():
    summary = pd.DataFrame(
        {
            "hidden_size": [32, 64],
            "mean_inner_validation_loss": [1.10, 1.00],
            "se_inner_validation_loss": [0.01, 0.03],
        }
    )
    selected, best, _ = select_smallest_one_se(summary)
    assert selected == 64
    assert best == 64


def test_one_se_fails_closed_without_uncertainty():
    summary = pd.DataFrame(
        {
            "hidden_size": [32, 64],
            "mean_inner_validation_loss": [1.02, 1.00],
            "se_inner_validation_loss": [0.01, float("nan")],
        }
    )
    with pytest.raises(ValueError, match="finite uncertainty"):
        select_smallest_one_se(summary)


def test_partial_screen_cannot_become_formal_by_lowering_cli_expectations():
    subject = pd.DataFrame(
        {
            "subject": [f"patient_{index}" for index in range(9)],
            "n_seeds": [1] * 9,
        }
    )
    requested, formal = _coverage_status(
        subject, failures=[], expected_patients=9, expected_seeds=1
    )
    assert requested is True
    assert formal is False


def test_frozen_13_by_3_contract_is_formally_eligible():
    subject = pd.DataFrame(
        {
            "subject": [f"patient_{index}" for index in range(13)],
            "n_seeds": [3] * 13,
        }
    )
    requested, formal = _coverage_status(
        subject, failures=[], expected_patients=13, expected_seeds=3
    )
    assert requested is True
    assert formal is True
