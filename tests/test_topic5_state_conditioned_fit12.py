import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_topic5_state_conditioned_prefix_fields import (
    event_primary_onset,
    events_after_prefix,
    parent_event_contract,
)
from scripts.build_topic5_state_conditioned_fit2_rnn_dataset import load_targets
from scripts.run_topic5_state_conditioned_fit12 import verify_fit2
from scripts.train_topic5_state_conditioned_rnn import swap_targets, target_column
from scripts.analyze_topic5_state_conditioned_fit2_rnn import history_pairing_null


def test_clinical_onset_is_primary_for_epilepsiae():
    row = {"clin_onset_epoch": 100.0, "eeg_onset_epoch": 140.0}
    assert event_primary_onset(row, "epilepsiae") == 100.0
    assert event_primary_onset(row, "yuquan") == 140.0


def test_prefix_filter_uses_exact_parent_events_and_clinical_time():
    parent = pd.DataFrame(
        [
            {"dataset": "epilepsiae", "subject": "epilepsiae_1", "seizure_idx": 0},
            {"dataset": "epilepsiae", "subject": "epilepsiae_1", "seizure_idx": 2},
        ]
    )
    inventory = [
        {"clin_onset_epoch": 90.0, "eeg_onset_epoch": 150.0},
        {"clin_onset_epoch": 200.0, "eeg_onset_epoch": 200.0},
        {"clin_onset_epoch": 130.0, "eeg_onset_epoch": 170.0},
    ]
    assert events_after_prefix("epilepsiae_1", parent, 100.0, inventory) == [2]


def test_fit2_gate_uses_strict_bb150_not_within_shaft():
    summary = {
        "cohort_statistics": [
            {
                "group_id": "strict_broadband",
                "n_subjects": 10,
                "n_seizures": 20,
                "data_median": 0.8,
                "null_median": 0.7,
                "margin_median": 0.1,
                "n_data_gt_null": 7,
                "wilcoxon_one_sided_data_gt_null_p": 0.02,
            },
            {
                "group_id": "all_phenotype_matched",
                "n_subjects": 11,
                "n_seizures": 25,
                "data_median": 0.8,
                "null_median": 0.7,
                "margin_median": 0.1,
                "n_data_gt_null": 8,
                "wilcoxon_one_sided_data_gt_null_p": 0.01,
            },
        ]
    }
    verdict = verify_fit2(summary)
    assert verdict["fit2_pass"]
    assert "within_shaft" not in verdict["checks"]


def test_parent_contract_is_frozen_17_by_167():
    frame = parent_event_contract()
    assert frame.subject.nunique() == 17
    assert len(frame) == 167
    assert set(frame.phenotype) == {"strict_broadband", "gamma_nonbroadband"}


def test_fit2_rnn_target_is_ab_swap_invariant():
    cfg = {
        "target": {
            "primary_label_column": "target_scaffold_margin_bb150",
            "swap_equivariance": "invariant",
        }
    }
    y = np.array([-0.1, 0.2])
    assert target_column(cfg) == "target_scaffold_margin_bb150"
    np.testing.assert_array_equal(swap_targets(y, cfg), y)


def test_legacy_signed_target_keeps_sign_equivariance():
    cfg = {"target": {"swap_equivariance": "sign_flip"}}
    y = np.array([-0.1, 0.2])
    np.testing.assert_array_equal(swap_targets(y, cfg), -y)


def test_fit2_rnn_target_is_strict_bb150_channel_shuffle_margin():
    cfg = {
        "cohort": {
            "parent_event_table": (
                "results/topic5_state_conditioned_predictor/"
                "fit12_clinical_bb150/fit2/"
                "fig6_fit2_clinical_onset_scaffold_event.csv"
            )
        },
        "target": {"primary_group": "strict_broadband"},
    }
    frame = load_targets(cfg)
    assert frame.subject.nunique() == 13
    assert len(frame) == 71
    assert set(frame.time_reference) == {"clinical_onset"}
    assert set(frame.band) == {"broadband_1_150"}
    np.testing.assert_allclose(
        frame.target_scaffold_margin_bb150,
        frame.observed - frame.null_median,
    )


def test_history_pairing_null_does_not_treat_seeds_as_samples():
    rows = []
    for subject in ("s1", "s2"):
        for seed in (1, 2, 3):
            for seizure_idx, target in enumerate((0.0, 1.0)):
                rows.append(
                    {
                        "subject": subject,
                        "seed": seed,
                        "seizure_idx": seizure_idx,
                        "target": target,
                        "rnn_prediction": target,
                        "rnn_absolute_error": 0.0,
                    }
                )
    result = history_pairing_null(
        pd.DataFrame(rows), "target", draws=4000, seed=7
    )
    assert result["n_eligible_subjects"] == 2
    # Two patients with two events provide only 2^2 independent pairings.
    assert result["empirical_p_observed_lower"] >= 0.20


def test_fit2_rnn_event_attrition_is_frozen_and_fail_closed():
    frame = pd.read_csv(
        "results/topic5_state_conditioned_predictor/"
        "dataset_fit2_clinical_bb150/event_attrition.csv"
    )
    assert len(frame) == 71
    assert int(frame.eligible_history_target_pair.sum()) == 11
    excluded = frame[~frame.eligible_history_target_pair.astype(bool)]
    assert set(excluded.reason) == {
        "too_few_definite_interictal_history_events"
    }
