import numpy as np
import torch

from src.topic5_symmetric_axis_propagation_state_v2_2 import (
    event_first_patient_mean,
    normalized_event_nll,
    seed_median_patient_metric,
)


def test_event_first_prevents_long_events_from_becoming_pseudoreplicates():
    events = [[0.0] * 100, [10.0]]
    assert np.isclose(event_first_patient_mean(events), 5.0)
    assert not np.isclose(event_first_patient_mean(events), 10.0 / 101.0)


def test_seed_aggregation_is_median_after_patient_event_mean():
    seed_events = [
        [[1.0, 3.0], [2.0]],
        [[10.0], [10.0]],
        [[3.0], [3.0]],
    ]
    assert np.isclose(seed_median_patient_metric(seed_events), 3.0)


def test_normalized_nll_divides_each_decision_by_eligible_contacts():
    value = normalized_event_nll(
        [torch.tensor(-4.0), torch.tensor(-2.0)], [4, 1]
    )
    assert torch.isclose(value, torch.tensor(1.5))
