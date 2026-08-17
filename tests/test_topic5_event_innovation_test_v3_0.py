import numpy as np

from scripts.run_topic5_event_innovation_v3_0_local_response import ResponseRows
from src.topic5_event_innovation_data import ContinuitySequence
from src.topic5_event_innovation_test_v3_0 import (
    combine_response_rows,
    fit_final_test_innovations,
)
from src.topic5_event_innovation_v3_0 import fit_rank_state_basis


def _rows(group):
    return ResponseRows(
        event_index=np.asarray([group]),
        group=np.asarray([0], dtype=np.int32),
        pre_state=np.asarray([[1.0]]),
        future_state=np.asarray([[2.0]]),
        past_state=np.asarray([[0.0]]),
        innovation_state=np.asarray([[0.5]]),
        nuisance=np.asarray([[0.0, 0.0, 0.0]]),
        observed_future_field=np.asarray([[0.2, 0.8]]),
        future_support=np.asarray([[1, 1]]),
        future_windows=[np.asarray([group])],
    )


def test_combined_response_rows_keep_groups_disjoint():
    combined = combine_response_rows([_rows(1), _rows(2)])
    assert combined.event_index.tolist() == [1, 2]
    assert combined.group.tolist() == [0, 1]
    assert len(combined.future_windows) == 2


def test_final_observer_refits_train_validation_and_emits_test_only():
    rng = np.random.default_rng(2)
    rank = rng.uniform(size=(180, 3))
    participation = np.ones_like(rank, dtype=bool)

    def sequence(name, start, stop):
        indices = np.arange(start, stop)
        return ContinuitySequence(
            continuity_unit_id=name,
            event_indices=indices,
            event_times=indices.astype(float),
            source_ids=np.asarray([name] * len(indices)),
        )

    sequences = {
        "train": [sequence("train", 0, 60)],
        "validation": [sequence("validation", 60, 120)],
        "test": [sequence("test", 120, 180)],
    }
    basis = fit_rank_state_basis(rank[:60], 2)
    innovations = fit_final_test_innovations(
        {"rank": rank, "participation": participation},
        sequences,
        basis,
        {"ladder": "pre20", "alpha": 1.0},
        {"observer_minimum_observations": 10},
    )
    assert min(innovations) == 140
    assert max(innovations) == 179
    assert all(value[0].shape == (3,) for value in innovations.values())
