from __future__ import annotations

import numpy as np

from scripts.run_topic5_event_innovation_v3_0_phase1_measurement import (
    train_contact_backbone,
    unit_balanced_dense_fields,
)
from src.topic5_event_innovation_data import ContinuitySequence


def _sequence(name: str, indices: np.ndarray) -> ContinuitySequence:
    return ContinuitySequence(
        continuity_unit_id=name,
        event_indices=indices,
        event_times=indices.astype(float),
        source_ids=np.repeat(name, len(indices)),
    )


def test_train_backbone_masks_nonparticipating_phantom_rank():
    rank = np.array([[0.0, 99.0], [1.0, 0.0], [99.0, 1.0]])
    participation = np.array([[1, 0], [1, 1], [0, 1]], dtype=bool)
    backbone, support = train_contact_backbone(rank, participation, np.arange(3))
    np.testing.assert_allclose(backbone, [0.5, 0.5])
    np.testing.assert_array_equal(support, [2, 2])


def test_dense_field_weights_give_each_continuity_unit_equal_total_mass():
    rank = np.arange(20, dtype=float).reshape(10, 2)
    participation = np.ones_like(rank, dtype=bool)
    fields, support, weights = unit_balanced_dense_fields(
        rank,
        participation,
        [_sequence("long", np.arange(7)), _sequence("short", np.arange(7, 10))],
        window=2,
    )
    assert fields.shape == support.shape == (6, 2)
    np.testing.assert_allclose(weights[:5].sum(), 1.0)
    np.testing.assert_allclose(weights[5:].sum(), 1.0)
