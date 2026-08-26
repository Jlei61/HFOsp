import numpy as np
import torch

from src.topic5_continuous_marked_state_r1.baseline import (
    ExactHistoryMarkDecoder, HistoryIntensity,
)
from src.topic5_continuous_marked_state_r1.r1_2 import (
    FullAnchorDesign, FrozenEmbeddingStateModel,
    filtered_anchor_states, memoryless_anchor_states,
)
from src.topic5_continuous_marked_state_r1.r1_2b_diagnostics import (
    strict_matched_wrong_time_permutations,
)


def _checkpoint() -> dict:
    timing = HistoryIntensity(11, history_visible=True)
    mark = ExactHistoryMarkDecoder(
        11, 2, np.zeros((1, 2, 2), dtype=np.float32)
    )
    return {
        "timing": {"history": timing.state_dict()},
        "mark": {"history": mark.state_dict()},
    }


def _design() -> FullAnchorDesign:
    n = 8
    time = np.arange(n, dtype=np.float64) * 2000.0
    history = np.zeros((n, 11), dtype=np.float32)
    history[:, 1] = np.linspace(-1, 1, n)
    return FullAnchorDesign(
        subject="toy", anchor_time=time, anchor_split=np.ones(n, dtype=np.int8),
        anchor_session=np.zeros(n, dtype=np.int64), anchor_history=history,
        event_time=time + 1.0, event_split=np.ones(n, dtype=np.int8),
        event_session=np.zeros(n, dtype=np.int64),
        event_source_anchor=np.arange(n, dtype=np.int64),
        event_history=history.copy(),
        event_group_ids=np.tile(np.asarray([[0, -1]], dtype=np.int64), (n, 1)),
        event_group_count=np.ones(n, dtype=np.int64),
        quadrature_time=time + 0.5,
        quadrature_split=np.ones(n, dtype=np.int8),
        quadrature_session=np.zeros(n, dtype=np.int64),
        quadrature_source_anchor=np.arange(n, dtype=np.int64),
        quadrature_history=history.copy(),
        quadrature_weight_seconds=np.ones(n, dtype=np.float64),
        session_label=np.asarray([0], dtype=np.int64),
        session_start=np.asarray([0.0], dtype=np.float64),
    )


def test_memoryless_anchor_codes_do_not_carry_previous_observation() -> None:
    torch.manual_seed(3)
    model = FrozenEmbeddingStateModel(
        _checkpoint(), 11, 2, np.zeros((1, 2, 2), dtype=np.float32),
        observation_dim=64, state_dim=2,
    )
    design = _design()
    embedding = np.zeros((8, 64), dtype=np.float32)
    embedding[0, 0] = 10.0
    persistent = filtered_anchor_states(model, design, embedding, device="cpu")
    memoryless = memoryless_anchor_states(model, design, embedding, device="cpu")
    assert not torch.allclose(persistent[1], memoryless[1])
    changed = embedding.copy(); changed[0, 0] = -10.0
    memoryless_changed = memoryless_anchor_states(model, design, changed, device="cpu")
    assert torch.equal(memoryless[1], memoryless_changed[1])


def test_strict_swap_returns_multiple_separated_same_session_donors() -> None:
    design = _design()
    permutation, matched, audit = strict_matched_wrong_time_permutations(
        design, np.ones(8), n_donors=3, min_separation_seconds=1800.0
    )
    assert permutation.shape == (3, 8)
    assert matched.all()
    for donor in permutation:
        assert np.all(np.abs(design.anchor_time[donor] - design.anchor_time) >= 1800.0)
    assert audit["n_donors"] == 3
