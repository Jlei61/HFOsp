import numpy as np
import pytest
import torch

from src.topic5_continuous_marked_state_r1.baseline import (
    ExactHistoryMarkDecoder, HistoryIntensity,
)
from src.topic5_continuous_marked_state_r1.observer import ObservationTransformer
from src.topic5_continuous_marked_state_r1.r1_2b import (
    JointLastLayerStateModel, LastSpatialObserver, fit_joint_t1,
)


def _observer() -> ObservationTransformer:
    torch.manual_seed(4)
    return ObservationTransformer(
        3, d_model=64, n_heads=4, temporal_layers=1,
        spatial_layers=1, raw_enabled=True,
    )


def _checkpoint() -> dict:
    timing = HistoryIntensity(3, history_visible=True)
    mark = ExactHistoryMarkDecoder(
        3, 2, np.zeros((1, 2, 2), dtype=np.float32)
    )
    return {
        "timing": {"history": timing.state_dict()},
        "mark": {"history": mark.state_dict()},
    }


def test_zero_raw_gain_makes_joint_arms_exactly_paired() -> None:
    source = _observer()
    explicit = LastSpatialObserver(source, raw_enabled=False).eval()
    raw = LastSpatialObserver(source, raw_enabled=True).eval()
    base = torch.randn(5, 2, 64)
    raw_node = torch.randn(5, 2, 64)
    mask = torch.tensor([[1, 1], [1, 0], [1, 1], [0, 1], [1, 1]], dtype=torch.bool)
    with torch.no_grad():
        left = explicit(base, raw_node, mask)
        right = raw(base, raw_node, mask)
    assert float(raw.raw_gain) == 0.0
    assert torch.equal(left, right)


def test_only_last_spatial_observer_tail_is_present_and_trainable() -> None:
    model = JointLastLayerStateModel(
        _checkpoint(), 3, 2, np.zeros((1, 2, 2), dtype=np.float32),
        _observer(), raw_enabled=True, state_dim=2,
    )
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    assert any(name.startswith("last_observer.spatial") for name in names)
    assert "last_observer.raw_gain" in names
    assert not any("tokenizer" in name or "raw.transformer" in name for name in names)
    assert not any(name.startswith("timing_baseline") for name in names)
    assert not any(name.startswith("mark_baseline") for name in names)


def test_joint_fit_rejects_nonregistered_lr_ratio() -> None:
    model = JointLastLayerStateModel(
        _checkpoint(), 3, 2, np.zeros((1, 2, 2), dtype=np.float32),
        _observer(), raw_enabled=False, state_dim=2,
    )
    with pytest.raises(ValueError, match="0.1 x state LR"):
        fit_joint_t1(
            model, None, None, None, None, device="cpu",
            state_learning_rate=3e-4, observer_learning_rate=2e-5,
        )
