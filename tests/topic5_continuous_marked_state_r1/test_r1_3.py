import numpy as np
import torch
from types import SimpleNamespace

from src.topic5_continuous_marked_state_r1.baseline import (
    ExactHistoryMarkDecoder, HistoryIntensity,
)
from src.topic5_continuous_marked_state_r1.observer import ObservationTransformer
from src.topic5_continuous_marked_state_r1.r1_3 import (
    FullTargetObserverStateModel, _set_trainable, fit_target_observer,
)
import src.topic5_continuous_marked_state_r1.r1_3 as r1_3_module


def _checkpoint() -> dict:
    timing = HistoryIntensity(3, history_visible=True)
    mark = ExactHistoryMarkDecoder(
        3, 2, np.zeros((1, 2, 2), dtype=np.float32)
    )
    return {
        "timing": {"history": timing.state_dict()},
        "mark": {"history": mark.state_dict()},
    }


def _model(use_raw: bool) -> FullTargetObserverStateModel:
    observer = ObservationTransformer(
        13, d_model=64, n_heads=4, temporal_layers=2,
        spatial_layers=1, raw_enabled=True,
    )
    return FullTargetObserverStateModel(
        _checkpoint(), 3, 2, np.zeros((1, 2, 2), dtype=np.float32),
        observer, use_raw=use_raw, state_dim=2,
    )


def test_raw_arm_trains_tokenizer_and_every_temporal_block() -> None:
    model = _model(True)
    names = _set_trainable(model, stage="observer_alignment")
    assert "observer.raw.tokenizer.weight" in names
    assert any(name.startswith("observer.raw.transformer.layers.0") for name in names)
    assert any(name.startswith("observer.raw.transformer.layers.1") for name in names)
    assert not any(name.startswith("state.generator") for name in names)
    assert not any(name.startswith("state.correction") for name in names)
    assert not any(name.startswith("state_timing") for name in names)
    assert not any(name.startswith("state_contact") for name in names)
    assert not any(name.startswith("state_size") for name in names)
    assert not any(name.startswith("observer.spatial") for name in names)
    assert not any(name == "observer.pool_token" for name in names)


def test_raw_joint_stage_does_not_get_extra_common_training() -> None:
    model = _model(True)
    names = _set_trainable(model, stage="joint_alignment")
    assert any(name.startswith("observer.raw") for name in names)
    assert not any(name.startswith("state.correction") for name in names)
    assert not any(name.startswith("state_") for name in names)
    assert not any(name.startswith("observer.spatial") for name in names)


def test_explicit_arm_does_not_train_unused_raw_stack() -> None:
    model = _model(False)
    names = _set_trainable(model, stage="joint_alignment")
    assert any(name.startswith("observer.explicit") for name in names)
    assert any(name.startswith("state.correction") for name in names)
    assert not any(name.startswith("observer.raw") for name in names)
    assert not any(name.startswith("state.generator") for name in names)


def test_nonzero_raw_gate_propagates_gradient_to_tokenizer() -> None:
    torch.manual_seed(7)
    model = _model(True)
    _set_trainable(model, stage="observer_alignment")
    with torch.no_grad():
        model.observer.raw_gain.fill_(0.02)
    batch = {
        "explicit": torch.randn(2, 2, 13),
        "waveform": torch.randn(2, 2, 256),
        "sample_valid": torch.ones(2, 2, 256, dtype=torch.bool),
        "contact_mask": torch.ones(2, 2, dtype=torch.bool),
        "coordinates": torch.randn(2, 2, 3),
        "coordinate_valid": torch.ones(2, 2, dtype=torch.bool),
        "shaft_index": torch.zeros(2, 2, dtype=torch.long),
    }
    model.observation_embedding(batch).square().mean().backward()
    gradient = model.observer.raw.tokenizer.weight.grad
    assert gradient is not None and float(gradient.abs().sum()) > 0.0


def test_raw_epoch_zero_is_exact_paired_explicit_baseline(monkeypatch) -> None:
    model = _model(True)
    with torch.no_grad():
        model.observer.raw_gain.fill_(0.02)
    observed_gain = []

    class Design:
        subject = "epilepsiae_620"
        anchor_time = np.arange(10, dtype=np.float64)

        @staticmethod
        def anchor_ids(split):
            assert split == "train"
            return np.arange(10, dtype=np.int64)

    def fake_materialize(current, *args, **kwargs):
        observed_gain.append(float(current.observer.raw_gain.detach()))
        return np.zeros((10, 64), dtype=np.float32)

    def fake_evaluate(current, *args, **kwargs):
        # The paired gate-zero checkpoint is deliberately best.
        return SimpleNamespace(
            joint_nll_per_event=1.0 + float(current.observer.raw_gain.detach())
        )

    monkeypatch.setattr(r1_3_module, "materialize_embedding", fake_materialize)
    monkeypatch.setattr(r1_3_module, "evaluate_full_t1", fake_evaluate)
    monkeypatch.setattr(
        r1_3_module, "train_epoch", lambda *args, **kwargs: {
            "raw_tokenizer": 1.0,
            "raw_temporal_layer_0": 1.0,
            "raw_temporal_layer_1": 1.0,
        },
    )
    trace = fit_target_observer(
        model, Design(), object(), device="cpu",
        observer_epochs=1, joint_epochs=0, chunk_anchors=2,
    )
    assert observed_gain[0] == 0.0
    assert observed_gain[1] > 0.0
    assert trace.selected_total_epoch == 0
    assert float(model.observer.raw_gain.detach()) == 0.0
