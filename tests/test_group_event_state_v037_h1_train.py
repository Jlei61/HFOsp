from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from src.topic5_group_event_state.v037.ctssm import DualStreamEventCTSSM
from src.topic5_group_event_state.v037.h1_train import (
    EventStateComputer,
    GridEventStateComputer,
    NestedH1Readout,
    _equal_horizon_mean,
    _fixed_features,
)


def _readout() -> NestedH1Readout:
    widths = {
        "count": 1, "burden": 2, "community": 3, "coupling": 4,
        "mixture": 3, "embedding": 2, "mark": 5,
    }
    model = NestedH1Readout(2, 3, 4, 2, 3, widths, 2)
    with torch.no_grad():
        for module in model.q.values(): module.weight.zero_(); module.bias.zero_()
        for module in model.bmark.values(): module.weight.fill_(1.0)
        for module in model.state.values():
            module.weight.copy_(
                torch.arange(module.weight.numel(), dtype=module.weight.dtype).reshape_as(module.weight) / 10.0
            )
        for module in model.random.values(): module.weight.fill_(1.0)
    return model


def test_burden_heads_cannot_read_grammar_stream() -> None:
    model = _readout(); q = torch.zeros(1, 2); b = torch.zeros(1, 7)
    base = model.predict(q, bmark=b)
    changed = b.clone(); changed[:, 3:] = 9.0
    after = model.predict(q, bmark=changed)
    assert torch.equal(base["count"], after["count"])
    assert torch.equal(base["burden"], after["burden"])
    assert not torch.equal(base["community"], after["community"])


def test_grammar_heads_cannot_read_burden_stream() -> None:
    model = _readout(); q = torch.zeros(1, 2); b = torch.zeros(1, 7)
    base = model.predict(q, bmark=b)
    changed = b.clone(); changed[:, :3] = 9.0
    after = model.predict(q, bmark=changed)
    assert not torch.equal(base["count"], after["count"])
    assert torch.equal(base["community"], after["community"])


def test_state_streams_are_endpoint_separated() -> None:
    model = _readout(); q = torch.zeros(1, 2); state = torch.tensor([[1.0, -1.0, 0.0, 1.0, -1.0]])
    base = model.predict(q, state=state)
    grammar_changed = state.clone(); grammar_changed[:, 2:] = torch.tensor([8.0, -4.0, 2.0])
    after = model.predict(q, state=grammar_changed)
    assert torch.equal(base["count"], after["count"])
    assert not torch.equal(base["community"], after["community"])


def test_loss_weights_physical_horizons_equally_not_by_anchor_count() -> None:
    loss = torch.tensor([[1.0, 10.0], [1.0, 10.0], [1.0, 999.0]])
    mask = torch.tensor([[True, True], [True, True], [True, False]])
    value = _equal_horizon_mean(loss, mask)
    assert torch.isclose(value, torch.tensor(5.5))


def _grid_data(mark_at_boundary: float):
    rate = SimpleNamespace(
        anchor_time=np.asarray([300.0, 600.0]),
        segment=np.asarray([0, 0]),
        segment_bounds=np.asarray([[0.0, 900.0]]),
    )
    return SimpleNamespace(
        rate=rate,
        event_time=np.asarray([100.0, 299.0, 300.0, 599.0]),
        event_segment=np.asarray([0, 0, 0, 0]),
        burden_mark=np.asarray([[1.0], [2.0], [mark_at_boundary], [4.0]], dtype=np.float32),
        grammar_mark=np.ones((4, 1), dtype=np.float32),
    )


def test_hierarchical_grid_state_is_strictly_causal_at_bin_boundary() -> None:
    torch.manual_seed(3)
    model = DualStreamEventCTSSM(
        1, 1, taus_seconds=(600.0,), burden_channels_per_tau=1,
        grammar_channels_per_tau=1,
    )
    a = GridEventStateComputer(_grid_data(3.0), model, torch.device("cpu"))()
    b = GridEventStateComputer(_grid_data(30.0), model, torch.device("cpu"))()
    # The event exactly at 300 s is invisible to the 300 s query, but belongs
    # to the completed [300, 600) bin seen at 600 s.
    assert torch.equal(a[0], b[0])
    assert not torch.equal(a[1], b[1])


def test_hierarchical_grid_state_has_first_step_gradient() -> None:
    torch.manual_seed(4)
    model = DualStreamEventCTSSM(
        1, 1, taus_seconds=(600.0, 3600.0), burden_channels_per_tau=1,
        grammar_channels_per_tau=1,
    )
    value = GridEventStateComputer(_grid_data(3.0), model, torch.device("cpu"))()
    value[-1].square().sum().backward()
    assert model.burden_input.weight.grad is not None
    assert torch.isfinite(model.burden_input.weight.grad).all()
    assert float(model.burden_input.weight.grad.abs().sum()) > 0.0


def test_event_free_observed_segment_keeps_zero_history_rows_aligned() -> None:
    rate = SimpleNamespace(
        anchor_time=np.asarray([100.0, 200.0, 1000.0, 1100.0]),
        segment=np.asarray([0, 0, 1, 1]),
    )
    data = SimpleNamespace(
        rate=rate,
        event_time=np.asarray([50.0, 150.0]),
        event_segment=np.asarray([0, 0]),
        burden_mark=np.asarray([[1.0], [2.0]], dtype=np.float32),
        grammar_mark=np.asarray([[0.25], [0.75]], dtype=np.float32),
    )
    fixed = _fixed_features(data, torch.device("cpu"), (600.0, 3600.0))
    assert fixed.shape[0] == rate.anchor_time.size
    assert torch.equal(fixed[2:], torch.zeros_like(fixed[2:]))

    model = DualStreamEventCTSSM(
        1, 1, taus_seconds=(600.0, 3600.0), burden_channels_per_tau=1,
        grammar_channels_per_tau=1,
    )
    learned = EventStateComputer(data, model, torch.device("cpu"))()
    assert learned.shape[0] == rate.anchor_time.size
    assert torch.equal(learned[2:], torch.zeros_like(learned[2:]))
    learned[:2].square().sum().backward()
    assert float(model.burden_input.weight.grad.abs().sum()) > 0.0
