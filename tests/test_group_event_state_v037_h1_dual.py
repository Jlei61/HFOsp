from __future__ import annotations

import torch

from src.topic5_group_event_state.v037.h1_dual_train import NestedDualReadout


def _model() -> NestedDualReadout:
    widths = {"count": 1, "burden": 2, "community": 3, "coupling": 2,
              "mixture": 2, "embedding": 2, "mark": 3}
    model = NestedDualReadout(2, 2, 3, 4, 5, 2, 3, widths, 2)
    with torch.no_grad():
        for family in (model.q, model.bmark, model.background_current,
                       model.background_state, model.event, model.random):
            for layer in family.values():
                layer.weight.copy_(
                    torch.arange(layer.weight.numel(), dtype=layer.weight.dtype).reshape_as(layer.weight) / 10.0
                )
                if layer.bias is not None: layer.bias.zero_()
    return model


def test_dual_event_streams_remain_endpoint_separated() -> None:
    model = _model(); q = torch.zeros(2, 2)
    event = torch.tensor([[1.0, -1.0, 0.0, 1.0, -1.0], [-1.0, 1.0, 1.0, 0.0, -1.0]])
    base = model.predict(q, event_state=event)
    grammar_changed = event.clone(); grammar_changed[:, 2:] *= 4.0
    after = model.predict(q, event_state=grammar_changed)
    assert torch.equal(base["count"], after["count"])
    assert not torch.equal(base["community"], after["community"])


def test_background_stream_can_change_both_burden_and_grammar_heads() -> None:
    model = _model(); q = torch.zeros(2, 2)
    background = torch.tensor([[1.0, -1.0, 0.0, 2.0, -2.0], [-1.0, 1.0, 2.0, 0.0, -2.0]])
    base = model.predict(q)
    after = model.predict(q, background_state=background)
    assert not torch.equal(base["count"], after["count"])
    assert not torch.equal(base["community"], after["community"])
