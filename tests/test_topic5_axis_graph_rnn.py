import numpy as np
import pytest

from src.topic5_axis_graph_rnn import (
    AxisStructuredGraphRNN,
    structured_next_set_stop_loss,
)


def _inputs(torch, batch_size=2, n_contacts=6):
    axis = torch.linspace(-1.0, 1.0, n_contacts)
    forward = torch.zeros(n_contacts, n_contacts)
    for source in range(n_contacts - 1):
        forward[source + 1, source] = 1.0
    reverse = forward.T.clone()
    left = torch.zeros(n_contacts, dtype=torch.bool)
    right = torch.zeros(n_contacts, dtype=torch.bool)
    left[:2] = True
    right[-2:] = True
    return {
        "contact_features": torch.randn(batch_size, n_contacts, 8),
        "contact_mask": torch.ones(
            batch_size, n_contacts, dtype=torch.bool
        ),
        "group_ids": torch.tensor(
            [
                [0, 1, 2, -1, -1, -1],
                [1, 0, -1, 2, -1, -1],
            ][:batch_size]
        ),
        "group_count": torch.full((batch_size,), 3, dtype=torch.long),
        "local_offset": torch.zeros(n_contacts, 1),
        "axis_coordinate": axis,
        "forward_graph": forward,
        "reverse_graph": reverse,
        "left_endpoint": left,
        "right_endpoint": right,
    }


@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4])
def test_structured_ranks_have_declared_state_shape_and_finite_loss(rank):
    torch = pytest.importorskip("torch")
    inputs = _inputs(torch)
    model = AxisStructuredGraphRNN(8, structured_rank=rank)
    output = model(**inputs)
    assert output["contact_logits"].shape == (2, 4, 6)
    assert output["stop_logits"].shape == (2, 4)
    assert output["latent_state"].shape == (2, 4, 6, rank)
    assert output["inhibitory_state"].shape == (2, 4)
    loss = structured_next_set_stop_loss(
        output, inputs["group_ids"], inputs["group_count"]
    )
    assert torch.isfinite(loss["total"])
    loss["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_rank0_is_order_free_given_the_same_recruited_set():
    torch = pytest.importorskip("torch")
    inputs = _inputs(torch)
    inputs["contact_features"] = inputs["contact_features"][:1].expand(
        2, -1, -1
    ).clone()
    inputs["group_ids"] = torch.tensor(
        [[0, 1, 2, -1, -1, -1], [1, 0, 2, -1, -1, -1]]
    )
    model = AxisStructuredGraphRNN(8, structured_rank=0)
    output = model(**inputs)
    # At step two, both prefixes contain exactly contacts 0 and 1.
    torch.testing.assert_close(
        output["contact_logits"][0, 2],
        output["contact_logits"][1, 2],
    )
    torch.testing.assert_close(
        output["stop_logits"][0, 2],
        output["stop_logits"][1, 2],
    )


def test_rank2_is_invariant_to_axis_sign_and_template_label_swap():
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    inputs = _inputs(torch, batch_size=1)
    model = AxisStructuredGraphRNN(8, structured_rank=2)
    reference = model(**inputs)
    swapped = dict(inputs)
    swapped["axis_coordinate"] = -inputs["axis_coordinate"]
    swapped["forward_graph"] = inputs["reverse_graph"]
    swapped["reverse_graph"] = inputs["forward_graph"]
    swapped["left_endpoint"] = inputs["right_endpoint"]
    swapped["right_endpoint"] = inputs["left_endpoint"]
    changed = model(**swapped)
    torch.testing.assert_close(
        reference["contact_logits"], changed["contact_logits"]
    )
    torch.testing.assert_close(
        reference["stop_logits"], changed["stop_logits"]
    )
    torch.testing.assert_close(
        reference["latent_state"][:, :, :, 0],
        changed["latent_state"][:, :, :, 1],
    )
    torch.testing.assert_close(
        reference["latent_state"][:, :, :, 1],
        changed["latent_state"][:, :, :, 0],
    )


def test_direction_lesion_changes_the_propagating_rank2_state():
    torch = pytest.importorskip("torch")
    torch.manual_seed(11)
    inputs = _inputs(torch, batch_size=1)
    model = AxisStructuredGraphRNN(8, structured_rank=2)
    intact = model(**inputs)
    lesioned = model(**inputs, lesion="direction_forward")
    assert torch.any(
        torch.abs(
            intact["latent_state"][:, 1:, :, 0]
            - lesioned["latent_state"][:, 1:, :, 0]
        )
        > 1e-6
    )
    assert not torch.allclose(
        intact["contact_logits"][:, 1:],
        lesioned["contact_logits"][:, 1:],
    )
    # After observing contact 0 at rank 0, its outgoing graph must already
    # affect the prediction of contact 1 at rank 1 (no one-rank delay).
    assert not torch.allclose(
        intact["contact_logits"][:, 1, 1],
        lesioned["contact_logits"][:, 1, 1],
    )


def test_all_declared_transition_gains_obey_sign_and_stability_constraints():
    torch = pytest.importorskip("torch")
    model = AxisStructuredGraphRNN(8, structured_rank=4)
    assert torch.all((model.alpha_by_type > 0.0) & (model.alpha_by_type < 1.0))
    assert torch.all(model.input_gain_by_type >= 0.0)
    assert torch.all(model.propagation_gain_by_type >= 0.0)
    assert torch.all(model.decay_by_type >= 0.0)
    assert torch.all(model.inhibition_gain_by_type >= 0.0)
    assert torch.all(model.output_gain_by_type >= 0.0)
    assert 0.0 < float(model.inhibitory_alpha) < 1.0
    assert float(model.inhibitory_drive) >= 0.0
    assert float(model.direction_competition) >= 0.0


@pytest.mark.parametrize("rank", [0, 2, 4])
def test_structured_rollout_never_repeats_contacts_and_terminates(rank):
    torch = pytest.importorskip("torch")
    inputs = _inputs(torch, batch_size=1)
    model = AxisStructuredGraphRNN(8, structured_rank=rank)
    groups, counts = model.rollout(
        inputs["contact_features"],
        inputs["contact_mask"],
        inputs["local_offset"],
        inputs["axis_coordinate"],
        inputs["forward_graph"],
        inputs["reverse_graph"],
        inputs["left_endpoint"],
        inputs["right_endpoint"],
        n_events=96,
        seed=19,
        batch_size=48,
    )
    assert groups.shape == (96, 6)
    assert np.all(counts <= 6)
    for event, count in zip(groups, counts):
        used = event[event >= 0]
        assert len(used) == int(count)
        assert len(np.unique(used)) == int(count)
