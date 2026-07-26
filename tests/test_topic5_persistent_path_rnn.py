import numpy as np
import pytest

from src.topic5_persistent_path_rnn import (
    PersistentPathModeRNN,
    persistent_mixture_loss,
)


def _inputs(torch, batch_size=2, n_contacts=5, n_components=4):
    features = torch.randn(batch_size, n_contacts, 4)
    features[:] = features[:1]
    mask = torch.ones(batch_size, n_contacts, dtype=torch.bool)
    groups = torch.tensor(
        [
            [0, 1, 2, 3, -1],
            [0, 1, 3, 2, -1],
        ][:batch_size],
        dtype=torch.long,
    )
    count = torch.full((batch_size,), 4, dtype=torch.long)
    graphs = torch.zeros(n_components, n_contacts, n_contacts)
    for component in range(n_components):
        if component % 2 == 0:
            for source in range(n_contacts - 1):
                graphs[component, source + 1, source] = 1.0
        else:
            for source in range(1, n_contacts):
                graphs[component, source - 1, source] = 1.0
    prior = torch.full((n_components,), 1.0 / n_components)
    left = torch.zeros(n_contacts, dtype=torch.bool)
    right = torch.zeros(n_contacts, dtype=torch.bool)
    left[0] = True
    right[-1] = True
    return {
        "contact_features": features,
        "contact_mask": mask,
        "group_ids": groups,
        "group_count": count,
        "local_offset": torch.zeros(n_contacts, 1),
        "component_graphs": graphs,
        "component_prior": prior,
        "left_endpoint": left,
        "right_endpoint": right,
    }


def test_persistent_model_shapes_loss_and_gradient():
    torch = pytest.importorskip("torch")
    torch.manual_seed(3)
    inputs = _inputs(torch)
    model = PersistentPathModeRNN(4)
    output = model(**inputs)
    assert output["component_contact_logits"].shape == (2, 4, 5, 5)
    assert output["component_stop_logits"].shape == (2, 4, 5)
    assert output["latent_state"].shape == (2, 4, 5, 5)
    loss = persistent_mixture_loss(
        output, inputs["group_ids"], inputs["group_count"]
    )
    assert loss["predictive_action_probability"].shape == (2, 5, 6)
    assert loss["component_posterior_trajectory"].shape == (2, 5, 4)
    torch.testing.assert_close(
        loss["final_component_posterior"].sum(1),
        torch.ones(2),
    )
    assert torch.isfinite(loss["total"])
    loss["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_prefix_prediction_does_not_read_future_ranks():
    torch = pytest.importorskip("torch")
    torch.manual_seed(5)
    inputs = _inputs(torch)
    model = PersistentPathModeRNN(4)
    output = model(**inputs)
    loss = persistent_mixture_loss(
        output, inputs["group_ids"], inputs["group_count"]
    )
    probability = loss["predictive_action_probability"]
    # Events share ranks 0 and 1, so the prediction before rank 2 is equal.
    torch.testing.assert_close(probability[0, :3], probability[1, :3])
    # After different rank-2 contacts are observed, the next prediction differs.
    assert not torch.allclose(probability[0, 3], probability[1, 3])


def test_component_label_and_axis_direction_swap_is_marginally_invariant():
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    inputs = _inputs(torch, batch_size=1, n_components=2)
    model = PersistentPathModeRNN(4)
    reference_output = model(**inputs)
    reference = persistent_mixture_loss(
        reference_output, inputs["group_ids"], inputs["group_count"]
    )
    swapped = dict(inputs)
    swapped["component_graphs"] = inputs["component_graphs"].flip(0)
    swapped["component_prior"] = inputs["component_prior"].flip(0)
    changed_output = model(**swapped)
    changed = persistent_mixture_loss(
        changed_output, inputs["group_ids"], inputs["group_count"]
    )
    torch.testing.assert_close(
        reference["predictive_action_probability"],
        changed["predictive_action_probability"],
    )
    torch.testing.assert_close(reference["event_nll"], changed["event_nll"])


def test_no_history_control_ignores_graph_values():
    torch = pytest.importorskip("torch")
    inputs = _inputs(torch, batch_size=1)
    model = PersistentPathModeRNN(4, use_recurrence=False)
    reference = model(**inputs)
    changed = dict(inputs)
    changed["component_graphs"] = torch.rand_like(
        inputs["component_graphs"]
    )
    output = model(**changed)
    torch.testing.assert_close(
        reference["component_contact_logits"],
        output["component_contact_logits"],
    )
    torch.testing.assert_close(
        reference["component_stop_logits"], output["component_stop_logits"]
    )


def test_rollout_selects_one_component_for_each_complete_event():
    torch = pytest.importorskip("torch")
    inputs = _inputs(torch, batch_size=1)
    model = PersistentPathModeRNN(4)
    groups, counts, components = model.rollout(
        inputs["contact_features"],
        inputs["contact_mask"],
        inputs["local_offset"],
        inputs["component_graphs"],
        inputs["component_prior"],
        inputs["left_endpoint"],
        inputs["right_endpoint"],
        n_events=64,
        seed=17,
        batch_size=32,
    )
    assert groups.shape == (64, 5)
    assert counts.shape == (64,)
    assert components.shape == (64,)
    assert np.all((components >= 0) & (components < 4))
    for event, count in zip(groups, counts):
        ranks = event[event >= 0]
        assert len(ranks) == int(count)
        assert len(np.unique(ranks)) == int(count)
