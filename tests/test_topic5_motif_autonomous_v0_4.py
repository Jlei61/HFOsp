"""The autonomous objective is the only new load-bearing piece, so it is pinned here.

Two tests in the first version of this file were tautologies and are called out in
their replacements below: one built a shuffled target array and then only asserted its
shape, and one compared a function against itself with identical arguments.  Both
passed for reasons unrelated to what they claimed to check, which is worse than having
no test at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_dynamical_motif_rnn_v0_1 import MotifConfig, MotifRNN  # noqa: E402
from src.topic5_dynamical_motif_rollout_v0_1 import (  # noqa: E402
    DecoderContract,
    SizeHead,
    stochastic_rollout,
)
from src.topic5_shared_propagation_field import conditional_k_subset_log_prob  # noqa: E402
from src.topic5_motif_autonomous_v0_4 import (  # noqa: E402
    MASK_PENALTY,
    apply_warm_start,
    autonomous_calibration_trace,
    build_autonomous_event_tensors,
    geometry_axis_angle,
    rotated_axis_angles,
    shaft_axis_angle,
    autonomous_loss,
    autonomous_trace,
    cardinality_log_likelihood,
    kinematic_endpoint_extrapolation,
    soft_available_logits,
    refit_stop_head_on_autonomous_states,
    soft_rank_set,
    spatial_parameter_hash,
    subset_log_likelihood,
)

MODELS = ("DM0_ISOTROPIC", "DM1_FREE_AXIS", "DM2_LOCAL_DIRECTIONAL",
          "DM3_AXIS_FEEDFORWARD_TRANSIENT")


def toy_model(n_contacts: int = 10, model_id: str = "DM1_FREE_AXIS", seed: int = 0):
    coords = np.column_stack([np.arange(n_contacts) * 4.0,
                              np.zeros(n_contacts)]).astype(np.float32)
    config = MotifConfig(
        model_id=model_id, n_contacts=n_contacts, n_nodes=n_contacts,
        observation_operator=np.eye(n_contacts, dtype=np.float32), node_xy_mm=coords,
        local_mask=np.ones((n_contacts, n_contacts), dtype=np.uint8),
        r_forward_mm=8.0, sigma_s_mm=4.0, seed=seed)
    torch.manual_seed(seed)
    return MotifRNN(config), SizeHead(n_contacts), torch.as_tensor(coords)


def toy_batch(n_contacts: int = 10, batch: int = 6, prefix_len: int = 2, steps: int = 4):
    torch.manual_seed(1)
    prefix = torch.zeros(batch, prefix_len, n_contacts)
    targets = torch.zeros(batch, steps, n_contacts)
    for event in range(batch):
        order = torch.randperm(n_contacts)[: prefix_len + steps]
        for position in range(prefix_len):
            prefix[event, position, order[position]] = 1.0
        for position in range(steps):
            targets[event, position, order[prefix_len + position]] = 1.0
    cardinality = targets.sum(-1).long().clamp(min=1)
    valid = torch.ones(batch, steps, dtype=torch.bool)
    return prefix, targets, cardinality, valid


# -- the likelihood must be the sampler's law -------------------------------


@pytest.mark.parametrize("indices", ([0], [0, 2], [0, 2, 4]))
def test_subset_likelihood_matches_the_repository_law_bit_for_bit(indices):
    """A per-contact softmax sum equals the subset law only for singletons.

    Measured against the exact law, the first version of this module was 1.0 nats out
    at cardinality two and 2.8 at three — enough to change the training gradient on
    every multi-contact rank set.
    """
    logits = torch.tensor([[0.5, -0.2, 1.3, 0.0, 0.7]])
    mask = torch.ones(1, 5, dtype=torch.bool)
    target = torch.zeros(1, 5)
    target[0, indices] = 1.0
    cardinality = torch.tensor([len(indices)])
    mine = subset_log_likelihood(logits, target, mask, cardinality)
    exact = conditional_k_subset_log_prob(logits, target.bool(), mask,
                                          cardinality=cardinality)
    assert torch.allclose(mine, exact, atol=1e-12)


def test_a_per_contact_softmax_sum_would_disagree_at_cardinality_two():
    """Pins the size of the error the delegation avoids, so the reuse is not cosmetic."""
    logits = torch.tensor([[0.5, -0.2, 1.3, 0.0, 0.7]])
    mask = torch.ones(1, 5, dtype=torch.bool)
    target = torch.zeros(1, 5)
    target[0, [0, 2]] = 1.0
    naive = (torch.log_softmax(logits, dim=-1) * target).sum(dim=-1)
    exact = subset_log_likelihood(logits, target, mask, torch.tensor([2]))
    assert abs(float(naive) - float(exact)) > 0.5


# -- the true future must not enter the support -----------------------------


def test_changing_the_first_horizon_truth_does_not_move_the_second_horizon_score():
    """The replacement for a tautological leak test.

    The old version shuffled a target array and then asserted only its shape, which
    could not fail.  This holds the trace fixed and changes ONLY the first horizon's
    truth: if the loss advanced its support with the observed rank set, the second
    horizon's score would move.  It moved by 0.025 before the fix.
    """
    model, size_head, coords = toy_model(n_contacts=8, model_id="DM2_LOCAL_DIRECTIONAL")
    prefix = torch.zeros(3, 2, 8)
    prefix[:, 0, 0] = 1.0
    prefix[:, 1, 1] = 1.0
    trace = autonomous_trace(model, size_head, prefix, coords, horizons=3)

    def second_horizon(first_truth: int) -> float:
        targets = torch.zeros(3, 3, 8)
        targets[:, 0, first_truth] = 1.0
        targets[:, 1, 4] = 1.0
        targets[:, 2, 5] = 1.0
        cardinality = targets.sum(-1).long().clamp(min=1)
        _, detail = autonomous_loss(trace, targets, cardinality,
                                    torch.ones(3, 3, dtype=torch.bool))
        return detail["h2_nll"]

    assert second_horizon(2) == pytest.approx(second_horizon(3), abs=1e-9)


def test_the_prefix_is_excluded_from_every_future_set():
    """Prefix contacts are observed, so excluding them is knowledge, not leakage."""
    model, size_head, coords = toy_model(n_contacts=8)
    prefix = torch.zeros(2, 2, 8)
    prefix[:, 0, 3] = 1.0
    prefix[:, 1, 6] = 1.0
    trace = autonomous_trace(model, size_head, prefix, coords, horizons=3)
    recruited = trace["observed_recruited"][0]
    assert float(recruited[3]) == 1.0 and float(recruited[6]) == 1.0
    assert float(recruited.sum()) == 2.0


def test_the_rollout_never_receives_a_true_future_contact():
    model, size_head, coords = toy_model()
    prefix, _, _, _ = toy_batch()
    with torch.no_grad():
        first = autonomous_trace(model, size_head, prefix, coords, horizons=4)
        second = autonomous_trace(model, size_head, prefix, coords, horizons=4)
    assert torch.equal(first["contact_logits"], second["contact_logits"])


def test_the_prefix_does_change_the_rollout():
    model, size_head, coords = toy_model()
    prefix, _, _, _ = toy_batch()
    with torch.no_grad():
        base = autonomous_trace(model, size_head, prefix, coords, horizons=4)
        moved = autonomous_trace(model, size_head, prefix.flip(dims=[1]), coords,
                                 horizons=4)
    assert not torch.allclose(base["contact_logits"], moved["contact_logits"])


# -- the feedback must be a legal rank set ----------------------------------


def test_soft_rank_set_entries_are_probabilities():
    """``probability * expected_size`` reached 2.9 on a single contact; this cannot."""
    model, size_head, coords = toy_model()
    prefix, _, _, _ = toy_batch()
    trace = autonomous_trace(model, size_head, prefix, coords, horizons=4)
    soft = trace["soft_rank_sets"]
    assert float(soft.min()) >= -1e-6
    assert float(soft.max()) <= 1.0 + 1e-6


def test_soft_rank_set_total_mass_is_the_expected_cardinality():
    torch.manual_seed(0)
    logits = torch.randn(4, 9)
    available = torch.ones(4, 9, dtype=torch.bool)
    size_logits = torch.randn(4, 4)
    soft = soft_rank_set(logits, available, size_logits, kmax=4)
    expected = (torch.softmax(size_logits, dim=-1)
                @ torch.arange(1, 5, dtype=torch.float32))
    assert torch.allclose(soft.sum(dim=-1), expected, atol=1e-4)


def test_soft_rank_set_puts_no_mass_on_the_observed_prefix():
    torch.manual_seed(0)
    logits = torch.randn(2, 6)
    available = torch.ones(2, 6, dtype=torch.bool)
    available[:, [1, 4]] = False
    soft = soft_rank_set(logits, available, torch.randn(2, 3), kmax=3)
    assert float(soft[:, [1, 4]].abs().max()) < 1e-6


# -- the prefix must be encoded exactly as the sampler encodes it ------------


@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize("prefix_len", (1, 2, 3))
def test_prefix_encoding_matches_the_stochastic_rollout(model_id, prefix_len):
    """Training and evaluation must share one direction-gate history.

    The sampler's order is: gate from the current displacement, advance the state with
    the pending input, THEN load the next observed set and update the displacement.
    Doing the step and the update in the same iteration left the two apart by 0.18 in
    logit units for the directed motif.  The loop below is transcribed from
    ``stochastic_rollout`` so that changing either encoder breaks this test.
    """
    from src.topic5_dynamical_motif_rnn_v0_1 import rollout_displacement_update
    from src.topic5_dynamical_motif_rollout_v0_1 import _direction_weight

    n_contacts = 8
    model, size_head, coords = toy_model(n_contacts=n_contacts, model_id=model_id)
    prefix = torch.zeros(2, prefix_len, n_contacts)
    for event in range(2):
        for position in range(prefix_len):
            prefix[event, position, (event + 2 * position) % n_contacts] = 1.0

    with torch.no_grad():
        # --- the sampler's encoder, transcribed ---
        state = torch.zeros(2, model.n_nodes)
        terms = model.recurrent_terms()
        unit, _ = model.axis_unit()
        counts = prefix.sum(-1, keepdim=True).clamp_min(1.0)
        pending = prefix[:, 0].clone()
        centroid_start = prefix[:, 0] @ coords / counts[:, 0]
        displacement = torch.zeros(2, 2)
        for position in range(1, prefix_len):
            gate = model.direction_gate(displacement, unit)
            state = model.step(state, pending, gate, terms,
                               _direction_weight(model, displacement, gate, unit))
            pending = prefix[:, position]
            centroid_now = (prefix[:, position] @ coords) / counts[:, position]
            displacement = rollout_displacement_update(
                displacement, centroid_start, centroid_now, position, "M2-2RANK")
        gate = model.direction_gate(displacement, unit)
        state = model.step(state, pending, gate, terms,
                           _direction_weight(model, displacement, gate, unit))
        reference = model.readout(state)

        # --- what training does; the mask penalty is added back to compare the raw
        # read-out, since nothing is recruited by the model before the first step ---
        trace = autonomous_trace(model, size_head, prefix, coords, horizons=1)
        mine = (trace["contact_logits"][:, 0]
                + MASK_PENALTY * trace["predicted_recruited_before"][:, 0])

    assert torch.allclose(mine, reference, atol=1e-6), (
        f"training and sampling encode the prefix differently: "
        f"max |delta| = {float((mine - reference).abs().max()):.4f}")


# -- the loss must be differentiable through the whole rollout --------------


def test_gradient_from_a_later_horizon_reaches_the_earlier_feedback():
    """A detached loop would train only the last step and look fine otherwise."""
    model, size_head, coords = toy_model()
    prefix, targets, cardinality, valid = toy_batch(steps=3)
    trace = autonomous_trace(model, size_head, prefix, coords, horizons=3)
    third_only, _ = autonomous_loss(trace, targets, cardinality, valid, horizons=(3,))
    third_only.backward()
    moved = [name for name, parameter in model.named_parameters()
             if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0]
    assert moved, "the third horizon's loss did not reach any operator parameter"


def test_loss_weights_only_the_primary_horizons_but_reports_all():
    model, size_head, coords = toy_model()
    prefix, targets, cardinality, valid = toy_batch(steps=5)
    trace = autonomous_trace(model, size_head, prefix, coords, horizons=5)
    total, detail = autonomous_loss(trace, targets, cardinality, valid)
    assert set(detail) == {f"h{h}_nll" for h in range(1, 6)}
    assert torch.isfinite(total)
    primary = sum(detail[f"h{h}_nll"] for h in (1, 2, 3)) / 3.0
    assert float(total) == pytest.approx(primary, rel=1e-5)


# -- the size head owns cardinality, the operator owns which contacts -------


def test_cardinality_term_moves_with_the_size_head_and_the_spatial_term_does_not():
    """The replacement for a test that compared a call against itself.

    Changing only the size logits must move the cardinality term and leave the spatial
    term untouched, otherwise a model could buy the spatial comparison by guessing how
    many contacts fire.
    """
    logits = torch.randn(4, 8)
    mask = torch.ones(4, 8, dtype=torch.bool)
    target = torch.zeros(4, 8)
    target[torch.arange(4), torch.randint(0, 8, (4,))] = 1.0
    cardinality = torch.ones(4, dtype=torch.long)

    spatial_before = subset_log_likelihood(logits, target, mask, cardinality)
    quiet = torch.zeros(4, 3)
    loud = torch.zeros(4, 3)
    loud[:, 0] = 5.0
    assert float(cardinality_log_likelihood(loud, cardinality - 1).mean()) > \
        float(cardinality_log_likelihood(quiet, cardinality - 1).mean())
    spatial_after = subset_log_likelihood(logits, target, mask, cardinality)
    assert torch.allclose(spatial_before, spatial_after)


def test_soft_mask_matches_the_hard_mask_in_the_one_hot_limit():
    logits = torch.tensor([[0.5, -0.2, 1.3, 0.0]])
    recruited = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    soft = torch.softmax(soft_available_logits(logits, recruited), dim=-1)
    hard = torch.softmax(logits.masked_fill(recruited > 0.5, -1e9), dim=-1)
    assert torch.allclose(soft, hard, atol=1e-9)


# -- the control the directed-transport motif has to beat -------------------


def test_kinematic_extrapolation_continues_the_last_step():
    centroids = np.array([[[0.0, 0.0], [2.0, 1.0]]])
    assert kinematic_endpoint_extrapolation(centroids, gain=1.0)[0] == pytest.approx(
        [4.0, 2.0])


def test_kinematic_extrapolation_needs_two_prefix_steps():
    with pytest.raises(ValueError):
        kinematic_endpoint_extrapolation(np.zeros((3, 1, 2)))


# -- calibration must happen on autonomous states, not teacher-forced ones ---


def test_the_calibration_trace_does_not_read_the_true_future():
    """Perturbing the truth beyond the prefix must leave every state untouched.

    The states, the logits and the support are the model's own; only the scoring
    masks may depend on the truth.  The companion test below shows the two support
    constructions genuinely disagree here, so this equality is not vacuous.
    """
    model, size_head, coords = toy_model()
    prefix, targets, _, valid = toy_batch()
    original = autonomous_calibration_trace(model, size_head, prefix, targets, valid,
                                            coords)

    shuffled = targets.clone()
    shuffled[:, 1:] = targets[:, 1:].flip(0)  # a different true future, same prefix
    perturbed = autonomous_calibration_trace(model, size_head, prefix, shuffled, valid,
                                             coords)

    for key in ("features", "contact_logits", "stop_logits", "available"):
        assert torch.equal(original[key], perturbed[key]), key


def test_the_autonomous_support_differs_from_the_teacher_forced_support():
    """Without this, the leak test above could pass on data where truth changes nothing.

    Teacher forcing would remove the true earlier contacts from the later support; the
    autonomous support removes only what the model itself predicted.
    """
    model, size_head, coords = toy_model()
    prefix, targets, _, valid = toy_batch()
    trace = autonomous_calibration_trace(model, size_head, prefix, targets, valid,
                                         coords)

    seen = torch.cumsum(targets, dim=1) - targets  # truth recruited strictly before t
    teacher_forced = (prefix.sum(1)[:, None, :] < 0.5) & (seen < 0.5)
    assert not torch.equal(trace["available"], teacher_forced)
    # and the disagreement is in the expected direction: truth-only exclusions are gone
    assert int((trace["available"] & ~teacher_forced).sum()) > 0


def test_the_stop_refit_leaves_the_spatial_operator_untouched():
    model, size_head, coords = toy_model()
    prefix, targets, _, valid = toy_batch()
    trace = autonomous_calibration_trace(model, size_head, prefix, targets, valid,
                                         coords)
    before = spatial_parameter_hash(model)
    refit_stop_head_on_autonomous_states(model, trace, max_epochs=20)
    assert spatial_parameter_hash(model) == before


def test_the_stop_refit_does_move_the_termination_head():
    """Guards the freeze from being total — a refit that changes nothing is not a refit."""
    model, size_head, coords = toy_model()
    prefix, targets, _, valid = toy_batch()
    trace = autonomous_calibration_trace(model, size_head, prefix, targets, valid,
                                         coords)
    head_before = [p.detach().clone() for n, p in model.named_parameters()
                   if n.startswith("stop_head")]
    report = refit_stop_head_on_autonomous_states(model, trace, max_epochs=100)
    head_after = [p.detach() for n, p in model.named_parameters()
                  if n.startswith("stop_head")]
    assert any(not torch.equal(a, b) for a, b in zip(head_before, head_after))
    assert report["stop_bce_after"] < report["stop_bce_before"]


def test_the_refit_restores_gradients_on_the_spatial_parameters():
    """A leaked ``requires_grad_(False)`` would silently freeze later training."""
    model, size_head, coords = toy_model()
    prefix, targets, _, valid = toy_batch()
    trace = autonomous_calibration_trace(model, size_head, prefix, targets, valid,
                                         coords)
    refit_stop_head_on_autonomous_states(model, trace, max_epochs=5)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_the_calibration_trace_carries_the_schema_the_calibrators_expect():
    model, size_head, coords = toy_model()
    prefix, targets, _, valid = toy_batch()
    trace = autonomous_calibration_trace(model, size_head, prefix, targets, valid,
                                         coords)
    for key in ("features", "contact_logits", "stop_logits", "target", "available",
                "predict", "is_last", "valid"):
        assert key in trace, key
    assert trace["features"].shape[-1] == 4
    assert trace["available"].dtype == torch.bool
    # the last scored step of an event is a STOP decision, never a contact prediction
    assert not bool((trace["predict"] & trace["is_last"]).any())


# -- training and generation must walk the same path ------------------------


def neutral_contract() -> DecoderContract:
    """Temperatures of one and empty diagnostics.

    The hidden state up to the first readout is computed before any temperature is
    applied, so these values cannot affect what this test compares.
    """
    return DecoderContract(
        contact_temperature=1.0, cardinality_temperature=1.0, stop_temperature=1.0,
        n_calibration_decisions=0, n_calibration_events=0,
        contact_nll_before=0.0, contact_nll_after=0.0,
        cardinality_nll_before=0.0, cardinality_nll_after=0.0,
        stop_bce_before=0.0, stop_bce_after=0.0,
        size_head_train_decisions=0, size_head_validation_nll=0.0)


def record_states(model, run) -> list[torch.Tensor]:
    """Run ``run()`` with a spy on ``readout`` and return the states it saw."""
    seen: list[torch.Tensor] = []
    original = model.readout

    def spy(state):
        seen.append(state.detach().clone())
        return original(state)

    model.readout = spy
    try:
        run()
    finally:
        del model.readout
    return seen


@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize("prefix_len", (1, 2, 3))
def test_the_first_generated_state_is_bit_identical_to_the_samplers(model_id, prefix_len):
    """Calls ``stochastic_rollout`` itself rather than transcribing its loop.

    Up to the first generated step neither path has fed anything back, so the two
    hidden states must agree exactly.  Any divergence in how the prefix is encoded —
    the order of the step and the displacement update, or which direction-weight helper
    is used — shows up here, and it is what makes training and evaluation one task.
    """
    model, size_head, coords = toy_model(model_id=model_id)
    prefix, _, _, _ = toy_batch(prefix_len=prefix_len)

    sampler = record_states(model, lambda: stochastic_rollout(
        model, size_head, neutral_contract(), prefix, coords.numpy(),
        torch.device("cpu"), mode="FIXED_H", horizon=1))
    mine = record_states(model, lambda: autonomous_trace(
        model, size_head, prefix, coords, horizons=1))

    assert torch.equal(mine[0], sampler[0])


# -- the prefix/future split ------------------------------------------------


def toy_ranks() -> tuple[np.ndarray, np.ndarray]:
    """Four events of length 6, 5, 3 and 2 on a ten-contact line."""
    ranks = -np.ones((4, 10), dtype=int)
    for event, length in enumerate((6, 5, 3, 2)):
        ranks[event, :length] = np.arange(length)
    coords = np.column_stack([np.arange(10) * 4.0, np.zeros(10)]).astype(np.float32)
    return ranks, coords


def test_events_too_short_to_have_a_future_are_dropped_and_counted():
    ranks, coords = toy_ranks()
    built = build_autonomous_event_tensors(ranks, coords, prefix_len=3, horizons=3)
    # lengths 6 and 5 survive a three-set prefix; 3 and 2 have nothing left to predict
    assert built["n_events_kept"] == 2
    assert built["n_events_too_short"] == 2
    assert list(built["event_index"]) == [0, 1]


def test_the_horizon_mask_marks_where_each_event_actually_ended():
    ranks, coords = toy_ranks()
    built = build_autonomous_event_tensors(ranks, coords, prefix_len=3, horizons=3)
    # event 0 has length 6: rank sets 3, 4, 5 exist -> three valid horizons
    # event 1 has length 5: rank sets 3, 4 exist    -> two valid horizons
    assert built["valid"].tolist() == [[True, True, True], [True, True, False]]
    assert built["horizon_coverage"] == [2, 2, 1]


def test_the_prefix_and_the_targets_do_not_overlap():
    ranks, coords = toy_ranks()
    built = build_autonomous_event_tensors(ranks, coords, prefix_len=2, horizons=3)
    overlap = built["prefix"].sum(1)[:, None, :] * built["targets"]
    assert float(overlap.sum()) == 0.0


def test_padding_beyond_an_event_carries_no_target_mass():
    ranks, coords = toy_ranks()
    built = build_autonomous_event_tensors(ranks, coords, prefix_len=3, horizons=3)
    assert float(built["targets"][1, 2].sum()) == 0.0
    assert int(built["cardinality"][1, 2]) == 0


def test_a_horizon_past_the_end_of_a_short_event_does_not_raise():
    """The padded steps of a short event must be skipped, not scored.

    The first version clamped their cardinality to one while their target stayed empty
    and relied on a mask applied afterwards; the exact subset law checks that pairing
    before the mask is reached and refused the batch.  Every toy event used above has a
    full-length future, which is why the fixtures alone could not catch it.
    """
    model, size_head, coords = toy_model()
    prefix, targets, cardinality, valid = toy_batch(steps=4)
    targets[0, 2:] = 0.0           # this event ended after two more rank sets
    cardinality[0, 2:] = 0
    valid[0, 2:] = False

    trace = autonomous_trace(model, size_head, prefix, coords, horizons=4)
    total, detail = autonomous_loss(trace, targets, cardinality, valid,
                                    horizons=(1, 2, 3))
    assert torch.isfinite(total)
    assert all(np.isfinite(value) for value in detail.values())


def test_an_all_empty_horizon_contributes_nothing_rather_than_failing():
    model, size_head, coords = toy_model()
    prefix, targets, cardinality, valid = toy_batch(steps=4)
    targets[:, 3] = 0.0
    cardinality[:, 3] = 0
    valid[:, 3] = False
    trace = autonomous_trace(model, size_head, prefix, coords, horizons=4)
    _, detail = autonomous_loss(trace, targets, cardinality, valid, horizons=(1, 2, 3))
    assert detail["h4_nll"] == 0.0


def test_the_calibration_trace_is_free_of_the_training_graph():
    """Every decoder fit calls backward repeatedly on these states.

    The first version left the graph attached and the STOP refit was patched locally;
    the size-head fit then hit the same error on the first real patient, eleven minutes
    into the run.  Detaching belongs at the trace, which is where every consumer reads.
    """
    model, size_head, coords = toy_model()
    prefix, targets, _, valid = toy_batch()
    trace = autonomous_calibration_trace(model, size_head, prefix, targets, valid,
                                         coords)
    for key in ("features", "contact_logits", "stop_logits"):
        assert trace[key].grad_fn is None, key
        assert not trace[key].requires_grad, key


def test_a_decoder_fit_can_backward_more_than_once_on_the_trace():
    """The property the detach exists for, exercised end to end."""
    model, size_head, coords = toy_model()
    prefix, targets, _, valid = toy_batch()
    trace = autonomous_calibration_trace(model, size_head, prefix, targets, valid,
                                         coords)
    report = refit_stop_head_on_autonomous_states(model, trace, max_epochs=3)
    assert report["n_decisions"] > 0

    head = SizeHead(10)
    optimiser = torch.optim.Adam(head.parameters(), lr=1e-2)
    rows = trace["predict"]
    x = trace["features"][rows]
    y = (trace["target"].sum(-1).long()[rows] - 1).clamp_min(0)
    for _ in range(3):
        loss = torch.nn.functional.cross_entropy(head(x), y)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()


# -- the warm start must not eat the optimisation start ---------------------


def test_the_axis_angle_survives_inheriting_the_parent():
    """The bug this helper exists for, at the value that made it invisible.

    A parent with no axis still stores ``theta = 0``; inheriting it after the angle was
    set returned every start to zero, so three angles became one run counted three
    times and the reported spread came entirely from something else.
    """
    parent, _, _ = toy_model(model_id="DM0_ISOTROPIC")
    state = {k: v.detach().clone() for k, v in parent.state_dict().items()}
    for angle in (0.0, np.pi / 3.0, 2.0 * np.pi / 3.0):
        child, _, _ = toy_model(model_id="DM1_FREE_AXIS")
        with torch.no_grad():
            child.theta.fill_(angle)
        apply_warm_start(child, state, ("theta", "eta_raw"), theta_init=angle)
        assert float(child.theta) == pytest.approx(angle)
        assert float(child.eta_raw) == 0.0


def test_different_angles_give_different_models_after_the_warm_start():
    """The property that matters downstream: the starts are actually distinct."""
    parent, _, _ = toy_model(model_id="DM0_ISOTROPIC")
    state = {k: v.detach().clone() for k, v in parent.state_dict().items()}
    hashes = set()
    for angle in (0.0, np.pi / 3.0, 2.0 * np.pi / 3.0):
        child, _, _ = toy_model(model_id="DM1_FREE_AXIS")
        apply_warm_start(child, state, ("theta", "eta_raw"), theta_init=angle)
        hashes.add(spatial_parameter_hash(child))
    assert len(hashes) == 3


def test_the_restored_angle_still_reproduces_the_parent_exactly():
    """Zero anisotropy must make the angle inert, or the chain is not nested."""
    parent, parent_head, coords = toy_model(model_id="DM0_ISOTROPIC")
    prefix, _, _, _ = toy_batch()
    state = {k: v.detach().clone() for k, v in parent.state_dict().items()}
    reference = autonomous_trace(parent, parent_head, prefix, coords, horizons=1)

    for angle in (0.0, np.pi / 3.0, 2.0 * np.pi / 3.0):
        child, child_head, _ = toy_model(model_id="DM1_FREE_AXIS")
        apply_warm_start(child, state, ("theta", "eta_raw"), theta_init=angle)
        child_head.load_state_dict(parent_head.state_dict())
        mine = autonomous_trace(child, child_head, prefix, coords, horizons=1)
        gap = float((mine["contact_logits"][:, 0]
                     - reference["contact_logits"][:, 0]).abs().max())
        assert gap < 1e-5, f"angle {angle} is not inert at zero anisotropy: {gap:.3e}"


# -- the axes the learned corridor has to beat ------------------------------


def test_the_geometry_axis_finds_the_direction_the_contacts_lie_along():
    for angle in (0.0, np.pi / 6.0, np.pi / 2.0, 2.0 * np.pi / 3.0):
        along = np.array([[t * np.cos(angle), t * np.sin(angle)]
                          for t in np.linspace(-10, 10, 11)])
        assert geometry_axis_angle(along) == pytest.approx(angle % np.pi, abs=1e-6)


def test_an_axis_is_taken_modulo_pi_so_a_flip_is_the_same_axis():
    points = np.array([[t, 0.3 * t] for t in np.linspace(-5, 5, 9)])
    assert geometry_axis_angle(points) == pytest.approx(
        geometry_axis_angle(-points), abs=1e-9)


def test_the_shaft_axis_is_the_arrangement_of_shafts_not_of_contacts():
    """Contacts run along x within each shaft; the shafts are stacked along y."""
    coordinates, shafts = [], []
    for shaft, y in enumerate((0.0, 10.0, 20.0)):
        for x in (0.0, 2.0, 4.0, 6.0):
            coordinates.append([x, y])
            shafts.append(f"S{shaft}")
    coordinates = np.array(coordinates)
    assert geometry_axis_angle(coordinates) == pytest.approx(np.pi / 2, abs=0.35)
    assert shaft_axis_angle(coordinates, shafts) == pytest.approx(np.pi / 2, abs=1e-6)


def test_a_single_shaft_implant_has_no_shaft_axis_and_says_so():
    """Falling back to the contact axis would report a control that was never run."""
    coordinates = np.array([[x, 0.0] for x in range(6)], dtype=float)
    with pytest.raises(ValueError, match="at least two shafts"):
        shaft_axis_angle(coordinates, ["S0"] * 6)


def test_the_rotated_axes_are_distinct_from_the_axis_they_rotate():
    base = 0.4
    rotated = rotated_axis_angles(base)
    assert len(set(rotated)) == 2
    assert all(abs(angle - base) > 1e-3 for angle in rotated)
    assert all(0.0 <= angle < np.pi for angle in rotated)


def test_rotating_by_pi_returns_the_same_axis():
    """The modulo is real: half a turn is not a different corridor."""
    assert rotated_axis_angles(0.4, offsets=(np.pi,))[0] == pytest.approx(0.4)
