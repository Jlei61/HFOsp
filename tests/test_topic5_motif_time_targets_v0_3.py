"""The time target is what this round rests on, so it is tested before anything runs.

The failure that would be invisible otherwise: a time target that is really just the
step index in disguise.  If that happened, every motif would score the same and the
round would repeat v0.1's null for a different reason.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_motif_time_targets_v0_3 import (  # noqa: E402
    TIME_BASELINES,
    RankOnlyTimeBaseline,
    TimeHead,
    adjacent_distance_time_relation,
    time_baseline_scores,
    build_event_tensors_with_time,
    build_time_targets,
    direction_persistence,
    distance_time_relation,
    recruited_field,
)


def toy_event(n_contacts: int = 6, n_events: int = 40, seed: int = 0):
    """Events that march along a line, with the lag growing with the distance walked."""
    rng = np.random.default_rng(seed)
    coords = np.column_stack([np.arange(n_contacts) * 5.0, np.zeros(n_contacts)])
    ranks = np.full((n_events, n_contacts), -1, dtype=np.int16)
    lag = np.zeros((n_events, n_contacts))
    for event in range(n_events):
        start = int(rng.integers(0, 2))
        # variable stride, so a step can be short or long and the increment is not a
        # function of the step index alone
        stride = rng.integers(1, 3, size=4)
        order, position = [], start
        for hop in stride:
            if position >= n_contacts:
                break
            order.append(int(position))
            position += int(hop)
        if len(order) < 3:
            order = list(range(start, min(start + 3, n_contacts)))
        clock = 0.0
        for step, contact in enumerate(order):
            ranks[event, contact] = step
            if step:
                clock += abs(coords[contact, 0] - coords[order[step - 1], 0]) / 10.0
            lag[event, contact] = clock
    return ranks, coords, lag


# -- the target itself ------------------------------------------------------


def test_time_increment_is_the_gap_between_consecutive_rank_sets():
    ranks = np.array([[0, 1, 2, -1]], dtype=np.int16)
    lag = np.array([[0.0, 2.0, 5.0, np.nan]])
    out = build_time_targets(ranks, lag)
    assert out["time_valid"][0, :2].all()
    assert not out["time_valid"][0, 2]
    assert out["time_delta"][0, 0] == pytest.approx(2.0)
    assert out["time_delta"][0, 1] == pytest.approx(3.0)


def test_steps_with_no_finite_lag_are_masked_not_imputed():
    """An imputed zero is indistinguishable from a genuinely fast step."""
    ranks = np.array([[0, 1, 2, -1]], dtype=np.int16)
    lag = np.array([[0.0, np.nan, 5.0, np.nan]])
    out = build_time_targets(ranks, lag)
    assert not bool(out["time_valid"][0, 0])
    assert not bool(out["time_valid"][0, 1])
    assert out["time_delta"][0, 0] == 0.0  # masked, so the stored value is inert


def test_time_target_is_not_just_the_step_index():
    """If the increment were a function of the step, the motifs could not differ on it."""
    ranks, coords, lag = toy_event()
    out = build_time_targets(ranks, lag)
    delta = out["time_delta"].numpy()
    valid = out["time_valid"].numpy()
    spread = [delta[:, step][valid[:, step]].std() for step in range(delta.shape[1])
              if valid[:, step].sum() > 3]
    assert spread and max(spread) > 1e-6


def test_tensors_stay_index_aligned_with_the_v0_1_builder():
    ranks, coords, lag = toy_event()
    tensors = build_event_tensors_with_time(ranks, coords, lag)
    assert tensors["time_delta"].shape == tensors["valid"].shape
    assert tensors["time_valid"].shape == tensors["valid"].shape
    # a time step can only be valid where the event itself is still running
    assert bool((~tensors["valid"] & tensors["time_valid"]).sum() == 0)


# -- the head and its reference --------------------------------------------


def test_time_head_owns_two_scalars_so_the_operator_does_the_work():
    head = TimeHead()
    assert sum(p.numel() for p in head.parameters()) == 2


def test_time_head_prefers_the_field_that_orders_the_steps_correctly():
    """A field that tracks the true delay must score better than a scrambled one."""
    torch.manual_seed(0)
    target = torch.rand(64, 5) * 3.0
    mask = torch.ones_like(target, dtype=torch.bool)
    informative = -torch.log1p(target)
    scrambled = informative[torch.randperm(informative.shape[0])]

    def best_nll(field):
        head = TimeHead()
        optimiser = torch.optim.Adam(head.parameters(), lr=0.05)
        for _ in range(400):
            optimiser.zero_grad()
            loss = head.nll(field, target, mask)
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            return float(head.nll(field, target, mask))

    assert best_nll(informative) < best_nll(scrambled) - 0.01


def test_rank_only_baseline_cannot_use_anything_but_the_step_index():
    """Two batches with identical step structure must get identical predictions."""
    torch.manual_seed(0)
    baseline = RankOnlyTimeBaseline(max_steps=5)
    target = torch.rand(16, 5)
    mask = torch.ones_like(target, dtype=torch.bool)
    shuffled_rows = target[torch.randperm(target.shape[0])]
    with torch.no_grad():
        assert baseline.nll(target, mask) == pytest.approx(
            float(baseline.nll(shuffled_rows, mask)), abs=1e-6)


def test_recruited_field_picks_the_contact_that_actually_fired():
    states = torch.tensor([[[1.0, 2.0, 3.0]]])                 # (1, 1, nodes)
    observation = torch.eye(3)
    target = torch.tensor([[[0.0, 1.0, 0.0]]])
    assert float(recruited_field(states, target, observation)) == pytest.approx(2.0)


# -- the secondary observables ---------------------------------------------


def test_direction_persistence_is_high_for_a_straight_march():
    ranks, coords, _ = toy_event()
    assert direction_persistence(ranks, coords) > 0.9


def test_direction_persistence_drops_when_the_order_is_scrambled():
    ranks, coords, _ = toy_event(n_contacts=8, n_events=200, seed=1)
    rng = np.random.default_rng(2)
    scrambled = ranks.copy()
    for row in scrambled:
        present = np.flatnonzero(row >= 0)
        row[present] = rng.permutation(row[present])
    assert direction_persistence(scrambled, coords) < direction_persistence(ranks, coords)


def test_distance_time_relation_recovers_a_planted_association():
    ranks, coords, lag = toy_event(n_contacts=10, n_events=400, seed=3)
    out = distance_time_relation(ranks, coords, lag)
    assert out["n_pairs"] > 200
    assert out["partial_spearman"] > 0.2


def test_distance_time_relation_is_null_when_time_is_unrelated_to_space():
    ranks, coords, _ = toy_event(n_contacts=10, n_events=400, seed=4)
    rng = np.random.default_rng(5)
    lag = np.zeros_like(ranks, dtype=float)
    for event, row in enumerate(ranks):
        present = np.flatnonzero(row >= 0)
        # time still grows with the step, but carries no distance information
        lag[event, present[np.argsort(row[present])]] = np.cumsum(rng.exponential(1.0, present.size))
    out = distance_time_relation(ranks, coords, lag)
    assert abs(out["partial_spearman"]) < 0.1


def test_losses_do_not_transform_the_target_a_second_time():
    """The caller standardises the target; transforming it again squashes it.

    The first run of this round scored every arm on ``log1p`` of an already
    standardised value, which clamps each negative standardised entry to zero and
    collapses most of the target.  Every arm then looked worse than a step-index
    baseline that was itself scoring a different quantity.  A target with negative
    entries makes that mistake visible: a perfect predictor must reach zero loss.
    """
    target = torch.tensor([[-2.0, -0.5, 0.0, 1.5]])
    mask = torch.ones_like(target, dtype=torch.bool)

    head = TimeHead()
    with torch.no_grad():
        head.time_offset.zero_()
        head.time_slope.fill_(1.0)
    # predict = offset - slope * field, so this field is the exact answer
    assert float(head.nll(-target, target, mask)) == pytest.approx(0.0, abs=1e-9)

    baseline = RankOnlyTimeBaseline(max_steps=4)
    with torch.no_grad():
        baseline.per_step.copy_(target[0])
    assert float(baseline.nll(target, mask)) == pytest.approx(0.0, abs=1e-9)


def test_rank_only_baseline_cannot_beat_the_per_step_means_it_is_fit_to():
    """A one-parameter-per-step model cannot do better than the per-step means.

    A score lower than that is the signature of the loss and the reference scoring
    different quantities, which is how the double transform stayed invisible.
    """
    torch.manual_seed(0)
    target = torch.randn(400, 5)
    mask = torch.ones_like(target, dtype=torch.bool)
    optimal = float(((target - target.mean(dim=0, keepdim=True)) ** 2).mean())

    baseline = RankOnlyTimeBaseline(max_steps=5)
    optimiser = torch.optim.Adam(baseline.parameters(), lr=0.1)
    for _ in range(500):
        optimiser.zero_grad()
        loss = baseline.nll(target, mask)
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        assert float(baseline.nll(target, mask)) >= optimal - 1e-6


# -- the fair baselines the motifs actually have to beat --------------------


def planted_time(n_contacts=10, n_events=600, seed=0, distance_weight=1.0):
    """Events whose step duration is a step effect plus a distance effect."""
    rng = np.random.default_rng(seed)
    coords = np.column_stack([np.arange(n_contacts) * 5.0, np.zeros(n_contacts)])
    ranks = np.full((n_events, n_contacts), -1, dtype=np.int16)
    lag = np.zeros((n_events, n_contacts))
    for event in range(n_events):
        order, position = [], int(rng.integers(0, 3))
        while position < n_contacts and len(order) < 5:
            order.append(int(position))
            position += int(rng.integers(1, 4))
        if len(order) < 3:
            continue
        clock = 0.0
        for step, contact in enumerate(order):
            ranks[event, contact] = step
            if step:
                span = abs(coords[contact, 0] - coords[order[step - 1], 0])
                clock += 0.2 * (step + 1) + distance_weight * 0.1 * span + rng.normal(0, 0.05)
            lag[event, contact] = clock
    return ranks, coords, lag


def baseline_inputs(ranks, coords, lag):
    from src.topic5_motif_time_targets_v0_3 import build_event_tensors_with_time

    tensors = build_event_tensors_with_time(ranks, coords, lag)
    delta = tensors["time_delta"].numpy()
    mask = tensors["time_valid"].numpy()
    centroid = tensors["centroid"].numpy()
    distance = np.zeros_like(delta)
    distance[:, :-1] = np.linalg.norm(centroid[:, 1:] - centroid[:, :-1], axis=-1)
    target = np.log1p(delta)
    n_events = ranks.shape[0]
    train = np.zeros(n_events, dtype=bool)
    test = np.zeros(n_events, dtype=bool)
    train[: int(0.7 * n_events)] = True
    test[int(0.7 * n_events):] = True
    return tensors, target, mask, distance, train, test


def test_distance_baseline_beats_step_only_when_distance_drives_the_duration():
    """The reference must be able to absorb the destination distance.

    The motif reads its field at the contact that actually fired next, so it knows the
    destination.  If the baseline could not use that same distance, the motif would be
    credited for geometry the reference was simply denied.
    """
    ranks, coords, lag = planted_time(distance_weight=1.0)
    tensors, target, mask, distance, train, test = baseline_inputs(ranks, coords, lag)
    scores = time_baseline_scores(target, mask, distance, tensors["target"].numpy(),
                                  train, test)
    assert scores["STEP_DISTANCE"] < scores["STEP_ONLY"] * 0.95


def test_distance_baseline_does_not_help_when_duration_ignores_distance():
    ranks, coords, lag = planted_time(distance_weight=0.0, seed=4)
    tensors, target, mask, distance, train, test = baseline_inputs(ranks, coords, lag)
    scores = time_baseline_scores(target, mask, distance, tensors["target"].numpy(),
                                  train, test)
    assert scores["STEP_DISTANCE"] > scores["STEP_ONLY"] * 0.9


def test_every_baseline_level_is_reported():
    ranks, coords, lag = planted_time()
    tensors, target, mask, distance, train, test = baseline_inputs(ranks, coords, lag)
    scores = time_baseline_scores(target, mask, distance, tensors["target"].numpy(),
                                  train, test)
    assert set(scores) == set(TIME_BASELINES)


def test_adjacent_clue_matches_the_quantity_the_model_predicts():
    """The all-pairs statistic and the model's target are different estimands."""
    ranks, coords, lag = planted_time(distance_weight=1.0, n_events=900, seed=7)
    tensors, *_ = baseline_inputs(ranks, coords, lag)
    out = adjacent_distance_time_relation(tensors)
    assert out["n_steps"] > 200
    assert out["adjacent_partial_spearman"] > 0.2


def test_adjacent_clue_is_null_when_duration_ignores_distance():
    ranks, coords, lag = planted_time(distance_weight=0.0, n_events=900, seed=8)
    tensors, *_ = baseline_inputs(ranks, coords, lag)
    out = adjacent_distance_time_relation(tensors)
    assert abs(out["adjacent_partial_spearman"]) < 0.1


# -- the free arm must differ from the motifs in exactly one way ------------


def test_free_arm_reuses_the_motif_cell_and_changes_only_the_recurrent_drive():
    """A free *linear* cell would confound free structure with linear dynamics."""
    from src.topic5_dynamical_motif_rnn_v0_1 import MotifConfig, MotifRNN
    from src.topic5_motif_time_targets_v0_3 import FreeLowRankDrive

    n_nodes = 12
    coords = np.column_stack([np.arange(n_nodes) * 3.0, np.zeros(n_nodes)]).astype(np.float32)
    config = MotifConfig(
        model_id="DM0_ISOTROPIC", n_contacts=n_nodes, n_nodes=n_nodes,
        observation_operator=np.eye(n_nodes, dtype=np.float32), node_xy_mm=coords,
        local_mask=np.ones((n_nodes, n_nodes), dtype=np.uint8), r_forward_mm=6.0,
        sigma_s_mm=3.0)
    structured = MotifRNN(config)
    free = FreeLowRankDrive(config, rank=3, seed=0)

    shared = ("input_gain", "node_bias", "contact_bias", "readout_gain", "kappa_logit")
    for name in shared:
        assert hasattr(structured, name) and hasattr(free, name), name
    # the free arm keeps the saturating leaky cell: zero drive must give the same step
    h = torch.zeros(2, n_nodes)
    x = torch.zeros(2, n_nodes)
    x[:, 0] = 1.0
    with torch.no_grad():
        free.free_left.zero_()
        free.free_right.zero_()
        terms_free = free.recurrent_terms()
        gate = torch.zeros(2)
        stepped = free.step(h, x, gate, terms_free)
        # a linear cell would return the pre-activation; a tanh cell saturates
        assert float(stepped.abs().max()) <= 1.0
