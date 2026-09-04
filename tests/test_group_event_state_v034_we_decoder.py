"""Frozen tissue-decoder S_P scorer: parity, per-event scores, alignment and pair utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from src.topic5_lbss_rnn_v0_2 import LBSSConfig, LBSSModel, build_pool_contract
from src.topic5_wiring_economy_rnn import build_event_tensors, next_rank_stop_loss, cardinality_conditioned_nll
from src.topic5_group_event_state.v033_training_lab.sg_o2 import GrammarPairs
from src.topic5_group_event_state.v034_spatial_state.we_decoder import (
    FrozenDecoderBundle, WEStateScorer, align_events, forward_with_h0, per_event_scores, restrict_pairs,
    split_pairs_by_time,
)

CPU = torch.device("cpu")


def _toy_model(n_contacts: int = 4, n_nodes: int = 9, seed: int = 0) -> LBSSModel:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 30, size=(n_nodes, 2))
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1)).astype(np.float32)
    h = np.zeros((n_contacts, n_nodes), np.float32)
    for c in range(n_contacts):
        h[c, c * 2] = 1.0
    pools = build_pool_contract(d, 0.3, 0.3, 2.0)
    model = LBSSModel(LBSSConfig(arm="L3_LOCAL_PLUS_LEARNED_LR", n_contacts=n_contacts, n_nodes=n_nodes,
                                 observation_operator=h, node_distance_mm=d, local_mask=pools.local_mask,
                                 extra_local_pool=pools.extra_local_pool, nonlocal_pool=pools.nonlocal_pool,
                                 k_added=pools.k_added, seed=seed, state_dim=1))
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.3 * torch.randn_like(p))
    model.freeze_mask(); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _toy_ranks(n_events: int = 12, n_contacts: int = 4, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ranks = np.full((n_events, n_contacts), -1, np.int16)
    for e in range(n_events):
        k = int(rng.integers(1, n_contacts + 1))
        order = rng.permutation(n_contacts)[:k]
        ranks[e, order] = np.arange(k)
    return ranks


def _bundle(model, ranks, times):
    return FrozenDecoderBundle(model=model, ranks=ranks, event_abs_time=times, contact_names=("a", "b", "c", "d"),
                               unit_dir=Path("."), cache_dir=Path("."), metrics={}, split=np.zeros(len(times), np.int8))


def test_zero_h0_reproduces_the_frozen_decoder_and_single_event_scores_match_batch_loss():
    model = _toy_model()
    ranks = _toy_ranks()
    t = build_event_tensors(ranks)
    logits, stops = model(t["x"], t["recruited"], t["valid"])
    l2, s2 = forward_with_h0(model, t["x"], t["recruited"], t["valid"], None)
    assert torch.equal(logits, l2) and torch.equal(stops, s2)
    scores = per_event_scores(logits, stops, t)
    for e in range(ranks.shape[0]):
        single = {k: v[e:e + 1] for k, v in t.items()}
        loss, nb, sb = next_rank_stop_loss(logits[e:e + 1], stops[e:e + 1], single["target"], single["available"],
                                           single["valid"], single["is_last"])
        assert float(scores["grammar"][e]) == pytest.approx(float(loss), abs=1e-6)
        predict = single["valid"] & ~single["is_last"]
        cnll = cardinality_conditioned_nll(logits[e:e + 1], single["target"], single["available"], predict)
        if float(predict.sum()) > 0:
            assert float(scores["contact_nll"][e]) == pytest.approx(float(cnll), abs=1e-6)


def test_scorer_bias_and_state_paths_change_scores_and_carry_gradient():
    model = _toy_model()
    ranks = _toy_ranks()
    times = np.arange(ranks.shape[0], dtype=np.float64) * 10.0
    scorer = WEStateScorer(_bundle(model, ranks, times), state_dim=6, rank=3)
    t = build_event_tensors(ranks)
    state = torch.randn(ranks.shape[0], 6)
    zero = scorer.scores(t, state, use_bias=False, use_state=False)["grammar"]
    with torch.no_grad():
        scorer.h0_bias.add_(0.5)
        scorer.to_h0[1].weight.mul_(100.0)
    adapter = scorer.scores(t, state, use_bias=True, use_state=False)["grammar"]
    learned = scorer.scores(t, state, use_bias=True, use_state=True)["grammar"]
    assert not torch.allclose(zero, adapter) and not torch.allclose(adapter, learned)
    learned.sum().backward()
    assert scorer.to_h0[0].weight.grad is not None and float(scorer.to_h0[0].weight.grad.abs().sum()) > 0
    assert all(p.grad is None for p in model.parameters())      # decoder stays frozen


def test_align_events_matches_within_tolerance_and_marks_missing():
    cache = np.array([10.0, 20.0, 30.0, 40.0])
    ours = np.array([10.0004, 15.0, 20.0, 40.0009, 41.0])
    idx = align_events(ours, cache)
    assert idx.tolist() == [0, -1, 1, 3, -1]


def test_restrict_and_split_pairs_keep_equal_anchor_weights_and_chronology():
    pairs = GrammarPairs(anchor_rows=np.array([5, 9, 2]), pair_anchor=np.array([0, 0, 1, 2, 2, 2]),
                         pair_event=np.array([0, 1, 2, 3, 4, 5]), pair_weight=np.array([1 / 6, 1 / 6, 1 / 3, 1 / 9, 1 / 9, 1 / 9])).validate()
    keep = np.array([True, False, False, True, True, True])
    r = restrict_pairs(pairs, keep)
    assert r.anchor_rows.tolist() == [5, 2] and abs(r.pair_weight.sum() - 1) < 1e-9
    assert np.allclose(r.pair_weight, [0.5, 1 / 6, 1 / 6, 1 / 6])
    anchor_time = np.zeros(10); anchor_time[[5, 9, 2]] = [100.0, 300.0, 200.0]
    head, tail = split_pairs_by_time(pairs, anchor_time, 0.34)
    assert tail.anchor_rows.tolist() == [9] and sorted(head.anchor_rows.tolist()) == [2, 5]
    assert abs(head.pair_weight.sum() - 1) < 1e-9 and abs(tail.pair_weight.sum() - 1) < 1e-9
