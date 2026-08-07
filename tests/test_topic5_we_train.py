"""Contract tests for the WE-SLP-RNN v0.3 training unit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_topic5_we_unit import (  # noqa: E402
    DEFAULTS,
    rank_disagreement,
    resolve_batch,
    sequence_agreement,
    shuffle_targets,
    train_unit,
)


def test_batch_gives_every_patient_at_least_eight_updates():
    assert resolve_batch(249) == 32
    assert resolve_batch(105) == 14
    assert resolve_batch(100_000) == 1024
    for n in (105, 249, 1000, 8000, 100_000):
        assert int(np.ceil(n / resolve_batch(n))) >= 8


def test_shuffled_targets_keep_the_participating_set_and_length():
    rng = np.random.default_rng(0)
    ranks = np.full((50, 8), -1, np.int16)
    for e in range(50):
        k = rng.integers(2, 8)
        picks = rng.choice(8, size=k, replace=False)
        ranks[e, picks] = np.arange(k)
    out = shuffle_targets(ranks, seed=1)
    assert np.array_equal(out >= 0, ranks >= 0)
    for a, b in zip(ranks, out):
        assert sorted(a[a >= 0].tolist()) == sorted(b[b >= 0].tolist())
    assert not np.array_equal(out, ranks)


def test_sequence_agreement_is_one_for_a_perfect_reproduction():
    observed = np.array([0, 1, 2, -1])
    assert sequence_agreement(observed, [[0], [1], [2]]) == 1.0
    assert sequence_agreement(observed, [[2], [1], [0]]) == -1.0


def test_rank_disagreement_is_zero_for_identical_repertoires():
    a = [[[0], [1], [2]], [[3], [4]]]
    assert rank_disagreement(a, a) == 0.0
    b = [[[0], [2], [1]], [[3], [4]]]
    assert 0.0 < rank_disagreement(a, b) < 1.0


def _tiny_cache(tmp_path: Path, n_events=180, n_contacts=6, n_nodes=12) -> str:
    rng = np.random.default_rng(0)
    fit_id = "test__shared"
    cache = tmp_path / "cache" / fit_id
    cache.mkdir(parents=True)
    contacts = rng.uniform(-15, 15, size=(n_contacts, 2))
    nodes = rng.uniform(-15, 15, size=(n_nodes, 2))
    d = np.linalg.norm(contacts[:, None] - nodes[None], axis=-1)
    w = np.exp(-(d ** 2) / (2 * 6.0 ** 2))
    w[d > 18.0] = 0.0
    w[w.sum(1) == 0, 0] = 1.0
    np.savez(cache / "plane.npz", contacts_xy_mm=contacts.astype(np.float32),
             nodes_xy_mm=nodes.astype(np.float32),
             H=(w / w.sum(1, keepdims=True)).astype(np.float32),
             D_mm=np.linalg.norm(nodes[:, None] - nodes[None], axis=-1).astype(np.float32),
             sigma_mm=np.array([6.0], np.float32), scale_mm=np.array([30.0], np.float32))
    ranks = np.full((n_events, n_contacts), -1, np.int16)
    for e in range(n_events):
        k = int(rng.integers(3, n_contacts + 1))
        picks = rng.permutation(n_contacts)[:k]
        ranks[e, picks] = np.arange(k)
    split = np.zeros(n_events, np.int8)
    split[int(0.7 * n_events):int(0.85 * n_events)] = 1
    split[int(0.85 * n_events):] = 2
    np.savez(cache / "events.npz", ranks=ranks, split=split,
             mode=(rng.integers(0, 2, n_events)).astype(np.int8))
    (cache / "provenance.json").write_text(json.dumps({
        "scope": "shared", "subject": "test", "n_contacts": n_contacts,
        "n_nodes": n_nodes, "label_coverage": 1.0}))
    return fit_id


def _cfg(**over):
    cfg = dict(DEFAULTS)
    cfg.update({"cell": "rnn", "epochs_warmup": 1, "epochs_rewire": 2,
                "epochs_freeze": 3, "rollout_events": 8})
    cfg.update(over)
    return cfg


def test_hitting_the_ceiling_is_recorded_as_not_converged(tmp_path):
    fit_id = _tiny_cache(tmp_path)
    m = train_unit(fit_id, "SPATIAL_SET", 0, _cfg(patience=999), tmp_path,
                   torch.device("cpu"))
    assert m["hit_ceiling"] is True
    assert m["converged"] is False
    assert json.loads((tmp_path / "per_subject" / fit_id / "SPATIAL_SET_rnn" / "seed0"
                       / "DONE.json").read_text())["converged"] is False


def test_stopping_early_is_recorded_as_converged(tmp_path):
    fit_id = _tiny_cache(tmp_path)
    m = train_unit(fit_id, "SPATIAL_SET", 0,
                   _cfg(patience=1, epochs_freeze=200, min_relative_improvement=0.5),
                   tmp_path, torch.device("cpu"))
    assert m["converged"] is True
    assert m["n_epochs"] < 1 + 2 + 200


def test_the_early_stopping_clock_does_not_run_while_the_mask_moves(tmp_path):
    # With patience 1 and an impossible improvement bar, the run must still last
    # at least through warmup + rewiring before it is allowed to stop.
    fit_id = _tiny_cache(tmp_path)
    m = train_unit(fit_id, "SPATIAL_SET", 0,
                   _cfg(epochs_warmup=2, epochs_rewire=5, epochs_freeze=200,
                        patience=1, min_relative_improvement=0.99),
                   tmp_path, torch.device("cpu"))
    assert m["n_epochs"] >= 2 + 5 + 1


def test_unit_records_the_graph_and_its_wiring_cost(tmp_path):
    fit_id = _tiny_cache(tmp_path)
    train_unit(fit_id, "SPATIAL_SET", 0, _cfg(), tmp_path, torch.device("cpu"))
    graph = np.load(tmp_path / "per_subject" / fit_id / "SPATIAL_SET_rnn" / "seed0" / "graph.npz")
    assert set(graph.files) == {"mask", "initial_mask", "strength", "D_mm"}
    assert graph["mask"].sum() == graph["initial_mask"].sum()


def test_static_arm_writes_no_graph_and_zero_wiring_cost(tmp_path):
    fit_id = _tiny_cache(tmp_path)
    m = train_unit(fit_id, "STATIC_CONTACT", 0, _cfg(), tmp_path, torch.device("cpu"))
    assert m["c_wiring"] == 0.0 and m["edge_count"] == 0
    assert not (tmp_path / "per_subject" / fit_id / "STATIC_CONTACT_rnn" / "seed0" / "graph.npz").exists()


def test_shuffled_control_lands_in_its_own_directory(tmp_path):
    fit_id = _tiny_cache(tmp_path)
    m = train_unit(fit_id, "SPATIAL_SET", 0, _cfg(), tmp_path, torch.device("cpu"),
                   shuffled=True)
    assert m["shuffled_targets"] is True
    assert (tmp_path / "per_subject" / fit_id / "SPATIAL_SET_shuffled_rnn" / "seed0"
            / "metrics.json").exists()


def test_thin_fits_are_flagged(tmp_path):
    fit_id = _tiny_cache(tmp_path, n_events=180)
    m = train_unit(fit_id, "SPATIAL_SET", 0, _cfg(), tmp_path, torch.device("cpu"))
    assert m["thin"] is True and m["n_train"] < 500
