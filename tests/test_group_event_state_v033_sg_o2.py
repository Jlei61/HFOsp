"""Contracts for the staged human S_G O2 training diagnostic."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.topic5_group_event_state.v033_training_lab.contact_grammar import LegacyContactGrammar
from src.topic5_group_event_state.v033_training_lab.sg_o2 import (
    FrozenLegacyStateScorer,
    GrammarPairs,
    SGO2ArchConfig,
    SGO2EventEncoder,
    assert_training_phases,
    staged_o2_plan,
    validate_o2_smoke_lease,
)
from src.topic5_rank_distribution import FullHistorySequenceGRU, next_set_stop_loss


def _arch(**kwargs) -> SGO2ArchConfig:
    return SGO2ArchConfig(**kwargs).validate()


def test_required_o2_architecture_surface_and_successive_plan() -> None:
    plan = staged_o2_plan()
    assert plan["launch_policy"].startswith("one stage at a time")
    assert [c["width"] for c in plan["stages"][1]["cells"]] == [32, 64, 128]
    structure = plan["stages"][2]["cells_template"]
    assert {(c["depth"], c["residual"], c["norm"]) for c in structure} == {
        (d, r, n) for d in (2, 4) for r in (False, True) for n in ("pre", "post")
    }
    later = plan["stages"][3]["cells_template"]
    assert {(c["init"], c["update_gate"]) for c in later} == {
        (i, g) for i in ("xavier", "orthogonal") for g in (False, True)
    }
    assert "only if" in plan["stages"][4]["requires"]


@pytest.mark.parametrize("depth", [2, 4])
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("norm", ["pre", "post"])
@pytest.mark.parametrize("init", ["xavier", "orthogonal"])
@pytest.mark.parametrize("gate", [False, True])
def test_o2_encoder_cells_run_and_gate_is_explicit(
    depth: int, residual: bool, norm: str, init: str, gate: bool
) -> None:
    torch.manual_seed(3)
    cfg = _arch(width=32, depth=depth, residual=residual, norm=norm,
                init=init, update_gate=gate)
    model = SGO2EventEncoder(
        8, cfg, mark_columns=(0, 1, 2, 3, 4, 5), scaffold_columns=(6, 7)
    )
    write, update_gate = model(torch.randn(7, 8))
    assert write.shape == update_gate.shape == (7, cfg.write_width)
    if gate:
        assert torch.all((update_gate > 0) & (update_gate < 1))
    else:
        assert torch.equal(update_gate, torch.ones_like(update_gate))
    assert len(model.blocks) == depth
    assert all(block.residual is residual and block.norm_position == norm
               for block in model.blocks)


def test_mark_scaffold_split_is_optional_not_default() -> None:
    joint = SGO2EventEncoder(
        8, _arch(input_routing="joint"),
        mark_columns=(0, 1, 2, 3, 4, 5), scaffold_columns=(6, 7),
    )
    split = SGO2EventEncoder(
        8, _arch(input_routing="mark_scaffold_split"),
        mark_columns=(0, 1, 2, 3, 4, 5), scaffold_columns=(6, 7),
    )
    x = torch.randn(5, 8)
    assert joint(x)[0].shape == split(x)[0].shape == (5, 4)
    assert joint.input_joint is not None and split.input_joint is None


def test_development_seizure_sealed_and_unknown_phases_fail_closed() -> None:
    assert_training_phases(["STATE_TRAIN", "STATE_SELECTION"])
    for phase in ("DEVELOPMENT_EVALUATION", "SEALED", "TEST", "SEIZURE", "CALIBRATION"):
        with pytest.raises(PermissionError):
            assert_training_phases(["STATE_TRAIN", phase])


def test_lease_must_explicitly_authorize_one_worker_o2_smoke(tmp_path) -> None:
    path = tmp_path / "lease.json"
    good = {
        "status": "ACTIVE_O2_IMPLEMENTATION_SMOKE_ONLY", "max_workers": 1,
        "allowed_work": ["one S_G O2 resource smoke"],
        "allowed_subjects": ["epilepsiae_916"],
        "allowed_gpu_indices": [1],
        "max_jobs_per_gpu_before_sentinel_review": 1,
    }
    path.write_text(json.dumps(good))
    assert validate_o2_smoke_lease(path, subject="epilepsiae_916")["max_workers"] == 1
    good["status"] = "HOLD"
    path.write_text(json.dumps(good))
    with pytest.raises(PermissionError):
        validate_o2_smoke_lease(path, subject="epilepsiae_916")


def test_pair_weights_are_equal_anchor_then_mean_event() -> None:
    pairs = GrammarPairs(
        anchor_rows=np.array([2, 5]),
        pair_anchor=np.array([0, 0, 1]),
        pair_event=np.array([3, 4, 8]),
        pair_weight=np.array([0.25, 0.25, 0.5]),
    ).validate()
    assert pairs.pair_weight.sum() == pytest.approx(1.0)


def test_frozen_legacy_scorer_preserves_old_loss_and_gradients_only_residual() -> None:
    torch.manual_seed(4)
    base = FullHistorySequenceGRU(
        5, hidden_size=8, contact_embedding_dim=8,
        contact_encoder_hidden=8, local_offset_dim=2,
    )
    decoder = LegacyContactGrammar(
        base, np.random.default_rng(2).normal(size=(4, 5)).astype(np.float32),
        np.ones(4, dtype=bool), local_offset_dim=2,
    )
    scorer = FrozenLegacyStateScorer(decoder, state_dim=6, rank=2)
    groups = torch.tensor([[0, 1, -1, -1], [0, 0, 1, -1]])
    counts = torch.tensor([2, 2])
    expected = next_set_stop_loss(decoder(groups, counts), groups, counts)["event_nll"]
    state = torch.zeros(2, 6, requires_grad=True)
    observed = scorer.event_nll(groups, counts, state)
    # Zero state is exact prefix-only parity because every residual has no bias.
    assert torch.equal(expected, observed)
    observed.mean().backward()
    assert state.grad is not None
    assert all(parameter.grad is None for parameter in decoder.parameters())
    assert all(not parameter.requires_grad for parameter in decoder.parameters())

