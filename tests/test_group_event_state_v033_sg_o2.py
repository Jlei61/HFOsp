"""Contracts for the staged human S_G O2 training diagnostic."""

from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.topic5_group_event_state.v033_training_lab.contact_grammar import LegacyContactGrammar
from src.topic5_group_event_state.v033_training_lab.sg_o2 import (
    FrozenLegacyStateScorer,
    FrozenO1Recipe,
    GrammarPairs,
    SGO2ArchConfig,
    SGO2EventEncoder,
    SGO2TrainConfig,
    _build_o1_optimizer,
    assert_training_phases,
    ensure_pairing_manifest,
    freeze_o1_optimizer_recipe,
    load_frozen_o1_recipe,
    o2_cell_contract,
    resolve_o2_output_dir,
    staged_o2_plan,
    validate_o2_lease,
    validate_o2_smoke_lease,
    validate_resume_payload,
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
    assert len(plan["stages"]) == 4
    assert "only if" in plan["optional_architecture_diagnostic"]["requires"]
    assert "stage S3" in plan["optional_architecture_diagnostic"]["requires"]


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
        "expires_at": (dt.datetime.now().astimezone() + dt.timedelta(hours=1)).isoformat(),
    }
    path.write_text(json.dumps(good))
    assert validate_o2_smoke_lease(path, subject="epilepsiae_916")["max_workers"] == 1
    good["status"] = "HOLD"
    path.write_text(json.dumps(good))
    with pytest.raises(PermissionError):
        validate_o2_smoke_lease(path, subject="epilepsiae_916")


def test_smoke_and_full_training_are_not_cap_aliases() -> None:
    SGO2TrainConfig(
        run_kind="resource_smoke", smoke_train_anchors=2,
        smoke_inner_anchors=2,
    ).validate()
    SGO2TrainConfig(run_kind="full_training").validate()
    with pytest.raises(ValueError, match="requires both"):
        SGO2TrainConfig(run_kind="resource_smoke").validate()
    with pytest.raises(ValueError, match="forbids smoke caps"):
        SGO2TrainConfig(
            run_kind="full_training", smoke_train_anchors=100,
            smoke_inner_anchors=100,
        ).validate()


def test_full_lease_cannot_be_reused_as_smoke_or_wrong_gpu(tmp_path) -> None:
    path = tmp_path / "lease.json"
    payload = {
        "status": "ACTIVE_O2_FULL_TRAINING", "max_workers": 1,
        "allowed_work": ["S_G O2 full training"],
        "allowed_subjects": ["epilepsiae_916"], "allowed_gpu_indices": [1],
        "max_jobs_per_gpu_before_sentinel_review": 1,
        "expires_at": (dt.datetime.now().astimezone() + dt.timedelta(hours=1)).isoformat(),
    }
    path.write_text(json.dumps(payload))
    assert validate_o2_lease(
        path, subject="epilepsiae_916", run_kind="full_training",
        device=torch.device("cuda:1"),
    )["status"].startswith("ACTIVE")
    with pytest.raises(PermissionError, match="resource smoke"):
        validate_o2_lease(path, subject="epilepsiae_916", run_kind="resource_smoke")
    with pytest.raises(PermissionError, match="outside lease"):
        validate_o2_lease(
            path, subject="epilepsiae_916", run_kind="full_training",
            device=torch.device("cuda:0"),
        )


def _o1_files(tmp_path):
    recipe = {
        "optimizer": "adamw", "schedule": "constant", "betas": [0.9, 0.999],
        "eps": 1e-8, "weight_decay": 1e-4, "grad_clip": 0.5,
        "lr": {"encoder_weights": 3e-4, "encoder_bias": 1e-4, "adapter_w": 2e-3},
    }
    cell = {
        "format": "group_event_state_v0_3_3_o1_optimizer_cell_v1",
        "cell_id": "selected", "config_hash": "a" * 64, "recipe": recipe,
    }
    study = {
        "format": "group_event_state_v0_3_3_o1_optimizer_study_v1",
        "study_content_hash": "b" * 64,
        "scientific_scope": {
            "development_evaluation_read": False,
            "sealed_partition_opened": False,
            "selection_phase": "STATE_SELECTION",
        },
        "cells": [{"cell_id": "selected", "config_hash": "a" * 64}],
    }
    study_path, cell_path = tmp_path / "study.json", tmp_path / "cell.json"
    study_path.write_text(json.dumps(study)); cell_path.write_text(json.dumps(cell))
    return study_path, cell_path


def test_o1_recipe_is_frozen_hashed_and_parameter_groups_reused(tmp_path) -> None:
    study, cell = _o1_files(tmp_path)
    path = tmp_path / "o1_recipe.json"
    freeze_o1_optimizer_recipe(
        study_manifest_path=study, cell_manifest_path=cell, output_path=path
    )
    recipe = load_frozen_o1_recipe(path)
    assert recipe.gradient_clip == 0.5 and recipe.selected_cell_id == "selected"
    base = FullHistorySequenceGRU(
        5, hidden_size=8, contact_embedding_dim=8,
        contact_encoder_hidden=8, local_offset_dim=2,
    )
    decoder = LegacyContactGrammar(
        base, np.ones((4, 5), dtype=np.float32), np.ones(4, dtype=bool),
        local_offset_dim=2,
    )
    from src.topic5_group_event_state.v033_training_lab.sg_o2 import SGO2Model
    model = SGO2Model(
        decoder, in_dim=8, arch=_arch(), mark_columns=range(6),
        scaffold_columns=(6, 7),
    )
    _optimizer, contract = _build_o1_optimizer(model, recipe)
    assert contract["lr_by_group"] == {
        "encoder_weights": 3e-4, "encoder_bias": 1e-4, "adapter_w": 2e-3,
    }
    changed = json.loads(path.read_text()); changed["lr_encoder_weights"] = 9.0
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="content hash"):
        load_frozen_o1_recipe(path)


def test_cell_contract_pairs_same_seed_and_resume_rejects_other_cell(tmp_path) -> None:
    recipe = FrozenO1Recipe(
        source_path=str(tmp_path / "recipe.json"), source_sha256="1" * 64,
        content_hash="2" * 64, optimizer="adamw", schedule="constant",
        betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4,
        gradient_clip=1.0, lr_encoder_weights=1e-3,
        lr_encoder_bias=1e-3, lr_adapter_w=1e-3,
        selected_cell_id="selected", o1_study_hash="3" * 64,
    ).validate()
    data = SimpleNamespace(provenance={"human_input_sha256": "4" * 64, "split_hash": "5" * 64})
    cfg = SGO2TrainConfig(run_kind="full_training", seed=91)
    a = o2_cell_contract(
        subject="epilepsiae_916", stage="S1", pairing_id="pair91",
        arch=_arch(width=32), train_cfg=cfg, o1_recipe=recipe,
        data=data, grammar_hash="6" * 64,
    )
    b = o2_cell_contract(
        subject="epilepsiae_916", stage="S1", pairing_id="pair91",
        arch=_arch(width=64), train_cfg=cfg, o1_recipe=recipe,
        data=data, grammar_hash="6" * 64,
    )
    assert a["pairing_id"] == b["pairing_id"] == "pair91"
    assert a["train_config"]["seed"] == b["train_config"]["seed"] == 91
    assert a["contract_hash"] != b["contract_hash"]
    pairing = tmp_path / "pairing.json"
    ensure_pairing_manifest(pairing, contract=a)
    ensure_pairing_manifest(pairing, contract=b)
    valid = {"format": "group_event_state_v0_3_3_sg_o2_resume",
             "contract_hash": a["contract_hash"], "last_completed_step": 3, "history": []}
    validate_resume_payload(valid, contract_hash=a["contract_hash"])
    with pytest.raises(PermissionError):
        validate_resume_payload(valid, contract_hash=b["contract_hash"])


def test_pairing_manifest_rejects_a_seed_change(tmp_path) -> None:
    recipe = FrozenO1Recipe(
        source_path=str(tmp_path / "recipe.json"), source_sha256="1" * 64,
        content_hash="2" * 64, optimizer="adamw", schedule="constant",
        betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4,
        gradient_clip=1.0, lr_encoder_weights=1e-3,
        lr_encoder_bias=1e-3, lr_adapter_w=1e-3,
        selected_cell_id="selected", o1_study_hash="3" * 64,
    ).validate()
    data = SimpleNamespace(provenance={"human_input_sha256": "4" * 64, "split_hash": "5" * 64})
    common = dict(
        subject="epilepsiae_916", stage="S1", pairing_id="locked",
        arch=_arch(), o1_recipe=recipe, data=data, grammar_hash="6" * 64,
    )
    first = o2_cell_contract(
        **common, train_cfg=SGO2TrainConfig(run_kind="full_training", seed=10)
    )
    changed = o2_cell_contract(
        **common, train_cfg=SGO2TrainConfig(run_kind="full_training", seed=11)
    )
    path = tmp_path / "pairing.json"
    ensure_pairing_manifest(path, contract=first)
    with pytest.raises(PermissionError, match="differ"):
        ensure_pairing_manifest(path, contract=changed)


def test_output_namespace_cannot_alias_smoke_and_full(monkeypatch, tmp_path) -> None:
    import src.topic5_group_event_state.v033_training_lab.sg_o2 as module
    monkeypatch.setattr(module, "O2_ROOT", tmp_path)
    full = tmp_path / "epilepsiae_916" / "full_training" / "pair" / "S1" / "cell"
    observed, pairing = resolve_o2_output_dir(
        subject="epilepsiae_916", run_kind="full_training", pairing_id="pair",
        stage="S1", requested=full,
    )
    assert observed == full.resolve()
    assert pairing == full.parent / "pairing_manifest.json"
    smoke = tmp_path / "epilepsiae_916" / "resource_smoke" / "pair" / "S1" / "cell"
    with pytest.raises(PermissionError, match="canonical full_training"):
        resolve_o2_output_dir(
            subject="epilepsiae_916", run_kind="full_training", pairing_id="pair",
            stage="S1", requested=smoke,
        )


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
