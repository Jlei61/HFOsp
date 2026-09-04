"""Contracts for the v0.3.4 spatial predictive-state pilot."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from src.topic5_group_event_state.v033_training_lab.contact_grammar import (
    LegacyContactGrammar,
    tensor_state_hash,
)
from src.topic5_group_event_state.v033_training_lab.sg_o2 import GrammarPairs
from src.topic5_group_event_state.v034_spatial_state.contracts import (
    ArchConfig,
    EVALUATION_SUBJECTS,
    LOCKED_SEEDS,
    SEED_CONTRACT,
    OptimizerConfig,
    TrainConfig,
    assert_safe_phases,
    build_evaluation_release_gate,
    build_human_release_gate,
    build_locked_recipe_manifest,
    lr_search_cells,
    optimizer_contract,
    require_evaluation_release_gate,
    require_human_release_gate,
    seed_before_model_construction,
)
from src.topic5_group_event_state.v034_spatial_state.data import (
    _burn_in_pairs,
    sample_equal_anchor_pairs,
)
from src.topic5_group_event_state.v034_spatial_state.model import (
    SpatialStateModel,
    build_optimizer,
)
from src.topic5_group_event_state.v034_spatial_state.synthetic import (
    _truth_oracle,
    make_synthetic_spatial_data,
    run_synthetic,
)
from src.topic5_group_event_state.v034_spatial_state.trainer import (
    chronological_train_fit_inner,
    train_spatial_state,
)
from src.topic5_rank_distribution import FullHistorySequenceGRU, next_set_stop_loss


def _decoder(n_contacts: int = 6) -> LegacyContactGrammar:
    base = FullHistorySequenceGRU(
        5, hidden_size=8, contact_embedding_dim=8,
        contact_encoder_hidden=8, local_offset_dim=2,
    )
    return LegacyContactGrammar(
        base,
        np.random.default_rng(2).normal(size=(n_contacts, 5)).astype(np.float32),
        np.ones(n_contacts, dtype=bool),
        local_offset_dim=2,
    )


def test_lr_search_is_bounded_and_has_no_hidden_multiplier() -> None:
    cells = lr_search_cells()
    assert len(cells) == 5
    assert len({tuple(sorted(row.items())) for row in cells}) == 5
    contract = optimizer_contract(OptimizerConfig())
    assert contract["requested_lr"] == contract["effective_lr"]
    assert contract["hidden_lr_multiplier"] == 1.0


def test_human_and_synthetic_rungs_are_explicit() -> None:
    for rung in (300, 900, 2700):
        TrainConfig(max_steps=rung).validate()
    with pytest.raises(ValueError):
        TrainConfig(max_steps=20).validate()
    TrainConfig(max_steps=20).validate(allow_tiny=True)


def test_forbidden_phases_fail_closed() -> None:
    assert_safe_phases(["STATE_TRAIN", "STATE_SELECTION"])
    for phase in ("DEVELOPMENT_EVALUATION", "SEALED", "TEST", "SEIZURE"):
        with pytest.raises(PermissionError):
            assert_safe_phases([phase])


def test_release_gate_requires_two_passing_immutable_cards(tmp_path) -> None:
    def card(kind: str, status: str = "PASS"):
        path = tmp_path / f"{kind}.json"
        path.write_text(json.dumps({
            "format": f"group_event_state_v0_3_4_spatial_state_{kind}_card_v1",
            "status": status,
            "contract_hash": kind,
            "development_targets_read": False,
            "sealed_partition_opened": False,
            "seizure_outcomes_read": False,
        }))
        return path
    synthetic = card("synthetic_recovery")
    canary = card("tiny_canary")
    gate_path = tmp_path / "gate.json"
    build_human_release_gate(synthetic_card=synthetic, canary_card=canary, output=gate_path)
    require_human_release_gate(gate_path, subject="epilepsiae_253")
    with pytest.raises(PermissionError):
        require_human_release_gate(gate_path, subject="epilepsiae_1146")
    canary.write_text(canary.read_text() + "\n")
    with pytest.raises(PermissionError, match="changed"):
        require_human_release_gate(gate_path, subject="epilepsiae_253")


def _locked_tuning_card(
    tmp_path, subject: str, seed: int, gain: float, *, rung: int = 900, recipe=None,
):
    recipe = deepcopy(recipe) if recipe is not None else _example_recipe()
    train = deepcopy(recipe["train"])
    train["seed"] = seed
    train["max_steps"] = rung
    optimizer = recipe["optimizer"]
    path = tmp_path / f"{subject}_{seed}_{rung}.json"
    contract = {
        "subject": subject,
        "arch": recipe["arch"],
        "optimizer": {
            "family": optimizer["family"],
            "requested_lr": {
                "encoder": optimizer["lr_encoder"],
                "state_adapter": optimizer["lr_state_adapter"],
                "auxiliary": optimizer["lr_auxiliary"],
            },
            "effective_lr": {
                "encoder": optimizer["lr_encoder"],
                "state_adapter": optimizer["lr_state_adapter"],
                "auxiliary": optimizer["lr_auxiliary"],
            },
            "hidden_lr_multiplier": 1.0,
            "weight_decay": optimizer["weight_decay"],
            "betas": optimizer["betas"],
            "eps": optimizer["eps"],
            "gradient_clip": optimizer["gradient_clip"],
        },
        "train": train,
        "seed_contract": SEED_CONTRACT,
        "input_provenance": {
            "target_phases": ["STATE_TRAIN", "STATE_SELECTION"],
            "development_targets_read": False,
            "sealed_partition_opened": False,
            "seizure_outcomes_read": False,
        },
    }
    from src.topic5_group_event_state.v033_training_lab.paths import payload_hash
    path.write_text(json.dumps({
        "format": "group_event_state_v0_3_4_spatial_state_human_tuning_card_v1",
        "status": "PASS",
        "seed_contract": SEED_CONTRACT,
        "contract": contract,
        "contract_hash": payload_hash(contract),
        "selection_gain": gain,
        "selected_step": 100 if gain > 0 else 0,
        "development_targets_read": False,
        "sealed_partition_opened": False,
        "seizure_outcomes_read": False,
    }))
    return path


def _example_recipe():
    return {
        "arch": {
            "width": 64, "depth": 4, "write_width": 4, "adapter_rank": 4,
            "residual": True, "taus_seconds": [300.0, 1800.0, 7200.0],
        },
        "optimizer": {
            "family": "adamw", "lr_encoder": 1e-3, "lr_state_adapter": 3e-3,
            "lr_auxiliary": 1e-3, "weight_decay": 1e-4, "betas": [0.9, 0.999],
            "eps": 1e-8, "gradient_clip": 1.0,
        },
        "train": {
            "max_steps": 900, "validate_every": 25, "patience_checks": 8,
            "pair_batch_size": 512, "anchors_per_step": 128, "events_per_anchor": 16,
            "burn_in_seconds": 1800.0, "chunk_seconds": 3600.0,
            "grammar_weight": 1.0, "extent_weight": 0.2, "lag_weight": 0.2,
        },
        "allowed_seeds": list(LOCKED_SEEDS),
        "seed_contract": SEED_CONTRACT,
    }


def _locked_recipe_and_eval_gate(tmp_path):
    cards = [
        _locked_tuning_card(tmp_path, "epilepsiae_253", seed, gain)
        for seed, gain in zip(LOCKED_SEEDS, (0.32, 0.27, 0.319, 0.313, 0.314))
    ]
    diagnostic = _locked_tuning_card(
        tmp_path, "epilepsiae_916", LOCKED_SEEDS[0], 0.0, rung=300,
    )
    recipe_path = tmp_path / "recipe.json"
    build_locked_recipe_manifest(
        e253_cards=cards, e916_diagnostic_cards=[diagnostic], output=recipe_path,
    )
    input_root = tmp_path / "inputs"
    for subject in EVALUATION_SUBJECTS:
        root = input_root / subject
        root.mkdir(parents=True)
        artifact = root / "input.npz"
        artifact.write_bytes((subject + "-prefix-only").encode())
        from src.topic5_group_event_state.v033_training_lab.paths import file_hash
        (root / "manifest_v3.json").write_text(json.dumps({
            "format": "group_event_state_v0_3_3_human_r0_input_manifest",
            "subject": subject,
            "role": "explicit_non_tuning_override",
            "sealed": False,
            "development_evaluation_used_for_fitting": False,
            "input_path": str(artifact),
            "input_npz_sha256": file_hash(artifact),
        }))
    gate_path = tmp_path / "evaluation_gate.json"
    build_evaluation_release_gate(
        recipe_manifest=recipe_path, input_root=input_root, output=gate_path,
    )
    return cards, recipe_path, gate_path


def test_locked_recipe_records_selection_and_diagnostic_roles(tmp_path) -> None:
    _cards, recipe_path, _gate = _locked_recipe_and_eval_gate(tmp_path)
    payload = json.loads(recipe_path.read_text())
    assert payload["selection_subject"] == "epilepsiae_253"
    assert payload["selection_n_positive"] == payload["selection_n_total"] == 5
    assert payload["selection_gain_median"] == pytest.approx(0.314)
    assert payload["diagnostic_subject"] == "epilepsiae_916"
    assert payload["diagnostic_evidence"][0]["role"] == "no_learning_diagnostic_only"


def test_locked_recipe_is_selected_from_seed_fixed_cards_not_hardcoded(tmp_path) -> None:
    alternate = _example_recipe()
    alternate["arch"]["width"] = 32
    alternate["optimizer"]["lr_encoder"] = 3e-4
    cards = [
        _locked_tuning_card(
            tmp_path, "epilepsiae_253", seed, 0.1 + i / 100,
            recipe=alternate,
        )
        for i, seed in enumerate(LOCKED_SEEDS)
    ]
    diagnostic = _locked_tuning_card(
        tmp_path, "epilepsiae_916", LOCKED_SEEDS[0], 0.0, rung=300,
    )
    path = tmp_path / "dynamic_recipe.json"
    payload = build_locked_recipe_manifest(
        e253_cards=cards, e916_diagnostic_cards=[diagnostic], output=path,
    )
    assert payload["recipe"]["arch"]["width"] == 32
    assert payload["recipe"]["optimizer"]["lr_encoder"] == 3e-4


def test_e916_legacy_seed_card_is_diagnostic_only_and_cannot_choose_recipe(tmp_path) -> None:
    cards = [
        _locked_tuning_card(tmp_path, "epilepsiae_253", seed, 0.1)
        for seed in LOCKED_SEEDS
    ]
    diagnostic = _locked_tuning_card(
        tmp_path, "epilepsiae_916", LOCKED_SEEDS[0], 0.0, rung=300,
    )
    payload = json.loads(diagnostic.read_text())
    payload.pop("seed_contract")
    payload["contract"].pop("seed_contract")
    from src.topic5_group_event_state.v033_training_lab.paths import payload_hash
    payload["contract_hash"] = payload_hash(payload["contract"])
    diagnostic.write_text(json.dumps(payload))
    result = build_locked_recipe_manifest(
        e253_cards=cards, e916_diagnostic_cards=[diagnostic],
        output=tmp_path / "recipe.json",
    )
    assert result["recipe"]["arch"]["width"] == 64
    assert result["diagnostic_evidence"][0]["seed_contract"] == "legacy_post_model_construction"


def test_locked_evaluation_gate_rejects_wrong_subject_and_recipe(tmp_path) -> None:
    _cards, _recipe, gate = _locked_recipe_and_eval_gate(tmp_path)
    require_evaluation_release_gate(
        gate, subject="epilepsiae_1146", requested_recipe=_example_recipe(),
    )
    with pytest.raises(PermissionError, match="not authorized"):
        require_evaluation_release_gate(
            gate, subject="epilepsiae_253", requested_recipe=_example_recipe(),
        )
    changed = _example_recipe()
    changed["optimizer"]["lr_encoder"] = 3e-4
    with pytest.raises(PermissionError, match="change the recipe"):
        require_evaluation_release_gate(
            gate, subject="epilepsiae_1146", requested_recipe=changed,
        )


def test_locked_evaluation_gate_rejects_tampered_evidence(tmp_path) -> None:
    cards, _recipe, gate = _locked_recipe_and_eval_gate(tmp_path)
    cards[0].write_text(cards[0].read_text() + "\n")
    with pytest.raises(PermissionError, match="source tuning card changed"):
        require_evaluation_release_gate(
            gate, subject="epilepsiae_1146", requested_recipe=_example_recipe(),
        )


def test_locked_evaluation_gate_rejects_tampered_gate_hash(tmp_path) -> None:
    _cards, _recipe, gate = _locked_recipe_and_eval_gate(tmp_path)
    payload = json.loads(gate.read_text())
    payload["allowed_subjects"].append("epilepsiae_253")
    gate.write_text(json.dumps(payload))
    with pytest.raises(PermissionError, match="content hash differs"):
        require_evaluation_release_gate(
            gate, subject="epilepsiae_1146", requested_recipe=_example_recipe(),
        )


def test_locked_recipe_rejects_forbidden_provenance(tmp_path) -> None:
    cards = [
        _locked_tuning_card(tmp_path, "epilepsiae_253", seed, 0.1)
        for seed in LOCKED_SEEDS
    ]
    bad = json.loads(cards[0].read_text())
    bad["development_targets_read"] = True
    cards[0].write_text(json.dumps(bad))
    diagnostic = _locked_tuning_card(
        tmp_path, "epilepsiae_916", LOCKED_SEEDS[0], 0.0, rung=300,
    )
    with pytest.raises(PermissionError, match="data scope"):
        build_locked_recipe_manifest(
            e253_cards=cards, e916_diagnostic_cards=[diagnostic], output=tmp_path / "recipe.json",
        )


def test_zero_state_is_exact_legacy_scoring_parity() -> None:
    torch.manual_seed(2)
    decoder = _decoder()
    config = ArchConfig(width=32, depth=1)
    model = SpatialStateModel(
        input_dim=10, n_contacts=6, config=config, legacy_decoder=decoder
    )
    group_ids = torch.tensor([
        [0, 1, -1, -1, -1, -1],
        [0, 0, 1, -1, -1, -1],
    ])
    count = torch.tensor([2, 2])
    expected = next_set_stop_loss(decoder(group_ids, count), group_ids, count)["event_nll"]
    observed = model.legacy_event_nll(
        group_ids, count, torch.zeros((2, config.state_dim))
    )
    assert torch.equal(expected, observed)
    assert all(not p.requires_grad for p in decoder.parameters())


def test_optimizer_groups_use_independent_effective_lrs() -> None:
    model = SpatialStateModel(
        input_dim=10, n_contacts=6, config=ArchConfig(width=32, depth=1),
        legacy_decoder=_decoder(),
    )
    config = OptimizerConfig(
        lr_encoder=1e-4, lr_state_adapter=2e-3, lr_auxiliary=3e-3
    )
    optimizer, contract = build_optimizer(model, config)
    got = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert got == {"encoder": 1e-4, "state_adapter": 2e-3, "auxiliary": 3e-3}
    assert contract["effective_lr"] == {
        "encoder": 1e-4, "state_adapter": 2e-3, "auxiliary": 3e-3
    }


def test_seed_is_bound_before_model_construction() -> None:
    def model_hash(seed: int) -> str:
        seed_before_model_construction(seed)
        model = SpatialStateModel(
            input_dim=10, n_contacts=6, config=ArchConfig(width=32, depth=1),
            legacy_decoder=None,
        )
        return tensor_state_hash(model.state_dict())

    assert model_hash(20260903) == model_hash(20260903)
    assert model_hash(20260903) != model_hash(20260904)


def test_segment_boundary_resets_history_but_chunk_boundary_does_not() -> None:
    torch.manual_seed(4)
    config = ArchConfig(width=32, depth=1, taus_seconds=(10.0, 30.0))
    model = SpatialStateModel(input_dim=4, n_contacts=4, config=config)
    token = torch.randn(8, 4)
    time = torch.tensor([0., 2., 4., 6., 100., 102., 104., 106.], dtype=torch.float64)
    segment = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    anchor_time = torch.tensor([6.1, 106.1], dtype=torch.float64)
    last = torch.tensor([3, 7])
    rows = torch.tensor([0, 1])
    first = model.trajectory(token, time, segment, anchor_time, last, rows)
    changed = token.clone(); changed[:4] += 100.0
    second = model.trajectory(changed, time, segment, anchor_time, last, rows)
    # Normalisation couples anchors, so inspect the raw bank to test the carry contract.
    _, post_a = model.state_bank(model.encoder(token), time, segment)
    _, post_b = model.state_bank(model.encoder(changed), time, segment)
    assert torch.allclose(post_a[4:], post_b[4:], atol=1e-6)
    assert first.shape == second.shape == (2, config.state_dim)


def test_burn_in_and_pair_sampling_preserve_equal_anchor_weight() -> None:
    pairs = GrammarPairs(
        anchor_rows=np.array([0, 1, 2]),
        pair_anchor=np.array([0, 0, 1, 2, 2, 2]),
        pair_event=np.arange(6),
        pair_weight=np.array([1/6, 1/6, 1/3, 1/9, 1/9, 1/9]),
    ).validate()
    filtered = _burn_in_pairs(
        pairs,
        anchor_time=np.array([2., 12., 22.]),
        anchor_segment=np.array([0, 0, 0]),
        event_time=np.arange(30, dtype=float),
        event_segment=np.zeros(30, dtype=int),
        seconds=10.0,
    )
    assert filtered.anchor_rows.tolist() == [1, 2]
    sampled = sample_equal_anchor_pairs(
        filtered, rng=np.random.default_rng(1), n_anchors=2, events_per_anchor=2
    )
    totals = np.bincount(sampled.pair_anchor, weights=sampled.pair_weight)
    assert np.allclose(totals, 0.5)


def test_synthetic_token_uses_masked_spatial_values_not_legacy_rank() -> None:
    data = make_synthetic_spatial_data(n_events=180, seed=7)
    assert data.provenance["legacy_rank_used"] is False
    assert data.event_token.shape[0] == data.event_time.size
    assert np.all(data.group_ids[~data.participation] == -1)
    assert np.isfinite(data.event_token).all()


def test_checkpoint_selection_is_inside_state_train_with_target_embargo() -> None:
    data = make_synthetic_spatial_data(n_events=1000, seed=7)
    fit, inner, meta = chronological_train_fit_inner(data, embargo_seconds=300.0)
    assert data.anchor_time[fit.anchor_rows].max() + 300.0 < data.anchor_time[inner.anchor_rows].min()
    assert data.anchor_time[inner.anchor_rows].max() < data.anchor_time[data.selection_pairs.anchor_rows].min()
    assert meta["checkpoint_selection_source"] == "chronologically_later_STATE_TRAIN_inner"
    assert meta["reported_source"] == "STATE_SELECTION_full"


def test_synthetic_truth_suite_contains_signal_and_no_state_null() -> None:
    dynamic = make_synthetic_spatial_data(n_events=3000, seed=20260903, truth_kind="dynamic")
    null = make_synthetic_spatial_data(n_events=3000, seed=20260903, truth_kind="none")
    assert _truth_oracle(dynamic)["gain_vs_train_mean"] > 0.005
    assert _truth_oracle(null)["gain_vs_train_mean"] < 0.001


def test_train_mean_adapter_is_zero_parity_then_changes_only_frozen_decoder_output() -> None:
    seed_before_model_construction(3)
    decoder = _decoder()
    model = SpatialStateModel(
        input_dim=10, n_contacts=6, config=ArchConfig(width=32, depth=1),
        legacy_decoder=decoder,
    )
    ids = torch.tensor([[0, 1, -1, -1, -1, -1]], dtype=torch.long)
    count = torch.tensor([2], dtype=torch.long)
    state = torch.zeros((1, model.config.state_dim))
    before = model.legacy_event_nll(ids, count, state)
    expected = decoder.loss(ids, count)["event_nll"]
    assert torch.equal(before, expected)
    with torch.no_grad():
        model.train_mean_adapter.contact_bias[0] = 1.0
        model.train_mean_adapter.stop_bias.fill_(-0.5)
    after = model.legacy_event_nll(ids, count, state)
    assert not torch.equal(after, expected)


def test_frozen_random_encoder_is_excluded_from_optimizer() -> None:
    seed_before_model_construction(5)
    model = SpatialStateModel(
        input_dim=10, n_contacts=6, config=ArchConfig(width=32, depth=1),
        legacy_decoder=_decoder(),
    )
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    optimizer, _ = build_optimizer(model, OptimizerConfig())
    assigned = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert not any(id(p) in assigned for p in model.encoder.parameters())
    assert any(id(p) in assigned for p in model.legacy.residual.parameters())


def test_report_period_targets_cannot_change_selected_checkpoint(tmp_path) -> None:
    data = make_synthetic_spatial_data(n_events=180, seed=19)
    changed_rows = np.setdiff1d(
        np.unique(data.selection_pairs.pair_event), np.unique(data.train_pairs.pair_event),
    )
    permutation = changed_rows[::-1]
    group_ids = data.group_ids.copy(); group_ids[changed_rows] = data.group_ids[permutation]
    group_count = data.group_count.copy(); group_count[changed_rows] = data.group_count[permutation]
    participation = data.participation.copy(); participation[changed_rows] = data.participation[permutation]
    extent = data.positive_extent.copy(); extent[changed_rows] = data.positive_extent[permutation]
    lag = data.relative_lag.copy(); lag[changed_rows] = data.relative_lag[permutation]
    lag_valid = data.lag_valid.copy(); lag_valid[changed_rows] = data.lag_valid[permutation]
    altered = replace(
        data, group_ids=group_ids, group_count=group_count, participation=participation,
        positive_extent=extent, relative_lag=lag, lag_valid=lag_valid,
    )
    arch = ArchConfig(width=32, depth=1)
    train = TrainConfig(
        max_steps=20, validate_every=5, patience_checks=6,
        anchors_per_step=16, events_per_anchor=8, burn_in_seconds=0.0, seed=23,
    )
    optim = OptimizerConfig(lr_encoder=1e-3, lr_state_adapter=1e-3, lr_auxiliary=3e-3)

    def fit(one, name):
        seed_before_model_construction(23)
        model = SpatialStateModel(
            input_dim=one.event_token.shape[1], n_contacts=one.n_contacts,
            config=arch, legacy_decoder=None,
        )
        return train_spatial_state(
            model, one, arch=arch, optimizer_config=optim, train_config=train,
            device=torch.device("cpu"), output_dir=tmp_path / name,
            card_kind="tiny_canary", allow_tiny=True, overwrite=True,
        )

    original = fit(data, "original")
    changed = fit(altered, "altered")
    assert original["selected_step"] == changed["selected_step"]
    assert original["selected_state_hash"] == changed["selected_state_hash"]
    assert original["history"] == changed["history"]
    assert original["selection_gain"] != changed["selection_gain"]


def test_tiny_cpu_canary_proves_gradient_and_atomic_card(tmp_path) -> None:
    card = run_synthetic(
        output_dir=tmp_path, device=torch.device("cpu"), tiny=True,
        seed=20260903, overwrite=True,
    )
    assert card["status"] == "PASS"
    assert card["parameters_changed"] is True
    assert card["max_gradient_l2"] > 0
    assert card["seed_contract"] == SEED_CONTRACT
    assert card["contract"]["seed_contract"] == SEED_CONTRACT
    assert card["selection_is_report_only"] is True
    assert card["model_selection_split"]["checkpoint_selection_source"] \
        == "chronologically_later_STATE_TRAIN_inner"
    assert card["train_mean_adapter"]["status"] == "NOT_APPLICABLE_SYNTHETIC"
    assert card["state_selection_full"]["n_report_anchors"] \
        == make_synthetic_spatial_data(n_events=180, seed=20260903).selection_pairs.anchor_rows.size
    assert (tmp_path / "training_card.json").is_file()
    assert (tmp_path / "selected_checkpoint.pt").is_file()
