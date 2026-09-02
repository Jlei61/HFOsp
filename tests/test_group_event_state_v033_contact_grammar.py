from __future__ import annotations

import copy

import numpy as np
import torch

from src.topic5_group_event_state.v033_training_lab.contact_grammar import (
    CalibrationData,
    LegacyContactGrammar,
    _group_count,
    build_subject_grammar,
    direct_legacy_score,
    load_calibrated_legacy_grammar,
    tensor_state_hash,
)
from src.topic5_rank_distribution import FullHistorySequenceGRU, next_set_stop_loss


def _toy_model() -> LegacyContactGrammar:
    torch.manual_seed(4)
    base = FullHistorySequenceGRU(
        8, hidden_size=7, contact_embedding_dim=6,
        contact_encoder_hidden=9, local_offset_dim=3,
    )
    features = np.random.default_rng(3).normal(size=(5, 8)).astype(np.float32)
    return LegacyContactGrammar(base, features, np.ones(5, bool), local_offset_dim=3)


def test_wrapper_is_exactly_the_legacy_next_set_stop_score():
    model = _toy_model()
    groups = torch.tensor([[0, 0, 1, -1, -1], [0, -1, 1, 1, -1]])
    count = torch.tensor([2, 2])
    outputs = model(groups, count)
    wrapped = direct_legacy_score(model, groups, count)
    direct = next_set_stop_loss(outputs, groups, count)
    assert torch.equal(wrapped["step_mask"], direct["step_mask"])
    assert torch.allclose(wrapped["event_nll"], direct["event_nll"], atol=0, rtol=0)


def test_tied_group_contract_is_old_any_member_not_exact_subset():
    model = _toy_model()
    groups = torch.tensor([[0, 0, -1, -1, -1]])
    count = _group_count(groups.numpy())
    score = model.loss(groups, torch.from_numpy(count))["event_nll"]
    # The legacy score remains finite for a tied set and uses one logsumexp
    # numerator for that set.  This test prevents an exact-subset scorer from
    # being silently substituted into the v0.3.3 calibration runner.
    assert torch.isfinite(score).all()


def test_e916_builder_imports_hyperparameters_but_no_template_weights(monkeypatch):
    torch.manual_seed(9)
    template = FullHistorySequenceGRU(
        8, hidden_size=5, contact_embedding_dim=4,
        contact_encoder_hidden=6, local_offset_dim=2,
    )
    payload = {
        "control": "full_history_gru",
        "model_kwargs": {
            "hidden_size": 5, "contact_embedding_dim": 4,
            "contact_encoder_hidden": 6, "local_offset_dim": 2,
        },
        "model_state": copy.deepcopy(template.state_dict()),
        "heldout_local_offset": torch.full((3, 2), 99.0),
        "heldout_subject": "epilepsiae_253",
        "ictal_target_read": False,
    }
    monkeypatch.setattr(
        "src.topic5_group_event_state.v033_training_lab.contact_grammar._template_payload",
        lambda _path: payload,
    )
    monkeypatch.setattr(
        "src.topic5_group_event_state.v033_training_lab.contact_grammar._sha256",
        lambda _path: "0" * 64,
    )
    data = CalibrationData(
        subject="epilepsiae_916", contact_names=("A1", "A2", "A3"),
        contact_features=np.zeros((3, 8), np.float32),
        contact_mask=np.ones(3, bool), group_ids=np.zeros((4, 3), np.int64),
        group_count=np.ones(4, np.int64), fit_rows=np.arange(2),
        inner_rows=np.arange(2, 4), partition={}, feature_provenance={},
    )
    grammar, provenance = build_subject_grammar(data, seed=123)
    assert provenance["base_weight_source"] == "deterministic_random_initialization"
    assert provenance["other_patient_weights_loaded"] is False
    assert torch.count_nonzero(grammar.local_offset).item() == 0
    assert any(
        not torch.equal(grammar.base.state_dict()[k], template.state_dict()[k])
        for k in template.state_dict()
        if template.state_dict()[k].dtype.is_floating_point
    )


def test_e253_uses_heldout_base_but_discards_old_local_offset(monkeypatch):
    torch.manual_seed(11)
    template = FullHistorySequenceGRU(
        8, hidden_size=5, contact_embedding_dim=4,
        contact_encoder_hidden=6, local_offset_dim=2,
    )
    payload = {
        "control": "full_history_gru",
        "model_kwargs": {
            "hidden_size": 5, "contact_embedding_dim": 4,
            "contact_encoder_hidden": 6, "local_offset_dim": 2,
        },
        "model_state": copy.deepcopy(template.state_dict()),
        "heldout_local_offset": torch.full((3, 2), 99.0),
        "heldout_subject": "epilepsiae_253",
        "ictal_target_read": False,
    }
    monkeypatch.setattr(
        "src.topic5_group_event_state.v033_training_lab.contact_grammar._template_payload",
        lambda _path: payload,
    )
    monkeypatch.setattr(
        "src.topic5_group_event_state.v033_training_lab.contact_grammar._sha256",
        lambda _path: "0" * 64,
    )
    data = CalibrationData(
        subject="epilepsiae_253", contact_names=("A1", "A2", "A3"),
        contact_features=np.zeros((3, 8), np.float32),
        contact_mask=np.ones(3, bool), group_ids=np.zeros((4, 3), np.int64),
        group_count=np.ones(4, np.int64), fit_rows=np.arange(2),
        inner_rows=np.arange(2, 4), partition={}, feature_provenance={},
    )
    grammar, provenance = build_subject_grammar(data, seed=123)
    assert provenance["base_weight_source"] == "locked_leave_one_patient_out_shared_base"
    assert all(
        torch.equal(grammar.base.state_dict()[k], template.state_dict()[k])
        for k in template.state_dict()
    )
    assert torch.count_nonzero(grammar.local_offset).item() == 0


def test_checkpoint_loader_rejects_scoring_upgrade(tmp_path):
    path = tmp_path / "wrong.pt"
    torch.save({
        "format": "group_event_contact_grammar_v0_3_3_legacy_scoring",
        "scoring_contract": {
            "name": "legacy_next_set_or_STOP", "exact_subset_likelihood": True,
        },
    }, path)
    try:
        load_calibrated_legacy_grammar(path)
    except ValueError as exc:
        assert "scoring contract" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("exact-subset drift was accepted")


def test_calibrated_artifact_reload_is_logit_and_loss_identical(tmp_path):
    model = _toy_model().eval()
    groups = torch.tensor([[0, 0, 1, -1, -1], [0, -1, 1, 1, -1]])
    count = torch.tensor([2, 2])
    with torch.no_grad():
        before_outputs = model(groups, count)
        before_loss = model.loss(groups, count)["event_nll"]
    payload = {
        "format": "group_event_contact_grammar_v0_3_3_legacy_scoring",
        "model_state": model.base.state_dict(),
        "local_offset": model.local_offset.detach(),
        "contact_features": model.contact_features.detach(),
        "contact_mask": model.contact_mask.detach(),
        "architecture_hyperparameters": {
            "hidden_size": model.base.hidden_size,
            "contact_embedding_dim": model.base.contact_embedding_dim,
            "contact_encoder_hidden": 9,
            "local_offset_dim": model.base.local_offset_dim,
        },
        "scoring_contract": {
            "name": "legacy_next_set_or_STOP", "exact_subset_likelihood": False,
        },
        "calibrated_grammar_frozen": True,
        "base_tensor_hash": tensor_state_hash(model.state_dict()),
    }
    path = tmp_path / "grammar.pt"
    torch.save(payload, path)
    loaded, _ = load_calibrated_legacy_grammar(path)
    with torch.no_grad():
        after_outputs = loaded(groups, count)
        after_loss = loaded.loss(groups, count)["event_nll"]
    for key in ("contact_logits", "stop_logits", "candidate_mask"):
        assert torch.equal(before_outputs[key], after_outputs[key])
    assert torch.equal(before_loss, after_loss)
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
