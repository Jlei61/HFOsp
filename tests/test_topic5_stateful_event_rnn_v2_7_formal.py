import json
from pathlib import Path

import torch

from scripts import run_topic5_stateful_event_rnn_v2_7_formal as formal
from src.topic5_stateful_event_rnn_v2_6 import (
    StatefulProfile,
    StatefulTrainingTrace,
)


def test_formal_fit_adapter_uses_repaired_v27(monkeypatch):
    observed = {}

    def fake_fit(profile, datasets, encoder, config, scales, seed):
        observed["args"] = (profile, datasets, encoder, config, scales, seed)
        return "fit", 2.0

    monkeypatch.setattr(formal.selection, "fit_profile", fake_fit)
    result = formal.fit_profile("ignored", 1, 2, 3, 4, 5, 6)
    assert result == ("fit", 2.0)
    assert observed["args"] == (1, 2, 3, 4, 5, 6)


def test_frozen_validation_is_verified():
    config_path = formal.DEFAULT_CONFIG.resolve()
    config = __import__("yaml").safe_load(config_path.read_text())
    output = formal.ROOT / config["output_root"]
    state = formal.verify_frozen(config_path, output)
    assert state["status"] == "ALL_PATIENT_VALIDATION_PROFILES_FROZEN"
    assert state["test_results_read_during_selection"] is False


def test_checkpoint_contract_is_v27(tmp_path: Path):
    class Value:
        pass

    fitted = Value()
    fitted.trained_model = torch.nn.Linear(1, 1)
    fitted.nested_model = torch.nn.Linear(1, 1)
    fitted.feature_mean = [0.0]
    fitted.feature_scale = [1.0]
    fitted.profile = StatefulProfile(
        cell="rnn", hidden_size=1, num_layers=1, dropout=0.0,
        optimizer="adamw", learning_rate=0.001, weight_decay=0.0,
        normalization="zscore", tbptt_length=1, update_chunks=1,
        gradient_clip=1.0, participation_weight=1.0,
        input_layer_norm=False, hidden_layer_norm=False,
    )
    fitted.trace = StatefulTrainingTrace(
        train_loss=[], validation_trained_propagation=[],
        validation_nested_propagation=[], gradient_norm_mean=[],
        gradient_norm_max=[], clipped_fraction=[], state_norm_mean=[],
        best_trained_epoch=0, best_nested_epoch=-1, stopped_epoch=0,
        finite=True,
    )
    encoder = Value()
    encoder.centers = encoder.feature_mean = encoder.feature_scale = [0.0]
    encoder.rank_prior = [0.5]
    encoder.n_modes = 2
    ewma = Value()
    ewma.decay = 0.9
    ewma.alpha = 1.0
    ewma.feature_mean = [0.0]
    ewma.feature_scale = [1.0]
    ewma.ridge = Value()
    ewma.ridge.coef_ = [[1.0]]
    ewma.ridge.intercept_ = [0.0]
    path = tmp_path / "checkpoint.pt"
    formal.save_checkpoint(
        path, fitted, encoder, ewma, "subject", 1,
        {"maximum_epochs": 2, "minimum_epochs": 1, "patience": 1},
    )
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["contract"] == "topic5_stateful_event_sequence_rnn_v2_7"
    assert checkpoint["parent_v2_6"]["frozen_status"] == (
        "ALL_PATIENT_VALIDATION_PROFILES_FROZEN"
    )
