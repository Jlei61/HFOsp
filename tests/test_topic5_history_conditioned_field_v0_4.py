import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import spearmanr

from scripts.summarize_topic5_history_conditioned_field_v0_4 import (
    _ensemble_true,
    _exact_signed_rank,
)
from src.topic5_history_rnn import TimeDecayHistoryGRU
from src.topic5_static_anchored_history_residual import (
    DualCandidateResidualHead,
    compose_static_residual,
    fixed_time_aware_summary,
    patient_balanced_soft_maxab,
    run_history_to_cutoff,
    safe_unit_residual,
    soft_maxab_score,
    unit_eps,
)
from src.topic5_static_ab import load_frozen_static_scaffold


def test_zero_gain_exactly_returns_static_and_tiny_residual_is_zero():
    static = torch.tensor([3.0, -1.0, 2.0, 0.5])
    residual = torch.tensor([0.2, -0.1, 0.5, -0.3])
    torch.testing.assert_close(
        compose_static_residual(static, residual, 0.0), unit_eps(static), rtol=0, atol=0
    )
    tiny = torch.full((4,), 1e-12)
    torch.testing.assert_close(
        safe_unit_residual(tiny, norm_threshold=1e-6),
        torch.zeros_like(tiny),
        rtol=0,
        atol=0,
    )


def test_soft_maxab_is_ab_and_sign_invariant():
    a = torch.tensor([-2.0, -1.0, 1.0, 3.0])
    b = torch.tensor([1.0, -2.0, 4.0, 0.0])
    target_rank = torch.tensor([1.0, 2.0, 3.0, 4.0])
    reference = soft_maxab_score(a, b, target_rank)
    torch.testing.assert_close(reference, soft_maxab_score(b, a, target_rank))
    torch.testing.assert_close(reference, soft_maxab_score(-a, -b, target_rank))


def test_fixed_time_summary_uses_frozen_two_hour_weighting():
    embedding = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    event_time = torch.tensor([0.0, 7200.0])
    summary = fixed_time_aware_summary(
        embedding, event_time, cutoff_time=7200.0, tau_hours=2.0
    )
    weight_old = np.exp(-1.0)
    expected = torch.tensor(
        [weight_old / (weight_old + 1.0), 2.0 / (weight_old + 1.0)],
        dtype=summary.dtype,
    )
    torch.testing.assert_close(summary[:2], expected)
    assert summary.shape == (10,)


def test_state_is_decayed_from_last_event_to_cutoff():
    torch.manual_seed(1)
    history = TimeDecayHistoryGRU(3, 4, initial_half_life_hours=2.0)
    embedding = torch.randn(5, 3)
    event_time = torch.arange(5, dtype=torch.float32) * 60.0
    at_last = run_history_to_cutoff(
        history, embedding, event_time, cutoff_time=event_time[-1], chunk_events=2
    )
    later = run_history_to_cutoff(
        history,
        embedding,
        event_time,
        cutoff_time=event_time[-1] + 7200.0,
        chunk_events=2,
    )
    assert torch.linalg.vector_norm(later) < torch.linalg.vector_norm(at_last)


def test_chunking_preserves_output_and_gradient():
    torch.manual_seed(7)
    first = TimeDecayHistoryGRU(3, 4, initial_half_life_hours=2.0)
    second = copy.deepcopy(first)
    event_time = torch.cumsum(torch.rand(9) * 120.0, dim=0)
    embedding_a = torch.randn(9, 3, requires_grad=True)
    embedding_b = embedding_a.detach().clone().requires_grad_(True)
    output_a = run_history_to_cutoff(
        first, embedding_a, event_time, cutoff_time=event_time[-1] + 300, chunk_events=2
    )
    output_b = run_history_to_cutoff(
        second, embedding_b, event_time, cutoff_time=event_time[-1] + 300, chunk_events=99
    )
    torch.testing.assert_close(output_a, output_b, rtol=1e-6, atol=1e-7)
    output_a.square().sum().backward()
    output_b.square().sum().backward()
    torch.testing.assert_close(embedding_a.grad, embedding_b.grad, rtol=1e-5, atol=1e-7)
    for parameter_a, parameter_b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(parameter_a.grad, parameter_b.grad, rtol=1e-5, atol=1e-7)


def test_gain_is_near_static_without_sigmoid_saturation():
    head = DualCandidateResidualHead(4, 3, initial_gain=1e-3)
    torch.testing.assert_close(head.gains, torch.full((2,), 1e-3))
    head.gains.sum().backward()
    assert torch.all(torch.abs(head.raw_gain.grad) > 0.05)


def test_patient_loss_is_invariant_to_repeating_all_seizures():
    a = torch.tensor([-2.0, -1.0, 1.0, 3.0])
    b = torch.tensor([1.0, -2.0, 4.0, 0.0])
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    one = patient_balanced_soft_maxab([(a, b, target)])
    repeated = patient_balanced_soft_maxab([(a, b, target)] * 5)
    torch.testing.assert_close(one, repeated)


def test_primary_endpoint_and_no_retrain_sensitivity_are_frozen():
    config = json.loads(
        Path("config/topic5_history_conditioned_field_refinement_v0_4.json").read_text()
    )
    assert config["primary_target"] == "clinical_onset_[0,10]s_1-45Hz_contact_energy"
    assert "1-150Hz" in config["sensitivity_target"]
    assert "no_retrain" in config["sensitivity_target"]
    assert config["target_seeds"] == [11, 29, 47]


def test_seed_ensemble_averages_fields_before_exact_scoring():
    rows = []
    target = np.arange(6, dtype=float)
    predictions = {
        11: np.array([0, 1, 2, 3, 5, 4], dtype=float),
        29: np.array([0, 1, 2, 4, 3, 5], dtype=float),
        47: np.array([0, 1, 2, 3, 4, 5], dtype=float),
    }
    for seed, prediction in predictions.items():
        for contact, value in enumerate(prediction):
            rows.append(
                {
                    "subject": "p1",
                    "seizure_id": "s1",
                    "seizure_idx": 0,
                    "contact": f"c{contact}",
                    "model": "M3_JOINT_RNN",
                    "seed": seed,
                    "draw": -1,
                    "prediction_a": value,
                    "prediction_b": -value,
                    "target_1_45": target[contact],
                    "target_1_150": target[contact],
                }
            )
    ensemble, metrics = _ensemble_true(pd.DataFrame(rows))
    expected = np.mean(np.stack(list(predictions.values())), axis=0)
    np.testing.assert_allclose(
        ensemble.sort_values("contact").prediction_a.to_numpy(), expected
    )
    assert metrics.maxab_1_45.iloc[0] == pytest.approx(
        abs(spearmanr(expected, target).statistic)
    )


def test_exact_signed_rank_excludes_numerical_zero_ties():
    result = _exact_signed_rank(np.array([1.0, 1.0, 0.0, 5e-12]))
    assert result["n_positive"] == 2
    assert result["n_negative"] == 0
    assert result["n_tie"] == 2
    assert result["n_nonzero"] == 2
    assert result["p_two_sided_exact"] == pytest.approx(0.5)


def test_frozen_static_loader_uses_exact_contact_names(tmp_path):
    source = (
        tmp_path
        / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
    )
    source.mkdir(parents=True)
    payload = {
        "status": "ok",
        "interictal_field": {
            "status": "ok",
            "contact_order": ["B2", "A1", "C3"],
            "field_models": {
                "own_a": {"template_field": [2.0, 1.0, 3.0]},
                "own_b": {"template_field": [-2.0, -1.0, -3.0]},
            },
        },
    }
    (source / "p1.json").write_text(json.dumps(payload))
    result = load_frozen_static_scaffold(
        tmp_path, "p1", np.asarray(["A1", "X9", "C3", "B2"])
    )
    np.testing.assert_array_equal(result["scaffold_valid"], [True, False, True, True])
    np.testing.assert_allclose(
        result["scaffold_field_a"][[0, 2, 3]], [-1.0, 1.0, 0.0]
    )
    assert np.isnan(result["scaffold_field_a"][1])
