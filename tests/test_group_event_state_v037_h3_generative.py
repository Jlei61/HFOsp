from __future__ import annotations

import numpy as np
import torch

from src.topic5_group_event_state.v037.h3_generative import (
    H3Config,
    H3Data,
    PhysiologicalGenerativeState,
    _causal_delayed_inputs,
    train_h3_subject,
)
from src.topic5_group_event_state.v037.h3_persistent import (
    _fit_count_edge,
    _fit_gaussian_edge,
)


def _model() -> PhysiologicalGenerativeState:
    torch.manual_seed(4)
    model = PhysiologicalGenerativeState(
        3, 2, 2, 2, H3Config(seed=4, taus_seconds=(600.0, 3600.0), channels_per_tau=1)
    )
    with torch.no_grad():
        model.count_feedback_to_count.weight.fill_(0.2)
        model.mark_feedback_to_count.weight.fill_(0.1)
    return model


def test_h3_m0_is_event_free_and_edges_are_nested() -> None:
    model = _model()
    time = torch.arange(5, dtype=torch.float64) * 300.0
    segment = torch.zeros(5, dtype=torch.long)
    context = torch.zeros(5, 3)
    count = torch.ones(5, 1)
    mark = torch.ones(5, 2)
    zero_count = torch.zeros_like(count); zero_mark = torch.zeros_like(mark)
    m0 = model(time, segment, context, count, mark, "M0_common_drive")[0]
    m0_zero = model(time, segment, context, zero_count, zero_mark, "M0_common_drive")[0]
    assert torch.allclose(m0, m0_zero)
    m0_output = model(time, segment, context, count, mark, "M0_common_drive")[1]
    with torch.no_grad():
        saved = model.count_feedback_to_count.weight.clone()
        model.count_feedback_to_count.weight.fill_(99.0)
    m0_large_unused_jump = model(time, segment, context, count, mark, "M0_common_drive")[1]
    assert torch.allclose(m0_output, m0_large_unused_jump)
    with torch.no_grad():
        model.count_feedback_to_count.weight.copy_(saved)
    m1 = model(time, segment, context, count, mark, "M1_count_feedback")[0]
    assert not torch.allclose(m1, m0)
    m2 = model(time, segment, context, count, mark, "M2_mark_feedback")[0]
    assert not torch.allclose(m2, m1)


def test_h3_jump_affects_only_later_grid_blocks() -> None:
    model = _model()
    time = torch.arange(4, dtype=torch.float64) * 300.0
    segment = torch.zeros(4, dtype=torch.long)
    context = torch.zeros(4, 3)
    count = torch.zeros(4, 1); count[1] = 1.0
    mark = torch.zeros(4, 2)
    state = model(time, segment, context, count, mark, "M1_count_feedback")[0]
    assert torch.allclose(state[:2], torch.zeros_like(state[:2]))
    assert torch.linalg.vector_norm(state[2]) > 0


def test_h3_emission_never_reads_contemporaneous_context() -> None:
    model = _model()
    time = torch.arange(4, dtype=torch.float64) * 300.0
    segment = torch.zeros(4, dtype=torch.long)
    count = torch.zeros(4, 1)
    mark = torch.zeros(4, 2)
    context = torch.zeros(4, 3)
    reference = model(time, segment, context, count, mark, "M0_common_drive")
    perturbed = context.clone(); perturbed[2, 0] = 100.0
    changed = model(time, segment, perturbed, count, mark, "M0_common_drive")
    # Row 2 is still the prediction made at the end of row 1.  The newly
    # observed context can first alter an emission at row 3.
    for ref, new in zip(reference[1:], changed[1:]):
        assert torch.allclose(ref[2], new[2])
    assert not torch.allclose(reference[1][3], changed[1][3])


def test_h3_low_rank_mark_input_can_predict_full_mark_target() -> None:
    model = PhysiologicalGenerativeState(
        3, 2, 5, 2, H3Config(seed=5, taus_seconds=(600.0, 3600.0), channels_per_tau=1)
    )
    time = torch.arange(4, dtype=torch.float64) * 300.0
    output = model(
        time, torch.zeros(4, dtype=torch.long), torch.zeros(4, 3),
        torch.zeros(4, 1), torch.zeros(4, 2), "M2_mark_feedback",
    )
    assert output[2].shape == (4, 5)


def test_h3_causal_delay_uses_only_older_same_segment_rows() -> None:
    n = 8
    data = H3Data(
        subject="synthetic", time=np.arange(n, dtype=float) * 300.0,
        segment=np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
        phase=np.asarray(["FIT"] * n), exposure_seconds=np.full(n, 300.0),
        count=np.arange(n, dtype=np.float32), count_input=np.arange(n, dtype=np.float32)[:, None],
        grammar_input=np.arange(n * 2, dtype=np.float32).reshape(n, 2),
        grammar_target=np.zeros((n, 2), dtype=np.float32), grammar_valid=np.ones(n, bool),
        background_target=np.zeros((n, 2), dtype=np.float32), background_valid=np.ones(n, bool),
        context=np.zeros((n, 1), dtype=np.float32), context_names=("x",), transforms={},
    )
    count, grammar, valid = _causal_delayed_inputs(data, 600.0)
    assert np.array_equal(valid, [False, False, True, True, False, False, True, True])
    assert count[2, 0] == data.count_input[0, 0]
    assert grammar[7, 0] == data.grammar_input[5, 0]


def test_h3_card_does_not_claim_equal_complete_model_capacity(tmp_path) -> None:
    rng = np.random.default_rng(17)
    n = 30
    phase = np.asarray(["FIT"] * 18 + ["INNER"] * 6 + ["SELECTION"] * 6)
    data = H3Data(
        subject="synthetic_contract", time=np.arange(n, dtype=float) * 300.0,
        segment=np.zeros(n, dtype=np.int64), phase=phase,
        exposure_seconds=np.full(n, 300.0), count=rng.poisson(2.0, n).astype(np.float32),
        count_input=rng.normal(size=(n, 1)).astype(np.float32),
        grammar_input=rng.normal(size=(n, 2)).astype(np.float32),
        grammar_target=rng.normal(size=(n, 5)).astype(np.float32),
        grammar_valid=np.ones(n, bool),
        background_target=rng.normal(size=(n, 2)).astype(np.float32),
        background_valid=np.ones(n, bool), context=rng.normal(size=(n, 3)).astype(np.float32),
        context_names=("a", "b", "c"), transforms={},
    )
    card = train_h3_subject(
        data,
        H3Config(seed=17, taus_seconds=(600.0, 3600.0), channels_per_tau=1,
                 max_steps=1, validate_every=1, patience_checks=1),
        device=torch.device("cpu"), out_dir=tmp_path,
    )
    assert card["format"].endswith("_v3")
    assert card["common_core_and_intercept_frozen_across_nested_models"] is True
    assert card["feedback_edges_add_zero_bias_source_specific_readouts"] is True
    assert card["complete_models_have_equal_parameter_count"] is False
    assert card["edge_estimand_scope"]["long_horizon_feedback_status"] == "NOT_ESTIMATED_IN_V0_3_7_PRIMARY"


def test_persistent_gaussian_edge_recovers_signal_and_keeps_exact_null_zero() -> None:
    x = torch.linspace(-2.0, 2.0, 180)[:, None]
    x = torch.cat((x, torch.sin(2.0 * x), torch.cos(3.0 * x)), dim=1)
    base = torch.zeros(180, 2)
    fit = torch.arange(0, 120); inner = torch.arange(120, 180)
    target = x @ torch.tensor([[0.8, -0.4], [0.2, 0.5], [-0.3, 0.1]])
    weight, audit = _fit_gaussian_edge(x, target, base, fit, inner)
    assert audit["status"] == "ESTIMATED"
    assert audit["selected_zero_edge"] is False
    assert audit["penalty_grid_diagnostics"]["grid_upper_edge"] == 1e4
    assert audit["penalty_grid_diagnostics"]["saturated_at_upper_edge"] is False
    assert torch.mean((x[inner] @ weight.T - target[inner]).square()) < 1e-4

    null_weight, null_audit = _fit_gaussian_edge(x, base, base, fit, inner)
    assert null_audit["selected_zero_edge"] is True
    assert null_audit["penalty_grid_diagnostics"]["selected_zero_edge"] is True
    assert null_audit["penalty_grid_diagnostics"]["saturated_at_upper_edge"] is False
    assert torch.count_nonzero(null_weight) == 0


def test_persistent_count_edge_recovers_signal_and_keeps_constant_null_zero() -> None:
    pattern = torch.tensor([-1.0, 1.0]).repeat(120)[:, None]
    base_log_rate = torch.full((240,), float(np.log(6.0)))
    exposure = torch.full((240,), 300.0)
    fit = torch.arange(0, 160); inner = torch.arange(160, 240)
    count = torch.where(pattern[:, 0] > 0, 12.0, 2.0)
    weight, audit = _fit_count_edge(
        pattern, base_log_rate, count, exposure, torch.tensor(3.0),
        fit, inner, 300.0,
    )
    assert audit["status"] == "ESTIMATED"
    assert audit["selected_zero_edge"] is False
    assert audit["penalty_grid_diagnostics"]["grid_upper_edge"] == 1e4
    assert audit["penalty_grid_diagnostics"]["saturated_at_upper_edge"] is False
    assert weight[0, 0] > 0.2

    constant = torch.full((240,), 6.0)
    null_weight, null_audit = _fit_count_edge(
        pattern, base_log_rate, constant, exposure, torch.tensor(3.0),
        fit, inner, 300.0,
    )
    assert null_audit["selected_zero_edge"] is True
    assert null_audit["penalty_grid_diagnostics"]["selected_zero_edge"] is True
    assert null_audit["penalty_grid_diagnostics"]["saturated_at_upper_edge"] is False
    assert torch.count_nonzero(null_weight) == 0
