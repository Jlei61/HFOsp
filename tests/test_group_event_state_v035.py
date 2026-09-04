from __future__ import annotations

import numpy as np
import torch
from torch import nn
from types import SimpleNamespace

from src.topic5_group_event_state.v035.contracts import RATE_TAUS_SECONDS, RateTrainConfig
from src.topic5_group_event_state.v035.dynamic_rate import DynamicRateModel, _causal_features
from src.topic5_group_event_state.v035.long_windows import (
    exposure_and_gap_count,
    merge_artificial_cuts,
    plan_horizon_specific_split,
)
from src.topic5_group_event_state.v035.background_rate import _align_fixed_background
from src.topic5_group_event_state.v035.feedback_models import (
    Block,
    _common_time_features,
    _nested_arm_admissibility,
)
from src.topic5_group_event_state.v035.functional_readouts import Endpoint, _select_and_score
from src.topic5_group_event_state.v035.full_mark_state import (
    FullMarkData,
    FullMarkTrainConfig,
    PhysicalFutureHead,
    configure_event_input_view,
    _fit_period_mean_state,
)
from src.topic5_group_event_state.model import DataShape, EncoderConfig, EventEncoder
from src.topic5_group_event_state.v035.stepwise_decoder import StepwiseAdapterConfig, StepwiseConditionedDecoder
from src.topic5_group_event_state.v035.stepwise_auxiliary import _open_loop_state
from src.topic5_group_event_state.v035 import stepwise_train
from src.topic5_group_event_state.v035.seizure_transfer import (
    _hazard_rows,
    _hazard_rows_observed_support,
    _risk_scores,
)
from src.topic5_group_event_state.v034_spatial_state.data import V035_EXTENSION_SUBJECTS


def test_dynamic_rate_static_is_exact_nested_special_case() -> None:
    cfg = RateTrainConfig(max_steps_static=1, max_steps_dynamic=1, max_steps_residual=1)
    model = DynamicRateModel(7, cfg)
    q = torch.randn(5, 7)
    with torch.no_grad():
        model.dynamic.weight.zero_()
        model.residual[-1].weight.zero_()
    base = model(q, dynamic=False, residual=False)
    assert torch.equal(base, model(q, dynamic=True, residual=False))
    assert torch.equal(base, model(q, dynamic=True, residual=True))


def test_dynamic_rate_gate_initialisation_preserves_exact_static_nesting() -> None:
    for gate in (-5.0, -1.0):
        model = DynamicRateModel(7, RateTrainConfig(residual_gate_logit=gate))
        q = torch.randn(5, 7)
        base = model(q, dynamic=False, residual=False)
        assert torch.equal(base, model(q, dynamic=True, residual=True))


def test_dynamic_rate_count_mean_scales_with_observed_exposure() -> None:
    model = DynamicRateModel(3, RateTrainConfig(horizons_seconds=(3600.0,)))
    q = torch.zeros(2, 3)
    exposure = torch.tensor([[3600.0], [1800.0]])
    mean = torch.exp(model(q, dynamic=False, residual=False, exposure_seconds=exposure))
    assert torch.allclose(mean[0], 2.0 * mean[1])


def test_long_window_merges_only_short_nonseizure_cuts_and_counts_real_holes() -> None:
    source = np.asarray([[0.0, 100.0], [120.0, 200.0], [400.0, 500.0]])
    merged, audit = merge_artificial_cuts(source, (), max_gap_seconds=60.0)
    assert np.array_equal(merged, np.asarray([[0.0, 200.0], [400.0, 500.0]]))
    assert audit.merged_artificial_cuts == 1
    # The merged intervals are state-carry support only.  Exposure is always
    # computed from the original coverage, so the bridged 20 s remain missing.
    exposure, gaps = exposure_and_gap_count(source, np.asarray([50.0]), np.asarray([450.0]))
    assert np.isclose(exposure[0], 180.0)
    assert gaps[0] == 2
    protected, audit = merge_artificial_cuts(
        source[:2], ({"onset_epoch": 105.0, "offset_epoch": 115.0},), max_gap_seconds=60.0,
    )
    assert protected.shape[0] == 2
    assert audit.protected_seizure_gaps == 1


def test_ten_minute_state_carry_does_not_turn_missing_time_into_exposure() -> None:
    source = np.asarray([[0.0, 100.0], [650.0, 800.0]])
    carry, audit = merge_artificial_cuts(source, (), max_gap_seconds=600.0)
    assert np.array_equal(carry, np.asarray([[0.0, 800.0]]))
    assert audit.merged_gap_seconds == 550.0
    exposure, gaps = exposure_and_gap_count(source, np.asarray([50.0]), np.asarray([750.0]))
    assert np.isclose(exposure[0], 150.0)
    assert gaps[0] == 1


def test_horizon_specific_split_reserves_three_horizons_for_final_holdout() -> None:
    segments = np.asarray([[0.0, 1000.0]])
    legacy = {"20pct": 0.0, "60pct": 600.0, "70pct": 700.0, "80pct": 1000.0}
    plan = plan_horizon_specific_split(segments, legacy, 100.0)
    assert plan.status == "ESTIMABLE"
    assert plan.boundaries == {"20pct": 0.0, "60pct": 500.0, "70pct": 700.0, "80pct": 1000.0}
    assert np.isclose(plan.exposure_seconds["INNER"], 200.0)
    assert np.isclose(plan.exposure_seconds["SELECTION"], 300.0)


def test_observed_support_hazard_does_not_treat_a_gap_as_silence() -> None:
    anchors, bins, labels, weights = _hazard_rows_observed_support(
        np.asarray([0.0]), np.asarray(["SELECTION"]),
        np.asarray([[0.0, 300.0], [600.0, 900.0]]),
        {"SELECTION": 900.0}, np.asarray([], dtype=np.float64),
    )
    # With the registered five-minute hazard bin, only the two observed bins
    # enter.  The unobserved 300--600 s bin supplies no no-event evidence.
    assert anchors.tolist() == [0, 0]
    assert bins.tolist() == [0, 2]
    assert labels.tolist() == [0.0, 0.0]
    assert weights.tolist() == [1.0, 1.0]


def test_observed_support_contract_rejects_pooled_horizons() -> None:
    with np.testing.assert_raises(ValueError):
        RateTrainConfig(
            horizons_seconds=(300.0, 1800.0),
            window_contract="observed_support",
        ).validate()


def test_physical_future_head_mark_state_is_exact_q_only_special_case() -> None:
    model = PhysicalFutureHead(q_dim=5, state_dim=4, n_horizons=3, n_contacts=7)
    q, state = torch.randn(6, 5), torch.randn(6, 4)
    q_only = model.predictions(q, None)
    full = model.predictions(q, state)
    assert all(torch.equal(a, b) for a, b in zip(q_only, full))


def test_mark_state_uses_the_registered_q_timescale_bank() -> None:
    assert tuple(FullMarkTrainConfig().state_taus_seconds) == tuple(RATE_TAUS_SECONDS)


def test_event_encoder_modality_ablation_does_not_leak_size_or_delay_span() -> None:
    cfg = EncoderConfig(
        use_participation=False, use_exact_delay=False, use_tied_groups=False,
        use_legacy_rank=False, use_waveform=False, use_multiband=False,
        use_geometry=False, d_contact=8, d_event=6, n_attention_heads=2,
        n_attention_layers=1, dropout=0.0,
    )
    shape = DataShape(
        n_contacts=3, n_bands=1, n_band_features=1, n_cross_band_pairs=1,
        n_views=1, n_waveform_samples=4, n_envelope_bins=2,
        n_background_features=1, band_available=(True,),
    )
    model = EventEncoder(cfg, shape, geometry=None).eval()
    batch = {
        "participation": torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool),
        "rel_delay": torch.tensor([[0.0, float("nan"), float("nan")], [0.0, 1.0, 2.0]]),
        "contact_ok": torch.ones(2, 3, dtype=torch.bool),
    }
    event, _ = model(batch)
    assert torch.equal(event[0], event[1])


def test_mark_shuffle_preserves_times_and_payload_multiset_but_breaks_alignment() -> None:
    n = 6
    data = FullMarkData(
        subject="toy", seq=None, event_time=np.arange(n, dtype=np.float64),
        event_segment=np.asarray([0, 0, 0, 1, 1, 1]),
        phase=np.asarray(["FIT"] * n), source_position=np.arange(10, 10 + n),
        input_source_position=np.arange(10, 10 + n), input_view="full_mark",
            input_view_details={}, q_context=None, decoder_index=None, next_index=None,
            event_offsets=(1, 5, 20),
        grid_time=None, grid_segment=None, grid_phase=None, grid_q=None,
        grid_source_event=None, grid_source_dt=None, future_count=None,
        future_count_log_offset=None,
        future_valid=None, future_seizure_count=None, future_participation=None,
            future_participation_valid=None, future_extent=None,
            physical_horizons_seconds=(300.0, 1800.0, 7200.0), provenance={},
    )
    got = configure_event_input_view(data, FullMarkTrainConfig(input_view="mark_shuffle", seed=7))
    assert np.array_equal(got.event_time, data.event_time)
    assert np.array_equal(np.sort(got.input_source_position), np.sort(data.source_position))
    assert np.all(got.input_source_position != data.source_position)
    assert got.input_view_details["marginal_mark_payload_preserved"] is True


def test_constant_mark_state_control_is_computed_from_fit_rows_only() -> None:
    states = np.asarray([[1.0, 3.0], [3.0, 5.0], [100.0, 200.0]], dtype=np.float32)
    got = _fit_period_mean_state(states, np.asarray([0, 1]))
    assert np.array_equal(got, np.asarray([2.0, 4.0], dtype=np.float32))


def test_causal_features_exclude_event_exactly_at_anchor_and_reset_segment() -> None:
    anchor = np.array([10.0, 20.0, 110.0])
    aseg = np.array([0, 0, 1])
    event = np.array([5.0, 10.0, 105.0])
    eseg = np.array([0, 0, 1])
    mark = np.ones((3, 1))
    bounds = np.array([[0.0, 30.0], [100.0, 130.0]])
    out, _ = _causal_features(anchor, aseg, event, eseg, mark, bounds, (10.0,), np.ones(3, bool))
    # At t=10 only the event at 5 is visible. At t=20 both segment-0 events
    # are visible. Segment 1 starts from its own event and cannot inherit them.
    assert np.isclose(out[0, 0], np.log1p(6.0 * np.exp(-0.5)))
    assert np.isclose(out[1, 0], np.log1p(6.0 * (np.exp(-1.0) + np.exp(-1.5))))
    assert np.isclose(out[2, 0], out[0, 0])


def test_causal_features_session_position_ignores_segment_end() -> None:
    # Two anchors at the same elapsed time since their segment start must
    # receive identical session-position features even though one segment
    # ends much later.  The former (t - lo) / (hi - lo) fraction leaked the
    # segment end, which coincides with the next seizure onset.
    anchor = np.array([10.0, 110.0])
    aseg = np.array([0, 1])
    event = np.array([5.0, 105.0])
    eseg = np.array([0, 1])
    mark = np.ones((2, 1))
    bounds = np.array([[0.0, 30.0], [100.0, 100000.0]])
    out, names = _causal_features(anchor, aseg, event, eseg, mark, bounds, (10.0,), np.ones(2, bool))
    assert names[-1] == "segment_elapsed_over_8h"
    assert "segment_fraction" not in names
    # Same elapsed time since the last event and since the segment start; the
    # clock terms legitimately differ because the absolute times differ.
    assert np.isclose(out[0, -4], out[1, -4])
    assert np.isclose(out[0, -1], out[1, -1])
    # Bounded on the fixed 8 h scale, never on the segment's own length.
    assert np.isclose(out[0, -1], 10.0 / 28800.0)


def test_fixed_background_is_recent_strictly_past_and_same_segment() -> None:
    anchor = np.asarray([40.0, 100.0, 140.0, 205.0])
    segment = np.asarray([0, 0, 0, 1])
    bounds = np.asarray([[0.0, 150.0], [200.0, 260.0]])
    starts = np.asarray([8.0, 68.0, 145.0, 198.0])
    ends = np.asarray([10.0, 70.0, 147.0, 200.0])
    values = np.arange(8, dtype=np.float32).reshape(4, 2)
    out, available, age, donor = _align_fixed_background(
        anchor, segment, bounds, starts, ends, values, max_age_seconds=60.0,
    )
    # t=40 and t=100 see their latest causal observations in segment 0.
    assert available[:2].tolist() == [True, True]
    assert donor[:2].tolist() == [0, 1]
    assert np.allclose(age[:2], [30.0, 30.0])
    # t=140 cannot see the observation that ends at 147 (future).  The older
    # observation is too stale.  t=205 cannot import the observation that
    # started before its segment boundary even though it ended at the boundary.
    assert available[2:].tolist() == [False, False]
    assert np.isnan(out[2:, :2]).all()


def test_h3_common_drive_receives_event_count_window_duration() -> None:
    blocks = [
        Block("FIT", 0, 100.0, 700.0, 2500.0, np.arange(5), np.arange(5, 8)),
        Block("FIT", 0, 400.0, 700.0, 2500.0, np.arange(5), np.arange(5, 8)),
    ]
    common = _common_time_features(blocks, np.asarray([[0.0, 3000.0]]))
    assert common.shape == (2, 7)
    assert np.isclose(common[0, 0], np.log1p(600.0))  # exposure duration, not position
    assert np.isclose(common[1, 0], np.log1p(300.0))
    assert common[0, 0] != common[1, 0]


def test_h3_common_drive_position_never_uses_segment_end() -> None:
    # Target segments end exactly at seizure onsets for most patients, so any
    # dependence on the segment end hands every arm a countdown to the next
    # seizure.  Only elapsed time since the segment start is admissible.
    blocks = [Block("FIT", 0, 100.0, 700.0, 2500.0, np.arange(5), np.arange(5, 8))]
    short = _common_time_features(blocks, np.asarray([[0.0, 3000.0]]))
    long = _common_time_features(blocks, np.asarray([[0.0, 300000.0]]))
    assert np.array_equal(short, long)
    assert np.isclose(short[0, 5], 100.0 / 28800.0)
    assert np.isclose(short[0, 6], 700.0 / 28800.0)


def test_h3_nested_arm_admissibility_is_sign_blind_and_checks_both_splits() -> None:
    parent = {"inner_mse": 2.0, "selection_mse": 3.0}
    stable = _nested_arm_admissibility(
        parent, {"inner_mse": 7.9, "selection_mse": 1.0},
    )
    assert stable["admissible"] is True
    # This arm happens to improve on SELECTION, but its INNER fit explodes.
    # It must still be withheld; the rule cannot depend on effect direction.
    unstable_inner = _nested_arm_admissibility(
        parent, {"inner_mse": 8.1, "selection_mse": 1.0},
    )
    assert unstable_inner["admissible"] is False
    assert unstable_inner["parent_relative_mse_ratio"]["selection"] < 1.0
    unstable_selection = _nested_arm_admissibility(
        parent, {"inner_mse": 1.0, "selection_mse": 12.1},
    )
    assert unstable_selection["admissible"] is False


def test_h3_nested_arm_admissibility_rejects_a_diverged_parent() -> None:
    # A parent that explodes while the child stays bounded is the same
    # numerical divergence wearing a favourable sign: the "gain" is the
    # parent's failure, not evidence for the added slot.
    diverged_parent = {"inner_mse": 1.0, "selection_mse": 50.0}
    verdict = _nested_arm_admissibility(diverged_parent, {"inner_mse": 1.0, "selection_mse": 1.0})
    assert verdict["admissible"] is False
    assert any("parent" in reason for reason in verdict["reasons"])
    boundary = _nested_arm_admissibility({"inner_mse": 1.0, "selection_mse": 3.9},
                                         {"inner_mse": 1.0, "selection_mse": 1.0})
    assert boundary["admissible"] is True


def test_h3_admissibility_rejects_parent_and_child_that_diverge_together() -> None:
    # Both arms are far worse than predicting the FIT mean, so their ratio is
    # meaningless; the contrast must be withheld once the null is known.
    parent = {"inner_mse": 30.0, "selection_mse": 40.0}
    child = {"inner_mse": 28.0, "selection_mse": 12.0}
    null = {"inner_mse": 1.1, "selection_mse": 0.9}
    ratio_only = _nested_arm_admissibility(parent, child)
    assert ratio_only["admissible"] is True
    absolute = _nested_arm_admissibility(parent, child, null=null)
    assert absolute["admissible"] is False
    assert any("fit_mean_null" in reason for reason in absolute["reasons"])
    bounded = _nested_arm_admissibility({"inner_mse": 1.5, "selection_mse": 2.0},
                                        {"inner_mse": 1.2, "selection_mse": 1.8}, null=null)
    assert bounded["admissible"] is True


class _DummyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n_nodes = 3
        self.state_dim = 1
        self.n_contacts = 3
        self.weight = nn.Parameter(torch.eye(3))

    def _step(self, h, x):
        return torch.tanh(h @ self.weight.T + x)

    def _readout(self, h):
        return h

    def _stop(self, h, t_norm, recruited_fraction):
        return h.mean(-1) + t_norm + recruited_fraction

    def forward(self, x, recruited, valid):
        h = torch.zeros(x.shape[0], 3)
        logits, stops = [], []
        for step in range(x.shape[1]):
            h = self._step(h, x[:, step])
            logits.append(self._readout(h))
            t = torch.full((x.shape[0],), step / 2)
            stops.append(self._stop(h, t, recruited[:, step].mean(-1)))
        return torch.stack(logits, 1), torch.stack(stops, 1)


def test_stepwise_adapter_starts_at_exact_parity_and_can_backpropagate_to_context() -> None:
    decoder = _DummyDecoder()
    model = StepwiseConditionedDecoder(decoder, StepwiseAdapterConfig(context_dim=4, rank=3))
    x = torch.randn(2, 3, 3)
    recruited = torch.zeros(2, 3, 3)
    valid = torch.ones(2, 3, dtype=torch.bool)
    context = torch.randn(2, 4, requires_grad=True)
    ref = decoder(x, recruited, valid)
    got = model(x, recruited, valid, context, use_static=True, use_dynamic=True)
    assert torch.equal(ref[0], got[0])
    assert torch.equal(ref[1], got[1])
    # Move one output map, as happens on the first optimiser step; the frozen
    # decoder then passes a real gradient back into the context producer.
    with torch.no_grad():
        model.dynamic.contact.weight.fill_(0.01)
    logits, stops = model(x, recruited, valid, context, use_static=True, use_dynamic=True)
    (logits.sum() + stops.sum()).backward()
    assert context.grad is not None and float(context.grad.abs().sum()) > 0
    assert all(parameter.requires_grad is False for parameter in decoder.parameters())


def test_stepwise_shift_support_means_follow_the_scorer_endpoint_set() -> None:
    """The donor-valid means must be derived from whatever the scorer returns.

    A hard-coded endpoint list silently diverges from ``pair_scores`` and then
    raises KeyError deep inside a queued unit, so the contract is checked here.
    """
    import types

    anchor_rows = np.arange(4)
    pairs = types.SimpleNamespace(
        anchor_rows=anchor_rows, pair_anchor=anchor_rows, pair_event=anchor_rows,
        pair_weight=np.full(4, 0.25),
    )
    data = types.SimpleNamespace(
        anchor_time=np.asarray([0.0, 100.0, 200.0, 300.0]),
        event_segment=np.zeros(4, dtype=np.int64),
        last_event_pos=np.arange(4),
    )
    prep = types.SimpleNamespace(selection_pairs=pairs, data=data, context=None)
    donor_valid = np.asarray([True, True, False, False])
    scores = {"grammar": torch.tensor([1.0, 3.0, 100.0, 100.0]),
              "contact_nll": torch.tensor([2.0, 4.0, 50.0, 50.0])}
    calls = {"n": 0}

    def fake_pair_scores(model, prep_, pairs_, context, *, use_static, use_dynamic, **kwargs):
        calls["n"] += 1
        return {k: v.clone() for k, v in scores.items()}

    original_pair_scores = stepwise_train.pair_scores
    original_shift = stepwise_train._shift_context
    stepwise_train.pair_scores = fake_pair_scores
    stepwise_train._shift_context = lambda prep_, horizon=7200.0: (None, donor_valid)
    try:
        means, arrays = stepwise_train._evaluate(None, prep)
    finally:
        stepwise_train.pair_scores = original_pair_scores
        stepwise_train._shift_context = original_shift

    assert calls["n"] == 4
    assert set(means["block_shift"]) == set(scores) | {"n_anchors"}
    assert set(means["rate_dynamic_on_shift_support"]) == set(scores) | {"n_anchors"}
    assert means["block_shift"]["n_anchors"] == 2
    # Only the two donor-valid anchors enter either side of the timing contrast.
    assert np.isclose(means["block_shift"]["grammar"], 2.0)
    assert np.isclose(means["rate_dynamic_on_shift_support"]["grammar"], 2.0)
    assert np.isnan(arrays["block_shift_grammar"][~donor_valid]).all()


def test_future_mark_state_uses_exact_physical_time_without_event_updates() -> None:
    state = torch.tensor([[2.0, 4.0], [9.0, 9.0]])
    mean = torch.zeros(1, 2)
    taus = torch.tensor([[1.0, 2.0]])
    data = SimpleNamespace(event_time=np.asarray([10.0, 12.0]))
    got = _open_loop_state(state, mean, taus, data, np.asarray([0]), np.asarray([1]))
    want = torch.tensor([[2.0 * np.exp(-2.0), 4.0 * np.exp(-1.0)]], dtype=state.dtype)
    assert torch.allclose(got, want, atol=1e-7)


def test_unestimable_functional_readout_keeps_registered_null_arms_explicit() -> None:
    endpoint = Endpoint(
        "missing_endpoint",
        np.zeros((6, 1), dtype=np.float32),
        np.zeros((6, 1), dtype=bool),
    )
    q = np.zeros((6, 2), dtype=np.float32)
    state = np.zeros((6, 3), dtype=np.float32)
    result = _select_and_score(
        endpoint, q, state,
        np.asarray([0, 1]), np.asarray([2, 3]), np.asarray([4, 5]),
        state.copy(), np.zeros(6, dtype=bool),
    )
    assert result["block_shift_state"]["status"] == "NOT_ESTIMABLE"
    assert result["fit_period_mean_state"]["status"] == "NOT_ESTIMABLE"
    assert result["correct_state_on_shift_support"]["status"] == "NOT_ESTIMABLE"


def test_functional_timing_contrast_scores_correct_state_on_shift_support_only() -> None:
    rng = np.random.default_rng(0)
    n = 60
    state = rng.normal(size=(n, 2)).astype(np.float32)
    q = rng.normal(size=(n, 1)).astype(np.float32)
    values = (state[:, :1] * 2.0 + rng.normal(scale=0.1, size=(n, 1))).astype(np.float32)
    endpoint = Endpoint("toy", values, np.ones((n, 1), dtype=bool))
    fit, inner, selection = np.arange(0, 30), np.arange(30, 40), np.arange(40, 60)
    shifted = state.copy()
    shifted[selection] = state[np.roll(selection, 10)]
    shift_valid = np.zeros(n, dtype=bool)
    shift_valid[selection[:8]] = True  # only part of SELECTION has a distant donor
    result = _select_and_score(endpoint, q, state, fit, inner, selection, shifted, shift_valid)
    assert result["block_shift_state"]["n_values"] == 8
    assert result["correct_state_on_shift_support"]["n_values"] == 8
    assert result["q_plus_state"]["n_values"] == 20
    expected = (result["block_shift_state"]["selection_loss"]
                - result["correct_state_on_shift_support"]["selection_loss"])
    assert np.isclose(result["contrasts"]["correct_time_gain_over_shift"], expected)
    assert result["contrasts"]["correct_time_gain_over_shift"] > 0


def test_hazard_counts_seizure_at_observed_segment_right_boundary() -> None:
    anchor_time = np.asarray([700.0])
    segment = np.asarray([0])
    phase = np.asarray(["FIT"])
    segment_bounds = np.asarray([[0.0, 1000.0]])
    anchors, bins, labels = _hazard_rows(
        anchor_time, segment, phase, segment_bounds, {"FIT": 1000.0},
        np.asarray([1000.0]),
    )
    assert anchors.tolist() == [0]
    assert bins.tolist() == [0]
    assert labels.tolist() == [1.0]


def test_risk_score_keeps_observed_event_in_partially_followed_horizon() -> None:
    probability = np.full((1, 72), 0.25, dtype=np.float64)
    result = _risk_scores(
        probability, np.asarray([800.0]), np.asarray([0]), np.asarray([1000.0]),
        np.asarray([0]), np.asarray([[0.0, 1000.0]]), observation_hi=1000.0,
    )
    assert result["5min"]["n_anchors"] == 1
    assert result["5min"]["n_positive"] == 1


def test_risk_score_withholds_a_horizon_whose_eligibility_is_outcome_determined() -> None:
    """Long horizons keep only anchors whose seizure was observed.

    With no anchor whose observation window covers the horizon, every survivor
    is a positive; a Brier score over that set is not a forecast score.
    """
    probability = np.full((2, 72), 0.25, dtype=np.float64)
    # Both anchors sit less than one horizon before the end of the observation,
    # so neither has full follow-up; only the one with a seizure survives.
    result = _risk_scores(
        probability, np.asarray([600.0, 700.0]), np.asarray([0, 1]),
        np.asarray([800.0]), np.asarray([0, 0]), np.asarray([[0.0, 1000.0]]),
        observation_hi=1000.0,
    )
    assert result["30min"]["status"] == "NOT_ESTIMABLE"
    assert result["30min"]["n_full_followup"] == 0
    assert result["30min"]["n_positive"] == result["30min"]["n_anchors"]
    assert "withheld_brier" in result["30min"]
    # The short horizon still has a genuine negative and stays estimable.
    assert result["5min"]["n_full_followup"] > 0
    assert result["5min"]["outcome_dependent_eligibility"] is False


def test_risk_score_drops_partially_followed_no_event_interval() -> None:
    probability = np.full((1, 72), 0.25, dtype=np.float64)
    result = _risk_scores(
        probability, np.asarray([800.0]), np.asarray([0]), np.asarray([], dtype=np.float64),
        np.asarray([0]), np.asarray([[0.0, 1000.0]]), observation_hi=1000.0,
    )
    assert result["5min"]["status"] == "NOT_ESTIMABLE"


def test_v035_second_wave_subjects_are_explicit_not_implicit_allowlist_growth() -> None:
    assert V035_EXTENSION_SUBJECTS == (
        "epilepsiae_1096", "epilepsiae_384", "epilepsiae_1125",
    )
