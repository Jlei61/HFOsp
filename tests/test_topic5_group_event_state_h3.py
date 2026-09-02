"""Regression tests for the H3 event-feedback line.

Each test names the contract clause it defends (``docs/archive/topic5/
group_event_state_v0_2_h3_contract_clauses_2026-09-01.md``).  They are cheap and
synthetic on purpose: they prove the implementation matches the contract, which
is a different and much weaker claim than proving anything about H3.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state_h3.features import build_block_targets, EventFeatures
from src.topic5_group_event_state_h3.models import H3Config, build_model
from src.topic5_group_event_state_h3.support import (
    SPLIT_FRACTIONS,
    SPLIT_NAMES,
    Interval,
    build_coverage_segments,
    cut_intervals_at_seizures,
    segment_anchor_grid,
    select_disjoint_anchors,
    split_by_physical_time,
    tile_blocks,
)


# --------------------------------------------------------------------------- A1


def test_a1_recording_gap_breaks_the_segment_and_a_seam_does_not():
    blocks = [(0.0, 3600.0), (3600.5, 7200.0), (10800.0, 14400.0)]
    segments = build_coverage_segments(blocks)
    assert segments == [(0.0, 7200.0), (10800.0, 14400.0)]


def test_a1_refuses_to_invent_coverage_when_given_nothing():
    with pytest.raises(ValueError):
        build_coverage_segments([])


# --------------------------------------------------------------------------- A2/A3


def test_a2_a3_seizure_removes_onset_to_offset_plus_postictal():
    segments = [(0.0, 20000.0)]
    cut = cut_intervals_at_seizures(segments, [(5000.0, 5100.0)], postictal_exclusion_s=3600.0)
    assert cut == [(0.0, 5000.0), (8700.0, 20000.0)]


def test_a2_no_block_can_contain_a_seizure_onset():
    segments = build_coverage_segments([(0.0, 40000.0)])
    cut = cut_intervals_at_seizures(segments, [(20000.0, 20050.0)], postictal_exclusion_s=0.0)
    intervals = split_by_physical_time(cut)
    for horizon in (5, 30):
        for block in tile_blocks("s", intervals, horizon, disjoint_exposure=True):
            assert not (block.exposure_start < 20000.0 < block.target_stop)


# --------------------------------------------------------------------------- A4


def test_a4_split_uses_recorded_time_not_event_count_and_cuts_intervals():
    # Two segments of very different length: an event-count split would be free to
    # put the boundary anywhere, a recorded-time split may not.
    intervals = split_by_physical_time([(0.0, 300.0), (1000.0, 1700.0)])
    seconds: dict[str, float] = {}
    for interval in intervals:
        seconds[interval.split] = seconds.get(interval.split, 0.0) + interval.duration
    for name, fraction in zip(SPLIT_NAMES, SPLIT_FRACTIONS):
        assert math.isclose(seconds[name], 1000.0 * fraction, rel_tol=1e-6)
    assert math.isclose(sum(seconds.values()), 1000.0, rel_tol=1e-9)


def test_a4_no_block_spans_two_splits():
    intervals = split_by_physical_time([(0.0, 100000.0)])
    bounds = {(i.split, i.start, i.stop) for i in intervals}
    for block in tile_blocks("s", intervals, 30, disjoint_exposure=True):
        assert any(
            start - 1e-6 <= block.exposure_start and block.target_stop <= stop + 1e-6
            for split, start, stop in bounds
            if split == block.split
        )


# --------------------------------------------------------------------------- A7


def test_a7_target_blocks_are_disjoint_and_are_never_truncated():
    intervals = split_by_physical_time([(0.0, 100000.0)])
    blocks = [b for b in tile_blocks("s", intervals, 30, disjoint_exposure=False)]
    for block in blocks:
        assert math.isclose(block.target_stop - block.anchor, 1800.0, rel_tol=1e-9)
    for split in {b.split for b in blocks}:
        spans = sorted(
            (b.anchor, b.target_stop) for b in blocks if b.split == split
        )
        for (a1, b1), (a2, _b2) in zip(spans, spans[1:]):
            assert b1 <= a2 + 1e-6, "target blocks must not overlap"


def test_a7_disjoint_exposure_mode_keeps_exposures_apart_too():
    intervals = split_by_physical_time([(0.0, 100000.0)])
    blocks = tile_blocks("s", intervals, 30, disjoint_exposure=True)
    for split in {b.split for b in blocks}:
        spans = sorted(
            (b.exposure_start, b.target_stop) for b in blocks if b.split == split
        )
        for (_a1, b1), (a2, _b2) in zip(spans, spans[1:]):
            assert b1 <= a2 + 1e-6


def test_a7_eligibility_and_estimator_share_one_grid():
    """A block count is only honest if the estimator uses the same anchors."""

    intervals = split_by_physical_time([(0.0, 100000.0)])
    grid = segment_anchor_grid(0.0, 100000.0)
    chosen = select_disjoint_anchors(grid, intervals, 30, disjoint_exposure=False)
    tiled = tile_blocks("s", intervals, 30, disjoint_exposure=False)
    assert [round(a, 6) for a, _s, _g in chosen] == [round(b.anchor, 6) for b in tiled]
    assert set(np.round([a for a, _s, _g in chosen], 6)) <= set(np.round(grid, 6))


# --------------------------------------------------------------------------- G1/H3


def _tiny_model(arm: str, d_state: int = 8, seed: int = 0):
    cfg = H3Config(d_state=d_state, horizons_minutes=(5,), chunk_steps=16)
    return build_model(arm, cfg, n_drive_features=3, n_count_features=4, n_mark_features=6, seed=seed)


def test_g1_tau_uses_exp_clamp_and_can_reach_hours():
    model = _tiny_model("M0_no_feedback")
    with torch.no_grad():
        model.log_tau.fill_(math.log(5.0 * 3600.0))
    tau = model.taus()
    assert torch.all(tau > 3599.0), "a slow state must be able to be hours long"
    with torch.no_grad():
        model.log_tau.fill_(math.log(1e9))
    assert torch.allclose(model.taus(), torch.full_like(model.taus(), 6.0 * 3600.0))


def test_h3_m0_state_is_bit_identical_when_phantom_events_are_inserted():
    """``M0``'s free dynamics must not learn the event count through step count.

    An additive per-event drive -- the shape the v0.1 model used -- fails this.
    """

    torch.manual_seed(0)
    model = _tiny_model("M0_no_feedback")
    d = model.cfg.d_state

    dt_a = torch.tensor([10.0, 20.0, 30.0, 0.0])
    drive_a = torch.randn(4, d)
    drive_a[1:] = drive_a[0]  # one constant background cell
    impulse_a = torch.zeros(4, d)
    want = torch.tensor([3])
    state_a, final_a = model.rollout(dt_a, drive_a, impulse_a, want)

    # Same physical span, five extra "events" inserted inside the same cell.
    dt_b = torch.tensor([4.0, 6.0, 5.0, 15.0, 12.0, 8.0, 10.0, 0.0])
    drive_b = drive_a[0].unsqueeze(0).expand(8, d).contiguous()
    impulse_b = torch.zeros(8, d)
    state_b, final_b = model.rollout(dt_b, drive_b, impulse_b, torch.tensor([7]))

    assert torch.allclose(state_a, state_b, atol=1e-5), (state_a, state_b)
    assert torch.allclose(final_a, final_b, atol=1e-5)


def test_h3_m1_state_does_change_when_events_are_inserted():
    """The mirror image: the arm that *has* an edge must actually use it."""

    torch.manual_seed(0)
    model = _tiny_model("M1_count_rate_feedback")
    with torch.no_grad():
        model.count_adapter.up.weight.normal_(0.0, 0.5)
    d = model.cfg.d_state
    drive = torch.zeros(8, d)
    dt = torch.tensor([4.0, 6.0, 5.0, 15.0, 12.0, 8.0, 10.0, 0.0])
    quiet = torch.zeros(8, d)
    busy = quiet.clone()
    busy[2] = model.event_impulse(torch.randn(1, 4), torch.randn(1, 6))[0]
    _s0, final_quiet = model.rollout(dt, drive, quiet, torch.tensor([7]))
    _s1, final_busy = model.rollout(dt, drive, busy, torch.tensor([7]))
    assert not torch.allclose(final_quiet, final_busy)


def test_g2_windowed_carry_matches_one_uninterrupted_pass():
    """Split pass + carry must equal the uninterrupted causal pass."""

    torch.manual_seed(1)
    model = _tiny_model("M2_mark_specific_feedback")
    with torch.no_grad():
        model.count_adapter.up.weight.normal_(0.0, 0.3)
        model.mark_adapter.up.weight.normal_(0.0, 0.3)
    n, d = 40, model.cfg.d_state
    dt = torch.rand(n) * 50.0
    drive = torch.randn(n, d)
    impulse = torch.randn(n, d) * 0.1
    want = torch.arange(n)

    whole, final_whole = model.rollout(dt, drive, impulse, want, chunk=n)
    first, carry = model.rollout(dt[:17], drive[:17], impulse[:17], torch.arange(17), chunk=5)
    second, final_split = model.rollout(
        dt[17:], drive[17:], impulse[17:], torch.arange(n - 17), state_init=carry, chunk=7
    )
    joined = torch.cat([first, second], dim=0)
    assert torch.allclose(whole, joined, atol=1e-4), (whole - joined).abs().max()
    assert torch.allclose(final_whole, final_split, atol=1e-4)


def test_rollout_chunking_does_not_change_the_answer():
    torch.manual_seed(2)
    model = _tiny_model("M1_count_rate_feedback")
    n, d = 64, model.cfg.d_state
    dt = torch.rand(n) * 100.0
    drive = torch.randn(n, d)
    impulse = torch.randn(n, d) * 0.05
    want = torch.arange(0, n, 7)
    a, fa = model.rollout(dt, drive, impulse, want, chunk=8)
    b, fb = model.rollout(dt, drive, impulse, want, chunk=n)
    assert torch.allclose(a, b, atol=1e-4)
    assert torch.allclose(fa, fb, atol=1e-4)


def test_rollout_survives_a_recording_length_gap_without_nan():
    """Across a 40-hour gap the decay underflows; a ratio form would give 0/0."""

    model = _tiny_model("M0_no_feedback")
    n, d = 8, model.cfg.d_state
    dt = torch.tensor([10.0, 144000.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0])
    drive = torch.randn(n, d)
    impulse = torch.zeros(n, d)
    states, final = model.rollout(dt, drive, impulse, torch.arange(n), chunk=n)
    assert torch.isfinite(states).all()
    assert torch.isfinite(final).all()


# --------------------------------------------------------------------------- F6 zero truth


def _synthetic_features(n: int, n_contacts: int = 4, seed: int = 0) -> EventFeatures:
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.exponential(20.0, n)) + 1000.0
    part = rng.random((n, n_contacts)) < 0.5
    mark = rng.normal(size=(n, 6)).astype(np.float32)
    count = np.stack(
        [np.ones(n), np.log1p(part.sum(1)), part.mean(1), np.log1p(np.diff(t, prepend=t[0]))],
        axis=1,
    ).astype(np.float32)
    return EventFeatures(
        t_abs=t,
        count_features=count,
        mark_features=mark,
        mark_group_slices={"a": (0, 3), "b": (3, 6)},
        count_feature_names=["occurrence", "log1p_size", "size_fraction", "log1p_dt_prev"],
        mark_feature_names=[f"m{i}" for i in range(6)],
        participation=part,
        size=part.sum(1).astype(np.float32),
        band_available=np.ones(3, dtype=bool),
    )


def test_block_targets_are_half_open_and_prefix_summed():
    feats = _synthetic_features(200, seed=3)
    anchors = np.array([feats.t_abs[10], feats.t_abs[50]])
    targets = build_block_targets(feats, anchors, 100.0)
    for i, anchor in enumerate(anchors):
        inside = (feats.t_abs >= anchor) & (feats.t_abs < anchor + 100.0)
        assert targets.count[i] == inside.sum()
        if inside.any():
            assert np.allclose(
                targets.mark_mean[i], feats.mark_features[inside].mean(0), atol=1e-4
            )


def test_an_event_exactly_on_the_anchor_belongs_to_the_future_block():
    """Half-open by contract: it is part of what is predicted, not of the past."""

    feats = _synthetic_features(50, seed=4)
    anchor = np.array([feats.t_abs[7]])
    targets = build_block_targets(feats, anchor, 60.0)
    assert targets.count[0] >= 1
    assert np.isclose(feats.t_abs[7], anchor[0])


def test_f6_constant_and_drift_zero_truth_leave_the_edge_at_zero_effect():
    """With no event-dependent signal, an added edge must not manufacture one.

    The adapter is initialised with a zero output matrix, so at initialisation
    ``M1`` and ``M0`` are *the same function*.  A future refactor that seeded the
    edge non-zero would let an arm win before it had learned anything.
    """

    m0 = _tiny_model("M0_no_feedback", seed=7)
    m1 = _tiny_model("M1_count_rate_feedback", seed=7)
    m2 = _tiny_model("M2_mark_specific_feedback", seed=7)
    x_count = torch.randn(5, 4)
    x_mark = torch.randn(5, 6)
    assert torch.allclose(m1.event_impulse(x_count, x_mark), torch.zeros(5, m1.cfg.d_state))
    assert torch.allclose(m2.event_impulse(x_count, x_mark), torch.zeros(5, m2.cfg.d_state))
    assert torch.allclose(m0.event_impulse(x_count, x_mark), torch.zeros(5, m0.cfg.d_state))


def test_f1_disabling_the_edge_reproduces_the_no_feedback_state_exactly():
    """``no_event_feedback`` must be the same model with the edge switched off."""

    torch.manual_seed(5)
    model = _tiny_model("M2_mark_specific_feedback")
    with torch.no_grad():
        model.count_adapter.up.weight.normal_(0.0, 0.4)
        model.mark_adapter.up.weight.normal_(0.0, 0.4)
    x_count, x_mark = torch.randn(6, 4), torch.randn(6, 6)
    off = model.event_impulse(x_count, x_mark, enable_count=False, enable_mark=False)
    assert torch.allclose(off, torch.zeros_like(off))
    scale = 1.0 / (model.mean_event_rate_hz * model.taus()).clamp_min(1e-6)
    only_count = model.event_impulse(x_count, x_mark, enable_mark=False)
    assert torch.allclose(only_count, model.count_adapter(x_count) * scale.unsqueeze(0))
    both = model.event_impulse(x_count, x_mark)
    only_mark = model.event_impulse(x_count, x_mark, enable_count=False)
    assert torch.allclose(both, only_count + only_mark, atol=1e-6)


def test_the_event_edge_is_rate_normalised_so_two_patients_are_comparable():
    """A dense patient's edge must not be 30x a sparse patient's by construction.

    Under a constant rate ``r`` the linear state settles at ``r * tau * u``.  With
    a raw kick that steady state scales with the patient's IED rate, so the same
    learned gain would mean a state of order 100 in one patient and 3 in another,
    and only the first would leave the decoder's range.  Normalising by ``r * tau``
    makes the gain a dimensionless fraction of that steady state.
    """

    cfg = H3Config(d_state=8, horizons_minutes=(5,), chunk_steps=16)
    dense = build_model("M1_count_rate_feedback", cfg, 3, 4, 6, 0, mean_event_rate_hz=2.0)
    sparse = build_model("M1_count_rate_feedback", cfg, 3, 4, 6, 0, mean_event_rate_hz=0.02)
    for model in (dense, sparse):
        with torch.no_grad():
            model.count_adapter.up.weight.fill_(0.1)
            model.log_tau.fill_(math.log(600.0))
    x = torch.ones(1, 4)
    steady_dense = float(dense.mean_event_rate_hz) * 600.0 * dense.event_impulse(x, torch.zeros(1, 6))
    steady_sparse = float(sparse.mean_event_rate_hz) * 600.0 * sparse.event_impulse(x, torch.zeros(1, 6))
    assert torch.allclose(steady_dense, steady_sparse, atol=1e-5)
    assert float(steady_dense.abs().max()) < 1.0


def test_m2_is_nested_over_m1():
    """The acceptance rule compares M2 to M1 at the same count and time."""

    model = _tiny_model("M2_mark_specific_feedback")
    assert model.count_adapter is not None and model.mark_adapter is not None
    m1 = _tiny_model("M1_count_rate_feedback")
    assert m1.count_adapter is not None and m1.mark_adapter is None
    m0 = _tiny_model("M0_no_feedback")
    assert m0.count_adapter is None and m0.mark_adapter is None


def test_negative_binomial_logpmf_matches_scipy_shape_at_known_values():
    """A wrong parameterisation would move every arm by the same amount and hide."""

    model = _tiny_model("M0_no_feedback")
    states = torch.zeros(3, model.cfg.d_state)
    with torch.no_grad():
        model.decoder.count["5"].weight.zero_()
        model.decoder.count["5"].bias.copy_(torch.tensor([math.log(4.0), math.log(2.0)]))
        out = model.score_blocks(
            states, 5,
            torch.tensor([0, 4, 10]),
            torch.tensor([False, True, True]),
            torch.zeros(3, 6),
        )
    mu, phi = 4.0, 2.0
    for i, k in enumerate([0, 4, 10]):
        expect = (
            math.lgamma(k + phi) - math.lgamma(phi) - math.lgamma(k + 1)
            + phi * math.log(phi / (phi + mu)) + k * math.log(mu / (phi + mu))
        )
        assert abs(float(out["count"][i]) - expect) < 1e-4


def test_mark_score_is_zero_where_the_block_had_no_events():
    """The mark endpoint is conditional on the block containing events."""

    model = _tiny_model("M0_no_feedback")
    out = model.score_blocks(
        torch.zeros(2, model.cfg.d_state), 5,
        torch.tensor([0, 3]),
        torch.tensor([False, True]),
        torch.zeros(2, 6),
    )
    assert torch.allclose(out["mark"][0], torch.zeros(6))
    assert not torch.allclose(out["mark"][1], torch.zeros(6))


# --------------------------------------------------------------------------- A5/A6


def test_a6_an_anchor_is_read_before_an_event_standing_on_the_same_instant():
    """An event at exactly the anchor time belongs to the block, not to its past."""

    from src.topic5_group_event_state_h3.models import KIND_ANCHOR, KIND_CELL, KIND_EVENT
    from src.topic5_group_event_state_h3.timeline import build_segment_timeline

    start, stop = 0.0, 3600.0
    anchor_hit = 300.0  # on the 5-minute grid
    events = np.array([120.0, anchor_hit, 900.0])
    tl = build_segment_timeline(
        0, start, stop, events,
        (np.array([0.0, 30.0]), np.zeros((2, 2), np.float32)),
        start, stop,
    )
    at_instant = np.flatnonzero(tl.step_time == anchor_hit)
    kinds = tl.step_kind[at_instant].tolist()
    assert KIND_ANCHOR in kinds and KIND_EVENT in kinds
    assert kinds.index(KIND_ANCHOR) < kinds.index(KIND_EVENT)
    # and the cell boundary, if it lands here too, comes before both
    if KIND_CELL in kinds:
        assert kinds.index(KIND_CELL) < kinds.index(KIND_ANCHOR)


def test_a4_a_block_that_would_cross_a_split_is_marked_invalid():
    from src.topic5_group_event_state_h3.timeline import build_segment_timeline, label_anchors

    intervals = split_by_physical_time([(0.0, 36000.0)])
    tl = build_segment_timeline(
        0, 0.0, 36000.0, np.array([100.0, 200.0]),
        (np.array([0.0, 30.0]), np.zeros((2, 2), np.float32)),
        0.0, 36000.0,
    )
    split, valid = label_anchors(tl, intervals, [5, 30, 120])
    stops = {i.split: i.stop for i in intervals}
    for horizon in (5, 30, 120):
        span = horizon * 60.0
        for i, anchor in enumerate(tl.anchor_time):
            if not split[i]:
                continue
            expected = (anchor + span) <= stops[split[i]] + 1e-6
            assert bool(valid[horizon][i]) == expected


def test_a5_the_state_resets_at_the_start_of_every_coverage_segment():
    """Two segments with identical inputs must produce identical states."""

    torch.manual_seed(3)
    model = _tiny_model("M1_count_rate_feedback")
    d = model.cfg.d_state
    dt = torch.tensor([10.0, 10.0, 10.0, 0.0])
    drive = torch.randn(4, d)
    impulse = torch.zeros(4, d)
    first, _ = model.rollout(dt, drive, impulse, torch.arange(4))
    # a "new segment" is a fresh rollout with state_init=None
    second, _ = model.rollout(dt, drive, impulse, torch.arange(4))
    assert torch.allclose(first, second)
    # and carrying a state in must change the answer, or the reset means nothing
    carried, _ = model.rollout(
        dt, drive, impulse, torch.arange(4), state_init=torch.ones(d) * 5.0
    )
    assert not torch.allclose(first, carried)


# --------------------------------------------------------------------------- F2/F4/F6


def test_f4_content_perturbation_preserves_event_count_and_instants_exactly():
    """The content estimand is only a content estimand if nothing else moved."""

    from src.topic5_group_event_state_h3.perturb import state_matched_marks

    torch.manual_seed(9)
    n_events, n_mark = 7, 6
    marks = torch.randn(40, n_mark)

    class _Data:
        mark_features = marks

    recipient_rows = torch.arange(n_events)
    recipient_state = torch.randn(n_events, 5)
    donor_rows = torch.arange(20, 40)
    donor_states = torch.randn(20, 5)
    replaced = state_matched_marks(
        _Data(), recipient_rows, recipient_state, donor_rows, donor_states
    )
    # one row out, one row in: the event count is untouched by construction
    assert replaced.shape == (n_events, n_mark)
    # every replacement really comes from the donor pool, never from the recipient
    for row in replaced:
        assert any(torch.allclose(row, marks[int(d)]) for d in donor_rows)
    # and the donors are state-matched, not arbitrary: the chosen donor is the
    # nearest one in standardised state space
    mu, sd = donor_states.mean(0, keepdim=True), donor_states.std(0, keepdim=True).clamp_min(1e-6)
    a, b = (recipient_state - mu) / sd, (donor_states - mu) / sd
    expected = donor_rows[torch.cdist(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0).argmin(dim=1)]
    assert torch.allclose(replaced, marks[expected])


def test_f6_a_constant_and_a_drifting_zero_truth_produce_no_event_edge_signal():
    """Zero-truth regression: with no event-dependent structure, nothing to find.

    Kept as a unit test rather than a human arm, exactly as the plan asks.  It
    checks the generator, not the estimator: an ``intercept_only`` recording must
    have a latent that never moves, and a ``linear_drift`` one must move
    monotonically and by time alone.
    """

    from src.topic5_group_event_state_h3.synthetic import generate

    flat = generate("intercept_only", hours=6.0, seed=0)
    assert float(flat.latent.std()) == 0.0

    drift = generate("linear_drift", hours=6.0, seed=0)
    assert np.all(np.diff(drift.latent) >= -1e-12)
    assert float(drift.latent[-1] - drift.latent[0]) > 0.5

    # and the two feedback truths must actually differ from the no-feedback one
    zero = generate("zero_feedback", hours=6.0, seed=0)
    count = generate("count_feedback", hours=6.0, seed=0)
    assert not np.allclose(zero.latent, count.latent)
    assert np.isfinite(count.latent).all() and float(np.abs(count.latent).max()) <= 4.0


def test_f8_the_primary_perturbation_set_is_exactly_three_arms():
    from src.topic5_group_event_state_h3.perturb import PRIMARY_ARMS, SECONDARY_ARMS

    assert PRIMARY_ARMS == (
        "real_sequence", "no_event_feedback", "state_matched_mark_replacement",
    )
    assert set(PRIMARY_ARMS) & set(SECONDARY_ARMS) == set()
