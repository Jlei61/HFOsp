"""A0 regression tests for session-preserving training (clause C2, EI 3).

The engineering appendix names three minimum tests for this layer, and all three
are here: one uninterrupted forward must equal the chunked carry, state must be
reset across a gap or seizure, and shuffling chunk order inside a segment must
change the answer (otherwise "we carry state" is not actually doing anything).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.topic5_group_event_state.model import DataShape, EncoderConfig, StateConfig
from src.topic5_group_event_state.v02 import marks as M
from src.topic5_group_event_state.v02 import producers as P
from src.topic5_group_event_state.v02 import timeline as T
from src.topic5_group_event_state.v02.subject import SubjectTimeline, SubjectTimelineConfig
from src.topic5_group_event_state.v02.baseline import build_baseline_features
from src.topic5_group_event_state.v02.targets import FutureTargetBuilder


N_CONTACTS, N_BANDS, N_VIEWS, N_CTX, N_ENV, N_BF, N_PAIR, N_BG = 4, 2, 1, 64, 8, 5, 1, 3


class FakeSequence:
    """The slice of ``SubjectSequence`` that the session pass actually touches."""

    def __init__(self, n: int, rng: np.random.Generator, t_abs: np.ndarray):
        self.n = n
        self.rng = rng
        self.arrays = {
            "waveform": rng.normal(size=(n, N_CONTACTS, N_VIEWS, N_CTX)).astype(np.float32),
            "band_envelope": np.abs(rng.normal(size=(n, N_CONTACTS, N_BANDS, N_ENV))).astype(np.float32),
            "band_features": rng.normal(size=(n, N_CONTACTS, N_BANDS, N_BF)).astype(np.float32),
            "cross_band_lag": rng.normal(size=(n, N_CONTACTS, N_PAIR)).astype(np.float32),
            "background": rng.normal(size=(n, N_CONTACTS, N_BG)).astype(np.float32),
        }
        part = rng.random((n, N_CONTACTS)) < 0.6
        part[~part.any(1), 0] = True
        self.participation = part
        self.rel_delay = np.where(part, rng.random((n, N_CONTACTS)) * 0.03, np.nan).astype(np.float32)
        self.t_abs = t_abs

    def gather_positions(self, pos: np.ndarray) -> dict[str, np.ndarray]:
        pos = np.asarray(pos, dtype=np.int64)
        out = {k: v[pos] for k, v in self.arrays.items()}
        out.update({
            "participation": self.participation[pos],
            "contact_ok": np.ones((pos.size, N_CONTACTS), dtype=bool),
            "rel_delay": self.rel_delay[pos],
            "tied_group_id": np.zeros((pos.size, N_CONTACTS), dtype=np.int16),
            "legacy_rank": np.zeros((pos.size, N_CONTACTS), dtype=np.int16),
            "background_age": np.full(pos.size, 10.0, dtype=np.float32),
            "has_waveform": np.ones(pos.size, dtype=bool),
            "time_to_next_seizure": np.full(pos.size, 1e6, dtype=np.float32),
            "t_abs": self.t_abs[pos],
            "dt_prev": np.ones(pos.size, dtype=np.float32),
            "new_session": np.zeros(pos.size, dtype=bool),
            "history": np.zeros((pos.size, 4), dtype=np.float32),
        })
        return out


def _timeline(seed: int = 0, n_per_segment: int = 300, gap: float = 100_000.0):
    rng = np.random.default_rng(seed)
    sessions = [
        T.RecordedSession(0, 0.0, 20_000.0),
        T.RecordedSession(1, gap, gap + 20_000.0),
    ]
    segments = T.build_carry_segments(sessions)
    times = np.concatenate([
        np.sort(rng.uniform(s.start_epoch + 10.0, s.stop_epoch - 10.0, n_per_segment))
        for s in segments
    ])
    seg_of = T.assign_events_to_segments(times, segments)
    split = T.physical_time_split(segments, (0.7, 0.1, 0.2))
    seq = FakeSequence(times.size, rng, times)
    train_pos = np.flatnonzero(times < split.boundary_epochs[0])
    marks = M.build_event_marks(
        seq.participation, seq.rel_delay, seq.arrays["band_features"],
        band_available=(True,) * N_BANDS, band_names=tuple("ab"[:N_BANDS]),
        train_positions=train_pos, n_components=3, seed=0,
    )
    grid = T.build_anchor_grid(segments, split, times,
                               horizons_seconds=(300.0, 1800.0))
    baseline = build_baseline_features(
        grid, segments, times, seg_of, marks,
        seizure_onsets=np.zeros(0), seizure_offsets=np.zeros(0),
    )
    cfg = SubjectTimelineConfig(horizons_seconds=(300.0, 1800.0))
    tl = SubjectTimeline(
        subject="synthetic", dataset="synthetic", config=cfg, index={},
        segments=segments, split=split, stream_positions=np.arange(times.size),
        event_times=times, event_segment=seg_of, marks=marks,
        builder=FutureTargetBuilder(marks), grid=grid, baseline=baseline,
        excluded={},
    )
    return tl, seq


def _model(tl, use_future: bool, seed: int = 0):
    shape = DataShape(
        n_contacts=N_CONTACTS, n_bands=N_BANDS, n_band_features=N_BF,
        n_cross_band_pairs=N_PAIR, n_views=N_VIEWS, n_waveform_samples=N_CTX,
        n_envelope_bins=N_ENV, n_background_features=N_BG,
        band_available=(True,) * N_BANDS,
    )
    cfg = P.ProducerConfig(
        name="test", use_future_heads=use_future,
        encoder=EncoderConfig(use_waveform=True, use_multiband=True, use_geometry=False,
                              d_contact=16, d_event=16, n_attention_heads=2,
                              waveform_channels=4, dropout=0.0),
        state=StateConfig(d_fast=8, d_slow=4),
        chunk_events=64, batch_segments=2, amp=False,
    )
    torch.manual_seed(seed)
    model = P.GroupEventStateProducer(
        cfg, shape, None, tl.n_dims, tl.grid.horizons_seconds,
        torch.Generator().manual_seed(seed),
    )
    model.eval()
    return model, cfg


def _anchor_states(model, tl, seq, cfg, chunk: int, batch: int):
    targets = P.build_anchor_targets(tl, None)
    with torch.no_grad():
        _m, extra = P.run_session_pass(
            model, tl, seq, P.full_segment_ranges(tl), targets,
            torch.device("cpu"), replace(cfg, chunk_events=chunk, batch_segments=batch),
            train=False, rng=np.random.default_rng(0), collect_states=True,
        )
    return extra["anchor_state"], targets


# --------------------------------------------------------------------- EI 3 (1)


def test_one_uninterrupted_forward_equals_the_chunked_carry() -> None:
    """A chunk edge must only detach the graph; it must not change the state."""

    tl, seq = _timeline()
    model, cfg = _model(tl, use_future=False)
    whole, targets = _anchor_states(model, tl, seq, cfg, chunk=10_000, batch=2)
    chunked, _ = _anchor_states(model, tl, seq, cfg, chunk=16, batch=2)
    seen = targets.last_event_pos >= 0
    assert seen.sum() > 20
    assert np.allclose(whole[seen], chunked[seen], atol=1e-4)


def test_slot_count_does_not_change_the_state_of_any_anchor() -> None:
    """Segments are independent chains, so how many run side by side is irrelevant."""

    tl, seq = _timeline()
    model, cfg = _model(tl, use_future=False)
    one, targets = _anchor_states(model, tl, seq, cfg, chunk=64, batch=1)
    two, _ = _anchor_states(model, tl, seq, cfg, chunk=64, batch=2)
    seen = targets.last_event_pos >= 0
    assert np.allclose(one[seen], two[seen], atol=1e-4)


# --------------------------------------------------------------------- EI 3 (2)


def test_state_is_reset_across_a_recording_gap() -> None:
    """Changing the first segment must leave the second segment's anchors alone."""

    tl, seq = _timeline()
    model, cfg = _model(tl, use_future=False)
    before, targets = _anchor_states(model, tl, seq, cfg, chunk=64, batch=2)

    first = np.flatnonzero(tl.event_segment == 0)
    seq.arrays["waveform"][first] *= 7.3
    seq.arrays["band_features"][first] += 5.0
    after, _ = _anchor_states(model, tl, seq, cfg, chunk=64, batch=2)

    seg_of_anchor = tl.grid.segment_index[targets.anchor_index]
    second = (seg_of_anchor == 1) & (targets.last_event_pos >= 0)
    firstseg = (seg_of_anchor == 0) & (targets.last_event_pos >= 0)
    assert second.sum() > 10 and firstseg.sum() > 10
    assert np.allclose(before[second], after[second], atol=1e-5)
    assert not np.allclose(before[firstseg], after[firstseg], atol=1e-3)


def test_a_seizure_splits_one_session_into_two_independent_chains() -> None:
    tl, seq = _timeline()
    seizures = [{"onset_epoch": 8_000.0, "offset_epoch": 8_100.0}]
    segments = T.build_carry_segments(
        [T.RecordedSession(0, 0.0, 20_000.0)], seizures,
        postictal_exclusion_seconds=3600.0, min_segment_seconds=0.0,
    )
    assert len(segments) == 2
    assert segments[0].stop_epoch == 8_000.0
    assert segments[1].start_epoch == 8_100.0 + 3600.0


# --------------------------------------------------------------------- EI 3 (3)


def test_shuffling_the_chunk_order_inside_a_segment_changes_the_answer() -> None:
    """If order did not matter, "we carry state in order" would be vacuous."""

    tl, seq = _timeline()
    model, cfg = _model(tl, use_future=False)
    ordered, targets = _anchor_states(model, tl, seq, cfg, chunk=64, batch=2)

    rng = np.random.default_rng(0)
    seg0 = np.flatnonzero(tl.event_segment == 0)
    perm = seg0.copy()
    rng.shuffle(perm)
    for key in seq.arrays:
        seq.arrays[key][seg0] = seq.arrays[key][perm]
    seq.participation[seg0] = seq.participation[perm]
    shuffled, _ = _anchor_states(model, tl, seq, cfg, chunk=64, batch=2)

    seg_of_anchor = tl.grid.segment_index[targets.anchor_index]
    inside = (seg_of_anchor == 0) & (targets.last_event_pos >= 0)
    assert not np.allclose(ordered[inside], shuffled[inside], atol=1e-3)


# --------------------------------------------------------------------- gradients


def test_every_module_including_the_future_heads_actually_updates() -> None:
    """SP A1.3: confirm the 5/30 min heads and the upstream encoder all move."""

    tl, seq = _timeline()
    model, cfg = _model(tl, use_future=True)
    targets = P.build_anchor_targets(tl, "train")
    model.future.initialise_from_targets(P._future_marginals(tl, targets))
    before = P._param_snapshot(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    P.run_session_pass(
        model, tl, seq, P.split_segment_ranges(tl, "train"), targets,
        torch.device("cpu"), cfg, train=True, optimizer=opt,
        weights={"future_0": 1.0, "future_1": 1.0}, rng=np.random.default_rng(0),
    )
    moved = P._update_magnitude(before, model)
    for group in ("encoder", "state", "heads", "future"):
        assert moved.get(group, 0.0) > 0.0, f"{group} did not move"


def test_future_loss_weights_are_frozen_after_the_initial_balance() -> None:
    """SP 2: weights are set once on TRAIN at init and never re-tuned."""

    tl, seq = _timeline()
    model, cfg = _model(tl, use_future=True)
    targets = P.build_anchor_targets(tl, "train")
    weights = P.balance_future_weights(
        model, tl, seq, P.split_segment_ranges(tl, "train"), targets,
        torch.device("cpu"), cfg, n_chunks=2, seed=0,
    )
    assert set(weights) == {"future_0", "future_1"}
    assert all(1e-3 <= v <= 1e3 for v in weights.values())
    again = P.balance_future_weights(
        model, tl, seq, P.split_segment_ranges(tl, "train"), targets,
        torch.device("cpu"), cfg, n_chunks=2, seed=0,
    )
    assert weights == again


def test_padded_slots_contribute_no_loss() -> None:
    """A batch wider than the number of segments must score exactly the same."""

    tl, seq = _timeline()
    model, cfg = _model(tl, use_future=False)
    targets = P.build_anchor_targets(tl, "train")
    ranges = P.split_segment_ranges(tl, "train")
    with torch.no_grad():
        tight, _ = P.run_session_pass(
            model, tl, seq, ranges, targets, torch.device("cpu"),
            replace(cfg, batch_segments=len(ranges)), train=False,
            rng=np.random.default_rng(0),
        )
        padded, _ = P.run_session_pass(
            model, tl, seq, ranges, targets, torch.device("cpu"),
            replace(cfg, batch_segments=len(ranges) + 5), train=False,
            rng=np.random.default_rng(0),
        )
    for key in tight:
        assert tight[key] == pytest.approx(padded[key], rel=1e-5, abs=1e-6)


# --------------------------------------------------------------------- A4 probes


def test_resetting_every_event_leaves_exactly_one_event_of_memory() -> None:
    """A4: K=1 means one-event memory, not zero.

    The reset happens before the step, so the state an anchor reads still holds
    the single event that immediately preceded it -- and nothing earlier.  The
    test perturbs every event *except* those immediate predecessors and requires
    the anchor states not to move; then perturbs a predecessor and requires that
    they do.
    """

    tl, seq = _timeline()
    model, cfg = _model(tl, use_future=False)
    targets = P.build_anchor_targets(tl, None)

    def _states() -> np.ndarray:
        with torch.no_grad():
            _m, extra = P.run_session_pass(
                model, tl, seq, P.full_segment_ranges(tl), targets,
                torch.device("cpu"), cfg, train=False, collect_states=True,
                rng=np.random.default_rng(0), reset_every_events=1,
            )
        return extra["anchor_state"]

    seen = targets.last_event_pos >= 0
    assert seen.sum() > 20
    predecessors = np.unique(targets.last_event_pos[seen])
    others = np.setdiff1d(np.arange(tl.event_times.size), predecessors)

    before = _states()
    for key in ("waveform", "band_features"):
        seq.arrays[key][others] *= 11.0
    assert np.allclose(before[seen], _states()[seen], atol=1e-5)

    for key in ("waveform", "band_features"):
        seq.arrays[key][predecessors] *= 11.0
    assert not np.allclose(before[seen], _states()[seen], atol=1e-3)


def test_a_physical_time_reset_shortens_memory_without_touching_the_clock() -> None:
    """A4: a 5 min reset must differ from full history but keep the anchor grid."""

    tl, seq = _timeline()
    model, cfg = _model(tl, use_future=False)
    targets = P.build_anchor_targets(tl, None)

    def _states(seconds: float) -> np.ndarray:
        with torch.no_grad():
            _m, extra = P.run_session_pass(
                model, tl, seq, P.full_segment_ranges(tl), targets,
                torch.device("cpu"), cfg, train=False, collect_states=True,
                rng=np.random.default_rng(0), reset_every_seconds=seconds,
            )
        return extra["anchor_state"]

    full = _states(0.0)
    short = _states(300.0)
    seen = targets.last_event_pos >= 0
    assert not np.allclose(full[seen], short[seen], atol=1e-4)
    assert np.isfinite(short[seen]).all()


# ------------------------------------------- what the memoryless control carries


def test_the_memoryless_arm_carries_only_the_time_since_the_last_event() -> None:
    """The whole H1 reading rests on this, so it is pinned rather than asserted.

    With ``persistent=False`` the update returns the learned initial vector, so an
    anchor's state is ``bias + (init - bias) * exp(-dt / tau)`` -- a 96-dimensional
    exponential basis in the time since the last event and nothing else.  It is
    therefore the control that separates "the model carries history" from "the
    baseline's single log(1 + dt) column was too rigid", which the block-shift
    null cannot do: shifting destroys the dt correspondence too.
    """

    tl, seq = _timeline()
    model, cfg = _model(tl, use_future=True)
    model.state.cfg = replace(model.state.cfg, persistent=False)
    targets = P.build_anchor_targets(tl, None)

    def _states() -> np.ndarray:
        with torch.no_grad():
            _m, extra = P.run_session_pass(
                model, tl, seq, P.full_segment_ranges(tl), targets,
                torch.device("cpu"), cfg, train=False, collect_states=True,
                rng=np.random.default_rng(0),
            )
        return extra["anchor_state"]

    before = _states()
    for key in ("waveform", "band_features", "band_envelope"):
        seq.arrays[key] *= 9.0
    seen = targets.last_event_pos >= 0
    assert seen.sum() > 20
    assert np.allclose(before[seen], _states()[seen], atol=1e-6), (
        "a memoryless state moved when only event content changed"
    )

    # Anchors that share a dt must share a state exactly.
    dt = np.round(tl.grid.seconds_since_last_event[seen], 9)
    states = before[seen]
    order = np.argsort(dt, kind="stable")
    dts, sts = dt[order], states[order]
    tied = np.flatnonzero(np.diff(dts) == 0)
    if tied.size:
        assert np.abs(sts[tied] - sts[tied + 1]).max() < 1e-6
