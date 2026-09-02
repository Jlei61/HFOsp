"""Toy scaffold for v0.3.3 assay tests (helper module, not a test module).

Two recorded sessions separated by a gap, one seizure inside the first session
(so there is a post-seizure target segment in the same session), a 5-minute
anchor grid, the frozen 60/70/80 recorded-time partition, three horizons and a
small contact vocabulary.  Everything the real scaffold exposes, at toy size.
"""
from __future__ import annotations

import numpy as np

from src.topic5_group_event_state.v02.timeline import RecordedSession, build_anchor_grid, build_carry_segments
from src.topic5_group_event_state.v032_eval.partition import eval_partition
from src.topic5_group_event_state.v033_evaluator import boundaries as B
from src.topic5_group_event_state.v033_evaluator.scaffold import Scaffold

HORIZONS = (300.0, 1800.0, 7200.0)


def toy_scaffold(seed: int = 0, *, rate_per_second: float = 0.03, n_contacts: int = 6,
                 carry: str = "segment") -> Scaffold:
    rng = np.random.default_rng(seed)
    sessions = [RecordedSession(0, 0.0, 60_000.0), RecordedSession(1, 80_000.0, 130_000.0)]
    seizures = [{"onset_epoch": 30_000.0, "offset_epoch": 30_090.0}]
    segments = build_carry_segments(sessions, seizures, postictal_exclusion_seconds=3_600.0,
                                    min_segment_seconds=300.0)
    partition = eval_partition(segments)
    events = np.sort(np.concatenate([
        rng.uniform(s.start_epoch, s.stop_epoch, int(s.duration_seconds * rate_per_second)) for s in segments
    ]))
    grid = build_anchor_grid(segments, partition, events, horizons_seconds=HORIZONS,
                             grid_seconds=300.0, min_warmup_seconds=300.0)
    event_segment = B.anchor_carry_index(events, segments)
    event_session = np.asarray([segments[i].session_id for i in event_segment], dtype=np.int64)
    if carry == "segment":
        event_carry, anchor_carry, last = event_segment, grid.segment_index, grid.last_event_pos
    else:
        event_carry, anchor_carry = event_session, grid.session_id
        last = B.carry_last_event(events, event_carry, np.ones(events.size, bool), grid.t_anchor, anchor_carry)
    eligible = np.stack([B.target_window_valid(grid.t_anchor, h, segments, partition) for h in HORIZONS], axis=1)
    participation = rng.uniform(size=(events.size, n_contacts)) < 0.35
    log_mu_h = {int(h): np.full(grid.n_anchors, np.log(max(rate_per_second * h, 1e-3))) + rng.normal(0, 0.05, grid.n_anchors)
                for h in HORIZONS}
    return Scaffold(
        subject="toy", horizons=HORIZONS, t_anchor=grid.t_anchor, anchor_carry=np.asarray(anchor_carry, np.int64),
        anchor_phase=partition.labels_of(grid.t_anchor), eligible=eligible,
        window_lo=grid.window_lo, window_hi=grid.window_hi, last_event_pos=np.asarray(last, np.int64),
        event_times=events, event_carry=np.asarray(event_carry, np.int64), event_phase=partition.labels_of(events),
        participation=participation, log_mu_h=log_mu_h, log_r_h={int(h): np.log(5.0) for h in HORIZONS},
        segment_bounds=np.array([[s.start_epoch, s.stop_epoch] for s in segments]),
        phase_bounds=np.array([list(partition.bounds(p)) for p in ("base_fit", "inner_val", "dev_val", "dev_test")]),
        carry=carry, provenance={"toy": True, "seed": seed},
    )
