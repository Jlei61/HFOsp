"""A forked run's snapshot schedule has to live on the forked state's clock.

The bug this guards is silent and it corrupted every derived wear number in one batch of arms.  A
loaded state resumes the step counter of the trajectory it came from -- 12 s in, 240000 steps -- but
the schedule was built from zero.  Two things then go wrong at once, and neither raises:

* only the overlapping part of the schedule ever fires, so a 30 s run keeps 18 s of snapshots and
  the rest of the trajectory is simply absent;
* the template's own t=0 capture is still in the table, and its wear is zero because nothing had
  run when it was taken.  One row is enough to report a discharge as starting from no wear, its
  minimum wear as zero, and its first crossing of the departure level as 0 ms -- which is what the
  first three arms reported while actually starting at 0.436.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_topic4_fcxr_lc4_from_discharge as FK  # noqa: E402

DT, RUN_MS, SNAP_MS = 0.05, 30000.0, 250.0
FORK_STEP = 240000                                  # 12 s in, the state the arms fork from


def _state(step=FORK_STEP, stale=True):
    slow = types.SimpleNamespace(_step_i=step, snapshots={}, _snap_steps={})
    if stale:
        slow.snapshots["t0"] = dict(step=0, z_E=np.ones(4))     # the template's own capture
    return types.SimpleNamespace(slow=slow)


def test_the_schedule_covers_the_whole_run_after_a_fork():
    s = _state()
    base = FK._rekey_snapshots(s, RUN_MS, SNAP_MS, DT)
    assert base == FORK_STEP
    steps = sorted(s.slow._snap_steps)
    assert steps[0] == FORK_STEP, "the first capture must be at the fork, not before it"
    assert steps[-1] == FORK_STEP + int(round(RUN_MS / DT)), (
        "the last capture must be at the end of the run; a schedule built from zero stops early")
    assert len(steps) == int(RUN_MS / SNAP_MS) + 1


def test_the_template_capture_is_dropped():
    """Its wear is zero because nothing had run, and one such row poisons start, minimum and
    crossing all at once."""
    s = _state(stale=True)
    FK._rekey_snapshots(s, RUN_MS, SNAP_MS, DT)
    assert s.slow.snapshots == {}


def test_times_reported_against_the_fork_start_at_zero():
    s = _state()
    base = FK._rekey_snapshots(s, RUN_MS, SNAP_MS, DT)
    absolute = np.asarray(sorted(s.slow._snap_steps), float) * DT
    relative = absolute - base * DT
    assert relative[0] == 0.0 and relative[-1] == RUN_MS


def test_a_run_that_does_not_fork_is_unaffected():
    s = _state(step=0, stale=False)
    assert FK._rekey_snapshots(s, RUN_MS, SNAP_MS, DT) == 0
    assert min(s.slow._snap_steps) == 0
