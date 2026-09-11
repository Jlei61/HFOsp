"""Sparse future-block targets: prefix sums, never a dense tensor (clause C5).

A dense target for this question would be
``(anchors x horizons x contacts x bands)`` -- for one Epilepsiae patient that is
~40k anchors x 3 horizons x 16 contacts x 5 bands, repeated for every arm, and it
would still be the *wrong* object because a window holds a variable number of
events.

The trick that removes it: every endpoint the plan asks for is a **per-event
proper score**, and every per-event proper score used here depends on the window
only through first and second moments plus counts.  So

    count            -> prefix event counts
    participation    -> prefix sums of the boolean participation matrix
    continuous marks -> prefix sums of x and x**2 over valid events

and a window statistic is one subtraction.  The result is not an approximation of
the dense computation; ``test_window_statistics_match_brute_force`` checks it is
bit-comparable to a literal loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .marks import EventMarks


@dataclass(frozen=True)
class WindowStats:
    """Sufficient statistics of every event inside a set of half-open windows."""

    count: np.ndarray               # (A,) int64 -- all events, the p(N) target
    n_valid_mark: np.ndarray        # (A,) int64 -- events with a finite mark
    sum_participation: np.ndarray   # (A, C) int64
    sum_x: np.ndarray               # (A, D) float64, valid events only
    sum_x2: np.ndarray              # (A, D) float64, valid events only

    @property
    def n_windows(self) -> int:
        return int(self.count.size)


class FutureTargetBuilder:
    """Prefix sums of one patient's mark stream."""

    def __init__(self, marks: EventMarks):
        self.marks = marks
        n = marks.n_events
        part = marks.participation.astype(np.int64)
        valid = marks.valid
        x = np.where(valid[:, None], marks.continuous, 0.0)
        self._cum_n = np.arange(n + 1, dtype=np.int64)
        self._cum_valid = np.concatenate([[0], np.cumsum(valid.astype(np.int64))])
        self._cum_part = np.concatenate(
            [np.zeros((1, marks.n_contacts), dtype=np.int64), np.cumsum(part, axis=0)]
        )
        self._cum_x = np.concatenate(
            [np.zeros((1, marks.n_continuous)), np.cumsum(x, axis=0)]
        )
        self._cum_x2 = np.concatenate(
            [np.zeros((1, marks.n_continuous)), np.cumsum(x ** 2, axis=0)]
        )

    def prefix_bytes(self) -> int:
        return int(
            self._cum_n.nbytes
            + self._cum_valid.nbytes
            + self._cum_part.nbytes
            + self._cum_x.nbytes
            + self._cum_x2.nbytes
        )

    def window_stats(self, lo: np.ndarray, hi: np.ndarray) -> WindowStats:
        """Statistics of events ``[lo, hi)`` for many windows at once."""

        a = np.asarray(lo, dtype=np.int64)
        b = np.asarray(hi, dtype=np.int64)
        if a.shape != b.shape:
            raise ValueError("lo and hi must have the same shape")
        if np.any(b < a):
            raise ValueError("window end precedes its start")
        n = self.marks.n_events
        if np.any(a < 0) or np.any(b > n):
            raise ValueError("window indices outside the event stream")
        return WindowStats(
            count=(self._cum_n[b] - self._cum_n[a]),
            n_valid_mark=(self._cum_valid[b] - self._cum_valid[a]),
            sum_participation=(self._cum_part[b] - self._cum_part[a]),
            sum_x=(self._cum_x[b] - self._cum_x[a]),
            sum_x2=(self._cum_x2[b] - self._cum_x2[a]),
        )
