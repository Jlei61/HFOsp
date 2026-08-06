"""Where an event started, and which of the two source regions started it.

The subject figure's two middle columns are keyed by *source identity*: one column per
template source region.  Choosing which event goes in which column, and where to put the
star that marks the event's source, therefore needs the event's actual ignition site --
not the sign of a correlation.

The distinction matters here because most pre-entry events are local patches around one
core rather than transits between the two.  For a patch, the sign of an onset-vs-axis
correlation reports which flank of the ignition point is longer, which is a different
question from where the event began.
"""
from __future__ import annotations

import numpy as np

EARLY_Q = 10.0      # per cent; the leading edge whose centroid locates the ignition
MIN_CELLS = 20      # below this an "event" is too sparse to locate
CLOSER_BY = 2.0     # a source region owns the ignition only if it is this many times nearer


def event_ignition_xy(onset_ms, pos, early_q=EARLY_Q, min_cells=MIN_CELLS):
    """Centroid of the first cells to fire, or None when too few cells fired.

    ``onset_ms`` is per-cell first-spike time with NaN for cells that never fired.
    """
    onset_ms = np.asarray(onset_ms, float)
    pos = np.asarray(pos, float)
    if onset_ms.shape[0] != pos.shape[0]:
        raise ValueError(f"onset ({onset_ms.shape[0]}) and pos ({pos.shape[0]}) disagree")
    fin = np.isfinite(onset_ms)
    if int(fin.sum()) < min_cells:
        return None
    t = onset_ms[fin]
    lead = t <= np.percentile(t, early_q)
    return pos[fin][lead].mean(axis=0)


def assign_to_source(ignition_xy, core_a, core_b, closer_by=CLOSER_BY):
    """Which source region ignited: ``"a"``, ``"b"``, or None when neither clearly owns it.

    A region owns the ignition when it is at least ``closer_by`` times nearer than the
    other.  An event starting midway between them is not attributable to either end, and
    saying so is the point -- forcing it onto the nearer region would put an event with no
    resolvable origin into a column that claims one.
    """
    if ignition_xy is None:
        return None
    ignition_xy = np.asarray(ignition_xy, float)
    d_a = float(np.linalg.norm(ignition_xy - np.asarray(core_a, float)))
    d_b = float(np.linalg.norm(ignition_xy - np.asarray(core_b, float)))
    if d_a <= 0.0 and d_b <= 0.0:
        return None
    if d_b >= closer_by * d_a:
        return "a"
    if d_a >= closer_by * d_b:
        return "b"
    return None


def classify_events(onset_maps, pos, core_a, core_b, **kw):
    """Per-event ignition site and source assignment, in input order."""
    out = []
    for onset in np.asarray(onset_maps, float):
        xy = event_ignition_xy(onset, pos, **{k: v for k, v in kw.items()
                                              if k in ("early_q", "min_cells")})
        side = assign_to_source(xy, core_a, core_b,
                                **{k: v for k, v in kw.items() if k == "closer_by"})
        d_a = d_b = None
        if xy is not None:
            d_a = float(np.linalg.norm(xy - np.asarray(core_a, float)))
            d_b = float(np.linalg.norm(xy - np.asarray(core_b, float)))
        out.append(dict(ignition_xy=(None if xy is None else [float(xy[0]), float(xy[1])]),
                        source=side, dist_a_mm=d_a, dist_b_mm=d_b))
    return out


def pick_representatives(classified):
    """One event per source region: the one whose ignition sits nearest that region.

    Returns ``(idx_a, idx_b)`` with None where a region ignited nothing.
    """
    def _nearest(side, key):
        cand = [(i, c[key]) for i, c in enumerate(classified) if c["source"] == side]
        return min(cand, key=lambda p: p[1])[0] if cand else None
    return _nearest("a", "dist_a_mm"), _nearest("b", "dist_b_mm")
