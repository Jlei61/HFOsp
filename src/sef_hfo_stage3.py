"""Pure helpers for the Stage 3 stochastic-template-train pilot (cm-SNN two coexisting foci).

Core-level onset + collision/source labeling per
docs/superpowers/specs/2026-06-13-sef-hfo-snn-stage3-stochastic-template-train-design.md §1-§2.
These are deliberately side-effect-free (no engine, no I/O) so the labeling contract is unit-tested
independently of the simulation. The runner imports them; the simulation-integration lives there.

Design note (P1-4 fix): the source label is a CORE-LEVEL onset (when a focus's E-cell population
crosses a participation threshold), NOT a single-cell first spike, so a lone early spike at the
wrong end cannot flip the label.
"""
import math
from typing import List, Sequence, Tuple

import numpy as np


def core_participation_threshold(n_core_cells, n_min):
    """Active-fraction threshold for declaring a core 'ignited': max(1% of its cells, N_min cells).

    Returns a fraction in [0, 1]. The 1% floor guards tiny N_min on a large core; the N_min count
    guards a small core where 1% would be <1 cell.
    """
    if n_core_cells <= 0:
        raise ValueError("n_core_cells must be positive")
    return max(0.01, n_min / n_core_cells)


def first_crossing_time(series, bin_w, threshold, t_offset=0.0):
    """Absolute time (ms) of the FIRST bin in `series` at or above `threshold`, else None.

    `series` is a core's binned active-fraction restricted to the event window; bin i spans
    [t_offset + i*bin_w, t_offset + (i+1)*bin_w). Returns the bin's left edge time.
    """
    for i, v in enumerate(series):
        if v >= threshold:
            return t_offset + i * bin_w
    return None


def label_event(onset_neg, onset_pos, delta_onset, readable):
    """Hidden source label for one event: 'neg' | 'pos' | 'collision' | 'ambiguous' (spec §1).

    - not `readable` (axis unreadable: n_part < PART_MIN or axis_err is None) -> 'ambiguous'
    - neither core crosses (both onsets None) -> 'ambiguous'
    - both cores cross within `delta_onset` of each other -> 'collision'
    - otherwise -> the earlier core ('neg' or 'pos')

    A None onset means that core never crossed; it is treated as +inf, so a single-core crossing
    yields that core's label (NOT 'collision', NOT 'ambiguous' — one end genuinely fired).
    """
    if not readable:
        return "ambiguous"
    neg = math.inf if onset_neg is None else onset_neg
    pos = math.inf if onset_pos is None else onset_pos
    if neg == math.inf and pos == math.inf:
        return "ambiguous"
    if abs(neg - pos) <= delta_onset:          # both finite & near-simultaneous (inf-finite is huge)
        return "collision"
    return "neg" if neg < pos else "pos"


def collision_free_blocks(event_times, clean_for_timing
                          ) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """Partition events into maximal runs of CONSECUTIVE clean events (time order).

    Each run -> one (t_first, t_last) block. Censored events (clean_for_timing False)
    get block_id -1 and break the run, so they fall outside every block range and no
    transition is ever counted across them (spec §2, P1-1). The returned block ranges
    are the legacy `block_time_ranges` consumed by src.template_temporal_pairing.

    Returns (block_time_ranges, block_id_per_event) aligned to the INPUT order.
    """
    times = np.asarray(event_times, float)
    clean = np.asarray(clean_for_timing, bool)
    order = np.argsort(times, kind="stable")
    block_id = np.full(times.size, -1, dtype=int)
    blocks: List[Tuple[float, float]] = []
    cur: List[int] = []   # original indices of the current consecutive-clean run

    def _close() -> None:
        if cur:
            ts = times[cur]
            blocks.append((float(ts.min()), float(ts.max())))
            for j in cur:
                block_id[j] = len(blocks) - 1
            cur.clear()

    for i in order:
        if clean[i]:
            cur.append(int(i))
        else:
            _close()
    _close()
    return blocks, block_id


def synthetic_label_sequence(labels, mode: str, rng) -> np.ndarray:
    """Re-arrange a binary label array holding the MARGINAL COUNTS fixed (spec §4, P1-5).

    mode='alternating' -> maximal ping-pong; 'sticky' -> two maximal runs;
    'shuffle' -> random permutation (independent). The caller leaves event TIMES and
    the collision/block structure unchanged; only the label order is replaced.
    """
    labels = np.asarray(labels, int)
    if mode == "shuffle":
        return rng.permutation(labels)
    classes, counts = np.unique(labels, return_counts=True)
    if classes.size != 2:
        raise ValueError(f"synthetic controls require 2 classes, got {classes.tolist()}")
    a, b = int(classes[0]), int(classes[1])
    na, nb = int(counts[0]), int(counts[1])
    if mode == "sticky":
        return np.array([a] * na + [b] * nb, dtype=int)
    if mode == "alternating":
        k = min(na, nb)
        head = np.empty(2 * k, dtype=int)
        head[0::2] = a
        head[1::2] = b
        tail = [a] * (na - k) if na > nb else [b] * (nb - k)
        return np.concatenate([head, np.array(tail, dtype=int)])
    raise ValueError(f"unknown mode {mode!r}")


def entry_jitter_stats(first_contacts: Sequence) -> dict:
    """Distribution of the per-event first-active contact for one direction (spec §3.3).

    A single fixed contact -> n_unique 1, top1 1.0 (locked, no jitter). A wandering
    small group -> n_unique>1 with high top-k coverage (echoes real 'top-3 cover ~74%').
    None entries are dropped.
    """
    fc = [c for c in first_contacts if c is not None]
    n = len(fc)
    if n == 0:
        return {"n": 0, "n_unique": 0, "top1_fraction": float("nan"),
                "top3_fraction": float("nan"), "counts": {}}
    vals, cnts = np.unique(np.array(fc, dtype=object), return_counts=True)
    cnts_desc = np.sort(cnts)[::-1]
    return {
        "n": n,
        "n_unique": int(vals.size),
        "top1_fraction": float(cnts_desc[0]) / n,
        "top3_fraction": float(cnts_desc[:3].sum()) / n,
        "counts": {str(vals[i]): int(cnts[i]) for i in range(vals.size)},
    }
