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


def synthetic_label_sequence(labels, mode: str, rng, block_id=None) -> np.ndarray:
    """Re-arrange a binary label array holding the MARGINAL COUNTS fixed (spec §4, P1-5).

    mode='alternating' -> maximal ping-pong; 'sticky' -> two maximal runs;
    'shuffle' -> random permutation (independent). The caller leaves event TIMES and
    the collision/block structure unchanged; only the label order is replaced.

    block_id (P1-b): when given (aligned to `labels`, -1 = censored/out-of-block), the control
    is generated INDEPENDENTLY within each collision-free block, because the block-aware timing
    test resets runs at block boundaries (`compute_runs`). A globally-sticky sequence whose run
    straddles a boundary would be split by the test and under-state stickiness; per-block
    generation makes the control maximally dependent AS THE TEST MEASURES IT. Censored events
    (block_id < 0) keep their original label (they fall outside every block range). Single-class
    blocks (only one end fired) are left as-is — they cannot alternate/split.
    """
    labels = np.asarray(labels, int)
    if block_id is None:
        return _rearrange_labels(labels, mode, rng, allow_single_class=False)
    block_id = np.asarray(block_id, int)
    out = labels.copy()
    for b in np.unique(block_id):
        if b < 0:
            continue
        idx = np.flatnonzero(block_id == b)
        out[idx] = _rearrange_labels(labels[idx], mode, rng, allow_single_class=True)
    return out


def _rearrange_labels(labels, mode, rng, allow_single_class) -> np.ndarray:
    """Marginal-preserving re-arrangement of ONE label run (the per-block / global primitive)."""
    labels = np.asarray(labels, int)
    if mode == "shuffle":
        return rng.permutation(labels)
    classes, counts = np.unique(labels, return_counts=True)
    if classes.size == 1:
        if allow_single_class:
            return labels.copy()
        raise ValueError(f"synthetic controls require 2 classes, got {classes.tolist()}")
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


def core_active_fraction(spk, core_e_idx, dt, bin_ms, t_on, t_off) -> np.ndarray:
    """Binned active fraction of ONE core's E cells inside [t_on, t_off]. `core_e_idx` are
    E-NEURON column indices of `spk` (the (n_steps, NE) E-spike-bool matrix). Bin i spans
    [t_on + i*bin_ms, ...); pair with first_crossing_time(..., t_offset=t_on) for an absolute onset.
    Pure numpy (no engine) so the labeling contract is unit-testable without a simulation."""
    core_e_idx = np.asarray(core_e_idx)
    bs = max(1, int(round(bin_ms / dt)))
    s, e = int(round(t_on / dt)), int(round(t_off / dt))
    seg = np.asarray(spk)[s:e][:, core_e_idx]
    nb = seg.shape[0] // bs
    if nb == 0 or core_e_idx.size == 0:
        return np.zeros(0)
    binned = seg[:nb * bs].reshape(nb, bs, -1).any(axis=1)
    return binned.mean(axis=1)


def build_sidecar(ev_recs, spk, core_masks, NE, *, dt, bin_ms, part_min, delta_onset, n_min) -> dict:
    """Per RETURNED event (== legacy-record column order) hidden core-level source label + collision
    flag + clean_for_timing. Pure given (ev_recs, spk, core_masks): no sim, no I/O -> unit-testable.

    Aligns 1:1 to the record columns (the runner builds the record from the SAME returned events, in
    the SAME order), so `event_id` == lagPat column index and `raw_event_index` keeps the original
    detect_events position (plan P1-3). The packedTimes hard assert (packed[:,0] == t_on/1000) holds
    because both the sidecar `t_on` and the record window onset derive from this same `ev_recs` t_on.
    """
    neg_idx = np.flatnonzero(np.asarray(core_masks[0])[:NE])
    pos_idx = np.flatnonzero(np.asarray(core_masks[1])[:NE])
    frac_neg = core_participation_threshold(neg_idx.size, n_min) if neg_idx.size else 1.0
    frac_pos = core_participation_threshold(pos_idx.size, n_min) if pos_idx.size else 1.0
    sidecar: List[dict] = []
    for raw_i, e in enumerate(ev_recs):
        if not e["returned"]:
            continue
        af_neg = core_active_fraction(spk, neg_idx, dt, bin_ms, e["t_on"], e["t_off"])
        af_pos = core_active_fraction(spk, pos_idx, dt, bin_ms, e["t_on"], e["t_off"])
        on_neg = first_crossing_time(af_neg, bin_ms, frac_neg, t_offset=e["t_on"])
        on_pos = first_crossing_time(af_pos, bin_ms, frac_pos, t_offset=e["t_on"])
        readable = e["axis_err"] is not None and e["n_part"] >= part_min
        hidden = label_event(on_neg, on_pos, delta_onset, readable)
        reason = ("unreadable_axis" if not readable else
                  "no_core_crossing" if (on_neg is None and on_pos is None) else
                  "simultaneous_onset" if hidden == "collision" else "none")
        clean_t = (hidden in ("neg", "pos") and readable and e["axis_err"] < 25)
        diag_k = 8
        sidecar.append(dict(
            event_id=len(sidecar), raw_event_index=raw_i, t_on=e["t_on"], t_off=e["t_off"],
            event_peak_t=e.get("event_peak_t"), hidden_source_label=hidden,
            core_onset_neg=(None if on_neg is None else round(on_neg, 1)),
            core_onset_pos=(None if on_pos is None else round(on_pos, 1)),
            collision_reason=reason, clean_for_timing=bool(clean_t),
            # diagnostic (user 2026-06-13): the first bins of each core's active fraction let an
            # auditor tell a REAL co-ignition (both cores ramp together) from a too-sensitive 1%
            # onset (one core just tickles threshold). thresholds echoed per-event for self-containment.
            core_frac_neg_first_bins=np.round(af_neg[:diag_k], 4).tolist(),
            core_frac_pos_first_bins=np.round(af_pos[:diag_k], 4).tolist(),
            core_threshold_neg=round(frac_neg, 4), core_threshold_pos=round(frac_pos, 4),
            n_part=e["n_part"], axis_err=e["axis_err"], sign=e["sign"],
            readability=e.get("readability")))
    # block_id_after_collision_censoring: censored (non-clean) events break the consecutive-clean
    # run so no transition is counted across them (spec §2, P1-1). Reuses collision_free_blocks.
    if sidecar:
        _bt = np.array([s["t_on"] for s in sidecar], float)
        _bc = np.array([s["clean_for_timing"] for s in sidecar], bool)
        _, _bid = collision_free_blocks(_bt, _bc)
        for s, b in zip(sidecar, _bid):
            s["block_id_after_collision_censoring"] = int(b)
    n_coll = sum(1 for s in sidecar if s["hidden_source_label"] == "collision")
    return dict(n_record_events=len(sidecar),
                config=dict(delta_onset=delta_onset, n_min=n_min,
                            n_core_neg=int(neg_idx.size), n_core_pos=int(pos_idx.size),
                            frac_neg=round(frac_neg, 4), frac_pos=round(frac_pos, 4)),
                collision_rate=round(n_coll / max(1, len(sidecar)), 4),
                events=sidecar)


def pilot_gate(collision_rate, neg_clean, pos_clean, n_events, ambiguous,
               bidir_seed_frac=1.0, sign_ok=True,
               coll_max=0.30, clean_min=3, min_events=6):
    """Pre-registered Stage 3 regime-screen gate, applied per (sep,std,mean[,drive]) cell on the
    MEDIAN over its seeds. Returns (passed: bool, reason: str, flags: dict).

    Encodes the conclusion (MEMORY: a gate must encode the conclusion, not just existence): a cell
    only PASSES if it is a usable BALANCED bidirectional low-collision regime. Failures carry a
    specific reason so a too-cold ('no_events') / co-igniting ('high_collision') / one-end-dominant
    ('source_imbalance') / single-direction ('unidirectional') regime can never silently pass.

    `bidir_seed_frac` = fraction of the cell's seeds with BOTH directions present; only enforced
    when `sign_ok` (the oneend sign sanity validated the read-out direction). If sign is NOT
    trusted, direction is ignored and the source-based sub-gates decide.
    """
    amb_rate = round((ambiguous or 0) / n_events, 3) if n_events else None
    flags = {
        "collision_ok": collision_rate is not None and collision_rate < coll_max,
        "source_balance_ok": (neg_clean or 0) >= clean_min and (pos_clean or 0) >= clean_min,
        "enough_events": (n_events or 0) >= min_events,
        "bidir_ok": (not sign_ok) or bidir_seed_frac >= 0.5,
        "ambiguous_rate": amb_rate,
    }
    if not flags["enough_events"]:
        return False, "no_events", flags
    if not flags["collision_ok"]:
        return False, "high_collision", flags
    if not flags["source_balance_ok"]:
        return False, "source_imbalance", flags
    if not flags["bidir_ok"]:
        return False, "unidirectional", flags
    return True, "pass", flags
