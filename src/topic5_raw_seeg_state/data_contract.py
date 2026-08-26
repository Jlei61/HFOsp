"""Continuous-time index + eligibility layer for the Raw-SEEG R0.1 model.

Owner: **Worker A**.  Everything here compiles against the frozen constants in
:mod:`src.topic5_raw_seeg_state.contract` — no constant is redefined locally.

Scientific spec : ``docs/archive/topic5/raw_seeg_state_scientific_spec_2026-08-21.md`` §4
Execution plan  : ``docs/archive/topic5/raw_seeg_state_execution_plan_2026-08-21.md`` §3 (A1-A4)

What this module produces (per subject)
---------------------------------------
1. ``build_subject_blocks``   -> recorded block intervals + session id + split
2. ``build_contact_metadata`` -> the bipolar montage that IS the model channel set
3. ``build_minute_index``     -> the 60 s eligibility grid (context / horizon flags)

Conventions that downstream workers must honour
-----------------------------------------------
``native_index_anode`` / ``native_index_cathode``
    0-based index into the **full native channel list of the source file**:

    * Epilepsiae -> position in the ``.head`` ``elec_names`` list, i.e. the
      column order of the interleaved int16 ``.data`` array.
    * Yuquan     -> position in the EDF *signal* list (all ``n_signals``
      entries, including non-SEEG and ``EDF Annotations``).

    ``data_audit.json`` stores ``native_channel_names`` per subject so a
    consumer can re-derive the mapping by name instead of trusting the integer.

``channel_index``
    0-based row order of ``contact_metadata``, sorted by ``(shaft, shaft_index)``.
    Every downstream array (raw cache columns, spectral target, decoder rows)
    must use this order.

``contact_valid`` vs ``coord_valid`` vs ``coord_mode``  (coordinator ruling 1)
    Two independent axes.  ``contact_valid`` is **electrical**: both endpoints
    exist in the native layout of every dev block, both are intracranial, and
    the native index is addressable.  ``coord_valid`` is **anatomical**: the
    contact has an mm coordinate.  A missing coordinate NEVER clears
    ``contact_valid`` — five Yuquan subjects (chenziyang / gaolan / hanyuxuan /
    sunyuanxin / wangyiyang) have perfectly good recordings but no electrode
    localisation anywhere on the mount.  ``drop_reason`` still records
    ``missing_coordinate`` so the two axes stay auditable, and ``coord_mode``
    is ``contract.COORD_MODE_FULL`` when at least one contact of the subject
    has a coordinate, else ``contract.COORD_MODE_TOPOLOGY_ONLY``.

``shaft_index``
    0-based position of the bipolar pair **along its shaft**, taken from the
    anode ordinal order and re-densified over ``contact_valid`` rows so the
    values are gap-free per shaft (the model embeds it as a categorical).
    Invalid rows carry ``-1``.  ``anode``/``cathode`` keep the true contact
    names, so the original ordinals are never lost.

Seizure guard source  (coordinator ruling 2)
    Yuquan guards are the UNION of ``contract.YUQUAN_SEIZURE_INVENTORY`` and the
    raw EDF-annotation scan in ``contract.YUQUAN_SUPPLEMENTARY_SEIZURE_DIR``
    (``pr1_seizure_<subject>.json``), de-duplicated by onset within 1 s.  Any
    onset without a *usable* offset (missing, or ``offset <= onset``) uses
    ``contract.SEIZURE_OFFSET_FALLBACK_SECONDS``.  ``seizure_guard_source`` is
    one of ``inventory`` / ``inventory+annotation_scan`` /
    ``annotation_scan_only`` / ``none_found``.  **``none_found`` is not a pass:
    absence of an annotation is not evidence of absence of a seizure**, and
    those subjects stay ``DEGRADED_NO_SEIZURE_GUARD``.

``n_valid_contacts`` / ``minute_usable`` at stage A3
    ``build_minute_index`` writes ``n_valid_contacts = -1`` and
    ``minute_usable = covered & guard_free & (session_id >= 0)``.  The artifact
    rule of spec §4.3 (6 robust SD broadband outlier, ADC saturation,
    ``MINUTE_MIN_VALID_CONTACT_FRACTION``) needs the decimated signal, which is
    Worker B's pass.  **B must call**
    :func:`refine_minute_index_with_artifacts` with its per-contact-minute
    validity mask; that function recomputes ``n_valid_contacts``,
    ``minute_usable``, ``ctx_ok`` and every ``h*_ok`` consistently.  It is the
    only sanctioned way to update those columns.

Rules implemented verbatim from the spec / task contract
--------------------------------------------------------
* session: ``gap > SESSION_JOIN_SECONDS`` opens a new session (``==`` does not).
* coverage: a minute is ``covered`` when strictly more than
  ``MINUTE_COVERAGE_FRACTION * 60`` seconds of it lie inside recorded blocks.
  (Spec §4.3 prose says ">= 95 %"; the task contract pins 57 s -> False and
  58 s -> True, which is the strict reading.  The two differ only on an exact
  tie and the strict form is the conservative one.  ``COVERAGE_RULE`` records
  the choice in ``data_audit.json``.)
* coverage is additionally False for the single minute that would straddle
  ``dev_end_epoch`` — reading it would touch the sealed partition.
* guard: ``[eeg_onset - PREICTAL_GUARD_SECONDS, eeg_offset + POSTICTAL_GUARD_SECONDS]``,
  with ``eeg_offset := eeg_onset + 120 s`` when the inventory offset is missing.
  Seizures *after* ``dev_end`` are still used, because their preictal guard
  reaches back into the dev window.
* horizon ``h`` at minute ``t`` is eligible when the whole closed range
  ``[t - CONTEXT_MINUTES + 1, t + h]`` is guard-free, stays inside one session
  (id >= 0) and one split, ``ctx_ok[t]`` holds, ``minute_usable[t + h]`` holds
  and ``t + h`` is inside the grid.  Intervening minutes may be *uncovered*
  (spec §4.4 explicitly allows micro-gaps).
"""

from __future__ import annotations

import glob
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.topic5_raw_seeg_state import contract

# Re-use, do not re-invent (CLAUDE.md §6).
from src.preprocessing import (  # noqa: E402
    ElectrodeParser,
    _build_bipolar_pairs,
    _parse_edf_header_for_streaming,
    _read_epilepsiae_head_for_streaming,
    read_edf_record_info,
)
from src.epilepsiae_dataset import SCALP_OR_AUX_CHANNELS  # noqa: E402
from src.seeg_coord_loader import load_subject_coords  # noqa: E402


SECONDS_PER_MINUTE = 60.0
COVERAGE_MIN_SECONDS = contract.MINUTE_COVERAGE_FRACTION * SECONDS_PER_MINUTE
COVERAGE_RULE = ("max_contiguous_recorded_seconds > MINUTE_COVERAGE_FRACTION*60 "
                 "(strict) AND minute end <= dev_end_epoch")
MISSING_OFFSET_FALLBACK_SECONDS = contract.SEIZURE_OFFSET_FALLBACK_SECONDS
SEIZURE_DEDUP_SECONDS = 1.0
NATIVE_INDEX_REFERENCE = {
    "epilepsiae": "index into .head elec_names (== .data interleaved column order)",
    "yuquan": "index into the EDF signal list (all n_signals entries)",
}

_SPLIT_CODE = {"train": 0, "validation": 1, "sealed": 2}


# ---------------------------------------------------------------------------
# cached frozen inputs
# ---------------------------------------------------------------------------

_CACHE: Dict[str, object] = {}


def _epilepsiae_blocks() -> pd.DataFrame:
    if "epi_blocks" not in _CACHE:
        _CACHE["epi_blocks"] = pd.read_csv(contract.EPILEPSIAE_BLOCK_INVENTORY)
    return _CACHE["epi_blocks"]  # type: ignore[return-value]


def _yuquan_blocks() -> pd.DataFrame:
    if "yq_blocks" not in _CACHE:
        _CACHE["yq_blocks"] = pd.read_csv(contract.YUQUAN_BLOCK_INVENTORY)
    return _CACHE["yq_blocks"]  # type: ignore[return-value]


def _epilepsiae_seizures() -> pd.DataFrame:
    if "epi_sz" not in _CACHE:
        _CACHE["epi_sz"] = pd.read_csv(contract.EPILEPSIAE_SEIZURE_INVENTORY)
    return _CACHE["epi_sz"]  # type: ignore[return-value]


def _yuquan_seizures() -> pd.DataFrame:
    if "yq_sz" not in _CACHE:
        _CACHE["yq_sz"] = pd.read_csv(contract.YUQUAN_SEIZURE_INVENTORY)
    return _CACHE["yq_sz"]  # type: ignore[return-value]


def split_subject_key(subject: str) -> Tuple[str, str]:
    """``"epilepsiae_958"`` -> ``("epilepsiae", "958")``."""
    dataset, _, native = subject.partition("_")
    if dataset not in ("epilepsiae", "yuquan") or not native:
        raise ValueError(f"unrecognised cohort subject key: {subject!r}")
    return dataset, native


# ---------------------------------------------------------------------------
# A1 primitives — sessions, grid, coverage, guards, splits
# ---------------------------------------------------------------------------


def assign_sessions(
    block_starts: Sequence[float],
    block_ends: Sequence[float],
    join_seconds: float = contract.SESSION_JOIN_SECONDS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological session assignment.

    Returns ``(gap_to_prev_sec, opens_session, session_id)`` in **sorted**
    (chronological) order.  ``gap_to_prev_sec[0]`` is NaN.  A gap of exactly
    ``join_seconds`` does *not* open a session (rule is ``>``).
    """
    starts = np.asarray(block_starts, dtype=float)
    ends = np.asarray(block_ends, dtype=float)
    if starts.shape != ends.shape:
        raise ValueError("block_starts and block_ends must have the same shape")
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    n = starts.size
    gap = np.full(n, np.nan, dtype=float)
    if n > 1:
        gap[1:] = starts[1:] - ends[:-1]
    opens = np.zeros(n, dtype=bool)
    if n:
        opens[0] = True
    if n > 1:
        opens[1:] = gap[1:] > join_seconds
    session_id = np.cumsum(opens) - 1
    return gap, opens, session_id.astype(np.int64)


def session_extents(
    block_starts: Sequence[float],
    block_ends: Sequence[float],
    session_id: Sequence[int],
) -> np.ndarray:
    """``(n_sessions, 2)`` array of ``[session_start, session_end]``.

    Input must already be chronological and aligned with ``assign_sessions``'
    output ordering.
    """
    starts = np.asarray(block_starts, dtype=float)
    ends = np.asarray(block_ends, dtype=float)
    sid = np.asarray(session_id, dtype=np.int64)
    if sid.size == 0:
        return np.zeros((0, 2), dtype=float)
    n_sessions = int(sid.max()) + 1
    out = np.zeros((n_sessions, 2), dtype=float)
    for s in range(n_sessions):
        sel = sid == s
        out[s, 0] = float(starts[sel].min())
        out[s, 1] = float(ends[sel].max())
    return out


def minute_grid_starts(first_epoch: float, dev_end_epoch: float) -> np.ndarray:
    """Minute-grid origins: ``first_epoch + 60 k`` for every ``k`` with start < seal."""
    span = float(dev_end_epoch) - float(first_epoch)
    if span <= 0:
        return np.zeros(0, dtype=float)
    n = int(math.ceil(span / SECONDS_PER_MINUTE))
    starts = float(first_epoch) + SECONDS_PER_MINUTE * np.arange(n, dtype=float)
    return starts[starts < float(dev_end_epoch)]


def _merge_intervals(starts: np.ndarray, ends: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Union of possibly overlapping intervals, sorted by start."""
    if starts.size == 0:
        return starts, ends
    order = np.argsort(starts, kind="stable")
    s, e = starts[order], ends[order]
    out_s = [s[0]]
    out_e = [e[0]]
    for i in range(1, s.size):
        if s[i] <= out_e[-1]:
            out_e[-1] = max(out_e[-1], e[i])
        else:
            out_s.append(s[i])
            out_e.append(e[i])
    return np.asarray(out_s, dtype=float), np.asarray(out_e, dtype=float)


def minute_covered_seconds(
    minute_starts: Sequence[float],
    block_starts: Sequence[float],
    block_ends: Sequence[float],
    minute_seconds: float = SECONDS_PER_MINUTE,
) -> np.ndarray:
    """Recorded seconds inside each ``[minute_start, minute_start + 60)`` window.

    Overlapping blocks are merged first, so slightly overlapping EDF files
    (the Yuquan header route routinely yields ~-0.06 s "gaps") cannot make a
    minute look more than 100 % covered.
    """
    m = np.asarray(minute_starts, dtype=float)
    s, e = _merge_intervals(
        np.asarray(block_starts, dtype=float), np.asarray(block_ends, dtype=float)
    )
    if s.size == 0:
        return np.zeros(m.shape, dtype=float)
    lengths = e - s
    cum = np.concatenate([[0.0], np.cumsum(lengths)])[:-1]  # cum[i] = length before i

    def _prefix(t: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(s, t, side="right") - 1
        out = np.zeros(t.shape, dtype=float)
        ok = idx >= 0
        if ok.any():
            j = idx[ok]
            out[ok] = cum[j] + np.clip(t[ok] - s[j], 0.0, lengths[j])
        return out

    return _prefix(m + minute_seconds) - _prefix(m)


def minute_max_contiguous_seconds(
    minute_starts: Sequence[float],
    block_starts: Sequence[float],
    block_ends: Sequence[float],
    minute_seconds: float = SECONDS_PER_MINUTE,
) -> np.ndarray:
    """Longest UNBROKEN recorded stretch inside each minute.

    This, not the summed coverage, is what decides ``covered``. Epilepsiae's
    hourly blocks abut with ~1 s gaps, so a minute straddling a block boundary
    sums to 59 of 60 s and would pass a summed-coverage test -- while actually
    containing a recorder discontinuity. Splicing across that gap injects a step
    into a 60 s Welch estimate and splatters broadband power into every band of
    that minute's target, so such a minute must not be a training target at all.
    Dropping it costs about one minute per block boundary (~2 % for a subject
    with 256 hourly blocks) and removes a whole class of silent artifact.
    """
    m = np.asarray(minute_starts, dtype=float)
    s_, e_ = _merge_intervals(
        np.asarray(block_starts, dtype=float), np.asarray(block_ends, dtype=float)
    )
    out = np.zeros(m.shape, dtype=float)
    if s_.size == 0 or m.size == 0:
        return out
    order = np.argsort(m, kind="stable")
    ms = m[order]
    acc = np.zeros(ms.shape, dtype=float)
    for a, b in zip(s_, e_):
        lo = int(np.searchsorted(ms, a - minute_seconds, side="right"))
        hi = int(np.searchsorted(ms, b, side="left"))
        if hi <= lo:
            continue
        seg = ms[lo:hi]
        ov = np.minimum(b, seg + minute_seconds) - np.maximum(a, seg)
        np.maximum(acc[lo:hi], np.clip(ov, 0.0, minute_seconds), out=acc[lo:hi])
    out[order] = acc
    return out


def covered_from_seconds(
    covered_seconds: Sequence[float],
    threshold_seconds: float = COVERAGE_MIN_SECONDS,
) -> np.ndarray:
    """Strict coverage rule — see ``COVERAGE_RULE``."""
    return np.asarray(covered_seconds, dtype=float) > float(threshold_seconds)


def minute_session_ids(
    minute_starts: Sequence[float],
    extents: np.ndarray,
    minute_seconds: float = SECONDS_PER_MINUTE,
) -> np.ndarray:
    """Session containing each minute's midpoint; ``-1`` when none does."""
    m = np.asarray(minute_starts, dtype=float) + 0.5 * minute_seconds
    out = np.full(m.shape, -1, dtype=np.int64)
    if extents.size == 0:
        return out
    idx = np.searchsorted(extents[:, 0], m, side="right") - 1
    ok = idx >= 0
    if ok.any():
        j = idx[ok]
        inside = m[ok] < extents[j, 1]
        sel = np.flatnonzero(ok)[inside]
        out[sel] = j[inside]
    return out


def guard_intervals_from_seizures(
    onsets: Sequence[float],
    offsets: Sequence[float],
    preictal_seconds: float = contract.PREICTAL_GUARD_SECONDS,
    postictal_seconds: float = contract.POSTICTAL_GUARD_SECONDS,
) -> np.ndarray:
    """``(n, 2)`` guard intervals; onsets without a usable offset fall back to +120 s.

    "Usable" means finite AND strictly greater than the onset — a zero-duration
    annotation mark (``offset == onset``) is an onset without an offset, and both
    the frozen inventory and the annotation scan contain such marks.
    """
    on = np.asarray(onsets, dtype=float)
    off = np.asarray(offsets, dtype=float)
    keep = np.isfinite(on)
    on, off = on[keep], off[keep]
    fallback = ~np.isfinite(off) | (off <= on)
    off = np.where(fallback, on + MISSING_OFFSET_FALLBACK_SECONDS, off)
    if on.size == 0:
        return np.zeros((0, 2), dtype=float)
    guards = np.stack([on - float(preictal_seconds), off + float(postictal_seconds)], axis=1)
    return guards[np.argsort(guards[:, 0], kind="stable")]


def count_offset_fallbacks(onsets: Sequence[float], offsets: Sequence[float]) -> int:
    on = np.asarray(onsets, dtype=float)
    off = np.asarray(offsets, dtype=float)
    keep = np.isfinite(on)
    on, off = on[keep], off[keep]
    return int(np.sum(~np.isfinite(off) | (off <= on)))


def merge_seizure_sources(
    primary: Tuple[Sequence[float], Sequence[float]],
    supplement: Tuple[Sequence[float], Sequence[float]],
    dedup_seconds: float = SEIZURE_DEDUP_SECONDS,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Union of two (onsets, offsets) sources, de-duplicated by onset.

    A supplementary onset within ``dedup_seconds`` of a primary onset is the
    same seizure and is dropped.  Returns ``(onsets, offsets, n_from_supplement)``
    sorted chronologically.  Rows with a non-finite onset are discarded from both
    sources — an onset is the only thing a guard cannot be built without.
    """
    def _clean(pair):
        on = np.asarray(pair[0], dtype=float)
        off = np.asarray(pair[1], dtype=float)
        if on.size == 0:
            return np.zeros(0), np.zeros(0)
        keep = np.isfinite(on)
        return on[keep], off[keep]

    p_on, p_off = _clean(primary)
    s_on, s_off = _clean(supplement)

    add_on, add_off = [], []
    for o, f in zip(s_on, s_off):
        if p_on.size and np.min(np.abs(p_on - o)) <= float(dedup_seconds):
            continue
        if add_on and np.min(np.abs(np.asarray(add_on) - o)) <= float(dedup_seconds):
            continue
        add_on.append(float(o))
        add_off.append(float(f))

    on = np.concatenate([p_on, np.asarray(add_on, dtype=float)])
    off = np.concatenate([p_off, np.asarray(add_off, dtype=float)])
    order = np.argsort(on, kind="stable")
    return on[order], off[order], len(add_on)


def minute_guard_free(
    minute_starts: Sequence[float],
    guard_intervals: np.ndarray,
    minute_seconds: float = SECONDS_PER_MINUTE,
) -> np.ndarray:
    """False when the minute window overlaps ANY guard interval."""
    m = np.asarray(minute_starts, dtype=float)
    free = np.ones(m.shape, dtype=bool)
    if guard_intervals is None or len(guard_intervals) == 0:
        return free
    g = np.asarray(guard_intervals, dtype=float).reshape(-1, 2)
    for lo, hi in g:
        free &= ~((m < hi) & (m + minute_seconds > lo))
    return free


def block_split_labels(
    block_starts: Sequence[float], train_end_epoch: float, dev_end_epoch: float
) -> np.ndarray:
    """``train`` / ``validation`` / ``sealed`` by block *start* time."""
    s = np.asarray(block_starts, dtype=float)
    out = np.full(s.shape, "validation", dtype=object)
    out[s < float(train_end_epoch)] = "train"
    out[s >= float(dev_end_epoch)] = "sealed"
    return out


def minute_split_labels(minute_starts: Sequence[float], train_end_epoch: float) -> np.ndarray:
    """``train`` / ``validation`` — the grid never reaches the sealed bound."""
    s = np.asarray(minute_starts, dtype=float)
    out = np.full(s.shape, "validation", dtype=object)
    out[s < float(train_end_epoch)] = "train"
    return out


# ---------------------------------------------------------------------------
# A3 primitives — context / horizon eligibility
# ---------------------------------------------------------------------------


def _true_prefix(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(np.asarray(x, dtype=np.int64))])


def _change_prefix(codes: np.ndarray) -> np.ndarray:
    """``out[k]`` = number of value changes among indices ``0..k-1``."""
    codes = np.asarray(codes)
    if codes.size == 0:
        return np.zeros(1, dtype=np.int64)
    diff = np.concatenate([[0], (codes[1:] != codes[:-1]).astype(np.int64)])
    return np.cumsum(diff)


def compute_eligibility_flags(
    session_id: Sequence[int],
    split: Sequence[str],
    minute_usable: Sequence[bool],
    guard_free: Sequence[bool],
    context_minutes: int = contract.CONTEXT_MINUTES,
    horizons: Sequence[int] = contract.HORIZONS_MIN,
) -> Dict[str, np.ndarray]:
    """``ctx_ok`` plus one ``h{H}_ok`` array per horizon.

    ``ctx_ok[t]``  : minutes ``[t-C+1, t]`` are all usable, one session (>= 0),
                     one split.
    ``h{H}_ok[t]`` : ``ctx_ok[t]`` AND ``minute_usable[t+H]`` AND the whole
                     closed range ``[t-C+1, t+H]`` is guard-free, single-session
                     (>= 0) and single-split AND ``t+H`` is inside the grid.
                     Intervening minutes may be uncovered (spec §4.4).
    """
    sid = np.asarray(session_id, dtype=np.int64)
    usable = np.asarray(minute_usable, dtype=bool)
    gfree = np.asarray(guard_free, dtype=bool)
    split_codes = np.array([_SPLIT_CODE.get(str(s), 3) for s in np.asarray(split)], dtype=np.int64)
    n = sid.size
    C = int(context_minutes)

    usable_pref = _true_prefix(usable)
    guard_pref = _true_prefix(gfree)
    sess_change = _change_prefix(sid)
    split_change = _change_prefix(split_codes)

    t = np.arange(n, dtype=np.int64)
    lo = t - C + 1
    has_ctx = lo >= 0

    def _all_true(pref: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (pref[b + 1] - pref[a]) == (b - a + 1)

    def _constant(change: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (change[b] - change[a]) == 0

    ctx_ok = np.zeros(n, dtype=bool)
    if has_ctx.any():
        tt, ll = t[has_ctx], lo[has_ctx]
        ctx_ok[has_ctx] = (
            _all_true(usable_pref, ll, tt)
            & _constant(sess_change, ll, tt)
            & _constant(split_change, ll, tt)
            & (sid[tt] >= 0)
        )

    out: Dict[str, np.ndarray] = {"ctx_ok": ctx_ok}
    for h in horizons:
        h = int(h)
        hi = t + h
        ok = has_ctx & (hi < n)
        flag = np.zeros(n, dtype=bool)
        if ok.any():
            tt, ll, hh = t[ok], lo[ok], hi[ok]
            flag[ok] = (
                ctx_ok[tt]
                & usable[hh]
                & _all_true(guard_pref, ll, hh)
                & _constant(sess_change, ll, hh)
                & _constant(split_change, ll, hh)
                & (sid[hh] >= 0)
            )
        out[f"h{h}_ok"] = flag
    return out


def refine_minute_index_with_artifacts(
    window_index: pd.DataFrame,
    artifact_mask: np.ndarray,
    min_valid_fraction: float = contract.MINUTE_MIN_VALID_CONTACT_FRACTION,
    context_minutes: int = contract.CONTEXT_MINUTES,
    horizons: Sequence[int] = contract.HORIZONS_MIN,
) -> pd.DataFrame:
    """Sanctioned Worker-B update hook for the artifact columns.

    Parameters
    ----------
    window_index
        One subject's frame as produced by :func:`build_minute_index`, in
        ``minute_index`` order.
    artifact_mask
        ``(n_minutes, n_contacts)`` boolean, **True = contact-minute survives
        the artifact rule**.  Columns must be the ``contact_valid`` subset of
        ``contact_metadata`` in ``channel_index`` order; ``n_contacts`` is the
        denominator of the valid-contact fraction.

    Returns a **new** frame; the input is never mutated.
    """
    df = window_index.copy()
    mask = np.asarray(artifact_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[0] != len(df):
        raise ValueError(
            f"artifact_mask must be (n_minutes={len(df)}, n_contacts); got {mask.shape}"
        )
    n_contacts = int(mask.shape[1])
    if n_contacts == 0:
        raise ValueError("artifact_mask has zero contacts")

    n_valid = mask.sum(axis=1).astype(np.int64)
    frac_ok = (n_valid / float(n_contacts)) >= float(min_valid_fraction)

    covered = df["covered"].to_numpy(dtype=bool)
    guard_free = df["guard_free"].to_numpy(dtype=bool)
    sid = df["session_id"].to_numpy(dtype=np.int64)
    usable = covered & guard_free & (sid >= 0) & frac_ok

    df["n_valid_contacts"] = n_valid
    df["minute_usable"] = usable
    flags = compute_eligibility_flags(
        session_id=sid,
        split=df["split"].to_numpy(),
        minute_usable=usable,
        guard_free=guard_free,
        context_minutes=context_minutes,
        horizons=horizons,
    )
    for key, value in flags.items():
        df[key] = value
    return df[list(contract.WINDOW_INDEX_COLUMNS)]


# ---------------------------------------------------------------------------
# A2 primitives — bipolar montage
# ---------------------------------------------------------------------------


def dense_shaft_index(shafts: Sequence[str], valid: Sequence[bool]) -> np.ndarray:
    """0-based, gap-free position along each shaft among the ``valid`` rows.

    Rows must already be in ``(shaft, anode ordinal)`` order.  Invalid rows get
    ``-1``: they are not model channels, so they must not consume a categorical
    embedding slot.
    """
    shafts = [str(x) for x in shafts]
    valid = np.asarray(valid, dtype=bool)
    out = np.full(len(shafts), -1, dtype=np.int64)
    counters: Dict[str, int] = {}
    for i, sh in enumerate(shafts):
        if not valid[i]:
            continue
        out[i] = counters.get(sh, 0)
        counters[sh] = out[i] + 1
    return out


def bipolar_pairs_from_labels(
    labels: Sequence[str], keep_indices: Optional[Sequence[int]] = None
) -> List[Dict[str, object]]:
    """Adjacent-contact, same-shaft bipolar pairs, sorted by ``(shaft, shaft_index)``.

    ``labels`` is the **full native** channel list; ``keep_indices`` selects the
    intracranial subset.  Returned ``native_index_*`` are indices into
    ``labels``.  Pairing is adjacent-index only: a missing ``A2`` never yields
    ``A1-A3`` (delegated to :func:`src.preprocessing._build_bipolar_pairs`).
    """
    labels = list(labels)
    keep = list(range(len(labels))) if keep_indices is None else [int(i) for i in keep_indices]
    sub = [labels[i] for i in keep]
    pairs, pair_labels = _build_bipolar_pairs(sub)
    out: List[Dict[str, object]] = []
    for (ia, ib), name in zip(pairs, pair_labels):
        anode, cathode = name.split("-", 1)
        shaft, anode_ordinal = ElectrodeParser.parse(anode)
        out.append(
            {
                "channel_name": name,
                "anode": anode,
                "cathode": cathode,
                "shaft": shaft,
                "anode_ordinal": int(anode_ordinal),
                "native_index_anode": keep[ia],
                "native_index_cathode": keep[ib],
            }
        )
    out.sort(key=lambda d: (str(d["shaft"]), int(d["anode_ordinal"])))
    dense = dense_shaft_index(
        [str(d["shaft"]) for d in out], np.ones(len(out), dtype=bool)
    )
    for d, k in zip(out, dense):
        d["shaft_index"] = int(k)
    return out


# ---------------------------------------------------------------------------
# per-subject block layer (A1)
# ---------------------------------------------------------------------------


class TimebaseUnavailable(RuntimeError):
    """Raised when a subject's block intervals cannot be reconstructed."""


def _yuquan_edf_header(path: Path) -> Dict[str, object]:
    """Header-only description of one Yuquan EDF (no MNE, no signal read).

    ``native_labels`` is the FULL signal list (length ``n_signals``) so that
    ``seeg_idx`` entries index directly into it — that is the ``native_index``
    reference for the Yuquan half of the cohort.
    """
    info = read_edf_record_info(path)
    parsed = _parse_edf_header_for_streaming(Path(path))
    with open(path, "rb") as fh:
        fixed = fh.read(256)
        n_signals = int(float(fixed[252:256].decode("ascii", errors="ignore").strip()))
        label_bytes = fh.read(16 * n_signals)
    native_labels = [
        label_bytes[i * 16: (i + 1) * 16].decode("latin1", errors="ignore").strip()
        for i in range(n_signals)
    ]
    if not np.isfinite(info["start_epoch"]) or info["duration_sec"] <= 0:
        raise TimebaseUnavailable(f"EDF header gives no usable timebase: {path}")
    return {
        "block_id": Path(path).stem,
        "start_epoch": float(info["start_epoch"]),
        "end_epoch": float(info["end_epoch"]),
        "duration_sec": float(info["duration_sec"]),
        "sfreq": float(parsed["sfreq"]),
        "n_signals": n_signals,
        "native_labels": native_labels,
        "seeg_labels": list(parsed["seeg_labels"]),
        "seeg_idx": [int(i) for i in parsed["seeg_idx"]],
        "path": str(path),
    }


def _yuquan_edf_paths(native: str) -> List[Path]:
    root = contract.YUQUAN_EDF_ROOT / native
    return [Path(p) for p in sorted(glob.glob(str(root / "*.edf")))]


def build_subject_blocks(subject: str) -> pd.DataFrame:
    """Recorded block intervals for one cohort subject (``DATASET_MANIFEST_COLUMNS``)."""
    dataset, native = split_subject_key(subject)
    sp = contract.load_subject_splits()[subject]

    if dataset == "epilepsiae":
        inv = _epilepsiae_blocks()
        rows = inv[inv["subject"].astype(int) == int(native)].copy()
        rows = rows[rows["data_exists"].astype(bool) & rows["head_exists"].astype(bool)]
        if rows.empty:
            raise TimebaseUnavailable(f"{subject}: no usable rows in the block inventory")
        rows = rows.sort_values("block_start_epoch", kind="stable").reset_index(drop=True)
        starts = rows["block_start_epoch"].to_numpy(dtype=float)
        ends = rows["block_end_epoch"].to_numpy(dtype=float)
        frame = pd.DataFrame(
            {
                "block_id": rows["block_id"].astype(str).to_numpy(),
                "block_start_epoch": starts,
                "block_end_epoch": ends,
                "duration_sec": ends - starts,
                "native_sampling_rate": rows["sample_rate_sql"].to_numpy(dtype=float),
                "n_channels_native": rows["n_channels_sql"].to_numpy(dtype=float),
                "source_path": rows["data_path"].astype(str).to_numpy(),
                "source_kind": "sql_block_inventory",
            }
        )
    else:
        inv = _yuquan_blocks()
        rows = inv[inv["subject"].astype(str) == native].copy()
        if not rows.empty:
            rows = rows.sort_values("block_start_epoch", kind="stable").reset_index(drop=True)
            starts = rows["block_start_epoch"].to_numpy(dtype=float)
            ends = rows["block_end_epoch"].to_numpy(dtype=float)
            frame = pd.DataFrame(
                {
                    "block_id": rows["block_id"].astype(str).to_numpy(),
                    "block_start_epoch": starts,
                    "block_end_epoch": ends,
                    "duration_sec": rows["duration_sec"].to_numpy(dtype=float),
                    "native_sampling_rate": rows["sample_rate"].to_numpy(dtype=float),
                    "n_channels_native": rows["n_channels_total"].to_numpy(dtype=float),
                    "source_path": rows["edf_path"].astype(str).to_numpy(),
                    "source_kind": "yuquan_block_inventory",
                }
            )
        else:
            paths = _yuquan_edf_paths(native)
            if not paths:
                raise TimebaseUnavailable(f"{subject}: no EDF files under {contract.YUQUAN_EDF_ROOT / native}")
            heads = sorted((_yuquan_edf_header(p) for p in paths), key=lambda h: h["start_epoch"])
            frame = pd.DataFrame(
                {
                    "block_id": [h["block_id"] for h in heads],
                    "block_start_epoch": [h["start_epoch"] for h in heads],
                    "block_end_epoch": [h["end_epoch"] for h in heads],
                    "duration_sec": [h["duration_sec"] for h in heads],
                    "native_sampling_rate": [h["sfreq"] for h in heads],
                    "n_channels_native": [float(h["n_signals"]) for h in heads],
                    "source_path": [h["path"] for h in heads],
                    "source_kind": "edf_header",
                }
            )
            starts = frame["block_start_epoch"].to_numpy(dtype=float)
            ends = frame["block_end_epoch"].to_numpy(dtype=float)

    if not np.isfinite(starts).all() or not np.isfinite(ends).all():
        raise TimebaseUnavailable(f"{subject}: non-finite block epochs in the timebase source")
    if (ends < starts).any():
        raise TimebaseUnavailable(f"{subject}: block_end_epoch < block_start_epoch")

    gap, opens, session_id = assign_sessions(starts, ends)
    frame["subject"] = subject
    frame["dataset"] = dataset
    frame["session_id"] = session_id
    frame["gap_to_prev_sec"] = gap
    frame["opens_session"] = opens
    frame["split"] = block_split_labels(starts, sp.train_end_epoch, sp.dev_end_epoch)
    frame["n_channels_native"] = frame["n_channels_native"].astype(float)
    return frame[list(contract.DATASET_MANIFEST_COLUMNS)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# per-subject contact layer (A2)
# ---------------------------------------------------------------------------


def _epilepsiae_block_layout(head_path: Path) -> Tuple[List[str], List[int], float]:
    head = _read_epilepsiae_head_for_streaming(Path(head_path))
    names = [str(n) for n in head["channel_names"]]
    keep = [i for i, n in enumerate(names) if n.strip().upper() not in SCALP_OR_AUX_CHANNELS]
    return names, keep, float(head["sample_freq"])


def _addressable_pairs(names: Sequence[str], keep: Sequence[int]) -> Tuple[Tuple[int, str], ...]:
    """``(native_index, cleaned_channel_name)`` for the kept channels of one block.

    The cleaned name is what ``bipolar_pairs_from_labels`` puts in ``anode`` /
    ``cathode`` (``"EEG A1-Ref"`` -> ``"A1"``), so the two sides compare.
    """
    return tuple((int(k), ElectrodeParser.clean_name(str(names[k]))) for k in keep)


def _subject_dev_layouts(subject: str, blocks: pd.DataFrame) -> Dict[str, object]:
    """Native channel layout of every dev (non-sealed) block of a subject.

    Returns the canonical (first-block) layout plus the set of ``(native_index,
    name)`` pairs present in **every** dev block, so callers can invalidate
    channels that are not consistently addressable.
    """
    dataset, _ = split_subject_key(subject)
    dev = blocks[blocks["split"] != "sealed"]
    if dev.empty:
        raise TimebaseUnavailable(f"{subject}: no dev-window blocks")

    canonical_names: List[str] = []
    canonical_keep: List[int] = []
    per_block: List[Tuple[str, Tuple[Tuple[int, str], ...]]] = []
    rates: List[float] = []

    if dataset == "epilepsiae":
        inv = _epilepsiae_blocks()
        head_by_block = dict(
            zip(inv["block_id"].astype(str), inv["head_path"].astype(str))
        )
        for i, block_id in enumerate(dev["block_id"].astype(str)):
            names, keep, sf = _epilepsiae_block_layout(Path(head_by_block[block_id]))
            per_block.append((block_id, _addressable_pairs(names, keep)))
            rates.append(sf)
            if i == 0:
                canonical_names, canonical_keep = names, keep
    else:
        for i, path in enumerate(dev["source_path"].astype(str)):
            head = _yuquan_edf_header(Path(path))
            names = list(head["native_labels"])
            keep = list(head["seeg_idx"])
            per_block.append((str(head["block_id"]), _addressable_pairs(names, keep)))
            rates.append(float(head["sfreq"]))
            if i == 0:
                canonical_names, canonical_keep = names, keep

    common = set(per_block[0][1])
    for _, pairs in per_block[1:]:
        common &= set(pairs)
    consistent = len(set(p[1] for p in per_block)) == 1

    return {
        "canonical_names": canonical_names,
        "canonical_keep": canonical_keep,
        "common_pairs": common,
        "consistent": bool(consistent),
        "n_dev_blocks": len(per_block),
        "native_rates": sorted({float(r) for r in rates}),
        "n_layouts": len(set(p[1] for p in per_block)),
    }


def assemble_contact_rows(
    subject: str,
    dataset: str,
    pairs: Sequence[Dict[str, object]],
    coords: np.ndarray,
    coord_mapped: Sequence[bool],
    coord_space: str,
    addressable: Sequence[bool],
) -> pd.DataFrame:
    """Build ``CONTACT_METADATA_COLUMNS`` with contact/coord validity decoupled.

    ``contact_valid`` = ``addressable`` only (electrical well-formedness).
    ``coord_valid``   = ``coord_mapped`` only (anatomy).  A missing coordinate is
    recorded in ``drop_reason`` but never removes the channel.
    """
    coords = np.asarray(coords, dtype=float).reshape(len(pairs), 3)
    coord_mapped = np.asarray(coord_mapped, dtype=bool)
    addressable = np.asarray(addressable, dtype=bool)
    coord_mode = (
        contract.COORD_MODE_FULL
        if bool((coord_mapped & addressable).any())
        else contract.COORD_MODE_TOPOLOGY_ONLY
    )
    shaft_index = dense_shaft_index([str(p["shaft"]) for p in pairs], addressable)

    rows = []
    for i, p in enumerate(pairs):
        reasons = []
        if not addressable[i]:
            reasons.append("inconsistent_native_index")
        if not coord_mapped[i]:
            reasons.append("missing_coordinate")
        rows.append(
            {
                "subject": subject,
                "dataset": dataset,
                "channel_index": i,
                "channel_name": p["channel_name"],
                "anode": p["anode"],
                "cathode": p["cathode"],
                "shaft": p["shaft"],
                "shaft_index": int(shaft_index[i]),
                "x_mm": float(coords[i, 0]),
                "y_mm": float(coords[i, 1]),
                "z_mm": float(coords[i, 2]),
                "coord_space": coord_space,
                "coord_valid": bool(coord_mapped[i]),
                "native_index_anode": int(p["native_index_anode"]),
                "native_index_cathode": int(p["native_index_cathode"]),
                "contact_valid": bool(addressable[i]),
                "drop_reason": "|".join(reasons),
                "coord_mode": coord_mode,
            }
        )
    return pd.DataFrame(rows, columns=list(contract.CONTACT_METADATA_COLUMNS))


def build_contact_metadata(subject: str, blocks: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Bipolar montage + coordinates (``CONTACT_METADATA_COLUMNS``)."""
    dataset, native = split_subject_key(subject)
    blocks = build_subject_blocks(subject) if blocks is None else blocks
    layout = _subject_dev_layouts(subject, blocks)

    pairs = bipolar_pairs_from_labels(layout["canonical_names"], layout["canonical_keep"])
    if not pairs:
        raise TimebaseUnavailable(f"{subject}: no bipolar pairs could be formed")

    names = [str(p["channel_name"]) for p in pairs]
    coord_space = "unavailable"
    coords = np.full((len(pairs), 3), np.nan, dtype=float)
    mapped = np.zeros(len(pairs), dtype=bool)
    coord_note = ""
    try:
        res = load_subject_coords(dataset, native, names, allow_voxel_fallback=False)
    except FileNotFoundError as exc:
        if dataset == "epilepsiae":
            try:
                res = load_subject_coords(dataset, native, names, allow_voxel_fallback=True)
                coord_note = f"voxel_fallback:{exc}"
            except Exception as exc2:  # pragma: no cover - defensive
                res = None
                coord_note = f"coords_unavailable:{exc2}"
        else:
            res = None
            coord_note = f"coords_unavailable:{exc}"
    except Exception as exc:  # pragma: no cover - defensive
        res = None
        coord_note = f"coords_unavailable:{exc}"

    if res is not None:
        coords = np.asarray(res.coords_array_in_requested_order, dtype=float)
        mapped = np.asarray(res.mapped_mask_in_requested_order, dtype=bool)
        coord_space = f"{res.coord_space}[{res.coord_units}]"

    common = layout["common_pairs"]
    addressable = np.array(
        [
            ((int(p["native_index_anode"]), str(p["anode"])) in common)
            and ((int(p["native_index_cathode"]), str(p["cathode"])) in common)
            for p in pairs
        ],
        dtype=bool,
    )
    df = assemble_contact_rows(
        subject, dataset, pairs, coords, mapped, coord_space, addressable
    )
    df.attrs["coord_note"] = coord_note
    df.attrs["layout"] = {k: v for k, v in layout.items() if k not in ("common_pairs",)}
    return df


# ---------------------------------------------------------------------------
# per-subject guard layer
# ---------------------------------------------------------------------------


def load_yuquan_annotation_scan(native: str) -> Tuple[np.ndarray, np.ndarray, Optional[Path]]:
    """Onsets/offsets from ``pr1_seizure_<subject>.json`` (raw EDF-annotation scan).

    This is the supplementary guard source of coordinator ruling 2.  The frozen
    inventory drops marks that fail ``has_complete_eeg_interval``, which silently
    loses real onsets; this scan keeps them.
    """
    path = contract.YUQUAN_SUPPLEMENTARY_SEIZURE_DIR / f"pr1_seizure_{native}.json"
    if not path.exists():
        return np.zeros(0), np.zeros(0), None
    payload = json.loads(path.read_text())
    onsets, offsets = [], []
    for entry in payload.get("files", []):
        for iv in entry.get("seizure_intervals", []) or []:
            on = iv.get("onset_epoch")
            if on is None:
                continue
            onsets.append(float(on))
            off = iv.get("offset_epoch")
            offsets.append(float(off) if off is not None else float("nan"))
    return (
        np.asarray(onsets, dtype=float),
        np.asarray(offsets, dtype=float),
        path,
    )


def load_seizure_table(subject: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """``(onsets, offsets, meta)`` — the guard source for one subject.

    Epilepsiae uses the frozen SQL inventory alone.  Yuquan uses the UNION of the
    frozen inventory and the raw EDF-annotation scan (coordinator ruling 2),
    de-duplicated by onset within ``SEIZURE_DEDUP_SECONDS``.
    """
    dataset, native = split_subject_key(subject)
    supp_path: Optional[Path] = None
    if dataset == "epilepsiae":
        table = _epilepsiae_seizures()
        rows = table[table["subject"].astype(int) == int(native)]
        source = str(contract.EPILEPSIAE_SEIZURE_INVENTORY)
        inv_on = rows["eeg_onset_epoch"].to_numpy(dtype=float)
        inv_off = rows["eeg_offset_epoch"].to_numpy(dtype=float)
        sup_on = np.zeros(0)
        sup_off = np.zeros(0)
    else:
        table = _yuquan_seizures()
        rows = table[table["subject"].astype(str) == native]
        source = str(contract.YUQUAN_SEIZURE_INVENTORY)
        inv_on = rows["eeg_onset_epoch"].to_numpy(dtype=float)
        inv_off = rows["eeg_offset_epoch"].to_numpy(dtype=float)
        sup_on, sup_off, supp_path = load_yuquan_annotation_scan(native)

    onsets, offsets, n_from_supplement = merge_seizure_sources(
        (inv_on, inv_off), (sup_on, sup_off)
    )
    n_inventory = int(np.sum(np.isfinite(inv_on)))
    if n_inventory and n_from_supplement:
        guard_source = "inventory+annotation_scan"
    elif n_inventory:
        guard_source = "inventory"
    elif n_from_supplement:
        guard_source = "annotation_scan_only"
    else:
        guard_source = "none_found"

    meta = {
        "seizure_inventory": source,
        "supplementary_scan": str(supp_path) if supp_path is not None else None,
        "subject_present_in_seizure_inventory": bool(n_inventory > 0),
        "n_seizures_inventory": n_inventory,
        "n_seizures_from_supplement": int(n_from_supplement),
        "n_seizures_total": int(onsets.size),
        "n_onset_missing": int(np.sum(~np.isfinite(inv_on)) + np.sum(~np.isfinite(sup_on))),
        "n_offset_fallback": count_offset_fallbacks(onsets, offsets),
        "seizure_guard_source": guard_source,
        "no_annotation_is_not_no_seizure": (
            "seizure_guard_source == 'none_found' means no seizure was annotated "
            "anywhere in this subject's EDF/SQL metadata. It is NOT evidence that "
            "no seizure occurred; ictal exclusion cannot be guaranteed."
        ),
    }
    return onsets, offsets, meta


# ---------------------------------------------------------------------------
# per-subject minute layer (A3)
# ---------------------------------------------------------------------------


def build_minute_index(subject: str, blocks: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Minute eligibility grid (``WINDOW_INDEX_COLUMNS``)."""
    blocks = build_subject_blocks(subject) if blocks is None else blocks
    sp = contract.load_subject_splits()[subject]
    dev_end = sp.dev_end_epoch

    starts = blocks["block_start_epoch"].to_numpy(dtype=float)
    ends = blocks["block_end_epoch"].to_numpy(dtype=float)
    session_id_blocks = blocks["session_id"].to_numpy(dtype=np.int64)

    minute_starts = minute_grid_starts(float(starts.min()), dev_end)
    if minute_starts.size == 0:
        raise TimebaseUnavailable(f"{subject}: empty minute grid (first block at/after the seal)")

    # coverage is measured against blocks clipped at the seal, and the minute
    # that would straddle the seal is never covered.
    clipped_ends = np.minimum(ends, dev_end)
    keep = clipped_ends > starts
    covered_seconds = minute_covered_seconds(minute_starts, starts[keep], clipped_ends[keep])
    # The decision uses the longest UNBROKEN stretch, not the sum -- see
    # minute_max_contiguous_seconds for why a spliced minute is worse than a
    # dropped one.
    contiguous_seconds = minute_max_contiguous_seconds(
        minute_starts, starts[keep], clipped_ends[keep])
    covered = covered_from_seconds(contiguous_seconds) & (
        minute_starts + SECONDS_PER_MINUTE <= dev_end
    )

    extents = session_extents(starts, ends, session_id_blocks)
    session_id = minute_session_ids(minute_starts, extents)

    onsets, offsets, _ = load_seizure_table(subject)
    guards = guard_intervals_from_seizures(onsets, offsets)
    guard_free = minute_guard_free(minute_starts, guards)

    split = minute_split_labels(minute_starts, sp.train_end_epoch)
    minute_usable = covered & guard_free & (session_id >= 0)

    df = pd.DataFrame(
        {
            "subject": subject,
            "minute_index": np.arange(minute_starts.size, dtype=np.int64),
            "minute_start_epoch": minute_starts,
            "session_id": session_id,
            "split": split,
            "covered": covered,
            "covered_seconds": covered_seconds,
            "contiguous_seconds": contiguous_seconds,
            "guard_free": guard_free,
            "n_valid_contacts": np.full(minute_starts.size, -1, dtype=np.int64),
            "minute_usable": minute_usable,
        }
    )
    flags = compute_eligibility_flags(
        session_id=session_id,
        split=split,
        minute_usable=minute_usable,
        guard_free=guard_free,
    )
    for key, value in flags.items():
        df[key] = value
    df = df[list(contract.WINDOW_INDEX_COLUMNS)]
    contract.assert_not_sealed(subject, df["minute_start_epoch"].to_numpy())
    return df


# ---------------------------------------------------------------------------
# subject assembly (A1 + A2 + A3 -> eligibility row + audit entry)
# ---------------------------------------------------------------------------


@dataclass
class SubjectBuild:
    subject: str
    dataset: str
    status: str
    blocks: pd.DataFrame = field(default_factory=pd.DataFrame)
    contacts: pd.DataFrame = field(default_factory=pd.DataFrame)
    minutes: pd.DataFrame = field(default_factory=pd.DataFrame)
    eligibility: Dict[str, object] = field(default_factory=dict)
    audit: Dict[str, object] = field(default_factory=dict)


def _empty_eligibility(subject: str, dataset: str, status: str) -> Dict[str, object]:
    row = {c: "" for c in contract.ELIGIBILITY_COLUMNS}
    row["subject"] = subject
    row["dataset"] = dataset
    row["status"] = status
    return row


def build_subject(subject: str) -> SubjectBuild:
    """Full A1-A3 build for one subject; never raises for data problems."""
    dataset, _ = split_subject_key(subject)
    sp = contract.load_subject_splits()[subject]
    audit: Dict[str, object] = {
        "subject": subject,
        "dataset": dataset,
        "split_bounds": {
            "first_epoch": sp.first_epoch,
            "train_end_epoch": sp.train_end_epoch,
            "dev_end_epoch": sp.dev_end_epoch,
            "sealed_first_epoch": sp.sealed_first_epoch,
        },
        "coverage_rule": COVERAGE_RULE,
        "native_index_reference": NATIVE_INDEX_REFERENCE[dataset],
        "flags": [],
    }

    try:
        blocks = build_subject_blocks(subject)
    except Exception as exc:
        audit["error"] = f"{type(exc).__name__}: {exc}"
        status = "BLOCKED_NO_TIMEBASE"
        audit["flags"].append(status)
        return SubjectBuild(subject, dataset, status, eligibility=_empty_eligibility(subject, dataset, status), audit=audit)

    try:
        minutes = build_minute_index(subject, blocks)
    except Exception as exc:
        audit["error"] = f"{type(exc).__name__}: {exc}"
        status = "BLOCKED_NO_TIMEBASE"
        audit["flags"].append(status)
        return SubjectBuild(subject, dataset, status, blocks=blocks,
                            eligibility=_empty_eligibility(subject, dataset, status), audit=audit)

    contacts = pd.DataFrame()
    contact_error = None
    try:
        contacts = build_contact_metadata(subject, blocks)
    except Exception as exc:
        contact_error = f"{type(exc).__name__}: {exc}"
        audit["contact_error"] = contact_error

    # ---- block-level audit ------------------------------------------------
    dev_blocks = blocks[blocks["split"] != "sealed"]
    b_start = blocks["block_start_epoch"].to_numpy(dtype=float)
    b_end = blocks["block_end_epoch"].to_numpy(dtype=float)
    merged_s, merged_e = _merge_intervals(b_start, b_end)
    recorded_hours_total = float(np.sum(merged_e - merged_s) / 3600.0)
    gaps = blocks["gap_to_prev_sec"].to_numpy(dtype=float)
    largest_gap = float(np.nanmax(gaps)) if np.isfinite(gaps).any() else 0.0

    dev_rates = sorted({float(r) for r in dev_blocks["native_sampling_rate"].dropna()})
    all_rates = sorted({float(r) for r in blocks["native_sampling_rate"].dropna()})
    nyquist_limited = bool(any(r < 512.0 for r in dev_rates))

    # ---- seizure / guard audit -------------------------------------------
    onsets, offsets, sz_meta = load_seizure_table(subject)
    n_sz_dev = int(np.sum(np.isfinite(onsets) & (onsets < sp.dev_end_epoch)))
    guard_minutes = int(np.sum(minutes["covered"].to_numpy() & ~minutes["guard_free"].to_numpy()))

    # ---- minute / eligibility --------------------------------------------
    covered = minutes["covered"].to_numpy(dtype=bool)
    usable = minutes["minute_usable"].to_numpy(dtype=bool)
    is_train = minutes["split"].to_numpy() == "train"
    is_val = minutes["split"].to_numpy() == "validation"
    dev_covered_hours = float(covered.sum() / 60.0)
    train_hours = float((covered & is_train).sum() / 60.0)
    val_hours = float((covered & is_val).sum() / 60.0)
    train_usable_hours = float((usable & is_train).sum() / 60.0)
    val_usable_hours = float((usable & is_val).sum() / 60.0)
    # A cap of None means "cache every covered dev minute" (contract lifted the
    # caps on 2026-08-21 once the measured zstd ratio came in at 4.68x).
    cached_train_hours = float(train_usable_hours if contract.CACHE_TRAIN_HOURS_CAP is None
                               else min(train_usable_hours, contract.CACHE_TRAIN_HOURS_CAP))
    cached_val_hours = float(val_usable_hours if contract.CACHE_VAL_HOURS_CAP is None
                             else min(val_usable_hours, contract.CACHE_VAL_HOURS_CAP))

    horizon_counts: Dict[str, int] = {}
    for h in contract.HORIZONS_MIN:
        col = minutes[f"h{h}_ok"].to_numpy(dtype=bool)
        horizon_counts[f"n_train_h{h}"] = int((col & is_train).sum())
        horizon_counts[f"n_val_h{h}"] = int((col & is_val).sum())

    # counts restricted to the hours the cache budget will actually hold
    capped = _cache_capped_counts(minutes, cached_train_hours, cached_val_hours)

    n_sessions = int(minutes.loc[minutes["session_id"] >= 0, "session_id"].nunique())

    # ---- contacts ---------------------------------------------------------
    if len(contacts):
        n_bipolar = int(len(contacts))
        n_valid = int(contacts["contact_valid"].sum())
        n_coord = int((contacts["coord_valid"] & contacts["contact_valid"]).sum())
        coord_space = str(contacts["coord_space"].iloc[0])
        coord_mode = str(contacts["coord_mode"].iloc[0])
        layout = dict(contacts.attrs.get("layout", {}))
        audit["coord_note"] = contacts.attrs.get("coord_note", "")
        audit["native_channel_names"] = list(layout.pop("canonical_names", []))
        audit["native_channel_indices_kept"] = layout.pop("canonical_keep", [])
        audit["channel_order_consistent_across_dev_blocks"] = bool(layout.get("consistent", False))
        audit["n_native_layouts_in_dev"] = int(layout.get("n_layouts", 0))
        audit["n_channels_native"] = int(len(audit["native_channel_names"]))
    else:
        n_bipolar = n_valid = n_coord = 0
        coord_space = "unavailable"
        coord_mode = contract.COORD_MODE_TOPOLOGY_ONLY
        audit["channel_order_consistent_across_dev_blocks"] = False
        audit["n_native_layouts_in_dev"] = 0
        audit["n_channels_native"] = 0

    # ---- status -----------------------------------------------------------
    guard_source = str(sz_meta["seizure_guard_source"])
    status = "OK"
    if contact_error is not None:
        status = "BLOCKED_NO_CONTACTS"
    elif n_valid == 0:
        status = "BLOCKED_NO_VALID_CONTACTS"
    elif n_valid < 20:
        status = "DEGRADED_FEW_CONTACTS"
    if guard_source == "none_found" and status == "OK":
        status = "DEGRADED_NO_SEIZURE_GUARD"
    if guard_source == "none_found":
        audit["flags"].append("no_seizure_annotation_found")
    if not sz_meta["subject_present_in_seizure_inventory"]:
        audit["flags"].append("no_seizure_inventory_row")
    if sz_meta["n_seizures_from_supplement"]:
        audit["flags"].append("guard_extended_by_annotation_scan")
    if coord_mode == contract.COORD_MODE_TOPOLOGY_ONLY:
        audit["flags"].append("coord_mode_shaft_index_only")
    if not audit.get("channel_order_consistent_across_dev_blocks", True):
        audit["flags"].append("native_channel_order_varies_across_dev_blocks")
    if nyquist_limited:
        audit["flags"].append("nyquist_limited")

    audit.update(
        {
            "native_rates_dev": dev_rates,
            "native_rates_all_blocks": all_rates,
            "nyquist_limited": nyquist_limited,
            "n_blocks": int(len(blocks)),
            "n_dev_blocks": int(len(dev_blocks)),
            "n_sessions": n_sessions,
            "n_sessions_all_blocks": int(blocks["session_id"].nunique()),
            "largest_gap_sec": largest_gap,
            "recorded_hours_total": recorded_hours_total,
            "dev_covered_hours": dev_covered_hours,
            "train_hours": train_hours,
            "val_hours": val_hours,
            "train_usable_hours": train_usable_hours,
            "val_usable_hours": val_usable_hours,
            "guard_hours_removed": guard_minutes / 60.0,
            "n_minutes_in_grid": int(len(minutes)),
            "n_bipolar_channels": n_bipolar,
            "n_bipolar_valid": n_valid,
            "n_bipolar_with_coords": n_coord,
            "coord_space": coord_space,
            "coord_mode": coord_mode,
            "coord_valid_fraction": (round(n_coord / n_valid, 4) if n_valid else 0.0),
            "seizures": {**sz_meta, "n_seizures_in_dev": n_sz_dev},
            "horizon_counts": horizon_counts,
            "horizon_counts_within_cache_cap": capped,
            "cached_hours_are_projected": True,
            "status": status,
        }
    )
    audit["checks"] = _subject_checks(subject, blocks, minutes, contacts, sp.dev_end_epoch)

    row = {
        "subject": subject,
        "dataset": dataset,
        "n_contacts": n_valid,
        "native_rates": "|".join(f"{int(r)}" for r in dev_rates),
        "nyquist_limited": nyquist_limited,
        "recorded_hours_total": round(recorded_hours_total, 4),
        "dev_covered_hours": round(dev_covered_hours, 4),
        "train_hours": round(train_hours, 4),
        "val_hours": round(val_hours, 4),
        "cached_train_hours": round(cached_train_hours, 4),
        "cached_val_hours": round(cached_val_hours, 4),
        "n_sessions": n_sessions,
        "n_seizures_in_dev": n_sz_dev,
        "n_seizures_from_supplement": int(sz_meta["n_seizures_from_supplement"]),
        "seizure_guard_source": guard_source,
        "coord_mode": coord_mode,
        "guard_hours_removed": round(guard_minutes / 60.0, 4),
        **horizon_counts,
        "pilot_tier": "",
        "status": status,
    }
    row = {c: row.get(c, "") for c in contract.ELIGIBILITY_COLUMNS}
    return SubjectBuild(subject, dataset, status, blocks, contacts, minutes, row, audit)


def _cache_capped_counts(
    minutes: pd.DataFrame, cached_train_hours: float, cached_val_hours: float
) -> Dict[str, int]:
    """Horizon counts if only the most recent cached hours of each split exist.

    Purely informational (spec §4.5 is an engineering budget); the parquet keeps
    the full dev grid so Worker B can decide the exact truncation.
    """
    out: Dict[str, int] = {}
    usable = minutes["minute_usable"].to_numpy(dtype=bool)
    split = minutes["split"].to_numpy()
    keep = np.zeros(len(minutes), dtype=bool)
    for name, cap_hours in (("train", cached_train_hours), ("validation", cached_val_hours)):
        sel = np.flatnonzero((split == name) & usable)
        if sel.size == 0:
            continue
        budget = int(round(cap_hours * 60))
        chosen = sel[-budget:] if budget > 0 else sel[:0]
        if chosen.size:
            keep[chosen.min(): chosen.max() + 1] = True
    for h in contract.HORIZONS_MIN:
        col = minutes[f"h{h}_ok"].to_numpy(dtype=bool) & keep
        out[f"n_train_h{h}"] = int((col & (split == "train")).sum())
        out[f"n_val_h{h}"] = int((col & (split == "validation")).sum())
    return out


def _subject_checks(
    subject: str,
    blocks: pd.DataFrame,
    minutes: pd.DataFrame,
    contacts: pd.DataFrame,
    dev_end: float,
) -> Dict[str, str]:
    """PASS/FAIL for the spec §9 conditions that are checkable at stage A."""
    checks: Dict[str, str] = {}

    m = minutes["minute_start_epoch"].to_numpy(dtype=float)
    checks["window_index_below_seal"] = "PASS" if (m.size and m.max() < dev_end) else "FAIL"
    checks["minute_grid_monotone"] = (
        "PASS" if bool(np.all(np.diff(m) > 0)) else "FAIL"
    )

    dev = blocks[blocks["split"] != "sealed"]
    checks["dev_blocks_start_below_seal"] = (
        "PASS" if bool((dev["block_start_epoch"].to_numpy(dtype=float) < dev_end).all()) else "FAIL"
    )
    bs = blocks["block_start_epoch"].to_numpy(dtype=float)
    be = blocks["block_end_epoch"].to_numpy(dtype=float)
    checks["block_times_monotone"] = (
        "PASS" if bool(np.all(np.diff(bs) >= 0)) and bool(np.all(be >= bs)) else "FAIL"
    )
    checks["block_epochs_finite"] = (
        "PASS" if bool(np.isfinite(bs).all() and np.isfinite(be).all()) else "FAIL"
    )
    checks["minute_epochs_finite"] = "PASS" if bool(np.isfinite(m).all()) else "FAIL"

    if len(contacts):
        idx = contacts["channel_index"].to_numpy()
        order_ok = bool(np.array_equal(idx, np.arange(len(contacts))))
        ordinals = [ElectrodeParser.parse(a)[1] for a in contacts["anode"]]
        sort_key = list(zip(contacts["shaft"].tolist(), ordinals))
        checks["channel_index_is_shaft_sorted"] = (
            "PASS" if order_ok and sort_key == sorted(sort_key) else "FAIL"
        )
        checks["shaft_index_dense_per_shaft"] = _check_dense_shaft_index(contacts)
        checks["coord_mode_consistent"] = (
            "PASS" if contacts["coord_mode"].nunique() == 1 else "FAIL"
        )
        checks["channel_names_unique"] = (
            "PASS" if contacts["channel_name"].is_unique else "FAIL"
        )
        checks["channel_order_consistent_across_dev_blocks"] = (
            "PASS" if bool(contacts.attrs.get("layout", {}).get("consistent", False)) else "FAIL"
        )
    else:
        checks["channel_index_is_shaft_sorted"] = "FAIL"
        checks["shaft_index_dense_per_shaft"] = "FAIL"
        checks["coord_mode_consistent"] = "FAIL"
        checks["channel_names_unique"] = "FAIL"
        checks["channel_order_consistent_across_dev_blocks"] = "FAIL"

    # eligible windows must never straddle a guard minute or a session boundary
    checks["horizon_flags_guard_clean"] = _check_horizon_guard_clean(minutes)
    return checks


def _check_dense_shaft_index(contacts: pd.DataFrame) -> str:
    """Valid rows of each shaft must carry shaft_index 0..k-1 with no gaps."""
    valid = contacts[contacts["contact_valid"]]
    for _, grp in valid.groupby("shaft"):
        got = sorted(int(x) for x in grp["shaft_index"])
        if got != list(range(len(grp))):
            return "FAIL"
    if len(contacts) > len(valid):
        if not (contacts.loc[~contacts["contact_valid"], "shaft_index"] == -1).all():
            return "FAIL"
    return "PASS"


def _check_horizon_guard_clean(minutes: pd.DataFrame) -> str:
    guard_free = minutes["guard_free"].to_numpy(dtype=bool)
    sid = minutes["session_id"].to_numpy(dtype=np.int64)
    n = len(minutes)
    C = contract.CONTEXT_MINUTES
    for h in contract.HORIZONS_MIN:
        idx = np.flatnonzero(minutes[f"h{h}_ok"].to_numpy(dtype=bool))
        for t in idx[:: max(1, idx.size // 200 or 1)]:  # sample up to ~200 windows
            lo, hi = t - C + 1, t + h
            if lo < 0 or hi >= n:
                return "FAIL"
            if not guard_free[lo: hi + 1].all():
                return "FAIL"
            if sid[lo] < 0 or not np.all(sid[lo: hi + 1] == sid[lo]):
                return "FAIL"
    return "PASS"


__all__ = [
    "SubjectBuild",
    "TimebaseUnavailable",
    "COVERAGE_MIN_SECONDS",
    "COVERAGE_RULE",
    "MISSING_OFFSET_FALLBACK_SECONDS",
    "NATIVE_INDEX_REFERENCE",
    "assign_sessions",
    "session_extents",
    "minute_grid_starts",
    "minute_covered_seconds",
    "covered_from_seconds",
    "minute_session_ids",
    "guard_intervals_from_seizures",
    "count_offset_fallbacks",
    "minute_guard_free",
    "block_split_labels",
    "minute_split_labels",
    "compute_eligibility_flags",
    "refine_minute_index_with_artifacts",
    "bipolar_pairs_from_labels",
    "dense_shaft_index",
    "assemble_contact_rows",
    "merge_seizure_sources",
    "load_yuquan_annotation_scan",
    "build_subject_blocks",
    "build_contact_metadata",
    "build_minute_index",
    "build_subject",
    "load_seizure_table",
    "split_subject_key",
]
