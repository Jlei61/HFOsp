"""Two scales, not one.

`N_obs` is the smallest window at which the repertoire rises above this patient's own
chance level; it becomes the model step, because a state that persists for 500 events cut
into 500-event blocks leaves one block per state and no dwell to observe.

`N_break` is the smallest larger window at which the chronological halves agree worse than
random halves — where a window starts averaging over a state change.

Scales with too few independent windows are skipped, not counted as failures, and a
leading run of below-chance scales is expected rather than disqualifying.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.topic5_slow_state_repertoire import FAMILIES, family_agreement, local_repertoire

BELOW, RELIABLE, BREAK = "BELOW_CHANCE", "RELIABLE", "CHRONOLOGY_BREAK"
TOO_FEW = "UNRESOLVED_TOO_FEW_WINDOWS"
UNRESOLVED_FAMILIES = "UNRESOLVED_FAMILIES"


def _local_repertoire_kwargs(floors: Mapping[str, int]) -> dict[str, int]:
    """Map config-style floor keys onto `local_repertoire`'s parameter names.

    The frozen config (`config/topic5_slow_state_v4_0.yaml`) spells the pair floor
    `min_pair_coparticipation_count`; `local_repertoire` (Task 4) takes `min_pair_count`.
    `min_participation_count` is spelled the same on both sides. Splatting a raw config
    dict into `local_repertoire` raises `TypeError` (unexpected keyword argument) — this
    accepts either spelling so a caller can pass the raw config subset or an
    already-renamed dict without that trap.
    """
    merged = dict(floors)
    if "min_pair_count" not in merged and "min_pair_coparticipation_count" in merged:
        merged["min_pair_count"] = merged["min_pair_coparticipation_count"]
    return {
        "min_participation_count": int(merged["min_participation_count"]),
        "min_pair_count": int(merged["min_pair_count"]),
    }


def window_agreements(
    rank: np.ndarray,
    participation: np.ndarray,
    group_ids: np.ndarray,
    *,
    random_half_draws: int,
    null_draws: int,
    seed: int,
    floors: Mapping[str, int],
) -> dict[str, Any]:
    """The three agreements of one event window (§6.3).

    - `random_half` — many **random** assignments of this window's own events into two
      halves: the sampling-noise ceiling. One list of successfully-computed agreement
      values per family, length up to `random_half_draws` (a draw that lands on too few
      finite pairs for a family contributes no value to that family's list, per
      `family_agreement`'s own `None` contract — the return type is `list[float]`, not
      `list[float | None]`, so `None` draws are dropped rather than stored).
    - `chronological` — this window's **first half vs second half by event position**
      (not randomly assigned): one value per family, or `None` when that split could not
      be computed.
    - `contact_null` — the same random-half machinery, but with **contact identity (the
      column order of `rank`/`participation`/`group_ids`) permuted on one side before
      `local_repertoire` is computed for it**. This is this patient's own chance level:
      it holds the event-count and split mechanics identical to `random_half` and changes
      only which physical contact each array position refers to, so any leftover
      agreement is attributable to chance column alignment rather than genuine per-contact
      structure. A null that instead permuted event order would change nothing, because
      every descriptor in `local_repertoire` is an aggregate over events per contact
      (order-independent within a half).

    Every draw is taken from a single `numpy.random.default_rng(seed)` instance created
    once at the top of this function, so the whole function is deterministic given its
    inputs — no draw reseeds independently.
    """
    rank = np.asarray(rank, dtype=float)
    participation = np.asarray(participation)
    group_ids = np.asarray(group_ids)
    n_events, n_contacts = rank.shape
    rng = np.random.default_rng(seed)
    kwargs = _local_repertoire_kwargs(floors)

    def _repertoire(event_idx: np.ndarray, contact_perm: np.ndarray | None = None) -> dict[str, Any]:
        if contact_perm is None:
            return local_repertoire(
                rank[event_idx], participation[event_idx], group_ids[event_idx], **kwargs
            )
        cols = np.ix_(event_idx, contact_perm)
        return local_repertoire(rank[cols], participation[cols], group_ids[cols], **kwargs)

    def _random_half_split() -> tuple[np.ndarray, np.ndarray]:
        perm = rng.permutation(n_events)
        half = n_events // 2
        return perm[:half], perm[half:]

    random_half: dict[str, list[float]] = {family: [] for family in FAMILIES}
    for _ in range(int(random_half_draws)):
        left_idx, right_idx = _random_half_split()
        agreement = family_agreement(_repertoire(left_idx), _repertoire(right_idx))
        for family in FAMILIES:
            value = agreement[family]
            if value is not None:
                random_half[family].append(value)

    half = n_events // 2
    chrono_agreement = family_agreement(
        _repertoire(np.arange(0, half)), _repertoire(np.arange(half, n_events))
    )
    chronological: dict[str, float | None] = {
        family: chrono_agreement[family] for family in FAMILIES
    }

    contact_null: dict[str, list[float]] = {family: [] for family in FAMILIES}
    for _ in range(int(null_draws)):
        left_idx, right_idx = _random_half_split()
        contact_perm = rng.permutation(n_contacts)
        agreement = family_agreement(
            _repertoire(left_idx), _repertoire(right_idx, contact_perm=contact_perm)
        )
        for family in FAMILIES:
            value = agreement[family]
            if value is not None:
                contact_null[family].append(value)

    return {
        "random_half": random_half,
        "chronological": chronological,
        "contact_null": contact_null,
    }


def window_state(
    agreements: Mapping[str, Any], *, alpha: float, min_resolved_families: int
) -> str:
    """One of BELOW_CHANCE / RELIABLE / CHRONOLOGY_BREAK / UNRESOLVED_FAMILIES (§6.3).

    Per family, in order:

    1. **Resolved** only when both its `random_half` and `contact_null` lists are
       non-empty — a family with either entirely empty (no draw produced a finite
       agreement) cannot be tested and does not enter the vote below.
    2. **Above chance** when the random-half *median* exceeds (strictly) the contact
       null's own 95th percentile (`q95`) — each family judged against its own null,
       never a shared/absolute threshold (§6.3: contact counts range ~8-16 in this
       cohort and a constant threshold would systematically penalise high-dimensional
       patients).
    3. Not above chance -> that family's vote is `BELOW_CHANCE`.
    4. Above chance -> `CHRONOLOGY_BREAK` when the chronological value sits below the
       random-half distribution's `alpha`-quantile (`np.percentile(random_half, 100 *
       alpha)`); otherwise `RELIABLE`. A family whose chronological value is `None`
       (that specific split could not be computed) is never treated as break evidence —
       absence of a comparison is not evidence a break occurred — so it defaults to
       `RELIABLE` provided it cleared the above-chance test.

    Fewer than `min_resolved_families` resolved families short-circuits to
    `UNRESOLVED_FAMILIES` before any vote is taken.

    Window-level verdict is by **strict majority** (more than half) of resolved
    families' votes, checked `BELOW_CHANCE` then `CHRONOLOGY_BREAK`; anything short of a
    majority for either defaults to `RELIABLE`. The brief states the majority rule
    explicitly only for `CHRONOLOGY_BREAK` ("... in a majority of resolved families");
    this extends the same rule to `BELOW_CHANCE` by symmetry as a controller-level
    completion (not a separately specified plan clause), and treats "no majority for a
    problem state" as `RELIABLE` rather than inventing a fifth label for a 3-family tie.
    """
    random_half = agreements["random_half"]
    contact_null = agreements["contact_null"]
    chronological = agreements["chronological"]

    labels: list[str] = []
    for family in FAMILIES:
        rh = random_half.get(family) or []
        cn = contact_null.get(family) or []
        if not rh or not cn:
            continue  # unresolved: not enough draws to judge this family at all

        median_rh = float(np.median(rh))
        q95_null = float(np.percentile(cn, 95))
        if median_rh <= q95_null:
            labels.append(BELOW)
            continue

        chrono = chronological.get(family)
        alpha_quantile = float(np.percentile(rh, 100.0 * float(alpha)))
        if chrono is not None and float(chrono) < alpha_quantile:
            labels.append(BREAK)
        else:
            labels.append(RELIABLE)

    if len(labels) < int(min_resolved_families):
        return UNRESOLVED_FAMILIES

    n = len(labels)
    if labels.count(BELOW) * 2 > n:
        return BELOW
    if labels.count(BREAK) * 2 > n:
        return BREAK
    return RELIABLE


def scale_states(windows_states: Sequence[str], *, min_windows: int) -> str:
    """Majority state over one scale's independent primary windows (§6.4).

    Fewer than `min_windows` windows -> `UNRESOLVED_TOO_FEW_WINDOWS`: a scale is
    evaluated only with enough independent primary windows behind it, and is excluded
    from `select_scales`'s pattern rather than counted as a failure.

    Otherwise the most frequent window state wins (plurality; with only two possible
    contenders in the common `min_windows=5`-with-3-families setting this is usually a
    true majority too, but the rule here is "most votes", not ">50%"). On an exact
    tie between two or more states, the earliest in
    `(BELOW_CHANCE, RELIABLE, CHRONOLOGY_BREAK, UNRESOLVED_FAMILIES)` wins — a
    controller decision (task-6 ambiguity resolution) to keep the reduction
    deterministic, not a rule derived from the plan text.
    """
    states = list(windows_states)
    if len(states) < int(min_windows):
        return TOO_FEW

    order = (BELOW, RELIABLE, BREAK, UNRESOLVED_FAMILIES)
    counts = {label: states.count(label) for label in order}
    best = max(counts.values())
    for label in order:
        if counts[label] == best:
            return label
    raise ValueError(f"unexpected window state(s) outside the closed label set: {states!r}")


def select_scales(states: Mapping[int, str]) -> dict[str, Any]:
    evaluated = [
        (size, states[size]) for size in sorted(states) if states[size] != TOO_FEW
    ]
    labels = [state for _, state in evaluated]
    empty = {
        "n_obs": None,
        "n_break": None,
        "n_last_reliable": None,
        "dwell_interval": None,
    }
    if RELIABLE not in labels:
        return {**empty, "status": "UNRESOLVED_SCALE"}

    first = labels.index(RELIABLE)
    last = len(labels) - 1 - labels[::-1].index(RELIABLE)
    leading_ok = all(state == BELOW for state in labels[:first])
    middle_ok = all(state == RELIABLE for state in labels[first : last + 1])
    trailing_ok = all(state == BREAK for state in labels[last + 1 :])
    if not (leading_ok and middle_ok and trailing_ok):
        return {**empty, "status": "UNRESOLVED_NONMONOTONE"}

    n_obs = evaluated[first][0]
    n_last_reliable = evaluated[last][0]
    n_break = evaluated[last + 1][0] if last + 1 < len(evaluated) else None
    return {
        "n_obs": n_obs,
        "n_break": n_break,
        "n_last_reliable": n_last_reliable,
        "dwell_interval": (n_last_reliable, n_break),
        "status": "SCALE_RESOLVED",
    }
