"""Two scales, not one.

`N_obs` is the smallest window at which the repertoire rises above this patient's own
chance level; it becomes the model step, because a state that persists for 500 events cut
into 500-event blocks leaves one block per state and no dwell to observe.

`N_break` is the smallest larger window at which the chronological halves agree worse than
random halves — where a window starts averaging over a state change.

Scales with too few independent windows are skipped, not counted as failures, and a
leading run of below-chance scales is expected rather than disqualifying.

A dropped mid-grid scale (`UNRESOLVED_TOO_FEW_WINDOWS` or `UNRESOLVED_FAMILIES`) can
fabricate apparent monotonicity: `select_scales` removes it before pattern matching, so it
never sees what state that scale was actually in. A reported `N_break` or dwell interval
whose two grid points are not adjacent in the original grid spans a dropped scale, and must
be read together with which grid point was skipped rather than as if the grid were
contiguous.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.topic5_slow_state_repertoire import FAMILIES, family_agreement, local_repertoire

BELOW, RELIABLE, BREAK = "BELOW_CHANCE", "RELIABLE", "CHRONOLOGY_BREAK"
TOO_FEW = "UNRESOLVED_TOO_FEW_WINDOWS"
UNRESOLVED_FAMILIES = "UNRESOLVED_FAMILIES"
# rev3 R3-A: a window whose above-chance families cannot agree among themselves on
# whether chronology broke (an exact tie in the break vote) is discordant, not reliable.
UNRESOLVED_FAMILY_DISCORDANCE = "UNRESOLVED_FAMILY_DISCORDANCE"
# rev3 R3-B: a scale whose evaluable windows cannot reach a strict majority on any one
# of BELOW_CHANCE / RELIABLE / CHRONOLOGY_BREAK, after non-evaluable windows are
# dropped and the minimum is re-checked against the surviving count.
UNRESOLVED_MIXED_WINDOWS = "UNRESOLVED_MIXED_WINDOWS"
# Window-level verdicts that carry no vote at the scale level: window_state could not
# reach a judgement about this window at all, so it must be dropped before scale_states
# counts anything, and must not be able to satisfy min_windows either.
NON_EVALUABLE_WINDOW_STATES = (UNRESOLVED_FAMILIES, UNRESOLVED_FAMILY_DISCORDANCE)
# rev3 follow-up (recurrence of fix round 1's C2 under a new label): a scale whose
# windows could not reach a majority (UNRESOLVED_MIXED_WINDOWS) is undecided, not
# failed, and must be dropped before select_scales pattern-matches, the same as
# UNRESOLVED_TOO_FEW_WINDOWS and UNRESOLVED_FAMILIES already are -- otherwise one
# undecidable mid-grid scale throws the whole patient to UNRESOLVED_NONMONOTONE.
# UNRESOLVED_FAMILY_DISCORDANCE is deliberately NOT included here: it is a
# window-level verdict (window_state's output), never a scale-level one -- scale_states
# only ever returns TOO_FEW, BELOW, RELIABLE, BREAK, or UNRESOLVED_MIXED_WINDOWS, so it
# can never reach select_scales as a value in its `states` mapping. Adding it here
# would be dead defensive code for an input select_scales cannot receive.
NOT_EVALUATED = (TOO_FEW, UNRESOLVED_FAMILIES, UNRESOLVED_MIXED_WINDOWS)

# I3 fix: a family must clear a minimum number of *finite* draws on both `random_half` and
# `contact_null` before it counts as "resolved" in `window_state` -- a single finite null
# draw makes its q95 trivially easy to beat, biasing toward RELIABLE and a smaller N_obs.
# `window_state` receives only `agreements`/`alpha`/`min_resolved_families` (the frozen
# Interfaces block), not `null_draws` itself, so `max(20, null_draws // 10)` is evaluated
# here against the frozen config's `null_draws: 200` / `random_half_draws: 200`
# (config/topic5_slow_state_v4_0.yaml) -> max(20, 200 // 10) = 20. If `null_draws` is ever
# reconfigured away from 200 this constant does not auto-scale and must be revisited.
MIN_FINITE_DRAWS_FOR_RESOLUTION = max(20, 200 // 10)


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
    """One of BELOW_CHANCE / RELIABLE / CHRONOLOGY_BREAK / UNRESOLVED_FAMILIES /
    UNRESOLVED_FAMILY_DISCORDANCE (§6.3).

    **Resolved.** A family counts as resolved only when ALL three hold:

    1. its `random_half` list has at least `MIN_FINITE_DRAWS_FOR_RESOLUTION` finite
       values (fix I3 — a single finite null draw makes that draw's implied q95
       trivially easy to beat; a near-empty null must not be usable as a chance floor);
    2. its `contact_null` list also has at least `MIN_FINITE_DRAWS_FOR_RESOLUTION`
       finite values, for the same reason;
    3. its `chronological` value is not `None` (fix I7 — the conservative reading: a
       family whose chronological split could not be computed is not evidence of
       *anything*, above-chance or break, so it does not count as resolved at all,
       rather than defaulting to "no break evidence" as an earlier revision did. This
       biases the estimator toward *fewer* resolved families rather than toward a
       later/wider `N_break`/dwell built partly on families that were never actually
       compared chronologically).

    Fewer than `min_resolved_families` resolved -> `UNRESOLVED_FAMILIES`, checked before
    any vote below.

    **Per-family tests**, computed for every resolved family:

    - *above chance*: random-half **median** exceeds (strictly) contact-null's own 95th
      percentile (`q95`) — each family judged against its own null, never a
      shared/absolute threshold (§6.3: contact counts range ~8-16 in this cohort and a
      constant threshold would systematically penalise high-dimensional patients).

    ... and, only for families that test above chance:

    - *chronology below alpha*: the chronological value is strictly less than the
      random-half distribution's `alpha`-quantile (`np.percentile(random_half, 100 *
      alpha)`).

    **Window-level verdict**, in this exhaustive, explicit precedence (fix C1 — RELIABLE
    must never be a bare fallthrough for "neither majority was reached"; it requires
    positive above-chance evidence):

    1. fewer than `min_resolved_families` resolved -> `UNRESOLVED_FAMILIES`;
    2. `BELOW_CHANCE` **unless** a strict majority (more than half) of resolved families
       are above chance — this is the complement of "majority above chance", so a tie
       (e.g. 1-of-2, 2-of-4) or a minority also falls here, not into `RELIABLE`;
    3. among windows that passed step 2 (a majority above chance), `CHRONOLOGY_BREAK`
       when a strict majority of the **above-chance** resolved families (fix round 2,
       Item A — deliberately a *different*, smaller denominator than step 2's "all
       resolved families") show chronology below alpha, because a family that cannot be
       told apart from its own contact-permuted null has no meaningful chronological
       comparison to cast in the first place — its first-half-versus-second-half
       difference is noise about a quantity that was never estimable, so it must not be
       able to swing the break vote either way;
    4. `RELIABLE` when a strict majority of the **same above-chance** denominator show
       NO break — the complement of step 3 on that denominator;
    5. `UNRESOLVED_FAMILY_DISCORDANCE` (rev3 R3-A) when neither step 3 nor step 4 reaches
       a strict majority — an exact tie among the above-chance families' break votes
       (e.g. 1-of-2, 2-of-4) is discordance among the families themselves, never
       evidence of reliability. `RELIABLE` must never be a bare fallthrough for "the
       break vote didn't reach a majority either way", the same C1 principle step 2
       already applies to the above-chance vote: RELIABLE requires *reaching* step 4
       via its own strict majority, not landing there because nothing else matched.
    """
    random_half = agreements["random_half"]
    contact_null = agreements["contact_null"]
    chronological = agreements["chronological"]

    above_chance: list[bool] = []
    chrono_below_alpha: list[bool] = []  # only for families that ARE above chance
    for family in FAMILIES:
        rh = random_half.get(family) or []
        cn = contact_null.get(family) or []
        chrono = chronological.get(family)
        if (
            len(rh) < MIN_FINITE_DRAWS_FOR_RESOLUTION
            or len(cn) < MIN_FINITE_DRAWS_FOR_RESOLUTION
            or chrono is None
        ):
            continue  # unresolved: not enough evidence to judge this family at all

        median_rh = float(np.median(rh))
        q95_null = float(np.percentile(cn, 95))
        is_above_chance = median_rh > q95_null
        above_chance.append(is_above_chance)

        if is_above_chance:
            alpha_quantile = float(np.percentile(rh, 100.0 * float(alpha)))
            chrono_below_alpha.append(float(chrono) < alpha_quantile)

    n = len(above_chance)
    if n < int(min_resolved_families):
        return UNRESOLVED_FAMILIES

    if not (sum(above_chance) * 2 > n):
        return BELOW

    n_above_chance = len(chrono_below_alpha)
    n_break_votes = sum(chrono_below_alpha)
    if n_break_votes * 2 > n_above_chance:
        return BREAK

    n_no_break_votes = n_above_chance - n_break_votes
    if n_no_break_votes * 2 > n_above_chance:
        return RELIABLE

    # rev3 R3-A: neither a strict break-majority (step 3) nor a strict no-break-majority
    # (step 4) was reached -- an exact tie among the above-chance families. This must
    # not fall through to RELIABLE.
    return UNRESOLVED_FAMILY_DISCORDANCE


def scale_states(windows_states: Sequence[str], *, min_windows: int) -> str:
    """Majority state over one scale's independent primary windows (§6.4).

    rev3 R3-B: this used to reduce to the *mode* (most frequent label, tie-broken by a
    fixed order) — a window state could win with as little as 2-of-5 support whenever
    the other 3 were split across the remaining labels. That is not "most windows agree
    on this scale's state"; it is "this was merely the largest minority". The rule is
    now a strict majority of the *evaluable* windows, computed as follows:

    1. Every element of `windows_states` is validated against this closed label set up
       front, so an unrecognised state raises `ValueError` immediately rather than being
       silently uncounted.
    2. `UNRESOLVED_FAMILIES` and `UNRESOLVED_FAMILY_DISCORDANCE` windows never vote —
       `window_state` reached no judgement about them at all, so they are dropped before
       anything is counted (`NON_EVALUABLE_WINDOW_STATES`).
    3. `min_windows` is re-checked against the SURVIVING evaluable count, not the
       original `len(windows_states)` — dropping non-evaluable windows can itself take a
       scale below the minimum, and that must read as `UNRESOLVED_TOO_FEW_WINDOWS`, not
       as a vote among however few evaluable windows happen to remain.
    4. `BELOW_CHANCE` / `RELIABLE` / `CHRONOLOGY_BREAK` is returned only on a strict
       majority (more than half) of the evaluable windows. Otherwise —
       `UNRESOLVED_MIXED_WINDOWS`: the evaluable windows exist and clear the minimum, but
       cannot agree on a single scale state.
    """
    states = list(windows_states)

    known = (
        BELOW,
        RELIABLE,
        BREAK,
        UNRESOLVED_FAMILIES,
        UNRESOLVED_FAMILY_DISCORDANCE,
        UNRESOLVED_MIXED_WINDOWS,
    )
    unknown = sorted(set(states) - set(known))
    if unknown:
        raise ValueError(f"unexpected window state(s) outside the closed label set: {unknown!r}")

    evaluable = [state for state in states if state not in NON_EVALUABLE_WINDOW_STATES]
    if len(evaluable) < int(min_windows):
        return TOO_FEW

    n = len(evaluable)
    for label in (BELOW, RELIABLE, BREAK):
        if evaluable.count(label) * 2 > n:
            return label
    return UNRESOLVED_MIXED_WINDOWS


def select_scales(states: Mapping[int, str]) -> dict[str, Any]:
    evaluated = [
        (size, states[size]) for size in sorted(states) if states[size] not in NOT_EVALUATED
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
