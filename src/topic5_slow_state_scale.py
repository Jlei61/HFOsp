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
# select_scales' own return `status` values. Never valid inputs anywhere -- not to
# window_state, not to scale_states, not to select_scales itself -- only outputs.
UNRESOLVED_SCALE = "UNRESOLVED_SCALE"
UNRESOLVED_NONMONOTONE = "UNRESOLVED_NONMONOTONE"

# fix-round-4, ITEM 1 (structural): ONE canonical set of every label that means "this
# could not be decided", at any level -- window-level (window_state's output, consumed
# by scale_states) or scale-level (scale_states' output, consumed by select_scales).
# Before this round there were two separate lists with different membership
# (the old NON_EVALUABLE_WINDOW_STATES lacked UNRESOLVED_MIXED_WINDOWS, the old
# NOT_EVALUATED lacked UNRESOLVED_FAMILY_DISCORDANCE) and they drifted out of sync four
# review rounds in a row: each new "unresolved" label was added to the alphabet and to
# ONE filter, but not reliably to the other. `scale_states` and `select_scales` both
# filter with this single set now. A label that cannot occur as an input at a given
# level (e.g. UNRESOLVED_MIXED_WINDOWS can never legitimately reach scale_states,
# because window_state never produces it) simply never matches anything there, so one
# superset is safe, and adding a future "unresolved" label requires exactly one edit:
# add it here.
NOT_EVALUABLE = frozenset(
    {TOO_FEW, UNRESOLVED_FAMILIES, UNRESOLVED_FAMILY_DISCORDANCE, UNRESOLVED_MIXED_WINDOWS}
)
# Disjoint from NOT_EVALUABLE by construction: these are never things a filter should
# drop FROM an input, because they can never legitimately appear IN one -- they are
# select_scales' own return statuses. Kept separately (rather than folded into
# NOT_EVALUABLE) so the meta-test below can classify every "UNRESOLVED_*"/TOO_FEW
# module constant into exactly one of the two sets and assert the sets never overlap.
OUTPUT_ONLY_STATUSES = frozenset({UNRESOLVED_SCALE, UNRESOLVED_NONMONOTONE})

# rev3 R3-C, deferred minor from the fix-round-4 ledger: `scale_states` used to validate
# against a hand-written tuple that duplicated the alphabet, so adding a label to
# NOT_EVALUABLE without also editing that tuple would make `scale_states` REJECT its own
# upstream output -- the same drift family as the four recurrences NOT_EVALUABLE already
# collapsed, in the one place the meta-test did not reach. Deriving it means one edit
# still suffices. This deliberately also admits TOO_FEW, which the hand-written tuple
# omitted: like every other member of NOT_EVALUABLE it is now dropped rather than raising,
# which is the safe direction and the same "one superset is safe" argument NOT_EVALUABLE
# is built on.
DECIDED_STATES = (BELOW, RELIABLE, BREAK)
KNOWN_WINDOW_STATES = frozenset(DECIDED_STATES) | NOT_EVALUABLE

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


def _backbone_offset(values: np.ndarray) -> np.ndarray:
    """The amount actually subtracted: the backbone where it exists, else zero.

    Both halves of the nan contract come from this one line. A descriptor this window
    could not estimate is `nan` and stays `nan` (`nan - anything` is `nan`), so
    residualising never invents a number where the window had none. A descriptor the
    *backbone* could not estimate contributes `0.0`, so residualising never destroys a
    number the window did estimate — `nan` would silently drop that contact or pair out
    of every agreement for the rest of the run.
    """
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values), values, 0.0)


def _permuted_pair_backbone(
    pair_effects: np.ndarray,
    pair_index: Sequence[tuple[int, int]],
    contact_perm: np.ndarray,
) -> np.ndarray:
    """The pair main effects re-expressed in a contact-permuted column layout.

    `local_repertoire` enumerates pairs as `i < j` over *array positions*. When the
    contact-null permutes columns first, position `i` holds physical contact
    `contact_perm[i]`, so the pair at layout position `(i, j)` is the physical ordered
    pair `(a, b) = (contact_perm[i], contact_perm[j])` and its precedence is
    `P(a before b)`. The backbone stores `P(a before b)` for `a < b` only, and
    precedence is its own complement — `P(b before a) = 1 - P(a before b)` holds exactly
    per window (earlier + tied/2 counts are complementary) and therefore also for the
    mean over the windows that estimated both. So a pair the permutation reverses takes
    `1 - effect`, which negates its residual; that reversal is part of what "contact
    identity was scrambled" means and it is applied to the null side only.
    """
    lookup = {tuple(pair): position for position, pair in enumerate(pair_index)}
    out = np.empty(len(pair_index), dtype=float)
    for position, (i, j) in enumerate(pair_index):
        a, b = int(contact_perm[i]), int(contact_perm[j])
        out[position] = (
            pair_effects[lookup[(a, b)]] if a < b else 1.0 - pair_effects[lookup[(b, a)]]
        )
    return out


def _residualise_descriptors(
    repertoire: Mapping[str, Any],
    backbone: Mapping[str, Any],
    *,
    contact_perm: np.ndarray | None = None,
) -> dict[str, Any]:
    """One window's descriptors minus the patient's backbone main effects (§6.6).

    `contact_perm` is the permutation `window_agreements` applied to the columns before
    computing this repertoire, or `None`. It is **not** ignored: the main effect belongs
    to a physical contact, so the backbone is reindexed by the same permutation and each
    position has its own contact's effect removed. Subtracting position-wise instead
    would leave the null side carrying a difference between two contacts' backbones
    while the observed side carries none, and the two sides would no longer be the same
    quantity.
    """
    n_contacts = len(np.asarray(repertoire["participation_rate"]))
    n_pairs = len(np.asarray(repertoire["precedence"]))
    if len(np.asarray(backbone["participation_rate"])) != n_contacts:
        raise ValueError(
            "backbone contact count does not match this window's — the backbone would be "
            "subtracted from the wrong contacts"
        )
    if len(np.asarray(backbone["precedence"])) != n_pairs:
        raise ValueError(
            "backbone pair count does not match this window's — the backbone would be "
            "subtracted from the wrong pairs"
        )

    contact_effects = {
        family: np.asarray(backbone[family], dtype=float)
        for family in ("participation_rate", "masked_mean_rank")
    }
    pair_effects = np.asarray(backbone["precedence"], dtype=float)
    if contact_perm is not None:
        contact_effects = {
            family: values[contact_perm] for family, values in contact_effects.items()
        }
        pair_effects = _permuted_pair_backbone(
            pair_effects, backbone["pair_index"], contact_perm
        )

    out = dict(repertoire)
    for family, effects in contact_effects.items():
        out[family] = np.asarray(repertoire[family], dtype=float) - _backbone_offset(effects)
    out["precedence"] = np.asarray(repertoire["precedence"], dtype=float) - _backbone_offset(
        pair_effects
    )
    return out


def window_agreements(
    rank: np.ndarray,
    participation: np.ndarray,
    group_ids: np.ndarray,
    *,
    random_half_draws: int,
    null_draws: int,
    seed: int,
    floors: Mapping[str, int],
    residualise: bool = False,
    backbone: Mapping[str, Any] | None = None,
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

    **`residualise` / `backbone` (rev3 R3-C, §6.6).** The raw descriptors are dominated
    by the patient's stable backbone — the fixed source, relay and sink structure that
    split-half and odd-even analyses established years ago. Run on them, a small window
    looks reliable merely because the backbone reproduces and a large window may never
    break because the backbone does not drift, so the scale curve would restate the known
    backbone result as a slow-state timescale. Each patient therefore gets **two** curves
    from this one function:

    - `residualise=False` (default, behaviour unchanged) — the raw curve. Quality control
      only: "can a window this size see this patient's stable repertoire at all". Yields
      `N_obs_backbone`.
    - `residualise=True` — `backbone` (from `estimate_backbone`, fitted on the TRAIN
      windows only) is subtracted from **every** repertoire this function computes, on
      both sides of every split and on the null side too, before any agreement is taken.
      Yields `N_obs_state`, and `N_break` comes from this curve's chronological values.

    `backbone` is ignored when `residualise=False`, so a caller may hold one backbone and
    loop over both settings. `residualise=True` without a backbone raises rather than
    silently returning the raw curve under the residual curve's name.
    """
    if residualise and backbone is None:
        raise ValueError(
            "residualise=True needs a backbone — pass estimate_backbone(train_repertoires); "
            "without one this would silently return the raw curve"
        )
    rank = np.asarray(rank, dtype=float)
    participation = np.asarray(participation)
    group_ids = np.asarray(group_ids)
    n_events, n_contacts = rank.shape
    rng = np.random.default_rng(seed)
    kwargs = _local_repertoire_kwargs(floors)

    def _repertoire(event_idx: np.ndarray, contact_perm: np.ndarray | None = None) -> dict[str, Any]:
        if contact_perm is None:
            out = local_repertoire(
                rank[event_idx], participation[event_idx], group_ids[event_idx], **kwargs
            )
        else:
            cols = np.ix_(event_idx, contact_perm)
            out = local_repertoire(rank[cols], participation[cols], group_ids[cols], **kwargs)
        if not residualise:
            return out
        return _residualise_descriptors(out, backbone, contact_perm=contact_perm)

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

    1. Every element of `windows_states` is validated against `KNOWN_WINDOW_STATES` up
       front, so an unrecognised state raises `ValueError` immediately rather than being
       silently uncounted. That set is *derived* from `DECIDED_STATES` and
       `NOT_EVALUABLE` rather than restated, so a future label added to `NOT_EVALUABLE`
       is accepted here in the same edit instead of being rejected as unknown.
    2. `UNRESOLVED_FAMILIES` and `UNRESOLVED_FAMILY_DISCORDANCE` windows never vote —
       `window_state` reached no judgement about them at all, so they are dropped before
       anything is counted (`NOT_EVALUABLE`, shared with `select_scales`).
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

    unknown = sorted(set(states) - KNOWN_WINDOW_STATES)
    if unknown:
        raise ValueError(f"unexpected window state(s) outside the closed label set: {unknown!r}")

    evaluable = [state for state in states if state not in NOT_EVALUABLE]
    if len(evaluable) < int(min_windows):
        return TOO_FEW

    n = len(evaluable)
    for label in DECIDED_STATES:
        if evaluable.count(label) * 2 > n:
            return label
    return UNRESOLVED_MIXED_WINDOWS


def select_scales(states: Mapping[int, str]) -> dict[str, Any]:
    """Pattern-match a monotone BELOW* RELIABLE+ BREAK* run over the evaluated scales.

    Filters with the same `NOT_EVALUABLE` set `scale_states` uses to drop non-evaluable
    window states -- a scale whose own state could not be decided (any member of
    `NOT_EVALUABLE`) is dropped before pattern matching, not treated as an observed
    state the monotone run must account for (fix-round-4 ITEM 1; this function and
    `scale_states` used to filter with two separately-maintained lists that drifted out
    of sync across four review rounds).
    """
    evaluated = [
        (size, states[size]) for size in sorted(states) if states[size] not in NOT_EVALUABLE
    ]
    labels = [state for _, state in evaluated]
    empty = {
        "n_obs": None,
        "n_break": None,
        "n_last_reliable": None,
        "dwell_interval": None,
    }
    if RELIABLE not in labels:
        return {**empty, "status": UNRESOLVED_SCALE}

    first = labels.index(RELIABLE)
    last = len(labels) - 1 - labels[::-1].index(RELIABLE)
    leading_ok = all(state == BELOW for state in labels[:first])
    middle_ok = all(state == RELIABLE for state in labels[first : last + 1])
    trailing_ok = all(state == BREAK for state in labels[last + 1 :])
    if not (leading_ok and middle_ok and trailing_ok):
        return {**empty, "status": UNRESOLVED_NONMONOTONE}

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
