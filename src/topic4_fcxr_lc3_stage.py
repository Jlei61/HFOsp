"""How far around the loop an arm actually got.

The loop being tracked is: repeated interictal events -> wear accumulates ->
the tissue enters a bounded high state -> the relay terminates it -> wear
recovers -> interictal events return.  Five stages name where an arm stopped:

``ONE_SHOT``            entered, but without a preceding train -- the high state
                        was reached in one go, so nothing accumulated into it
``IED_TRAIN_NO_ONSET``  events came and went; the tissue never entered
``ONSET_NO_OFFSET``     entered and never stopped
``OFFSET_NO_RECOVERY``  entered and stopped, but did not come back to the
                        interictal neighbourhood
``FULL_LIFECYCLE``      entered, stopped, and returned to interictal events
                        inside the frozen reference distribution

Returning is judged against the frozen baseline's own event distribution, not
against a population rate.  A tissue can sit at a plausible mean rate while
emitting no discrete events at all, which is how a clamped relay first read as
"back to the interictal rate" on this stage before the retraction.  So the
return test asks for returning events whose count, duration and participation
land inside the reference band -- silence and a smouldering train both fail it,
and they fail it for different reasons that the record keeps separate.
"""
from __future__ import annotations

STAGE_SCHEMA = "fcxr-lc3-lifecycle-stage-1.0"
ACCUMULATION_BAR = 3        # same bar the entry ledger classifies against

STAGE_ORDER = (
    "IED_TRAIN_NO_ONSET",
    "ONE_SHOT",
    "ONSET_NO_OFFSET",
    "OFFSET_NO_RECOVERY",
    "FULL_LIFECYCLE",
)


def reference_band(baseline) -> dict:
    """The frozen baseline's full event distribution, not just its rate edges.

    ``band`` on disk carries only the event-rate interval; the shape of a
    returning event lives one level up as the 34 reference events' own durations
    and participations.  Without them the shape check silently does not run, so
    they are folded in here rather than left to each caller.
    """
    band = dict(baseline["band"])
    durations = baseline.get("event_durations_ms") or []
    participation = baseline.get("event_participation") or []
    if durations:
        band["dur_lo_ms"], band["dur_hi_ms"] = float(min(durations)), float(max(durations))
    if participation:
        band["part_lo"], band["part_hi"] = float(min(participation)), float(max(participation))
    band["n_reference_events"] = len(durations)
    return band


def returned_to_reference(*, n_returning_after_offset, event_rate_hz, band,
                          durations_ms=(), participation=()) -> dict:
    """Did the tissue come back to the frozen baseline's own event distribution?

    ``band`` is the frozen LC1 reference: an event-rate interval plus duration
    and participation ranges.  Zero events fails on count before any rate is
    consulted, because a mean rate computed over a silent window is not evidence
    of quiet interictal activity.
    """
    for key in ("event_rate_lo", "event_rate_hi"):
        if key not in band:
            raise ValueError(f"reference band needs {key}")
    if n_returning_after_offset <= 0:
        return dict(returned=False, reason="no returning events after offset",
                    n_returning=int(n_returning_after_offset))
    rate_in = bool(band["event_rate_lo"] <= event_rate_hz <= band["event_rate_hi"])
    checks = dict(event_rate=rate_in)
    for name, values, lo_key, hi_key in (
            ("duration", durations_ms, "dur_lo_ms", "dur_hi_ms"),
            ("participation", participation, "part_lo", "part_hi")):
        if values and lo_key in band and hi_key in band:
            inside = sum(1 for v in values if band[lo_key] <= v <= band[hi_key])
            checks[name] = bool(inside == len(values))
    returned = all(checks.values())
    return dict(returned=returned, checks=checks,
                n_returning=int(n_returning_after_offset),
                event_rate_hz=float(event_rate_hz),
                reason=("inside the frozen reference distribution" if returned
                        else "returning events outside the frozen reference "
                             + ", ".join(k for k, v in checks.items() if not v)))


def lifecycle_stage(*, onset_ms, offset_ms, n_returning_before_onset,
                    return_check=None) -> dict:
    """The furthest stage this arm reached, with the reason it stopped there.

    ``return_check`` is the mapping ``returned_to_reference`` produces; ``None``
    means the return was never measured, which is not the same as failing it and
    is recorded as such.
    """
    if onset_ms is None:
        return dict(schema=STAGE_SCHEMA, stage="IED_TRAIN_NO_ONSET",
                    reason="no onset in the observed window")
    if n_returning_before_onset is not None and n_returning_before_onset < ACCUMULATION_BAR:
        return dict(schema=STAGE_SCHEMA, stage="ONE_SHOT",
                    reason=(f"onset preceded by {n_returning_before_onset} returning "
                            f"events, below the {ACCUMULATION_BAR}-event accumulation bar"))
    if offset_ms is None:
        return dict(schema=STAGE_SCHEMA, stage="ONSET_NO_OFFSET",
                    reason="entered and did not terminate in the observed window")
    if return_check is None:
        return dict(schema=STAGE_SCHEMA, stage="OFFSET_NO_RECOVERY",
                    reason="terminated, but the return was never measured",
                    return_measured=False)
    if not return_check["returned"]:
        return dict(schema=STAGE_SCHEMA, stage="OFFSET_NO_RECOVERY",
                    reason=f"terminated, but {return_check['reason']}",
                    return_measured=True, return_check=return_check)
    return dict(schema=STAGE_SCHEMA, stage="FULL_LIFECYCLE",
                reason="entered after accumulation, terminated, and returned",
                return_measured=True, return_check=return_check)


def stage_index(stage) -> int:
    """Position in the loop, for ordering arms by how far they got."""
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown stage {stage!r}")
    return STAGE_ORDER.index(stage)
