"""The registered four-way temporal-geometry label, emitted from frozen evidence.

The plan asks the slow-vector stage for exactly one of ``DX_GEOMETRIC_PATH_PRESENT
/ _ABSENT / DX_DYNAMIC_VECTOR_MISALIGNED / DX_MAP_UNRESOLVED``, and says plainly
that it authorizes nothing: it neither opens nor closes the dynamic stage.  The
stage ran without emitting it, so it is derived here from the frozen cells and
the drift vectors those runs already stored.

Two rules carry the weight.

**A quiet cell is only evidence of quiet if it was watched for longer than
ignition takes.**  Every frozen low cell ran 1500 ms; the no-kick trajectories
ignite at 4000-6000 ms.  Reading those 42 cells as "the quiet state never
departs" would report the screen window, not the tissue -- the same failure that
already produced two retractions on this stage, where a window opening on a
state transition was read as the state, and a clamped relay that could not refill
was read as the tissue being silent.  So window adequacy is checked against the
observed ignition times before absence is allowed to mean anything.

**A mean is not a direction.**  Released high states drift to higher relay on the
array mean while their cores drift to lower relay, so the mean and the cores
point at opposite sides of the return boundary.  Both are returned; neither is
allowed to stand alone.
"""
from __future__ import annotations

PATHLABEL_SCHEMA = "fcxr-lc3-path-label-1.0"

HIGH_BRANCH_LABELS = ("FINITE_HIGH_FIXED", "FINITE_HIGH_ORBIT")
QUIET_LABELS = ("INTERICTAL_WORKPOINT",)
REGION_KEYS = ("core_A", "core_B", "axial", "off_axis")


def window_is_adequate(cells, reference_ms) -> dict:
    """Was the quiet side watched for at least as long as ignition takes?

    ``reference_ms`` is the fastest observed no-kick ignition.  A cell that
    stayed quiet for less than that has not been given the chance to depart, so
    its label is uninformative about departure however clean it looks.
    """
    if not cells:
        raise ValueError("window adequacy needs at least one cell")
    if not (isinstance(reference_ms, (int, float)) and reference_ms > 0):
        raise ValueError("reference_ms must be a positive ignition time")
    longest = max(float(c["total_ms"]) for c in cells)
    n_adequate = sum(1 for c in cells if float(c["total_ms"]) >= float(reference_ms))
    return dict(
        adequate=bool(n_adequate > 0), n_adequate=n_adequate, n_cells=len(cells),
        longest_window_ms=longest, reference_ms=float(reference_ms),
        shortfall_ms=max(0.0, float(reference_ms) - longest),
    )


def has_high_branch(cells) -> bool:
    """Does any cell hold a high branch?"""
    return any(c["resolved_label"] in HIGH_BRANCH_LABELS for c in cells)


def return_bracket(cells) -> dict:
    """The relay interval across which a started high branch stops surviving.

    A bracket needs a surviving cell above a quiet cell on the same wear field,
    so it is derived per field and only then pooled.
    """
    by_field = {}
    for c in cells:
        by_field.setdefault(c["d_label"], []).append(c)
    brackets = {}
    for field, group in by_field.items():
        survives = sorted(float(c["a_x"]) for c in group
                          if c["resolved_label"] in HIGH_BRANCH_LABELS)
        quiets = sorted(float(c["a_x"]) for c in group
                        if c["resolved_label"] not in HIGH_BRANCH_LABELS)
        if not survives or not quiets:
            continue
        below = [a for a in quiets if a < min(survives)]
        if below:
            brackets[field] = (max(below), min(survives))
    return dict(present=bool(brackets), per_field=brackets,
                n_fields=len(by_field), n_bracketed=len(brackets))


def drift_toward_return(vectors, bracket_top) -> dict:
    """Does the measured drift carry a high state down toward the return bracket?

    Returning requires relay to fall.  Reported on the array mean and on each
    region separately: on this stage they disagree, and collapsing them to the
    mean erases the only component that moves the right way.
    """
    considered = [v for v in vectors
                  if v.get("state_kind") == "high" and float(v["a_x"]) >= float(bracket_top)]
    if not considered:
        return dict(measured=False, reason="no high-state vector at or above the bracket")
    mean_down = [v for v in considered if float(v["dot_mean_a_X_per_s"]) < 0.0]
    regions_down = {
        k: sum(1 for v in considered if float(v["regional_X_change"][k]) < 0.0)
        for k in REGION_KEYS}
    return dict(
        measured=True, n_vectors=len(considered),
        n_mean_toward_return=len(mean_down),
        mean_reaches_return=bool(mean_down),
        regions_toward_return={k: n for k, n in regions_down.items() if n},
        any_region_reaches_return=any(regions_down.values()),
    )


def temporal_geometry_label(*, low_cells, high_cells, vectors, ignition_times_ms) -> dict:
    """One of the four registered labels, with the evidence that produced it.

    Order matters.  Adequacy is asked before absence, because an inadequate
    window makes absence unreadable rather than negative; and the entry side is
    asked before the vectors, because a path needs both ends before its
    direction is worth checking.
    """
    if not ignition_times_ms:
        raise ValueError("adequacy needs at least one observed ignition time")
    fastest = min(float(t) for t in ignition_times_ms)
    entry_seen = has_high_branch(low_cells)
    adequacy = window_is_adequate(low_cells, fastest)
    ret = return_bracket(high_cells)
    top = min((v[1] for v in ret["per_field"].values()), default=None)
    drift = (drift_toward_return(vectors, top) if top is not None
             else dict(measured=False, reason="no return bracket to aim at"))

    if not entry_seen and not adequacy["adequate"]:
        label = "DX_MAP_UNRESOLVED"
        why = (f"no frozen quiet cell departed, but the longest quiet window is "
               f"{adequacy['longest_window_ms']:.0f} ms against a fastest observed "
               f"ignition of {fastest:.0f} ms, so departure was never given the "
               f"time it takes; absence here is the screen window, not the tissue")
    elif not entry_seen:
        label = "DX_GEOMETRIC_PATH_ABSENT"
        why = (f"{adequacy['n_adequate']} quiet cells were watched past the "
               f"{fastest:.0f} ms ignition time and none departed")
    elif not ret["present"]:
        label = "DX_GEOMETRIC_PATH_ABSENT"
        why = "quiet cells depart but no started high branch stops surviving"
    elif not (drift.get("mean_reaches_return") or drift.get("any_region_reaches_return")):
        label = "DX_DYNAMIC_VECTOR_MISALIGNED"
        why = ("both brackets exist, but no measured drift component carries a "
               "high state toward the return bracket")
    else:
        label = "DX_GEOMETRIC_PATH_PRESENT"
        why = "both brackets exist and measured drift connects them in projection"

    return dict(
        schema=PATHLABEL_SCHEMA, label=label, reason=why,
        authorizes_nothing=("registered as non-gating: it neither opens nor "
                            "closes the dynamic stage"),
        entry=dict(any_quiet_cell_departed=entry_seen, window=adequacy),
        return_bracket=ret, drift=drift,
        fastest_observed_ignition_ms=fastest,
    )
