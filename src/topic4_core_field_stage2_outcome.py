"""Stage 2 outcome classification, frozen BEFORE Stage 2 runs (spec 8.1 / 8.2).

The ORDER is part of the contract: it decides which label is reported when several
conditions hold at once. Evaluated on held-out seeds only.
"""
from __future__ import annotations

import itertools

import numpy as np

# Short-circuit order, first match wins.
OUTCOME_ORDER = (
    "FAIL_CLOSED",
    "SIMULATOR_OVERFIT",
    "ONE_DIRECTION_ONLY",
    "FIELD_NONIDENTIFIABLE",
    "UNIDENTIFIABLE",
    "AXIS_ONLY_SUFFICIENT",
    "UNIFORM_CORRIDOR_SUFFICIENT",
    "MANUAL_HEURISTIC_RETAINED",
    "LOW_COVERAGE_WIN",
    "RECOVERED_NONTRIVIAL_FIELD",
)

ALLOWED_STATEMENTS = {
    "FAIL_CLOSED":
        "artifacts are incomplete or inconsistent; nothing may be concluded",
    "SIMULATOR_OVERFIT":
        "the optimisation fitted the simulator's own random realisation, not a "
        "transferable field structure",
    "ONE_DIRECTION_ONLY":
        "the field recovers one propagation direction; it may NOT be described as "
        "reproducing the patient's two-template set",
    "FIELD_NONIDENTIFIABLE":
        "the score does not pin down the field's shape -- report the equivalent-optimum "
        "family, never a single solution",
    "UNIDENTIFIABLE":
        "the learned field is unstable across restarts and seeds; report the instability, "
        "not the best single run",
    "AXIS_ONLY_SUFFICIENT":
        "the score is explained by propagation along the fixed axis; the field's shape "
        "makes no detectable contribution",
    "UNIFORM_CORRIDOR_SUFFICIENT":
        "a uniform axial corridor at the same budget is enough; the data do not support "
        "a more structured field",
    "MANUAL_HEURISTIC_RETAINED":
        "the hand-placed two-end cores remain the better working point; the inversion "
        "found nothing better",
    "LOW_COVERAGE_WIN":
        "the apparent win comes with materially lower contact recruitment and does not "
        "count towards the primary claim",
    "RECOVERED_NONTRIVIAL_FIELD":
        "a non-trivial pathology field inverted from contact ranks beats both pure "
        "geometry and a uniform corridor at the same budget",
}

SIGN_MAJORITY = 11          # of 12 held-out seeds, matching the Stage 1 sign convention
FAMILY_CORR_FLOOR = 0.50    # spec 8.2
RESTART_CORR_FLOOR = 0.50   # spec 8.1


def _beats(d):
    """A difference counts as a win only if it is positive AND consistently signed."""
    return bool(d["mean"] > 0 and d["n_above"] >= SIGN_MAJORITY)


def _loses(d):
    return bool(d["mean"] < 0 and (d["n"] - d["n_above"]) >= SIGN_MAJORITY)


def classify_stage2(res):
    """Pure function. `res` is the held-out summary; see the tests for its shape."""
    def out(name, **extra):
        return dict(outcome=name, allowed_statement=ALLOWED_STATEMENTS[name], **extra)

    if not res.get("integrity_ok", False):
        return out("FAIL_CLOSED")

    train, held = float(res["train_delta"]), float(res["heldout_delta"])
    if train > 0 and held <= 0:
        return out("SIMULATOR_OVERFIT", train_delta=train, heldout_delta=held)

    if not res["bidirectional_gate"]["passed"]:
        return out("ONE_DIRECTION_ONLY")

    fam = res["family"]
    if float(fam["median_field_corr"]) < FAMILY_CORR_FLOOR:
        return out("FIELD_NONIDENTIFIABLE", family=dict(fam))

    if float(res["restart_field_corr_median"]) < RESTART_CORR_FLOOR:
        return out("UNIDENTIFIABLE",
                   restart_field_corr_median=float(res["restart_field_corr_median"]))

    if not _beats(res["vs_axis_only"]):
        return out("AXIS_ONLY_SUFFICIENT", vs_axis_only=dict(res["vs_axis_only"]))

    if not _beats(res["vs_uniform"]):
        return out("UNIFORM_CORRIDOR_SUFFICIENT", vs_uniform=dict(res["vs_uniform"]))

    if _loses(res["vs_manual_projected"]):
        return out("MANUAL_HEURISTIC_RETAINED",
                   vs_manual_projected=dict(res["vs_manual_projected"]))

    cov = res["coverage"]
    if float(cov["learned"]) < float(cov["manual_smooth"]) - float(cov["margin"]):
        return out("LOW_COVERAGE_WIN", coverage=dict(cov))

    return out("RECOVERED_NONTRIVIAL_FIELD")


classify_stage2.ALLOWED_STATEMENTS = ALLOWED_STATEMENTS


def equivalent_optimum_family(scores, fields, paired_sd):
    """Every candidate within one paired sd of the best held-out score.

    Showing only the single best solution over-interprets: spatially different
    fields can share a score. If the family disagrees, the score does not pin the
    field (spec 8.2).
    """
    scores = np.asarray(scores, float)
    best = float(np.nanmax(scores))
    keep = [i for i in range(scores.size)
            if np.isfinite(scores[i]) and scores[i] >= best - float(paired_sd)]
    members = [np.asarray(fields[i], float) for i in keep]
    corrs = [float(np.corrcoef(a, b)[0, 1]) for a, b in itertools.combinations(members, 2)]
    stacked = np.vstack(members) if members else np.zeros((0, 0))
    return dict(
        n_members=len(members),
        member_indices=keep,
        median_field_corr=float(np.median(corrs)) if corrs else 1.0,
        min_field_corr=float(np.min(corrs)) if corrs else 1.0,
        mean_field=stacked.mean(axis=0).tolist() if len(members) else [],
        sd_field=stacked.std(axis=0).tolist() if len(members) else [],
    )
