"""Stage 3 outcome taxonomy, frozen before any simulation (spec section 9.4).

Two lessons from Stage 2 are built into this module.

First, ordering. Stage 2 put SIMULATOR_OVERFIT at rule 1, it fired, and every
downstream scientific question became unanswerable -- including the one the
stage existed to ask. Here POSITION_UNIDENTIFIABLE comes first instead, because
position instability is the more basic failure: if independent restarts return
fields that do not resemble each other, there is no stable object whose
train-versus-held-out gap is worth discussing.

Second, and more important: reordering alone would just replay the same
occlusion in the other direction. So classify_stage3 returns every component --
position stability, held-out transfer, the axis relation and the full list of
triggered conditions -- alongside the short-circuit label. The primary outcome
is for one-line reporting; it is never the only thing a reader can see.
"""
from __future__ import annotations

import math

OUTCOME_ORDER = (
    "FAIL_CLOSED",              # 0
    "POSITION_UNIDENTIFIABLE",  # 1
    "SIMULATOR_OVERFIT",        # 2
    "AXIS_REDISCOVERED",        # 3
    "AXIS_NOT_REQUIRED",        # 4
    "AXIS_INCONCLUSIVE",        # 5
)

R_BAR_SD_MAX_MM = 1.0        # cross-restart spread of the transverse centroid
FIELD_CORR_MIN = 0.5         # cross-restart field similarity
NEAR_AXIS_MM = 1.0           # |r_bar| below this counts as on the axis
NEAR_AXIS_MASS = 0.7         # with at least this much mass within 2 mm
OFF_AXIS_MM = 2.0            # |r_bar| above this counts as off the axis

REQUIRED = ("restart_field_corr_median", "restart_r_bar_sd", "train_delta",
            "heldout_delta", "r_bar", "c_axis_2mm", "heldout_delta_vs_onaxis")

ALLOWED_STATEMENTS = {
    "FAIL_CLOSED":
        "artefacts incomplete or inconsistent; no scientific statement is licensed",
    "POSITION_UNIDENTIFIABLE":
        "this read-out cannot determine where the pathology sits; report the "
        "family of fields, never a single solution",
    "SIMULATOR_OVERFIT":
        "the search fitted this simulator's particular random draws rather than "
        "transferable field structure",
    "AXIS_REDISCOVERED":
        "given a network whose propagation direction was already set from the "
        "patient's ranks, a position-free field search also concentrated "
        "excitability near that direction",
    "AXIS_NOT_REQUIRED":
        "the frozen anisotropy constrains the propagation direction, but the "
        "pathology's position is not constrained to lie along it",
    "AXIS_INCONCLUSIVE":
        "the learned position is neither clearly on nor clearly off the axis at "
        "this resolution",
}


def _finite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def classify_stage3(results):
    """Full verdict. `results` must carry every key in REQUIRED plus integrity_ok.

    Returns primary_outcome together with position_stable, transfers_to_heldout,
    axis_relation and all_triggered_conditions, so no single label can hide the
    others (spec section 9.4; Stage 2's occlusion is the reason).
    """
    integrity = bool(results.get("integrity_ok", False))
    missing = [k for k in REQUIRED if not _finite(results.get(k))]

    def _num(k):
        return float(results[k]) if _finite(results.get(k)) else float("nan")

    r_bar, c2 = _num("r_bar"), _num("c_axis_2mm")
    sd, corr = _num("restart_r_bar_sd"), _num("restart_field_corr_median")
    train, held = _num("train_delta"), _num("heldout_delta")
    vs_onaxis = _num("heldout_delta_vs_onaxis")

    position_stable = (None if not (_finite(sd) and _finite(corr))
                       else bool(sd < R_BAR_SD_MAX_MM and corr >= FIELD_CORR_MIN))
    transfers = None if not _finite(held) else bool(held > 0)

    if not (_finite(r_bar) and _finite(c2)):
        axis_relation = "inconclusive"
    elif abs(r_bar) < NEAR_AXIS_MM and c2 >= NEAR_AXIS_MASS:
        axis_relation = "near"
    elif abs(r_bar) > OFF_AXIS_MM:
        axis_relation = "off"
    else:
        axis_relation = "inconclusive"

    triggered = []
    if not integrity or missing:
        triggered.append(0)
    if position_stable is False:
        triggered.append(1)
    if _finite(train) and _finite(held) and train > 0 and held <= 0:
        triggered.append(2)
    if axis_relation == "near":
        triggered.append(3)
    if axis_relation == "off" and _finite(vs_onaxis) and vs_onaxis >= 0:
        triggered.append(4)

    primary = OUTCOME_ORDER[min(triggered)] if triggered else "AXIS_INCONCLUSIVE"
    if not triggered:
        triggered = [5]

    return dict(
        primary_outcome=primary,
        position_stable=position_stable,
        transfers_to_heldout=transfers,
        axis_relation=axis_relation,
        all_triggered_conditions=sorted(triggered),
        allowed_statement=ALLOWED_STATEMENTS[primary],
        missing_fields=missing,
        measurements=dict(r_bar=r_bar, c_axis_2mm=c2, restart_r_bar_sd=sd,
                          restart_field_corr_median=corr, train_delta=train,
                          heldout_delta=held, heldout_delta_vs_onaxis=vs_onaxis),
        thresholds=dict(r_bar_sd_max_mm=R_BAR_SD_MAX_MM,
                        field_corr_min=FIELD_CORR_MIN,
                        near_axis_mm=NEAR_AXIS_MM, near_axis_mass=NEAR_AXIS_MASS,
                        off_axis_mm=OFF_AXIS_MM))
