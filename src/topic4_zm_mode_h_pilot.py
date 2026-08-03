"""Fail-closed adjudication for the small state-selective-H SNN pilot.

The pilot is a causal mechanism test, not a new parameter screen.  A positive
result requires the M-gated arm to leave the high-activity episode while its
matched no-M-gate arm remains active or runs away.  Returning interictal events
are reported separately from offset.
"""
from __future__ import annotations


VERSION = "topic4_zm_mode_h_pilot_v1_2026-08-02"


def _is_offset(row: dict) -> bool:
    return row.get("episode_status") == "onset_durable_offset"


def _is_persistent(row: dict) -> bool:
    return bool(row.get("runaway")) or (
        row.get("onset_ms") is not None and row.get("offset_ms") is None
    )


def adjudicate_mode_h_pilot(rows: dict[str, dict]) -> dict:
    """Classify the fixed-anchor H pilot without promoting offset to recovery."""
    # A negative result needs only the two actually engaged M-gated arms: if
    # neither exits, a no-M-gate control cannot turn it into a lifecycle.  A
    # positive causal-exit claim still requires the matched open-gate arm.
    required = {"baseline", "rho05_mc30", "rho1_mc30"}
    missing = sorted(required.difference(rows))
    if missing:
        return {"version": VERSION, "verdict": "NO_EVIDENCE", "missing": missing}

    h_peak = max(
        float(row.get("H_peak", 0.0) or 0.0)
        for key, row in rows.items() if key != "baseline"
    )
    if h_peak < 0.05:
        return {
            "version": VERSION,
            "verdict": "NO_GO_H_NOT_ENGAGED",
            "H_peak": h_peak,
        }

    causal_pairs = []
    pairs = []
    for strength, m_half, gated_key, open_key in (
        (0.5, 45.0, "rho05_gate", "rho05_nomgate"),
        (1.0, 45.0, "rho1_gate", "rho1_nomgate"),
        (0.5, 30.0, "rho05_mc30", "rho05_nomgate"),
        (1.0, 30.0, "rho1_mc30", "rho1_nomgate"),
    ):
        if gated_key in rows and open_key in rows:
            pairs.append((strength, m_half, gated_key, open_key))
    for strength, m_half, gated_key, open_key in pairs:
        gated, open_arm = rows[gated_key], rows[open_key]
        if _is_offset(gated) and _is_persistent(open_arm):
            causal_pairs.append({
                "rho_mode_H": strength,
                "m_mode_half": m_half,
                "gated_arm": gated_key,
                "no_M_gate_arm": open_key,
                "returning_event": bool(gated.get("returning_event")),
                "returning_distribution": bool(gated.get("returning_distribution")),
                "z_post_offset_recovery": gated.get("z_post_offset_recovery"),
            })
    if causal_pairs:
        best = max(
            causal_pairs,
            key=lambda row: (row["returning_distribution"], row["returning_event"]),
        )
        if best["returning_distribution"]:
            verdict = "PROVISIONAL_LIFECYCLE_CANDIDATE"
        elif best["returning_event"]:
            verdict = "M_GATED_EXIT_WITH_RETURNING_EVENT"
        else:
            verdict = "M_GATED_EXIT_WITHOUT_INTERICTAL_RETURN"
        return {
            "version": VERSION,
            "verdict": verdict,
            "causal_pairs": causal_pairs,
            "claim_boundary": (
                "seed-1 fixed-anchor candidate; healthy specificity, longer return, and locked seeds remain required"
            ),
        }

    open_control = {
        "rho05_gate": "rho05_nomgate", "rho05_mc30": "rho05_nomgate",
        "rho1_gate": "rho1_nomgate", "rho1_mc30": "rho1_nomgate",
    }
    observed_exit = [
        key for key, row in rows.items()
        if key in open_control and _is_offset(row) and open_control[key] not in rows
    ]
    if observed_exit:
        return {
            "version": VERSION,
            "verdict": "EXIT_OBSERVED_MATCHED_CONTROL_PENDING",
            "exit_arms": observed_exit,
            "claim_boundary": "offset is not causal until the matched no-M-gate arm persists",
        }

    baseline_gap = rows["baseline"].get("post_onset_deep_gap_fraction")
    bridged = []
    gated_keys = [
        key for key in ("rho05_gate", "rho1_gate", "rho05_mc30", "rho1_mc30")
        if key in rows
    ]
    for key in gated_keys:
        gap = rows[key].get("post_onset_deep_gap_fraction")
        if (
            baseline_gap is not None and gap is not None
            and float(baseline_gap) - float(gap) >= 0.10
            and not rows[key].get("runaway")
        ):
            bridged.append(key)
    contained = []
    comparison_pairs = [
        (gated, open_arm) for gated, open_arm in (
            ("rho05_gate", "rho05_nomgate"), ("rho1_gate", "rho1_nomgate"),
            ("rho05_mc30", "rho05_nomgate"), ("rho1_mc30", "rho1_nomgate")
        ) if gated in rows and open_arm in rows
    ]
    for gated_key, open_key in comparison_pairs:
        if not rows[gated_key].get("runaway") and rows[open_key].get("runaway"):
            contained.append(gated_key)
    if contained:
        verdict = "M_GATE_CONTAINS_H_BUT_NO_EXIT"
    elif bridged:
        verdict = "H_BRIDGES_GAPS_BUT_NO_EXIT"
    elif any(row.get("runaway") for key, row in rows.items() if key != "baseline"):
        verdict = "H_OVERAMPLIFIES_EXISTING_HIGH_BRANCH"
    else:
        verdict = "NO_LIFECYCLE_DIRECTION"
    return {
        "version": VERSION,
        "verdict": verdict,
        "gap_bridging_arms": bridged,
        "contained_arms": contained,
        "claim_boundary": "no causal durable offset with returning interictal dynamics",
    }
