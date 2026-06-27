#!/usr/bin/env python3
"""M3 Task 4: local-W pre-registration freeze.

SKELETON — DO NOT freeze until BOTH prerequisites are met:
  1. kick_calibration.json exists (Task 1.5 calibration run completed).
  2. per-subject AF/LR medians are available from Task-0 audit.

When run prematurely this script exits LOUDLY (per CLAUDE.md §6: loud failure
beats silent contamination).  It will NOT hardcode the tolerance band and will
NOT fabricate per-subject median values.

Once both prerequisites are met, this script reads:
  - results/topic4_sef_hfo/m3_local_w/kick_calibration/kick_calibration.json
      → calibrated kick_boost + [Δ1, Δ2]
  - per-subject AF/LR medians array (n≈23 subjects, one value per subject)
      → passed via --af-medians-json or produced by a future Task-0 re-run
  then writes:
  - results/topic4_sef_hfo/m3_local_w/preregistration.json

Fields in preregistration.json (spec §6.3 + plan Task 4)
---------------------------------------------------------
  h_scheme_plan          : {primary, secondary, negative_controls}
  A_p_primary            : "E_spike_count" (per spec §5.2)
  calibrated_kick_boost  : float [from kick_calibration.json]
  calibrated_win_ms      : [Δ1, Δ2] [from kick_calibration.json]
  calibration_rationale  : str
  layer2_tolerance_band  : {af_lo, af_hi, lr_lo, lr_hi}  [from subject_tolerance_band()]
  r0_r4_thresholds       : {z, T_quiet_ms, T_sustain_ms, T_gap_ms,
                             early_recruitment_pct, axis_alignment_angle_deg}
  preregistration_date   : ISO date
  notes                  : list of DRAFT warnings

Usage (once kick_calibration prerequisite is met)
------------------------------------
  python3 scripts/run_m3_localw_preregistration.py \\
      --calib-json  results/topic4_sef_hfo/m3_local_w/kick_calibration/kick_calibration.json \\
      --out-dir  results/topic4_sef_hfo/m3_local_w

--af-medians-json defaults to results/topic4_sef_hfo/event_extent_audit/per_subject_extent_medians.json
(produced by scripts/dump_event_extent_per_subject_medians.py).  The file must contain
top-level keys "per_subject_af_medians" and "per_subject_lr_medians" as lists of floats
(one per subject, n≈23).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# DRAFT constants for R0–R4 numeric thresholds (spec §6.3)
# These are labeled DRAFT because they must be reviewed at preregistration time.
# Change them only at preregistration, before any SNN dynamics results are seen.
# ---------------------------------------------------------------------------

# return-to-baseline gate (R0–R2 / R3 / R4 boundary)
_DRAFT_Z = 2.0              # A(t) < μ_A + z·σ_A to count as "quiet"
_DRAFT_T_QUIET_MS = 300.0   # duration the network must stay quiet (ms) after event

# sustained recruitment gate (R4 boundary)
_DRAFT_T_SUSTAIN_MS = 500.0  # A(t) > threshold for this long → sustained

# repeated-wave gate (R4a boundary)
_DRAFT_T_GAP_MS = 200.0      # inter-wave interval < this → repeated-wave regime

# R4a (W-aligned) vs R4b (tonic runaway) discrimination
_DRAFT_EARLY_RECRUITMENT_PCT = 20.0   # first X% of event duration defines "early phase"
_DRAFT_AXIS_ALIGNMENT_ANGLE_DEG = 30.0  # early-phase axis must be within this of W_shape axis


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="M3 Task 4: local-W pre-registration freeze (scaffold)"
    )
    p.add_argument(
        "--calib-json",
        default="results/topic4_sef_hfo/m3_local_w/kick_calibration/kick_calibration.json",
        help="Path to kick_calibration.json from Task 1.5",
    )
    p.add_argument(
        "--af-medians-json",
        default="results/topic4_sef_hfo/event_extent_audit/per_subject_extent_medians.json",
        help=(
            "Path to JSON with per-subject AF/LR medians "
            "(keys: per_subject_af_medians, per_subject_lr_medians). "
            "Default: per_subject_extent_medians.json produced by "
            "scripts/dump_event_extent_per_subject_medians.py."
        ),
    )
    p.add_argument(
        "--band-q-lo", type=float, default=10.0,
        help="Lower percentile for Layer-2 tolerance band (default 10th)"
    )
    p.add_argument(
        "--band-q-hi", type=float, default=90.0,
        help="Upper percentile for Layer-2 tolerance band (default 90th)"
    )
    p.add_argument(
        "--out-dir",
        default="results/topic4_sef_hfo/m3_local_w",
        help="Output directory; preregistration.json written here",
    )
    return p


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def _load_calibration(calib_path: str) -> dict:
    calib_abs = _resolve_path(calib_path)
    if not os.path.exists(calib_abs):
        print(
            f"\n[PENDING] kick_calibration.json not found at:\n  {calib_abs}\n"
            "Task 1.5 calibration sweep must be completed before preregistration.\n"
            "Run scripts/run_m3_kick_calibration.py first."
        )
        sys.exit(1)
    with open(calib_abs, encoding="utf-8") as f:
        return json.load(f)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_per_subject_medians(af_json_path: str) -> tuple[list[float], list[float], dict]:
    """Load per-subject AF/LR medians from per_subject_extent_medians.json.

    The file is produced by scripts/dump_event_extent_per_subject_medians.py and
    contains per-SUBJECT observed median axial fraction / lateral ratio (one value
    per subject, n≈23).  Must NOT be cohort_summary.json — that file stores only
    cohort-level medians plus per-EVENT arrays.
    """
    abs_af = _resolve_path(af_json_path)

    # Guard: detect accidental pass of cohort_summary.json
    cohort_summary_path = _resolve_path(
        "results/topic4_sef_hfo/event_extent_audit/cohort_summary.json"
    )
    if os.path.abspath(abs_af) == os.path.abspath(cohort_summary_path):
        print(
            "\n[BLOCKED] --af-medians-json points to cohort_summary.json.\n"
            "That file contains COHORT-LEVEL medians and per-EVENT arrays — NOT "
            "per-SUBJECT medians.  Use per_subject_extent_medians.json instead\n"
            "(produced by scripts/dump_event_extent_per_subject_medians.py)."
        )
        sys.exit(1)

    if not os.path.exists(abs_af):
        print(
            f"\n[PENDING] af-medians-json file not found at:\n  {abs_af}\n"
            "Run scripts/dump_event_extent_per_subject_medians.py first."
        )
        sys.exit(1)

    with open(abs_af, encoding="utf-8") as f:
        data = json.load(f)

    required_keys = {"per_subject_af_medians", "per_subject_lr_medians"}
    missing = required_keys - set(data.keys())
    if missing:
        print(
            f"\n[BLOCKED] af-medians-json at {abs_af} is missing required keys: {missing}\n"
            "Expected keys: per_subject_af_medians (list of floats, n≈23) and "
            "per_subject_lr_medians (list of floats, n≈23)."
        )
        sys.exit(1)

    af_medians = list(data["per_subject_af_medians"])
    lr_medians = list(data["per_subject_lr_medians"])

    if len(af_medians) < 5:
        print(
            f"\n[BLOCKED] per_subject_af_medians has only {len(af_medians)} entries. "
            "Expected n≈23. Refusing to compute a tolerance band from too few subjects."
        )
        sys.exit(1)
    if len(af_medians) != len(lr_medians):
        print(
            f"\n[BLOCKED] per_subject_af_medians (n={len(af_medians)}) and "
            f"per_subject_lr_medians (n={len(lr_medians)}) have different lengths."
        )
        sys.exit(1)

    # Provenance verification (review P1, 2026-06-22): the band's 10/90 percentiles come from
    # this sidecar's per-subject arrays; prove they belong to the SAME audit run as the
    # cohort_summary.json this prereg freezes against, by matching the recorded cohort hash.
    prov = data.get("provenance")
    if not prov or "cohort_summary_sha256" not in prov:
        print(
            f"\n[BLOCKED] {abs_af} has no provenance block (cohort_summary_sha256).\n"
            "Re-run scripts/dump_event_extent_per_subject_medians.py (it now writes a "
            "same-source proof + sha256 provenance) before freezing."
        )
        sys.exit(1)
    if os.path.exists(cohort_summary_path):
        cur_sha = _sha256(cohort_summary_path)
        if cur_sha != prov["cohort_summary_sha256"]:
            print(
                "\n[BLOCKED] cohort_summary.json changed since the sidecar was built:\n"
                f"  sidecar recorded : {prov['cohort_summary_sha256']}\n"
                f"  current file     : {cur_sha}\n"
                "The Layer-2 band would no longer be from the same audit as the cohort verdict. "
                "Re-run scripts/dump_event_extent_per_subject_medians.py against the current "
                "cohort_summary.json."
            )
            sys.exit(1)

    return af_medians, lr_medians, prov


def _compute_tolerance_band(af_medians: list[float], lr_medians: list[float],
                             q_lo: float, q_hi: float) -> dict:
    """Thin wrapper around subject_tolerance_band from src/topic4_m3_acceptance.py."""
    from src.topic4_m3_acceptance import subject_tolerance_band
    band = subject_tolerance_band(af_medians, lr_medians, q=(q_lo, q_hi))
    band["n_subjects"] = len(af_medians)
    band["q_lo"] = q_lo
    band["q_hi"] = q_hi
    return band


def _build_preregistration(calib: dict, band: dict,
                             af_medians: list[float], lr_medians: list[float],
                             layer2_provenance: dict,
                             args: argparse.Namespace) -> dict:
    """Assemble the preregistration dict. No fabrication allowed."""
    return {
        "preregistration_date": date.today().isoformat(),
        "status": "FROZEN",
        "h_scheme_plan": {
            "primary": "h_post",
            "secondary": "h_hybrid",
            "negative_controls": ["uniform_h", "shuffled_h"],
            "note": (
                "h_post = norm(row-sum of W_resp) — target-bin recruitability. "
                "Must NOT be computed from a row-normalized W; W_resp must remain unnormalized. "
                "h_hybrid = 0.5*(h_post + h_out). "
                "uniform_h: h_i=1 for all bins (global excitability control, C5). "
                "shuffled_h: h_i randomly permuted across bins (spatial structure control, C5)."
            ),
        },
        "A_p_primary": {
            "metric": "E_spike_count",
            "definition": (
                "Number of E-cell spikes in bin p within [t_kick + Δ1, t_kick + Δ2]. "
                "W_resp[p,q] = clip(mean_seed(A_p|kick@q) - mean_seed(A_p|sham), 0, inf). "
                "Diagonal zeroed. W_resp NOT row-normalized."
            ),
        },
        "calibrated_kick_boost": calib["calibrated_kick_boost"],
        "calibrated_win_ms": calib["calibrated_win_ms"],
        "calibration_rationale": calib.get("rationale", ""),
        "calibration_sweep_parameters": calib.get("sweep_parameters", {}),
        "layer2_tolerance_band": band,
        "layer2_provenance": layer2_provenance,
        "layer2_note": (
            "Band computed via subject_tolerance_band(ref_per_subject_af, ref_per_subject_lr, "
            f"q=({band['q_lo']}, {band['q_hi']})) on n={band['n_subjects']} real subjects. "
            "Layer-2 PASS = model per-subject AF median in [af_lo, af_hi] AND "
            "LR median in [lr_lo, lr_hi] AND AF >= min_af (see layer2_equivalence). "
            "NOT a 'not-rejected p-value' gate."
        ),
        "r0_r4_thresholds": {
            "z": _DRAFT_Z,
            "T_quiet_ms": _DRAFT_T_QUIET_MS,
            "T_sustain_ms": _DRAFT_T_SUSTAIN_MS,
            "T_gap_ms": _DRAFT_T_GAP_MS,
            "early_recruitment_pct": _DRAFT_EARLY_RECRUITMENT_PCT,
            "axis_alignment_angle_deg": _DRAFT_AXIS_ALIGNMENT_ANGLE_DEG,
            "definitions": {
                "return_to_baseline": (
                    "Event classified as time-limited (R0–R3) if A(t) < mu_A + z*sigma_A "
                    "continuously for T_quiet_ms after event peak. "
                    "mu_A, sigma_A from inter-event baseline windows."
                ),
                "sustained_recruitment": (
                    "Event classified as sustained (R4 candidate) if A(t) > mu_A + z*sigma_A "
                    "for > T_sustain_ms, OR no quiet interval detected, "
                    "OR repeated waves with inter-wave gap < T_gap_ms."
                ),
                "R4a_vs_R4b": (
                    "R4a (W-aligned sustained): early recruitment axis theta_early "
                    "(computed from the first early_recruitment_pct% of recruited bins) "
                    "aligns with W_shape principal axis within axis_alignment_angle_deg. "
                    "R4b (tonic runaway): sustained but no propagation structure "
                    "(theta_early does NOT align with W_shape axis). "
                    "Only R4a supports the 'same-W supercritical expansion' claim (spec C3/C4)."
                ),
            },
            "DRAFT_warning": (
                "All numeric values above are DRAFT defaults from spec §6.3. "
                "They must be reviewed and confirmed at preregistration time "
                "(before any SNN dynamics results are examined). "
                "Do not adjust them after seeing results."
            ),
        },
        "conditioned_on_ignition_note": (
            "Primary propagation analysis uses fixed finite-pulse to condition on ignition "
            "(same onset patch + same kick per mu value). "
            "Secondary: spontaneous events with matched initial active mass (10–20ms). "
            "See spec §6.3 condition-on-ignition clause."
        ),
        "claim_discipline": {
            "C1": "Criticality = rho(W_step) ≈ 1 (recruitment operator spectral radius). NOT resting-state max Re lambda.",
            "C3": "Self-limitation = temporal (return to baseline). NOT spatial containment. Axial reach is descriptive only.",
            "C4": "Ictal simulation = synthetic feasibility bridge only. Does NOT explain clinical seizure onset.",
            "C5": "mu must couple via h(W). Uniform-mu and shuffled-h are controls, not mechanisms.",
            "C6": "W_kicked decomposed into three non-interchangeable objects: W_resp (unnormalized → h), W_step (src-normalized → Lambda_0 = rho), W_shape (row-normalized → axis).",
        },
        "w_step_valid_src_rule": (
            "The W_step sensitivity口径 (injected_mass normalizer) uses the SAME valid_src "
            "mask as the main口径 (src_mass normalizer). Specifically: "
            "make_step_operator(W_resp, src_mass, injected_mass=injected_mass, "
            "src_mass_floor=...) always excludes columns where src_mass[q] < src_mass_floor, "
            "even when injected_mass is the denominator. "
            "Rationale: a source bin whose measured kick response (src_mass) is below the floor "
            "is an unreliable propagation source regardless of which normalizer is used; "
            "excluding it is a data-quality filter on the source, not a numerical-blowup guard. "
            "Using the same valid_src mask for both口径 ensures the main-vs-sensitivity "
            "comparison varies ONLY the normalizer (src_mass vs injected_mass), "
            "not which source bins are included. "
            "Do NOT relax the src_mass_floor exclusion under the injected_mass sensitivity path."
        ),
    }


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    # Step 1: load calibration (exits loudly if missing)
    calib = _load_calibration(args.calib_json)

    # Step 2: load per-subject AF/LR medians + same-source provenance (exits loudly otherwise)
    af_medians, lr_medians, layer2_prov = _load_per_subject_medians(args.af_medians_json)

    # Step 3: compute tolerance band via the canonical helper
    band = _compute_tolerance_band(
        af_medians, lr_medians,
        q_lo=args.band_q_lo, q_hi=args.band_q_hi,
    )

    # Step 4: assemble and write preregistration.json
    prereg = _build_preregistration(calib, band, af_medians, lr_medians, layer2_prov, args)

    out_dir_abs = _resolve_path(args.out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)
    out_path = os.path.join(out_dir_abs, "preregistration.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prereg, f, indent=2, ensure_ascii=False)

    print(f"[OK] preregistration.json frozen → {out_path}")
    print(f"     kick_boost = {prereg['calibrated_kick_boost']}")
    print(f"     win_ms     = {prereg['calibrated_win_ms']}")
    print(f"     Layer-2 band (AF): [{band['af_lo']:.3f}, {band['af_hi']:.3f}]")
    print(f"     Layer-2 band (LR): [{band['lr_lo']:.3f}, {band['lr_hi']:.3f}]")
    print(f"     n_subjects = {band['n_subjects']}")
    print(
        "\nWARNING: preregistration.json is now frozen. "
        "Do NOT modify R0–R4 thresholds or h_scheme after this point."
    )


if __name__ == "__main__":
    main()
