#!/usr/bin/env python
"""Topic 5 V3p preictal trajectory -- summary + tier verdict (Task 9, integration).

Plain-language question: across the two preictal co-primary legs (H3p-b
non-axial flux amplification, H3p-c dominant-mode shift toward non-axis),
how many subjects show a rising-toward-onset trend that survives EVERY
robustness check already computed by the trajectory runner (onset jitter,
leave-one-contact, axis-only control, near-onset dependence, label-null
power), and does the COHORT as a whole shift in the expected direction once
the two legs' p-values are jointly (Holm) corrected for having been
pre-registered together? H3p-a (axial weakening) is context only: it can
only STRENGTHEN a subject's story, it never creates ``subject_support`` on
its own.

This script does NOT run any new permutation nulls -- it reads the flags,
slopes, and p-values already written by ``scripts/run_topic5_v3p_trajectory.py``
(Tasks 7+8) into ``v3p_trajectory_subject.csv`` and derives the summary from
them. ``tier`` (0-4; 5 is model-side, out of scope) is assigned ONLY here --
no earlier task computes it, and no per-subject CSV (this one or the
trajectory one) carries a ``tier`` column; ``tier`` lives only in
``v3p_cohort_tier.json``.

narrow (7 subjects) is PRIMARY; broad is REPLICATION and is reported at TWO
levels per the rev2 cohort-expansion decision (spec Sec 10): ``broad`` (=
``broad_expanded``, the curated core of 9 plus admitted epilepsiae
candidates) and ``broad_core`` (the same 9 curated subjects only, sliced out
of the SAME CSV via the ``in_broad_core`` column). narrow and broad are
NEVER pooled into one statistic. tier 4 requires the direction to replicate
on BOTH ``broad_expanded`` AND ``broad_core`` -- expanding the roster is only
allowed to ADD power, never to manufacture a replication that the originally
curated 9 do not themselves show. ``tier_broad_core`` reports what the tier
ladder would say if broad's replication arm were the curated 9 alone, so a
reader can see directly whether the expansion changed the verdict.

See docs/superpowers/specs/2026-07-03-topic5-v3p-preictal-trajectory-design.md
Sec 8 (statistics + tier) and Sec 10 (cohort expansion, rev2).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.topic5_v3p_preictal_trajectory import load_v3p_config  # noqa: E402

_RESULTS_BASE = _ROOT / "results/topic5_ictal_recruitment/v3p_preictal_trajectory"

SUMMARY_COLS = [
    "subject", "cohort", "in_broad_core", "status", "geometry_sufficient",
    "excluded_from_denominator",
    "net_offaxis_flux_surplus_slope", "net_offaxis_flux_slope_z",
    "mode_shift_density_surplus_slope", "mode_shift_density_slope_z",
    "nonaxis_flux_amplification_supported", "mode_transition_supported",
    "axis_weakening_supportive",
    "subject_support", "support_driver",
]


# ---------------------------------------------------------------------------
# CSV loading + per-subject gate stack (string-valued DictReader rows -> typed)
# ---------------------------------------------------------------------------
def _to_bool(s) -> bool:
    return str(s) == "True"


def _to_float(s) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _load_trajectory_rows(csv_path: Path) -> list[dict] | None:
    """Raw ``v3p_trajectory_subject.csv`` rows for one cohort, or ``None`` if
    the file is absent (a pipeline-staging gap, NOT "0 subjects qualify")."""
    if not csv_path.exists():
        return None
    with csv_path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _subject_row(raw: dict, alpha: float) -> dict:
    """One ``v3p_summary_subject.csv`` row: the ``subject_support`` gate
    stack for EACH leg using that leg's OWN robustness columns (brief Sec
    "Core logic"), joined by OR -- H3p-a never enters this OR, it only sets
    the separate ``axis_weakening_supportive`` context flag.

    ``single_contact_driven`` only exists for the mode-shift (H3p-c) leg in
    the trajectory CSV (there is no flux-leg analogue of "one contact's
    eigenvector weight dominates"), so it gates ONLY ``mode_transition_supported``.
    ``label_null_underpowered`` is a subject-level (not leg-level) QC column
    and gates BOTH legs.
    """
    status = raw["status"]
    excluded = bool(status != "ok")
    label_null_underpowered = _to_bool(raw["label_null_underpowered"])

    h3pb_path = bool(
        _to_bool(raw["module_support_flag_b"])
        and _to_bool(raw["onset_jitter_pass_b"])
        and _to_bool(raw["leave_one_contact_flux_pass"])
        and _to_bool(raw["axis_only_flux_control_pass"])
        and not _to_bool(raw["near_onset_dependent_b"])
        and not label_null_underpowered
    )
    h3pc_path = bool(
        _to_bool(raw["module_support_flag_c"])
        and _to_bool(raw["onset_jitter_pass_c"])
        and not _to_bool(raw["single_contact_driven"])
        and _to_bool(raw["leave_one_contact_mode_pass"])
        and _to_bool(raw["axis_only_mode_control_pass"])
        and not _to_bool(raw["near_onset_dependent_c"])
        and not label_null_underpowered
    )
    subject_support = bool((h3pb_path or h3pc_path) and not excluded)

    if excluded:
        support_driver = ""
    elif h3pb_path and h3pc_path:
        support_driver = "H3p-b+H3p-c"
    elif h3pb_path:
        support_driver = "H3p-b"
    elif h3pc_path:
        support_driver = "H3p-c"
    else:
        support_driver = "none"

    # H3p-a "strengthens, never sole" (spec Sec 2/Sec 11): direction correct
    # (slope < 0), a reliable |beta_axis| baseline, AND its OWN label-null
    # significant -- never feeds subject_support above.
    beta_axis_strength_slope = _to_float(raw["beta_axis_strength_slope"])
    p_label_slope_a = _to_float(raw["p_label_slope_a"])
    axis_weakening_supportive = bool(
        np.isfinite(beta_axis_strength_slope) and beta_axis_strength_slope < 0
        and _to_bool(raw["beta_axis_reliable"])
        and np.isfinite(p_label_slope_a) and p_label_slope_a < alpha
    )

    return {
        "subject": raw["subject"], "cohort": raw["cohort"],
        "in_broad_core": _to_bool(raw["in_broad_core"]),
        "status": status, "geometry_sufficient": _to_bool(raw["geometry_sufficient"]),
        "excluded_from_denominator": excluded,
        "net_offaxis_flux_surplus_slope": _to_float(raw["net_offaxis_flux_surplus_slope"]),
        "net_offaxis_flux_slope_z": _to_float(raw["net_offaxis_flux_slope_z"]),
        "mode_shift_density_surplus_slope": _to_float(raw["mode_shift_density_surplus_slope"]),
        "mode_shift_density_slope_z": _to_float(raw["mode_shift_density_slope_z"]),
        "nonaxis_flux_amplification_supported": h3pb_path,
        "mode_transition_supported": h3pc_path,
        "axis_weakening_supportive": axis_weakening_supportive,
        "subject_support": subject_support, "support_driver": support_driver,
    }


# ---------------------------------------------------------------------------
# cohort-level stats: Holm-corrected co-primary Wilcoxon on subject z-values
# ---------------------------------------------------------------------------
def _wilcoxon_greater(values: np.ndarray) -> float:
    """One-sided-greater Wilcoxon signed-rank p on subject-level ``slope_label_z``.

    Guards scipy's raise-on-degenerate-input (all differences zero) and an
    underpowered draw (<2 finite values, or 0 non-zero values) to a NaN
    return instead of crashing the whole cohort summary.
    """
    finite = values[np.isfinite(values)]
    nonzero = finite[finite != 0]
    if finite.size < 2 or nonzero.size < 1:
        return float("nan")
    try:
        _, p = wilcoxon(finite, alternative="greater")
    except ValueError:
        return float("nan")
    return float(p)


def _holm_correct_2(p1: float, p2: float) -> tuple[float, float]:
    """Holm step-down correction for the 2 pre-registered co-primary legs
    (H3p-b, H3p-c; ``config/topic5_v3p.yaml: co_primary.correction: holm``).

    ``m=2`` always, regardless of how many of the two raw p-values happen to
    be finite this run. Sort ascending -- NaN sorts last (numpy convention:
    "worst"); the smaller finite p is multiplied by 2, the larger by 1; then
    enforce non-decreasing order (standard Holm step-down running max). A
    NaN input stays NaN in its own slot and never receives a multiplier, but
    does not change the OTHER, finite leg's multiplier.
    """
    vals = np.array([p1, p2], dtype=float)
    order = np.argsort(vals)
    holm = np.full(2, np.nan)
    running_max = -np.inf
    for rank, idx in enumerate(order):
        p = vals[idx]
        if not np.isfinite(p):
            continue
        adjusted = min((2 - rank) * p, 1.0)
        running_max = max(running_max, adjusted)
        holm[idx] = running_max
    return float(holm[0]), float(holm[1])


_UNAVAILABLE_BLOCK = {
    "available": False,
    "n_total": 0, "n_eligible": 0, "n_excluded": 0, "n_subject_support": 0,
    "p_wilcoxon_b": float("nan"), "p_wilcoxon_c": float("nan"),
    "p_holm_b": float("nan"), "p_holm_c": float("nan"),
    "median_slope_z_b": float("nan"), "median_slope_z_c": float("nan"),
    "cohort_b_pass": False, "cohort_c_pass": False, "cohort_pass": False,
}


def _cohort_block(rows: list[dict], alpha: float) -> dict:
    """One cohort's (narrow / broad_expanded / broad_core) aggregate stats.

    ``rows`` are ALREADY-typed ``_subject_row`` dicts. The denominator for
    both ``n_subject_support`` and the Wilcoxon population is
    ``n_eligible`` = subjects NOT excluded (``status=='ok'``, i.e. neither
    ``geometry_insufficient`` nor feasibility-insufficient nor a compute
    error) -- brief: "geometry_insufficient/status==skipped subjects
    EXCLUDED from the denominator."
    """
    included = [r for r in rows if not r["excluded_from_denominator"]]
    n_total = len(rows)
    n_eligible = len(included)
    n_support = sum(1 for r in included if r["subject_support"])

    z_b = np.array([r["net_offaxis_flux_slope_z"] for r in included], dtype=float)
    z_c = np.array([r["mode_shift_density_slope_z"] for r in included], dtype=float)
    z_b = z_b[np.isfinite(z_b)]
    z_c = z_c[np.isfinite(z_c)]

    p_wilcoxon_b = _wilcoxon_greater(z_b)
    p_wilcoxon_c = _wilcoxon_greater(z_c)
    p_holm_b, p_holm_c = _holm_correct_2(p_wilcoxon_b, p_wilcoxon_c)

    med_b = float(np.median(z_b)) if z_b.size else float("nan")
    med_c = float(np.median(z_c)) if z_c.size else float("nan")

    cohort_b_pass = bool(np.isfinite(p_holm_b) and p_holm_b < alpha and med_b > 0)
    cohort_c_pass = bool(np.isfinite(p_holm_c) and p_holm_c < alpha and med_c > 0)

    return {
        "available": True,
        "n_total": n_total, "n_eligible": n_eligible, "n_excluded": n_total - n_eligible,
        "n_subject_support": n_support,
        "p_wilcoxon_b": p_wilcoxon_b, "p_wilcoxon_c": p_wilcoxon_c,
        "p_holm_b": p_holm_b, "p_holm_c": p_holm_c,
        "median_slope_z_b": med_b, "median_slope_z_c": med_c,
        "cohort_b_pass": cohort_b_pass, "cohort_c_pass": cohort_c_pass,
        "cohort_pass": bool(cohort_b_pass or cohort_c_pass),
    }


# ---------------------------------------------------------------------------
# tier verdict (narrow primary; broad_expanded + broad_core NEVER pooled)
# ---------------------------------------------------------------------------
def _replication_check(narrow_block: dict, other_block: dict) -> bool:
    """Does ``other_block`` replicate narrow's passing leg(s), in the SAME
    direction? Requires narrow itself to already cohort-pass (an
    unreplicated narrow has nothing to replicate) AND the SAME leg (b or c)
    to ALSO cohort-pass in ``other_block`` -- replicating via the OTHER leg
    is not replication of the finding narrow actually made.
    """
    narrow_cohort_pass = bool(narrow_block["cohort_b_pass"] or narrow_block["cohort_c_pass"])
    leg_keys = (("b", "cohort_b_pass"), ("c", "cohort_c_pass"))
    narrow_legs = {leg for leg, key in leg_keys if narrow_block[key]}
    other_legs = {leg for leg, key in leg_keys if other_block[key]}
    return bool(narrow_cohort_pass and (narrow_legs & other_legs))


def _ladder(narrow_tier3_met: bool, replicates: bool, n_support: int, direction_correct: bool) -> int:
    """The shared 0-4 tier ladder (spec Sec 8), parameterized only by WHICH
    "does it replicate" boolean is supplied -- this is what lets ``tier``
    (gated on broad_expanded AND broad_core) and ``tier_broad_core`` (gated
    on broad_core alone) share one implementation without duplicating the
    0/1/2/3 rungs.
    """
    if narrow_tier3_met and replicates:
        return 4
    if narrow_tier3_met:
        return 3
    if n_support >= 1:
        return 2
    if direction_correct:
        return 1
    return 0


def _tier_verdict(
    narrow_block: dict, broad_block: dict, broad_core_block: dict, min_subject_support_narrow: int,
) -> dict:
    """tier 0-4 (5 is model-side, out of this script's scope).

    narrow tier-3 needs BOTH a Holm-passed leg (``cohort_b_pass`` or
    ``cohort_c_pass``) AND >= ``min_subject_support_narrow`` subjects that
    individually clear the FULL per-subject gate stack (spec Sec 8) -- a
    cohort-level Wilcoxon shift alone is not enough without individually
    robust subjects behind it, and vice versa (subject_support count alone,
    below the Holm bar, caps at tier 2 -- see the tier-2 rung below).

    tier 4 requires the direction to hold on BOTH ``broad_expanded`` and
    ``broad_core`` (rev2): expanding the roster only adds power, it must
    never manufacture a replication the curated 9 do not themselves show.
    ``tier_broad_core`` is the SAME ladder computed with broad_core as the
    sole replication arm, reported separately so a reader can see directly
    whether the expansion changed the verdict.
    """
    narrow_cohort_pass = bool(narrow_block["cohort_b_pass"] or narrow_block["cohort_c_pass"])
    narrow_support_ok = bool(narrow_block["n_subject_support"] >= min_subject_support_narrow)
    narrow_tier3_met = bool(narrow_cohort_pass and narrow_support_ok)

    broad_expanded_replicates = _replication_check(narrow_block, broad_block)
    broad_core_replicates = _replication_check(narrow_block, broad_core_block)

    narrow_direction_correct = bool(
        (np.isfinite(narrow_block["median_slope_z_b"]) and narrow_block["median_slope_z_b"] > 0)
        or (np.isfinite(narrow_block["median_slope_z_c"]) and narrow_block["median_slope_z_c"] > 0)
    )
    n_support = narrow_block["n_subject_support"]

    tier = _ladder(
        narrow_tier3_met, bool(broad_expanded_replicates and broad_core_replicates),
        n_support, narrow_direction_correct,
    )
    tier_broad_core = _ladder(narrow_tier3_met, broad_core_replicates, n_support, narrow_direction_correct)

    return {
        "tier": tier, "tier_broad_core": tier_broad_core,
        "state_v3p_supported": bool(tier >= 3),
        "pre_registered_negative": bool(tier <= 1),
        "narrow_cohort_pass": narrow_cohort_pass, "narrow_tier3_met": narrow_tier3_met,
        "broad_expanded_replicates": broad_expanded_replicates,
        "broad_core_replicates": broad_core_replicates,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["narrow", "broad"], default=None,
                     help="default: both narrow and broad")
    ap.add_argument("--indir", default=None,
                     help="parent dir containing narrow/ and broad/ subdirs with "
                          "v3p_trajectory_subject.csv; default: "
                          "results/topic5_ictal_recruitment/v3p_preictal_trajectory")
    ap.add_argument("--outdir", default=None,
                     help="default: <indir>/<cohort>; requires --cohort when processing "
                          "both (2 cohorts can't share 1 outdir)")
    args = ap.parse_args(argv)

    if args.outdir and args.cohort is None:
        ap.error("--outdir requires --cohort (default both-cohort mode writes 2 separate directories)")

    requested = [args.cohort] if args.cohort else ["narrow", "broad"]
    indir_base = Path(args.indir) if args.indir else _RESULTS_BASE

    v3pcfg = load_v3p_config()
    alpha = float(v3pcfg["nulls"]["alpha"])
    min_support_narrow = int(v3pcfg["summary"]["min_subject_support_narrow"])

    # ALWAYS load both cohorts (tier needs both regardless of which cohort's
    # own CSV/JSON gets written this invocation); a cohort not in `requested`
    # tolerates a missing file (degrades to the unavailable block) but a
    # REQUESTED cohort missing its file raises loudly.
    raw_by_cohort: dict[str, list[dict] | None] = {}
    subject_rows: dict[str, list[dict]] = {}
    blocks: dict[str, dict] = {}
    for c in ("narrow", "broad"):
        raw = _load_trajectory_rows(indir_base / c / "v3p_trajectory_subject.csv")
        raw_by_cohort[c] = raw
        if raw is None:
            if c in requested:
                raise FileNotFoundError(
                    f"cohort {c!r}: missing v3p_trajectory_subject.csv under {indir_base / c} -- "
                    f"run scripts/run_topic5_v3p_trajectory.py --cohort {c} first"
                )
            subject_rows[c] = []
            blocks[c] = dict(_UNAVAILABLE_BLOCK)
            continue
        typed = [_subject_row(r, alpha) for r in raw]
        subject_rows[c] = typed
        blocks[c] = _cohort_block(typed, alpha)

    # broad_core: the SAME broad rows, sliced by `in_broad_core` (spec Sec 10
    # rev2) -- never its own CLI cohort, never its own CSV, always derived.
    if raw_by_cohort["broad"] is None:
        blocks["broad_core"] = dict(_UNAVAILABLE_BLOCK)
    else:
        core_rows = [r for r in subject_rows["broad"] if r["in_broad_core"]]
        blocks["broad_core"] = _cohort_block(core_rows, alpha)

    verdict = _tier_verdict(blocks["narrow"], blocks["broad"], blocks["broad_core"], min_support_narrow)
    payload = {
        "tier": verdict["tier"], "tier_broad_core": verdict["tier_broad_core"],
        "state_v3p_supported": verdict["state_v3p_supported"],
        "pre_registered_negative": verdict["pre_registered_negative"],
        "alpha": alpha, "min_subject_support_narrow": min_support_narrow,
        "narrow_cohort_pass": verdict["narrow_cohort_pass"],
        "narrow_tier3_met": verdict["narrow_tier3_met"],
        "broad_expanded_replicates": verdict["broad_expanded_replicates"],
        "broad_core_replicates": verdict["broad_core_replicates"],
        "narrow": blocks["narrow"], "broad": blocks["broad"], "broad_core": blocks["broad_core"],
    }

    for c in requested:
        outdir = Path(args.outdir) if args.outdir else indir_base / c
        outdir.mkdir(parents=True, exist_ok=True)

        out_csv = outdir / "v3p_summary_subject.csv"
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SUMMARY_COLS)
            w.writeheader()
            w.writerows(subject_rows[c])

        out_json = outdir / "v3p_cohort_tier.json"
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

        print(
            f"[done] {c}: {len(subject_rows[c])} subjects -> {out_csv} "
            f"(n_subject_support={blocks[c]['n_subject_support']}/{blocks[c]['n_eligible']}) "
            f"-> {out_json} (tier={verdict['tier']} tier_broad_core={verdict['tier_broad_core']} "
            f"state_v3p_supported={verdict['state_v3p_supported']})",
            flush=True,
        )


if __name__ == "__main__":
    main()
