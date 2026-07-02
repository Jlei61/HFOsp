#!/usr/bin/env python
"""Topic 5 V3a mode-transition — summary + tier verdict (Task 10, integration).

Plain-language question: across the two co-primary endpoints (H3b avalanche
compartment flux, H3c dynamics mode-shift), how many subjects actually show
axial-weakening-into-seizure evidence that survives EVERY robustness check
built for that endpoint (onset jitter, leave-one-contact, axis-only control,
and — for H3b only — the common-drive control), and does the COHORT as a
whole shift in the expected direction once the two endpoints' p-values are
jointly (Holm) corrected for having been pre-registered together? H3a
(susceptibility) is context only: it never creates support on its own, it
can only strengthen the story around a subject/cohort that already has H3b
or H3c support.

This script does NOT run any new permutation nulls — it reads the flags,
deltas, and p-values already written by ``run_topic5_v3_{avalanche,
dynamics,susceptibility}.py`` and joins them by subject. ``tier`` (0-4; 5 is
reserved for V3b) is assigned ONLY here — no earlier task computes it.

See docs/superpowers/plans/2026-07-02-topic5-v3a-mode-transition.md Task 10
and docs/superpowers/specs/2026-07-02-topic5-v3a-mode-transition-design.md §7.
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

from src.topic5_v3_mode_transition import load_v3_config  # noqa: E402

_RESULTS_BASE = _ROOT / "results/topic5_ictal_recruitment/v3_mode_transition"

SUMMARY_COLS = [
    "subject", "cohort", "geometry_insufficient",
    "avalanche_status", "dynamics_status", "susceptibility_status",
    "h3b_path", "h3c_path", "subject_support", "support_driver",
    "common_drive_downgrade", "h3a_strengthens",
    "delta_net_offaxis_flux_surplus", "module_direction_correct_h3b",
    "delta_mode_shift_density", "module_direction_correct_h3c",
]


# ---------------------------------------------------------------------------
# CSV loading + per-endpoint views (string-valued DictReader rows -> typed)
# ---------------------------------------------------------------------------
def _to_bool(s) -> bool:
    return str(s) == "True"


def _to_float(s) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _load_rows(csv_path: Path) -> dict | None:
    """``{subject: raw DictReader row}``, or ``None`` if the file is absent.

    ``None`` (missing file) is a DIFFERENT condition from "file exists but a
    given subject's row has ``status=='skipped'``" — the latter is a real,
    per-subject geometry/compute outcome; the former means this endpoint has
    not been run for this cohort yet (a pipeline-staging gap, not a subject
    property) and must not be silently read as "0 subjects qualify".
    """
    if not csv_path.exists():
        return None
    with csv_path.open(newline="") as fh:
        return {r["subject"]: r for r in csv.DictReader(fh)}


_AVALANCHE_DEFAULT = {
    "status": "skipped", "geometry_sufficient": False,
    "module_support_flag": False, "onset_jitter_pass": False,
    "leave_one_contact_pass": False, "axis_only_control_pass": False,
    "common_drive_sensitive": True, "module_direction_correct": False,
    "delta_net_offaxis_flux_surplus": float("nan"),
}


def _avalanche_view(row: dict | None) -> dict:
    """H3b fields ``subject_support``/``geometry_insufficient`` need. A
    missing row (subject absent from this cohort's avalanche CSV) defaults
    to the same shape a ``status=='skipped'`` row would have.
    """
    if row is None:
        return dict(_AVALANCHE_DEFAULT)
    return {
        "status": row["status"],
        "geometry_sufficient": _to_bool(row["geometry_sufficient"]),
        "module_support_flag": _to_bool(row["module_support_flag"]),
        "onset_jitter_pass": _to_bool(row["onset_jitter_pass"]),
        "leave_one_contact_pass": _to_bool(row["leave_one_contact_pass"]),
        "axis_only_control_pass": _to_bool(row["axis_only_control_pass"]),
        "common_drive_sensitive": _to_bool(row["common_drive_sensitive"]),
        "module_direction_correct": _to_bool(row["module_direction_correct"]),
        "delta_net_offaxis_flux_surplus": _to_float(row["delta_net_offaxis_flux_surplus"]),
    }


_DYNAMICS_DEFAULT = {
    "status": "skipped", "geometry_sufficient": False,
    "module_support_flag": False, "onset_jitter_pass": False,
    "leave_one_contact_mode_shift_pass": False, "axis_only_control_pass": False,
    "single_contact_driven": True, "module_direction_correct": False,
    "delta_mode_shift_density": float("nan"),
}


def _dynamics_view(row: dict | None) -> dict:
    """H3c fields ``subject_support``/``geometry_insufficient`` need."""
    if row is None:
        return dict(_DYNAMICS_DEFAULT)
    return {
        "status": row["status"],
        "geometry_sufficient": _to_bool(row["geometry_sufficient"]),
        "module_support_flag": _to_bool(row["module_support_flag"]),
        "onset_jitter_pass": _to_bool(row["onset_jitter_pass"]),
        "leave_one_contact_mode_shift_pass": _to_bool(row["leave_one_contact_mode_shift_pass"]),
        "axis_only_control_pass": _to_bool(row["axis_only_control_pass"]),
        "single_contact_driven": _to_bool(row["single_contact_driven"]),
        "module_direction_correct": _to_bool(row["module_direction_correct"]),
        "delta_mode_shift_density": _to_float(row["delta_mode_shift_density"]),
    }


_SUSCEPTIBILITY_DEFAULT = {
    "status": "skipped", "module_direction_correct": False, "beta_axis_P3_reliable": False,
    "module_null_pass": False,
}


def _susceptibility_view(row: dict | None) -> dict:
    """H3a is context-only: no ``geometry_sufficient`` column (Task 9
    contract), never gates ``geometry_insufficient`` or `subject_support``.
    """
    if row is None:
        return dict(_SUSCEPTIBILITY_DEFAULT)
    return {
        "status": row["status"],
        "module_direction_correct": _to_bool(row["module_direction_correct"]),
        "beta_axis_P3_reliable": _to_bool(row["beta_axis_P3_reliable"]),
        "module_null_pass": _to_bool(row["module_null_pass"]),
    }


# ---------------------------------------------------------------------------
# per-subject join + gate stack
# ---------------------------------------------------------------------------
def _subject_row(subject: str, cohort: str, av: dict, dyn: dict, susc: dict) -> dict:
    """One ``v3_summary_subject.csv`` row: the endpoint-appropriate
    ``subject_support`` gate stack (each co-primary uses ITS OWN robustness
    columns), the H3a context flag, and the "downgrade" audit trail.
    """
    geometry_insufficient = (
        av["status"] == "skipped" or not av["geometry_sufficient"]
        or dyn["status"] == "skipped" or not dyn["geometry_sufficient"]
    )

    h3b_path = bool(
        av["module_support_flag"] and av["onset_jitter_pass"]
        and av["leave_one_contact_pass"] and av["axis_only_control_pass"]
        and not av["common_drive_sensitive"]
    )
    h3c_path = bool(
        dyn["module_support_flag"] and dyn["onset_jitter_pass"]
        and dyn["leave_one_contact_mode_shift_pass"] and dyn["axis_only_control_pass"]
        and not dyn["single_contact_driven"]
    )
    subject_support = bool((h3b_path or h3c_path) and not geometry_insufficient)

    if geometry_insufficient:
        support_driver = ""
    elif h3b_path and h3c_path:
        support_driver = "H3b+H3c"
    elif h3b_path:
        support_driver = "H3b"
    elif h3c_path:
        support_driver = "H3c"
    else:
        support_driver = "none"

    # The "downgrade": H3b's own flag said yes, but the common-drive control
    # vetoed it — record this regardless of whether H3c ends up carrying
    # subject_support anyway, so the audit trail never silently disappears.
    common_drive_downgrade = bool(av["module_support_flag"] and av["common_drive_sensitive"])
    # H3a strengthens the narrative only when it is SIGNIFICANT -- i.e. it
    # also clears its own Delta-null gate (module_null_pass = p_spatial_delta
    # < alpha AND p_label_delta < alpha, Task 9). Direction-correct + a
    # reliable P3 baseline alone is not "significant". It can never set
    # subject_support either way.
    h3a_strengthens = bool(
        susc["module_direction_correct"] and susc["beta_axis_P3_reliable"]
        and susc["module_null_pass"]
    )

    return {
        "subject": subject, "cohort": cohort,
        "geometry_insufficient": geometry_insufficient,
        "avalanche_status": av["status"], "dynamics_status": dyn["status"],
        "susceptibility_status": susc["status"],
        "h3b_path": h3b_path, "h3c_path": h3c_path,
        "subject_support": subject_support, "support_driver": support_driver,
        "common_drive_downgrade": common_drive_downgrade,
        "h3a_strengthens": h3a_strengthens,
        "delta_net_offaxis_flux_surplus": av["delta_net_offaxis_flux_surplus"],
        "module_direction_correct_h3b": bool(av["module_direction_correct"]),
        "delta_mode_shift_density": dyn["delta_mode_shift_density"],
        "module_direction_correct_h3c": bool(dyn["module_direction_correct"]),
    }


# ---------------------------------------------------------------------------
# cohort-level stats: Holm-corrected co-primary Wilcoxon
# ---------------------------------------------------------------------------
def _wilcoxon_greater(deltas: np.ndarray) -> float:
    """One-sided-greater Wilcoxon signed-rank p on subject-level deltas.

    Guards scipy's raise-on-degenerate-input (all differences zero) and an
    underpowered draw (<2 finite deltas, or 0 non-zero deltas) to a NaN
    return instead of crashing the whole cohort summary.
    """
    finite = deltas[np.isfinite(deltas)]
    nonzero = finite[finite != 0]
    if finite.size < 2 or nonzero.size < 1:
        return float("nan")
    try:
        _, p = wilcoxon(finite, alternative="greater")
    except ValueError:
        return float("nan")
    return float(p)


def _holm_correct_2(p1: float, p2: float) -> tuple[float, float]:
    """Holm step-down correction for the 2 pre-registered co-primary
    endpoints (``statistics.co_primary_correction: holm``, plan §7).

    ``m=2`` always, regardless of how many of the two raw p-values happen to
    be finite this run (the pre-registration fixes the FAMILY size, not this
    run's data availability). Sort ascending — NaN sorts last (numpy
    convention: "worst"); the smaller finite p is multiplied by 2, the
    larger by 1; then enforce non-decreasing order (standard Holm step-down
    running max). A NaN input stays NaN in its own slot and never receives a
    multiplier, but does not change the OTHER, finite endpoint's multiplier.
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


_UNAVAILABLE_COHORT_BLOCK = {
    "available": False,
    "n_geometry_sufficient": 0, "n_geometry_insufficient": 0, "n_subject_support": 0,
    "p_h3b": float("nan"), "p_h3c": float("nan"),
    "p_holm_h3b": float("nan"), "p_holm_h3c": float("nan"),
    "median_delta_h3b": float("nan"), "median_delta_h3c": float("nan"),
    "cohort_h3b_pass": False, "cohort_h3c_pass": False,
}


def _build_cohort(cohort: str, alpha: float, required: bool) -> tuple[list, dict]:
    """Load + join one cohort's 3 CSVs. Returns ``(subject_rows, block)``.

    ``avalanche``/``dynamics`` are the 2 co-primary CSVs; if EITHER is
    entirely missing (not just some rows skipped) and this cohort was not
    explicitly requested (``required=False`` — it's only being read for the
    tier's replication check), degrade to the all-default "unavailable"
    block rather than reporting a fabricated "0 geometry-sufficient"
    (CLAUDE.md #6: a stub-like silent return must not be mistaken for a real
    negative result). A REQUESTED cohort missing either file raises loudly —
    the user asked to summarize a cohort this script cannot summarize yet.
    ``susceptibility`` is context-only (H3a never gates anything) and is
    always tolerated missing, for any cohort.
    """
    base = _RESULTS_BASE / cohort
    av_rows = _load_rows(base / "v3_avalanche_subject.csv")
    dyn_rows = _load_rows(base / "v3_dynamics_subject.csv")
    susc_rows = _load_rows(base / "v3_susceptibility_subject.csv") or {}

    missing = [name for name, rows in (("avalanche", av_rows), ("dynamics", dyn_rows)) if rows is None]
    if missing:
        if required:
            raise FileNotFoundError(
                f"cohort {cohort!r}: missing required input CSV(s) {missing} under {base} -- "
                f"run scripts/run_topic5_v3_{{avalanche,dynamics}}.py --cohort {cohort} first"
            )
        return [], dict(_UNAVAILABLE_COHORT_BLOCK)

    subjects = sorted(set(av_rows) | set(dyn_rows) | set(susc_rows))
    subject_rows = [
        _subject_row(
            s, cohort,
            _avalanche_view(av_rows.get(s)),
            _dynamics_view(dyn_rows.get(s)),
            _susceptibility_view(susc_rows.get(s)),
        )
        for s in subjects
    ]

    included = [r for r in subject_rows if not r["geometry_insufficient"]]
    n_geo_suff = len(included)
    n_geo_insuff = len(subject_rows) - n_geo_suff
    n_support = sum(1 for r in included if r["subject_support"])

    deltas_h3b = np.array([r["delta_net_offaxis_flux_surplus"] for r in included], dtype=float)
    deltas_h3c = np.array([r["delta_mode_shift_density"] for r in included], dtype=float)
    deltas_h3b = deltas_h3b[np.isfinite(deltas_h3b)]
    deltas_h3c = deltas_h3c[np.isfinite(deltas_h3c)]

    p_h3b = _wilcoxon_greater(deltas_h3b)
    p_h3c = _wilcoxon_greater(deltas_h3c)
    if not np.isfinite(p_h3b):
        print(f"[warn] {cohort}: p_h3b is NaN (n_finite_deltas={deltas_h3b.size}) "
              "-- insufficient signal for Wilcoxon", flush=True)
    if not np.isfinite(p_h3c):
        print(f"[warn] {cohort}: p_h3c is NaN (n_finite_deltas={deltas_h3c.size}) "
              "-- insufficient signal for Wilcoxon", flush=True)
    p_holm_h3b, p_holm_h3c = _holm_correct_2(p_h3b, p_h3c)

    med_h3b = float(np.median(deltas_h3b)) if deltas_h3b.size else float("nan")
    med_h3c = float(np.median(deltas_h3c)) if deltas_h3c.size else float("nan")

    cohort_h3b_pass = bool(np.isfinite(p_holm_h3b) and p_holm_h3b < alpha and med_h3b > 0)
    cohort_h3c_pass = bool(np.isfinite(p_holm_h3c) and p_holm_h3c < alpha and med_h3c > 0)

    block = {
        "available": True,
        "n_geometry_sufficient": n_geo_suff, "n_geometry_insufficient": n_geo_insuff,
        "n_subject_support": n_support,
        "p_h3b": p_h3b, "p_h3c": p_h3c,
        "p_holm_h3b": p_holm_h3b, "p_holm_h3c": p_holm_h3c,
        "median_delta_h3b": med_h3b, "median_delta_h3c": med_h3c,
        "cohort_h3b_pass": cohort_h3b_pass, "cohort_h3c_pass": cohort_h3c_pass,
    }
    return subject_rows, block


# ---------------------------------------------------------------------------
# tier verdict (single overall verdict; narrow primary, broad replication)
# ---------------------------------------------------------------------------
def _tier_verdict(narrow_block: dict, broad_block: dict) -> dict:
    """tier 0-4 (5 reserved for V3b, never emitted here); narrow + broad are
    NEVER pooled — broad only promotes narrow's tier via same-endpoint,
    same-direction replication (``cohort_*_pass`` already requires
    median>0, so "same direction" needs no separate check here).
    """
    narrow_cohort_pass = bool(narrow_block["cohort_h3b_pass"] or narrow_block["cohort_h3c_pass"])
    broad_cohort_pass = bool(broad_block["cohort_h3b_pass"] or broad_block["cohort_h3c_pass"])

    endpoint_keys = (("h3b", "cohort_h3b_pass"), ("h3c", "cohort_h3c_pass"))
    narrow_endpoints = {e for e, k in endpoint_keys if narrow_block[k]}
    broad_endpoints = {e for e, k in endpoint_keys if broad_block[k]}
    broad_replicates = bool(narrow_cohort_pass and (narrow_endpoints & broad_endpoints))

    narrow_direction_correct = bool(
        (np.isfinite(narrow_block["median_delta_h3b"]) and narrow_block["median_delta_h3b"] > 0)
        or (np.isfinite(narrow_block["median_delta_h3c"]) and narrow_block["median_delta_h3c"] > 0)
    )

    if narrow_cohort_pass and broad_replicates:
        tier = 4
    elif narrow_cohort_pass:
        tier = 3
    elif narrow_block["n_subject_support"] >= 1:
        tier = 2
    elif narrow_direction_correct:
        tier = 1
    else:
        tier = 0

    return {
        "tier": tier, "state_v3_supported": bool(tier >= 3),
        "narrow_cohort_pass": narrow_cohort_pass,
        "broad_cohort_pass": broad_cohort_pass,
        "broad_replicates": broad_replicates,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["narrow", "broad"], default=None,
                     help="default: both narrow and broad")
    ap.add_argument("--outdir", default=None,
                     help="default: results/.../v3_mode_transition/<cohort>; "
                          "requires --cohort when processing both (2 cohorts can't share 1 outdir)")
    args = ap.parse_args()

    if args.outdir and args.cohort is None:
        ap.error("--outdir requires --cohort (default both-cohort mode writes 2 separate directories)")

    requested = [args.cohort] if args.cohort else ["narrow", "broad"]

    cfg = load_v3_config()
    alpha = float(cfg["nulls"]["alpha"])

    subject_rows_by_cohort = {}
    blocks = {}
    for c in ("narrow", "broad"):
        subject_rows, block = _build_cohort(c, alpha, required=(c in requested))
        subject_rows_by_cohort[c] = subject_rows
        blocks[c] = block

    verdict = _tier_verdict(blocks["narrow"], blocks["broad"])
    payload = {
        "tier": verdict["tier"], "state_v3_supported": verdict["state_v3_supported"],
        "alpha": alpha,
        "narrow_cohort_pass": verdict["narrow_cohort_pass"],
        "broad_cohort_pass": verdict["broad_cohort_pass"],
        "broad_replicates": verdict["broad_replicates"],
        "narrow": blocks["narrow"], "broad": blocks["broad"],
    }

    for c in requested:
        outdir = Path(args.outdir) if args.outdir else _RESULTS_BASE / c
        outdir.mkdir(parents=True, exist_ok=True)

        out_csv = outdir / "v3_summary_subject.csv"
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SUMMARY_COLS)
            w.writeheader()
            w.writerows(subject_rows_by_cohort[c])

        out_json = outdir / "v3_cohort_tier.json"
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

        print(
            f"[done] {c}: {len(subject_rows_by_cohort[c])} subjects -> {out_csv} "
            f"(n_subject_support={blocks[c]['n_subject_support']}/{blocks[c]['n_geometry_sufficient']}) "
            f"-> {out_json} (tier={verdict['tier']} state_v3_supported={verdict['state_v3_supported']})",
            flush=True,
        )


if __name__ == "__main__":
    main()
