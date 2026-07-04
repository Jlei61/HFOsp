#!/usr/bin/env python
"""Topic 5 V3p — preictal-only non-axial trajectory: feasibility pilot (Task 1, DECISION GATE).

Per subject: build the all-clean contact pool + interictal HFO participation
+ axis/non-axis-strict/ambiguous classification + geometry gate via the
frozen V3a `scripts._topic5_v3_io.classify_subject_contacts` (single source
of truth; READ-ONLY reuse -- no V3a file is touched), then count PREICTAL
P0/P1/P2/P3 sliding windows per seizure (frozen `phase_bin_range` +
`sliding_windows`, V3a real-relt window contract) to gauge whether there is
enough preictal data for a per-seizure trend slope.

Adds the rev2 cohort-expansion axis-quality gate (curated roster subjects
are grandfathered `admitted=True`; expansion candidates are admitted only if
they pass geometry + axis-rank-distinctness + participation-gap + have a
rank-displacement JSON) and a label-null permutability estimate (how many
distinct axis/non-axis shaft-constrained label permutations are actually
reachable -- too few means the label-null p-value would be under-resolved).

Writes `feasibility.csv` for the human pilot-lock decision (Task 1 Step 6,
done by the controller -- NOT this script). See
docs/superpowers/plans/2026-07-03-topic5-v3p-preictal-trajectory.md Task 1.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._topic5_v3_io import CACHE, classify_subject_contacts  # noqa: E402
from scripts.run_topic5_ictal_field_dynamics import SUBJECTS_BY_SUB  # noqa: E402
from src.topic5_v3_mode_transition import load_v3_config, phase_bin_range, sliding_windows  # noqa: E402
from src.topic5_v3p_preictal_trajectory import load_v3p_config  # noqa: E402

PREICTAL_PHASES = ["P0", "P1", "P2", "P3"]
RANK_DISPLACEMENT_JSON_DIR = _ROOT / "results/interictal_propagation_masked/rank_displacement/per_subject"

CSV_COLS = [
    "subject", "cohort", "roster_status", "n_seizures",
    "n_windows_P0", "n_windows_P1", "n_windows_P2", "n_windows_P3",
    "n_windows_full_total", "n_windows_guard_total", "usable_pre_sec",
    "n_contacts_all_clean", "n_axis", "n_nonaxis", "n_ambiguous",
    "n_shaft_with_axis_and_nonaxis", "n_unique_label_permutations_est", "label_null_underpowered",
    "geometry_sufficient", "axis_rank_distinct", "axis_participation_gap",
    "has_rank_displacement_json", "axis_quality_gate_pass", "admitted",
    "n_seizures_ge_min_windows", "cohort_viable",
]

# Defaults for a subject whose context/cache cannot be loaded at all (this
# also covers `cohort="yuquan"`, which has no real SUBSTRATE wiring yet in
# `scripts.run_topic5_ictal_field_dynamics` -- a KeyError there lands here)
# -- the row still exists (never silently drop a subject); a [skip] line on
# stdout carries the reason (feasibility.csv's column contract is fixed, no
# skip_reason column). `admitted` is computed after this dict is merged in
# (roster is grandfathered regardless of load success).
_FAILURE_ROW = {
    "n_seizures": 0,
    "n_windows_P0": 0.0, "n_windows_P1": 0.0, "n_windows_P2": 0.0, "n_windows_P3": 0.0,
    "n_windows_full_total": 0.0, "n_windows_guard_total": 0.0,
    "usable_pre_sec": float("nan"),
    "n_contacts_all_clean": 0, "n_axis": 0, "n_nonaxis": 0, "n_ambiguous": 0,
    "n_shaft_with_axis_and_nonaxis": 0,
    "n_unique_label_permutations_est": 1.0,
    "label_null_underpowered": True,
    "geometry_sufficient": False,
    "axis_rank_distinct": 0,
    "axis_participation_gap": float("nan"),
    "has_rank_displacement_json": False,
    "axis_quality_gate_pass": False,
    "n_seizures_ge_min_windows": 0,
}


def _median_or(values: list, default: float) -> float:
    return float(np.median(values)) if values else default


def _phase_windows(relt: np.ndarray, onset: float, offset: float, dur: float, phase: str, cfg: dict) -> list:
    """(ws, we) sliding-window index pairs for one phase bin (real relt, V3a window contract)."""
    rng = phase_bin_range(relt, onset, offset, dur, phase, cfg)
    if rng is None:
        return []
    return sliding_windows(relt, rng[0], rng[1], cfg["phases"]["window_sec"], cfg["phases"]["step_sec"])


def _label_permutations_est(cc: dict) -> float:
    """exp(sum over shafts-with-both of log C(n_clean_in_shaft, k_axis_in_shaft)).

    An all-axis or all-nonaxis shaft contributes C(n, 0) = C(n, n) = 1 (log
    0), so restricting the sum to shafts-with-both is exact, not an
    approximation: it mirrors `label_permute`'s shaft-constrained null,
    where a uniform-label shaft has exactly one reachable relabeling
    (itself).
    """
    shaft_by_name = cc["shaft_by_name"]
    by_shaft: dict = {}
    for name in cc["is_axis"]:
        by_shaft.setdefault(shaft_by_name[name], []).append(True)
    for name in cc["is_nonaxis_strict"]:
        by_shaft.setdefault(shaft_by_name[name], []).append(False)

    log_total = 0.0
    for is_axis_flags in by_shaft.values():
        n_clean = len(is_axis_flags)
        k_axis = sum(is_axis_flags)
        if 0 < k_axis < n_clean:  # shaft-with-both only; all other shafts contribute log(1)=0
            log_total += math.log(math.comb(n_clean, k_axis))
    return math.exp(log_total)


def run_subject(ds_sid: str, load_cohort: str, roster_status: str, v3cfg: dict, v3pcfg: dict) -> dict:
    row = {"subject": ds_sid, "cohort": load_cohort, "roster_status": roster_status, **_FAILURE_ROW}
    try:
        cc = classify_subject_contacts(ds_sid, load_cohort, v3cfg)
        meta = cc["meta"]
        data = np.load(CACHE / f"{ds_sid}.npz", allow_pickle=True)

        usable_pre_l: list = []
        phase_counts = {ph: [] for ph in PREICTAL_PHASES}
        total_l, guard_l = [], []
        n_used = 0
        n_ge_min = 0
        min_windows = v3pcfg["preictal"]["min_windows_for_slope"]
        guard_end_sec = v3pcfg["preictal"]["span_guard_rel"][1]  # e.g. -20.0

        for si in meta.get("eligible_idxs", []):
            sz = meta.get("seizure", {}).get(str(si))
            relt_key = f"bb_relt__{si}"
            if sz is None or relt_key not in data.files:
                continue
            onset = float(sz["eeg_onset_rel"])
            offset = float(sz["eeg_offset_rel"])
            dur = float(sz["eeg_duration_sec"])
            relt = np.asarray(data[relt_key], dtype=float)
            n_used += 1

            pre_vals = relt[relt < 0.0]
            usable_pre_l.append(
                min(float(-pre_vals.min()), v3cfg["phases"]["span_pre_sec"]) if pre_vals.size else 0.0
            )

            sz_windows = {ph: _phase_windows(relt, onset, offset, dur, ph, v3cfg) for ph in PREICTAL_PHASES}
            for ph in PREICTAL_PHASES:
                phase_counts[ph].append(len(sz_windows[ph]))
            total_this = sum(len(w) for w in sz_windows.values())
            total_l.append(total_this)
            if total_this >= min_windows:
                n_ge_min += 1
            guard_l.append(sum(
                1 for windows in sz_windows.values() for (ws, we) in windows
                if (float(np.mean(relt[ws:we])) - onset) <= guard_end_sec
            ))

        ta_rank = {
            c["name"]: float(c["typical_rank"])
            for c in cc["ctx"]["ta"]["channels"]
            if np.isfinite(c.get("typical_rank", np.nan))
        }
        axis_rank_distinct = len({ta_rank[n] for n in cc["is_axis"] if n in ta_rank})

        axis_part = [cc["participation"][n] for n in cc["is_axis"]]
        nonaxis_part = [cc["participation"][n] for n in cc["is_nonaxis_strict"]]
        axis_participation_gap = (
            min(axis_part) - max(nonaxis_part) if axis_part and nonaxis_part else float("nan")
        )

        has_rank_displacement_json = (RANK_DISPLACEMENT_JSON_DIR / f"{ds_sid}.json").exists()

        n_unique_label_permutations_est = _label_permutations_est(cc)
        label_null_underpowered = (
            n_unique_label_permutations_est < v3pcfg["nulls_v3p"]["label_null_min_unique_perms"]
        )

        gate_cfg = v3pcfg["cohort_expansion"]["axis_quality_gate"]
        axis_quality_gate_pass = bool(
            cc["geometry_sufficient"]
            and axis_rank_distinct >= gate_cfg["axis_rank_min_distinct"]
            and axis_participation_gap >= gate_cfg["axis_participation_gap_min"]
            and has_rank_displacement_json
        )

        row.update({
            "n_seizures": n_used,
            "n_windows_P0": _median_or(phase_counts["P0"], 0.0),
            "n_windows_P1": _median_or(phase_counts["P1"], 0.0),
            "n_windows_P2": _median_or(phase_counts["P2"], 0.0),
            "n_windows_P3": _median_or(phase_counts["P3"], 0.0),
            "n_windows_full_total": _median_or(total_l, 0.0),
            "n_windows_guard_total": _median_or(guard_l, 0.0),
            "usable_pre_sec": _median_or(usable_pre_l, float("nan")),
            "n_contacts_all_clean": len(cc["all_clean"]),
            "n_axis": cc["n_axis"],
            "n_nonaxis": cc["n_nonaxis"],
            "n_ambiguous": cc["n_ambiguous"],
            "n_shaft_with_axis_and_nonaxis": cc["shafts_with_both"],
            "n_unique_label_permutations_est": n_unique_label_permutations_est,
            "label_null_underpowered": bool(label_null_underpowered),
            "geometry_sufficient": bool(cc["geometry_sufficient"]),
            "axis_rank_distinct": axis_rank_distinct,
            "axis_participation_gap": axis_participation_gap,
            "has_rank_displacement_json": bool(has_rank_displacement_json),
            "axis_quality_gate_pass": axis_quality_gate_pass,
            "n_seizures_ge_min_windows": n_ge_min,
        })
    except Exception as exc:  # noqa: BLE001 - never silently drop a subject
        print(f"[skip] {ds_sid} ({load_cohort}/{roster_status}): {type(exc).__name__}: {exc}", flush=True)

    row["admitted"] = bool(
        roster_status == "roster" or (roster_status == "candidate" and row["axis_quality_gate_pass"])
    )
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["narrow", "broad"], required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument(
        "--subjects", nargs="+", default=None,
        help="bare subject ids (e.g. 253 1125), filters both the roster and (if requested) the candidates",
    )
    ap.add_argument(
        "--include-candidates", action="store_true",
        help="also probe cohort_expansion.candidates_epilepsiae (broad context) + candidates_yuquan (yuquan context)",
    )
    args = ap.parse_args(argv)

    outdir = (
        Path(args.outdir) if args.outdir
        else _ROOT / "results/topic5_ictal_recruitment/v3p_preictal_trajectory/feasibility" / args.cohort
    )
    outdir.mkdir(parents=True, exist_ok=True)

    v3cfg = load_v3_config()
    v3pcfg = load_v3p_config()

    entries = [(ds_sid, args.cohort, "roster") for ds_sid in SUBJECTS_BY_SUB[args.cohort]]
    if args.include_candidates:
        exp = v3pcfg["cohort_expansion"]
        entries += [(ds_sid, "broad", "candidate") for ds_sid in exp.get("candidates_epilepsiae", [])]
        entries += [(ds_sid, "yuquan", "candidate") for ds_sid in exp.get("candidates_yuquan", [])]

    if args.subjects:
        wanted = set(args.subjects)
        entries = [e for e in entries if e[0].split("_", 1)[1] in wanted]

    rows = [
        run_subject(ds_sid, load_cohort, roster_status, v3cfg, v3pcfg)
        for ds_sid, load_cohort, roster_status in entries
    ]

    n_qualify_roster = sum(
        1 for r in rows
        if r["roster_status"] == "roster" and r["geometry_sufficient"] and r["n_seizures_ge_min_windows"] >= 1
    )
    cohort_viable = n_qualify_roster >= 4
    for r in rows:
        r["cohort_viable"] = cohort_viable

    with open(outdir / "feasibility.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(rows)

    n_roster = sum(1 for r in rows if r["roster_status"] == "roster")
    print(
        f"[done] {len(rows)} rows ({n_roster} roster) -> {outdir / 'feasibility.csv'} "
        f"(cohort_viable={cohort_viable}, {n_qualify_roster}/{n_roster} roster "
        f"geometry_sufficient AND n_seizures_ge_min_windows>=1)",
        flush=True,
    )


if __name__ == "__main__":
    main()
