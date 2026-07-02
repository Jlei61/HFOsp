#!/usr/bin/env python
"""Topic 5 V3a mode-transition — feasibility pilot (Task 3, PILOT-LOCK GATE).

Per subject: build the all-clean contact pool from the ictal field long
cache, compute interictal HFO participation per contact, classify contacts
into axis / non-axis-strict / ambiguous (frozen `classify_contacts`), check
the axis/non-axis geometry gate (frozen `geometry_sufficient`), and count
P3/I1 sliding windows per seizure (frozen `phase_bin_range` +
`sliding_windows`). Writes `feasibility.csv` for the human pilot-lock
decision (Task 3 Step 6, done by the controller — NOT this script). See
docs/superpowers/plans/2026-07-02-topic5-v3a-mode-transition.md Task 3.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._topic5_v2_crit_io import load_context, shaft_of  # noqa: E402
from scripts.run_topic5_ictal_field_dynamics import SUBJECTS_BY_SUB  # noqa: E402
from src.interictal_propagation import load_subject_propagation_events  # noqa: E402
from src.topic5_v3_mode_transition import (  # noqa: E402
    classify_contacts,
    geometry_sufficient,
    i1_range,
    load_v3_config,
    phase_bin_range,
    sliding_windows,
)

CACHE = _ROOT / "results/topic5_ictal_recruitment/ictal_field_long_cache"
LAGPAT_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")

CSV_COLS = [
    "subject", "cohort", "n_seizures", "eeg_onset_rel_median", "eeg_offset_rel_median",
    "duration_median", "usable_pre_sec", "usable_ictal_sec", "n_contacts_all_clean",
    "n_axis", "n_nonaxis", "n_ambiguous", "n_windows_P3", "n_windows_I1",
    "i1_eligible", "geometry_sufficient",
]

# Defaults for a subject whose context/cache cannot be loaded at all — the row
# still exists (never silently drop a subject); a [skip] line on stdout carries
# the reason (feasibility.csv's column contract is fixed, no skip_reason column).
_FAILURE_ROW = {
    "n_seizures": 0,
    "eeg_onset_rel_median": float("nan"),
    "eeg_offset_rel_median": float("nan"),
    "duration_median": float("nan"),
    "usable_pre_sec": float("nan"),
    "usable_ictal_sec": float("nan"),
    "n_contacts_all_clean": 0,
    "n_axis": 0,
    "n_nonaxis": 0,
    "n_ambiguous": 0,
    "n_windows_P3": 0.0,
    "n_windows_I1": 0.0,
    "i1_eligible": False,
    "geometry_sufficient": False,
}


def _median_or(values: list, default: float) -> float:
    return float(np.median(values)) if values else default


def _n_windows(relt: np.ndarray, onset: float, offset: float, dur: float, phase: str, cfg: dict) -> int:
    rng = phase_bin_range(relt, onset, offset, dur, phase, cfg)
    if rng is None:
        return 0
    return len(sliding_windows(relt, rng[0], rng[1], cfg["phases"]["window_sec"], cfg["phases"]["step_sec"]))


def _load_participation(subj: str, all_clean: list) -> tuple[dict, str]:
    """Interictal HFO participation per clean contact.

    The 0.0 default for contacts absent from the lagPat pool IS the non-axis
    definition (a contact that never fires an interictal HFO has
    participation 0 < thresh -> non-axis-strict). On lagPat load failure,
    participation is all-0 for every clean contact (classification still
    proceeds via axis_template_names) rather than crashing the subject.
    """
    try:
        ev = load_subject_propagation_events(LAGPAT_ROOT / subj / "all_recs")
        part_raw = {n: float(np.mean(ev["bools"][i])) for i, n in enumerate(ev["channel_names"])}
        return {n: part_raw.get(n, 0.0) for n in all_clean}, ""
    except Exception as exc:  # noqa: BLE001 - external mount, any failure must not crash the cohort
        return {n: 0.0 for n in all_clean}, f"lagpat_load_failed:{type(exc).__name__}:{exc}"


def _axis_template_names(ctx: dict, all_clean_set: set) -> list:
    """Names with finite ``typical_rank`` in either template, intersected with ``all_clean``."""
    names = set()
    for rec in (ctx["ta"], ctx["tb"]):
        for c in rec["channels"]:
            r = c.get("typical_rank", np.nan)
            if np.isfinite(r) and c["name"] in all_clean_set:
                names.add(c["name"])
    return sorted(names)


def run_subject(ds_sid: str, cohort: str, cfg: dict) -> dict:
    _, subj = ds_sid.split("_", 1)
    row = {"subject": ds_sid, "cohort": cohort, **_FAILURE_ROW}
    try:
        ctx = load_context(ds_sid, cohort)
        data = np.load(CACHE / f"{ds_sid}.npz", allow_pickle=True)
        cache_names = [str(x) for x in data["channels"]]
        meta = json.loads((CACHE / f"{ds_sid}.json").read_text())

        drops = set(map(str, meta.get("drops", [])))
        all_clean = [n for n in cache_names if n not in drops]
        all_clean_set = set(all_clean)

        participation, skip_reason = _load_participation(subj, all_clean)
        if skip_reason:
            print(f"[warn] {ds_sid} ({cohort}): {skip_reason}", flush=True)

        axis_template_names = _axis_template_names(ctx, all_clean_set)
        cl = classify_contacts(
            all_clean, axis_template_names, participation,
            cfg["geometry"]["nonaxis_hfo_participation_max"],
        )
        shafts_with_both = len(
            {shaft_of(n) for n in cl["is_axis"]} & {shaft_of(n) for n in cl["is_nonaxis_strict"]}
        )
        geom_ok, _geom_reason = geometry_sufficient(cl["n_axis"], cl["n_nonaxis"], shafts_with_both, cfg)

        onset_l, offset_l, dur_l, usable_pre_l, usable_ictal_l = [], [], [], [], []
        n_windows_p3_l, n_windows_i1_l = [], []
        i1_eligible_any = False
        n_used = 0
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
            onset_l.append(onset)
            offset_l.append(offset)
            dur_l.append(dur)

            pre_vals = relt[relt < 0.0]
            usable_pre_l.append(
                min(float(-pre_vals.min()), cfg["phases"]["span_pre_sec"]) if pre_vals.size else 0.0
            )
            usable_ictal_l.append(min(dur, max(0.0, float(relt.max()) - onset)))
            n_windows_p3_l.append(_n_windows(relt, onset, offset, dur, "P3", cfg))

            _, _, elig = i1_range(onset, offset, dur, cfg)
            if elig:
                i1_eligible_any = True
                n_windows_i1_l.append(_n_windows(relt, onset, offset, dur, "I1", cfg))

        row.update({
            "n_seizures": n_used,
            "eeg_onset_rel_median": _median_or(onset_l, float("nan")),
            "eeg_offset_rel_median": _median_or(offset_l, float("nan")),
            "duration_median": _median_or(dur_l, float("nan")),
            "usable_pre_sec": _median_or(usable_pre_l, float("nan")),
            "usable_ictal_sec": _median_or(usable_ictal_l, float("nan")),
            "n_contacts_all_clean": len(all_clean),
            "n_axis": cl["n_axis"],
            "n_nonaxis": cl["n_nonaxis"],
            "n_ambiguous": cl["n_ambiguous"],
            "n_windows_P3": _median_or(n_windows_p3_l, 0.0),
            "n_windows_I1": _median_or(n_windows_i1_l, 0.0),
            "i1_eligible": bool(i1_eligible_any),
            "geometry_sufficient": bool(geom_ok),
        })
    except Exception as exc:  # noqa: BLE001 - never silently drop a subject
        print(f"[skip] {ds_sid} ({cohort}): {type(exc).__name__}: {exc}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["narrow", "broad"], required=True)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    outdir = (
        Path(args.outdir) if args.outdir
        else _ROOT / "results/topic5_ictal_recruitment/v3_mode_transition" / args.cohort
    )
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_v3_config()
    subjects = SUBJECTS_BY_SUB[args.cohort]
    rows = [run_subject(ds_sid, args.cohort, cfg) for ds_sid in subjects]

    with open(outdir / "feasibility.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(rows)

    n_qualify = sum(1 for r in rows if r["geometry_sufficient"] and r["i1_eligible"])
    print(
        f"[done] {len(rows)} subjects -> {outdir / 'feasibility.csv'} "
        f"({n_qualify}/{len(rows)} geometry_sufficient AND i1_eligible)",
        flush=True,
    )


if __name__ == "__main__":
    main()
