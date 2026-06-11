#!/usr/bin/env python3
"""PILOT analysis: does broadening the lagPat pool reveal propagation structure
beyond the rate/SOZ-restricted core?  (PASS/FAIL gates pre-specified.)

For each pilot subject (broad lagPat in results/lagpat_broad_pilot/<subj>):
  - core channels   = the current narrow refined template pool
  - added channels  = broad pool minus core
  - participation floor: channel must participate in >= PART_FLOOR of events in
    BOTH split-halves (else rank is phantom-dominated → excluded).
  - reproducibility: per-channel split-half shift of median RANK-FRACTION
    ((rank-1)/(n_participating-1), scale-free 0=first..1=last). A channel is
    "reproducible" if its shift <= the core channels' 75th-percentile shift
    (i.e. its propagation position is as stable as the core's = real ORDER
    structure, which is rate-orthogonal).
  - anatomical cross-check: does a reproducible ADDED channel fall in the
    report-documented spike/spread network (the network the tight template
    under-counted)?

DECISION (per subject):
  - reproducible added channels that map onto the documented spread network
    => broadening RECOVERS the under-counted epileptogenic network
       (= structure beyond the rate-core; fixes the §5.4 coverage confound).
  - added channels mostly non-reproducible / off-network
    => CONFIRMS focal-core (honest bound).
NOT load-bearing: broad-vs-narrow template similarity (event-set also changes).
"""
from __future__ import annotations
import json, re, sys, csv
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.interictal_propagation import load_subject_propagation_events  # noqa: E402

PART_FLOOR = 0.10
BROAD_DIR = REPO / "results" / "lagpat_broad_pilot"
NARROW = REPO / "results" / "interictal_propagation_masked" / "per_subject"
OUT = REPO / "results" / "lagpat_broad_pilot"

def expand(spec):
    """'F1-7' -> {F1..F7}; 'F6-7' -> {F6,F7}; 'C4-10' -> {C4..C10}; 'D13-14'."""
    out = set()
    for tok in re.findall(r"([A-Z]+'?)(\d+)-(\d+)|([A-Z]+'?)(\d+)", spec):
        if tok[0]:
            el, a, b = tok[0], int(tok[1]), int(tok[2])
            for n in range(a, b + 1):
                out.add(f"{el}{n}")
        else:
            out.add(f"{tok[3]}{tok[4]}")
    return out

# report-documented networks (onset + interictal + ictal-spread), from §5.4 case-review
CLINICAL = {
    "zhaojinrui": expand("F1-7 B7-10 C4-10 D5-9 E1-5 G1-3 K9-15"),
    "chenziyang": expand("D2-5 D12-14 E1-7 B1-15 H1-9 F1-2 G1-4 G9-10 A8-10 C8-10"),
}

def base(ch):
    """alias-left bipolar name 'F6' (=F6-F7) -> {'F6','F7'} membership keys."""
    m = re.match(r"([A-Z]+'?)(\d+)", str(ch))
    if not m:
        return {str(ch)}
    el, n = m.group(1), int(m.group(2))
    return {f"{el}{n}", f"{el}{n+1}"}  # alias-left covers contact n and n+1

def narrow_core(subj):
    f = NARROW / f"yuquan_{subj}.json"
    if not f.exists():
        return set()
    return {str(c) for c in json.load(open(f))["channel_names"]}

def rank_fraction(ranks, bools):
    """per-event scale-free position: (rank-1)/(npart-1) for participating ch."""
    rf = np.full(ranks.shape, np.nan)
    for e in range(ranks.shape[1]):
        part = bools[:, e]
        npart = int(part.sum())
        if npart < 2:
            continue
        r = ranks[:, e]
        rf[part, e] = (r[part] - np.nanmin(r[part])) / (npart - 1)
    return rf

def analyze(subj):
    d = load_subject_propagation_events(BROAD_DIR / subj)
    ch = [str(c) for c in d["channel_names"]]
    ranks = np.asarray(d["ranks"], float)
    bools = np.asarray(d["bools"]) > 0
    n_ev = ranks.shape[1]
    half = n_ev // 2
    A, B = slice(0, half), slice(half, 2 * half)
    rf = rank_fraction(ranks, bools)
    core = narrow_core(subj)
    clin = CLINICAL.get(subj, set())
    rows = []
    for i, c in enumerate(ch):
        partA, partB = bools[i, A].mean(), bools[i, B].mean()
        is_core = c in core
        in_clin = bool(base(c) & clin)
        if partA < PART_FLOOR or partB < PART_FLOOR:
            rows.append(dict(ch=c, is_core=is_core, in_clinical=in_clin,
                             part=round(float(bools[i].mean()), 3),
                             shift=None, reproducible=None, note="below_floor"))
            continue
        fa = rf[i, A][~np.isnan(rf[i, A])]
        fb = rf[i, B][~np.isnan(rf[i, B])]
        shift = abs(np.median(fa) - np.median(fb))
        rows.append(dict(ch=c, is_core=is_core, in_clinical=in_clin,
                         part=round(float(bools[i].mean()), 3),
                         shift=round(float(shift), 3), reproducible=None, note=""))
    core_shifts = [r["shift"] for r in rows if r["is_core"] and r["shift"] is not None]
    bar = float(np.percentile(core_shifts, 75)) if core_shifts else None
    for r in rows:
        if r["shift"] is not None and bar is not None:
            r["reproducible"] = bool(r["shift"] <= bar)
    added = [r for r in rows if not r["is_core"]]
    added_floor = [r for r in added if r["reproducible"] is not None]
    repro_added = [r for r in added_floor if r["reproducible"]]
    repro_added_onnet = [r for r in repro_added if r["in_clinical"]]
    summary = dict(
        subject=subj, n_broad=len(ch), n_core=len(core), n_events=n_ev,
        core_shift_bar_p75=round(bar, 3) if bar else None,
        n_added=len(added), n_added_above_floor=len(added_floor),
        n_added_reproducible=len(repro_added),
        n_added_reproducible_on_clinical_net=len(repro_added_onnet),
        reproducible_added_channels=[r["ch"] for r in repro_added],
        on_net_added_channels=[r["ch"] for r in repro_added_onnet],
    )
    return summary, rows

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows, summaries = [], []
    for subj in ["zhaojinrui", "chenziyang"]:
        if not (BROAD_DIR / subj).exists():
            print(f"[skip] {subj}: no broad lagPat yet")
            continue
        s, rows = analyze(subj)
        summaries.append(s)
        for r in rows:
            r["subject"] = subj
            all_rows.append(r)
        print(f"\n=== {subj} ===")
        for k, v in s.items():
            print(f"  {k}: {v}")
    if all_rows:
        cols = ["subject", "ch", "is_core", "in_clinical", "part", "shift",
                "reproducible", "note"]
        with open(OUT / "pilot_channel_analysis.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k, "") for k in cols})
        json.dump(summaries, open(OUT / "pilot_summary.json", "w"), indent=2)
        print(f"\nwrote {OUT/'pilot_channel_analysis.csv'} + pilot_summary.json")

if __name__ == "__main__":
    main()
