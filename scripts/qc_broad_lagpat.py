#!/usr/bin/env python3
"""Broad-lagPat QC table (per user 2026-06-07 plan, steps 1 + 3).

For each subject with broad lagPat in results/lagpat_broad/<subj>/:
  narrow_n_ch, broad_n_ch, broad_n_events,
  per-event participation (min/25/med/75/max),
  added_channels (broad - narrow), added_in_clinical_network (frac),
  + 3 eligibility checks:
    [c1] n_broad close to 20 (>=15)
    [c2] median per-event participation reasonable (>=5; 3-5 under 20ch = noise)
    [c3] events not crashed (>=100 for stable clustering)
  eligible_for_kmeans = c1 & c2 & c3 ; else broad-ineligible (flagged, not silent).

Goal: confirm broad lagPat captures real participable outer-ring channels, not
phantom noise, BEFORE running KMeans.
"""
from __future__ import annotations
import sys, json, re, csv, glob, os
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.interictal_propagation import load_subject_propagation_events  # noqa: E402

BROAD = REPO / "results" / "lagpat_broad"
NARROW = REPO / "results" / "interictal_propagation_masked" / "per_subject"
NETS = BROAD / "yuquan_clinical_networks.json"

C1_MIN_BROAD = 15
C2_MIN_MED_PART = 5
C3_MIN_EVENTS = 100

def base(ch):
    m = re.match(r"([A-Z]+'?)(\d+)", str(ch).replace("’", "'"))
    if not m:
        return {str(ch)}
    el, n = m.group(1), int(m.group(2))
    return {f"{el}{n}", f"{el}{n+1}"}

def narrow_channels(subj):
    f = NARROW / f"yuquan_{subj}.json"
    if not f.exists():
        return None
    return {str(c) for c in json.load(open(f))["channel_names"]}

def main():
    nets = json.load(open(NETS)) if NETS.exists() else {}
    rows = []
    subj_dirs = sorted(d for d in glob.glob(str(BROAD / "*")) if os.path.isdir(d))
    for sd in subj_dirs:
        subj = os.path.basename(sd)
        if not glob.glob(os.path.join(sd, "*_lagPat.npz")):
            continue
        try:
            d = load_subject_propagation_events(Path(sd))
        except Exception as e:
            rows.append(dict(subject=subj, status=f"load_error:{str(e)[:40]}"))
            continue
        ch = [str(c) for c in d["channel_names"]]
        bools = np.asarray(d["bools"]) > 0
        n_ev = bools.shape[1]
        part = bools.sum(axis=0)  # participating channels per event
        narrow = narrow_channels(subj)
        added = sorted(set(ch) - narrow) if narrow else sorted(ch)
        net = set(nets.get(subj, {}).get("network", []))
        added_in_net = [c for c in added if base(c) & net] if net else []
        c1 = len(ch) >= C1_MIN_BROAD
        c2 = float(np.median(part)) >= C2_MIN_MED_PART
        c3 = n_ev >= C3_MIN_EVENTS
        rows.append(dict(
            subject=subj, status="ok",
            narrow_n_ch=(len(narrow) if narrow else None),
            broad_n_ch=len(ch), broad_n_events=int(n_ev),
            part_min=int(part.min()) if n_ev else 0,
            part_p25=float(np.percentile(part, 25)) if n_ev else 0,
            part_med=float(np.median(part)) if n_ev else 0,
            part_p75=float(np.percentile(part, 75)) if n_ev else 0,
            part_max=int(part.max()) if n_ev else 0,
            n_added=len(added),
            n_added_in_clinical_net=len(added_in_net),
            frac_added_in_net=(round(len(added_in_net) / len(added), 2) if added else None),
            c1_nbroad_ok=c1, c2_participation_ok=c2, c3_events_ok=c3,
            eligible_for_kmeans=bool(c1 and c2 and c3),
            added_channels=";".join(added),
        ))
    cols = ["subject", "status", "narrow_n_ch", "broad_n_ch", "broad_n_events",
            "part_min", "part_p25", "part_med", "part_p75", "part_max",
            "n_added", "n_added_in_clinical_net", "frac_added_in_net",
            "c1_nbroad_ok", "c2_participation_ok", "c3_events_ok",
            "eligible_for_kmeans", "added_channels"]
    out = BROAD / "qc_table.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    # console
    print(f"{'subject':<14}{'narrow':<7}{'broad':<6}{'events':<8}{'partMed':<8}{'added':<6}{'inNet':<6}{'eligible'}")
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['subject']:<14}{r['status']}")
            continue
        print(f"{r['subject']:<14}{str(r['narrow_n_ch']):<7}{r['broad_n_ch']:<6}"
              f"{r['broad_n_events']:<8}{r['part_med']:<8}{r['n_added']:<6}"
              f"{r['n_added_in_clinical_net']:<6}{r['eligible_for_kmeans']}")
    elig = [r["subject"] for r in rows if r.get("eligible_for_kmeans")]
    inelig = [r["subject"] for r in rows if r["status"] == "ok" and not r.get("eligible_for_kmeans")]
    print(f"\neligible (n={len(elig)}): {elig}")
    print(f"broad-ineligible (n={len(inelig)}): {inelig}")
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
