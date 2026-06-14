#!/usr/bin/env python3
"""Aggregate Stage 3 regime-screen runs across one or more result dirs, apply the pre-registered
pilot gate per (sep,std,mean,drive) cell, and emit (1) a per-run raw CSV and (2) a per-cell gate
CSV with pass/fail reason. Params are read from each readout's `config` (reliable) with sep_frac
falling back to the tag (older runs predate sep_frac-in-config).

The gate itself lives in src.sef_hfo_stage3.pilot_gate (unit-tested) — this script only groups and
reports, so the auto-refine decision rests on tested logic, not ad-hoc thresholds here.
"""
import os
import re
import csv
import glob
import json
import argparse
import statistics as st
from collections import defaultdict

import numpy as np

import sys
sys.path.insert(0, os.getcwd())
from src.sef_hfo_stage3 import pilot_gate  # noqa: E402


def _sep_from_tag(tag, cfg):
    if cfg.get("sep_frac") is not None:
        return float(cfg["sep_frac"])
    m = re.search(r"s(?:ep|f)([\d.]+)", tag)
    return float(m.group(1)) if m else 0.6


def coignition_diag(sidecar):
    rows = []
    for e in sidecar.get("events", []):
        if e.get("hidden_source_label") != "collision":
            continue
        fn, fp = e.get("core_frac_neg_first_bins") or [0], e.get("core_frac_pos_first_bins") or [0]
        tn, tp = e.get("core_threshold_neg") or 0, e.get("core_threshold_pos") or 0
        rows.append((max(fn) / tn if tn else 0.0, max(fp) / tp if tp else 0.0))
    if not rows:
        return None, None
    arr = np.array(rows)
    return round(float(arr[:, 0].mean()), 1), round(float(arr[:, 1].mean()), 1)


def parse_run(rf, dirpath):
    tag = os.path.basename(rf)[len("readout_"):-len(".json")]
    r = json.load(open(rf))
    cfg = r.get("config", {})
    sc = r.get("stage3_source_counts") or {}
    det = r.get("detector", {})
    cn = cp = None
    sf = os.path.join(dirpath, f"sidecar_{tag}.json")
    if os.path.exists(sf):
        try:
            cn, cp = coignition_diag(json.load(open(sf)))
        except Exception:
            pass
    return {"tag": tag, "lesion": cfg.get("lesion"), "sep_frac": _sep_from_tag(tag, cfg),
            "core_std": cfg.get("core_std"), "core_mean": cfg.get("core_mean"),
            "drive": cfg.get("drive"), "seed": cfg.get("seed"), "n_events": r.get("n_events"),
            "collision_rate": sc.get("collision_rate"), "neg_clean": sc.get("neg_clean"),
            "pos_clean": sc.get("pos_clean"), "collision": sc.get("collision"),
            "ambiguous": sc.get("ambiguous"), "n_fwd": r.get("n_clean_forward"),
            "n_rev": r.get("n_clean_reverse"), "true_floor": det.get("true_inter_event_floor"),
            "peak": det.get("peak"), "coign_neg_x_thresh": cn, "coign_pos_x_thresh": cp}


def _med(rs, k):
    v = [x[k] for x in rs if x.get(k) is not None]
    return st.median(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+", required=True, help="one or more result dirs to scan")
    ap.add_argument("--out", default=None, help="dir to write CSVs (default: first --dir)")
    ap.add_argument("--sign-ok", action="store_true", default=True,
                    help="readout direction is trusted (oneend sign sanity passed) -> enforce bidir")
    a = ap.parse_args()
    out = a.out or a.dir[0]
    rows = []
    for d in a.dir:
        for rf in sorted(glob.glob(os.path.join(d, "readout_*.json"))):
            try:
                rows.append(parse_run(rf, d))
            except Exception as ex:
                print(f"[skip] {rf}: {ex}")
    raw_cols = ["tag", "lesion", "sep_frac", "core_std", "core_mean", "drive", "seed", "n_events",
                "collision_rate", "neg_clean", "pos_clean", "collision", "ambiguous", "n_fwd",
                "n_rev", "true_floor", "peak", "coign_neg_x_thresh", "coign_pos_x_thresh"]
    with open(os.path.join(out, "regime_screen_raw.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=raw_cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in raw_cols})

    # per-cell gate (twoend_equal + twoend_deph grouped by param tuple, median over seeds)
    cells = defaultdict(list)
    for r in rows:
        if r["lesion"] in ("twoend_equal", "twoend_deph"):
            cells[(r["lesion"], r["sep_frac"], r["core_std"], r["core_mean"], r["drive"])].append(r)
    cell_cols = ["lesion", "sep_frac", "core_std", "core_mean", "drive", "n_seeds", "med_collision",
                 "med_neg_clean", "med_pos_clean", "med_n_events", "bidir_seed_frac", "ambiguous_rate",
                 "passed", "reason"]
    cell_rows = []
    for key, rs in sorted(cells.items(), key=lambda kv: str(kv[0])):
        lesion, sep, std, mean, drive = key
        coll, neg, pos = _med(rs, "collision_rate"), _med(rs, "neg_clean"), _med(rs, "pos_clean")
        nev, amb = _med(rs, "n_events"), _med(rs, "ambiguous")
        bidir = sum(1 for x in rs if (x["n_fwd"] or 0) > 0 and (x["n_rev"] or 0) > 0) / len(rs)
        passed, reason, flags = pilot_gate(coll, neg, pos, nev, amb, bidir_seed_frac=bidir,
                                           sign_ok=a.sign_ok)
        cell_rows.append({"lesion": lesion, "sep_frac": sep, "core_std": std, "core_mean": mean,
                          "drive": drive, "n_seeds": len(rs), "med_collision": coll,
                          "med_neg_clean": neg, "med_pos_clean": pos, "med_n_events": nev,
                          "bidir_seed_frac": round(bidir, 2), "ambiguous_rate": flags["ambiguous_rate"],
                          "passed": passed, "reason": reason})
    with open(os.path.join(out, "regime_screen_cells.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cell_cols)
        w.writeheader()
        for c in cell_rows:
            w.writerow(c)

    # stdout verdict
    sign = [r for r in rows if r["lesion"] in ("oneend_neg", "oneend_pos")]
    print(f"[aggregate] {len(rows)} runs, {len(cell_rows)} cells -> {out}/regime_screen_cells.csv")
    if sign:
        print("SIGN sanity (oneend known source):")
        for r in sorted(sign, key=lambda x: x["tag"]):
            print(f"  {r['lesion']:11} seed{r['seed']}: fwd={r['n_fwd']} rev={r['n_rev']}")
    passed = [c for c in cell_rows if c["passed"]]
    print(f"PASSING cells: {len(passed)}")
    for c in passed:
        print(f"  PASS {c['lesion']} sep{c['sep_frac']} std{c['core_std']} m{c['core_mean']} drive{c['drive']}")
    # closest non-passing twoend_equal cells (lowest collision among those that fired both ends a bit)
    near = [c for c in cell_rows if c["lesion"] == "twoend_equal" and not c["passed"]
            and (c["med_neg_clean"] or 0) > 0 and (c["med_pos_clean"] or 0) > 0]
    near.sort(key=lambda c: (c["med_collision"] if c["med_collision"] is not None else 9,
                             -(min(c["med_neg_clean"] or 0, c["med_pos_clean"] or 0))))
    print("CLOSEST near-miss cells (both ends fired, lowest collision):")
    for c in near[:5]:
        print(f"  sep{c['sep_frac']} std{c['core_std']} m{c['core_mean']} drive{c['drive']} | "
              f"coll={c['med_collision']} neg/pos={c['med_neg_clean']}/{c['med_pos_clean']} "
              f"amb_rate={c['ambiguous_rate']} reason={c['reason']}")


if __name__ == "__main__":
    main()
