#!/usr/bin/env python3
"""Event-count learning curve: how many events does de novo recovery of the full-recording
propagation structure need, and do real quiet windows fall ON that curve or BELOW it?

For each subject:
  - RANDOM-DRAW curve: draw M random events from the WHOLE recording (M = 5,10,20,50,100,...),
    n_rep times, recompute three recoveries vs the full-recording structure, take median.
    This isolates the PURE event-count effect (time scrambled).
  - REAL low-event WINDOWS: the same three recoveries on contiguous 1h windows.

Comparison the curve answers:
  - real window ON the random curve (same M, same recovery)  -> failure is pure event scarcity.
  - real window BELOW the random curve (same M, worse)        -> quiet periods themselves differ
    (state change), not just fewer events.

Three recoveries reported SEPARATELY (different questions):
  - axis_signed   : same propagation DIRECTION (which end is source) -- hardest
  - axis_abs (|p|): same propagation AXIS LINE (which channels are extremes), direction ignored
  - endpoint      : same ENDPOINT SET (first+last to fire), discrete, direction-agnostic (KMeans-union)

Rate is an OPERATIONAL REFERENCE only (rate top-k vs the same full endpoints / full count),
NOT the ground truth. Ground truth = the full-recording structure throughout.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Number of distinct clusters")
warnings.filterwarnings("ignore", message="An input array is constant")

from sklearn.cluster import KMeans

from src.interictal_propagation import load_subject_propagation_events
from src.lagpat_rank_audit import mask_phantom_ranks, build_masked_kmeans_features
from src.low_rate_template_stability import (
    align_template_events, time_windows, _spearman, denovo_window_axis,
    window_endpoint_stability_denovo,
)

OUT_DIR = _ROOT / "results" / "topic4_sef_hfo" / "low_rate_template_stability"
YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
M_GRID = [5, 10, 20, 50, 100, 200, 400, 800]
N_REP = 40
WINDOW_S = 3600.0
K = 2
MIN_CH = 3


def _subject_dir(ds, subj):
    return YUQUAN_ROOT / subj if ds == "yuquan" else EPILEPSIAE_ROOT / subj / "all_recs"


def _load(ds, subj):
    ev = load_subject_propagation_events(_subject_dir(ds, subj))
    if not ev["channel_names"] or ev["ranks"].size == 0:
        return None
    times = ev["event_abs_times"]
    if not np.all(np.isfinite(times)):
        times = np.where(np.isfinite(times), times, ev["event_rel_times"])
    ranks = np.asarray(ev["ranks"], float)
    bools = np.asarray(ev["bools"]) > 0
    masked = mask_phantom_ranks(ranks, bools, normalize=True)
    labels = KMeans(n_clusters=2, n_init=5, random_state=0).fit_predict(
        build_masked_kmeans_features(ranks, bools))
    aligned, _ = align_template_events(masked, labels)
    full_axis = np.array([np.nanmean(r) if np.any(~np.isnan(r)) else np.nan for r in aligned])
    full_count = bools.sum(axis=1).astype(float)
    return dict(ds=ds, subj=subj, times=times, ranks=ranks, bools=bools,
                full_axis=full_axis, full_count=full_count)


def _recovery(full, event_idx, seed=0):
    """Three recoveries + rate references for one event-index set."""
    ranks, bools = full["ranks"], full["bools"]
    full_axis, full_count = full["full_axis"], full["full_count"]
    ev = np.asarray(event_idx, dtype=int)

    # axis recoveries (signed direction, |.| axis line)
    den_axis = denovo_window_axis(ranks[:, ev], bools[:, ev])
    signed = _spearman(den_axis, full_axis, MIN_CH)
    # rate-axis reference: window count vs full count over common channels
    win_count = (bools[:, ev]).sum(axis=1).astype(float)
    common = np.isfinite(den_axis) & np.isfinite(full_axis)
    rate_axis = _spearman(np.where(common, win_count, np.nan),
                          np.where(common, full_count, np.nan), MIN_CH)
    # endpoint recovery (direction-agnostic KMeans-union) + rate-endpoint reference
    ep = window_endpoint_stability_denovo(ranks, bools, full_axis, full_count, ev, k=K)
    return {"axis_signed": signed,
            "axis_abs": abs(signed) if signed == signed else float("nan"),
            "endpoint": ep["endpoint_jaccard"],
            "rate_axis": rate_axis,
            "rate_endpoint": ep["rate_topk_jaccard"]}


def _median_finite(vals):
    a = [v for v in vals if v == v]
    return float(np.median(a)) if a else float("nan")


def _subject(full):
    n_ev = len(full["times"])
    rng = np.random.default_rng(0)

    # RANDOM-DRAW learning curve
    curve = {}
    for M in M_GRID:
        if M > n_ev:
            continue
        recs = [_recovery(full, rng.choice(n_ev, size=M, replace=False)) for _ in range(N_REP)]
        curve[M] = {k: _median_finite([r[k] for r in recs])
                    for k in ("axis_signed", "axis_abs", "endpoint", "rate_axis", "rate_endpoint")}

    # REAL contiguous windows
    real = []
    for w in time_windows(full["times"], WINDOW_S):
        r = _recovery(full, w)
        r["n_events"] = int(len(w))
        real.append(r)

    return {"dataset": full["ds"], "subject": full["subj"], "n_events_total": n_ev,
            "random_curve": curve, "real_windows": real}


def _cohort_curve(per):
    """Cohort median of random-draw curve at each M, and real-window medians binned by M."""
    out = {"random_curve": {}, "real_binned": {}}
    for M in M_GRID:
        rows = [p["random_curve"][M] for p in per if M in p["random_curve"]]
        if rows:
            out["random_curve"][str(M)] = {
                "n_subjects": len(rows),
                **{k: _median_finite([r[k] for r in rows])
                   for k in ("axis_signed", "axis_abs", "endpoint", "rate_axis", "rate_endpoint")}}
    # real windows binned to the same M grid (nearest log-bin edges)
    edges = [0, 7, 15, 35, 75, 150, 300, 600, 1e9]
    labels = [5, 10, 20, 50, 100, 200, 400, 800]
    for lbl, lo, hi in zip(labels, edges[:-1], edges[1:]):
        rows = [w for p in per for w in p["real_windows"] if lo < w["n_events"] <= hi]
        if rows:
            out["real_binned"][str(lbl)] = {
                "n_windows": len(rows),
                **{k: _median_finite([r[k] for r in rows])
                   for k in ("axis_signed", "axis_abs", "endpoint", "rate_axis", "rate_endpoint")}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="")
    args = ap.parse_args()

    cohort = json.loads((OUT_DIR.parent / "soz_localization" / "cohort.json").read_text())
    kept = cohort["kept"]
    if args.subjects:
        want = set(args.subjects.split(","))
        kept = [r for r in kept if r["subject"] in want]

    per, skipped = [], []
    for rec in kept:
        try:
            full = _load(rec["dataset"], rec["subject"])
            if full is None:
                skipped.append(rec["subject"]); continue
            per.append(_subject(full))
            print(f"  {rec['dataset']:<10}{rec['subject']:<12} n_ev={per[-1]['n_events_total']:>6} "
                  f"curve_M={list(per[-1]['random_curve'].keys())}", flush=True)
        except Exception as e:
            skipped.append(rec["subject"])
            print(f"  [skip {rec['subject']}] {type(e).__name__}: {e}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "per_subject_learning_curve").mkdir(exist_ok=True)
    for p in per:
        (OUT_DIR / "per_subject_learning_curve" / f"{p['dataset']}_{p['subject']}.json"
         ).write_text(json.dumps(p, indent=2, ensure_ascii=False))

    out = {"meta": {"analysis": "de novo event-count learning curve (LR-7)",
                    "question": "How many events does de novo recovery need, and do real quiet "
                                "windows fall ON the random-draw curve (pure scarcity) or BELOW it "
                                "(quiet-period state change)?",
                    "ground_truth": "full-recording propagation structure (NOT rate)",
                    "recoveries": "axis_signed (direction), axis_abs (axis line), endpoint (KMeans-union set)",
                    "rate_role": "operational reference only",
                    "M_grid": M_GRID, "n_rep": N_REP, "window_s": WINDOW_S},
           "skipped": skipped,
           "cohort_all": _cohort_curve(per),
           "cohort_epilepsiae": _cohort_curve([p for p in per if p["dataset"] == "epilepsiae"]),
           "cohort_yuquan": _cohort_curve([p for p in per if p["dataset"] == "yuquan"])}
    (OUT_DIR / "cohort_learning_curve.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("\n=== Event-count learning curve (random draw vs real quiet windows) ===")
    print("    recovery of full-recording structure; ground truth = whole recording (not rate)")
    for lbl, agg in (("ALL", out["cohort_all"]),):
        print(f"\n  {lbl}: random-draw curve (axis_signed / axis_abs / endpoint):")
        for M, d in agg["random_curve"].items():
            print(f"    M={M:>4}: signed={d['axis_signed']:+.2f} |p|={d['axis_abs']:.2f} "
                  f"endpoint={d['endpoint']:.2f}  (rate_axis={d['rate_axis']:+.2f} rate_ep={d['rate_endpoint']:.2f}, n={d['n_subjects']})")
        print(f"  {lbl}: REAL quiet windows binned by event count:")
        for M, d in agg["real_binned"].items():
            print(f"    M~{M:>4}: signed={d['axis_signed']:+.2f} |p|={d['axis_abs']:.2f} "
                  f"endpoint={d['endpoint']:.2f}  (n_win={d['n_windows']})")
    print(f"\n  wrote {OUT_DIR / 'cohort_learning_curve.json'}")


if __name__ == "__main__":
    main()
