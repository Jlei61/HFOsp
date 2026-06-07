#!/usr/bin/env python3
"""Direction-agnostic endpoint stability: can a short window re-discover the
full-recording endpoint channels (first + last to fire)?

Primary question: KMeans k=2 on the window's events → union of cluster extremes →
Jaccard vs full-recording endpoints. No direction disambiguation needed.

Rate = operational reference only.
Count-matched null subtracts the random-draw floor.
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

from scipy.stats import wilcoxon
from sklearn.cluster import KMeans

from src.interictal_propagation import load_subject_propagation_events
from src.lagpat_rank_audit import mask_phantom_ranks, build_masked_kmeans_features
from src.low_rate_template_stability import (
    align_template_events, time_windows, stratify_by_event_count, m_bucket,
    window_endpoint_stability_denovo,
)

OUT_DIR = _ROOT / "results" / "topic4_sef_hfo" / "low_rate_template_stability"
YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
K = 2          # top-k source + top-k sink = 4 endpoint channels total
WINDOW_S = 3600.0
N_NULL = 50
MIN_WINDOWS = 3


def _subject_dir(ds, subj):
    return YUQUAN_ROOT / subj if ds == "yuquan" else EPILEPSIAE_ROOT / subj / "all_recs"


def _null_m(m):
    if m <= 25: return m
    if m <= 60: return 5 * round(m / 5)
    if m <= 150: return 10 * round(m / 10)
    if m <= 400: return 25 * round(m / 25)
    return 100 * round(m / 100)


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


def _subject(full):
    ds, subj = full["ds"], full["subj"]
    ranks, bools = full["ranks"], full["bools"]
    full_axis, full_count = full["full_axis"], full["full_count"]
    n_ev = len(full["times"])
    wins = time_windows(full["times"], WINDOW_S)

    null_cache = {}
    def null_endpoint(m):
        key = _null_m(m)
        if key not in null_cache:
            rng = np.random.default_rng(3000 + key)
            ng = []
            for _ in range(N_NULL):
                ev = rng.choice(n_ev, size=min(key, n_ev), replace=False)
                r = window_endpoint_stability_denovo(ranks, bools, full_axis, full_count, ev, k=K)
                if np.isfinite(r["endpoint_jaccard"]):
                    ng.append(r["endpoint_jaccard"] - r["rate_topk_jaccard"])
            null_cache[key] = float(np.median(ng)) if ng else float("nan")
        return null_cache[key]

    rows, dropped = [], 0
    for w in wins:
        r = window_endpoint_stability_denovo(ranks, bools, full_axis, full_count, w, k=K)
        if not np.isfinite(r["endpoint_jaccard"]):
            dropped += 1; continue
        gap = r["endpoint_jaccard"] - r["rate_topk_jaccard"]
        null_gap = null_endpoint(r["n_events"])
        rows.append({**r, "gap": gap,
                     "excess": gap - null_gap if np.isfinite(null_gap) else float("nan"),
                     "m_bucket": m_bucket(r["n_events"])})

    if len(rows) < MIN_WINDOWS:
        return {"dataset": ds, "subject": subj, "insufficient": True,
                "reason": f"only {len(rows)} computable windows"}

    for r, s in zip(rows, stratify_by_event_count([r["n_events"] for r in rows], 3)):
        r["stratum"] = s

    def med(key, sub): return float(np.median([r[key] for r in sub if np.isfinite(r.get(key, np.nan))])) if sub else float("nan")

    by_s = {}
    for s in ("low", "mid", "high"):
        sub = [r for r in rows if r["stratum"] == s]
        by_s[s] = {"n": len(sub),
                   "endpoint_jaccard": med("endpoint_jaccard", sub),
                   "rate_topk_jaccard": med("rate_topk_jaccard", sub),
                   "excess": med("excess", sub)}
    low = [r for r in rows if r["stratum"] == "low"]
    low_exc = [r["excess"] for r in low if np.isfinite(r["excess"])]
    m_graded = {}
    for b in ("<=2(unresolvable)", "3-4", "5-20", "21-100", ">100"):
        sub = [r for r in rows if r["m_bucket"] == b]
        exc = [r["excess"] for r in sub if np.isfinite(r["excess"])]
        m_graded[b] = {"n": len(sub),
                       "endpoint_jaccard": med("endpoint_jaccard", sub),
                       "rate_topk_jaccard": med("rate_topk_jaccard", sub),
                       "excess": float(np.median(exc)) if exc else float("nan")}
    return {"dataset": ds, "subject": subj, "insufficient": False, "k": K,
            "n_windows_total": len(wins), "n_windows_computable": len(rows), "n_dropped": dropped,
            "by_stratum": by_s,
            "primary_low_endpoint_jaccard": by_s["low"]["endpoint_jaccard"],
            "primary_low_excess": med("excess", low) if len(low_exc) >= 2 else float("nan"),
            "m_graded": m_graded,
            "windows": [{"n_events": r["n_events"], "endpoint_jaccard": r["endpoint_jaccard"],
                         "rate_topk_jaccard": r["rate_topk_jaccard"], "gap": r["gap"],
                         "excess": r["excess"], "stratum": r["stratum"], "m_bucket": r["m_bucket"]}
                        for r in rows]}


def _wilcoxon_gt0(vals):
    a = np.asarray([v for v in vals if v == v], float)
    try: return float(wilcoxon(a, alternative="greater").pvalue) if a.size and np.any(a != 0) else float("nan")
    except ValueError: return float("nan")


def _cohort(per, sel):
    rows = [p for p in per if not p.get("insufficient") and sel(p)]
    def col(key): return np.asarray([p[key] for p in rows if p.get(key) == p.get(key)], float)
    ej = col("primary_low_endpoint_jaccard"); exc = col("primary_low_excess")
    by_s = {}
    for s in ("low", "mid", "high"):
        ej_s = [p["by_stratum"][s]["endpoint_jaccard"] for p in rows if p["by_stratum"][s]["n"]]
        rt_s = [p["by_stratum"][s]["rate_topk_jaccard"] for p in rows if p["by_stratum"][s]["n"]]
        by_s[s] = {"n": len(ej_s),
                   "median_endpoint_jaccard": float(np.median(ej_s)) if ej_s else float("nan"),
                   "median_rate_topk_jaccard": float(np.median(rt_s)) if rt_s else float("nan")}
    mg = {}
    for b in ("<=2(unresolvable)", "3-4", "5-20", "21-100", ">100"):
        e = [p["m_graded"][b]["excess"] for p in rows if p["m_graded"][b]["n"]
             and p["m_graded"][b]["excess"] == p["m_graded"][b]["excess"]]
        mg[b] = {"n": len(e), "median_excess": float(np.median(e)) if e else float("nan")}
    return {"n": len(rows),
            "median_low_endpoint_jaccard": float(np.median(ej)) if ej.size else float("nan"),
            "median_low_excess": float(np.median(exc)) if exc.size else float("nan"),
            "n_excess_gt0": int(np.sum(exc > 0)), "n_excess": int(exc.size),
            "wilcoxon_p_excess": _wilcoxon_gt0(exc),
            "wilcoxon_p_jaccard_gt0": _wilcoxon_gt0(ej),
            "by_stratum": by_s, "m_graded": mg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="")
    args = ap.parse_args()

    cohort = json.loads((_ROOT / "results/topic4_sef_hfo/soz_localization/cohort.json").read_text())
    kept = cohort["kept"]
    if args.subjects:
        want = set(args.subjects.split(","))
        kept = [r for r in kept if r["subject"] in want]

    per, skipped = [], []
    for rec in kept:
        try:
            full = _load(rec["dataset"], rec["subject"])
            if full is None: skipped.append({"subject": rec["subject"], "reason": "no events"}); continue
            res = _subject(full)
        except Exception as e:
            skipped.append({"subject": rec["subject"], "reason": f"{type(e).__name__}: {e}"})
            print(f"  [skip {rec['subject']}] {e}", flush=True); continue
        per.append(res)
        if res["insufficient"]:
            print(f"  {res['dataset']:<10}{res['subject']:<12} INSUFFICIENT", flush=True)
        else:
            ej = res["primary_low_endpoint_jaccard"]; exc = res["primary_low_excess"]
            print(f"  {res['dataset']:<10}{res['subject']:<12} "
                  f"low endpoint_J={ej:+.2f} rate_J={res['by_stratum']['low']['rate_topk_jaccard']:+.2f} "
                  f"excess={exc:+.2f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "per_subject_endpoint_stability").mkdir(exist_ok=True)
    for p in per:
        (OUT_DIR / "per_subject_endpoint_stability" / f"{p['dataset']}_{p['subject']}.json"
         ).write_text(json.dumps(p, indent=2, ensure_ascii=False))

    out = {"meta": {"analysis": "direction-agnostic endpoint stability (LR-7 redesign)",
                    "scientific_question": "Can a short window re-discover the full-recording "
                        "endpoint channels (first + last to fire)?",
                    "method": "KMeans k=2 on window events -> union of cluster extremes -> "
                        "Jaccard vs full-recording endpoints. No direction disambiguation.",
                    "rate_role": "operational reference (top-2k by count vs same full endpoints)",
                    "k_endpoint": K, "window_s": WINDOW_S, "n_null": N_NULL},
           "skipped": skipped,
           "cohort_all": _cohort(per, lambda p: True),
           "cohort_epilepsiae": _cohort(per, lambda p: p["dataset"] == "epilepsiae"),
           "cohort_yuquan": _cohort(per, lambda p: p["dataset"] == "yuquan"),
           "per_subject": [{"dataset": p["dataset"], "subject": p["subject"],
                            "insufficient": p.get("insufficient", False),
                            "primary_low_endpoint_jaccard": p.get("primary_low_endpoint_jaccard"),
                            "primary_low_excess": p.get("primary_low_excess"),
                            "by_stratum": p.get("by_stratum", {})}
                           for p in per]}
    (OUT_DIR / "cohort_endpoint_stability.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\n=== Direction-agnostic endpoint stability ===")
    print("    Primary: endpoint_jaccard (window union-of-cluster-extremes vs full-recording endpoints)")
    print("    Rate = operational reference (top-2k by count vs same full endpoints)")
    for label, agg in (("ALL", out["cohort_all"]), ("EPI", out["cohort_epilepsiae"]), ("YUQ", out["cohort_yuquan"])):
        a = agg; bs = a["by_stratum"]
        print(f"\n  {label} (n={a['n']}):")
        for s in ("low", "mid", "high"):
            print(f"    {s:<4}: endpoint_J={bs[s]['median_endpoint_jaccard']:.2f} "
                  f"rate_J={bs[s]['median_rate_topk_jaccard']:.2f} (n={bs[s]['n']})")
        print(f"    low EXCESS (endpoint - null): {a['median_low_excess']:+.3f} "
              f"({a['n_excess_gt0']}/{a['n_excess']}, p={a['wilcoxon_p_excess']:.3f})")
        mg = a["m_graded"]
        print("    M-graded excess: " + "  ".join(f"{b}:{mg[b]['median_excess']:+.2f}(n{mg[b]['n']})" for b in mg))
    print(f"\n  insufficient: {[p['subject'] for p in per if p.get('insufficient')]}")
    print(f"  wrote {OUT_DIR / 'cohort_endpoint_stability.json'}")


if __name__ == "__main__":
    main()
