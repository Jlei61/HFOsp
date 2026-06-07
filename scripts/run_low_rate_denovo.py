#!/usr/bin/env python3
"""De novo layer (LR-7): the HARDER complement to read-back.

Read-back (run_low_rate_template_stability.py) let each low-event window borrow the
full-recording event labels to orient its source->sink axis. De novo FORBIDS that: every
window must re-cluster and orient using ONLY its own events. The single change is the window
axis; rate, the common-channel mask, and the count-matched null structure are identical, so
the GLOBAL and DE NOVO arms are scored on the SAME index sets -> excess_global - excess_denovo
is a PAIRED contrast = the literal cost of forbidding the peek.

PRE-SPECIFIED (before the run, per advisor 2026-06-07):
  - PRIMARY de novo metric = SIGNED template recovery (apples-to-apples with the +0.131
    read-back anchor). The window axis is self-oriented by its own larger sub-cluster; a
    reverse-dominated window scores NEGATIVE (honest polarity penalty, no peeking rescues it).
  - DECOMPOSITION (reported alongside, not cherry-picked): |.| axis-line recovery (removes the
    polarity penalty) + polarity-free source-UNION-sink endpoint recovery.
  - LEAD interpretation with ABSOLUTE recovery levels (can you discover it at all), not only
    the null-corrected gap (viability is an absolute-level question).
  - The GLOBAL arm must reproduce the main read-back EXCESS (+0.131) = built-in consistency check.

A weak / null de novo does NOT negate read-back (it is a stress test, not a refutation), and a
positive de novo is still exploratory (no held-out). Framing locked by user.
"""
from __future__ import annotations

import os
# Single-thread the math libs BEFORE numpy/sklearn import. We make ~10^5 tiny k=2 KMeans calls;
# sklearn spins an OpenMP pool PER call, so multi-thread setup overhead (~29ms/call) dominated and
# the run stalled. Single-threaded each call is ~1-2ms.
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

from scipy.stats import wilcoxon
from sklearn.cluster import KMeans

from src.interictal_propagation import load_subject_propagation_events
from src.lagpat_rank_audit import mask_phantom_ranks, build_masked_kmeans_features
from src.low_rate_template_stability import (
    align_template_events, stratify_by_event_count, time_windows, m_bucket,
    window_recovery_paired, count_matched_null_gap_paired, window_endpoint_union_denovo,
)

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Number of distinct clusters")
warnings.filterwarnings("ignore", message="An input array is constant")

OUT_DIR = _ROOT / "results" / "topic4_sef_hfo" / "low_rate_template_stability"
YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
MIN_CH = 3
MIN_WINDOWS_PER_STRATUM = 2
N_NULL = 50               # paired null re-clusters every draw -> kept lighter than read-back's 100
K_ENDPOINT = 2
M_BUCKETS = ("<=2(unresolvable)", "3-4", "5-20", "21-100", ">100")


def _subject_dir(ds, subj):
    return YUQUAN_ROOT / subj if ds == "yuquan" else EPILEPSIAE_ROOT / subj / "all_recs"


def _null_m(m):
    """Representative event-count for null caching. The count-matched null gap is smooth in M,
    so we compute it exactly where the effect is sharp (M<=25) and on a coarsening grid above,
    collapsing ~150 distinct window sizes per multi-day subject to ~30 null computations."""
    if m <= 25:
        return m
    if m <= 60:
        return 5 * round(m / 5)
    if m <= 150:
        return 10 * round(m / 10)
    if m <= 400:
        return 25 * round(m / 25)
    return 100 * round(m / 100)


def _load_full(ds, subj):
    """Full-recording global alignment (read-back target) + raw arrays. SEED-INDEPENDENT
    (global KMeans is always random_state=0; only per-window de novo varies with seed), so
    each multi-day subject is loaded ONCE and reused across axis/endpoint/all seeds."""
    ev = load_subject_propagation_events(_subject_dir(ds, subj))
    chan = ev["channel_names"]
    if not chan or ev["ranks"].size == 0:
        return None
    times = ev["event_abs_times"]
    if not np.all(np.isfinite(times)):
        times = np.where(np.isfinite(times), times, ev["event_rel_times"])
    ranks = np.asarray(ev["ranks"], float)
    bools = np.asarray(ev["bools"]) > 0
    masked = mask_phantom_ranks(ranks, bools, normalize=True)
    labels = KMeans(n_clusters=2, n_init=5, random_state=0).fit_predict(
        build_masked_kmeans_features(ranks, bools))
    aligned, align_meta = align_template_events(masked, labels)
    full_axis = np.array([np.nanmean(r) if np.any(~np.isnan(r)) else np.nan for r in aligned])
    full_count = bools.sum(axis=1).astype(float)
    return dict(ds=ds, subj=subj, times=times, ranks=ranks, bools=bools, aligned=aligned,
                full_axis=full_axis, full_count=full_count, align_meta=align_meta)


def _subject(full, window_seconds, denovo_seed=0, n_null=N_NULL):
    ds, subj = full["ds"], full["subj"]
    aligned, ranks, bools = full["aligned"], full["ranks"], full["bools"]
    full_axis, full_count = full["full_axis"], full["full_count"]
    dk = {"random_state": denovo_seed}
    n_ev = len(full["times"])
    wins = time_windows(full["times"], window_seconds)

    # cache the paired null by event count M (same-M windows share the null distribution -> exact)
    null_cache = {}

    def paired_null(m):
        key = _null_m(m)
        if key not in null_cache:
            null_cache[key] = count_matched_null_gap_paired(
                aligned, ranks, bools, full_axis, full_count, key, n_ev,
                np.random.default_rng(1000 + key), n_null=n_null, min_ch=MIN_CH, **dk)
        return null_cache[key]

    rows, dropped = [], 0
    for w in wins:
        rep = window_recovery_paired(aligned, ranks, bools, full_axis, full_count, w,
                                     min_ch=MIN_CH, **dk)
        if not (np.isfinite(rep["template_repro_global"]) and
                np.isfinite(rep["template_repro_denovo_signed"]) and np.isfinite(rep["rate_repro"])):
            dropped += 1
            continue
        m = rep["n_events"]
        ng, nd, nda = paired_null(m)
        rep["gap_global"] = rep["template_repro_global"] - rep["rate_repro"]
        rep["gap_denovo_signed"] = rep["template_repro_denovo_signed"] - rep["rate_repro"]
        rep["gap_denovo_abs"] = rep["template_repro_denovo_abs"] - rep["rate_repro"]
        rep["excess_global"] = rep["gap_global"] - ng if np.isfinite(ng) else float("nan")
        rep["excess_denovo_signed"] = rep["gap_denovo_signed"] - nd if np.isfinite(nd) else float("nan")
        rep["excess_denovo_abs"] = rep["gap_denovo_abs"] - nda if np.isfinite(nda) else float("nan")
        rep["m_bucket"] = m_bucket(m)
        rows.append(rep)

    if len(rows) < 3:
        return {"dataset": ds, "subject": subj, "insufficient": True,
                "reason": f"only {len(rows)} computable windows (this scale can't resolve it)",
                "n_windows_total": len(wins), "n_windows_dropped": dropped}

    counts = [r["n_events"] for r in rows]
    for r, s in zip(rows, stratify_by_event_count(counts, 3)):
        r["stratum"] = s

    def med(key, sub):
        v = [r[key] for r in sub if np.isfinite(r.get(key, np.nan))]
        return float(np.median(v)) if v else float("nan")

    by_stratum = {}
    for s in ("low", "mid", "high"):
        sub = [r for r in rows if r["stratum"] == s]
        by_stratum[s] = {"n_windows": len(sub),
                         "median_template_global": med("template_repro_global", sub),
                         "median_template_denovo_signed": med("template_repro_denovo_signed", sub),
                         "median_template_denovo_abs": med("template_repro_denovo_abs", sub),
                         "median_rate": med("rate_repro", sub)}
    low = [r for r in rows if r["stratum"] == "low"]
    enough = len([r for r in low if np.isfinite(r["excess_denovo_signed"])]) >= MIN_WINDOWS_PER_STRATUM

    # paired peek-cost (global - de novo) per low window, then median
    peek_cost = [r["excess_global"] - r["excess_denovo_signed"] for r in low
                 if np.isfinite(r["excess_global"]) and np.isfinite(r["excess_denovo_signed"])]

    m_graded = {}
    for b in M_BUCKETS:
        sub = [r for r in rows if r["m_bucket"] == b]
        m_graded[b] = {"n": len(sub),
                       "median_excess_denovo_signed": med("excess_denovo_signed", sub),
                       "median_excess_denovo_abs": med("excess_denovo_abs", sub),
                       "median_excess_global": med("excess_global", sub)}

    return {"dataset": ds, "subject": subj, "insufficient": False,
            "denovo_seed": denovo_seed,
            "reversed_template": full["align_meta"]["reversed"],
            "n_windows_total": len(wins), "n_windows_computable": len(rows), "n_windows_dropped": dropped,
            "by_stratum": by_stratum,
            "primary_low_excess_denovo_signed": med("excess_denovo_signed", low) if enough else float("nan"),
            "low_excess_denovo_abs": med("excess_denovo_abs", low) if enough else float("nan"),
            "low_excess_global": med("excess_global", low) if enough else float("nan"),
            "low_gap_denovo_signed_RAW": med("gap_denovo_signed", low) if enough else float("nan"),
            "peek_cost_low": float(np.median(peek_cost)) if len(peek_cost) >= MIN_WINDOWS_PER_STRATUM else float("nan"),
            "m_graded": m_graded,
            "windows": [{k: r.get(k) for k in
                         ("n_events", "template_repro_global", "template_repro_denovo_signed",
                          "template_repro_denovo_abs", "rate_repro", "n_common_channels",
                          "excess_global", "excess_denovo_signed", "excess_denovo_abs",
                          "stratum", "m_bucket")} for r in rows]}


def _endpoint_subject(full, window_seconds, denovo_seed=0, n_null=40):
    """Secondary: polarity-free source-UNION-sink endpoint recovery (de novo). M-cached null."""
    ds, subj = full["ds"], full["subj"]
    aligned, ranks, bools = full["aligned"], full["ranks"], full["bools"]
    full_axis, full_count = full["full_axis"], full["full_count"]
    dk = {"random_state": denovo_seed}
    n_ev = len(full["times"])
    wins = time_windows(full["times"], window_seconds)

    null_cache = {}

    def endpoint_null(m):
        key = _null_m(m)
        if key not in null_cache:
            rng = np.random.default_rng(2000 + key)
            ng = []
            for _ in range(n_null):
                evn = rng.choice(n_ev, size=min(key, n_ev), replace=False)
                rn = window_endpoint_union_denovo(aligned, ranks, bools, full_axis, full_count, evn,
                                                  k=K_ENDPOINT, min_ch=MIN_CH, **dk)
                if np.isfinite(rn["endpoint_union_jaccard"]) and np.isfinite(rn["rate_topk_jaccard"]):
                    ng.append(rn["endpoint_union_jaccard"] - rn["rate_topk_jaccard"])
            null_cache[key] = float(np.median(ng)) if ng else float("nan")
        return null_cache[key]

    rows = []
    for w in wins:
        rep = window_endpoint_union_denovo(aligned, ranks, bools, full_axis, full_count, w,
                                           k=K_ENDPOINT, min_ch=MIN_CH, **dk)
        if not (np.isfinite(rep["endpoint_union_jaccard"]) and np.isfinite(rep["rate_topk_jaccard"])):
            continue
        gap = rep["endpoint_union_jaccard"] - rep["rate_topk_jaccard"]
        null_gap = endpoint_null(int(len(w)))
        rows.append({"n_events": int(len(w)), "gap": gap,
                     "excess": gap - null_gap if np.isfinite(null_gap) else float("nan")})
    if len(rows) < 3:
        return {"dataset": ds, "subject": subj, "insufficient": True}
    for r, s in zip(rows, stratify_by_event_count([r["n_events"] for r in rows], 3)):
        r["stratum"] = s
    low = [r for r in rows if r["stratum"] == "low"]
    le = [r["excess"] for r in low if np.isfinite(r["excess"])]
    lg = [r["gap"] for r in low]
    return {"dataset": ds, "subject": subj, "insufficient": False,
            "low_endpoint_excess": float(np.median(le)) if len(le) >= MIN_WINDOWS_PER_STRATUM else float("nan"),
            "low_endpoint_gap_RAW": float(np.median(lg)) if len(low) >= MIN_WINDOWS_PER_STRATUM else float("nan")}


def _wilcoxon_gt0(vals):
    a = np.asarray([v for v in vals if v == v], dtype=float)
    try:
        return float(wilcoxon(a, alternative="greater").pvalue) if a.size and np.any(a != 0) else float("nan")
    except ValueError:
        return float("nan")


def _cohort(per_subject, sel):
    rows = [p for p in per_subject if not p.get("insufficient") and sel(p)]
    def col(key):
        return np.asarray([p[key] for p in rows if p.get(key) == p.get(key)], float)
    primary = col("primary_low_excess_denovo_signed")
    strat = {}
    for s in ("low", "mid", "high"):
        sub = [p for p in rows if p["by_stratum"][s]["n_windows"]]
        strat[s] = {k: float(np.median([p["by_stratum"][s][k] for p in sub
                                        if p["by_stratum"][s][k] == p["by_stratum"][s][k]]))
                    if sub else float("nan")
                    for k in ("median_template_global", "median_template_denovo_signed",
                              "median_template_denovo_abs", "median_rate")}
        strat[s]["n"] = len(sub)
    mg = {}
    for b in M_BUCKETS:
        e = [p["m_graded"][b]["median_excess_denovo_signed"] for p in rows
             if p["m_graded"][b]["n"] and p["m_graded"][b]["median_excess_denovo_signed"]
             == p["m_graded"][b]["median_excess_denovo_signed"]]
        mg[b] = {"n": len(e), "median_excess_denovo_signed": float(np.median(e)) if e else float("nan")}
    return {"n": len(rows),
            "PRIMARY_median_low_excess_denovo_signed": float(np.median(primary)) if primary.size else float("nan"),
            "n_primary_gt0": int(np.sum(primary > 0)), "n_primary": int(primary.size),
            "wilcoxon_p_primary": _wilcoxon_gt0(primary),
            "median_low_excess_denovo_abs": float(np.median(col("low_excess_denovo_abs"))) if col("low_excess_denovo_abs").size else float("nan"),
            "median_low_excess_global_CONSISTENCY": float(np.median(col("low_excess_global"))) if col("low_excess_global").size else float("nan"),
            "median_low_gap_denovo_signed_RAW": float(np.median(col("low_gap_denovo_signed_RAW"))) if col("low_gap_denovo_signed_RAW").size else float("nan"),
            "median_peek_cost_low": float(np.median(col("peek_cost_low"))) if col("peek_cost_low").size else float("nan"),
            "n_peek_cost_gt0": int(np.sum(col("peek_cost_low") > 0)),
            "by_stratum_cohort": strat, "m_graded_cohort": mg}


def _endpoint_cohort(per, sel):
    rows = [p for p in per if not p.get("insufficient") and sel(p)]
    exc = np.asarray([p["low_endpoint_excess"] for p in rows if p["low_endpoint_excess"] == p["low_endpoint_excess"]], float)
    raw = np.asarray([p["low_endpoint_gap_RAW"] for p in rows if p["low_endpoint_gap_RAW"] == p["low_endpoint_gap_RAW"]], float)
    return {"n": len(rows),
            "median_low_endpoint_excess": float(np.median(exc)) if exc.size else float("nan"),
            "n_excess_gt0": int(np.sum(exc > 0)), "n_excess": int(exc.size),
            "wilcoxon_p_excess": _wilcoxon_gt0(exc),
            "median_low_endpoint_gap_RAW": float(np.median(raw)) if raw.size else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-min", type=float, default=60.0)
    ap.add_argument("--subjects", default="")
    ap.add_argument("--seeds", default="0,1,2", help="de novo KMeans seeds for headline robustness")
    args = ap.parse_args()
    window_seconds = args.window_min * 60.0
    seeds = [int(s) for s in args.seeds.split(",")]

    cohort = json.loads((_ROOT / "results/topic4_sef_hfo/soz_localization/cohort.json").read_text())
    kept = cohort["kept"]
    if args.subjects:
        want = set(args.subjects.split(","))
        kept = [r for r in kept if r["subject"] in want]

    # load each subject ONCE (multi-day epilepsiae loads are expensive); reuse across
    # axis + endpoint + all seeds. primary run = first seed (full per-subject output).
    primary_seed = seeds[0]
    per_subject, per_endpoint, skipped = [], [], []
    per_subject_by_seed = {sd: [] for sd in seeds}
    for rec in kept:
        ds, subj = rec["dataset"], rec["subject"]
        try:
            full = _load_full(ds, subj)
            if full is None:
                skipped.append({"subject": subj, "reason": "no events"})
                continue
            res = _subject(full, window_seconds, denovo_seed=primary_seed)
            eres = _endpoint_subject(full, window_seconds, denovo_seed=primary_seed)
            per_subject_by_seed[primary_seed].append(res)
            for sd in seeds:
                if sd != primary_seed:
                    per_subject_by_seed[sd].append(_subject(full, window_seconds, denovo_seed=sd))
        except Exception as e:  # noqa: BLE001
            skipped.append({"subject": subj, "reason": f"{type(e).__name__}: {e}"})
            print(f"  [skip {subj}] {type(e).__name__}: {e}", flush=True)
            continue
        per_subject.append(res)
        if eres is not None:
            per_endpoint.append(eres)
        if res["insufficient"]:
            print(f"  {ds:<10}{subj:<12} INSUFFICIENT: {res['reason']}", flush=True)
        else:
            print(f"  {ds:<10}{subj:<12} rev={int(res['reversed_template'])} "
                  f"win={res['n_windows_computable']:>3} denovo_signed_excess={res['primary_low_excess_denovo_signed']:+.2f} "
                  f"abs={res['low_excess_denovo_abs']:+.2f} global={res['low_excess_global']:+.2f} peek_cost={res['peek_cost_low']:+.2f}", flush=True)

    # seed robustness: cohort PRIMARY signed excess at each seed (reuses already-scored per-seed lists)
    seed_robustness = {}
    for sd in seeds:
        c = _cohort(per_subject_by_seed[sd], lambda p: True)
        seed_robustness[str(sd)] = {"median_primary": c["PRIMARY_median_low_excess_denovo_signed"],
                                    "n_gt0": c["n_primary_gt0"], "n": c["n_primary"],
                                    "wilcoxon_p": c["wilcoxon_p_primary"]}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "per_subject_denovo").mkdir(exist_ok=True)
    for p in per_subject:
        (OUT_DIR / "per_subject_denovo" / f"{p['dataset']}_{p['subject']}.json").write_text(
            json.dumps(p, indent=2, ensure_ascii=False))

    out = {"meta": {"analysis": "de novo short-window template discovery (LR-7)",
                    "window_minutes": args.window_min, "min_ch": MIN_CH, "n_null": N_NULL,
                    "primary_metric": "SIGNED de novo template recovery excess in LOW-event windows (pre-specified)",
                    "decomposition": "|.| axis-line excess + polarity-free endpoint-union (reported alongside)",
                    "paired_contrast": "peek_cost = excess_global - excess_denovo_signed (cost of forbidding the read-back peek)",
                    "consistency": "median_low_excess_global should reproduce the read-back main result (+0.131)",
                    "framing": "stress test of read-back, NOT a refutation; exploratory (no held-out)",
                    "denovo_seeds": seeds, "n_subjects": len(per_subject)},
           "skipped": skipped,
           "seed_robustness": seed_robustness,
           "cohort_all": _cohort(per_subject, lambda p: True),
           "cohort_epilepsiae": _cohort(per_subject, lambda p: p["dataset"] == "epilepsiae"),
           "cohort_yuquan": _cohort(per_subject, lambda p: p["dataset"] == "yuquan"),
           "endpoint_union_secondary": {
               "note": "polarity-free source-UNION-sink endpoint recovery (de novo) vs rate top-2k",
               "cohort_all": _endpoint_cohort(per_endpoint, lambda p: True),
               "cohort_epilepsiae": _endpoint_cohort(per_endpoint, lambda p: p["dataset"] == "epilepsiae"),
               "cohort_yuquan": _endpoint_cohort(per_endpoint, lambda p: p["dataset"] == "yuquan"),
               "per_subject": per_endpoint}}
    suffix = "" if args.window_min == 60.0 else f"_{int(args.window_min)}min"
    (OUT_DIR / f"cohort_denovo{suffix}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\n=== De novo short-window template discovery (window={args.window_min}min) ===")
    print("    LEAD = absolute recovery levels (can you discover it?); EXCESS = null-corrected gap vs rate")
    for label, agg in (("ALL", out["cohort_all"]), ("EPILEPSIAE", out["cohort_epilepsiae"]), ("YUQUAN", out["cohort_yuquan"])):
        a = agg
        bs = a["by_stratum_cohort"]
        print(f"\n  {label} (n={a['n']}):")
        for s in ("low", "mid", "high"):
            print(f"      {s:<4}: global={bs[s]['median_template_global']:.2f} "
                  f"denovo_signed={bs[s]['median_template_denovo_signed']:.2f} "
                  f"denovo_abs={bs[s]['median_template_denovo_abs']:.2f} rate={bs[s]['median_rate']:.2f} (n={bs[s]['n']})")
        print(f"      PRIMARY de novo SIGNED excess (low): {a['PRIMARY_median_low_excess_denovo_signed']:+.3f} "
              f"({a['n_primary_gt0']}/{a['n_primary']}, p={a['wilcoxon_p_primary']:.3f})")
        print(f"      decomposition: |.|-abs excess={a['median_low_excess_denovo_abs']:+.3f}  "
              f"global(consistency, ~+0.131)={a['median_low_excess_global_CONSISTENCY']:+.3f}  "
              f"peek_cost(global-denovo)={a['median_peek_cost_low']:+.3f}")
        mg = a["m_graded_cohort"]
        print("      M-graded de novo signed excess: " +
              "  ".join(f"{b}:{mg[b]['median_excess_denovo_signed']:+.2f}(n{mg[b]['n']})" for b in mg))
    print("\n  seed robustness (cohort PRIMARY signed excess):")
    for sd, r in seed_robustness.items():
        print(f"      seed {sd}: {r['median_primary']:+.3f} ({r['n_gt0']}/{r['n']}, p={r['wilcoxon_p']:.3f})")
    e = out["endpoint_union_secondary"]["cohort_all"]
    print(f"\n  endpoint-union secondary (ALL n={e['n']}): RAW {e['median_low_endpoint_gap_RAW']:+.3f} | "
          f"EXCESS {e['median_low_endpoint_excess']:+.3f} ({e['n_excess_gt0']}/{e['n_excess']}, p={e['wilcoxon_p_excess']:.3f})")
    print(f"\n  insufficient: {[p['subject'] for p in per_subject if p['insufficient']]}")
    print(f"  skipped: {[s['subject'] for s in skipped]}")
    print(f"\nwrote {OUT_DIR / f'cohort_denovo{suffix}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
