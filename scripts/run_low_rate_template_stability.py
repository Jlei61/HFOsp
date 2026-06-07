#!/usr/bin/env python3
"""Low-rate-window template stability (plan 2026-06-07 task spec; the GOOD question).

For each subject, partition the recording into fixed-duration windows. In quiet (low-event)
windows the firing-COUNT ranking gets jittery. Question: does the propagation TEMPLATE axis
still reproduce the full-recording template better than count reproduces its own full ranking,
specifically in the LOW-event windows? Universe = all lagPat channels (NOT SOZ-restricted).
Reversal handled by global event-label axis alignment (no per-window winner pick).

PRIMARY (pre-specified): per subject median(template_repro - rate_repro) over LOW-event
windows; cohort one-sided Wilcoxon > 0. Window = 1h (30min sensitivity via --window-min).

HARD GUARDRAILS (task spec): no static SOZ AUC; no post-hoc window-definition cherry-picking;
no best-of-fwd/rev winner trick; report negatives as negatives; if low windows lack events to
form a template, say "this scale can't resolve it" — do not rescue.
"""
from __future__ import annotations

import argparse
import json
import sys
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
    align_template_events, window_reproductions, stratify_by_event_count, time_windows,
    count_matched_null_gap, m_bucket,
)

OUT_DIR = _ROOT / "results" / "topic4_sef_hfo" / "low_rate_template_stability"
YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
MIN_CH = 3                # >=3 channels with finite rank needed to form a template axis in a window
MIN_WINDOWS_PER_STRATUM = 2


def _subject_dir(ds, subj):
    return YUQUAN_ROOT / subj if ds == "yuquan" else EPILEPSIAE_ROOT / subj / "all_recs"


def _subject(ds, subj, window_seconds, n_null=100):
    ev = load_subject_propagation_events(_subject_dir(ds, subj))
    chan = ev["channel_names"]
    if not chan or ev["ranks"].size == 0:
        return None
    times = ev["event_abs_times"]
    if not np.all(np.isfinite(times)):
        times = np.where(np.isfinite(times), times, ev["event_rel_times"])
    masked = mask_phantom_ranks(np.asarray(ev["ranks"], float), np.asarray(ev["bools"]) > 0, normalize=True)
    labels = KMeans(n_clusters=2, n_init=5, random_state=0).fit_predict(
        build_masked_kmeans_features(ev["ranks"], np.asarray(ev["bools"]) > 0))
    aligned, align_meta = align_template_events(masked, labels)
    full_axis = np.array([np.nanmean(r) if np.any(~np.isnan(r)) else np.nan for r in aligned])
    full_count = (np.asarray(ev["bools"]) > 0).sum(axis=1).astype(float)

    n_ev = len(times)
    rng = np.random.default_rng(0)
    wins = time_windows(times, window_seconds)
    rows, dropped = [], 0
    for w in wins:
        rep = window_reproductions(aligned, ev["bools"], full_axis, full_count, w, min_ch=MIN_CH)
        if not (np.isfinite(rep["template_repro"]) and np.isfinite(rep["rate_repro"])):
            dropped += 1
            continue
        rep["gap"] = rep["template_repro"] - rep["rate_repro"]
        # count-matched null: time-scrambled same-M draws -> estimator-smoothness floor.
        # excess = observed gap - null gap = time-structured count drift beyond sampling.
        null_gap = count_matched_null_gap(aligned, ev["bools"], full_axis, full_count,
                                          rep["n_events"], n_ev, rng, n_null=n_null, min_ch=MIN_CH)
        rep["null_gap"] = null_gap
        rep["excess"] = rep["gap"] - null_gap if np.isfinite(null_gap) else float("nan")
        rep["m_bucket"] = m_bucket(rep["n_events"])
        rows.append(rep)
    if len(rows) < 3:
        return {"dataset": ds, "subject": subj, "insufficient": True,
                "reason": f"only {len(rows)} computable windows (this scale can't resolve it)",
                "n_windows_total": len(wins), "n_windows_dropped": dropped}

    counts = [r["n_events"] for r in rows]
    strata = stratify_by_event_count(counts, n_strata=3)
    for r, s in zip(rows, strata):
        r["stratum"] = s
    by_stratum = {}
    for s in ("low", "mid", "high"):
        sub = [r for r in rows if r["stratum"] == s]
        by_stratum[s] = {
            "n_windows": len(sub),
            "median_template_repro": float(np.median([r["template_repro"] for r in sub])) if sub else float("nan"),
            "median_rate_repro": float(np.median([r["rate_repro"] for r in sub])) if sub else float("nan"),
            "median_delta_template_minus_rate": float(np.median([r["template_repro"] - r["rate_repro"] for r in sub])) if sub else float("nan"),
        }
    low = [r for r in rows if r["stratum"] == "low"]
    low_exc = [r["excess"] for r in low if np.isfinite(r["excess"])]
    primary_delta = (float(np.median([r["gap"] for r in low]))
                     if len(low) >= MIN_WINDOWS_PER_STRATUM else float("nan"))
    primary_low_excess = (float(np.median(low_exc))
                          if len(low_exc) >= MIN_WINDOWS_PER_STRATUM else float("nan"))
    # M-graded (the effect is not uniform; peaks at intermediate M, unresolvable at M<=2)
    m_graded = {}
    for b in ("<=2(unresolvable)", "3-4", "5-20", "21-100", ">100"):
        sub = [r for r in rows if r["m_bucket"] == b]
        exc = [r["excess"] for r in sub if np.isfinite(r["excess"])]
        m_graded[b] = {"n": len(sub),
                       "median_gap": float(np.median([r["gap"] for r in sub])) if sub else float("nan"),
                       "median_excess": float(np.median(exc)) if exc else float("nan")}
    return {"dataset": ds, "subject": subj, "insufficient": False,
            "reversed_template": align_meta["reversed"], "centroid_corr": align_meta["centroid_corr"],
            "n_windows_total": len(wins), "n_windows_computable": len(rows), "n_windows_dropped": dropped,
            "by_stratum": by_stratum, "primary_low_delta": primary_delta,
            "primary_low_excess": primary_low_excess, "m_graded": m_graded,
            "windows": [{"n_events": r["n_events"], "template_repro": r["template_repro"],
                         "rate_repro": r["rate_repro"], "rate_repro_allch": r.get("rate_repro_allch"),
                         "n_common_channels": r.get("n_common_channels"),
                         "gap": r["gap"], "null_gap": r["null_gap"],
                         "excess": r["excess"], "stratum": r["stratum"], "m_bucket": r["m_bucket"]}
                        for r in rows]}


def _wilcoxon_gt0(vals):
    a = np.asarray([v for v in vals if v == v], dtype=float)
    try:
        return float(wilcoxon(a, alternative="greater").pvalue) if a.size and np.any(a != 0) else float("nan")
    except ValueError:
        return float("nan")


def _cohort_test(per_subject, label_sel):
    rows = [p for p in per_subject if not p.get("insufficient") and label_sel(p)
            and p["primary_low_delta"] == p["primary_low_delta"]]
    deltas = np.array([p["primary_low_delta"] for p in rows])
    # honest primary = null-corrected EXCESS (time-structured drift beyond sampling)
    exc = [p["primary_low_excess"] for p in rows if p["primary_low_excess"] == p["primary_low_excess"]]
    # per-stratum cohort medians (descriptive crossover, NOT the test statistic)
    strat = {}
    for s in ("low", "mid", "high"):
        t = [p["by_stratum"][s]["median_template_repro"] for p in rows if p["by_stratum"][s]["n_windows"]]
        r = [p["by_stratum"][s]["median_rate_repro"] for p in rows if p["by_stratum"][s]["n_windows"]]
        strat[s] = {"n": len(t), "median_template_repro": float(np.median(t)) if t else float("nan"),
                    "median_rate_repro": float(np.median(r)) if r else float("nan")}
    # M-graded cohort excess
    mg = {}
    for b in ("<=2(unresolvable)", "3-4", "5-20", "21-100", ">100"):
        e = [p["m_graded"][b]["median_excess"] for p in rows
             if p["m_graded"][b]["n"] and p["m_graded"][b]["median_excess"] == p["m_graded"][b]["median_excess"]]
        mg[b] = {"n": len(e), "median_excess": float(np.median(e)) if e else float("nan")}
    return {"n": len(rows),
            "median_low_delta_RAW": float(np.median(deltas)) if deltas.size else float("nan"),
            "n_delta_gt0": int(np.sum(deltas > 0)), "wilcoxon_p_raw_low": _wilcoxon_gt0(deltas),
            "median_low_excess_NULLCORRECTED": float(np.median(exc)) if exc else float("nan"),
            "n_excess_gt0": int(np.sum(np.asarray(exc) > 0)), "n_excess": len(exc),
            "wilcoxon_p_excess_low": _wilcoxon_gt0(exc),
            "by_stratum_cohort": strat, "m_graded_cohort_excess": mg}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-min", type=float, default=60.0, help="window duration in minutes (default 60)")
    ap.add_argument("--subjects", default="")
    args = ap.parse_args()
    window_seconds = args.window_min * 60.0

    cohort = json.loads((_ROOT / "results/topic4_sef_hfo/soz_localization/cohort.json").read_text())
    kept = cohort["kept"]
    if args.subjects:
        want = set(args.subjects.split(","))
        kept = [r for r in kept if r["subject"] in want]

    per_subject, skipped = [], []
    for rec in kept:
        try:
            res = _subject(rec["dataset"], rec["subject"], window_seconds)
        except Exception as e:  # noqa: BLE001
            skipped.append({"subject": rec["subject"], "reason": f"{type(e).__name__}: {e}"})
            print(f"  [skip {rec['subject']}] {type(e).__name__}: {e}")
            continue
        if res is None:
            skipped.append({"subject": rec["subject"], "reason": "no events"})
            continue
        per_subject.append(res)
        if res["insufficient"]:
            print(f"  {res['dataset']:<10}{res['subject']:<12} INSUFFICIENT: {res['reason']}")
        else:
            ld = res["primary_low_delta"]
            print(f"  {res['dataset']:<10}{res['subject']:<12} rev={int(res['reversed_template'])} "
                  f"win={res['n_windows_computable']:>3}(drop{res['n_windows_dropped']}) "
                  f"low Δ(tmpl-rate)={ld:+.2f}" if ld == ld else
                  f"  {res['dataset']:<10}{res['subject']:<12} low stratum too small")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "per_subject").mkdir(exist_ok=True)
    for p in per_subject:
        (OUT_DIR / "per_subject" / f"{p['dataset']}_{p['subject']}.json").write_text(
            json.dumps(p, indent=2, ensure_ascii=False))

    out = {"meta": {"analysis": "low-rate-window template stability",
                    "window_minutes": args.window_min, "min_ch": MIN_CH,
                    "universe": "all lagPat channels (NOT SOZ-restricted)",
                    "primary": "per-subject median(template_repro - rate_repro) in LOW-event windows; cohort Wilcoxon>0",
                    "guardrails": "no static SOZ AUC; no winner trick; reversal via global event-label axis alignment; negatives reported",
                    "n_subjects": len(per_subject),
                    "n_insufficient": sum(1 for p in per_subject if p["insufficient"])},
           "skipped": skipped,
           "cohort_all": _cohort_test(per_subject, lambda p: True),
           "cohort_epilepsiae": _cohort_test(per_subject, lambda p: p["dataset"] == "epilepsiae"),
           "cohort_yuquan": _cohort_test(per_subject, lambda p: p["dataset"] == "yuquan")}
    suffix = "" if args.window_min == 60.0 else f"_{int(args.window_min)}min"
    (OUT_DIR / f"cohort{suffix}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\n=== Low-rate-window template stability (window={args.window_min}min) ===")
    print("    RAW = template_repro - rate_repro in low windows; EXCESS = RAW - count-matched null (honest, time-structured)")
    for label, agg in (("ALL", out["cohort_all"]), ("EPILEPSIAE", out["cohort_epilepsiae"]), ("YUQUAN", out["cohort_yuquan"])):
        a = agg
        print(f"  {label:<11}(n={a['n']:>2}): RAW Δ={a['median_low_delta_RAW']:+.3f} ({a['n_delta_gt0']}/{a['n']}, p={a['wilcoxon_p_raw_low']:.3f}) "
              f"| EXCESS Δ={a['median_low_excess_NULLCORRECTED']:+.3f} ({a['n_excess_gt0']}/{a['n_excess']}, p={a['wilcoxon_p_excess_low']:.3f})")
        bs = a["by_stratum_cohort"]
        for s in ("low", "mid", "high"):
            print(f"      {s:<4}: template={bs[s]['median_template_repro']:.2f} rate={bs[s]['median_rate_repro']:.2f} (n={bs[s]['n']})")
        mg = a["m_graded_cohort_excess"]
        print("      M-graded excess: " + "  ".join(f"{b}:{mg[b]['median_excess']:+.2f}(n{mg[b]['n']})" for b in mg))
    print(f"\n  insufficient: {[p['subject'] for p in per_subject if p['insufficient']]}")
    print(f"  skipped: {[s['subject'] for s in skipped]}")
    print(f"\nwrote {OUT_DIR / f'cohort{suffix}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
