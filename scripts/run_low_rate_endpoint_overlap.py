#!/usr/bin/env python3
"""SECONDARY analysis: do the source/sink ENDPOINT SETS drift in low-event windows?

The main analysis (run_low_rate_template_stability.py) measures whole-axis ORDERING
reproduction. This complements it with the discrete ENDPOINT question (user review): full
data defines source/sink endpoints; each low-event window defines its own; compare Jaccard,
vs the analogous rate top-k Jaccard, with the same count-matched null. Reference-recovery
(window projects onto the full axis via global labels; no re-clustering). All lagPat channels.
"""
from __future__ import annotations

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
    align_template_events, time_windows, stratify_by_event_count, window_endpoint_overlaps,
)

OUT_DIR = _ROOT / "results" / "topic4_sef_hfo" / "low_rate_template_stability"
YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
K_ENDPOINT = 2          # 2 source + 2 sink endpoint channels
WINDOW_S = 3600.0
N_NULL = 100
MIN_LOW = 2


def _subject_dir(ds, subj):
    return YUQUAN_ROOT / subj if ds == "yuquan" else EPILEPSIAE_ROOT / subj / "all_recs"


def _subject(ds, subj):
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
    aligned, _ = align_template_events(masked, labels)
    full_axis = np.array([np.nanmean(r) if np.any(~np.isnan(r)) else np.nan for r in aligned])
    full_count = (np.asarray(ev["bools"]) > 0).sum(axis=1).astype(float)
    n_ev = len(times)
    rng = np.random.default_rng(0)

    rows, dropped = [], 0
    for w in time_windows(times, WINDOW_S):
        rep = window_endpoint_overlaps(aligned, ev["bools"], full_axis, full_count, w, k=K_ENDPOINT)
        if not np.isfinite(rep["endpoint_jaccard"]) or not np.isfinite(rep["rate_topk_jaccard"]):
            dropped += 1
            continue
        gap = rep["endpoint_jaccard"] - rep["rate_topk_jaccard"]
        # count-matched null on the same gap
        ng = []
        for _ in range(N_NULL):
            evn = rng.choice(n_ev, size=min(len(w), n_ev), replace=False)
            rn = window_endpoint_overlaps(aligned, ev["bools"], full_axis, full_count, evn, k=K_ENDPOINT)
            if np.isfinite(rn["endpoint_jaccard"]) and np.isfinite(rn["rate_topk_jaccard"]):
                ng.append(rn["endpoint_jaccard"] - rn["rate_topk_jaccard"])
        null_gap = float(np.median(ng)) if ng else float("nan")
        rows.append({"n_events": rep["n_events"] if "n_events" in rep else int(len(w)),
                     "endpoint_jaccard": rep["endpoint_jaccard"], "rate_topk_jaccard": rep["rate_topk_jaccard"],
                     "gap": gap, "excess": gap - null_gap if np.isfinite(null_gap) else float("nan")})
    if len(rows) < 3:
        return {"dataset": ds, "subject": subj, "insufficient": True,
                "reason": f"only {len(rows)} computable endpoint windows (this scale can't resolve it)"}
    counts = [r["n_events"] for r in rows]
    for r, s in zip(rows, stratify_by_event_count(counts, 3)):
        r["stratum"] = s
    low = [r for r in rows if r["stratum"] == "low"]
    low_exc = [r["excess"] for r in low if np.isfinite(r["excess"])]
    by = {s: {"n": sum(r["stratum"] == s for r in rows),
              "median_endpoint_jaccard": float(np.median([r["endpoint_jaccard"] for r in rows if r["stratum"] == s])) if any(r["stratum"] == s for r in rows) else float("nan"),
              "median_rate_jaccard": float(np.median([r["rate_topk_jaccard"] for r in rows if r["stratum"] == s])) if any(r["stratum"] == s for r in rows) else float("nan")}
          for s in ("low", "mid", "high")}
    return {"dataset": ds, "subject": subj, "insufficient": False, "k": K_ENDPOINT,
            "n_windows_computable": len(rows), "n_windows_dropped": dropped, "by_stratum": by,
            "primary_low_delta": float(np.median([r["gap"] for r in low])) if len(low) >= MIN_LOW else float("nan"),
            "primary_low_excess": float(np.median(low_exc)) if len(low_exc) >= MIN_LOW else float("nan")}


def _coh(per, sel):
    rows = [p for p in per if not p.get("insufficient") and sel(p)]
    exc = np.array([p["primary_low_excess"] for p in rows if p["primary_low_excess"] == p["primary_low_excess"]])
    raw = np.array([p["primary_low_delta"] for p in rows if p["primary_low_delta"] == p["primary_low_delta"]])
    def wp(a):
        try:
            return float(wilcoxon(a, alternative="greater").pvalue) if a.size and np.any(a != 0) else float("nan")
        except ValueError:
            return float("nan")
    return {"n": len(rows), "median_low_excess": float(np.median(exc)) if exc.size else float("nan"),
            "n_excess_gt0": int(np.sum(exc > 0)), "n_excess": int(exc.size), "wilcoxon_p_excess_low": wp(exc),
            "median_low_delta_RAW": float(np.median(raw)) if raw.size else float("nan"), "wilcoxon_p_raw_low": wp(raw)}


def main() -> int:
    cohort = json.loads((OUT_DIR.parent / "soz_localization" / "cohort.json").read_text())
    per = []
    for rec in cohort["kept"]:
        try:
            r = _subject(rec["dataset"], rec["subject"])
        except Exception as e:  # noqa: BLE001
            print(f"  [skip {rec['subject']}] {type(e).__name__}: {e}")
            continue
        if r is None:
            continue
        per.append(r)
        if r["insufficient"]:
            print(f"  {r['dataset']:<10}{r['subject']:<12} INSUFFICIENT: {r['reason']}")
        else:
            print(f"  {r['dataset']:<10}{r['subject']:<12} low Δ(endpt-rate)={r['primary_low_delta']:+.2f} excess={r['primary_low_excess']:+.2f}")
    out = {"meta": {"analysis": "low-rate-window source/sink ENDPOINT overlap (secondary)",
                    "k_endpoint": K_ENDPOINT, "window_s": WINDOW_S, "n_null": N_NULL,
                    "note": "endpoint = top-k source ∪ top-k sink on the full-recording axis, common channels; "
                            "vs rate top-k; null-corrected EXCESS is the honest number"},
           "cohort_all": _coh(per, lambda p: True),
           "cohort_epilepsiae": _coh(per, lambda p: p["dataset"] == "epilepsiae"),
           "cohort_yuquan": _coh(per, lambda p: p["dataset"] == "yuquan"),
           "per_subject": per}
    (OUT_DIR / "cohort_endpoint_overlap.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n=== ENDPOINT overlap secondary (k=2 source + 2 sink; EXCESS = null-corrected) ===")
    for lbl, a in (("ALL", out["cohort_all"]), ("EPI", out["cohort_epilepsiae"]), ("YUQ", out["cohort_yuquan"])):
        print(f"  {lbl:<4}(n={a['n']:>2}): RAW Δ={a['median_low_delta_RAW']:+.3f}(p{a['wilcoxon_p_raw_low']:.3f}) | "
              f"EXCESS Δ={a['median_low_excess']:+.3f} ({a['n_excess_gt0']}/{a['n_excess']}, p={a['wilcoxon_p_excess_low']:.3f})")
    print(f"\nwrote {OUT_DIR / 'cohort_endpoint_overlap.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
