#!/usr/bin/env python3
"""SEF-HFO MAIN analysis (plan v4): SOZ-internal source-rank stability over sampling.

Within each subject's SOZ∩U, does the propagation SOURCE top-k stay closer to the
full-recording target than the firing-RATE top-k under short event budgets? A
count-matched null (random-M events) separates sampling noise from real temporal drift.

Consumes cohort.json (frozen universe + soz_in_universe). masked-only per-event ranks.
Reports rate vs source separately for epilepsiae (primary) and yuquan.
Usage: python scripts/run_sef_hfo_soz_reliability.py [--subjects sub1,sub2] [--quick]
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

from src.sef_hfo_soz_localization import soz_internal_source_stability, _READOUT_SPECS
from src.interictal_propagation import load_subject_propagation_events
from src.lagpat_rank_audit import build_masked_kmeans_features
from sklearn.cluster import KMeans

GEOM_READOUTS = ["source", "sink", "endpoint", "source_fwd", "source_rev"]  # vs rate

OUT_DIR = _ROOT / "results" / "topic4_sef_hfo" / "soz_localization"
YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")

MIN_SOZ_U = 3           # need >=3 SOZ-active channels for a within-SOZ ranking
M_GRID = [50, 100, 200, 500]
N_NULL = 200
N_STARTS = 12


def _subject_dir(dataset: str, subject: str) -> Path:
    if dataset == "yuquan":
        return YUQUAN_ROOT / subject
    return EPILEPSIAE_ROOT / subject / "all_recs"


def _k_for(n_soz_u: int) -> int:
    """The 'few leading source channels': leading-half capped at 3."""
    return max(1, min(3, n_soz_u // 2))


def _subject_stability(rec: dict, m_grid, n_null, n_starts) -> dict | None:
    soz_u = rec["soz_in_universe"]
    if len(soz_u) < MIN_SOZ_U:
        return None
    ev = load_subject_propagation_events(_subject_dir(rec["dataset"], rec["subject"]))
    chan = ev["channel_names"]
    if not chan or ev["ranks"].size == 0:
        return None
    # keep only soz_u channels actually present in the loaded event contract
    soz_u = [c for c in soz_u if c in chan]
    if len(soz_u) < MIN_SOZ_U:
        return None
    times = ev["event_abs_times"]
    if not np.all(np.isfinite(times)):  # need event times to window; fall back to rel
        times = np.where(np.isfinite(times), times, ev["event_rel_times"])
    # per-event template labels (k=2) for reversal-robust source_fwd/source_rev (S2)
    feats = build_masked_kmeans_features(ev["ranks"], np.asarray(ev["bools"]) > 0)
    labels = KMeans(n_clusters=2, n_init=5, random_state=0).fit_predict(feats)
    k = _k_for(len(soz_u))
    res = soz_internal_source_stability(
        ev["ranks"], ev["bools"], times, chan, soz_u,
        M_grid=m_grid, k=k, n_null=n_null, n_starts=n_starts, seed=0,
        labels=labels, fwd_id=0, rev_id=1)
    res.update({"dataset": rec["dataset"], "subject": rec["subject"],
                "n_soz_u": len(soz_u), "k": k, "total_hours": rec["total_hours"],
                "soz_coverage": rec["soz_coverage"]})
    return res


def _paired_geom_vs_rate(rows, M, readout, metric):
    """Paired (geom readout vs rate) over subjects where both are finite at budget M."""
    from scipy.stats import wilcoxon
    pairs = []
    for p in rows:
        c = p["curves"].get(M, {})
        g, rt = c.get(readout, {}).get(metric), c.get("rate", {}).get(metric)
        if g is not None and rt is not None and g == g and rt == rt:
            pairs.append((g, rt))
    if not pairs:
        return None
    g = np.array([x[0] for x in pairs]); rt = np.array([x[1] for x in pairs])
    d = g - rt
    try:
        pv = float(wilcoxon(g, rt, alternative="greater").pvalue) if np.any(d != 0) else float("nan")
    except ValueError:
        pv = float("nan")
    return {"n": len(pairs), "median_geom": float(np.median(g)), "median_rate": float(np.median(rt)),
            "median_delta": float(np.median(d)), "n_geom_ge_rate": int(np.sum(d >= 0)),
            "wilcoxon_p_geom_gt_rate": pv}


def _aggregate(per_subject, m_grid) -> dict:
    """Per readout x metric (jaccard/spearman) x M: paired geom-vs-rate + drift counts."""
    out = {}
    for M in m_grid:
        rows = [p for p in per_subject if M in p["curves"]]
        if not rows:
            continue
        block = {"n_subjects": len(rows), "by_readout": {}}
        for ro in GEOM_READOUTS:
            block["by_readout"][ro] = {
                "jaccard": _paired_geom_vs_rate(rows, M, ro, "jaccard_obs"),
                "spearman": _paired_geom_vs_rate(rows, M, ro, "spearman_obs"),
            }
        # stratified source vs rate. NOTE: the per-event KMeans label is arbitrary (0/1 = random
        # which physical template), so max(fwd,rev) is winner's-curse biased UP and mean(fwd,rev)
        # is diluted DOWN. Neither is the decisive test — see scripts/run_soz_aligned_source_test.py
        # (window-independent rank_a alignment). Both reported here only for transparency.
        from scipy.stats import wilcoxon

        def _strat_vs_rate(reducer):
            pairs = []
            for p in rows:
                c = p["curves"].get(M, {})
                cand = [c.get(r, {}).get("jaccard_obs") for r in ("source_fwd", "source_rev")]
                cand = [x for x in cand if x is not None and x == x]
                rt = c.get("rate", {}).get("jaccard_obs")
                if cand and rt is not None and rt == rt:
                    pairs.append((reducer(cand), rt))
            if not pairs:
                return None
            g = np.array([x[0] for x in pairs]); rt = np.array([x[1] for x in pairs]); d = g - rt
            try:
                pv = float(wilcoxon(g, rt, alternative="greater").pvalue) if np.any(d != 0) else float("nan")
            except ValueError:
                pv = float("nan")
            return {"n": len(pairs), "median_geom": float(np.median(g)), "median_rate": float(np.median(rt)),
                    "median_delta": float(np.median(d)), "n_geom_ge_rate": int(np.sum(d >= 0)),
                    "wilcoxon_p_geom_gt_rate": pv}

        block["stratified_source_max_SELECTION_BIASED_do_not_report"] = _strat_vs_rate(max)
        block["stratified_source_mean_diluted"] = _strat_vs_rate(lambda a: float(np.mean(a)))
        # drift counts (rate vs endpoint vs source): contiguous-window Jaccard in low tail of null
        def drift_n(ro):
            return int(np.sum([1 for p in rows
                               if p["curves"].get(M, {}).get(ro, {}).get("drift_frac", 1.0) is not None
                               and p["curves"][M][ro].get("drift_frac", 1.0) == p["curves"][M][ro].get("drift_frac", 1.0)
                               and p["curves"][M][ro]["drift_frac"] < 0.05]))
        block["drift_counts"] = {ro: drift_n(ro) for ro in ("rate", "source", "endpoint")}
        out[M] = block
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="", help="comma-separated subset (smoke test)")
    ap.add_argument("--quick", action="store_true", help="small null/grid for a fast smoke test")
    args = ap.parse_args()

    cohort = json.loads((OUT_DIR / "cohort.json").read_text())
    kept = cohort["kept"]
    if args.subjects:
        want = set(args.subjects.split(","))
        kept = [r for r in kept if r["subject"] in want]
    m_grid = [50, 100] if args.quick else M_GRID
    n_null = 30 if args.quick else N_NULL
    n_starts = 4 if args.quick else N_STARTS

    per_subject = []
    skipped = []
    for rec in kept:
        try:
            res = _subject_stability(rec, m_grid, n_null, n_starts)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip {rec['dataset']}/{rec['subject']}] {type(e).__name__}: {e}")
            skipped.append({"subject": rec["subject"], "reason": f"{type(e).__name__}: {e}"})
            continue
        if res is None:
            print(f"  [skip {rec['dataset']}/{rec['subject']}] |SOZ∩U|<{MIN_SOZ_U} or no events")
            skipped.append({"subject": rec["subject"], "reason": f"|SOZ∩U|<{MIN_SOZ_U} or no events"})
            continue
        per_subject.append(res)
        c = res["curves"].get(m_grid[0], {})
        g = lambda ro: c.get(ro, {}).get("jaccard_obs", float("nan"))
        print(f"  {res['dataset']:<10}{res['subject']:<12} nSOZ∩U={res['n_soz_u']:>2} k={res['k']} "
              f"n_ev={res['n_events']:>6}  M={m_grid[0]} Jacc: rate={g('rate'):.2f} endpoint={g('endpoint'):.2f} "
              f"source={g('source'):.2f} src_fwd={g('source_fwd'):.2f}")

    (OUT_DIR / "per_subject").mkdir(parents=True, exist_ok=True)
    for p in per_subject:
        (OUT_DIR / "per_subject" / f"reliability_{p['dataset']}_{p['subject']}.json").write_text(
            json.dumps(p, indent=2, ensure_ascii=False))

    agg_all = _aggregate(per_subject, m_grid)
    agg_epi = _aggregate([p for p in per_subject if p["dataset"] == "epilepsiae"], m_grid)
    agg_yq = _aggregate([p for p in per_subject if p["dataset"] == "yuquan"], m_grid)
    out = {"meta": {"analysis": "SOZ-internal readout stability (plan v4 main, review-fixed)",
                    "M_grid": m_grid, "n_null": n_null, "n_starts": n_starts,
                    "min_soz_u": MIN_SOZ_U, "n_subjects": len(per_subject),
                    "readouts": ["rate"] + GEOM_READOUTS,
                    "note": "geom readouts vs rate; endpoint=reversal-invariant, source/sink=reversal-sensitive, "
                            "source_fwd/rev=template-stratified; rate=participation count in group-propagation events"},
           "skipped": skipped,
           "cohort_all": agg_all, "cohort_epilepsiae_primary": agg_epi, "cohort_yuquan": agg_yq}
    suffix = "_quick" if args.quick else ""
    (OUT_DIR / f"reliability{suffix}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\n=== SOZ-internal readout stability (n={len(per_subject)}) — median Jaccard(window top-k, full top-k) vs RATE ===")
    print(f"    (each geom readout vs rate; Δ>0 = geom MORE stable than rate; p = one-sided geom>rate)")
    for label, agg in (("ALL", agg_all), ("EPILEPSIAE(primary)", agg_epi), ("YUQUAN", agg_yq)):
        print(f"  --- {label} ---")
        for M in m_grid:
            if M not in agg:
                continue
            b = agg[M]
            j = b["by_readout"]
            def fmt(ro):
                d = j[ro]["jaccard"]
                return f"{ro}:Δ{d['median_delta']:+.2f}(p{d['wilcoxon_p_geom_gt_rate']:.2f},{d['n_geom_ge_rate']}/{d['n']})" if d else f"{ro}:NA"
            bs = b.get("best_stratified_source_jaccard")
            bs_s = f"  best_strat_src:Δ{bs['median_delta']:+.2f}(p{bs['wilcoxon_p_geom_gt_rate']:.2f})" if bs else ""
            rate_med = j["endpoint"]["jaccard"]["median_rate"] if j["endpoint"]["jaccard"] else float("nan")
            print(f"    M={M:>4}(n={b['n_subjects']:>2}) rate_med={rate_med:.2f} | " +
                  "  ".join(fmt(ro) for ro in ("endpoint", "source", "sink")) + bs_s +
                  f" | drift rate/src/endp={b['drift_counts']['rate']}/{b['drift_counts']['source']}/{b['drift_counts']['endpoint']}")
    if skipped:
        print(f"\n  skipped subjects: {skipped}")
    print(f"\nwrote {OUT_DIR / f'reliability{suffix}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
