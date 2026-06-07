#!/usr/bin/env python3
"""DECISIVE count-matched null for the low-rate-window template-stability claim.

CONCERN: template_repro (Spearman on a bounded *mean rank* axis) may beat rate_repro
(Spearman on a small-N integer participation COUNT) simply because a bounded average is an
inherently smoother / more stable estimator than a coarse integer count at small sample size
-- i.e. a METRIC artifact, not time-structured count drift.

DISCRIMINATOR: rebuild the exact reference the claim used (aligned axis, full_axis,
full_count, via the identical runner code incl. KMeans(n_init=5, random_state=0)). Then for
each subject's LOW-stratum windows, draw N=100 RANDOM (time-scrambled) subsamples of the SAME
per-window event count M_w from the WHOLE recording, compute template_repro and rate_repro for
each via the SAME window_reproductions helper / same min_ch=3 drop rule, and form:
    rand_delta_w  = median over valid subsamples of (template_repro - rate_repro)
    random_gap    = median over low windows of rand_delta_w
    observed_gap  = primary_low_delta (from per-subject JSON; median over low windows)
Plus the paired excess (obs_delta_w - rand_delta_w) at IDENTICAL M_w, the cleanest isolation
of time-contiguity.

(a) random_gap ~= observed_gap  -> advantage is mechanistic (rank smoother than count), NOT time
(b) observed_gap >  random_gap  -> contiguous quiet windows degrade rate MORE than random -> genuine
                                    time-structured count drift on top of the estimator effect.

We do NOT match finite-channel-count per window: at fixed M, fewer finite channels in
contiguous quiet hours IS the time-structure signal; matching it would erase the test.
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

from sklearn.cluster import KMeans

from src.interictal_propagation import load_subject_propagation_events
from src.lagpat_rank_audit import mask_phantom_ranks, build_masked_kmeans_features
from src.low_rate_template_stability import align_template_events, window_reproductions

OUT_DIR = _ROOT / "results" / "topic4_sef_hfo" / "low_rate_template_stability"
PER_SUBJECT = OUT_DIR / "per_subject"
YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
MIN_CH = 3
N_DRAWS = 100

# The 8 smallest-M subjects (where the integer-count coarseness concern bites hardest).
DEFAULT_SUBJECTS = [
    ("epilepsiae", "1084"), ("epilepsiae", "442"), ("yuquan", "wangyiyang"),
    ("epilepsiae", "139"), ("yuquan", "liyouran"), ("epilepsiae", "590"),
    ("yuquan", "huanghanwen"), ("epilepsiae", "548"),
]


def _subject_dir(ds, subj):
    return YUQUAN_ROOT / subj if ds == "yuquan" else EPILEPSIAE_ROOT / subj / "all_recs"


def _build_reference(ds, subj):
    """Reconstruct EXACTLY what the runner built (run_low_rate_template_stability._subject)."""
    ev = load_subject_propagation_events(_subject_dir(ds, subj))
    bools = np.asarray(ev["bools"]) > 0
    masked = mask_phantom_ranks(np.asarray(ev["ranks"], float), bools, normalize=True)
    labels = KMeans(n_clusters=2, n_init=5, random_state=0).fit_predict(
        build_masked_kmeans_features(ev["ranks"], bools))
    aligned, align_meta = align_template_events(masked, labels)
    full_axis = np.array([np.nanmean(r) if np.any(~np.isnan(r)) else np.nan for r in aligned])
    full_count = bools.sum(axis=1).astype(float)
    n_ev = bools.shape[1]
    return aligned, bools, full_axis, full_count, n_ev, align_meta


def _random_null_for_window(aligned, bools, full_axis, full_count, n_ev, M, rng):
    """N_DRAWS random size-M subsamples from the whole recording; median (template-rate) gap
    over VALID draws (both repros finite under min_ch=3). Returns (rand_delta, n_valid)."""
    t_reps, r_reps, deltas = [], [], []
    for _ in range(N_DRAWS):
        idx = rng.choice(n_ev, size=M, replace=False)
        rep = window_reproductions(aligned, bools, full_axis, full_count, idx, min_ch=MIN_CH)
        if np.isfinite(rep["template_repro"]) and np.isfinite(rep["rate_repro"]):
            t_reps.append(rep["template_repro"])
            r_reps.append(rep["rate_repro"])
            deltas.append(rep["template_repro"] - rep["rate_repro"])
    if not deltas:
        return {"rand_delta": float("nan"), "n_valid": 0,
                "rand_template": float("nan"), "rand_rate": float("nan"),
                "delta_p05": float("nan"), "delta_p95": float("nan")}
    d = np.array(deltas)
    return {"rand_delta": float(np.median(d)), "n_valid": len(d),
            "rand_template": float(np.median(t_reps)), "rand_rate": float(np.median(r_reps)),
            "delta_p05": float(np.percentile(d, 5)), "delta_p95": float(np.percentile(d, 95))}


def _subject(ds, subj, seed):
    aligned, bools, full_axis, full_count, n_ev, align_meta = _build_reference(ds, subj)

    js = json.loads((PER_SUBJECT / f"{ds}_{subj}.json").read_text())
    low_windows = [w for w in js["windows"] if w["stratum"] == "low"]
    observed_gap = js["primary_low_delta"]

    # Faithfulness gate (two parts):
    #  (1) my rebuilt KMeans+alignment reproduce the runner's whole-recording reference:
    #      reversed_template flag AND centroid_corr must match the JSON. This is the real test
    #      that aligned/full_axis are the SAME reference the claim used.
    #  (2) JSON self-consistency: median of the JSON's stored low-window deltas == primary_low_delta.
    align_match = (bool(align_meta["reversed"]) == bool(js["reversed_template"])
                   and abs(float(align_meta["centroid_corr"]) - float(js["centroid_corr"])) < 1e-9)
    recomputed_obs_gap = float(np.median([w["template_repro"] - w["rate_repro"] for w in low_windows]))
    json_consistent = bool(abs(recomputed_obs_gap - observed_gap) < 1e-9)
    gate_ok = bool(align_match and json_consistent)

    rng = np.random.default_rng(seed)
    per_window = []
    for w in low_windows:
        M = int(w["n_events"])
        null = _random_null_for_window(aligned, bools, full_axis, full_count, n_ev, M, rng)
        obs_delta_w = w["template_repro"] - w["rate_repro"]
        per_window.append({
            "M": M,
            "obs_template": w["template_repro"], "obs_rate": w["rate_repro"],
            "obs_delta": obs_delta_w,
            "rand_delta": null["rand_delta"], "n_valid_draws": null["n_valid"],
            "rand_template": null["rand_template"], "rand_rate": null["rand_rate"],
            "rand_delta_p05": null["delta_p05"], "rand_delta_p95": null["delta_p95"],
            "excess_obs_minus_rand": (obs_delta_w - null["rand_delta"]) if null["n_valid"] else float("nan"),
        })

    rand_deltas = np.array([pw["rand_delta"] for pw in per_window if np.isfinite(pw["rand_delta"])])
    excesses = np.array([pw["excess_obs_minus_rand"] for pw in per_window
                         if np.isfinite(pw["excess_obs_minus_rand"])])
    random_gap = float(np.median(rand_deltas)) if rand_deltas.size else float("nan")
    median_excess = float(np.median(excesses)) if excesses.size else float("nan")
    # Cohort-level random gap spread: median over windows of per-window p05/p95.
    rand_p05 = np.array([pw["rand_delta_p05"] for pw in per_window if np.isfinite(pw["rand_delta_p05"])])
    rand_p95 = np.array([pw["rand_delta_p95"] for pw in per_window if np.isfinite(pw["rand_delta_p95"])])

    return {
        "dataset": ds, "subject": subj,
        "n_ev_total": int(n_ev),
        "n_low_windows": len(low_windows),
        "M_values": sorted(int(w["n_events"]) for w in low_windows),
        "M_median": int(np.median([w["n_events"] for w in low_windows])),
        "faithfulness_gate_ok": gate_ok,
        "align_match": align_match,
        "json_consistent": json_consistent,
        "recomputed_obs_gap": recomputed_obs_gap,
        "observed_gap": observed_gap,
        "random_gap": random_gap,
        "random_gap_p05_median": float(np.median(rand_p05)) if rand_p05.size else float("nan"),
        "random_gap_p95_median": float(np.median(rand_p95)) if rand_p95.size else float("nan"),
        "median_excess_obs_minus_rand": median_excess,
        "n_windows_with_valid_null": int(rand_deltas.size),
        "min_valid_draw_frac": float(min((pw["n_valid_draws"] for pw in per_window), default=0)) / N_DRAWS,
        "per_window": per_window,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="", help="comma list ds:subj; default = 8 smallest-M")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="count_matched_null.json", help="output filename under OUT_DIR")
    args = ap.parse_args()

    if args.subjects:
        subjects = [tuple(s.split(":")) for s in args.subjects.split(",")]
    else:
        subjects = DEFAULT_SUBJECTS

    results = []
    print(f"=== Count-matched null (N={N_DRAWS} random size-M draws per low window, min_ch={MIN_CH}) ===")
    print(f"{'ds':<11}{'subj':<13}{'gate':<5}{'medM':>5}{'obsGap':>9}{'randGap':>9}"
          f"{'rand[p5,p95]':>20}{'excess':>9}{'minValid':>9}")
    for i, (ds, subj) in enumerate(subjects):
        res = _subject(ds, subj, args.seed + i)
        results.append(res)
        print(f"{res['dataset']:<11}{res['subject']:<13}"
              f"{'OK' if res['faithfulness_gate_ok'] else 'FAIL':<5}"
              f"{res['M_median']:>5}{res['observed_gap']:>+9.3f}{res['random_gap']:>+9.3f}"
              f"  [{res['random_gap_p05_median']:+.2f},{res['random_gap_p95_median']:+.2f}]"
              f"{res['median_excess_obs_minus_rand']:>+9.3f}{res['min_valid_draw_frac']:>9.2f}")

    obs = np.array([r["observed_gap"] for r in results])
    rnd = np.array([r["random_gap"] for r in results])
    exc = np.array([r["median_excess_obs_minus_rand"] for r in results
                    if np.isfinite(r["median_excess_obs_minus_rand"])])
    n_obs_gt_rand = int(np.sum(obs > rnd))
    summary = {
        "meta": {
            "analysis": "count-matched (time-scrambled) null for low-rate-window template stability",
            "n_draws_per_window": N_DRAWS, "min_ch": MIN_CH, "seed": args.seed,
            "M_matching": "per-window M_w (not single median M)",
            "discriminator": "(a) random~observed => metric artifact (rank smoother than small-N count); "
                             "(b) observed>random => genuine time-structured count drift on top",
        },
        "n_subjects": len(results),
        "all_gates_ok": bool(all(r["faithfulness_gate_ok"] for r in results)),
        "cohort_median_observed_gap": float(np.median(obs)),
        "cohort_median_random_gap": float(np.median(rnd)),
        "cohort_median_excess": float(np.median(exc)) if exc.size else float("nan"),
        "n_subjects_obs_gt_rand": n_obs_gt_rand,
        "per_subject": results,
    }
    out_path = OUT_DIR / args.out
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\ncohort: median observed_gap={np.median(obs):+.3f}  median random_gap={np.median(rnd):+.3f}  "
          f"median excess={np.median(exc) if exc.size else float('nan'):+.3f}  "
          f"obs>rand in {n_obs_gt_rand}/{len(results)}")
    print(f"all faithfulness gates ok: {summary['all_gates_ok']}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
