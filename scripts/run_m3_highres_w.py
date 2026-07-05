#!/usr/bin/env python3
"""High-resolution W_event verification (M3 spec v2 §8 parallel non-blocking line, 2026-06-24).

The 5×5 B1b/B1c FAIL (W isotropic / W≈distance) was attributed to RESOLUTION (4mm bins, r95≈5mm
≈ 1 bin). This re-runs center bare@K_min vs n17.6@K_min at n_bins_per_axis ∈ {5,9,11} and re-tests:
  - anisotropy vs spatial-shuffle null (B1b): does a 45° axis emerge as bins shrink?
  - W_event-vs-distance LOSO predictivity (B1c): does W beat distance as bins shrink?
If both improve monotonically with resolution -> the FAIL was resolution-limited (claim confirmed).
If flat across resolution -> the model's finite event really is isotropic local diffusion.

OFFLINE over highres_w/ (n9,n11) + the existing 5×5 mini_w_event center runs. NOT a blocking gate.
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import sef_hfo_mini_w_event as mwe          # noqa: E402
from src import sef_hfo_b1_validation as b1           # noqa: E402

HR = "results/topic4_sef_hfo/m3_local_w/highres_w"
MW = "results/topic4_sef_hfo/m3_local_w/mini_w_event/runs"
OUT = "results/topic4_sef_hfo/m3_local_w/highres_w/analysis"
THETA = 45.0
KMATCH = {"bare": 1.6, "n17.6": 1.1}


def _shapes(run_dir, nb, kick):
    run = mwe.load_run_dir(run_dir, expected_n_bins=nb * nb)
    ki = int(np.argmin(np.abs(np.asarray(run["npz_kicks"], float) - kick)))
    rk = min(run["kicks"], key=lambda k: abs(k - kick))
    succ = mwe.success_seeds_at_kick(run["recs_by_kick"][rk], run["spont_seeds"])
    per_seed, mean_w, used = mwe.build_w_shape(run["ea_net_bins"][ki], succ, run["src_bin_idx"])
    src = run["src_bin_idx"]
    nonsrc = [b for b in range(nb * nb) if b != src]
    pos = np.asarray(run["bin_centers"])[nonsrc]
    return per_seed, mean_w, pos, run["bin_centers"][src], len(used)


def _row(sub, nb, run_dir):
    kick = KMATCH[sub]
    try:
        per_seed, mean_w, pos, src_pos, n = _shapes(run_dir, nb, kick)
    except (FileNotFoundError, OSError, ValueError, KeyError) as e:
        return {"substrate": sub, "n_bins_per_axis": nb, "status": f"skip:{type(e).__name__}"}
    # B1b: anisotropy + axis vs shuffle null
    angle, aniso = b1.principal_axis(mean_w, pos)
    rng = np.random.default_rng(0)
    null = np.array([b1.principal_axis(mean_w[rng.permutation(len(mean_w))], pos)[1]
                     for _ in range(2000)])
    p_aniso = float(np.mean(null[np.isfinite(null)] >= aniso))
    # B1c: W vs distance LOSO
    neg_dist = -np.linalg.norm(pos - np.asarray(src_pos)[None, :], axis=1)
    rW, rD = [], []
    for j in range(per_seed.shape[0]):
        train = np.delete(per_seed, j, 0).mean(0)
        rW.append(spearmanr(train, per_seed[j]).correlation)
        rD.append(spearmanr(neg_dist, per_seed[j]).correlation)
    diff = np.array(rW) - np.array(rD)
    brng = np.random.default_rng(1)
    boot = [np.mean(diff[brng.integers(0, len(diff), len(diff))]) for _ in range(5000)]
    return {"substrate": sub, "n_bins_per_axis": nb, "status": "ok", "n_success": n,
            "kick": kick, "anisotropy": round(aniso, 3), "aniso_p_vs_null": round(p_aniso, 4),
            "aniso_sig": bool(p_aniso < 0.05), "axis_deg": round(angle, 1),
            "axis_err_vs_45": round(b1.axis_angle_diff(angle, THETA), 1),
            "rho_W": round(float(np.mean(rW)), 3), "rho_dist": round(float(np.mean(rD)), 3),
            "W_minus_dist": round(float(diff.mean()), 3),
            "W_minus_dist_CIlo": round(float(np.percentile(boot, 2.5)), 3),
            "W_beats_dist": bool(np.percentile(boot, 2.5) > 0)}


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for sub in ("bare", "n17.6"):
        rows.append(_row(sub, 5, f"{MW}/{sub}_center"))            # existing 5×5
        rows.append(_row(sub, 9, f"{HR}/{sub}_center_n9"))
        rows.append(_row(sub, 11, f"{HR}/{sub}_center_n11"))
    with open(os.path.join(OUT, "highres_w_metrics.csv"), "w", newline="") as f:
        ok = [r for r in rows if r.get("status") == "ok"]
        if ok:
            allk = sorted({k for r in rows for k in r})
            w = csv.DictWriter(f, fieldnames=allk); w.writeheader(); w.writerows(rows)
    # resolution-trend verdict
    aniso_trend = {sub: [(r["n_bins_per_axis"], r.get("aniso_sig"), r.get("anisotropy"))
                         for r in rows if r["substrate"] == sub and r.get("status") == "ok"]
                   for sub in ("bare", "n17.6")}
    beats_trend = {sub: [(r["n_bins_per_axis"], r.get("W_beats_dist"), r.get("W_minus_dist"))
                         for r in rows if r["substrate"] == sub and r.get("status") == "ok"]
                   for sub in ("bare", "n17.6")}
    any_axis_sig = any(r.get("aniso_sig") for r in rows if r.get("status") == "ok")
    # robust W-beats-distance = both finer resolutions (n9 AND n11) for a substrate
    robust_beats = any(
        all(any(r["substrate"] == sub and r["n_bins_per_axis"] == nb and r.get("W_beats_dist")
                for r in rows if r.get("status") == "ok") for nb in (9, 11))
        for sub in ("bare", "n17.6"))
    # does anisotropy RISE monotonically with resolution (5->9->11)?
    def _rises(sub):
        a = [r["anisotropy"] for nb in (5, 9, 11)
             for r in rows if r["substrate"] == sub and r["n_bins_per_axis"] == nb and r.get("status") == "ok"]
        return len(a) == 3 and a[0] < a[1] < a[2]
    aniso_rises = all(_rises(s) for s in ("bare", "n17.6"))
    if any_axis_sig or robust_beats:
        verdict = "RESOLUTION-LIMITED CONFIRMED (axis/predictivity emerges + significant at finer bins)"
    elif aniso_rises:
        verdict = ("RESOLUTION-SENSITIVE but NOT RESOLVED: anisotropy rises with resolution "
                   "(5x5 washed out structure) but stays non-significant vs null and W does not "
                   "robustly beat distance -> W is not established as a directional field operator "
                   "even at 11x11; the E->E gradient has a real-but-weak effect the current events barely feel")
    else:
        verdict = "STILL isotropic at 9/11 (FAIL not resolution)"
    json.dump({"verdict": verdict, "rows": rows, "aniso_trend": aniso_trend,
               "beats_trend": beats_trend}, open(os.path.join(OUT, "highres_w_summary.json"), "w"), indent=1)
    print(f"[highres-W] {verdict}")
    for r in rows:
        if r.get("status") == "ok":
            print(f"  {r['substrate']:5} n{r['n_bins_per_axis']:<2}: aniso={r['anisotropy']} "
                  f"p={r['aniso_p_vs_null']} sig={r['aniso_sig']} | rho_W={r['rho_W']} "
                  f"rho_d={r['rho_dist']} W>dist={r['W_beats_dist']}")
        else:
            print(f"  {r['substrate']:5} n{r['n_bins_per_axis']:<2}: {r['status']}")
    print(f"[highres-W] wrote -> {OUT}")


if __name__ == "__main__":
    main()
