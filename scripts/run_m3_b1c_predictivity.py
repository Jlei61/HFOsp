#!/usr/bin/env python3
"""B1c — ordering predictivity (M3 mini-W_event validation, 2026-06-24).

Question: does W_event predict which bins are recruited early BETTER than pure distance from
the source? If W_event only equals distance, the finite event is just local diffusion, not a
mechanistically-informative propagation operator.

Activation TIME is not emitted (ea_net_bins is the early-window summed response), so per the
plan we use early-response RANK as the proxy for recruitment order (documented). Predictors
'local baseline rate' (per-bin) and 'K_min susceptibility map' (per-bin) are NOT available in
the current artifacts -> DATA_MISSING.md. Available comparison: W_event vs distance.

Leave-one-seed-out: train W_event = mean shape over the other success seeds; on the held-out
seed compare Spearman(W_event, observed) vs Spearman(-distance, observed); paired across folds.
Center primary (bare@1.6, n17.6@1.1), EA-local-returned successful events only. OFFLINE.
Outputs b1_validation/b1c_predictivity/.
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import sef_hfo_mini_w_event as mwe          # noqa: E402
from src import sef_hfo_b1_validation as b1           # noqa: E402

ROOT = "results/topic4_sef_hfo/m3_local_w/mini_w_event/runs"
OUT = "results/topic4_sef_hfo/m3_local_w/mini_w_event/b1_validation/b1c_predictivity"
KICK = {"bare": 1.6, "n17.6": 1.1}


def _success_shapes(run, kick):
    ki = int(np.argmin(np.abs(np.asarray(run["npz_kicks"], float) - kick)))
    rk = min(run["kicks"], key=lambda k: abs(k - kick))
    succ = mwe.success_seeds_at_kick(run["recs_by_kick"][rk], run["spont_seeds"])
    per_seed, _, used = mwe.build_w_shape(run["ea_net_bins"][ki], succ, run["src_bin_idx"])
    return per_seed, used


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    rows = []
    per_sub = {}
    for sub in ("bare", "n17.6"):
        run = mwe.load_run_dir(f"{ROOT}/{sub}_center")
        n_bins = run["ea_net_bins"].shape[2]
        src = run["src_bin_idx"]
        nonsrc = [b for b in range(n_bins) if b != src]
        pos = np.asarray(run["bin_centers"])[nonsrc]
        src_pos = np.asarray(run["bin_centers"])[src]
        neg_dist = -np.linalg.norm(pos - src_pos[None, :], axis=1)   # closer -> higher
        per_seed, used = _success_shapes(run, KICK[sub])
        n = per_seed.shape[0]
        rho_W, rho_d, topW, topd = [], [], [], []
        for j in range(n):                       # leave-one-seed-out
            train = np.delete(per_seed, j, axis=0).mean(0)
            obs = per_seed[j]
            rho_W.append(spearmanr(train, obs).correlation)
            rho_d.append(spearmanr(neg_dist, obs).correlation)
            topW.append(b1.top_k_overlap(train, obs, 3))
            topd.append(b1.top_k_overlap(neg_dist, obs, 3))
        rho_W = np.array(rho_W); rho_d = np.array(rho_d)
        diff = rho_W - rho_d
        # bootstrap CI of the mean paired difference
        rng = np.random.default_rng(0)
        boot = [np.mean(diff[rng.integers(0, n, n)]) for _ in range(5000)]
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        try:
            wp = float(wilcoxon(rho_W, rho_d).pvalue) if n >= 6 else float("nan")
        except ValueError:
            wp = float("nan")
        rows.append({
            "substrate": sub, "kick": KICK[sub], "n_seeds": n,
            "rho_W_mean": round(float(rho_W.mean()), 3), "rho_dist_mean": round(float(rho_d.mean()), 3),
            "diff_mean": round(float(diff.mean()), 3),
            "diff_CI_lo": round(ci[0], 3), "diff_CI_hi": round(ci[1], 3),
            "wilcoxon_p": round(wp, 4) if wp == wp else "nan",
            "top3_W_mean": round(float(np.mean(topW)), 3),
            "top3_dist_mean": round(float(np.mean(topd)), 3),
            "W_beats_dist": bool(ci[0] > 0),
        })
        per_sub[sub] = (rho_W, rho_d)

    with open(os.path.join(OUT, "b1c_predictivity_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # verdict: W_event must BEAT distance (paired diff CI lower bound > 0) in both substrates
    beats = [r["W_beats_dist"] for r in rows]
    close = all(abs(r["diff_mean"]) < 0.05 for r in rows)
    if all(beats):
        verdict = "PASS"
    elif any(beats):
        verdict = "WEAK"
    else:
        verdict = "FAIL"          # distance ties or beats W_event -> local diffusion
    if verdict == "FAIL" and close:
        verdict = "FAIL (W_event ≈ distance: local diffusion)"

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.3))
    x = np.arange(len(rows)); wm = 0.35
    ax.bar(x - wm / 2, [r["rho_W_mean"] for r in rows], wm, label="W_event", color="tab:green")
    ax.bar(x + wm / 2, [r["rho_dist_mean"] for r in rows], wm, label="distance", color="tab:grey")
    for i, r in enumerate(rows):
        ax.text(i, max(r["rho_W_mean"], r["rho_dist_mean"]) + 0.02,
                f"Δ={r['diff_mean']:+.3f}\nCI[{r['diff_CI_lo']:+.2f},{r['diff_CI_hi']:+.2f}]",
                ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{r['substrate']}@{r['kick']}" for r in rows])
    ax.set_ylabel("LOSO Spearman rho with held-out event"); ax.set_ylim(0, 1.05)
    ax.set_title(f"B1c — W_event vs distance predictivity — verdict {verdict}")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "b1c_W_vs_distance.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(OUT, "b1c_summary.md"), "w") as f:
        f.write("# B1c — ordering predictivity (W_event vs distance)\n\n")
        f.write(f"**Verdict: {verdict}**\n\n")
        f.write("Proxy: early-response RANK (activation time not emitted). Predictor = leave-one-"
                "seed-out W_event mean shape vs -distance from source; paired across held-out seeds.\n\n")
        for r in rows:
            f.write(f"- {r['substrate']}@{r['kick']} (n={r['n_seeds']}): rho_W={r['rho_W_mean']}, "
                    f"rho_dist={r['rho_dist_mean']}, diff={r['diff_mean']} "
                    f"CI[{r['diff_CI_lo']},{r['diff_CI_hi']}], wilcoxon p={r['wilcoxon_p']}; "
                    f"top3 W={r['top3_W_mean']} vs dist={r['top3_dist_mean']}; "
                    f"W_beats_dist={r['W_beats_dist']}\n")
        f.write("\nVerdict rule: PASS = paired-diff CI lower bound > 0 (W beats distance) in BOTH "
                "substrates; WEAK = one; FAIL = distance ties/beats. If W_event ~ distance, the "
                "finite event is LOCAL DIFFUSION, not a directional propagation operator.\n\n")
        f.write("**DATA_MISSING** (see DATA_MISSING.md): per-bin baseline rate and per-bin K_min "
                "susceptibility predictors are not in current artifacts; only W_event-vs-distance "
                "was testable.\n")
    json.dump({"verdict": verdict, "rows": rows}, open(os.path.join(OUT, "b1c_summary.json"), "w"), indent=1)

    with open(os.path.join(OUT, "DATA_MISSING.md"), "w") as f:
        f.write("# B1c — data not available in current artifacts\n\n")
        f.write("Tested: W_event vs distance (per-bin early-response rank proxy).\n\n")
        f.write("NOT testable without new runner fields:\n")
        f.write("1. **per-bin local baseline rate** — the runner emits only the DIFFERENCED "
                "ea_net_bins (kick - sham); to use 'local rate' as a predictor the runner must "
                "also emit the per-bin sham (core_only / no_core_no_kick) early-window counts.\n")
        f.write("2. **per-bin K_min / ignition susceptibility map** — only 5 source K_min values "
                "exist (center + 4 R_src=4mm); a per-bin susceptibility predictor needs each bin "
                "kicked (a dense source sweep), not just 5 sources.\n")
        f.write("3. **per-bin activation TIME** — would let order be true recruitment timing "
                "rather than early-response rank; needs the runner to emit per-bin first-spike / "
                "onset time in the event-aligned window.\n")
    print(f"[B1c] verdict={verdict}")
    for r in rows:
        print(f"  {r['substrate']}@{r['kick']}: rho_W={r['rho_W_mean']} rho_dist={r['rho_dist_mean']} "
              f"diff={r['diff_mean']} CI[{r['diff_CI_lo']},{r['diff_CI_hi']}]")
    print(f"[B1c] wrote -> {OUT}")


if __name__ == "__main__":
    main()
