"""V2 bridge — long-axis control + temporal evolution + local-participation (no SNN re-run; existing artifacts).

Addresses the review: the interictal contact-order template is nearly the fixed long-axis coordinate itself
(|Spearman| > 0.95), so the pre-runaway maxAB could be carried by geometry alone. This script quantifies:
  (A) axis-only maxAB (contact long-axis coord AS the template) + within-shaft null,
  (B) template maxAB - axis-only maxAB increment, and partial Spearman(template, energy | axis),
  (C) the temporal evolution of contact maxAB / p across the fixed windows, and
  (D) the per-contact energy vs local-participation Spearman (+ hottest/coldest contacts).
Writes axis_and_temporal_control.json and a 2-panel supplementary figure.
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import src.topic4_mz_early_field_bridge as B  # noqa: E402

V2 = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_early_field_bridge_v2_zm_tau500")
FIGDIR = os.path.join(ROOT, "results", "paper-ready-figure", "fig_mz_early_bridge_v2_zm_tau500", "figures")
SEEDS = (1, 3, 4)
WINDOWS = ("early_0_25_ms", "early_25_50_ms", "early_0_50_ms", "early_50_100_ms", "early_0_100_ms")
NULLCFG = dict(n_permutations=10000, seed=0, max_exact_permutations=50000)


def _shafts(names):
    return np.array([re.match(r"[A-Za-z]+", n).group(0) for n in names], object)


def _partial_spearman(x, y, z):
    """partial Spearman(x, y | z): rank all, linearly residualize x and y on z, correlate residuals."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    xr, yr, zr = rankdata(x[m]), rankdata(y[m]), rankdata(z[m])
    z1 = np.c_[np.ones_like(zr), zr]

    def resid(a):
        coef, _, _, _ = np.linalg.lstsq(z1, a, rcond=None)
        return a - z1 @ coef
    r, p = spearmanr(resid(xr), resid(yr))
    return float(r), float(p)


def analyze():
    rows = []
    for s in SEEDS:
        sd = os.path.join(V2, "per_seed", f"seed{s}")
        so = np.load(os.path.join(sd, "slowoff.npz"), allow_pickle=True)
        na = np.load(os.path.join(sd, "native.npz"), allow_pickle=True)
        tm = np.load(os.path.join(sd, "templates.npz"), allow_pickle=True)
        bm = json.load(open(os.path.join(sd, "bridge_metrics.json")))
        names = [str(x) for x in so["names"]]
        shafts = _shafts(names)
        axis = np.asarray(so["contact_axis"], float)
        energy = np.asarray(na["contact_energy__early_0_50_ms"], float)
        tB = np.asarray(tm["contact_B"], float)

        # (A) axis-only maxAB: the contact long-axis coord itself as an ordinal "template", both directions
        ar_a, ar_d = B._ordinal_rank(axis), B._ordinal_rank(-axis)
        obs_ax = B.maxab_observed(ar_a, ar_d, energy, support_a=np.isfinite(ar_a), support_b=np.isfinite(ar_d))
        null_ax = B.maxab_permutation_null(ar_a, ar_d, energy, support_a=np.isfinite(ar_a),
                                           support_b=np.isfinite(ar_d), groups=shafts, **NULLCFG)
        # (B) template maxAB + increment + template-axis corr + partial(template, energy | axis)
        ca = bm["by_window"]["early_0_50_ms"]["contact"]["all_support"]
        tmpl_maxab = ca["maxab"]["rho_maxab"]
        tmpl_p = ca["within_shaft_null"]["p_one_sided"]
        mfin = np.isfinite(tB) & np.isfinite(axis)
        corr_tmpl_axis = float(spearmanr(tB[mfin], axis[mfin]).correlation)
        partial_r, partial_p = _partial_spearman(tB, energy, axis)
        # (C) temporal windows
        windows = {}
        for wk in WINDOWS:
            w = bm["by_window"].get(wk, {}).get("contact", {}).get("all_support", {})
            lp = (bm["by_window"].get(wk, {}).get("local_participation") or {})
            windows[wk] = {"maxab": (w.get("maxab") or {}).get("rho_maxab"),
                           "within_shaft_p": (w.get("within_shaft_null") or {}).get("p_one_sided"),
                           "lp_median": lp.get("median")}
        # (D) energy vs local-participation
        lp0 = np.asarray(bm["by_window"]["early_0_50_ms"]["local_participation"]["per_contact"], float)
        mlp = np.isfinite(lp0) & np.isfinite(energy)
        elp_r = float(spearmanr(energy[mlp], lp0[mlp]).correlation)
        order = np.argsort(energy[mlp])
        rows.append({
            "seed": s,
            "template_maxab": tmpl_maxab, "template_within_shaft_p": tmpl_p,
            "axis_only_maxab": obs_ax["rho_maxab"], "axis_only_within_shaft_p": null_ax["p_one_sided"],
            "template_minus_axis": float(tmpl_maxab - obs_ax["rho_maxab"]),
            "corr_template_vs_axis": corr_tmpl_axis,
            "partial_template_energy_given_axis_r": partial_r, "partial_p": partial_p,
            "windows": windows,
            "energy_localparticipation_spearman": elp_r,
            "hot3_participation": [round(float(x), 3) for x in lp0[mlp][order[-3:]]],
            "cold3_participation": [round(float(x), 3) for x in lp0[mlp][order[:3]]],
        })
    out = {
        "experiment": "V2 bridge long-axis control + temporal evolution + local participation (no re-sim)",
        "interpretation": (
            "(A) the fixed contact long-axis coordinate ALONE predicts the pre-runaway early energy field "
            "(axis-only maxAB significant in 3/3 seeds); the interictal contact-order template is nearly the "
            "axis itself (|Spearman|>0.95) and adds only a modest maxAB increment. (B) after controlling for "
            "the axis, a residual template->energy association survives in 2/3 seeds (seed1/seed3), marginal "
            "in seed4 -> 'template beyond geometry' is partially, not cleanly, established. (C) the bridge is "
            "temporally early: strong in 0-50 ms, weaker by 50-100 ms (seed1 loses significance). (D) high-"
            "energy contacts co-occur with strong nearby recruitment (energy-participation Spearman ~0.90-0.94), "
            "but energy and participation both vary along the axis, so local exclusivity is NOT established."),
        "rows": rows,
    }
    json.dump(out, open(os.path.join(V2, "axis_and_temporal_control.json"), "w"), indent=2)
    return rows


def plot(rows):
    fig, (axB, axT) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    x = np.arange(len(SEEDS)); w = 0.38
    # Panel B: template maxAB vs axis-only maxAB
    tmpl = [r["template_maxab"] for r in rows]
    axo = [r["axis_only_maxab"] for r in rows]
    axB.bar(x - w / 2, axo, w, label="long-axis coord only", color="#9aa7b0", edgecolor="black", lw=0.6)
    axB.bar(x + w / 2, tmpl, w, label="interictal template", color="#b85450", edgecolor="black", lw=0.6)
    for i, r in enumerate(rows):
        axB.text(i, max(tmpl[i], axo[i]) + 0.015,
                 f"+{r['template_minus_axis']:.02f}\npartial p={r['partial_p']:.03f}", ha="center", va="bottom", fontsize=7)
    axB.set_xticks(x); axB.set_xticklabels([f"seed {s}" for s in SEEDS])
    axB.set_ylim(0, 1.15); axB.set_ylabel("contact maxAB (0-50 ms)")
    axB.set_title("Long-axis geometry alone already predicts;\ntemplate ≈ axis (|r|>0.95) adds a small increment", fontsize=9.5)
    axB.legend(fontsize=8, frameon=False, loc="lower center")
    # Panel T: temporal maxAB across windows + participation trend
    disp = ["0-25", "25-50", "50-100"]
    keys = ["early_0_25_ms", "early_25_50_ms", "early_50_100_ms"]
    cols = {1: "#1f77b4", 3: "#2ca02c", 4: "#d62728"}
    for r in rows:
        mv = [r["windows"][k]["maxab"] for k in keys]
        axT.plot(disp, mv, "-o", color=cols[r["seed"]], label=f"seed {r['seed']} maxAB", lw=1.6, ms=5)
        for j, k in enumerate(keys):
            p = r["windows"][k]["within_shaft_p"]
            if isinstance(p, (int, float)) and p < 0.05:
                axT.text(j, mv[j] + 0.02, "*", ha="center", va="bottom", fontsize=13, color=cols[r["seed"]])
    axT.set_ylim(0.4, 1.05); axT.set_ylabel("contact maxAB")
    axT.set_xlabel("window after t_recruit (ms)")
    axT.set_title("Early stereotyped axis readout weakens by 50-100 ms\n(* within-shaft p<0.05)", fontsize=9.5)
    axT.legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("E1146 z+m V2 — geometry control + temporal evolution (supplementary; not Figure 5; n=3 seeds, one substrate)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    png = os.path.join(FIGDIR, "fig_mz_v2_axis_temporal_supp.png")
    fig.savefig(png, dpi=170, bbox_inches="tight"); fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png} (+pdf)")


if __name__ == "__main__":
    rows = analyze()
    for r in rows:
        print(f"seed{r['seed']}: tmpl={r['template_maxab']:.3f} axis-only={r['axis_only_maxab']:.3f}(p={r['axis_only_within_shaft_p']:.4f}) "
              f"incr=+{r['template_minus_axis']:.3f} corr(tmpl,axis)={r['corr_template_vs_axis']:.3f} "
              f"partial r={r['partial_template_energy_given_axis_r']:.3f} p={r['partial_p']:.3f} | E-LP rho={r['energy_localparticipation_spearman']:.3f}")
    plot(rows)
