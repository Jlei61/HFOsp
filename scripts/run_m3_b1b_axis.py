#!/usr/bin/env python3
"""B1b — axis / anisotropy of the W_event shape (M3 mini-W_event validation, 2026-06-24).

Question: does the early-recruitment shape reflect the E->E long axis (theta_EE=45 deg),
or is it just an isotropic local spread?

For the mean W_shape (source-excluded) of each (substrate, source) at its representative
EA-local kick: principal-axis angle + anisotropy ratio, vs a SPATIAL-SHUFFLE null (permute
which non-source bin carries which weight, >=1000x) -> is the observed anisotropy above chance?
Axis error vs theta_EE=45 deg (axes undirected, mod 180). Per the plan: if the shape is near
isotropic the ANGLE is not interpretable -> judged by anisotropy.

PRIMARY = center (bare@1.6, n17.6@1.1). Off-axis R_src=4mm sources are EDGE-SENSITIVE
(boundary clamps r95) -> reported as SENSITIVITY only, not primary.

RESOLUTION CAVEAT (recorded in the summary): bins are 4mm (L=20 / 5) and the event r95~5mm,
so the shape is dominated by the 4 orthogonal immediate neighbours (0/90 deg) — at this
coarse resolution the W_shape cannot cleanly resolve a 45 deg diagonal axis. A FAIL/WEAK here
is a resolution statement, not proof the connectivity is isotropic.

OFFLINE over existing ea_net_bins.npz. Outputs to b1_validation/b1b_axis/.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import sef_hfo_mini_w_event as mwe          # noqa: E402
from src import sef_hfo_b1_validation as b1           # noqa: E402

ROOT = "results/topic4_sef_hfo/m3_local_w/mini_w_event/runs"
OUT = "results/topic4_sef_hfo/m3_local_w/mini_w_event/b1_validation/b1b_axis"
THETA_EE = 45.0
# representative EA kick per source (its own K_min-ish; from the K_min(q) table)
KICK = {("bare", "center"): 1.6, ("n17.6", "center"): 1.1,
        ("bare", "+axis"): 1.2, ("n17.6", "+axis"): 1.1,
        ("bare", "-axis"): 1.2, ("n17.6", "-axis"): 1.2,
        ("bare", "+offaxis"): 1.0, ("n17.6", "+offaxis"): 1.0,
        ("bare", "-offaxis"): 1.0, ("n17.6", "-offaxis"): 1.0}


def _mean_shape(run, kick):
    ki = int(np.argmin(np.abs(np.asarray(run["npz_kicks"], float) - kick)))
    rk = min(run["kicks"], key=lambda k: abs(k - kick))
    succ = mwe.success_seeds_at_kick(run["recs_by_kick"][rk], run["spont_seeds"])
    per_seed, mean_w, used = mwe.build_w_shape(run["ea_net_bins"][ki], succ, run["src_bin_idx"])
    return mean_w, len(used)


def _shuffle_null(weights, pos, n_null, rng_seed):
    """Null anisotropy/axis by permuting which bin carries which weight (breaks spatial
    structure, preserves the weight value set)."""
    rng = np.random.default_rng(rng_seed)
    anis, ang_err = [], []
    for _ in range(n_null):
        wp = weights[rng.permutation(len(weights))]
        a, an = b1.principal_axis(wp, pos)
        anis.append(an if np.isfinite(an) else np.nan)
        ang_err.append(b1.axis_angle_diff(a, THETA_EE))
    return np.asarray(anis), np.asarray(ang_err)


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    rows = []
    fig_data = {}
    for sub in ("bare", "n17.6"):
        for src_name in ("center", "+axis", "-axis", "+offaxis", "-offaxis"):
            run = mwe.load_run_dir(f"{ROOT}/{sub}_{src_name}")
            n_bins = run["ea_net_bins"].shape[2]
            src = run["src_bin_idx"]
            nonsrc = [b for b in range(n_bins) if b != src]
            pos = np.asarray(run["bin_centers"])[nonsrc]
            kick = KICK[(sub, src_name)]
            mean_w, n_used = _mean_shape(run, kick)
            angle, aniso = b1.principal_axis(mean_w, pos)
            null_an, _ = _shuffle_null(mean_w, pos, 2000, rng_seed=0)
            p_aniso = float(np.mean(null_an[np.isfinite(null_an)] >= aniso))
            null_p95 = float(np.nanpercentile(null_an, 95))
            ang_err = b1.axis_angle_diff(angle, THETA_EE)
            is_primary = (src_name == "center")
            rows.append({
                "substrate": sub, "source": src_name, "kick": kick, "n_used": n_used,
                "tier": "primary" if is_primary else "sensitivity(edge)",
                "anisotropy": round(aniso, 3), "null_aniso_p95": round(null_p95, 3),
                "p_aniso_ge_null": round(p_aniso, 4),
                "axis_angle_deg": round(angle, 1), "axis_err_vs_45": round(ang_err, 1),
                "aniso_sig": bool(p_aniso < 0.05),
            })
            if is_primary:
                fig_data[sub] = (mean_w.copy(), aniso, null_an, angle, ang_err, p_aniso)

    with open(os.path.join(OUT, "b1b_axis_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # verdict from the PRIMARY (center) rows
    center = [r for r in rows if r["source"] == "center"]
    any_sig = any(r["aniso_sig"] for r in center)
    near45 = all(r["axis_err_vs_45"] <= 25 for r in center)
    if any_sig and near45:
        verdict = "PASS"
    elif any_sig:
        verdict = "WEAK"        # some anisotropy but axis not at 45 (or resolution-limited)
    else:
        verdict = "FAIL"        # not above isotropic null at this resolution

    # figure: center null vs observed anisotropy
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for sub, color in [("bare", "tab:blue"), ("n17.6", "tab:orange")]:
        if sub in fig_data:
            _, aniso, null_an, angle, ang_err, p = fig_data[sub]
            ax[0].hist(null_an[np.isfinite(null_an)], bins=40, alpha=0.4, color=color,
                       density=True, label=f"{sub} null")
            ax[0].axvline(aniso, color=color, lw=2,
                          label=f"{sub} obs={aniso:.2f} (p={p:.2f})")
    ax[0].set_xlabel("anisotropy ratio (lambda1/lambda2)"); ax[0].set_ylabel("density")
    ax[0].set_title("center anisotropy vs spatial-shuffle null"); ax[0].legend(fontsize=8)
    # axis angle vs 45 ref
    cs = [r for r in rows if r["source"] == "center"]
    ax[1].bar([r["substrate"] for r in cs], [r["axis_err_vs_45"] for r in cs],
              color=["tab:blue", "tab:orange"])
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_ylabel("axis error vs theta_EE=45 (deg)")
    ax[1].set_title("center principal-axis error vs E->E (45 deg)")
    fig.suptitle(f"B1b axis/anisotropy (center primary) — verdict {verdict}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "figures", "b1b_anisotropy_null.png"), dpi=130); plt.close(fig)

    with open(os.path.join(OUT, "b1b_summary.md"), "w") as f:
        f.write("# B1b — axis / anisotropy\n\n")
        f.write(f"**Verdict: {verdict}** (from center primary).\n\n")
        for r in center:
            f.write(f"- {r['substrate']} center @ kick {r['kick']} (n={r['n_used']}): "
                    f"anisotropy={r['anisotropy']} (null p95={r['null_aniso_p95']}, "
                    f"p={r['p_aniso_ge_null']}), axis={r['axis_angle_deg']}deg, "
                    f"err vs 45={r['axis_err_vs_45']}deg, sig={r['aniso_sig']}\n")
        f.write("\n**RESOLUTION CAVEAT**: bins are 4mm (L=20/5), event r95~5mm ~ 1 bin, so the "
                "shape is dominated by the 4 orthogonal immediate neighbours and cannot cleanly "
                "resolve a 45deg diagonal axis. A WEAK/FAIL here is a resolution statement, NOT "
                "proof the connectivity is isotropic. Off-axis R_src=4mm rows are edge-sensitive "
                "(sensitivity only).\n\n")
        f.write("Verdict rule: PASS = center anisotropy sig > shuffle-null AND axis within 25deg "
                "of 45; WEAK = anisotropy sig but axis off / resolution-limited; FAIL = not above "
                "isotropic null at this resolution.\n")
    json.dump({"verdict": verdict, "center_rows": center}, open(os.path.join(OUT, "b1b_summary.json"), "w"), indent=1)
    print(f"[B1b] verdict={verdict}")
    for r in center:
        print(f"  {r['substrate']} center: aniso={r['anisotropy']} p={r['p_aniso_ge_null']} "
              f"axis={r['axis_angle_deg']} err45={r['axis_err_vs_45']} sig={r['aniso_sig']}")
    print(f"[B1b] wrote -> {OUT}")


if __name__ == "__main__":
    main()
