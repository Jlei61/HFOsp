#!/usr/bin/env python3
"""B1d — matched-shape equivalence (M3 mini-W_event validation, 2026-06-24).

Question: does n17.6 only LOWER the finite-event threshold, or also CHANGE the early
propagation shape of the successful events?

Primary comparison (each substrate at its OWN K_min, EA-local-returned successful events):
  bare  center @ K=1.6   vs   n17.6 center @ K=1.1
Sensitivity: n17.6 center @ K=1.2.

Equivalence is judged against the WITHIN-substrate split-half similarity (the ceiling for
"same shape" given seed noise), NOT against p>0.05:
  cross_median >= min(within_bare, within_core) - delta,   delta in {0.05, 0.10, 0.15}.

OFFLINE over existing runs/ ea_net_bins.npz (no SNN re-run). load_run_dir fails closed on
stale/mixed (4x4) artifacts. Outputs to b1_validation/b1d_matched_shape/.
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
OUT = "results/topic4_sef_hfo/m3_local_w/mini_w_event/b1_validation/b1d_matched_shape"
METRICS = ("cosine", "pearson", "spearman")


def _kick_index(run, kick):
    return int(np.argmin(np.abs(np.asarray(run["npz_kicks"], float) - kick)))


def _shapes_at_kick(run, kick):
    """(per_seed_shapes, mean_shape, used_seeds, success_recs) at the kick (matched grid kick)."""
    ki = _kick_index(run, kick)
    rk = min(run["kicks"], key=lambda k: abs(k - kick))
    recs = run["recs_by_kick"][rk]
    succ = mwe.success_seeds_at_kick(recs, run["spont_seeds"])
    per_seed, mean_w, used = mwe.build_w_shape(run["ea_net_bins"][ki], succ, run["src_bin_idx"])
    succ_recs = [r for r in recs if int(r["seed"]) in set(used)]
    return per_seed, mean_w, used, succ_recs


def _rep_seed(per_seed, used, succ_recs):
    """Index into per_seed of the representative (median-r95) success seed (no cherry-pick)."""
    from src.sef_hfo_event_figure import median_representative
    r95 = {int(r["seed"]): r["r95_ea"] for r in succ_recs}
    rep = median_representative(list(used), [r95[s] for s in used])
    return used.index(int(rep)), int(rep)


def _grid(vec, nb, src):
    g = np.full(nb * nb, np.nan)
    nonsrc = [b for b in range(nb * nb) if b != src]
    g[nonsrc] = vec
    return g.reshape(nb, nb)


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    bare = mwe.load_run_dir(f"{ROOT}/bare_center")
    core = mwe.load_run_dir(f"{ROOT}/n17.6_center")
    n_bins = bare["ea_net_bins"].shape[2]          # full grid (incl. source bin) = 25
    nb = int(round(np.sqrt(n_bins)))               # 5
    assert nb * nb == n_bins, f"n_bins={n_bins} not a square grid"
    src = bare["src_bin_idx"]
    nonsrc = [b for b in range(n_bins) if b != src]
    pos = np.asarray(bare["bin_centers"])[nonsrc]

    ps_bare, mean_bare, used_bare, recs_bare = _shapes_at_kick(bare, 1.6)
    ps_core, mean_core, used_core, recs_core = _shapes_at_kick(core, 1.1)
    ps_core12, mean_core12, used_core12, recs_core12 = _shapes_at_kick(core, 1.2)

    # --- deterministic metrics on the substrate-mean shapes -------------------
    def det_metrics(mA, mB):
        d = {m: b1.shape_similarity(mA, mB, m) for m in METRICS}
        cA, cB = b1.weighted_centroid(mA, pos), b1.weighted_centroid(mB, pos)
        d["centroid_dist_mm"] = float(np.linalg.norm(cA - cB))
        for k in (3, 5):
            d[f"top{k}_overlap"] = b1.top_k_overlap(mA, mB, k)
        aA, anA = b1.principal_axis(mA, pos)
        aB, anB = b1.principal_axis(mB, pos)
        d.update(axis_angle_A=aA, axis_angle_B=aB, axis_angle_diff=b1.axis_angle_diff(aA, aB),
                 anisotropy_A=anA, anisotropy_B=anB, anisotropy_diff=abs(anA - anB))
        return d

    det_primary = det_metrics(mean_bare, mean_core)        # bare@1.6 vs core@1.1
    det_sens = det_metrics(mean_bare, mean_core12)          # bare@1.6 vs core@1.2

    # --- within / cross similarity distributions (per metric) -----------------
    dist = {}
    for m in METRICS:
        wb = b1.split_half_similarity(ps_bare, m, n_splits=500, rng_seed=0)
        wc = b1.split_half_similarity(ps_core, m, n_splits=500, rng_seed=0)
        cr = b1.cross_subsample_similarity(ps_bare, ps_core, m, n_sub=500, rng_seed=0)
        dist[m] = {"within_bare": wb, "within_core": wc, "cross": cr,
                   "floor": min(wb["median"], wc["median"])}

    # --- equivalence sensitivity (primary metric cosine) ----------------------
    equiv_rows = []
    for m in METRICS:
        floor = dist[m]["floor"]; cr = dist[m]["cross"]["median"]
        for delta in (0.05, 0.10, 0.15):
            equiv_rows.append({"metric": m, "delta": delta,
                               "cross_median": round(cr, 4), "within_floor": round(floor, 4),
                               "threshold": round(floor - delta, 4),
                               "equivalent": bool(cr >= floor - delta)})

    # --- verdict --------------------------------------------------------------
    cos = dist["cosine"]; cr_cos = cos["cross"]["median"]; floor_cos = cos["floor"]
    deficit = floor_cos - cr_cos
    strong = (deficit <= 0.05 and det_primary["cosine"] >= 0.8
              and det_primary["top3_overlap"] >= 2 / 3 and det_primary["axis_angle_diff"] <= 20)
    if strong:
        verdict = "PASS"
    elif deficit <= 0.15 and det_primary["cosine"] >= 0.6:
        verdict = "WEAK"
    else:
        verdict = "FAIL"

    # --- write CSVs -----------------------------------------------------------
    with open(os.path.join(OUT, "b1d_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "metric_or_quantity", "value"])
        for name, dd in [("bare1.6_vs_core1.1", det_primary), ("bare1.6_vs_core1.2", det_sens)]:
            for k, v in dd.items():
                w.writerow([name, k, round(float(v), 4)])
        for m in METRICS:
            for kind in ("within_bare", "within_core", "cross"):
                w.writerow([f"dist_{m}", kind + "_median", round(dist[m][kind]["median"], 4)])
    with open(os.path.join(OUT, "b1d_equivalence_sensitivity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(equiv_rows[0].keys())); w.writeheader()
        w.writerows(equiv_rows)

    summary = {
        "n_success": {"bare@1.6": len(used_bare), "core@1.1": len(used_core),
                      "core@1.2": len(used_core12)},
        "det_primary_bare1.6_vs_core1.1": {k: round(float(v), 4) for k, v in det_primary.items()},
        "cosine_within_floor": round(floor_cos, 4), "cosine_cross_median": round(cr_cos, 4),
        "cosine_deficit_floor_minus_cross": round(deficit, 4),
        "verdict": verdict,
    }

    # --- figures --------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bc = np.asarray(bare["bin_centers"]); ext = [bc[:, 0].min(), bc[:, 0].max(),
                                                 bc[:, 1].min(), bc[:, 1].max()]
    # 1-3: mean heatmaps + difference
    gb, gc = _grid(mean_bare, nb, src), _grid(mean_core, nb, src)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for a, g, t in [(ax[0], gb, f"bare@1.6 mean W_shape (n={len(used_bare)})"),
                    (ax[1], gc, f"n17.6@1.1 mean W_shape (n={len(used_core)})")]:
        im = a.imshow(g, origin="lower", extent=ext, cmap="viridis"); a.set_title(t)
        fig.colorbar(im, ax=a, fraction=0.046); a.set_xlabel("x(mm)"); a.set_ylabel("y(mm)")
    dg = gb - gc
    vlim = np.nanmax(np.abs(dg)) if np.any(np.isfinite(dg)) else 1.0
    im = ax[2].imshow(dg, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ax[2].set_title("difference (bare − n17.6)"); fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.suptitle(f"B1d matched-shape — cos(means)={det_primary['cosine']:.2f}  "
                 f"axisΔ={det_primary['axis_angle_diff']:.0f}°  top3={det_primary['top3_overlap']:.2f}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "figures", "b1d_mean_shapes.png"), dpi=130); plt.close(fig)

    # 4: within vs cross similarity (cosine) bar
    fig, a = plt.subplots(figsize=(6, 4))
    labels = ["within bare", "within n17.6", "cross"]
    meds = [cos["within_bare"]["median"], cos["within_core"]["median"], cr_cos]
    los = [cos["within_bare"]["q25"], cos["within_core"]["q25"], cos["cross"]["q25"]]
    his = [cos["within_bare"]["q75"], cos["within_core"]["q75"], cos["cross"]["q75"]]
    a.bar(labels, meds, color=["tab:blue", "tab:orange", "tab:green"])
    a.errorbar(labels, meds, yerr=[np.subtract(meds, los), np.subtract(his, meds)],
               fmt="none", ecolor="k", capsize=4)
    a.axhline(floor_cos - 0.05, ls="--", color="grey", label="floor − 0.05")
    a.set_ylabel("cosine similarity (median, IQR)"); a.set_ylim(0, 1.02)
    a.set_title(f"B1d within vs cross (cosine) — verdict {verdict}"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "b1d_within_vs_cross.png"),
                                    dpi=130); plt.close(fig)

    # 5: representative-seed (median r95) heatmap montage (no re-sim; from npz)
    ib, sb = _rep_seed(ps_bare, used_bare, recs_bare)
    ic, sc = _rep_seed(ps_core, used_core, recs_core)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
    for a, g, t in [(ax[0], _grid(ps_bare[ib], nb, src), f"bare@1.6 rep seed {sb} (median r95)"),
                    (ax[1], _grid(ps_core[ic], nb, src), f"n17.6@1.1 rep seed {sc} (median r95)")]:
        im = a.imshow(g, origin="lower", extent=ext, cmap="viridis"); a.set_title(t)
        fig.colorbar(im, ax=a, fraction=0.046); a.set_xlabel("x(mm)"); a.set_ylabel("y(mm)")
    fig.suptitle("B1d representative-event W_shape (median-r95 seed, not cherry-picked)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "figures", "b1d_representative_montage.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(OUT, "b1d_summary.md"), "w") as f:
        f.write("# B1d — matched-shape equivalence\n\n")
        f.write(f"**Verdict: {verdict}**\n\n")
        f.write(f"- n_success: bare@1.6={len(used_bare)}, n17.6@1.1={len(used_core)}, "
                f"n17.6@1.2={len(used_core12)}\n")
        f.write(f"- cosine(mean bare@1.6, mean n17.6@1.1) = {det_primary['cosine']:.3f}; "
                f"pearson={det_primary['pearson']:.3f}; spearman={det_primary['spearman']:.3f}\n")
        f.write(f"- top3_overlap={det_primary['top3_overlap']:.2f}, "
                f"top5_overlap={det_primary['top5_overlap']:.2f}, "
                f"centroid_dist={det_primary['centroid_dist_mm']:.2f}mm\n")
        f.write(f"- axis: bare={det_primary['axis_angle_A']:.0f}°, "
                f"n17.6={det_primary['axis_angle_B']:.0f}°, diff={det_primary['axis_angle_diff']:.0f}°; "
                f"anisotropy bare={det_primary['anisotropy_A']:.2f}, "
                f"n17.6={det_primary['anisotropy_B']:.2f}\n")
        f.write(f"- cosine within-floor={floor_cos:.3f} (within_bare="
                f"{cos['within_bare']['median']:.3f}, within_core={cos['within_core']['median']:.3f}); "
                f"cross_median={cr_cos:.3f}; deficit(floor−cross)={deficit:.3f}\n")
        f.write("- equivalence (cosine): "
                + ", ".join(f"delta={d}:{('PASS' if cr_cos>=floor_cos-d else 'no')}"
                            for d in (0.05, 0.10, 0.15)) + "\n\n")
        f.write("Equivalence is judged against within-substrate split-half similarity "
                "(seed-noise ceiling), not p>0.05. Sensitivity vs n17.6@1.2 in b1d_metrics.csv.\n\n")
        f.write("Verdict rule: PASS = deficit<=0.05 AND cos(means)>=0.8 AND top3>=0.67 AND "
                "axisΔ<=20°; WEAK = deficit<=0.15 AND cos>=0.6; else FAIL.\n")
    json.dump(summary, open(os.path.join(OUT, "b1d_summary.json"), "w"), indent=1)
    print(f"[B1d] verdict={verdict}  cos(means)={det_primary['cosine']:.3f}  "
          f"cross={cr_cos:.3f} floor={floor_cos:.3f} deficit={deficit:.3f}")
    print(f"[B1d] wrote -> {OUT}")


if __name__ == "__main__":
    main()
