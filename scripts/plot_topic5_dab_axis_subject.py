#!/usr/bin/env python3
"""Per-subject: interictal A/B earliness contrast D_AB laid onto SEEG contact
coordinates, and the spatial axis it defines.

D_AB (reuses V3d src.topic5_scaffold_ab_contrast.build_D_AB): per joint contact,
eA - eB where eA = -zscore(rank_a), eB = -zscore(rank_b). D_AB > 0 = this contact
leads in template A and lags in template B (A-source side); D_AB < 0 = B-source
side. Source ranks = rank-displacement accepted-template pair (INTERICTAL); this
is the Core-1 axis construction, distinct from V3d's ictal-energy C_AB.

Axis = least-squares gradient of D_AB on 3D coords (direction along which D_AB is
organized). The load-bearing question this figure answers BY EYE: does D_AB
polarize into two spatially SEPARATED poles, or is it a gradient along a SINGLE
electrode shaft (within-shaft) -- the case the within-shaft null adjudicates.
"""
import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib import font_manager, gridspec
from matplotlib.lines import Line2D

_CJK = font_manager.FontProperties(family="Noto Sans CJK JP", size=8.8)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.topic5_scaffold_ab_contrast import build_D_AB, template_pair_tier  # noqa: E402
from src.propagation_skeleton_geometry import parse_shaft  # noqa: E402
from src.seeg_coord_loader import load_subject_coords  # noqa: E402

RANKDISP = {
    "narrow": REPO / "results/interictal_propagation_masked/rank_displacement/per_subject",
    "broad": REPO / "results/interictal_propagation_masked_broad/rank_displacement/per_subject",
}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]


def morans_i(coords, y, rng, n_perm=999):
    """Spatial autocorrelation of D_AB (inverse-distance weights) + permutation
    p. Unlike a linear-gradient R², Moran's I is NOT inflated by single-shaft
    collinearity, so it is the honest 'is D_AB spatially structured' test."""
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    with np.errstate(divide="ignore"):
        W = 1.0 / d
    W[~np.isfinite(W)] = 0.0
    S0 = W.sum()
    z = y - y.mean()
    denom = float((z ** 2).sum())
    n = len(y)
    if S0 <= 0 or denom <= 0:
        return float("nan"), float("nan")

    def _I(zz):
        return (n / S0) * float(zz @ W @ zz) / denom

    i_obs = _I(z)
    perm = np.array([_I(rng.permutation(z)) for _ in range(n_perm)])
    p = (1 + int(np.sum(perm >= i_obs))) / (n_perm + 1)
    return float(i_obs), float(p)


def compute(dataset, subject, pool):
    f = RANKDISP[pool] / f"{dataset}_{subject}.json"
    d = json.loads(f.read_text())
    p = d["pairs"][0]
    names = list(p["channel_names"])
    ra = np.array(p["rank_a_dense_full"], float)
    rb = np.array(p["rank_b_dense_full"], float)
    jv = np.array(p["joint_valid"], bool)
    soz = set(d.get("soz_channels", []))
    sw = p.get("swap_sweep", {})

    jn = [names[i] for i in np.where(jv)[0]]
    dd = build_D_AB(ra[jv], rb[jv])
    dab_joint = dd["D_AB"]

    cr = load_subject_coords(dataset, subject, jn)
    C = np.asarray(cr.coords_array_in_requested_order, float)
    mp = np.asarray(cr.mapped_mask_in_requested_order, bool)
    en = [jn[i] for i in np.where(mp)[0]]
    X = C[mp]
    y = dab_joint[mp]
    shafts = np.array([parse_shaft(c)[0] for c in en])
    if len(en) < 6:
        raise ValueError(f"only {len(en)} mapped joint contacts (<6)")

    # Least-squares gradient of D_AB over 3D coords -> spatial axis direction.
    Xc = X - X.mean(0)
    yc = y - y.mean()
    if (yc ** 2).sum() < 1e-9:
        raise ValueError("D_AB constant across contacts (no gradient)")
    beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    R2 = float(1 - ((yc - Xc @ beta) ** 2).sum() / (yc ** 2).sum())
    u = beta / np.linalg.norm(beta)
    along = Xc @ u
    if np.corrcoef(along, y)[0, 1] < 0:  # orient +axis toward A-source pole
        u, along = -u, -along
    resid = Xc - np.outer(along, u)
    w = np.linalg.svd(resid, full_matrices=False)[2][0]  # main perpendicular
    perp = resid @ w

    # Poles = extreme terciles of D_AB.
    order = np.argsort(y)
    k = max(2, len(en) // 3)
    b_idx, a_idx = order[:k], order[-k:]
    capA, capB = X[a_idx].mean(0), X[b_idx].mean(0)
    a_shafts = sorted(set(shafts[a_idx]))
    b_shafts = sorted(set(shafts[b_idx]))
    shared = sorted(set(a_shafts) & set(b_shafts))

    # Within-shaft fraction of D_AB variance: 1 - between-shaft SS / total SS.
    smean = np.array([y[shafts == shafts[i]].mean() for i in range(len(en))])
    between = ((smean - y.mean()) ** 2).sum()
    within_frac = float(1 - between / (yc ** 2).sum()) if (yc ** 2).sum() > 0 else np.nan
    sep_mm = float(np.linalg.norm(capA - capB))
    moran_i, moran_p = morans_i(X, y, np.random.default_rng(0))

    # Case type (descriptive; the quantitative gate remains the within-shaft
    # null). Spatial structure is judged by Moran's I (collinearity-free), NOT
    # by linear R² which is inflated for single-shaft geometry. cross_shaft =
    # the two D_AB poles sit on different electrode shafts.
    cross_shaft = len(shared) == 0
    if np.isnan(moran_p) or moran_p >= 0.05:
        case = "unstructured"
    elif within_frac >= 0.8 and not cross_shaft:
        case = "single_shaft"
    elif cross_shaft or within_frac < 0.5:
        case = "cross_shaft"
    else:
        case = "mixed"

    return dict(
        dataset=dataset, subject=subject, pool=pool,
        names=en, X=X, DAB=y, shafts=shafts, soz=soz,
        along=along, perp=perp, u=u, w=w,
        rho_AB=dd["rho_AB"], tier=template_pair_tier(dd["rho_AB"]),
        rank_a=ra[jv][mp], rank_b=rb[jv][mp],
        R2=R2, within_frac=within_frac, moran_i=moran_i, moran_p=moran_p,
        a_idx=a_idx, b_idx=b_idx, capA=capA, capB=capB,
        a_shafts=a_shafts, b_shafts=b_shafts, shared_shafts=shared,
        sep_mm=sep_mm, cross_shaft=cross_shaft, case=case,
        n_joint=len(en), n_shafts=len(set(shafts)),
        swap_class=sw.get("swap_class"), decision_k=sw.get("decision_k"),
    )


def plot(r, out):
    vmax = float(np.abs(r["DAB"]).max())
    uniq = sorted(set(r["shafts"]))
    mk = {s: MARKERS[i % len(MARKERS)] for i, s in enumerate(uniq)}
    cmap = plt.get_cmap("tab10")
    scol = {s: cmap(i % 10) for i, s in enumerate(uniq)}

    fig = plt.figure(figsize=(13.5, 6.2))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.1, 1.0],
                           wspace=0.28, hspace=0.35, left=0.06, right=0.97,
                           top=0.9, bottom=0.1)

    # ---- Panel 1: anatomical view. Project contacts onto their two principal
    # spatial axes (real electrode geometry preserved) so "one shaft vs
    # separated poles" is visible by eye; color = D_AB.
    ax1 = fig.add_subplot(gs[:, 0])
    mean = r["X"].mean(0)
    Xc = r["X"] - mean
    e1, e2 = np.linalg.svd(Xc, full_matrices=False)[2][:2]
    px, py = Xc @ e1, Xc @ e2
    for s in uniq:
        idx = np.where(r["shafts"] == s)[0]
        edge = ["k" if r["names"][i] in r["soz"] else "0.4" for i in idx]
        lw = [1.8 if r["names"][i] in r["soz"] else 0.4 for i in idx]
        ax1.scatter(px[idx], py[idx], c=r["DAB"][idx], cmap="coolwarm",
                    vmin=-vmax, vmax=vmax, marker=mk[s], s=185,
                    edgecolors=edge, linewidths=lw, zorder=3)
    for i, nm in enumerate(r["names"]):
        ax1.annotate(nm, (px[i], py[i]), fontsize=6, ha="left", va="bottom",
                     xytext=(4, 3), textcoords="offset points", color="0.25", zorder=4)
    bA = ((r["capA"] - mean) @ e1, (r["capA"] - mean) @ e2)
    bB = ((r["capB"] - mean) @ e1, (r["capB"] - mean) @ e2)
    ax1.annotate("", xy=bA, xytext=bB,
                 arrowprops=dict(arrowstyle="-|>", color="0.2", lw=2.4,
                                 mutation_scale=24), zorder=2)
    ax1.annotate("B-source", bB, color="steelblue", fontsize=9, weight="bold",
                 ha="center", va="top", xytext=(0, -9), textcoords="offset points")
    ax1.annotate("A-source", bA, color="firebrick", fontsize=9, weight="bold",
                 ha="center", va="bottom", xytext=(0, 9), textcoords="offset points")
    ax1.set_xlabel("principal spatial axis 1 (mm)")
    ax1.set_ylabel("principal spatial axis 2 (mm)")
    ax1.set_title("Contacts in SEEG space, colored by A/B earliness contrast\n"
                  "(grey arrow = D_AB spatial axis, B-source → A-source)", fontsize=10.5)
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(alpha=0.25)
    sm = plt.cm.ScalarMappable(cmap="coolwarm",
                               norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    cb = fig.colorbar(sm, ax=ax1, fraction=0.046, pad=0.02)
    cb.set_label("D_AB   (red = leads in A / lags in B      blue = leads in B / lags in A)",
                 fontsize=8)
    shaft_handles = [Line2D([0], [0], marker=mk[s], color="0.4", linestyle="none",
                            markersize=8, label=s) for s in uniq]
    soz_handle = [Line2D([0], [0], marker="o", markerfacecolor="none",
                         markeredgecolor="k", linestyle="none", markersize=9,
                         markeredgewidth=1.6, label="clinical SOZ")]
    ax1.legend(handles=shaft_handles + soz_handle, title="electrode shaft",
               loc="upper left", fontsize=7.5, title_fontsize=8, framealpha=0.9)

    # ---- Panel 2: D_AB gradient along the axis, colored by shaft.
    ax2 = fig.add_subplot(gs[0, 1])
    for s in uniq:
        m = r["shafts"] == s
        ax2.scatter(r["along"][m], r["DAB"][m], color=scol[s], marker=mk[s],
                    s=70, label=s, zorder=3)
    xs = np.array([r["along"].min(), r["along"].max()])
    b1 = np.polyfit(r["along"], r["DAB"], 1)
    ax2.plot(xs, np.polyval(b1, xs), color="0.3", lw=1.4, ls="--", zorder=2)
    ax2.axhline(0, color="0.7", lw=0.8, zorder=1)
    ax2.set_xlabel("position along D_AB axis (mm)")
    ax2.set_ylabel("D_AB")
    ax2.set_title(f"Is the gradient one shaft or cross-shaft?   (spatial R² = {r['R2']:.2f})",
                  fontsize=10)
    ax2.legend(fontsize=7.5, title="shaft", title_fontsize=8, ncol=2, framealpha=0.9)
    ax2.grid(alpha=0.25)

    # ---- Panel 3: metrics block (English monospace) + plain-language read (CJK).
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    case = r["case"]
    read = {
        "unstructured": "D_AB 在空间上没有显著结构 (Moran's I 不显著)，这个被试读不出一条轴。",
        "single_shaft": (f"退化情况:D_AB 有空间结构但主要沿单根电极杆 ({'/'.join(r['shared_shafts'])})。"
                         "这只是一类退化、不是失败——空间信息基本来自一根杆;"
                         "跨杆的被试在 broad 池里更多。"),
        "cross_shaft": "跨杆:D_AB 两极落在不同电极杆/变异多在杆间,空间信息更强的一类 (定量仍走同杆零假设)。",
        "mixed": "单杆/跨杆之间:D_AB 有空间结构但杆内、杆间混合。",
    }[case]
    lines = [
        f"subject: {r['dataset']} {r['subject']}   pool = {r['pool']}",
        f"templates: rho(A,B) = {r['rho_AB']:+.2f}  [{r['tier']}]",
        f"joint contacts = {r['n_joint']}   shafts = {r['n_shafts']}",
        "",
        f"spatial structure: Moran I = {r['moran_i']:.2f} (p={r['moran_p']:.3f})",
        f"within-shaft variance = {r['within_frac']*100:.0f}%   (R2 = {r['R2']:.2f})",
        f"A-source pole: {', '.join(r['a_shafts'])}   B-source pole: {', '.join(r['b_shafts'])}",
        f"share shaft: {', '.join(r['shared_shafts']) or 'none (cross-shaft)'}   sep = {r['sep_mm']:.0f} mm",
        f"case: {case}",
    ]
    ax3.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=7.8,
             family="monospace", transform=ax3.transAxes)
    ax3.text(0.0, 0.34, "读: " + "\n".join(textwrap.wrap(read, width=22)),
             va="top", ha="left", fontproperties=_CJK, transform=ax3.transAxes)

    fig.suptitle(
        f"{r['dataset']} {r['subject']}  —  interictal A/B earliness contrast D_AB "
        f"and the spatial axis it defines",
        fontsize=12.5, weight="bold")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out}")


CASES = ["cross_shaft", "mixed", "single_shaft", "unstructured"]
CASE_COL = {"cross_shaft": "#2c7fb8", "mixed": "#7fbc41",
            "single_shaft": "#fdae61", "unstructured": "0.6"}
POOL_MK = {"narrow": "o", "broad": "s"}
CSV_COLS = ["dataset", "subject", "pool", "status", "n_joint", "n_shafts", "rho_AB",
            "tier", "moran_i", "moran_p", "R2", "within_frac", "sep_mm", "cross_shaft",
            "case", "a_shafts", "b_shafts", "shared_shafts", "swap_class", "decision_k"]


def sweep(pool):
    rows = []
    for f in sorted(RANKDISP[pool].glob("*.json")):
        ds, subj = f.stem.split("_", 1)
        try:
            r = compute(ds, subj, pool)
            rows.append(dict(
                dataset=ds, subject=subj, pool=pool, status="ok",
                n_joint=r["n_joint"], n_shafts=r["n_shafts"],
                rho_AB=round(r["rho_AB"], 3), tier=r["tier"],
                moran_i=round(r["moran_i"], 3), moran_p=round(r["moran_p"], 3),
                R2=round(r["R2"], 3),
                within_frac=round(r["within_frac"], 3), sep_mm=round(r["sep_mm"], 1),
                cross_shaft=r["cross_shaft"], case=r["case"],
                a_shafts="/".join(r["a_shafts"]), b_shafts="/".join(r["b_shafts"]),
                shared_shafts="/".join(r["shared_shafts"]),
                swap_class=r["swap_class"], decision_k=r["decision_k"]))
        except Exception as e:  # coord miss / low-n / degenerate -> record, keep going
            rows.append(dict(dataset=ds, subject=subj, pool=pool, status=str(e)[:70]))
    return rows


def plot_cohort(rows, out):
    ok = [x for x in rows if x["status"] == "ok"]
    pools = ["narrow", "broad"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # Panel A: spatial structure (Moran's I, collinearity-free) vs within-shaft
    # fraction. Top-right = single-shaft gradient; top-left = cross-shaft axis;
    # bottom = spatially unstructured.
    for x in ok:
        recip = x["tier"] == "reciprocal"  # only reciprocal pairs give a clean D_AB
        axA.scatter(x["within_frac"], x["moran_i"], c=CASE_COL[x["case"]],
                    marker=POOL_MK[x["pool"]], s=95 if recip else 72,
                    edgecolors="black" if recip else "0.55",
                    linewidths=1.8 if recip else 0.4,
                    zorder=4 if recip else 3, alpha=0.92)
    axA.axvline(0.5, color="0.75", ls=":", lw=1)
    axA.axvline(0.8, color="0.75", ls="--", lw=1)
    axA.axhline(0.0, color="0.6", lw=0.8)
    for x in ok:
        if x["subject"] == "1146":
            axA.annotate(f"E1146/{x['pool']}", (x["within_frac"], x["moran_i"]),
                         fontsize=7, xytext=(4, -9), textcoords="offset points")

    def _recip_cs(pool):
        rs = [x for x in ok if x["pool"] == pool and x["tier"] == "reciprocal"]
        return sum(1 for x in rs if x["case"] == "cross_shaft"), len(rs)
    bn, bd = _recip_cs("broad")
    nn, nd = _recip_cs("narrow")
    axA.set_xlabel("within-shaft fraction of D_AB variance  (1.0 = single-shaft gradient)")
    axA.set_ylabel("Moran's I of D_AB  (spatial structure)")
    axA.set_title("Each subject: spatial structure (↑) vs within-shaft locality (→)\n"
                  f"thick black edge = reciprocal templates (ρ≤−0.5), the clean D_AB regime: "
                  f"broad {bn}/{bd} cross-shaft, narrow {nn}/{nd}", fontsize=9.5)
    axA.set_xlim(-0.02, 1.02)
    case_h = [Line2D([0], [0], marker="o", color="none", markerfacecolor=CASE_COL[c],
                     markeredgecolor="0.3", linestyle="none", markersize=8, label=c)
              for c in CASES]
    pool_h = [Line2D([0], [0], marker=POOL_MK[p], color="0.35", linestyle="none",
                     markersize=8, markerfacecolor="0.85", label=f"{p} pool") for p in pools]
    axA.legend(handles=case_h + pool_h, fontsize=7.5, loc="lower left", ncol=2,
               framealpha=0.9)
    axA.grid(alpha=0.25)

    # Panel B: case-type counts, narrow vs broad (does broad give more cross-shaft?).
    width = 0.38
    xpos = np.arange(len(CASES))
    pcol = {"narrow": "#a6cee3", "broad": "#1f78b4"}
    for i, p in enumerate(pools):
        counts = [sum(1 for x in ok if x["pool"] == p and x["case"] == c) for c in CASES]
        npool = sum(1 for x in ok if x["pool"] == p)
        bars = axB.bar(xpos + (i - 0.5) * width, counts, width,
                       label=f"{p} (n={npool})", color=pcol[p], edgecolor="0.3")
        for b, ct in zip(bars, counts):
            if ct:
                axB.text(b.get_x() + b.get_width() / 2, ct + 0.15, str(ct),
                         ha="center", fontsize=8)
    axB.set_xticks(xpos)
    axB.set_xticklabels(CASES, rotation=12)
    axB.set_ylabel("subjects")
    axB.set_title("Case-type counts: narrow vs broad pool")
    axB.legend(fontsize=8.5)
    axB.grid(alpha=0.25, axis="y")

    fig.suptitle("Cohort D_AB axis review — is the A/B earliness contrast a within-shaft "
                 "gradient or a cross-shaft axis?", fontsize=12.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out}")


def run_cohort():
    outdir = REPO / "results/topic5_dab_axis"
    all_rows = []
    for pool in ["narrow", "broad"]:
        rows = sweep(pool)
        all_rows += rows
        ok = [x for x in rows if x["status"] == "ok"]
        summ = ", ".join(f"{c}={sum(1 for x in ok if x['case'] == c)}" for c in CASES)
        print(f"[{pool}] {len(ok)}/{len(rows)} ok   {summ}")
    csv_path = outdir / "cohort_dab_axis_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        for row in all_rows:
            w.writerow({c: row.get(c, "") for c in CSV_COLS})
    print(f"wrote {csv_path}")
    plot_cohort(all_rows, outdir / "figures/cohort_dab_axis_summary.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="epilepsiae")
    ap.add_argument("--subject", default="1146")
    ap.add_argument("--pool", default="narrow", choices=["narrow", "broad"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--cohort", action="store_true",
                    help="sweep all subjects in both pools; write CSV + summary figure")
    a = ap.parse_args()
    if a.cohort:
        run_cohort()
        return
    r = compute(a.dataset, a.subject, a.pool)
    out = a.out or str(REPO / f"results/topic5_dab_axis/figures/{a.dataset}_{a.subject}_{a.pool}_dab_axis.png")
    print(f"[{r['dataset']} {r['subject']} {r['pool']}] n_joint={r['n_joint']} "
          f"rho_AB={r['rho_AB']:+.3f} ({r['tier']}) R2={r['R2']:.3f} "
          f"within_shaft={r['within_frac']*100:.0f}% sep={r['sep_mm']:.1f}mm case={r['case']}")
    plot(r, out)


if __name__ == "__main__":
    main()
