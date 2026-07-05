"""Paper-ready real-vs-model interictal template comparison (Topic 4 Fig4C).

A = the patient's REAL interictal templates (E1146 t_a / t_b per-channel typical_rank).
B = the subject-SNN MODEL templates (forward / reverse cluster mean within-event rank from the
    spontaneous twoend readout used in Fig4A/B).

Shows whether the model readout reproduces the real interictal template ORDER (Spearman per
channel). Plotting-only; consumes the same model readout + the real propagation geometry. No re-sim.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/topic4_sef_hfo/field_swap_subject_snn"
# Real interictal templates live under the montage-matched propagation_geometry tree. Subjects whose
# narrow montage is degenerate (e.g. yuquan_zhangjinhan: 5 ch narrow vs 20 ch broad with C6/C7/F5/F6)
# must read the broad templates -- same montage the SNN cores were placed in.
_GEO = {
    "narrow": ROOT / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects",
    "broad": ROOT / "results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects",
}
A_COL, B_COL = "#1f77b4", "#d62728"   # template-A/forward = blue, template-B/reverse = red


def _real_templates(subject, montage="narrow"):
    geo = _GEO[montage]
    out = {}
    for tpl in ("t_a", "t_b"):
        g = json.load(open(geo / f"{subject}_{tpl}.json"))
        out[tpl] = {c["name"]: c["typical_rank"] for c in g["channels"] if c.get("typical_rank") is not None}
    return out


def _model_templates(tag):
    ro = json.load(open(RUN / f"readout_{tag}.json"))
    k_dir = int(ro.get("k_dir", 2))
    events = [e for e in ro["events"] if e.get("sign") is not None and e.get("n_part", 0) >= 2 * k_dir]
    names = sorted({n for e in events for n, v in (e.get("ranks") or {}).items() if v is not None})
    R = np.full((len(names), len(events)), np.nan)
    for j, e in enumerate(events):
        for i, n in enumerate(names):
            v = (e.get("ranks") or {}).get(n)
            if v is not None:
                R[i, j] = v
    # model fwd/rev templates built by event SIGN directly (NOT by mapping KMeans clusters -- for a
    # one-direction readout the two clusters are sub-patterns of the SAME direction, so a cluster->fwd/rev
    # mapping is fake). Consistent with the Fig4B gate (plot_fig_subject_snn_kmeans2.py).
    signs = np.array([e["sign"] for e in events])

    def meanrank(mask):
        out = {}
        for i, n in enumerate(names):
            v = R[i, mask]; v = v[np.isfinite(v)]
            if v.size:
                out[n] = float(v.mean())
        return out
    return {"forward": meanrank(signs > 0), "reverse": meanrank(signs < 0)}


def _axis_order(subject, tag, names):
    fd = np.load(RUN / f"figdata_{tag}.npz", allow_pickle=True)
    all_names = [str(x) for x in fd["names"]]
    contacts = np.asarray(fd["contacts"], float)
    reg = fd["reg"].item(); center = np.asarray(reg["center"]); axis = np.asarray(reg["axis_unit"])
    keep = [n for n in names if n in all_names]
    proj = {n: float((contacts[all_names.index(n)] - center) @ axis) for n in keep}
    return sorted(keep, key=lambda n: proj[n])


def _norm(d, order):
    v = np.array([d.get(n, np.nan) for n in order], float)
    fin = np.isfinite(v)
    if fin.sum() < 2:
        return v
    lo, hi = np.nanmin(v), np.nanmax(v)
    return (v - lo) / (hi - lo) if hi > lo else v


def _stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def _sim_matrix(model, real, B=10000, seed=0):
    """2x2 Spearman (rows model fwd/rev, cols data t_a/t_b) + DIRECTIONAL channel-shuffle
    permutation p (diagonal tested one-sided positive, off-diagonal one-sided negative -- the
    swap-predicted direction)."""
    rng = np.random.default_rng(seed)
    rows = [("forward", model["forward"]), ("reverse", model["reverse"])]
    cols = [("t_a", real["t_a"]), ("t_b", real["t_b"])]
    M = np.full((2, 2), np.nan); P = np.full((2, 2), np.nan)
    for i, (_, mv) in enumerate(rows):
        for j, (_, dv) in enumerate(cols):
            common = sorted(set(mv) & set(dv))
            if len(common) < 4:
                continue
            a = np.array([mv[n] for n in common]); b = np.array([dv[n] for n in common])
            rho = float(spearmanr(a, b).correlation); M[i, j] = rho
            null = np.array([spearmanr(a, rng.permutation(b)).correlation for _ in range(B)])
            diag = (i == j)   # diagonal -> positive tail; off-diagonal -> negative tail
            P[i, j] = ((1 + np.sum(null >= rho)) / (B + 1)) if diag else ((1 + np.sum(null <= rho)) / (B + 1))
    return M, P


def _matrix_panel(ax, M, P):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")  # square cells (not squished)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["t_a", "t_b"], fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["fwd", "rev"], fontsize=9)
    ax.set_xlabel("real template", fontsize=9); ax.set_ylabel("model", fontsize=9)
    for i in range(2):
        for j in range(2):
            if np.isfinite(M[i, j]):
                ax.text(j, i, _stars(P[i, j]), ha="center", va="center", fontsize=15,
                        color="white" if abs(M[i, j]) > 0.55 else "black")
    ax.set_title("C  similarity\n(★ = perm p)", fontsize=10)
    return im


def _panel(ax, a_prof, b_prof, order, a_lbl, b_lbl, title):
    y = np.arange(len(order))
    for prof, col, lbl in ((a_prof, A_COL, a_lbl), (b_prof, B_COL, b_lbl)):
        v = _norm(prof, order); fin = np.isfinite(v)
        ax.plot(v[fin], y[fin], "-o", color=col, ms=4, lw=1.6, label=lbl)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=8)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(-0.05, 1.05); ax.set_xlabel("normalized rank (early → late)", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", color="0.92", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def compose(subject, tag, fig_name, montage="narrow"):
    real = _real_templates(subject, montage)
    model = _model_templates(tag)
    order = _axis_order(subject, tag, sorted(set(real["t_a"]) | set(model["forward"]) | set(model["reverse"])))

    def corr(a, b):
        common = [n for n in order if n in a and n in b]
        return (float(spearmanr([a[n] for n in common], [b[n] for n in common]).correlation),
                len(common)) if len(common) >= 4 else (float("nan"), len(common))
    r_fa, n_fa = corr(model["forward"], real["t_a"])
    r_rb, n_rb = corr(model["reverse"], real["t_b"])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.6), facecolor="white")
    _panel(axes[0], real["t_a"], real["t_b"], order, "real t_a", "real t_b",
           "A  real interictal templates")
    _panel(axes[1], model["forward"], model["reverse"], order, "model forward", "model reverse",
           "B  subject-SNN model templates")
    fig.suptitle(f"{subject}: model forward vs real t_a  ρ={r_fa:+.2f} (n={n_fa}) | "
                 f"model reverse vs real t_b  ρ={r_rb:+.2f} (n={n_rb})", fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    outdir = ROOT / f"results/paper-ready-figure/{fig_name}/figures"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{fig_name}_realvsmodel"
    png, pdf = outdir / f"{stem}.png", outdir / f"{stem}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    meta = {"figure": stem, "subject": subject, "model_tag": tag,
            "model_forward_vs_real_t_a_spearman": r_fa, "n_common_fa": n_fa,
            "model_reverse_vs_real_t_b_spearman": r_rb, "n_common_rb": n_rb,
            "channels": order,
            "notes": ["Plotting-only; no SNN rerun.",
                      "Profiles min-max normalized per template for display; consistency judged by Spearman (order).",
                      "Positive forward~t_a and reverse~t_b => model reproduces the real interictal template order.",
                      "The 2x2 similarity matrix (stars) lives in Fig4B (kmeans2) rightmost panel; combined-S null in *_similarity."]}
    (outdir / f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {png}\nwrote {pdf}\nforward~t_a rho={r_fa:+.3f} (n={n_fa}) | reverse~t_b rho={r_rb:+.3f} (n={n_rb})")
    return outdir


def main():
    os.chdir(ROOT)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="epilepsiae_1146")
    ap.add_argument("--tag", default="epilepsiae_1146_twoend_equal_tsrc_s3")
    ap.add_argument("--fig-name", default="fig_subject_snn_epilepsiae_1146")
    ap.add_argument("--montage", default="narrow", choices=["narrow", "broad"])
    a = ap.parse_args()
    compose(a.subject, a.tag, a.fig_name, a.montage)


if __name__ == "__main__":
    main()
