"""Model-vs-data template similarity statistic (Topic 4 Fig4D).

Quantifies how similar the subject-SNN MODEL templates (forward/reverse cluster mean rank) are to
the patient's REAL interictal templates (t_a/t_b typical_rank), with a CHANNEL-SHUFFLE permutation
null (controls for the small channel count).

Two panels:
  A  2x2 Spearman similarity matrix  rows = model {forward, reverse}, cols = data {t_a, t_b};
     each cell annotated with rho and its channel-shuffle permutation p.
  B  null distribution of the combined swap-consistency statistic
     S = mean(diagonal rho) - mean(off-diagonal rho)  with the observed S + permutation p.

Honesty: the model cores were PLACED at the data template-source channels, so the ENDPOINT match is
partly built in; the permutation p tests whether the *full channel order* (incl. intermediate
channels) aligns beyond chance, NOT construction specificity. The proper specificity control is the
core-location / axis-rotation null (re-sim with mislocated cores) -- see README.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures.plot_fig_subject_snn_realvsmodel import _real_templates, _model_templates

B_PERM = 10000


def _spear(a, b, names):
    common = [n for n in names if n in a and n in b]
    if len(common) < 4:
        return float("nan"), len(common), None, None
    va = np.array([a[n] for n in common]); vb = np.array([b[n] for n in common])
    rho = float(spearmanr(va, vb).correlation)
    return rho, len(common), va, vb


def _perm_p(va, vb, rho_obs, rng, B=B_PERM):
    """one-sided (positive) channel-shuffle permutation p for spearman(va, vb)."""
    null = np.array([spearmanr(va, rng.permutation(vb)).correlation for _ in range(B)])
    p_pos = (1 + np.sum(null >= rho_obs)) / (B + 1)
    return float(p_pos), null


def compose(subject, tag, fig_name, montage="narrow"):
    real = _real_templates(subject, montage)
    model = _model_templates(tag)
    rng = np.random.default_rng(0)

    rows = [("forward", model["forward"]), ("reverse", model["reverse"])]
    cols = [("t_a", real["t_a"]), ("t_b", real["t_b"])]
    M = np.full((2, 2), np.nan); P = np.full((2, 2), np.nan); N = np.zeros((2, 2), int)
    for i, (_, mvec) in enumerate(rows):
        for j, (_, dvec) in enumerate(cols):
            names = sorted(set(mvec) | set(dvec))
            rho, n, va, vb = _spear(mvec, dvec, names)
            M[i, j], N[i, j] = rho, n
            if va is not None:
                P[i, j], _ = _perm_p(va, vb, rho, rng)

    # combined swap-consistency S on the channel set common to ALL four
    common = sorted(set(model["forward"]) & set(model["reverse"]) & set(real["t_a"]) & set(real["t_b"]))
    mf = np.array([model["forward"][n] for n in common]); mr = np.array([model["reverse"][n] for n in common])
    ta = np.array([real["t_a"][n] for n in common]); tb = np.array([real["t_b"][n] for n in common])

    def swap_S(ta_, tb_):
        d = 0.5 * (spearmanr(mf, ta_).correlation + spearmanr(mr, tb_).correlation)
        o = 0.5 * (spearmanr(mf, tb_).correlation + spearmanr(mr, ta_).correlation)
        return d - o
    S_obs = swap_S(ta, tb)
    # null: permute the data channel labels (same perm on t_a & t_b -> preserves their relation,
    # breaks model<->data alignment)
    S_null = np.empty(B_PERM)
    for b in range(B_PERM):
        perm = rng.permutation(len(common))
        S_null[b] = swap_S(ta[perm], tb[perm])
    p_S = (1 + np.sum(S_null >= S_obs)) / (B_PERM + 1)

    # ---- figure: combined swap-consistency permutation null (the 2x2 matrix lives in Fig4C now) ----
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2), facecolor="white")
    ax.hist(S_null, bins=45, color="0.7", edgecolor="0.5", lw=0.3)
    ax.axvline(S_obs, color="#d62728", lw=2.2, label=f"observed S={S_obs:+.2f}")
    ax.set_xlabel("swap-consistency  S = mean(diag ρ) − mean(off-diag ρ)", fontsize=10)
    ax.set_ylabel("channel-shuffle null count", fontsize=10)
    ax.set_title(f"model–data swap-consistency permutation test\n"
                 f"S={S_obs:+.2f}, p={p_S:.4f}  (n_ch={len(common)}, B={B_PERM})", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()

    outdir = ROOT / f"results/paper-ready-figure/{fig_name}/figures"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{fig_name}_similarity"
    png, pdf = outdir / f"{stem}.png", outdir / f"{stem}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white"); plt.close(fig)
    meta = {"figure": stem, "subject": subject, "model_tag": tag,
            "spearman_matrix_rows_model_fwd_rev_cols_data_ta_tb": M.tolist(),
            "perm_p_matrix": P.tolist(), "n_common_matrix": N.tolist(),
            "swap_consistency_S_observed": float(S_obs), "swap_consistency_S_perm_p": float(p_S),
            "n_channels_S": len(common), "B_permutations": B_PERM,
            "method": "per-cell one-sided channel-shuffle permutation Spearman; S=mean(diag)-mean(offdiag), "
                      "null permutes data channel labels (same perm on t_a/t_b).",
            "caveat": ["Model cores placed at data template-source channels -> ENDPOINT match partly built in. "
                       "Permutation p tests full-channel-order alignment beyond chance, NOT construction specificity. "
                       "Proper specificity control = core-location / axis-rotation null (re-sim, mislocated cores)."]}
    (outdir / f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {png}\nwrote {pdf}")
    print(f"2x2 rho:\n{np.round(M,3)}\nperm p:\n{np.round(P,4)}\nS_obs={S_obs:+.3f} p_S={p_S:.4f} (n_ch={len(common)})")
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
