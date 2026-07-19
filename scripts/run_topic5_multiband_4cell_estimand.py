#!/usr/bin/env python3
"""§六 — Unified multiband estimand across the R2/R3 x subject_fixed/frozen_per_model 2x2.

Consumes the per-band subject-level D and coherent null draws N saved by the two
smoothing-policy runs (each `multiband_subject_null_draws.npz` carries R3 as
``D``/``N`` and R2 as ``D_r2``/``N_r2``). For each method cell m, subject s, band b,
draw k:

    c[s,b,m]     = median_k N[s,b,m,k]
    Eobs[s,b,m]  = D[s,b,m] - c[s,b,m]
    Enull[s,b,m,k] = N[s,b,m,k] - c[s,b,m]
    Tobs[b,m]    = median_subject Eobs[s,b,m]        (the cohort bar)
    Tnull[b,m,k] = median_subject Enull[s,b,m,k]

Outputs a per-cell seven-band coherent maxT AND a joint 28-family (4 cells x 7
bands) maxT, verifies the vectorized pFWER against a per-draw reference, and
saves the full D/N/Eobs/Enull tensors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ROOT = _ROOT / "results/topic5_ictal_recruitment/field_concordance_grid_method_sensitivity"
POLICY_DIRS = {"subject_fixed": ROOT / "n161_subject_fixed",
               "frozen_per_model": ROOT / "n161_frozen_per_model"}
BANDS = ["delta_HYP_slow", "theta_preictal_PAC", "alpha_sharp_leq13", "beta_LVFA_low",
         "gamma_LVFA", "hg_low_ripple", "ripple_high"]


def _load_cells(policy_dirs):
    """Return cells {name: (D[s,b], N[s,b,k])} for the 4 method x policy combinations."""
    cells = {}
    for policy, d in policy_dirs.items():
        npz = np.load(d / "multiband_subject_null_draws.npz", allow_pickle=True)
        cells[f"R3::{policy}"] = (np.asarray(npz["D"], float), np.asarray(npz["N"], float))
        cells[f"R2::{policy}"] = (np.asarray(npz["D_r2"], float), np.asarray(npz["N_r2"], float))
    return cells


def _maxt_from_tnull(Tobs, Tnull, family_mask):
    """One-sided upper maxT p over an arbitrary family (mask over the flattened (b,m) axis).

    Tobs, Tnull flattened over (band, cell): Tobs (F,), Tnull (F, K). family_mask (F,) bool.
    M[k] = max over family members of Tnull[:,k]; p[f] = (1+#{M>=Tobs[f]})/(K+1).
    """
    K = Tnull.shape[1]
    M = np.max(np.where(family_mask[:, None], Tnull, -np.inf), axis=0)   # (K,)
    p = np.array([(1 + int(np.sum(M >= Tobs[f] - 1e-15))) / (K + 1) for f in range(Tobs.size)])
    return p, M


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-fixed-dir", default=str(POLICY_DIRS["subject_fixed"]))
    ap.add_argument("--frozen-per-model-dir", default=str(POLICY_DIRS["frozen_per_model"]))
    ap.add_argument("--outdir", default=str(ROOT))
    args = ap.parse_args()
    policy_dirs = {"subject_fixed": Path(args.subject_fixed_dir),
                   "frozen_per_model": Path(args.frozen_per_model_dir)}
    outdir = Path(args.outdir)
    for d in policy_dirs.values():
        if not (d / "multiband_subject_null_draws.npz").exists():
            raise SystemExit(f"missing policy run output: {d} (run both smoothing policies first)")
    cells = _load_cells(policy_dirs)
    cell_names = list(cells)                       # 4 cells
    n_cell = len(cell_names)
    n_band = len(BANDS)
    n_subj, _, n_perm = cells[cell_names[0]][1].shape

    # assemble D[s,b,m], N[s,b,m,k]
    D = np.full((n_subj, n_band, n_cell), np.nan)
    N = np.full((n_subj, n_band, n_cell, n_perm), np.nan)
    for mi, name in enumerate(cell_names):
        Dm, Nm = cells[name]
        D[:, :, mi] = Dm
        N[:, :, mi, :] = Nm
    c = np.median(N, axis=3)                        # (s,b,m)
    Eobs = D - c
    Enull = N - c[..., None]
    Tobs = np.median(Eobs, axis=0)                 # (b,m)  cohort bar
    Tnull = np.median(Enull, axis=0)               # (b,m,k)

    # flatten (band, cell)
    Tobs_f = Tobs.reshape(-1)                       # (F,) F = n_band*n_cell = 28
    Tnull_f = Tnull.reshape(n_band * n_cell, n_perm)
    cell_of = np.array([mi for _b in range(n_band) for mi in range(n_cell)])
    band_of = np.array([_b for _b in range(n_band) for mi in range(n_cell)])

    rows = []
    # per-cell seven-band coherent maxT (family = the 7 bands of that cell)
    percell_p = np.full(Tobs_f.size, np.nan)
    for mi in range(n_cell):
        fam = (cell_of == mi)
        p, _ = _maxt_from_tnull(Tobs_f, Tnull_f, fam)
        percell_p[fam] = p[fam]
    # joint 28-family maxT (family = all cells x bands)
    joint_p, M_joint = _maxt_from_tnull(Tobs_f, Tnull_f, np.ones(Tobs_f.size, bool))

    for f in range(Tobs_f.size):
        rows.append({
            "cell": cell_names[cell_of[f]], "band": BANDS[band_of[f]],
            "Tobs_cohort_bar": float(Tobs_f[f]),
            "per_cell_seven_band_maxt_pfwer": float(percell_p[f]),
            "joint_28_family_maxt_pfwer": float(joint_p[f]),
            "n_positive_subjects": int(np.sum(Eobs[:, band_of[f], cell_of[f]] > 0)),
        })

    # --- formal 2x2 contrasts on Eobs (P1-1: test the DIFFERENCES directly, not
    # "R3 significant while R2 is not"). Fold band -> subject median, paired across
    # subjects with a two-sided Wilcoxon + a subject sign-flip permutation p. ---
    from scipy.stats import wilcoxon

    def _idx(name):
        return cell_names.index(name)
    i_r3s, i_r2s = _idx("R3::subject_fixed"), _idx("R2::subject_fixed")
    i_r3f, i_r2f = _idx("R3::frozen_per_model"), _idx("R2::frozen_per_model")
    readout = 0.5 * ((Eobs[:, :, i_r3s] - Eobs[:, :, i_r2s]) + (Eobs[:, :, i_r3f] - Eobs[:, :, i_r2f]))
    sigma = 0.5 * ((Eobs[:, :, i_r3f] - Eobs[:, :, i_r3s]) + (Eobs[:, :, i_r2f] - Eobs[:, :, i_r2s]))
    interaction = ((Eobs[:, :, i_r3f] - Eobs[:, :, i_r2f]) - (Eobs[:, :, i_r3s] - Eobs[:, :, i_r2s]))

    def _sign_flip_p(x, n_perm=100000, seed=20260719):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            return float("nan")
        obs = abs(float(np.mean(x)))
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_perm, x.size))
        null = np.abs((signs * x[None, :]).mean(axis=1))
        return float((1 + int(np.sum(null >= obs - 1e-15))) / (n_perm + 1))

    contrast_rows = []
    for cname, mat in [("readout_R3_minus_R2", readout), ("sigma_frozen_minus_subjectfixed", sigma),
                       ("interaction_readout_x_sigma", interaction)]:
        subj = np.nanmedian(mat, axis=1)                 # band -> subject median
        subj = subj[np.isfinite(subj)]
        try:
            wp = float(wilcoxon(subj, alternative="two-sided").pvalue) if not np.allclose(subj, 0) else float("nan")
        except ValueError:
            wp = float("nan")
        contrast_rows.append({
            "contrast": cname, "n_subjects": int(subj.size),
            "median_effect": float(np.median(subj)),
            "iqr_low": float(np.percentile(subj, 25)), "iqr_high": float(np.percentile(subj, 75)),
            "n_positive": int(np.sum(subj > 0)),
            "paired_two_sided_wilcoxon_p": wp,
            "subject_sign_flip_p": _sign_flip_p(subj)})

    # --- verification: vectorized joint pFWER == per-draw reference ---
    M_ref = np.array([max(Tnull_f[f, k] for f in range(Tobs_f.size)) for k in range(n_perm)])
    joint_ref = np.array([(1 + int(np.sum(M_ref >= Tobs_f[f] - 1e-15))) / (n_perm + 1)
                          for f in range(Tobs_f.size)])
    max_p_err = float(np.max(np.abs(joint_p - joint_ref)))
    assert max_p_err < 1e-12, f"joint maxT vectorized != per-draw reference ({max_p_err})"

    import pandas as pd
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / "multiband_4cell_estimand.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(outdir / "multiband_4cell_contrasts.csv", index=False)
    np.savez_compressed(outdir / "multiband_4cell_tensors.npz",
                        D=D, N=N, Eobs=Eobs, Enull=Enull, Tobs=Tobs, Tnull=Tnull,
                        cells=np.array(cell_names), bands=np.array(BANDS))
    summary = {
        "contract": "multiband_4cell_estimand_v1",
        "cells": cell_names, "bands": BANDS, "n_subjects": int(n_subj), "n_perm": int(n_perm),
        "estimand": "Eobs[s,b,m]=D-median_k N; Tobs[b,m]=median_subject Eobs (cohort bar)",
        "families": {"per_cell": "seven bands within each method cell",
                     "joint": "28 = 4 cells x 7 bands (single-step maxT)"},
        "verification_joint_maxt_vectorized_vs_perdraw_max_abs_p_err": max_p_err,
        "n_joint_significant_0p05": int(np.sum(joint_p < 0.05)),
        "n_per_cell_significant_0p05": int(np.sum(percell_p < 0.05)),
        "formal_contrasts": {r["contrast"]: {"median_effect": r["median_effect"],
                                             "paired_wilcoxon_p": r["paired_two_sided_wilcoxon_p"],
                                             "sign_flip_p": r["subject_sign_flip_p"]}
                             for r in contrast_rows},
        "attribution_note": "readout/sigma/interaction are DIRECT paired contrasts on Eobs (band->subject "
                            "fold). Do NOT attribute a difference from 'one cell significant, another not' "
                            "(significance-difference fallacy); use these contrasts.",
        "claim_boundary": "a cell being significant while another is not is NOT evidence that the "
                          "methods differ; report the shared positive structure and the direct contrasts.",
    }
    (outdir / "multiband_4cell_estimand_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print("\n--- per (cell, band) Tobs / per-cell pFWER / joint-28 pFWER ---")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
