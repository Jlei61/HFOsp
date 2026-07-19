#!/usr/bin/env python3
"""Axis-only comparison: endpoint-axis R3 vs gradient-primary R3 (seven-band).

Both runs share the IDENTICAL 17/167 event list, common mask (verified: byte-equal
coherent-permutation hashes), [0,10]s activations, sigma rule, N=161 grid, and
coherent all-contact null; only the projection axis differs (endpoint source->sink
cores, per-template A/B, vs gradient shared-else-own). This script reports, PER
BAND:
  - side-by-side D / cohort Delta / coherent cohort spatial-null p / seven-band maxT pFWER;
  - the DIRECT per-subject margin contrast endpoint-margin - gradient-margin
    (margin = D - median_k N), with median effect, n positive, paired two-sided
    Wilcoxon and a subject sign-flip p; folded per band and band->subject.

This direct paired contrast is the answer to "did the axis move the seven-band
result", NOT "endpoint has more stars than gradient" (significance-difference
fallacy). The axis+routing confound (endpoint per-template vs gradient shared-else-own)
is stated in every output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.topic5_gradient_grid_field as gg
from scipy.stats import wilcoxon

MS = _ROOT / "results/topic5_ictal_recruitment/field_concordance_grid_method_sensitivity"
EP = _ROOT / "results/topic5_ictal_recruitment/field_concordance_grid_endpoint_axis"
GRADIENT_DIR = MS / "n161_subject_fixed"
ENDPOINT_DIR = EP / "n161_endpoint"
OUT = EP
BANDS = ["delta_HYP_slow", "theta_preictal_PAC", "alpha_sharp_leq13", "beta_LVFA_low",
         "gamma_LVFA", "hg_low_ripple", "ripple_high"]


def _sign_flip_p(x, n_perm=100000, seed=20260719):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    obs = abs(float(np.mean(x)))
    rng = np.random.default_rng(seed)
    null = np.abs((rng.choice([-1.0, 1.0], size=(n_perm, x.size)) * x[None, :]).mean(axis=1))
    return float((1 + int(np.sum(null >= obs - 1e-15))) / (n_perm + 1))


def _perm_identity(grad_dir, ep_dir):
    g = pd.read_csv(grad_dir / "permutation_mapping_audit_summary.csv").set_index(
        ["subject", "seizure_idx"]).mapping_sha256.sort_index()
    e = pd.read_csv(ep_dir / "permutation_mapping_audit_summary.csv").set_index(
        ["subject", "seizure_idx"]).mapping_sha256.sort_index()
    return bool(g.index.equals(e.index) and (g == e).all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gradient-dir", default=str(GRADIENT_DIR))
    ap.add_argument("--endpoint-dir", default=str(ENDPOINT_DIR))
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args()
    gdir, edir, outdir = Path(args.gradient_dir), Path(args.endpoint_dir), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    perm_identical = _perm_identity(gdir, edir)

    gc = pd.read_csv(gdir / "multiband_cohort.csv").set_index("band")
    ec = pd.read_csv(edir / "multiband_cohort.csv").set_index("band")
    gs = pd.read_csv(gdir / "multiband_subject.csv")
    es = pd.read_csv(edir / "multiband_subject.csv")
    merged = gs.merge(es, on=["band", "subject"], suffixes=("_grad", "_ep"))
    merged["margin_diff"] = merged["delta_ep"] - merged["delta_grad"]   # endpoint - gradient margin

    # per-band side-by-side + direct paired contrast
    per_band = []
    for b in BANDS:
        sub = merged[merged.band == b]
        d = sub["margin_diff"].dropna().values
        try:
            wp = float(wilcoxon(sub["delta_ep"], sub["delta_grad"], alternative="two-sided").pvalue)
        except ValueError:
            wp = float("nan")
        per_band.append({
            "band": b,
            "gradient_delta_cohort": float(gc.loc[b, "delta_cohort_median"]),
            "endpoint_delta_cohort": float(ec.loc[b, "delta_cohort_median"]),
            "gradient_cohort_spatial_null_p": float(gc.loc[b, "coherent_cohort_spatial_null_p"]),
            "endpoint_cohort_spatial_null_p": float(ec.loc[b, "coherent_cohort_spatial_null_p"]),
            "gradient_seven_band_maxt_pfwer": float(gc.loc[b, "seven_band_maxt_pfwer"]),
            "endpoint_seven_band_maxt_pfwer": float(ec.loc[b, "seven_band_maxt_pfwer"]),
            "margin_diff_median_endpoint_minus_gradient": float(np.median(d)) if d.size else float("nan"),
            "n_subjects_endpoint_gt_gradient": int(np.sum(d > 0)),
            "direct_paired_wilcoxon_p": wp,
            "direct_sign_flip_p": _sign_flip_p(d),
        })
    pb = pd.DataFrame(per_band)
    pb.to_csv(outdir / "axis_only_endpoint_vs_gradient_per_band.csv", index=False)

    # folded band -> subject: one margin_diff per subject (median over bands)
    subj_fold = merged.groupby("subject")["margin_diff"].median()
    d_fold = subj_fold.dropna().values
    try:
        wp_fold = float(wilcoxon(d_fold, alternative="two-sided").pvalue) if not np.allclose(d_fold, 0) else float("nan")
    except ValueError:
        wp_fold = float("nan")
    contrast = {
        "contract": "axis_only_endpoint_vs_gradient_v1",
        "held_constant": "17/167 events, common mask, [0,10]s activation, sigma rule, N=161, "
                         "coherent all-contact null (same physical mapping)",
        "changed": "projection axis only: endpoint source->sink cores (per-template A/B) "
                   "vs gradient shared-else-own",
        "confound": "endpoint is per-template A/B while gradient-primary is shared-else-own; this "
                    "measures the endpoint package vs the gradient-primary package, NOT the axis alone",
        "identical_pipeline_verified_perm_hashes": perm_identical,
        "band_to_subject_folded_margin_diff": {
            "median_endpoint_minus_gradient": float(np.median(d_fold)) if d_fold.size else float("nan"),
            "n_subjects_endpoint_gt_gradient": int(np.sum(d_fold > 0)),
            "n_subjects": int(d_fold.size),
            "paired_two_sided_wilcoxon_p": wp_fold,
            "subject_sign_flip_p": _sign_flip_p(d_fold),
        },
        "seven_band_pfwer_pass_count": {
            "gradient": int((pb.gradient_seven_band_maxt_pfwer < 0.05).sum()),
            "endpoint": int((pb.endpoint_seven_band_maxt_pfwer < 0.05).sum()),
        },
        "interpretation_note": "pass-count delta (gradient vs endpoint) is descriptive only; the axis "
                               "effect is the DIRECT paired margin contrast above.",
    }
    (outdir / "axis_only_endpoint_vs_gradient_summary.json").write_text(json.dumps(contrast, indent=2, default=str))
    print("identical-pipeline perm hashes:", perm_identical)
    print(pb.round(4).to_string(index=False))
    print("\nband->subject folded margin_diff (endpoint - gradient):",
          json.dumps(contrast["band_to_subject_folded_margin_diff"], default=str))
    print("seven-band pFWER pass:", contrast["seven_band_pfwer_pass_count"])


if __name__ == "__main__":
    main()
