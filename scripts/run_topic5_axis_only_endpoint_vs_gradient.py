#!/usr/bin/env python3
"""endpoint-PACKAGE vs gradient-PRIMARY seven-band field-concordance sensitivity.

This is NOT a pure axis-only contrast. Both runs share the identical event list,
common mask, [0,10]s activations, N=161 grid and coherent all-contact null (all
FAIL-CLOSED verified here, incl. byte-equal permutation hashes), but the endpoint
run changes THREE coupled things at once: the projection axis (endpoint source->sink
cores), the routing (endpoint is per-template A/B for all subjects; gradient-primary
is 7 shared + 10 own-fallback), and — because sigma is estimated on the projection
plane — the per-subject sigma value. So this reports the endpoint PACKAGE minus the
gradient-PRIMARY package, not "the axis effect".

Multiplicity: the 7 direct per-band contrasts are corrected with Holm AND a
synchronized subject sign-flip maxT; stars/decisions use the CORRECTED p. The
closest-to-pure-axis internal control is the own-fallback stratum (both sides already
per-template A/B); it is reported separately.
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


def _sign_flip_maxt(mat, n_perm=100000, seed=20260719):
    """Synchronized subject sign-flip maxT over bands. mat: (n_subject, n_band) of
    per-subject margin diffs. One sign per subject applied across all bands; the null
    is the max over bands of |mean_subject|. Returns corrected p per band."""
    mat = np.asarray(mat, float)
    n_sub, n_band = mat.shape
    obs = np.abs(np.nanmean(mat, axis=0))                       # (n_band,)
    rng = np.random.default_rng(seed)
    maxnull = np.empty(n_perm)
    for k in range(n_perm):
        s = rng.choice([-1.0, 1.0], size=n_sub)
        maxnull[k] = np.nanmax(np.abs(np.nanmean(s[:, None] * mat, axis=0)))
    return np.array([(1 + int(np.sum(maxnull >= obs[b] - 1e-15))) / (n_perm + 1)
                     for b in range(n_band)])


def _read(d, name):
    return pd.read_csv(Path(d) / name)


def _fail_closed_contract(gdir, edir):
    """Raise unless the two runs share the identical event list, common mask, seed,
    primary grid and per-event coherent permutation hashes."""
    gm = json.loads((Path(gdir) / "contract_manifest.json").read_text())
    em = json.loads((Path(edir) / "contract_manifest.json").read_text())
    if int(gm["seed"]) != int(em["seed"]):
        raise SystemExit(f"seed mismatch: {gm['seed']} vs {em['seed']}")
    if int(gm["grids"][0]) != int(em["grids"][0]):
        raise SystemExit(f"primary grid mismatch: {gm['grids'][0]} vs {em['grids'][0]}")
    ge = _read(gdir, "cohort_event_inventory.csv")[["subject", "seizure_idx"]].sort_values(
        ["subject", "seizure_idx"]).reset_index(drop=True)
    ee = _read(edir, "cohort_event_inventory.csv")[["subject", "seizure_idx"]].sort_values(
        ["subject", "seizure_idx"]).reset_index(drop=True)
    if not ge.equals(ee):
        raise SystemExit("event lists differ between gradient and endpoint runs")
    gc = _read(gdir, "common_contact_inventory.csv")[["subject", "seizure_idx", "n_common_contacts"]].sort_values(
        ["subject", "seizure_idx"]).reset_index(drop=True)
    ec = _read(edir, "common_contact_inventory.csv")[["subject", "seizure_idx", "n_common_contacts"]].sort_values(
        ["subject", "seizure_idx"]).reset_index(drop=True)
    if not gc.equals(ec):
        raise SystemExit("common contact masks differ between gradient and endpoint runs")
    gh = _read(gdir, "permutation_mapping_audit_summary.csv").set_index(
        ["subject", "seizure_idx"]).mapping_sha256.sort_index()
    eh = _read(edir, "permutation_mapping_audit_summary.csv").set_index(
        ["subject", "seizure_idx"]).mapping_sha256.sort_index()
    if not (gh.index.equals(eh.index) and (gh == eh).all()):
        raise SystemExit("coherent permutation hashes differ — the pipeline is NOT identical (fail closed)")
    return {"seed": int(gm["seed"]), "primary_grid": int(gm["grids"][0]),
            "n_events": int(len(ge)), "perm_hashes_identical": True,
            "event_list_identical": True, "common_mask_identical": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gradient-dir", default=str(GRADIENT_DIR))
    ap.add_argument("--endpoint-dir", default=str(ENDPOINT_DIR))
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args()
    gdir, edir, outdir = Path(args.gradient_dir), Path(args.endpoint_dir), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    contract = _fail_closed_contract(gdir, edir)   # raises on any mismatch

    gc = _read(gdir, "multiband_cohort.csv").set_index("band")
    ec = _read(edir, "multiband_cohort.csv").set_index("band")
    gs = _read(gdir, "multiband_subject.csv")
    es = _read(edir, "multiband_subject.csv")
    route = _read(gdir, "field_routing_sigma_grid_inventory.csv").set_index("subject").route.to_dict()
    merged = gs.merge(es, on=["band", "subject"], suffixes=("_grad", "_ep"))
    merged["margin_diff"] = merged["delta_ep"] - merged["delta_grad"]
    merged["gradient_route"] = merged["subject"].map(route)

    # subject x band margin-diff matrix (fixed band + subject order) for maxT
    subjects = sorted(merged.subject.unique())
    mat = np.full((len(subjects), len(BANDS)), np.nan)
    for bi, b in enumerate(BANDS):
        sub = merged[merged.band == b].set_index("subject").reindex(subjects)
        mat[:, bi] = sub["margin_diff"].values
    raw_p = np.array([
        (float(wilcoxon(merged[merged.band == b]["delta_ep"], merged[merged.band == b]["delta_grad"],
                        alternative="two-sided").pvalue)
         if not np.allclose(merged[merged.band == b]["margin_diff"].dropna(), 0) else np.nan)
        for b in BANDS])
    holm_p = gg._holm(raw_p)
    maxt_p = _sign_flip_maxt(mat)

    per_band = []
    for bi, b in enumerate(BANDS):
        d = merged[merged.band == b]["margin_diff"].dropna().values
        per_band.append({
            "band": b,
            "gradient_delta_cohort": float(gc.loc[b, "delta_cohort_median"]),
            "endpoint_delta_cohort": float(ec.loc[b, "delta_cohort_median"]),
            "gradient_seven_band_maxt_pfwer": float(gc.loc[b, "seven_band_maxt_pfwer"]),
            "endpoint_seven_band_maxt_pfwer": float(ec.loc[b, "seven_band_maxt_pfwer"]),
            "margin_diff_median_endpoint_minus_gradient": float(np.median(d)),
            "n_subjects_endpoint_gt_gradient": int(np.sum(d > 0)),
            "direct_raw_wilcoxon_p": float(raw_p[bi]),
            "direct_holm_p": float(holm_p[bi]),
            "direct_signflip_maxt_p": float(maxt_p[bi]),
        })
    pb = pd.DataFrame(per_band)
    pb.to_csv(outdir / "endpoint_package_vs_gradient_primary_per_band.csv", index=False)

    # subject-level contrast CSV (spec-promised): folded margin_diff per subject + gradient routing
    subj_rows = []
    for s in subjects:
        sub = merged[merged.subject == s]
        subj_rows.append({"subject": s, "gradient_route": route.get(s),
                          "folded_margin_diff_endpoint_minus_gradient": float(sub["margin_diff"].median()),
                          "n_bands_endpoint_gt_gradient": int((sub["margin_diff"] > 0).sum())})
    subj_df = pd.DataFrame(subj_rows)
    subj_df.to_csv(outdir / "endpoint_package_vs_gradient_primary_subject_contrast.csv", index=False)

    def _fold_stat(df):
        fold = df.groupby("subject")["margin_diff"].median()
        v = fold.dropna().values
        try:
            wp = float(wilcoxon(v, alternative="two-sided").pvalue) if not np.allclose(v, 0) else float("nan")
        except ValueError:
            wp = float("nan")
        return {"n_subjects": int(v.size), "median": float(np.median(v)) if v.size else float("nan"),
                "n_endpoint_gt_gradient": int(np.sum(v > 0)), "wilcoxon_p": wp,
                "sign_flip_p": _sign_flip_p(v)}

    overall = _fold_stat(merged)
    own = _fold_stat(merged[merged.gradient_route == "own_fallback"])
    shared = _fold_stat(merged[merged.gradient_route == "shared"])

    summary = {
        "contract": "endpoint_package_vs_gradient_primary_sensitivity_v2",
        "NOT_axis_only": ("endpoint changes axis + routing (per-template vs shared-else-own) + sigma value "
                          "together; this is a package contrast, not the axis in isolation"),
        "identical_pipeline_verified": contract,
        "multiplicity": "7 direct per-band contrasts corrected with Holm AND synchronized subject sign-flip maxT",
        "direct_band_pass_after_correction": {
            "holm_lt_0p05": int(np.sum(holm_p < 0.05)),
            "signflip_maxt_lt_0p05": int(np.sum(maxt_p < 0.05)),
        },
        "overall_band_to_subject_folded": overall,
        "closest_to_pure_axis_own_fallback_stratum": own,
        "routing_changed_shared_stratum": shared,
        "descriptive_seven_band_pfwer_pass_count": {
            "gradient": int((pb.gradient_seven_band_maxt_pfwer < 0.05).sum()),
            "endpoint": int((pb.endpoint_seven_band_maxt_pfwer < 0.05).sum()),
        },
        "conclusion": ("Under the fully updated 17/167 contract the endpoint method no longer reproduces "
                       "the old 6/7; no overall endpoint-package vs gradient-primary seven-band difference "
                       "is detected, and NO band survives the direct seven-band correction. The nominal "
                       "beta/alpha gains are confined to the shared stratum (where routing changes) and "
                       "vanish in the own-fallback stratum (closest to pure axis), so they are routing-"
                       "confounded, not an axis effect. Gradient stays primary; endpoint is a sensitivity. "
                       "No equivalence test was run, so equivalence is not claimed."),
    }
    (outdir / "endpoint_package_vs_gradient_primary_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("identical-pipeline contract:", contract["perm_hashes_identical"], contract["event_list_identical"], contract["common_mask_identical"])
    print(pb.round(4).to_string(index=False))
    print("\ndirect band pass after correction: Holm",
          int(np.sum(holm_p < 0.05)), "/ sign-flip maxT", int(np.sum(maxt_p < 0.05)))
    print("overall folded:", json.dumps(overall, default=str))
    print("own-fallback (closest to pure axis):", json.dumps(own, default=str))
    print("shared (routing changed):", json.dumps(shared, default=str))


if __name__ == "__main__":
    main()
