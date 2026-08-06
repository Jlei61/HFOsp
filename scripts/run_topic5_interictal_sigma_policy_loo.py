#!/usr/bin/env python3
"""§五 — Outcome-blind interictal smoothing-policy adjudication (leave-one-contact-out).

Decides which smoothing policy (subject_fixed vs frozen_per_model) is more
geometrically supported WITHOUT any ictal outcome: on each subject's frozen
TA/TB interictal plane, each supported contact is held out and its earliness is
reconstructed by kernel regression from the remaining contacts, using the sigma
that policy assigns to that template. Lower reconstruction error = better-matched
smoothing scale. No sigma-multiplier scan; sigmas are exactly the frozen
subject_fixed / frozen_per_model values. Shared-route subjects are identical
across policies (both use the shared-plane sigma for A and B).

Outputs to results/topic5_ictal_recruitment/field_concordance_grid_method_sensitivity/ :
    interictal_sigma_policy_loo_contact.csv
    interictal_sigma_policy_subject.csv
    interictal_sigma_policy_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.topic5_gradient_grid_field as gg
from scripts.run_topic5_figure3_ictal_grid_rebuild import SubjectField, load_parent_events

OUT = _ROOT / "results/topic5_ictal_recruitment/field_concordance_grid_method_sensitivity"
POLICIES = ["subject_fixed", "frozen_per_model"]


def _template_errors(pts, support, earliness, sigma):
    recon = gg.loo_contact_reconstruction(pts, support, earliness, sigma)
    err = recon - np.asarray(earliness, float)
    ok = np.isfinite(err)
    return recon, err, ok


def _template_metrics(err, support, ok):
    e = err[ok]
    w = np.asarray(support, float)[ok]
    if e.size == 0:
        return dict(n=0, unweighted_rmse=np.nan, median_abs_error=np.nan, support_weighted_rmse=np.nan)
    return dict(
        n=int(e.size),
        unweighted_rmse=float(np.sqrt(np.mean(e ** 2))),
        median_abs_error=float(np.median(np.abs(e))),
        support_weighted_rmse=float(np.sqrt(np.sum(w * e ** 2) / np.sum(w))) if np.sum(w) > 0 else np.nan,
    )


def main():
    events = load_parent_events()
    subjects = list(dict.fromkeys(events.subject.tolist()))
    contact_rows, subject_rows = [], []
    for s in subjects:
        sf = SubjectField(s)
        tmpl = {"A": (sf.pts_a, sf.support_a, sf.earliness_a, sf.contact_order),
                "B": (sf.pts_b, sf.support_b, sf.earliness_b, sf.contact_order)}
        for policy in POLICIES:
            sigma_a, sigma_b = sf.sigmas(policy)
            sigmas = {"A": sigma_a, "B": sigma_b}
            per_template = {}
            for tkey, (pts, sup, earl, names) in tmpl.items():
                recon, err, ok = _template_errors(pts, sup, earl, sigmas[tkey])
                m = _template_metrics(err, sup, ok)
                per_template[tkey] = m
                subject_rows.append({"subject": s, "route": sf.route, "policy": policy,
                                     "template": tkey, "sigma": sigmas[tkey], **m})
                for i, nm in enumerate(names):
                    if ok[i]:
                        contact_rows.append({
                            "subject": s, "route": sf.route, "policy": policy, "template": tkey,
                            "contact": str(nm), "sigma": sigmas[tkey],
                            "earliness_true": float(earl[i]), "earliness_recon": float(recon[i]),
                            "abs_error": float(abs(err[i])), "support": float(sup[i])})
            # fold A/B -> subject
            subject_rows.append({
                "subject": s, "route": sf.route, "policy": policy, "template": "AB_mean",
                "sigma": np.nan,
                "unweighted_rmse": float(np.nanmean([per_template["A"]["unweighted_rmse"],
                                                     per_template["B"]["unweighted_rmse"]])),
                "median_abs_error": float(np.nanmean([per_template["A"]["median_abs_error"],
                                                      per_template["B"]["median_abs_error"]])),
                "support_weighted_rmse": float(np.nanmean([per_template["A"]["support_weighted_rmse"],
                                                           per_template["B"]["support_weighted_rmse"]])),
                "n": int(per_template["A"]["n"] + per_template["B"]["n"])})

    OUT.mkdir(parents=True, exist_ok=True)
    contact_df = pd.DataFrame(contact_rows)
    subject_df = pd.DataFrame(subject_rows)
    contact_df.to_csv(OUT / "interictal_sigma_policy_loo_contact.csv", index=False)
    subject_df.to_csv(OUT / "interictal_sigma_policy_subject.csv", index=False)

    ab = subject_df[subject_df.template == "AB_mean"]
    # shared subjects must be identical across policies (fail-closed check)
    shared_diff = 0.0
    for s in ab.subject.unique():
        sf_row = ab[(ab.subject == s) & (ab.policy == "subject_fixed")].iloc[0]
        fp_row = ab[(ab.subject == s) & (ab.policy == "frozen_per_model")].iloc[0]
        if sf_row.route == "shared":
            shared_diff = max(shared_diff, abs(sf_row.unweighted_rmse - fp_row.unweighted_rmse))
    summary = {
        "contract": "interictal_sigma_policy_loo_v1",
        "adjudication": "outcome-blind leave-one-contact-out interictal earliness reconstruction",
        "n_subjects": int(ab.subject.nunique()),
        "shared_route_policy_identity_max_abs_diff": float(shared_diff),
        "cohort_median_AB_unweighted_rmse": {
            p: float(np.nanmedian(ab[ab.policy == p].unweighted_rmse)) for p in POLICIES},
        "cohort_median_AB_support_weighted_rmse": {
            p: float(np.nanmedian(ab[ab.policy == p].support_weighted_rmse)) for p in POLICIES},
        "own_route_only_median_AB_unweighted_rmse": {
            p: float(np.nanmedian(ab[(ab.policy == p) & (ab.route == "own_fallback")].unweighted_rmse))
            for p in POLICIES},
        "note": "lower RMSE = better geometric support for that smoothing scale; NOT an ictal outcome. "
                "Shared route identical across policies by construction.",
    }
    (OUT / "interictal_sigma_policy_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
