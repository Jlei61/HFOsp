#!/usr/bin/env python3
"""Cohort runner: propagation entry-dispersion vs single-template-noise null.

For every masked PR-2 per-subject cluster, ask whether the earliest-detected
("entry") channel is dispersed BEYOND what a single noisy propagation template
predicts -- the load-bearing test for an "entry region" over the simpler "one
noisy template" explanation. See src/propagation_entry_dispersion.py for the
scientific framing and the contract clauses.

Inputs : results/interictal_propagation_masked/per_subject/*.json  (masked
         PR-2 labels + templates; phantom-rank safe)
         raw *_lagPat_withFreqCent.npz under the dataset roots (event matrices)
         SEEG coords via src.seeg_coord_loader (Epilepsiae MNI mm / Yuquan native)
Outputs: results/propagation_entry_dispersion/{cohort_summary.json,.csv,
         per_subject/<dataset>_<subject>.json}

Discipline: NEVER pool Epilepsiae (MNI) and Yuquan (native RAS) coordinates;
only per-subject scalar radii are aggregated, dataset-stratified.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interictal_propagation import (
    load_subject_propagation_events,
    _valid_event_indices,
    _center_rank_matrix,
)
from src.propagation_entry_dispersion import analyze_cluster, MIN_PARTICIPATING
from src.seeg_coord_loader import load_subject_coords

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("entry_dispersion")

# Path resolution -- matches scripts/run_interictal_propagation.py (scripts is
# not an importable package in this repo; every runner replicates this).
YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
MASKED_PER_SUBJECT = Path("results/interictal_propagation_masked/per_subject")
OUT_DIR = Path("results/propagation_entry_dispersion")


def _subject_dir(dataset: str, subject: str) -> Path:
    if dataset == "yuquan":
        return YUQUAN_ROOT / subject
    legacy = EPILEPSIAE_ROOT / subject / "all_recs"
    return legacy if legacy.exists() else EPILEPSIAE_ROOT / subject


def _load_coords(dataset: str, subject: str, channel_names: List[str]):
    """Return (coords (n,3) mm, mapped (n,) bool, coord_space) or (None,None,None)."""
    try:
        cr = load_subject_coords(dataset, subject, channel_names)
        if cr.coord_units != "mm":
            logger.warning("%s/%s coords not mm (%s) -> skip spatial",
                           dataset, subject, cr.coord_units)
            return None, None, None
        return (cr.coords_array_in_requested_order,
                cr.mapped_mask_in_requested_order, cr.coord_space)
    except Exception as ex:  # noqa: BLE001 - coords are optional
        logger.info("%s/%s no coords (%s)", dataset, subject, type(ex).__name__)
        return None, None, None


def analyze_subject(dataset: str, subject: str, json_path: Path,
                    n_reps: int) -> Optional[Dict[str, Any]]:
    d = json.load(open(json_path))
    ac = d.get("adaptive_cluster", {})
    chosen_k = ac.get("chosen_k")
    if not chosen_k or chosen_k < 2 or not ac.get("clusters"):
        return None

    sd = _subject_dir(dataset, subject)
    loaded = load_subject_propagation_events(sd)
    ranks, bools = loaded["ranks"], loaded["bools"]
    chn = loaded["channel_names"]

    # (2) CHANNEL-ALIGN, (3) LABEL-ALIGN -- raise loudly on mismatch.
    if chn != d["channel_names"]:
        raise ValueError(f"{dataset}/{subject}: loader channel_names != JSON "
                         f"channel_names (alignment broken)")
    labels = np.array(ac["labels"])
    ve = _valid_event_indices(bools, min_participating=MIN_PARTICIPATING)
    if labels.size != ve.size:
        raise ValueError(f"{dataset}/{subject}: labels {labels.size} != "
                         f"valid_events {ve.size} (LABEL-ALIGN)")

    centered_meta = _center_rank_matrix(ranks, bools, min_participation=10)
    coords, mapped, coord_space = _load_coords(dataset, subject, chn)

    # subject-level forward/reverse signature: most anti-correlated cluster pair
    corr = np.array(ac.get("inter_cluster_corr_matrix", []), dtype=float)
    min_corr = float(np.nanmin(corr)) if corr.size and np.isfinite(corr).any() else float("nan")

    # cluster-quality fields for the blend/continuum confound cross-tab (advisor):
    # low silhouette / mixture flag => a 'cluster' may be a loose blend or a
    # 1-D continuum, which mechanically inflates entry dispersion.
    mix = d.get("mixture", {})
    sil_chosen = next((s.get("median_silhouette") for s in ac.get("scan", [])
                       if s.get("k") == chosen_k), None)

    clusters_out: List[Dict[str, Any]] = []
    for ci, clu in enumerate(ac["clusters"]):
        cev = ve[labels == ci]
        res = analyze_cluster(
            ranks, bools, cev, clu["template_rank"], centered_meta,
            coords, mapped, n_reps=n_reps, seed=ci,
        )
        res["cluster_id"] = int(ci)
        res["fraction"] = float(clu.get("fraction", np.nan))
        res["template_raw_tau"] = float(clu.get("raw_tau", np.nan))
        clusters_out.append(res)

    return {
        "dataset": dataset,
        "subject": subject,
        "chosen_k": int(chosen_k),
        "stable_k": ac.get("stable_k"),
        "chosen_reason": ac.get("chosen_reason"),
        "n_channels": len(chn),
        "channel_names": chn,
        "coord_space": coord_space,
        "n_mapped_channels": int(mapped.sum()) if mapped is not None else 0,
        "coords_mm": coords.tolist() if coords is not None else None,
        "mapped": mapped.tolist() if mapped is not None else None,
        "min_inter_cluster_corr": min_corr,
        "silhouette_chosen_k": sil_chosen,
        "silhouette_k2": mix.get("silhouette_k2"),
        "dip_p": mix.get("dip_p"),
        "is_mixture": mix.get("is_mixture"),
        "possible_mixture": mix.get("possible_mixture"),
        "clusters": clusters_out,
    }


_CSV_COLUMNS = [
    "dataset", "subject", "n_channels", "chosen_k", "cluster_id", "n_events", "fraction",
    "neff", "top_share",
    "null_neff_pooled_mean", "p_neff_excess_pooled", "p_neff_concentrated_pooled",
    "null_neff_gauss_mean", "p_neff_excess_gauss", "p_neff_concentrated_gauss",
    "verdict",
    "obs_radius_mm", "null_radius_mean", "p_radius_excess",
    "n_mapped_entry", "n_mapped_channels_subject",
    "shape_spearman_prob_vs_trank", "modal_trank",
    "secondary_peak_trank", "secondary_peak_share",
    "downstream_cross_entry_rho", "downstream_n_groups",
    "within_cluster_tau", "centered_tau_full", "centered_tau_delta",
    "silhouette_chosen_k", "silhouette_k2", "dip_p", "is_mixture",
    "min_inter_cluster_corr", "coord_space",
]


def _verdict(p_exc_pool, p_exc_gauss, p_con_pool, alpha=0.05):
    """Three-way verdict from the null bracket. 'robust_excess' = exceeds BOTH
    nulls (entry dispersion beyond even a per-channel-noise template)."""
    if p_exc_pool is None:
        return "skipped"
    if p_exc_pool < alpha and p_exc_gauss is not None and p_exc_gauss < alpha:
        return "robust_excess"
    if p_exc_pool < alpha:
        return "fragile_excess"          # pooled-only (heteroscedasticity-explained)
    if p_con_pool is not None and p_con_pool < alpha:
        return "concentrated"            # endpoint-leaning
    return "consistent_one_template"


def _cluster_csv_row(subj: Dict[str, Any], c: Dict[str, Any]) -> Dict[str, Any]:
    n = c.get("null", {})
    sh = c.get("shape", {})
    di = c.get("downstream_invariance", {})
    st = c.get("stability", {})
    p_exc_pool = n.get("p_neff_excess")
    p_exc_gauss = n.get("p_neff_excess_gauss")
    p_con_pool = n.get("p_neff_concentrated")
    return {
        "dataset": subj["dataset"], "subject": subj["subject"],
        "n_channels": subj.get("n_channels"),
        "chosen_k": subj["chosen_k"], "cluster_id": c["cluster_id"],
        "n_events": c.get("n_events"), "fraction": round(c.get("fraction", float("nan")), 4),
        "neff": _r(c.get("neff")), "top_share": _r(c.get("top_share")),
        "null_neff_pooled_mean": _r(n.get("null_neff", {}).get("mean")),
        "p_neff_excess_pooled": _r(p_exc_pool),
        "p_neff_concentrated_pooled": _r(p_con_pool),
        "null_neff_gauss_mean": _r(n.get("null_neff_gauss", {}).get("mean")),
        "p_neff_excess_gauss": _r(p_exc_gauss),
        "p_neff_concentrated_gauss": _r(n.get("p_neff_concentrated_gauss")),
        "verdict": _verdict(p_exc_pool, p_exc_gauss, p_con_pool),
        "obs_radius_mm": _r(n.get("obs_radius_mm")),
        "null_radius_mean": _r(n.get("null_radius_mm", {}).get("mean")),
        "p_radius_excess": _r(n.get("p_radius_excess")),
        # CLUSTER-level: # of entry contacts that have coordinates (NOT the
        # subject's total mapped channels). Feeds the compact-sub-region next step.
        "n_mapped_entry": n.get("obs_radius_n_mapped_entry"),
        "n_mapped_channels_subject": subj["n_mapped_channels"],
        "shape_spearman_prob_vs_trank": _r(sh.get("spearman_prob_vs_trank")),
        "modal_trank": sh.get("modal_trank"),
        "secondary_peak_trank": sh.get("secondary_peak_trank"),
        "secondary_peak_share": _r(sh.get("secondary_peak_share")),
        "downstream_cross_entry_rho": _r(di.get("median_cross_entry_spearman")),
        "downstream_n_groups": di.get("n_groups"),
        "within_cluster_tau": _r(c.get("template_raw_tau")),
        "centered_tau_full": _r(st.get("centered_tau_full")),
        "centered_tau_delta": _r(st.get("centered_tau_delta")),
        "silhouette_chosen_k": _r(subj.get("silhouette_chosen_k")),
        "silhouette_k2": _r(subj.get("silhouette_k2")),
        "dip_p": _r(subj.get("dip_p")),
        "is_mixture": subj.get("is_mixture"),
        "min_inter_cluster_corr": _r(subj.get("min_inter_cluster_corr")),
        "coord_space": subj.get("coord_space"),
    }


def _r(x, nd=4):
    return round(float(x), nd) if x is not None and np.isfinite(float(x)) else None


def _cohort_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Headline (advisor 2026-06-08): LEAD with rejection of fixed single-point
    ignition (entry dispersed in ~every cluster); report excess-dispersion as
    a liberal/conservative BRACKET (pooled vs per-channel-Gaussian null) and the
    ROBUST subset (exceeds BOTH), then expose the cluster-quality / montage
    confound so the cross-tab can decide framing. NOT a region confirmation."""
    analyzed = [r for r in rows if r.get("p_neff_excess_pooled") is not None]
    alpha = 0.05

    def cnt(pred):
        return sum(1 for r in analyzed if pred(r))

    def frac(n):
        return _r(n / len(analyzed)) if analyzed else None

    verdicts = {}
    for r in analyzed:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1

    # entry dispersed at all (vs a degenerate single-point ignition: top_share~1
    # / neff~1). Report how many clusters have neff clearly > 1 (dispersed entry).
    dispersed = cnt(lambda r: (r.get("neff") or 0) > 1.5)

    excess_pool = cnt(lambda r: r["p_neff_excess_pooled"] < alpha)
    excess_gauss = cnt(lambda r: (r.get("p_neff_excess_gauss") is not None
                                  and r["p_neff_excess_gauss"] < alpha))
    robust = cnt(lambda r: r["verdict"] == "robust_excess")
    concen = cnt(lambda r: r["verdict"] == "concentrated")
    spatial = [r for r in analyzed if r.get("p_radius_excess") is not None]
    rad_excess = sum(1 for r in spatial if r["p_radius_excess"] < alpha)

    def med(key, pool=analyzed):
        v = [r[key] for r in pool if r.get(key) is not None]
        return _r(np.median(v)) if v else None

    # confound cross-tab: robust-excess clusters vs the rest, on cluster quality
    def stratum(pool, key):
        v = [r[key] for r in pool if r.get(key) is not None]
        return _r(np.median(v)) if v else None
    rob = [r for r in analyzed if r["verdict"] == "robust_excess"]
    nonrob = [r for r in analyzed if r["verdict"] != "robust_excess"]

    by_ds = {}
    for ds in ("epilepsiae", "yuquan"):
        sub = [r for r in analyzed if r["dataset"] == ds]
        if sub:
            by_ds[ds] = {
                "n_analyzed": len(sub),
                "robust_excess": sum(1 for r in sub if r["verdict"] == "robust_excess"),
                "fragile_excess": sum(1 for r in sub if r["verdict"] == "fragile_excess"),
                "concentrated": sum(1 for r in sub if r["verdict"] == "concentrated"),
                "consistent_one_template": sum(1 for r in sub if r["verdict"] == "consistent_one_template"),
            }

    return {
        "n_clusters_total": len(rows),
        "n_clusters_analyzed": len(analyzed),
        "alpha": alpha,
        "FIRM_rejection_fixed_single_point": {
            "n_clusters_entry_dispersed_neff_gt_1.5": dispersed,
            "fraction": frac(dispersed),
            "median_top_share": med("top_share"),
            "note": "entry is NOT a single fixed point in ~every cluster -> strict "
                    "two-fixed-endpoint ignition rejected (the firm result).",
        },
        "excess_dispersion_bracket": {
            "pooled_liberal": {"n": excess_pool, "fraction": frac(excess_pool)},
            "gauss_conservative": {"n": excess_gauss, "fraction": frac(excess_gauss)},
            "robust_both_nulls": {"n": robust, "fraction": frac(robust)},
            "note": "NOT confirmation of an independent entry structure: these "
                    "nulls reject single-template better than they confirm one; "
                    "robust subset is confounded by cluster quality / montage size "
                    "(see cross_tab). Confirmed picture = one jittery stereotyped "
                    "pathway whose early end is a small entry group that takes turns.",
        },
        "verdict_counts": verdicts,
        "endpoint_concentrated": {"n": concen, "fraction": frac(concen)},
        "spatial_radius_excess_pooled": {
            "n_spatial": len(spatial), "n": rad_excess,
            "fraction": _r(rad_excess / len(spatial)) if spatial else None,
        },
        "confound_cross_tab_robust_vs_rest": {
            "robust_excess": {
                "n": len(rob),
                "median_within_cluster_tau": stratum(rob, "within_cluster_tau"),
                "median_silhouette_k2": stratum(rob, "silhouette_k2"),
                "median_n_channels": stratum(rob, "n_channels"),
                "n_is_mixture": sum(1 for r in rob if r.get("is_mixture")),
            },
            "rest": {
                "n": len(nonrob),
                "median_within_cluster_tau": stratum(nonrob, "within_cluster_tau"),
                "median_silhouette_k2": stratum(nonrob, "silhouette_k2"),
                "median_n_channels": stratum(nonrob, "n_channels"),
                "n_is_mixture": sum(1 for r in nonrob if r.get("is_mixture")),
            },
        },
        "by_dataset": by_ds,
        "medians": {
            "neff": med("neff"), "top_share": med("top_share"),
            "shape_spearman_prob_vs_trank": med("shape_spearman_prob_vs_trank"),
            "downstream_cross_entry_rho": med("downstream_cross_entry_rho"),
            "centered_tau_delta": med("centered_tau_delta"),
            "n_channels": med("n_channels"),
        },
    }


def _write_cohort(all_rows: List[Dict[str, Any]], n_subjects: int) -> Dict[str, Any]:
    summary = _cohort_summary(all_rows)
    summary["n_subjects"] = n_subjects
    with open(OUT_DIR / "cohort_summary.json", "w") as fh:
        json.dump({"summary": summary, "clusters": all_rows}, fh, indent=2)
    with open(OUT_DIR / "cohort_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N subjects")
    ap.add_argument("--only", type=str, default="", help="comma list dataset_subject")
    ap.add_argument("--shard", type=str, default="", help="i/n: process files[i::n]")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip compute; rebuild cohort_summary from per_subject/*.json")
    args = ap.parse_args()

    (OUT_DIR / "per_subject").mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        all_rows = []
        stems = []
        for pf in sorted((OUT_DIR / "per_subject").glob("*.json")):
            subj = json.load(open(pf))
            stems.append(pf.stem)
            for c in subj.get("clusters", []):
                all_rows.append(_cluster_csv_row(subj, c))
        _write_cohort(all_rows, n_subjects=len(stems))
        logger.info("AGGREGATED %d subjects, %d clusters", len(stems), len(all_rows))
        return

    files = sorted(MASKED_PER_SUBJECT.glob("*.json"))
    only = set(args.only.split(",")) if args.only else None
    if only:
        files = [f for f in files if f.stem in only]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        files = files[i::n]
    if args.limit:
        files = files[: args.limit]

    all_rows: List[Dict[str, Any]] = []
    subjects_done = 0
    for f in files:
        stem = f.stem
        dataset = "yuquan" if stem.startswith("yuquan_") else "epilepsiae"
        subject = stem[len(dataset) + 1:]
        try:
            res = analyze_subject(dataset, subject, f, args.n_reps)
        except Exception as ex:  # noqa: BLE001
            logger.error("FAILED %s: %s", stem, ex)
            continue
        if res is None:
            continue
        with open(OUT_DIR / "per_subject" / f"{stem}.json", "w") as fh:
            json.dump(res, fh, indent=2)
        for c in res["clusters"]:
            all_rows.append(_cluster_csv_row(res, c))
        subjects_done += 1
        n_excess = sum(1 for c in res["clusters"]
                       if c.get("null", {}).get("p_neff_excess", 1) < 0.05)
        logger.info("%s: k=%d  excess-dispersion clusters=%d/%d",
                    stem, res["chosen_k"], n_excess, len(res["clusters"]))

    summary = _write_cohort(all_rows, n_subjects=subjects_done)

    logger.info("DONE: %d subjects, %d clusters", subjects_done, len(all_rows))
    br = summary["excess_dispersion_bracket"]
    logger.info("Cohort headline: entry dispersed %s/%s | excess bracket "
                "pooled=%s gauss=%s robust=%s | verdicts=%s",
                summary["FIRM_rejection_fixed_single_point"]["n_clusters_entry_dispersed_neff_gt_1.5"],
                summary["n_clusters_analyzed"],
                br["pooled_liberal"]["n"], br["gauss_conservative"]["n"],
                br["robust_both_nulls"]["n"], summary["verdict_counts"])


if __name__ == "__main__":
    main()
