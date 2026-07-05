#!/usr/bin/env python3
"""Topic 5 event-resolved interictal axis_bias — PILOT runner (secondary, exploratory).

Spec: docs/superpowers/specs/2026-06-25-topic5-event-resolved-axis-bias-design.md (v2)
Plan: docs/superpowers/plans/2026-06-25-topic5-event-resolved-axis-bias.md (Task 7)

Per pilot subject: load broad A/B labels + per-event ranks (C1-guarded), map cluster_id→t_a/t_b
(C2), build each class's own plane + pinned sigma + subject-mean ictal field (C3/C4/C6), then
compute the per-event mirror-invariant field alignment M (primary), the 1D companion M1d, the
block-level A/B separation null R2 (C7), and participation diagnostics. Writes per_subject JSON
+ pilot_summary.json.

HARD STOP: cohort run is forbidden here (spec §8). This script refuses to run without --pilot or
--subjects, and never aggregates a cohort verdict — that awaits the human advisor (user) sign-off.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.topic5_event_resolved_alignment as erm
from src.propagation_contact_plane_readout import make_plane_grid

GEOM_BROAD = Path("results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects")
ICTAL_CACHE = Path("results/topic5_ictal_recruitment/t0_feature_cache")   # same cache the A-line primary used
OUT = Path("results/topic5_ictal_recruitment/event_resolved_alignment")
RNG_SEED = 20260625
PILOT_SUBJECTS = ["epilepsiae_1077", "epilepsiae_1125", "epilepsiae_922", "yuquan_zhangbichen"]
ACTIVATION_KEY = {"broadband": "bb_auc", "hfa": "hfa_auc"}


def _vec_in_order(name_to_val, order):
    return np.array([name_to_val.get(n, np.nan) for n in order], float)


def _subsample_bundle(bundle, max_per_class, rng):
    """Cap events per class for the (expensive) per-event field metric. The per-event alignment
    DISTRIBUTION + dispersion are well-estimated from ~1500 events/class; blocks stay represented.
    Fixed-RNG, documented subsample (spec §3.4 descriptive tier). diagnostics run on FULL data."""
    labels = bundle["labels"]
    keep = []
    for g in (0, 1):
        idx = np.where(labels == g)[0]
        if idx.size > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    out = dict(bundle)
    out["masked"] = bundle["masked"][:, keep]
    out["bools"] = bundle["bools"][:, keep]
    out["labels"] = bundle["labels"][keep]
    out["block_ids"] = bundle["block_ids"][keep]
    out["event_abs_times"] = bundle["event_abs_times"][keep]
    out["valid_ev"] = bundle["valid_ev"][keep]
    out["n_analyzed"] = int(keep.size)
    return out


def _subject_ictal_by_channel(ds_sid, activation_key):
    npz = ICTAL_CACHE / f"{ds_sid}.npz"
    js = ICTAL_CACHE / f"{ds_sid}.json"
    if not npz.exists() or not js.exists():
        return None, "no_ictal_cache"
    data = np.load(npz, allow_pickle=True)
    meta = json.load(open(js))
    elig = meta.get("eligible_idxs")           # C6: from sidecar JSON, not npz
    if not elig:
        return None, "no_eligible_idxs"
    chans = [str(x) for x in data["channels"]]
    cols = []
    for idx in elig:
        k = f"{activation_key}__{idx}"
        if k in data.files:
            cols.append(np.asarray(data[k], float))
    if not cols:
        return None, "no_activation_keys"
    mean_act = np.nanmean(np.vstack(cols), axis=0)         # subject-mean over eligible seizures
    return {n: float(v) for n, v in zip(chans, mean_act) if np.isfinite(v)}, "ok"


def _run_subject(ds_sid, *, activation_key, rng, n_perm_m1d, n_perm_r2, max_per_class):
    dataset, subject = ds_sid.split("_", 1)
    ta_f = GEOM_BROAD / f"{ds_sid}_t_a.json"
    tb_f = GEOM_BROAD / f"{ds_sid}_t_b.json"
    if not (ta_f.exists() and tb_f.exists()):
        return {"subject_id": ds_sid, "status": "no_broad_planes"}
    try:
        bundle = erm.load_event_labels_ranks(dataset, subject)
    except FileNotFoundError as e:
        return {"subject_id": ds_sid, "status": f"load_error:{e}"}
    except ValueError as e:
        return {"subject_id": ds_sid, "status": f"c1_violation:{e}"}

    plane_a = json.load(open(ta_f)); plane_b = json.load(open(tb_f))
    order = bundle["channel_names"]
    ta_rank = _vec_in_order({c["name"]: c["typical_rank"] for c in plane_a["channels"]}, order)
    tb_rank = _vec_in_order({c["name"]: c["typical_rank"] for c in plane_b["channels"]}, order)
    c0 = np.asarray(bundle["cluster_template_ranks"][0], float)
    c1 = np.asarray(bundle["cluster_template_ranks"][1], float)
    cmap = erm.map_clusters_to_templates(c0, c1, ta_rank, tb_rank)
    if cmap["ambiguous"]:
        return {"subject_id": ds_sid, "status": "cluster_map_ambiguous",
                "diag_minus_offdiag": cmap["diag_minus_offdiag"], "corr_matrix": cmap["corr_matrix"]}

    plane_of = {"t_a": plane_a, "t_b": plane_b}
    plane_by_label = {k: plane_of[cmap["map"][k]] for k in (0, 1)}

    ictal_by_ch, ist = _subject_ictal_by_channel(ds_sid, activation_key)
    if ictal_by_ch is None:
        return {"subject_id": ds_sid, "status": ist}

    X, Y = make_plane_grid()
    sigma_by_label, ictal_field_by_label = {}, {}
    for k in (0, 1):
        pl = plane_by_label[k]
        sig = erm.class_template_sigma(pl, X=X, Y=Y)
        sigma_by_label[k] = sig
        ictal_field_by_label[k] = erm.make_subject_ictal_field(pl, ictal_by_ch, sigma=sig, X=X, Y=Y)

    # diagnostics on FULL data (true participation); expensive metrics on a fixed subsample
    diag = erm.participation_diagnostics(bundle["bools"], bundle["labels"], bundle["block_ids"])
    bs = _subsample_bundle(bundle, max_per_class, rng)

    M = erm.per_event_field_alignment(bs, plane_by_label=plane_by_label,
                                      ictal_field_by_label=ictal_field_by_label,
                                      sigma_by_label=sigma_by_label)
    M1d = erm.per_event_1d_alignment(bs, ictal_by_ch, n_perm=n_perm_m1d, rng=rng)

    # R2 (P0 fix): pass BOTH per-event aligns (under each class plane) so the within-block
    # label permutation re-picks under the shuffled label's plane (pays plane-selection cost)
    align0_arr = np.array([r["align0"] if r["align0"] is not None else np.nan for r in M["per_event"]], float)
    align1_arr = np.array([r["align1"] if r["align1"] is not None else np.nan for r in M["per_event"]], float)
    R2 = erm.class_separation_block_null(align0_arr, align1_arr, bs["labels"], bs["block_ids"],
                                         n_perm=n_perm_r2, rng=rng)

    # per-class M summary (S1a: direct dispersion = the std effect)
    by_class = {}
    for g in (0, 1):
        vals = np.array([r["align"] for r in M["per_event"]
                         if r["status"] == "ok" and r["label"] == g], float)
        if vals.size:
            top = sorted([r for r in M["per_event"] if r["status"] == "ok" and r["label"] == g],
                         key=lambda r: -r["align"])[:5]
            m1d_vals = [r["align1d"] for r in M1d.get("per_event", []) if r["label"] == g]
            by_class[f"class_{g}"] = {
                "template": cmap["map"][g], "n_usable": int(vals.size),
                "median_align": float(np.median(vals)),
                "iqr_align": float(np.subtract(*np.percentile(vals, [75, 25]))),
                "std_align": float(np.std(vals)),
                "aligns": [round(float(v), 4) for v in vals],          # per-event distribution (for figures)
                "m1d_aligns": [round(float(v), 4) for v in m1d_vals],  # replay-adjacent companion distribution
                "top5_events": [{"event_idx": r["event_idx"], "abs_time": r["abs_time"],
                                 "block_id": r["block_id"], "align": r["align"]} for r in top],
            }
    return {
        "subject_id": ds_sid, "dataset": dataset, "status": "ok",
        "activation": activation_key, "n_channels_broad": len(order),
        # full-data counts are PARTICIPATION-only; all inference is on the analyzed subsample
        "n_events_valid_full": int(bundle["labels"].size), "n_blocks_full": bundle["n_blocks"],
        "n_events_analyzed": int(bs["n_analyzed"]), "max_per_class": max_per_class,
        "cluster_map": cmap["map"], "cluster_map_margin": cmap["diag_minus_offdiag"],
        # M / R2 inference basis (subsample): report usable n + usable/R2 blocks NEXT TO every p
        "M_usable_fraction": M["usable_fraction"], "M_n_usable": M["n_usable"],
        "M_n_blocks_usable": M["n_blocks_usable"], "M_by_class": by_class,
        "M1d_eligible": M1d.get("eligible"), "M1d_usable_fraction": M1d.get("usable_fraction"),
        "R2_separation": R2, "R2_n_events": R2.get("n"), "R2_n_blocks": R2.get("n_blocks"),
        "participation_full": diag,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)        # HARD STOP: no implicit cohort
    g.add_argument("--pilot", action="store_true", help="run the 4 locked PILOT_SUBJECTS")
    g.add_argument("--subjects", nargs="+", help="explicit ds_sid list (e.g. epilepsiae_1077)")
    ap.add_argument("--activation", choices=list(ACTIVATION_KEY), default="broadband")
    ap.add_argument("--n-perm-m1d", type=int, default=200)
    ap.add_argument("--n-perm-r2", type=int, default=1000)
    ap.add_argument("--max-per-class", type=int, default=1500,
                    help="cap events/class for the per-event field metric (subsample, fixed RNG)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    subjects = PILOT_SUBJECTS if args.pilot else args.subjects
    outdir = Path(args.out); (outdir / "per_subject").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    akey = ACTIVATION_KEY[args.activation]

    summary = []
    for ds_sid in subjects:
        print(f"[run] {ds_sid} ...", flush=True)
        res = _run_subject(ds_sid, activation_key=akey, rng=rng,
                           n_perm_m1d=args.n_perm_m1d, n_perm_r2=args.n_perm_r2,
                           max_per_class=args.max_per_class)
        json.dump(res, open(outdir / "per_subject" / f"{ds_sid}.json", "w"), indent=2)
        st = res.get("status")
        line = f"    status={st}"
        if st == "ok":
            line += (f" | M usable={res['M_usable_fraction']:.2f} analyzed={res['n_events_analyzed']}"
                     f" R2_blocks={res['R2_n_blocks']}"
                     f" | R2 Δmed={res['R2_separation'].get('delta_median_obs'):.3f}"
                     f" p={res['R2_separation'].get('delta_median_null_p')}")
        print(line, flush=True)
        summary.append({k: res.get(k) for k in
                        ("subject_id", "status", "n_channels_broad", "n_events_valid_full",
                         "n_events_analyzed", "n_blocks_full", "R2_n_blocks",
                         "cluster_map", "M_usable_fraction", "M_n_usable")})
    json.dump({"note": "PILOT — exploratory secondary; cohort verdict awaits advisor (user) sign-off.",
               "activation": args.activation, "subjects": subjects, "summary": summary},
              open(outdir / "pilot_summary.json", "w"), indent=2)
    print(f"[done] wrote {outdir/'pilot_summary.json'}")


if __name__ == "__main__":
    main()
