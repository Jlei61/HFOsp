#!/usr/bin/env python3
"""Topic 5 — TA/TB field-reversal §6a Option-B axis-level robustness: cohort runner.

Spec: docs/superpowers/specs/2026-07-06-topic5-tatb-field-reversal-design.md §6a
Pilot (n=5, broad only): scripts/pilot_topic5_axis_robustness.py + .superpowers/sdd/pilot_axis_report.md

Reframed claim (Option-B, NOT "field denoises"): reading propagation direction by
electrode/shaft order (coordinate-blind, `sequence_axis`) can badly mislead on some
subjects; using real coordinates (`raw_contact_axis`) avoids it; smoothing (`field_axis`)
adds nothing beyond plain coordinate LS.

Per subject x substrate: load the A/B interictal labels + per-event ranks (C1-guarded),
map cluster_id -> t_a/t_b (C2), pick the t_a plane as the shared reference frame (P0 §3.1;
same convention as run_topic5_field_reversal.py). Per class (TA, TB): the three axes (full
class, no split) + held-out split scores (a few dozen splits) for all three axis types, then
fold TA/TB to one per-subject number BY MEAN. broad and narrow are run and reported
SEPARATELY, never pooled (spec §8). Every requested subject gets a per-subject JSON with an
explicit `reason` (no_planes / load_error / c1_violation / plane_not_built /
cluster_map_ambiguous / ok) -- never a silent skip or a crash.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.topic5_event_resolved_alignment import (
    load_event_labels_ranks, map_clusters_to_templates, class_aggregate_contact_values,
    class_template_sigma, field_from_contact_values, build_plane_xy)
from src.propagation_contact_plane_readout import make_plane_grid, S_THRESH
from src.topic5_axis_robustness import (
    raw_contact_axis, field_axis_from_field, sequence_axis, axis_angle, cos_unit,
    axis_robustness_splits)

OUT = Path("results/topic5_ictal_recruitment/field_reversal/axis_robustness")
RNG_SEED = 20260706
REASONS = ("no_planes", "load_error", "c1_violation", "plane_not_built",
           "cluster_map_ambiguous", "ok")
DIVERGENCE_LARGE_DEG = 45.0
DIVERGENCE_HUGE_DEG = 90.0


def _vec_in_order(name_to_val, order):
    return np.array([name_to_val.get(n, np.nan) for n in order], float)


def _plane_usable(plane: dict) -> bool:
    """False for status-only records (e.g. status="descriptive_only") that have no
    "channels" key -- the geometry plane was never built for this subject/template."""
    return isinstance(plane, dict) and "channels" in plane


def pick_reference(cmap, plane_a, plane_b):
    """P0 reference frame = the plane mapped to t_a (same convention as run_topic5_field_reversal.py)."""
    inv = {v: k for k, v in cmap["map"].items()}          # {"t_a":label, "t_b":label}
    plane_of = {"t_a": plane_a, "t_b": plane_b}
    return plane_of["t_a"], inv["t_a"], inv["t_b"]


def _geom(input_results_root: Path) -> dict:
    return {
        "broad":  input_results_root / "spatial_modulation" / "propagation_geometry_broad"
                  / "observation_readout" / "real_subjects",
        "narrow": input_results_root / "spatial_modulation" / "propagation_geometry"
                  / "observation_readout" / "real_subjects",
    }


def _class_axes(bundle, plane_ref, label, *, X, Y, sigma, n_split, rng, s_thresh=S_THRESH) -> dict:
    """Full-class (no-split) three axes + divergence/agreement scalars, plus the held-out
    split scores (a few dozen splits), for ONE class label (TA or TB)."""
    cav = class_aggregate_contact_values(bundle, label)
    plane_xy = build_plane_xy(plane_ref)
    rc = raw_contact_axis(cav, plane_xy)
    seq = sequence_axis(cav, plane_xy)
    v = {n: d["value"] for n, d in cav.items()}
    s = {n: d["support"] for n, d in cav.items()}
    field = field_from_contact_values(plane_ref, v, support_by_name=s, sigma=sigma, X=X, Y=Y,
                                      s_thresh=s_thresh)
    fa = field_axis_from_field(field, X, Y, s_thresh)

    splits = axis_robustness_splits(bundle, plane_ref, label, X=X, Y=Y, sigma=sigma,
                                    n_split=n_split, rng=rng, s_thresh=s_thresh)
    # Each axis type is scored independently per split (axis_robustness_splits: one axis
    # failing to resolve, e.g. sequence_axis on a single-shaft montage, does not discard the
    # other two) -- so each column's own finite entries are its own usable-split set.
    raw_rhos = [d["raw_rho"] for d in splits if np.isfinite(d["raw_rho"])]
    field_rhos = [d["field_rho"] for d in splits if np.isfinite(d["field_rho"])]
    seq_rhos = [d["sequence_rho"] for d in splits if np.isfinite(d["sequence_rho"])]

    return {
        "n_events": int(np.sum(np.asarray(bundle["labels"]) == label)),
        "raw_contact_unit": rc["unit"], "field_unit": fa["unit"], "sequence_unit": seq["unit"],
        "raw_contact_ok": bool(rc["ok"]), "field_ok": bool(fa["ok"]), "sequence_ok": bool(seq["ok"]),
        "angle_sequence_raw": axis_angle(seq["unit"], rc["unit"]),
        "cos_raw_field": cos_unit(rc["unit"], fa["unit"]),
        "held_out_n_splits_attempted": len(splits),
        "held_out_n_splits_raw": len(raw_rhos),
        "held_out_n_splits_field": len(field_rhos),
        "held_out_n_splits_sequence": len(seq_rhos),
        "held_out_raw_median": float(np.median(raw_rhos)) if raw_rhos else float("nan"),
        "held_out_field_median": float(np.median(field_rhos)) if field_rhos else float("nan"),
        "held_out_sequence_median": float(np.median(seq_rhos)) if seq_rhos else float("nan"),
    }


def _fold_mean(a, b):
    """Fold TA's and TB's class-level value to ONE per-subject number BY MEAN (spec §6a
    Option-B). Skips whichever side is non-finite rather than propagating NaN -- if only one
    class produced a usable value, that value stands alone; NaN only if neither did."""
    vals = [v for v in (a, b) if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _run_subject(ds_sid, substrate, *, geom_dir, X, Y, rng, n_split):
    dataset, subject = ds_sid.split("_", 1)
    out = {"ds_sid": ds_sid, "dataset": dataset, "subject": subject, "substrate": substrate}

    ta_f = geom_dir / f"{ds_sid}_t_a.json"
    tb_f = geom_dir / f"{ds_sid}_t_b.json"
    if not (ta_f.exists() and tb_f.exists()):
        out.update(reason="no_planes")
        return out

    try:
        bundle = load_event_labels_ranks(dataset, subject, broad=(substrate == "broad"))
    except FileNotFoundError as e:
        out.update(reason="load_error", detail=str(e))
        return out
    except ValueError as e:
        out.update(reason="c1_violation", detail=str(e))
        return out

    plane_a = json.load(open(ta_f)); plane_b = json.load(open(tb_f))
    if not (_plane_usable(plane_a) and _plane_usable(plane_b)):
        out.update(reason="plane_not_built",
                   plane_a_status=plane_a.get("status"), plane_b_status=plane_b.get("status"))
        return out

    order = bundle["channel_names"]
    out["n_channels"] = len(order)
    ta_rank = _vec_in_order({c["name"]: c["typical_rank"] for c in plane_a["channels"]}, order)
    tb_rank = _vec_in_order({c["name"]: c["typical_rank"] for c in plane_b["channels"]}, order)
    c0 = np.asarray(bundle["cluster_template_ranks"][0], float)
    c1 = np.asarray(bundle["cluster_template_ranks"][1], float)
    cmap = map_clusters_to_templates(c0, c1, ta_rank, tb_rank)
    if cmap["ambiguous"]:
        out.update(reason="cluster_map_ambiguous", diag_minus_offdiag=cmap["diag_minus_offdiag"])
        return out
    out["cluster_map"] = cmap["map"]

    # P0: reference frame = t_a's plane; ONE sigma derived from it, reused for TA, TB, AND
    # every held-out split of both classes (never re-derived per class or per split).
    plane_ref, ta_label, tb_label = pick_reference(cmap, plane_a, plane_b)
    sigma = class_template_sigma(plane_ref, X=X, Y=Y)

    per_class = {}
    for label, cls_name in [(ta_label, "TA"), (tb_label, "TB")]:
        per_class[cls_name] = _class_axes(bundle, plane_ref, label, X=X, Y=Y, sigma=sigma,
                                          n_split=n_split, rng=rng)
    out["per_class"] = per_class
    out["base_sigma"] = float(sigma)

    ta, tb = per_class["TA"], per_class["TB"]
    # TA/TB axis anti-parallel (spec §6a secondary readout): cos(raw_contact_TA, -raw_contact_TB).
    tb_neg_unit = tuple(-c for c in tb["raw_contact_unit"])
    out["subject_summary"] = {
        "angle_sequence_raw_mean": _fold_mean(ta["angle_sequence_raw"], tb["angle_sequence_raw"]),
        "cos_raw_field_mean": _fold_mean(ta["cos_raw_field"], tb["cos_raw_field"]),
        "held_out_raw_mean": _fold_mean(ta["held_out_raw_median"], tb["held_out_raw_median"]),
        "held_out_field_mean": _fold_mean(ta["held_out_field_median"], tb["held_out_field_median"]),
        "held_out_sequence_mean": _fold_mean(ta["held_out_sequence_median"], tb["held_out_sequence_median"]),
        "cos_ta_tb_axis_anti": cos_unit(ta["raw_contact_unit"], tb_neg_unit),
    }
    out["reason"] = "ok"
    return out


# --------------------------------------------------------------------------- cohort stats
def _paired_wilcoxon(pairs, alternative="greater"):
    """One-sided (default) paired Wilcoxon signed-rank test over (a, b) tuples; only finite
    pairs count. n_wins_a = how many subjects have a > b (paired direction check)."""
    pairs = [(a, b) for a, b in pairs if np.isfinite(a) and np.isfinite(b)]
    n = len(pairs)
    if n < 1:
        return {"n": 0, "statistic": None, "p_value": None, "n_wins_a": None}
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    n_wins_a = int(np.sum(a > b))
    if np.allclose(a, b):
        return {"n": n, "statistic": None, "p_value": None, "n_wins_a": n_wins_a,
                "note": "all paired differences ~0"}
    try:
        stat, p = wilcoxon(a, b, alternative=alternative)
        return {"n": n, "statistic": float(stat), "p_value": float(p), "n_wins_a": n_wins_a}
    except ValueError as e:
        return {"n": n, "statistic": None, "p_value": None, "n_wins_a": n_wins_a, "note": str(e)}


def _divergence_stats(ok_records):
    """Cohort distribution of angle(sequence_axis, raw_contact_axis) (per-subject, mean-folded
    over TA/TB) -- median/IQR + count/fraction of subjects diverging >45 deg and >90 deg."""
    vals = [r["subject_summary"]["angle_sequence_raw_mean"] for r in ok_records]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return {"n": 0}
    arr = np.asarray(vals, float)
    return {
        "n": len(vals),
        "median": float(np.median(arr)),
        "iqr": [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))],
        "n_gt45": int(np.sum(arr > DIVERGENCE_LARGE_DEG)),
        "frac_gt45": float(np.mean(arr > DIVERGENCE_LARGE_DEG)),
        "n_gt90": int(np.sum(arr > DIVERGENCE_HUGE_DEG)),
        "frac_gt90": float(np.mean(arr > DIVERGENCE_HUGE_DEG)),
        "values_by_subject": {r["ds_sid"]: float(r["subject_summary"]["angle_sequence_raw_mean"])
                              for r in ok_records
                              if np.isfinite(r["subject_summary"]["angle_sequence_raw_mean"])},
    }


def _aggregate_cohort(records):
    accountability = {r: 0 for r in REASONS}
    for rec in records:
        accountability[rec["reason"]] = accountability.get(rec["reason"], 0) + 1
    ok = [r for r in records if r["reason"] == "ok"]

    divergence = _divergence_stats(ok)

    pairs_raw_seq = [(r["subject_summary"]["held_out_raw_mean"],
                      r["subject_summary"]["held_out_sequence_mean"]) for r in ok]
    raw_beats_seq_all = _paired_wilcoxon(pairs_raw_seq, alternative="greater")

    large_div = [r for r in ok if np.isfinite(r["subject_summary"]["angle_sequence_raw_mean"])
                and r["subject_summary"]["angle_sequence_raw_mean"] > DIVERGENCE_LARGE_DEG]
    pairs_raw_seq_large = [(r["subject_summary"]["held_out_raw_mean"],
                            r["subject_summary"]["held_out_sequence_mean"]) for r in large_div]
    raw_beats_seq_large = _paired_wilcoxon(pairs_raw_seq_large, alternative="greater")
    raw_beats_seq_large["n_subjects_large_divergence"] = len(large_div)

    pairs_field_raw = [(r["subject_summary"]["held_out_field_mean"],
                        r["subject_summary"]["held_out_raw_mean"]) for r in ok]
    field_vs_raw = _paired_wilcoxon(pairs_field_raw, alternative="two-sided")
    finite_fr = [(f, r) for f, r in pairs_field_raw if np.isfinite(f) and np.isfinite(r)]
    field_vs_raw["n_field_wins"] = sum(1 for f, r in finite_fr if f > r)
    field_vs_raw["n_paired"] = len(finite_fr)

    cos_vals = [r["subject_summary"]["cos_raw_field_mean"] for r in ok]
    cos_vals = [v for v in cos_vals if np.isfinite(v)]
    anti_vals = [r["subject_summary"]["cos_ta_tb_axis_anti"] for r in ok]
    anti_vals = [v for v in anti_vals if np.isfinite(v)]

    return {
        "n_subjects": len(records), "n_ok": len(ok), "accountability": accountability,
        "divergence_sequence_vs_raw_contact_deg": divergence,
        "raw_contact_beats_sequence_wilcoxon_all_ok": raw_beats_seq_all,
        "raw_contact_beats_sequence_wilcoxon_large_divergence_subset": raw_beats_seq_large,
        "field_vs_raw_contact_near_null_wilcoxon": field_vs_raw,
        "cos_raw_field_median": (float(np.median(cos_vals)) if cos_vals else float("nan")),
        "cos_ta_tb_axis_antiparallel_median": (float(np.median(anti_vals)) if anti_vals else float("nan")),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)      # HARD STOP: no implicit cohort run
    g.add_argument("--subjects", nargs="+", help="explicit ds_sid list (e.g. epilepsiae_1077)")
    g.add_argument("--cohort", action="store_true",
                   help="discover all subjects per substrate (glob GEOM[substrate]/*_t_a.json)")
    ap.add_argument("--substrate", choices=["broad", "narrow", "both"], default="both")
    ap.add_argument("--input-results-root", default="/home/honglab/leijiaxin/HFOsp/results",
                    help="root containing spatial_modulation/propagation_geometry{,_broad} "
                         "(labels+geometry live in the main tree, gitignored, not the worktree)")
    ap.add_argument("--n-split", type=int, default=30, help="held-out splits per class (a few dozen)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    GEOM = _geom(Path(args.input_results_root))
    substrates = ["broad", "narrow"] if args.substrate == "both" else [args.substrate]
    outdir = Path(args.out)
    X, Y = make_plane_grid()
    rng = np.random.default_rng(RNG_SEED)

    cohort_by_substrate = {}
    for substrate in substrates:
        geom_dir = GEOM[substrate]
        (outdir / "per_subject" / substrate).mkdir(parents=True, exist_ok=True)
        if args.cohort:
            subjects = sorted(p.name[:-len("_t_a.json")] for p in geom_dir.glob("*_t_a.json"))
        else:
            subjects = list(args.subjects)

        records = []
        for ds_sid in subjects:
            print(f"[run] {substrate}/{ds_sid} ...", flush=True)
            res = _run_subject(ds_sid, substrate, geom_dir=geom_dir, X=X, Y=Y, rng=rng,
                               n_split=args.n_split)
            json.dump(res, open(outdir / "per_subject" / substrate / f"{ds_sid}.json", "w"), indent=2)
            print(f"    reason={res['reason']}", flush=True)
            records.append(res)
        cohort_by_substrate[substrate] = _aggregate_cohort(records)

    cohort_summary = {
        "note": "broad and narrow reported separately; never pooled (spec §8, §6a).",
        "n_split": args.n_split,
        "divergence_thresholds_deg": {"large": DIVERGENCE_LARGE_DEG, "huge": DIVERGENCE_HUGE_DEG},
        **cohort_by_substrate,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump(cohort_summary, open(outdir / "cohort_summary.json", "w"), indent=2)
    print(f"[done] wrote {outdir/'cohort_summary.json'}")


if __name__ == "__main__":
    main()
