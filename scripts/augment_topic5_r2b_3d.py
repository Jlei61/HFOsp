"""Topic5 R2b native-3D sensitivity augment — R2b + R2_nm on a common coord subset.

Defensive sensitivity check for the contact-similarity ladder: does the
"same-plane geometry helps" readout survive when we swap the normalized 2D
contact-plane projection for native 3D Euclidean distance (mm)?

Per subject we rebuild the ladder context FROM SOURCE (matched channels + plane
pts + frozen sigma + per-seizure activation vectors) via the runner's `_ctx`,
then restrict to the COMMON channel subset that has BOTH a 2D plane entry AND a
valid mm 3D coord, and recompute two rungs on that identical subset:

  R2_nm : 2D-plane Gaussian-smoothed contact similarity, NO mirror,  sigma=sigma_xy.
  R2b   : native-3D (mm) Gaussian-smoothed contact similarity, NO mirror,
          sigma=median 3D nearest-neighbor spacing.

Both go through the SAME within-shaft per-seizure null harness (subject_null,
per-draw maxAB, median-over-seizures fold). Primary delta = R2b - R2_nm (both
no-mirror, common subset). Secondary DESCRIPTIVE delta = R2b - stored R2
(mirror, full) — a connector to the published R1/R2/R3 figure, NOT a verdict.

Units are a HARD gate (P1-3): coords must be mm; voxel / loader-raise / missing
coords -> subject r2b_status=NA_*, never a silent fallback. Never pools
cross-dataset point clouds (per-subject only).

Reuses (does not reimplement): run_topic5_contact_similarity._ctx (loaders),
src.topic5_contact_similarity.{contact_corr,subject_null,median_nn_spacing},
src.seeg_coord_loader.{load_subject_coords,assert_coord_result_is_mm_for_main_analysis},
src.propagation_skeleton_geometry.parse_shaft.
See docs/superpowers/plans/2026-07-01-topic5-r2b-3d-sensitivity.md Task 3.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered")

from scripts.run_topic5_contact_similarity import (
    _ctx, ACTIVATION_KEY, _negligible, _bootstrap_median_ci, SESOI,
)
from src.topic5_contact_similarity import contact_corr, subject_null, median_nn_spacing
from src.seeg_coord_loader import (
    load_subject_coords, assert_coord_result_is_mm_for_main_analysis,
)
from src.propagation_skeleton_geometry import parse_shaft

MIN_CH = 6
MIN_SHAFTS = 2
MIN_FINITE_PER_SZ = 6
RNG_SEED = 20260614
DEF_ROOT = "results"
DEF_OUT = "results/topic5_ictal_recruitment/contact_similarity"

# every non-"ok" value augment_subject() can return (module docstring, augment_subject).
NA_REASONS = ("NA_ineligible", "NA_coords", "NA_units", "NA_insufficient",
              "NA_degenerate", "NA_no_null")

COVERAGE_COLUMNS = ["subject_id", "n_matched_2d", "n_coord_mapped_3d", "n_common",
                    "n_shafts_common", "coord_space", "coord_units", "r2b_status",
                    "missing_channels"]


def _maxab_nomirror(rank_a, rank_b, source_pts, support, sigma):
    """Return stat(value) = maxAB over templates of abs(kernel contact_corr, NO mirror).

    Distinct from polarity_free_maxab: that helper's _abs_mirror maxes over the
    identity AND y-mirrored eval (a 2D-plane symmetry trick). Here mirror is held
    OFF for both R2_nm (2D) and R2b (3D) — 3D coords are absolute/anatomical, and
    the plan pins the primary comparison at mirror=False for both rungs.
    """
    ra = np.asarray(rank_a, float)
    rb = None if rank_b is None else np.asarray(rank_b, float)

    def _abs_t(rank, value):
        c = contact_corr(rank, value, mode="kernel", source_pts=source_pts,
                         support=support, sigma=sigma, mirror=False)
        return abs(c) if np.isfinite(c) else np.nan

    def stat(value):
        r_a = _abs_t(ra, value)
        if rb is None:
            return float(r_a) if np.isfinite(r_a) else np.nan
        r_b = _abs_t(rb, value)
        cand = [x for x in (r_a, r_b) if np.isfinite(x)]
        return float(max(cand)) if cand else np.nan

    return stat


def _stored_cross_check(ds_sid, activation, input_results_root):
    """Read stored cohort_summary_{activation}.json R1/R2/R3 within_shaft obs for this
    subject (provenance + the mirror/full R2 baseline for r2b_minus_r2main). No R3 recompute."""
    f = (Path(input_results_root) / "topic5_ictal_recruitment" / "contact_similarity"
         / f"cohort_summary_{activation}.json")
    if not f.exists():
        return {"present": False, "reason": "no_cohort_summary", "reference": str(f)}
    summary = json.load(open(f))
    by_id = {s["subject_id"]: s for s in summary.get("per_subject", [])}
    s = by_id.get(ds_sid)
    if s is None or s.get("status") != "ok":
        return {"present": False, "reason": "subject_not_in_summary", "reference": str(f)}

    def _obs(rung):
        return float(s[rung]["within_shaft"].get("obs_subject", np.nan))

    have_all = all(r in s and "within_shaft" in s[r] for r in ("R1", "R2", "R3"))
    return {"present": bool(have_all), "reference": str(f),
            "r1_obs": _obs("R1") if "R1" in s else None,
            "r2_obs": _obs("R2") if "R2" in s else None,   # mirror, full-channel
            "r3_obs": _obs("R3") if "R3" in s else None}


def augment_subject(ds_sid, *, activation="broadband", B=1000, seed=RNG_SEED,
                    input_results_root=DEF_ROOT):
    """R2b (native-3D) + R2_nm (2D no-mirror) on the common coord-mapped subset.

    Returns a per-subject dict (see module docstring / plan Task 3 contract).
    r2b_status is one of: ok / NA_ineligible / NA_coords / NA_units /
    NA_insufficient / NA_degenerate / NA_no_null.
    """
    out = {"subject_id": ds_sid, "dataset": ds_sid.split("_", 1)[0],
           "activation": activation, "B": int(B), "seed": int(seed)}

    # Clause 1: rebuild context FROM SOURCE (matched channels, plane pts, frozen
    # sigma, per-seizure activation vectors) — NOT from stored per_subject JSON.
    ctx = _ctx(ds_sid, activation, input_results_root)
    if ctx is None:
        return {**out, "r2b_status": "NA_ineligible"}
    names_m = list(ctx["names_m"])
    out["n_matched_2d"] = len(names_m)

    # Clause 2: mm units HARD GATE (P1-3). load raise -> NA_coords; assert raise -> NA_units.
    ds, subj = ds_sid.split("_", 1)
    try:
        cr = load_subject_coords(ds, subj, names_m, allow_voxel_fallback=False)
    except (FileNotFoundError, ValueError) as e:
        return {**out, "r2b_status": "NA_coords", "reason": str(e)}
    out["coord_space"] = cr.coord_space
    out["coord_units"] = cr.coord_units
    try:
        assert_coord_result_is_mm_for_main_analysis(cr)
    except ValueError as e:
        return {**out, "r2b_status": "NA_units", "reason": str(e)}

    # Clause 7: channel_names ordering — coords must be aligned to names_m before indexing.
    if list(cr.channel_names_requested) != names_m:
        return {**out, "r2b_status": "NA_coords",
                "reason": "coord channel order != matched channel order"}

    # Clause 3: COMMON subset = matched channels with a finite mm 3D coord.
    coords_all = cr.coords_array_in_requested_order
    coord_mapped_mask = (cr.mapped_mask_in_requested_order
                         & np.isfinite(coords_all).all(axis=1))
    common_idx = np.where(coord_mapped_mask)[0]
    common_names = [names_m[i] for i in common_idx]
    out["n_coord_mapped_3d"] = int(coord_mapped_mask.sum())
    out["n_common"] = int(common_idx.size)
    out["missing_channels"] = [names_m[i] for i in range(len(names_m))
                               if not coord_mapped_mask[i]]
    n_shafts_common = len({parse_shaft(n)[0] for n in common_names})
    out["n_shafts_common"] = int(n_shafts_common)

    # per-seizure finite-on-common gate: keep seizures with >=6 finite on the common subset.
    sz_common = {}
    for idx, v in ctx["sz_vals"].items():
        vc = np.asarray(v, float)[common_idx]
        if int(np.isfinite(vc).sum()) >= MIN_FINITE_PER_SZ:
            sz_common[idx] = vc
    out["n_seizures_common"] = len(sz_common)

    if out["n_common"] < MIN_CH or n_shafts_common < MIN_SHAFTS or not sz_common:
        return {**out, "r2b_status": "NA_insufficient"}

    # Clause 4: build R2_nm (2D-plane) + R2b (native-3D) on the SAME common subset.
    src2d = np.asarray(ctx["source_pts"], float)[common_idx]      # (n_common, 2)
    support_c = np.asarray(ctx["support"], float)[common_idx]
    coords3d_c = coords_all[common_idx]                            # (n_common, 3) mm
    sigma_xy = float(ctx["sigma"])
    sigma_3d = median_nn_spacing(coords3d_c)
    out["sigma_xy"] = sigma_xy
    out["sigma_3d"] = float(sigma_3d)
    if not (sigma_3d > 0):
        return {**out, "r2b_status": "NA_degenerate"}

    rank_a = np.asarray(ctx["rank_a"], float)[common_idx]
    rank_b = (np.asarray(ctx["rank_b"], float)[common_idx]
              if ctx["rank_b"] is not None else None)

    r2nm_stat = _maxab_nomirror(rank_a, rank_b, src2d, support_c, sigma_xy)
    r2b_stat = _maxab_nomirror(rank_a, rank_b, coords3d_c, support_c, sigma_3d)

    # SAME null harness for both: within-shaft, per-draw maxAB, median-over-seizures fold.
    r2nm = subject_null(r2nm_stat, sz_common, common_names,
                        shuffle="within_shaft", B=B, seed=seed)
    r2b = subject_null(r2b_stat, sz_common, common_names,
                       shuffle="within_shaft", B=B, seed=seed)
    out["R2_nm"] = r2nm
    out["R2b"] = r2b
    if "obs_subject" not in r2nm or "obs_subject" not in r2b:
        return {**out, "r2b_status": "NA_no_null"}

    # Clause 5: deltas. Primary no-mirror common-subset delta; secondary descriptive connector.
    out["r2b_minus_r2nm"] = float(r2b["obs_subject"] - r2nm["obs_subject"])

    # Clause 6: cross-check stored R1/R2/R3 (record r3 obs for provenance; NO R3 recompute).
    xc = _stored_cross_check(ds_sid, activation, input_results_root)
    out["stored_cross_check"] = xc
    out["r3_obs_stored"] = xc.get("r3_obs")
    r2main = xc.get("r2_obs")
    out["r2b_minus_r2main"] = (float(r2b["obs_subject"] - r2main)
                               if r2main is not None and np.isfinite(r2main) else None)

    out["r2b_status"] = "ok"
    return out


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _write_coverage_csv(path, results):
    """r2b_coverage_{activation}.csv — one row per subject, exact COVERAGE_COLUMNS."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COVERAGE_COLUMNS)
        for r in results:
            w.writerow([
                r["subject_id"], r.get("n_matched_2d", ""), r.get("n_coord_mapped_3d", ""),
                r.get("n_common", ""), r.get("n_shafts_common", ""), r.get("coord_space", ""),
                r.get("coord_units", ""), r["r2b_status"],
                ";".join(r.get("missing_channels") or []),
            ])


def _null_insufficient(res):
    """True if EITHER recomputed rung's within-shaft null was underpowered (few
    effective shuffles). M-1: r2b_status=='ok' only guarantees obs_subject is
    finite for both rungs -- it does NOT mean the null threshold is trustworthy,
    so this must be checked and reported separately, not inferred from status."""
    return any(res.get(rung, {}).get("status") == "INSUFFICIENT_NULL"
               for rung in ("R2_nm", "R2b"))


def _build_summary(results, *, activation, B, seed):
    """r2b_summary_{activation}.json payload: per-subject trimmed fields + cohort
    r2b_minus_r2nm median/CI/SESOI verdict, computed on r2b_status=='ok' subjects
    only (M-1), plus the INSUFFICIENT_NULL and NA-reason breakdowns."""
    n_na_by_reason = {reason: 0 for reason in NA_REASONS}
    for r in results:
        if r["r2b_status"] != "ok":
            n_na_by_reason[r["r2b_status"]] = n_na_by_reason.get(r["r2b_status"], 0) + 1

    ok = [r for r in results if r["r2b_status"] == "ok"]
    n_ok_insufficient_null = sum(1 for r in ok if _null_insufficient(r))

    per_subject = [{
        "subject_id": r["subject_id"],
        "R2_nm": r.get("R2_nm"),
        "R2b": r.get("R2b"),
        "r2b_minus_r2nm": r.get("r2b_minus_r2nm"),
        "r2b_minus_r2main": r.get("r2b_minus_r2main"),
        "r2b_status": r["r2b_status"],
        # not part of the Task 4 minimal schema, but needed by the R1/R2_nm/R2b
        # ladder figure (Panel B) without re-reading per_subject_r2b/*.json.
        "r1_obs_stored": (r.get("stored_cross_check") or {}).get("r1_obs"),
    } for r in results]

    summary = {
        "activation": activation, "B": int(B), "seed": int(seed),
        "n_subjects": len(results), "n_ok": len(ok),
        "n_na_by_reason": n_na_by_reason,
        "n_ok_insufficient_null": n_ok_insufficient_null,
    }

    deltas = [r.get("r2b_minus_r2nm") for r in ok]
    deltas = [d for d in deltas if d is not None and np.isfinite(d)]
    if deltas:
        lo, hi = _bootstrap_median_ci(deltas, seed)
        summary["r2b_minus_r2nm_median"] = float(np.median(deltas))
        summary["r2b_minus_r2nm_ci"] = [lo, hi]
        summary["r2b_minus_r2nm_negligible"] = _negligible(lo, hi, SESOI)
    else:
        summary["r2b_minus_r2nm_median"] = None
        summary["r2b_minus_r2nm_ci"] = None
        summary["r2b_minus_r2nm_negligible"] = None

    summary["per_subject"] = per_subject
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation", choices=list(ACTIVATION_KEY), default="broadband")
    ap.add_argument("--B", type=int, default=1000, help="null draws per seizure (smoke: 20-50)")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--input-results-root", default=DEF_ROOT,
                    help="root holding the T0 cache + axis records + stored cohort summaries")
    ap.add_argument("--out-dir", default=DEF_OUT)
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    root = Path(args.input_results_root)
    cache_dir = root / "topic5_ictal_recruitment" / "t0_feature_cache"
    out_dir = Path(args.out_dir)
    (out_dir / "per_subject_r2b").mkdir(parents=True, exist_ok=True)

    cohort = sorted(p.stem for p in cache_dir.glob("*.npz"))
    if args.subjects:
        cohort = [s for s in cohort if s in set(args.subjects)]
    print(f"[r2b-3d] activation={args.activation} B={args.B} seed={args.seed} | "
          f"{len(cohort)} cached subjects", flush=True)

    results = []
    for ds_sid in cohort:
        res = augment_subject(ds_sid, activation=args.activation, B=args.B,
                              seed=args.seed, input_results_root=args.input_results_root)
        results.append(res)
        json.dump(_to_jsonable(res),
                  open(out_dir / "per_subject_r2b" / f"{ds_sid}.json", "w"),
                  indent=2, ensure_ascii=False)
        st = res["r2b_status"]
        if st == "ok":
            print(f"  {ds_sid}: ok | R2_nm={res['R2_nm']['obs_subject']:.3f} "
                  f"R2b={res['R2b']['obs_subject']:.3f} "
                  f"r2b-r2nm={res['r2b_minus_r2nm']:+.3f} "
                  f"n_common={res['n_common']}/{res['n_matched_2d']} "
                  f"shafts={res['n_shafts_common']}", flush=True)
        else:
            print(f"  {ds_sid}: {st} "
                  f"(n_common={res.get('n_common')}, "
                  f"n_matched={res.get('n_matched_2d')})", flush=True)

    _write_coverage_csv(out_dir / f"r2b_coverage_{args.activation}.csv", results)
    summary = _build_summary(results, activation=args.activation, B=args.B, seed=args.seed)
    json.dump(_to_jsonable(summary),
              open(out_dir / f"r2b_summary_{args.activation}.json", "w"),
              indent=2, ensure_ascii=False)

    print(f"\n[r2b-3d] n_ok={summary['n_ok']}/{len(results)} "
          f"| n_ok_insufficient_null={summary['n_ok_insufficient_null']}", flush=True)
    print(f"  NA breakdown: {summary['n_na_by_reason']}", flush=True)
    if summary["r2b_minus_r2nm_median"] is not None:
        lo, hi = summary["r2b_minus_r2nm_ci"]
        print(f"  r2b_minus_r2nm median={summary['r2b_minus_r2nm_median']:+.4f} "
              f"CI=[{lo:+.4f},{hi:+.4f}] negligible(|.|<{SESOI})="
              f"{summary['r2b_minus_r2nm_negligible']}", flush=True)
    print(f"wrote {out_dir}/r2b_coverage_{args.activation}.csv, "
          f"r2b_summary_{args.activation}.json", flush=True)


if __name__ == "__main__":
    main()
