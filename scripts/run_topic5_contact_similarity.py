"""Topic5 contact-similarity ladder runner — three rungs through one null harness.

Question: when we say the ictal activation 'recruits along the interictal axis', does
that claim NEED the smoothed 2D field, or is it already there in the raw per-contact
numbers? We climb a 3-rung ladder on the SAME matched contacts / same plane / same frozen
sigma / same per-seizure activation vectors, changing ONLY how similarity is measured:

  R1 (raw)         : Pearson over contacts, no geometry at all.
  R2 (same-plane)  : same as R1 but both vectors Gaussian-smoothed AT the contacts
                     (no grid, no pixel reweighting) — isolates 'same-plane smoothing'.
  R3 (field)       : the published A-line statistic — smooth onto an 81x81 grid and take
                     |mirror-invariant field corr| — isolates the 'grid' contribution.

Each rung is polarity-free maxAB (max over the two interictal templates t_a / t_b), folded
per-subject as median-over-seizures, with the SAME per-seizure paired null harness as the
A-line (within_shaft / channel / anchor_matched). Paired subject-level deltas read out where
the signal lives: smooth_delta = R2 - R1 (same-plane smoothing), grid_delta = R3 - R2 (grid).
A sigma sweep (x0.5 / x1 / x2) on R2 and a geometry-free sequence track (Spearman / Kendall)
are sanity side-channels. R3 is byte-faithful to run_topic5_axis_alignment.py's max_ab
statistic, so the cohort run cross-checks R3.within_shaft.obs_subject against the published
axis_alignment_<act>_max_ab_B1000.json real_median_abs_corr (atol=0.03).

Reuses: src.topic5_contact_similarity (R1/R2 + null fold), src.topic5_axis_alignment (join +
shuffles), src.propagation_contact_plane_readout (plane / field smooth / mirror-invariant corr).
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

from src.topic5_contact_similarity import polarity_free_maxab, sequence_maxab, subject_null
from src.topic5_axis_alignment import make_field_record, matched_channels, channel_shuffle
from src.propagation_contact_plane_readout import (
    make_plane_grid, R_smooth_rank, corr_pair_mirror_invariant, S_THRESH, OVERLAP_MIN,
)
from src.propagation_skeleton_geometry import parse_shaft

# activation name -> T0 cache key prefix (mirrors run_topic5_axis_alignment.ACTIVATION_KEY)
ACTIVATION_KEY = {"broadband": "bb_auc", "hfa": "hfa_auc", "ramp": "ramp", "ei": "ei_like"}

SESOI = 0.05                 # smallest grid_delta we'd care about (equivalence band)
SIGMA_SWEEP = (0.5, 1.0, 2.0)
RNG_SEED = 20260614
DEF_ROOT = "results"
DEF_OUT = "results/topic5_ictal_recruitment/contact_similarity"


def _abs_corr(Fi, Fj):
    """Local copy of run_topic5_axis_alignment._abs_corr (script-level there, NOT importable).
    R3 must be byte-faithful to the A-line, so this replicates it exactly."""
    r = corr_pair_mirror_invariant(Fi["T"], Fi["S"], Fj["T"], Fj["S"], S_THRESH, OVERLAP_MIN)["corr"]
    return abs(r) if r is not None and np.isfinite(r) else np.nan


def _ctx(ds_sid, activation, input_results_root=DEF_ROOT):
    """Per-subject context shared by ALL rungs (matched channels, plane pts/support, frozen
    sigma, interictal t_a/t_b fields, per-seizure activation + baseline-activity anchor).
    Loaders mirror run_topic5_axis_alignment._subject exactly. Returns None if ineligible."""
    root = Path(input_results_root)
    cache_dir = root / "topic5_ictal_recruitment" / "t0_feature_cache"
    axis_dir = root / "spatial_modulation" / "propagation_geometry" / "observation_readout" / "real_subjects"
    act_key = ACTIVATION_KEY[activation]

    axis_f = axis_dir / f"{ds_sid}_t_a.json"
    npz_f = cache_dir / f"{ds_sid}.npz"
    if not axis_f.exists() or not npz_f.exists():
        return None
    axis = json.load(open(axis_f))
    if not axis.get("channels"):
        return None
    data = np.load(npz_f, allow_pickle=True)
    meta = json.load(open(cache_dir / f"{ds_sid}.json"))
    cache_names = [str(x) for x in data["channels"]]
    cidx = {n: i for i, n in enumerate(cache_names)}

    matched = matched_channels(axis, {n: 0.0 for n in cache_names})
    if len(matched) < 6:
        return None
    names_m = [c["name"] for c in matched]
    m_in_cache = np.array([cidx[n] for n in names_m])
    has_anchor = any(k.startswith("bact__") for k in data.files)

    X, Y = make_plane_grid()
    rank_a = [float(c["typical_rank"]) for c in matched]
    F_inter_a = R_smooth_rank(make_field_record(matched, rank_a), X, Y, None, S_THRESH)
    sigma = float(F_inter_a["sigma_xy"])          # frozen on t_a; reused for t_b + every draw

    # t_b interictal field on the SAME matched contacts (t_b ranks joined BY NAME)
    rank_b, F_inter_b = None, None
    axis_f_b = axis_dir / f"{ds_sid}_t_b.json"
    if axis_f_b.exists():
        axis_b = json.load(open(axis_f_b))
        if axis_b.get("channels"):
            matched_b = matched_channels(axis_b, {n: 0.0 for n in cache_names})
            b_rank = {c["name"]: float(c["typical_rank"]) for c in matched_b}
            inter_b = [b_rank.get(n, np.nan) for n in names_m]
            if int(np.isfinite(np.asarray(inter_b, float)).sum()) >= 4:
                rank_b = inter_b
                F_inter_b = R_smooth_rank(make_field_record(matched, inter_b), X, Y, sigma, S_THRESH)

    source_pts = np.array([[c["x_norm"], c["y_norm"]] for c in matched], float)  # (n,2), NOT column_stack
    support = np.array([c["support"] for c in matched], float)

    sz_vals, anchor = {}, {}
    for idx in meta["eligible_idxs"]:
        key = f"{act_key}__{idx}"
        if key not in data.files:
            continue
        vals = data[key][m_in_cache].astype(float)
        if int(np.isfinite(vals).sum()) < 6:
            continue
        sz_vals[idx] = vals
        if has_anchor and f"bact__{idx}" in data.files:
            anchor[idx] = data[f"bact__{idx}"][m_in_cache].astype(float)
    if not sz_vals:
        return None

    n_shafts = len({parse_shaft(n)[0] for n in names_m})
    return {
        "matched": matched, "names_m": names_m, "rank_a": rank_a, "rank_b": rank_b,
        "X": X, "Y": Y, "F_inter_a": F_inter_a, "F_inter_b": F_inter_b,
        "sigma": sigma, "source_pts": source_pts, "support": support,
        "sz_vals": sz_vals, "anchor": (anchor if anchor else None), "n_shafts": n_shafts,
        "has_tb": rank_b is not None,
    }


def _stats(ctx):
    """The three per-seizure statistic closures over the shared context."""
    ra, rb = ctx["rank_a"], ctx["rank_b"]
    sp, su, sg = ctx["source_pts"], ctx["support"], ctx["sigma"]

    def R1(v):
        return polarity_free_maxab(ra, rb, v, mode="raw", source_pts=sp, support=su, sigma=sg)

    def R2(v):
        return polarity_free_maxab(ra, rb, v, mode="kernel", source_pts=sp, support=su, sigma=sg)

    def R3(v):
        F = lambda vals: R_smooth_rank(make_field_record(ctx["matched"], vals),
                                       ctx["X"], ctx["Y"], sg, S_THRESH)
        r_a = _abs_corr(ctx["F_inter_a"], F(v))
        if ctx["F_inter_b"] is None:
            return float(r_a) if np.isfinite(r_a) else np.nan
        r_b = _abs_corr(ctx["F_inter_b"], F(v))
        vals = [x for x in (r_a, r_b) if np.isfinite(x)]
        return float(max(vals)) if vals else np.nan

    return {"R1": R1, "R2": R2, "R3": R3}


def run_subject(ds_sid, *, activation="broadband", B=1000, seed=RNG_SEED,
                negative_control=False, input_results_root=DEF_ROOT):
    ctx = _ctx(ds_sid, activation, input_results_root)
    if ctx is None:
        return {"subject_id": ds_sid, "status": "ineligible"}
    if ctx["n_shafts"] < 2:
        return {"subject_id": ds_sid, "status": "single_shaft"}
    if negative_control:   # bad-data gate: spatially scramble each seizure's activation once
        rng = np.random.default_rng(seed)
        ctx["sz_vals"] = {i: channel_shuffle(v, rng) for i, v in ctx["sz_vals"].items()}
    stats = _stats(ctx)
    out = {"subject_id": ds_sid, "dataset": ds_sid.split("_", 1)[0], "status": "ok",
           "activation": activation, "B": B, "seed": seed, "sigma_xy": ctx["sigma"],
           "n_matched_channels": len(ctx["matched"]), "n_seizures": len(ctx["sz_vals"]),
           "n_shafts": ctx["n_shafts"], "has_tb": ctx["has_tb"]}
    for name, fn in stats.items():
        rung = {}
        for nm in ("within_shaft", "channel", "anchor_matched"):
            if nm == "anchor_matched":
                if ctx["anchor"] is None:
                    rung[nm] = {"status": "no_anchor"}
                    continue
                # mirror A-line: only draw nulls for seizures whose bact anchor is finite
                sz_anchor = {idx: v for idx, v in ctx["sz_vals"].items()
                             if idx in ctx["anchor"] and np.all(np.isfinite(ctx["anchor"][idx]))}
                if not sz_anchor:
                    rung[nm] = {"status": "no_anchor"}
                    continue
                rung[nm] = subject_null(fn, sz_anchor, ctx["names_m"], shuffle=nm,
                                        B=B, seed=seed, anchor_by_sz=ctx["anchor"])
                continue
            rung[nm] = subject_null(fn, ctx["sz_vals"], ctx["names_m"], shuffle=nm,
                                    B=B, seed=seed, anchor_by_sz=None)
        out[name] = rung
    # sigma sweep on R2 (same-plane smoothing scale sensitivity), within-shaft null
    out["R2_sigma_sweep"] = {}
    for k in SIGMA_SWEEP:
        def R2k(v, kk=k):
            return polarity_free_maxab(ctx["rank_a"], ctx["rank_b"], v, mode="kernel",
                                       source_pts=ctx["source_pts"], support=ctx["support"],
                                       sigma=ctx["sigma"] * kk)
        out["R2_sigma_sweep"][f"{k}x"] = subject_null(R2k, ctx["sz_vals"], ctx["names_m"],
                                                      shuffle="within_shaft", B=B, seed=seed)
    # sequence sanity (no geometry): Spearman / Kendall over contacts
    out["sequence"] = {}
    for method in ("spearman", "kendall"):
        def seq(v, mm=method):
            return sequence_maxab(ctx["rank_a"], ctx["rank_b"], v, method=mm)
        out["sequence"][method] = subject_null(seq, ctx["sz_vals"], ctx["names_m"],
                                               shuffle="within_shaft", B=B, seed=seed)
    # paired subject-level deltas (real obs, deterministic): where does the signal live?
    g = lambda r: out[r]["within_shaft"].get("obs_subject", float("nan"))
    out["smooth_delta"] = float(g("R2") - g("R1"))   # same-plane smoothing contribution
    out["grid_delta"] = float(g("R3") - g("R2"))     # grid contribution
    return out


# --------------------------------------------------------------------------- cohort

def _bootstrap_median_ci(values, seed, n_boot=2000):
    v = np.asarray(values, float)
    boot = [float(np.median(np.random.default_rng(seed + b).choice(v, v.size)))
            for b in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def _r3_cross_check(subjects, maxab_ref, activation, *, atol=0.03):
    """R3.within_shaft.obs_subject must reproduce the published A-line max_ab
    real_median_abs_corr (same statistic, same RNG_SEED) within MC tolerance."""
    ref_f = maxab_ref / f"axis_alignment_{activation}_max_ab_B1000.json"
    if not ref_f.exists():
        return {"status": "no_reference", "reference": str(ref_f)}
    ref = json.load(open(ref_f))
    ref_by = {r["subject_id"]: r for r in ref.get("per_subject", []) if r.get("status") == "ok"}
    rows, max_abs = [], 0.0
    for s in subjects:
        if s.get("status") != "ok" or s["subject_id"] not in ref_by:
            continue
        r3_obs = float(s["R3"]["within_shaft"]["obs_subject"])
        ref_val = float(ref_by[s["subject_id"]]["real_median_abs_corr"])
        delta = r3_obs - ref_val
        rows.append({"subject_id": s["subject_id"], "r3_obs": r3_obs,
                     "maxab_ref": ref_val, "delta": delta, "within_tol": bool(abs(delta) <= atol)})
        max_abs = max(max_abs, abs(delta))
    return {"status": "ok", "reference": str(ref_f), "atol": atol, "n_compared": len(rows),
            "max_abs_delta": max_abs, "n_mismatch": sum(1 for r in rows if not r["within_tol"]),
            "per_subject": rows}


def _pass(rung):
    p = rung.get("within_shaft", {}).get("passed") if "within_shaft" in rung else None
    return p


def _write_cohort_csv(path, subjects):
    cols = ["subject_id", "dataset", "status", "n_seizures", "n_matched_channels", "n_shafts",
            "sigma_xy", "R1_pass", "R2_pass", "R3_pass", "smooth_delta", "grid_delta"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for s in subjects:
            if s.get("status") != "ok":
                w.writerow([s["subject_id"], s.get("dataset", ""), s.get("status", ""),
                            "", "", "", "", "", "", "", "", ""])
                continue
            w.writerow([s["subject_id"], s["dataset"], "ok", s["n_seizures"],
                        s["n_matched_channels"], s["n_shafts"], round(s["sigma_xy"], 5),
                        _pass(s["R1"]), _pass(s["R2"]), _pass(s["R3"]),
                        round(s["smooth_delta"], 5), round(s["grid_delta"], 5)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-results-root", default=DEF_ROOT,
                    help="root holding the gitignored T0 cache + axis records (default: results)")
    ap.add_argument("--out-dir", default=DEF_OUT)
    ap.add_argument("--activation", choices=list(ACTIVATION_KEY), default="broadband")
    ap.add_argument("--B", type=int, default=1000, help="null draws per seizure (smoke: 20-50)")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    root = Path(args.input_results_root)
    cache_dir = root / "topic5_ictal_recruitment" / "t0_feature_cache"
    maxab_ref = root / "topic5_ictal_recruitment" / "axis_alignment"
    out_dir = Path(args.out_dir)
    (out_dir / "per_subject").mkdir(parents=True, exist_ok=True)

    cohort = sorted(p.stem for p in cache_dir.glob("*.npz"))
    if args.subjects:
        cohort = [s for s in cohort if s in set(args.subjects)]
    print(f"[contact-similarity] activation={args.activation} B={args.B} seed={args.seed} | "
          f"{len(cohort)} cached subjects", flush=True)

    subjects = []
    for ds_sid in cohort:
        res = run_subject(ds_sid, activation=args.activation, B=args.B, seed=args.seed,
                          input_results_root=args.input_results_root)
        subjects.append(res)
        json.dump(res, open(out_dir / "per_subject" / f"{ds_sid}.json", "w"),
                  indent=2, ensure_ascii=False)
        if res["status"] == "ok":
            r3_obs = res['R3']['within_shaft'].get('obs_subject', float('nan'))
            print(f"  {ds_sid}: R1={_pass(res['R1'])} R2={_pass(res['R2'])} R3={_pass(res['R3'])} | "
                  f"smooth_d={res['smooth_delta']:+.3f} grid_d={res['grid_delta']:+.3f} "
                  f"(R3_obs={r3_obs:.3f}) n_sz={res['n_seizures']}",
                  flush=True)
        else:
            print(f"  {ds_sid}: {res['status']}", flush=True)

    ok = [s for s in subjects if s.get("status") == "ok"]
    summary = {
        "activation": args.activation, "B": args.B, "seed": args.seed,
        "n_subjects": len(subjects), "n_ok": len(ok),
        "n_pass_R1_within_shaft": sum(bool(_pass(s["R1"])) for s in ok),
        "n_pass_R2_within_shaft": sum(bool(_pass(s["R2"])) for s in ok),
        "n_pass_R3_within_shaft": sum(bool(_pass(s["R3"])) for s in ok),
    }
    if ok:
        deltas = np.array([s["grid_delta"] for s in ok if np.isfinite(s["grid_delta"])], float)
        smooth = np.array([s["smooth_delta"] for s in ok], float)
        summary["smooth_delta_median"] = float(np.median(smooth))
        slo, shi = _bootstrap_median_ci(smooth, args.seed)
        summary["smooth_delta_ci"] = [slo, shi]
        if deltas.size > 0:
            summary["grid_delta_median"] = float(np.median(deltas))
            glo, ghi = _bootstrap_median_ci(deltas, args.seed)
            summary["grid_delta_ci"] = [glo, ghi]
            # TOST-style equivalence: is the grid contribution negligible (|median| within SESOI)?
            summary["grid_negligible"] = bool(glo > -SESOI and ghi < SESOI)
        else:
            summary["grid_delta_ci"] = None
            summary["grid_negligible"] = None
    summary["r3_cross_check"] = _r3_cross_check(subjects, maxab_ref, args.activation)
    summary["per_subject"] = subjects

    json.dump(summary, open(out_dir / "cohort_summary.json", "w"), indent=2, ensure_ascii=False)
    _write_cohort_csv(out_dir / "cohort_summary.csv", subjects)

    cc = summary["r3_cross_check"]
    print(f"\nwrote {out_dir}/cohort_summary.{{json,csv}}  (n_ok={len(ok)})")
    if ok:
        if summary.get("grid_delta_ci") is not None:
            print(f"  grid_delta median={summary['grid_delta_median']:+.4f} "
                  f"CI=[{summary['grid_delta_ci'][0]:+.4f},{summary['grid_delta_ci'][1]:+.4f}] "
                  f"negligible(|.|<{SESOI})={summary['grid_negligible']}")
        else:
            print("  grid_delta: all subjects filtered (non-finite grid_delta)")
        print(f"  smooth_delta median={summary['smooth_delta_median']:+.4f}")
        print(f"  pass within_shaft: R1={summary['n_pass_R1_within_shaft']} "
              f"R2={summary['n_pass_R2_within_shaft']} R3={summary['n_pass_R3_within_shaft']} /{len(ok)}")
    if cc["status"] == "ok":
        print(f"  R3 cross-check vs {Path(cc['reference']).name}: "
              f"max|Δ|={cc['max_abs_delta']:.4f} (atol={cc['atol']}) "
              f"mismatch={cc['n_mismatch']}/{cc['n_compared']}")
    else:
        print(f"  R3 cross-check: {cc['status']} ({cc.get('reference')})")


if __name__ == "__main__":
    main()
