#!/usr/bin/env python3
"""Topic 5 — TA/TB interictal field-reversal gate: runner (broad + narrow substrates).

Spec: docs/superpowers/specs/2026-07-06-topic5-tatb-field-reversal-design.md
Plan: docs/superpowers/plans/2026-07-06-topic5-tatb-field-reversal.md (Task 8)

Per subject x substrate: load the A/B interictal labels + per-event ranks (C1-guarded),
map cluster_id -> t_a/t_b (C2), pick the t_a plane as the shared reference frame (P0 §3.1),
then run the within-shaft reversal gate (primary, §4), the channel-shuffle floor (§4), the
non-inferential random-split contrast (§4), the contact-level head-to-head (§5), the LOO
field-vs-contact reproducibility supplement (§6), and a {0.5,1,2}x bandwidth sweep (§7).

broad and narrow are run and reported SEPARATELY, never pooled (spec §8). Every requested
subject gets a per-subject JSON with an explicit `reason` (no_planes / load_error /
c1_violation / cluster_map_ambiguous / insufficient_overlap / degenerate_null / ok) — a
degenerate/ambiguous/insufficient result IS a recorded outcome (the accountability table),
never a silent skip or a crash.
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
    load_event_labels_ranks, map_clusters_to_templates, class_aggregate_contact_values)
from src.propagation_contact_plane_readout import make_plane_grid
from src.topic5_field_reversal import (
    within_shaft_reversal_gate, channel_floor, random_split_contrast,
    contact_reversal_gate, loo_reproducibility, cohort_binomial)

OUT = Path("results/topic5_ictal_recruitment/field_reversal")
RNG_SEED = 20260706
SIGMA_MULTS = (0.5, 1.0, 2.0)
# Reason taxonomy (spec §8 accountability table). "ok" == a real gate ran (not degenerate,
# not insufficient overlap); every other value is a recorded skip, never a crash.
REASONS = ("no_planes", "load_error", "c1_violation", "plane_not_built",
           "cluster_map_ambiguous", "insufficient_overlap", "degenerate_null", "ok")


def _vec_in_order(name_to_val, order):
    return np.array([name_to_val.get(n, np.nan) for n in order], float)


def _plane_usable(plane: dict) -> bool:
    """False for status-only records (e.g. status="descriptive_only") that have no
    "channels" key -- the geometry plane was never built for this subject/template."""
    return isinstance(plane, dict) and "channels" in plane


def pick_reference(cmap, plane_a, plane_b):
    """P0 reference frame = the plane mapped to t_a. Returns (plane_ref, ta_label, tb_label)."""
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


def _run_subject(ds_sid, substrate, *, geom_dir, X, Y, rng, n_perm, n_split, loo_split, min_eff):
    dataset, subject = ds_sid.split("_", 1)
    out = {"ds_sid": ds_sid, "dataset": dataset, "subject": subject, "substrate": substrate}

    ta_f = geom_dir / f"{ds_sid}_t_a.json"
    tb_f = geom_dir / f"{ds_sid}_t_b.json"
    if not (ta_f.exists() and tb_f.exists()):
        out.update(reason="no_planes", status="no_planes")
        return out

    try:
        bundle = load_event_labels_ranks(dataset, subject, broad=(substrate == "broad"))
    except FileNotFoundError as e:
        out.update(reason="load_error", status="load_error", detail=str(e))
        return out
    except ValueError as e:
        out.update(reason="c1_violation", status="c1_violation", detail=str(e))
        return out

    plane_a = json.load(open(ta_f)); plane_b = json.load(open(tb_f))
    if not (_plane_usable(plane_a) and _plane_usable(plane_b)):
        # NARROW-substrate status-only records (status="descriptive_only") have no "channels"
        # key -- the geometry plane was never built for this subject. Record and move on;
        # do not crash the whole cohort on plane_a["channels"]/plane_b["channels"].
        out.update(reason="plane_not_built", status="plane_not_built",
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
        out.update(reason="cluster_map_ambiguous", status="cluster_map_ambiguous",
                    diag_minus_offdiag=cmap["diag_minus_offdiag"], corr_matrix=cmap["corr_matrix"])
        return out
    out["cluster_map"] = cmap["map"]

    # P0: reference frame = t_a's plane; cav0/cav1 are always TA/TB (template-relative, not
    # raw cluster indices) — matches build_reversal_fields' field0=TA/field1=TB convention.
    plane_ref, ta_label, tb_label = pick_reference(cmap, plane_a, plane_b)
    cav0 = class_aggregate_contact_values(bundle, ta_label)
    cav1 = class_aggregate_contact_values(bundle, tb_label)

    gate = within_shaft_reversal_gate(plane_ref, cav0, cav1, X=X, Y=Y, sigma=None,
                                      n_perm=n_perm, rng=rng, min_eff=min_eff)
    if gate["degenerate_null"]:
        reason = "degenerate_null"
    elif gate["insufficient_overlap"]:
        reason = "insufficient_overlap"
    else:
        reason = "ok"
    out.update(reason=reason, status=reason, gate=gate)

    # Same frame/grid/sigma/s_thresh for every secondary metric (§3.1 P0 shared-frame discipline).
    sigma = gate["sigma"]
    floor = channel_floor(plane_ref, cav0, cav1, X=X, Y=Y, sigma=sigma, n_perm=n_perm, rng=rng)
    split = random_split_contrast(bundle, plane_ref, X=X, Y=Y, sigma=sigma, n_split=n_split, rng=rng)
    contact = contact_reversal_gate(cav0, cav1, n_perm=n_perm, rng=rng, min_eff=min_eff)
    loo = loo_reproducibility(bundle, plane_ref, n_split=loo_split, rng=rng, sigma=sigma)

    # Bandwidth sweep (§7): only the gate is re-run, at {0.5,1,2}x the primary sigma. The 1.0x
    # entry is NOT re-run — it is exactly the step-5 primary gate above (same sigma, same
    # meaning); re-running it would burn n_perm draws for a value we already have.
    sweep = {}
    for mult in SIGMA_MULTS:
        key = f"{mult}x"
        if mult == 1.0:
            sweep[key] = gate
        else:
            sweep[key] = within_shaft_reversal_gate(
                plane_ref, cav0, cav1, X=X, Y=Y, sigma=mult * sigma,
                n_perm=n_perm, rng=rng, min_eff=min_eff)

    out.update(channel_floor=floor, random_split=split, contact_gate=contact,
               loo=loo, sweep=sweep, base_sigma=float(sigma))
    return out


def _wilcoxon_field_vs_contact(ok_records):
    """§6 supplement, cohort level: paired field_rho vs contact_rho over ok subjects."""
    pairs = [(r["loo"]["field_rho"], r["loo"]["contact_rho"]) for r in ok_records]
    pairs = [(f, c) for f, c in pairs if np.isfinite(f) and np.isfinite(c)]
    if not pairs:
        return {"n": 0, "statistic": None, "p_value": None}
    f = np.array([p[0] for p in pairs]); c = np.array([p[1] for p in pairs])
    try:
        stat, p = wilcoxon(f, c)
        return {"n": len(pairs), "statistic": float(stat), "p_value": float(p)}
    except ValueError as e:
        return {"n": len(pairs), "statistic": None, "p_value": None, "note": str(e)}


def _aggregate_sweep(ok_records):
    """§7 bandwidth sensitivity, cohort level: for each sigma multiplier, how many ok
    subjects still pass the within-shaft reversal gate (cohort binomial) + median
    signed_corr. Answers 'is the reversal robust to bandwidth choice, or a smoothing
    artifact?'. Keyed by multiplier ('0.5x'/'1.0x'/'2.0x'); 1.0x IS the primary gate."""
    out = {}
    if not ok_records:
        return out
    keys = sorted({k for r in ok_records for k in r.get("sweep", {})},
                  key=lambda s: float(s.rstrip("x")))
    for key in keys:
        entries = [r["sweep"][key] for r in ok_records if key in r.get("sweep", {})]
        passed = [bool(e["passed"]) for e in entries if e.get("passed") is not None]
        corrs = [e["signed_corr"] for e in entries
                 if e.get("signed_corr") is not None and np.isfinite(e["signed_corr"])]
        out[key] = {"n": len(passed), "n_pass": int(sum(passed)),
                    "binomial": cohort_binomial(passed) if passed else None,
                    "median_signed_corr": float(np.median(corrs)) if corrs else None}
    return out


def _aggregate_cohort(records):
    accountability = {r: 0 for r in REASONS}
    for rec in records:
        accountability[rec["reason"]] = accountability.get(rec["reason"], 0) + 1
    # reason == "ok" already means "gate ran, not degenerate, overlap sufficient" (see
    # _run_subject) — this IS the "non-degenerate ok subjects" cohort set, no extra filtering.
    ok = [r for r in records if r["reason"] == "ok"]
    binom = cohort_binomial([r["gate"]["passed"] for r in ok])
    wil = _wilcoxon_field_vs_contact(ok)
    return {"n_subjects": len(records), "n_ok": len(ok),
            "accountability": accountability, "binomial": binom,
            "field_vs_contact_wilcoxon": wil,
            "bandwidth_sweep": _aggregate_sweep(ok)}


def _sensitivity_broad_vs_narrow(records_by_substrate):
    """§8 sensitivity: subjects ok in BOTH substrates, never pooled."""
    broad = {r["ds_sid"]: r for r in records_by_substrate["broad"] if r["reason"] == "ok"}
    narrow = {r["ds_sid"]: r for r in records_by_substrate["narrow"] if r["reason"] == "ok"}
    common = sorted(set(broad) & set(narrow))
    per_subject = []
    for ds_sid in common:
        b, n = broad[ds_sid], narrow[ds_sid]
        pb, pn = bool(b["gate"]["passed"]), bool(n["gate"]["passed"])
        per_subject.append({"ds_sid": ds_sid,
                            "signed_corr_broad": b["gate"]["signed_corr"],
                            "signed_corr_narrow": n["gate"]["signed_corr"],
                            "passed_broad": pb, "passed_narrow": pn})
    concordance = {
        "both_pass": sum(1 for p in per_subject if p["passed_broad"] and p["passed_narrow"]),
        "broad_only": sum(1 for p in per_subject if p["passed_broad"] and not p["passed_narrow"]),
        "narrow_only": sum(1 for p in per_subject if not p["passed_broad"] and p["passed_narrow"]),
        "neither": sum(1 for p in per_subject if not p["passed_broad"] and not p["passed_narrow"]),
    }
    n_b, k_b = len(broad), sum(1 for r in broad.values() if r["gate"]["passed"])
    n_n, k_n = len(narrow), sum(1 for r in narrow.values() if r["gate"]["passed"])
    return {"n_common_ok": len(common), "per_subject": per_subject,
            "pass_concordance_2x2": concordance,
            "summary": (f"broad {k_b}/{n_b} vs narrow {k_n}/{n_n} passed own within-shaft gate; "
                        f"of the {len(common)} subjects ok in BOTH, {concordance['both_pass']} "
                        f"passed both (never pooled)")}


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
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-split", type=int, default=200)
    ap.add_argument("--loo-split", type=int, default=50)
    ap.add_argument("--min-eff", type=int, default=6)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    GEOM = _geom(Path(args.input_results_root))
    substrates = ["broad", "narrow"] if args.substrate == "both" else [args.substrate]
    outdir = Path(args.out)
    X, Y = make_plane_grid()
    rng = np.random.default_rng(RNG_SEED)

    records_by_substrate, cohort_by_substrate = {}, {}
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
                               n_perm=args.n_perm, n_split=args.n_split,
                               loo_split=args.loo_split, min_eff=args.min_eff)
            json.dump(res, open(outdir / "per_subject" / substrate / f"{ds_sid}.json", "w"), indent=2)
            print(f"    reason={res['reason']}", flush=True)
            records.append(res)
        records_by_substrate[substrate] = records
        cohort_by_substrate[substrate] = _aggregate_cohort(records)

    cohort_summary = {
        "note": "broad and narrow reported separately; never pooled (spec §8).",
        "n_perm": args.n_perm, "n_split": args.n_split,
        "loo_split": args.loo_split, "min_eff": args.min_eff,
        **cohort_by_substrate,
    }
    if "broad" in records_by_substrate and "narrow" in records_by_substrate:
        cohort_summary["sensitivity_broad_vs_narrow"] = \
            _sensitivity_broad_vs_narrow(records_by_substrate)
    else:
        cohort_summary["sensitivity_broad_vs_narrow"] = \
            {"status": "requires_both_substrates_in_same_run"}

    outdir.mkdir(parents=True, exist_ok=True)
    json.dump(cohort_summary, open(outdir / "cohort_summary.json", "w"), indent=2)
    print(f"[done] wrote {outdir/'cohort_summary.json'}")


if __name__ == "__main__":
    main()
