#!/usr/bin/env python
"""Topic 5 V2 Phase 2 — join the three state-layer legs into one exploratory summary.

Combines susceptibility K_t (primary feature line_length_rate), dynamics M_loading /
lambda-trend, and avalanche ATM forward-flow per subject. `state_leg_supported` is a
DESCRIPTIVE flag that NEVER upgrades a claim without Phase-1 Gate A: it requires
within-shaft-strong spatial null + non-weak order null + at least one leg's null passing.
While Phase-1 nulls are pending it is False by construction. Subject unit; broad/narrow
never pooled; cohort denominator from phase1_cohort_manifest when present.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = _ROOT / "results/topic5_ictal_recruitment/v2_criticality"
BAND_SCAN_ROOT = _ROOT / "results/topic5_ictal_recruitment/v2_band_scan"
PRIMARY_SUSC_FEATURE = "line_length_rate"

SUMMARY_COLUMNS = [
    "subject", "axis_set", "status", "state_band",
    "K_signed_oriented", "K_spatial_empirical_p", "K_order_empirical_p",
    "M_loading_spearman", "M_phase_empirical_p", "M_block_empirical_p", "cv_r2",
    "var_meaningful_flag", "lambda_trend_spearman", "lambda_trend_phase_empirical_p",
    "atm_forward_displacement", "atm_spatial_empirical_p", "atm_order_empirical_p",
    "spatial_null_strength", "order_null_strength", "tier", "state_leg_supported",
]


def _read(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


def _by_subject(rows, pick=None):
    out = {}
    for r in rows:
        if pick and not pick(r):
            continue
        out[r["subject"]] = r
    return out


def _sig(p, alpha):
    try:
        return float(p) < alpha
    except (TypeError, ValueError):
        return False


def state_leg_supported(row, alpha):
    """DESCRIPTIVE support flag — never upgrades without Gate A (spec §1.1)."""
    if row["status"] != "ok":
        return False
    if row.get("spatial_null_strength") != "within_shaft_strong":
        return False
    if row.get("order_null_strength") == "weak_downgrade":
        return False
    k_sig = _sig(row.get("K_spatial_empirical_p"), alpha) and _sig(row.get("K_order_empirical_p"), alpha)
    m_sig = (str(row.get("var_meaningful_flag")) == "True"
             and _sig(row.get("M_phase_empirical_p"), alpha) and _sig(row.get("M_block_empirical_p"), alpha))
    a_sig = _sig(row.get("atm_spatial_empirical_p"), alpha) and _sig(row.get("atm_order_empirical_p"), alpha)
    return bool(k_sig or m_sig or a_sig)


def _manifest_denominator(axis_set):
    for name in ("phase1_cohort_manifest.csv",):
        p = BAND_SCAN_ROOT / name
        if p.exists():
            rows = _read(p)
            inc = [r for r in rows if r.get("axis_set") == axis_set and str(r.get("included")) == "True"]
            if inc:
                return len(inc), "phase1_cohort_manifest"
    return None, "manifest_pending"


def build_summary(axis_set, base_dir, cfg):
    alpha = float(cfg["nulls"]["alpha"])
    susc = _by_subject(_read(base_dir / "phase2_susceptibility_subject.csv"),
                       pick=lambda r: r.get("feature") == PRIMARY_SUSC_FEATURE or r["status"] != "ok")
    dyn = _by_subject(_read(base_dir / "phase2_dynamics_subject.csv"))
    aval = _by_subject(_read(base_dir / "phase2_avalanche_subject.csv"))
    subjects = sorted(set(susc) | set(dyn) | set(aval))

    rows = []
    for s in subjects:
        k, d, a = susc.get(s, {}), dyn.get(s, {}), aval.get(s, {})
        status = "ok" if "ok" in (k.get("status"), d.get("status"), a.get("status")) else "skipped"
        row = {c: "" for c in SUMMARY_COLUMNS}
        row.update(
            subject=s, axis_set=axis_set, status=status, state_band=cfg["state_band"],
            K_signed_oriented=k.get("K_signed_oriented", ""),
            K_spatial_empirical_p=k.get("K_spatial_empirical_p", ""),
            K_order_empirical_p=k.get("K_order_empirical_p", ""),
            M_loading_spearman=d.get("M_loading_spearman", ""),
            M_phase_empirical_p=d.get("M_phase_empirical_p", ""),
            M_block_empirical_p=d.get("M_block_empirical_p", ""),
            cv_r2=d.get("cv_r2", ""), var_meaningful_flag=d.get("var_meaningful_flag", ""),
            lambda_trend_spearman=d.get("lambda_trend_spearman", ""),
            lambda_trend_phase_empirical_p=d.get("lambda_trend_phase_empirical_p", ""),
            atm_forward_displacement=a.get("atm_forward_displacement", ""),
            atm_spatial_empirical_p=a.get("atm_spatial_empirical_p", ""),
            atm_order_empirical_p=a.get("atm_order_empirical_p", ""),
            spatial_null_strength=k.get("spatial_null_strength", a.get("spatial_null_strength", "")),
            order_null_strength=k.get("order_null_strength", d.get("order_null_strength", "")),
            tier=cfg.get("tier", "exploratory"),
        )
        row["state_leg_supported"] = state_leg_supported(row, alpha)
        rows.append(row)
    return rows


def _median(rows, col):
    vals = []
    for r in rows:
        try:
            vals.append(float(r[col]))
        except (TypeError, ValueError):
            pass
    return float(np.median(vals)) if vals else None


def cohort_line(rows, axis_set):
    ok = [r for r in rows if r["status"] == "ok"]
    denom, denom_src = _manifest_denominator(axis_set)
    return {
        "axis_set": axis_set, "tier": "exploratory",
        "n_subjects_ok": len(ok), "n_subjects_valid_denominator": denom,
        "denominator_source": denom_src,
        "median_K_signed_oriented_line_length_rate": _median(ok, "K_signed_oriented"),
        "median_M_loading_spearman": _median(ok, "M_loading_spearman"),
        "median_lambda_trend_spearman": _median(ok, "lambda_trend_spearman"),
        "median_atm_forward_displacement": _median(ok, "atm_forward_displacement"),
        "n_state_leg_supported": sum(1 for r in ok if r["state_leg_supported"]),
        "note": "EXPLORATORY peri-ictal susceptibility; NOT forecasting; state_leg_supported "
                "requires Phase-1 nulls + Gate A; broad/narrow never pooled.",
    }


def main(argv=None):
    from src.topic5_v2_criticality import load_phase2_config
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--substrate", choices=["broad", "narrow"], default="broad")
    ap.add_argument("--basedir", default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    cfg = load_phase2_config()
    base_dir = Path(args.basedir) if args.basedir else (OUT_ROOT / args.substrate)
    outdir = Path(args.outdir) if args.outdir else base_dir
    outdir.mkdir(parents=True, exist_ok=True)

    rows = build_summary(args.substrate, base_dir, cfg)
    out_csv = outdir / "phase2_criticality_summary.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    cohort = cohort_line(rows, args.substrate)
    (outdir / "phase2_criticality_cohort.json").write_text(json.dumps(cohort, indent=2))
    print(f"[summary] {args.substrate}: {len(rows)} subjects, "
          f"{cohort['n_state_leg_supported']} state_leg_supported (of {cohort['n_subjects_ok']} ok) -> {out_csv}")
    return out_csv


if __name__ == "__main__":
    sys.path.insert(0, str(_ROOT))
    main()
