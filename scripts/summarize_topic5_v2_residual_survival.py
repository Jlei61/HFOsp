#!/usr/bin/env python
"""Topic5 V2 Phase-1-v2 — W1 residual-survival summary (single combined artifact).

`phase1_gate_summary.csv` reflects only the LAST feature the gate script ran (usually raw), so a
reader easily misreads it as "the gate result" and misses the residual-survival ladder. This script
recomputes the cohort-perm FWER survival (`max_over_bands_p`) for ALL three features
{raw, common_resid, aperiodic_resid} x both pools {narrow, broad} and writes ONE table:

    {V2_ROOT}/phase1_residual_survival_summary.csv
      columns: substrate, feature, band, cohort_perm_p_spatial, cohort_perm_delta_spatial,
               max_over_bands_p, survive   (survive = max_over_bands_p < alpha)

This is DESCRIPTIVE W1 residual survival under the WEAK spatial null (formal within-shaft Gate A is
NOT evaluated here). It reuses build_gate_rows() from the gate script — same cohort-perm machinery,
no re-implementation.

Usage:  python scripts/summarize_topic5_v2_residual_survival.py [--outdir <root>]
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.topic5_v2_band_scan import load_phase1_config           # noqa: E402
from run_topic5_v2_gates import build_gate_rows, V2_ROOT         # noqa: E402

FEATURES = ["raw", "common_resid", "aperiodic_resid"]
SUBSTRATES = ["narrow", "broad"]
OUT_COLS = ["substrate", "feature", "band", "cohort_perm_p_spatial",
            "cohort_perm_delta_spatial", "max_over_bands_p", "survive"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=None, help="root (reads {root}/{substrate}); default results tree")
    args = ap.parse_args()

    cfg = load_phase1_config()
    alpha = float(cfg["nulls"]["alpha"])
    primary7 = [b[0] for b in cfg["bands"]["primary"]]          # canonical FWER family order
    ripple = {"hg_low_ripple", "ripple_high"}
    outroot = Path(args.outdir) if args.outdir else V2_ROOT

    out_rows = []
    print(f"=== V2 Phase-1-v2 residual survival (alpha={alpha}; survive = max_over_bands_p<alpha) ===")
    for substrate in SUBSTRATES:
        sub_dir = outroot / substrate
        for feature in FEATURES:
            try:
                rows, _family = build_gate_rows(sub_dir, substrate, feature, cfg)
            except FileNotFoundError as e:
                print(f"[skip] {substrate}/{feature}: {e}")
                continue
            by_band = {r["band"]: r for r in rows}
            survivors = []
            for band in primary7:
                r = by_band.get(band)
                if r is None:
                    continue
                mob = r["max_over_bands_p"]
                surv = (mob is not None) and (float(mob) < alpha)
                if surv:
                    survivors.append(band)
                out_rows.append({
                    "substrate": substrate, "feature": feature, "band": band,
                    "cohort_perm_p_spatial": r.get("cohort_perm_p_spatial"),
                    "cohort_perm_delta_spatial": r.get("cohort_perm_delta_spatial"),
                    "max_over_bands_p": mob, "survive": int(surv)})
            rip = [b for b in survivors if b in ripple]
            print(f"  {substrate:6s} {feature:16s}  survivors {len(survivors)}/7: {survivors}"
                  f"   ripple: {rip}")

    out_path = outroot / "phase1_residual_survival_summary.csv"
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"[done] {len(out_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
