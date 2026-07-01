"""Topic 5 V2 Phase 1 — Task 4: legacy bb/hfa reproduction QC (per axis_set, HARD GATE).

QC-1 ONLY (unmasked, all-channel reproduction). This is the linchpin gate: it proves the V2
backbone faithfully recomputes the committed legacy `align_maxab` numbers BEFORE the pipeline is
extended to the masked multi-band cache. If any subject drifts past tolerance, exit non-zero and
STOP — a real orchestration bug OR data/code drift since the CSV was committed, NOT to be papered
over. (QC-2, the v2 fixed-mask cross-check, needs the Task-6 cache and is a Task-7 concern; NOT here.)

Reproduction (faithful): reuse `run_subject(ds_sid, substrate)` directly (same load_context +
per-eligible-seizure onset/offset window slide + `window_maxab` on the UNMASKED `bb_zt`/`hfa_zt`
long cache as the old pipeline; parity_fail seizures skipped inside run_subject). Per (subject, band):
new_subject_median = nanmedian(align_maxab over ALL rows for that subject+band). From the legacy CSV
(LEGACY_CSV_BY_AXIS[substrate]) take old_subject_median over the IDENTICAL filter (same ds_sid + band,
all rows). delta = new - old, expected ~0 (deterministic recompute).

Plan: docs/superpowers/plans/2026-07-01-topic5-v2-phase1-band-scan-backbone.md Task 4.
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.run_topic5_ictal_field_dynamics import CACHE, SUBJECTS_BY_SUB, run_subject
from src.topic5_v2_band_scan import load_phase1_config

# Patch B: legacy per-axis committed CSVs (source of `old_subject_median`).
LEGACY_CSV_BY_AXIS = {
    "broad": _ROOT / "results/topic5_ictal_recruitment/field_dynamics/per_seizure_metrics.csv",
    "narrow": _ROOT / "results/topic5_ictal_recruitment/field_dynamics_narrow/per_seizure_metrics.csv",
}
OUT_ROOT = _ROOT / "results/topic5_ictal_recruitment/v2_band_scan"
BANDS = ("bb", "hfa")
QC_COLS = ["subject", "axis_set", "band", "n_seizures", "n_windows",
           "old_subject_median", "new_subject_median", "delta"]


def _new_subject_medians(ds_sid, substrate):
    """Recompute align_maxab from the EXISTING unmasked long cache via run_subject (rev1 path).

    Returns {band: (median, n_seizures, n_windows)} where the filter is (this subject, band, ALL
    rows). parity_fail seizures are skipped INSIDE run_subject (never appended), which mirrors the
    committed CSV (only parity_fail=False rows) — so old/new see the same row set."""
    rows, _subj = run_subject(ds_sid, substrate)
    out = {}
    for band in BANDS:
        vals = [r["align_maxab"] for r in rows if r["band"] == band]
        seiz = {r["seizure_idx"] for r in rows if r["band"] == band}
        med = float(np.nanmedian(vals)) if vals else float("nan")
        out[band] = (med, len(seiz), len(vals))
    return out


def _old_subject_median(legacy_rows, ds_sid, band):
    """IDENTICAL filter to the new side: ds_sid==X AND band==Y, over ALL rows -> nanmedian.

    Match on ds_sid (not the short `subject`) so the same subject number across datasets can never
    collide. The committed CSV holds only parity_fail=False rows, so 'all rows' == the new row set."""
    vals = [float(r["align_maxab"]) if r["align_maxab"] != "" else float("nan")
            for r in legacy_rows if r["ds_sid"] == ds_sid and r["band"] == band]
    return float(np.nanmedian(vals)) if vals else float("nan")


def run_axis(substrate, out_root, tol):
    legacy_rows = list(csv.DictReader(open(LEGACY_CSV_BY_AXIS[substrate])))
    out_rows, failures = [], []
    for ds_sid in SUBJECTS_BY_SUB[substrate]:
        if not (CACHE / f"{ds_sid}.npz").exists():
            print(f"[skip] {ds_sid} no long cache", flush=True)
            continue
        new = _new_subject_medians(ds_sid, substrate)
        subject = ds_sid.split("_", 1)[1]
        for band in BANDS:
            new_med, n_seiz, n_win = new[band]
            old_med = _old_subject_median(legacy_rows, ds_sid, band)
            delta = new_med - old_med
            out_rows.append(dict(subject=subject, axis_set=substrate, band=band,
                                 n_seizures=n_seiz, n_windows=n_win,
                                 old_subject_median=old_med, new_subject_median=new_med,
                                 delta=delta))
            ok = np.isfinite(delta) and abs(delta) <= tol
            if not ok:
                failures.append((subject, band, delta))
            print(f"[{substrate}] {subject:>5} {band:>3}: n_sz={n_seiz:>2} n_win={n_win:>3} "
                  f"old={old_med:.6f} new={new_med:.6f} delta={delta:+.6f}"
                  f"{'  <-- OVER TOL' if not ok else ''}", flush=True)

    outdir = out_root / substrate
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "phase1_qc_legacy_reproduction.csv"
    with open(outpath, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=QC_COLS)
        w.writeheader()
        w.writerows(out_rows)
    max_abs = max((abs(r["delta"]) for r in out_rows if np.isfinite(r["delta"])), default=float("nan"))
    print(f"[done] {substrate}: {len(out_rows)} rows -> {outpath} | max|delta|={max_abs:.2e} "
          f"| tol={tol} | failures={len(failures)}", flush=True)
    return out_rows, failures, outpath


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--substrate", choices=list(LEGACY_CSV_BY_AXIS), default="broad")
    ap.add_argument("--outdir", default=None,
                    help="override output ROOT (default results/.../v2_band_scan); "
                         "writes {outdir}/{axis_set}/phase1_qc_legacy_reproduction.csv")
    args = ap.parse_args()
    tol = float(load_phase1_config()["tolerances"]["legacy_subject_median_abs"])
    out_root = Path(args.outdir) if args.outdir else OUT_ROOT
    _rows, failures, _outpath = run_axis(args.substrate, out_root, tol)
    if failures:
        print(f"[FAIL] {args.substrate} legacy reproduction over tol (|delta|>{tol}): "
              + ", ".join(f"{s}/{b}={d:+.4f}" for s, b, d in failures), file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
