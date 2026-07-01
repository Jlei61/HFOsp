#!/usr/bin/env python
"""Topic 5 V2 Phase 2 (2A) — preictal susceptibility field K_t alignment to G_HFO.

EXPLORATORY peri-ictal susceptibility. Per contact, the late-minus-early CHANGE in
{variance, lag1_autocorr, line_length_rate} across the preictal window is the
susceptibility field; K = contact_alignment(field, G_HFO). Question: does rising
susceptibility concentrate on the fixed interictal HFO rank axis?

Spatial + order nulls come from Phase-1 `src/topic5_v2_band_scan.py`
(spatial_constrained_permute / order_null_rank_pair). While Phase 1 is in flight the
OBSERVED K is reported and the null columns carry `pending_phase1` — never a fabricated
null (skipped/pending != negative). Subject unit; broad/narrow never pooled.
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

from scripts._topic5_v2_crit_io import (  # noqa: E402
    load_subject_preictal, window_index_range, get_contact_alignment, get_null_fns,
)
from scripts.run_topic5_ictal_field_dynamics import SUBJECTS_BY_SUB  # noqa: E402
from src.topic5_v2_criticality import load_phase2_config, contact_susceptibility  # noqa: E402

OUT_ROOT = _ROOT / "results/topic5_ictal_recruitment/v2_criticality"
ORIENTED_TEMPLATE = "a"

COLUMNS = [
    "subject", "axis_set", "status", "skip_reason", "available_pre_sec", "required_pre_sec",
    "state_band", "feature", "K_signed_oriented", "K_abs",
    "K_spatial_null_z", "K_spatial_empirical_p", "K_order_null_z", "K_order_empirical_p",
    "spatial_null_strength", "order_null_strength", "n_contacts", "n_seizures",
    "n_seizures_total", "align_source", "null_source", "tier",
]


def _r(x, nd=4):
    return round(float(x), nd) if x is not None and np.isfinite(x) else ""


def subject_field(sub, cfg, feature):
    """Median-over-seizures per-contact susceptibility delta for one feature."""
    early = cfg["preictal"]["early_baseline_rel"]
    late = cfg["preictal"]["late_preictal_rel"]
    names = sub["mapped"]
    per_contact = {n: [] for n in names}
    for s in sub["seizures"]:
        e_idx = window_index_range(s["relt"], early[0], early[1])
        l_idx = window_index_range(s["relt"], late[0], late[1])
        if e_idx is None or l_idx is None:
            continue
        feats = contact_susceptibility(s["E"], e_idx, l_idx)
        vals = feats[feature]
        for i, n in enumerate(names):
            if np.isfinite(vals[i]):
                per_contact[n].append(float(vals[i]))
    return {n: float(np.median(v)) for n, v in per_contact.items() if v}


def run_subject(ds_sid, substrate, cfg, n_perm, seed):
    align_fn, align_source = get_contact_alignment()
    null_fns, null_source = get_null_fns()
    subj = ds_sid.split("_", 1)[1] if "_" in ds_sid else ds_sid
    features = list(cfg["susceptibility"]["features"])

    base = {c: "" for c in COLUMNS}
    base.update(subject=subj, axis_set=substrate, state_band=cfg["state_band"],
                required_pre_sec=cfg["preictal"]["min_required_pre_sec"],
                align_source=align_source, null_source=null_source,
                tier=cfg.get("tier", "exploratory"))

    sub = load_subject_preictal(ds_sid, substrate, cfg)
    base.update(available_pre_sec=round(sub["available_pre_sec"], 2),
                n_seizures=sub["n_seizures"], n_seizures_total=sub.get("n_seizures_total", sub["n_seizures"]),
                n_contacts=sub["n_contacts"])
    if sub["status"] != "ok":
        return [{**base, "status": "skipped", "skip_reason": sub["skip_reason"], "feature": ""}]

    rows = []
    for feature in features:
        field = subject_field(sub, cfg, feature)
        al = align_fn(field, sub["ta_rank"], sub["tb_rank"], ORIENTED_TEMPLATE)
        row = {**base, "status": "ok", "skip_reason": "", "feature": feature,
               "K_signed_oriented": _r(al["align_signed_oriented"]),
               "K_abs": _r(al["align_abs_maxab_contact"]),
               "spatial_null_strength": "pending_phase1",
               "order_null_strength": "pending_phase1"}
        if null_fns is not None:
            # Stage C: full spatial + order null via Phase-1 band_scan (wired + tested when landed).
            _apply_nulls(row, field, sub, align_fn, null_fns, cfg, n_perm, seed)
        rows.append(row)
    return rows


def _apply_nulls(row, field, sub, align_fn, null_fns, cfg, n_perm, seed):  # pragma: no cover - Stage C
    """Placeholder for the Phase-1 spatial + order null (activated once band_scan lands)."""
    raise NotImplementedError("Phase-1 null builders not yet integrated (Stage C).")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--substrate", choices=["broad", "narrow"], default="broad")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--n-perm", type=int, default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    cfg = load_phase2_config()
    n_perm = args.n_perm if args.n_perm is not None else int(cfg["nulls"]["n_perm_smoke"])
    seed = int(cfg["nulls"]["seed"])
    subjects = args.subjects or SUBJECTS_BY_SUB[args.substrate]
    outdir = Path(args.outdir) if args.outdir else (OUT_ROOT / args.substrate)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds_sid in subjects:
        try:
            rows.extend(run_subject(ds_sid, args.substrate, cfg, n_perm, seed))
        except Exception as exc:
            subj = ds_sid.split("_", 1)[1] if "_" in ds_sid else ds_sid
            r = {c: "" for c in COLUMNS}
            r.update(subject=subj, axis_set=args.substrate, status="skipped",
                     skip_reason=f"error:{type(exc).__name__}:{exc}", feature="",
                     state_band=cfg["state_band"], tier=cfg.get("tier", "exploratory"))
            rows.append(r)
            print(f"[WARN] {ds_sid}: {type(exc).__name__}: {exc}", file=sys.stderr)

    out_csv = outdir / "phase2_susceptibility_subject.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    n_ok = len({r["subject"] for r in rows if r["status"] == "ok"})
    print(f"[susceptibility] {args.substrate}: {n_ok} subjects ok, {len(rows)} rows, "
          f"nulls={null_source_note(rows)} -> {out_csv}")
    return out_csv


def null_source_note(rows):
    srcs = {r.get("null_source", "") for r in rows}
    return "+".join(sorted(s for s in srcs if s)) or "none"


if __name__ == "__main__":
    main()
