#!/usr/bin/env python
"""Topic 5 V2 Phase 2 (2C) — preictal avalanche transition flow along G_HFO.

EXPLORATORY. Threshold the preictal state envelope (z>thr) into activations, build the
avalanche transition matrix (ATM) over the late-preictal window, and ask whether the
activation cascade flows FORWARD along the fixed interictal HFO rank axis G_HFO.

PRIMARY direction metric = ATM forward displacement (expected next-rank minus current-rank);
direction index = robustness; rank-coupling Spearman = DESCRIPTIVE only (conflates
self-persistence with flow). NO power-law exponent. Spatial + order nulls come from Phase-1
`src/topic5_v2_band_scan.py`; until it lands they carry `pending_phase1` (never fabricated).
Subject unit; broad/narrow never pooled.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._topic5_v2_crit_io import (  # noqa: E402
    load_subject_preictal, window_index_range, get_null_fns,
)
from scripts.run_topic5_ictal_field_dynamics import SUBJECTS_BY_SUB  # noqa: E402
from src.topic5_v2_criticality import (  # noqa: E402
    load_phase2_config, activations_from_z, avalanche_atm, branching_ratio,
    atm_forward_displacement, atm_direction_index, atm_rank_coupling_spearman,
)

OUT_ROOT = _ROOT / "results/topic5_ictal_recruitment/v2_criticality"

COLUMNS = [
    "subject", "axis_set", "status", "skip_reason", "available_pre_sec", "state_band",
    "atm_forward_displacement", "atm_direction_index", "atm_rank_coupling_spearman",
    "branching_late", "branching_trend_spearman",
    "atm_spatial_null_z", "atm_spatial_empirical_p", "atm_order_null_z", "atm_order_empirical_p",
    "spatial_null_strength", "order_null_strength", "n_active_bins", "n_transitions",
    "activation_rate", "n_contacts", "n_seizures", "n_seizures_total", "null_source", "tier",
]


def _r(x, nd=4):
    return round(float(x), nd) if x is not None and np.isfinite(x) else ""


def _rank_vec(sub):
    """G_HFO rank per matched contact (template A, the pre-fixed orientation)."""
    return np.array([sub["ta_rank"].get(n, np.nan) for n in sub["mapped"]], dtype=float)


def subject_avalanche(sub, cfg):
    """Pool per-seizure late-preictal ATMs (equal weight) + branching stats."""
    thr = float(cfg["avalanche"]["z_threshold"])
    late = cfg["preictal"]["late_preictal_rel"]
    atms, branch_late, brtrend = [], [], []
    n_active_bins = n_transitions = 0
    active_frac = []
    for s in sub["seizures"]:
        l_idx = window_index_range(s["relt"], late[0], late[1])
        if l_idx is None:
            continue
        z_late = s["E"][:, l_idx[0]:l_idx[1]]
        active = activations_from_z(z_late, thr)
        atm = avalanche_atm(active)
        if atm.sum() > 0:
            atms.append(atm)
            n_transitions += int(active[:, :-1].any(axis=0).sum())  # bins with >=1 -> next
        n_active_bins += int(active.any(axis=0).sum())
        active_frac.append(float(active.mean()))
        b = branching_ratio(active)
        if np.isfinite(b):
            branch_late.append(b)
        brtrend.append(_branching_trend(s, cfg, thr))

    pooled = np.mean(atms, axis=0) if atms else np.zeros((sub["n_contacts"], sub["n_contacts"]))
    rv = _rank_vec(sub)
    brtrend = [b for b in brtrend if np.isfinite(b)]
    return {
        "atm": pooled, "rank_vec": rv,
        "fwd": atm_forward_displacement(pooled, rv),
        "dir": atm_direction_index(pooled, rv),
        "coup": atm_rank_coupling_spearman(pooled, rv),
        "branching_late": float(np.median(branch_late)) if branch_late else np.nan,
        "branching_trend": float(np.median(brtrend)) if brtrend else np.nan,
        "n_active_bins": n_active_bins, "n_transitions": n_transitions,
        "activation_rate": float(np.mean(active_frac)) if active_frac else np.nan,
        "n_atms": len(atms),
    }


def _branching_trend(s, cfg, thr):
    """Spearman(branching_ratio, window center) across the preictal span for one seizure."""
    relt = s["relt"]
    win, step = float(cfg["preictal"]["window_sec"]), float(cfg["preictal"]["step_sec"])
    start, centers, brs = float(relt.min()), [], []
    while start + win <= float(relt.max()) + 1e-9:
        rng = window_index_range(relt, start, start + win)
        start += step
        if rng is None or (rng[1] - rng[0]) < 3:
            continue
        active = activations_from_z(s["E"][:, rng[0]:rng[1]], thr)
        b = branching_ratio(active)
        if np.isfinite(b):
            centers.append(float(np.mean(relt[rng[0]:rng[1]])))
            brs.append(b)
    if len(brs) >= 4 and np.std(centers) > 0 and np.std(brs) > 0:
        return float(spearmanr(brs, centers).statistic)
    return np.nan


def run_subject(ds_sid, substrate, cfg, n_perm, seed):
    null_fns, null_source = get_null_fns()
    subj = ds_sid.split("_", 1)[1] if "_" in ds_sid else ds_sid
    base = {c: "" for c in COLUMNS}
    base.update(subject=subj, axis_set=substrate, state_band=cfg["state_band"],
                null_source=null_source, tier=cfg.get("tier", "exploratory"))

    sub = load_subject_preictal(ds_sid, substrate, cfg)
    base.update(available_pre_sec=round(sub["available_pre_sec"], 2),
                n_seizures=sub["n_seizures"], n_seizures_total=sub.get("n_seizures_total", sub["n_seizures"]),
                n_contacts=sub["n_contacts"])
    if sub["status"] != "ok":
        return {**base, "status": "skipped", "skip_reason": sub["skip_reason"]}

    av = subject_avalanche(sub, cfg)
    row = {**base, "status": "ok", "skip_reason": "",
           "atm_forward_displacement": _r(av["fwd"]), "atm_direction_index": _r(av["dir"]),
           "atm_rank_coupling_spearman": _r(av["coup"]),
           "branching_late": _r(av["branching_late"]), "branching_trend_spearman": _r(av["branching_trend"]),
           "spatial_null_strength": "pending_phase1", "order_null_strength": "pending_phase1",
           "n_active_bins": av["n_active_bins"], "n_transitions": av["n_transitions"],
           "activation_rate": _r(av["activation_rate"])}
    if null_fns is not None:
        _apply_nulls(row, sub, av, null_fns, cfg, n_perm, seed)  # Stage C
    return row


def _apply_nulls(row, sub, av, null_fns, cfg, n_perm, seed):  # pragma: no cover - Stage C
    """Spatial + order null on atm_forward_displacement (Stage C). Until wired, preserve
    OBSERVED ATM stats + flag loudly (never skip-all, never fabricate a null)."""
    row["spatial_null_strength"] = "band_scan_present_stage_c_unwired"
    row["order_null_strength"] = "band_scan_present_stage_c_unwired"
    print(f"[NOTE] {row['subject']}: Phase-1 null fns present but Stage-C null loop not yet "
          f"wired; observed ATM kept, null pending.", file=sys.stderr)


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
            rows.append(run_subject(ds_sid, args.substrate, cfg, n_perm, seed))
        except Exception as exc:
            subj = ds_sid.split("_", 1)[1] if "_" in ds_sid else ds_sid
            r = {c: "" for c in COLUMNS}
            r.update(subject=subj, axis_set=args.substrate, status="skipped",
                     skip_reason=f"error:{type(exc).__name__}:{exc}",
                     state_band=cfg["state_band"], tier=cfg.get("tier", "exploratory"))
            rows.append(r)
            print(f"[WARN] {ds_sid}: {type(exc).__name__}: {exc}", file=sys.stderr)

    out_csv = outdir / "phase2_avalanche_subject.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(r["status"] == "ok" for r in rows)
    print(f"[avalanche] {args.substrate}: {n_ok}/{len(rows)} ok -> {out_csv}")
    return out_csv


if __name__ == "__main__":
    main()
