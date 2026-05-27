"""SEF-ITP Phase 3 runner — per-seizure peri-ictal recruitment / expansion.

Plan: docs/superpowers/plans/2026-05-24-topic4-phase3-h5-per-seizure-recruitment-plan.md
Module: src/sef_itp_phase3.py
Framework: docs/topic4_sef_itp_framework.md v1.0.7

Usage:
    python scripts/run_sef_itp_phase3.py --all
    python scripts/run_sef_itp_phase3.py --subject epilepsiae_1146
    python scripts/run_sef_itp_phase3.py --all --include-expansion

Reads:
  - Phase 2 cohort gate: results/topic4_sef_itp/phase2_temporal_x_geometry/per_subject/*.json
  - PR-2 masked:         results/interictal_propagation_masked/per_subject/<...>.json
  - rank_displacement:   results/interictal_propagation_masked/rank_displacement/per_subject/<...>.json
  - lagPat NPZ:          /mnt/.../...
  - Seizure inventories: results/{dataset_inventory/yuquan_seizure_inventory,epilepsiae_seizure_inventory}.csv

Writes per-subject:
  results/topic4_sef_itp/phase3_ictal_adjacent/per_subject/<dataset>_<sid>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.sef_itp_phase1 import resolve_lagpat_subject_dir
from src.interictal_propagation import load_subject_propagation_events, _valid_event_indices
from src.seeg_coord_loader import load_subject_coords
from src.sef_itp_phase3 import (
    _load_seizure_times,
    _load_real_recording_block_ranges,
    _enumerate_peri_ictal_windows,
    _pick_matched_baseline_windows,
    _tz_for_dataset,
    compute_window_metrics,
    compute_delta_metrics,
    PERI_NOMINAL_MIN,
    COVERAGE_FLOOR,
    GUARD_HOURS_PRIMARY,
    GUARD_HOURS_SENSITIVITY,
    HOUR_TOLERANCE_H,
    BASELINE_WINDOW_MIN,
    BASELINE_STRIDE_MIN,
    N_BASELINE_MIN,
    MIN_EVENTS_PER_WINDOW,
    MIN_EVENTS_PER_CLUSTER_WINDOW,
    SWAP_SWEEP_N_PERM_PRIMARY,
    SWAP_SWEEP_N_PERM_BASELINE,
    __version__ as PHASE3_VERSION,
)


SCHEMA_VERSION = "sef_itp_phase3_v1_2026_05_24"

PHASE2_PER_SUBJECT_DIR = Path("results/topic4_sef_itp/phase2_temporal_x_geometry/per_subject")
PHASE0A_DIR = Path("results/interictal_propagation_masked/per_subject")
RANK_DISPLACEMENT_DIR = Path("results/interictal_propagation_masked/rank_displacement/per_subject")
OUT_DIR = Path("results/topic4_sef_itp/phase3_ictal_adjacent/per_subject")


def _read_phase2_cohort() -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for p in sorted(PHASE2_PER_SUBJECT_DIR.glob("*.json")):
        stem = p.stem
        dataset, _, subject = stem.partition("_")
        out.append((dataset, subject))
    return out


def _sensitivity_expansion_cohort() -> List[Tuple[str, str]]:
    p2_stems = {p.stem for p in PHASE2_PER_SUBJECT_DIR.glob("*.json")}
    extras: List[Tuple[str, str]] = []
    for p in sorted(RANK_DISPLACEMENT_DIR.glob("*.json")):
        if p.stem in p2_stems:
            continue
        d = json.loads(p.read_text())
        if d.get("stable_k") != 2:
            continue
        pairs = d.get("pairs") or []
        sc = (pairs[0] or {}).get("swap_sweep", {}).get("swap_class") if pairs else None
        if sc not in ("strict", "candidate"):
            continue
        dataset, _, subject = p.stem.partition("_")
        extras.append((dataset, subject))
    return extras


def _load_subject_swap_class(dataset: str, subject_id: str) -> Tuple[Optional[str], Optional[int]]:
    p = RANK_DISPLACEMENT_DIR / f"{dataset}_{subject_id}.json"
    if not p.exists():
        return None, None
    d = json.loads(p.read_text())
    pairs = d.get("pairs") or []
    if not pairs:
        return None, None
    ss = (pairs[0] or {}).get("swap_sweep") or {}
    return ss.get("swap_class"), ss.get("decision_k")


def _load_subject_arrays(
    dataset: str, subject_id: str,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    List[Tuple[float, float]], List[str], Optional[np.ndarray]
]:
    """Load event arrays + coords + REAL recording blocks (not event-derived).

    Phase 3 v1.1: block_time_ranges loaded from inventory CSVs (SQL/EDF head),
    NOT from load_subject_propagation_events (which derives blocks from event
    first/last time and is event-presence biased). See _load_real_recording_block_ranges
    docstring.
    """
    phase0a = json.loads((PHASE0A_DIR / f"{dataset}_{subject_id}.json").read_text())
    labels_all = np.asarray(phase0a["adaptive_cluster"]["labels"], dtype=int)
    channel_names: List[str] = list(phase0a["channel_names"])
    lagpat_dir = resolve_lagpat_subject_dir(dataset, subject_id)
    loaded = load_subject_propagation_events(lagpat_dir)
    raw_times = np.asarray(loaded["event_abs_times"], dtype=float)
    bools_full = np.asarray(loaded["bools"], dtype=int)
    ranks_full = np.asarray(loaded["ranks"], dtype=float)
    # Real recording block ranges from inventory CSVs (NOT event-derived). user catch
    # 2026-05-24: event-derived blocks miss recording extent when many events fail to
    # detect; AGENTS.md trust order SQL > head > legacy > event timing.
    blocks: List[Tuple[float, float]] = _load_real_recording_block_ranges(dataset, subject_id)
    if not blocks:
        # Fallback to event-derived if inventory CSV miss (defensive; should not happen
        # for Phase 2 cohort which has full inventory coverage)
        blocks = [tuple(tr) for tr in loaded["block_time_ranges"]]
    valid_idx = _valid_event_indices(bools_full, min_participating=3)
    if valid_idx.size != labels_all.size:
        raise RuntimeError(
            f"valid/labels mismatch for {dataset}_{subject_id}: "
            f"valid={valid_idx.size} labels={labels_all.size}"
        )
    times = raw_times[valid_idx]
    finite_mask = np.isfinite(times) & (labels_all >= 0)
    event_abs_times = times[finite_mask]
    labels = labels_all[finite_mask]
    bools = bools_full[:, valid_idx][:, finite_mask].astype(bool)
    ranks = ranks_full[:, valid_idx][:, finite_mask]
    try:
        cr = load_subject_coords(dataset=dataset, subject_id=subject_id, channel_names_requested=channel_names)
        coords = cr.coords_array_in_requested_order
    except Exception:
        coords = None
    return event_abs_times, labels, ranks, bools, blocks, channel_names, coords


def _run_subject(
    dataset: str, subject_id: str,
    guard_hours: float = GUARD_HOURS_PRIMARY,
    n_perm_primary: int = SWAP_SWEEP_N_PERM_PRIMARY,
    n_perm_baseline: int = SWAP_SWEEP_N_PERM_BASELINE,
    cluster_a: int = 0, cluster_b: int = 1,
    seed: int = 0,
) -> Dict[str, Any]:
    stem = f"{dataset}_{subject_id}"
    out: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase3_version": PHASE3_VERSION,
        "dataset": dataset,
        "subject_id": subject_id,
        "params": {
            "guard_hours_primary": guard_hours,
            "peri_nominal_minutes": PERI_NOMINAL_MIN,
            "coverage_floor": COVERAGE_FLOOR,
            "hour_tolerance_h": HOUR_TOLERANCE_H,
            "baseline_window_minutes": BASELINE_WINDOW_MIN,
            "baseline_stride_minutes": BASELINE_STRIDE_MIN,
            "n_baseline_min": N_BASELINE_MIN,
            "min_events_per_cluster_window": MIN_EVENTS_PER_CLUSTER_WINDOW,
            "n_perm_primary": n_perm_primary,
            "n_perm_baseline": n_perm_baseline,
            "cluster_a": cluster_a, "cluster_b": cluster_b,
            "seed": seed,
        },
    }
    swap_class, dk = _load_subject_swap_class(dataset, subject_id)
    out["swap_class_full_data"] = swap_class
    out["global_decision_k"] = dk

    seizures = _load_seizure_times(dataset, subject_id)
    out["n_seizures_total"] = len(seizures)
    if not seizures:
        out["exit_reason"] = "no_seizures_in_inventory"
        return out

    try:
        evt_times, labels, ranks, bools, blocks, channel_names, coords = _load_subject_arrays(
            dataset, subject_id,
        )
    except Exception as e:
        out["exit_reason"] = f"load_failed:{type(e).__name__}:{e}"
        return out

    out["n_events_valid"] = int(evt_times.size)
    out["n_blocks"] = len(blocks)
    out["total_recording_seconds"] = float(sum(b[1] - b[0] for b in blocks))
    if labels.size == 0:
        out["exit_reason"] = "no_valid_events"
        return out

    tz_name = _tz_for_dataset(dataset)
    peri_records = _enumerate_peri_ictal_windows(seizures, blocks)
    out["n_peri_records"] = len(peri_records)

    per_seizure_out: List[Dict[str, Any]] = []
    t0 = time.time()
    for ridx, pr in enumerate(peri_records):
        sz_rec: Dict[str, Any] = {
            "seizure_id": pr["seizure_id"],
            "seizure_onset_t": pr["onset"],
            "seizure_offset_t": pr["offset"],
            "classification": pr.get("classification"),
        }
        for side in ("pre", "post"):
            side_rec: Dict[str, Any] = {
                "t_start": pr[side]["t_start"], "t_end": pr[side]["t_end"],
                "effective_seconds": pr[side]["effective_seconds"],
                "coverage": pr[side]["coverage"],
                "qualifies_coverage": pr[side]["qualifies"],
            }
            if not pr[side]["qualifies"]:
                side_rec["exit_reason"] = "coverage_below_floor"
                sz_rec[side] = side_rec
                continue
            # Compute peri-ictal window metrics (v1.1 gate: also check total n_events ≥
            # MIN_EVENTS_PER_WINDOW per plan §1 v4 catch — locked at B0 audit time = 30)
            peri_m = compute_window_metrics(
                t_start=pr[side]["t_start"], t_end=pr[side]["t_end"],
                event_abs_times=evt_times, labels=labels, ranks=ranks, bools=bools,
                coords=coords, channel_names=channel_names,
                cluster_a=cluster_a, cluster_b=cluster_b,
                effective_seconds=pr[side]["effective_seconds"],
                n_perm=n_perm_primary, seed=seed + ridx * 10,
                min_events_per_cluster=MIN_EVENTS_PER_CLUSTER_WINDOW,
                min_events_per_window=MIN_EVENTS_PER_WINDOW,
            )
            side_rec["peri_metrics"] = peri_m
            if peri_m.get("exit_reason") != "ok":
                side_rec["exit_reason"] = f"peri:{peri_m.get('exit_reason')}"
                sz_rec[side] = side_rec
                continue
            # Pick matched baselines
            matched = _pick_matched_baseline_windows(
                pr[side]["t_start"], pr[side]["t_end"], seizures, blocks,
                tz_name=tz_name, guard_hours=guard_hours,
            )
            side_rec["n_baseline_matched"] = len(matched)
            if len(matched) < N_BASELINE_MIN:
                side_rec["exit_reason"] = f"insufficient_baselines:{len(matched)}<{N_BASELINE_MIN}"
                sz_rec[side] = side_rec
                continue
            # Compute metrics for each baseline window
            baseline_metrics: List[Dict[str, Any]] = []
            for bi, m in enumerate(matched):
                bm = compute_window_metrics(
                    t_start=m["t_start"], t_end=m["t_end"],
                    event_abs_times=evt_times, labels=labels, ranks=ranks, bools=bools,
                    coords=coords, channel_names=channel_names,
                    cluster_a=cluster_a, cluster_b=cluster_b,
                    effective_seconds=m["effective_seconds"],
                    n_perm=n_perm_baseline, seed=seed + ridx * 100 + bi,
                    min_events_per_cluster=MIN_EVENTS_PER_CLUSTER_WINDOW,
                    min_events_per_window=MIN_EVENTS_PER_WINDOW,
                )
                baseline_metrics.append(bm)
            # Δmetrics
            delta = compute_delta_metrics(peri_m, baseline_metrics)
            side_rec["n_baseline_qualifying"] = delta["n_baseline_qualifying"]
            side_rec["deltas"] = delta
            # Don't store full baseline_metrics list (too large); store summary
            n_ok = sum(1 for b in baseline_metrics if b.get("exit_reason") == "ok")
            side_rec["baseline_summary"] = {
                "n_total_matched": len(matched),
                "n_window_metrics_ok": n_ok,
                "n_qualifying_for_delta": delta["n_baseline_qualifying"],
            }
            side_rec["exit_reason"] = delta.get("exit_reason", "ok")
            sz_rec[side] = side_rec
        per_seizure_out.append(sz_rec)
        # Progress every 5 seizures
        if (ridx + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  [{stem}] seizure {ridx + 1}/{len(peri_records)} done ({elapsed:.1f}s)",
                  flush=True)

    out["per_seizure"] = per_seizure_out
    out["exit_reason"] = "ok"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", help="dataset_subject e.g. epilepsiae_1146")
    parser.add_argument("--all", action="store_true", help="Run Phase 2 cohort")
    parser.add_argument("--include-expansion", action="store_true",
                        help="Also run sensitivity expansion cohort")
    parser.add_argument("--guard-hours", type=float, default=GUARD_HOURS_PRIMARY,
                        help=f"Baseline guard hours (default {GUARD_HOURS_PRIMARY})")
    parser.add_argument("--n-perm-primary", type=int, default=SWAP_SWEEP_N_PERM_PRIMARY)
    parser.add_argument("--n-perm-baseline", type=int, default=SWAP_SWEEP_N_PERM_BASELINE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-suffix", default="",
                        help="Optional suffix on per-subject JSON filename (e.g. '_guard4h_sensitivity')")
    args = parser.parse_args()

    if args.subject:
        ds, _, sub = args.subject.partition("_")
        cohort = [(ds, sub)]
    elif args.all:
        cohort = _read_phase2_cohort()
        if args.include_expansion:
            cohort = cohort + _sensitivity_expansion_cohort()
    else:
        parser.error("provide --subject or --all")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Phase3] running {len(cohort)} subjects, guard_hours={args.guard_hours}")
    print(f"[Phase3] output: {OUT_DIR}")

    for ds, sub in cohort:
        print(f"[Phase3] {ds}_{sub} ...", flush=True)
        t_start = time.time()
        try:
            rec = _run_subject(
                ds, sub,
                guard_hours=args.guard_hours,
                n_perm_primary=args.n_perm_primary,
                n_perm_baseline=args.n_perm_baseline,
                seed=args.seed,
            )
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            continue
        out_path = OUT_DIR / f"{ds}_{sub}{args.out_suffix}.json"
        out_path.write_text(json.dumps(rec, indent=2, default=str))
        elapsed = time.time() - t_start
        n_ok = sum(1 for sz in (rec.get("per_seizure") or [])
                   if any(sz.get(side, {}).get("exit_reason") == "ok" for side in ("pre", "post")))
        print(f"  → {out_path.name}: n_seizures_total={rec.get('n_seizures_total')}, "
              f"n_ok_at_least_one_side={n_ok}, exit={rec.get('exit_reason')}, t={elapsed:.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
