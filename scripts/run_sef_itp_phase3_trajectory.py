"""Phase 3 v2 runner — trajectory + alt endpoints + electrode coverage.

For each subject in Phase 3 cohort, for each seizure:
  - Compute continuous trajectory of geometry metrics across -24h to +24h around onset
  - Use 3 endpoint modes: swap_k (variable-k), top_k_3, top_k_5 (fixed-k PR-6 style)
  - Compute electrode coverage diagnostic (one per subject)

Outputs:
  results/topic4_sef_itp/phase3_ictal_adjacent/v2_trajectory/per_subject/<dataset>_<sid>.json
  results/topic4_sef_itp/phase3_ictal_adjacent/v2_trajectory/cohort_summary.json
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
)
from src.sef_itp_phase3_trajectory import (
    compute_trajectory_for_seizure,
    compute_trajectory_full_sweep,
    compute_electrode_coverage,
    __version__ as PHASE3_TRAJ_VERSION,
)


SCHEMA_VERSION = "sef_itp_phase3_trajectory_v1_2026_05_24"

PHASE2_PER_SUBJECT_DIR = Path("results/topic4_sef_itp/phase2_temporal_x_geometry/per_subject")
PHASE0A_DIR = Path("results/interictal_propagation_masked/per_subject")
RANK_DISPLACEMENT_DIR = Path("results/interictal_propagation_masked/rank_displacement/per_subject")
SOZ_YUQUAN = Path("results/yuquan_soz_core_channels.json")
SOZ_EPI = Path("results/epilepsiae_soz_core_channels.json")
OUT_DIR = Path("results/topic4_sef_itp/phase3_ictal_adjacent/v2_trajectory")


def _read_cohort(full_pr2: bool = False) -> List[Tuple[str, str]]:
    """Cohort selector.

    full_pr2=False (legacy): swap-positive only (PR-2 + rank-displacement candidate).
    full_pr2=True (per user 2026-05-24): full PR-2 cohort with stable_k=2, no swap
      filter — per-event geometry doesn't depend on swap evidence. Includes none-class
      subjects as fully analyzed (not just background control).
    """
    if not full_pr2:
        out: List[Tuple[str, str]] = []
        for p in sorted(PHASE2_PER_SUBJECT_DIR.glob("*.json")):
            stem = p.stem
            dataset, _, subject = stem.partition("_")
            out.append((dataset, subject))
        p2_stems = {p.stem for p in PHASE2_PER_SUBJECT_DIR.glob("*.json")}
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
            out.append((dataset, subject))
        return out

    # Full PR-2 cohort
    out: List[Tuple[str, str]] = []
    seen = set()
    for p in sorted(PHASE0A_DIR.glob("*.json")):
        if p.is_dir():
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("adaptive_cluster", {}).get("stable_k") != 2:
            continue
        stem = p.stem
        if stem in seen:
            continue
        seen.add(stem)
        dataset, _, subject = stem.partition("_")
        out.append((dataset, subject))
    return out


def _load_subject_swap_class(dataset: str, subject_id: str) -> Tuple[Optional[str], Optional[int], Optional[List[str]]]:
    """Returns (swap_class, decision_k, swap_k_channel_names_full_data)."""
    p = RANK_DISPLACEMENT_DIR / f"{dataset}_{subject_id}.json"
    if not p.exists():
        return None, None, None
    d = json.loads(p.read_text())
    pairs = d.get("pairs") or []
    if not pairs:
        return None, None, None
    ss = pairs[0].get("swap_sweep") or {}
    dk = ss.get("decision_k")
    sc = ss.get("swap_class")
    # Derive swap-k endpoint channels using derive_swap_endpoint
    from src.rank_displacement import derive_swap_endpoint
    rank_a_dense = np.asarray(pairs[0].get("rank_a_dense_full", []), dtype=float)
    ch_names = pairs[0].get("channel_names") or d.get("channel_names")
    swap_chs: Optional[List[str]] = None
    if rank_a_dense.size and ch_names and dk:
        try:
            swap_chs = derive_swap_endpoint(ch_names, rank_a_dense, int(dk))
        except (ValueError, KeyError):
            swap_chs = None
    return sc, dk, swap_chs


def _load_clinical_soz(dataset: str, subject_id: str) -> Optional[List[str]]:
    p = SOZ_YUQUAN if dataset == "yuquan" else SOZ_EPI
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get(str(subject_id))


def _run_subject(dataset: str, subject_id: str, seed: int = 0,
                 full_sweep: bool = False, skip_per_seizure: bool = False) -> Dict[str, Any]:
    stem = f"{dataset}_{subject_id}"
    out: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "trajectory_version": PHASE3_TRAJ_VERSION,
        "dataset": dataset, "subject_id": subject_id,
    }
    sc, dk, swap_chs_full = _load_subject_swap_class(dataset, subject_id)
    out["swap_class_full_data"] = sc
    out["global_decision_k"] = dk
    out["swap_k_endpoint_channels_full_data"] = swap_chs_full

    seizures = _load_seizure_times(dataset, subject_id)
    out["n_seizures_total"] = len(seizures)

    # Always compute electrode coverage even if no seizures (it's about anatomy, not events)
    try:
        phase0a = json.loads((PHASE0A_DIR / f"{dataset}_{subject_id}.json").read_text())
        channel_names: List[str] = list(phase0a["channel_names"])
        try:
            cr = load_subject_coords(dataset=dataset, subject_id=subject_id,
                                     channel_names_requested=channel_names)
            coords = cr.coords_array_in_requested_order
        except Exception:
            coords = None
        soz_chs = _load_clinical_soz(dataset, subject_id)
        out["electrode_coverage"] = compute_electrode_coverage(
            channel_names=channel_names, coords=coords,
            swap_k_channel_names=swap_chs_full or [],
            soz_channels=soz_chs,
        )
    except Exception as e:
        out["electrode_coverage_error"] = f"{type(e).__name__}:{e}"

    if not seizures:
        out["exit_reason"] = "no_seizures_in_inventory"
        out["per_seizure_trajectory"] = []
        return out

    # Load arrays
    try:
        labels_all = np.asarray(phase0a["adaptive_cluster"]["labels"], dtype=int)
        lagpat_dir = resolve_lagpat_subject_dir(dataset, subject_id)
        loaded = load_subject_propagation_events(lagpat_dir)
    except Exception as e:
        out["exit_reason"] = f"load_failed:{type(e).__name__}:{e}"
        return out
    raw_times = np.asarray(loaded["event_abs_times"], dtype=float)
    bools_full = np.asarray(loaded["bools"], dtype=int)
    ranks_full = np.asarray(loaded["ranks"], dtype=float)
    blocks: List[Tuple[float, float]] = _load_real_recording_block_ranges(dataset, subject_id)
    if not blocks:
        blocks = [tuple(tr) for tr in loaded["block_time_ranges"]]
    valid_idx = _valid_event_indices(bools_full, min_participating=3)
    if valid_idx.size != labels_all.size:
        out["exit_reason"] = "valid_event_count_mismatch"
        return out
    times = raw_times[valid_idx]
    finite_mask = np.isfinite(times) & (labels_all >= 0)
    event_abs_times = times[finite_mask]
    labels = labels_all[finite_mask]
    bools = bools_full[:, valid_idx][:, finite_mask].astype(bool)
    ranks = ranks_full[:, valid_idx][:, finite_mask]

    # Trajectory per seizure (skip if --skip-per-seizure)
    per_sz_traj: List[Dict[str, Any]] = []
    if skip_per_seizure:
        for sz in seizures:
            per_sz_traj.append({
                "seizure_id": sz["id"],
                "seizure_onset_t": sz["onset"],
                "seizure_offset_t": sz["offset"],
                "classification": sz.get("classification"),
                "n_trajectory_windows": 0,
                "trajectory": [],
            })
        out["per_seizure_trajectory"] = per_sz_traj
    else:
        for si, sz in enumerate(seizures):
            traj = compute_trajectory_for_seizure(
                seizure_onset_t=sz["onset"], seizure_offset_t=sz["offset"],
                event_abs_times=event_abs_times, labels=labels, ranks=ranks, bools=bools,
                coords=coords, channel_names=channel_names, blocks=blocks,
                seed=seed + si * 1000,
            )
            per_sz_traj.append({
                "seizure_id": sz["id"],
                "seizure_onset_t": sz["onset"],
                "seizure_offset_t": sz["offset"],
                "classification": sz.get("classification"),
                "n_trajectory_windows": len(traj),
                "trajectory": traj,
            })
        out["per_seizure_trajectory"] = per_sz_traj

    if full_sweep:
        seizure_onsets = [sz["onset"] for sz in seizures]
        try:
            fs = compute_trajectory_full_sweep(
                event_abs_times=event_abs_times, labels=labels, ranks=ranks, bools=bools,
                coords=coords, channel_names=channel_names,
                blocks=blocks, seizure_onsets=seizure_onsets,
                seed=seed + 50000,
            )
            out["full_sweep_trajectory"] = fs
            out["n_full_sweep_windows"] = len(fs)
        except Exception as e:
            out["full_sweep_error"] = f"{type(e).__name__}:{e}"

    out["exit_reason"] = "ok"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", help="dataset_subject e.g. epilepsiae_1146")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full-sweep", action="store_true",
                        help="Also compute full-recording sweep (not seizure-anchored)")
    parser.add_argument("--full-pr2-cohort", action="store_true",
                        help="Use full PR-2 cohort (stable_k=2, no swap filter)")
    parser.add_argument("--skip-per-seizure", action="store_true",
                        help="Skip per-seizure trajectory (only full sweep). Speeds up ~10x.")
    args = parser.parse_args()

    if args.subject:
        ds, _, sub = args.subject.partition("_")
        cohort = [(ds, sub)]
    elif args.all:
        cohort = _read_cohort(full_pr2=args.full_pr2_cohort)
    else:
        parser.error("provide --subject or --all")

    out_dir = OUT_DIR / "per_subject"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Phase3 v2] running {len(cohort)} subjects → {out_dir}")
    for ds, sub in cohort:
        print(f"[Phase3 v2] {ds}_{sub} ...", flush=True)
        t_start = time.time()
        try:
            rec = _run_subject(ds, sub, seed=args.seed, full_sweep=args.full_sweep,
                               skip_per_seizure=args.skip_per_seizure)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            continue
        out_path = out_dir / f"{ds}_{sub}.json"
        out_path.write_text(json.dumps(rec, indent=2, default=str))
        n_sz_with_traj = sum(1 for sz in (rec.get("per_seizure_trajectory") or [])
                              if sz.get("n_trajectory_windows", 0) > 0)
        elapsed = time.time() - t_start
        print(f"  → {out_path.name}: n_sz_total={rec.get('n_seizures_total')}, "
              f"n_sz_with_traj={n_sz_with_traj}, exit={rec.get('exit_reason')}, t={elapsed:.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
