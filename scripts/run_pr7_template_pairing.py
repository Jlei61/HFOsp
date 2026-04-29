#!/usr/bin/env python3
"""PR-7 Template Antagonistic Temporal Pairing runner.

Step 2 deliverable: cohort audit (`--audit`) + per-subject pairing/null
analysis (`--per-subject`).

Plan: docs/archive/topic1/pr7_template_antagonistic_pairing_plan_2026-04-28.md.
The audit reuses PR-6's `cohort_audit.csv` (forward_reverse_reproduced OR
rule) and augments with PR-7-specific eligibility fields:
  - n_events_total, n_T_a, n_T_b, min_cluster_n
  - n_blocks, total_coverage_hours, max_block_hours
  - 5 pre-registered eligibility flags + final cohort assignment

Usage:
    python scripts/run_pr7_template_pairing.py --audit
    python scripts/run_pr7_template_pairing.py --per-subject
    python scripts/run_pr7_template_pairing.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interictal_propagation import (  # noqa: E402
    _valid_event_indices,
    load_subject_propagation_events,
)
from src.template_temporal_pairing import (  # noqa: E402
    cohort_paired_test,
    compute_burst_diagnostic_with_nulls,
    compute_pairing_with_nulls,
    compute_transition_odds,
    evaluate_pass_criteria,
)


# ---------------------------------------------------------------------------
# Loader for *_lagPat_withFreqCent.npz (matches PR-2 adaptive_cluster JSONs)
# ---------------------------------------------------------------------------
def _record_name_from_freqcent(path: Path) -> str:
    """Strip suffix `_lagPat_withFreqCent` from filename stem."""
    stem = path.stem  # e.g. FC1047XY_lagPat_withFreqCent
    suffix = "_lagPat_withFreqCent"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def _safe_float_scalar(value: Any) -> float:
    try:
        return float(np.asarray(value).reshape(()).item())
    except Exception:  # noqa: BLE001
        return float("nan")


def _load_subject_with_freqcent(subject_dir: Path) -> Optional[Dict[str, Any]]:
    """Load `*_lagPat_withFreqCent.npz` blocks with matching packed times.

    PR-2 adaptive_cluster JSONs were produced from the FreqCent (10ch full)
    pipeline. `load_subject_propagation_events` in src/interictal_propagation.py
    loads the older 7ch `*_lagPat.npz`, which does NOT match stored cluster
    labels. PR-7 must use FreqCent to align with stored adaptive_cluster.labels.

    Returns dict with bools (n_ch, n_events), event_abs_times (n_events,),
    block_time_ranges (list of (start, end) tuples), or None if no FreqCent
    files present.
    """
    subject_dir = Path(subject_dir)
    files = sorted(subject_dir.glob("*_lagPat_withFreqCent.npz"))
    if not files:
        return None

    channel_names: List[str] = []
    channel_index: Dict[str, int] = {}
    block_records: List[Dict[str, Any]] = []

    for npz_path in files:
        try:
            lp = np.load(npz_path, allow_pickle=True)
        except Exception:  # noqa: BLE001
            continue
        if "eventsBool" not in lp.files or "chnNames" not in lp.files:
            continue
        bools = np.asarray(lp["eventsBool"]) > 0
        chns = [str(x) for x in list(lp["chnNames"])]
        start_t = _safe_float_scalar(lp["start_t"]) if "start_t" in lp.files else float("nan")
        if bools.ndim != 2 or bools.size == 0:
            continue
        n_ev = bools.shape[1]
        n_ch = min(bools.shape[0], len(chns))
        bools = bools[:n_ch, :n_ev]
        chns = chns[:n_ch]

        record = _record_name_from_freqcent(npz_path)
        packed_path = npz_path.with_name(f"{record}_packedTimes_withFreqCent.npy")
        event_rel_times = np.full(n_ev, np.nan, dtype=float)
        event_abs_times = np.full(n_ev, np.nan, dtype=float)
        if packed_path.exists():
            packed = np.asarray(np.load(packed_path), dtype=float)
            if packed.ndim == 2 and packed.shape[1] >= 1:
                m = min(n_ev, packed.shape[0])
                bools = bools[:, :m]
                event_rel_times = packed[:m, 0].astype(float, copy=False)
                if np.isfinite(start_t):
                    event_abs_times = event_rel_times + start_t
                n_ev = m

        for ch in chns:
            if ch not in channel_index:
                channel_index[ch] = len(channel_names)
                channel_names.append(ch)

        block_records.append(
            {
                "record": record,
                "start_t": start_t,
                "chns": chns,
                "bools": bools,
                "event_rel_times": event_rel_times,
                "event_abs_times": event_abs_times,
            }
        )

    if not block_records:
        return None

    block_records.sort(
        key=lambda r: (r["start_t"] if np.isfinite(r["start_t"]) else float("inf"),)
    )

    n_ch_total = len(channel_names)
    bools_blocks: List[np.ndarray] = []
    times_blocks: List[np.ndarray] = []
    block_time_ranges: List[Tuple[float, float]] = []

    for r in block_records:
        big = np.zeros((n_ch_total, r["bools"].shape[1]), dtype=bool)
        for src_idx, ch in enumerate(r["chns"]):
            big[channel_index[ch], :] = r["bools"][src_idx, :]
        bools_blocks.append(big)
        times_blocks.append(r["event_abs_times"])
        finite = r["event_abs_times"][np.isfinite(r["event_abs_times"])]
        if finite.size:
            block_time_ranges.append((float(finite.min()), float(finite.max())))
        else:
            block_time_ranges.append((float("nan"), float("nan")))

    return {
        "bools": np.concatenate(bools_blocks, axis=1),
        "event_abs_times": np.concatenate(times_blocks),
        "block_time_ranges": block_time_ranges,
        "channel_names": channel_names,
        "n_blocks": len(block_records),
    }


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PER_SUBJECT_DIR = ROOT / "results" / "interictal_propagation" / "per_subject"
PR6_AUDIT_CSV = (
    ROOT
    / "results"
    / "interictal_propagation"
    / "template_anchoring"
    / "cohort_audit.csv"
)

YUQUAN_RAW_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_RAW_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")

OUT_DIR = ROOT / "results" / "interictal_propagation" / "template_pairing"
PER_SUBJECT_OUT = OUT_DIR / "per_subject"
PER_SUBJECT_BURST_OUT = OUT_DIR / "per_subject_burst"
AUDIT_CSV = OUT_DIR / "pr7_cohort_audit.csv"
COHORT_SUMMARY_JSON = OUT_DIR / "cohort_summary.json"


# ---------------------------------------------------------------------------
# Pre-registered eligibility thresholds (plan §5; do NOT relax)
# ---------------------------------------------------------------------------
MIN_N_EVENTS_TOTAL = 300
MIN_CLUSTER_COUNT = 75
MIN_N_BLOCKS = 3
MIN_COVERAGE_HOURS = 6.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_subject_dir(dataset: str, subject_id: str) -> Optional[Path]:
    if dataset == "yuquan":
        cand = YUQUAN_RAW_ROOT / subject_id
        return cand if cand.exists() else None
    if dataset == "epilepsiae":
        cand = EPILEPSIAE_RAW_ROOT / subject_id / "all_recs"
        return cand if cand.exists() else None
    return None


def _per_subject_json_path(dataset: str, subject_id: str) -> Path:
    return PER_SUBJECT_DIR / f"{dataset}_{subject_id}.json"


def _load_pr6_audit_rows() -> List[Dict[str, str]]:
    if not PR6_AUDIT_CSV.exists():
        raise FileNotFoundError(
            f"PR-6 cohort_audit.csv not found at {PR6_AUDIT_CSV}; "
            "run scripts/run_pr6_template_anchoring.py --audit first."
        )
    with PR6_AUDIT_CSV.open() as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _bool_from_pr6(value: str) -> bool:
    return value.strip().lower() == "true"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coverage_stats(
    block_time_ranges: List[Tuple[float, float]],
) -> Tuple[int, float, float]:
    """Return (n_blocks, total_coverage_hours, max_block_hours)."""
    if not block_time_ranges:
        return 0, 0.0, 0.0
    durations_hours = [
        (b_end - b_start) / 3600.0 for (b_start, b_end) in block_time_ranges
    ]
    return (
        len(block_time_ranges),
        float(sum(durations_hours)),
        float(max(durations_hours)),
    )


# ---------------------------------------------------------------------------
# Audit row construction
# ---------------------------------------------------------------------------
AUDIT_FIELDS = [
    "subject_id",
    "dataset",
    "stable_k",
    "n_ch",
    "endpoint_defined_pr6",
    "forward_reverse_reproduced",
    "n_events_total",
    "n_T_a",
    "n_T_b",
    "min_cluster_n",
    "n_blocks",
    "total_coverage_hours",
    "max_block_hours",
    "cond1_endpoint_defined",
    "cond2_fwdrev_reproduced",
    "cond3_n_events_ge_300",
    "cond4_min_cluster_ge_75",
    "cond5_blocks_or_coverage_ok",
    "exit_reason",
    "h1_primary_pass",
    "h2_negative_pass",
    "case_series_eligible",
]


def _audit_one_subject(pr6_row: Dict[str, str]) -> Dict[str, Any]:
    subject_id = pr6_row.get("subject_id", "").strip()
    dataset = pr6_row.get("dataset", "").strip()
    stable_k = _safe_int(pr6_row.get("stable_k"), 0)
    n_ch = _safe_int(pr6_row.get("n_ch"), 0)
    endpoint_defined_pr6 = _bool_from_pr6(pr6_row.get("endpoint_defined", "False"))
    fwdrev = _bool_from_pr6(pr6_row.get("forward_reverse_reproduced", "False"))

    row: Dict[str, Any] = {
        "subject_id": subject_id,
        "dataset": dataset,
        "stable_k": stable_k,
        "n_ch": n_ch,
        "endpoint_defined_pr6": endpoint_defined_pr6,
        "forward_reverse_reproduced": fwdrev,
        "n_events_total": 0,
        "n_T_a": 0,
        "n_T_b": 0,
        "min_cluster_n": 0,
        "n_blocks": 0,
        "total_coverage_hours": 0.0,
        "max_block_hours": 0.0,
        "cond1_endpoint_defined": endpoint_defined_pr6,
        "cond2_fwdrev_reproduced": fwdrev,
        "cond3_n_events_ge_300": False,
        "cond4_min_cluster_ge_75": False,
        "cond5_blocks_or_coverage_ok": False,
        "exit_reason": "",
        "h1_primary_pass": False,
        "h2_negative_pass": False,
        "case_series_eligible": False,
    }

    if not endpoint_defined_pr6:
        row["exit_reason"] = "endpoint_undefined_pr6"
        return row
    if stable_k != 2:
        row["exit_reason"] = "k!=2"
        return row

    json_path = _per_subject_json_path(dataset, subject_id)
    if not json_path.exists():
        row["exit_reason"] = "missing_per_subject_json"
        return row

    with json_path.open() as fh:
        psj = json.load(fh)

    labels = np.asarray(
        psj.get("adaptive_cluster", {}).get("labels", []), dtype=int
    )
    n_events_total = int(labels.size)
    n_T_a = int(np.sum(labels == 0))
    n_T_b = int(np.sum(labels == 1))
    min_cluster_n = min(n_T_a, n_T_b) if n_events_total > 0 else 0

    row.update(
        {
            "n_events_total": n_events_total,
            "n_T_a": n_T_a,
            "n_T_b": n_T_b,
            "min_cluster_n": min_cluster_n,
        }
    )

    subject_dir = _resolve_subject_dir(dataset, subject_id)
    if subject_dir is None:
        row["exit_reason"] = "raw_data_root_unmounted"
        return row
    try:
        loaded = load_subject_propagation_events(subject_dir)
    except Exception as exc:  # noqa: BLE001
        row["exit_reason"] = f"load_failed:{type(exc).__name__}"
        return row

    block_time_ranges: List[Tuple[float, float]] = [
        tuple(b) for b in loaded.get("block_time_ranges", [])
    ]
    n_blocks, coverage_hours, max_block_hours = _coverage_stats(block_time_ranges)
    row.update(
        {
            "n_blocks": n_blocks,
            "total_coverage_hours": coverage_hours,
            "max_block_hours": max_block_hours,
        }
    )

    cond3 = n_events_total >= MIN_N_EVENTS_TOTAL
    cond4 = min_cluster_n >= MIN_CLUSTER_COUNT
    cond5 = (n_blocks >= MIN_N_BLOCKS) or (coverage_hours >= MIN_COVERAGE_HOURS)
    row.update(
        {
            "cond3_n_events_ge_300": cond3,
            "cond4_min_cluster_ge_75": cond4,
            "cond5_blocks_or_coverage_ok": cond5,
        }
    )

    size_block_pass = cond3 and cond4 and cond5
    h1_pass = endpoint_defined_pr6 and fwdrev and size_block_pass
    h2_pass = endpoint_defined_pr6 and (not fwdrev) and size_block_pass
    case_series = endpoint_defined_pr6 and (not size_block_pass)

    row.update(
        {
            "h1_primary_pass": h1_pass,
            "h2_negative_pass": h2_pass,
            "case_series_eligible": case_series,
        }
    )

    if not size_block_pass:
        reasons = []
        if not cond3:
            reasons.append(f"n_events<{MIN_N_EVENTS_TOTAL}")
        if not cond4:
            reasons.append(f"min_cluster<{MIN_CLUSTER_COUNT}")
        if not cond5:
            reasons.append(
                f"n_blocks<{MIN_N_BLOCKS}_AND_coverage<{MIN_COVERAGE_HOURS}h"
            )
        row["exit_reason"] = ";".join(reasons)
    elif not (h1_pass or h2_pass):
        row["exit_reason"] = "endpoint_undefined_for_main_cohorts"

    return row


# ---------------------------------------------------------------------------
# Audit driver
# ---------------------------------------------------------------------------
def run_audit() -> List[Dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pr6_rows = _load_pr6_audit_rows()
    print(f"Loaded {len(pr6_rows)} PR-6 audit rows from {PR6_AUDIT_CSV}")

    audit_rows: List[Dict[str, Any]] = []
    for pr6_row in pr6_rows:
        result = _audit_one_subject(pr6_row)
        audit_rows.append(result)

    with AUDIT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=AUDIT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in audit_rows:
            writer.writerow(r)

    n_h1 = sum(1 for r in audit_rows if r["h1_primary_pass"])
    n_h2 = sum(1 for r in audit_rows if r["h2_negative_pass"])
    n_case = sum(1 for r in audit_rows if r["case_series_eligible"])
    print(f"\nWrote {AUDIT_CSV} ({len(audit_rows)} rows)")
    print(f"  H1 primary cohort (fwd/rev reproduced + 5 conds): n={n_h1}")
    print(f"  H2 negative cohort (non-fwdrev + 5 conds):        n={n_h2}")
    print(f"  Case-series eligible (endpoint_defined no size):  n={n_case}")

    if n_h1 < 4:
        print(
            "\nWARNING: H1 primary cohort < 4 — per plan §9.7, fall back to "
            "case-series; do NOT relax thresholds to inflate cohort."
        )
    return audit_rows


# ---------------------------------------------------------------------------
# Per-subject driver
# ---------------------------------------------------------------------------
DEFAULT_DELTA_T_GRID: Tuple[float, ...] = (1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 1800.0, 3600.0)
DEFAULT_NULLS: Tuple[str, ...] = ("N0", "N1", "N2", "N3")


def _run_one_subject_pairing(
    subject_id: str,
    dataset: str,
    n_perm: int,
    n2_window_seconds: float,
    delta_t_grid: Tuple[float, ...],
    nulls: Tuple[str, ...],
    seed: int,
) -> Optional[Dict[str, Any]]:
    json_path = _per_subject_json_path(dataset, subject_id)
    if not json_path.exists():
        print(f"  [skip] {dataset}/{subject_id}: missing per-subject JSON")
        return None

    with json_path.open() as fh:
        psj = json.load(fh)
    labels = np.asarray(
        psj.get("adaptive_cluster", {}).get("labels", []), dtype=int
    )

    subject_dir = _resolve_subject_dir(dataset, subject_id)
    if subject_dir is None:
        print(f"  [skip] {dataset}/{subject_id}: raw data root unmounted")
        return None
    # Use FreqCent loader (matches PR-2 adaptive_cluster JSONs); the older
    # `*_lagPat.npz` 7ch slice does NOT align with stored labels.
    loaded = _load_subject_with_freqcent(subject_dir)
    if loaded is None:
        print(f"  [skip] {dataset}/{subject_id}: no *_lagPat_withFreqCent.npz")
        return None
    raw_times = np.asarray(loaded["event_abs_times"], dtype=float)
    bools = np.asarray(loaded["bools"], dtype=int)
    block_time_ranges = [tuple(b) for b in loaded.get("block_time_ranges", [])]

    # PR-2 cluster labels correspond to events surviving _valid_event_indices
    # (default min_participating=3). Align by applying the same filter to
    # raw event_abs_times before joining with labels.
    valid_idx = _valid_event_indices(bools, min_participating=3)
    if valid_idx.size != labels.size:
        print(
            f"  [skip] {dataset}/{subject_id}: valid_event_count {valid_idx.size} "
            f"!= label count {labels.size}"
        )
        return None
    times = raw_times[valid_idx]

    finite_mask = np.isfinite(times) & (labels >= 0)
    times = times[finite_mask]
    labels = labels[finite_mask]
    if times.size == 0:
        print(f"  [skip] {dataset}/{subject_id}: no finite events after filter")
        return None

    t0 = time.time()
    pairing = compute_pairing_with_nulls(
        event_abs_times=times,
        cluster_labels=labels,
        block_time_ranges=block_time_ranges,
        delta_t_grid=delta_t_grid,
        n_perm=n_perm,
        nulls=nulls,
        n2_window_seconds=n2_window_seconds,
        seed=seed,
    )
    transition = compute_transition_odds(times, labels, block_time_ranges)
    elapsed = time.time() - t0

    # JSON-serialize: convert numpy ints/floats and inner lists
    def _to_jsonable(x: Any) -> Any:
        if isinstance(x, dict):
            return {str(k): _to_jsonable(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_to_jsonable(v) for v in x]
        if isinstance(x, tuple):
            return [_to_jsonable(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        return x

    out_record = {
        "subject_id": subject_id,
        "dataset": dataset,
        "n_events_used": int(times.size),
        "n_T_a": int(np.sum(labels == 0)),
        "n_T_b": int(np.sum(labels == 1)),
        "n_blocks": len(block_time_ranges),
        "delta_t_grid": list(delta_t_grid),
        "nulls_run": list(nulls),
        "n_perm": int(n_perm),
        "n2_window_seconds": float(n2_window_seconds),
        "seed": int(seed),
        "pairing_with_nulls": _to_jsonable(pairing),
        "transition_odds": _to_jsonable(transition),
        "elapsed_seconds": float(elapsed),
    }

    PER_SUBJECT_OUT.mkdir(parents=True, exist_ok=True)
    out_path = PER_SUBJECT_OUT / f"{dataset}_{subject_id}.json"
    with out_path.open("w") as fh:
        json.dump(out_record, fh, indent=2)
    print(f"  [done] {dataset}/{subject_id} in {elapsed:.1f}s -> {out_path.name}")
    return out_record


def run_per_subject(
    cohort: str = "h1_primary",
    n_perm: int = 1000,
    n2_window_seconds: float = 1800.0,
    delta_t_grid: Tuple[float, ...] = DEFAULT_DELTA_T_GRID,
    nulls: Tuple[str, ...] = DEFAULT_NULLS,
    seed: int = 0,
    only: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Run pairing+nulls for every subject passing the chosen cohort flag.

    cohort ∈ {'h1_primary', 'h2_negative', 'all_eligible'}.
    'all_eligible' = h1 ∪ h2 (cohort runs both for cohort_summary).
    """
    if not AUDIT_CSV.exists():
        raise FileNotFoundError(
            f"PR-7 audit CSV not found at {AUDIT_CSV}; run --audit first."
        )

    with AUDIT_CSV.open() as fh:
        rows = list(csv.DictReader(fh))

    def _is_target(row: Dict[str, str]) -> bool:
        if cohort == "h1_primary":
            return row.get("h1_primary_pass") == "True"
        if cohort == "h2_negative":
            return row.get("h2_negative_pass") == "True"
        if cohort == "all_eligible":
            return (
                row.get("h1_primary_pass") == "True"
                or row.get("h2_negative_pass") == "True"
            )
        raise ValueError(f"Unknown cohort: {cohort}")

    targets = [r for r in rows if _is_target(r)]
    if only:
        only_set = set(only)
        targets = [r for r in targets if r["subject_id"] in only_set]

    print(
        f"Running pairing+nulls for cohort='{cohort}' on {len(targets)} subjects"
        f" (n_perm={n_perm}, n2_window={n2_window_seconds}s, "
        f"|Δt|={len(delta_t_grid)}, nulls={nulls})"
    )
    results: List[Dict[str, Any]] = []
    for r in targets:
        rec = _run_one_subject_pairing(
            subject_id=r["subject_id"],
            dataset=r["dataset"],
            n_perm=n_perm,
            n2_window_seconds=n2_window_seconds,
            delta_t_grid=delta_t_grid,
            nulls=nulls,
            seed=seed,
        )
        if rec is not None:
            results.append(rec)
    return results


# ---------------------------------------------------------------------------
# Step 3.5 burst-level diagnostic — per-subject + cohort aggregation
# ---------------------------------------------------------------------------
def _run_one_subject_burst(
    subject_id: str,
    dataset: str,
    n_perm: int,
    n2_window_seconds: float,
    nulls: Tuple[str, ...],
    seed: int,
) -> Optional[Dict[str, Any]]:
    json_path = _per_subject_json_path(dataset, subject_id)
    if not json_path.exists():
        print(f"  [skip] {dataset}/{subject_id}: missing per-subject JSON")
        return None
    with json_path.open() as fh:
        psj = json.load(fh)
    labels = np.asarray(
        psj.get("adaptive_cluster", {}).get("labels", []), dtype=int
    )

    subject_dir = _resolve_subject_dir(dataset, subject_id)
    if subject_dir is None:
        print(f"  [skip] {dataset}/{subject_id}: raw data root unmounted")
        return None
    loaded = _load_subject_with_freqcent(subject_dir)
    if loaded is None:
        print(f"  [skip] {dataset}/{subject_id}: no *_lagPat_withFreqCent.npz")
        return None
    raw_times = np.asarray(loaded["event_abs_times"], dtype=float)
    bools = np.asarray(loaded["bools"], dtype=int)
    block_time_ranges = [tuple(b) for b in loaded.get("block_time_ranges", [])]

    valid_idx = _valid_event_indices(bools, min_participating=3)
    if valid_idx.size != labels.size:
        print(
            f"  [skip] {dataset}/{subject_id}: valid_event_count {valid_idx.size} "
            f"!= label count {labels.size}"
        )
        return None
    times = raw_times[valid_idx]
    finite_mask = np.isfinite(times) & (labels >= 0)
    times = times[finite_mask]
    labels = labels[finite_mask]
    if times.size == 0:
        print(f"  [skip] {dataset}/{subject_id}: no finite events after filter")
        return None

    t0 = time.time()
    diag = compute_burst_diagnostic_with_nulls(
        event_abs_times=times,
        cluster_labels=labels,
        block_time_ranges=block_time_ranges,
        n_perm=n_perm,
        nulls=nulls,
        n2_window_seconds=n2_window_seconds,
        seed=seed,
    )
    elapsed = time.time() - t0

    def _to_jsonable(x: Any) -> Any:
        if isinstance(x, dict):
            return {str(k): _to_jsonable(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_to_jsonable(v) for v in x]
        if isinstance(x, tuple):
            return [_to_jsonable(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        return x

    record = {
        "subject_id": subject_id,
        "dataset": dataset,
        "n_events_used": int(times.size),
        "n_T_a": int(np.sum(labels == 0)),
        "n_T_b": int(np.sum(labels == 1)),
        "n_blocks": len(block_time_ranges),
        "nulls_run": list(nulls),
        "n_perm": int(n_perm),
        "n2_window_seconds": float(n2_window_seconds),
        "seed": int(seed),
        "burst_diagnostic": _to_jsonable(diag),
        "elapsed_seconds": float(elapsed),
    }

    PER_SUBJECT_BURST_OUT.mkdir(parents=True, exist_ok=True)
    out = PER_SUBJECT_BURST_OUT / f"{dataset}_{subject_id}.json"
    with out.open("w") as fh:
        json.dump(record, fh, indent=2)
    rll_n2 = diag["lift"].get("N2", {}).get("run_length_lift", float("nan"))
    git_n2 = diag["lift"].get("N2", {}).get("gap_to_iei_lift", float("nan"))
    lag1_n2 = diag["lag1_same_excess"].get("N2", float("nan"))
    print(
        f"  [done] {dataset}/{subject_id} in {elapsed:.1f}s "
        f"(N2: rll={rll_n2:.3f}, gap_iei={git_n2:.3f}, lag1_excess={lag1_n2:+.4f})"
    )
    return record


def run_burst_diagnostic(
    cohort: str = "h1_primary",
    n_perm: int = 500,
    n2_window_seconds: float = 1800.0,
    nulls: Tuple[str, ...] = ("N1", "N2"),
    seed: int = 0,
    only: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if not AUDIT_CSV.exists():
        raise FileNotFoundError(
            f"PR-7 audit CSV not found at {AUDIT_CSV}; run --audit first."
        )

    with AUDIT_CSV.open() as fh:
        rows = list(csv.DictReader(fh))

    def _is_target(row: Dict[str, str]) -> bool:
        if cohort == "h1_primary":
            return row.get("h1_primary_pass") == "True"
        if cohort == "h2_negative":
            return row.get("h2_negative_pass") == "True"
        if cohort == "all_eligible":
            return (
                row.get("h1_primary_pass") == "True"
                or row.get("h2_negative_pass") == "True"
            )
        raise ValueError(f"Unknown cohort: {cohort}")

    targets = [r for r in rows if _is_target(r)]
    if only:
        only_set = set(only)
        targets = [r for r in targets if r["subject_id"] in only_set]

    print(
        f"Burst diagnostic for cohort='{cohort}' on {len(targets)} subjects"
        f" (n_perm={n_perm}, nulls={nulls}, n2_window={n2_window_seconds}s)"
    )
    results: List[Dict[str, Any]] = []
    for r in targets:
        rec = _run_one_subject_burst(
            subject_id=r["subject_id"],
            dataset=r["dataset"],
            n_perm=n_perm,
            n2_window_seconds=n2_window_seconds,
            nulls=nulls,
            seed=seed,
        )
        if rec is not None:
            results.append(rec)
    return results


# ---------------------------------------------------------------------------
# Cohort aggregation (Step 3 deliverable)
# ---------------------------------------------------------------------------
def _classify_subjects_by_cohort() -> Dict[str, List[Tuple[str, str]]]:
    """Read pr7_cohort_audit.csv and return mapping cohort -> [(dataset, subject_id)]."""
    if not AUDIT_CSV.exists():
        raise FileNotFoundError(f"Audit not found at {AUDIT_CSV}; run --audit first")
    with AUDIT_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    out = {"h1_primary": [], "h2_negative": []}
    for r in rows:
        ds = r["dataset"]
        sid = r["subject_id"]
        if r.get("h1_primary_pass") == "True":
            out["h1_primary"].append((ds, sid))
        if r.get("h2_negative_pass") == "True":
            out["h2_negative"].append((ds, sid))
    return out


def _load_per_subject_record(dataset: str, subject_id: str) -> Optional[Dict[str, Any]]:
    p = PER_SUBJECT_OUT / f"{dataset}_{subject_id}.json"
    if not p.exists():
        return None
    with p.open() as fh:
        return json.load(fh)


def _gather_excess_per_subject(
    members: List[Tuple[str, str]],
    null_id: str,
    delta_t: float,
    field: str = "excess",
) -> Dict[str, float]:
    """Read per-subject JSONs and pull lift[null_id][Δt][field] keyed by subject."""
    dt_str = f"{delta_t}"
    excess: Dict[str, float] = {}
    for ds, sid in members:
        rec = _load_per_subject_record(ds, sid)
        if rec is None:
            continue
        try:
            v = rec["pairing_with_nulls"]["lift"][null_id][dt_str][field]
        except KeyError:
            continue
        excess[f"{ds}_{sid}"] = float(v)
    return excess


def _gather_transition_per_subject(
    members: List[Tuple[str, str]],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for ds, sid in members:
        rec = _load_per_subject_record(ds, sid)
        if rec is None:
            continue
        out[f"{ds}_{sid}"] = rec.get("transition_odds", {})
    return out


def aggregate_cohort(
    delta_t_grid: Tuple[float, ...] = DEFAULT_DELTA_T_GRID,
    nulls: Tuple[str, ...] = DEFAULT_NULLS,
) -> Dict[str, Any]:
    """Build cohort_summary.json for H1 primary + H2 negative cohorts.

    For each cohort × null × Δt, compute:
      - paired Wilcoxon (greater) p, sign-test p, median, n
      - per-subject excess values (key-aligned)

    For H1 primary, additionally evaluate the triple-gate PASS criteria
    (10s primary + 30s sensitivity).
    """
    cohorts = _classify_subjects_by_cohort()
    summary: Dict[str, Any] = {
        "delta_t_grid_seconds": list(delta_t_grid),
        "nulls": list(nulls),
        "cohort_membership": {
            cohort: [{"dataset": ds, "subject_id": sid} for ds, sid in members]
            for cohort, members in cohorts.items()
        },
        "cohorts": {},
    }

    for cohort_name, members in cohorts.items():
        if not members:
            summary["cohorts"][cohort_name] = {
                "n_members": 0,
                "skipped": "empty cohort",
            }
            continue

        per_null: Dict[str, Any] = {}
        for null_id in nulls:
            per_dt: Dict[str, Any] = {}
            for dt in delta_t_grid:
                excess = _gather_excess_per_subject(members, null_id, dt, "excess")
                a_to_b = _gather_excess_per_subject(members, null_id, dt, "a_to_b_lift")
                b_to_a = _gather_excess_per_subject(members, null_id, dt, "b_to_a_lift")
                wilc = cohort_paired_test(excess, alternative="greater")
                per_dt[f"{dt}"] = {
                    "n_subjects": len(excess),
                    "excess_per_subject": excess,
                    "median_excess": wilc["median"],
                    "wilcoxon_greater_p": wilc["wilcoxon_p"],
                    "sign_test_greater_p": wilc["sign_test_p"],
                    "a_to_b_lift_per_subject": a_to_b,
                    "b_to_a_lift_per_subject": b_to_a,
                }
            per_null[null_id] = per_dt
        cohort_block: Dict[str, Any] = {
            "n_members": len(members),
            "by_null": per_null,
            "transition_odds_per_subject": _gather_transition_per_subject(members),
        }

        # Triple-gate PASS for H1 primary cohort (N2 main null + N3 robustness)
        if cohort_name == "h1_primary":
            triple_gate: Dict[str, Any] = {}
            for null_id in ("N2", "N3"):
                e10 = _gather_excess_per_subject(members, null_id, 10.0, "excess")
                e30 = _gather_excess_per_subject(members, null_id, 30.0, "excess")
                if not e10 or not e30:
                    triple_gate[null_id] = {"skipped": "missing per-subject data"}
                    continue
                triple_gate[null_id] = evaluate_pass_criteria(e10, e30)
            cohort_block["triple_gate_pass"] = triple_gate

        summary["cohorts"][cohort_name] = cohort_block

    # Step 3.5 burst-level diagnostic block (per-cohort), if available
    summary["burst_diagnostic_per_cohort"] = {}
    for cohort_name, members in cohorts.items():
        if not members:
            continue
        per_subject_burst: Dict[str, Dict[str, Any]] = {}
        for ds, sid in members:
            p = PER_SUBJECT_BURST_OUT / f"{ds}_{sid}.json"
            if not p.exists():
                continue
            with p.open() as fh:
                rec = json.load(fh)
            per_subject_burst[f"{ds}_{sid}"] = {
                "empirical": rec["burst_diagnostic"]["empirical"],
                "lift": rec["burst_diagnostic"]["lift"],
                "lag1_same_excess": rec["burst_diagnostic"]["lag1_same_excess"],
                "n_events_used": rec["n_events_used"],
            }
        if not per_subject_burst:
            continue
        # cohort-level summary by null
        cohort_burst_summary: Dict[str, Any] = {
            "n_with_burst_data": len(per_subject_burst),
            "per_subject": per_subject_burst,
            "by_null": {},
        }
        for null_id in ("N1", "N2"):
            rll = {
                k: v["lift"].get(null_id, {}).get("run_length_lift", float("nan"))
                for k, v in per_subject_burst.items()
            }
            git = {
                k: v["lift"].get(null_id, {}).get("gap_to_iei_lift", float("nan"))
                for k, v in per_subject_burst.items()
            }
            l1ex = {
                k: v["lag1_same_excess"].get(null_id, float("nan"))
                for k, v in per_subject_burst.items()
            }
            rll_arr = np.asarray([v for v in rll.values() if np.isfinite(v)])
            git_arr = np.asarray([v for v in git.values() if np.isfinite(v)])
            l1_arr = np.asarray([v for v in l1ex.values() if np.isfinite(v)])
            cohort_burst_summary["by_null"][null_id] = {
                "run_length_lift": rll,
                "gap_to_iei_lift": git,
                "lag1_same_excess": l1ex,
                "n_subjects": int(rll_arr.size),
                "median_run_length_lift": (
                    float(np.median(rll_arr)) if rll_arr.size else float("nan")
                ),
                "n_subjects_run_length_lift_gt_1": int(np.sum(rll_arr > 1.0)),
                "median_gap_to_iei_lift": (
                    float(np.median(git_arr)) if git_arr.size else float("nan")
                ),
                "n_subjects_gap_to_iei_lift_gt_1": int(np.sum(git_arr > 1.0)),
                "median_lag1_same_excess": (
                    float(np.median(l1_arr)) if l1_arr.size else float("nan")
                ),
                "n_subjects_lag1_excess_positive": int(np.sum(l1_arr > 0.0)),
            }
        summary["burst_diagnostic_per_cohort"][cohort_name] = cohort_burst_summary

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with COHORT_SUMMARY_JSON.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote {COHORT_SUMMARY_JSON}")

    for cohort_name, block in summary["cohorts"].items():
        if block.get("skipped"):
            print(f"  [{cohort_name}] {block['skipped']}")
            continue
        print(f"  [{cohort_name}] n={block['n_members']}")
        if "triple_gate_pass" in block:
            for null_id, gate in block["triple_gate_pass"].items():
                if gate.get("skipped"):
                    print(f"    {null_id} triple-gate: {gate['skipped']}")
                    continue
                print(
                    f"    {null_id} triple-gate: PASS={gate['pass']} | "
                    f"wilc(10s)={gate['wilcoxon_10s']:.4f}, "
                    f"sign(10s)={gate['sign_10s']:.4f}, "
                    f"med(10s)={gate['median_10s']:+.4f}, "
                    f"med(30s)={gate['median_30s']:+.4f}"
                )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="PR-7 Template Pairing runner")
    parser.add_argument("--audit", action="store_true", help="Run cohort audit")
    parser.add_argument(
        "--per-subject", action="store_true", help="Run per-subject pairing+nulls"
    )
    parser.add_argument(
        "--cohort-stats",
        action="store_true",
        help="Aggregate per-subject JSONs -> cohort_summary.json + triple-gate",
    )
    parser.add_argument(
        "--burst-diagnostic",
        action="store_true",
        help="Step 3.5 burst-level diagnostic on a cohort (post-hoc exploratory)",
    )
    parser.add_argument("--all", action="store_true", help="Audit then per-subject")
    parser.add_argument(
        "--cohort",
        type=str,
        default="all_eligible",
        choices=["h1_primary", "h2_negative", "all_eligible"],
        help="Which cohort to run for --per-subject (default: all_eligible)",
    )
    parser.add_argument(
        "--n-perm", type=int, default=1000,
        help="Permutations for --per-subject (event-level pairing). Default 1000.",
    )
    parser.add_argument(
        "--burst-n-perm", type=int, default=500,
        help="Permutations for --burst-diagnostic (run/lag1 metrics). Default 500.",
    )
    parser.add_argument("--n2-window-min", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated subject_ids to restrict --per-subject to",
    )
    args = parser.parse_args()

    if not (
        args.audit
        or args.per_subject
        or args.cohort_stats
        or args.burst_diagnostic
        or args.all
    ):
        parser.error(
            "Must specify one of --audit / --per-subject / --cohort-stats "
            "/ --burst-diagnostic / --all"
        )

    if args.audit or args.all:
        run_audit()

    if args.per_subject or args.all:
        only_list = (
            [s.strip() for s in args.only.split(",")] if args.only else None
        )
        run_per_subject(
            cohort=args.cohort,
            n_perm=args.n_perm,
            n2_window_seconds=args.n2_window_min * 60.0,
            seed=args.seed,
            only=only_list,
        )

    if args.burst_diagnostic:
        only_list = (
            [s.strip() for s in args.only.split(",")] if args.only else None
        )
        run_burst_diagnostic(
            cohort=args.cohort,
            n_perm=args.burst_n_perm,
            n2_window_seconds=args.n2_window_min * 60.0,
            seed=args.seed,
            only=only_list,
        )

    if args.cohort_stats or args.all:
        aggregate_cohort()


if __name__ == "__main__":
    main()
