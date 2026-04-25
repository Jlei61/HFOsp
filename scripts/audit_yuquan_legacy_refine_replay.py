"""Track B — Yuquan legacy-refine replay comparator.

For every record produced by `scripts/run_yuquan_legacy_refine_replay.py`
under `<out_root>/<subject>/`, pair the replay output against the legacy
ground truth at `<DATA_ROOT>/<subject>/.legacy_backup/<stem>_*` and
compute per-record / per-subject / cohort numerical equivalence metrics.

Two-axis alignment:
  rows    by `chnNames` — required exact set match; new arrays are
          row-permuted to the legacy order. If sets differ, the record
          fails with `chn_set_mismatch` and arrays are NOT diffed.
  columns by `packedTimes` — 1:1 nearest-window match within
          `pack_match_tol_ms` (default = pack_win_sec/2 = 150 ms).
          Unmatched columns counted; `eventsBool` / `lagPatRank` /
          `lagPatRaw` only diffed on aligned indices.

Provenance gate: `verdict_record` refuses to verdict any record whose
manifest reports a `gpu_npz_used` or `refine_npz_used` under
`DETECT_ROOT` — that would be a same-source run masquerading as a
legacy-refine replay.

Cohort verdict: `pack_lag_parity_pass` iff over all replayed subjects /
records (passing the provenance gate):
  - `chnNames` exact (after row-permute)
  - `packedTimes` aligned columns 100% (zero unmatched on both sides)
  - max column edge delta ≤ packed tolerance
  - `eventsBool` exact
  - `lagPatRank` exact (finite columns)
  - `lagPatRaw` maxabs ≤ lag_raw tolerance

Default tolerances are intentionally tight; the `gaolan` preflight
locks the production tolerance via `--lag-raw-tol-sec` /
`--packed-tol-sec` overrides.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_yuquan_lagpat_backfill import (  # noqa: E402
    DATA_ROOT,
    DETECT_ROOT,
    YUQUAN_SAME_SOURCE_SUBJECTS,
)


# ---------------------------------------------------------------------------
# Default tolerances — overridden by the preflight on gaolan via CLI.
# ---------------------------------------------------------------------------

DEFAULT_PACK_MATCH_TOL_MS = 150.0      # 0.3 s pack_win_sec / 2
DEFAULT_PACKED_TOL_SEC = 1.25e-3       # 1 sample at 800 Hz
DEFAULT_LAGRAW_TOL_SEC = 1e-9          # tight FP default; preflight may relax


# ---------------------------------------------------------------------------
# Comparator — two-axis alignment, no naive shape diff.
# ---------------------------------------------------------------------------


def _to_dict(npz_or_dict) -> Dict[str, np.ndarray]:
    if isinstance(npz_or_dict, dict):
        return npz_or_dict
    if isinstance(npz_or_dict, (str, Path)):
        d = np.load(npz_or_dict, allow_pickle=True)
        return {k: d[k] for k in d.files}
    # numpy NpzFile
    return {k: npz_or_dict[k] for k in npz_or_dict.files}


def _packed_to_arr(p) -> np.ndarray:
    if isinstance(p, np.ndarray):
        return p.astype(np.float64)
    if isinstance(p, (str, Path)):
        return np.load(p).astype(np.float64)
    raise TypeError(f"unsupported packedTimes type: {type(p)}")


def _match_packed(
    legacy_pt: np.ndarray, new_pt: np.ndarray, *, tol_sec: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """1:1 nearest-onset matcher between legacy and new packed-window arrays
    (each shape (n, 2) with [start, end] in seconds). Match cost = onset
    distance (legacy_pt[:,0] vs new_pt[:,0]). Pairs with distance ≤ tol_sec
    are eligible; greedy on sorted-by-distance candidates.

    Returns: (matched_pairs, unmatched_legacy_idx, unmatched_new_idx)."""
    n_l = len(legacy_pt)
    n_n = len(new_pt)
    if n_l == 0 or n_n == 0:
        return [], list(range(n_l)), list(range(n_n))
    legacy_starts = legacy_pt[:, 0].astype(np.float64)
    new_starts = new_pt[:, 0].astype(np.float64)
    order_n = np.argsort(new_starts, kind="mergesort")
    new_sorted = new_starts[order_n]
    candidates: List[Tuple[float, int, int]] = []
    for li, ls in enumerate(legacy_starts):
        lo = np.searchsorted(new_sorted, ls - tol_sec, side="left")
        hi = np.searchsorted(new_sorted, ls + tol_sec, side="right")
        for k in range(lo, hi):
            ni = int(order_n[k])
            d = abs(ls - new_starts[ni])
            if d <= tol_sec:
                candidates.append((d, li, ni))
    candidates.sort()
    used_l: set = set()
    used_n: set = set()
    matched: List[Tuple[int, int]] = []
    for _, li, ni in candidates:
        if li in used_l or ni in used_n:
            continue
        matched.append((li, ni))
        used_l.add(li)
        used_n.add(ni)
    unmatched_l = [i for i in range(n_l) if i not in used_l]
    unmatched_n = [i for i in range(n_n) if i not in used_n]
    return matched, unmatched_l, unmatched_n


def compare_lagpat_pair(
    new_lagpat,
    legacy_lagpat,
    new_packed=None,
    legacy_packed=None,
    *,
    pack_match_tol_ms: float = DEFAULT_PACK_MATCH_TOL_MS,
    packed_tol_sec: float = DEFAULT_PACKED_TOL_SEC,
    lag_raw_tol_sec: float = DEFAULT_LAGRAW_TOL_SEC,
) -> Dict[str, object]:
    """Compare a (new, legacy) lagPat + packedTimes pair after row+column
    alignment. See module docstring."""
    n = _to_dict(new_lagpat)
    l = _to_dict(legacy_lagpat)

    # packedTimes: prefer separate packed arrays; fall back to lagpat dict.
    if new_packed is not None:
        new_pt = _packed_to_arr(new_packed)
    elif "packedTimes" in n:
        new_pt = _packed_to_arr(n["packedTimes"])
    else:
        new_pt = np.zeros((0, 2), dtype=np.float64)
    if legacy_packed is not None:
        legacy_pt = _packed_to_arr(legacy_packed)
    elif "packedTimes" in l:
        legacy_pt = _packed_to_arr(l["packedTimes"])
    else:
        legacy_pt = np.zeros((0, 2), dtype=np.float64)

    # --- Row alignment ---
    new_chn = [str(c) for c in n["chnNames"]]
    legacy_chn = [str(c) for c in l["chnNames"]]
    set_eq = set(new_chn) == set(legacy_chn)

    failures: List[str] = []
    res: Dict[str, object] = {
        "chn_match": bool(set_eq),
        "chn_set_legacy": list(legacy_chn),
        "chn_set_new": list(new_chn),
        "packed_n_legacy": int(len(legacy_pt)),
        "packed_n_new": int(len(new_pt)),
        "packed_n_matched": 0,
        "packed_unmatched_legacy": 0,
        "packed_unmatched_new": 0,
        "packed_max_edge_delta_sec": float("nan"),
        "events_bool_exact": False,
        "events_bool_n_diff": -1,
        "lag_rank_exact": False,
        "lag_rank_n_diff": -1,
        "lag_raw_maxabs_sec": float("nan"),
        "lag_raw_median_abs_sec": float("nan"),
        "lag_raw_p99_abs_sec": float("nan"),
        "lag_raw_nan_pattern_match": False,
        "start_t_delta_sec": float("nan"),
        "failures": failures,
    }

    if not set_eq:
        failures.append("chn_set_mismatch")
        return res

    # Permute new rows to legacy order.
    legacy_idx_by_name = {c: i for i, c in enumerate(legacy_chn)}
    new_idx_by_name = {c: i for i, c in enumerate(new_chn)}
    perm = np.array([new_idx_by_name[c] for c in legacy_chn], dtype=np.int64)

    new_raw = np.asarray(n["lagPatRaw"])[perm]
    new_rank = np.asarray(n["lagPatRank"])[perm]
    new_bool = np.asarray(n["eventsBool"])[perm]
    legacy_raw = np.asarray(l["lagPatRaw"])
    legacy_rank = np.asarray(l["lagPatRank"])
    legacy_bool = np.asarray(l["eventsBool"])

    # --- Column alignment ---
    matched_pairs, unmatched_l, unmatched_n = _match_packed(
        legacy_pt, new_pt, tol_sec=pack_match_tol_ms / 1000.0,
    )
    res["packed_n_matched"] = len(matched_pairs)
    res["packed_unmatched_legacy"] = len(unmatched_l)
    res["packed_unmatched_new"] = len(unmatched_n)

    if matched_pairs:
        l_idx = np.array([p[0] for p in matched_pairs], dtype=np.int64)
        n_idx = np.array([p[1] for p in matched_pairs], dtype=np.int64)
        edge_delta = np.max(np.abs(legacy_pt[l_idx] - new_pt[n_idx]))
        res["packed_max_edge_delta_sec"] = float(edge_delta)
        if edge_delta > packed_tol_sec:
            failures.append("packed_edge_above_tolerance")
    if unmatched_l or unmatched_n:
        failures.append("unmatched_columns")

    if not matched_pairs:
        # No aligned columns — short-circuit the rest.
        res["events_bool_n_diff"] = -1
        res["lag_rank_n_diff"] = -1
        return res

    # --- Aligned-index diffs ---
    aligned_legacy_bool = legacy_bool[:, l_idx]
    aligned_new_bool = new_bool[:, n_idx]
    n_bool_diff = int(np.sum(aligned_legacy_bool != aligned_new_bool))
    res["events_bool_exact"] = bool(n_bool_diff == 0)
    res["events_bool_n_diff"] = n_bool_diff
    if n_bool_diff > 0:
        failures.append("eventsBool_mismatch")

    aligned_legacy_rank = legacy_rank[:, l_idx]
    aligned_new_rank = new_rank[:, n_idx]
    # Rank exactness on finite columns only — a column where any cell is NaN
    # in either side is a "non-finite column" and excluded from the rank
    # diff count (rank int does not have NaN; we proxy via lagPatRaw NaN).
    aligned_legacy_raw = legacy_raw[:, l_idx]
    aligned_new_raw = new_raw[:, n_idx]
    finite_cols = np.all(np.isfinite(aligned_legacy_raw), axis=0) \
        & np.all(np.isfinite(aligned_new_raw), axis=0)
    n_rank_diff = int(np.sum(
        aligned_legacy_rank[:, finite_cols] != aligned_new_rank[:, finite_cols]
    ))
    res["lag_rank_exact"] = bool(n_rank_diff == 0)
    res["lag_rank_n_diff"] = n_rank_diff
    if n_rank_diff > 0:
        failures.append("rank_mismatch")

    nan_mask_l = ~np.isfinite(aligned_legacy_raw)
    nan_mask_n = ~np.isfinite(aligned_new_raw)
    res["lag_raw_nan_pattern_match"] = bool(np.array_equal(nan_mask_l, nan_mask_n))
    if not res["lag_raw_nan_pattern_match"]:
        failures.append("lag_raw_nan_pattern_mismatch")

    finite_cells = ~(nan_mask_l | nan_mask_n)
    if finite_cells.any():
        diff = (aligned_legacy_raw - aligned_new_raw)[finite_cells]
        abs_diff = np.abs(diff)
        res["lag_raw_maxabs_sec"] = float(np.max(abs_diff))
        res["lag_raw_median_abs_sec"] = float(np.median(abs_diff))
        res["lag_raw_p99_abs_sec"] = float(np.percentile(abs_diff, 99))
        if res["lag_raw_maxabs_sec"] > lag_raw_tol_sec:
            failures.append("raw_above_tolerance")

    # ---- Phase A baseline metrics (loose tolerance kept for backward
    # compatibility with `scripts/validate_pack_against_legacy.py`).
    # Phase A accepted lagPatRaw at median ≤ 5 ms / p95 ≤ 20 ms / RMSE ≤ 10 ms
    # and lagPatRank at `full_event_match_rate ≥ 0.95` /
    # `participating_only_rate ≥ 0.99`. We surface those numbers next to the
    # strict ones so the comparator does not double-count the same data and
    # so the operator can choose the right contract for the cohort.
    phaseA = {
        "lag_raw_median_abs_ms": float("nan"),
        "lag_raw_p95_abs_ms": float("nan"),
        "lag_raw_rmse_ms": float("nan"),
        "lag_raw_phaseA_pass": False,
        "lag_rank_full_event_match_rate": float("nan"),
        "lag_rank_participating_only_match_rate": float("nan"),
        "lag_rank_phaseA_pass": False,
    }
    if finite_cells.any():
        diff_ms_arr = (aligned_legacy_raw - aligned_new_raw)[finite_cells] * 1000.0
        abs_ms = np.abs(diff_ms_arr)
        phaseA["lag_raw_median_abs_ms"] = float(np.median(abs_ms))
        phaseA["lag_raw_p95_abs_ms"] = float(np.percentile(abs_ms, 95))
        phaseA["lag_raw_rmse_ms"] = float(np.sqrt(np.mean(diff_ms_arr ** 2)))
        phaseA["lag_raw_phaseA_pass"] = bool(
            phaseA["lag_raw_median_abs_ms"] <= 5.0
            and phaseA["lag_raw_p95_abs_ms"] <= 20.0
            and phaseA["lag_raw_rmse_ms"] <= 10.0
        )
    # full-event rank match (any aligned column where every channel's rank matches)
    legacy_bool_aligned = aligned_legacy_bool.astype(bool)
    if aligned_legacy_rank.size:
        eq = (aligned_legacy_rank == aligned_new_rank)
        phaseA["lag_rank_full_event_match_rate"] = float(np.all(eq, axis=0).mean())
        if legacy_bool_aligned.any():
            partic_match = float(eq[legacy_bool_aligned].mean())
        else:
            partic_match = float("nan")
        phaseA["lag_rank_participating_only_match_rate"] = partic_match
        phaseA["lag_rank_phaseA_pass"] = bool(
            phaseA["lag_rank_full_event_match_rate"] >= 0.95
            and (np.isnan(partic_match) or partic_match >= 0.99)
        )
    res["phaseA"] = phaseA

    # start_t comparison if present
    if "start_t" in n and "start_t" in l:
        res["start_t_delta_sec"] = float(abs(float(n["start_t"]) - float(l["start_t"])))

    return res


# ---------------------------------------------------------------------------
# Provenance gate
# ---------------------------------------------------------------------------


def provenance_violates_gate(summary: Dict, record_manifest: Dict) -> bool:
    """Return True if either the subject-level `refine_npz_used` or the
    per-record `gpu_npz_used` lives under `DETECT_ROOT`. That would mean
    the record was produced by the same-source contract, not the
    legacy-refine replay, and the comparator must refuse to verdict it."""
    detect_str = str(DETECT_ROOT)
    refine_used = str(summary.get("refine_npz_used", ""))
    gpu_used = str(record_manifest.get("gpu_npz_used", ""))
    return refine_used.startswith(detect_str) or gpu_used.startswith(detect_str)


# ---------------------------------------------------------------------------
# Per-subject + cohort audit
# ---------------------------------------------------------------------------


def _audit_subject(
    *,
    subject: str,
    replay_dir: Path,
    legacy_backup_dir: Path,
    pack_match_tol_ms: float,
    packed_tol_sec: float,
    lag_raw_tol_sec: float,
) -> Dict[str, object]:
    summary_path = replay_dir / "summary.json"
    manifest_path = replay_dir / "manifest.json"
    if not (summary_path.exists() and manifest_path.exists()):
        return {
            "subject": subject,
            "status": "not_replayed",
            "n_records": 0,
            "records": [],
        }
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    block_manifests = {b["record"]: b for b in manifest.get("blocks", [])}

    rec_results: List[Dict[str, object]] = []
    for rec in summary.get("records", []):
        record = rec.get("record")
        if rec.get("status") != "ok":
            rec_results.append({
                "record": record,
                "status": rec.get("status"),
                "skip_reason": rec.get("skip_reason"),
                "verdict": "skipped_by_replay",
            })
            continue

        # Provenance gate
        block_manifest = block_manifests.get(record, {})
        merged_rec = {**rec, **block_manifest}  # rec carries gpu_npz_used too
        if provenance_violates_gate(summary, merged_rec):
            rec_results.append({
                "record": record,
                "status": "ok",
                "verdict": "provenance_violation",
                "failures": ["provenance_violation"],
            })
            continue

        replay_lagpat = replay_dir / f"{record}_lagPat.npz"
        replay_packed = replay_dir / f"{record}_packedTimes.npy"
        legacy_lagpat = legacy_backup_dir / f"{record}_lagPat.npz"
        legacy_packed = legacy_backup_dir / f"{record}_packedTimes.npy"
        if not (legacy_lagpat.exists() and legacy_packed.exists()):
            rec_results.append({
                "record": record,
                "status": "ok",
                "verdict": "no_legacy_ground_truth",
            })
            continue
        if not (replay_lagpat.exists() and replay_packed.exists()):
            rec_results.append({
                "record": record,
                "status": "ok",
                "verdict": "replay_output_missing",
            })
            continue

        cmp = compare_lagpat_pair(
            replay_lagpat, legacy_lagpat,
            new_packed=replay_packed, legacy_packed=legacy_packed,
            pack_match_tol_ms=pack_match_tol_ms,
            packed_tol_sec=packed_tol_sec,
            lag_raw_tol_sec=lag_raw_tol_sec,
        )
        strict_pass = not cmp["failures"]
        # Pack-stage parity is the structural claim: chnNames + packedTimes
        # exact, eventsBool exact, NaN pattern matched. Centroid + rank
        # numerical drift is its own axis (Phase A baseline).
        pack_stage_failures = [
            f for f in cmp["failures"]
            if f in {"chn_set_mismatch", "unmatched_columns",
                     "packed_edge_above_tolerance", "eventsBool_mismatch",
                     "lag_raw_nan_pattern_mismatch"}
        ]
        pack_stage_pass = not pack_stage_failures
        phaseA = cmp.get("phaseA", {})
        phaseA_pass = (
            pack_stage_pass
            and bool(phaseA.get("lag_raw_phaseA_pass"))
            and bool(phaseA.get("lag_rank_phaseA_pass"))
        )
        if strict_pass:
            verdict = "strict_pass"
        elif phaseA_pass:
            verdict = "phaseA_baseline_pass"
        elif pack_stage_pass:
            verdict = "pack_stage_only"
        else:
            verdict = "fail"
        rec_results.append({
            "record": record,
            "status": "ok",
            "verdict": verdict,
            "pack_stage_pass": pack_stage_pass,
            "phaseA_pass": phaseA_pass,
            "strict_pass": strict_pass,
            **cmp,
        })

    n_strict = sum(1 for r in rec_results if r.get("verdict") == "strict_pass")
    n_phaseA = sum(1 for r in rec_results if r.get("verdict") == "phaseA_baseline_pass")
    n_pack_only = sum(1 for r in rec_results if r.get("verdict") == "pack_stage_only")
    n_fail = sum(1 for r in rec_results if r.get("verdict") == "fail")
    n_provenance = sum(1 for r in rec_results if r.get("verdict") == "provenance_violation")
    n_no_legacy = sum(1 for r in rec_results if r.get("verdict") == "no_legacy_ground_truth")
    n_skipped = sum(1 for r in rec_results if r.get("verdict") == "skipped_by_replay")
    n_pack_stage = sum(1 for r in rec_results if r.get("pack_stage_pass") is True)
    n_phaseA_total = sum(1 for r in rec_results if r.get("phaseA_pass") is True)

    return {
        "subject": subject,
        "status": "audited",
        "n_records": len(rec_results),
        "n_strict_pass": n_strict,
        "n_phaseA_pass": n_phaseA + n_strict,
        "n_pack_stage_pass": n_pack_stage,
        "n_phaseA_baseline_pass_total": n_phaseA_total,
        "n_pack_stage_only": n_pack_only,
        "n_fail": n_fail,
        "n_provenance_violation": n_provenance,
        "n_no_legacy_ground_truth": n_no_legacy,
        "n_skipped_by_replay": n_skipped,
        "records": rec_results,
    }


def _cohort_verdict(per_subject: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    audited = [v for v in per_subject.values() if v["status"] == "audited"]
    n_subjects = len(audited)
    n_records = sum(int(v["n_records"]) for v in audited)
    n_strict = sum(int(v.get("n_strict_pass", 0)) for v in audited)
    n_phaseA = sum(int(v.get("n_phaseA_baseline_pass_total", 0)) for v in audited)
    n_pack_stage = sum(int(v.get("n_pack_stage_pass", 0)) for v in audited)
    n_fail = sum(int(v.get("n_fail", 0)) for v in audited)
    n_prov = sum(int(v.get("n_provenance_violation", 0)) for v in audited)
    n_no_legacy = sum(int(v.get("n_no_legacy_ground_truth", 0)) for v in audited)

    failure_classes: Counter = Counter()
    for v in audited:
        for r in v.get("records", []):
            for f in r.get("failures", []) or []:
                failure_classes[f] += 1

    pass_eligible = n_records - n_no_legacy - n_prov

    # Three-tier verdict: strict (plan), phaseA-baseline (already-validated
    # historical thresholds), pack-stage-only (structural reproduction
    # without centroid parity).
    strict_pass = (n_subjects > 0 and pass_eligible > 0
                   and pass_eligible == n_strict and n_prov == 0)
    phaseA_pass = (n_subjects > 0 and pass_eligible > 0
                   and pass_eligible == n_phaseA and n_prov == 0)
    pack_stage_pass = (n_subjects > 0 and pass_eligible > 0
                       and pass_eligible == n_pack_stage and n_prov == 0)

    if strict_pass:
        label = "strict_replay_pass"
    elif phaseA_pass:
        label = "phaseA_baseline_pass"
    elif pack_stage_pass:
        label = "pack_stage_pass_only"
    else:
        label = "fail"

    return {
        "cohort_label": label,
        "strict_replay_pass": bool(strict_pass),
        "phaseA_baseline_pass": bool(phaseA_pass),
        "pack_stage_pass_only": bool(pack_stage_pass),
        "n_subjects_audited": n_subjects,
        "n_records_total": n_records,
        "n_strict_pass": n_strict,
        "n_phaseA_baseline_pass": n_phaseA,
        "n_pack_stage_pass": n_pack_stage,
        "n_fail": n_fail,
        "n_provenance_violation": n_prov,
        "n_no_legacy_ground_truth": n_no_legacy,
        "n_pass_eligible": pass_eligible,
        "failure_class_counts": dict(failure_classes),
    }


def _render_markdown(report: Dict[str, object]) -> str:
    lines: List[str] = []
    cv = report["cohort_verdict"]
    lines.append("# Yuquan legacy-refine replay numerical audit")
    lines.append("")
    lines.append(f"- schema: `{report['schema_version']}`")
    lines.append(f"- replay root: `{report['out_root']}`")
    lines.append(f"- legacy backup root template: `<DATA_ROOT>/<subject>/.legacy_backup`")
    lines.append("")
    lines.append("## Tolerances (strict, plan-defined)")
    lines.append("")
    tol = report["tolerances"]
    lines.append(f"- `pack_match_tol_ms`: {tol['pack_match_tol_ms']} ms")
    lines.append(f"- `packed_tol_sec`:    {tol['packed_tol_sec']} s")
    lines.append(f"- `lag_raw_tol_sec`:   {tol['lag_raw_tol_sec']} s")
    lines.append("")
    lines.append("## Phase A baseline thresholds (loose, historically validated)")
    lines.append("")
    lines.append("- `lag_raw_median_abs_ms` ≤ 5.0 (Phase A A4)")
    lines.append("- `lag_raw_p95_abs_ms`    ≤ 20.0 (Phase A A4)")
    lines.append("- `lag_raw_rmse_ms`       ≤ 10.0 (Phase A A4)")
    lines.append("- `lag_rank_full_event_match_rate` ≥ 0.95 (Phase A A5)")
    lines.append("- `lag_rank_participating_only_match_rate` ≥ 0.99 (Phase A A5)")
    lines.append("")
    lines.append("## Cohort verdict")
    lines.append("")
    lines.append(f"- **label**: `{cv['cohort_label']}`")
    lines.append(f"- strict (plan, lag_raw maxabs ≤ tol + lag_rank exact + pack-stage exact): "
                 f"`{cv['strict_replay_pass']}`")
    lines.append(f"- Phase A baseline (lag_raw within Phase A bounds + lag_rank ≥ 0.95 + pack-stage exact): "
                 f"`{cv['phaseA_baseline_pass']}`")
    lines.append(f"- pack-stage-only (chnNames + packedTimes + eventsBool exact, ignoring centroid drift): "
                 f"`{cv['pack_stage_pass_only']}`")
    lines.append("")
    lines.append(f"- subjects audited: {cv['n_subjects_audited']}")
    lines.append(f"- records total: {cv['n_records_total']}")
    lines.append(f"- pass-eligible (excluding no_legacy + provenance_violation): {cv['n_pass_eligible']}")
    lines.append(f"- strict pass:          {cv['n_strict_pass']} / {cv['n_pass_eligible']}")
    lines.append(f"- Phase A baseline pass: {cv['n_phaseA_baseline_pass']} / {cv['n_pass_eligible']}")
    lines.append(f"- pack-stage pass:       {cv['n_pack_stage_pass']} / {cv['n_pass_eligible']}")
    lines.append(f"- fail:                  {cv['n_fail']}")
    lines.append(f"- provenance violation:  {cv['n_provenance_violation']}")
    lines.append(f"- no legacy ground truth: {cv['n_no_legacy_ground_truth']}")
    if cv["failure_class_counts"]:
        lines.append("")
        lines.append("### Failure class counts")
        lines.append("")
        for k, v in sorted(cv["failure_class_counts"].items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Per-subject summary")
    lines.append("")
    lines.append("| subject | status | n_rec | pack | phaseA | strict | fail | prov | no_legacy |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in YUQUAN_SAME_SOURCE_SUBJECTS:
        v = report["per_subject"].get(s)
        if v is None:
            continue
        lines.append(
            f"| `{s}` | {v['status']} | {v.get('n_records', 0)} | "
            f"{v.get('n_pack_stage_pass', 0)} | "
            f"{v.get('n_phaseA_baseline_pass_total', 0)} | "
            f"{v.get('n_strict_pass', 0)} | "
            f"{v.get('n_fail', 0)} | "
            f"{v.get('n_provenance_violation', 0)} | "
            f"{v.get('n_no_legacy_ground_truth', 0)} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


SCHEMA_VERSION = "yuquan_legacy_refine_replay_audit_v1_2026Q2"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Track B — Yuquan legacy-refine replay comparator"
    )
    parser.add_argument(
        "--out-root", type=str,
        default=str(REPO_ROOT / "results" / "lagpat_backfill_legacy_refine_replay"),
        help="Replay output root produced by `run_yuquan_legacy_refine_replay.py`."
    )
    parser.add_argument(
        "--audit-out-dir", type=str,
        default=str(REPO_ROOT / "results" / "lagpat_backfill" / "_audit" / "legacy_refine_replay"),
        help="Where to write cohort_replay_audit.{json,md} + per_subject/."
    )
    parser.add_argument("--only-subject", action="append", default=None,
                        help="Restrict to listed subject(s); repeatable.")
    parser.add_argument("--pack-match-tol-ms", type=float, default=DEFAULT_PACK_MATCH_TOL_MS)
    parser.add_argument("--packed-tol-sec", type=float, default=DEFAULT_PACKED_TOL_SEC)
    parser.add_argument("--lag-raw-tol-sec", type=float, default=DEFAULT_LAGRAW_TOL_SEC)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    audit_out_dir = Path(args.audit_out_dir)
    audit_out_dir.mkdir(parents=True, exist_ok=True)
    per_dir = audit_out_dir / "per_subject"
    per_dir.mkdir(parents=True, exist_ok=True)

    subjects = (
        list(args.only_subject) if args.only_subject else list(YUQUAN_SAME_SOURCE_SUBJECTS)
    )

    per_subject: Dict[str, Dict[str, object]] = {}
    for s in subjects:
        replay_dir = out_root / s
        legacy_backup_dir = DATA_ROOT / s / ".legacy_backup"
        v = _audit_subject(
            subject=s,
            replay_dir=replay_dir,
            legacy_backup_dir=legacy_backup_dir,
            pack_match_tol_ms=args.pack_match_tol_ms,
            packed_tol_sec=args.packed_tol_sec,
            lag_raw_tol_sec=args.lag_raw_tol_sec,
        )
        per_subject[s] = v
        (per_dir / f"{s}.json").write_text(
            json.dumps(v, indent=2, ensure_ascii=False, default=str)
        )
        print(f"[{s}] {v['status']} n_rec={v['n_records']} "
              f"pass={v.get('n_pass', 0)} fail={v.get('n_fail', 0)} "
              f"prov={v.get('n_provenance_violation', 0)} "
              f"no_legacy={v.get('n_no_legacy_ground_truth', 0)}")

    cohort_verdict = _cohort_verdict(per_subject)
    report: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "out_root": str(out_root),
        "audit_out_dir": str(audit_out_dir),
        "data_root": str(DATA_ROOT),
        "detect_root": str(DETECT_ROOT),
        "tolerances": {
            "pack_match_tol_ms": args.pack_match_tol_ms,
            "packed_tol_sec": args.packed_tol_sec,
            "lag_raw_tol_sec": args.lag_raw_tol_sec,
        },
        "cohort_verdict": cohort_verdict,
        "per_subject": per_subject,
    }
    json_path = audit_out_dir / "cohort_replay_audit.json"
    md_path = audit_out_dir / "cohort_replay_audit.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    md_path.write_text(_render_markdown(report))
    print(f"\nwrote: {json_path}")
    print(f"wrote: {md_path}")
    print(
        f"cohort label={cohort_verdict['cohort_label']} "
        f"strict={cohort_verdict['n_strict_pass']}/{cohort_verdict['n_pass_eligible']} "
        f"phaseA={cohort_verdict['n_phaseA_baseline_pass']}/{cohort_verdict['n_pass_eligible']} "
        f"pack-stage={cohort_verdict['n_pack_stage_pass']}/{cohort_verdict['n_pass_eligible']} "
        f"fail={cohort_verdict['n_fail']} prov_viol={cohort_verdict['n_provenance_violation']}"
    )
    return 0 if cohort_verdict["cohort_label"] != "fail" else 1


if __name__ == "__main__":
    sys.exit(main())
