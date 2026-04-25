"""Track A — Yuquan detector error attribution at the event level.

Pairs the new detector's `<record>_gpu.npz` (under `results/hfo_detection/`)
with the legacy `<record>_gpu.npz` left in `<raw>/` for every record where
both exist, and reports per-channel one-to-one onset matching at three
tolerance buckets (1.25 ms / 6.25 ms / 25 ms — Yuquan resamples to 800 Hz,
so 1 sample = 1.25 ms).

Cohort coverage (per the contract):

  - usable     14 subjects, ≤177 records
  - degenerate  1 subject  (pengzihang, only 1/12 records has legacy gpu)
  - no_legacy_gpu 6 subjects (excluded from the verdict denominator)

Cohort verdict (two layers, no preset 0.99):

  Layer 1 — no_coarse_logic_shift:
    - alias-collapsed `legacy_chn_containment_in_new` median ≥ 0.95
      (= |legacy_alias ∩ new_alias| / |legacy_alias|; new being a
      superset of legacy is acceptable, since the new code legitimately
      adds bipolar pairs the legacy didn't include);
    - |global_onset_shift_ms| ≤ 1.25 ms (1 sample) per usable subject;
    - tight-match fraction (events matched within 1 sample) ≥ 0.85
      across usable subjects.

  Layer 2 — given Layer 1, label the residual:
    - threshold_sensitive_drift_only: unmatched fraction
      (= max(unmatched_legacy/n_legacy, unmatched_new/n_new)) ≤ 0.30
      across all usable records (matches the documented ±20% events
      drift budget);
    - unexplained_event_set_difference: residual exceeds that budget
      and cannot be explained by boundary/near-duplicate events.

Output:
  - `cohort_detector_attribution.json`
  - `cohort_detector_attribution.md`
  - `per_subject/<subject>.json`
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
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
    alias_bipolar_to_left_with_arbitration,
)

# ---------------------------------------------------------------------------
# Tolerance defaults — anchored to Yuquan's 800 Hz detector resample rate
# (subject_params.json::yuquan._defaults.resample_sfreq = 800).
# 1 sample = 1.25 ms.
# ---------------------------------------------------------------------------

YUQUAN_DETECTOR_RESAMPLE_HZ = 800.0
DEFAULT_TOL_TIGHT_MS = 1.25     # 1 sample at 800 Hz
DEFAULT_TOL_MEDIUM_MS = 6.25    # 5 samples
DEFAULT_TOL_LOOSE_MS = 25.0     # 20 samples


# ---------------------------------------------------------------------------
# Event matcher — globally tighter pair beats local greedy.
# ---------------------------------------------------------------------------


def match_events_one_to_one(
    legacy: Sequence[Sequence[float]],
    new: Sequence[Sequence[float]],
    *,
    tol_loose_ms: float = DEFAULT_TOL_LOOSE_MS,
    tol_medium_ms: float = DEFAULT_TOL_MEDIUM_MS,
    tol_tight_ms: float = DEFAULT_TOL_TIGHT_MS,
    record_last_sec: Optional[float] = None,
    boundary_window_sec: float = 0.2,
    near_duplicate_tol_ms: Optional[float] = None,
) -> Dict[str, object]:
    """Greedy 1:1 onset matcher with global cost ordering.

    Build all candidate (legacy_i, new_j) pairs with onset distance
    ≤ `tol_loose_ms`; sort by absolute distance ascending (ties broken by
    legacy_idx then new_idx for determinism); greedy-assign without
    re-using any index. This is equivalent to Hungarian on a 1-D
    candidate set since the cost is the absolute onset difference.

    Returns a dict (see module docstring).
    """
    if near_duplicate_tol_ms is None:
        near_duplicate_tol_ms = tol_tight_ms

    legacy_starts = np.array([float(e[0]) for e in legacy], dtype=np.float64)
    new_starts = np.array([float(e[0]) for e in new], dtype=np.float64)
    legacy_ends = np.array(
        [float(e[1]) for e in legacy], dtype=np.float64
    ) if legacy else np.zeros(0)
    new_ends = np.array(
        [float(e[1]) for e in new], dtype=np.float64
    ) if new else np.zeros(0)

    n_l = len(legacy)
    n_n = len(new)

    # Candidate generation via sorted onset arrays + bisect window.
    candidates: List[Tuple[float, int, int]] = []
    if n_l > 0 and n_n > 0:
        order_n = np.argsort(new_starts, kind="mergesort")
        new_starts_sorted = new_starts[order_n]
        tol_loose_s = tol_loose_ms / 1000.0
        for li, ls in enumerate(legacy_starts):
            lo = np.searchsorted(new_starts_sorted, ls - tol_loose_s, side="left")
            hi = np.searchsorted(new_starts_sorted, ls + tol_loose_s, side="right")
            for k in range(lo, hi):
                ni = int(order_n[k])
                delta_ms = abs(ls - new_starts[ni]) * 1000.0
                if delta_ms <= tol_loose_ms:
                    candidates.append((delta_ms, li, ni))

    candidates.sort()  # ascending by delta_ms, then li, then ni — stable

    matched: List[Dict[str, object]] = []
    used_l: set = set()
    used_n: set = set()
    for delta_ms, li, ni in candidates:
        if li in used_l or ni in used_n:
            continue
        signed_delta = (new_starts[ni] - legacy_starts[li]) * 1000.0
        l_dur = legacy_ends[li] - legacy_starts[li]
        n_dur = new_ends[ni] - new_starts[ni]
        dur_delta_ms = (n_dur - l_dur) * 1000.0
        if delta_ms <= tol_tight_ms:
            bucket = "tight"
        elif delta_ms <= tol_medium_ms:
            bucket = "medium"
        else:
            bucket = "loose"
        matched.append({
            "legacy_idx": li,
            "new_idx": ni,
            "onset_delta_ms": float(signed_delta),
            "abs_onset_delta_ms": float(delta_ms),
            "duration_delta_ms": float(dur_delta_ms),
            "bucket": bucket,
        })
        used_l.add(li)
        used_n.add(ni)

    unmatched_l = [i for i in range(n_l) if i not in used_l]
    unmatched_n = [i for i in range(n_n) if i not in used_n]

    def _boundary(onset_sec: float) -> bool:
        if record_last_sec is None:
            return False
        return (
            onset_sec <= boundary_window_sec
            or (record_last_sec - onset_sec) <= boundary_window_sec
        )

    def _nearest_dist_ms(onset_sec: float, others: np.ndarray) -> Optional[float]:
        if others.size == 0:
            return None
        return float(np.min(np.abs(others - onset_sec)) * 1000.0)

    unmatched_legacy_detail = [
        {
            "idx": i,
            "onset_sec": float(legacy_starts[i]),
            "nearest_new_dist_ms": _nearest_dist_ms(legacy_starts[i], new_starts),
            "boundary_like": _boundary(float(legacy_starts[i])),
        }
        for i in unmatched_l
    ]
    unmatched_new_detail = [
        {
            "idx": i,
            "onset_sec": float(new_starts[i]),
            "nearest_legacy_dist_ms": _nearest_dist_ms(new_starts[i], legacy_starts),
            "boundary_like": _boundary(float(new_starts[i])),
        }
        for i in unmatched_n
    ]

    # Near-duplicate detection — sub-sample-spaced events on each side.
    def _near_dup_pairs(starts: np.ndarray) -> List[Tuple[int, int]]:
        if starts.size < 2:
            return []
        order = np.argsort(starts, kind="mergesort")
        sorted_s = starts[order]
        out: List[Tuple[int, int]] = []
        tol_s = near_duplicate_tol_ms / 1000.0
        for k in range(len(sorted_s) - 1):
            if (sorted_s[k + 1] - sorted_s[k]) <= tol_s:
                a = int(order[k])
                b = int(order[k + 1])
                out.append((min(a, b), max(a, b)))
        return out

    return {
        "matched": matched,
        "unmatched_legacy_idx": unmatched_l,
        "unmatched_new_idx": unmatched_n,
        "unmatched_legacy_detail": unmatched_legacy_detail,
        "unmatched_new_detail": unmatched_new_detail,
        "near_duplicate_legacy_idx_pairs": _near_dup_pairs(legacy_starts),
        "near_duplicate_new_idx_pairs": _near_dup_pairs(new_starts),
        "n_legacy": n_l,
        "n_new": n_n,
    }


def estimate_global_onset_shift_ms(match_result: Dict[str, object]) -> float:
    """Median signed onset_delta_ms across matched pairs. NaN if no matches."""
    deltas = [m["onset_delta_ms"] for m in match_result.get("matched", [])]
    if not deltas:
        return float("nan")
    return float(np.median(deltas))


# ---------------------------------------------------------------------------
# Cohort coverage classification
# ---------------------------------------------------------------------------


def _classify_subject(subject: str) -> Tuple[str, List[Path]]:
    """Return (bucket, legacy_gpu_paths). Bucket ∈ {usable, degenerate,
    no_legacy_gpu}. legacy_gpu_paths only includes those that pair with a
    new gpu_npz under DETECT_ROOT (else an audit can't pair them)."""
    sd = DATA_ROOT / subject
    nd = DETECT_ROOT / subject
    legacy_gpus = sorted(sd.glob("*_gpu.npz"))
    if not legacy_gpus:
        return ("no_legacy_gpu", [])
    pairs = [p for p in legacy_gpus if (nd / p.name).exists()]
    if not pairs:
        return ("no_legacy_gpu", [])

    n_edfs = max(1, len(list(sd.glob("*.edf"))))
    coverage = len(pairs) / n_edfs
    if coverage < 0.5:
        return ("degenerate", pairs)
    return ("usable", pairs)


# ---------------------------------------------------------------------------
# Per-record audit
# ---------------------------------------------------------------------------


@dataclass
class _RecordAudit:
    record: str
    legacy_gpu_path: str
    new_gpu_path: str
    n_chn_legacy: int
    n_chn_new: int
    chn_jaccard_raw: float
    chn_jaccard_alias: float
    legacy_containment_in_new: float
    n_shared_chn: int
    n_legacy_events_total: int
    n_new_events_total: int
    n_matched_total: int
    n_matched_tight: int
    n_matched_medium: int
    n_matched_loose: int
    n_unmatched_legacy: int
    n_unmatched_new: int
    n_unmatched_legacy_boundary: int
    n_unmatched_new_boundary: int
    global_onset_shift_ms: float
    onset_shift_iqr_ms: float
    n_near_dup_legacy_pairs: int
    n_near_dup_new_pairs: int


def _alias_collapse_for_audit(names: Sequence[str], counts_proxy: Sequence[int]):
    """Map a per-channel name list to alias-collapsed (left-contact) names
    via the same rule used by the live alias arbitrator. `counts_proxy` is
    only used for arbitration tie-breaking."""
    aliases, _, _ = alias_bipolar_to_left_with_arbitration(
        list(names), np.asarray(counts_proxy, dtype=np.int64)
    )
    return set(aliases.keys()), aliases


def _audit_record(
    *, subject: str, record: str, legacy_gpu_path: Path, new_gpu_path: Path,
) -> _RecordAudit:
    legacy = np.load(legacy_gpu_path, allow_pickle=True)
    new = np.load(new_gpu_path, allow_pickle=True)
    legacy_names = [str(x) for x in legacy["chns_names"]]
    new_names = [str(x) for x in new["chns_names"]]
    legacy_dets = legacy["whole_dets"]
    new_dets = new["whole_dets"]
    legacy_counts = legacy["events_count"].astype(np.int64) if "events_count" in legacy.files \
        else np.array([len(legacy_dets[i]) for i in range(len(legacy_dets))])
    new_counts = new["events_count"].astype(np.int64) if "events_count" in new.files \
        else np.array([len(new_dets[i]) for i in range(len(new_dets))])

    legacy_set_raw = set(legacy_names)
    new_set_raw = set(new_names)
    chn_jaccard_raw = (
        len(legacy_set_raw & new_set_raw) / len(legacy_set_raw | new_set_raw)
        if (legacy_set_raw or new_set_raw) else 1.0
    )

    legacy_alias_set, legacy_aliases = _alias_collapse_for_audit(legacy_names, legacy_counts)
    new_alias_set, new_aliases = _alias_collapse_for_audit(new_names, new_counts)
    chn_jaccard_alias = (
        len(legacy_alias_set & new_alias_set) / len(legacy_alias_set | new_alias_set)
        if (legacy_alias_set or new_alias_set) else 1.0
    )
    legacy_containment_in_new = (
        len(legacy_alias_set & new_alias_set) / len(legacy_alias_set)
        if legacy_alias_set else 1.0
    )

    shared_alias = legacy_alias_set & new_alias_set
    n_legacy_total = 0
    n_new_total = 0
    n_matched_total = 0
    n_t = n_m = n_l_loose = 0
    n_unmatched_l = n_unmatched_n = 0
    n_unmatched_l_b = n_unmatched_n_b = 0
    deltas: List[float] = []
    n_near_dup_l = n_near_dup_n = 0

    record_last_sec: Optional[float] = None
    # heuristic: last event end time across legacy+new (cheap proxy for record duration)
    all_ends: List[float] = []
    for ai in shared_alias:
        if ai in legacy_aliases:
            li = legacy_names.index(legacy_aliases[ai].orig)
            for ev in legacy_dets[li]:
                all_ends.append(float(ev[1]))
        if ai in new_aliases:
            ni = new_names.index(new_aliases[ai].orig)
            for ev in new_dets[ni]:
                all_ends.append(float(ev[1]))
    if all_ends:
        record_last_sec = float(max(all_ends))

    for ai in shared_alias:
        l_orig = legacy_aliases[ai].orig
        n_orig = new_aliases[ai].orig
        try:
            li = legacy_names.index(l_orig)
            ni = new_names.index(n_orig)
        except ValueError:
            continue
        legacy_ev = list(legacy_dets[li]) if legacy_dets[li] is not None else []
        new_ev = list(new_dets[ni]) if new_dets[ni] is not None else []
        n_legacy_total += len(legacy_ev)
        n_new_total += len(new_ev)
        m = match_events_one_to_one(
            legacy_ev,
            new_ev,
            record_last_sec=record_last_sec,
        )
        for mm in m["matched"]:
            n_matched_total += 1
            deltas.append(mm["onset_delta_ms"])
            if mm["bucket"] == "tight":
                n_t += 1
            elif mm["bucket"] == "medium":
                n_m += 1
            else:
                n_l_loose += 1
        n_unmatched_l += len(m["unmatched_legacy_idx"])
        n_unmatched_n += len(m["unmatched_new_idx"])
        n_unmatched_l_b += sum(1 for u in m["unmatched_legacy_detail"] if u["boundary_like"])
        n_unmatched_n_b += sum(1 for u in m["unmatched_new_detail"] if u["boundary_like"])
        n_near_dup_l += len(m["near_duplicate_legacy_idx_pairs"])
        n_near_dup_n += len(m["near_duplicate_new_idx_pairs"])

    if deltas:
        d = np.array(deltas, dtype=np.float64)
        global_shift = float(np.median(d))
        iqr = float(np.percentile(d, 75) - np.percentile(d, 25))
    else:
        global_shift = float("nan")
        iqr = float("nan")

    return _RecordAudit(
        record=record,
        legacy_gpu_path=str(legacy_gpu_path),
        new_gpu_path=str(new_gpu_path),
        n_chn_legacy=len(legacy_names),
        n_chn_new=len(new_names),
        chn_jaccard_raw=float(chn_jaccard_raw),
        chn_jaccard_alias=float(chn_jaccard_alias),
        legacy_containment_in_new=float(legacy_containment_in_new),
        n_shared_chn=len(shared_alias),
        n_legacy_events_total=int(n_legacy_total),
        n_new_events_total=int(n_new_total),
        n_matched_total=int(n_matched_total),
        n_matched_tight=int(n_t),
        n_matched_medium=int(n_m),
        n_matched_loose=int(n_l_loose),
        n_unmatched_legacy=int(n_unmatched_l),
        n_unmatched_new=int(n_unmatched_n),
        n_unmatched_legacy_boundary=int(n_unmatched_l_b),
        n_unmatched_new_boundary=int(n_unmatched_n_b),
        global_onset_shift_ms=global_shift,
        onset_shift_iqr_ms=iqr,
        n_near_dup_legacy_pairs=int(n_near_dup_l),
        n_near_dup_new_pairs=int(n_near_dup_n),
    )


# ---------------------------------------------------------------------------
# Verdict computation
# ---------------------------------------------------------------------------


def _subject_global_shift(records: Sequence[_RecordAudit]) -> float:
    shifts = [r.global_onset_shift_ms for r in records if not np.isnan(r.global_onset_shift_ms)]
    if not shifts:
        return float("nan")
    return float(np.median(shifts))


def _classify_residual(
    rec_audits: Sequence[_RecordAudit],
    *,
    unmatched_budget_fraction: float = 0.30,
) -> Tuple[str, Dict[str, object]]:
    """Layer-2 residual classification given Layer 1 holds.

    The phaseD documented detector drift is ±10–20% events_count. This
    classifier accepts up to 30% as the threshold-sensitive band (extra
    headroom for the ±20% being a per-channel observation, not an
    absolute bound). Any record whose unmatched fraction exceeds the
    budget on either side and isn't dominated by boundary/near-duplicate
    events is flagged as `unexplained_event_set_difference`.
    """
    n_unmatched_l = sum(r.n_unmatched_legacy for r in rec_audits)
    n_unmatched_n = sum(r.n_unmatched_new for r in rec_audits)
    n_legacy = sum(r.n_legacy_events_total for r in rec_audits)
    n_new = sum(r.n_new_events_total for r in rec_audits)
    n_boundary = sum(
        r.n_unmatched_legacy_boundary + r.n_unmatched_new_boundary for r in rec_audits
    )
    n_total_unmatched = n_unmatched_l + n_unmatched_n
    unmatched_legacy_frac = n_unmatched_l / n_legacy if n_legacy > 0 else 0.0
    unmatched_new_frac = n_unmatched_n / n_new if n_new > 0 else 0.0
    max_unmatched_frac = max(unmatched_legacy_frac, unmatched_new_frac)
    boundary_frac = (n_boundary / n_total_unmatched) if n_total_unmatched > 0 else 1.0
    near_dup_frac = (
        sum(r.n_near_dup_legacy_pairs + r.n_near_dup_new_pairs for r in rec_audits)
        / max(1, n_total_unmatched)
    )

    info = {
        "n_legacy_events_total": int(n_legacy),
        "n_new_events_total": int(n_new),
        "n_unmatched_legacy": int(n_unmatched_l),
        "n_unmatched_new": int(n_unmatched_n),
        "unmatched_legacy_fraction": float(unmatched_legacy_frac),
        "unmatched_new_fraction": float(unmatched_new_frac),
        "max_unmatched_fraction": float(max_unmatched_frac),
        "unmatched_budget_fraction": float(unmatched_budget_fraction),
        "n_boundary_like": int(n_boundary),
        "boundary_fraction_of_unmatched": float(boundary_frac),
        "near_duplicate_pair_count": int(
            sum(r.n_near_dup_legacy_pairs + r.n_near_dup_new_pairs for r in rec_audits)
        ),
        "near_duplicate_pair_fraction_of_unmatched": float(near_dup_frac),
    }

    if max_unmatched_frac <= unmatched_budget_fraction:
        return ("threshold_sensitive_drift_only", info)
    # Budget exceeded — but boundary-dominated residual is still excusable.
    if boundary_frac >= 0.7:
        return ("threshold_sensitive_drift_only", info)
    return ("unexplained_event_set_difference", info)


def _cohort_verdict(
    per_subject: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    """Combine per-subject verdicts across the usable bucket.

    Layer 1 uses **legacy_containment_in_new** (asymmetric) instead of
    Jaccard so the new pipeline being a superset of legacy doesn't
    spuriously fail. The pass criterion is: legacy channels are mostly
    all in new + no global time shift + most matched events are tight.
    """
    usable = {s: v for s, v in per_subject.items() if v["bucket"] == "usable"}
    if not usable:
        return {
            "label": "no_usable_subjects",
            "layer1_no_coarse_logic_shift": False,
            "layer2_label": None,
            "denominator_subjects": 0,
            "denominator_records": 0,
        }

    containments = [v["legacy_containment_in_new_subject_median"] for v in usable.values()]
    containment_median = float(np.median(containments))
    shifts = [abs(v["global_onset_shift_ms"]) for v in usable.values()
              if not np.isnan(v["global_onset_shift_ms"])]
    max_abs_shift = float(np.max(shifts)) if shifts else float("nan")
    tight_fracs = [v["tight_match_fraction_subject"] for v in usable.values()
                   if not np.isnan(v["tight_match_fraction_subject"])]
    tight_min = float(np.min(tight_fracs)) if tight_fracs else float("nan")

    layer1 = (
        containment_median >= 0.95
        and (np.isnan(max_abs_shift) or max_abs_shift <= DEFAULT_TOL_TIGHT_MS)
        and (np.isnan(tight_min) or tight_min >= 0.85)
    )

    layer2_labels = Counter(v["residual_label"] for v in usable.values())
    if layer2_labels.get("unexplained_event_set_difference", 0) == 0:
        layer2 = "threshold_sensitive_drift_only"
    else:
        layer2 = "unexplained_event_set_difference"

    if not layer1:
        label = "coarse_logic_divergence"
    elif layer2 == "threshold_sensitive_drift_only":
        label = "physical_FP_only_threshold_sensitive"
    else:
        label = "unexplained_residual"

    n_records = sum(v["n_records_audited"] for v in usable.values())

    return {
        "label": label,
        "layer1_no_coarse_logic_shift": bool(layer1),
        "layer2_label": layer2,
        "legacy_containment_in_new_cohort_median": containment_median,
        "max_abs_global_onset_shift_ms": max_abs_shift,
        "min_tight_match_fraction_across_usable": tight_min,
        "n_subjects_per_residual_label": dict(layer2_labels),
        "denominator_subjects": len(usable),
        "denominator_records": int(n_records),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _summarize_subject(
    subject: str, bucket: str, pairs: Sequence[Path], out_dir: Path,
) -> Dict[str, object]:
    if bucket == "no_legacy_gpu":
        return {
            "subject": subject,
            "bucket": bucket,
            "n_records_audited": 0,
            "records": [],
            "verdict": "uncomparable_at_detector",
        }

    rec_audits: List[_RecordAudit] = []
    for legacy_p in pairs:
        record = legacy_p.stem.replace("_gpu", "")
        new_p = DETECT_ROOT / subject / legacy_p.name
        if not new_p.exists():
            continue
        try:
            rec_audits.append(_audit_record(
                subject=subject,
                record=record,
                legacy_gpu_path=legacy_p,
                new_gpu_path=new_p,
            ))
        except Exception as exc:  # robust against malformed npz
            rec_audits.append(_RecordAudit(  # type: ignore[arg-type]
                record=record,
                legacy_gpu_path=str(legacy_p),
                new_gpu_path=str(new_p),
                n_chn_legacy=0, n_chn_new=0,
                chn_jaccard_raw=float("nan"), chn_jaccard_alias=float("nan"),
                n_shared_chn=0,
                n_legacy_events_total=0, n_new_events_total=0,
                n_matched_total=0, n_matched_tight=0, n_matched_medium=0, n_matched_loose=0,
                n_unmatched_legacy=0, n_unmatched_new=0,
                n_unmatched_legacy_boundary=0, n_unmatched_new_boundary=0,
                global_onset_shift_ms=float("nan"), onset_shift_iqr_ms=float("nan"),
                n_near_dup_legacy_pairs=0, n_near_dup_new_pairs=0,
            ))
            print(f"  [{subject}/{record}] error: {exc!r}")

    chn_alias_jaccs = [r.chn_jaccard_alias for r in rec_audits if not np.isnan(r.chn_jaccard_alias)]
    subj_chn_alias_med = float(np.median(chn_alias_jaccs)) if chn_alias_jaccs else float("nan")
    containments = [
        r.legacy_containment_in_new for r in rec_audits
        if not np.isnan(r.legacy_containment_in_new)
    ]
    subj_containment_med = float(np.median(containments)) if containments else float("nan")
    subj_global_shift = _subject_global_shift(rec_audits)
    n_t = sum(r.n_matched_tight for r in rec_audits)
    n_total_matched = sum(r.n_matched_total for r in rec_audits)
    tight_frac_subj = (n_t / n_total_matched) if n_total_matched > 0 else float("nan")
    residual_label, residual_info = _classify_residual(rec_audits)

    if bucket == "degenerate":
        verdict = "degenerate_evidence"
    else:
        verdict = "evaluated"

    rec_dicts = [r.__dict__ for r in rec_audits]
    subj_summary: Dict[str, object] = {
        "subject": subject,
        "bucket": bucket,
        "n_records_audited": len(rec_audits),
        "alias_chn_jaccard_subject_median": subj_chn_alias_med,
        "legacy_containment_in_new_subject_median": subj_containment_med,
        "global_onset_shift_ms": subj_global_shift,
        "tight_match_fraction_subject": float(tight_frac_subj),
        "residual_label": residual_label,
        "residual_info": residual_info,
        "verdict": verdict,
        "records": rec_dicts,
    }

    per_dir = out_dir / "per_subject"
    per_dir.mkdir(parents=True, exist_ok=True)
    (per_dir / f"{subject}.json").write_text(
        json.dumps(subj_summary, indent=2, ensure_ascii=False, default=str)
    )
    return subj_summary


def _render_markdown(report: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Yuquan detector event-level attribution")
    lines.append("")
    lines.append(f"- schema: `{report['schema_version']}`")
    lines.append(f"- detector resample: **{YUQUAN_DETECTOR_RESAMPLE_HZ} Hz**, "
                 f"1 sample = {1000.0 / YUQUAN_DETECTOR_RESAMPLE_HZ:.3g} ms")
    lines.append(f"- tolerance buckets: tight ≤ {DEFAULT_TOL_TIGHT_MS} ms (1 sample); "
                 f"medium ≤ {DEFAULT_TOL_MEDIUM_MS} ms (5 samples); "
                 f"loose ≤ {DEFAULT_TOL_LOOSE_MS} ms (20 samples)")
    lines.append("")
    cv = report["cohort_verdict"]
    lines.append(f"## Cohort verdict: **{cv['label']}**")
    lines.append("")
    lines.append(f"- denominator: **{cv['denominator_subjects']} subjects, "
                 f"{cv['denominator_records']} records** (usable bucket only; "
                 f"degenerate + no_legacy_gpu excluded from verdict)")
    lines.append(f"- layer 1 no_coarse_logic_shift: `{cv['layer1_no_coarse_logic_shift']}`")
    lines.append(f"- layer 2 label: `{cv['layer2_label']}`")
    if "legacy_containment_in_new_cohort_median" in cv:
        lines.append(f"- legacy_containment_in_new cohort median: "
                     f"{cv['legacy_containment_in_new_cohort_median']:.4f} (≥ 0.95 to pass layer 1)")
    if "max_abs_global_onset_shift_ms" in cv:
        lines.append(f"- max |global_onset_shift| across usable subjects: "
                     f"{cv['max_abs_global_onset_shift_ms']} ms (≤ {DEFAULT_TOL_TIGHT_MS} to pass layer 1)")
    if "min_tight_match_fraction_across_usable" in cv:
        lines.append(f"- min tight-match fraction across usable: "
                     f"{cv['min_tight_match_fraction_across_usable']:.4f} (≥ 0.85 to pass layer 1)")
    lines.append("")
    lines.append("## Cohort coverage")
    lines.append("")
    bcounts = report["bucket_counts"]
    lines.append(f"- usable: **{bcounts['usable']}** / 21")
    lines.append(f"- degenerate: **{bcounts['degenerate']}** / 21")
    lines.append(f"- no_legacy_gpu: **{bcounts['no_legacy_gpu']}** / 21")
    lines.append("")
    lines.append("## Per-subject summary")
    lines.append("")
    lines.append("| subject | bucket | n_rec | legacy_in_new | tight_frac | global_shift_ms | unmatched_max | residual | verdict |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|---|")
    for s in YUQUAN_SAME_SOURCE_SUBJECTS:
        v = report["per_subject"].get(s)
        if v is None:
            continue
        cont = v.get("legacy_containment_in_new_subject_median", float("nan"))
        tight = v.get("tight_match_fraction_subject", float("nan"))
        shift = v.get("global_onset_shift_ms", float("nan"))
        info = v.get("residual_info", {}) or {}
        unmatched_max = info.get("max_unmatched_fraction", float("nan"))
        cont_str = "nan" if (cont is None or (isinstance(cont, float) and np.isnan(cont))) \
            else f"{cont:.3f}"
        tight_str = "nan" if (tight is None or (isinstance(tight, float) and np.isnan(tight))) \
            else f"{tight:.3f}"
        shift_str = "nan" if (shift is None or (isinstance(shift, float) and np.isnan(shift))) \
            else f"{shift:+.2f}"
        unmatched_str = "nan" if (unmatched_max is None or (isinstance(unmatched_max, float) and np.isnan(unmatched_max))) \
            else f"{unmatched_max:.3f}"
        lines.append(
            f"| `{s}` | {v['bucket']} | {v.get('n_records_audited', 0)} | {cont_str} | "
            f"{tight_str} | {shift_str} | {unmatched_str} | "
            f"{v.get('residual_label', '-')} | {v.get('verdict', '-')} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


SCHEMA_VERSION = "yuquan_detector_attribution_v1_2026Q2"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Track A — Yuquan detector event-level attribution audit"
    )
    parser.add_argument(
        "--out-dir", type=str,
        default=str(REPO_ROOT / "results" / "lagpat_backfill" / "_audit" / "detector_attribution"),
        help="Output directory for cohort_detector_attribution.{json,md} + per_subject/."
    )
    parser.add_argument(
        "--only-subject", type=str, default=None,
        help="Restrict to a single subject (for debugging)."
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = (
        [args.only_subject] if args.only_subject else list(YUQUAN_SAME_SOURCE_SUBJECTS)
    )
    bucket_counts = {"usable": 0, "degenerate": 0, "no_legacy_gpu": 0}
    per_subject: Dict[str, Dict[str, object]] = {}
    for s in subjects:
        bucket, pairs = _classify_subject(s)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        print(f"[{s}] bucket={bucket} pairs={len(pairs)}")
        per_subject[s] = _summarize_subject(s, bucket, pairs, out_dir)

    cohort_verdict = _cohort_verdict(per_subject)
    report: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "cohort_size": len(subjects),
        "subjects": subjects,
        "bucket_counts": bucket_counts,
        "tolerance_ms": {
            "tight": DEFAULT_TOL_TIGHT_MS,
            "medium": DEFAULT_TOL_MEDIUM_MS,
            "loose": DEFAULT_TOL_LOOSE_MS,
        },
        "detector_resample_hz": YUQUAN_DETECTOR_RESAMPLE_HZ,
        "data_root": str(DATA_ROOT),
        "detect_root": str(DETECT_ROOT),
        "cohort_verdict": cohort_verdict,
        "per_subject": per_subject,
    }

    json_path = out_dir / "cohort_detector_attribution.json"
    md_path = out_dir / "cohort_detector_attribution.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    md_path.write_text(_render_markdown(report))

    print(f"\nwrote: {json_path}")
    print(f"wrote: {md_path}")
    print(f"cohort verdict: {cohort_verdict['label']} "
          f"(usable={bucket_counts['usable']} degenerate={bucket_counts['degenerate']} "
          f"no_legacy_gpu={bucket_counts['no_legacy_gpu']})")

    # exit 0 even on coarse_logic_divergence — Track A is a report, not a gate
    return 0


if __name__ == "__main__":
    sys.exit(main())
