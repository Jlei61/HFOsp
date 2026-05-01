"""Stage C: structural audit — new vs legacy Epilepsiae lagPat.

Plan: docs/archive/epilepsiae_lagpat/epilepsiae_lagpat_backfill_plan_2026-04-29.md §4.

Compares per record:
  - new lagPat (results/epilepsiae_lagpat_backfill/<subject>/<stem>_lagPat.npz)
  - legacy lagPat (/mnt/epilepsia_data/interilca_inter_results/all_data_lns/<subject>/all_recs/<stem>_lagPat.npz)

5 dimensions of structural similarity (NOT numerical parity):
  1. Channel set Jaccard overlap
  2. Event count ratio
  3. Per-event participation distribution KS test
  4. Lag span median diff
  5. Mean rank-vector Pearson r over shared channels

User risk callout (2026-04-30): zero-event records and record-set alignment must be
EXPLICIT fields, never silently averaged into ratio / KS / rank correlation.

Outputs:
  results/epilepsiae_lagpat_backfill/audit/per_record_audit.csv
  results/epilepsiae_lagpat_backfill/audit/per_subject_audit.csv
  results/epilepsiae_lagpat_backfill/audit/cohort_audit_summary.json

Stage D scope is decided by `decision_for_stage_d` in cohort summary; user-gated
per plan §10b R2.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# nanmean on an all-NaN slice is the expected behavior here (channels that
# never participate get NaN mean rank — caller filters them out).
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_epilepsiae_lagpat_backfill import COHORT_SUBJECTS  # type: ignore

LEGACY_LAGPAT_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
NEW_LAGPAT_ROOT = Path("results/epilepsiae_lagpat_backfill")
AUDIT_OUTPUT_ROOT = NEW_LAGPAT_ROOT / "audit"

# ---------------------------------------------------------------------------
# Bucket classification + decision rules (per plan §4 Task C.2)
# ---------------------------------------------------------------------------

_BUCKET_ORDER = {"stable": 2, "moderate_drift": 1, "large_drift": 0}


def _classify_chn_bucket(jaccard: Optional[float]) -> str:
    """Plan §4 Task C.2 table: chn_overlap_jaccard thresholds."""
    if jaccard is None or (isinstance(jaccard, float) and math.isnan(jaccard)):
        return "large_drift"  # missing similarity => conservative worst case
    if jaccard >= 0.7:
        return "stable"
    if jaccard >= 0.5:
        return "moderate_drift"
    return "large_drift"


def _classify_count_bucket(ratio: Optional[float]) -> str:
    """Plan §4 Task C.2 table: count_ratio thresholds.

    stable: [0.7, 1.4]
    moderate_drift: [0.5, 0.7) or (1.4, 2.0]
    large_drift: < 0.5 or > 2.0 or NaN
    """
    if ratio is None or (isinstance(ratio, float) and (math.isnan(ratio) or math.isinf(ratio))):
        return "large_drift"
    if 0.7 <= ratio <= 1.4:
        return "stable"
    if 0.5 <= ratio <= 2.0:
        return "moderate_drift"
    return "large_drift"


def _assign_subject_bucket(chn_bucket: str, count_bucket: str) -> str:
    """Take the more conservative (worse) of the two."""
    chn_rank = _BUCKET_ORDER.get(chn_bucket, 0)
    count_rank = _BUCKET_ORDER.get(count_bucket, 0)
    if chn_rank <= count_rank:
        return chn_bucket
    return count_bucket


def _decide_stage_d(stable: int, moderate: int, large: int) -> str:
    """Plan §4 Task C.2 decision table.

    enter_full: stable >= 14 AND large <= 2
    enter_smoke: 5 <= stable <= 13 OR 3 <= large <= 7
    pause: large >= 8 OR stable < 5
    """
    if stable < 5 or large >= 8:
        return "pause"
    if stable >= 14 and large <= 2:
        return "enter_full"
    return "enter_smoke"


# ---------------------------------------------------------------------------
# Path / discovery helpers
# ---------------------------------------------------------------------------


def _legacy_subject_dir(subject: str) -> Path:
    return LEGACY_LAGPAT_ROOT / subject / "all_recs"


def _new_subject_dir(subject: str) -> Path:
    return NEW_LAGPAT_ROOT / subject


def _list_legacy_stems(subject: str) -> List[str]:
    d = _legacy_subject_dir(subject)
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*_lagPat.npz")):
        if "withFreqCent" in p.name:
            continue
        out.append(p.name[: -len("_lagPat.npz")])
    return out


def _list_new_stems(subject: str) -> List[str]:
    d = _new_subject_dir(subject)
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*_lagPat.npz")):
        out.append(p.name[: -len("_lagPat.npz")])
    return out


def _record_pair_status(subject: str) -> Dict[str, str]:
    """Returns {stem: status} where status in {paired, new_only, legacy_only}."""
    new_stems = set(_list_new_stems(subject))
    legacy_stems = set(_list_legacy_stems(subject))
    out: Dict[str, str] = {}
    for s in sorted(new_stems | legacy_stems):
        if s in new_stems and s in legacy_stems:
            out[s] = "paired"
        elif s in new_stems:
            out[s] = "new_only"
        else:
            out[s] = "legacy_only"
    return out


def _load_lagpat(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load a *_lagPat.npz; return None on load error."""
    try:
        z = np.load(path, allow_pickle=True)
    except Exception:
        return None
    out: Dict[str, np.ndarray] = {}
    for key in ("chnNames", "lagPatRaw", "lagPatRank", "eventsBool", "start_t"):
        if key in z.files:
            out[key] = z[key]
    if "lagPatRaw" not in out or "chnNames" not in out:
        return None
    return out


# ---------------------------------------------------------------------------
# Per-record metric computation
# ---------------------------------------------------------------------------


def _shape_n_events(arr: Optional[np.ndarray]) -> int:
    """lagPatRaw is (n_chns, n_events); be tolerant of empty / 1-D."""
    if arr is None:
        return 0
    a = np.asarray(arr)
    if a.ndim == 0 or a.size == 0:
        return 0
    if a.ndim == 1:
        return 0
    return int(a.shape[1])


def _shape_n_chns(rec: Optional[Dict[str, np.ndarray]]) -> int:
    if rec is None:
        return 0
    cn = rec.get("chnNames")
    if cn is None:
        return 0
    return int(np.asarray(cn).size)


def _participation_vec(rec: Dict[str, np.ndarray]) -> np.ndarray:
    """Per-event n_participating from eventsBool. Returns shape (n_events,)."""
    eb = np.asarray(rec.get("eventsBool"))
    if eb.ndim != 2 or eb.size == 0:
        return np.empty((0,), dtype=float)
    return eb.sum(axis=0).astype(float)


def _lag_span_med(rec: Dict[str, np.ndarray]) -> Optional[float]:
    """Median per-event lag span. NaN-aware: new pipeline lagPatRaw has NaN
    for non-participating channels; legacy lagPatRaw is dense (already pruned)."""
    raw = np.asarray(rec.get("lagPatRaw"), dtype=float)
    if raw.ndim != 2 or raw.size == 0:
        return None
    with np.errstate(invalid="ignore"):
        col_max = np.nanmax(raw, axis=0)
        col_min = np.nanmin(raw, axis=0)
    span = col_max - col_min
    span = span[np.isfinite(span)]
    if span.size == 0:
        return None
    return float(np.median(span))


def _mean_rank_per_chn(rec: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Mean rank per channel, masked by participation.

    For new pipeline: lagPatRank uses _legacy_hist_mean_rank fallback (rank=ci)
    for non-participating channels. Averaging without masking would dilute the
    template by these fallback ranks. Mask by eventsBool > 0 instead.
    Channels with zero participation get NaN (excluded from correlation).
    """
    rank = np.asarray(rec.get("lagPatRank"), dtype=float)
    eb = np.asarray(rec.get("eventsBool"), dtype=float)
    chn_names = [str(c) for c in np.asarray(rec.get("chnNames"))]
    if rank.ndim != 2 or rank.size == 0 or len(chn_names) == 0:
        return {}
    if rank.shape[0] != len(chn_names):
        return {}
    # Mask: True where channel participates in event
    if eb.shape == rank.shape:
        mask = eb > 0
        rank_masked = np.where(mask, rank, np.nan)
        # nanmean of all-NaN row issues a RuntimeWarning; suppress.
        with np.errstate(invalid="ignore"):
            means = np.nanmean(rank_masked, axis=1)
    else:
        means = rank.mean(axis=1)
    out: Dict[str, float] = {}
    for i, ch in enumerate(chn_names):
        v = float(means[i])
        if not math.isnan(v):
            out[ch] = v
    return out


def _pearson_r(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Pearson r with NaN guard. Returns None if undefined."""
    if a.size < 2 or b.size != a.size:
        return None
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    r = np.corrcoef(a, b)[0, 1]
    if math.isnan(r):
        return None
    return float(r)


def _ks_2samp_p(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """KS 2-sample p-value; needs both samples nonempty."""
    if a.size == 0 or b.size == 0:
        return None
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        return None
    res = ks_2samp(a, b)
    return float(res.pvalue)


def compute_record_metrics(
    new_rec: Optional[Dict[str, np.ndarray]],
    legacy_rec: Optional[Dict[str, np.ndarray]],
) -> Dict[str, object]:
    """Compute the 5-dimensional structural-similarity metrics for one record.

    Risk-aware: zero-event records and missing-side records get EXPLICIT *_eligible
    flags; downstream aggregation must filter on those flags before taking medians.
    """
    n_chns_new = _shape_n_chns(new_rec)
    n_chns_legacy = _shape_n_chns(legacy_rec)
    n_ev_new = _shape_n_events(new_rec.get("lagPatRaw") if new_rec else None)
    n_ev_legacy = _shape_n_events(legacy_rec.get("lagPatRaw") if legacy_rec else None)

    chn_set_new = (
        {str(c) for c in np.asarray(new_rec["chnNames"])} if new_rec else set()
    )
    chn_set_legacy = (
        {str(c) for c in np.asarray(legacy_rec["chnNames"])} if legacy_rec else set()
    )
    shared_chns = sorted(chn_set_new & chn_set_legacy)
    union_chns = chn_set_new | chn_set_legacy

    # Channel Jaccard: only computable when both have ≥ 1 chn
    chn_jaccard: Optional[float]
    if chn_set_new and chn_set_legacy:
        chn_jaccard = len(chn_set_new & chn_set_legacy) / len(union_chns)
        chn_jaccard_eligible = True
    else:
        chn_jaccard = None
        chn_jaccard_eligible = False

    # Asymmetric coverage (advisor 2026-04-30):
    # cov_legacy_in_new = how much of legacy chns appears in new (1.0 => legacy ⊂ new)
    # cov_new_in_legacy = how much of new chns appears in legacy (low => new added channels)
    if chn_set_legacy:
        cov_legacy_in_new: Optional[float] = len(chn_set_new & chn_set_legacy) / len(chn_set_legacy)
    else:
        cov_legacy_in_new = None
    if chn_set_new:
        cov_new_in_legacy: Optional[float] = len(chn_set_new & chn_set_legacy) / len(chn_set_new)
    else:
        cov_new_in_legacy = None

    # Count ratio: only when both have ≥ 1 events
    if n_ev_new >= 1 and n_ev_legacy >= 1:
        count_ratio: Optional[float] = n_ev_new / n_ev_legacy
        count_ratio_eligible = True
    else:
        count_ratio = None
        count_ratio_eligible = False

    # Participation KS: both ≥ 1 event
    if n_ev_new >= 1 and n_ev_legacy >= 1:
        part_new = _participation_vec(new_rec)  # type: ignore[arg-type]
        part_legacy = _participation_vec(legacy_rec)  # type: ignore[arg-type]
        ks_p = _ks_2samp_p(part_new, part_legacy)
        ks_eligible = ks_p is not None
    else:
        ks_p = None
        ks_eligible = False

    # Lag span: both ≥ 1 event
    if n_ev_new >= 1 and n_ev_legacy >= 1:
        span_new = _lag_span_med(new_rec)  # type: ignore[arg-type]
        span_legacy = _lag_span_med(legacy_rec)  # type: ignore[arg-type]
        if span_new is not None and span_legacy is not None:
            lag_span_diff = span_new - span_legacy
            lag_span_eligible = True
        else:
            lag_span_diff = None
            lag_span_eligible = False
    else:
        span_new = None
        span_legacy = None
        lag_span_diff = None
        lag_span_eligible = False

    # Rank template Pearson r: shared ≥ 3 AND both have events
    if len(shared_chns) >= 3 and n_ev_new >= 1 and n_ev_legacy >= 1:
        mean_rank_new = _mean_rank_per_chn(new_rec)  # type: ignore[arg-type]
        mean_rank_legacy = _mean_rank_per_chn(legacy_rec)  # type: ignore[arg-type]
        a = np.array([mean_rank_new[c] for c in shared_chns if c in mean_rank_new and c in mean_rank_legacy], dtype=float)
        b = np.array([mean_rank_legacy[c] for c in shared_chns if c in mean_rank_new and c in mean_rank_legacy], dtype=float)
        if a.size >= 3:
            rank_r = _pearson_r(a, b)
            rank_eligible = rank_r is not None
        else:
            rank_r = None
            rank_eligible = False
    else:
        rank_r = None
        rank_eligible = False

    return {
        "n_chns_new": n_chns_new,
        "n_chns_legacy": n_chns_legacy,
        "n_chns_shared": len(shared_chns),
        "n_chns_union": len(union_chns),
        "n_events_new": n_ev_new,
        "n_events_legacy": n_ev_legacy,
        "zero_event_new": n_ev_new == 0,
        "zero_event_legacy": n_ev_legacy == 0,
        "both_zero": (n_ev_new == 0 and n_ev_legacy == 0),
        "chn_overlap_jaccard": chn_jaccard,
        "chn_overlap_jaccard_eligible": chn_jaccard_eligible,
        "cov_legacy_in_new": cov_legacy_in_new,
        "cov_new_in_legacy": cov_new_in_legacy,
        "count_ratio": count_ratio,
        "count_ratio_eligible": count_ratio_eligible,
        "participation_ks_p": ks_p,
        "participation_ks_eligible": ks_eligible,
        "lag_span_med_new": span_new,
        "lag_span_med_legacy": span_legacy,
        "lag_span_diff": lag_span_diff,
        "lag_span_eligible": lag_span_eligible,
        "rank_template_corr": rank_r,
        "rank_template_eligible": rank_eligible,
    }


# ---------------------------------------------------------------------------
# Per-subject audit
# ---------------------------------------------------------------------------


def audit_subject(subject: str) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Run the audit for one subject. Returns (per_record_rows, subject_summary)."""
    pairs = _record_pair_status(subject)
    per_record: List[Dict[str, object]] = []

    for stem in sorted(pairs.keys()):
        status = pairs[stem]
        new_path = _new_subject_dir(subject) / f"{stem}_lagPat.npz"
        legacy_path = _legacy_subject_dir(subject) / f"{stem}_lagPat.npz"
        new_rec = _load_lagpat(new_path) if status in ("paired", "new_only") else None
        legacy_rec = (
            _load_lagpat(legacy_path) if status in ("paired", "legacy_only") else None
        )
        metrics = compute_record_metrics(new_rec, legacy_rec)
        row = {
            "subject": subject,
            "stem": stem,
            "record_status": status,
            **metrics,
        }
        per_record.append(row)

    # Per-subject medians, only on eligible (paired + eligible) records
    paired_rows = [r for r in per_record if r["record_status"] == "paired"]

    def _med(rows: List[Dict[str, object]], key: str, eligible_key: str) -> Optional[float]:
        vals = [
            r[key] for r in rows
            if r.get(eligible_key) is True and r.get(key) is not None
            and not (isinstance(r[key], float) and math.isnan(r[key]))  # type: ignore[arg-type]
        ]
        if not vals:
            return None
        return float(np.median(np.array(vals, dtype=float)))

    chn_jacc_med = _med(paired_rows, "chn_overlap_jaccard", "chn_overlap_jaccard_eligible")
    cov_legacy_in_new_med = _med(paired_rows, "cov_legacy_in_new", "chn_overlap_jaccard_eligible")
    cov_new_in_legacy_med = _med(paired_rows, "cov_new_in_legacy", "chn_overlap_jaccard_eligible")
    count_ratio_med = _med(paired_rows, "count_ratio", "count_ratio_eligible")
    ks_p_med = _med(paired_rows, "participation_ks_p", "participation_ks_eligible")
    lag_span_diff_med = _med(paired_rows, "lag_span_diff", "lag_span_eligible")
    rank_corr_med = _med(paired_rows, "rank_template_corr", "rank_template_eligible")

    chn_bucket = _classify_chn_bucket(chn_jacc_med)
    count_bucket = _classify_count_bucket(count_ratio_med)
    subject_bucket = _assign_subject_bucket(chn_bucket, count_bucket)

    n_total = len(per_record)
    n_paired = sum(1 for r in per_record if r["record_status"] == "paired")
    n_new_only = sum(1 for r in per_record if r["record_status"] == "new_only")
    n_legacy_only = sum(1 for r in per_record if r["record_status"] == "legacy_only")
    n_both_zero = sum(1 for r in paired_rows if r["both_zero"])
    n_zero_new = sum(1 for r in paired_rows if r["zero_event_new"] and not r["zero_event_legacy"])
    n_zero_legacy = sum(1 for r in paired_rows if r["zero_event_legacy"] and not r["zero_event_new"])

    summary = {
        "subject": subject,
        "n_records_total": n_total,
        "n_records_paired": n_paired,
        "n_records_new_only": n_new_only,
        "n_records_legacy_only": n_legacy_only,
        "n_records_both_zero": n_both_zero,
        "n_records_zero_new_only": n_zero_new,
        "n_records_zero_legacy_only": n_zero_legacy,
        "n_eligible_chn_jaccard": sum(
            1 for r in paired_rows if r.get("chn_overlap_jaccard_eligible") is True
        ),
        "n_eligible_count_ratio": sum(
            1 for r in paired_rows if r.get("count_ratio_eligible") is True
        ),
        "n_eligible_ks": sum(
            1 for r in paired_rows if r.get("participation_ks_eligible") is True
        ),
        "n_eligible_lag_span": sum(
            1 for r in paired_rows if r.get("lag_span_eligible") is True
        ),
        "n_eligible_rank": sum(
            1 for r in paired_rows if r.get("rank_template_eligible") is True
        ),
        "chn_overlap_jaccard_med": chn_jacc_med,
        "cov_legacy_in_new_med": cov_legacy_in_new_med,
        "cov_new_in_legacy_med": cov_new_in_legacy_med,
        "count_ratio_med": count_ratio_med,
        "participation_ks_p_med": ks_p_med,
        "lag_span_diff_med": lag_span_diff_med,
        "rank_template_corr_med": rank_corr_med,
        "chn_bucket": chn_bucket,
        "count_bucket": count_bucket,
        "subject_bucket": subject_bucket,
    }
    return per_record, summary


# ---------------------------------------------------------------------------
# Cohort writer
# ---------------------------------------------------------------------------


def _write_per_record_csv(rows: List[Dict[str, object]], path: Path) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            row = {}
            for k, v in r.items():
                if v is None:
                    row[k] = ""
                elif isinstance(v, bool):
                    row[k] = "True" if v else "False"
                elif isinstance(v, float) and math.isnan(v):
                    row[k] = ""
                else:
                    row[k] = v
            writer.writerow(row)


def _write_per_subject_csv(rows: List[Dict[str, object]], path: Path) -> None:
    _write_per_record_csv(rows, path)


def _write_cohort_summary_json(
    per_subject: List[Dict[str, object]],
    path: Path,
    *,
    all_records: Optional[List[Dict[str, object]]] = None,
) -> None:
    buckets: Dict[str, List[str]] = {"stable": [], "moderate_drift": [], "large_drift": []}
    for s in per_subject:
        b = str(s["subject_bucket"])
        buckets.setdefault(b, []).append(str(s["subject"]))

    counts = {k: len(v) for k, v in buckets.items()}
    decision = _decide_stage_d(
        stable=counts.get("stable", 0),
        moderate=counts.get("moderate_drift", 0),
        large=counts.get("large_drift", 0),
    )
    # Cohort-level coverage stats — interpretation is data-driven, not hardcoded.
    cov_l_in_n_meds = [
        s.get("cov_legacy_in_new_med") for s in per_subject
        if s.get("cov_legacy_in_new_med") is not None
    ]
    cov_n_in_l_meds = [
        s.get("cov_new_in_legacy_med") for s in per_subject
        if s.get("cov_new_in_legacy_med") is not None
    ]
    n_subjects_legacy_subset_of_new = sum(
        1 for v in cov_l_in_n_meds
        if v is not None and float(v) >= 0.999
    )
    n_subjects_new_subset_of_legacy = sum(
        1 for v in cov_n_in_l_meds
        if v is not None and float(v) >= 0.999
    )
    # Fraction of paired-eligible records with exact chn-set match.
    if all_records is None:
        n_paired_eligible = 0
        n_exact_chn_match = 0
    else:
        paired_for_match = [
            r for r in all_records
            if r.get("record_status") == "paired"
            and r.get("chn_overlap_jaccard_eligible") is True
            and r.get("cov_legacy_in_new") is not None
            and r.get("cov_new_in_legacy") is not None
        ]
        n_paired_eligible = len(paired_for_match)
        n_exact_chn_match = sum(
            1 for r in paired_for_match
            if float(r["cov_legacy_in_new"]) == 1.0
            and float(r["cov_new_in_legacy"]) == 1.0
        )

    summary = {
        "n_subjects_audited": len(per_subject),
        "bucket_counts": counts,
        "subject_lists": buckets,
        "decision_for_stage_d": decision,
        "decision_rule_reference": (
            "plan §4 Task C.2: enter_full requires stable>=14 AND large<=2; "
            "pause if stable<5 OR large>=8; else enter_smoke."
        ),
        "metric_aggregation_note": (
            "Per-subject medians use only paired records that pass each metric's "
            "*_eligible flag. Zero-event and unpaired records are tracked in the "
            "n_records_* counts but excluded from medians."
        ),
        "channel_coverage_observation": {
            "n_subjects_with_legacy_med_subset_of_new": n_subjects_legacy_subset_of_new,
            "n_subjects_with_new_med_subset_of_legacy": n_subjects_new_subset_of_legacy,
            "n_subjects_total": len(per_subject),
            "cov_legacy_in_new_cohort_median": (
                float(np.median(np.array(cov_l_in_n_meds, dtype=float)))
                if cov_l_in_n_meds else None
            ),
            "cov_new_in_legacy_cohort_median": (
                float(np.median(np.array(cov_n_in_l_meds, dtype=float)))
                if cov_n_in_l_meds else None
            ),
            "n_paired_eligible_records": n_paired_eligible,
            "n_records_with_exact_chn_match": n_exact_chn_match,
            "fraction_records_exact_chn_match": (
                float(n_exact_chn_match) / n_paired_eligible
                if n_paired_eligible else None
            ),
            "field_definitions": (
                "cov_legacy_in_new = |legacy ∩ new| / |legacy| (fraction of legacy "
                "chns kept in new core); cov_new_in_legacy = |legacy ∩ new| / |new| "
                "(fraction of new chns kept in legacy core). Read both directions "
                "before drawing conclusions about asymmetry."
            ),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def run_cohort(subjects: Tuple[str, ...]) -> None:
    AUDIT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_records: List[Dict[str, object]] = []
    all_summaries: List[Dict[str, object]] = []
    for s in subjects:
        print(f"[audit] {s} ...", flush=True)
        per_rec, summary = audit_subject(str(s))
        all_records.extend(per_rec)
        all_summaries.append(summary)
        print(
            f"  bucket={summary['subject_bucket']} chn_med={summary['chn_overlap_jaccard_med']} "
            f"count_med={summary['count_ratio_med']} paired={summary['n_records_paired']}",
            flush=True,
        )

    _write_per_record_csv(all_records, AUDIT_OUTPUT_ROOT / "per_record_audit.csv")
    _write_per_subject_csv(all_summaries, AUDIT_OUTPUT_ROOT / "per_subject_audit.csv")
    _write_cohort_summary_json(
        all_summaries,
        AUDIT_OUTPUT_ROOT / "cohort_audit_summary.json",
        all_records=all_records,
    )
    print(f"[audit] wrote {AUDIT_OUTPUT_ROOT}/", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=str, default=None, help="Audit a single subject")
    parser.add_argument("--all", action="store_true", help="Audit the canonical 20-subject cohort")
    args = parser.parse_args()

    if args.subject:
        per_rec, summary = audit_subject(args.subject)
        AUDIT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        _write_per_record_csv(per_rec, AUDIT_OUTPUT_ROOT / f"per_record_audit_{args.subject}.csv")
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        return

    if args.all:
        run_cohort(COHORT_SUBJECTS)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
