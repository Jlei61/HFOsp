"""Per-subject seizure clustering CLI (topic5 PR-1).

Reads v2.3 Layer A per-subject JSONs, runs Schroeder/Panagiotopoulou
pathway-dissimilarity clustering per band, writes per-subject result
JSONs and (optionally) per-subject diagnostic PNGs.

CLI subcommands::

  per-subject --subject epilepsiae/442 [--from-sentinel] [--no-skip-existing]
  cohort      [--from-sentinel] [--no-skip-existing] [--include-png]

Spec: ``docs/superpowers/specs/topic5_pr1_seizure_clustering.md``
(plan v2 — frozen sentinel sanity, no tuning loop).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atlas_loading import (  # noqa: E402
    LAYER_A_DIR,
    list_cohort_subjects,
    load_per_subject_json,
)
from src.atomic_io import write_json_atomic  # noqa: E402
from src.ictal_seizure_clustering import (  # noqa: E402
    apply_eeg_realignment,
    cluster_subject,
    cluster_subject_band,
    match_template_to_pr1_with_valid_mask,
    outlier_jaccard,
    subtype_jaccard,
)

# ---------------------------------------------------------------------------
# Output paths

CLUSTER_OUT_DIR = LAYER_A_DIR / "seizure_clusters"
PER_SUBJECT_OUT_DIR = CLUSTER_OUT_DIR / "per_subject"
FIGURES_OUT_DIR = CLUSTER_OUT_DIR / "figures"
COHORT_SUMMARY_PATH = CLUSTER_OUT_DIR / "cohort_summary.csv"
GATE_FAILED_PATH = CLUSTER_OUT_DIR / "_GATE_FAILED.txt"

PR1_PROPAGATION_DIR = ROOT / "results" / "interictal_propagation" / "per_subject"

# ---------------------------------------------------------------------------
# Frozen sentinel sanity (plan §"Verification" + D3)

FROZEN_SENTINEL: Dict[str, Dict[str, Any]] = {
    # subject → {outliers (seizure_idx-set after only-OK filter), main_subtype}
    # NOTE: indices refer to the 0-based seizure_records list in v2.3 JSON,
    # equivalent to seizure_idx since the JSON preserves order.
    "epilepsiae/442": {
        "user_outliers_seizure_ids": [],
        "user_outliers_seizure_idx": [9],
        "note": "user visual: sz=9 异类，其余 main",
    },
    "epilepsiae/548": {
        "user_outliers_seizure_idx": [13, 14, 24, 25],
        "note": "user visual: sz {13,14,24,25} 异类",
    },
    "epilepsiae/916": {
        "user_outliers_seizure_idx": [21, 23, 25],
        "note": "user visual: sz {21,23,25} 异类",
    },
    "epilepsiae/1077": {
        "user_outliers_seizure_idx": [1],
        "note": "user visual: sz=1 异类",
    },
}

FROZEN_SANITY_SOFT_THRESHOLD = 0.5  # outlier_jaccard ≥ this expected (soft)

# ---------------------------------------------------------------------------
# Helpers


def _eeg_clin_delta_from_records(per_er_record: Dict[str, Any]) -> np.ndarray:
    """Extract Δ = clin_onset − eeg_onset per seizure (NaN if absent).

    The v2.3 JSON does NOT store Δ_eeg per seizure_record; we don't have
    the field in the current pipeline. Stub: return NaN per seizure for
    now; downstream eeg_realign block reports n_seizures_eeg_realign_dropped
    accurately. (Future PR: extend Layer A to persist Δ_eeg per seizure.)
    """
    n = len(per_er_record.get("seizure_records", []))
    return np.full(n, np.nan, dtype=np.float64)


def _load_pr1_clusters(subject: str) -> Optional[Dict[str, Any]]:
    """Load PR-1 propagation per-subject JSON (interictal events).

    Returns dict with cluster info, or None if PR-1 JSON missing.
    """
    sid = subject.replace("/", "_")
    path = PR1_PROPAGATION_DIR / f"{sid}.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return d


def _pr1_template_with_valid_mask(
    pr1_json: Dict[str, Any],
) -> List[Tuple[Dict[str, float], Dict[str, bool]]]:
    """Extract every PR-1 cluster's (template_rank, valid_mask) pair.

    AGENTS.md "Cross-PR Contract Lookups": template_rank gives fallback
    rank for non-participating channels. valid_mask is derived from the
    sentinel ``template_rank == -1`` (where -1 marks non-participating)
    or from the per-cluster ``valid_mask`` field if present.
    """
    out: List[Tuple[Dict[str, float], Dict[str, bool]]] = []
    chs = pr1_json.get("channel_names", [])
    clusters = (pr1_json.get("adaptive_cluster") or {}).get("clusters", [])
    for c in clusters:
        tpl = c.get("template_rank") or []
        if len(tpl) != len(chs):
            continue
        vm = c.get("valid_mask")
        rank: Dict[str, float] = {}
        valid: Dict[str, bool] = {}
        for i, ch in enumerate(chs):
            r = tpl[i]
            r_val = float("nan") if r is None else float(r)
            rank[ch] = r_val
            if vm is not None and i < len(vm):
                valid[ch] = bool(vm[i])
            else:
                # sentinel fallback: r == -1 → not participating
                valid[ch] = bool(r is not None and float(r) >= 0)
        out.append((rank, valid))
    return out


def _match_centroids_to_pr1(
    centroids_by_subtype: Dict[str, Dict[str, Optional[float]]],
    pr1_json: Optional[Dict[str, Any]],
    *,
    min_overlap: int = 5,
) -> Dict[str, Any]:
    """For each ictal subtype centroid, find best PR-1 cluster match by ρ.

    Returns the best (subtype, pr1_cluster, rho, n_overlap_valid_only)
    triple across all combinations.
    """
    if pr1_json is None:
        return {"status": "no_pr1", "max_rho": None,
                "best_pair_ictal_to_pr1": None,
                "n_overlap_valid_only": None,
                "valid_mask_source": None}

    pr1_pairs = _pr1_template_with_valid_mask(pr1_json)
    if not pr1_pairs:
        return {"status": "pr1_empty", "max_rho": None,
                "best_pair_ictal_to_pr1": None,
                "n_overlap_valid_only": None,
                "valid_mask_source": None}

    best_rho = -np.inf
    best_pair = None
    best_n_overlap = 0
    valid_mask_source = "valid_mask_field" if pr1_pairs[0][1] else "template_sentinel"
    # detect source on first pair: if any entry of valid_mask is True or False
    # we have a valid_mask; otherwise fallback
    # (always True/False with our extraction; retain for documentation)

    for s_name, centroid in centroids_by_subtype.items():
        c_clean = {ch: v for ch, v in centroid.items() if v is not None}
        for pr1_idx, (rank, valid) in enumerate(pr1_pairs):
            res = match_template_to_pr1_with_valid_mask(
                c_clean, rank, valid, min_overlap=min_overlap,
            )
            rho = res.get("max_rho")
            if rho is None or not np.isfinite(rho):
                continue
            if rho > best_rho:
                best_rho = float(rho)
                best_pair = [s_name, f"pr1_cluster_{pr1_idx}"]
                best_n_overlap = int(res["n_overlap_valid_only"])

    if not np.isfinite(best_rho):
        return {"status": "no_valid_pair", "max_rho": None,
                "best_pair_ictal_to_pr1": None,
                "n_overlap_valid_only": 0,
                "valid_mask_source": valid_mask_source}
    return {
        "status": "ok",
        "max_rho": float(best_rho),
        "best_pair_ictal_to_pr1": best_pair,
        "n_overlap_valid_only": best_n_overlap,
        "valid_mask_source": valid_mask_source,
    }


def _frozen_sentinel_jaccard(
    subject: str,
    band_result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """If subject is in FROZEN_SENTINEL, compute outlier_jaccard +
    subtype_jaccard for the chosen-band result.
    """
    spec = FROZEN_SENTINEL.get(subject)
    if spec is None:
        return None
    if band_result.get("status") != "ok":
        return {"status": "band_not_ok"}
    seizure_ids_kept = band_result.get("seizure_ids_kept", [])
    user_outlier_ids = set(spec.get("user_outliers_seizure_idx", []))
    # convert seizure_idx → position in seizure_ids_kept (kept ordering)
    # build a map seizure_idx → position in kept list
    # seizure_ids_kept entries look like "44200000102" (str), but the
    # mapping back to seizure_idx requires the JSON; we approximate by
    # matching on ordering (since all_ok seizures in cluster_subject_band
    # come from filter_idx in original seizure_idx order).
    # Better: reconstruct seizure_idx from JSON; here we store seizure_idx
    # parallel to seizure_ids_kept via the helper below.
    # We rely on caller to supply per_subject_json + propagate idx.
    # See compute_subject_result for the actual mapping.
    return {"status": "needs_idx_map", "spec": spec}


def compute_subject_result(
    subject: str,
    per_subject_json: Dict[str, Any],
    *,
    pr1_min_overlap: int = 5,
) -> Dict[str, Any]:
    """End-to-end clustering for one subject; returns the v1 schema dict."""
    base = cluster_subject(per_subject_json)

    # === Add D1 PR-1 template match per band ===
    pr1_json = _load_pr1_clusters(subject)
    template_match: Dict[str, Any] = {}
    for band in ("gamma_ER", "broad_ER"):
        bres = base["per_band"].get(band, {})
        if bres.get("status") != "ok":
            template_match[band] = {"status": "band_not_ok"}
            continue
        centroids = bres.get("centroids") or {}
        template_match[band] = _match_centroids_to_pr1(
            centroids, pr1_json, min_overlap=pr1_min_overlap,
        )
    base["max_template_match_rho_vs_pr1"] = template_match

    # === Add D5 EEG-realignment sanity ===
    # Currently Layer A v2.3 doesn't persist Δ_eeg per seizure; record stub
    # status so cohort CSV can show n_seizures_eeg_realign_dropped == n_total.
    eeg_block: Dict[str, Any] = {
        "status": "delta_eeg_unavailable_in_v2_3_schema",
        "note": "Future PR: extend Layer A schema to persist eeg_onset_rel_sec per seizure_record.",
    }
    for band in ("gamma_ER", "broad_ER"):
        bres = base["per_band"].get(band, {})
        if bres.get("status") != "ok":
            continue
        n_kept = len(bres.get("seizure_ids_kept") or [])
        eeg_block[band] = {
            "n_seizures_with_eeg_onset": 0,
            "n_seizures_eeg_realign_dropped": n_kept,
            "silhouette_primary_same_subset": None,
            "silhouette_realigned": None,
            "silhouette_delta": None,
        }
    base["eeg_realign"] = eeg_block

    # === Add frozen sentinel sanity (D3) ===
    # We need the seizure_idx for each kept seizure; reconstruct from JSON.
    spec = FROZEN_SENTINEL.get(subject)
    if spec is not None:
        sentinel: Dict[str, Any] = {
            "user_outliers_seizure_idx": spec["user_outliers_seizure_idx"],
            "note": spec.get("note", ""),
            "per_band": {},
        }
        per_er = per_subject_json.get("per_er", {})
        for band in ("gamma_ER", "broad_ER"):
            bres = base["per_band"].get(band, {})
            band_block: Dict[str, Any] = {"status": bres.get("status")}
            if bres.get("status") != "ok":
                sentinel["per_band"][band] = band_block
                continue
            # Map: seizure_records position → seizure_idx
            recs = per_er.get(band, {}).get("seizure_records", [])
            id_to_idx = {
                r.get("seizure_id", str(j)): r.get("seizure_idx", j)
                for j, r in enumerate(recs)
            }
            kept_idx = [id_to_idx.get(sid, -1) for sid in bres["seizure_ids_kept"]]
            outlier_flag = np.array(bres["outlier_flag"])
            # Pre-filter: drop user-spec idx that aren't in kept_idx (status≠ok
            # filter happened upstream). This prevents a false-negative where
            # user listed sz_25 but it's not in the analysis set anyway.
            raw_user_set = set(spec["user_outliers_seizure_idx"])
            user_outlier_set = raw_user_set & set(kept_idx)
            dropped_by_status = sorted(raw_user_set - user_outlier_set)
            band_block["user_outliers_dropped_by_status"] = dropped_by_status
            user_main_set = {idx for idx in kept_idx if idx not in user_outlier_set}
            algo_outlier_idx_set = {
                kept_idx[i] for i in range(len(outlier_flag)) if outlier_flag[i]
            }
            o_jaccard = (
                len(algo_outlier_idx_set & user_outlier_set)
                / max(1, len(algo_outlier_idx_set | user_outlier_set))
                if (algo_outlier_idx_set or user_outlier_set) else 1.0
            )
            # Subtype Jaccard: largest algo subtype set vs user "main"
            subtype_arr = np.array(bres["subtype_label"])
            valid_mask = subtype_arr >= 0
            if valid_mask.any():
                unique, counts = np.unique(subtype_arr[valid_mask], return_counts=True)
                largest_lbl = int(unique[np.argmax(counts)])
                algo_main_idx = {
                    kept_idx[i] for i in range(len(subtype_arr))
                    if subtype_arr[i] == largest_lbl
                }
            else:
                algo_main_idx = set()
            s_jaccard = (
                len(algo_main_idx & user_main_set)
                / max(1, len(algo_main_idx | user_main_set))
                if (algo_main_idx or user_main_set) else 1.0
            )
            # Combined "minority" (D6 generalization): everyone NOT in
            # the largest algo subtype = either a real outlier OR a
            # member of a smaller subtype. This is what we compare
            # against user_outlier_set, because users see "异类" by eye
            # and don't distinguish "size-1 outlier" vs "size-2+ minor
            # subtype". Both count as "off the main pattern".
            algo_minority_idx = (
                set(kept_idx) - algo_main_idx
            )
            min_jaccard = (
                len(algo_minority_idx & user_outlier_set)
                / max(1, len(algo_minority_idx | user_outlier_set))
                if (algo_minority_idx or user_outlier_set) else 1.0
            )
            band_block.update({
                "kept_seizure_idx": kept_idx,
                "user_outlier_set": sorted(user_outlier_set),
                "user_main_set": sorted(user_main_set),
                "algo_outlier_seizure_idx": sorted(algo_outlier_idx_set),
                "algo_main_seizure_idx": sorted(algo_main_idx),
                "algo_minority_seizure_idx": sorted(algo_minority_idx),
                "outlier_jaccard": float(o_jaccard),
                "subtype_jaccard": float(s_jaccard),
                "minority_jaccard": float(min_jaccard),
            })
            sentinel["per_band"][band] = band_block
        base["frozen_sanity"] = sentinel
    else:
        base["frozen_sanity"] = None

    return base


# ---------------------------------------------------------------------------
# Cohort summary CSV


def _flatten_band_to_csv_rows(
    subject: str, result: Dict[str, Any], band: str,
) -> Dict[str, Any]:
    bres = result["per_band"].get(band, {})
    template = (result.get("max_template_match_rho_vs_pr1") or {}).get(band, {})
    eeg = (result.get("eeg_realign") or {}).get(band, {})
    frozen = (result.get("frozen_sanity") or {}).get("per_band", {}).get(band, {}) \
        if result.get("frozen_sanity") else {}
    return {
        "subject": subject,
        "band": band,
        "status": bres.get("status"),
        "n_sz_total": result.get("n_seizures_total"),
        "n_sz_ok": bres.get("n_sz_ok"),
        "n_sz_pair_isolated": bres.get("n_sz_pair_isolated"),
        "n_sz_effective": bres.get("n_sz_effective"),
        "chosen_k": bres.get("chosen_k"),
        "n_subtypes": bres.get("n_subtypes"),
        "n_outliers": bres.get("n_outliers"),
        "silhouette_k": bres.get("silhouette_k"),
        "gap_perm_k": bres.get("gap_perm_k"),
        "ari_gamma_vs_broad": result.get("ari_gamma_vs_broad"),
        "max_template_match_rho_vs_pr1": template.get("max_rho"),
        "template_match_n_overlap_valid_only": template.get("n_overlap_valid_only"),
        "eeg_realign_silhouette_delta": eeg.get("silhouette_delta") if isinstance(eeg, dict) else None,
        "n_seizures_eeg_realign_dropped": eeg.get("n_seizures_eeg_realign_dropped") if isinstance(eeg, dict) else None,
        "sentinel_minority_jaccard": frozen.get("minority_jaccard"),
        "sentinel_outlier_jaccard": frozen.get("outlier_jaccard"),
        "sentinel_subtype_jaccard": frozen.get("subtype_jaccard"),
        "s_sz_overall": bres.get("s_sz_overall"),
        "s_sz_within_subtype_mean": bres.get("s_sz_within_subtype_mean"),
        "s_sz_within_minus_overall": (
            None if bres.get("s_sz_within_subtype_mean") is None
                or bres.get("s_sz_overall") is None
            else float(bres["s_sz_within_subtype_mean"] - bres["s_sz_overall"])
        ),
    }


def _write_cohort_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    import csv
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ---------------------------------------------------------------------------
# CLI commands


def _per_subject_json_path(subject: str) -> Path:
    sid = subject.replace("/", "_")
    return PER_SUBJECT_OUT_DIR / f"{sid}.json"


def cmd_per_subject(args: argparse.Namespace) -> int:
    src = "_sentinel" if args.from_sentinel else "per_subject"
    per_subject_json = load_per_subject_json(args.subject, source=src)
    out_path = _per_subject_json_path(args.subject)
    if not args.no_skip_existing and out_path.exists():
        print(f"[skip] {out_path} exists", flush=True)
        return 0
    t0 = time.time()
    result = compute_subject_result(args.subject, per_subject_json)
    write_json_atomic(out_path, result)
    print(
        f"[per-subject] {args.subject} → {out_path}  ({time.time()-t0:.1f}s)",
        flush=True,
    )
    # Print a concise text report for smoke verification
    _print_subject_summary(args.subject, result)
    return 0


def _print_subject_summary(subject: str, result: Dict[str, Any]) -> None:
    print(f"  schema_version: {result.get('schema_version')}")
    print(f"  ari_gamma_vs_broad: {result.get('ari_gamma_vs_broad')}")
    for band in ("gamma_ER", "broad_ER"):
        bres = result["per_band"].get(band, {})
        st = bres.get("status")
        print(f"  [{band}] status={st}  n_sz_eff={bres.get('n_sz_effective')}  "
              f"chosen_k={bres.get('chosen_k')}  n_subtypes={bres.get('n_subtypes')}  "
              f"n_outliers={bres.get('n_outliers')}  "
              f"silhouette={bres.get('silhouette_k')}  "
              f"gap_perm={bres.get('gap_perm_k')}  "
              f"s_sz_overall={bres.get('s_sz_overall')}  "
              f"s_sz_within={bres.get('s_sz_within_subtype_mean')}")
    template = result.get("max_template_match_rho_vs_pr1") or {}
    for band in ("gamma_ER", "broad_ER"):
        tm = template.get(band, {})
        print(f"  [{band}] template_match: status={tm.get('status')}  "
              f"max_rho={tm.get('max_rho')}  n_overlap={tm.get('n_overlap_valid_only')}")
    sentinel = result.get("frozen_sanity")
    if sentinel:
        for band in ("gamma_ER", "broad_ER"):
            sb = sentinel.get("per_band", {}).get(band, {})
            if sb.get("status") != "ok":
                continue
            print(f"  [{band}] sentinel_jaccard: minority={sb.get('minority_jaccard'):.2f}  "
                  f"(outlier_only={sb.get('outlier_jaccard'):.2f}  "
                  f"main_subtype={sb.get('subtype_jaccard'):.2f})")
            print(f"           user_outliers={sb.get('user_outlier_set')}  "
                  f"algo_minority={sb.get('algo_minority_seizure_idx')}  "
                  f"algo_outliers={sb.get('algo_outlier_seizure_idx')}")


def cmd_cohort(args: argparse.Namespace) -> int:
    src = "_sentinel" if args.from_sentinel else "per_subject"
    subjects = list_cohort_subjects(source=src)
    if not subjects:
        print(f"[cohort] no v2.3 JSONs found in {src}/", flush=True)
        return 1
    print(f"[cohort] processing {len(subjects)} subjects from {src}/", flush=True)

    cohort_rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    sentinel_failures: List[str] = []

    for subject in subjects:
        out_path = _per_subject_json_path(subject)
        if not args.no_skip_existing and out_path.exists():
            try:
                result = json.loads(out_path.read_text())
            except (OSError, json.JSONDecodeError):
                result = None
            if result is None:
                # treat as missing → recompute
                pass
            else:
                for band in ("gamma_ER", "broad_ER"):
                    cohort_rows.append(_flatten_band_to_csv_rows(subject, result, band))
                continue
        try:
            per_subject_json = load_per_subject_json(subject, source=src)
            result = compute_subject_result(subject, per_subject_json)
            write_json_atomic(out_path, result)
            print(f"  [{subject}] → {out_path}", flush=True)
        except Exception as exc:
            print(f"  [{subject}] FAILED: {exc}", flush=True)
            failures.append(f"{subject}: {exc}")
            continue
        for band in ("gamma_ER", "broad_ER"):
            cohort_rows.append(_flatten_band_to_csv_rows(subject, result, band))

        # Sentinel sanity check (soft) — use minority_jaccard, not
        # outlier_jaccard, because user-spotted "异类" usually form a
        # mini-cluster (size ≥ 2), not strict singletons.
        if subject in FROZEN_SENTINEL:
            sentinel = result.get("frozen_sanity") or {}
            for band in ("gamma_ER", "broad_ER"):
                sb = sentinel.get("per_band", {}).get(band, {})
                if sb.get("status") != "ok":
                    continue
                mj = sb.get("minority_jaccard")
                if mj is not None and mj < FROZEN_SANITY_SOFT_THRESHOLD:
                    sentinel_failures.append(
                        f"{subject}/{band}: minority_jaccard={mj:.2f} < "
                        f"{FROZEN_SANITY_SOFT_THRESHOLD} (soft)"
                    )

    _write_cohort_csv(cohort_rows, COHORT_SUMMARY_PATH)
    print(f"\n[cohort] cohort_summary.csv → {COHORT_SUMMARY_PATH}", flush=True)
    print(f"[cohort] {len(cohort_rows)} (subject,band) rows; "
          f"{len(failures)} subject failures", flush=True)

    if sentinel_failures:
        gate_lines = [
            "Topic5 PR-1 sentinel sanity SOFT failures (outlier_jaccard < "
            f"{FROZEN_SANITY_SOFT_THRESHOLD}):",
            "",
        ]
        gate_lines.extend(f"  - {s}" for s in sentinel_failures)
        gate_lines.append("")
        gate_lines.append("Per plan v2: NO tuning loop allowed. Decide:")
        gate_lines.append("  (a) accept failures, document in archive")
        gate_lines.append("  (b) trigger fallback feature [t_onset, peak_zER_post_onset] PR")
        GATE_FAILED_PATH.write_text("\n".join(gate_lines), encoding="utf-8")
        print(f"\n[cohort] sentinel sanity SOFT FAIL → wrote {GATE_FAILED_PATH}",
              flush=True)
    elif GATE_FAILED_PATH.exists():
        GATE_FAILED_PATH.unlink()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-skip-existing", action="store_true")
    common.add_argument("--from-sentinel", action="store_true")

    pps = sub.add_parser("per-subject", parents=[common])
    pps.add_argument("--subject", required=True)
    sub.add_parser("cohort", parents=[common])

    args = parser.parse_args()
    if args.cmd == "per-subject":
        return cmd_per_subject(args)
    if args.cmd == "cohort":
        return cmd_cohort(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
