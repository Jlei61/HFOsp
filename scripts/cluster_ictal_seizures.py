"""Per-subject seizure clustering CLI (topic5 PR-1).

Reads v2.3 Layer A per-subject JSONs, runs Schroeder/Panagiotopoulou
pathway-dissimilarity clustering per band, writes per-subject result
JSONs and (optionally) per-subject 4-panel summary PNGs.

CLI subcommands::

  per-subject --subject epilepsiae/442 [--from-sentinel] [--no-skip-existing]
  cohort      [--from-sentinel] [--no-skip-existing]
  render      [--subject epilepsiae/442 | --all] [--feature-mode zer_binned]

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
    _empty_band_result,
    apply_eeg_realignment,
    cluster_subject,
    cluster_subject_band,
    cluster_subject_band_zer,
    match_template_to_pr1_with_valid_mask,
    outlier_jaccard,
    subtype_jaccard,
)
from src.ictal_zer_features import (  # noqa: E402
    DEFAULT_BINS_SEC,
    extract_zer_binned_for_subject,
    stack_features_to_matrix,
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
        "over_split_flag": bres.get("over_split_flag"),
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


def _per_subject_json_path(subject: str, *, feature_mode: str = "t_onset") -> Path:
    sid = subject.replace("/", "_")
    if feature_mode == "t_onset":
        return PER_SUBJECT_OUT_DIR / f"{sid}.json"
    if feature_mode == "zer_binned":
        return PER_SUBJECT_OUT_DIR / f"{sid}__zer_binned.json"
    raise ValueError(f"unknown feature_mode={feature_mode}")


def compute_subject_result_zer(
    subject: str,
    per_subject_json: Dict[str, Any],
) -> Dict[str, Any]:
    """End-to-end: extract z-ER binned features per band, cluster, output.

    Sentinel sanity (D6 + minority_jaccard) is computed identically to
    the t_onset path, but on the new feature's subtype labels.
    """
    out: Dict[str, Any] = {
        "schema_version": "topic5_pr1_seizure_clustering_zer_v1",
        "subject": subject,
        "n_seizures_total": per_subject_json.get("n_seizures_total"),
        "feature_mode": "zer_binned",
        "bins_sec": [list(b) for b in DEFAULT_BINS_SEC],
        "per_band": {},
    }

    n_bins = len(DEFAULT_BINS_SEC)
    per_er = per_subject_json.get("per_er", {})

    band_sz_indices: Dict[str, List[int]] = {}
    band_results: Dict[str, Any] = {}

    for band in ("gamma_ER", "broad_ER"):
        rec = per_er.get(band)
        if not rec:
            band_results[band] = {"band": band, "status": "missing_band"}
            continue
        # Collect ok seizure indices (other status → skip pre-extraction)
        sz_idx_ok = [
            int(r["seizure_idx"])
            for r in rec.get("seizure_records", [])
            if r.get("status") == "ok"
        ]
        band_sz_indices[band] = sz_idx_ok
        if len(sz_idx_ok) < 5:
            band_results[band] = _empty_band_result(
                band, "insufficient_n",
                n_sz_total=len(rec.get("seizure_records", [])),
                n_sz_ok=len(sz_idx_ok),
            )
            continue

        print(f"  [{band}] extracting z-ER for {len(sz_idx_ok)} ok seizures...",
               flush=True)
        t0 = time.time()
        features, sids, ch_names, drop_reasons = extract_zer_binned_for_subject(
            subject, band, sz_idx_ok,
        )
        feature_matrix, kept_ids = stack_features_to_matrix(features, sids)
        n_dropped = sum(1 for r in drop_reasons if r)
        print(f"  [{band}] z-ER extracted in {time.time()-t0:.1f}s "
               f"({n_dropped} dropped)", flush=True)

        if feature_matrix.shape[1] < 5:
            stub = _empty_band_result(
                band, "insufficient_n_after_extraction",
                n_sz_total=len(rec.get("seizure_records", [])),
                n_sz_ok=len(sz_idx_ok),
                n_sz_effective=int(feature_matrix.shape[1]),
            )
            stub["n_kept_after_extraction"] = int(feature_matrix.shape[1])
            band_results[band] = stub
            continue

        band_block = cluster_subject_band_zer(
            feature_matrix, kept_ids,
            band=band,
            channels_for_centroid=ch_names,
            n_bins=n_bins,
        )
        band_results[band] = band_block

    out["per_band"] = band_results

    # ARI between two feature_mode bands (within zer_binned cohort)
    g = band_results.get("gamma_ER", {})
    b = band_results.get("broad_ER", {})
    ari = None
    if (g.get("status") == "ok" and b.get("status") == "ok"
            and g.get("subtype_label") and b.get("subtype_label")):
        from sklearn.metrics import adjusted_rand_score
        g_ids = g.get("seizure_ids_kept", [])
        b_ids = b.get("seizure_ids_kept", [])
        common_ids = [s for s in g_ids if s in b_ids]
        if len(common_ids) >= 4:
            g_lookup = {sid: lbl for sid, lbl in zip(g_ids, g["subtype_label"])}
            b_lookup = {sid: lbl for sid, lbl in zip(b_ids, b["subtype_label"])}
            g_lab = np.array([g_lookup[s] for s in common_ids])
            b_lab = np.array([b_lookup[s] for s in common_ids])
            try:
                ari = float(adjusted_rand_score(g_lab, b_lab))
            except Exception:
                ari = None
    out["ari_gamma_vs_broad"] = ari

    # Cross-feature ARI: compare with existing t_onset result if present
    t_onset_path = _per_subject_json_path(subject, feature_mode="t_onset")
    out["ari_zer_vs_t_onset"] = {"per_band": {}}
    if t_onset_path.exists():
        try:
            t_onset_result = json.loads(t_onset_path.read_text())
            from sklearn.metrics import adjusted_rand_score
            for band in ("gamma_ER", "broad_ER"):
                g = band_results.get(band, {})
                t_b = t_onset_result.get("per_band", {}).get(band, {})
                if (g.get("status") == "ok" and t_b.get("status") == "ok"
                        and g.get("subtype_label") and t_b.get("subtype_label")):
                    g_ids = g.get("seizure_ids_kept", [])
                    t_ids = t_b.get("seizure_ids_kept", [])
                    common = [s for s in g_ids if s in t_ids]
                    if len(common) >= 4:
                        g_lookup = {sid: lbl for sid, lbl in zip(g_ids, g["subtype_label"])}
                        t_lookup = {sid: lbl for sid, lbl in zip(t_ids, t_b["subtype_label"])}
                        g_lab = np.array([g_lookup[s] for s in common])
                        t_lab = np.array([t_lookup[s] for s in common])
                        out["ari_zer_vs_t_onset"]["per_band"][band] = {
                            "ari": float(adjusted_rand_score(g_lab, t_lab)),
                            "n_common": len(common),
                        }
                    else:
                        out["ari_zer_vs_t_onset"]["per_band"][band] = {
                            "ari": None, "n_common": len(common),
                        }
        except (OSError, json.JSONDecodeError):
            pass

    # Sentinel sanity (D3 + D6) — same logic as t_onset path but on zer subtypes
    spec = FROZEN_SENTINEL.get(subject)
    if spec is not None:
        sentinel: Dict[str, Any] = {
            "user_outliers_seizure_idx": spec["user_outliers_seizure_idx"],
            "note": spec.get("note", ""),
            "per_band": {},
        }
        for band in ("gamma_ER", "broad_ER"):
            bres = band_results.get(band, {})
            band_block: Dict[str, Any] = {"status": bres.get("status")}
            if bres.get("status") != "ok":
                sentinel["per_band"][band] = band_block
                continue
            recs = per_er.get(band, {}).get("seizure_records", [])
            id_to_idx = {
                r.get("seizure_id", str(j)): r.get("seizure_idx", j)
                for j, r in enumerate(recs)
            }
            kept_idx = [id_to_idx.get(sid, -1) for sid in bres["seizure_ids_kept"]]
            outlier_flag = np.array(bres["outlier_flag"])
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
            algo_minority_idx = set(kept_idx) - algo_main_idx
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
        out["frozen_sanity"] = sentinel
    else:
        out["frozen_sanity"] = None

    return out


def cmd_per_subject(args: argparse.Namespace) -> int:
    src = "_sentinel" if args.from_sentinel else "per_subject"
    per_subject_json = load_per_subject_json(args.subject, source=src)
    fm = args.feature_mode
    out_path = _per_subject_json_path(args.subject, feature_mode=fm)
    if not args.no_skip_existing and out_path.exists():
        print(f"[skip] {out_path} exists", flush=True)
        return 0
    t0 = time.time()
    if fm == "t_onset":
        result = compute_subject_result(args.subject, per_subject_json)
    elif fm == "zer_binned":
        result = compute_subject_result_zer(args.subject, per_subject_json)
    else:
        raise ValueError(f"unknown feature_mode={fm}")
    write_json_atomic(out_path, result)
    print(
        f"[per-subject:{fm}] {args.subject} → {out_path}  ({time.time()-t0:.1f}s)",
        flush=True,
    )
    _print_subject_summary(args.subject, result)
    return 0


def _print_subject_summary(subject: str, result: Dict[str, Any]) -> None:
    print(f"  schema_version: {result.get('schema_version')}")
    print(f"  ari_gamma_vs_broad: {result.get('ari_gamma_vs_broad')}")
    cross = result.get("ari_zer_vs_t_onset")
    if cross:
        for band, info in (cross.get("per_band") or {}).items():
            print(f"  [{band}] ari_zer_vs_t_onset = {info.get('ari')}  "
                  f"(n_common={info.get('n_common')})")
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
        if tm:
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
    fm = args.feature_mode
    subjects = list_cohort_subjects(source=src)
    if not subjects:
        print(f"[cohort] no v2.3 JSONs found in {src}/", flush=True)
        return 1
    print(f"[cohort] processing {len(subjects)} subjects from {src}/  "
          f"feature_mode={fm}", flush=True)

    cohort_rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    sentinel_failures: List[str] = []

    for subject in subjects:
        out_path = _per_subject_json_path(subject, feature_mode=fm)
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
            if fm == "t_onset":
                result = compute_subject_result(subject, per_subject_json)
            elif fm == "zer_binned":
                result = compute_subject_result_zer(subject, per_subject_json)
            else:
                raise ValueError(f"unknown feature_mode={fm}")
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

    out_csv = (
        COHORT_SUMMARY_PATH if fm == "t_onset"
        else COHORT_SUMMARY_PATH.with_name(
            COHORT_SUMMARY_PATH.stem + f"__{fm}.csv"
        )
    )
    _write_cohort_csv(cohort_rows, out_csv)
    print(f"\n[cohort] cohort_summary.csv → {out_csv}", flush=True)
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


# ---------------------------------------------------------------------------
# Per-subject 4-panel render (Step 11)


PER_SUBJECT_PNG_DIR = FIGURES_OUT_DIR / "per_subject"


def _render_band_block(
    fig, gs_band, *, band: str, bres: Dict[str, Any],
    per_er_record: Dict[str, Any], onset_kept: np.ndarray,
    show_xlabel: bool,
) -> None:
    """Render one band's 4-panel block (left: dendro+heatmap, right: MDS+t_ER).

    gs_band is a 2x2 SubplotSpec with height_ratios=[1, 3] for left col
    (dendro thin, heatmap thick) and [1, 1] for right col (MDS, t_ER).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import scipy.cluster.hierarchy as sch
    from matplotlib.colors import TwoSlopeNorm
    from src.ictal_seizure_plotting import (
        compute_mds_2d,
        subtype_color_palette,
        subtype_sort_indices,
        channel_sort_by_subtype_means,
    )

    # --- subgrids: 2x2 with custom row heights for left vs right ---
    sub_left = gs_band[:, 0].subgridspec(2, 1, height_ratios=[1, 3], hspace=0.05)
    sub_right = gs_band[:, 1].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.30)

    ax_dendro = fig.add_subplot(sub_left[0, 0])
    ax_heat = fig.add_subplot(sub_left[1, 0])
    ax_mds = fig.add_subplot(sub_right[0, 0])
    ax_ter = fig.add_subplot(sub_right[1, 0])

    if bres.get("status") != "ok":
        for ax, label in [
            (ax_dendro, "dendrogram"), (ax_heat, "pairwise heatmap"),
            (ax_mds, "MDS"), (ax_ter, "cluster-stratified t_ER"),
        ]:
            ax.text(0.5, 0.5,
                    f"{band}\nstatus={bres.get('status')}\n[{label}: n/a]",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, color="#7f8c8d")
            ax.axis("off")
        return

    D = np.asarray(bres["D"], dtype=np.float64)
    Z = np.asarray(bres["Z"], dtype=np.float64)
    sids_kept = list(bres["seizure_ids_kept"])
    subtype_label = list(bres["subtype_label"])
    outlier_flag = list(bres["outlier_flag"])
    n_sub = int(bres["n_subtypes"])

    # Map seizure_id -> seizure_idx using per_er record
    id_to_idx: Dict[str, int] = {}
    for r in per_er_record.get("seizure_records", []):
        id_to_idx[str(r["seizure_id"])] = int(r["seizure_idx"])
    sz_idx_kept = [id_to_idx.get(sid, -1) for sid in sids_kept]

    cols, outlier_color = subtype_color_palette(max(n_sub, 1))
    point_colors = [
        outlier_color if outlier_flag[i] or subtype_label[i] < 0
        else cols[int(subtype_label[i]) % len(cols)]
        for i in range(len(subtype_label))
    ]

    # --- LEFT: dendrogram ---
    dendro = sch.dendrogram(
        Z, ax=ax_dendro, no_labels=True, color_threshold=0.0,
        above_threshold_color="#34495e",
    )
    leaf_order = dendro["leaves"]
    ax_dendro.set_title(
        f"{band}: dendrogram (UPGMA on 1−Spearman)",
        fontsize=10,
    )
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])
    for spine in ("top", "right", "left"):
        ax_dendro.spines[spine].set_visible(False)

    # --- LEFT: sorted pairwise heatmap, in dendrogram leaf order ---
    D_sorted = D[np.ix_(leaf_order, leaf_order)]
    im = ax_heat.imshow(D_sorted, cmap="viridis", vmin=0, vmax=2,
                          aspect="equal", interpolation="nearest", origin="upper")
    n = D_sorted.shape[0]
    ax_heat.set_xticks(np.arange(n))
    ax_heat.set_yticks(np.arange(n))
    leaf_labels = [str(sz_idx_kept[i]) for i in leaf_order]
    ax_heat.set_xticklabels(leaf_labels, fontsize=6, rotation=90)
    ax_heat.set_yticklabels(leaf_labels, fontsize=6)
    ax_heat.set_xlabel("seizure_idx (dendrogram order)" if show_xlabel else "",
                        fontsize=9)
    # Color tick labels by subtype
    for i, ti in enumerate(leaf_order):
        c = point_colors[ti]
        ax_heat.get_xticklabels()[i].set_color(c)
        ax_heat.get_yticklabels()[i].set_color(c)
    cb = fig.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02)
    cb.set_label("1−Spearman", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # --- RIGHT TOP: MDS scatter ---
    if n >= 2:
        emb = compute_mds_2d(D, random_state=0)
        for i in range(n):
            marker = "X" if (outlier_flag[i] or subtype_label[i] < 0) else "o"
            ax_mds.scatter(emb[i, 0], emb[i, 1], c=point_colors[i], s=80,
                            marker=marker, edgecolors="black", linewidths=0.5)
            ax_mds.text(emb[i, 0], emb[i, 1] + 0.01, str(sz_idx_kept[i]),
                          fontsize=6, ha="center", va="bottom")
    else:
        ax_mds.text(0.5, 0.5, "n<2: no MDS",
                    transform=ax_mds.transAxes, ha="center", va="center")
    ax_mds.set_title(
        f"{band}: MDS-2D  (k={bres['chosen_k']}, sil={bres['silhouette_k']:.2f}, "
        f"gap={bres['gap_perm_k']:.2f})", fontsize=10,
    )
    ax_mds.set_xticks([]); ax_mds.set_yticks([])
    # legend (subtypes only, outliers as 'X')
    handles = []
    for s in range(n_sub):
        handles.append(mpatches.Patch(color=cols[s], label=f"subtype {s}"))
    if any(outlier_flag) or any(s < 0 for s in subtype_label):
        handles.append(mpatches.Patch(color=outlier_color, label="outlier"))
    ax_mds.legend(handles=handles, loc="best", fontsize=7, framealpha=0.7)

    # --- RIGHT BOTTOM: cluster-stratified t_ER matrix ---
    sort_idx = subtype_sort_indices(subtype_label, outlier_flag)
    onset_sorted = onset_kept[:, sort_idx]
    ch_order = channel_sort_by_subtype_means(onset_kept, subtype_label)
    onset_view = onset_sorted[ch_order, :]
    norm = TwoSlopeNorm(vmin=-120.0, vcenter=0.0, vmax=30.0)
    ax_ter.imshow(onset_view, aspect="auto", cmap="RdBu_r", norm=norm,
                    interpolation="nearest")
    nan_mask = np.isnan(onset_view)
    if nan_mask.any():
        gray_layer = np.zeros((*onset_view.shape, 4))
        gray_layer[nan_mask] = (0.84, 0.84, 0.84, 1.0)
        ax_ter.imshow(gray_layer, aspect="auto", interpolation="nearest")
    n_ch_eff = onset_view.shape[0]
    n_sz_eff = onset_view.shape[1]
    ax_ter.set_xticks(np.arange(n_sz_eff))
    sz_labels = [str(sz_idx_kept[i]) for i in sort_idx]
    sz_label_colors = [point_colors[i] for i in sort_idx]
    ax_ter.set_xticklabels(sz_labels, fontsize=6, rotation=90)
    for tk, c in zip(ax_ter.get_xticklabels(), sz_label_colors):
        tk.set_color(c)
    ax_ter.set_yticks([])  # too many channels to label
    ax_ter.set_ylabel(f"{n_ch_eff} channels (sorted by main subtype t_onset)",
                        fontsize=8)
    ax_ter.set_xlabel("seizure_idx (grouped by subtype)" if show_xlabel else "",
                        fontsize=9)
    ax_ter.set_title(f"{band}: cluster-stratified t_ER_onset (sec)",
                       fontsize=10)
    # Vertical separators between subtype groups
    cur = 0
    seen = []
    for i in sort_idx:
        s = subtype_label[i] if not outlier_flag[i] and subtype_label[i] >= 0 else -1
        if s not in seen:
            seen.append(s)
            if cur > 0:
                ax_ter.axvline(cur - 0.5, color="black", linewidth=1.0, alpha=0.6)
        cur += 1


def render_subject_pdf(
    subject: str, result: Dict[str, Any],
    per_subject_json: Dict[str, Any], out_path: Path,
) -> Path:
    """Render the per-subject 4-panel summary (gamma top, broad bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.atlas_loading import build_onset_matrix

    fig = plt.figure(figsize=(20, 22), facecolor="white")
    fig.suptitle(
        f"{subject}  |  topic5 PR-1 z-ER seizure subtyping  "
        f"(n_sz_total={result.get('n_seizures_total')})",
        fontsize=14, y=0.998,
    )
    outer = fig.add_gridspec(2, 1, hspace=0.18, top=0.97, bottom=0.02,
                              left=0.04, right=0.99)
    for row, band in enumerate(("gamma_ER", "broad_ER")):
        bres = result["per_band"].get(band, {})
        per_er_rec = per_subject_json["per_er"][band]
        # Build onset matrix on union channels of seizure_records, then slice to kept
        channels = list(per_er_rec.get("channels") or [])
        if not channels:
            ch_set = set()
            for r in per_er_rec.get("seizure_records", []):
                ch_set.update((r.get("channel_onsets") or {}).keys())
            channels = sorted(ch_set)
        onset_full, statuses, sz_ids_full = build_onset_matrix(per_er_rec, channels)
        # Slice columns to kept seizure_ids in their result-order
        sids_kept = list(bres.get("seizure_ids_kept") or [])
        if sids_kept and bres.get("status") == "ok":
            sid_to_col = {sid: j for j, sid in enumerate(sz_ids_full)}
            kept_cols = [sid_to_col.get(s) for s in sids_kept if s in sid_to_col]
            onset_kept = onset_full[:, kept_cols]
        else:
            onset_kept = onset_full
        # Each band block is its own 2x2 sub-gridspec (rows=2, cols=2)
        gs_band = outer[row].subgridspec(2, 2, hspace=0.30, wspace=0.18,
                                            width_ratios=[1, 1])
        _render_band_block(
            fig, gs_band, band=band, bres=bres,
            per_er_record=per_er_rec, onset_kept=onset_kept,
            show_xlabel=(row == 1),  # only bottom band labels x
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def cmd_render(args: argparse.Namespace) -> int:
    src = "_sentinel" if args.from_sentinel else "per_subject"
    fm = args.feature_mode
    suffix = "" if fm == "t_onset" else "__zer_binned"
    if args.all:
        subjects = list_cohort_subjects(source=src)
    elif args.subject:
        subjects = [args.subject]
    else:
        print("ERROR: render requires --subject or --all", file=sys.stderr)
        return 2
    out_dir = PER_SUBJECT_PNG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    n_done, n_skip, n_fail = 0, 0, 0
    for subj in subjects:
        sid = subj.replace("/", "_")
        json_path = _per_subject_json_path(subj, feature_mode=fm)
        if not json_path.exists():
            print(f"  [{subj}] SKIP: cluster JSON missing ({json_path.name})",
                  flush=True)
            n_skip += 1
            continue
        try:
            result = json.loads(json_path.read_text())
            per_subj = load_per_subject_json(subj, source=src)
            out_path = out_dir / f"{sid}{suffix}.png"
            render_subject_pdf(subj, result, per_subj, out_path)
            print(f"  [{subj}] → {out_path.name}", flush=True)
            n_done += 1
        except Exception as exc:
            print(f"  [{subj}] FAILED: {exc}", flush=True)
            n_fail += 1
    print(f"\n[render] done={n_done}, skipped={n_skip}, failed={n_fail}",
          flush=True)
    return 0 if n_fail == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-skip-existing", action="store_true")
    common.add_argument("--from-sentinel", action="store_true")
    common.add_argument("--feature-mode", default="t_onset",
                         choices=("t_onset", "zer_binned"),
                         help="Clustering feature: per-channel t_onset (default) "
                              "or per-channel × time-bin z-ER mean tensor")

    pps = sub.add_parser("per-subject", parents=[common])
    pps.add_argument("--subject", required=True)
    sub.add_parser("cohort", parents=[common])
    rps = sub.add_parser("render", parents=[common])
    rps.add_argument("--subject", default=None)
    rps.add_argument("--all", action="store_true")

    args = parser.parse_args()
    if args.cmd == "per-subject":
        return cmd_per_subject(args)
    if args.cmd == "cohort":
        return cmd_cohort(args)
    if args.cmd == "render":
        return cmd_render(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
