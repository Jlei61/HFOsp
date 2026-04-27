#!/usr/bin/env python3
"""PR-6 Template Endpoint Anatomical Anchoring runner.

Step 2 deliverable (audit + per-subject) and Step 3+ (cohort statistics).
Plan: docs/archive/topic1/pr6_template_endpoint_anchoring_plan_2026-04-25.md

Usage:
    # Cohort audit -> cohort_audit.csv
    python scripts/run_pr6_template_anchoring.py --audit

    # Per-subject endpoint/middle SOZ enrichment
    python scripts/run_pr6_template_anchoring.py --per-subject

    # Cohort H1/H2/H3 statistics + dataset-specific sensitivity
    python scripts/run_pr6_template_anchoring.py --cohort

    # Run all three in order
    python scripts/run_pr6_template_anchoring.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interictal_propagation import (
    _valid_event_indices,
    compute_time_split_reproducibility,
    load_subject_propagation_events,
)
from src.template_anatomical_anchoring import (
    NODE_CLASSES,
    audit_subject_eligibility,
    classify_template_pair_nodes,
    cohort_sign_test,
    cohort_wilcoxon,
    compute_split_half_endpoint_jaccards,
    compute_subject_delta,
    compute_template_anchoring,
    compute_template_anchoring_by_coreness,
    compute_template_coreness,
    compute_template_pair_geometry,
    forward_reverse_swap_check,
    soz_breakdown_by_node_class,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PER_SUBJECT_DIR = ROOT / "results" / "interictal_propagation" / "per_subject"
YUQUAN_SOZ_PATH = ROOT / "results" / "yuquan_soz_core_channels.json"
EPILEPSIAE_SOZ_PATH = ROOT / "results" / "epilepsiae_soz_core_channels.json"
EPILEPSIAE_FOCUS_REL_PATH = ROOT / "results" / "epilepsiae_electrode_focus_rel.json"

# Raw lagPat data roots (matches scripts/run_interictal_propagation.py).
# Required to derive per-cluster valid_mask honestly — see _legacy_hist_mean_rank
# fallback `template[ci] = ci` for non-participating channels, which would
# otherwise pollute endpoint/middle extraction with channels that never
# actually appeared in this cluster's events.
YUQUAN_RAW_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_RAW_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")

OUT_DIR = ROOT / "results" / "interictal_propagation" / "template_anchoring"
PER_SUBJECT_OUT = OUT_DIR / "per_subject"
AUDIT_CSV = OUT_DIR / "cohort_audit.csv"
COHORT_SUMMARY = OUT_DIR / "cohort_summary.json"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def _parse_per_subject_filename(stem: str) -> Tuple[str, str]:
    """`yuquan_chenziyang` -> ('yuquan', 'chenziyang'); `epilepsiae_1073` -> ('epilepsiae', '1073')."""
    if stem.startswith("yuquan_"):
        return "yuquan", stem[len("yuquan_") :]
    if stem.startswith("epilepsiae_"):
        return "epilepsiae", stem[len("epilepsiae_") :]
    raise ValueError(f"Unknown dataset prefix in filename: {stem}")


def _subject_raw_dir(dataset: str, subject_id: str) -> Path:
    if dataset == "yuquan":
        return YUQUAN_RAW_ROOT / subject_id
    return EPILEPSIAE_RAW_ROOT / subject_id / "all_recs"


def _load_candidate(path: Path, soz_yuquan: Dict, soz_epi: Dict, focus_rel: Dict) -> Dict[str, Any]:
    data = _load_json(path)
    dataset, subject_id = _parse_per_subject_filename(path.stem)

    ac = data.get("adaptive_cluster") or {}
    stable_k = ac.get("stable_k")
    clusters = ac.get("clusters") or []
    template_ranks = [c.get("template_rank") for c in clusters]
    channel_names = data.get("channel_names") or []
    labels = ac.get("labels")

    soz_dict = soz_yuquan if dataset == "yuquan" else soz_epi
    soz_channels = list(soz_dict.get(subject_id) or [])
    focus_rel_dict = focus_rel.get(subject_id) if dataset == "epilepsiae" else None

    return {
        "subject_id": subject_id,
        "dataset": dataset,
        "stable_k": stable_k,
        "soz_channels": soz_channels,
        "channel_names": channel_names,
        "template_ranks": template_ranks,
        "labels": labels,
        "focus_rel": focus_rel_dict,
        "raw_path": str(path),
        "raw_subject_dir": str(_subject_raw_dir(dataset, subject_id)),
        "inter_cluster_corr_matrix": ac.get("inter_cluster_corr_matrix"),
        "time_split_reproducibility": data.get("time_split_reproducibility") or {},
    }


# ---------------------------------------------------------------------------
# Per-cluster valid_mask from raw events
# ---------------------------------------------------------------------------
def _load_bools_and_channels(subject_dir: Path) -> Optional[Dict[str, Any]]:
    """Inline minimal lagPat loader that prefers `*_lagPat_withFreqCent.npz`
    (the full-channel pipeline used by adaptive_cluster JSONs) and falls back
    to `*_lagPat.npz`.  Returns dict with 'bools' (n_ch, n_events), 'ranks'
    (n_ch, n_events, NaN-padded for non-participating block channels), and
    'channel_names' (union across blocks ordered by first appearance, matching
    the per_subject JSON convention).  ranks is needed by Step 5a coreness
    sensitivity; bools alone is enough for the Step 2 valid_mask path."""
    fc_files = sorted(subject_dir.glob("*_lagPat_withFreqCent.npz"))
    plain_files = sorted(subject_dir.glob("*_lagPat.npz"))
    files = fc_files if fc_files else plain_files
    if not files:
        return None

    channel_names: List[str] = []
    channel_index: Dict[str, int] = {}
    block_records: List[Dict[str, Any]] = []
    for f in files:
        try:
            lp = np.load(f, allow_pickle=True)
        except Exception:
            continue
        if (
            "eventsBool" not in lp.files
            or "chnNames" not in lp.files
            or "lagPatRank" not in lp.files
        ):
            continue
        bools = np.asarray(lp["eventsBool"]) > 0
        ranks = np.asarray(lp["lagPatRank"], dtype=float)
        chns = [str(x) for x in list(lp["chnNames"])]
        start_t = float(lp["start_t"]) if "start_t" in lp.files else float("nan")
        if bools.ndim != 2 or bools.size == 0 or ranks.ndim != 2:
            continue
        n_ev = min(bools.shape[1], ranks.shape[1])
        n_ch = min(bools.shape[0], ranks.shape[0], len(chns))
        bools = bools[:n_ch, :n_ev]
        ranks = ranks[:n_ch, :n_ev]
        chns = chns[:n_ch]
        for ch in chns:
            if ch not in channel_index:
                channel_index[ch] = len(channel_names)
                channel_names.append(ch)
        block_records.append(
            {"bools": bools, "ranks": ranks, "chns": chns, "start_t": start_t}
        )

    if not block_records:
        return None

    # Sort by start_t (matches load_subject_propagation_events)
    block_records.sort(
        key=lambda r: (
            r["start_t"] if np.isfinite(r["start_t"]) else float("inf"),
        )
    )

    n_ch_total = len(channel_names)
    bool_blocks: List[np.ndarray] = []
    rank_blocks: List[np.ndarray] = []
    block_id_arrays: List[np.ndarray] = []
    for block_idx, r in enumerate(block_records):
        big_bools = np.zeros((n_ch_total, r["bools"].shape[1]), dtype=bool)
        big_ranks = np.full((n_ch_total, r["ranks"].shape[1]), np.nan, dtype=float)
        for src_idx, ch in enumerate(r["chns"]):
            tgt_idx = channel_index[ch]
            big_bools[tgt_idx, :] = r["bools"][src_idx, :]
            big_ranks[tgt_idx, :] = r["ranks"][src_idx, :]
        bool_blocks.append(big_bools)
        rank_blocks.append(big_ranks)
        block_id_arrays.append(
            np.full(r["bools"].shape[1], block_idx, dtype=int)
        )

    full_bools = np.concatenate(bool_blocks, axis=1) if bool_blocks else None
    full_ranks = np.concatenate(rank_blocks, axis=1) if rank_blocks else None
    full_block_ids = (
        np.concatenate(block_id_arrays) if block_id_arrays else None
    )
    if full_bools is None or full_ranks is None or full_block_ids is None:
        return None
    return {
        "bools": full_bools,
        "ranks": full_ranks,
        "block_ids": full_block_ids,
        "channel_names": channel_names,
        "source_glob": "withFreqCent" if fc_files else "plain",
    }


def _resolve_cluster_data(
    cand: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Load raw ranks/bools, intersect with valid events used by adaptive
    cluster, and return everything Step 5a needs to derive both per-cluster
    valid_mask AND per-cluster coreness composite.

    Returns dict with:
      ``valid_ranks`` (n_ch, n_valid_events) – NaN for non-participating
      ``valid_bools`` (n_ch, n_valid_events)
      ``labels`` (n_valid_events,)
      ``n_clusters`` (int)
    or ``None`` when raw NPZ missing / channel order mismatch / event-count
    drift relative to JSON labels.
    """
    sd = Path(cand["raw_subject_dir"])
    if not sd.exists():
        return None

    loaded = _load_bools_and_channels(sd)
    if loaded is None:
        return None
    bools = loaded["bools"]
    ranks = loaded["ranks"]
    if (
        bools.ndim != 2
        or bools.size == 0
        or ranks.ndim != 2
        or ranks.shape != bools.shape
    ):
        return None
    raw_channel_names = list(loaded["channel_names"])
    json_channel_names = list(cand["channel_names"])
    if raw_channel_names != json_channel_names:
        return None

    valid_events = _valid_event_indices(bools, min_participating=3)
    labels_raw = cand.get("labels")
    if labels_raw is None:
        return None
    labels = np.asarray(labels_raw, dtype=int)
    if valid_events.size != labels.size:
        return None
    n_clusters = int(np.max(labels)) + 1 if labels.size else 0
    if n_clusters < 2:
        return None

    return {
        "valid_ranks": ranks[:, valid_events],
        "valid_bools": bools[:, valid_events],
        "labels": labels,
        "n_clusters": n_clusters,
        "full_ranks": ranks,
        "full_bools": bools,
        "full_block_ids": loaded["block_ids"],
        "valid_event_indices": valid_events,
    }


def compute_split_half_robustness(
    cand: Dict[str, Any],
    cluster_data: Dict[str, Any],
    n_endpoint: int = 3,
) -> Dict[str, Any]:
    """Step 5b: invoke compute_time_split_reproducibility inline (so the
    Step 1 split-half extension fields are populated for THIS run, not relying
    on legacy per_subject JSONs predating Step 1) and return per-split Jaccard
    summaries.  Synthetic event_abs_times = arange so monotonic-by-construction;
    real block_ids drive odd/even split."""
    full_bools = cluster_data["full_bools"]
    full_ranks = cluster_data["full_ranks"]
    block_ids = cluster_data["full_block_ids"]
    valid_event_indices = cluster_data["valid_event_indices"]
    labels = cluster_data["labels"]
    n_clusters = cluster_data["n_clusters"]
    n_events = full_bools.shape[1]

    event_abs_times = np.arange(n_events, dtype=float)
    try:
        repro = compute_time_split_reproducibility(
            ranks=full_ranks,
            bools=full_bools,
            event_abs_times=event_abs_times,
            block_ids=block_ids,
            chosen_k=n_clusters,
            adaptive_labels=labels,
            valid_event_indices=valid_event_indices,
        )
    except Exception as exc:
        return {"exit_reason": f"split_half_failed:{exc}"}

    out: Dict[str, Any] = {"per_split": {}}
    for split_name, split_rec in (repro.get("splits") or {}).items():
        if not isinstance(split_rec, dict):
            continue
        per_cluster = compute_split_half_endpoint_jaccards(
            channel_names=cand["channel_names"],
            cluster_rank_a=split_rec.get("cluster_rank_a") or [],
            cluster_valid_mask_a=split_rec.get("cluster_valid_mask_a") or [],
            cluster_rank_b_matched_to_a=split_rec.get("cluster_rank_b_matched_to_a")
            or [],
            cluster_valid_mask_b_matched_to_a=split_rec.get(
                "cluster_valid_mask_b_matched_to_a"
            )
            or [],
            n=n_endpoint,
        )
        # Subject-level summary: mean jaccard across non-exit clusters
        valid_clusters = [c for c in per_cluster if c.get("exit_reason") is None]
        if valid_clusters:
            mean_src = float(
                np.mean([c["jaccard_source"] for c in valid_clusters])
            )
            mean_snk = float(
                np.mean([c["jaccard_sink"] for c in valid_clusters])
            )
            mean_ep = float(
                np.mean([c["jaccard_endpoint"] for c in valid_clusters])
            )
        else:
            mean_src = mean_snk = mean_ep = float("nan")
        out["per_split"][split_name] = {
            "n_events_a": split_rec.get("n_events_a"),
            "n_events_b": split_rec.get("n_events_b"),
            "mean_match_corr": split_rec.get("mean_match_corr"),
            "per_cluster": per_cluster,
            "subject_mean_jaccard_source": mean_src,
            "subject_mean_jaccard_sink": mean_snk,
            "subject_mean_jaccard_endpoint": mean_ep,
        }
    return out


def compute_per_cluster_valid_mask(cand: Dict[str, Any]) -> Optional[List[List[bool]]]:
    """Re-derive per-cluster valid_mask = "channel participated in at least one
    event of this cluster".  Backwards-compat entry; new callers should prefer
    `_resolve_cluster_data` to avoid double-loading the raw NPZ."""
    cd = _resolve_cluster_data(cand)
    if cd is None:
        return None
    valid_bools = cd["valid_bools"]
    labels = cd["labels"]
    masks: List[List[bool]] = []
    for k in range(cd["n_clusters"]):
        cluster_mask = labels == k
        if not np.any(cluster_mask):
            masks.append([False] * valid_bools.shape[0])
            continue
        cluster_bools = valid_bools[:, cluster_mask]
        valid_per_channel = np.sum(cluster_bools > 0, axis=1) > 0
        masks.append(valid_per_channel.tolist())
    return masks


def load_all_candidates() -> List[Dict[str, Any]]:
    if not PER_SUBJECT_DIR.exists():
        raise FileNotFoundError(f"Missing {PER_SUBJECT_DIR}")
    soz_yuquan = _load_json(YUQUAN_SOZ_PATH)
    soz_epi = _load_json(EPILEPSIAE_SOZ_PATH)
    focus_rel = _load_json(EPILEPSIAE_FOCUS_REL_PATH)

    paths = sorted(PER_SUBJECT_DIR.glob("*.json"))
    return [_load_candidate(p, soz_yuquan, soz_epi, focus_rel) for p in paths]


# ---------------------------------------------------------------------------
# Step 2a: audit
# ---------------------------------------------------------------------------
AUDIT_CSV_FIELDS = [
    "subject_id",
    "dataset",
    "stable_k",
    "n_ch",
    "n_soz_listed",
    "n_soz_matched",
    "endpoint_defined",
    "h1_primary_eligible",
    "pass",
    "exit_reason",
    "forward_reverse_full_data",
    "forward_reverse_reproduced_split_half",
    "forward_reverse_reproduced_odd_even",
    "forward_reverse_reproduced",
    "has_focus_rel",
]


def _decorate_audit_rows(rows: List[Dict[str, Any]], candidates: List[Dict[str, Any]]):
    """Annotate audit rows with PR-2.5 forward/reverse provenance.

    `forward_reverse_reproduced` follows PR-2.5 accepted rule: split-half OR
    odd/even reproduces.  Older runner only checked split-half and undercounted.
    """
    by_id = {c["subject_id"]: c for c in candidates}
    for row in rows:
        cand = by_id.get(row["subject_id"], {})
        tsr = cand.get("time_split_reproducibility") or {}
        splits = tsr.get("splits") or {}
        fh = splits.get("first_half_second_half") or {}
        oe = splits.get("odd_even_block") or {}
        repro_fh = bool(fh.get("forward_reverse_reproduced") or False)
        repro_oe = bool(oe.get("forward_reverse_reproduced") or False)
        row["forward_reverse_full_data"] = int(tsr.get("full_data_forward_reverse_pairs") or 0) > 0
        row["forward_reverse_reproduced_split_half"] = repro_fh
        row["forward_reverse_reproduced_odd_even"] = repro_oe
        row["forward_reverse_reproduced"] = bool(repro_fh or repro_oe)
        row["has_focus_rel"] = cand.get("focus_rel") is not None


def run_audit() -> List[Dict[str, Any]]:
    candidates = load_all_candidates()
    rows = audit_subject_eligibility(candidates)
    _decorate_audit_rows(rows, candidates)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with AUDIT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_CSV_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in AUDIT_CSV_FIELDS})

    # Console summary
    n_total = len(rows)
    n_pass = sum(1 for r in rows if r["pass"])
    n_endpoint_defined = sum(1 for r in rows if r["endpoint_defined"])
    n_h1 = sum(1 for r in rows if r["h1_primary_eligible"])
    n_fwd_rev = sum(
        1 for r in rows if r["pass"] and r["forward_reverse_reproduced"]
    )
    print(f"[audit] candidates={n_total}")
    print(f"[audit] endpoint_defined (n_ch>=6)={n_endpoint_defined}")
    print(f"[audit] h1_primary_eligible (n_ch>=7)={n_h1}")
    print(f"[audit] pass (=h1_primary_eligible)={n_pass}")
    print(f"[audit] forward_reverse reproduced within pass cohort={n_fwd_rev}")
    by_exit = {}
    for r in rows:
        by_exit.setdefault(r["exit_reason"], 0)
        by_exit[r["exit_reason"]] += 1
    print("[audit] exit_reason distribution:")
    for k, v in sorted(by_exit.items(), key=lambda x: -x[1]):
        print(f"        {str(k):<20} {v}")
    print(f"[audit] csv -> {AUDIT_CSV}")

    return rows


# ---------------------------------------------------------------------------
# Step 2b: per-subject endpoint/middle SOZ enrichment
# ---------------------------------------------------------------------------
def run_per_subject(audit_rows: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    candidates = load_all_candidates()
    by_id = {c["subject_id"]: c for c in candidates}

    if audit_rows is None:
        audit_rows = audit_subject_eligibility(candidates)
        _decorate_audit_rows(audit_rows, candidates)

    PER_SUBJECT_OUT.mkdir(parents=True, exist_ok=True)
    written: List[Dict[str, Any]] = []
    valid_mask_fail_subjects: List[str] = []

    for row in audit_rows:
        # Process both pass (H1) and endpoint_defined-only (case-series); skip
        # subjects where endpoint cannot be extracted at all.
        if not row["endpoint_defined"]:
            continue

        cand = by_id[row["subject_id"]]

        # PR-6 Step 1 review fix: derive per-cluster valid_mask honestly from
        # raw bools.  Without this, _legacy_hist_mean_rank's `template[ci]=ci`
        # fallback for non-participating channels would pollute endpoint/middle.
        # PR-6 Step 5a: also derive per-cluster coreness composite from raw
        # ranks for sensitivity analysis (parallel definition, plan §5.2).
        cluster_data = _resolve_cluster_data(cand)
        per_cluster_masks: List[List[bool]]
        per_cluster_coreness: List[Optional[Dict[str, Any]]]
        if cluster_data is None:
            valid_mask_source = "fallback_all_valid"
            valid_mask_fail_subjects.append(
                f"{cand['dataset']}_{cand['subject_id']}"
            )
            n_ch = len(cand["channel_names"])
            per_cluster_masks = [[True] * n_ch] * len(cand["template_ranks"])
            per_cluster_coreness = [None] * len(cand["template_ranks"])
        else:
            valid_mask_source = "raw_bools"
            valid_bools = cluster_data["valid_bools"]
            labels = cluster_data["labels"]
            n_clusters = cluster_data["n_clusters"]
            per_cluster_masks = []
            for k in range(n_clusters):
                cluster_mask = labels == k
                if not np.any(cluster_mask):
                    per_cluster_masks.append(
                        [False] * valid_bools.shape[0]
                    )
                    continue
                cluster_bools = valid_bools[:, cluster_mask]
                valid_per_channel = np.sum(cluster_bools > 0, axis=1) > 0
                per_cluster_masks.append(valid_per_channel.tolist())
            # Coreness composite per cluster
            per_cluster_coreness = compute_template_coreness(
                cluster_data["valid_ranks"],
                cluster_data["valid_bools"].astype(float),
                labels,
                n_clusters,
            )
        row["valid_mask_source"] = valid_mask_source

        per_template_records = []
        per_template_coreness_records = []
        for k_idx, tr in enumerate(cand["template_ranks"]):
            if tr is None:
                continue
            mask = (
                per_cluster_masks[k_idx]
                if k_idx < len(per_cluster_masks)
                else None
            )
            rec = compute_template_anchoring(
                channel_names=cand["channel_names"],
                template_rank=tr,
                soz_channels=cand["soz_channels"],
                focus_rel_dict=cand["focus_rel"],
                n=3,
                valid_mask=mask,
            )
            rec["cluster_id"] = k_idx
            rec["valid_mask"] = mask
            rec["n_valid_channels"] = (
                int(sum(mask)) if mask is not None else len(cand["channel_names"])
            )
            per_template_records.append(rec)

            # Step 5a coreness sensitivity (only if raw_bools path succeeded)
            coreness_rec = (
                per_cluster_coreness[k_idx]
                if k_idx < len(per_cluster_coreness)
                else None
            )
            if coreness_rec is not None:
                c_rec = compute_template_anchoring_by_coreness(
                    channel_names=cand["channel_names"],
                    coreness_record=coreness_rec,
                    soz_channels=cand["soz_channels"],
                    focus_rel_dict=cand["focus_rel"],
                    n=3,
                )
                c_rec["cluster_id"] = k_idx
                per_template_coreness_records.append(c_rec)

        delta = compute_subject_delta(
            per_template_records, focus_rel=cand["focus_rel"] is not None
        )
        delta_coreness = (
            compute_subject_delta(
                per_template_coreness_records,
                focus_rel=cand["focus_rel"] is not None,
            )
            if per_template_coreness_records
            else None
        )

        # H2 swap mechanism: only meaningful when the subject has a forward/reverse
        # pair AND we have at least 2 templates with valid endpoint extraction.
        swap_check: Optional[Dict[str, Any]] = None
        valid_templates = [
            r for r in per_template_records if r.get("exit_reason") is None
        ]
        if (
            row["forward_reverse_reproduced"]
            and len(valid_templates) >= 2
        ):
            t0 = valid_templates[0]
            t1 = valid_templates[1]
            swap_check = forward_reverse_swap_check(
                t0_source=t0["source"],
                t0_sink=t0["sink"],
                t1_source=t1["source"],
                t1_sink=t1["sink"],
                channel_names=cand["channel_names"],
                n_perm=1000,
                seed=0,
            )

        # Step 5b — split-half endpoint robustness (uses raw bools + ranks +
        # block_ids loaded by _resolve_cluster_data; only runs when cluster_data
        # available, since the synthetic event_abs_times path needs the full
        # bools/ranks arrays)
        split_half_robustness: Optional[Dict[str, Any]] = None
        if cluster_data is not None:
            split_half_robustness = compute_split_half_robustness(
                cand, cluster_data, n_endpoint=3
            )

        # Step 4 (upgraded) — template-pair geometry for all k=2 subjects
        # (not just forward/reverse).  Asks: are T0/T1 two independent
        # networks, two directions of the same network, or sharing high-HI
        # nodes with re-shuffled ordering?
        template_pair_geometry: Optional[Dict[str, Any]] = None
        node_anatomy: Optional[Dict[str, Any]] = None
        if (
            len(cand["template_ranks"]) >= 2
            and cand["template_ranks"][0] is not None
            and cand["template_ranks"][1] is not None
        ):
            template_pair_geometry = compute_template_pair_geometry(
                channel_names=cand["channel_names"],
                t0_template_rank=cand["template_ranks"][0],
                t1_template_rank=cand["template_ranks"][1],
                t0_valid_mask=per_cluster_masks[0]
                if len(per_cluster_masks) > 0
                else None,
                t1_valid_mask=per_cluster_masks[1]
                if len(per_cluster_masks) > 1
                else None,
                n=3,
            )

            # Step 4b — node-level template-pair anatomy.  Use the same
            # source/sink lists that geometry's helper would produce by
            # calling extract_endpoint_middle directly.
            from src.template_anatomical_anchoring import extract_endpoint_middle
            t0_parts = extract_endpoint_middle(
                cand["channel_names"],
                cand["template_ranks"][0],
                n=3,
                valid_mask=per_cluster_masks[0]
                if len(per_cluster_masks) > 0
                else None,
            )
            t1_parts = extract_endpoint_middle(
                cand["channel_names"],
                cand["template_ranks"][1],
                n=3,
                valid_mask=per_cluster_masks[1]
                if len(per_cluster_masks) > 1
                else None,
            )
            if (
                t0_parts.get("exit_reason") is None
                and t1_parts.get("exit_reason") is None
            ):
                node_classification = classify_template_pair_nodes(
                    channel_names=cand["channel_names"],
                    t0_source=t0_parts["source"],
                    t0_sink=t0_parts["sink"],
                    t1_source=t1_parts["source"],
                    t1_sink=t1_parts["sink"],
                )
                soz_breakdown = soz_breakdown_by_node_class(
                    node_classification,
                    soz_channels=cand["soz_channels"],
                    focus_rel_dict=cand["focus_rel"],
                )
                node_anatomy = {
                    "classification": node_classification,
                    "soz_breakdown": soz_breakdown,
                    "swap_minus_same_side_count": (
                        node_classification["counts"]["swap_node"]
                        - node_classification["counts"]["same_side_node"]
                    ),
                }

        out_obj = {
            "subject_id": row["subject_id"],
            "dataset": row["dataset"],
            "audit": row,
            "per_template": per_template_records,
            "subject_delta": delta,
            "h2_swap_check": swap_check,
            "per_template_coreness": per_template_coreness_records,
            "subject_delta_coreness": delta_coreness,
            "split_half_robustness": split_half_robustness,
            "template_pair_geometry": template_pair_geometry,
            "node_anatomy": node_anatomy,
        }
        out_path = PER_SUBJECT_OUT / f"{row['dataset']}_{row['subject_id']}.json"
        with out_path.open("w") as f:
            json.dump(out_obj, f, indent=2, default=_json_default)
        written.append(out_obj)

    print(f"[per-subject] wrote {len(written)} subject json files -> {PER_SUBJECT_OUT}")
    if valid_mask_fail_subjects:
        print(
            f"[per-subject] WARNING: valid_mask fallback applied to "
            f"{len(valid_mask_fail_subjects)} subject(s) (raw lagPat unreachable "
            f"or channel order mismatch): {valid_mask_fail_subjects}"
        )
    return written


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"Object of type {type(o)} not serializable")


# ---------------------------------------------------------------------------
# Step 3: cohort statistics (H1/H1b/H2/H3)
# ---------------------------------------------------------------------------
def run_cohort(per_subject_records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    if per_subject_records is None:
        if not PER_SUBJECT_OUT.exists():
            raise FileNotFoundError(
                f"{PER_SUBJECT_OUT} missing — run --per-subject first"
            )
        per_subject_records = []
        for p in sorted(PER_SUBJECT_OUT.glob("*.json")):
            with p.open() as f:
                per_subject_records.append(json.load(f))

    # H1 cohort: pooled (all h1_primary_eligible) subject deltas
    h1_subjects = [
        r for r in per_subject_records if r["audit"].get("h1_primary_eligible")
    ]
    h1_deltas_pooled = [
        r["subject_delta"]["delta_endpoint_vs_middle"] for r in h1_subjects
    ]
    h1_deltas_yuquan = [
        r["subject_delta"]["delta_endpoint_vs_middle"]
        for r in h1_subjects
        if r["dataset"] == "yuquan"
    ]
    h1_deltas_epi = [
        r["subject_delta"]["delta_endpoint_vs_middle"]
        for r in h1_subjects
        if r["dataset"] == "epilepsiae"
    ]

    h1_pooled = {
        "subject_ids": [r["subject_id"] for r in h1_subjects],
        "wilcoxon_greater": cohort_wilcoxon(h1_deltas_pooled, "greater"),
        "wilcoxon_two_sided": cohort_wilcoxon(h1_deltas_pooled, "two-sided"),
        "sign_test": cohort_sign_test(h1_deltas_pooled),
    }
    h1_yuquan = {
        "n": len(h1_deltas_yuquan),
        "wilcoxon_greater": cohort_wilcoxon(h1_deltas_yuquan, "greater"),
        "sign_test": cohort_sign_test(h1_deltas_yuquan),
    }
    h1_epi = {
        "n": len(h1_deltas_epi),
        "wilcoxon_greater": cohort_wilcoxon(h1_deltas_epi, "greater"),
        "sign_test": cohort_sign_test(h1_deltas_epi),
    }

    # H1b polarity (source vs sink): only on non-forward/reverse subset to avoid
    # cancellation
    h1b_subjects = [
        r
        for r in h1_subjects
        if not r["audit"].get("forward_reverse_reproduced")
    ]
    h1b_deltas = [
        r["subject_delta"]["delta_source_vs_sink"] for r in h1b_subjects
    ]
    h1b = {
        "n": len(h1b_deltas),
        "subject_ids": [r["subject_id"] for r in h1b_subjects],
        "wilcoxon_two_sided": cohort_wilcoxon(h1b_deltas, "two-sided"),
        "sign_test": cohort_sign_test(h1b_deltas),
    }

    # H2 forward/reverse swap mechanism
    h2_subjects = [
        r
        for r in per_subject_records
        if r["audit"].get("forward_reverse_reproduced")
        and r.get("h2_swap_check") is not None
    ]
    h2_records = []
    for r in h2_subjects:
        sc = r["h2_swap_check"]
        h2_records.append(
            {
                "subject_id": r["subject_id"],
                "dataset": r["dataset"],
                "swap_score": sc["swap_score"],
                "null_p": sc["null_p"],
                "null_95th": sc["null_95th"],
                "exceeds_null_95": sc["swap_score"] > sc["null_95th"],
            }
        )
    n_exceed = sum(1 for r in h2_records if r["exceeds_null_95"])
    h2 = {
        "n": len(h2_records),
        "n_exceeds_null_95": n_exceed,
        "per_subject": h2_records,
    }

    # H3 focus_rel (Epilepsiae only): i / l / e endpoint vs middle deltas
    h3_subjects = [
        r
        for r in h1_subjects
        if r["dataset"] == "epilepsiae"
        and r["audit"].get("has_focus_rel")
    ]
    h3 = {}
    for label in ("i", "l", "e"):
        key = f"delta_{label}_endpoint_vs_middle"
        deltas = []
        sids = []
        for r in h3_subjects:
            d = r["subject_delta"].get(key)
            if d is not None and np.isfinite(d):
                deltas.append(d)
                sids.append(r["subject_id"])
        h3[label] = {
            "n": len(deltas),
            "subject_ids": sids,
            "wilcoxon_greater": cohort_wilcoxon(deltas, "greater"),
            "wilcoxon_two_sided": cohort_wilcoxon(deltas, "two-sided"),
            "sign_test": cohort_sign_test(deltas),
        }

    # Step 5a sensitivity: parallel H1 with coreness-defined endpoints.
    # Reports direction agreement, NOT a parallel main metric — see plan §5.2.
    h1_coreness_subjects = [
        r
        for r in h1_subjects
        if r.get("subject_delta_coreness") is not None
        and np.isfinite(
            r["subject_delta_coreness"].get("delta_endpoint_vs_middle", float("nan"))
        )
    ]
    h1_coreness_deltas = [
        r["subject_delta_coreness"]["delta_endpoint_vs_middle"]
        for r in h1_coreness_subjects
    ]
    # Per-subject direction agreement: distinguish three regimes —
    #   same_sign      : both deltas have matching sign (or both ≈ 0)
    #   direction_discordant : strict sign opposition (main * coreness < 0,
    #                          neither is ≈ 0)
    #   one_is_zero    : exactly one of the two is ≈ 0 (technically not
    #                    same-sign, but not "direction-discordant" either)
    pair_records = []
    for r in h1_coreness_subjects:
        d_main = r["subject_delta"]["delta_endpoint_vs_middle"]
        d_core = r["subject_delta_coreness"]["delta_endpoint_vs_middle"]
        eps = 1e-12
        main_zero = abs(d_main) < eps
        core_zero = abs(d_core) < eps
        if main_zero and core_zero:
            same_sign = True
            direction_discordant = False
            one_is_zero = False
        elif main_zero or core_zero:
            same_sign = False
            direction_discordant = False
            one_is_zero = True
        else:
            same_sign = (d_main > 0) == (d_core > 0)
            direction_discordant = not same_sign
            one_is_zero = False
        pair_records.append(
            {
                "subject_id": r["subject_id"],
                "dataset": r["dataset"],
                "delta_main": d_main,
                "delta_coreness": d_core,
                "same_sign": same_sign,
                "direction_discordant": direction_discordant,
                "one_is_zero": one_is_zero,
            }
        )
    n_same_sign = sum(1 for p in pair_records if p["same_sign"])
    n_direction_discordant = sum(1 for p in pair_records if p["direction_discordant"])
    n_one_is_zero = sum(1 for p in pair_records if p["one_is_zero"])
    h1_coreness = {
        "n": len(h1_coreness_deltas),
        "subject_ids": [r["subject_id"] for r in h1_coreness_subjects],
        "wilcoxon_greater": cohort_wilcoxon(h1_coreness_deltas, "greater"),
        "wilcoxon_two_sided": cohort_wilcoxon(h1_coreness_deltas, "two-sided"),
        "sign_test": cohort_sign_test(h1_coreness_deltas),
        "n_same_sign_with_main": n_same_sign,
        "n_direction_discordant": n_direction_discordant,
        "n_one_is_zero": n_one_is_zero,
        "n_with_both_definitions": len(pair_records),
        "per_subject_pairs": pair_records,
    }

    # Step 5b — split-half endpoint robustness Jaccard cohort summary
    split_half_records: Dict[str, List[Dict[str, Any]]] = {
        "first_half_second_half": [],
        "odd_even_block": [],
    }
    for r in h1_subjects:
        shr = r.get("split_half_robustness") or {}
        per_split = shr.get("per_split") or {}
        for split_name, ps in per_split.items():
            if split_name not in split_half_records:
                continue
            mean_ep = ps.get("subject_mean_jaccard_endpoint")
            if mean_ep is None or not np.isfinite(mean_ep):
                continue
            split_half_records[split_name].append(
                {
                    "subject_id": r["subject_id"],
                    "dataset": r["dataset"],
                    "mean_jaccard_source": ps.get("subject_mean_jaccard_source"),
                    "mean_jaccard_sink": ps.get("subject_mean_jaccard_sink"),
                    "mean_jaccard_endpoint": ps.get("subject_mean_jaccard_endpoint"),
                    "mean_match_corr": ps.get("mean_match_corr"),
                }
            )

    split_half_summary: Dict[str, Any] = {}
    for split_name, recs in split_half_records.items():
        if not recs:
            split_half_summary[split_name] = {"n": 0}
            continue
        eps = [r["mean_jaccard_endpoint"] for r in recs]
        srcs = [r["mean_jaccard_source"] for r in recs]
        snks = [r["mean_jaccard_sink"] for r in recs]
        split_half_summary[split_name] = {
            "n": len(recs),
            "median_jaccard_endpoint": float(np.median(eps)),
            "median_jaccard_source": float(np.median(srcs)),
            "median_jaccard_sink": float(np.median(snks)),
            "iqr_jaccard_endpoint": [
                float(np.percentile(eps, 25)),
                float(np.percentile(eps, 75)),
            ],
            "n_subjects_endpoint_jaccard_below_0p4": int(sum(1 for e in eps if e < 0.4)),
            "per_subject": recs,
        }

    # Step 4 (upgraded) — template-pair geometry, stratified
    geom_records = []
    for r in per_subject_records:
        tpg = r.get("template_pair_geometry")
        if not tpg or tpg.get("exit_reason") is not None:
            continue
        # Endpoint-stable subset: split-half subject_mean_jaccard_endpoint >= 0.7
        shr = (r.get("split_half_robustness") or {}).get("per_split", {}) or {}
        fh = shr.get("first_half_second_half") or {}
        ep_stable = (
            fh.get("subject_mean_jaccard_endpoint") is not None
            and np.isfinite(fh.get("subject_mean_jaccard_endpoint", float("nan")))
            and fh["subject_mean_jaccard_endpoint"] >= 0.7
        )
        geom_records.append(
            {
                "subject_id": r["subject_id"],
                "dataset": r["dataset"],
                "h1_eligible": bool(r["audit"].get("h1_primary_eligible")),
                "endpoint_defined": bool(r["audit"].get("endpoint_defined")),
                "forward_reverse_reproduced": bool(
                    r["audit"].get("forward_reverse_reproduced")
                ),
                "endpoint_stable_split_half": bool(ep_stable),
                "jaccard_endpoint": tpg.get("jaccard_endpoint"),
                "jaccard_source_same": tpg.get("jaccard_source_same"),
                "jaccard_sink_same": tpg.get("jaccard_sink_same"),
                "jaccard_source_to_sink": tpg.get("jaccard_source_to_sink"),
                "jaccard_sink_to_source": tpg.get("jaccard_sink_to_source"),
                "swap_score": tpg.get("swap_score"),
                "same_side_score": tpg.get("same_side_score"),
                "spearman_rank_pair": tpg.get("spearman_rank_pair"),
            }
        )

    def _strata_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not records:
            return {"n": 0}
        keys = (
            "jaccard_endpoint",
            "jaccard_source_same",
            "jaccard_sink_same",
            "jaccard_source_to_sink",
            "jaccard_sink_to_source",
            "swap_score",
            "same_side_score",
            "spearman_rank_pair",
        )
        out: Dict[str, Any] = {"n": len(records)}
        for k in keys:
            vals = [
                r[k] for r in records if r.get(k) is not None and np.isfinite(r[k])
            ]
            out[f"median_{k}"] = float(np.median(vals)) if vals else float("nan")
            out[f"iqr_{k}"] = (
                [
                    float(np.percentile(vals, 25)),
                    float(np.percentile(vals, 75)),
                ]
                if vals
                else [float("nan"), float("nan")]
            )
        return out

    geom_h1 = [r for r in geom_records if r["h1_eligible"]]
    geom_fwdrev = [r for r in geom_records if r["forward_reverse_reproduced"]]
    geom_non_fwdrev = [
        r
        for r in geom_records
        if r["h1_eligible"] and not r["forward_reverse_reproduced"]
    ]
    geom_endpoint_stable = [
        r for r in geom_records if r["h1_eligible"] and r["endpoint_stable_split_half"]
    ]
    template_pair_geometry_summary = {
        "all_endpoint_defined": _strata_summary(geom_records),
        "h1_eligible": _strata_summary(geom_h1),
        "forward_reverse_reproduced": _strata_summary(geom_fwdrev),
        "non_forward_reverse_h1_eligible": _strata_summary(geom_non_fwdrev),
        "endpoint_stable_split_half_h1_eligible": _strata_summary(
            geom_endpoint_stable
        ),
        "per_subject": geom_records,
    }

    # Step 4b — node-level anatomy stratified summary
    anatomy_records = []
    for r in per_subject_records:
        na = r.get("node_anatomy")
        if not na:
            continue
        cls = na["classification"]
        soz = na["soz_breakdown"]
        anatomy_records.append(
            {
                "subject_id": r["subject_id"],
                "dataset": r["dataset"],
                "h1_eligible": bool(r["audit"].get("h1_primary_eligible")),
                "endpoint_defined": bool(r["audit"].get("endpoint_defined")),
                "forward_reverse_reproduced": bool(
                    r["audit"].get("forward_reverse_reproduced")
                ),
                "n_swap_node": cls["counts"]["swap_node"],
                "n_same_side_node": cls["counts"]["same_side_node"],
                "n_template_specific_endpoint": cls["counts"][
                    "template_specific_endpoint"
                ],
                "n_shared_endpoint_unassigned": cls["counts"][
                    "shared_endpoint_unassigned"
                ],
                "n_non_endpoint": cls["counts"]["non_endpoint"],
                "swap_minus_same_side_count": na.get("swap_minus_same_side_count"),
                # SOZ counts per class
                "soz_in_swap": soz.get("swap_node", {}).get("n_soz", 0),
                "soz_in_same_side": soz.get("same_side_node", {}).get("n_soz", 0),
                "soz_in_template_specific": soz.get(
                    "template_specific_endpoint", {}
                ).get("n_soz", 0),
                "soz_in_non_endpoint": soz.get("non_endpoint", {}).get("n_soz", 0),
                "frac_soz_swap": soz.get("swap_node", {}).get("frac_soz"),
                "frac_soz_same_side": soz.get("same_side_node", {}).get("frac_soz"),
                "frac_soz_template_specific": soz.get(
                    "template_specific_endpoint", {}
                ).get("frac_soz"),
                "frac_soz_non_endpoint": soz.get("non_endpoint", {}).get(
                    "frac_soz"
                ),
            }
        )

    def _anatomy_strata_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not records:
            return {"n": 0}
        keys_count = (
            "n_swap_node",
            "n_same_side_node",
            "n_template_specific_endpoint",
            "n_shared_endpoint_unassigned",
            "n_non_endpoint",
            "swap_minus_same_side_count",
        )
        out: Dict[str, Any] = {"n": len(records)}
        for k in keys_count:
            vals = [r[k] for r in records if r.get(k) is not None]
            if vals:
                out[f"median_{k}"] = float(np.median(vals))
                out[f"mean_{k}"] = float(np.mean(vals))
                out[f"sum_{k}"] = float(np.sum(vals))
            else:
                out[f"median_{k}"] = float("nan")
        # Pooled SOZ fraction by category (channel-pooled across cohort).
        # KEEP these for transparency, but they are NOT cohort claims — they
        # are channel-count-weighted aggregates and can be dominated by a few
        # high-n_ch subjects.  The valid cohort claims live in the
        # subject-level paired tests below.
        for cat_count, cat_soz in (
            ("n_swap_node", "soz_in_swap"),
            ("n_same_side_node", "soz_in_same_side"),
            ("n_template_specific_endpoint", "soz_in_template_specific"),
            ("n_non_endpoint", "soz_in_non_endpoint"),
        ):
            total_n = sum(int(r.get(cat_count, 0)) for r in records)
            total_soz = sum(int(r.get(cat_soz, 0)) for r in records)
            out[f"pooled_frac_soz_{cat_count.replace('n_', '')}"] = (
                total_soz / total_n if total_n > 0 else float("nan")
            )
            out[f"pooled_n_{cat_count.replace('n_', '')}"] = total_n
            out[f"pooled_n_soz_{cat_count.replace('n_', '')}"] = total_soz

        # === Subject-level paired tests (P0/P1 of Step 4b review) ===
        # 1) Swap-leaning geometry: per-subject `n_swap_node − n_same_side_node`,
        #    then cohort Wilcoxon (greater + two-sided) + sign-test against 0.
        swap_minus_same = [
            (r["subject_id"], int(r["n_swap_node"]) - int(r["n_same_side_node"]))
            for r in records
            if r.get("n_swap_node") is not None
            and r.get("n_same_side_node") is not None
        ]
        deltas_swap_same = [d for _, d in swap_minus_same]
        out["subject_level_swap_minus_same"] = {
            "n": len(swap_minus_same),
            "median": float(np.median(deltas_swap_same)) if deltas_swap_same else float("nan"),
            "n_positive": int(sum(1 for d in deltas_swap_same if d > 0)),
            "n_negative": int(sum(1 for d in deltas_swap_same if d < 0)),
            "n_zero": int(sum(1 for d in deltas_swap_same if d == 0)),
            "wilcoxon_greater": cohort_wilcoxon(deltas_swap_same, "greater"),
            "wilcoxon_two_sided": cohort_wilcoxon(deltas_swap_same, "two-sided"),
            "sign_test": cohort_sign_test(deltas_swap_same),
            "per_subject": [
                {"subject_id": sid, "swap_minus_same_count": d}
                for sid, d in swap_minus_same
            ],
        }

        # 2) Subject-level SOZ-frac swap vs template-specific (the headline
        #    claim that "swap nodes are NOT subject-level SOZ-enriched
        #    relative to template-specific endpoints").  Skip subjects where
        #    either fraction is undefined (zero count in that category).
        soz_pairs = []
        for r in records:
            fs = r.get("frac_soz_swap")
            ft = r.get("frac_soz_template_specific")
            if (
                fs is None
                or ft is None
                or not np.isfinite(fs)
                or not np.isfinite(ft)
            ):
                continue
            soz_pairs.append(
                {
                    "subject_id": r["subject_id"],
                    "frac_soz_swap": float(fs),
                    "frac_soz_template_specific": float(ft),
                    "delta": float(fs) - float(ft),
                }
            )
        deltas_soz = [p["delta"] for p in soz_pairs]
        out["subject_level_soz_swap_vs_template_specific"] = {
            "n": len(soz_pairs),
            "median_delta": float(np.median(deltas_soz)) if deltas_soz else float("nan"),
            "n_positive": int(sum(1 for d in deltas_soz if d > 0)),
            "n_negative": int(sum(1 for d in deltas_soz if d < 0)),
            "n_zero": int(sum(1 for d in deltas_soz if d == 0)),
            "wilcoxon_greater": cohort_wilcoxon(deltas_soz, "greater"),
            "wilcoxon_two_sided": cohort_wilcoxon(deltas_soz, "two-sided"),
            "sign_test": cohort_sign_test(deltas_soz),
            "per_subject": soz_pairs,
        }

        # 3) Same paired test for SOZ swap vs same-side (where both defined).
        soz_pairs_same = []
        for r in records:
            fs = r.get("frac_soz_swap")
            fsa = r.get("frac_soz_same_side")
            if (
                fs is None
                or fsa is None
                or not np.isfinite(fs)
                or not np.isfinite(fsa)
            ):
                continue
            soz_pairs_same.append(
                {
                    "subject_id": r["subject_id"],
                    "frac_soz_swap": float(fs),
                    "frac_soz_same_side": float(fsa),
                    "delta": float(fs) - float(fsa),
                }
            )
        deltas_soz_same = [p["delta"] for p in soz_pairs_same]
        out["subject_level_soz_swap_vs_same_side"] = {
            "n": len(soz_pairs_same),
            "median_delta": float(np.median(deltas_soz_same))
            if deltas_soz_same
            else float("nan"),
            "n_positive": int(sum(1 for d in deltas_soz_same if d > 0)),
            "n_negative": int(sum(1 for d in deltas_soz_same if d < 0)),
            "n_zero": int(sum(1 for d in deltas_soz_same if d == 0)),
            "wilcoxon_greater": cohort_wilcoxon(deltas_soz_same, "greater"),
            "wilcoxon_two_sided": cohort_wilcoxon(deltas_soz_same, "two-sided"),
            "sign_test": cohort_sign_test(deltas_soz_same),
            "per_subject": soz_pairs_same,
        }
        return out

    anatomy_summary = {
        "all_endpoint_defined": _anatomy_strata_summary(anatomy_records),
        "h1_eligible": _anatomy_strata_summary(
            [r for r in anatomy_records if r["h1_eligible"]]
        ),
        "forward_reverse_reproduced": _anatomy_strata_summary(
            [r for r in anatomy_records if r["forward_reverse_reproduced"]]
        ),
        "non_forward_reverse_h1_eligible": _anatomy_strata_summary(
            [
                r
                for r in anatomy_records
                if r["h1_eligible"] and not r["forward_reverse_reproduced"]
            ]
        ),
        "per_subject": anatomy_records,
    }

    summary = {
        "h1_pooled": h1_pooled,
        "h1_dataset_specific": {"yuquan": h1_yuquan, "epilepsiae": h1_epi},
        "h1_coreness_sensitivity": h1_coreness,
        "h1b_polarity_non_fwdrev": h1b,
        "h2_forward_reverse_swap": h2,
        "h3_focus_rel_epilepsiae": h3,
        "split_half_endpoint_robustness": split_half_summary,
        "template_pair_geometry": template_pair_geometry_summary,
        "node_anatomy": anatomy_summary,
        "n_per_subject_records": len(per_subject_records),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with COHORT_SUMMARY.open("w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    print(f"[cohort] H1 pooled n={h1_pooled['wilcoxon_greater']['n']} "
          f"median={h1_pooled['wilcoxon_greater']['median']:.4f} "
          f"p_greater={h1_pooled['wilcoxon_greater']['p_value']:.4g}")
    print(f"[cohort] H1 yuquan   n={h1_yuquan['n']}  | epilepsiae n={h1_epi['n']}")
    print(f"[cohort] H1 coreness n={h1_coreness['n']} "
          f"median={h1_coreness['wilcoxon_greater']['median']:.4f} "
          f"p_greater={h1_coreness['wilcoxon_greater']['p_value']:.4g} "
          f"same_sign={n_same_sign}/{len(pair_records)} "
          f"direction_discordant={n_direction_discordant} "
          f"one_is_zero={n_one_is_zero}")
    print(f"[cohort] H1b polarity (non-fwdrev) n={h1b['n']}")
    print(f"[cohort] H2 swap n={h2['n']} exceeding null_95th={n_exceed}")
    for label, rec in h3.items():
        print(f"[cohort] H3 {label} n={rec['n']} "
              f"median={rec['wilcoxon_greater']['median']}")
    for split_name, ss in split_half_summary.items():
        if ss.get("n", 0) == 0:
            print(f"[cohort] split-half {split_name}: n=0")
            continue
        print(
            f"[cohort] split-half {split_name} n={ss['n']} "
            f"median Jaccard endpoint={ss['median_jaccard_endpoint']:.3f} "
            f"source={ss['median_jaccard_source']:.3f} "
            f"sink={ss['median_jaccard_sink']:.3f} "
            f"endpoint<0.4 subjects={ss['n_subjects_endpoint_jaccard_below_0p4']}"
        )
    # Step 4b — node-level anatomy stratified summary
    print(
        "[cohort] node-level anatomy — stratified "
        "(median counts + subject-level paired tests):"
    )
    for stratum_name in (
        "h1_eligible",
        "forward_reverse_reproduced",
        "non_forward_reverse_h1_eligible",
    ):
        ss = anatomy_summary.get(stratum_name) or {"n": 0}
        if ss.get("n", 0) == 0:
            print(f"  {stratum_name}: n=0")
            continue
        print(
            f"  {stratum_name:<40} n={ss['n']:>3} "
            f"med swap={ss['median_n_swap_node']:.1f} "
            f"med same={ss['median_n_same_side_node']:.1f} "
            f"med tspec={ss['median_n_template_specific_endpoint']:.1f}"
        )
        sl_ss = ss.get("subject_level_swap_minus_same") or {}
        if sl_ss.get("n", 0) > 0:
            print(
                f"    swap−same per subject (n={sl_ss['n']}): "
                f"median={sl_ss['median']:+.1f} "
                f"pos/neg/zero={sl_ss['n_positive']}/{sl_ss['n_negative']}/{sl_ss['n_zero']} "
                f"Wilcoxon-greater p={sl_ss['wilcoxon_greater'].get('p_value')!r} "
                f"sign-test p={sl_ss['sign_test'].get('p_value')!r}"
            )
        sl_soz = ss.get("subject_level_soz_swap_vs_template_specific") or {}
        if sl_soz.get("n", 0) > 0:
            print(
                f"    SOZ frac (swap − tspec) per subject (n={sl_soz['n']}): "
                f"median={sl_soz['median_delta']:+.3f} "
                f"pos/neg/zero={sl_soz['n_positive']}/{sl_soz['n_negative']}/{sl_soz['n_zero']} "
                f"Wilcoxon-greater p={sl_soz['wilcoxon_greater'].get('p_value')!r}"
            )
        print(
            f"    [pooled-only, NOT cohort claim] SOZ frac: swap={ss.get('pooled_frac_soz_swap_node', float('nan')):.3f} "
            f"({ss.get('pooled_n_soz_swap_node', 0)}/{ss.get('pooled_n_swap_node', 0)}), "
            f"same={ss.get('pooled_frac_soz_same_side_node', float('nan')):.3f} "
            f"({ss.get('pooled_n_soz_same_side_node', 0)}/{ss.get('pooled_n_same_side_node', 0)}), "
            f"tspec={ss.get('pooled_frac_soz_template_specific_endpoint', float('nan')):.3f} "
            f"({ss.get('pooled_n_soz_template_specific_endpoint', 0)}/{ss.get('pooled_n_template_specific_endpoint', 0)})"
        )

    # Template-pair geometry stratified summary
    print(
        "[cohort] template-pair geometry — stratified (medians):"
    )
    for stratum_name in (
        "all_endpoint_defined",
        "h1_eligible",
        "forward_reverse_reproduced",
        "non_forward_reverse_h1_eligible",
        "endpoint_stable_split_half_h1_eligible",
    ):
        ss = template_pair_geometry_summary.get(stratum_name) or {"n": 0}
        if ss.get("n", 0) == 0:
            print(f"  {stratum_name}: n=0")
            continue
        print(
            f"  {stratum_name:<40} n={ss['n']:>3} "
            f"endpoint={ss['median_jaccard_endpoint']:+.3f} "
            f"src_same={ss['median_jaccard_source_same']:+.3f} "
            f"snk_same={ss['median_jaccard_sink_same']:+.3f} "
            f"src→snk={ss['median_jaccard_source_to_sink']:+.3f} "
            f"snk→src={ss['median_jaccard_sink_to_source']:+.3f} "
            f"spearman={ss['median_spearman_rank_pair']:+.3f}"
        )
    print(f"[cohort] summary -> {COHORT_SUMMARY}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", action="store_true", help="Step 2a: cohort audit -> cohort_audit.csv")
    ap.add_argument(
        "--per-subject",
        action="store_true",
        help="Step 2b: per-subject endpoint/middle JSON",
    )
    ap.add_argument(
        "--cohort",
        action="store_true",
        help="Step 3: H1/H1b/H2/H3 cohort statistics",
    )
    ap.add_argument("--all", action="store_true", help="Run audit + per-subject + cohort")
    args = ap.parse_args()

    if not (args.audit or args.per_subject or args.cohort or args.all):
        ap.print_help()
        sys.exit(2)

    audit_rows = None
    per_subj_recs = None
    if args.audit or args.all:
        audit_rows = run_audit()
    if args.per_subject or args.all:
        per_subj_recs = run_per_subject(audit_rows)
    if args.cohort or args.all:
        run_cohort(per_subj_recs)


if __name__ == "__main__":
    main()
