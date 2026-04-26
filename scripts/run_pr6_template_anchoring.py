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
    load_subject_propagation_events,
)
from src.template_anatomical_anchoring import (
    audit_subject_eligibility,
    cohort_sign_test,
    cohort_wilcoxon,
    compute_subject_delta,
    compute_template_anchoring,
    forward_reverse_swap_check,
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
    to `*_lagPat.npz`.  Returns dict with 'bools' (n_ch, n_events), and
    'channel_names' (union across blocks ordered by first appearance, matching
    the per_subject JSON convention)."""
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
        if "eventsBool" not in lp.files or "chnNames" not in lp.files:
            continue
        bools = np.asarray(lp["eventsBool"]) > 0
        chns = [str(x) for x in list(lp["chnNames"])]
        start_t = float(lp["start_t"]) if "start_t" in lp.files else float("nan")
        if bools.ndim != 2 or bools.size == 0:
            continue
        n_ch = min(bools.shape[0], len(chns))
        bools = bools[:n_ch, :]
        chns = chns[:n_ch]
        for ch in chns:
            if ch not in channel_index:
                channel_index[ch] = len(channel_names)
                channel_names.append(ch)
        block_records.append({"bools": bools, "chns": chns, "start_t": start_t})

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
    for r in block_records:
        big = np.zeros((n_ch_total, r["bools"].shape[1]), dtype=bool)
        for src_idx, ch in enumerate(r["chns"]):
            big[channel_index[ch], :] = r["bools"][src_idx, :]
        bool_blocks.append(big)
    full_bools = np.concatenate(bool_blocks, axis=1) if bool_blocks else None
    if full_bools is None:
        return None
    return {
        "bools": full_bools,
        "channel_names": channel_names,
        "source_glob": "withFreqCent" if fc_files else "plain",
    }


def compute_per_cluster_valid_mask(cand: Dict[str, Any]) -> Optional[List[List[bool]]]:
    """Re-derive per-cluster valid_mask = "channel participated in at least one
    event of this cluster" from the raw lagPat NPZ.  Returns one bool list per
    cluster, aligned to candidate's channel_names; or None if raw data missing
    or label/event count misaligned (caller decides how to handle)."""
    sd = Path(cand["raw_subject_dir"])
    if not sd.exists():
        return None

    loaded = _load_bools_and_channels(sd)
    if loaded is None:
        return None

    bools = loaded["bools"]
    if bools.ndim != 2 or bools.size == 0:
        return None
    raw_channel_names = list(loaded["channel_names"])
    json_channel_names = list(cand["channel_names"])
    if raw_channel_names != json_channel_names:
        return None

    valid_events = _valid_event_indices(bools, min_participating=3)
    labels = cand.get("labels")
    if labels is None:
        return None
    labels = np.asarray(labels, dtype=int)
    if valid_events.size != labels.size:
        return None

    n_clusters = int(np.max(labels)) + 1 if labels.size else 0
    if n_clusters < 2:
        return None

    valid_bools = bools[:, valid_events]  # (n_ch, n_valid_events)
    masks: List[List[bool]] = []
    for k in range(n_clusters):
        cluster_mask = labels == k
        if not np.any(cluster_mask):
            masks.append([False] * bools.shape[0])
            continue
        cluster_bools = valid_bools[:, cluster_mask]
        # channel participates if at least one event of this cluster activates it
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
        per_cluster_masks = compute_per_cluster_valid_mask(cand)
        valid_mask_source: str
        if per_cluster_masks is None:
            valid_mask_source = "fallback_all_valid"
            valid_mask_fail_subjects.append(
                f"{cand['dataset']}_{cand['subject_id']}"
            )
            # Fallback: treat all channels as valid (matches legacy behaviour
            # but flagged in audit row + summary so reviewers can see)
            n_ch = len(cand["channel_names"])
            per_cluster_masks = [[True] * n_ch] * len(cand["template_ranks"])
        else:
            valid_mask_source = "raw_bools"
        row["valid_mask_source"] = valid_mask_source

        per_template_records = []
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

        delta = compute_subject_delta(
            per_template_records, focus_rel=cand["focus_rel"] is not None
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

        out_obj = {
            "subject_id": row["subject_id"],
            "dataset": row["dataset"],
            "audit": row,
            "per_template": per_template_records,
            "subject_delta": delta,
            "h2_swap_check": swap_check,
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

    summary = {
        "h1_pooled": h1_pooled,
        "h1_dataset_specific": {"yuquan": h1_yuquan, "epilepsiae": h1_epi},
        "h1b_polarity_non_fwdrev": h1b,
        "h2_forward_reverse_swap": h2,
        "h3_focus_rel_epilepsiae": h3,
        "n_per_subject_records": len(per_subject_records),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with COHORT_SUMMARY.open("w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    print(f"[cohort] H1 pooled n={h1_pooled['wilcoxon_greater']['n']} "
          f"median={h1_pooled['wilcoxon_greater']['median']:.4f} "
          f"p_greater={h1_pooled['wilcoxon_greater']['p_value']:.4g}")
    print(f"[cohort] H1 yuquan   n={h1_yuquan['n']}  | epilepsiae n={h1_epi['n']}")
    print(f"[cohort] H1b polarity (non-fwdrev) n={h1b['n']}")
    print(f"[cohort] H2 swap n={h2['n']} exceeding null_95th={n_exceed}")
    for label, rec in h3.items():
        print(f"[cohort] H3 {label} n={rec['n']} "
              f"median={rec['wilcoxon_greater']['median']}")
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
