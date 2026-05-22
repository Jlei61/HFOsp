#!/usr/bin/env python3
"""Per-subject signed rank displacement runner.

Plan: docs/archive/topic1/pr6_supplementary_rank_displacement_plan_2026-05-06.md

Inputs (read from canonical repo root, even when running from a worktree):
  - results/interictal_propagation/per_subject/<dataset>_<subject>.json (PR-2)
  - results/interictal_propagation/template_anchoring/per_subject/<dataset>_<subject>.json (PR-6)
  - results/<dataset>_soz_core_channels.json

Outputs (written to canonical repo root):
  - results/interictal_propagation/rank_displacement/per_subject/<stem>.json
  - results/interictal_propagation/rank_displacement/cohort_summary.json

Sign anchor (plan §3.0): T_a = cluster with smaller cluster_id.
Forward/reverse-reproduced flag: OR rule (split-half OR odd-even),
per CLAUDE.md cross-PR contract lookup.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _canonical_data_root() -> Path:
    """Main repo root; canonical location of shared (gitignored) results/."""
    here = Path(__file__).resolve().parent
    common = subprocess.check_output(
        ["git", "-C", str(here), "rev-parse", "--git-common-dir"],
        text=True,
    ).strip()
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = (here / common_path).resolve()
    return common_path.parent


# Worktree root for src/ imports (rank_displacement.py is on the worktree branch)
WORKTREE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE_ROOT))

from src.rank_displacement import (  # noqa: E402
    aggregate_pair_metrics,
    cohort_sign_test_enrichment,
    compute_clinical_soz_set_relation,
    compute_swap_score_sweep,
    derive_swap_endpoint,
)

# Data root for shared results (read PR-2/PR-6 inputs, write rank_displacement outputs)
DATA_ROOT = _canonical_data_root()
PR2_DIR = DATA_ROOT / "results" / "interictal_propagation" / "per_subject"
PR6_DIR = (
    DATA_ROOT / "results" / "interictal_propagation" / "template_anchoring" / "per_subject"
)
OUT_DIR = DATA_ROOT / "results" / "interictal_propagation" / "rank_displacement"
OUT_PER_SUBJECT = OUT_DIR / "per_subject"
SOZ_FILES = {
    "epilepsiae": DATA_ROOT / "results" / "epilepsiae_soz_core_channels.json",
    "yuquan": DATA_ROOT / "results" / "yuquan_soz_core_channels.json",
}


def _apply_masked_paths() -> None:
    """Reassign module-level path globals to the `_masked` parallel tree.

    Step 5f.4 of the Topic 0 phantom-rank rerun roadmap: rank displacement
    consumes the masked PR-2 cluster JSONs + masked PR-6 template_anchoring
    per-subject JSONs; writes a parallel rank_displacement_masked tree. Pure
    path-routing change (no compute helper needs use_masked_features).
    """
    global PR2_DIR, PR6_DIR, OUT_DIR, OUT_PER_SUBJECT
    PR2_DIR = DATA_ROOT / "results" / "interictal_propagation_masked" / "per_subject"
    PR6_DIR = (
        DATA_ROOT
        / "results"
        / "interictal_propagation_masked"
        / "template_anchoring"
        / "per_subject"
    )
    OUT_DIR = (
        DATA_ROOT / "results" / "interictal_propagation_masked" / "rank_displacement"
    )
    OUT_PER_SUBJECT = OUT_DIR / "per_subject"


def load_soz_lookup() -> Dict[str, Dict[str, set]]:
    out: Dict[str, Dict[str, set]] = {}
    for ds, path in SOZ_FILES.items():
        if not path.exists():
            out[ds] = {}
            continue
        raw = json.loads(path.read_text())
        out[ds] = {sub: set(chs or []) for sub, chs in raw.items()}
    return out


def derive_fwd_rev_reproduced(pr2: dict) -> Optional[bool]:
    splits = pr2.get("time_split_reproducibility", {}).get("splits", {})
    fhsh = splits.get("first_half_second_half", {}).get("forward_reverse_reproduced")
    oeb = splits.get("odd_even_block", {}).get("forward_reverse_reproduced")
    flags = [v for v in (fhsh, oeb) if isinstance(v, bool)]
    if not flags:
        return None
    return any(flags)


def derive_fwd_rev_source(pr2: dict) -> str:
    splits = pr2.get("time_split_reproducibility", {}).get("splits", {})
    fhsh = splits.get("first_half_second_half", {}).get("forward_reverse_reproduced")
    oeb = splits.get("odd_even_block", {}).get("forward_reverse_reproduced")
    if fhsh and oeb:
        return "both"
    if fhsh:
        return "first_half_second_half"
    if oeb:
        return "odd_even_block"
    if fhsh is False or oeb is False:
        return "none"
    return "unknown"


def build_template_lookup(per_template: list, channel_names_master: list) -> Dict[int, Dict]:
    """Map cluster_id -> {valid_mask, source, sink, n_valid_channels} from PR-6."""
    out: Dict[int, Dict] = {}
    for entry in per_template:
        cid = entry.get("cluster_id")
        valid_mask = entry.get("valid_mask")
        if cid is None or valid_mask is None:
            continue
        if len(valid_mask) != len(channel_names_master):
            raise ValueError(
                f"PR-6 valid_mask length {len(valid_mask)} != "
                f"channel_names {len(channel_names_master)}"
            )
        out[int(cid)] = {
            "valid_mask": np.asarray(valid_mask, dtype=bool),
            "source": entry.get("source", []),
            "sink": entry.get("sink", []),
            "n_valid_channels": int(entry.get("n_valid_channels", sum(valid_mask))),
        }
    return out


def build_template_lookup_from_pr2(
    rank_lookup: Dict[int, np.ndarray],
    n_channels: int,
) -> Dict[int, Dict]:
    """Derive valid_mask from PR-2 template_rank sentinels (rank != -1).

    Used when PR-6 anchoring JSON is missing for a subject (e.g., dropped
    by PR-6 because of empty SOZ JSON, even though PR-2 stable_k=2 holds).
    Verified to match PR-6 valid_mask exactly on subjects where both exist.
    """
    out: Dict[int, Dict] = {}
    for cid, rank in rank_lookup.items():
        valid_mask = np.asarray(np.asarray(rank) != -1, dtype=bool)
        out[cid] = {
            "valid_mask": valid_mask,
            "source": [],
            "sink": [],
            "n_valid_channels": int(valid_mask.sum()),
        }
    return out


def _empty_set_relation(exit_reason: str) -> dict:
    return {
        "soz_source": "clinical",
        "n_E": None, "n_S": None, "n_L": None, "n_E_inter_S": None,
        "precision": None, "recall_within_lagPat": None,
        "coverage": None, "lagpat_baseline": None,
        "enrichment_over_lagPat": None,
        "typology": None, "informative": False,
        "exit_reason": exit_reason,
    }


def _compute_pair_set_relation(
    metrics: dict,
    soz_channels: set,
    soz_present: bool,
) -> dict:
    """Per-pair clinical SOZ set-relationship readout.

    Drops endpoint derivation when swap_sweep aborted; returns exit_reason
    accordingly. Spec §1: SOZ JSON 缺 subject → exit_reason='no_clinical_soz';
    swap_sweep ok with non-trivial decision_k → compute set_relation.
    """
    if not soz_present:
        return _empty_set_relation("no_clinical_soz")

    sw = metrics.get("swap_sweep") or {}
    if sw.get("exit_reason") != "ok" or sw.get("decision_k") is None:
        return _empty_set_relation("swap_sweep_unavailable")

    joint_valid = np.asarray(metrics["joint_valid"], dtype=bool)
    channel_names = metrics["channel_names"]
    rank_a_dense_full = np.asarray(metrics["rank_a_dense_full"], dtype=float)
    joint_chs = [ch for ch, v in zip(channel_names, joint_valid) if v]
    joint_dense = rank_a_dense_full[joint_valid]
    decision_k = int(sw["decision_k"])
    if 2 * decision_k > len(joint_chs):
        return _empty_set_relation("decision_k_exceeds_n_valid")

    endpoint_chs = derive_swap_endpoint(
        channel_names=joint_chs,
        rank_a_dense=joint_dense,
        decision_k=decision_k,
    )
    soz_in_lagpat = list(soz_channels & set(joint_chs))
    return compute_clinical_soz_set_relation(
        valid_chs=joint_chs,
        endpoint_chs=endpoint_chs,
        soz_chs=soz_in_lagpat,
    )


def process_subject(
    pr2_path: Path,
    pr6_path: Path,
    soz_lookup: Dict[str, Dict[str, set]],
) -> Optional[dict]:
    """Consume PR-2 cluster JSON. PR-6 anchoring JSON is OPTIONAL — when
    missing, valid_mask is derived from PR-2 template_rank sentinels
    (rank != -1). This lets the supplementary cohort include all
    stable_k=2 subjects regardless of PR-6 SOZ-driven exclusions.
    """
    pr2 = json.loads(pr2_path.read_text())
    dataset = pr2.get("dataset")
    subject = pr2.get("subject")
    channel_names = pr2.get("channel_names")
    if channel_names is None:
        return {
            "subject": subject,
            "dataset": dataset,
            "exit_reason": "no_channel_names",
        }

    ac = pr2.get("adaptive_cluster", {})
    stable_k = ac.get("stable_k")
    clusters = ac.get("clusters", [])
    rank_lookup: Dict[int, np.ndarray] = {}
    for c in clusters:
        cid = c.get("cluster_id")
        tr = c.get("template_rank")
        if cid is None or tr is None:
            continue
        if len(tr) != len(channel_names):
            raise ValueError(
                f"{subject}: template_rank length {len(tr)} != "
                f"channel_names {len(channel_names)}"
            )
        rank_lookup[int(cid)] = np.asarray(tr, dtype=float)

    # Build valid_mask from PR-6 if available, else fall back to PR-2 sentinel.
    if pr6_path.exists():
        pr6 = json.loads(pr6_path.read_text())
        template_lookup = build_template_lookup(pr6.get("per_template", []), channel_names)
        valid_mask_provenance = "pr6"
    else:
        pr6 = None
        template_lookup = build_template_lookup_from_pr2(rank_lookup, len(channel_names))
        valid_mask_provenance = "pr2_sentinel"

    common_cids = sorted(set(rank_lookup) & set(template_lookup))

    soz_dict_for_dataset = soz_lookup.get(dataset, {})
    soz_present = (str(subject) in soz_dict_for_dataset) or (subject in soz_dict_for_dataset)
    soz_channels: set = soz_dict_for_dataset.get(str(subject), set())
    if not soz_channels:
        soz_channels = soz_dict_for_dataset.get(subject, set())

    fwd_rev = derive_fwd_rev_reproduced(pr2)
    fwd_rev_source = derive_fwd_rev_source(pr2)

    pairs = []
    # combinations(sorted_iterable, 2) yields (smaller, larger), so cluster_id_a
    # is always the smaller cluster_id - this is the T_a anchor (plan §3.0).
    for cid_a, cid_b in combinations(common_cids, 2):
        rank_a = rank_lookup[cid_a]
        rank_b = rank_lookup[cid_b]
        v_a = template_lookup[cid_a]["valid_mask"]
        v_b = template_lookup[cid_b]["valid_mask"]
        metrics = aggregate_pair_metrics(
            rank_a=rank_a,
            rank_b=rank_b,
            valid_mask_a=v_a,
            valid_mask_b=v_b,
            channel_names=channel_names,
            soz_channels=soz_channels,
        )
        metrics["cluster_id_a"] = int(cid_a)
        metrics["cluster_id_b"] = int(cid_b)
        metrics["swap_sweep"] = compute_swap_score_sweep(
            rank_a=rank_a,
            rank_b=rank_b,
            valid_mask_a=v_a,
            valid_mask_b=v_b,
            n_perm=1000,
            seed=0,
        )
        metrics["clinical_soz_set_relation"] = _compute_pair_set_relation(
            metrics=metrics,
            soz_channels=soz_channels,
            soz_present=soz_present,
        )
        pairs.append(metrics)

    inter_corr = ac.get("inter_cluster_corr_matrix")
    geom = (pr6 or {}).get("template_pair_geometry") or {}
    h2 = (pr6 or {}).get("h2_swap_check") or {}

    return {
        "subject": subject,
        "dataset": dataset,
        "stable_k": stable_k,
        "n_channels": len(channel_names),
        "channel_names": channel_names,
        "soz_channels": sorted(soz_channels),
        "fwd_rev_reproduced": fwd_rev,
        "fwd_rev_source": fwd_rev_source,
        "valid_mask_provenance": valid_mask_provenance,
        "pr6_available": pr6 is not None,
        "pr6_swap_score": h2.get("swap_score"),
        "pr6_swap_null_p": h2.get("null_p"),
        "pr6_pair_geometry_spearman": geom.get("spearman_rank_pair"),
        "inter_cluster_corr_matrix": inter_corr,
        "pairs": pairs,
        "exit_reason": "ok",
    }


def _slim_for_cohort(record: dict) -> dict:
    """Drop heavy per-channel arrays; keep summary fields for cohort_summary."""
    if record is None:
        return record
    out = {k: v for k, v in record.items() if k not in {
        "channel_names", "inter_cluster_corr_matrix",
    }}
    slim_pairs = []
    for p in record.get("pairs", []):
        slim_pairs.append({
            k: v for k, v in p.items()
            if k not in {
                "rank_a_full", "rank_b_full",
                "rank_a_dense_full", "rank_b_dense_full",
                "signed_displacement_full", "signed_displacement_dense",
                "joint_valid", "soz_mask", "channel_names",
            }
        })
    out["pairs"] = slim_pairs
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional list of <dataset>_<subject> stems; default = all PR-2 JSONs.",
    )
    ap.add_argument(
        "--masked-features",
        action="store_true",
        help=(
            "Topic 0 Step 5f.4 masked rerun: read PR-2 + PR-6 JSONs from "
            "results/interictal_propagation_masked/{per_subject,template_anchoring/per_subject}/ "
            "and write to results/interictal_propagation_masked/rank_displacement/."
        ),
    )
    args = ap.parse_args()

    if args.masked_features:
        _apply_masked_paths()
        print(
            "[main] --masked-features: rank_displacement paths routed to "
            "results/interictal_propagation_masked/rank_displacement/"
        )

    OUT_PER_SUBJECT.mkdir(parents=True, exist_ok=True)
    soz_lookup = load_soz_lookup()

    if args.subjects:
        stems = args.subjects
    else:
        stems = sorted(
            p.stem for p in PR2_DIR.glob("*.json") if not p.stem.startswith("pr")
        )

    cohort: List[dict] = []
    for stem in stems:
        pr2_path = PR2_DIR / f"{stem}.json"
        pr6_path = PR6_DIR / f"{stem}.json"
        if not pr2_path.exists():
            continue
        result = process_subject(pr2_path, pr6_path, soz_lookup)
        if result is None:
            continue
        out_path = OUT_PER_SUBJECT / f"{stem}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        cohort.append(_slim_for_cohort(result))
        print(
            f"[{stem}] stable_k={result.get('stable_k')} "
            f"fwd_rev={result.get('fwd_rev_reproduced')} "
            f"n_pairs={len(result.get('pairs', []))} "
            f"exit={result.get('exit_reason')}"
        )

    cohort_path = OUT_DIR / "cohort_summary.json"
    cohort_path.write_text(json.dumps(cohort, indent=2, ensure_ascii=False))
    print(f"\nWrote {cohort_path} with {len(cohort)} subjects")

    # §9 clinical SOZ set-relationship summary
    soz_summary = compute_clinical_soz_summary(cohort)
    soz_summary_path = OUT_DIR / "clinical_soz_set_relation_summary.json"
    soz_summary_path.write_text(json.dumps(soz_summary, indent=2, ensure_ascii=False))
    print(
        f"Wrote {soz_summary_path}: strict_informative n="
        f"{soz_summary['strict_informative']['n_informative']}, "
        f"sign_p={soz_summary['strict_informative']['sign_test_p']}, "
        f"median_enrichment={soz_summary['strict_informative']['median_enrichment']}"
    )


def compute_clinical_soz_summary(cohort: list) -> dict:
    """Aggregate per-subject set_relation across cohort.

    Per spec §3 cohort divisioning:
      Primary cohort     = strict ∩ informative
      Sensitivity cohort = candidate ∩ informative
      Descriptive only   = degenerate (any class) + none ∩ informative

    Reports:
      - per-tier × per-typology counts (typology_distribution)
      - strict_informative / candidate_informative sign test + bootstrap CI
      - per-subject array (for figure: precision/recall/coverage/enrichment + tier)
    """
    rows = []  # [{dataset, subject, swap_class, set_rel, ...}]
    for record in cohort:
        if record.get("exit_reason") != "ok" or not record.get("pairs"):
            continue
        primary_pair = record["pairs"][0]
        sw = primary_pair.get("swap_sweep") or {}
        sr = primary_pair.get("clinical_soz_set_relation") or {}
        rows.append({
            "dataset": record.get("dataset"),
            "subject": record.get("subject"),
            "swap_class": sw.get("swap_class", "unknown"),
            "decision_k": sw.get("decision_k"),
            "set_rel": sr,
        })

    typology_distribution = {}  # tier -> typology -> count
    for tier in ("strict", "candidate", "none"):
        typology_distribution[tier] = {
            t: 0 for t in ("E_subset_S", "S_subset_E", "partial", "disjoint", "degenerate", "missing")
        }
    for r in rows:
        tier = r["swap_class"]
        if tier not in typology_distribution:
            continue
        sr = r["set_rel"] or {}
        if sr.get("exit_reason"):
            typology_distribution[tier]["missing"] += 1
        else:
            t = sr.get("typology") or "missing"
            typology_distribution[tier][t] = typology_distribution[tier].get(t, 0) + 1

    def _enr_for(tier: str) -> list:
        out = []
        for r in rows:
            if r["swap_class"] != tier:
                continue
            sr = r["set_rel"] or {}
            if sr.get("exit_reason"):
                continue
            if not sr.get("informative"):
                continue
            v = sr.get("enrichment_over_lagPat")
            if v is None:
                continue
            out.append(float(v))
        return out

    strict_enr = _enr_for("strict")
    candidate_enr = _enr_for("candidate")

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_subjects_total": len(rows),
        "typology_distribution": typology_distribution,
        "strict_informative": cohort_sign_test_enrichment(
            enrichments=strict_enr, n_boot=2000, seed=0,
        ),
        "candidate_informative": cohort_sign_test_enrichment(
            enrichments=candidate_enr, n_boot=2000, seed=0,
        ),
        "per_subject": [
            {
                "dataset": r["dataset"],
                "subject": r["subject"],
                "swap_class": r["swap_class"],
                "decision_k": r["decision_k"],
                **(r["set_rel"] or {}),
            }
            for r in rows
        ],
    }


if __name__ == "__main__":
    main()
