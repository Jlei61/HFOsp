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

from src.rank_displacement import aggregate_pair_metrics  # noqa: E402

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
    """Map cluster_id -> {valid_mask, source, sink, n_valid_channels}."""
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


def process_subject(
    pr2_path: Path,
    pr6_path: Path,
    soz_lookup: Dict[str, Dict[str, set]],
) -> Optional[dict]:
    pr2 = json.loads(pr2_path.read_text())
    if not pr6_path.exists():
        return {
            "subject": pr2.get("subject"),
            "dataset": pr2.get("dataset"),
            "exit_reason": "pr6_missing",
        }
    pr6 = json.loads(pr6_path.read_text())

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

    template_lookup = build_template_lookup(pr6.get("per_template", []), channel_names)
    common_cids = sorted(set(rank_lookup) & set(template_lookup))

    soz_channels: set = soz_lookup.get(dataset, {}).get(str(subject), set())
    if not soz_channels:
        soz_channels = soz_lookup.get(dataset, {}).get(subject, set())

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
        pairs.append(metrics)

    inter_corr = ac.get("inter_cluster_corr_matrix")
    geom = pr6.get("template_pair_geometry") or {}
    h2 = pr6.get("h2_swap_check") or {}

    return {
        "subject": subject,
        "dataset": dataset,
        "stable_k": stable_k,
        "n_channels": len(channel_names),
        "channel_names": channel_names,
        "soz_channels": sorted(soz_channels),
        "fwd_rev_reproduced": fwd_rev,
        "fwd_rev_source": fwd_rev_source,
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
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()
