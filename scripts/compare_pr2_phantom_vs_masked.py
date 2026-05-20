#!/usr/bin/env python3
"""Step 5a comparison — phantom (original) vs masked PR-2 per-subject results.

Reads paired per-subject JSONs from:
  results/interictal_propagation/per_subject/<sid>.json         (original)
  results/interictal_propagation_masked/per_subject/<sid>.json  (masked)

Emits:
  results/interictal_propagation_vs_masked/pr2_comparison.csv
  results/interictal_propagation_vs_masked/pr2_comparison_summary.md
  results/interictal_propagation_vs_masked/figures/
    cluster_fraction_shift.{png,pdf}
    label_jaccard_distribution.{png,pdf}
    README.md

Per-subject metrics:
  - chosen_k_orig / chosen_k_masked  (and whether they match)
  - stable_k_orig / stable_k_masked
  - cluster fractions (when k matches)
  - **label Jaccard / contingency** (only when k matches): the central
    question — "did 'which event goes in which cluster' change?"
    Jaccard is computed after the optimal cluster-id permutation
    (greedy max-overlap matching), so we measure structural agreement
    rather than label-id agreement.
  - lagpatrank_audit cross-check: load
    results/lagpatrank_audit/<sid>.json and compare its
    `ami_audit_minus_floor` against our per-subject AMI(orig labels,
    masked labels) — they should agree closely (different sampling but
    same underlying signal).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

REPO_ROOT = Path(__file__).resolve().parent.parent
ORIG_DIR = REPO_ROOT / "results" / "interictal_propagation" / "per_subject"
MASK_DIR = REPO_ROOT / "results" / "interictal_propagation_masked" / "per_subject"
AUDIT_DIR = REPO_ROOT / "results" / "lagpatrank_audit"
OUT_DIR = REPO_ROOT / "results" / "interictal_propagation_vs_masked"
STEP0_CSV = REPO_ROOT / "results" / "topic4_attractor" / "step0_audit.csv"


def _load_subject_list() -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    with open(STEP0_CSV) as f:
        for row in csv.DictReader(f):
            out.append((row["sid"], row["dataset"], row["subject"]))
    return out


def _read_pr2(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path) as f:
        full = json.load(f)
    ac = full.get("adaptive_cluster") or {}
    return {
        "chosen_k": ac.get("chosen_k"),
        "stable_k": ac.get("stable_k"),
        "labels": ac.get("labels"),
        "cluster_fractions": [c.get("fraction") for c in ac.get("clusters", [])],
        "n_valid_events": ac.get("n_valid_events"),
    }


def _read_audit_delta(sid: str) -> Optional[float]:
    p = AUDIT_DIR / f"{sid}.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    v = d.get("ami_audit_minus_floor")
    return float(v) if v is not None else None


def _best_perm_match(a: np.ndarray, b: np.ndarray, k: int) -> Dict[str, Any]:
    """Greedy max-overlap permutation of b's labels onto a's labels.

    Returns dict with:
      - mapping: dict mapping orig-b-label -> remapped-label
      - contingency: (k, k) array (rows = a, cols = remapped b)
      - jaccard_per_cluster: list[float]
      - jaccard_macro: mean of per-cluster Jaccards
      - exact_agreement: fraction of events where a == remapped b
    """
    cont = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(k):
            cont[i, j] = int(((a == i) & (b == j)).sum())
    best_perm = None
    best_diag = -1
    # k! permutations; cohort k_max=6 → 720 perms max, cheap.
    for perm in permutations(range(k)):
        diag = sum(cont[i, perm[i]] for i in range(k))
        if diag > best_diag:
            best_diag = diag
            best_perm = perm
    assert best_perm is not None
    mapping = {orig_j: best_perm.index(orig_j) for orig_j in range(k)} if False else {
        j: i for i, j in enumerate(best_perm)
    }
    # The line above is unintuitive — be explicit:
    # best_perm[i] = the original b-label assigned to row i (= a-label i)
    # So mapping from orig-b-label j to remapped label i is: i where best_perm[i] == j
    mapping = {int(best_perm[i]): int(i) for i in range(k)}

    b_remapped = np.array([mapping[int(x)] for x in b], dtype=int)
    cont_remapped = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(k):
            cont_remapped[i, j] = int(((a == i) & (b_remapped == j)).sum())

    jaccs = []
    for k_idx in range(k):
        intersect = int(((a == k_idx) & (b_remapped == k_idx)).sum())
        union = int(((a == k_idx) | (b_remapped == k_idx)).sum())
        jaccs.append(float(intersect / union) if union > 0 else float("nan"))
    macro = float(np.mean([j for j in jaccs if not np.isnan(j)])) if any(not np.isnan(j) for j in jaccs) else float("nan")
    exact = float((a == b_remapped).mean())
    return {
        "mapping": mapping,
        "contingency": cont_remapped.tolist(),
        "jaccard_per_cluster": jaccs,
        "jaccard_macro": macro,
        "exact_agreement": exact,
    }


def _compare_one(sid: str, dataset: str, subject: str) -> Dict[str, Any]:
    orig = _read_pr2(ORIG_DIR / f"{sid}.json")
    mask = _read_pr2(MASK_DIR / f"{sid}.json")
    audit_delta = _read_audit_delta(sid)
    row: Dict[str, Any] = {
        "sid": sid, "dataset": dataset, "subject": subject,
        "audit_ami_audit_minus_floor": audit_delta,
    }
    if orig is None:
        row["status"] = "orig_missing"
        return row
    if mask is None:
        row["status"] = "mask_missing"
        return row
    row["status"] = "ok"
    row["chosen_k_orig"] = orig["chosen_k"]
    row["chosen_k_masked"] = mask["chosen_k"]
    row["chosen_k_changed"] = bool(orig["chosen_k"] != mask["chosen_k"])
    row["stable_k_orig"] = orig["stable_k"]
    row["stable_k_masked"] = mask["stable_k"]
    row["stable_k_changed"] = bool(
        orig.get("stable_k") is not None
        and mask.get("stable_k") is not None
        and orig["stable_k"] != mask["stable_k"]
    )
    row["n_valid_orig"] = orig["n_valid_events"]
    row["n_valid_masked"] = mask["n_valid_events"]

    if (
        orig["chosen_k"] is None or mask["chosen_k"] is None
        or orig["chosen_k"] != mask["chosen_k"]
        or orig["labels"] is None or mask["labels"] is None
    ):
        row["jaccard_macro"] = float("nan")
        row["exact_agreement"] = float("nan")
        row["ami_orig_vs_masked"] = float("nan")
        return row

    a = np.asarray(orig["labels"], dtype=int)
    b = np.asarray(mask["labels"], dtype=int)
    if a.size != b.size:
        row["jaccard_macro"] = float("nan")
        row["exact_agreement"] = float("nan")
        row["ami_orig_vs_masked"] = float("nan")
        row["label_size_mismatch"] = True
        return row

    k = int(orig["chosen_k"])
    match = _best_perm_match(a, b, k)
    row["jaccard_macro"] = match["jaccard_macro"]
    row["exact_agreement"] = match["exact_agreement"]
    row["ami_orig_vs_masked"] = float(adjusted_mutual_info_score(a, b))

    # cluster fraction shift (orig vs masked) for downstream sanity
    orig_fracs = orig.get("cluster_fractions") or []
    mask_fracs = mask.get("cluster_fractions") or []
    if len(orig_fracs) == k and len(mask_fracs) == k:
        # Re-order masked fractions using the same permutation mapping
        remap = match["mapping"]
        mask_fracs_perm = [0.0] * k
        for orig_j, new_i in remap.items():
            mask_fracs_perm[new_i] = mask_fracs[orig_j]
        row["fraction_max_abs_shift"] = float(
            max(abs(o - m) for o, m in zip(orig_fracs, mask_fracs_perm))
        )
        row["fraction_orig_min"] = float(min(orig_fracs))
        row["fraction_masked_min"] = float(min(mask_fracs))
    return row


def _write_summary(rows: List[Dict[str, Any]]) -> str:
    ok = [r for r in rows if r["status"] == "ok"]
    lines = ["# Step 5a — PR-2 phantom vs masked comparison summary", ""]
    lines.append(f"- subjects: total **{len(rows)}**, status=ok **{len(ok)}**")
    lines.append("")

    if not ok:
        lines.append("(no ok comparisons yet — cohort run incomplete)")
        return "\n".join(lines)

    chosen_flips = [r for r in ok if r.get("chosen_k_changed")]
    stable_flips = [r for r in ok if r.get("stable_k_changed")]
    lines.append(f"- **chosen_k flips**: {len(chosen_flips)}/{len(ok)}")
    for r in chosen_flips:
        lines.append(f"  - `{r['sid']}`: chosen_k {r['chosen_k_orig']} → {r['chosen_k_masked']}")
    lines.append(f"- **stable_k flips**: {len(stable_flips)}/{len(ok)}")
    for r in stable_flips:
        lines.append(f"  - `{r['sid']}`: stable_k {r['stable_k_orig']} → {r['stable_k_masked']}")

    # Restrict to subjects where k matches for jaccard / AMI distribution
    matched = [r for r in ok if not r.get("chosen_k_changed") and np.isfinite(r.get("jaccard_macro", np.nan))]
    if matched:
        jaccs = np.array([r["jaccard_macro"] for r in matched])
        amis = np.array([r["ami_orig_vs_masked"] for r in matched])
        exact = np.array([r["exact_agreement"] for r in matched])
        lines.append("")
        lines.append(f"### Label-level shift (subjects where chosen_k unchanged, n={len(matched)})")
        lines.append("")
        lines.append("| metric | median | range |")
        lines.append("|---|---|---|")
        lines.append(f"| Jaccard (macro, best-perm) | {np.median(jaccs):.3f} | [{jaccs.min():.3f}, {jaccs.max():.3f}] |")
        lines.append(f"| exact agreement (best-perm) | {np.median(exact):.3f} | [{exact.min():.3f}, {exact.max():.3f}] |")
        lines.append(f"| AMI(orig labels, masked labels) | {np.median(amis):.3f} | [{amis.min():.3f}, {amis.max():.3f}] |")
        # Cross-check vs audit AMI
        audit_deltas = []
        cross = []
        for r in matched:
            if r.get("audit_ami_audit_minus_floor") is not None:
                audit_deltas.append(r["audit_ami_audit_minus_floor"])
                cross.append(r["ami_orig_vs_masked"])
        if audit_deltas:
            audit_deltas = np.array(audit_deltas)
            cross = np.array(cross)
            from scipy.stats import spearmanr
            rho, p = spearmanr(audit_deltas, cross)
            lines.append("")
            lines.append(
                f"Sanity vs lagpatrank_audit: Spearman ρ(audit Δ, PR-2 AMI) = "
                f"{rho:.3f} (p={p:.2e}, n={len(audit_deltas)}). Same-direction = "
                f"audit and PR-2-level rerun agree on which subjects are most affected."
            )
    return "\n".join(lines)


def _make_figures(rows: List[Dict[str, Any]]) -> None:
    """Two figures: cluster-fraction shift + label Jaccard distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ok = [r for r in rows if r["status"] == "ok"
          and not r.get("chosen_k_changed")
          and np.isfinite(r.get("jaccard_macro", np.nan))]
    if not ok:
        print("(no figures emitted — no matched-k subjects yet)")
        return

    # --- Jaccard distribution ---
    jaccs = np.array([r["jaccard_macro"] for r in ok])
    amis = np.array([r["ami_orig_vs_masked"] for r in ok])
    sk2 = [r for r in ok if r.get("stable_k_orig") == 2]
    skhi = [r for r in ok if r.get("stable_k_orig") is not None and r["stable_k_orig"] > 2]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.hist(jaccs, bins=20, color="#3F7A88", edgecolor="white")
    ax.set_xlabel("Jaccard (macro, best-perm)", fontsize=12)
    ax.set_ylabel("n subjects", fontsize=12)
    ax.set_title(f"PR-2 label-level agreement (n={len(ok)}, k unchanged)", fontsize=13, loc="left")
    ax.axvline(np.median(jaccs), color="0.3", linestyle="--",
               label=f"median = {np.median(jaccs):.2f}")
    ax.legend(loc="upper left", fontsize=10, frameon=False)

    ax2 = axes[1]
    ax2.scatter(
        [r["audit_ami_audit_minus_floor"] for r in ok if r.get("audit_ami_audit_minus_floor") is not None],
        [r["ami_orig_vs_masked"] for r in ok if r.get("audit_ami_audit_minus_floor") is not None],
        c="#3F7A88", s=60, edgecolors="#1F4F5C", linewidths=0.6,
    )
    ax2.set_xlabel("lagpatrank_audit ami_audit_minus_floor (Δ)", fontsize=12)
    ax2.set_ylabel("PR-2-level AMI(orig labels, masked labels)", fontsize=12)
    ax2.set_title("Sanity: audit signal vs PR-2 rerun AMI", fontsize=13, loc="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "label_jaccard_distribution.png", dpi=150)
    fig.savefig(fig_dir / "label_jaccard_distribution.pdf")
    plt.close(fig)

    # --- Cluster fraction shift ---
    fracs_shift = [r.get("fraction_max_abs_shift") for r in ok if r.get("fraction_max_abs_shift") is not None]
    if fracs_shift:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.hist(fracs_shift, bins=20, color="#C77E48", edgecolor="white")
        ax.set_xlabel("max |orig fraction − masked fraction| per subject", fontsize=12)
        ax.set_ylabel("n subjects", fontsize=12)
        ax.set_title(f"Cluster-fraction shift after masked re-rank (n={len(fracs_shift)})", fontsize=13, loc="left")
        ax.axvline(np.median(fracs_shift), color="0.3", linestyle="--",
                   label=f"median = {np.median(fracs_shift):.3f}")
        ax.legend(loc="upper right", fontsize=10, frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "cluster_fraction_shift.png", dpi=150)
        fig.savefig(fig_dir / "cluster_fraction_shift.pdf")
        plt.close(fig)

    # README
    (fig_dir / "README.md").write_text(
        "# Step 5a Phantom vs Masked PR-2 — comparison figures\n\n"
        "Source: `../pr2_comparison.csv` + `../pr2_comparison_summary.md`.\n"
        "Plan: `docs/topic0_methodology_audits.md` §5 + "
        "`docs/archive/topic0/lagpat_phantom_rank/rerun_roadmap_2026-05-20.md`.\n\n"
        "### label_jaccard_distribution.png\n"
        "Left: distribution of PR-2 label-level Jaccard between original "
        "(phantom) and masked clusterings (best-permutation), restricted to "
        "subjects where chosen_k is unchanged. Low values → 'WHICH event in "
        "which cluster' changed substantially. **关注点**：median 是否 < 0.5。\n\n"
        "Right: scatter of `ami_audit_minus_floor` (lagpatrank_audit Step 2) "
        "vs PR-2-level AMI on rerun output. Sanity-check that the two "
        "measurements agree on which subjects are most affected.\n\n"
        "### cluster_fraction_shift.png\n"
        "Per-subject max |orig − masked| cluster fraction after best-perm "
        "matching. Even when chosen_k is unchanged, the cluster size balance "
        "can shift dramatically (e.g. chengshuai: [0.458, 0.542] → [0.609, "
        "0.391]). **关注点**：median shift 是否 > 0.10。\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-progress", action="store_true",
                    help="Allow partial cohort comparison (skip subjects whose "
                         "masked JSON does not exist yet)")
    args = ap.parse_args()

    inv = _load_subject_list()
    rows: List[Dict[str, Any]] = []
    for sid, dataset, subject in inv:
        row = _compare_one(sid, dataset, subject)
        rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cohort_csv = OUT_DIR / "pr2_comparison.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(cohort_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary_md = _write_summary(rows)
    (OUT_DIR / "pr2_comparison_summary.md").write_text(summary_md)
    print(summary_md)

    _make_figures(rows)
    print(f"\nwrote -> {cohort_csv}")
    print(f"wrote -> {OUT_DIR / 'pr2_comparison_summary.md'}")
    print(f"wrote -> {OUT_DIR / 'figures/'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
