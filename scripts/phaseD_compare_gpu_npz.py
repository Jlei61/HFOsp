#!/usr/bin/env python3
"""Phase D: A/B compare new (legacy_align=True) `*_gpu.npz` vs legacy truth.

For each requested (subject, block) pair, compares:
- total events_count (sum-level drift)
- per-channel events_count (Pearson r, MAE, p95 abs ratio)
- top-K named channels' ratios (gaolan: B'13, B'14, A'10 vs D/D')
- channel-set overlap (Jaccard) after legacy-style left-contact alias

Outputs a markdown report.

Usage:
    python scripts/phaseD_compare_gpu_npz.py \
      --new-root results/hfo_detection \
      --legacy-root /mnt/yuquan_data/yuquan_24h_edf \
      --subject gaolan \
      --block FA0013KP \
      --report results/validation/phaseD/gaolan_FA0013KP_ab.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _bipolar_to_left_alias(name: str) -> str:
    """A1-A2 -> A1; A1 -> A1."""
    nm = str(name).strip()
    return nm.split("-", 1)[0].strip().upper() if "-" in nm else nm.upper()


def _load_gpu_npz(path: Path) -> Tuple[List[str], np.ndarray]:
    d = np.load(str(path), allow_pickle=True)
    chs = [str(x) for x in d["chns_names"].tolist()]
    ec = np.asarray(d["events_count"], dtype=np.int64)
    return chs, ec


def _alias_collapse(
    chs: List[str], ec: np.ndarray
) -> Tuple[List[str], np.ndarray, List[Tuple[str, str]]]:
    """Collapse channel list to legacy-style left-contact aliases.

    On collision, keep the entry with the higher events_count and report the
    losers as ``(loser_full_name, alias)`` tuples (for QC).
    """
    by_alias: Dict[str, Tuple[int, int]] = {}  # alias -> (idx, count)
    losers: List[Tuple[str, str]] = []
    for i, nm in enumerate(chs):
        a = _bipolar_to_left_alias(nm)
        c = int(ec[i])
        if a not in by_alias:
            by_alias[a] = (i, c)
        else:
            old_i, old_c = by_alias[a]
            if c > old_c:
                losers.append((chs[old_i], a))
                by_alias[a] = (i, c)
            else:
                losers.append((nm, a))
    aliases = sorted(by_alias.keys())
    out_ec = np.asarray([by_alias[a][1] for a in aliases], dtype=np.int64)
    return aliases, out_ec, losers


def compare_block(
    new_path: Path, legacy_path: Path
) -> Dict[str, object]:
    chs_new, ec_new = _load_gpu_npz(new_path)
    chs_leg, ec_leg = _load_gpu_npz(legacy_path)

    # Aliasing: legacy already monopolar; new is bipolar.
    al_new, ec_new_a, losers_new = _alias_collapse(chs_new, ec_new)
    al_leg, ec_leg_a, _ = _alias_collapse(chs_leg, ec_leg)

    set_new = set(al_new)
    set_leg = set(al_leg)
    common = sorted(set_new & set_leg)
    only_new = sorted(set_new - set_leg)
    only_leg = sorted(set_leg - set_new)
    jaccard = len(set_new & set_leg) / max(1, len(set_new | set_leg))

    map_new = dict(zip(al_new, ec_new_a))
    map_leg = dict(zip(al_leg, ec_leg_a))

    pairs = np.asarray(
        [(map_new[a], map_leg[a]) for a in common], dtype=np.float64
    )
    if pairs.shape[0] >= 2:
        # Pearson r
        a = pairs[:, 0] - pairs[:, 0].mean()
        b = pairs[:, 1] - pairs[:, 1].mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        pearson = float((a @ b) / denom)
    else:
        pearson = float("nan")

    # Per-channel new/legacy ratio (only for legacy != 0 to avoid div zero)
    safe_leg = np.where(pairs[:, 1] > 0, pairs[:, 1], np.nan)
    ratios = pairs[:, 0] / safe_leg
    finite = ratios[np.isfinite(ratios)]
    abs_log_ratio = (
        np.abs(np.log(np.where(finite > 0, finite, np.nan)))
        if finite.size
        else np.zeros((0,))
    )

    # Top per-channel ratio outliers (most-amplified and most-suppressed)
    sorted_idx_amp = np.argsort(-ratios)[: min(10, len(common))]
    sorted_idx_sup = np.argsort(ratios)[: min(10, len(common))]
    top_amp = [
        (common[i], float(map_new[common[i]]), float(map_leg[common[i]]),
         float(ratios[i]) if np.isfinite(ratios[i]) else float("nan"))
        for i in sorted_idx_amp
    ]
    top_sup = [
        (common[i], float(map_new[common[i]]), float(map_leg[common[i]]),
         float(ratios[i]) if np.isfinite(ratios[i]) else float("nan"))
        for i in sorted_idx_sup
        if np.isfinite(ratios[i])
    ]

    # Special channel families for gaolan diagnosis
    def _family_avg_ratio(prefixes: List[str]) -> float:
        rs: List[float] = []
        for a in common:
            if any(a.startswith(p) for p in prefixes):
                if map_leg[a] > 0:
                    rs.append(map_new[a] / map_leg[a])
        return float(np.median(rs)) if rs else float("nan")

    family_ratios = {
        "B'13_B'14_A'10": _family_avg_ratio(["B'13", "B'14", "A'10"]),
        "D_D_prime":      _family_avg_ratio(["D'", "D"]),  # crude; D matches both
        "C_low":          _family_avg_ratio(["C1", "C2", "C3"]),
    }

    return {
        "new_path": str(new_path),
        "legacy_path": str(legacy_path),
        "n_ch_new": len(chs_new),
        "n_ch_leg": len(chs_leg),
        "n_ch_new_aliased": len(al_new),
        "n_ch_leg_aliased": len(al_leg),
        "alias_collisions_new": len(losers_new),
        "n_common": len(common),
        "n_only_new": len(only_new),
        "n_only_leg": len(only_leg),
        "jaccard": jaccard,
        "sum_new": int(ec_new_a.sum()),
        "sum_leg": int(ec_leg_a.sum()),
        "sum_ratio": float(ec_new_a.sum()) / max(1.0, float(ec_leg_a.sum())),
        "pearson_r_common": pearson,
        "median_abs_log_ratio_common": (
            float(np.median(abs_log_ratio)) if abs_log_ratio.size else float("nan")
        ),
        "p95_abs_log_ratio_common": (
            float(np.percentile(abs_log_ratio, 95)) if abs_log_ratio.size else float("nan")
        ),
        "top_amplified": top_amp,
        "top_suppressed": top_sup,
        "family_ratios": family_ratios,
        "only_new_top": only_new[:10],
        "only_leg_top": only_leg[:10],
    }


def write_md_report(report_path: Path, header: str, blocks: List[Dict]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        f"# {header}",
        "",
        "## Summary",
        "",
        "| Block | n_new | n_leg | Jaccard | sum_new | sum_leg | ratio | Pearson r | med\\|log r\\| | p95\\|log r\\| |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for b in blocks:
        lines.append(
            "| {block} | {nn} | {nl} | {jac:.3f} | {sn} | {sl} | {sr:.3f} | "
            "{pr:.3f} | {ml:.3f} | {pl:.3f} |".format(
                block=Path(b["new_path"]).stem.replace("_gpu", ""),
                nn=b["n_ch_new_aliased"],
                nl=b["n_ch_leg_aliased"],
                jac=b["jaccard"],
                sn=b["sum_new"],
                sl=b["sum_leg"],
                sr=b["sum_ratio"],
                pr=b["pearson_r_common"],
                ml=b["median_abs_log_ratio_common"],
                pl=b["p95_abs_log_ratio_common"],
            )
        )
    lines.append("")

    for b in blocks:
        block = Path(b["new_path"]).stem.replace("_gpu", "")
        lines += [
            "",
            f"## {block}",
            "",
            f"- new: `{b['new_path']}` ({b['n_ch_new']} ch -> {b['n_ch_new_aliased']} aliased; "
            f"{b['alias_collisions_new']} collisions)",
            f"- legacy: `{b['legacy_path']}` ({b['n_ch_leg']} ch)",
            f"- common channels: {b['n_common']}, only-new: {b['n_only_new']}, only-leg: {b['n_only_leg']}",
            f"- Jaccard: {b['jaccard']:.3f}",
            f"- sum new={b['sum_new']}, legacy={b['sum_leg']}, ratio={b['sum_ratio']:.3f}",
            f"- Pearson r (common): {b['pearson_r_common']:.4f}",
            f"- median |log ratio|: {b['median_abs_log_ratio_common']:.3f} "
            f"(p95: {b['p95_abs_log_ratio_common']:.3f})",
            "",
            "**Family ratios (new/legacy, median across family channels)**:",
            "",
            "| Family | Ratio |",
            "|---|---|",
        ]
        for k, v in b["family_ratios"].items():
            lines.append(
                f"| {k} | {v:.3f} |" if np.isfinite(v) else f"| {k} | n/a |"
            )
        lines += [
            "",
            "**Top amplified channels (new/legacy ratio descending)**:",
            "",
            "| ch | new | legacy | ratio |",
            "|---|---|---|---|",
        ]
        for nm, n, l, r in b["top_amplified"]:
            lines.append(f"| {nm} | {n:.0f} | {l:.0f} | {r:.3f} |")
        lines += [
            "",
            "**Top suppressed channels (new/legacy ratio ascending)**:",
            "",
            "| ch | new | legacy | ratio |",
            "|---|---|---|---|",
        ]
        for nm, n, l, r in b["top_suppressed"]:
            lines.append(f"| {nm} | {n:.0f} | {l:.0f} | {r:.3f} |")
        if b["only_new_top"]:
            lines += [
                "",
                f"**only-new sample**: {', '.join(b['only_new_top'])}",
            ]
        if b["only_leg_top"]:
            lines += [
                f"**only-legacy sample**: {', '.join(b['only_leg_top'])}",
            ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-root", required=True, type=Path)
    ap.add_argument("--legacy-root", required=True, type=Path)
    ap.add_argument("--subject", required=True, type=str)
    ap.add_argument(
        "--block", default=None, type=str,
        help="block stem (e.g. FA0013KP). If omitted, scans all common blocks.",
    )
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args()

    new_dir = args.new_root / args.subject
    leg_dir = args.legacy_root / args.subject

    if args.block:
        new_p = new_dir / f"{args.block}_gpu.npz"
        leg_p = leg_dir / f"{args.block}_gpu.npz"
        blocks = [compare_block(new_p, leg_p)]
        header = f"Phase D A/B  -  {args.subject}  -  {args.block}"
    else:
        new_blocks = sorted(p.stem.replace("_gpu", "") for p in new_dir.glob("*_gpu.npz"))
        leg_blocks = sorted(p.stem.replace("_gpu", "") for p in leg_dir.glob("*_gpu.npz"))
        common = sorted(set(new_blocks) & set(leg_blocks))
        blocks = []
        for blk in common:
            blocks.append(
                compare_block(new_dir / f"{blk}_gpu.npz", leg_dir / f"{blk}_gpu.npz")
            )
        header = f"Phase D A/B  -  {args.subject}  -  {len(common)} common blocks"

    write_md_report(args.report, header, blocks)
    for b in blocks:
        print(
            f"{Path(b['new_path']).stem:25s}  jac={b['jaccard']:.3f}  "
            f"sum_ratio={b['sum_ratio']:.3f}  r={b['pearson_r_common']:.3f}  "
            f"p95|logr|={b['p95_abs_log_ratio_common']:.3f}"
        )
    print(f"\nReport: {args.report}")


if __name__ == "__main__":
    main()
