#!/usr/bin/env python3
"""Per-seed core_only spontaneous-ignition audit (Step B, reviewer 2026-06-23).

OFFLINE over per_seed_metrics.csv. Runs NO SNN and does NOT touch the calibration
runner's substrate-level majority-vote core_only_quiet gate (which is CORRECT).

The reframe: core_only was summarized by the MEAN, so one spontaneously-igniting
seed (w18.0: co_ds=1282 vs the other 7 ~19-51) dragged the substrate mean to 188.8
even though the median (32) ~= the bare sheet. The fix is in the analysis layer:
  1. a per-seed spontaneous-ignition flag (large no-kick core_only event),
  2. a robust (median + IQR) core_only summary instead of the mean,
  3. seed-level contamination tracking so a single igniter can be excluded as a
     sensitivity check (and never pollute a downstream W_event estimate).

w18.0 is used as a DETECTOR STRESS TEST (does the flag catch its 1/8 igniter?), not
as a B2 substrate. The B2 substrates (bare, n17.6) must show 0 flagged seeds.

Outputs (--out-dir):
  m3_core_only_seed_confounds_<date>.csv  — per (substrate, seed): co_only_max,
                                            bare background, spontaneous_ignition flag
  m3_core_only_robust_summary_<date>.md   — per substrate: median/IQR/mean/n_igniters
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_m3_finite_event_robustness import _read_csv, _fnum  # noqa: E402

# A no-kick core_only downstream count this far above the bare sheet = a real
# spontaneous event in that seed (not the low-magnitude background fluctuation that
# core_only_event_in_win flags at 2/8 for EVERY substrate). Both clauses required so
# a globally hot bare sheet is not flagged on the absolute floor alone.
SPONT_ABS_FLOOR = 100.0   # absolute core_only downstream spikes
SPONT_RATIO = 3.0         # AND >= this x the bare-sheet background


def spontaneous_ignition_flag(co_ds_max: float, bare_bg_med: float) -> bool:
    """True iff this seed's peak no-kick core_only activity is a real spontaneous event."""
    return bool(co_ds_max >= max(SPONT_ABS_FLOOR, SPONT_RATIO * bare_bg_med))


def summarize_core_only(co_ds_list: Sequence[float], bare_bg_med: float) -> Dict[str, float]:
    """Robust per-substrate core_only summary: median/IQR + igniter count (NOT mean-led)."""
    vals = sorted(float(v) for v in co_ds_list)
    n = len(vals)
    q25 = statistics.quantiles(vals, n=4)[0] if n >= 2 else vals[0]
    q75 = statistics.quantiles(vals, n=4)[2] if n >= 2 else vals[0]
    n_spont = sum(1 for v in vals if spontaneous_ignition_flag(v, bare_bg_med))
    return {
        "n_seeds": n,
        "n_spontaneous": n_spont,
        "median": statistics.median(vals),
        "q25": q25,
        "q75": q75,
        "mean": statistics.mean(vals),
        "max": vals[-1],
    }


def _per_seed_core_only(run_dir: str) -> Tuple[Dict[int, float], float]:
    """(seed -> max core_only_downstream over kicks/windows), bare-sheet bg median."""
    rows = _read_csv(os.path.join(run_dir, "per_seed_metrics.csv"))
    by_seed: Dict[int, float] = {}
    bg: List[float] = []
    for r in rows:
        s = int(_fnum(r, "seed"))
        co = _fnum(r, "core_only_downstream_resp")
        by_seed[s] = max(by_seed.get(s, 0.0), co)        # core_only is kick-indep; max over windows
        bg.append(_fnum(r, "no_core_no_kick_downstream"))
    bg_med = statistics.median([v for v in bg if v == v]) if bg else float("nan")
    return by_seed, bg_med


def audit(run_dirs: Dict[str, str]) -> Tuple[List[dict], Dict[str, dict]]:
    rows: List[dict] = []
    summary: Dict[str, dict] = {}
    for name, d in run_dirs.items():
        by_seed, bg_med = _per_seed_core_only(d)
        co_list = [by_seed[s] for s in sorted(by_seed)]
        summary[name] = {**summarize_core_only(co_list, bg_med), "bare_bg_med": bg_med}
        for s in sorted(by_seed):
            flag = spontaneous_ignition_flag(by_seed[s], bg_med)
            rows.append({
                "substrate": name, "seed": s,
                "core_only_max": round(by_seed[s], 1),
                "bare_bg_median": round(bg_med, 1),
                "spontaneous_ignition": int(flag),
            })
    return rows, summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--date", default="2026-06-23")
    # run-dir tag -> basename; allow both the original finescan_* and the ceiling dirs
    ap.add_argument("--dirs", nargs="+",
                    default=["finescan_bare", "finescan_n17.6", "finescan_n17.8",
                             "finescan_n18.0", "finescan_w18.0",
                             "kick_ceiling_bare", "kick_ceiling_n17.6"])
    args = ap.parse_args(argv)

    run_dirs = {}
    for tag in args.dirs:
        d = os.path.join(args.base, tag)
        if os.path.isfile(os.path.join(d, "per_seed_metrics.csv")):
            run_dirs[tag] = d
    os.makedirs(args.out_dir, exist_ok=True)

    rows, summary = audit(run_dirs)

    csv_path = os.path.join(args.out_dir, f"m3_core_only_seed_confounds_{args.date}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["substrate", "seed", "core_only_max",
                                           "bare_bg_median", "spontaneous_ignition"])
        w.writeheader()
        w.writerows(rows)

    # B2 robustness: the B2 substrates must carry zero contamination.
    b2_tags = [t for t in run_dirs if "bare" in t or "n17.6" in t]
    b2_clean = all(summary[t]["n_spontaneous"] == 0 for t in b2_tags)

    md_path = os.path.join(args.out_dir, f"m3_core_only_robust_summary_{args.date}.md")
    lines = [f"# core_only 逐 seed 自发点火审计（{args.date}）\n",
             "substrate-level 多数票 core_only_quiet 门**本就正确，未改动**。本审计在分析层加 "
             "per-seed 自发点火 flag + robust(median/IQR) 汇总，替代被离群 seed 拉偏的 mean。\n",
             f"判据：seed 自发点火 ⇔ core_only_max ≥ max({SPONT_ABS_FLOOR:g}, "
             f"{SPONT_RATIO:g}×bare 背景中位数)。\n",
             "\n| substrate | n_seeds | n_igniters | median | IQR(q25–q75) | mean(误导) | bare_bg | max |",
             "|---|---|---|---|---|---|---|---|"]
    for name in run_dirs:
        s = summary[name]
        lines.append(
            f"| {name} | {s['n_seeds']} | **{s['n_spontaneous']}** | {s['median']:.1f} | "
            f"{s['q25']:.1f}–{s['q75']:.1f} | {s['mean']:.1f} | {s['bare_bg_med']:.1f} | {s['max']:.0f} |")
    lines.append("\n## 读法\n")
    lines.append(
        "- **w18.0 = detector stress test**：含 1/8 自发点火 seed（core_only_max≈1282），"
        "mean 被拉到 ~189 但 median≈bare → flag 抓到它即说明 per-seed 检测有效。**w18.0 不进 B2 结论。**")
    lines.append(
        f"- **B2 稳健性**：B2 底物（bare、n17.6，含 ceiling）自发点火 seed 数 = "
        f"{'全 0（B2 不依赖任何污染 seed）✓' if b2_clean else '非 0 —— 需检查！'}")
    lines.append(
        "- **sensitivity exclusion**：下游（mini-W_event）应用同一 flag 逐 seed 剔除自发点火 seed，"
        "core_only 汇总一律用 median/IQR 不用 mean。flag 函数 `spontaneous_ignition_flag` 可被 import 复用。")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"audited {len(run_dirs)} run dirs; igniter seeds per substrate:")
    for name in run_dirs:
        print(f"  {name:20s} n_igniters={summary[name]['n_spontaneous']}/{summary[name]['n_seeds']}"
              f"  median={summary[name]['median']:.1f} mean={summary[name]['mean']:.1f}")
    print(f"B2 substrates (bare,n17.6) clean of contamination: {b2_clean}")
    print(f"outputs -> {csv_path}\n            {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
