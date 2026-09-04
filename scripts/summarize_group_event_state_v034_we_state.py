#!/usr/bin/env python3
"""Merge the frozen-decoder S_P state cards across decoder/state seeds and bootstrap the arm contrasts.

Per subject: per-anchor STATE_SELECTION scores are merged by the median across
seeds, then contrasted with a within-carry-segment moving-block bootstrap
(block_len 6, 1000 draws), matching the v0.3.3 cards.  Contrasts (positive
favours the second-named arm being better, i.e. lower NLL):

    adapter_gain   = zero    - adapter        state-free recalibration of the frozen decoder
    state_gain     = adapter - learned        what the cross-event state adds on top of the adapter
    total_gain     = zero    - learned
    beyond_const   = period_mean - learned    information a selection-period constant cannot give
    shift_cost     = shifted - learned        wrong-time penalty
    learned_vs_random = learned - random      negative favours the trained encoder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.topic5_group_event_state.v032_model.evaluate import block_bootstrap_mean_ci
from src.topic5_group_event_state.v033_training_lab.paths import atomic_write_json

OUT_ROOT = Path("/data/hfosp_group_event_state_v0_3_4/we_state")
BOOT = dict(block_len=6, n_boot=1000, seed=0)
SUBJECTS = ("epilepsiae_1146", "epilepsiae_583", "epilepsiae_548", "epilepsiae_922", "epilepsiae_253")
ARMS = ("zero", "adapter", "learned", "period_mean", "shifted", "random")
CONTRASTS = {
    "adapter_gain": ("zero", "adapter"), "state_gain": ("adapter", "learned"), "total_gain": ("zero", "learned"),
    "beyond_const": ("period_mean", "learned"), "shift_cost": ("shifted", "learned"), "learned_vs_random": ("learned", "random"),
}


def ci(values, segments):
    out = block_bootstrap_mean_ci(np.asarray(values, dtype=np.float64), np.asarray(segments), **BOOT)
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def fci(c):
    return f"{c['mean']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"


def summarize(root: Path, tag: str, metric: str) -> dict:
    rows = {}
    for subject in SUBJECTS:
        cards = sorted((root / tag / subject).glob("decoder_seed*/card.json"))
        if not cards:
            continue
        per = {a: [] for a in ARMS}
        seg = shift_ok = None
        stage_rows = []
        for card_path in cards:
            card = json.loads(card_path.read_text())
            arr = np.load(card["per_anchor_path"], allow_pickle=False)
            for a in ARMS:
                per[a].append(np.asarray(arr[f"{a}_{metric}"], dtype=np.float64))
            seg = np.asarray(arr["bootstrap_segment"]); shift_ok = np.asarray(arr["shift_valid"]).astype(bool)
            t_anchor = np.asarray(arr["anchor_time"], dtype=np.float64)
            stage_rows.append({k: {kk: v[kk] for kk in ("best_inner_val", "selected_step", "selected_at_init", "selected_at_budget_edge")}
                               for k, v in card["stages"].items()} | {"coverage": card["coverage"]["coverage_fraction"],
                                                                     "decoder_test_contact_nll": card["decoder_metrics"]["test"]["contact_nll"]})
        merged = {a: np.median(np.stack(per[a]), axis=0) for a in ARMS}
        merged["shifted"] = np.where(shift_ok, np.median(np.stack(per["shifted"]), axis=0), np.nan)
        contrasts = {name: ci(merged[a] - merged[b], seg) for name, (a, b) in CONTRASTS.items()}
        per_seed = {name: [float(np.nanmean(pa - pb)) for pa, pb in zip(per[a], per[b])] for name, (a, b) in CONTRASTS.items()}
        # independent 30-min windows within carry segments (same rule as DataView.blocks)
        windows = set()
        for sg in np.unique(seg):
            tt = t_anchor[seg == sg]
            windows.update((int(sg), int(v)) for v in np.floor((tt - tt.min()) / 1800.0))
        rows[subject] = {"n_seeds": len(cards), "n_anchors": int(seg.size), "n_blocks": len(windows), "n_segments": int(np.unique(seg).size),
                         "n_valid_donors": int(shift_ok.sum()), "arm_means": {a: float(np.nanmean(merged[a])) for a in ARMS},
                         "contrasts": contrasts, "per_seed_contrast_means": per_seed, "stages": stage_rows}
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=OUT_ROOT)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--metric", choices=("grammar", "contact_nll"), default="grammar")
    args = ap.parse_args()
    rows = summarize(args.root, args.tag, args.metric)
    lines = [f"| 患者 | seeds | 独立块 | 冻结解码器 zero | 无状态重标定 adapter_gain | 状态在 adapter 之上 state_gain | 常数解释不了 beyond_const | 错时代价 | 学到−随机 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s, r in rows.items():
        c = r["contrasts"]
        lines.append(f"| {s.replace('epilepsiae_', 'E')} | {r['n_seeds']} | {r['n_blocks']} | {r['arm_means']['zero']:.3f} | {fci(c['adapter_gain'])} | {fci(c['state_gain'])} | {fci(c['beyond_const'])} | {fci(c['shift_cost'])} | {fci(c['learned_vs_random'])} |")
    text = "\n".join(lines)
    print(text)
    out = args.root / args.tag / f"summary_{args.metric}.json"
    atomic_write_json(out, {"format": "group_event_state_v0_3_4_we_state_summary_v1", "metric": args.metric, "subjects": rows,
                            "development_targets_read": False})
    (args.root / args.tag / f"summary_{args.metric}.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
