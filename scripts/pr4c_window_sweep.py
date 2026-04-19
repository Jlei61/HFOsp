#!/usr/bin/env python3
"""PR-4C window-size sweep: data-driven window tuning, no L1/L2/L3 compute.

Loads event_abs_times and seizure_times for every Yuquan + Epilepsiae subject
and, for each candidate (baseline, pre, post) configuration, counts
`n_seizures_usable` (all three states non-empty with nearest-seizure
assignment) and per-state event retention. No Kendall-tau / Pearson-r runs.

Usage:
    python scripts/pr4c_window_sweep.py [--dataset epilepsiae|yuquan|both]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interictal_propagation import (  # noqa: E402
    _build_seizure_proximity_windows,
    _valid_event_indices,
    load_subject_propagation_events,
)
from src.event_periodicity import load_seizure_times  # noqa: E402


YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
RESULTS_DIR = Path("results/interictal_propagation")

YUQUAN_SUBJECTS = [
    "zhangkexuan", "pengzihang", "chengshuai", "huangwanling",
    "liyouran", "songzishuo", "zhangbichen", "zhaochenxi",
    "zhaojinrui", "zhourongxuan", "zhangjiaqi",
    "chenziyang", "hanyuxuan", "huanghanwen", "litengsheng",
    "xuxinyi", "zhangjinhan", "sunyuanxin",
]
EPILEPSIAE_SUBJECTS = [
    "1096", "1084", "958", "922", "590", "1150", "442", "1073",
    "253", "1146", "916", "620", "583", "548", "384", "139",
    "1125", "1077", "818", "635",
]


CONFIGS: List[Tuple[str, Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = [
    # (name, baseline, pre, post) all in hours relative to seizure onset
    ("12/6/6",  (-12.0, -6.0), (-6.0, -1.0), (1.0, 6.0)),   # current default
    ("8/4/4",   (-8.0,  -4.0), (-4.0, -1.0), (1.0, 4.0)),
    ("6/2/2",   (-6.0,  -2.0), (-2.0, -0.25), (0.25, 2.0)),
    ("4/1/1",   (-4.0,  -1.0), (-1.0, -0.25), (0.25, 1.0)),
    ("3/1/1",   (-3.0,  -1.0), (-1.0, -0.25), (0.25, 1.0)),
    ("2/0.5/1", (-2.0,  -0.5), (-0.5, -0.08), (0.08, 1.0)),
]


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout,
)
log = logging.getLogger("pr4c_window_sweep")


def _collect_subject_times(
    dataset: str, subject: str
) -> Dict[str, object] | None:
    root = YUQUAN_ROOT if dataset == "yuquan" else EPILEPSIAE_ROOT
    subject_dir = root / subject if dataset == "yuquan" else root / subject / "all_recs"
    if not subject_dir.exists() or not list(subject_dir.glob("*_lagPat.npz")):
        return None
    try:
        loaded = load_subject_propagation_events(subject_dir)
    except Exception as exc:
        log.warning("%s/%s: load failed: %s", dataset, subject, exc)
        return None
    sz_times = load_seizure_times(subject, dataset)
    valid_idx = _valid_event_indices(loaded["bools"], min_participating=3)
    event_times = np.asarray(loaded["event_abs_times"], dtype=float)[valid_idx]
    return {
        "dataset": dataset,
        "subject": subject,
        "n_events_valid": int(event_times.size),
        "n_seizures_total": int(len(sz_times)),
        "event_times": event_times,
        "seizure_times": [float(x) for x in sz_times],
    }


def _sweep_configs(
    event_times: np.ndarray, seizure_times: List[float]
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for name, base, pre, post in CONFIGS:
        prox = _build_seizure_proximity_windows(
            event_times, seizure_times,
            baseline_hours=base, pre_ictal_hours=pre, post_ictal_hours=post,
        )
        counts = prox["state_event_counts"]
        n_usable = len(prox["usable_windows"])
        out.append({
            "config": name,
            "baseline_h": [base[0], base[1]],
            "pre_h": [pre[0], pre[1]],
            "post_h": [post[0], post[1]],
            "n_seizures_usable": n_usable,
            "baseline_events": int(counts.get("baseline", 0)),
            "pre_events": int(counts.get("pre", 0)),
            "post_events": int(counts.get("post", 0)),
            "excluded_events": int(counts.get("excluded", 0)),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["yuquan", "epilepsiae", "both"], default="both")
    ap.add_argument("--out", default=str(RESULTS_DIR / "pr4c_window_sweep.json"))
    args = ap.parse_args()

    pairs: List[Tuple[str, List[str]]] = []
    if args.dataset in ("yuquan", "both"):
        pairs.append(("yuquan", YUQUAN_SUBJECTS))
    if args.dataset in ("epilepsiae", "both"):
        pairs.append(("epilepsiae", EPILEPSIAE_SUBJECTS))

    per_subject: List[Dict[str, object]] = []
    for dataset, subjects in pairs:
        for subject in subjects:
            rec = _collect_subject_times(dataset, subject)
            if rec is None:
                log.info("%s/%s: skipped (no data)", dataset, subject)
                continue
            n_sz = rec["n_seizures_total"]
            if n_sz == 0:
                log.info("%s/%s: 0 seizures, skip sweep", dataset, subject)
                per_subject.append({
                    "dataset": dataset, "subject": subject,
                    "n_events_valid": rec["n_events_valid"],
                    "n_seizures_total": 0,
                    "configs": [],
                })
                continue
            sweep = _sweep_configs(rec["event_times"], rec["seizure_times"])
            per_subject.append({
                "dataset": dataset, "subject": subject,
                "n_events_valid": rec["n_events_valid"],
                "n_seizures_total": n_sz,
                "configs": sweep,
            })
            line = f"{dataset}/{subject} sz={n_sz:3d}"
            for cfg in sweep:
                line += f"  {cfg['config']}:u={cfg['n_seizures_usable']}"
            log.info(line)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"configs": [c[0] for c in CONFIGS], "per_subject": per_subject}, f, indent=2)
    log.info("Saved %s", out_path)

    log.info("=== COHORT SUMMARY ===")
    for cfg_idx, (name, *_) in enumerate(CONFIGS):
        vals = [
            s["configs"][cfg_idx]
            for s in per_subject
            if s.get("configs") and len(s["configs"]) > cfg_idx
        ]
        if not vals:
            continue
        usable = [int(v["n_seizures_usable"]) for v in vals]
        total_sz = [int(s["n_seizures_total"]) for s in per_subject if s.get("configs")]
        subj_with_usable = sum(1 for u in usable if u > 0)
        log.info(
            "  %-10s total_usable=%4d  subjects_with_usable=%2d/%2d  median_usable_per_subject=%.1f  mean=%.1f",
            name,
            int(sum(usable)), subj_with_usable, len(usable),
            float(np.median(usable)), float(np.mean(usable)),
        )


if __name__ == "__main__":
    main()
