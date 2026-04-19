#!/usr/bin/env python3
"""Post-process pr4c_window_sweep.json: report per-state event counts per config."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sweep = json.load(open("results/interictal_propagation/pr4c_window_sweep.json"))
configs = sweep["configs"]
subjects = sweep["per_subject"]


def _per_config_stats():
    print(f"{'config':10} {'total_u':>8} {'subj_u':>8} "
          f"{'med_u':>6} {'mean_b':>8} {'med_b':>8} {'mean_p':>8} {'med_p':>8} "
          f"{'mean_po':>8} {'med_po':>8} {'retention':>10}")
    for ci, cfg_name in enumerate(configs):
        usable = []
        base_ev, pre_ev, post_ev, excl, total = [], [], [], [], []
        subj_has_any = 0
        for s in subjects:
            cs = s.get("configs") or []
            if ci >= len(cs):
                continue
            c = cs[ci]
            u = int(c["n_seizures_usable"])
            usable.append(u)
            if u > 0:
                subj_has_any += 1
            # per-state totals are across all seizures of this subject
            # but "usable" means both states non-empty for at least one seizure
            # we only look at per-state events inside usable windows, though:
            # approximation — for data sense, use global totals
            base_ev.append(int(c["baseline_events"]))
            pre_ev.append(int(c["pre_events"]))
            post_ev.append(int(c["post_events"]))
            excl.append(int(c["excluded_events"]))
            total.append(int(c["baseline_events"] + c["pre_events"] + c["post_events"] + c["excluded_events"]))
        retention = (
            float(np.sum([b + p + po for b, p, po in zip(base_ev, pre_ev, post_ev)]))
            / float(max(1, sum(total)))
        )
        print(f"{cfg_name:10} {sum(usable):8d} {subj_has_any:8d} "
              f"{float(np.median(usable)):6.1f} "
              f"{float(np.mean(base_ev)):8.0f} {float(np.median(base_ev)):8.0f} "
              f"{float(np.mean(pre_ev)):8.0f} {float(np.median(pre_ev)):8.0f} "
              f"{float(np.mean(post_ev)):8.0f} {float(np.median(post_ev)):8.0f} "
              f"{retention:9.2%}")


def _per_subject_counts_for_config(cfg_name: str):
    """For a given config, per-subject events in each state (cohort distribution)."""
    ci = configs.index(cfg_name)
    print(f"\n=== per-subject events at config {cfg_name} (epilepsiae only) ===")
    print(f"{'subject':12} {'n_sz':>5} {'u':>3} {'baseline':>10} {'pre':>10} {'post':>10} {'excluded':>10}")
    for s in subjects:
        if s["dataset"] != "epilepsiae":
            continue
        cs = s.get("configs") or []
        if ci >= len(cs):
            continue
        c = cs[ci]
        print(f"{s['subject']:12} {s['n_seizures_total']:5d} "
              f"{int(c['n_seizures_usable']):3d} "
              f"{int(c['baseline_events']):10d} {int(c['pre_events']):10d} "
              f"{int(c['post_events']):10d} {int(c['excluded_events']):10d}")


_per_config_stats()
for name in ("12/6/6", "4/1/1", "3/1/1", "2/0.5/1"):
    _per_subject_counts_for_config(name)
