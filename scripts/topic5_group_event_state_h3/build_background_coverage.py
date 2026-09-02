#!/usr/bin/env python3
"""How much clean background each patient's common-drive arm actually gets.

The background rule drops any 2 s window that overlaps an event core, so a patient
with a dense IED stream has almost no clean background windows left.  That is a
confound with a direction: it starves ``M0`` -- the arm whose whole case is the
common drive -- precisely in the patients where events are most frequent, and so
biases the comparison *towards* "the event edge helps".

It has to be measured and reported next to the result, not discovered afterwards.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state_h3.background import CELL_SECONDS, cell_background  # noqa: E402
from src.topic5_group_event_state_h3.io import write_json_atomic  # noqa: E402
from src.topic5_group_event_state_h3.support import (  # noqa: E402
    POSTICTAL_EXCLUSION_SECONDS,
    build_coverage_segments,
    cut_intervals_at_seizures,
    load_block_time_ranges,
    load_seizures,
    segment_anchor_grid,
    segment_bounds,
    split_by_physical_time,
)

V0_1 = Path("/home/honglab/leijiaxin/HFOsp/results/epi_prssm/group_event_state/v0_1")
DATASET = Path("/data/hfosp_group_event_state_v0_1/dataset")
BACKGROUND = Path("/data/hfosp_group_event_state_v0_2/agent_c/background")
FEATURES = Path("/data/hfosp_group_event_state_v0_2/agent_c/features")
OUT = ROOT / "results/epi_prssm/group_event_state/v0_2/h3/support/background_coverage.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()

    subjects = args.subjects or sorted(p.name for p in DATASET.iterdir() if (p / "index.json").exists())
    rows = []
    for subject in subjects:
        segments = build_coverage_segments(load_block_time_ranges(V0_1 / "block_inventory.csv", subject))
        cut = cut_intervals_at_seizures(
            segments, load_seizures(DATASET / subject / "index.json"),
            postictal_exclusion_s=POSTICTAL_EXCLUSION_SECONDS,
        )
        intervals = split_by_physical_time(cut)
        with np.load(BACKGROUND / f"{subject}.npz") as bg:
            anchor_time, anchor_features = bg["anchor_time"], bg["anchor_features"]
        with np.load(FEATURES / f"{subject}.npz") as feat:
            n_events = int(feat["t_abs"].size)

        total_cells = valid_cells = 0
        usable_seconds = 0.0
        for _seg, (lo, hi) in sorted(segment_bounds(intervals).items()):
            starts = lo + CELL_SECONDS * np.arange(int(np.floor((hi - lo) / CELL_SECONDS)) + 1)
            _values, valid = cell_background(starts, anchor_time, anchor_features)
            total_cells += int(valid.size)
            valid_cells += int(valid.sum())
        usable_seconds = sum(i.duration for i in intervals)

        rows.append(
            {
                "subject": subject,
                "usable_hours": usable_seconds / 3600.0,
                "n_events_interictal": n_events,
                "event_rate_hz": n_events / max(usable_seconds, 1.0),
                "n_background_anchors": int(anchor_time.size),
                "anchors_per_usable_hour": anchor_time.size / max(usable_seconds / 3600.0, 1e-9),
                "n_background_cells": total_cells,
                "background_cell_valid_fraction": valid_cells / max(total_cells, 1),
            }
        )

    rate = np.asarray([r["event_rate_hz"] for r in rows])
    frac = np.asarray([r["background_cell_valid_fraction"] for r in rows])
    ok = np.isfinite(rate) & np.isfinite(frac)
    rx = np.argsort(np.argsort(rate[ok])).astype(float)
    fx = np.argsort(np.argsort(frac[ok])).astype(float)
    rho = float(np.corrcoef(rx, fx)[0, 1]) if ok.sum() > 3 else float("nan")

    payload = {
        "note": (
            "background windows overlapping an event core are dropped by the v0.1 "
            "rule; a negative rank correlation here means the common-drive arm is "
            "starved exactly where events are dense, which biases the comparison "
            "towards the event-edge arms"
        ),
        "cell_seconds": CELL_SECONDS,
        "spearman_event_rate_vs_background_validity": rho,
        "n_subjects": len(rows),
        "subjects": sorted(rows, key=lambda r: -r["event_rate_hz"]),
    }
    write_json_atomic(payload, OUT)
    print(f"{'subject':26s} {'rate Hz':>8s} {'anch/h':>8s} {'cell valid':>11s}")
    for row in payload["subjects"]:
        print(f"{row['subject']:26s} {row['event_rate_hz']:8.4f} "
              f"{row['anchors_per_usable_hour']:8.1f} {row['background_cell_valid_fraction']:11.3f}")
    print(f"\nSpearman(event rate, background cell validity) = {rho:+.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
