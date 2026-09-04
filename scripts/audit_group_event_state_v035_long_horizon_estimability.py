#!/usr/bin/env python3
"""Re-count long H1/H2 support under the observed-exposure contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic5_group_event_state.v035.contracts import DATASET_ROOT, INPUT_ROOT, atomic_json  # noqa: E402
from src.topic5_group_event_state.v035.long_windows import (  # noqa: E402
    exposure_and_gap_count,
    merge_artificial_cuts,
    phase_for_times,
    plan_horizon_specific_split,
)

HORIZONS = (7200.0, 21600.0, 28800.0, 43200.0, 86400.0)
OFFSETS = (100, 500, 1000)
GRID_SECONDS = 300.0
MINIMUM_EXPOSURE_FRACTION = 0.5


def _nonoverlap_windows(starts: np.ndarray, horizon: float) -> int:
    chosen = 0
    available = -np.inf
    for value in np.sort(np.asarray(starts, dtype=np.float64)):
        if value >= available - 1e-9:
            chosen += 1
            available = float(value + horizon)
    return chosen


def _anchors(segments: np.ndarray, bounds: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    time, segment = [], []
    for seg, (lo, hi) in enumerate(segments):
        first = math.ceil(float(lo) / GRID_SECONDS) * GRID_SECONDS
        for value in np.arange(first, min(float(hi), bounds["80pct"]), GRID_SECONDS):
            if value >= bounds["20pct"]:
                time.append(float(value)); segment.append(seg)
    return np.asarray(time, dtype=np.float64), np.asarray(segment, dtype=np.int64)


def _wrong_time_support(
    anchor_time: np.ndarray, eligible: np.ndarray, donor_pool: np.ndarray, horizon: float,
) -> dict[str, int]:
    """Coarse support audit; final scoring additionally matches rate/exposure."""

    rows = np.flatnonzero(eligible)
    donors = np.flatnonzero(donor_pool)
    matched = 0
    for row in rows:
        clock_delta = np.abs((anchor_time[donors] - anchor_time[row]) % 86400.0)
        clock_delta = np.minimum(clock_delta, 86400.0 - clock_delta)
        ok = (np.abs(anchor_time[donors] - anchor_time[row]) >= horizon) & (clock_delta <= 7200.0)
        if np.any(ok):
            matched += 1
    return {"eligible_anchors": int(rows.size), "coarsely_matchable_anchors": int(matched)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=Path("/data/hfosp_group_event_state_v0_3_5_long_observed_support/estimability_v3.json"),
    )
    args = parser.parse_args()
    rows = []
    for manifest_path in sorted(INPUT_ROOT.glob("*/manifest_v3.json")):
        subject = manifest_path.parent.name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        legacy = {k: float(v) for k, v in manifest["report"]["phase_boundaries_epoch"].items()}
        index = json.loads((DATASET_ROOT / subject / "index.json").read_text(encoding="utf-8"))
        with np.load(manifest["input_path"], allow_pickle=False) as stored:
            original_segments = np.asarray(stored["target_segment_bounds"], dtype=np.float64)
            event_time = np.asarray(stored["event_time"], dtype=np.float64)
        carry_segments, merge_audit = merge_artificial_cuts(
            original_segments, index.get("seizures", ()), max_gap_seconds=600.0,
        )
        record = {
            "subject": subject, "merge_audit": merge_audit.__dict__,
            "safe_prefix_wall_hours": (legacy["80pct"] - legacy["20pct"]) / 3600.0,
            "physical": {}, "event_offsets": {},
        }
        for horizon in HORIZONS:
            plan = plan_horizon_specific_split(original_segments, legacy, horizon)
            item = {"split": plan.as_dict()}
            if plan.status == "ESTIMABLE" and plan.boundaries is not None:
                anchor_time, _ = _anchors(original_segments, plan.boundaries)
                phase = phase_for_times(anchor_time, plan.boundaries)
                stops = anchor_time + horizon
                exposure, gaps = exposure_and_gap_count(original_segments, anchor_time, stops)
                phase_hi = np.asarray([
                    {"FIT": plan.boundaries["60pct"], "INNER": plan.boundaries["70pct"],
                     "SELECTION": plan.boundaries["80pct"]}.get(str(value), -np.inf)
                    for value in phase
                ])
                eligible = (
                    np.isin(phase, ("FIT", "INNER", "SELECTION"))
                    & (stops <= phase_hi + 1e-9)
                    & (exposure >= MINIMUM_EXPOSURE_FRACTION * horizon)
                )
                support = {}
                for name in ("FIT", "INNER", "SELECTION"):
                    use = eligible & (phase == name)
                    support[name] = {
                        "eligible_anchors": int(use.sum()),
                        "nonoverlap_wall_windows": _nonoverlap_windows(anchor_time[use], horizon),
                        "median_exposure_fraction": float(np.median(exposure[use] / horizon)) if np.any(use) else None,
                        "windows_crossing_exposure_holes": int(np.sum(gaps[use] > 0)),
                    }
                item["support"] = support
                item["wrong_time_selection_support"] = _wrong_time_support(
                    anchor_time, eligible & (phase == "SELECTION"),
                    np.isin(phase, ("FIT", "INNER", "SELECTION")), horizon,
                )
            record["physical"][f"{int(horizon // 3600)}h"] = item

        phase = phase_for_times(event_time, legacy)
        carry_event_segment = np.full(event_time.shape, -1, dtype=np.int64)
        for seg, (lo, hi) in enumerate(carry_segments):
            carry_event_segment[(event_time >= lo) & (event_time < hi)] = seg
        for offset in OFFSETS:
            per_phase = {}
            for name in ("FIT", "INNER", "SELECTION"):
                counts = [
                    int(np.sum((phase == name) & (carry_event_segment == seg)))
                    for seg in np.unique(carry_event_segment[carry_event_segment >= 0])
                ]
                n = int(sum(counts))
                per_phase[name] = {
                    "events": n,
                    "same_segment_target_pairs": int(sum(max(0, value - offset) for value in counts)),
                    "nonoverlap_pairs_upper_bound": int(sum(value // (offset + 1) for value in counts)),
                }
            record["event_offsets"][str(offset)] = per_phase
        rows.append(record)

    summary = {}
    for horizon in HORIZONS:
        key = f"{int(horizon // 3600)}h"
        estimable = [r["subject"] for r in rows if r["physical"][key]["split"]["status"] == "ESTIMABLE"]
        scoreable = [
            r["subject"] for r in rows
            if r["physical"][key]["split"]["status"] == "ESTIMABLE"
            and r["physical"][key].get("support", {}).get("SELECTION", {}).get("nonoverlap_wall_windows", 0) >= 2
        ]
        summary[key] = {"split_estimable": estimable, "selection_has_at_least_2_nonoverlap_windows": scoreable}
    atomic_json(args.out, {
        "format": "group_event_state_v0_3_5_long_horizon_estimability_v3",
        "contract": {
            "window": "wall-clock window may cross exposure holes; likelihood offset uses observed seconds",
            "state_carry_gap_seconds": 600.0,
            "exposure_support": "original target coverage; short carried gaps retain zero exposure weight",
            "minimum_exposure_fraction": MINIMUM_EXPOSURE_FRACTION,
            "split": "per-horizon; FIT>=4H, INNER>=2H, final holdout>=3H observed exposure inside immutable <80pct prefix",
            "checkpoint_selection": "one horizon or event offset per training unit",
            "wrong_time": "cross-piece same-patient donor, >=H apart, clock time within 2h; final scorer also matches recent rate and exposure",
        },
        "summary": summary, "rows": rows,
        "development_targets_read": False, "sealed_partition_opened": False,
    })
    print(args.out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
