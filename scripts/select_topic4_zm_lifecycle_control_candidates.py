#!/usr/bin/env python3
"""Select phenotype-diverse persistent M-panel states for finite control."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint"


def control_ready(row):
    """A control source must be entered, persistent, active, and macroscopically visible."""
    return bool(
        row.get("status") == "complete"
        and row.get("onset_ms") is not None
        and row.get("offset_ms") is None
        and row.get("duration_right_censored") is True
        and float(row.get("g_M", 0.0)) > 0.0
        and float(row.get("median_energy_gain_db", float("-inf"))) >= 6.0
        and float(row.get("energy_occupancy_6db", 0.0)) >= 0.20
    )


def _priority(row):
    """Prefer a partially suppressed, still-active state near the exit boundary."""
    response = row.get("paired_M_response", {})
    ratio = response.get("ratio_tail_core_mean_hz", response.get("ratio_core_mean_hz"))
    if ratio is None:
        # Missing continuous evidence is allowed only as a low-priority fallback.
        return (2, float("inf"), -float(row["g_M"]), float(row["tau_M_ms"]))
    ratio = float(ratio)
    if 0.35 <= ratio <= 0.90:
        band = 0
    elif ratio > 0.90:
        band = 1
    else:
        # Very strong suppression is closer to prevention/silence than to a
        # controllable persistent carrier and is deliberately deprioritised.
        band = 2
    return (band, abs(ratio - 0.60), -float(row["g_M"]), float(row["tau_M_ms"]))


def select_candidates(surface, *, max_candidates=4, selection_ranks=None):
    selected = []
    ranks = sorted({int(row["selection_rank"]) for row in surface.get("rows", [])})
    if selection_ranks is not None:
        allowed = {int(value) for value in selection_ranks}
        ranks = [rank for rank in ranks if rank in allowed]
    for rank in ranks:
        candidates = [
            row for row in surface["rows"]
            if int(row["selection_rank"]) == rank and control_ready(row)
        ]
        if not candidates:
            continue
        chosen = min(candidates, key=_priority)
        selected.append({
            **chosen,
            "control_selection_role": "persistent_near_exit_boundary",
            "control_selection_priority": list(_priority(chosen)),
        })
    return selected[: int(max_candidates)]


def choose_control_time(fine_time_ms, core_rate_hz, *, onset_ms, delay_ms=1500.0):
    """Lock the first 50-ms active window after the registered post-onset delay."""
    time = np.asarray(fine_time_ms, float)
    core = np.asarray(core_rate_hz, float)
    if time.shape != core.shape or time.size < 2:
        raise ValueError("control timing requires aligned fine time/core-rate arrays")
    dt = float(np.median(np.diff(time)))
    window = max(1, int(round(50.0 / dt)))
    smooth = np.convolve(core, np.ones(window) / window, mode="valid")
    sustained = np.convolve((core >= 50.0).astype(np.int16), np.ones(window, np.int16), mode="valid") == window
    candidate_time = time[:smooth.size]
    earliest = float(onset_ms) + float(delay_ms)
    hits = np.flatnonzero((candidate_time >= earliest) & sustained)
    if not hits.size:
        raise ValueError("persistent candidate has no 50-ms active control window after onset+delay")
    index = int(hits[0])
    return {
        "control_t0_ms": float(candidate_time[index]),
        "control_timing_rule": "first_50ms_mean_core_ge_50Hz_after_onset_plus_1500ms",
        "uncontrolled_core_mean_hz_at_control": float(smooth[index]),
    }


def attach_control_times(rows):
    attached = []
    for row in rows:
        summary_path = row.get("summary_path")
        if summary_path is None:
            raise ValueError("control candidate is missing its uncontrolled summary path")
        trace_path = ROOT / Path(summary_path).parent / "traces.npz"
        with np.load(trace_path, allow_pickle=False) as data:
            timing = choose_control_time(
                data["fine_time_ms"], data["fine_core_rate_hz"], onset_ms=row["onset_ms"],
            )
        attached.append({**row, **timing})
    return attached


def build_selection(surface, *, source_path, max_candidates=4, selection_ranks=None):
    selected = select_candidates(
        surface, max_candidates=max_candidates, selection_ranks=selection_ranks,
    )
    if not selected:
        raise ValueError("M response surface contains no persistent control-ready candidate")
    return {
        "schema": "topic4_zm_lifecycle_control_selection_v1_2026-08-02",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_source": str(source_path),
        "semantic_scope": (
            "phenotype-diverse persistent seed1 checkpoint-fork states for finite-control "
            "development; not carrier acceptance"
        ),
        "n_selected": len(selected),
        "selected": attach_control_times(selected),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surface-json", type=Path, default=OUT / "m_response_surface.json")
    ap.add_argument("--output", type=Path, default=OUT / "control_selection.json")
    ap.add_argument("--max-candidates", type=int, default=4)
    ap.add_argument("--selection-ranks", help="optional comma-separated original M selection ranks")
    args = ap.parse_args()
    surface = json.loads(args.surface_json.read_text())
    payload = build_selection(
        surface,
        source_path=args.surface_json.relative_to(ROOT) if args.surface_json.is_absolute() else args.surface_json,
        max_candidates=args.max_candidates,
        selection_ranks=(
            None if not args.selection_ranks else
            [int(value) for value in args.selection_ranks.split(",") if value.strip()]
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
