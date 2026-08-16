#!/usr/bin/env python3
"""The single versioned adjudication entry for the LC6B atlas.

Round 2's outcome comparison lived in the plotting script, which is the wrong home for a verdict:
not versioned with the result, borrowing a tolerance registered for something else, and comparing
only two scalars.  This script is that comparison's only home now.  The plot reads what this writes.

It reports the registered classifier label and the outcome vector SEPARATELY.  They answer different
questions -- the label asks "what regime is this run in", the outcome vector asks "did two runs of
one field land in the same place" -- and collapsing them is how a drift-gate coin flip became three
bistability candidates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src/snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_topic4_fcxr_lc6b_atlas as ATLAS  # noqa: E402
import run_topic4_fcxr_lc6b_clamp_forks as CF  # noqa: E402
from src.topic4_fcxr_lc6b_outcome import (  # noqa: E402
    SCHEMA, TAIL_MS, coarse_spatial_map, outcome_distance, per_cell_rate_vector,
    phase_aligned_correlation, population_rate,
)


OUT = CF.OUT
ADJUDICATION = OUT / "atlas_outcome_adjudication.json"
#: The 32x32 grid the whole of LC6A/LC6B uses.  Same substrate, so the binning travels with it and
#: no substrate rebuild is needed to place spikes in space.
GRID = CF.NAT.OUT / "trajectories/C0/spatial_readouts.npz"


def _spikes(point_id):
    with np.load(OUT / f"atlas/{point_id}/spikes.npz", allow_pickle=False) as handle:
        return (np.asarray(handle["steps"], np.int64), np.asarray(handle["cells"], np.int32),
                int(handle["n_steps"][0]), int(handle["n_cells"][0]))


def _readouts(point_id, summary, cell_bins, occupancy):
    steps, cells, n_steps, n_cells = _spikes(point_id)
    return {
        "final_second_rate_hz": float(summary["verdict"]["per_second_mean_hz"][-1]),
        "median_active_area_mm2": float(summary["median_active_area_mm2"] or 0.0),
        "per_cell_rate_vector": per_cell_rate_vector(
            steps, cells, n_steps=n_steps, n_cells=n_cells),
        "population_rate": population_rate(steps, n_steps=n_steps, n_cells=n_cells),
        "coarse_spatial_map": coarse_spatial_map(
            steps, cells, cell_bins, occupancy, n_steps=n_steps),
    }


def adjudicate():
    atlas = json.loads((OUT / "natural_path_atlas.json").read_text())
    with np.load(GRID, allow_pickle=False) as handle:
        cell_bins = np.asarray(handle["cell_bins"], np.int64)
        occupancy = np.asarray(handle["occupancy"], float)

    per_field, all_pairs = {}, {}
    for field in atlas["fields_in_time_order"]:
        rows = {init: atlas["rows"][f"{field}__{init}"] for init in ATLAS.INITIALISATIONS}
        readouts = {init: _readouts(rows[init]["point_id"], rows[init], cell_bins, occupancy)
                    for init in ATLAS.INITIALISATIONS}
        primary = outcome_distance(readouts["locked_low"], readouts["locked_high"])
        # The path's own state is compared to both locked ones as well: an outcome that only the
        # trajectory's own state can reach would be invisible in a low-vs-high comparison.
        extra = {
            f"path_native_vs_{init}": outcome_distance(readouts["path_native"], readouts[init])
            for init in ("locked_low", "locked_high")
        }
        registered = {init: rows[init]["verdict"]["label"] for init in ATLAS.INITIALISATIONS}
        per_field[field] = {
            "relative_to_onset_ms": atlas["per_field"][field]["relative_to_onset_ms"],
            "D_mean": atlas["per_field"][field]["D_mean"],
            "h_gate_mean": atlas["per_field"][field]["h_gate_mean"],
            "registered_classifier_labels": registered,
            "registered_label_split_locked_low_vs_high": (
                registered["locked_low"] != registered["locked_high"]),
            "outcome_locked_low_vs_locked_high": primary,
            "outcome_path_native_vs_locked": extra,
            "all_three_same_outcome_regime": bool(
                primary["same_outcome_regime"]
                and all(row["same_outcome_regime"] for row in extra.values())),
        }
        all_pairs[field] = primary["same_outcome_regime"]

    every_field_single = all(row["all_three_same_outcome_regime"] for row in per_field.values())
    payload = {
        "schema": SCHEMA, "status": "COMPLETE",
        "scope": "LC6B round 2 natural-path atlas, one shared input stream",
        "tail_ms": TAIL_MS,
        "verdict": ("CANONICAL_PATH_COMMON_INPUT: NO_MACROSCOPIC_INITIALISATION_SPLIT_DETECTED; "
                    "MONOSTABLE_BOUNDED_BURST_REGIME_CANDIDATE"
                    if every_field_single else
                    "CANONICAL_PATH_COMMON_INPUT: INITIALISATION_SPLIT_CANDIDATE_AT_ONE_OR_MORE_FIELDS"),
        "every_field_single_outcome": every_field_single,
        "per_field": per_field,
        "registered_label_splits": [
            f for f, row in per_field.items() if row["registered_label_split_locked_low_vs_high"]],
        "registered_label_splits_with_same_outcome": [
            f for f, row in per_field.items()
            if row["registered_label_split_locked_low_vs_high"] and all_pairs[f]],
        "separation_of_concerns": (
            "the registered classifier label answers 'what regime is this ONE run in'; the outcome "
            "vector answers 'did TWO runs of one field land in the same place'.  Both are reported; "
            "neither is derived from the other."),
        "claim_boundary": (
            "No macroscopic initialisation split was detected under ONE shared input realisation. "
            "That is consistent with a single attractor and equally consistent with common-noise "
            "synchronisation of more than one attractor, which this design cannot separate. It does "
            "NOT establish that the bounded burst regime is attracting: no perturbation-return test "
            "has been run. It also does not license 'no carrier' -- a monostable oscillatory state "
            "can carry a seizure without any hysteresis."),
        "not_tested": ["perturbation_return", "second_input_stream_for_low_vs_high",
                       "termination", "lifecycle"],
        "source_sha256": ATLAS._source_hashes(),
    }
    CF.NAT._write_json(ADJUDICATION, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("LC6B outcome adjudication requires --confirm-run")
    payload = adjudicate()
    print(json.dumps({
        "verdict": payload["verdict"],
        "every_field_single_outcome": payload["every_field_single_outcome"],
        "registered_label_splits": payload["registered_label_splits"],
        "registered_label_splits_with_same_outcome":
            payload["registered_label_splits_with_same_outcome"],
        "per_field": {
            field: {
                "per_cell_r": round(row["outcome_locked_low_vs_locked_high"]
                                    ["per_cell_rate_vector_correlation"], 5),
                "zero_lag_r": round(row["outcome_locked_low_vs_locked_high"]
                                    ["population_rate_zero_lag_correlation"], 4),
                "phase_aligned_r": round(row["outcome_locked_low_vs_locked_high"]
                                         ["phase_aligned_population_rate_correlation"], 4),
                "lag_ms": row["outcome_locked_low_vs_locked_high"]["phase_alignment_lag_ms"],
                "spatial_r": round(row["outcome_locked_low_vs_locked_high"]
                                   ["coarse_spatial_map_correlation"], 5),
                "same_regime": row["all_three_same_outcome_regime"],
            }
            for field, row in payload["per_field"].items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
