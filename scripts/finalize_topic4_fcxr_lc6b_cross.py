#!/usr/bin/env python3
"""Adjudicate the D x H causal cross with the same versioned comparator round 2 uses.

"Did this cell enter the bursting regime" is answered as an outcome DISTANCE from the reference cell
where neither component was raised, not by a fresh threshold invented here.  Two cells that land in
the same place as D11_H11 did not enter; a cell that lands somewhere else did.

The two diagonal cells are round-2 atlas points recomputed by a different code path, so their spike
trains must be bitwise identical to the atlas.  A mismatch means the cross composition is not
equivalent to the atlas composition and the attribution built on it would be meaningless, so it
raises rather than publishes.
"""
from __future__ import annotations

import argparse
import hashlib
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
import run_topic4_fcxr_lc6b_dh_cross as CROSS  # noqa: E402
from src.topic4_fcxr_lc6b_outcome import (  # noqa: E402
    SCHEMA, coarse_spatial_map, outcome_distance, per_cell_rate_vector, population_rate,
)


OUT = CF.OUT
CROSS_DIR = OUT / "dh_cross"
ADJUDICATION = OUT / "dh_cross_adjudication.json"
GRID = CF.NAT.OUT / "trajectories/C0/spatial_readouts.npz"
REFERENCE = "D11_H11"           #: neither component raised; "entered" is measured against this


def _spike_sha(directory):
    with np.load(Path(directory) / "spikes.npz", allow_pickle=False) as handle:
        return hashlib.sha256(np.ascontiguousarray(handle["steps"]).tobytes()
                              + np.ascontiguousarray(handle["cells"]).tobytes()).hexdigest()


def _readouts(directory, summary, cell_bins, occupancy):
    with np.load(Path(directory) / "spikes.npz", allow_pickle=False) as handle:
        steps = np.asarray(handle["steps"], np.int64)
        cells = np.asarray(handle["cells"], np.int32)
        n_steps, n_cells = int(handle["n_steps"][0]), int(handle["n_cells"][0])
    return {
        "final_second_rate_hz": float(summary["verdict"]["per_second_mean_hz"][-1]),
        "median_active_area_mm2": float(summary["median_active_area_mm2"] or 0.0),
        "per_cell_rate_vector": per_cell_rate_vector(steps, cells, n_steps=n_steps, n_cells=n_cells),
        "population_rate": population_rate(steps, n_steps=n_steps, n_cells=n_cells),
        "coarse_spatial_map": coarse_spatial_map(steps, cells, cell_bins, occupancy, n_steps=n_steps),
    }


def adjudicate():
    with np.load(GRID, allow_pickle=False) as handle:
        cell_bins = np.asarray(handle["cell_bins"], np.int64)
        occupancy = np.asarray(handle["occupancy"], float)

    summaries, readouts = {}, {}
    for cell in CROSS.CELLS:
        for init in CROSS.INITS:
            key = f"{cell}__{init}"
            path = CROSS_DIR / key
            if not (path / "summary.json").is_file():
                raise RuntimeError(f"cross incomplete: {key}")
            summaries[key] = json.loads((path / "summary.json").read_text())
            readouts[key] = _readouts(path, summaries[key], cell_bins, occupancy)

    # free correctness check: a diagonal cell IS an atlas point, computed another way
    diagonal = {}
    for cell, field in CROSS.DIAGONAL_EQUIVALENT.items():
        for init in CROSS.INITS:
            cross_dir = CROSS_DIR / f"{cell}__{init}"
            atlas_dir = OUT / f"atlas/{field}__{init}"
            identical = _spike_sha(cross_dir) == _spike_sha(atlas_dir)
            diagonal[f"{cell}__{init}"] = {
                "atlas_point": f"{field}__{init}", "spike_train_identical": identical,
                "sha256": _spike_sha(cross_dir)}
            if not identical:
                raise RuntimeError(
                    f"{cell}__{init} does not reproduce atlas {field}__{init} spike for spike; the "
                    "cross composition is not equivalent to the atlas composition")

    entered, per_cell = {}, {}
    for cell in CROSS.CELLS:
        against_reference = {
            init: outcome_distance(readouts[f"{REFERENCE}__{init}"], readouts[f"{cell}__{init}"])
            for init in CROSS.INITS}
        # "entered" == landed somewhere OTHER than the reference cell
        entered[cell] = bool(cell != REFERENCE and not all(
            row["same_outcome_regime"] for row in against_reference.values()))
        per_cell[cell] = {
            "d_field": CROSS.CELLS[cell][0], "h_field": CROSS.CELLS[cell][1],
            "D_mean": summaries[f"{cell}__locked_low"]["d_field_D_mean"],
            "h_gate_mean": summaries[f"{cell}__locked_low"]["h_field_h_gate_mean"],
            "registered_labels": {init: summaries[f"{cell}__{init}"]["verdict"]["label"]
                                  for init in CROSS.INITS},
            "final_second_rate_hz": {init: summaries[f"{cell}__{init}"]["verdict"]["per_second_mean_hz"][-1]
                                     for init in CROSS.INITS},
            "median_active_area_mm2": {init: summaries[f"{cell}__{init}"]["median_active_area_mm2"]
                                       for init in CROSS.INITS},
            "distance_from_reference": against_reference,
            "entered_a_different_regime_than_reference": entered[cell],
            "initialisations_agree": outcome_distance(
                readouts[f"{cell}__locked_low"],
                readouts[f"{cell}__locked_high"])["same_outcome_regime"],
        }

    d_only, h_only, both = entered["D12_H11"], entered["D11_H12"], entered["D12_H12"]
    if h_only and not d_only:
        attribution = "H_OPENS_THE_BURSTING_REGIME_D_ALONE_DOES_NOT"
    elif d_only and not h_only:
        attribution = "D_OPENS_THE_BURSTING_REGIME_H_ALONE_DOES_NOT"
    elif d_only and h_only:
        attribution = "REDUNDANT_ENTRY_EITHER_COMPONENT_ALONE_OPENS_IT"
    elif both:
        attribution = "SYNERGY_ONLY_BOTH_TOGETHER_OPEN_IT"
    else:
        attribution = "NEITHER_CELL_LEFT_THE_REFERENCE_REGIME"

    payload = {
        "schema": SCHEMA, "status": "COMPLETE", "stage": "LC6B_ROUND_3_DH_CROSS",
        "reference_cell": REFERENCE,
        "attribution": attribution,
        "entered_by_cell": entered,
        "per_cell": per_cell,
        "diagonal_free_checks": diagonal,
        "amplitude_effect_of_D_once_H_has_opened_it": {
            "note": "D11_H12 vs D12_H12: what raising D adds once the H gate is already open",
            "median_active_area_mm2": {
                "D11_H12": per_cell["D11_H12"]["median_active_area_mm2"],
                "D12_H12": per_cell["D12_H12"]["median_active_area_mm2"]},
            "final_second_rate_hz": {
                "D11_H12": per_cell["D11_H12"]["final_second_rate_hz"],
                "D12_H12": per_cell["D12_H12"]["final_second_rate_hz"]},
        },
        "claim_boundary": (
            "One boundary (onset vs onset+1 s), one graph, one input stream, two shared "
            "initialisations.  The H gate at the onset field is exactly 0.0000, so the D-only cell "
            "tests raising D while the H actuator contributes nothing at all -- a strong condition, "
            "but tested at this boundary only.  Entry attribution says nothing about what drives the "
            "system PAST the regime: round 1 established that continued D deepening does that."),
        "not_tested": ["perturbation_return", "other boundaries along the path",
                       "second input stream", "termination", "lifecycle"],
        "source_sha256": CROSS._source_hashes(),
    }
    CF.NAT._write_json(ADJUDICATION, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("LC6B cross adjudication requires --confirm-run")
    payload = adjudicate()
    print(json.dumps({
        "attribution": payload["attribution"],
        "entered_by_cell": payload["entered_by_cell"],
        "diagonal_free_checks": {k: v["spike_train_identical"]
                                 for k, v in payload["diagonal_free_checks"].items()},
        "per_cell": {
            cell: {"D": row["D_mean"], "h_gate": row["h_gate_mean"],
                   "rate": row["final_second_rate_hz"], "area": row["median_active_area_mm2"],
                   "inits_agree": row["initialisations_agree"]}
            for cell, row in payload["per_cell"].items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
