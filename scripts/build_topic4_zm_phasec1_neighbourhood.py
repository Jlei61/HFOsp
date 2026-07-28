#!/usr/bin/env python3
"""Build immutable native and independent-dt/2 Phase-C1 coordinates.

This is the middle stage of the acyclic Phase-C lock:

    phasec_input_manifest.json -> coordinate manifests -> phasec_manifest.json

The builder never reads or writes a production Phase-C manifest and never runs
an SNN continuation.  Each resolution uses its own six independently captured
full slow fields.  The activity-independent anatomy panels remain selected by
the parent native configuration SHA at both resolutions.
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

import scripts.run_topic4_zm_branch_decision as R  # noqa: E402
import src.topic4_zm_checkpoint as CK  # noqa: E402
import src.topic4_zm_ictal_carrier as CG  # noqa: E402
import src.topic4_zm_phasec_contract as PCC  # noqa: E402
import src.topic4_zm_phasec_neighbourhood as N  # noqa: E402


OUT = ROOT / "results/topic4_sef_hfo/zm_phase_c_tonic_identity"
INPUT_MANIFEST = OUT / "phasec_input_manifest.json"
COORD_DIR = OUT / "phasec1_coordinates"
COORD_MANIFESTS = {
    "dt": OUT / "phasec1_coordinate_manifest_dt.json",
    "dt2": OUT / "phasec1_coordinate_manifest_dt2.json",
}
SEEDS_BY_RESOLUTION = {"dt": (1, 3, 4), "dt2": (1, 3)}
COORDINATE_SCHEMA = "zm_phasec1_coordinate_manifest_v2_2026-07-28"


def _sha_file(path):
    return PCC.sha256_file(Path(path))


def _load_input(path=INPUT_MANIFEST, *, allow_virtual=False):
    path = Path(path)
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            parent = json.load(handle)
        file_sha = _sha_file(path)
    elif allow_virtual:
        parent = PCC.build_input_manifest(ROOT)
        exact_bytes = (
            json.dumps(
                parent, indent=2, sort_keys=True, ensure_ascii=False,
                allow_nan=False,
            ) + "\n"
        ).encode("utf-8")
        file_sha = hashlib.sha256(exact_bytes).hexdigest()
    else:
        raise RuntimeError(f"Phase-C input manifest is missing: {path}")
    PCC.validate_manifest(parent)
    if (
        parent["schema"] != PCC.PHASEC_INPUT_VERSION
        or parent["production_authorized"] is not False
    ):
        raise RuntimeError(
            "coordinate builder requires the non-production Phase-C input lock"
        )
    expected = PCC.build_input_manifest(ROOT)
    PCC.assert_manifest_matches(parent, expected)
    if parent["c1"]["primary_cell_names"] != list(N.PRIMARY_CELL_NAMES):
        raise RuntimeError("input manifest primary C1 names differ")
    if parent["c1"]["secondary_shell_cell_names"] != list(N.SHELL_CELL_NAMES):
        raise RuntimeError("input manifest shell C1 names differ")
    return parent, file_sha


def _source_row(parent, resolution, seed):
    native = parent["per_seed"][str(seed)]
    if resolution == "dt":
        return {
            "config_sha": native["canonical_config_sha"],
            "anchor_path": native["anchor_path"],
            "anchor_file_sha256": native["anchor_file_sha256"],
            "c1_source_states": native["c1_source_states"],
            "panel_selection_config_sha": native[
                "panel_selection_config_sha"
            ],
            "panel_sha256": native["fixed_panels"]["panel_sha256"],
        }
    row = native.get("resolution_confirmations", {}).get("dt2")
    if not isinstance(row, dict):
        raise RuntimeError(f"seed {seed} lacks independent dt2 source lock")
    if row["parent_config_sha"] != native["canonical_config_sha"]:
        raise RuntimeError(f"seed {seed} dt2 parent/native config mismatch")
    return {
        "config_sha": row["config_sha"],
        "anchor_path": row["anchor_path"],
        "anchor_file_sha256": row["anchor_file_sha256"],
        "c1_source_states": row["c1_source_states"],
        "panel_selection_config_sha": row["panel_selection_config_sha"],
        "panel_sha256": row["fixed_panels"]["panel_sha256"],
    }


def _state_rows(source):
    rows = {}
    for ref in source["c1_source_states"]:
        key = (ref["fast_phase"], ref["bin_name"])
        if key in rows:
            raise RuntimeError(f"duplicate C1 source state {key}")
        rows[key] = ref
    expected = {
        (phase, stage)
        for phase in N.DEFAULT_PHASES
        for stage in N.PRIMARY_STAGES
    }
    if set(rows) != expected:
        raise RuntimeError(
            f"C1 source coverage mismatch: missing={sorted(expected-set(rows))} "
            f"extra={sorted(set(rows)-expected)}"
        )
    return rows


def _load_observed(source, ne, dt_ms):
    refs = _state_rows(source)
    observed = {phase: {} for phase in N.DEFAULT_PHASES}
    input_rows = []
    for phase in N.DEFAULT_PHASES:
        for stage in N.PRIMARY_STAGES:
            ref = refs[(phase, stage)]
            path = ROOT / ref["path"]
            if _sha_file(path) != ref["file_sha256"]:
                raise RuntimeError(f"C1 state file SHA drift: {ref['path']}")
            state, state_manifest = CK.load_state_npz(
                path,
                expected_config_sha=source["config_sha"],
                expected_dt=float(dt_ms),
            )
            if state_manifest["state_hash"] != ref["state_hash"]:
                raise RuntimeError(f"C1 semantic state hash drift: {ref['path']}")
            if state["slow.z"].shape[0] < ne or state["slow.m"].shape[0] < ne:
                raise RuntimeError(f"C1 slow field shorter than NE: {ref['path']}")
            slow = {
                "z": np.asarray(state["slow.z"][:ne], np.float64).copy(),
                "m": np.asarray(state["slow.m"][:ne], np.float64).copy(),
                "S_G": float(np.asarray(state["slow.S_G"])),
            }
            observed[phase][stage] = slow
            input_rows.append({
                "phase": phase,
                "stage": stage,
                "path": ref["path"],
                "file_sha256": ref["file_sha256"],
                "state_hash": ref["state_hash"],
                "slow_state_sha256": N.slow_state_sha256(slow),
            })
    return observed, input_rows


def _cell_metadata(cell, row):
    return {
        "cell_id": cell["cell_id"],
        "tier": cell["kind"],
        "array_row": int(row),
        "status": cell["status"],
        "reasons": list(cell["reasons"]),
        "clipped": bool(cell["clipped"]),
        "trajectory_id": cell["trajectory_id"],
        "path_index": int(cell["path_index"]),
        "path_coordinate": float(cell["path_coordinate"]),
        "path_direction": cell["path_direction"],
        "state_sha256": N.slow_state_sha256(cell["state"]),
        "summary7": [
            float(value) for value in np.asarray(cell["summary7"], np.float64)
        ],
        "standardized_distance_from_anchor_manifold": float(
            cell["standardized_distance_from_anchor_manifold"]
        ),
        "reconstruction_error_standardized_rms": float(
            cell["reconstruction_error_standardized_rms"]
        ),
    }


def _seed_coordinates(resolution, seed, source, geometry):
    ne = int(geometry["NE"])
    dt_ms = float(0.1 if resolution == "dt" else 0.05)
    observed, input_rows = _load_observed(source, ne, dt_ms)
    coordinates = N.build_coordinate_set(
        observed,
        core_mask=geometry["core"],
        axis_coord=geometry["along"],
        perpendicular_coord=geometry["perpendicular"],
    )
    arrays = N.coordinate_array_payload(coordinates)
    npz_bytes = N.deterministic_npz_bytes(arrays)
    npz_path = COORD_DIR / resolution / f"seed{seed}.npz"
    cells = list(coordinates["primary"]) + list(
        coordinates["secondary_shell"]
    )
    geometry_sha = hashlib.sha256(
        np.ascontiguousarray(
            np.column_stack([
                np.asarray(geometry["along"], np.float64),
                np.asarray(geometry["perpendicular"], np.float64),
                np.asarray(geometry["core"], np.uint8),
            ])
        ).tobytes()
    ).hexdigest()
    row = {
        "seed": seed,
        "resolution": resolution,
        "dt_ms": dt_ms,
        "config_sha": source["config_sha"],
        "panel_selection_config_sha": source["panel_selection_config_sha"],
        "panel_selection_resolution": "parent_native_dt",
        "panel_sha256": source["panel_sha256"],
        "anchor_path": source["anchor_path"],
        "anchor_file_sha256": source["anchor_file_sha256"],
        "input_states": input_rows,
        "geometry_sha256": geometry_sha,
        "npz_path": str(npz_path.relative_to(ROOT)),
        "npz_file_sha256": N.sha256_bytes(npz_bytes),
        "npz_semantic_sha256": N.semantic_array_sha256(arrays),
        "array_keys": sorted(arrays),
        "array_float_contract": "all floating arrays are exact float64",
        "n_E": ne,
        "n_primary": len(coordinates["primary"]),
        "n_secondary_shell": len(coordinates["secondary_shell"]),
        "n_primary_valid": sum(
            c["status"] == "valid" for c in coordinates["primary"]
        ),
        "n_secondary_shell_valid": sum(
            c["status"] == "valid"
            for c in coordinates["secondary_shell"]
        ),
        "cells": [
            _cell_metadata(cell, i) for i, cell in enumerate(cells)
        ],
        "basis_sha256": N.sha256_bytes(
            np.ascontiguousarray(
                arrays["basis_directions_standardized"]
            ).tobytes()
        ),
        "fullfield_mode_sign_alignment": coordinates["basis"][
            "fullfield_mode_sign_alignment"
        ],
        "summary7_contract": coordinates["summary7_contract"],
        "envelope_contract": {
            "full_field_quantiles": [0.005, 0.995],
            "summary7_quantiles": [0.005, 0.995],
            "iqr_pad": 0.25,
            "hard_bounds": {
                "z": [0.0, 1.0],
                "m_min": 0.0,
                "S_G": [0.0, 1.0],
            },
            "clipping_allowed": False,
        },
    }
    return row, npz_path, npz_bytes


def _geometry(seed):
    # Geometry and anatomy panels are intentionally inherited from native dt.
    # The dt2 substrate has identical positions/connectivity seed, but a
    # different numerical integration configuration.
    ctx = R.build_context(seed, resolution="dt")
    along, perpendicular = CG.axis_transverse_coords(
        ctx["S"]["posE"], ctx["S"]["src_xy"], ctx["S"]["axis_unit"]
    )
    return {
        "NE": int(ctx["S"]["NE"]),
        "core": np.asarray(ctx["core"], bool).copy(),
        "along": np.asarray(along, np.float64).copy(),
        "perpendicular": np.asarray(perpendicular, np.float64).copy(),
    }


def build_resolution(
    parent, parent_input_file_sha, resolution, geometry_by_seed
):
    rows = {}
    payloads = []
    for seed in SEEDS_BY_RESOLUTION[resolution]:
        geometry = geometry_by_seed[seed]
        source = _source_row(parent, resolution, seed)
        row, path, value = _seed_coordinates(
            resolution, seed, source, geometry
        )
        rows[str(seed)] = row
        payloads.append((path, value))
    producer_paths = (
        Path(__file__).resolve(),
        ROOT / "src/topic4_zm_phasec_neighbourhood.py",
        ROOT / "src/topic4_zm_phasec_contract.py",
    )
    semantic_payload = {
        "schema": COORDINATE_SCHEMA,
        "resolution": resolution,
        "neighbourhood_version": N.PHASEC_NEIGHBOURHOOD_VERSION,
        "parent_phasec_input_manifest_path": str(
            INPUT_MANIFEST.relative_to(ROOT)
        ),
        "parent_phasec_input_manifest_file_sha256": parent_input_file_sha,
        "parent_phasec_input_manifest_sha256": parent["manifest_sha256"],
        "producer_file_sha256": {
            str(path.relative_to(ROOT)): _sha_file(path)
            for path in producer_paths
        },
        "primary_cell_names": list(N.PRIMARY_CELL_NAMES),
        "secondary_shell_cell_names": list(N.SHELL_CELL_NAMES),
        "seeds": rows,
        "summary7_contract": {
            "names": list(N.SUMMARY7_NAMES),
            "units": list(N.SUMMARY7_UNITS),
            "definition": N.SUMMARY7_DEFINITION,
        },
        "claim_boundary": {
            "primary": "empirically_supported_convex_sensitivity",
            "secondary_shell": "nearby_extrapolated_candidate_only",
            "dynamic_reachability_established": False,
            "lifecycle_established": False,
        },
    }
    manifest = dict(semantic_payload)
    manifest["semantic_sha256"] = N.sha256_bytes(
        N.canonical_json_bytes(semantic_payload)
    )
    manifest["manifest_sha256"] = N.sha256_bytes(
        N.canonical_json_bytes(manifest)
    )
    return manifest, payloads


def build_all(resolutions=("dt", "dt2"), *, allow_virtual_input=False):
    parent, parent_input_file_sha = _load_input(
        allow_virtual=allow_virtual_input
    )
    required_seeds = sorted({
        seed
        for resolution in resolutions
        for seed in SEEDS_BY_RESOLUTION[resolution]
    })
    geometry_by_seed = {seed: _geometry(seed) for seed in required_seeds}
    return {
        resolution: build_resolution(
            parent, parent_input_file_sha, resolution, geometry_by_seed
        )
        for resolution in resolutions
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolution",
        choices=("dt", "dt2", "all"),
        default="all",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    resolutions = (
        ("dt", "dt2") if args.resolution == "all" else (args.resolution,)
    )
    built = build_all(
        resolutions, allow_virtual_input=bool(args.check_only)
    )
    if args.check_only:
        print(json.dumps({
            "status": "validated",
            "resolutions": {
                resolution: {
                    "manifest_sha256": manifest["manifest_sha256"],
                    "semantic_sha256": manifest["semantic_sha256"],
                    "seeds": sorted(int(k) for k in manifest["seeds"]),
                    "primary_per_seed": len(N.PRIMARY_CELL_NAMES),
                    "shell_per_seed": len(N.SHELL_CELL_NAMES),
                }
                for resolution, (manifest, _payloads) in built.items()
            },
        }, sort_keys=True))
        return 0
    statuses = {}
    for resolution, (manifest, payloads) in built.items():
        for path, value in payloads:
            statuses[str(path.relative_to(ROOT))] = N.write_bytes_once(
                path, value
            )
        manifest_path = COORD_MANIFESTS[resolution]
        statuses[str(manifest_path.relative_to(ROOT))] = N.write_json_once(
            manifest_path, manifest
        )
    print(json.dumps({"status": statuses}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
