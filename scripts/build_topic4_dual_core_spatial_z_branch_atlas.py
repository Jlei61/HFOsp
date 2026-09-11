#!/usr/bin/env python3
"""Assemble the verified dual-core spatial-Z continuation families.

The global-recruited family is stitched only at duplicated, bit-identical
restart states.  The independently continued core-A-entry family is retained
as a separate locus; this script explicitly tests rather than assumes that the
two families meet in the available continuation range.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_dual_core_spatial_z_bifurcation import (  # noqa: E402
    atomic_json,
    atomic_npz,
    sha256,
)


FIELDS = (
    "s", "mean_e_hz", "tangent_s", "core_a_hz", "core_b_hz",
    "surround_hz", "rates",
)


def _take(archive, prefix: str, selection) -> dict[str, np.ndarray]:
    return {
        field: np.asarray(archive[f"{prefix}__{field}"])[selection]
        for field in FIELDS
    }


def _assert_duplicate(left: dict, right: dict, label: str) -> None:
    # The restart re-normalises its arclength tangent; the corrected state and
    # all state-derived summaries, not tangent_s, are the identity contract.
    for field in ("s", "mean_e_hz", "core_a_hz", "core_b_hz",
                  "surround_hz", "rates"):
        if not np.array_equal(left[field][-2:], right[field][:2]):
            maximum = float(np.max(np.abs(
                np.asarray(left[field][-2:], float)
                - np.asarray(right[field][:2], float))))
            raise RuntimeError(
                f"{label}: restart states are not exact duplicates; "
                f"{field} max difference={maximum}")


def _concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        field: np.concatenate([part[field] for part in parts], axis=0)
        for field in FIELDS
    }


def _fold_indices(tangent) -> np.ndarray:
    values = np.asarray(tangent, float)
    return np.flatnonzero(values[:-1] * values[1:] <= 0.0)


def _closest_cross_family(entry: dict, recruited: dict,
                          *, s_tolerance: float) -> dict:
    best = None
    for entry_index, parameter in enumerate(entry["s"]):
        candidates = np.flatnonzero(
            np.abs(recruited["s"] - parameter) <= float(s_tolerance))
        if candidates.size == 0:
            continue
        distances = 1000.0 * np.sqrt(np.mean(
            (recruited["rates"][candidates]
             - entry["rates"][entry_index]) ** 2,
            axis=1,
        ))
        local = int(np.argmin(distances))
        record = {
            "entry_index": int(entry_index),
            "recruited_index": int(candidates[local]),
            "entry_s": float(parameter),
            "recruited_s": float(recruited["s"][candidates[local]]),
            "full_state_rms_hz": float(distances[local]),
        }
        if best is None or record["full_state_rms_hz"] < best[
                "full_state_rms_hz"]:
            best = record
    if best is None:
        raise RuntimeError("continuation families have no overlapping s samples")
    best["s_match_tolerance"] = float(s_tolerance)
    return best


def main() -> None:
    base = Path(
        "/data/hfosp_topic4_fig45_artifacts/fig5/"
        "data_driven_dual_core_spatial_z/bifurcation")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", default=str(base / "dualcore_spatial_z_bifurcation.json"))
    parser.add_argument(
        "--extension", action="append", default=None,
        help="ordered extension NPZ; defaults to the four audited restarts")
    parser.add_argument(
        "--out-prefix", default=str(
            base / "branch_atlas/dualcore_spatial_z_branch_atlas"))
    args = parser.parse_args()

    result_path = Path(args.result).resolve()
    base_arrays_path = result_path.with_suffix(".npz")
    base_arrays = np.load(base_arrays_path, allow_pickle=False)
    extension_paths = [Path(path).resolve() for path in args.extension] \
        if args.extension else [
            base / "branch_extensions/recruited_arc_plus8000.npz",
            base / "branch_extensions/recruited_arc_refined_bt10_plus3000.npz",
            base / "branch_extensions/recruited_arc_refined2_plus5000.npz",
            base / "branch_extensions/recruited_arc_refined3_plus10000.npz",
        ]
    extensions = [np.load(path, allow_pickle=False) for path in extension_paths]

    initial = _take(base_arrays, "recruited_arc", slice(None))
    extension = _take(extensions[0], "extension", slice(None))
    refined1 = _take(extensions[1], "extension", slice(None))
    refined2 = _take(extensions[2], "extension", slice(None))
    refined3 = _take(extensions[3], "extension", slice(None))
    _assert_duplicate(initial, extension, "initial to extension")
    _assert_duplicate(_take(extensions[0], "extension", slice(None, -10)),
                      refined1, "extension to refined1")
    _assert_duplicate(_take(extensions[1], "extension", slice(None, -10)),
                      refined2, "refined1 to refined2")
    _assert_duplicate(refined2, refined3, "refined2 to refined3")

    recruited = _concatenate([
        initial,
        _take(extensions[0], "extension", slice(2, -10)),
        _take(extensions[1], "extension", slice(2, -10)),
        _take(extensions[2], "extension", slice(2, -2)),
        _take(extensions[3], "extension", slice(2, None)),
    ])
    entry = {
        field: np.asarray(base_arrays[f"entry_saddle__{field}"])
        for field in FIELDS
    }
    recruited_folds = _fold_indices(recruited["tangent_s"])
    entry_folds = _fold_indices(entry["tangent_s"])
    closest = _closest_cross_family(entry, recruited, s_tolerance=2e-5)

    out_prefix = Path(args.out_prefix).resolve()
    arrays = {}
    for prefix, branch in (("global_recruited", recruited),
                           ("core_a_entry", entry)):
        for field, value in branch.items():
            arrays[f"{prefix}__{field}"] = value
    arrays["global_recruited__fold_indices"] = recruited_folds
    arrays["core_a_entry__fold_indices"] = entry_folds
    atomic_npz(out_prefix.with_suffix(".npz"), **arrays)
    payload = {
        "status": "DUAL_CORE_SPATIAL_Z_MULTIBRANCH_ATLAS_COMPLETE",
        "global_recruited_family": {
            "n_points": int(recruited["s"].size),
            "s_range": [float(np.min(recruited["s"])),
                        float(np.max(recruited["s"]))],
            "fold_count": int(recruited_folds.size),
            "fold_indices": recruited_folds.tolist(),
            "provenance": (
                "continued from the global tonic outer root through exact, "
                "bit-identical pseudo-arclength restart states"),
        },
        "core_a_entry_family": {
            "n_points": int(entry["s"].size),
            "s_range": [float(np.min(entry["s"])),
                        float(np.max(entry["s"]))],
            "fold_count": int(entry_folds.size),
            "fold_indices": entry_folds.tolist(),
            "provenance": (
                "independent continuation from the low-branch zero-mode "
                "partner; a core-A-localized spatial family"),
        },
        "cross_family_match": {
            **closest,
            "match_established": bool(closest["full_state_rms_hz"] < 1e-3),
            "interpretation": (
                "the available continuations are distinct full-state loci; "
                "no line is inserted between them"),
        },
        "stability_contract": (
            "branch loci and folds only; no line style in this atlas encodes "
            "zero-delay or delay-aware stability"),
        "source": {
            "result_json": {"path": str(result_path),
                            "sha256": sha256(result_path)},
            "base_npz": {"path": str(base_arrays_path),
                         "sha256": sha256(base_arrays_path)},
            "extensions": [
                {"path": str(path), "sha256": sha256(path)}
                for path in extension_paths
            ],
        },
    }
    atomic_json(payload, out_prefix.with_suffix(".json"))
    print(json.dumps({
        "status": payload["status"],
        "global_points": payload["global_recruited_family"]["n_points"],
        "global_folds": payload["global_recruited_family"]["fold_count"],
        "entry_points": payload["core_a_entry_family"]["n_points"],
        "entry_folds": payload["core_a_entry_family"]["fold_count"],
        "closest_full_state_rms_hz": closest["full_state_rms_hz"],
        "output": str(out_prefix.with_suffix(".json")),
    }, indent=2))


if __name__ == "__main__":
    main()
