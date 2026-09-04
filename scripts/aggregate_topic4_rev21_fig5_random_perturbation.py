#!/usr/bin/env python3
"""Aggregate the frozen low/early-ictal random-site responses for Fig. 5D."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_rev9l_forced_source_worker import _atomic_npz  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_state(path: Path, expected_label: str) -> tuple[dict, dict[str, np.ndarray]]:
    path = path.resolve()
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta.get("status") != "REV21_FIG5_RANDOM_PERTURBATION_COMPLETE":
        raise RuntimeError(f"{path}: incomplete perturbation artifact")
    if meta.get("state_label") != expected_label:
        raise RuntimeError(
            f"{path}: expected {expected_label}, found {meta.get('state_label')}")
    if not meta.get("resumed_sham_exact"):
        raise RuntimeError(f"{path}: sham did not exactly resume the source trajectory")
    if sha256(path) != meta["npz"]["sha256"]:
        raise RuntimeError(f"{path}: NPZ hash changed")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    return meta, arrays


def load_state_chunks(paths: list[Path], expected_label: str):
    loaded = [load_state(path, expected_label) for path in paths]
    reference_meta, reference_arrays = loaded[0]
    rows = []
    for meta, arrays in loaded:
        for key in (
            "candidate_id", "substrate", "topology_seed", "dynamics_seed",
            "checkpoint", "checkpoint_manifest", "site_contract",
            "dose_contract", "dose_cells", "window_ms", "response_window",
        ):
            if meta[key] != reference_meta[key]:
                raise RuntimeError(f"{expected_label} chunks have mismatched {key}")
        for key in ("positions_E", "contact_xy_mm"):
            if not np.array_equal(arrays[key], reference_arrays[key]):
                raise RuntimeError(f"{expected_label} chunks have mismatched {key}")
        rows.extend(meta["rows"])
    order = np.argsort(np.concatenate(
        [arrays["site_index"] for _, arrays in loaded]))
    combined = {
        "positions_E": reference_arrays["positions_E"],
        "contact_xy_mm": reference_arrays["contact_xy_mm"],
    }
    for key in (
        "site_index", "site_xy_mm", "excess_per_neuron_early",
        "excess_per_neuron_full", "excess_spikes_early", "e1_evaluable",
    ):
        combined[key] = np.concatenate(
            [arrays[key] for _, arrays in loaded], axis=0)[order]
    meta = dict(reference_meta)
    meta["rows"] = sorted(rows, key=lambda row: int(row["site_index"]))
    meta["chunks"] = [str(path.resolve()) for path in paths]
    return meta, combined


def site_response_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    absolute_total = float(np.sum(np.abs(values)))
    return {
        "mean_excess_spikes_0_50ms": float(np.mean(values)),
        "median_excess_spikes_0_50ms": float(np.median(values)),
        "minimum_excess_spikes_0_50ms": float(np.min(values)),
        "maximum_excess_spikes_0_50ms": float(np.max(values)),
        "positive_site_count": int(np.sum(values > 0)),
        "negative_site_count": int(np.sum(values < 0)),
        "largest_site_share_of_absolute_response": (
            0.0 if absolute_total == 0.0
            else float(np.max(np.abs(values)) / absolute_total)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low", type=Path, nargs="+", required=True)
    parser.add_argument("--early-ictal", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    low_meta, low = load_state_chunks(args.low, "low_activity")
    early_meta, early = load_state_chunks(args.early_ictal, "early_ictal")
    expected_sites = np.arange(16, dtype=np.int16)
    for label, arrays in (("low", low), ("early_ictal", early)):
        if not np.array_equal(arrays["site_index"], expected_sites):
            raise RuntimeError(
                f"{label}: artifact must contain every frozen site exactly once")
    for key in ("positions_E", "contact_xy_mm", "site_xy_mm"):
        if not np.array_equal(low[key], early[key]):
            raise RuntimeError(f"state contrast has mismatched {key}")
    for key in (
        "candidate_id", "substrate", "topology_seed", "dynamics_seed",
        "site_contract", "dose_contract", "dose_cells", "window_ms",
        "response_window",
    ):
        if low_meta[key] != early_meta[key]:
            raise RuntimeError(f"state contrast has mismatched {key}")
    if low_meta["checkpoint_manifest"] != early_meta["checkpoint_manifest"]:
        raise RuntimeError("states do not derive from the same checkpoint manifest")

    low_response = np.asarray(low["excess_per_neuron_early"], np.float32)
    early_response = np.asarray(early["excess_per_neuron_early"], np.float32)
    n_sites = len(expected_sites)
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        out,
        positions_E=np.asarray(low["positions_E"], np.float32),
        contact_xy_mm=np.asarray(low["contact_xy_mm"], np.float32),
        site_index=expected_sites,
        site_xy_mm=np.asarray(low["site_xy_mm"], np.float32),
        low_response_early=low_response,
        early_ictal_response_early=early_response,
        low_response_early_mean=np.mean(low_response, axis=0).astype(np.float32),
        early_ictal_response_early_mean=np.mean(
            early_response, axis=0).astype(np.float32),
        low_response_early_sem=(
            np.std(low_response, axis=0, ddof=1) / np.sqrt(n_sites)
        ).astype(np.float32),
        early_ictal_response_early_sem=(
            np.std(early_response, axis=0, ddof=1) / np.sqrt(n_sites)
        ).astype(np.float32),
        low_excess_spikes_early=np.asarray(
            low["excess_spikes_early"], np.float32),
        early_ictal_excess_spikes_early=np.asarray(
            early["excess_spikes_early"], np.float32),
        low_e1_evaluable=np.asarray(low["e1_evaluable"], bool),
        early_ictal_e1_evaluable=np.asarray(early["e1_evaluable"], bool),
    )
    summary = {
        "status": "REV21_FIG5_LOW_EARLY_ICTAL_CONTRAST_COMPLETE",
        "panel_semantic": (
            "same 16 frozen random perturbation locations, paired probe-minus-sham "
            "response, averaged separately before runaway and at early ictal state"
        ),
        "candidate_id": low_meta["candidate_id"],
        "substrate": low_meta["substrate"],
        "topology_seed": int(low_meta["topology_seed"]),
        "dynamics_seed": int(low_meta["dynamics_seed"]),
        "state_times_ms": {
            "low_activity": float(low_meta["checkpoint"]["time_ms"]),
            "early_ictal": float(early_meta["checkpoint"]["time_ms"]),
        },
        "checkpoint_manifest": low_meta["checkpoint_manifest"],
        "site_contract": low_meta["site_contract"],
        "dose_contract": low_meta["dose_contract"],
        "dose_cells": int(low_meta["dose_cells"]),
        "response_window": low_meta["response_window"],
        "aggregation": "equal-weight mean over 16 paired stratified-random sites",
        "all_sites_retained": True,
        "low_n_e1_evaluable": int(np.sum(low["e1_evaluable"])),
        "early_ictal_n_e1_evaluable": int(np.sum(early["e1_evaluable"])),
        "site_response_summary": {
            "low_activity": site_response_summary(low["excess_spikes_early"]),
            "early_ictal": site_response_summary(
                early["excess_spikes_early"]),
        },
        "low_rows": low_meta["rows"],
        "early_ictal_rows": early_meta["rows"],
        "sources": {
            "low": [{"path": str(path.resolve()),
                     "sha256": sha256(path.resolve())} for path in args.low],
            "early_ictal": [{"path": str(path.resolve()),
                             "sha256": sha256(path.resolve())}
                            for path in args.early_ictal],
        },
        "npz": {"path": str(out), "sha256": sha256(out)},
        "claim_boundary": (
            "single realized dual-core network, one dynamics seed and one frozen "
            "probe dose; this is a within-trajectory state contrast, not a "
            "population estimate across network seeds"
        ),
    }
    atomic_write_json(summary, str(out.with_suffix(".json")))
    print(json.dumps({
        "status": summary["status"],
        "out": str(out),
        "state_times_ms": summary["state_times_ms"],
        "low_n_e1_evaluable": summary["low_n_e1_evaluable"],
        "early_ictal_n_e1_evaluable": summary["early_ictal_n_e1_evaluable"],
    }, indent=2))


if __name__ == "__main__":
    main()
