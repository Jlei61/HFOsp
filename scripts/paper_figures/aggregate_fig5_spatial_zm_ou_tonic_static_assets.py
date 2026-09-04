#!/usr/bin/env python3
"""Aggregate exact event/energy and parallel probe chunks for static Fig. 5."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "src" / "snn_engine"
for search_path in (ROOT, ENGINE):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.paper_figures import (  # noqa: E402
    build_fig5_spatial_zm_ou_tonic_static_assets as build,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def join_chunks(paths, expected_label):
    records = []
    metas = []
    reference_positions = None
    reference_contacts = None
    for path in paths:
        path = path.resolve()
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["state_label"] != expected_label:
            raise RuntimeError(f"{path}: wrong state label")
        if not meta["sham_continuation_rate_bit_identical"]:
            raise RuntimeError(f"{path}: sham continuation was not exact")
        block = load_npz(path)
        if reference_positions is None:
            reference_positions = block["positions_E"]
            reference_contacts = block["contact_xy_mm"]
        elif not (
            np.array_equal(reference_positions, block["positions_E"])
            and np.array_equal(reference_contacts, block["contact_xy_mm"])
        ):
            raise RuntimeError("probe chunks use different frozen geometry")
        for local_index, site_index in enumerate(block["site_index"]):
            records.append({
                "site_index": int(site_index),
                "site_xy_mm": block["site_xy_mm"][local_index],
                "response_early": block["response_early"][local_index],
                "row": meta["rows"][local_index],
            })
        metas.append(meta)
    records.sort(key=lambda row: row["site_index"])
    if [row["site_index"] for row in records] != list(range(16)):
        raise RuntimeError(f"{expected_label}: chunks do not cover sites 0..15 once")
    reference = metas[0]
    for meta in metas[1:]:
        for key in ("checkpoint_time_ms", "site_contract", "dose_cells", "response"):
            if meta[key] != reference[key]:
                raise RuntimeError(f"{expected_label}: chunk contract drift in {key}")
    return {
        "positions_E": reference_positions,
        "contact_xy_mm": reference_contacts,
        "site_xy_mm": np.asarray([row["site_xy_mm"] for row in records], np.float32),
        "response_early": np.asarray(
            [row["response_early"] for row in records], np.float32),
        "rows": [row["row"] for row in records],
        "meta": reference,
        "paths": [str(path.resolve()) for path in paths],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-chunks", type=Path, nargs="+", required=True)
    parser.add_argument("--early-runaway-chunks", type=Path, nargs="+", required=True)
    parser.add_argument("--event-asset", type=Path, default=build.EVENT_NPZ)
    parser.add_argument("--out", type=Path, default=build.ASSET_NPZ)
    args = parser.parse_args()

    event = load_npz(args.event_asset.resolve())
    event_meta = json.loads(args.event_asset.with_suffix(".json").read_text())
    if event_meta["status"] != "FIG5_STATIC_EVENT_AND_ENERGY_COMPLETE":
        raise RuntimeError("event asset is not complete")
    if not all(row["bit_identical"]
               for row in event_meta["replay_comparisons"].values()):
        raise RuntimeError("event replay does not match the locked trajectory")
    low = join_chunks(args.low_chunks, "low_activity")
    high = join_chunks(args.early_runaway_chunks, "early_runaway")
    if not (
        np.array_equal(low["positions_E"], high["positions_E"])
        and np.array_equal(low["contact_xy_mm"], high["contact_xy_mm"])
        and np.array_equal(low["site_xy_mm"], high["site_xy_mm"])
        and np.array_equal(event["positions_E"], low["positions_E"])
        and np.array_equal(event["contact_xy_mm"], low["contact_xy_mm"])
    ):
        raise RuntimeError("event and perturbation assets use different geometry")

    output_arrays = dict(event)
    output_arrays.update({
        "site_xy_mm": low["site_xy_mm"],
        "low_response_early": low["response_early"],
        "early_runaway_response_early": high["response_early"],
        "low_response_early_mean": np.mean(
            low["response_early"], axis=0).astype(np.float32),
        "early_runaway_response_early_mean": np.mean(
            high["response_early"], axis=0).astype(np.float32),
    })
    args.out = args.out.resolve()
    atomic_npz(args.out, **output_arrays)
    payload = {
        "status": "FIG5_SPATIAL_ZM_OU_TONIC_STATIC_ASSETS_COMPLETE",
        "seed": 1842,
        "source": str(build.SOURCE_NPZ),
        "source_sha256": build.SOURCE_SHA256,
        "event_asset": str(args.event_asset.resolve()),
        "event_asset_sha256": sha256(args.event_asset.resolve()),
        "panel_C": {
            "event_selection": event_meta["event_selection"],
            "event_peak_ms": event_meta["event_peak_ms"],
            "event_window_ms": event_meta["event_window_ms"],
            "n_event_active_E": event_meta["n_event_active_E"],
            "n_contacts_with_local_first_spike": event_meta[
                "n_contacts_with_local_first_spike"],
            "event_order_measure": event_meta["event_order_measure"],
            "early_activity_energy_window_ms": event_meta[
                "early_activity_energy_window_ms"],
            "early_activity_energy_measure": event_meta[
                "early_activity_energy_measure"],
        },
        "panel_D": {
            "state_times_ms": {
                "low_activity": low["meta"]["checkpoint_time_ms"],
                "early_runaway": high["meta"]["checkpoint_time_ms"],
            },
            "site_contract": low["meta"]["site_contract"],
            "dose_cells": low["meta"]["dose_cells"],
            "dose_origin": (
                "frozen 16-cell weak dose inherited from the accepted Fig5 "
                "low-state assay; not selected from these response maps"),
            "response": low["meta"]["response"],
            "aggregation": "equal-weight mean over 16 paired stratified-random sites",
            "low_n_e1_evaluable": int(sum(
                bool(row["e1_evaluable"]) for row in low["rows"])),
            "early_runaway_n_e1_evaluable": int(sum(
                bool(row["e1_evaluable"]) for row in high["rows"])),
            "low_n_probe_attributable_event": int(sum(
                bool(row["probe_attributable_event_200ms"])
                for row in low["rows"])),
            "early_runaway_n_probe_attributable_event": int(sum(
                bool(row["probe_attributable_event_200ms"])
                for row in high["rows"])),
            "low_rows": low["rows"],
            "early_runaway_rows": high["rows"],
            "chunks": {"low": low["paths"], "early_runaway": high["paths"]},
        },
        "checkpoint_contract": event_meta["checkpoints"],
        "output_npz": str(args.out),
        "claim_boundary": (
            "single frozen network and one inherited weak dose; state-contrast "
            "mechanistic assay, not a population estimate across network seeds"),
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "out": str(args.out),
        "sha256": sha256(args.out),
        "low_evaluable": payload["panel_D"]["low_n_e1_evaluable"],
        "early_runaway_evaluable": payload["panel_D"][
            "early_runaway_n_e1_evaluable"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
