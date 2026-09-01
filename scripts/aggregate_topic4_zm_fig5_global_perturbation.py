#!/usr/bin/env python3
"""Aggregate paired full-sheet random-site perturbation chunks for Figure 5 D."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_rev9l_forced_source_worker import _atomic_npz  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402


def _load_chunks(paths, expected_label):
    rows = []
    blocks = []
    metas = []
    for raw in paths:
        path = Path(raw).resolve()
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["state_label"] != expected_label:
            raise RuntimeError(f"{path}: unexpected state {meta['state_label']}")
        with np.load(path, allow_pickle=False) as handle:
            block = {key: handle[key] for key in handle.files}
        rows.extend(meta["rows"])
        blocks.append(block)
        metas.append(meta)
    order = np.argsort(np.concatenate([b["site_index"] for b in blocks]))
    joined = {}
    for key in ("site_index", "site_xy_mm", "excess_per_neuron_early",
                "excess_per_neuron_full", "susceptibility",
                "excess_spikes_early", "e1_evaluable"):
        joined[key] = np.concatenate([b[key] for b in blocks], axis=0)[order]
    return joined, sorted(rows, key=lambda row: int(row["site_index"])), metas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-chunks", nargs="+", required=True)
    parser.add_argument("--runaway-chunks", nargs="+", required=True)
    parser.add_argument("--mode-snapshot", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    low, low_rows, low_meta = _load_chunks(args.low_chunks, "low_activity")
    high, high_rows, high_meta = _load_chunks(args.runaway_chunks, "runaway")
    expected = np.arange(16, dtype=int)
    if not np.array_equal(low["site_index"], expected):
        raise RuntimeError("low-state chunks do not contain each frozen site exactly once")
    if not np.array_equal(high["site_index"], expected):
        raise RuntimeError("runaway chunks do not contain each frozen site exactly once")
    if not np.allclose(low["site_xy_mm"], high["site_xy_mm"], atol=0, rtol=0):
        raise RuntimeError("low and runaway states use different random sites")
    reference = low_meta[0]
    for meta in low_meta + high_meta:
        if meta["seed"] != reference["seed"]:
            raise RuntimeError("chunk seed mismatch")
        if meta["site_contract"] != reference["site_contract"]:
            raise RuntimeError("chunk site contract mismatch")
        if meta["dose_cells"] != reference["dose_cells"]:
            raise RuntimeError("chunk probe-dose mismatch")
    with np.load(Path(args.low_chunks[0]), allow_pickle=False) as handle:
        positions = np.asarray(handle["positions_E"], np.float32)
        contacts = np.asarray(handle["contact_xy_mm"], np.float32)
    for raw in list(args.low_chunks[1:]) + list(args.runaway_chunks):
        with np.load(raw, allow_pickle=False) as handle:
            if not np.array_equal(positions, handle["positions_E"]):
                raise RuntimeError("chunk neuron sheet mismatch")
            if not np.array_equal(contacts, handle["contact_xy_mm"]):
                raise RuntimeError("chunk contact geometry mismatch")

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        out,
        positions_E=positions,
        contact_xy_mm=contacts,
        site_index=expected.astype(np.int16),
        site_xy_mm=np.asarray(low["site_xy_mm"], np.float32),
        low_response_early=np.asarray(low["excess_per_neuron_early"], np.float32),
        runaway_response_early=np.asarray(high["excess_per_neuron_early"], np.float32),
        low_response_early_mean=np.mean(
            low["excess_per_neuron_early"], axis=0).astype(np.float32),
        runaway_response_early_mean=np.mean(
            high["excess_per_neuron_early"], axis=0).astype(np.float32),
        low_response_early_sem=(
            np.std(low["excess_per_neuron_early"], axis=0, ddof=1)
            / np.sqrt(len(expected))).astype(np.float32),
        runaway_response_early_sem=(
            np.std(high["excess_per_neuron_early"], axis=0, ddof=1)
            / np.sqrt(len(expected))).astype(np.float32),
        low_excess_spikes_early=np.asarray(low["excess_spikes_early"], np.float32),
        runaway_excess_spikes_early=np.asarray(
            high["excess_spikes_early"], np.float32),
        low_e1_evaluable=np.asarray(low["e1_evaluable"], bool),
        runaway_e1_evaluable=np.asarray(high["e1_evaluable"], bool),
    )
    snapshot_meta = json.loads(Path(args.mode_snapshot).with_suffix(".json").read_text())
    summary = {
        "status": "ZM_FIG5_GLOBAL_PERTURBATION_COMPLETE",
        "seed": int(reference["seed"]),
        "state_times_ms": {
            "low_activity": float(low_meta[0]["checkpoint_time_ms"]),
            "runaway": float(high_meta[0]["checkpoint_time_ms"]),
        },
        "mode_snapshot": str(Path(args.mode_snapshot).resolve()),
        "runaway_template": snapshot_meta["selected"]["winning_template"],
        "runaway_template_identity_r": float(
            snapshot_meta["selected"]["selection_score"]),
        "site_contract": reference["site_contract"],
        "dose_cells": int(reference["dose_cells"]),
        "response_window": "paired probe-minus-sham descendant spikes, 0-50 ms",
        "aggregation": "equal-weight mean over 16 paired stratified-random sites",
        "low_n_e1_evaluable": int(np.sum(low["e1_evaluable"])),
        "runaway_n_e1_evaluable": int(np.sum(high["e1_evaluable"])),
        "low_rows": low_rows,
        "runaway_rows": high_rows,
        "chunks": {
            "low": [str(Path(path).resolve()) for path in args.low_chunks],
            "runaway": [str(Path(path).resolve()) for path in args.runaway_chunks],
        },
        "npz": str(out),
        "claim_boundary": (
            "single network and one frozen probe dose; site-averaged state response, "
            "not a causal population estimate across network seeds"),
    }
    atomic_write_json(summary, str(out.with_suffix(".json")))
    print(json.dumps({
        "out": str(out),
        "low_n_e1_evaluable": summary["low_n_e1_evaluable"],
        "runaway_n_e1_evaluable": summary["runaway_n_e1_evaluable"],
    }, indent=2))


if __name__ == "__main__":
    main()
