#!/usr/bin/env python3
"""Run one frozen Fig. 5D perturbation chunk from an exact checkpoint."""
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


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-label", required=True,
                        choices=("low_activity", "early_runaway"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--site-indices", required=True, type=int, nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    ckpt = build.patch_spatial_checkpoint_support()
    from src.topic4_zm_fig5 import stratified_random_sites
    from src.topic4_zm_ictal_transition import load_round_config

    source = json.loads(build.SOURCE_JSON.read_text())
    round_config = load_round_config(
        str(ROOT / "config/topic4_data_driven_zm_ictal_transition_v1.json"))
    state = ckpt.load(args.checkpoint)
    expected_time = (build.LOW_TIME_MS if args.state_label == "low_activity"
                     else build.HIGH_TIME_MS)
    if not np.isclose(state["absolute_time_ms"], expected_time, atol=1e-12):
        raise RuntimeError("checkpoint time does not match the requested state")
    sites = stratified_random_sites(
        n_side=build.N_SITE_SIDE, extent_mm=(0.0, 20.0),
        margin_mm=build.SITE_MARGIN_MM, seed=build.SITE_SEED)
    indices = sorted(set(int(value) for value in args.site_indices))
    if not indices or indices[0] < 0 or indices[-1] >= len(sites):
        raise ValueError("site indices fall outside the frozen site set")
    substrate, sham, rows, fields = build.probe_state(
        round_config, source, state, sites, site_indices=indices)

    dt_ms = float(substrate.engine["dt"])
    with np.load(build.SOURCE_NPZ, allow_pickle=False) as archived:
        start = int(round(expected_time / dt_ms))
        expected = np.asarray(archived["rate_E_hz"], np.float32)[
            start:start + len(sham["rate_E"])]
    observed = np.asarray(sham["rate_E"], np.float32)
    continuation_exact = bool(np.array_equal(observed, expected))
    if not continuation_exact:
        raise RuntimeError("chunk sham continuation diverged from the locked replay")

    args.out = args.out.resolve()
    atomic_npz(
        args.out,
        positions_E=np.asarray(substrate.positions_e, np.float32),
        contact_xy_mm=np.asarray(substrate.contact_xy, np.float32),
        site_index=np.asarray(indices, np.int16),
        site_xy_mm=np.asarray(sites[indices], np.float32),
        response_early=np.asarray(fields, np.float32),
        susceptibility=np.asarray([row["susceptibility"] for row in rows],
                                  np.float32),
        excess_spikes_early=np.asarray(
            [row["excess_spikes_early"] for row in rows], np.float32),
        e1_evaluable=np.asarray([row["e1_evaluable"] for row in rows], bool),
    )
    metadata = {
        "status": "FIG5_TONIC_PROBE_CHUNK_COMPLETE",
        "state_label": args.state_label,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "checkpoint_time_ms": expected_time,
        "site_indices": indices,
        "site_contract": {
            "kind": "one uniform random point per square stratum",
            "n_side": build.N_SITE_SIDE, "n_total": len(sites),
            "seed": build.SITE_SEED, "sheet_extent_mm": [0.0, 20.0],
            "edge_margin_mm": build.SITE_MARGIN_MM,
        },
        "dose_cells": build.PROBE_DOSE_CELLS,
        "response": "paired probe-minus-sham descendant spikes, 0-50 ms",
        "sham_continuation_rate_bit_identical": continuation_exact,
        "rows": rows,
        "npz": str(args.out),
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({
        "status": metadata["status"],
        "state": args.state_label,
        "indices": indices,
        "evaluable": int(np.sum([row["e1_evaluable"] for row in rows])),
        "npz": str(args.out),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
