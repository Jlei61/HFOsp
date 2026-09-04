#!/usr/bin/env python3
"""Run paired sham/probe responses at frozen rev21 Fig.5 state checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src" / "snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.prepare_topic4_rev21_fig5_checkpoints import (  # noqa: E402
    build_selected_substrate,
    load_selected_contract,
)
from scripts.run_topic4_rev9l_forced_source_worker import _atomic_npz  # noqa: E402
from scripts.run_topic4_zm_perturbation_worker import _continue  # noqa: E402
from src.snn_engine import checkpoint as ckpt  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_zm_fig5 import stratified_random_sites  # noqa: E402
from src.topic4_zm_perturbation import (  # noqa: E402
    in_window_ignition,
    response_metrics,
    select_packet,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path("/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/"
                "data_driven_dual_core_zm_transition")
    perturb = Path("/data/hfosp_topic4_fig45_artifacts/fig5/"
                   "data_driven_dual_core_spatial_z/perturbation")
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "config/topic4_rev21_dual_core_zm_transition.json")
    parser.add_argument(
        "--candidate-manifest", type=Path,
        default=base / "timescale/candidate_manifest.json")
    parser.add_argument(
        "--source-json", type=Path,
        default=base / "timescale/workers/"
        "rev21_ts_tz3000_ta500_topology_2542_dynamics_2642.json")
    parser.add_argument("--artifact-root", type=Path,
                        default=Path("/home/honglab/leijiaxin/HFOsp"))
    parser.add_argument(
        "--checkpoint-manifest", type=Path,
        default=perturb / "checkpoints/checkpoint_manifest.json")
    parser.add_argument("--state-label", choices=("low_activity", "early_ictal"),
                        required=True)
    parser.add_argument("--site-indices", type=int, nargs="+",
                        default=list(range(16)))
    parser.add_argument("--n-side", type=int, default=4)
    parser.add_argument("--site-seed", type=int, default=20260820)
    parser.add_argument("--margin-mm", type=float, default=1.2)
    parser.add_argument("--dose-cells", type=int, default=16)
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    started = time.time()
    artifact_root = args.artifact_root.resolve()
    config, candidate, source, source_npz = load_selected_contract(
        args.config.resolve(), args.candidate_manifest.resolve(),
        args.source_json.resolve(), artifact_root)
    checkpoint_manifest = json.loads(args.checkpoint_manifest.read_text())
    if checkpoint_manifest.get("status") != "REV21_FIG5_EXACT_STATE_CHECKPOINTS_COMPLETE":
        raise RuntimeError("exact checkpoint manifest is incomplete")
    if not checkpoint_manifest.get("replay_prefix_exact"):
        raise RuntimeError("checkpoint trajectory did not replay the source exactly")
    topology_seed = int(source["topology_seed"])
    dynamics_seed = int(source["dynamics_seed"])
    transition, substrate = build_selected_substrate(
        config, candidate, topology_seed=topology_seed,
        dynamics_seed=dynamics_seed, artifact_root=artifact_root)

    checkpoint_record = checkpoint_manifest["checkpoints"][args.state_label]
    checkpoint_path = Path(checkpoint_record["path"])
    if sha256(checkpoint_path) != checkpoint_record["sha256"]:
        raise RuntimeError("state checkpoint hash changed")
    state = ckpt.load(checkpoint_path)
    if not np.isclose(float(state["absolute_time_ms"]),
                      float(checkpoint_record["time_ms"]), atol=1e-9):
        raise RuntimeError("checkpoint time differs from manifest")

    sites = stratified_random_sites(
        n_side=int(args.n_side), extent_mm=(0.0, float(substrate.engine["L"])),
        margin_mm=float(args.margin_mm), seed=int(args.site_seed),
    )
    indices = sorted(set(int(value) for value in args.site_indices))
    if not indices or indices[0] < 0 or indices[-1] >= len(sites):
        raise ValueError("site indices fall outside the frozen random-site set")
    dt_ms = float(substrate.engine["dt"])
    window_ms = float(args.window_ms)
    sham, _ = _continue(
        substrate, transition, state, duration_ms=window_ms)
    offset = int(round(float(state["absolute_time_ms"]) / dt_ms))
    steps = int(round(window_ms / dt_ms))
    with np.load(source_npz, allow_pickle=False) as archive:
        reference_rate = np.asarray(
            archive["transition_rate_E_hz_raw"][offset:offset + steps],
            np.float32,
        )
    sham_rate = np.asarray(sham["rate_E"], np.float32)
    sham_exact = bool(np.array_equal(sham_rate, reference_rate))
    if not sham_exact:
        mismatch = int(np.flatnonzero(sham_rate != reference_rate)[0])
        raise RuntimeError(f"resumed sham diverged from source at local step {mismatch}")

    cmrun = substrate.extras["cmrun"]
    sham_active, active_dt = cmrun.active_fraction(
        np.asarray(sham["E_spk_bool"], bool), dt_ms, cmrun.BIN_MS)
    response_split_ms = float(transition["perturbation"]["response_split_ms"])
    packet_radius_mm = float(transition["perturbation"]["packet_radius_mm"])
    rows = []
    early_fields = []
    full_fields = []
    for site_index in indices:
        xy = np.asarray(sites[site_index], float)
        packet = select_packet(
            substrate.positions_e, xy, n_cells=int(args.dose_cells),
            radius_mm=packet_radius_mm,
        )
        probe, _ = _continue(
            substrate, transition, state, duration_ms=window_ms,
            packet=packet)
        probe_active, _ = cmrun.active_fraction(
            np.asarray(probe["E_spk_bool"], bool), dt_ms, cmrun.BIN_MS)
        metrics = response_metrics(
            probe, sham, dt_ms=dt_ms, positions_e=substrate.positions_e,
            packet_mask=packet, packet_xy=xy,
            envelope_probe=np.zeros((15, 1)),
            envelope_sham=np.zeros((15, 1)), envelope_dt_ms=2.0,
            inject_step=0, split_ms=response_split_ms, window_ms=window_ms,
        )
        regime = in_window_ignition(
            probe_active, sham_active, active_dt_ms=float(active_dt),
            detector_threshold=substrate.detector_threshold, inject_ms=0.0,
            window_ms=window_ms,
            probe_rate_hz=np.asarray(probe["rate_E"], float), dt_ms=dt_ms,
            es_thresh_hz=float(transition["simulation"]["es_thresh_hz"]),
            es_dur_ms=float(transition["simulation"]["es_dur_ms"]),
        )
        rows.append({
            "site_index": int(site_index),
            "site_xy_mm": [float(xy[0]), float(xy[1])],
            "dose_cells": int(args.dose_cells),
            "susceptibility": float(metrics["susceptibility"]),
            "excess_spikes_early": float(metrics["excess_spikes_early"]),
            "excess_spikes_late": float(metrics["excess_spikes_late"]),
            "r90_mm": float(metrics["r90_mm"]),
            **regime,
        })
        early_fields.append(metrics["excess_per_neuron_early"])
        full_fields.append(metrics["excess_per_neuron"])

    out = args.out
    if out is None:
        out = args.checkpoint_manifest.parent.parent / (
            f"dualcore_rev21_{args.state_label}_random_sites.npz")
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        out,
        positions_E=np.asarray(substrate.positions_e, np.float32),
        contact_xy_mm=np.asarray(substrate.contact_xy, np.float32),
        site_index=np.asarray(indices, np.int16),
        site_xy_mm=np.asarray(sites[indices], np.float32),
        excess_per_neuron_early=np.asarray(early_fields, np.float32),
        excess_per_neuron_full=np.asarray(full_fields, np.float32),
        excess_spikes_early=np.asarray(
            [row["excess_spikes_early"] for row in rows], np.float32),
        e1_evaluable=np.asarray([row["e1_evaluable"] for row in rows], bool),
    )
    payload = {
        "status": "REV21_FIG5_RANDOM_PERTURBATION_COMPLETE",
        "state_label": args.state_label,
        "candidate_id": source["candidate_id"],
        "substrate": "dualcore_s39 + Joint=1.25",
        "topology_seed": topology_seed,
        "dynamics_seed": dynamics_seed,
        "checkpoint": checkpoint_record,
        "checkpoint_manifest": {
            "path": str(args.checkpoint_manifest.resolve()),
            "sha256": sha256(args.checkpoint_manifest.resolve()),
        },
        "site_contract": {
            "kind": "one uniform random point per square stratum",
            "n_side": int(args.n_side), "n_total": int(len(sites)),
            "seed": int(args.site_seed),
            "sheet_extent_mm": [0.0, float(substrate.engine["L"])],
            "edge_margin_mm": float(args.margin_mm),
        },
        "site_indices": indices,
        "dose_contract": (
            "16-cell weak packet inherited from the prior Fig.5D protocol; "
            "not selected using the early-ictal response"
        ),
        "dose_cells": int(args.dose_cells),
        "window_ms": window_ms,
        "response_window": "paired probe-minus-sham descendant spikes, 0-50 ms",
        "resumed_sham_exact": sham_exact,
        "rows": rows,
        "npz": {"path": str(out), "sha256": sha256(out)},
        "wall_seconds": float(time.time() - started),
    }
    atomic_write_json(payload, str(out.with_suffix(".json")))
    print(json.dumps({
        "status": payload["status"], "state": args.state_label,
        "n_sites": len(indices), "resumed_sham_exact": sham_exact,
        "out": str(out), "wall_seconds": payload["wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
