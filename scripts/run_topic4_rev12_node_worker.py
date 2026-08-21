#!/usr/bin/env python3
"""Run one rev12-ND continuous Node field on one frozen SNN network."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src" / "snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.run_topic4_rev10_sa_spectral_field_worker import _contact_onsets  # noqa: E402
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _atomic_npz, _runtime_provenance,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_node_dualmode import event_source_onset_maps  # noqa: E402
from src.topic4_zm_ictal_transition import (  # noqa: E402
    build_substrate, load_round_config, make_external_drive,
)
from kick_probe import simulate_kick  # noqa: E402
from src.sef_hfo_events import detect_events  # noqa: E402
from src.sef_hfo_snn_adapter import snn_event_envelope  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else root / relative


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _config_at_commit(config_path: Path, commit: str) -> str | None:
    try:
        content = subprocess.check_output(
            ["git", "show", f"{commit}:{config_path.relative_to(ROOT)}"],
            cwd=ROOT, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, ValueError):
        return None
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path,
                        default=Path("/home/honglab/leijiaxin/HFOsp"))
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-npz", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config["scientific_role"] != "development_only_node_dualmode_refit":
        raise RuntimeError("rev12-ND scientific role changed")
    active_seeds = {
        int(seed) for key in (
            "canary_network_seeds", "fit_network_seeds",
            "selection_network_seeds", "confirmation_network_seeds",
        ) for seed in config["search"].get(key, [])
    }
    if args.seed not in active_seeds:
        parser.error("seed is outside every frozen rev12-ND pool")

    artifact_root = args.artifact_root.resolve()
    for record in config["inputs"].values():
        path = _resolve(artifact_root, record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    manifest_path = artifact_root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("rev12-ND candidate manifest is stale")
    matches = [
        row for row in manifest["candidates"]
        if row["candidate_id"] == args.candidate_id
    ]
    if len(matches) != 1:
        parser.error("candidate is outside the frozen rev12-ND manifest")
    candidate = matches[0]

    expected_commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    provenance = _runtime_provenance(expected_commit)
    provenance.update({
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "config_sha256_at_expected_commit": _config_at_commit(
            config_path, expected_commit,
        ),
        "systemd_unit": os.environ.get("REV12ND_SYSTEMD_UNIT"),
    })
    if (provenance["runtime_modules_dirty"]
            or not provenance["runtime_modules_match_expected_commit"]
            or provenance["config_sha256"] != provenance["config_sha256_at_expected_commit"]):
        raise RuntimeError("rev12-ND runtime modules or config are not frozen")

    output_root = artifact_root / config["output_root"]
    stem = f"{args.candidate_id}_seed_{args.seed}"
    out_json = args.out_json or output_root / "workers" / f"{stem}.json"
    out_npz = args.out_npz or output_root / "workers" / f"{stem}.npz"
    cache_dir = artifact_root / config["network_cache"]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    transition_path = _resolve(
        artifact_root, config["inputs"]["transition_config"]["path"],
    )
    transition = load_round_config(transition_path)
    substrate = build_substrate(
        transition, "node_baseline", args.seed, cache_dir=str(cache_dir),
        ee_dose=0.0, etoi_dose=0.0,
        node_candidate_override=candidate["node_field"],
        artifact_root=artifact_root,
    )
    simulation = config["search"]["simulation"]
    substrate.params.T = float(simulation["duration_ms"])
    substrate.net["rng"] = np.random.default_rng(args.seed)

    result = simulate_kick(
        substrate.params, substrate.net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=substrate.vtheta, slow=None,
        early_stop_runaway=bool(simulation["early_stop_runaway"]),
        external_e_rate_drive=make_external_drive(
            substrate, transition["spatial_ou"], args.seed,
        ),
    )
    spikes = np.asarray(result["E_spk_bool"], bool)
    cmrun = substrate.extras["cmrun"]
    active, active_dt = cmrun.active_fraction(
        spikes, float(substrate.engine["dt"]), cmrun.BIN_MS,
    )
    detected = detect_events(
        active, active_dt, event_on_frac=substrate.detector_threshold,
    )
    envelope, envelope_dt, _ = snn_event_envelope(
        spikes, substrate.positions_e, substrate.montage,
        float(substrate.engine["dt"]),
    )
    readout = config["search"]["contact_readout"]
    onset_rows, rank_rows, event_rows = [], [], []
    for index, event in enumerate(detected):
        onset, rank = _contact_onsets(
            envelope, envelope_dt, substrate.montage, substrate.valid_contacts,
            (event["t_on"], event["t_off"]),
            readout["participation_margin_fraction"],
            readout["timing_fraction"],
        )
        onset_rows.append(onset)
        rank_rows.append(rank)
        event_rows.append({
            "event_index": int(index), "t_on_ms": float(event["t_on"]),
            "t_off_ms": float(event["t_off"]),
            "duration_ms": float(event["dur_ms"]),
            "peak_active_fraction": float(event["peak_ext"]),
            "returned": bool(event["returned"]),
            "n_recruited_contacts": int(np.isfinite(onset).sum()),
        })
    onsets = np.asarray(onset_rows, float).reshape((-1, len(substrate.contact_names)))
    ranks = np.asarray(rank_rows, float).reshape((-1, len(substrate.contact_names)))
    returned = np.asarray([row["returned"] for row in event_rows], bool)
    event_t_on = np.asarray([row["t_on_ms"] for row in event_rows], float)
    source = event_source_onset_maps(
        spikes, substrate.positions_e, event_t_on, returned,
        dt_ms=float(substrate.engine["dt"]),
        sheet_mm=float(substrate.engine["L"]),
        bin_mm=float(config["source_topology"]["bin_mm"]),
    )
    del spikes

    _atomic_npz(
        out_npz,
        contact_names=np.asarray(substrate.contact_names, dtype="U16"),
        shaft_ids=np.asarray(substrate.shaft_ids, dtype="U8"),
        contact_xy_mm=np.asarray(substrate.contact_xy, np.float64),
        onsets=onsets.astype(np.float32), ranks=ranks.astype(np.float32),
        event_t_on_ms=event_t_on.astype(np.float32),
        event_t_off_ms=np.asarray([row["t_off_ms"] for row in event_rows], np.float32),
        event_returned=returned,
        active_fraction=np.asarray(active, np.float32),
        active_fraction_bin_ms=np.asarray(active_dt, float),
        contact_envelope=np.asarray(envelope, np.float32),
        contact_envelope_dt_ms=np.asarray(envelope_dt, float),
        source_onset_maps_ms=source["onset_maps_ms"],
        source_onset_evaluable=source["evaluable"],
        source_bin_mm=np.asarray(source["bin_mm"], float),
        source_sheet_mm=np.asarray(source["sheet_mm"], float),
        positions_E=np.asarray(substrate.positions_e, np.float32),
        h=np.asarray(substrate.h_e, np.float32),
        delta_vtheta=np.asarray(substrate.delta_vtheta, np.float32),
        edge_coefficients=np.asarray(substrate.edge_coefficients, np.float64),
    )
    payload = {
        "status": "REV12ND_NODE_WORKER_COMPLETE",
        "scientific_role": config["scientific_role"],
        "candidate_id": args.candidate_id,
        "field_sha256": candidate["node_field"]["field_sha256"],
        "seed": int(args.seed),
        "simulation": {
            "duration_ms": float(simulation["duration_ms"]),
            "runaway_early_stop_ms": result.get("runaway_early_stop_ms"),
            "wall_seconds": float(time.time() - started),
        },
        "events": event_rows,
        "source_topology": {
            "n_evaluable_returned_events": int(np.sum(source["evaluable"] & returned)),
            "bin_mm": source["bin_mm"],
            "window_ms": [-20.0, 80.0],
            "baseline_window_ms": [-120.0, -20.0],
        },
        "mechanism_freeze": {
            "EE": "off", "E_to_I": "off", "Z_M": "off",
            "edge_coefficients_all_zero": bool(np.allclose(substrate.edge_coefficients, 0.0)),
        },
        "arrays": {"path": str(out_npz), "sha256": _sha256(out_npz)},
        "provenance": provenance,
    }
    atomic_write_json(_json_safe(payload), str(out_json))
    print(json.dumps({
        "status": payload["status"], "candidate": args.candidate_id,
        "seed": args.seed, "n_returned": int(np.sum(returned)),
        "n_source_maps": payload["source_topology"]["n_evaluable_returned_events"],
        "output": str(out_json),
    }, indent=2))


if __name__ == "__main__":
    main()
