#!/usr/bin/env python3
"""Re-segment frozen whole-sheet movies into directed activity lineages."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src" / "snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.run_topic4_rev10_sa_spectral_field_worker import _contact_onsets  # noqa: E402
from scripts.run_topic4_rev12_node_worker import _event_peak_active_fraction  # noqa: E402
from scripts.run_topic4_rev9l_forced_source_worker import _atomic_npz  # noqa: E402
from src.sef_hfo_events import detect_events  # noqa: E402
from src.sef_hfo_observation import VirtualMontage  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_node_dualmode import (  # noqa: E402
    assign_detector_fragments_to_directed_lineages,
    cascade_event_windows,
    directed_lineage_onset_maps,
    directed_spatiotemporal_lineages,
)

ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


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


def _directed_bundle(source_npz: Path, source_payload: dict,
                     config: dict) -> dict:
    event_unit = config["event_unit"]
    readout = config["search"]["contact_readout"]
    with np.load(source_npz, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    active = np.asarray(arrays["active_fraction"], float)
    active_dt = float(arrays["active_fraction_bin_ms"])
    movie = np.asarray(arrays["sheet_activity_counts"])
    frame_ms = float(arrays["sheet_activity_frame_ms"])
    if not np.isclose(frame_ms, float(event_unit["movie_frame_ms"])):
        raise RuntimeError("source movie frame duration differs from directed contract")
    detector_threshold = float(source_payload["event_unit"]["event_on_threshold"])
    fragments = detect_events(
        active, active_dt, event_on_frac=detector_threshold,
    )
    expected_fragments = int(source_payload["event_unit"]["raw_detector_fragment_count"])
    if len(fragments) != expected_fragments:
        raise RuntimeError("reconstructed detector fragments differ from source worker")

    lineage = directed_spatiotemporal_lineages(
        movie, minimum_active_neurons=int(event_unit["minimum_active_neurons"]),
    )
    assignments = assign_detector_fragments_to_directed_lineages(
        movie, lineage["labels"], fragments, frame_ms=frame_ms,
        minimum_dominance=float(event_unit["minimum_dominance"]),
    )
    events, compounds = cascade_event_windows(
        lineage["components"], assignments, fragments, frame_ms=frame_ms,
        total_ms=len(active) * active_dt,
    )
    represented = sum(len(event["detector_fragment_indices"]) for event in events)
    if represented + len(compounds) != len(fragments):
        raise RuntimeError("directed event partition does not conserve fragments")

    names = np.asarray(arrays["contact_names"]).astype(str)
    montage = VirtualMontage(
        np.asarray(arrays["contact_xy_mm"], float), names.tolist(),
        provenance="rev12_directed_lineage_zero_simulation_resegmentation",
    )
    valid = np.ones(len(names), bool)
    envelope = np.asarray(arrays["contact_envelope"], float)
    envelope_dt = float(arrays["contact_envelope_dt_ms"])
    onset_rows, rank_rows, event_rows = [], [], []
    assignment_by_fragment = {
        int(row["detector_fragment_index"]): row for row in assignments
    }
    for event_index, event in enumerate(events):
        onset, rank = _contact_onsets(
            envelope, envelope_dt, montage, valid,
            (float(event["t_on"]), float(event["t_off"])),
            float(readout["participation_margin_fraction"]),
            float(readout["timing_fraction"]),
        )
        onset_rows.append(onset)
        rank_rows.append(rank)
        fragment_rows = [
            assignment_by_fragment[index]
            for index in event["detector_fragment_indices"]
        ]
        event_rows.append({
            "event_index": int(event_index),
            "t_on_ms": float(event["t_on"]),
            "t_off_ms": float(event["t_off"]),
            "trigger_t_on_ms": float(event["trigger_t_on"]),
            "trigger_t_off_ms": float(event["trigger_t_off"]),
            "duration_ms": float(event["dur_ms"]),
            "peak_active_fraction": _event_peak_active_fraction(
                event, active, active_dt,
            ),
            "returned": bool(event["returned"]),
            "n_recruited_contacts": int(np.sum(np.isfinite(onset))),
            "n_detector_fragments": int(len(event["detector_fragment_indices"])),
            "detector_fragment_indices": event["detector_fragment_indices"],
            "lineage_id": int(event["cascade_id"]),
            "minimum_fragment_dominance": float(min(
                row["dominant_activity_fraction"] for row in fragment_rows
            )),
            "maximum_fragment_collision_fraction": float(max(
                row["collision_activity_fraction"] for row in fragment_rows
            )),
        })
    onsets = np.asarray(onset_rows, float).reshape((-1, len(names)))
    ranks = np.asarray(rank_rows, float).reshape((-1, len(names)))
    returned = np.asarray([row["returned"] for row in event_rows], bool)
    source = directed_lineage_onset_maps(
        lineage["labels"], events, frame_ms=frame_ms,
    )
    active_local = movie >= int(event_unit["minimum_active_neurons"])
    active_mass = float(np.sum(movie[active_local]))
    collision_mass = float(np.sum(movie[lineage["collision_mask"]]))
    return {
        "arrays": arrays,
        "fragments": fragments,
        "assignments": assignments,
        "events": event_rows,
        "compounds": compounds,
        "onsets": onsets,
        "ranks": ranks,
        "returned": returned,
        "lineage": lineage,
        "source": source,
        "collision_mass_fraction": (
            collision_mass / active_mass if active_mass > 0.0 else 0.0
        ),
        "detector_threshold": detector_threshold,
        "active_dt": active_dt,
        "frame_ms": frame_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    expected = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip() != expected:
        raise RuntimeError("directed-lineage producer is not at expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/topic4_node_dualmode.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("directed-lineage producer paths are dirty")
    for record in config["inputs"].values():
        path = _resolve(artifact_root, record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"directed-lineage input changed: {record['path']}")
    manifest_path = artifact_root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest["config_sha256"] != _sha256(config_path):
        raise RuntimeError("directed-lineage manifest is stale")

    source_root = artifact_root / config["source_output_root"] / "workers"
    output_root = artifact_root / config["output_root"] / "workers"
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in config["search"]["fit_network_seeds"]]
    completed = []
    for candidate in manifest["candidates"]:
        for seed in seeds:
            stem = f"{candidate['candidate_id']}_seed_{seed}"
            source_npz = source_root / f"{stem}.npz"
            source_json = source_root / f"{stem}.json"
            if not source_npz.exists() or not source_json.exists():
                raise RuntimeError(f"missing source worker: {stem}")
            source_payload = json.loads(source_json.read_text())
            if _sha256(source_npz) != source_payload["arrays"]["sha256"]:
                raise RuntimeError(f"source worker hash changed: {stem}")
            bundle = _directed_bundle(source_npz, source_payload, config)
            arrays = bundle["arrays"]
            events = bundle["events"]
            assignments = bundle["assignments"]
            labels = bundle["lineage"]["labels"]
            label_dtype = np.int16 if np.max(labels, initial=0) <= 32767 else np.int32
            out_npz = output_root / f"{stem}.npz"
            out_json = output_root / f"{stem}.json"
            _atomic_npz(
                out_npz,
                contact_names=arrays["contact_names"],
                shaft_ids=arrays["shaft_ids"],
                contact_xy_mm=arrays["contact_xy_mm"],
                onsets=bundle["onsets"].astype(np.float32),
                ranks=bundle["ranks"].astype(np.float32),
                event_t_on_ms=np.asarray([row["t_on_ms"] for row in events], np.float32),
                event_t_off_ms=np.asarray([row["t_off_ms"] for row in events], np.float32),
                event_trigger_t_on_ms=np.asarray([
                    row["trigger_t_on_ms"] for row in events
                ], np.float32),
                event_returned=bundle["returned"],
                event_fragment_count=np.asarray([
                    row["n_detector_fragments"] for row in events
                ], np.int16),
                active_fraction=arrays["active_fraction"],
                active_fraction_bin_ms=arrays["active_fraction_bin_ms"],
                contact_envelope=arrays["contact_envelope"],
                contact_envelope_dt_ms=arrays["contact_envelope_dt_ms"],
                sheet_activity_counts=arrays["sheet_activity_counts"],
                sheet_activity_frame_ms=arrays["sheet_activity_frame_ms"],
                directed_lineage_labels=labels.astype(label_dtype),
                directed_lineage_collision_mask=bundle["lineage"]["collision_mask"],
                detector_fragment_dominant_lineage_id=np.asarray([
                    -1 if row["dominant_lineage_id"] is None
                    else row["dominant_lineage_id"] for row in assignments
                ], np.int32),
                detector_fragment_dominance=np.asarray([
                    row["dominant_activity_fraction"] for row in assignments
                ], np.float32),
                detector_fragment_collision_fraction=np.asarray([
                    row["collision_activity_fraction"] for row in assignments
                ], np.float32),
                detector_fragment_compound=np.asarray([
                    row["compound"] for row in assignments
                ], bool),
                source_onset_maps_ms=bundle["source"]["onset_maps_ms"],
                source_onset_evaluable=bundle["source"]["evaluable"],
                source_bin_mm=np.asarray(config["source_topology"]["bin_mm"], float),
                source_sheet_mm=arrays["source_sheet_mm"],
                positions_E=arrays["positions_E"], h=arrays["h"],
                delta_vtheta=arrays["delta_vtheta"],
                edge_coefficients=arrays["edge_coefficients"],
            )
            payload = {
                "status": "REV12ND_DIRECTED_LINEAGE_RESEGMENT_COMPLETE",
                "scientific_role": config["scientific_role"],
                "candidate_id": candidate["candidate_id"],
                "field_sha256": candidate["node_field"]["field_sha256"],
                "seed": seed,
                "simulation": source_payload["simulation"],
                "events": events,
                "event_unit": {
                    **config["event_unit"],
                    "event_on_threshold": bundle["detector_threshold"],
                    "raw_detector_fragment_count": len(bundle["fragments"]),
                    "episode_count": len(events),
                    "n_directed_roots": len(bundle["lineage"]["components"]),
                    "n_compound_detector_fragments": len(bundle["compounds"]),
                    "compound_detector_fragment_fraction": (
                        len(bundle["compounds"]) / max(1, len(bundle["fragments"]))
                    ),
                    "collision_activity_mass_fraction": bundle[
                        "collision_mass_fraction"
                    ],
                    "causal_rule": bundle["lineage"]["causal_rule"],
                    "compound_fragments": bundle["compounds"],
                },
                "source_topology": {
                    "n_evaluable_returned_events": int(np.sum(
                        bundle["source"]["evaluable"] & bundle["returned"]
                    )),
                    "source_map": config["source_topology"]["source_map"],
                    "bin_mm": config["source_topology"]["bin_mm"],
                },
                "mechanism_freeze": source_payload["mechanism_freeze"],
                "source_worker": {
                    "json": str(source_json), "json_sha256": _sha256(source_json),
                    "npz": str(source_npz), "npz_sha256": _sha256(source_npz),
                    "simulation_rerun": False,
                },
                "arrays": {"path": str(out_npz), "sha256": _sha256(out_npz)},
                "provenance": {
                    "git_commit": expected,
                    "expected_git_commit": expected,
                    "source_worker_git_commit": source_payload[
                        "provenance"
                    ]["git_commit"],
                    "config_path": str(config_path.relative_to(ROOT)),
                    "config_sha256": _sha256(config_path),
                    "runtime_paths_dirty": False,
                },
            }
            atomic_write_json(_json_safe(payload), str(out_json))
            completed.append({
                "candidate_id": candidate["candidate_id"], "seed": seed,
                "n_events": len(events), "n_returned": int(np.sum(bundle["returned"])),
                "compound_fraction": payload["event_unit"][
                    "compound_detector_fragment_fraction"
                ],
            })
            print(json.dumps(completed[-1]), flush=True)
    print(json.dumps({
        "status": "REV12ND_DIRECTED_LINEAGE_LIBRARY_COMPLETE",
        "n_workers": len(completed), "output": str(output_root),
    }, indent=2))


if __name__ == "__main__":
    main()
