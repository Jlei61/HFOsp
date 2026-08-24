#!/usr/bin/env python3
"""Materialize complete causal episodes from frozen SNN trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_rev10_sa_spectral_field_worker import _contact_onsets
from scripts.run_topic4_rev9l_forced_source_worker import _atomic_npz
from src.sef_hfo_events import detect_events
from src.topic4_core_field_runner import atomic_write_json
from src.topic4_node_dualmode import (
    annotate_population_excursions_with_lineages,
    directed_lineage_onset_maps,
    population_excursion_episodes,
)

ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value):
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _materialize(source_npz: Path, source_json: Path, config: dict) -> dict:
    payload = json.loads(source_json.read_text())
    if _sha256(source_npz) != payload["arrays"]["sha256"]:
        raise RuntimeError("source worker array hash changed")
    with np.load(source_npz, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    active = np.asarray(arrays["active_fraction"], float)
    active_dt = float(arrays["active_fraction_bin_ms"])
    envelope = np.asarray(arrays["contact_envelope"], float)
    envelope_dt = float(arrays["contact_envelope_dt_ms"])
    runtime = payload["event_unit"]
    threshold = float(runtime["event_on_threshold"])
    fast_tau = float(runtime.get("fast_tau_ms", runtime["fast_state_tau_ms"]))
    maximum_delay = float(runtime["maximum_delay_ms"])
    event_unit = config["event_unit"]
    multiple = float(event_unit["fast_state_decay_multiples"])
    reset_ms = multiple * fast_tau + maximum_delay
    fragments = detect_events(active, active_dt, event_on_frac=threshold)
    episodes = population_excursion_episodes(
        fragments, active, sample_dt_ms=active_dt,
        event_on_threshold=threshold,
        low_threshold_fraction=float(event_unit["low_threshold_fraction"]),
        reset_ms=reset_ms, pre_roll_ms=float(event_unit["pre_roll_ms"]),
    )
    movie = np.asarray(arrays["sheet_activity_counts"])
    frame_ms = float(arrays["sheet_activity_frame_ms"])
    labels = np.asarray(arrays["directed_lineage_labels"], int)
    episodes = annotate_population_excursions_with_lineages(
        episodes, movie, labels, frame_ms=frame_ms,
    )
    names = np.asarray(arrays["contact_names"]).astype(str)
    montage = SimpleNamespace(names=names)
    valid = np.ones(len(names), bool)
    readout = config["search"]["contact_readout"]
    onsets, ranks, rows = [], [], []
    for index, event in enumerate(episodes):
        onset, rank = _contact_onsets(
            envelope, envelope_dt, montage, valid,
            (float(event["t_on"]), float(event["t_off"])),
            float(readout["participation_margin_fraction"]),
            float(readout["timing_fraction"]),
        )
        onsets.append(onset)
        ranks.append(rank)
        rows.append({
            "event_index": int(index),
            "t_on_ms": float(event["t_on"]),
            "t_off_ms": float(event["t_off"]),
            "trigger_t_on_ms": float(event["trigger_t_on"]),
            "trigger_t_off_ms": float(event["trigger_t_off"]),
            "reset_start_ms": event["reset_start_ms"],
            "duration_ms": float(event["dur_ms"]),
            "returned": bool(event["returned"]),
            "n_recruited_contacts": int(np.sum(np.isfinite(onset))),
            "n_detector_fragments": int(len(event["detector_fragment_indices"])),
            "detector_fragment_indices": event["detector_fragment_indices"],
            "lineage_id": event["cascade_id"],
            "lineage_ids": event["lineage_ids"],
            "root_count": int(event["root_count"]),
            "root_attributable_activity_fraction": float(
                event["root_attributable_activity_fraction"]
            ),
            "collision_activity_fraction": float(
                event["collision_activity_fraction"]
            ),
        })
    onsets = np.asarray(onsets, float).reshape((-1, len(names)))
    ranks = np.asarray(ranks, float).reshape((-1, len(names)))
    source = directed_lineage_onset_maps(labels, episodes, frame_ms=frame_ms)
    return {
        "source_payload": payload, "source_arrays": arrays,
        "events": rows, "onsets": onsets, "ranks": ranks,
        "source": source, "fragments": fragments,
        "reset_ms": reset_ms, "fast_tau_ms": fast_tau,
        "maximum_delay_ms": maximum_delay, "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    artifact_root = args.artifact_root.resolve()
    expected = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip() != expected:
        raise RuntimeError("causal-episode materializer is not at expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/topic4_node_dualmode.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("causal-episode materializer paths are dirty")
    manifest_path = artifact_root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("selection_forbidden") is not True:
        raise RuntimeError("causal-episode materialization permits selection")
    source_root = artifact_root / config["source_output_root"] / "workers"
    output_root = artifact_root / config["output_root"] / "workers"
    output_root.mkdir(parents=True, exist_ok=True)
    excluded = {
        "onsets", "ranks", "event_t_on_ms", "event_t_off_ms",
        "event_trigger_t_on_ms", "event_returned", "event_fragment_count",
        "event_directed_root_id", "event_root_count", "source_onset_maps_ms",
        "source_onset_evaluable", "source_activity_counts",
        "source_activity_relative_ms",
    }
    completed = []
    for candidate in manifest["candidates"]:
        for seed in config["search"]["canary_network_seeds"]:
            stem = f"{candidate['candidate_id']}_seed_{int(seed)}"
            source_npz = source_root / f"{stem}.npz"
            source_json = source_root / f"{stem}.json"
            bundle = _materialize(source_npz, source_json, config)
            rows = bundle["events"]
            arrays = {
                key: value for key, value in bundle["source_arrays"].items()
                if key not in excluded and not key.startswith("lineage_sensitivity_")
            }
            out_npz = output_root / f"{stem}.npz"
            out_json = output_root / f"{stem}.json"
            _atomic_npz(
                out_npz, **arrays,
                onsets=bundle["onsets"].astype(np.float32),
                ranks=bundle["ranks"].astype(np.float32),
                event_t_on_ms=np.asarray([r["t_on_ms"] for r in rows], np.float32),
                event_t_off_ms=np.asarray([r["t_off_ms"] for r in rows], np.float32),
                event_trigger_t_on_ms=np.asarray([
                    r["trigger_t_on_ms"] for r in rows
                ], np.float32),
                event_returned=np.asarray([r["returned"] for r in rows], bool),
                event_fragment_count=np.asarray([
                    r["n_detector_fragments"] for r in rows
                ], np.int16),
                event_directed_root_id=np.asarray([
                    -1 if r["lineage_id"] is None else r["lineage_id"] for r in rows
                ], np.int32),
                event_root_count=np.asarray([r["root_count"] for r in rows], np.int32),
                source_onset_maps_ms=bundle["source"]["onset_maps_ms"],
                source_onset_evaluable=bundle["source"]["evaluable"],
                source_activity_counts=np.zeros((len(rows), 0, 20, 20), np.uint16),
                source_activity_relative_ms=np.asarray([], float),
            )
            source_payload = bundle["source_payload"]
            payload = {
                **source_payload,
                "status": "REV12ND_CAUSAL_EPISODE_MATERIALIZED",
                "events": rows,
                "event_unit": {
                    **config["event_unit"],
                    "event_on_threshold": bundle["threshold"],
                    "fast_tau_ms": bundle["fast_tau_ms"],
                    "maximum_delay_ms": bundle["maximum_delay_ms"],
                    "reset_ms": bundle["reset_ms"],
                    "raw_detector_fragment_count": len(bundle["fragments"]),
                    "episode_count": len(rows),
                    "all_detector_fragments_represented": (
                        sum(r["n_detector_fragments"] for r in rows)
                        == len(bundle["fragments"])
                    ),
                    "contact_geometry_used_for_boundary": False,
                },
                "contact_readout": config["search"]["contact_readout"],
                "source_worker": {
                    "json": str(source_json), "json_sha256": _sha256(source_json),
                    "npz": str(source_npz), "npz_sha256": _sha256(source_npz),
                    "simulation_rerun": False,
                },
                "arrays": {"path": str(out_npz), "sha256": _sha256(out_npz)},
                "provenance": {
                    "git_commit": expected, "expected_git_commit": expected,
                    "config_path": str(config_path.relative_to(ROOT)),
                    "config_sha256": _sha256(config_path),
                    "runtime_paths_dirty": False,
                },
            }
            atomic_write_json(_safe(payload), str(out_json))
            completed.append({
                "candidate_id": candidate["candidate_id"], "seed": int(seed),
                "n_events": len(rows),
                "n_returned": int(sum(r["returned"] for r in rows)),
            })
            print(json.dumps(completed[-1]), flush=True)
    print(json.dumps({
        "status": "REV12ND_CAUSAL_EPISODE_LIBRARY_MATERIALIZED",
        "n_workers": len(completed), "output": str(output_root),
    }, indent=2))


if __name__ == "__main__":
    main()
