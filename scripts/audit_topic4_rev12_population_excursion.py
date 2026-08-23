#!/usr/bin/env python3
"""Zero-simulation audit of population-excursion event units for rev12 Node."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_topic4_rev12_event_fragmentation import _natural_summary  # noqa: E402
from scripts.rescore_topic4_rev12_node_historical import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    _classifier_contract,
    _formal_clean_mask,
    _old_to_patient_label_map,
    _patient_data,
    _reorder_patient_contract,
    score_candidate,
)
from scripts.run_topic4_rev10_sa_spectral_field_worker import _contact_onsets  # noqa: E402
from src.topic4_node_dualmode import (  # noqa: E402
    fixed_projection_matrix,
    population_excursion_episodes,
)
from src.topic4_shaft_aware_direction import assign_direction_modes  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(
            json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _seed_delay_map(records: list[str]) -> dict[int, float]:
    output = {}
    for record in records:
        try:
            seed_text, delay_text = record.split("=", maxsplit=1)
            seed, delay = int(seed_text), float(delay_text)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "network delays must use SEED=DELAY_MS"
            ) from error
        if delay < 0.0 or not np.isfinite(delay) or seed in output:
            raise ValueError("network delays must be finite, nonnegative and unique")
        output[seed] = delay
    return output


def _population_worker(npz_path: Path, target_names: np.ndarray,
                       classifier: dict, label_map: np.ndarray, *,
                       reset_ms: float, low_fraction: float,
                       event_threshold: float, pre_roll_ms: float,
                       readout: dict) -> tuple[dict, dict]:
    payload = json.loads(npz_path.with_suffix(".json").read_text())
    with np.load(npz_path, allow_pickle=False) as loaded:
        source_names = np.asarray(loaded["contact_names"]).astype(str)
        envelope = np.asarray(loaded["contact_envelope"], float)
        envelope_dt = float(loaded["contact_envelope_dt_ms"])
        active = np.asarray(loaded["active_fraction"], float)
        active_dt = float(loaded["active_fraction_bin_ms"])
        t_on = np.asarray(loaded["event_t_on_ms"], float)
        t_off = np.asarray(loaded["event_t_off_ms"], float)
        returned = np.asarray(loaded["event_returned"], bool)
    fragments = [
        {
            "t_on": float(on), "t_off": float(off),
            "dur_ms": float(off - on + active_dt),
            "peak_ext": float(np.max(active[
                max(0, int(np.floor(on / active_dt))):
                min(len(active), int(np.ceil(off / active_dt)) + 1)
            ])),
            "returned": bool(is_returned),
        }
        for on, off, is_returned in zip(t_on, t_off, returned)
    ]
    episodes = population_excursion_episodes(
        fragments, active, sample_dt_ms=active_dt,
        event_on_threshold=event_threshold,
        low_threshold_fraction=low_fraction, reset_ms=reset_ms,
        pre_roll_ms=pre_roll_ms,
    )
    order = np.asarray([
        int(np.flatnonzero(source_names == name)[0]) for name in target_names
    ])
    montage = SimpleNamespace(names=source_names)
    onsets, ranks, selected = [], [], []
    for episode in episodes:
        if not episode["returned"]:
            continue
        onset, rank = _contact_onsets(
            envelope, envelope_dt, montage, np.ones(len(source_names), bool),
            (float(episode["t_on"]), float(episode["t_off"])),
            float(readout["participation_margin_fraction"]),
            float(readout["timing_fraction"]),
        )
        onsets.append(np.asarray(onset, float)[order])
        ranks.append(np.asarray(rank, float)[order])
        selected.append(episode)
    onsets = np.asarray(onsets, float).reshape((-1, len(target_names)))
    ranks = np.asarray(ranks, float).reshape((-1, len(target_names)))
    assigned = assign_direction_modes(
        onsets, groups=classifier["groups"], embedding=classifier["embedding"],
        classifier=classifier["classifier"],
    )
    labels = np.asarray(label_map, int)[np.asarray(assigned["labels"], int)]
    ood = np.asarray(assigned["ood"], bool)
    formal_clean = _formal_clean_mask(onsets, ood, classifier["groups"])
    worker = {
        "seed": int(payload["seed"]), "ranks": ranks, "onsets": onsets,
        "labels": labels, "ood": ood, "formal_clean": formal_clean,
        "n_detected": int(len(episodes)), "n_returned": int(len(selected)),
        "duration_ms": float(payload["simulation"]["duration_ms"]),
        "npz": str(npz_path), "npz_sha256": _sha256(npz_path),
    }
    counts = {
        "seed": int(payload["seed"]),
        "n_input_settled_episodes": int(len(fragments)),
        "n_population_excursions": int(len(episodes)),
        "n_returned_population_excursions": int(len(selected)),
        "n_multifragment_excursions": int(sum(
            row["fragment_count"] > 1 for row in episodes
        )),
        "maximum_fragments_per_excursion": int(max(
            (row["fragment_count"] for row in episodes), default=0,
        )),
        "median_returned_duration_ms": (
            float(np.median([row["dur_ms"] for row in selected]))
            if selected else None
        ),
    }
    return worker, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--decay-multiples", nargs="+", type=float, default=(4, 5, 6))
    parser.add_argument("--fast-tau-ms", type=float, default=20.0)
    parser.add_argument(
        "--network-delay-ms", nargs="+", required=True,
        help="actual frozen maximum delays as SEED=DELAY_MS",
    )
    parser.add_argument("--low-threshold-fraction", type=float, default=0.2)
    parser.add_argument("--pre-roll-ms", type=float, default=20.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    network_delays = _seed_delay_map(args.network_delay_ms)
    artifact_root = args.artifact_root.resolve()
    config = json.loads(args.config.resolve().read_text())
    summary = json.loads(args.summary.resolve().read_text())
    cohort = json.loads((ROOT / config["inputs"]["cohort_config"]["path"]).read_text())
    classifier_config = json.loads(
        (ROOT / config["inputs"]["classifier_config"]["path"]).read_text()
    )
    patient = _patient_data(cohort, artifact_root)
    classifier = _classifier_contract(classifier_config, artifact_root)
    semantics = _old_to_patient_label_map(patient, classifier)
    patient = _reorder_patient_contract(patient, classifier["names"])
    projections = fixed_projection_matrix(
        2 * len(patient["contact_names"]), n_directions=64, seed=20260821,
    )
    manifest = json.loads((artifact_root / config["candidate_manifest"]).read_text())
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    event_threshold = 0.0195703125
    rows = []
    output_root = artifact_root / config["output_root"]
    for multiple in args.decay_multiples:
        for candidate_row in summary["rows"]:
            workers, counts = [], []
            candidate_id = candidate_row["candidate_id"]
            for seed in summary["requested_seeds"]:
                seed = int(seed)
                if seed not in network_delays:
                    raise RuntimeError(f"missing frozen network delay for seed {seed}")
                reset_ms = (
                    float(multiple) * float(args.fast_tau_ms)
                    + network_delays[seed]
                )
                npz_path = output_root / "workers" / f"{candidate_id}_seed_{seed}.npz"
                worker, count = _population_worker(
                    npz_path, patient["contact_names"], classifier,
                    semantics["raw_to_patient"], reset_ms=reset_ms,
                    low_fraction=float(args.low_threshold_fraction),
                    event_threshold=event_threshold,
                    pre_roll_ms=float(args.pre_roll_ms),
                    readout=config["search"]["contact_readout"],
                )
                workers.append(worker)
                counts.append(count)
            score = score_candidate(
                candidates[candidate_id], workers, patient, projections,
                summary["component_calibration"],
            )
            rows.append({
                "candidate_id": candidate_id,
                "decay_multiple": float(multiple),
                "reset_ms_by_seed": {
                    str(seed): (
                        float(multiple) * float(args.fast_tau_ms)
                        + network_delays[int(seed)]
                    )
                    for seed in summary["requested_seeds"]
                },
                "score": score,
                "pooled_natural_kmeans": _natural_summary(workers, 20260823),
                "per_seed": counts,
            })
    payload = {
        "schema_id": "topic4_rev12_population_excursion_audit_v1",
        "status": "REV12ND_POPULATION_EXCURSION_AUDIT_COMPLETE",
        "config": {"path": str(args.config.resolve()), "sha256": _sha256(args.config.resolve())},
        "summary": {"path": str(args.summary.resolve()), "sha256": _sha256(args.summary.resolve())},
        "event_contract": {
            "boundary_signal": "population active fraction only",
            "event_on_threshold": event_threshold,
            "low_threshold_fraction": float(args.low_threshold_fraction),
            "fast_tau_ms": float(args.fast_tau_ms),
            "maximum_network_delay_ms_by_seed": {
                str(seed): delay for seed, delay in sorted(network_delays.items())
            },
            "decay_multiples": list(map(float, args.decay_multiples)),
            "pre_roll_ms": float(args.pre_roll_ms),
            "contact_geometry_used_for_boundary": False,
        },
        "rows": rows,
        "claim_boundary": (
            "Zero-simulation audit only. Previous Stage-B/C rankings are invalid until "
            "candidate ranking and natural K=2 are stable across the fast-state reset range."
        ),
    }
    output = args.out or output_root / "aggregate/population_excursion_audit.json"
    _atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
