#!/usr/bin/env python3
"""Run the first bounded native-activity regularization wave, then stop.

The controller waits for the independent contact-timing pilot to finish, freezes
three per-placement parents, a lambda scale, and six antithetic proposals before
dispatch.  Wave 1 uses a common physical candidate set for lambda=0 and all
regularized rankings, and never starts a second adaptive batch automatically.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_topic4_contact_timing_shape_pilot as timing  # noqa: E402
from scripts import run_topic4_multievent_distribution_v2_1 as engine  # noqa: E402
from scripts import run_topic4_xy_research as base  # noqa: E402
from scripts.audit_topic4_multievent_execution_parameters_v2_1 import audit as audit_execution  # noqa: E402
from src.topic4_native_activity_regularizer import (  # noqa: E402
    summarize_network_windows,
    unsupported_activity_for_window,
)

TIMING = ROOT / "results/topic4_sef_hfo/contact_timing_shape_pilot"
CANARY = ROOT / "results/topic4_sef_hfo/multievent_distribution_search_v2_1/native_activity_regularizer_canary"
OUT = ROOT / "results/topic4_sef_hfo/native_activity_regularized_search_v1"
PAIRS = [(6101, 7101), (6102, 7101)]
PRIMARY_REGULARIZER = dict(
    minimum_active_neurons=2, minimum_parent_support=0.001,
    reset_frames=18, background_quantile=0.95,
    response_tail_frames=13, response_tail_fraction=0.5,
    delay_rounding="nearest", baseline_frames=60,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _status(name: str, **payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base.write(OUT / "status.json", {
        "status": name, "updated_unix": time.time(), **payload,
    })


def _wait_for_timing() -> None:
    while True:
        status = base.read(TIMING / "status.json")
        name = status["status"]
        # Replay and held-back review never feed training selection.  Once B
        # scores and the TRAIN-only nomination are frozen, the three per-anchor
        # parents cannot change and wave 1 may run concurrently with replay.
        if ((TIMING / "B_scores.json").exists()
                and (TIMING / "nomination.json").exists()
                and name in {"RUNNING_replay", "PHYSICAL_PILOT_COMPLETE_ANALYZING",
                             "PILOT_COMPLETE_PENDING_SCIENTIFIC_REVIEW"}):
            return
        if name in {"ERROR_REVIEW_REQUIRED", "BLOCKED"}:
            raise RuntimeError(f"timing pilot cannot supply parents: {status}")
        _status("WAITING_FOR_TIMING_PILOT", timing_status=name)
        time.sleep(30)


def _support(topology_seed: int) -> np.ndarray:
    path = CANARY / f"support_topology_{topology_seed}_nearest.npz"
    with np.load(path) as arrays:
        return np.asarray(arrays["support_by_lag"])


def _regularizer_record(worker_path: str | Path, support: np.ndarray) -> dict:
    worker_path = Path(worker_path)
    worker = base.read(worker_path)
    observation_path, observation_npz, observation = engine.repaired_observation(worker_path)
    with np.load(worker_path.with_suffix(".npz")) as arrays:
        movie = np.asarray(arrays["sheet_activity_counts"], float)
        frame_ms = float(arrays["sheet_activity_frame_ms"])
    with np.load(observation_npz) as arrays:
        primary = np.asarray(arrays["primary_event_indices"], int)
    rows = []
    for event_id in primary:
        event = observation["events"][int(event_id)]
        start, stop = np.rint(np.asarray(event["window_ms"]) / frame_ms).astype(int)
        baseline_start = max(0, int(start) - PRIMARY_REGULARIZER["baseline_frames"])
        rows.append(unsupported_activity_for_window(
            movie, support, start_frame=int(start), stop_frame=int(stop),
            baseline_start_frame=baseline_start, baseline_stop_frame=int(start),
            minimum_active_neurons=PRIMARY_REGULARIZER["minimum_active_neurons"],
            minimum_parent_support=PRIMARY_REGULARIZER["minimum_parent_support"],
            reset_frames=PRIMARY_REGULARIZER["reset_frames"],
            background_quantile=PRIMARY_REGULARIZER["background_quantile"],
            response_tail_frames=PRIMARY_REGULARIZER["response_tail_frames"],
            response_tail_fraction=PRIMARY_REGULARIZER["response_tail_fraction"],
        ))
    summary = summarize_network_windows(rows)
    summary.update({
        "worker_path": str(worker_path),
        "topology_seed": int(worker["topology_seed"]),
        "dynamics_seed": int(worker["dynamics_seed"]),
        "n_detected_windows": int(len(observation["events"])),
        "primary_window_fraction": float(len(primary) / max(1, len(observation["events"]))),
        "physical_status": worker.get("physical_status"),
    })
    return summary


def _history_and_parents(design: dict, temporal, old) -> tuple[list[dict], list[dict]]:
    reports = [base.read(TIMING / f"{name}_scores.json") for name in ("baseline_train", "A", "B")]
    history = [row for report in reports for row in report["candidates"]]
    support = {seed: _support(seed) for seed, _ in PAIRS}
    scored = []
    for row in history:
        units = []
        for unit, record in sorted(row["units"].items()):
            reg = _regularizer_record(record["worker_path"], support[int(unit.split("_")[1])])
            units.append({"unit": unit, "patient": record["score"], "regularizer": reg})
        scored.append({
            "candidate_id": row["candidate_id"], "candidate": row["candidate"],
            "patient_loss": row["loss"], "ranking_eligible": row["ranking_eligible"],
            "mean_R": float(np.mean([unit["regularizer"]["mean_unsupported_fraction"] for unit in units])),
            "units": units,
        })
    parents = []
    for anchor_index, anchor in enumerate(design["anchors"], 1):
        prefix = f"tshape_anchor{anchor_index}_"
        pool = [row for row in scored if row["candidate_id"] == anchor["candidate_id"]
                or row["candidate_id"].startswith(prefix)]
        parents.append(min(
            [row for row in pool if row["ranking_eligible"]],
            key=lambda row: (row["patient_loss"], row["candidate_id"]),
        ))
    return scored, parents


def _freeze_lambda(history: list[dict]) -> dict:
    eligible = [row for row in history if row["ranking_eligible"]]
    losses = np.asarray([row["patient_loss"] for row in eligible], float)
    regularizer = np.asarray([row["mean_R"] for row in eligible], float)
    loss_iqr = float(np.ptp(np.quantile(losses, [0.25, 0.75])))
    regularizer_iqr = float(np.ptp(np.quantile(regularizer, [0.25, 0.75])))
    if loss_iqr <= 0.0 or regularizer_iqr <= 0.01:
        raise RuntimeError("history cannot provide a stable lambda scale")
    primary = float(0.25 * loss_iqr / regularizer_iqr)
    return {
        "rule": "one regularizer IQR equals 0.25 historical patient-loss IQR",
        "history_candidate_count": len(eligible),
        "patient_loss_iqr": loss_iqr, "regularizer_iqr": regularizer_iqr,
        "primary_lambda": primary,
        "lambda_grid": [0.0, 0.5 * primary, primary, 2.0 * primary],
        "lambda_is_mechanism_prior_not_patient_estimate": True,
    }


def _proposals(design: dict, parents: list[dict]) -> tuple[list[dict], list[dict]]:
    proposals, draws = [], []
    step = np.asarray(design["local_half_width"], float)
    low_global = np.asarray(design["global_lower_bounds"], float)
    high_global = np.asarray(design["global_upper_bounds"], float)
    for anchor_index, (anchor, parent_row) in enumerate(zip(design["anchors"], parents), 1):
        parent = parent_row["candidate"]
        origin = timing.vector(anchor)
        center = timing.vector(parent)
        low = np.maximum(low_global, origin - step)
        high = np.minimum(high_global, origin + step)
        rng = np.random.default_rng(821100 + anchor_index)
        direction = rng.choice([-1.0, 1.0], len(step)) * rng.uniform(0.35, 1.0, len(step))
        for sign, tag in ((1.0, "plus"), (-1.0, "minus")):
            raw = center + sign * 0.45 * step * direction
            width = high - low
            values = low + width - np.abs((raw - low) % (2 * width) - width)
            candidate_id = f"native_reg_anchor{anchor_index}_A_{tag}"
            candidate = timing.with_vector(
                parent, values, candidate_id, parent["candidate_id"], "native_reg_A",
            )
            proposals.append(candidate)
            draws.append({
                "candidate_id": candidate_id, "anchor_id": anchor["candidate_id"],
                "parent_id": parent["candidate_id"], "direction": direction.tolist(),
                "sign": sign, "scale": 0.45, "raw_values": raw.tolist(),
                "applied_values": values.tolist(), "bounds": [low.tolist(), high.tolist()],
            })
    return proposals, draws


def _execution(proposals: list[dict]) -> tuple[Path, Path, Path, Path]:
    folder = OUT / "execution/A"
    folder.mkdir(parents=True, exist_ok=True)
    config_path = folder / "execution_config.json"
    manifest_path = folder / "candidate_manifest.json"
    snapshot_path = folder / "runtime_snapshot.json"
    if not config_path.exists():
        source = TIMING / "execution/B"
        config = copy.deepcopy(base.read(source / "execution_config.json"))
        config.update({
            "output_root": str(folder), "candidate_manifest": str(manifest_path),
        })
        base.write(config_path, config)
        base.write(manifest_path, {
            "config_sha256": base.sha(config_path), "candidates": proposals,
            "phase": "A", "frozen_before_simulation": True,
        })
        source_snapshot = base.read(source / "runtime_snapshot.json")
        base.write(snapshot_path, {
            "source_hashes": source_snapshot["source_hashes"],
            "input_hashes": {
                str(config_path.resolve()): base.sha(config_path),
                str(manifest_path.resolve()): base.sha(manifest_path),
            },
            "identity_kind": "dependency_scoped_source_hash_snapshot",
            "not_final_substrate_freeze": True,
        })
    # A first dispatch made it through snapshot validation but was rejected by
    # the worker's closed scientific-role allow-list before simulation.  Repair
    # only that generated metadata when no worker artifact exists; the physical
    # config otherwise remains byte-for-byte inherited from timing B.
    inherited_role = base.read(TIMING / "execution/B/execution_config.json")["scientific_role"]
    current = base.read(config_path)
    completed = list((folder / "workers").glob("*.json")) if (folder / "workers").exists() else []
    if current["scientific_role"] != inherited_role:
        if completed:
            raise RuntimeError("cannot repair scientific role after a worker artifact exists")
        current["scientific_role"] = inherited_role
        base.write(config_path, current)
        manifest = base.read(manifest_path)
        manifest["config_sha256"] = base.sha(config_path)
        base.write(manifest_path, manifest)
        snapshot = base.read(snapshot_path)
        snapshot["input_hashes"] = {
            str(config_path.resolve()): base.sha(config_path),
            str(manifest_path.resolve()): base.sha(manifest_path),
        }
        base.write(snapshot_path, snapshot)
    if base.read(manifest_path)["candidates"] != proposals:
        raise RuntimeError("frozen wave-1 proposals changed")
    return folder, config_path, manifest_path, snapshot_path


def _guard(unit: dict, parent: dict) -> tuple[bool, list[str]]:
    reasons = []
    patient = unit["patient"]
    regularizer = unit["regularizer"]
    parent_regularizer = parent["regularizer"]
    if patient["N"] < 0.8 * parent["patient"]["N"]:
        reasons.append("primary_event_count_below_80pct_parent")
    mode_counts = patient.get("mode_counts") or []
    if len(mode_counts) != 2 or min(mode_counts) < 4:
        reasons.append("mode_collapsed_or_under_four_events")
    if regularizer["primary_window_fraction"] < 0.8 * parent_regularizer["primary_window_fraction"]:
        reasons.append("primary_window_fraction_below_80pct_parent")
    if regularizer["mean_background_mass_per_frame"] > 1.25 * parent_regularizer["mean_background_mass_per_frame"]:
        reasons.append("background_mass_above_125pct_parent")
    return not reasons, reasons


def run(workers: int) -> None:
    _wait_for_timing()
    if base.read(CANARY / "canary.json")["status"] != "CANARY_PASS_READY_FOR_FROZEN_LAMBDA":
        raise RuntimeError("native regularizer canary has not passed")
    design, temporal, old = timing.get_frozen()
    history, parents = _history_and_parents(design, temporal, old)
    lambda_contract = _freeze_lambda(history)
    proposals, draws = _proposals(design, parents)
    frozen = {
        "version": "native_activity_regularized_search_v1_wave1",
        "status": "FROZEN_BEFORE_NEW_SIMULATION",
        "question": "within matched patient observables, prefer activity supported by delayed recurrent recruitment",
        "patient_loss": "unchanged contact-timing pilot TRAIN objective",
        "regularizer": PRIMARY_REGULARIZER,
        "regularizer_role": "model-internal mechanism prior; not patient-derived truth or causal proof",
        "parents": parents, "lambda_contract": lambda_contract,
        "proposals": proposals, "draws": draws,
        "budget": {"conditions": 6, "topology_units_per_condition": 2, "simulations": 12,
                   "duration_ms": 24000, "maximum_workers": workers},
        "comparison": "same physical candidates ranked at lambda=0 and frozen lambda grid",
        "guards": ["event count", "both modes", "primary-window fraction", "background burden"],
        "stop": "after wave 1 scoring; no adaptive wave 2, replay, model freeze, or figure line",
        "time_pilot_review_opened_for_selection": False,
    }
    base.write(OUT / "design.json", frozen)
    folder, config_path, manifest_path, snapshot_path = _execution(proposals)
    engine.OUT = OUT
    jobs = [(candidate["candidate_id"], topology, dynamics)
            for candidate in proposals for topology, dynamics in PAIRS]
    engine._run_jobs(
        "A", jobs, maximum_workers=workers,
        execution=(folder, config_path, manifest_path, snapshot_path),
    )
    audit = audit_execution(
        require_complete=True, execution=folder, seed_pairs=PAIRS,
        output_path=OUT / "A_parameter_application_audit.json",
    )
    if audit["status"] != "PARAMETER_APPLICATION_AUDIT_PASS":
        raise RuntimeError("wave-1 actual parameter audit failed")
    support = {seed: _support(seed) for seed, _ in PAIRS}
    parent_by_anchor = {index: row for index, row in enumerate(parents, 1)}
    rows, detailed = [], []
    for candidate in proposals:
        anchor_index = int(candidate["candidate_id"].split("anchor")[1].split("_")[0])
        parent = parent_by_anchor[anchor_index]
        units = []
        for topology, dynamics in PAIRS:
            unit_name = f"topo_{topology}_dyn_{dynamics}"
            path = folder / "workers" / f"{candidate['candidate_id']}_{unit_name}.json"
            record, events = timing.score_record(
                path, temporal, old, candidate["candidate_id"], unit_name, "native_reg_A",
            )
            regularizer = _regularizer_record(path, support[topology])
            parent_unit = next(unit for unit in parent["units"] if unit["unit"] == unit_name)
            unit = {"unit": unit_name, "patient": record["score"], "regularizer": regularizer}
            passed, reasons = _guard(unit, parent_unit)
            unit.update({"guard_pass": passed, "guard_reasons": reasons})
            units.append(unit)
            detailed.append({
                "candidate_id": candidate["candidate_id"], "parent_id": parent["candidate_id"],
                "unit": unit_name, "patient_loss": record["score"]["loss"],
                "R_unsupported": regularizer["mean_unsupported_fraction"],
                "N": record["score"]["N"], "mode_counts": record["score"].get("mode_counts"),
                "primary_window_fraction": regularizer["primary_window_fraction"],
                "background_mass_per_frame": regularizer["mean_background_mass_per_frame"],
                "guard_pass": passed, "guard_reasons": ";".join(reasons),
            })
        eligible = all(unit["guard_pass"] and unit["patient"]["loss"] is not None for unit in units)
        patient_loss = float(np.mean([unit["patient"]["loss"] for unit in units]))
        regularizer_loss = float(np.mean([
            unit["regularizer"]["mean_unsupported_fraction"] for unit in units
        ]))
        row = {
            "candidate_id": candidate["candidate_id"], "parent_id": parent["candidate_id"],
            "eligible": eligible, "patient_loss": patient_loss,
            "R_unsupported": regularizer_loss,
        }
        for index, lam in enumerate(lambda_contract["lambda_grid"]):
            row[f"lambda_{index}"] = lam
            row[f"objective_{index}"] = patient_loss + lam * regularizer_loss
        rows.append(row)
    _write_csv(OUT / "wave1_units.csv", detailed)
    _write_csv(OUT / "wave1_tradeoff.csv", rows)
    base.write(OUT / "wave1_scores.json", {
        "status": "FIRST_WAVE_COMPLETE_PENDING_SCIENTIFIC_REVIEW",
        "lambda_contract": lambda_contract, "candidates": rows,
        "automatic_wave2": False, "model_frozen": False,
    })
    _status("FIRST_WAVE_COMPLETE_PENDING_SCIENTIFIC_REVIEW",
            complete=12, total=12, automatic_wave2=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.workers)
        except Exception as exc:
            _status("ERROR_REVIEW_REQUIRED", error=repr(exc))
            raise
