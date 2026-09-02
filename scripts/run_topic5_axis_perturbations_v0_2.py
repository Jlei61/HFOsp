#!/usr/bin/env python3
"""Run frozen axis, control, and empirical-chord interventions for Goal 5.2B."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    load_frozen_cell,
    parameter_state_sha256,
    parse_bool,
    sha256_file,
)
from src.topic5_latent_perturbation_v0_2 import DOSES, PRIMARY_DOSE  # noqa: E402
from src.topic5_latent_response_v0_2 import (  # noqa: E402
    prefix_ranks_for_references,
    project_centered_contact_response,
    raw_logits_stop,
    rollout_hidden_branches,
)
from scripts.freeze_topic5_latent_reference_states_v0_2 import (  # noqa: E402
    CONTROL_NAMES,
    FREEZE_REVISION,
    REFERENCE,
    reference_dir,
)
from scripts.run_topic5_latent_pass1_v0_2 import ANALYSIS_REVISION, OUT, PARENT  # noqa: E402


PERTURB = OUT / "axis_perturbation" / "responses"
PERTURB_REVISION = "PASS2_RESPONSE_R0_OPEN_CLOSED_LOOP_CONTINUOUS_FIELD"
HORIZON = 3
AXIS_NAMES = ("PROGRESS", "FIELD")


def response_dir(row: pd.Series) -> Path:
    return PERTURB / "per_cell" / str(row.fit_id) / str(row.public_arm) / f"seed{int(row.seed)}"


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def future_input(
    ranks: np.ndarray, event: np.ndarray, rank_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(ranks)[np.asarray(event, dtype=int)]
    requested = np.asarray(rank_index, dtype=int)
    maximum = np.max(np.where(rows >= 0, rows, -1), axis=1)
    valid = requested <= maximum
    x = (rows == requested[:, None]).astype(np.float32)
    recruited = ((rows >= 0) & (rows <= requested[:, None])).astype(np.uint8)
    return x, recruited, valid


@torch.no_grad()
def open_loop_pair(
    model: torch.nn.Module,
    h_left: np.ndarray,
    h_right: np.ndarray,
    recruited: np.ndarray,
    step: np.ndarray,
    event: np.ndarray,
    ranks: np.ndarray,
    denominator: np.ndarray,
    output_progress: np.ndarray,
    output_field: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    count, contacts = recruited.shape
    contact = np.full((count, HORIZON + 1, contacts), np.nan, dtype=np.float32)
    scores = np.full((count, HORIZON + 1, 2), np.nan, dtype=np.float32)
    stop_response = np.full((count, HORIZON + 1), np.nan, dtype=np.float32)
    logit_norm = np.full((count, HORIZON + 1), np.nan, dtype=np.float32)
    valid = np.zeros((count, HORIZON + 1), dtype=np.uint8)
    left = torch.as_tensor(h_left, dtype=torch.float32, device=device)
    right = torch.as_tensor(h_right, dtype=torch.float32, device=device)
    denom = np.asarray(denominator, dtype=np.float64)
    current_recruited = np.asarray(recruited, dtype=np.uint8)
    for tau in range(HORIZON + 1):
        current_step = np.asarray(step, dtype=int) + tau
        if tau == 0:
            use = np.ones(count, dtype=bool)
        else:
            x, current_recruited, use = future_input(ranks, event, current_step)
            x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
            next_left = model._step(left, x_tensor)
            next_right = model._step(right, x_tensor)
            use_tensor = torch.as_tensor(use[:, None], dtype=torch.bool, device=device)
            left = torch.where(use_tensor, next_left, left)
            right = torch.where(use_tensor, next_right, right)
        r_tensor = torch.as_tensor(current_recruited, dtype=torch.bool, device=device)
        k_tensor = torch.as_tensor(current_step, dtype=torch.long, device=device)
        left_logits, left_stop, _ = raw_logits_stop(model, left, k_tensor, r_tensor)
        right_logits, right_stop, _ = raw_logits_stop(model, right, k_tensor, r_tensor)
        difference = (
            right_logits.detach().cpu().numpy() - left_logits.detach().cpu().numpy()
        ) / denom[:, None]
        response_score = project_centered_contact_response(
            difference, output_progress, output_field
        )
        contact[use, tau] = difference[use]
        scores[use, tau] = response_score[use]
        stop_values = (
            right_stop.detach().cpu().numpy() - left_stop.detach().cpu().numpy()
        ) / denom
        stop_response[use, tau] = stop_values[use]
        logit_norm[use, tau] = np.linalg.norm(difference[use], axis=1)
        valid[use, tau] = 1
    return {
        "contact_response": contact,
        "scores": scores,
        "stop_response": stop_response,
        "logit_response_norm": logit_norm,
        "valid": valid,
    }


def paired_closed_loop(
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    h_left: np.ndarray,
    h_right: np.ndarray,
    recruited: np.ndarray,
    step: np.ndarray,
    prefix_ranks: np.ndarray,
    denominator: np.ndarray,
    output_progress: np.ndarray,
    output_field: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    count = len(h_left)
    result = rollout_hidden_branches(
        model, decoder,
        np.concatenate([h_left, h_right]),
        np.concatenate([recruited, recruited]),
        np.concatenate([step, step]),
        np.concatenate([prefix_ranks, prefix_ranks]),
        device, response_horizon=HORIZON,
    )
    denominator = np.asarray(denominator, dtype=np.float64)
    left_logits, right_logits = result["raw_logits"][:count], result["raw_logits"][count:]
    risk = np.isfinite(left_logits).all(axis=2) & np.isfinite(right_logits).all(axis=2)
    difference = (right_logits - left_logits) / denominator[:, None, None]
    scores = project_centered_contact_response(difference, output_progress, output_field)
    scores[~risk] = np.nan
    left_stop = result["stop_probability"][:count, : HORIZON + 1]
    right_stop = result["stop_probability"][count:, : HORIZON + 1]
    stop_response = (right_stop - left_stop) / denominator[:, None]
    stop_response[~risk] = np.nan
    left_field = result["terminal_start_removed_field"][:count]
    right_field = result["terminal_start_removed_field"][count:]
    terminal_difference = (right_field - left_field) / denominator[:, None]
    terminal_scores = project_centered_contact_response(
        terminal_difference, output_progress, output_field
    )
    terminal_stop = (
        result["generated_steps"][count:].astype(float)
        - result["generated_steps"][:count].astype(float)
    ) / denominator
    return {
        "scores": scores.astype(np.float32),
        "risk": risk.astype(np.uint8),
        "stop_response": stop_response.astype(np.float32),
        "terminal_scores": terminal_scores.astype(np.float32),
        "terminal_stop_length_response": terminal_stop.astype(np.float32),
        "left_terminal_ranks": result["terminal_ranks"][:count],
        "right_terminal_ranks": result["terminal_ranks"][count:],
        "left_stop_trajectory": result["stop_probability"][:count],
        "right_stop_trajectory": result["stop_probability"][count:],
    }


def run_pair_subset(
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    base_h: np.ndarray,
    direction: np.ndarray,
    magnitude: np.ndarray,
    eligible: np.ndarray,
    recruited: np.ndarray,
    step: np.ndarray,
    event: np.ndarray,
    ranks: np.ndarray,
    prefix_ranks: np.ndarray,
    output_progress: np.ndarray,
    output_field: np.ndarray,
    device: torch.device,
    *,
    central: bool,
    closed_loop: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    indices = np.flatnonzero(eligible)
    if not len(indices):
        return indices, np.empty(0), {}, None
    delta = magnitude[indices, None] * direction[indices]
    if central:
        left, right = base_h[indices] - delta, base_h[indices] + delta
        denominator = 2.0 * magnitude[indices]
    else:
        left, right = base_h[indices], base_h[indices] + delta
        denominator = magnitude[indices]
    open_result = open_loop_pair(
        model, left, right, recruited[indices], step[indices], event[indices], ranks,
        denominator, output_progress, output_field, device,
    )
    closed_result = None
    if closed_loop:
        closed_result = paired_closed_loop(
            model, decoder, left, right, recruited[indices], step[indices],
            prefix_ranks[indices], denominator, output_progress, output_field, device,
        )
    return indices, denominator, open_result, closed_result


def run_cell(row: pd.Series, device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    started = time.perf_counter()
    model, decoder, _, _ = load_frozen_cell(PARENT, row, device)
    model_hash = parameter_state_sha256(model)
    decoder_hash = parameter_state_sha256(decoder)
    frozen_dir = reference_dir(row)
    with np.load(frozen_dir / "reference_contract.npz", allow_pickle=False) as source:
        q = {key: np.asarray(source[key]) for key in source.files}
    cache = PARENT / "cache" / str(row.fit_id)
    with np.load(cache / "events.npz", allow_pickle=False) as source:
        ranks = np.asarray(source["ranks"])
    h = q["hidden"].astype(np.float64)
    recruited = q["recruited"].astype(np.uint8)
    step = q["step"].astype(int)
    event = q["reference_event_index"].astype(int)
    prefix = prefix_ranks_for_references(ranks, event, step)
    output_progress = q["contact_progress_axis"].astype(np.float64)
    output_field = q["contact_future_field_axis"].astype(np.float64)
    n_ref, contacts = len(h), int(row.n_contacts)

    axis_open_contact = np.full((n_ref, 2, len(DOSES), HORIZON + 1, contacts), np.nan, np.float32)
    axis_open_scores = np.full((n_ref, 2, len(DOSES), HORIZON + 1, 2), np.nan, np.float32)
    axis_open_stop = np.full((n_ref, 2, len(DOSES), HORIZON + 1), np.nan, np.float32)
    axis_open_logit_norm = np.full_like(axis_open_stop, np.nan)
    axis_open_valid = np.zeros_like(axis_open_stop, dtype=np.uint8)
    axis_closed_scores = np.full_like(axis_open_scores, np.nan)
    axis_closed_stop = np.full_like(axis_open_stop, np.nan)
    axis_closed_risk = np.zeros_like(axis_open_stop, dtype=np.uint8)
    axis_terminal_scores = np.full((n_ref, 2, len(DOSES), 2), np.nan, np.float32)
    axis_terminal_stop = np.full((n_ref, 2, len(DOSES)), np.nan, np.float32)
    axis_terminal_ranks = np.full((n_ref, 2, len(DOSES), 2, contacts), -1, np.int16)
    for axis_index in range(2):
        for dose_index, dose in enumerate(DOSES):
            direction = q["progress_axis"] if axis_index == 0 else q["field_axis"]
            magnitude = float(dose) * q["axis_local_sd"][:, axis_index]
            eligible = q["axis_support_checks"][:, axis_index, dose_index].all(axis=(1, 2))
            eligible &= np.isfinite(direction).all(axis=1) & np.isfinite(magnitude) & (magnitude > 1e-8)
            indices, _, open_result, closed_result = run_pair_subset(
                model, decoder, h, direction, magnitude, eligible, recruited, step, event,
                ranks, prefix, output_progress, output_field, device,
                central=True, closed_loop=True,
            )
            if not len(indices):
                continue
            axis_open_contact[indices, axis_index, dose_index] = open_result["contact_response"]
            axis_open_scores[indices, axis_index, dose_index] = open_result["scores"]
            axis_open_stop[indices, axis_index, dose_index] = open_result["stop_response"]
            axis_open_logit_norm[indices, axis_index, dose_index] = open_result["logit_response_norm"]
            axis_open_valid[indices, axis_index, dose_index] = open_result["valid"]
            assert closed_result is not None
            axis_closed_scores[indices, axis_index, dose_index] = closed_result["scores"]
            axis_closed_stop[indices, axis_index, dose_index] = closed_result["stop_response"]
            axis_closed_risk[indices, axis_index, dose_index] = closed_result["risk"]
            axis_terminal_scores[indices, axis_index, dose_index] = closed_result["terminal_scores"]
            axis_terminal_stop[indices, axis_index, dose_index] = closed_result["terminal_stop_length_response"]
            axis_terminal_ranks[indices, axis_index, dose_index, 0] = closed_result["left_terminal_ranks"]
            axis_terminal_ranks[indices, axis_index, dose_index, 1] = closed_result["right_terminal_ranks"]

    axis_arrays = {
        "axis_names": np.asarray(AXIS_NAMES), "doses": DOSES.astype(np.float32),
        "event_index": event.astype(np.int64), "step": step.astype(np.int16),
        "phase": q["phase"].astype(np.float32), "phase_target": q["phase_target"].astype(np.float32),
        "open_contact_response": axis_open_contact,
        "open_scores": axis_open_scores, "open_stop_response": axis_open_stop,
        "open_logit_response_norm": axis_open_logit_norm, "open_valid": axis_open_valid,
        "closed_scores": axis_closed_scores, "closed_stop_response": axis_closed_stop,
        "closed_risk": axis_closed_risk, "terminal_scores": axis_terminal_scores,
        "terminal_stop_length_response": axis_terminal_stop,
        "terminal_ranks": axis_terminal_ranks,
    }

    n_controls = len(CONTROL_NAMES)
    control_open_scores = np.full((n_ref, n_controls, HORIZON + 1, 2), np.nan, np.float32)
    control_open_norm = np.full((n_ref, n_controls, HORIZON + 1), np.nan, np.float32)
    control_open_valid = np.zeros((n_ref, n_controls, HORIZON + 1), np.uint8)
    control_closed_scores = np.full_like(control_open_scores, np.nan)
    control_closed_risk = np.zeros_like(control_open_valid)
    control_terminal_scores = np.full((n_ref, n_controls, 2), np.nan, np.float32)
    control_terminal_stop = np.full((n_ref, n_controls), np.nan, np.float32)
    for control_index in range(n_controls):
        direction = q["control_directions"][:, control_index].astype(np.float64)
        magnitude = PRIMARY_DOSE * q["control_local_sd"][:, control_index].astype(np.float64)
        eligible = q["control_support_checks"][:, control_index].all(axis=(1, 2))
        eligible &= np.isfinite(direction).all(axis=1) & np.isfinite(magnitude) & (magnitude > 1e-8)
        indices, _, open_result, closed_result = run_pair_subset(
            model, decoder, h, direction, magnitude, eligible, recruited, step, event,
            ranks, prefix, output_progress, output_field, device,
            central=True, closed_loop=True,
        )
        if not len(indices):
            continue
        control_open_scores[indices, control_index] = open_result["scores"]
        control_open_norm[indices, control_index] = open_result["logit_response_norm"]
        control_open_valid[indices, control_index] = open_result["valid"]
        assert closed_result is not None
        control_closed_scores[indices, control_index] = closed_result["scores"]
        control_closed_risk[indices, control_index] = closed_result["risk"]
        control_terminal_scores[indices, control_index] = closed_result["terminal_scores"]
        control_terminal_stop[indices, control_index] = closed_result["terminal_stop_length_response"]
    control_arrays = {
        "control_names": np.asarray(CONTROL_NAMES), "primary_dose": np.asarray(PRIMARY_DOSE, np.float32),
        "event_index": event.astype(np.int64), "step": step.astype(np.int16),
        "phase_target": q["phase_target"].astype(np.float32),
        "open_scores": control_open_scores, "open_logit_response_norm": control_open_norm,
        "open_valid": control_open_valid, "closed_scores": control_closed_scores,
        "closed_risk": control_closed_risk, "terminal_scores": control_terminal_scores,
        "terminal_stop_length_response": control_terminal_stop,
    }

    chords = pd.read_csv(frozen_dir / "chords.csv")
    n_chords = len(chords)
    chord_open_scores = np.full((n_chords, len(DOSES), HORIZON + 1, 2), np.nan, np.float32)
    chord_open_valid = np.zeros((n_chords, len(DOSES), HORIZON + 1), np.uint8)
    chord_closed_scores = np.full_like(chord_open_scores, np.nan)
    chord_closed_risk = np.zeros_like(chord_open_valid)
    chord_terminal_scores = np.full((n_chords, len(DOSES), 2), np.nan, np.float32)
    chord_terminal_stop = np.full((n_chords, len(DOSES)), np.nan, np.float32)
    if n_chords:
        source_index = chords["reference_index"].to_numpy(int)
        target_index = chords["target_reference_index"].to_numpy(int)
        chord_direction = h[target_index] - h[source_index]
        for dose_index, dose in enumerate(DOSES):
            magnitude = np.full(n_chords, float(dose), dtype=np.float64)
            eligible = chords[f"support_eta_{dose:.2f}"].map(parse_bool).to_numpy(bool)
            eligible &= np.isfinite(chord_direction).all(axis=1)
            indices = np.flatnonzero(eligible)
            if not len(indices):
                continue
            source = source_index[indices]
            delta = magnitude[indices, None] * chord_direction[indices]
            open_result = open_loop_pair(
                model, h[source], h[source] + delta, recruited[source], step[source],
                event[source], ranks, magnitude[indices], output_progress, output_field, device,
            )
            closed_result = paired_closed_loop(
                model, decoder, h[source], h[source] + delta, recruited[source], step[source],
                prefix[source], magnitude[indices], output_progress, output_field, device,
            )
            chord_open_scores[indices, dose_index] = open_result["scores"]
            chord_open_valid[indices, dose_index] = open_result["valid"]
            chord_closed_scores[indices, dose_index] = closed_result["scores"]
            chord_closed_risk[indices, dose_index] = closed_result["risk"]
            chord_terminal_scores[indices, dose_index] = closed_result["terminal_scores"]
            chord_terminal_stop[indices, dose_index] = closed_result["terminal_stop_length_response"]
    chord_arrays = {
        "doses": DOSES.astype(np.float32),
        "family": np.asarray(chords["family"].astype(str).tolist(), dtype="U16") if n_chords else np.asarray([], dtype="U1"),
        "reference_index": chords["reference_index"].to_numpy(np.int32) if n_chords else np.asarray([], np.int32),
        "target_reference_index": chords["target_reference_index"].to_numpy(np.int32) if n_chords else np.asarray([], np.int32),
        "u_difference": chords["u_difference"].to_numpy(np.float32) if n_chords else np.asarray([], np.float32),
        "direction_norm": chords["direction_norm"].to_numpy(np.float32) if n_chords else np.asarray([], np.float32),
        "open_scores": chord_open_scores, "open_valid": chord_open_valid,
        "closed_scores": chord_closed_scores, "closed_risk": chord_closed_risk,
        "terminal_scores": chord_terminal_scores,
        "terminal_stop_length_response": chord_terminal_stop,
    }

    primary_axis_eligible = q["axis_support_checks"][:, :, 1].all(axis=(2, 3))
    metrics = {
        "contract": "topic5_axis_perturbation_cell_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_revision": ANALYSIS_REVISION, "freeze_revision": FREEZE_REVISION,
        "perturbation_revision": PERTURB_REVISION, "status": "PASS",
        "patient": str(row.patient), "fit_id": str(row.fit_id),
        "public_arm": str(row.public_arm), "seed": int(row.seed),
        "n_reference_states": n_ref,
        "primary_progress_supported_states": int(primary_axis_eligible[:, 0].sum()),
        "primary_field_supported_states": int(primary_axis_eligible[:, 1].sum()),
        "n_high_u_chords": int((chords["family"] == "HIGH_U").sum()) if n_chords else 0,
        "n_small_u_chords": int((chords["family"] == "SMALL_U").sum()) if n_chords else 0,
        "model_hash_unchanged": model_hash == parameter_state_sha256(model),
        "decoder_hash_unchanged": decoder_hash == parameter_state_sha256(decoder),
        "reference_contract_sha256": sha256_file(frozen_dir / "reference_contract.npz"),
        "target_values_read": False, "elapsed_seconds": time.perf_counter() - started,
    }
    return axis_arrays, control_arrays, chord_arrays, metrics


def write_cell(row: pd.Series, axis: dict[str, np.ndarray], controls: dict[str, np.ndarray], chords: dict[str, np.ndarray], metrics: dict[str, object]) -> None:
    target = response_dir(row)
    target.mkdir(parents=True, exist_ok=True)
    failure = target / "FAILURE.json"
    if failure.is_file():
        failure.replace(target / "RECOVERED_FAILURE.json")
    write_npz(target / "axis_responses.npz", axis)
    write_npz(target / "control_responses.npz", controls)
    write_npz(target / "chord_responses.npz", chords)
    atomic_write_json(target / "metrics.json", metrics)
    atomic_write_json(target / "DONE.json", {
        "ok": True, "perturbation_revision": PERTURB_REVISION,
        "axis_sha256": sha256_file(target / "axis_responses.npz"),
        "controls_sha256": sha256_file(target / "control_responses.npz"),
        "chords_sha256": sha256_file(target / "chord_responses.npz"),
        "metrics_sha256": sha256_file(target / "metrics.json"),
        "target_values_read": False,
    })


def aggregate(manifest: pd.DataFrame) -> dict[str, object]:
    rows, missing = [], []
    for item in manifest.itertuples(index=False):
        row = pd.Series(item._asdict())
        target = response_dir(row)
        if not (target / "DONE.json").is_file():
            missing.append(f"{item.fit_id}/{item.public_arm}/seed{item.seed}")
            continue
        rows.append(json.loads((target / "metrics.json").read_text()))
    if rows:
        atomic_write_csv(PERTURB / "PERTURBATION_CELL_SUMMARY.csv", pd.DataFrame(rows))
    payload = {
        "contract": "topic5_axis_perturbation_execution_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_revision": ANALYSIS_REVISION, "freeze_revision": FREEZE_REVISION,
        "perturbation_revision": PERTURB_REVISION,
        "status": "PASS" if len(rows) == 630 and not missing else "INCOMPLETE",
        "scheduled_cells": 630, "completed_cells": len(rows),
        "reference_states": int(sum(int(row["n_reference_states"]) for row in rows)),
        "primary_progress_supported_states": int(sum(int(row["primary_progress_supported_states"]) for row in rows)),
        "primary_field_supported_states": int(sum(int(row["primary_field_supported_states"]) for row in rows)),
        "missing_count": len(missing), "missing_first20": missing[:20],
        "target_values_read": False,
    }
    atomic_write_json(PERTURB / "PERTURBATION_EXECUTION_STATUS.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cell-key")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    seal = json.loads((REFERENCE / "REFERENCE_FREEZE_SEAL.json").read_text())
    if not seal.get("sealed") or seal.get("freeze_revision") != FREEZE_REVISION:
        raise RuntimeError("response-blind reference freeze must be sealed before perturbation")
    if seal.get("manifest_sha256") != sha256_file(REFERENCE / "REFERENCE_STATE_MANIFEST.csv"):
        raise RuntimeError("frozen reference manifest content hash changed")
    manifest_all = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv")
    manifest = manifest_all.copy()
    if args.cell_key:
        fit, arm, seed_text = args.cell_key.split("/")
        manifest = manifest[
            manifest["fit_id"].eq(fit) & manifest["public_arm"].eq(arm)
            & manifest["seed"].eq(int(seed_text.removeprefix("seed")))
        ]
    elif args.limit is not None:
        manifest = manifest.iloc[: args.limit]
    device = torch.device(args.device)
    failures = []
    for position, (_, row) in enumerate(manifest.iterrows(), start=1):
        target = response_dir(row)
        if (target / "DONE.json").is_file() and not args.force:
            print(f"skip {position}/{len(manifest)} {row.fit_id}/{row.public_arm}/seed{row.seed}", flush=True)
            continue
        try:
            axis, controls, chords, metrics = run_cell(row, device)
            write_cell(row, axis, controls, chords, metrics)
            print(
                f"done {position}/{len(manifest)} {row.fit_id}/{row.public_arm}/seed{row.seed} "
                f"support={metrics['primary_progress_supported_states']}/"
                f"{metrics['primary_field_supported_states']} {metrics['elapsed_seconds']:.2f}s",
                flush=True,
            )
        except Exception as error:
            failures.append({
                "fit_id": row.fit_id, "public_arm": row.public_arm, "seed": int(row.seed),
                "error_type": type(error).__name__, "error": str(error),
            })
            atomic_write_json(target / "FAILURE.json", failures[-1])
            print(f"FAIL {row.fit_id}/{row.public_arm}/seed{row.seed}: {error}", flush=True)
    status = aggregate(manifest_all)
    print(json.dumps({"run_failures": failures, "aggregate": status}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
