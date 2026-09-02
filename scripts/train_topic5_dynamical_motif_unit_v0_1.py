#!/usr/bin/env python3
"""Train one Topic 5.2 dynamical motif unit (frame x model x seed).

Teacher-forced training with anchored joint fine-tuning from the previous
layer, calibration-split checkpoint selection, then a frozen stochastic
decoder.  Model-unseen events (``split == -1``) are never loaded here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_dynamical_motif_data_v0_1 import load_frame_unit  # noqa: E402
from src.topic5_dynamical_motif_rnn_v0_1 import (  # noqa: E402
    ALL_MODELS,
    MotifConfig,
    MotifRNN,
    NEW_PARAMETERS,
    WARM_START_PARENT,
    build_motif_event_tensors,
    freeze_direction_scale,
)
from src.topic5_dynamical_motif_rollout_v0_1 import (  # noqa: E402
    DecoderContract,
    calibrate_temperatures,
    fit_size_head,
    teacher_forced_traces,
)
from src.topic5_wiring_economy_rnn import (  # noqa: E402
    cardinality_conditioned_nll,
    next_rank_stop_loss,
)

DEFAULTS: dict[str, Any] = {
    "lr": 6e-3,
    "shared_lr_ratio": 0.2,
    "lambda_anchor": 0.03,
    "stop_weight": 1.0,
    "max_epochs": 1500,
    "patience": 40,
    "min_relative_improvement": 0.0,
    "max_batches_per_epoch": 120,
    "max_batch": 1024,
    "min_updates_per_epoch": 8,
    "gradient_clip": 5.0,
    "eval_batch": 1024,
    "resume_every_epochs": 25,
    "max_seconds": 7200,
}
THETA_INITS = (0.0, math.pi / 3.0, 2.0 * math.pi / 3.0)
SELECTION_METRICS = ("joint", "contact_nll")


def stable_seed(label: str, salt: int = 0) -> int:
    digest = hashlib.sha256(f"{label}|{salt}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonable) + "\n")
    temporary.replace(path)


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")


def resolve_batch(n_train: int, cfg: dict[str, Any]) -> int:
    return int(min(cfg["max_batch"],
                   max(1, int(np.ceil(n_train / cfg["min_updates_per_epoch"])))))


TENSOR_KEYS = ("x", "recruited", "displacement", "target", "available", "valid", "is_last")


def place_tensors(tensors: dict[str, torch.Tensor], device: torch.device,
                  budget_bytes: int = 3 << 30) -> tuple[dict[str, torch.Tensor], bool]:
    """Keep the whole padded event set on the device when it fits.

    Per-batch host indexing plus a host-to-device copy dominates the step cost
    for these small structured operators, and the padded tensors are at most a
    few hundred megabytes for every fit in this cohort.
    """
    total = sum(tensors[key].numel() * tensors[key].element_size() for key in TENSOR_KEYS)
    if device.type != "cuda" or total > budget_bytes:
        return tensors, False
    return {**tensors, **{key: tensors[key].to(device) for key in TENSOR_KEYS}}, True


@torch.no_grad()
def evaluate(model, tensors, indices, device, batch_size, stop_weight) -> dict[str, float]:
    model.eval()
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return {key: float("nan") for key in ("next_bce", "stop_bce", "contact_nll", "top1")}
    totals = {"next_bce": 0.0, "stop_bce": 0.0, "contact_nll": 0.0, "top1": 0.0}
    decisions = 0.0
    for begin in range(0, indices.size, int(batch_size)):
        chosen = torch.as_tensor(indices[begin:begin + int(batch_size)],
                                 device=tensors["x"].device)
        batch = {key: tensors[key][chosen].to(device) for key in TENSOR_KEYS}
        logits, stops, _ = model(batch["x"], batch["recruited"], batch["displacement"])
        _, next_bce, stop_bce = next_rank_stop_loss(
            logits, stops, batch["target"], batch["available"], batch["valid"], batch["is_last"],
            stop_weight=stop_weight,
        )
        predict = batch["valid"] & ~batch["is_last"]
        nll = cardinality_conditioned_nll(logits, batch["target"], batch["available"], predict)
        masked = logits.masked_fill(~batch["available"], -1e9)
        hit = ((batch["target"].gather(-1, masked.argmax(-1, keepdim=True)).squeeze(-1) > 0)
               & predict).float().sum()
        weight = float(predict.float().sum())
        totals["next_bce"] += float(next_bce) * weight
        totals["stop_bce"] += float(stop_bce) * weight
        totals["contact_nll"] += float(nll) * weight
        totals["top1"] += float(hit)
        decisions += weight
    result = {key: value / max(1.0, decisions) for key, value in totals.items()}
    result["n_continue_decisions"] = int(decisions)
    return result


def selection_score(record: dict[str, float], selection_metric: str,
                    stop_weight: float) -> float:
    """Checkpoint score; contact-only selection isolates the spatial task."""
    if selection_metric == "contact_nll":
        return float(record["contact_nll"])
    if selection_metric == "joint":
        return float(record["next_bce"] + stop_weight * record["stop_bce"])
    raise ValueError(f"unknown selection metric {selection_metric!r}")


# The grid is both the layer's starting point and the recorded dose-response
# profile of the motif at the previous layer's solution, so it includes the zero
# point and is symmetric where the parameter has a sign.
MOTIF_INIT_GRID = {
    "DM1_FREE_AXIS": {
        "theta": [i * math.pi / 12.0 for i in range(12)],
        "eta_raw": [0.0, 0.05, 0.10, 0.20, 0.40, 0.80],
    },
    "DM2_LOCAL_DIRECTIONAL": {
        "beta": [-2.0, -1.5, -1.0, -0.6, -0.3, -0.15, 0.0,
                 0.15, 0.3, 0.6, 1.0, 1.5, 2.0],
    },
    "DM3_AXIS_FEEDFORWARD_TRANSIENT": {
        "gamma_raw": [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
    "DM3_SYMMETRIC_MATCHED": {
        "gamma_raw": [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
    "DM3_AXIS_SHUFFLED_TRIANGULAR": {
        "gamma_raw": [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
    "DM3_GAIN_MEMORY": {"delta_g": [0.0, 0.1, 0.3, 0.7], "delta_kappa": [0.0, 0.3, 0.9, 1.8]},
}


@torch.no_grad()
def _axis_has_no_effect(model, tensors, calibration_idx, device, probe: int = 64) -> bool:
    """True when rotating the axis leaves the forward pass bit-identical."""
    if not isinstance(getattr(model, "theta", None), torch.nn.Parameter):
        return False
    indices = np.asarray(calibration_idx, dtype=int)[:probe]
    if indices.size == 0:
        return False
    chosen = torch.as_tensor(indices, device=tensors["x"].device)
    batch = {key: tensors[key][chosen].to(device) for key in ("x", "recruited", "displacement")}
    before = float(model.theta)
    reference = model(batch["x"], batch["recruited"], batch["displacement"])[0]
    model.theta.fill_(before + 0.7)
    rotated = model(batch["x"], batch["recruited"], batch["displacement"])[0]
    model.theta.fill_(before)
    return bool(torch.equal(reference, rotated))


@torch.no_grad()
def grid_initialise_new_parameters(model, tensors, calibration_idx, device, cfg,
                                   model_id: str, selection_metric: str = "joint") -> dict:
    """Pick the new layer's starting point on the calibration split.

    At ``eta = 0`` the axis angle has exactly zero gradient, so a layer that
    warm-starts at the nested-equivalent point can never discover an axis: the
    projection pins ``eta`` at zero whenever the arbitrary initial angle is a
    bad one.  A frozen grid over the layer's own parameters removes that
    pathology without changing the model family, and its landscape is a useful
    record in its own right.
    """
    grid = MOTIF_INIT_GRID.get(model_id)
    if not grid:
        return {"searched": False}
    grid = dict(grid)
    axis_free = False
    if "theta" not in grid and _axis_has_no_effect(model, tensors, calibration_idx, device):
        # The inherited axis is unidentified (the previous layer left eta, beta
        # and gamma at zero), so there is nothing to preserve and searching it
        # here is the only way this layer can find a direction at all.
        grid["theta"] = [i * math.pi / 6.0 for i in range(6)]
        axis_free = True
    names = list(grid)
    original = {name: float(getattr(model, name)) for name in names}
    combinations = [[]]
    for name in names:
        combinations = [row + [value] for row in combinations for value in grid[name]]
    landscape, best, best_values = [], float("inf"), None
    for values in combinations:
        for name, value in zip(names, values):
            getattr(model, name).fill_(float(value))
        score = evaluate(model, tensors, calibration_idx, device, cfg["eval_batch"],
                         float(cfg["stop_weight"]))
        total = selection_score(score, selection_metric, float(cfg["stop_weight"]))
        # The selection metric mixes next-rank and STOP; contact NLL scores only
        # "which contact comes next", which is what a transport motif changes.
        landscape.append({**dict(zip(names, values)), "selection_score": total,
                          "selection_metric": selection_metric,
                          "validation_score": (score["next_bce"]
                                               + float(cfg["stop_weight"]) * score["stop_bce"]),
                          "contact_nll": score["contact_nll"],
                          "next_bce": score["next_bce"], "stop_bce": score["stop_bce"]})
        if np.isfinite(total) and total < best:
            best, best_values = total, list(values)
    if best_values is None:
        for name, value in original.items():
            getattr(model, name).fill_(value)
        return {"searched": True, "selected": None, "landscape": landscape}
    for name, value in zip(names, best_values):
        getattr(model, name).fill_(float(value))
    best_contact = min(landscape, key=lambda row: row["contact_nll"])
    zero_row = min(landscape, key=lambda row: sum(
        abs(row[name]) for name in names if name != "theta"))
    return {
        "searched": True,
        "n_points": len(combinations),
        "axis_was_unidentified": bool(axis_free),
        "selected": dict(zip(names, best_values)),
        "selected_validation_score": best,
        "selection_metric": selection_metric,
        "best_contact_nll_point": best_contact,
        "zero_motif_contact_nll": zero_row["contact_nll"],
        "contact_nll_gain_at_best_point": zero_row["contact_nll"] - best_contact["contact_nll"],
        "landscape": landscape,
    }


def build_model(unit, model_id: str, seed: int, seed_index: int, sigma_s: float,
                gate_rule: str, device: torch.device) -> MotifRNN:
    config = MotifConfig(
        model_id=model_id,
        n_contacts=unit.n_contacts,
        n_nodes=unit.n_nodes,
        observation_operator=unit.H,
        node_xy_mm=unit.nodes_xy_mm,
        local_mask=unit.local_mask,
        r_forward_mm=unit.r_local_mm,
        sigma_s_mm=sigma_s,
        seed=seed,
        theta_init=THETA_INITS[seed_index % len(THETA_INITS)],
        gate_rule=gate_rule,
        shuffle_seed=stable_seed(f"{unit.unit_id}|shuffle", seed_index),
    )
    return MotifRNN(config).to(device)


def train_unit(
    frame: str,
    unit_id: str,
    model_id: str,
    seed_index: int,
    out_root: Path,
    device: torch.device,
    cfg: dict[str, Any],
    gate_rule: str = "M2-2RANK",
    warm_start: Path | None = None,
    freeze_shared: bool = False,
    resume: bool = True,
    tag: str = "formal",
    selection_metric: str = "joint",
) -> dict[str, Any]:
    started = time.time()
    unit = load_frame_unit(out_root, frame, unit_id)
    tensors = build_motif_event_tensors(unit.ranks, unit.contacts_xy_mm, gate_rule=gate_rule)
    tensors, tensors_on_device = place_tensors(tensors, device)
    train_idx = unit.indices(0)
    calibration_idx = unit.indices(1)
    development_idx = unit.indices(2)
    sigma_s = freeze_direction_scale(unit.ranks, unit.contacts_xy_mm, calibration_idx)

    seed = stable_seed(f"{frame}|{unit_id}|{model_id}", seed_index)
    torch.manual_seed(seed)
    model = build_model(unit, model_id, seed, seed_index, sigma_s, gate_rule, device)

    warm_state, warm_meta = None, None
    if warm_start is not None:
        payload = torch.load(warm_start, map_location="cpu", weights_only=False)
        warm_state = payload["model"]
        warm_meta = {"path": str(warm_start), "model_id": payload.get("model_id")}
        copied = model.load_warm_start(warm_state)
        warm_meta["copied_parameters"] = copied
        if model_id == "DM3_AXIS_SHUFFLED_TRIANGULAR":
            warm_meta["shuffle_calibration"] = model.calibrate_shuffle_radius()
    elif WARM_START_PARENT[model_id] is not None:
        raise RuntimeError(f"{model_id} requires a warm start from {WARM_START_PARENT[model_id]}")

    grid_report = {"searched": False}
    warm_start_score = None
    warm_start_state = None
    if warm_state is not None:
        # Snapshot and score the exact nested-equivalent point BEFORE the grid
        # moves the new parameters, so it can compete as an epoch -1 checkpoint.
        warm_start_state = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
        scored = evaluate(model, tensors, calibration_idx, device,
                          cfg["eval_batch"], float(cfg["stop_weight"]))
        warm_start_score = selection_score(
            scored, selection_metric, float(cfg["stop_weight"]))
        grid_report = grid_initialise_new_parameters(
            model, tensors, calibration_idx, device, cfg, model_id,
            selection_metric=selection_metric)

    new_names = set(NEW_PARAMETERS[model_id])
    shared_parameters = [(n, p) for n, p in model.named_parameters() if n not in new_names]
    new_parameters = [(n, p) for n, p in model.named_parameters() if n in new_names]
    if freeze_shared:
        for _, parameter in shared_parameters:
            parameter.requires_grad_(False)
    groups = [{"params": [p for _, p in new_parameters], "lr": cfg["lr"]}]
    if not freeze_shared:
        shared_lr = cfg["lr"] * (cfg["shared_lr_ratio"] if warm_state is not None else 1.0)
        groups.append({"params": [p for _, p in shared_parameters], "lr": shared_lr})
    if not groups[0]["params"]:
        groups = groups[1:]
    optimiser = torch.optim.Adam(groups)

    anchor_reference = None
    anchor_elements = 0
    if warm_state is not None and not freeze_shared:
        anchor_reference = {
            name: parameter.detach().clone()
            for name, parameter in shared_parameters
            if name in warm_state and warm_state[name].shape == parameter.shape
        }
        anchor_elements = sum(int(value.numel()) for value in anchor_reference.values())

    unit_dir = out_root / tag / frame / unit_id / model_id / f"seed{seed_index}"
    unit_dir.mkdir(parents=True, exist_ok=True)
    resume_path = unit_dir / "resume.pt"
    batch_size = resolve_batch(len(train_idx), cfg)
    rng = np.random.default_rng(seed)
    time_limited = False
    begin_epoch, best, best_epoch, best_state, stale, history = 0, float("inf"), -1, None, 0, []
    if warm_start_state is not None and np.isfinite(warm_start_score):
        best, best_epoch, best_state = float(warm_start_score), -1, warm_start_state
    if resume and resume_path.exists():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimiser.load_state_dict(payload["optimiser"])
        rng.bit_generator.state = payload["rng_state"]
        torch.set_rng_state(payload["torch_rng_state"])
        begin_epoch = int(payload["epoch"]) + 1
        best, best_epoch = float(payload["best"]), int(payload["best_epoch"])
        best_state = payload["best_state"]
        stale, history = int(payload["stale"]), list(payload["history"])

    for epoch in range(begin_epoch, int(cfg["max_epochs"])):
        model.train()
        order = rng.permutation(len(train_idx))
        n_batches = min(int(cfg["max_batches_per_epoch"]),
                        int(np.ceil(len(order) / batch_size)))
        epoch_loss = 0.0
        for step in range(n_batches):
            chunk = order[step * batch_size:(step + 1) * batch_size]
            if chunk.size == 0:
                break
            chosen = torch.as_tensor(train_idx[chunk], device=tensors["x"].device)
            batch = {key: tensors[key][chosen].to(device) for key in TENSOR_KEYS}
            logits, stops, _ = model(batch["x"], batch["recruited"], batch["displacement"])
            loss, _, _ = next_rank_stop_loss(
                logits, stops, batch["target"], batch["available"], batch["valid"],
                batch["is_last"], stop_weight=float(cfg["stop_weight"]),
            )
            if anchor_reference:
                # Mean squared drift per element, so one lambda means the same
                # thing for a 64-node and a 339-node patient.
                penalty = sum(
                    ((parameter - anchor_reference[name]) ** 2).sum()
                    for name, parameter in shared_parameters if name in anchor_reference
                ) / max(1, anchor_elements)
                loss = loss + float(cfg["lambda_anchor"]) * penalty
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], float(cfg["gradient_clip"]))
            optimiser.step()
            model.project_constraints()
            epoch_loss += float(loss)
        validation = evaluate(model, tensors, calibration_idx, device,
                              cfg["eval_batch"], float(cfg["stop_weight"]))
        score = selection_score(
            validation, selection_metric, float(cfg["stop_weight"]))
        history.append({"epoch": epoch, "train_loss": epoch_loss / max(1, n_batches),
                        "selection_score": score, "selection_metric": selection_metric,
                        "validation_score": (validation["next_bce"]
                                             + float(cfg["stop_weight"]) * validation["stop_bce"]),
                        **{f"validation_{k}": v
                                                      for k, v in validation.items()}})
        if not np.isfinite(score):
            raise RuntimeError(f"non-finite validation score at epoch {epoch}")
        margin = (abs(best) * float(cfg["min_relative_improvement"])
                  if np.isfinite(best) else 0.0)
        if score < best - margin - 1e-9:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= int(cfg["patience"]):
            break
        if time.time() - started > float(cfg["max_seconds"]):
            # Recorded, never silently hidden: a time-limited base leaves head-
            # room that a later layer could absorb, so the flag travels with the
            # unit into the evidence tables.
            time_limited = True
            break
        if (epoch + 1) % int(cfg["resume_every_epochs"]) == 0:
            torch.save({"model": model.state_dict(), "optimiser": optimiser.state_dict(),
                        "epoch": epoch, "best": best, "best_epoch": best_epoch,
                        "best_state": best_state, "stale": stale, "history": history,
                        "rng_state": rng.bit_generator.state,
                        "torch_rng_state": torch.get_rng_state()},
                       resume_path)
    if best_state is None:
        raise RuntimeError("training produced no finite checkpoint")
    model.load_state_dict(best_state)

    isolation = _component_isolation_replay(
        model, warm_state, model_id, tensors, calibration_idx, unit.indices(-1), device, cfg)

    train_trace = teacher_forced_traces(model, tensors, train_idx, device, cfg["eval_batch"])
    calibration_trace = teacher_forced_traces(model, tensors, calibration_idx, device,
                                              cfg["eval_batch"])
    size_head, size_report = fit_size_head(train_trace, calibration_trace, unit.n_contacts,
                                           seed, device)
    temperatures = calibrate_temperatures(calibration_trace, size_head, device)
    contract = DecoderContract(**{**temperatures, **size_report})

    metrics = {
        "contract": "topic5_dynamical_motif_unit_v0_1",
        "frame": frame,
        "unit_id": unit_id,
        "subject": unit.subject,
        "model_id": model_id,
        "seed_index": seed_index,
        "seed": seed,
        "gate_rule": gate_rule,
        "selection_metric": selection_metric,
        "freeze_shared": bool(freeze_shared),
        "n_contacts": unit.n_contacts,
        "n_nodes": unit.n_nodes,
        "n_train": int(len(train_idx)),
        "n_calibration": int(len(calibration_idx)),
        "n_development": int(len(development_idx)),
        "batch_size": batch_size,
        "sigma_s_mm": sigma_s,
        "r_forward_mm": unit.r_local_mm,
        "theta_init": THETA_INITS[seed_index % len(THETA_INITS)],
        "n_epochs": len(history),
        "best_epoch": best_epoch,
        "time_limited": bool(time_limited),
        "converged": bool(not time_limited and stale >= int(cfg["patience"])),
        "best_validation_score": best,
        "best_selection_score": best,
        "warm_start": warm_meta,
        "warm_start_validation_score": warm_start_score,
        "component_isolation": isolation,
        "motif_init_grid": grid_report,
        "calibration": evaluate(model, tensors, calibration_idx, device, cfg["eval_batch"],
                                float(cfg["stop_weight"])),
        "development": evaluate(model, tensors, development_idx, device, cfg["eval_batch"],
                                float(cfg["stop_weight"])),
        "decoder": contract.to_dict(),
        "numerical_audit": model.numerical_audit(),
        "parameter_drift": _parameter_drift(model, warm_state),
        "parameter_hashes": _parameter_hashes(model),
        "config": dict(cfg),
        "device": str(device),
        "tensors_on_device": bool(tensors_on_device),
        "seconds": time.time() - started,
        "target_values_read": False,
    }
    torch.save({"model": model.state_dict(), "model_id": model_id, "config": model.config.__dict__,
                "sigma_s_mm": sigma_s, "theta_init": THETA_INITS[seed_index % len(THETA_INITS)]},
               unit_dir / "checkpoint.pt")
    torch.save({"size_head": size_head.state_dict(), "contract": contract.to_dict()},
               unit_dir / "decoder.pt")
    write_json(unit_dir / "metrics.json", metrics)
    write_json(unit_dir / "history.json", history)
    write_json(unit_dir / "DONE.json", {
        "ok": True,
        "finite_nonreturning": bool(metrics["numerical_audit"]["recurrent_row_sum_max"] > 1.0),
        "seconds": metrics["seconds"],
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
    })
    if resume_path.exists():
        resume_path.unlink()
    return metrics


@torch.no_grad()
def _component_isolation_replay(model, warm_state, model_id, tensors, calibration_idx,
                                unseen_idx, device, cfg) -> dict:
    """Score the learned motif on top of the *previous* layer's shared solution.

    The joint fit lets the shared parameters move, so its gain mixes "the motif
    helped" with "the shared solution was not fully converged".  Restoring the
    parent's shared parameters and keeping only this layer's new parameters
    isolates the motif itself, and costs no training.
    """
    if warm_state is None or not NEW_PARAMETERS[model_id]:
        return {"available": False}
    learned = {name: float(getattr(model, name)) for name in NEW_PARAMETERS[model_id]}
    if model_id != "DM1_FREE_AXIS" and isinstance(getattr(model, "theta", None), torch.nn.Parameter):
        # The axis is not new at this layer, but a directional or feed-forward
        # motif is meaningless without the angle it was learned at.
        learned["theta"] = float(model.theta)
    current = {name: v.detach().cpu().clone() for name, v in model.state_dict().items()}
    model.load_state_dict({k: v.to(device) if torch.is_tensor(v) else v
                           for k, v in warm_state.items()}, strict=False)
    for name, value in learned.items():
        getattr(model, name).fill_(value)
    isolated_calibration = evaluate(model, tensors, calibration_idx, device,
                                    cfg["eval_batch"], float(cfg["stop_weight"]))
    isolated_unseen = evaluate(model, tensors, unseen_idx, device,
                               cfg["eval_batch"], float(cfg["stop_weight"]))
    for name in learned:
        getattr(model, name).fill_(0.0 if name != "theta" else learned["theta"])
    parent_calibration = evaluate(model, tensors, calibration_idx, device,
                                  cfg["eval_batch"], float(cfg["stop_weight"]))
    parent_unseen = evaluate(model, tensors, unseen_idx, device,
                             cfg["eval_batch"], float(cfg["stop_weight"]))
    model.load_state_dict({k: v.to(device) if torch.is_tensor(v) else v
                           for k, v in current.items()}, strict=False)
    stop_weight = float(cfg["stop_weight"])

    def total(record):
        return record["next_bce"] + stop_weight * record["stop_bce"]

    return {
        "available": True,
        "learned_motif_values": learned,
        "isolated_calibration": isolated_calibration,
        "parent_calibration": parent_calibration,
        "isolated_model_unseen": isolated_unseen,
        "parent_model_unseen": parent_unseen,
        "calibration_gain": total(parent_calibration) - total(isolated_calibration),
        "model_unseen_gain": total(parent_unseen) - total(isolated_unseen),
        "model_unseen_contact_nll_gain": (parent_unseen["contact_nll"]
                                          - isolated_unseen["contact_nll"]),
    }


def _parameter_drift(model: MotifRNN, warm_state: dict | None) -> dict:
    if warm_state is None:
        return {"warm_started": False}
    drift = {}
    for name, parameter in model.named_parameters():
        if name in warm_state and warm_state[name].shape == parameter.shape:
            reference = warm_state[name].to(parameter.device)
            drift[name] = {
                "l2": float(torch.linalg.vector_norm(parameter.detach() - reference)),
                "reference_l2": float(torch.linalg.vector_norm(reference)),
            }
    return {"warm_started": True, "per_parameter": drift,
            "total_l2": float(np.sqrt(sum(v["l2"] ** 2 for v in drift.values())))}


def _parameter_hashes(model: MotifRNN) -> dict[str, str]:
    out = {}
    for name, parameter in model.named_parameters():
        array = parameter.detach().cpu().numpy()
        out[name] = hashlib.sha256(np.ascontiguousarray(array)).hexdigest()[:16]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS))
    parser.add_argument("--seed-index", type=int, required=True)
    parser.add_argument("--out-root", type=Path,
                        default=ROOT / "results/topic5_dynamical_motif_rnn_v0_1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gate-rule", default="M2-2RANK")
    parser.add_argument("--warm-start", type=Path, default=None)
    parser.add_argument("--freeze-shared", action="store_true")
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--selection-metric", choices=SELECTION_METRICS, default="joint")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--lambda-anchor", type=float, default=None)
    parser.add_argument("--shared-lr-ratio", type=float, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    for key, value in (("max_epochs", args.max_epochs), ("lambda_anchor", args.lambda_anchor),
                       ("shared_lr_ratio", args.shared_lr_ratio)):
        if value is not None:
            cfg[key] = value
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    warm = args.warm_start
    if warm is None and WARM_START_PARENT[args.model] is not None:
        warm = (args.out_root / args.tag / args.frame / args.unit_id
                / WARM_START_PARENT[args.model] / f"seed{args.seed_index}" / "checkpoint.pt")
    unit_dir = args.out_root / args.tag / args.frame / args.unit_id / args.model / f"seed{args.seed_index}"
    try:
        metrics = train_unit(
            args.frame, args.unit_id, args.model, args.seed_index, args.out_root, device, cfg,
            gate_rule=args.gate_rule, warm_start=warm, freeze_shared=args.freeze_shared,
            resume=not args.no_resume, tag=args.tag,
            selection_metric=args.selection_metric,
        )
        print(json.dumps({k: metrics[k] for k in
                          ("unit_id", "model_id", "seed_index", "n_epochs", "best_epoch",
                           "best_validation_score", "seconds")}, default=_jsonable), flush=True)
    except Exception as error:  # noqa: BLE001 - a failed unit must be recorded, not silent
        unit_dir.mkdir(parents=True, exist_ok=True)
        write_json(unit_dir / "FAILED.json",
                   {"error": f"{type(error).__name__}: {error}",
                    "frame": args.frame, "unit_id": args.unit_id, "model": args.model,
                    "seed_index": args.seed_index})
        raise


if __name__ == "__main__":
    main()
