"""Training loop for S_P with atomic resume and fixed chronological selection."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import random
import resource
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from src.topic5_group_event_state.v033_training_lab.contact_grammar import (
    tensor_state_hash,
)
from src.topic5_group_event_state.v033_training_lab.paths import (
    atomic_write_json,
    atomic_write_torch,
    current_commit,
    payload_hash,
)
from src.topic5_group_event_state.v033_training_lab.sg_o2 import GrammarPairs
from src.topic5_group_event_state.v032_model.shift import block_circular_donor

from .contracts import (
    FORMAT_PREFIX,
    SEED_CONTRACT,
    ArchConfig,
    OptimizerConfig,
    TrainConfig,
)
from .data import SpatialData, sample_equal_anchor_pairs
from .model import SpatialStateModel, build_optimizer


def _weighted_training_baselines(data: SpatialData, pairs: GrammarPairs) -> dict[str, np.ndarray | float]:
    event = pairs.pair_event
    weight = np.asarray(pairs.pair_weight, dtype=np.float64)
    part = data.participation[event].astype(np.float64)
    freq = np.clip((weight[:, None] * part).sum(0), 0.01, 0.99)
    contact_logits = np.log(freq / (1.0 - freq))
    extent = float(np.sum(weight * data.positive_extent[event]))
    cont_p = float(np.clip(np.sum(weight * (data.group_count[event] > 1)), 0.01, 0.99))
    lag_mean = np.zeros(data.n_contacts, dtype=np.float64)
    lag_scale = np.ones(data.n_contacts, dtype=np.float64)
    for c in range(data.n_contacts):
        valid = data.lag_valid[event, c]
        if not np.any(valid):
            continue
        w = weight[valid]
        w = w / w.sum()
        x = data.relative_lag[event[valid], c]
        lag_mean[c] = float(np.sum(w * x))
        lag_scale[c] = max(float(np.sqrt(np.sum(w * (x - lag_mean[c]) ** 2))), 1e-4)
    return {
        "contact_logits": contact_logits.astype(np.float32),
        "log_extent": math.log(max(extent, 1e-3)),
        "continue_logit": math.log(cont_p / (1.0 - cont_p)),
        "lag_mean": lag_mean.astype(np.float32),
        "lag_scale": lag_scale.astype(np.float32),
    }


def set_training_baselines(
    model: SpatialStateModel, data: SpatialData, pairs: GrammarPairs | None = None,
) -> Mapping[str, Any]:
    baseline = _weighted_training_baselines(data, pairs or data.train_pairs)
    model.functional.set_training_baselines(
        contact_logits=torch.as_tensor(baseline["contact_logits"]),
        log_extent=float(baseline["log_extent"]),
        continue_logit=float(baseline["continue_logit"]),
        lag_mean=torch.as_tensor(baseline["lag_mean"]),
        lag_scale=torch.as_tensor(baseline["lag_scale"]),
    )
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else float(value)
        for key, value in baseline.items()
    }


def _to_device(data: SpatialData, device: torch.device) -> dict[str, Tensor]:
    return {
        "event_token": torch.as_tensor(data.event_token, dtype=torch.float32, device=device),
        "event_time": torch.as_tensor(data.event_time, dtype=torch.float64, device=device),
        "event_segment": torch.as_tensor(data.event_segment, dtype=torch.long, device=device),
        "anchor_time": torch.as_tensor(data.anchor_time, dtype=torch.float64, device=device),
        "last_event_pos": torch.as_tensor(data.last_event_pos, dtype=torch.long, device=device),
        "group_ids": torch.as_tensor(data.group_ids, dtype=torch.long, device=device),
        "group_count": torch.as_tensor(data.group_count, dtype=torch.long, device=device),
        "participation": torch.as_tensor(data.participation, dtype=torch.bool, device=device),
        "positive_extent": torch.as_tensor(data.positive_extent, dtype=torch.float32, device=device),
        "relative_lag": torch.as_tensor(data.relative_lag, dtype=torch.float32, device=device),
        "lag_valid": torch.as_tensor(data.lag_valid, dtype=torch.bool, device=device),
    }


def _states(model: SpatialStateModel, tensors: Mapping[str, Tensor], train_pairs: GrammarPairs) -> Tensor:
    train_rows = torch.as_tensor(
        np.unique(train_pairs.anchor_rows), dtype=torch.long,
        device=tensors["event_token"].device,
    )
    return model.trajectory(
        tensors["event_token"], tensors["event_time"], tensors["event_segment"],
        tensors["anchor_time"], tensors["last_event_pos"], train_rows,
    )


def _subset_anchor_pairs(pairs: GrammarPairs, local_rows: np.ndarray) -> GrammarPairs:
    """Keep selected local anchors and restore equal-anchor/equal-event weights."""

    local = np.asarray(local_rows, dtype=np.int64)
    if local.size == 0:
        raise ValueError("grammar-pair subset has no anchors")
    keep_anchor = np.zeros(pairs.anchor_rows.size, dtype=bool)
    keep_anchor[local] = True
    remap = np.full(pairs.anchor_rows.size, -1, dtype=np.int64)
    remap[local] = np.arange(local.size, dtype=np.int64)
    keep_pair = keep_anchor[pairs.pair_anchor]
    pair_anchor = remap[pairs.pair_anchor[keep_pair]]
    counts = np.bincount(pair_anchor, minlength=local.size).astype(np.float64)
    weight = 1.0 / (local.size * counts[pair_anchor])
    return GrammarPairs(
        anchor_rows=pairs.anchor_rows[local],
        pair_anchor=pair_anchor,
        pair_event=pairs.pair_event[keep_pair],
        pair_weight=weight,
    ).validate()


def chronological_train_fit_inner(
    data: SpatialData,
    *,
    inner_fraction: float = 0.20,
    embargo_seconds: float = 1800.0,
) -> tuple[GrammarPairs, GrammarPairs, dict[str, Any]]:
    """Split STATE_TRAIN itself; STATE_SELECTION remains report-only.

    The embargo keeps a fit anchor's 30-minute future block from reaching the
    first rolling-inner anchor.  This makes checkpoint selection independent
    of every target later reported from STATE_SELECTION.
    """

    pairs = data.train_pairs
    times = np.asarray(data.anchor_time[pairs.anchor_rows], dtype=np.float64)
    if times.size < 10 or np.any(np.diff(times) < 0):
        raise ValueError("STATE_TRAIN anchors are too few or not chronological")
    n_inner = max(2, int(math.ceil(times.size * float(inner_fraction))))
    report_start = float(data.anchor_time[data.selection_pairs.anchor_rows].min())
    inner_candidates = np.flatnonzero(times + float(embargo_seconds) < report_start)
    if inner_candidates.size < 2:
        raise ValueError("STATE_TRAIN has no inner targets fully embargoed from STATE_SELECTION")
    inner_local = inner_candidates[-min(n_inner, inner_candidates.size):]
    inner_start = float(times[inner_local[0]])
    fit_local = np.flatnonzero(times + float(embargo_seconds) < inner_start)
    if fit_local.size < 4 or inner_local.size < 2:
        raise ValueError("STATE_TRAIN cannot support fit/rolling-inner plus embargo")
    fit = _subset_anchor_pairs(pairs, fit_local)
    inner = _subset_anchor_pairs(pairs, inner_local)
    meta = {
        "fit_source": "early_STATE_TRAIN",
        "checkpoint_selection_source": "chronologically_later_STATE_TRAIN_inner",
        "reported_source": "STATE_SELECTION_full",
        "inner_fraction": float(inner_fraction),
        "embargo_seconds": float(embargo_seconds),
        "n_fit_anchors": int(fit.anchor_rows.size),
        "n_inner_anchors": int(inner.anchor_rows.size),
        "n_report_anchors": int(data.selection_pairs.anchor_rows.size),
        "fit_last_anchor_epoch": float(data.anchor_time[fit.anchor_rows].max()),
        "inner_first_anchor_epoch": inner_start,
        "inner_last_target_before_report": bool(
            float(data.anchor_time[inner.anchor_rows].max()) + float(embargo_seconds) < report_start
        ),
        "report_first_anchor_epoch": report_start,
    }
    return fit, inner, meta


def _loss(
    model: SpatialStateModel,
    data: SpatialData,
    tensors: Mapping[str, Tensor],
    states: Tensor,
    pairs: GrammarPairs,
    config: TrainConfig,
) -> tuple[Tensor, dict[str, float]]:
    device = states.device
    event = torch.as_tensor(pairs.pair_event, dtype=torch.long, device=device)
    anchor_rows = pairs.anchor_rows[pairs.pair_anchor]
    source = torch.as_tensor(anchor_rows, dtype=torch.long, device=device)
    weight = torch.as_tensor(pairs.pair_weight, dtype=torch.float64, device=device)
    state = states[source]
    aux = model.functional.losses(
        state,
        participation=tensors["participation"][event],
        positive_extent=tensors["positive_extent"][event],
        group_count=tensors["group_count"][event],
        relative_lag=tensors["relative_lag"][event],
        lag_valid=tensors["lag_valid"][event],
    )
    grammar = model.legacy_event_nll(
        tensors["group_ids"][event], tensors["group_count"][event], state
    ) if model.legacy is not None else aux["subset_nll"] + aux["continue_nll"]
    components = {
        "grammar": grammar,
        "subset": aux["subset_nll"],
        "continue": aux["continue_nll"],
        "extent": aux["extent_nll"],
        "lag": aux["lag_nll"],
    }
    combined = (
        float(config.grammar_weight) * grammar
        + float(config.extent_weight) * aux["extent_nll"]
        + float(config.lag_weight) * aux["lag_nll"]
    )
    total = torch.sum(combined.to(torch.float64) * weight)
    metrics = {
        name: float(torch.sum(value.to(torch.float64) * weight).detach().cpu())
        for name, value in components.items()
    }
    metrics["total"] = float(total.detach().cpu())
    return total, metrics


@torch.no_grad()
def evaluate(
    model: SpatialStateModel,
    data: SpatialData,
    tensors: Mapping[str, Tensor],
    pairs: GrammarPairs,
    config: TrainConfig,
    *,
    reference_pairs: GrammarPairs | None = None,
    state_override: Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    state = state_override
    if state is None:
        state = _states(model, tensors, reference_pairs or data.train_pairs)
    metrics = {name: 0.0 for name in ("grammar", "subset", "continue", "extent", "lag", "total")}
    batch = int(config.pair_batch_size)
    for lo in range(0, pairs.pair_event.size, batch):
        hi = min(lo + batch, pairs.pair_event.size)
        piece = GrammarPairs(
            anchor_rows=pairs.anchor_rows,
            pair_anchor=pairs.pair_anchor[lo:hi],
            pair_event=pairs.pair_event[lo:hi],
            pair_weight=pairs.pair_weight[lo:hi],
        )
        _value, got = _loss(model, data, tensors, state, piece, config)
        for name, value in got.items():
            metrics[name] += float(value)
    return metrics


def _fit_train_mean_adapter(
    model: SpatialStateModel,
    data: SpatialData,
    tensors: Mapping[str, Tensor],
    fit_pairs: GrammarPairs,
    inner_pairs: GrammarPairs,
    config: TrainConfig,
    *,
    learning_rate: float,
) -> dict[str, Any]:
    """Fit and freeze a no-state recalibration before recurrent-state fitting."""

    adapter = model.train_mean_adapter
    if adapter is None:
        return {"status": "NOT_APPLICABLE_SYNTHETIC", "selected_step": 0}
    adapter.set_trainable(True)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(learning_rate), weight_decay=0.0)
    dim = int(model.config.state_dim)
    # The baseline is shared across state seeds; only the recurrent model is a
    # stochastic repeat.  Keeping this seed fixed prevents baseline noise from
    # masquerading as state-seed heterogeneity.
    baseline_seed = 20260903
    rng = np.random.default_rng(baseline_seed)
    best = math.inf
    best_step = 0
    best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    history: list[dict[str, float | int]] = []
    stale = 0
    max_steps = (
        min(max(int(config.max_steps), 300), 900)
        if data.subject.startswith("synthetic") else 900
    )
    for step in range(1, max_steps + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        sampled = sample_equal_anchor_pairs(
            fit_pairs, rng=rng,
            n_anchors=min(config.anchors_per_step, fit_pairs.anchor_rows.size),
            events_per_anchor=config.events_per_anchor,
        )
        event = torch.as_tensor(sampled.pair_event, dtype=torch.long, device=tensors["event_token"].device)
        weight = torch.as_tensor(sampled.pair_weight, dtype=torch.float64, device=event.device)
        zero = torch.zeros((event.numel(), dim), dtype=torch.float32, device=event.device)
        loss = torch.sum(model.legacy_event_nll(
            tensors["group_ids"][event], tensors["group_count"][event], zero,
        ).to(torch.float64) * weight)
        loss.backward(); optimizer.step()
        if step % config.validate_every != 0 and step != max_steps:
            continue
        zero_all = torch.zeros(
            (data.anchor_time.size, dim), dtype=torch.float32, device=event.device,
        )
        inner = evaluate(
            model, data, tensors, inner_pairs, config,
            reference_pairs=fit_pairs, state_override=zero_all,
        )["grammar"]
        history.append({"step": step, "fit_grammar": float(loss.detach().cpu()), "inner_grammar": inner})
        if np.isfinite(inner) and inner < best - 1e-6:
            best = float(inner); best_step = step
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config.patience_checks:
            break
    adapter.load_state_dict(best_state)
    adapter.set_trainable(False)
    return {
        "status": "FITTED_AND_FROZEN",
        "fit_source": "early_STATE_TRAIN_only",
        "selection_source": "late_STATE_TRAIN_inner_only",
        "selected_step": int(best_step),
        "steps_run": int(history[-1]["step"] if history else 0),
        "best_inner_grammar": float(best),
        "learning_rate": float(learning_rate),
        "seed": baseline_seed,
        "history": history,
        "contact_bias_l2": float(adapter.contact_bias.detach().norm().cpu()),
        "stop_bias": float(adapter.stop_bias.detach().cpu()),
        "best_at_budget_boundary": bool(history and best_step == history[-1]["step"]),
    }


def _anchor_carry(data: SpatialData) -> np.ndarray:
    last = data.last_event_pos
    return np.where(last >= 0, data.event_segment[np.maximum(last, 0)], -1)


def _rolling_prefix_state(
    states: Tensor, data: SpatialData, reference_pairs: GrammarPairs, report_pairs: GrammarPairs,
) -> Tensor:
    """Causal expanding mean of inferred states, initialised from TRAIN history."""

    out = torch.zeros_like(states)
    carry = _anchor_carry(data)
    ref = set(int(v) for v in reference_pairs.anchor_rows.tolist())
    report = set(int(v) for v in report_pairs.anchor_rows.tolist())
    for seg in np.unique(carry):
        rows = [int(v) for v in np.flatnonzero(carry == seg)]
        history = [states[row] for row in rows if row in ref]
        running = torch.stack(history).sum(0) if history else torch.zeros(states.shape[1], device=states.device)
        count = len(history)
        for row in rows:
            if row not in report:
                continue
            # The anchor state uses only events strictly before its target block,
            # so including it in the running level remains causal.
            running = running + states[row]
            count += 1
            out[row] = running / max(count, 1)
    return out


def _trainable_state(model: SpatialStateModel) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("legacy.decoder.")
    }


def _load_trainable_state(model: SpatialStateModel, saved: Mapping[str, Tensor]) -> None:
    state = model.state_dict()
    for name, value in saved.items():
        if name not in state or name.startswith("legacy.decoder."):
            raise ValueError("resume/best state has unexpected tensor")
        state[name] = value
    model.load_state_dict(state)


def train_spatial_state(
    model: SpatialStateModel,
    data: SpatialData,
    *,
    arch: ArchConfig,
    optimizer_config: OptimizerConfig,
    train_config: TrainConfig,
    device: torch.device,
    output_dir: Path,
    card_kind: str,
    allow_tiny: bool = False,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train one cell. Selection is chronological and output is resumable."""

    arch.validate(); optimizer_config.validate(); train_config.validate(allow_tiny=allow_tiny)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    card_path = output_dir / "training_card.json"
    resume_path = output_dir / "resume.pt"
    if card_path.exists() and not overwrite:
        return __import__("json").loads(card_path.read_text(encoding="utf-8"))
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device)
    # Chunking is numerical/TBPTT plumbing only: carry is retained and not
    # detached.  Bind the requested physical-time chunk explicitly so the
    # effective run cannot drift from its manifest.
    model.state_bank.chunk_seconds = float(train_config.chunk_seconds)
    model.state_bank.detach_chunks = False
    target_horizon = float(data.provenance.get(
        "future_horizon_seconds", 300.0 if data.subject.startswith("synthetic") else 1800.0,
    ))
    fit_pairs, inner_pairs, split_contract = chronological_train_fit_inner(
        data, embargo_seconds=target_horizon,
    )
    baseline = set_training_baselines(model, data, fit_pairs)
    tensors = _to_device(data, device)
    train_mean = _fit_train_mean_adapter(
        model, data, tensors, fit_pairs, inner_pairs, train_config,
        learning_rate=3e-3,
    )
    optimizer, opt_contract = build_optimizer(model, optimizer_config)
    contract_body = {
        "subject": data.subject,
        "card_kind": card_kind,
        "arch": asdict(arch),
        "optimizer": opt_contract,
        "train": asdict(train_config),
        "seed_contract": SEED_CONTRACT,
        "encoder_role": (
            "trainable_learned" if any(p.requires_grad for p in model.encoder.parameters())
            else "frozen_random_reservoir_control"
        ),
        "model_selection": split_contract,
        "train_mean_adapter": {
            "role": "no_state_recalibration",
            "fit": "early_STATE_TRAIN",
            "selection": "late_STATE_TRAIN_inner",
            "frozen_during_state_training": True,
        },
        "input_provenance": dict(data.provenance),
        "scoring": "legacy_next_set_or_STOP_plus_positive_extent_plus_masked_relative_lag"
            if model.legacy is not None else "synthetic_conditional_subset_continue_extent_lag",
    }
    contract_hash = payload_hash(contract_body)
    # STATE_SELECTION is a full, report-only period.  All checkpoint decisions
    # are made on the rolling inner split carved from STATE_TRAIN above.
    initial = evaluate(
        model, data, tensors, inner_pairs, train_config, reference_pairs=fit_pairs,
    )
    initial_hash = tensor_state_hash(_trainable_state(model))
    best = float(initial["total"])
    best_step = 0
    best_state = _trainable_state(model)
    history: list[dict[str, Any]] = []
    stale = 0
    start = 1
    max_grad = 0.0
    if resume:
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved.get("contract_hash") != contract_hash:
            raise PermissionError("resume contract differs")
        _load_trainable_state(model, saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        best = float(saved["best"]); best_step = int(saved["best_step"])
        best_state = dict(saved["best_state"]); history = list(saved["history"])
        stale = int(saved["stale"]); max_grad = float(saved["max_grad"])
        start = int(saved["last_step"]) + 1
    started = time.monotonic()
    rng = np.random.default_rng(train_config.seed)
    for step in range(start, train_config.max_steps + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        pairs = sample_equal_anchor_pairs(
            fit_pairs,
            rng=rng,
            n_anchors=train_config.anchors_per_step,
            events_per_anchor=train_config.events_per_anchor,
        )
        state = _states(model, tensors, fit_pairs)
        loss, fit = _loss(model, data, tensors, state, pairs, train_config)
        loss.backward()
        params = [p for p in model.parameters() if p.requires_grad]
        grad = float(torch.sqrt(sum(
            torch.sum(p.grad.detach().float().square()) for p in params if p.grad is not None
        )).cpu())
        max_grad = max(max_grad, grad)
        torch.nn.utils.clip_grad_norm_(params, optimizer_config.gradient_clip)
        optimizer.step()
        if step % train_config.validate_every != 0 and step != train_config.max_steps:
            continue
        selected = evaluate(
            model, data, tensors, inner_pairs, train_config, reference_pairs=fit_pairs,
        )
        history.append({"step": step, "fit": fit, "selection": selected, "gradient_l2": grad})
        if np.isfinite(selected["total"]) and selected["total"] < best - 1e-6:
            best = float(selected["total"]); best_step = step
            best_state = _trainable_state(model); stale = 0
        else:
            stale += 1
        atomic_write_torch(resume_path, {
            "format": f"{FORMAT_PREFIX}_resume_v1",
            "contract_hash": contract_hash,
            "last_step": step,
            "model_state": _trainable_state(model),
            "optimizer_state": optimizer.state_dict(),
            "best": best,
            "best_step": best_step,
            "best_state": best_state,
            "history": history,
            "stale": stale,
            "max_grad": max_grad,
        })
        if stale >= train_config.patience_checks:
            break
    last_state_hash = tensor_state_hash(_trainable_state(model))
    optimizer_parameters_changed = last_state_hash != initial_hash
    _load_trainable_state(model, best_state)
    selected = evaluate(
        model, data, tensors, inner_pairs, train_config, reference_pairs=fit_pairs,
    )
    report_pairs = data.selection_pairs
    report_states = _states(model, tensors, fit_pairs)
    zero_states = torch.zeros_like(report_states)
    report_zero = evaluate(
        model, data, tensors, report_pairs, train_config,
        reference_pairs=fit_pairs, state_override=zero_states,
    )
    report_learned = evaluate(
        model, data, tensors, report_pairs, train_config,
        reference_pairs=fit_pairs, state_override=report_states,
    )
    report_rows = torch.as_tensor(report_pairs.anchor_rows, dtype=torch.long, device=device)
    period_states = torch.zeros_like(report_states)
    period_states[report_rows] = report_states[report_rows].mean(0, keepdim=True)
    report_period = evaluate(
        model, data, tensors, report_pairs, train_config,
        reference_pairs=fit_pairs, state_override=period_states,
    )
    rolling_states = _rolling_prefix_state(report_states, data, fit_pairs, report_pairs)
    report_rolling = evaluate(
        model, data, tensors, report_pairs, train_config,
        reference_pairs=fit_pairs, state_override=rolling_states,
    )
    carry = _anchor_carry(data)
    donor = block_circular_donor(
        data.anchor_time, carry, report_pairs.anchor_rows,
        horizon=target_horizon, fraction=0.5,
    )
    valid_donor = donor >= 0
    shifted_states = report_states.clone()
    if np.any(valid_donor):
        target_rows = report_pairs.anchor_rows[valid_donor]
        donor_rows = report_pairs.anchor_rows[donor[valid_donor]]
        shifted_states[torch.as_tensor(target_rows, dtype=torch.long, device=device)] = \
            report_states[torch.as_tensor(donor_rows, dtype=torch.long, device=device)]
        valid_local = np.flatnonzero(valid_donor)
        shifted_pairs = _subset_anchor_pairs(report_pairs, valid_local)
        report_shifted = evaluate(
            model, data, tensors, shifted_pairs, train_config,
            reference_pairs=fit_pairs, state_override=shifted_states,
        )
        report_learned_shiftable = evaluate(
            model, data, tensors, shifted_pairs, train_config,
            reference_pairs=fit_pairs, state_override=report_states,
        )
    else:
        report_shifted = {name: math.nan for name in report_learned}
        report_learned_shiftable = {name: math.nan for name in report_learned}
    final_hash = tensor_state_hash(_trainable_state(model))
    selected_parameters_changed = final_hash != initial_hash
    finite = np.isfinite([row["selection"]["total"] for row in history]).all() if history else False
    status = "PASS" if optimizer_parameters_changed and finite and max_grad > 0 else "FAIL"
    card = {
        "format": f"{FORMAT_PREFIX}_{card_kind}_card_v1",
        "status": status,
        "contract_hash": contract_hash,
        "seed_contract": SEED_CONTRACT,
        "contract": contract_body,
        "baseline_parameters": baseline,
        "train_mean_adapter": train_mean,
        "model_selection_split": split_contract,
        "initial_inner": initial,
        "selected_inner": selected,
        # Compatibility aliases now explicitly refer to TRAIN-inner, not the
        # later STATE_SELECTION reporting period.
        "initial_selection": initial,
        "selected_selection": selected,
        "state_selection_full": {
            "train_mean_no_state": report_zero,
            "learned_correct_time": report_learned,
            "selection_period_mean_noncausal_oracle": report_period,
            "rolling_prefix_level_causal": report_rolling,
            "block_circular_wrong_time": report_shifted,
            "learned_correct_time_on_wrong_time_support": report_learned_shiftable,
            "n_valid_wrong_time_anchors": int(valid_donor.sum()),
            "n_report_anchors": int(report_pairs.anchor_rows.size),
        },
        "selection_gain": float(report_zero["total"] - report_learned["total"]),
        "inner_gain": float(initial["total"] - selected["total"]),
        "period_level_gain": float(report_zero["total"] - report_period["total"]),
        "beyond_period_gain": float(report_period["total"] - report_learned["total"]),
        "rolling_level_gain": float(report_zero["total"] - report_rolling["total"]),
        "wrong_time_cost": float(
            report_shifted["total"] - report_learned_shiftable["total"]
        ),
        "selected_step": best_step,
        "steps_run": history[-1]["step"] if history else 0,
        "history": history,
        "max_gradient_l2": max_grad,
        "parameters_changed": optimizer_parameters_changed,
        "selected_parameters_changed": selected_parameters_changed,
        "selection_is_report_only": True,
        "first_validation_best": bool(best_step == train_config.validate_every),
        "best_at_last_validation": bool(history and best_step == history[-1]["step"]),
        "initial_state_hash": initial_hash,
        "last_state_hash": last_state_hash,
        "selected_state_hash": final_hash,
        "resources": {
            "wall_seconds": time.monotonic() - started,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "implementation_commit": current_commit(),
        "development_targets_read": False,
        "sealed_partition_opened": False,
        "seizure_outcomes_read": False,
    }
    atomic_write_json(card_path, card)
    atomic_write_torch(output_dir / "selected_checkpoint.pt", {
        "format": f"{FORMAT_PREFIX}_checkpoint_v1",
        "contract_hash": contract_hash,
        "state_dict": best_state,
        "selected_step": best_step,
    })
    return card
