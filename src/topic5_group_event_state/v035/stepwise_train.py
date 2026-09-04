"""Train nested static and dynamic step-wise adapters on a frozen decoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from src.topic5_group_event_state.v032_model.shift import block_circular_donor
from src.topic5_group_event_state.v033_training_lab.paths import atomic_write_npz
from src.topic5_group_event_state.v033_training_lab.sg_o2 import GrammarPairs
from src.topic5_group_event_state.v034_spatial_state.data import SpatialData, sample_equal_anchor_pairs
from src.topic5_group_event_state.v034_spatial_state.we_decoder import (
    FrozenDecoderBundle, align_events, decoder_tensors, restrict_pairs, split_pairs_by_time,
)
from src.topic5_group_event_state.v034_spatial_state.we_state import per_anchor, weighted

from .contracts import FORMAT_PREFIX, atomic_json, seed_all
from .stepwise_decoder import StepwiseAdapterConfig, StepwiseConditionedDecoder


@dataclass(frozen=True)
class StepwiseTrainConfig:
    rank: int = 8
    max_steps_static: int = 900
    max_steps_dynamic: int = 1800
    validate_every: int = 25
    patience_checks: int = 10
    anchors_per_step: int = 128
    events_per_anchor: int = 16
    inner_val_fraction: float = 0.15
    lr_static: float = 3e-3
    lr_dynamic: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 2.0
    seed: int = 20260903


@dataclass
class PreparedStepwise:
    data: SpatialData
    bundle: FrozenDecoderBundle
    dec_tensors: dict[str, Tensor]
    cache_index: np.ndarray
    fit_pairs: GrammarPairs
    inner_pairs: GrammarPairs
    selection_pairs: GrammarPairs
    context: Tensor


def _map_context(data: SpatialData, trajectory_path: Path) -> np.ndarray:
    with np.load(trajectory_path, allow_pickle=False) as z:
        time_ = np.asarray(z["anchor_time"], dtype=np.float64)
        q = np.asarray(z["q_standardized"], dtype=np.float32)
    pos = np.searchsorted(time_, data.anchor_time)
    ok = (pos < time_.size) & (np.abs(time_[np.minimum(pos, time_.size - 1)] - data.anchor_time) < 1e-3)
    out = np.zeros((data.anchor_time.size, q.shape[1]), dtype=np.float32)
    out[ok] = q[pos[ok]]
    # Spatial anchors that survived its own burn-in should all exist on the
    # same five-minute rate grid.  Failing here is safer than zero-filling a
    # scientific arm.
    required = np.unique(np.concatenate((data.train_pairs.anchor_rows, data.selection_pairs.anchor_rows)))
    if not np.all(ok[required]):
        raise ValueError(f"dynamic q trajectory misses {int((~ok[required]).sum())} required contact anchors")
    return out


def prepare_stepwise(data: SpatialData, bundle: FrozenDecoderBundle, trajectory_path: Path, device: torch.device,
                     config: StepwiseTrainConfig) -> PreparedStepwise:
    cache_index = align_events(data.event_time, bundle.event_abs_time)
    keep = cache_index >= 0
    train_pairs = restrict_pairs(data.train_pairs, keep)
    selection_pairs = restrict_pairs(data.selection_pairs, keep)
    fit, inner = split_pairs_by_time(train_pairs, data.anchor_time, config.inner_val_fraction)
    context = torch.as_tensor(_map_context(data, trajectory_path), dtype=torch.float32, device=device)
    return PreparedStepwise(data=data, bundle=bundle, dec_tensors=decoder_tensors(bundle, device),
                            cache_index=cache_index, fit_pairs=fit, inner_pairs=inner,
                            selection_pairs=selection_pairs, context=context)


def pair_scores(model: StepwiseConditionedDecoder, prep: PreparedStepwise, pairs: GrammarPairs,
                context: Tensor | None, *, use_static: bool, use_dynamic: bool,
                grad: bool = False, batch_size: int = 1024) -> dict[str, Tensor]:
    device = prep.context.device
    cache = torch.as_tensor(prep.cache_index[pairs.pair_event], dtype=torch.long, device=device)
    anchor = torch.as_tensor(pairs.anchor_rows[pairs.pair_anchor], dtype=torch.long, device=device)
    out: dict[str, list[Tensor]] = {"grammar": [], "contact_nll": [], "stop_bce": [], "next_bce": []}
    for lo in range(0, cache.numel(), batch_size):
        rows = cache[lo:lo + batch_size]
        batch = {k: v[rows] for k, v in prep.dec_tensors.items()}
        c = None if context is None else context[anchor[lo:lo + batch_size]]
        if grad:
            score = model.scores(batch, c, use_static=use_static, use_dynamic=use_dynamic)
        else:
            with torch.no_grad():
                score = model.scores(batch, c, use_static=use_static, use_dynamic=use_dynamic)
        for key in out:
            out[key].append(score[key])
    return {key: torch.cat(value) for key, value in out.items()}


def _fit(
    model: StepwiseConditionedDecoder, prep: PreparedStepwise, config: StepwiseTrainConfig,
    *, stage: str, params: list[torch.nn.Parameter], lr: float,
) -> dict[str, Any]:
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    use_dynamic = stage == "dynamic"
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed + (11 if stage == "static" else 22))

    def evaluate(pairs: GrammarPairs) -> float:
        model.eval()
        score = pair_scores(model, prep, pairs, prep.context if use_dynamic else None,
                            use_static=True, use_dynamic=use_dynamic)
        return float(weighted(score["grammar"], pairs))

    best = evaluate(prep.inner_pairs)
    best_step, stale = 0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith("decoder.")}
    history = [{"step": 0, "inner_nll": best}]
    max_steps = config.max_steps_static if stage == "static" else config.max_steps_dynamic
    for step in range(1, max_steps + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        pairs = sample_equal_anchor_pairs(prep.fit_pairs, rng=rng, n_anchors=config.anchors_per_step,
                                          events_per_anchor=config.events_per_anchor)
        score = pair_scores(model, prep, pairs, prep.context if use_dynamic else None,
                            use_static=True, use_dynamic=use_dynamic, grad=True)
        loss = weighted(score["grammar"], pairs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, config.gradient_clip)
        optimizer.step()
        if step % config.validate_every == 0 or step == max_steps:
            value = evaluate(prep.inner_pairs)
            history.append({"step": step, "inner_nll": value, "fit_nll": float(loss.detach().cpu())})
            if math.isfinite(value) and value < best - 1e-6:
                best, best_step, stale = value, step, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith("decoder.")}
            else:
                stale += 1
            if stale >= config.patience_checks:
                break
    current = model.state_dict()
    for key, value in best_state.items():
        current[key] = value.to(current[key].device)
    model.load_state_dict(current)
    for p in model.parameters():
        p.requires_grad_(False)
    return {"stage": stage, "selected_step": best_step, "steps_run": history[-1]["step"],
            "selected_at_init": best_step == 0, "selected_at_budget_edge": best_step == max_steps,
            "best_inner_nll": best, "history": history}


def _shift_context(prep: PreparedStepwise, horizon: float = 7200.0) -> tuple[Tensor, np.ndarray]:
    sel = prep.selection_pairs
    last = prep.data.last_event_pos
    carry = np.where(last >= 0, prep.data.event_segment[np.maximum(last, 0)], -1)
    donor = block_circular_donor(prep.data.anchor_time, carry, sel.anchor_rows, horizon=horizon, fraction=0.5)
    shifted = prep.context.clone()
    ok = donor >= 0
    rows = torch.as_tensor(sel.anchor_rows, dtype=torch.long, device=prep.context.device)
    if ok.any():
        shifted[rows[torch.as_tensor(ok, device=rows.device)]] = prep.context[rows[torch.as_tensor(donor[ok], device=rows.device)]]
    return shifted, ok


def _evaluate(model: StepwiseConditionedDecoder, prep: PreparedStepwise) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    sel = prep.selection_pairs
    shifted, ok = _shift_context(prep)
    specs = {
        "frozen": (None, False, False),
        "static": (None, True, False),
        "rate_dynamic": (prep.context, True, True),
        "block_shift": (shifted, True, True),
    }
    means, arrays = {}, {}
    score_keys: tuple[str, ...] = ()
    for name, (context, use_static, use_dynamic) in specs.items():
        score = pair_scores(model, prep, sel, context, use_static=use_static, use_dynamic=use_dynamic)
        means[name] = {key: float(weighted(value, sel)) for key, value in score.items()}
        score_keys = tuple(score)
        for key, value in score.items():
            a = per_anchor(value.detach().cpu().numpy(), sel)
            if name == "block_shift":
                a[~ok] = np.nan
            arrays[f"{name}_{key}"] = a
    arrays["anchor_time"] = prep.data.anchor_time[sel.anchor_rows]
    arrays["anchor_segment"] = prep.data.event_segment[np.maximum(prep.data.last_event_pos[sel.anchor_rows], 0)]
    arrays["shift_valid"] = ok.astype(np.uint8)
    # Anchors without a distant donor keep their correct-time context in the
    # shifted tensor, so a pair-weighted mean over all anchors dilutes the null
    # with correct-time scores.  Report the null and its like-for-like
    # correct-time companion on donor-valid anchors only (review 2026-09-04).
    ok_rows = np.flatnonzero(ok)
    # Keys come from the scorer itself; hard-coding them silently breaks when
    # ``pair_scores`` reports a different endpoint set.
    keys = score_keys
    means["block_shift"] = {
        key: (float(np.nanmean(arrays[f"block_shift_{key}"][ok_rows])) if ok_rows.size else None) for key in keys
    }
    means["block_shift"]["n_anchors"] = int(ok_rows.size)
    means["rate_dynamic_on_shift_support"] = {
        key: (float(np.mean(arrays[f"rate_dynamic_{key}"][ok_rows])) if ok_rows.size else None) for key in keys
    }
    means["rate_dynamic_on_shift_support"]["n_anchors"] = int(ok_rows.size)
    return means, arrays


def run_stepwise_subject(
    data: SpatialData, bundle: FrozenDecoderBundle, trajectory_path: Path, config: StepwiseTrainConfig,
    *, device: torch.device, out_dir: Path, overwrite: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    card_path = out_dir / "card.json"
    if card_path.exists() and not overwrite:
        return json.loads(card_path.read_text(encoding="utf-8"))
    started = time.time(); seed_all(config.seed)
    prep = prepare_stepwise(data, bundle, trajectory_path, device, config)
    model = StepwiseConditionedDecoder(
        bundle.model, StepwiseAdapterConfig(context_dim=prep.context.shape[1], rank=config.rank),
    ).to(device)
    # Exact construction parity against the unmodified decoder.
    rows = torch.as_tensor(prep.cache_index[prep.fit_pairs.pair_event[:16]], dtype=torch.long, device=device)
    batch = {k: v[rows] for k, v in prep.dec_tensors.items()}
    with torch.no_grad():
        reference = bundle.model(batch["x"], batch["recruited"], batch["valid"])
        got = model(batch["x"], batch["recruited"], batch["valid"], prep.context[:rows.numel()],
                    use_static=True, use_dynamic=True)
        parity = max(float((a - b).abs().max().cpu()) for a, b in zip(reference, got))
    if parity != 0.0:
        raise RuntimeError(f"stepwise zero-adapter parity failed: {parity}")
    stages = {
        "static": _fit(model, prep, config, stage="static", params=list(model.static.parameters()), lr=config.lr_static),
    }
    stages["dynamic"] = _fit(model, prep, config, stage="dynamic", params=list(model.dynamic.parameters()), lr=config.lr_dynamic)
    means, arrays = _evaluate(model, prep)
    npz = out_dir / "per_anchor_scores.npz"; atomic_write_npz(npz, arrays)
    torch.save({"adapter": {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("decoder.")},
                "config": asdict(config), "rate_trajectory": str(trajectory_path)}, out_dir / "adapter.pt")
    card = {
        "format": f"{FORMAT_PREFIX}_stepwise_decoder_card_v1", "subject": data.subject,
        "decoder_unit": str(bundle.unit_dir), "rate_trajectory": str(trajectory_path),
        "config": asdict(config), "zero_adapter_parity_max_abs": parity, "stages": stages,
        "selection_means": means, "per_anchor_path": str(npz),
        "coverage": {"fit_anchors": int(prep.fit_pairs.anchor_rows.size),
                     "inner_anchors": int(prep.inner_pairs.anchor_rows.size),
                     "selection_anchors": int(prep.selection_pairs.anchor_rows.size)},
        "elapsed_seconds": time.time() - started, "development_targets_read": False,
        "sealed_partition_opened": False, "seizure_outcomes_read": False,
    }
    atomic_json(card_path, card)
    return card


def _oracle_context_for_pairs(prep: PreparedStepwise, pairs: tuple[GrammarPairs, ...]) -> Tensor:
    """Intentional future-participation leak used only as an assay ceiling."""
    n_anchor = prep.data.anchor_time.size
    n_contacts = prep.bundle.model.n_contacts
    total = np.zeros((n_anchor, n_contacts), dtype=np.float32)
    count = np.zeros(n_anchor, dtype=np.float32)
    for group in pairs:
        for local, anchor_row in enumerate(group.anchor_rows):
            event_rows = group.pair_event[group.pair_anchor == local]
            cache = prep.cache_index[event_rows]
            cache = cache[cache >= 0]
            if cache.size:
                total[int(anchor_row)] = (prep.bundle.ranks[cache] >= 0).mean(axis=0)
                count[int(anchor_row)] = 1.0
    total[count == 0] = 0.0
    return torch.as_tensor(total, dtype=torch.float32, device=prep.context.device)


def run_stepwise_future_oracle(
    data: SpatialData, bundle: FrozenDecoderBundle, trajectory_path: Path, base_adapter: Path,
    config: StepwiseTrainConfig, *, device: torch.device, out_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Verify that the new every-step interface detects an answer-leaking state."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    card_path = out_dir / "card.json"
    if card_path.exists() and not overwrite:
        return json.loads(card_path.read_text(encoding="utf-8"))
    seed_all(config.seed)
    prep = prepare_stepwise(data, bundle, trajectory_path, device, config)
    oracle = _oracle_context_for_pairs(prep, (prep.fit_pairs, prep.inner_pairs, prep.selection_pairs))
    prep = replace(prep, context=oracle)
    model = StepwiseConditionedDecoder(
        bundle.model, StepwiseAdapterConfig(context_dim=bundle.model.n_contacts, rank=config.rank),
    ).to(device)
    saved = torch.load(base_adapter, map_location="cpu", weights_only=False)["adapter"]
    current = model.state_dict()
    for key, value in saved.items():
        if key.startswith("static.") and key in current and current[key].shape == value.shape:
            current[key] = value
    model.load_state_dict(current)
    stage = _fit(model, prep, config, stage="dynamic", params=list(model.dynamic.parameters()),
                 lr=config.lr_dynamic)
    means, arrays = _evaluate(model, prep)
    npz = out_dir / "per_anchor_scores.npz"; atomic_write_npz(npz, arrays)
    card = {
        "format": f"{FORMAT_PREFIX}_stepwise_future_oracle_card_v1", "subject": data.subject,
        "seed": config.seed, "base_adapter": str(base_adapter), "stage": stage,
        "selection_means": {
            "static": means["static"], "future_oracle": means["rate_dynamic"],
            "oracle_block_shift": means["block_shift"],
        },
        "oracle_semantics": "intentional leakage: mean participation of the exact future events paired to each anchor",
        "state_entry": "per-step hidden FiLM plus contact-specific and STOP shifts",
        "scientific_claim_allowed": False, "assay_sensitivity_only": True,
        "development_targets_read": False, "sealed_partition_opened": False,
        "seizure_outcomes_read": False,
    }
    atomic_json(card_path, card); return card
