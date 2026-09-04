"""Cross-event S_P state feeding the frozen tissue decoder's initial state (nested arms + trainer).

Ladder scored on the same per-event decoder objective (see ``we_decoder``):

    zero        frozen decoder, h0 = 0                       (no adapter, no state)
    adapter     h0 = b, b fitted on TRAIN without any state  (state-free recalibration)
    learned     h0 = b + A(s_t), encoder + A trained, b frozen from the adapter stage
    period_mean learned model, s replaced by its mean over the scored period (input only)
    shifted     learned model, same-carry-segment block-circular wrong-time s
    random      encoder frozen at its random initialisation, only A trained

Selection of every trained stage uses a *rolling inner-validation* tail of the
TRAIN anchors (chronologically last ``inner_val_fraction``); STATE_SELECTION is
scored, never selected on.  Nothing here reads development targets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from src.topic5_group_event_state.v032_model.shift import block_circular_donor
from src.topic5_group_event_state.v032_model.state import MarkedLeakyBank
from src.topic5_group_event_state.v033_training_lab.paths import atomic_write_json, atomic_write_npz, atomic_write_torch, current_commit
from src.topic5_group_event_state.v033_training_lab.sg_o2 import GrammarPairs

from .contracts import ArchConfig, seed_before_model_construction
from .data import SpatialData, sample_equal_anchor_pairs
from .model import SpatialEventEncoder
from .trainer import _to_device
from .we_decoder import (
    FrozenDecoderBundle, WEStateScorer, align_events, decoder_tensors, restrict_pairs, split_pairs_by_time,
)

FORMAT = "group_event_state_v0_3_4_we_state_card_v1"


@dataclass(frozen=True)
class WEStateConfig:
    width: int = 64
    depth: int = 4
    write_width: int = 4
    adapter_rank: int = 4
    taus_seconds: tuple[float, ...] = (300.0, 1800.0, 7200.0)
    residual: bool = True
    lr_encoder: float = 3e-4
    lr_h0: float = 3e-3
    lr_bias: float = 3e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    max_steps: int = 900
    validate_every: int = 25
    patience_checks: int = 8
    anchors_per_step: int = 128
    events_per_anchor: int = 16
    inner_val_fraction: float = 0.15
    burn_in_seconds: float = 1800.0
    chunk_seconds: float = 3600.0
    stop_weight: float = 1.0
    seed: int = 20260903

    def arch(self) -> ArchConfig:
        return ArchConfig(width=self.width, depth=self.depth, write_width=self.write_width,
                          adapter_rank=self.adapter_rank, residual=self.residual, taus_seconds=self.taus_seconds)


class WESpatialStateModel(nn.Module):
    def __init__(self, *, input_dim: int, bundle: FrozenDecoderBundle, config: WEStateConfig) -> None:
        super().__init__()
        arch = config.arch().validate()
        self.encoder = SpatialEventEncoder(input_dim, arch)
        self.state_bank = MarkedLeakyBank(arch.taus_seconds, arch.write_width, chunk_seconds=config.chunk_seconds,
                                          detach_chunks=False)
        self.scorer = WEStateScorer(bundle, state_dim=arch.state_dim, rank=arch.adapter_rank,
                                    stop_weight=config.stop_weight)

    def trajectory(self, t: Mapping[str, Tensor], train_rows: Tensor) -> Tensor:
        write = self.encoder(t["event_token"])
        _pre, post = self.state_bank(write, t["event_time"], t["event_segment"])
        anchor = self.state_bank.anchor(post, t["event_time"], t["anchor_time"], t["last_event_pos"])
        reference = anchor[train_rows]
        centre = reference.mean(0)
        scale = reference.std(0, unbiased=False).clamp_min(1e-4)
        return (anchor - centre) / scale


@dataclass
class Prepared:
    data: SpatialData
    bundle: FrozenDecoderBundle
    tensors: dict[str, Tensor]
    dec_tensors: dict[str, Tensor]
    cache_index: np.ndarray
    fit_pairs: GrammarPairs
    inner_pairs: GrammarPairs
    train_pairs: GrammarPairs
    selection_pairs: GrammarPairs
    train_rows: Tensor
    coverage: dict[str, Any] = field(default_factory=dict)


def prepare(data: SpatialData, bundle: FrozenDecoderBundle, config: WEStateConfig, device: torch.device) -> Prepared:
    cache_index = align_events(data.event_time, bundle.event_abs_time)
    keep = cache_index >= 0
    train_pairs = restrict_pairs(data.train_pairs, keep)
    selection_pairs = restrict_pairs(data.selection_pairs, keep)
    fit_pairs, inner_pairs = split_pairs_by_time(train_pairs, data.anchor_time, config.inner_val_fraction)
    if float(data.anchor_time[fit_pairs.anchor_rows].max()) >= float(data.anchor_time[inner_pairs.anchor_rows].min()):
        raise RuntimeError("rolling inner-validation is not strictly after the fit anchors")
    tensors = _to_device(data, device)
    train_rows = torch.as_tensor(np.unique(data.train_pairs.anchor_rows), dtype=torch.long, device=device)
    coverage = {
        "n_events_total": int(data.event_time.size), "n_events_decoder_covered": int(keep.sum()),
        "coverage_fraction": float(keep.mean()),
        "train_pairs_before": int(data.train_pairs.pair_event.size), "train_pairs_after": int(train_pairs.pair_event.size),
        "selection_pairs_before": int(data.selection_pairs.pair_event.size), "selection_pairs_after": int(selection_pairs.pair_event.size),
        "fit_anchors": int(fit_pairs.anchor_rows.size), "inner_val_anchors": int(inner_pairs.anchor_rows.size),
        "selection_anchors": int(selection_pairs.anchor_rows.size),
    }
    return Prepared(data=data, bundle=bundle, tensors=tensors, dec_tensors=decoder_tensors(bundle, device),
                    cache_index=cache_index, fit_pairs=fit_pairs, inner_pairs=inner_pairs, train_pairs=train_pairs,
                    selection_pairs=selection_pairs, train_rows=train_rows, coverage=coverage)


def pair_scores(model: WESpatialStateModel, prep: Prepared, pairs: GrammarPairs, states: Tensor | None, *,
                use_bias: bool, use_state: bool, batch: int = 2048, grad: bool = False) -> dict[str, Tensor]:
    """Per-pair scores (grammar objective, contact NLL) for one arm."""

    device = prep.tensors["event_token"].device
    cache_rows = torch.as_tensor(prep.cache_index[pairs.pair_event], dtype=torch.long, device=device)
    anchor_rows = torch.as_tensor(pairs.anchor_rows[pairs.pair_anchor], dtype=torch.long, device=device)
    out: dict[str, list[Tensor]] = {"grammar": [], "contact_nll": []}
    for lo in range(0, cache_rows.numel(), batch):
        rows = cache_rows[lo:lo + batch]
        b = {k: v[rows] for k, v in prep.dec_tensors.items()}
        s = None if states is None else states[anchor_rows[lo:lo + batch]]
        if grad:
            sc = model.scorer.scores(b, s, use_bias=use_bias, use_state=use_state)
        else:
            with torch.no_grad():
                sc = model.scorer.scores(b, s, use_bias=use_bias, use_state=use_state)
        out["grammar"].append(sc["grammar"])
        out["contact_nll"].append(sc["contact_nll"])
    return {k: torch.cat(v) for k, v in out.items()}


def weighted(values: Tensor, pairs: GrammarPairs) -> Tensor:
    w = torch.as_tensor(pairs.pair_weight, dtype=torch.float64, device=values.device)
    return torch.sum(values.to(torch.float64) * w)


def per_anchor(values: np.ndarray, pairs: GrammarPairs) -> np.ndarray:
    n = int(pairs.anchor_rows.size)
    sums = np.bincount(pairs.pair_anchor, weights=values, minlength=n)
    counts = np.bincount(pairs.pair_anchor, minlength=n).astype(np.float64)
    return sums / np.maximum(counts, 1.0)


def _states(model: WESpatialStateModel, prep: Prepared) -> Tensor:
    return model.trajectory(prep.tensors, prep.train_rows)


def fit_stage(model: WESpatialStateModel, prep: Prepared, config: WEStateConfig, *, stage: str, use_bias: bool,
              use_state: bool, train_encoder: bool, train_h0: bool, train_bias: bool, seed: int) -> dict[str, Any]:
    """Train one stage; select the checkpoint on the rolling inner-validation tail only."""

    groups = []
    if train_encoder:
        groups.append({"name": "encoder", "params": list(model.encoder.parameters()), "lr": config.lr_encoder})
    if train_h0:
        groups.append({"name": "to_h0", "params": list(model.scorer.to_h0.parameters()), "lr": config.lr_h0})
    if train_bias:
        groups.append({"name": "h0_bias", "params": [model.scorer.h0_bias], "lr": config.lr_bias})
    trainable = [p for g in groups for p in g["params"]]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in trainable:
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    rng = np.random.default_rng(seed)

    def evaluate(pairs: GrammarPairs) -> float:
        model.eval()
        with torch.no_grad():
            states = _states(model, prep) if use_state else None
            sc = pair_scores(model, prep, pairs, states, use_bias=use_bias, use_state=use_state)
        return float(weighted(sc["grammar"], pairs))

    best = evaluate(prep.inner_pairs)
    best_step, stale = 0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith("scorer.decoder.")}
    history = [{"step": 0, "inner_val": best}]
    max_grad = 0.0
    for step in range(1, config.max_steps + 1):
        model.train(); model.scorer.decoder.eval()
        optimizer.zero_grad(set_to_none=True)
        pairs = sample_equal_anchor_pairs(prep.fit_pairs, rng=rng, n_anchors=config.anchors_per_step,
                                          events_per_anchor=config.events_per_anchor)
        states = _states(model, prep) if use_state else None
        sc = pair_scores(model, prep, pairs, states, use_bias=use_bias, use_state=use_state, grad=True)
        loss = weighted(sc["grammar"], pairs)
        loss.backward()
        g = float(torch.sqrt(sum(p.grad.detach().float().square().sum() for p in trainable if p.grad is not None)).cpu())
        max_grad = max(max_grad, g)
        torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
        optimizer.step()
        if step % config.validate_every == 0 or step == config.max_steps:
            value = evaluate(prep.inner_pairs)
            history.append({"step": step, "inner_val": value, "fit_loss": float(loss.detach().cpu()), "grad": g})
            if math.isfinite(value) and value < best - 1e-6:
                best, best_step, stale = value, step, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith("scorer.decoder.")}
            else:
                stale += 1
            if stale >= config.patience_checks:
                break
    current = model.state_dict()
    for k, v in best_state.items():
        current[k] = v.to(current[k].device)
    model.load_state_dict(current)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return {"stage": stage, "best_inner_val": best, "selected_step": best_step, "steps_run": history[-1]["step"],
            "n_validations": len(history) - 1, "max_gradient_l2": max_grad, "history": history,
            "selected_at_budget_edge": bool(best_step == config.max_steps),
            "selected_at_init": bool(best_step == 0)}


def evaluate_arms(model: WESpatialStateModel, prep: Prepared, *, random_model: WESpatialStateModel | None,
                  fraction: float = 0.5) -> dict[str, Any]:
    """All arms on STATE_SELECTION (and zero/adapter/learned on the fit and inner-val sets)."""

    data = prep.data
    sel = prep.selection_pairs
    device = prep.tensors["event_token"].device
    last = data.last_event_pos
    carry = np.where(last >= 0, data.event_segment[np.maximum(last, 0)], -1)
    donor = block_circular_donor(data.anchor_time, carry, sel.anchor_rows, horizon=1800.0, fraction=fraction)
    ok = donor >= 0
    with torch.no_grad():
        states = _states(model, prep)
        s_sel = states[torch.as_tensor(sel.anchor_rows, dtype=torch.long, device=device)]
        shifted_full = states.clone()
        if ok.any():
            shifted_full[torch.as_tensor(sel.anchor_rows[ok], device=device)] = s_sel[torch.as_tensor(donor[ok], device=device)]
        period_full = states.clone()
        period_full[torch.as_tensor(sel.anchor_rows, device=device)] = s_sel.mean(0, keepdim=True).expand_as(s_sel)
        arms = {
            "zero": pair_scores(model, prep, sel, None, use_bias=False, use_state=False),
            "adapter": pair_scores(model, prep, sel, None, use_bias=True, use_state=False),
            "learned": pair_scores(model, prep, sel, states, use_bias=True, use_state=True),
            "period_mean": pair_scores(model, prep, sel, period_full, use_bias=True, use_state=True),
            "shifted": pair_scores(model, prep, sel, shifted_full, use_bias=True, use_state=True),
        }
        if random_model is not None:
            r_states = _states(random_model, prep)
            arms["random"] = pair_scores(random_model, prep, sel, r_states, use_bias=True, use_state=True)
        context = {}
        for name, pairs in (("fit", prep.fit_pairs), ("inner_val", prep.inner_pairs)):
            context[name] = {
                "zero": float(weighted(pair_scores(model, prep, pairs, None, use_bias=False, use_state=False)["grammar"], pairs)),
                "adapter": float(weighted(pair_scores(model, prep, pairs, None, use_bias=True, use_state=False)["grammar"], pairs)),
                "learned": float(weighted(pair_scores(model, prep, pairs, states, use_bias=True, use_state=True)["grammar"], pairs)),
            }
    per_anchor_arrays = {}
    means = {}
    for name, sc in arms.items():
        g = sc["grammar"].detach().cpu().numpy().astype(np.float64)
        c = sc["contact_nll"].detach().cpu().numpy().astype(np.float64)
        pa = per_anchor(g, sel)
        pc = per_anchor(c, sel)
        if name == "shifted":
            pa[~ok] = np.nan; pc[~ok] = np.nan
        per_anchor_arrays[f"{name}_grammar"] = pa
        per_anchor_arrays[f"{name}_contact_nll"] = pc
        means[name] = {"grammar": float(weighted(sc["grammar"], sel)), "contact_nll": float(weighted(sc["contact_nll"], sel))}
    per_anchor_arrays["bootstrap_segment"] = carry[sel.anchor_rows].astype(np.int64)
    per_anchor_arrays["shift_valid"] = ok.astype(np.uint8)
    per_anchor_arrays["anchor_rows"] = sel.anchor_rows.astype(np.int64)
    per_anchor_arrays["anchor_time"] = data.anchor_time[sel.anchor_rows].astype(np.float64)
    per_anchor_arrays["state_selection_mean_abs"] = np.asarray([float(s_sel.mean(0).abs().mean().cpu())])
    return {"selection_means": means, "context": context, "per_anchor": per_anchor_arrays,
            "n_valid_donors": int(ok.sum()), "n_selection_anchors": int(sel.anchor_rows.size)}


def run_subject(*, data: SpatialData, bundle: FrozenDecoderBundle, config: WEStateConfig, device: torch.device,
                out_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    card_path = out_dir / "card.json"
    if card_path.exists() and not overwrite:
        return json.loads(card_path.read_text())
    started = time.time()
    seed_before_model_construction(config.seed)
    prep = prepare(data, bundle, config, device)
    model = WESpatialStateModel(input_dim=data.event_token.shape[1], bundle=bundle, config=config).to(device)
    # parity: h0 = 0 must reproduce the frozen decoder exactly on a few events
    with torch.no_grad():
        rows = torch.as_tensor(prep.cache_index[prep.train_pairs.pair_event[:16]], dtype=torch.long, device=device)
        b = {k: v[rows] for k, v in prep.dec_tensors.items()}
        from src.topic5_wiring_economy_rnn import next_rank_stop_loss
        logits, stops = bundle.model(b["x"], b["recruited"], b["valid"])
        ref = next_rank_stop_loss(logits, stops, b["target"], b["available"], b["valid"], b["is_last"])[0]
        mine = model.scorer.scores(b, None, use_bias=False, use_state=False)
        from .we_decoder import forward_with_h0
        l2, s2 = forward_with_h0(bundle.model, b["x"], b["recruited"], b["valid"], None)
        parity = float((l2 - logits).abs().max().cpu()) + float((s2 - stops).abs().max().cpu())
    if parity != 0.0:
        raise RuntimeError(f"zero-state parity failed: {parity:g}")
    stages = {}
    stages["adapter"] = fit_stage(model, prep, config, stage="adapter", use_bias=True, use_state=False,
                                  train_encoder=False, train_h0=False, train_bias=True, seed=config.seed + 11)
    stages["learned"] = fit_stage(model, prep, config, stage="learned", use_bias=True, use_state=True,
                                  train_encoder=True, train_h0=True, train_bias=False, seed=config.seed + 22)
    # random-encoder twin: same seed, encoder frozen at init, bias copied from the adapter stage
    seed_before_model_construction(config.seed)
    random_model = WESpatialStateModel(input_dim=data.event_token.shape[1], bundle=bundle, config=config).to(device)
    with torch.no_grad():
        random_model.scorer.h0_bias.copy_(model.scorer.h0_bias)
    stages["random"] = fit_stage(random_model, prep, config, stage="random", use_bias=True, use_state=True,
                                 train_encoder=False, train_h0=True, train_bias=False, seed=config.seed + 33)
    arms = evaluate_arms(model, prep, random_model=random_model)
    per_anchor_path = out_dir / "per_anchor.npz"
    atomic_write_npz(per_anchor_path, arms.pop("per_anchor"))
    atomic_write_torch(out_dir / "learned_state.pt", {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("scorer.decoder.")})
    card = {
        "format": FORMAT, "subject": data.subject, "config": asdict(config), "decoder_unit": str(bundle.unit_dir),
        "decoder_metrics": {k: bundle.metrics.get(k) for k in ("arm", "seed", "converged", "best_epoch", "n_epochs", "validation", "test")},
        "coverage": prep.coverage, "zero_state_parity_error": parity, "stages": stages, **arms,
        "per_anchor_path": str(per_anchor_path), "elapsed_seconds": time.time() - started,
        "selection_rule": "rolling inner-validation = chronologically last inner_val_fraction of TRAIN anchors; STATE_SELECTION scored only",
        "implementation_commit": current_commit(), "development_targets_read": False,
        "sealed_partition_opened": False, "seizure_outcomes_read": False,
    }
    atomic_write_json(card_path, card)
    return card
