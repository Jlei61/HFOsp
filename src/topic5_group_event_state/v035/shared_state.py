"""Shared multi-horizon S_N/S_G producers for Group-Event State v0.3.5.

This is deliberately separate from the completed per-horizon L0 baselines.
One producer is selected once on a common chronological split and then replayed
once.  Horizon-specific evaluators may differ, but all read the same frozen
trajectory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from src.topic5_group_event_state.v034_spatial_state.we_decoder import (
    FrozenDecoderBundle,
    decoder_tensors,
)

from .contracts import FORMAT_PREFIX, atomic_json, seed_all
from .full_mark_state import (
    FullMarkData,
    FullMarkStateModel,
    FullMarkTrainConfig,
    _chunks,
    _grid_states_from_event_post,
    _physical_loss,
    _physical_tensors,
    _target_scores,
    _to_device,
    collect_all_states,
    configure_event_input_view,
    fit_physical_q_baseline,
)
from .grammar_targets import (
    GrammarBlockTargets,
    GrammarDictionary,
    aggregate_grammar_blocks,
    fit_grammar_dictionary,
)


@dataclass(frozen=True)
class SharedProducerConfig:
    family: str
    base: FullMarkTrainConfig
    block_grammar_weight: float = 1.0
    local_grammar_weight: float = 1.0
    physical_weight: float = 1.0
    requested_communities: int = 4
    requested_repertoires: int = 6

    def validate(self) -> "SharedProducerConfig":
        if self.family not in {"S_N", "S_G"}:
            raise ValueError("shared family must be S_N or S_G")
        expected = "sn" if self.family == "S_N" else "sg"
        if self.base.objective_family != expected:
            raise ValueError(f"{self.family} requires objective_family={expected}")
        self.base.validate()
        return self


@dataclass(frozen=True)
class SharedGrammarData:
    dictionary: GrammarDictionary
    targets: GrammarBlockTargets


def build_shared_grammar_data(data: FullMarkData, config: SharedProducerConfig) -> SharedGrammarData:
    raw_rows = data.seq.order[data.source_position]
    part = np.asarray(data.seq.arrays["participation"][raw_rows], dtype=bool)
    delay = np.asarray(data.seq.arrays["relative_delay"][raw_rows], dtype=np.float32)
    tied = np.asarray(data.seq.arrays["tied_group_id"][raw_rows], dtype=np.int16)
    band = np.asarray(data.seq.arrays["band_features"][raw_rows], dtype=np.float32)
    # The dictionary may use CALIBRATION/FIT only.  INNER is reserved for the
    # single producer checkpoint and SELECTION is never opened during fitting.
    dictionary_rows = np.flatnonzero(np.isin(data.phase, ("CALIBRATION", "FIT")))
    dictionary = fit_grammar_dictionary(
        part,
        delay,
        tied,
        band,
        band_available=data.seq.index["band_available"],
        band_names=data.seq.index["bands"],
        fit_rows=dictionary_rows,
        seed=int(config.base.seed),
        requested_communities=int(config.requested_communities),
        requested_repertoires=int(config.requested_repertoires),
    )
    targets = aggregate_grammar_blocks(
        grid_time=data.grid_time,
        horizons_seconds=data.physical_horizons_seconds,
        future_valid=data.future_valid,
        event_time=data.event_time,
        participation=part,
        tied_group_id=tied,
        dictionary=dictionary,
    )
    return SharedGrammarData(dictionary=dictionary, targets=targets)


class SharedGrammarHead(nn.Module):
    """Nested q-only and state-residual block grammar heads."""

    def __init__(self, q_dim: int, state_dim: int, n_horizon: int,
                 n_community: int, n_repertoire: int, embedding_dim: int) -> None:
        super().__init__()
        self.n_horizon = int(n_horizon)
        self.widths = {
            "community": int(n_community),
            "coupling": int(n_community * n_community),
            "mixture": int(n_repertoire),
            "embedding": int(embedding_dim),
        }
        self.q_heads = nn.ModuleDict({
            key: nn.Linear(q_dim, self.n_horizon * width)
            for key, width in self.widths.items()
        })
        self.state_heads = nn.ModuleDict({
            key: nn.Linear(state_dim, self.n_horizon * width, bias=False)
            for key, width in self.widths.items()
        })
        for module in self.q_heads.values():
            nn.init.zeros_(module.weight); nn.init.zeros_(module.bias)
        for module in self.state_heads.values():
            nn.init.zeros_(module.weight)

    def predictions(self, q: Tensor, state: Tensor | None) -> dict[str, Tensor]:
        output = {}
        for key, width in self.widths.items():
            value = self.q_heads[key](q).reshape(q.shape[0], self.n_horizon, width)
            if state is not None:
                value = value + self.state_heads[key](state).reshape(
                    q.shape[0], self.n_horizon, width
                )
            output[key] = value
        return output

    def q_parameters(self) -> list[nn.Parameter]:
        return [p for module in self.q_heads.values() for p in module.parameters()]

    def state_parameters(self) -> list[nn.Parameter]:
        return [p for module in self.state_heads.values() for p in module.parameters()]


def _simplex_loss(logit: Tensor, target: Tensor, valid: Tensor) -> tuple[Tensor | None, int]:
    terms = []
    for j in range(valid.shape[1]):
        if bool(valid[:, j].any()):
            logp = torch.log_softmax(logit[:, j][valid[:, j]], dim=-1)
            terms.append(-(target[:, j][valid[:, j]] * logp).sum(-1).mean())
    return (torch.stack(terms).mean() if terms else None), int(valid.sum().detach())


def grammar_block_loss(
    head: SharedGrammarHead,
    q: Tensor,
    state: Tensor | None,
    targets: Mapping[str, Tensor],
) -> tuple[Tensor | None, dict[str, float | int | None]]:
    pred = head.predictions(q, state)
    terms: list[Tensor] = []
    metrics: dict[str, float | int | None] = {}
    for name, valid_name in (
        ("community", "community_valid"),
        ("coupling", "coupling_valid"),
        ("mixture", "mixture_valid"),
    ):
        value, n = _simplex_loss(pred[name], targets[name], targets[valid_name])
        metrics[f"{name}_loss"] = None if value is None else float(value.detach())
        metrics[f"n_{name}"] = n
        if value is not None:
            terms.append(value)
    valid = targets["embedding_valid"]
    embedding_terms = []
    for j in range(valid.shape[1]):
        if bool(valid[:, j].any()):
            embedding_terms.append(
                (pred["embedding"][:, j][valid[:, j]]
                 - targets["embedding"][:, j][valid[:, j]]).square().mean()
            )
    embedding = torch.stack(embedding_terms).mean() if embedding_terms else None
    metrics["embedding_loss"] = None if embedding is None else float(embedding.detach())
    metrics["n_embedding"] = int(valid.sum().detach())
    if embedding is not None:
        terms.append(embedding)
    return (torch.stack(terms).mean() if terms else None), metrics


def _grammar_tensors(
    grammar: SharedGrammarData,
    rows: np.ndarray,
    device: torch.device,
) -> dict[str, Tensor]:
    t = grammar.targets
    return {
        "community": torch.as_tensor(t.community_occupancy[rows], dtype=torch.float32, device=device),
        "community_valid": torch.as_tensor(t.community_valid[rows], dtype=torch.bool, device=device),
        "coupling": torch.as_tensor(t.cross_community_coupling[rows], dtype=torch.float32, device=device),
        "coupling_valid": torch.as_tensor(t.coupling_valid[rows], dtype=torch.bool, device=device),
        "mixture": torch.as_tensor(t.repertoire_mixture[rows], dtype=torch.float32, device=device),
        "mixture_valid": torch.as_tensor(t.repertoire_valid[rows], dtype=torch.bool, device=device),
        "embedding": torch.as_tensor(t.repertoire_embedding_mean[rows], dtype=torch.float32, device=device),
        "embedding_valid": torch.as_tensor(t.repertoire_embedding_valid[rows], dtype=torch.bool, device=device),
    }


def fit_grammar_q_baseline(
    head: SharedGrammarHead,
    data: FullMarkData,
    grammar: SharedGrammarData,
    config: FullMarkTrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    fit_rows = np.flatnonzero(data.grid_phase == "FIT")
    inner_rows = np.flatnonzero(data.grid_phase == "INNER")
    q_fit = torch.as_tensor(data.grid_q[fit_rows], dtype=torch.float32, device=device)
    q_inner = torch.as_tensor(data.grid_q[inner_rows], dtype=torch.float32, device=device)
    y_fit = _grammar_tensors(grammar, fit_rows, device)
    y_inner = _grammar_tensors(grammar, inner_rows, device)
    params = head.q_parameters()
    optimizer = torch.optim.AdamW(params, lr=config.physical_baseline_lr,
                                  weight_decay=config.weight_decay)
    best, best_step, stale = math.inf, 0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    history = []
    for step in range(config.physical_baseline_steps + 1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = grammar_block_loss(head, q_fit, None, y_fit)
            if loss is None:
                raise ValueError("no FIT block-grammar target")
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, config.gradient_clip); optimizer.step()
        if step % 25 == 0 or step == config.physical_baseline_steps:
            with torch.no_grad():
                loss, metrics = grammar_block_loss(head, q_inner, None, y_inner)
            value = float(loss.detach()) if loss is not None else math.inf
            history.append({"step": step, "inner_loss": value, **metrics})
            if np.isfinite(value) and value < best - 1e-6:
                best, best_step, stale = value, step, 0
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            else:
                stale += 1
            if stale >= 8:
                break
    head.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    for parameter in head.q_parameters():
        parameter.requires_grad_(False)
    return {"selected_step": best_step, "best_inner_loss": best, "history": history}


def _grid_loss(
    family: str,
    model: FullMarkStateModel,
    grammar_head: SharedGrammarHead | None,
    grammar: SharedGrammarData | None,
    data: FullMarkData,
    rows: np.ndarray,
    state: Tensor,
    device: torch.device,
) -> tuple[Tensor | None, dict[str, float]]:
    physical = _physical_tensors(data, rows, device)
    p_loss, _ = _physical_loss(
        model.physical_head, physical[0], state, *physical[1:],
        endpoint_family="sn" if family == "S_N" else "sg",
    )
    parts: dict[str, Tensor] = {}
    if p_loss is not None:
        parts["burden" if family == "S_N" else "contact_field"] = p_loss
    if family == "S_G":
        if grammar_head is None or grammar is None:
            raise ValueError("S_G requires grammar block data/head")
        g_loss, _ = grammar_block_loss(
            grammar_head,
            physical[0],
            state,
            _grammar_tensors(grammar, rows, device),
        )
        if g_loss is not None:
            parts["block_grammar"] = g_loss
    if not parts:
        return None, {}
    return torch.stack(list(parts.values())).mean(), {
        key: float(value.detach()) for key, value in parts.items()
    }


def run_shared_phase(
    model: FullMarkStateModel,
    grammar_head: SharedGrammarHead | None,
    grammar: SharedGrammarData | None,
    data: FullMarkData,
    bundle_tensors: dict[str, Tensor],
    phase: str,
    config: SharedProducerConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    train = optimizer is not None
    model.train(train); model.decoder.decoder.eval()
    if grammar_head is not None:
        grammar_head.train(train)
    losses: list[float] = []
    component: dict[str, list[float]] = {}
    for seg in np.unique(data.event_segment):
        rows = np.flatnonzero(data.event_segment == seg)
        if not np.any(data.phase[rows] == phase):
            continue
        state = model.state.initial(1, device)
        previous = float(data.event_time[rows[0]])
        for chunk in _chunks(rows, data.event_time, config.base.chunk_events,
                             config.base.chunk_seconds):
            batch = _to_device(data.seq.gather_positions(data.input_source_position[chunk]), device)
            q_all = torch.as_tensor(data.q_context[chunk], dtype=torch.float32, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=config.base.amp and device.type == "cuda"):
                embedding = model.encode_events(batch)
            post, scored_state, scored_rows, scored_q = [], [], [], []
            for local, row in enumerate(chunk):
                dt = torch.tensor([max(0.0, float(data.event_time[row]) - previous)], device=device)
                pre = model.state.evolve(state, dt)
                state = model.state.update(
                    pre,
                    embedding[local:local + 1].float()
                    - model.expected_from_q(q_all[local:local + 1]),
                )
                post.append(state)
                if data.phase[row] == phase:
                    scored_state.append(state); scored_rows.append(int(row)); scored_q.append(q_all[local:local + 1])
                previous = float(data.event_time[row])
            terms: list[Tensor] = []
            if config.family == "S_G" and scored_rows:
                anchors = np.asarray(scored_rows, dtype=np.int64)
                s = torch.cat(scored_state, 0); q = torch.cat(scored_q, 0)
                local_terms = []
                for j, weight in enumerate(config.base.offset_weights):
                    score = _target_scores(model, bundle_tensors, data, anchors, s, q, j)
                    if score is not None:
                        local_terms.append(float(weight) * score["grammar"].mean())
                if local_terms:
                    local_loss = torch.stack(local_terms).sum() / max(sum(config.base.offset_weights), 1e-8)
                    terms.append(float(config.local_grammar_weight) * local_loss)
                    component.setdefault("local_grammar", []).append(float(local_loss.detach()))
            grid_rows = np.flatnonzero(
                (data.grid_phase == phase)
                & np.isin(data.grid_source_event, chunk)
                & (data.grid_source_event >= 0)
            )
            if grid_rows.size and post:
                source_local = np.searchsorted(chunk, data.grid_source_event[grid_rows])
                if np.any(source_local >= chunk.size) or not np.array_equal(
                    chunk[source_local], data.grid_source_event[grid_rows]
                ):
                    raise ValueError("shared grid anchor source is not in its event chunk")
                source_state = torch.cat(post, 0)[torch.as_tensor(source_local, device=device)]
                dt_grid = torch.as_tensor(data.grid_source_dt[grid_rows], dtype=torch.float32, device=device)
                grid_state = model.state.evolve(source_state, dt_grid)
                grid_loss, pieces = _grid_loss(
                    config.family, model, grammar_head, grammar, data, grid_rows, grid_state, device,
                )
                if grid_loss is not None:
                    terms.append(float(config.physical_weight) * grid_loss)
                    for key, value in pieces.items():
                        component.setdefault(key, []).append(value)
            if terms:
                loss = torch.stack(terms).mean(); losses.append(float(loss.detach()))
                if train:
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    params = [p for p in model.parameters() if p.requires_grad]
                    if grammar_head is not None:
                        params += [p for p in grammar_head.parameters() if p.requires_grad]
                    norm = torch.nn.utils.clip_grad_norm_(params, config.base.gradient_clip)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("non-finite shared-producer gradient")
                    optimizer.step()
            state = state.detach()
    return {
        "phase": phase,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "components": {key: float(np.mean(value)) for key, value in component.items()},
        "n_chunks_scored": len(losses),
    }


def train_shared_producer(
    data: FullMarkData,
    bundle: FrozenDecoderBundle,
    base_adapter: Path,
    config: SharedProducerConfig,
    *,
    device: torch.device,
    out_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    config.validate(); seed_all(config.base.seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    card_path = out_dir / "card.json"
    if card_path.exists() and not overwrite:
        return json.loads(card_path.read_text(encoding="utf-8"))
    started = time.time()
    data = configure_event_input_view(data, config.base)
    model = FullMarkStateModel(data, bundle, base_adapter, config.base, device).to(device)
    physical_q = fit_physical_q_baseline(model, data, config.base, device)
    grammar = build_shared_grammar_data(data, config) if config.family == "S_G" else None
    grammar_head = None
    grammar_q = None
    if grammar is not None:
        grammar_head = SharedGrammarHead(
            data.q_context.shape[1], model.state.cfg.state_dim,
            len(data.physical_horizons_seconds), grammar.dictionary.n_communities,
            grammar.dictionary.n_repertoires,
            grammar.dictionary.event_repertoire_embedding.shape[1],
        ).to(device)
        grammar_q = fit_grammar_q_baseline(grammar_head, data, grammar, config.base, device)

    groups = [
        {"params": model.event_encoder.parameters(), "lr": config.base.encoder_lr},
        {"params": [model.timing_token], "lr": config.base.encoder_lr},
        {"params": model.state.parameters(), "lr": config.base.state_lr},
        {"params": model.expected_from_q.parameters(), "lr": config.base.state_lr},
        {"params": model.physical_head.state_parameters(), "lr": config.base.adapter_lr},
    ]
    if config.family == "S_G":
        groups.append({"params": model.m_adapter.parameters(), "lr": config.base.adapter_lr})
        assert grammar_head is not None
        groups.append({"params": grammar_head.state_parameters(), "lr": config.base.adapter_lr})
    else:
        for parameter in model.m_adapter.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(groups, weight_decay=config.base.weight_decay)
    tensors = decoder_tensors(bundle, device)
    best, best_epoch, best_model, best_grammar = math.inf, -1, None, None
    stale, history = 0, []
    for epoch in range(config.base.max_epochs):
        fit = run_shared_phase(model, grammar_head, grammar, data, tensors, "FIT", config, device, optimizer)
        inner = run_shared_phase(model, grammar_head, grammar, data, tensors, "INNER", config, device, None)
        history.append({"epoch": epoch, "fit": fit, "inner": inner})
        value = inner["mean_loss"]
        if value is not None and np.isfinite(value) and value < best - 1e-5:
            best, best_epoch, stale = float(value), epoch, 0
            best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                          if not k.startswith("decoder.decoder.")}
            best_grammar = None if grammar_head is None else {
                k: v.detach().cpu().clone() for k, v in grammar_head.state_dict().items()
            }
        else:
            stale += 1
        atomic_json(out_dir / "progress.json", {
            "format": f"{FORMAT_PREFIX}_shared_producer_progress_v1",
            "family": config.family, "subject": data.subject, "history": history,
        })
        if epoch + 1 >= config.base.min_epochs and stale >= config.base.patience:
            break
    if best_model is None:
        raise RuntimeError("shared producer produced no finite INNER checkpoint")
    current = model.state_dict()
    for key, value in best_model.items():
        current[key] = value.to(device)
    model.load_state_dict(current)
    if grammar_head is not None and best_grammar is not None:
        grammar_head.load_state_dict({k: v.to(device) for k, v in best_grammar.items()})
    pre, post = collect_all_states(model, data, device, config.base)
    trajectory = out_dir / "state_trajectory.npz"
    with trajectory.open("wb") as handle:
        np.savez_compressed(
            handle,
            event_time=data.event_time.astype(np.float64),
            event_segment=data.event_segment.astype(np.int64),
            phase=data.phase,
            state_pre=pre,
            state_post=post,
            q_context=data.q_context.astype(np.float32),
            fixed_taus_seconds=model.state.taus.detach().cpu().numpy().astype(np.float32),
            state_mean=model.state.mean.detach().cpu().numpy().astype(np.float32),
            producer_family=np.asarray(config.family),
            shared_horizons_seconds=np.asarray(data.physical_horizons_seconds, dtype=np.float64),
        )
    checkpoint = out_dir / "checkpoint.pt"
    torch.save({
        "family": config.family,
        "subject": data.subject,
        "base_adapter": str(base_adapter),
        "producer": best_model,
        "grammar_head": best_grammar,
        "config": {"family": config.family, "base": asdict(config.base),
                   "block_grammar_weight": config.block_grammar_weight,
                   "local_grammar_weight": config.local_grammar_weight,
                   "physical_weight": config.physical_weight,
                   "requested_communities": config.requested_communities,
                   "requested_repertoires": config.requested_repertoires},
    }, checkpoint)
    dictionary_path = None
    if grammar is not None:
        dictionary_path = out_dir / "grammar_dictionary.npz"
        with dictionary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                community_of_contact=grammar.dictionary.community_of_contact,
                repertoire_centres=grammar.dictionary.repertoire_centres,
                event_repertoire_embedding=grammar.dictionary.event_repertoire_embedding,
                event_repertoire_label=grammar.dictionary.event_repertoire_label,
                provenance_json=np.asarray(json.dumps(grammar.dictionary.provenance, sort_keys=True)),
            )
    card = {
        "format": f"{FORMAT_PREFIX}_shared_producer_card_v1",
        "family": config.family,
        "subject": data.subject,
        "seed": int(config.base.seed),
        "shared_horizons_seconds": list(data.physical_horizons_seconds),
        "selected_epoch": best_epoch,
        "best_inner_loss": best,
        "history": history,
        "physical_q_baseline": physical_q,
        "grammar_q_baseline": grammar_q,
        "checkpoint": str(checkpoint),
        "state_trajectory": str(trajectory),
        "grammar_dictionary": None if dictionary_path is None else str(dictionary_path),
        "producer_contract": "one checkpoint and one causal trajectory shared by every horizon",
        "selection_targets_read": False,
        "development_targets_read": False,
        "sealed_partition_opened": False,
        "seizure_outcomes_read": False,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(card_path, card)
    return card


def update_checkpoint_registry(path: Path, card: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path)
    registry = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {
            "format": f"{FORMAT_PREFIX}_shared_checkpoint_registry_v1",
            "entries": {},
            "development_targets_read": False,
            "sealed_partition_opened": False,
        }
    )
    key = f"{card['subject']}::{card['family']}::seed{card['seed']}"
    registry["entries"][key] = {
        "subject": card["subject"], "family": card["family"], "seed": card["seed"],
        "checkpoint": card["checkpoint"], "state_trajectory": card["state_trajectory"],
        "shared_horizons_seconds": card["shared_horizons_seconds"],
        "selected_epoch": card["selected_epoch"],
        "selection_targets_read": card["selection_targets_read"],
    }
    atomic_json(path, registry)
    return registry
