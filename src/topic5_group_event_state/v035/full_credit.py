"""Exact within-episode credit assignment for the shared S_N/S_G producer.

Microchunks in this module are activation-checkpoint units only.  Unlike the
archived v0.3.5 loop, state is never detached between them and the optimiser is
stepped only after the complete causal episode has been backpropagated.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .full_mark_state import (
    FullMarkData,
    FullMarkStateModel,
    _chunks,
    _physical_tensors,
    _target_scores,
    _to_device,
)
from .shared_state import (
    SharedGrammarData,
    SharedGrammarHead,
    SharedProducerConfig,
    _grammar_tensors,
    _grid_loss,
)


COMPONENTS = ("local_grammar", "burden", "contact_field", "block_grammar")


def _chunk_forward(
    state_in: Tensor,
    *,
    chunk: np.ndarray,
    previous_time: float,
    model: FullMarkStateModel,
    grammar_head: SharedGrammarHead | None,
    grammar: SharedGrammarData | None,
    data: FullMarkData,
    bundle_tensors: dict[str, Tensor],
    phase: str,
    config: SharedProducerConfig,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pure checkpoint body: state-out, local loss, components, scored flag."""

    batch = _to_device(data.seq.gather_positions(data.input_source_position[chunk]), device)
    q_all = torch.as_tensor(data.q_context[chunk], dtype=torch.float32, device=device)
    with torch.autocast(
        "cuda", dtype=torch.bfloat16,
        enabled=config.base.amp and device.type == "cuda",
    ):
        embedding = model.encode_events(batch)

    state = state_in
    previous = float(previous_time)
    post: list[Tensor] = []
    scored_state: list[Tensor] = []
    scored_rows: list[int] = []
    scored_q: list[Tensor] = []
    for local, row in enumerate(chunk):
        dt = torch.tensor(
            [max(0.0, float(data.event_time[row]) - previous)],
            dtype=torch.float32,
            device=device,
        )
        pre = model.state.evolve(state, dt)
        state = model.state.update(
            pre,
            model.innovation(embedding[local : local + 1], q_all[local : local + 1]),
        )
        post.append(state)
        if data.phase[row] == phase:
            scored_state.append(state)
            scored_rows.append(int(row))
            scored_q.append(q_all[local : local + 1])
        previous = float(data.event_time[row])

    terms: list[Tensor] = []
    values: dict[str, float] = {}
    if config.family == "S_G" and scored_rows:
        anchors = np.asarray(scored_rows, dtype=np.int64)
        states = torch.cat(scored_state, 0)
        q = torch.cat(scored_q, 0)
        local_terms = []
        for j, weight in enumerate(config.base.offset_weights):
            score = _target_scores(model, bundle_tensors, data, anchors, states, q, j)
            if score is not None:
                local_terms.append(float(weight) * score["grammar"].mean())
        if local_terms:
            local_loss = torch.stack(local_terms).sum() / max(
                sum(config.base.offset_weights), 1e-8
            )
            terms.append(float(config.local_grammar_weight) * local_loss)
            values["local_grammar"] = float(local_loss.detach())

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
            raise ValueError("full-credit grid anchor source is not in checkpoint chunk")
        source_state = torch.cat(post, 0)[
            torch.as_tensor(source_local, dtype=torch.long, device=device)
        ]
        dt_grid = torch.as_tensor(
            data.grid_source_dt[grid_rows], dtype=torch.float32, device=device
        )
        grid_state = model.state.evolve(source_state, dt_grid)
        grid_loss, pieces = _grid_loss(
            config.family,
            model,
            grammar_head,
            grammar,
            data,
            grid_rows,
            grid_state,
            device,
            block_grammar_weight=config.block_grammar_weight,
        )
        if grid_loss is not None:
            terms.append(float(config.physical_weight) * grid_loss)
            values.update(pieces)

    if terms:
        loss = torch.stack(terms).mean()
        scored = torch.ones((), dtype=torch.float32, device=device)
    else:
        # Preserve the state graph through warm-up chunks even when they do not
        # themselves contain a scored anchor.
        loss = state.sum() * 0.0
        scored = torch.zeros((), dtype=torch.float32, device=device)
    component = torch.as_tensor(
        [values.get(name, np.nan) for name in COMPONENTS],
        dtype=torch.float32,
        device=device,
    )
    return state, loss, component, scored


def run_shared_phase_full_credit(
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
    """Run a phase with no gradient truncation inside a causal episode."""

    if config.base.credit_assignment_mode != "checkpointed_episode":
        raise ValueError("full-credit runner requires checkpointed_episode mode")
    train = optimizer is not None
    model.train(train)
    model.decoder.decoder.eval()
    if grammar_head is not None:
        grammar_head.train(train)
    episode_losses: list[float] = []
    component_values: dict[str, list[float]] = defaultdict(list)
    gradient_norms: list[float] = []
    chunks_scored = 0
    chunks_total = 0
    episode_spans = []

    for seg in np.unique(data.event_segment):
        all_rows = np.flatnonzero(data.event_segment == seg)
        target_rows = all_rows[data.phase[all_rows] == phase]
        if target_rows.size == 0:
            continue
        # Replay from the true episode boundary through the last target anchor;
        # later-phase events are not even materialised during this phase.
        rows = all_rows[all_rows <= target_rows[-1]]
        state = model.state.initial(1, device)
        previous = float(data.event_time[rows[0]])
        losses: list[Tensor] = []
        components: list[Tensor] = []
        scored_flags: list[Tensor] = []
        for raw_chunk in _chunks(
            rows, data.event_time, config.base.chunk_events, config.base.chunk_seconds
        ):
            chunk = np.asarray(raw_chunk, dtype=np.int64)
            chunk_previous = previous

            def body(current: Tensor, *, _chunk=chunk, _previous=chunk_previous):
                return _chunk_forward(
                    current,
                    chunk=_chunk,
                    previous_time=_previous,
                    model=model,
                    grammar_head=grammar_head,
                    grammar=grammar,
                    data=data,
                    bundle_tensors=bundle_tensors,
                    phase=phase,
                    config=config,
                    device=device,
                )

            if train:
                state, loss, component, scored = checkpoint(
                    body, state, use_reentrant=False, preserve_rng_state=True
                )
            else:
                with torch.no_grad():
                    state, loss, component, scored = body(state)
            losses.append(loss)
            components.append(component)
            scored_flags.append(scored)
            previous = float(data.event_time[chunk[-1]])
            chunks_total += 1

        flags = torch.stack(scored_flags)
        n_scored = int(flags.detach().sum().item())
        if n_scored == 0:
            continue
        loss_stack = torch.stack(losses)
        episode_loss = (loss_stack * flags).sum() / flags.sum().clamp_min(1.0)
        if train:
            optimizer.zero_grad(set_to_none=True)
            episode_loss.backward()
            params = [p for p in model.parameters() if p.requires_grad]
            if grammar_head is not None:
                params += [p for p in grammar_head.parameters() if p.requires_grad]
            norm = torch.nn.utils.clip_grad_norm_(params, config.base.gradient_clip)
            if not torch.isfinite(norm):
                raise FloatingPointError("non-finite full-credit producer gradient")
            optimizer.step()
            gradient_norms.append(float(norm))
        episode_losses.append(float(episode_loss.detach()))
        chunks_scored += n_scored
        matrix = torch.stack(components).detach().cpu().numpy()
        for index, name in enumerate(COMPONENTS):
            values = matrix[:, index]
            component_values[name].extend(values[np.isfinite(values)].tolist())
        episode_spans.append(float(data.event_time[rows[-1]] - data.event_time[rows[0]]))

    return {
        "phase": phase,
        "mean_loss": float(np.mean(episode_losses)) if episode_losses else None,
        "components": {
            key: float(np.mean(value)) for key, value in component_values.items() if value
        },
        "n_episodes_scored": len(episode_losses),
        "n_checkpoint_chunks": chunks_total,
        "n_chunks_scored": chunks_scored,
        "max_credit_span_seconds": max(episode_spans, default=0.0),
        "median_credit_span_seconds": float(np.median(episode_spans)) if episode_spans else 0.0,
        "gradient_norm_median": float(np.median(gradient_norms)) if gradient_norms else None,
        "credit_assignment_contract": (
            "no detach inside true episode; microchunks are activation checkpoints only"
        ),
    }
