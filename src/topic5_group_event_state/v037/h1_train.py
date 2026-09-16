"""Nested multi-horizon H1 training for the v0.3.7 observer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import subprocess
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from src.topic5_group_event_state.v035.dynamic_rate import negative_binomial_nll

from .baselines import fixed_mark_ewma, mark_ewma_features_at_queries
from .contracts import atomic_json, sha256_file
from .ctssm import DualStreamEventCTSSM, dual_stream_features_at_queries
from .h1_data import H1SubjectData


@dataclass(frozen=True)
class H1TrainConfig:
    taus_seconds: tuple[float, ...] = (
        600.0, 1800.0, 3600.0, 7200.0, 14400.0, 28800.0, 57600.0,
    )
    burden_channels_per_tau: int = 2
    grammar_channels_per_tau: int = 3
    lr_q: float = 3e-3
    lr_bmark: float = 2e-3
    lr_state: float = 3e-4
    lr_state_head: float = 1e-3
    lr_random_head: float = 1e-3
    weight_decay: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_steps_q: int = 900
    max_steps_bmark: int = 1200
    max_steps_state: int = 1800
    max_steps_random: int = 900
    validate_every: int = 25
    patience_checks: int = 16
    gradient_clip: float = 2.0
    state_readout_init_std: float = 1e-2
    warmup_steps_state: int = 0
    # A small non-zero optimiser start may help diagnose optimisation, but the
    # fitted parent remains the scientific step-zero candidate (see
    # `_train_stage`).  The default keeps exact nested parity.
    bmark_readout_init_std: float = 0.0
    warmup_steps_bmark: int = 0
    weight_decay_bmark: float | None = None
    seed: int = 20260904


class NestedH1Readout(nn.Module):
    ENDPOINTS = ("count", "burden", "community", "coupling", "mixture", "embedding", "mark")

    def __init__(
        self,
        q_dim: int,
        bmark_burden_dim: int,
        bmark_grammar_dim: int,
        state_burden_dim: int,
        state_grammar_dim: int,
        widths: Mapping[str, int],
        n_horizon: int,
        state_readout_init_std: float = 1e-2,
        bmark_readout_init_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.widths = dict(widths)
        self.n_horizon = int(n_horizon)
        self.bmark_burden_dim = int(bmark_burden_dim)
        self.state_burden_dim = int(state_burden_dim)
        self.q = nn.ModuleDict()
        self.bmark = nn.ModuleDict()
        self.state = nn.ModuleDict()
        self.random = nn.ModuleDict()
        for name in self.ENDPOINTS:
            width = self.n_horizon * int(self.widths[name])
            burden_endpoint = name in {"count", "burden"}
            b_dim = bmark_burden_dim if burden_endpoint else bmark_grammar_dim
            state_dim = state_burden_dim if burden_endpoint else state_grammar_dim
            self.q[name] = nn.Linear(q_dim, width)
            self.bmark[name] = nn.Linear(b_dim, width, bias=False)
            self.state[name] = nn.Linear(state_dim, width, bias=False)
            self.random[name] = nn.Linear(state_dim, width, bias=False)
            # B_rate is the innermost learned control and has no fitted parent.
            # A framework-default random linear map made its step-zero score
            # seed-dependent and could be selected by chance.  Its scientific
            # zero-increment candidate is therefore deterministic.
            nn.init.zeros_(self.q[name].weight)
            nn.init.zeros_(self.q[name].bias)
            # The optimiser start and the scientific parent candidate are
            # deliberately separate.  `_train_stage` always retains the
            # zero-increment parent even when this tensor starts non-zero.
            if float(bmark_readout_init_std) > 0.0:
                nn.init.normal_(self.bmark[name].weight, std=float(bmark_readout_init_std))
            else:
                nn.init.zeros_(self.bmark[name].weight)
            nn.init.normal_(self.state[name].weight, std=float(state_readout_init_std))
            nn.init.zeros_(self.random[name].weight)
        self.log_dispersion = nn.Parameter(torch.zeros(n_horizon))

    def predict(
        self,
        q: Tensor,
        bmark: Tensor | None = None,
        state: Tensor | None = None,
        random_state: Tensor | None = None,
    ) -> dict[str, Tensor]:
        out = {}
        for name in self.ENDPOINTS:
            burden_endpoint = name in {"count", "burden"}
            value = self.q[name](q)
            if bmark is not None:
                b_value = (
                    bmark[:, :self.bmark_burden_dim]
                    if burden_endpoint else bmark[:, self.bmark_burden_dim:]
                )
                value = value + self.bmark[name](b_value)
            if state is not None:
                s_value = (
                    state[:, :self.state_burden_dim]
                    if burden_endpoint else state[:, self.state_burden_dim:]
                )
                normalised = torch.nn.functional.layer_norm(s_value, (s_value.shape[-1],))
                value = value + self.state[name](normalised)
            if random_state is not None:
                r_value = (
                    random_state[:, :self.state_burden_dim]
                    if burden_endpoint else random_state[:, self.state_burden_dim:]
                )
                normalised_random = torch.nn.functional.layer_norm(
                    r_value, (r_value.shape[-1],)
                )
                value = value + self.random[name](normalised_random)
            out[name] = value.reshape(q.shape[0], self.n_horizon, self.widths[name])
        return out

    def parameters_for(self, stage: str) -> list[nn.Parameter]:
        if stage == "q":
            return [*self.q.parameters(), self.log_dispersion]
        if stage == "bmark":
            return list(self.bmark.parameters())
        if stage == "state":
            return list(self.state.parameters())
        if stage == "random":
            return list(self.random.parameters())
        raise KeyError(stage)


def _continuous_standardise(target: np.ndarray, valid: np.ndarray, fit_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(target, dtype=np.float64)
    centre = np.zeros(value.shape[1:], dtype=np.float64)
    scale = np.ones(value.shape[1:], dtype=np.float64)
    for h in range(value.shape[1]):
        rows = fit_rows[valid[fit_rows, h]]
        if rows.size:
            centre[h] = np.nanmedian(value[rows, h], axis=0)
            scale[h] = 1.4826 * np.nanmedian(np.abs(value[rows, h] - centre[h]), axis=0)
            scale[h] = np.where(np.isfinite(scale[h]) & (scale[h] > 1e-5), scale[h], 1.0)
    return np.clip((value - centre) / scale, -12.0, 12.0).astype(np.float32), centre, scale


def _target_bundle(data: H1SubjectData, device: torch.device) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Any]]:
    rate = data.rate
    fit = np.flatnonzero(rate.phase == "FIT")
    g = data.grammar_targets
    continuous = {
        "burden": (data.burden_mark_mean_target, data.burden_mark_mean_valid),
        "embedding": (g.repertoire_embedding_mean, g.repertoire_embedding_valid),
        "mark": (data.grammar_mark_mean_target, data.grammar_mark_mean_valid),
    }
    target: dict[str, Tensor] = {
        "count": torch.as_tensor(rate.target_count[..., None], dtype=torch.float32, device=device),
        "community": torch.as_tensor(g.community_occupancy, dtype=torch.float32, device=device),
        "coupling": torch.as_tensor(g.cross_community_coupling, dtype=torch.float32, device=device),
        "mixture": torch.as_tensor(g.repertoire_mixture, dtype=torch.float32, device=device),
    }
    valid: dict[str, Tensor] = {
        "count": torch.as_tensor(rate.target_valid, dtype=torch.bool, device=device),
        "community": torch.as_tensor(g.community_valid, dtype=torch.bool, device=device),
        "coupling": torch.as_tensor(g.coupling_valid, dtype=torch.bool, device=device),
        "mixture": torch.as_tensor(g.repertoire_valid, dtype=torch.bool, device=device),
    }
    scales: dict[str, Any] = {}
    for name, (value, mask) in continuous.items():
        z, centre, scale = _continuous_standardise(value, mask, fit)
        target[name] = torch.as_tensor(z, dtype=torch.float32, device=device)
        valid[name] = torch.as_tensor(mask, dtype=torch.bool, device=device)
        scales[name] = {"centre": centre.tolist(), "scale": scale.tolist()}
    return target, valid, scales


def _standardise_features(value: np.ndarray | Tensor, fit_rows: np.ndarray, device: torch.device) -> tuple[Tensor, np.ndarray, np.ndarray]:
    x = value.detach().cpu().numpy() if isinstance(value, Tensor) else np.asarray(value)
    centre = np.nanmedian(x[fit_rows], axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(x[fit_rows] - centre), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-5), scale, 1.0)
    z = np.clip((x - centre) / scale, -12.0, 12.0).astype(np.float32)
    return torch.as_tensor(z, device=device), centre, scale



def independent_window_count(
    anchor_time: np.ndarray,
    horizon_seconds: float,
    *,
    segment: np.ndarray | None = None,
) -> int:
    """Greedy count of non-overlapping ``horizon_seconds`` windows.

    Held-out rows are five-minute anchors, so a 24 h evaluation slab yields
    ~200 rows at every horizon while admitting only a handful of genuinely
    independent 6-8 h windows.  Reporting the row count as sample size
    overstates the evidence by one to two orders of magnitude, which is the
    same mistake the seizure layer already corrected by counting seizures
    instead of grid rows.  Windows never bridge a recording segment.
    """
    times = np.asarray(anchor_time, dtype=np.float64).reshape(-1)
    if times.size == 0:
        return 0
    horizon = float(horizon_seconds)
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon_seconds must be positive and finite")
    if segment is None:
        groups = [times]
    else:
        seg = np.asarray(segment).reshape(-1)
        if seg.size != times.size:
            raise ValueError("segment must align with anchor_time")
        groups = [times[seg == value] for value in np.unique(seg)]
    total = 0
    for group in groups:
        ordered = np.sort(group)
        last = -np.inf
        for value in ordered:
            if value >= last + horizon:
                total += 1
                last = float(value)
    return int(total)


def _code_provenance(source: Path | None = None) -> dict[str, Any]:
    """Record which source produced a card.

    v0.3.7 summarised 19:37 cards with 21:55 code because nothing on the card
    said which version wrote it.  A stale card is now detectable.
    """
    module = Path(__file__).resolve() if source is None else Path(source).resolve()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=module.parent, text=True
        ).strip()
    except Exception:  # pragma: no cover - provenance must never break a run
        commit = "unavailable"
    return {
        "git_commit": commit,
        "source_file": module.name,
        "source_sha256": sha256_file(module),
    }


def _fixed_features(data: H1SubjectData, device: torch.device, taus: tuple[float, ...]) -> Tensor:
    chunks, rows = [], []
    for segment in np.unique(data.event_segment):
        er = np.flatnonzero(data.event_segment == segment)
        ar = np.flatnonzero(data.rate.segment == segment)
        if er.size == 0 or ar.size == 0:
            continue
        event_time = torch.as_tensor(data.event_time[er], dtype=torch.float64, device=device)
        query_time = torch.as_tensor(data.rate.anchor_time[ar], dtype=torch.float64, device=device)
        output = fixed_mark_ewma(
            event_time,
            torch.as_tensor(data.burden_mark[er], dtype=torch.float32, device=device),
            torch.as_tensor(data.grammar_mark[er], dtype=torch.float32, device=device),
            taus_seconds=taus,
        )
        chunks.append(mark_ewma_features_at_queries(output, event_time, query_time, taus_seconds=taus))
        rows.append(torch.as_tensor(ar, dtype=torch.long, device=device))
    if not chunks:
        raise ValueError("no event-bearing carry segment for fixed marked-history features")
    row = torch.cat(rows)
    value = torch.cat(chunks)
    if torch.unique(row).numel() != row.numel():
        raise ValueError("fixed marked-history query rows overlap")
    # A valid carry segment may contain recorded anchors but no IED.  Those
    # anchors are genuine zero-history observations; dropping them shortens
    # B_mark and silently breaks its one-to-one alignment with q/targets.
    full = value.new_zeros((data.rate.anchor_time.size, value.shape[1]))
    return full.index_copy(0, row, value)


class EventStateComputer:
    def __init__(self, data: H1SubjectData, model: DualStreamEventCTSSM, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.n_query = int(data.rate.anchor_time.size)
        self.chains = []
        for segment in np.unique(data.event_segment):
            er = np.flatnonzero(data.event_segment == segment)
            ar = np.flatnonzero(data.rate.segment == segment)
            if er.size == 0 or ar.size == 0:
                continue
            self.chains.append((
                torch.as_tensor(ar, dtype=torch.long, device=device),
                torch.as_tensor(data.event_time[er], dtype=torch.float64, device=device),
                torch.as_tensor(data.rate.anchor_time[ar], dtype=torch.float64, device=device),
                torch.as_tensor(data.burden_mark[er], dtype=torch.float32, device=device),
                torch.as_tensor(data.grammar_mark[er], dtype=torch.float32, device=device),
            ))

    def __call__(self) -> Tensor:
        values, rows = [], []
        for ar, event_time, query_time, burden, grammar in self.chains:
            output = self.model(event_time, burden, grammar)
            values.append(dual_stream_features_at_queries(output, event_time, query_time))
            rows.append(ar)
        if not rows:
            raise ValueError("no event-bearing carry segment for event-state queries")
        row = torch.cat(rows)
        value = torch.cat(values)
        if torch.unique(row).numel() != row.numel():
            raise ValueError("event-state query rows overlap")
        # As above, an event-free but observed segment carries a defined zero
        # event-history state.  index_copy keeps gradients from every
        # event-bearing row while restoring the complete anchor axis.
        full = value.new_zeros((self.n_query, value.shape[1]))
        return full.index_copy(0, row, value)


def _grid_features_at_queries(output, frame_time: Tensor, query_time: Tensor) -> Tensor:
    """State after the last fully observed grid bin, propagated to each query.

    A frame stamped at ``t`` summarises only events in ``[t-grid, t)``.  It is
    therefore visible at a query exactly at ``t``; events at the query itself
    are never included.
    """

    predecessor = torch.searchsorted(frame_time, query_time, right=True) - 1
    valid = predecessor >= 0
    safe = predecessor.clamp_min(0)
    lag = (query_time - frame_time[safe]).clamp_min(0.0)
    tau = output.tau_seconds.to(device=query_time.device, dtype=torch.float64)
    phi = torch.exp(-lag[..., None] / tau).to(output.post_burden.dtype)
    burden = phi[..., None] * output.post_burden[safe]
    mass = phi[..., None] * output.post_grammar_mass[safe]
    grammar = output.post_grammar_composition[safe]
    burden = torch.where(valid[..., None, None], burden, torch.zeros_like(burden))
    mass = torch.where(valid[..., None, None], mass, torch.zeros_like(mass))
    grammar = torch.where(valid[..., None, None], grammar, torch.zeros_like(grammar))
    return torch.cat(
        (burden.flatten(-2), grammar.flatten(-2), torch.log1p(mass).flatten(-2)), dim=-1
    )


class GridEventStateComputer:
    """Five-minute hierarchical observer built from completed causal bins.

    The recurrent depth is the number of physical-time bins rather than the
    number of IEDs.  Burden is summed within a bin.  Conditional grammar is a
    count-weighted mean with the event count retained as reliability mass.
    """

    def __init__(
        self,
        data: H1SubjectData,
        model: DualStreamEventCTSSM,
        device: torch.device,
        *,
        grid_seconds: float = 300.0,
        query_time: np.ndarray | None = None,
        query_segment: np.ndarray | None = None,
    ) -> None:
        if grid_seconds <= 0:
            raise ValueError("grid_seconds must be positive")
        self.model = model
        self.device = device
        self.grid_seconds = float(grid_seconds)
        qtime = data.rate.anchor_time if query_time is None else np.asarray(query_time, dtype=np.float64)
        qsegment = data.rate.segment if query_segment is None else np.asarray(query_segment, dtype=np.int64)
        if qtime.shape != qsegment.shape:
            raise ValueError("query_time and query_segment must have identical shape")
        self.n_query = int(qtime.size)
        self.chains = []
        for segment in np.unique(qsegment):
            qr = np.flatnonzero(qsegment == segment)
            er = np.flatnonzero(data.event_segment == segment)
            if qr.size == 0:
                continue
            lo = float(data.rate.segment_bounds[int(segment), 0])
            last_query = float(np.max(qtime[qr]))
            first_end = (math.floor(lo / self.grid_seconds) + 1.0) * self.grid_seconds
            if first_end > last_query + 1e-6:
                # Queries before the first completed bin correctly receive a
                # zero state; keep a single zero-weight frame after them.
                frame_time = np.asarray([first_end], dtype=np.float64)
            else:
                n_frame = int(math.floor((last_query - first_end) / self.grid_seconds)) + 1
                frame_time = first_end + self.grid_seconds * np.arange(n_frame, dtype=np.float64)
            event_time = data.event_time[er]
            left_edges = frame_time - self.grid_seconds
            left = np.searchsorted(event_time, left_edges, side="left")
            right = np.searchsorted(event_time, frame_time, side="left")
            count = (right - left).astype(np.float32)
            burden = np.zeros((frame_time.size, data.burden_mark.shape[1]), dtype=np.float32)
            grammar = np.zeros((frame_time.size, data.grammar_mark.shape[1]), dtype=np.float32)
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                if b > a:
                    burden[index] = data.burden_mark[er[a:b]].mean(axis=0)
                    grammar[index] = data.grammar_mark[er[a:b]].mean(axis=0)
            self.chains.append((
                torch.as_tensor(qr, dtype=torch.long, device=device),
                torch.as_tensor(frame_time, dtype=torch.float64, device=device),
                torch.as_tensor(qtime[qr], dtype=torch.float64, device=device),
                torch.as_tensor(burden, dtype=torch.float32, device=device),
                torch.as_tensor(grammar, dtype=torch.float32, device=device),
                torch.as_tensor(count, dtype=torch.float32, device=device),
                lo,
            ))

    def __call__(self) -> Tensor:
        values, rows = [], []
        for row, frame_time, query_time, burden, grammar, count, initial_time in self.chains:
            output = self.model(
                frame_time, burden, grammar, event_weight=count,
                initial_time=initial_time,
            )
            values.append(_grid_features_at_queries(output, frame_time, query_time))
            rows.append(row)
        if not rows:
            raise ValueError("no grid-state query chains")
        row = torch.cat(rows)
        value = torch.cat(values)
        ordered = torch.argsort(row)
        if row.numel() != self.n_query or not torch.equal(row[ordered], torch.arange(self.n_query, device=row.device)):
            raise ValueError("grid-state queries are not a complete one-to-one partition")
        return value[ordered]


def _endpoint_losses(
    prediction: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
    valid: Mapping[str, Tensor],
    exposure: Tensor,
    dispersion: Tensor,
    rows: Tensor,
) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    count_valid = valid["count"][rows]
    log_mu = prediction["count"][rows, :, 0] + torch.log((exposure[rows] / 60.0).clamp_min(1e-6))
    count_loss = negative_binomial_nll(target["count"][rows, :, 0], log_mu, dispersion)
    result["count"] = _equal_horizon_mean(count_loss, count_valid)
    for name in ("community", "coupling", "mixture"):
        mask = valid[name][rows]
        logp = torch.log_softmax(prediction[name][rows], dim=-1)
        loss = -(target[name][rows] * logp).sum(-1)
        result[name] = _equal_horizon_mean(loss, mask)
    for name in ("burden", "embedding", "mark"):
        mask = valid[name][rows]
        loss = (prediction[name][rows] - target[name][rows]).square().mean(-1)
        result[name] = _equal_horizon_mean(loss, mask)
    return result


def _equal_horizon_mean(loss: Tensor, mask: Tensor) -> Tensor:
    """Give every estimable physical horizon equal weight.

    Pooling all valid cells would silently let short horizons dominate because
    they have more eligible anchors.  This helper first averages within each
    horizon and then averages the horizon estimates.
    """

    values = []
    for horizon in range(loss.shape[1]):
        use = mask[:, horizon]
        if bool(use.any()):
            values.append(loss[use, horizon].mean())
    if not values:
        return loss.sum() * 0.0
    return torch.stack(values).mean()


def _selection_score(
    prediction: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
    valid: Mapping[str, Tensor],
    exposure: Tensor,
    dispersion: Tensor,
    selection: Tensor,
    selection_np: np.ndarray,
    horizons_seconds: tuple[float, ...],
) -> dict[str, Any]:
    losses = _endpoint_losses(prediction, target, valid, exposure, dispersion, selection)
    by_horizon: dict[str, Any] = {}
    for h, seconds in enumerate(horizons_seconds):
        horizon_valid = {
            name: mask & (
                torch.arange(mask.shape[1], device=mask.device)[None, :] == h
            )
            for name, mask in valid.items()
        }
        if not bool(horizon_valid["count"][selection].any()):
            by_horizon[str(int(seconds))] = None
            continue
        horizon_losses = _endpoint_losses(
            prediction, target, horizon_valid, exposure, dispersion, selection
        )
        by_horizon[str(int(seconds))] = {
            "total": float(_weighted_total(horizon_losses)),
            "endpoints": {name: float(value) for name, value in horizon_losses.items()},
            "n": {
                name: int(horizon_valid[name][selection_np, h].sum())
                for name in horizon_valid
            },
        }
    return {
        "total": float(_weighted_total(losses)),
        "endpoints": {name: float(value) for name, value in losses.items()},
        "n_by_horizon": {
            name: [int(mask[selection_np, h].sum()) for h in range(len(horizons_seconds))]
            for name, mask in valid.items()
        },
        "by_horizon": by_horizon,
    }


def _weighted_total(losses: Mapping[str, Tensor]) -> Tensor:
    weights = {
        "count": 1.0, "burden": 0.5, "community": 1.0, "coupling": 1.0,
        "mixture": 1.0, "embedding": 0.5, "mark": 0.5,
    }
    return sum(weights[name] * value for name, value in losses.items()) / sum(weights.values())


def _train_stage(
    *,
    stage: str,
    readout: NestedH1Readout,
    q: Tensor,
    bmark: Tensor,
    state_computer: EventStateComputer | None,
    fixed_random: Tensor | None,
    target: Mapping[str, Tensor],
    valid: Mapping[str, Tensor],
    exposure: Tensor,
    fit_rows: Tensor,
    inner_rows: Tensor,
    config: H1TrainConfig,
) -> dict[str, Any]:
    for parameter in readout.parameters(): parameter.requires_grad_(False)
    parameters = readout.parameters_for(stage)
    for parameter in parameters: parameter.requires_grad_(True)
    if stage == "state":
        assert state_computer is not None
        parameters += list(state_computer.model.parameters())
        for parameter in state_computer.model.parameters(): parameter.requires_grad_(True)
        groups = [
            {"params": readout.parameters_for("state"), "lr": config.lr_state_head},
            {"params": list(state_computer.model.parameters()), "lr": config.lr_state},
        ]
        optimizer = torch.optim.AdamW(
            groups, weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon,
        )
        max_steps = config.max_steps_state
    else:
        lr = {"q": config.lr_q, "bmark": config.lr_bmark, "random": config.lr_random_head}[stage]
        max_steps = {"q": config.max_steps_q, "bmark": config.max_steps_bmark, "random": config.max_steps_random}[stage]
        decay = config.weight_decay
        if stage == "bmark" and config.weight_decay_bmark is not None:
            decay = float(config.weight_decay_bmark)
        optimizer = torch.optim.AdamW(
            parameters, lr=lr, weight_decay=decay,
            betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon,
        )

    def predict() -> dict[str, Tensor]:
        if stage == "q": return readout.predict(q)
        if stage == "bmark": return readout.predict(q, bmark=bmark)
        if stage == "random": return readout.predict(q, bmark=bmark, random_state=fixed_random)
        return readout.predict(q, bmark=bmark, state=state_computer())

    # Every added arm is nested above an already fitted parent.  A non-zero
    # readout initialisation is useful for upstream gradients, but it must not
    # replace the parent as the step-zero candidate: otherwise a lucky random
    # offset can be called a fitted B_mark/state gain, or an unlucky one can
    # make the child arm weaker than its parent.  Keep an explicit zero-
    # increment parent snapshot and require training to beat that snapshot.
    parent_prediction = None
    parent_readout = None
    if stage != "q":
        if stage == "bmark":
            parent_prediction = readout.predict(q)
        else:
            parent_prediction = readout.predict(q, bmark=bmark)
        parent_readout = {
            key: value.detach().cpu().clone()
            for key, value in readout.state_dict().items()
        }
        prefix = f"{stage}."
        for key in tuple(parent_readout):
            if key.startswith(prefix):
                parent_readout[key].zero_()

    with torch.no_grad():
        optimiser_initial = float(_weighted_total(_endpoint_losses(
            predict(), target, valid, exposure, readout.log_dispersion, inner_rows
        )))
        parent = (
            optimiser_initial if parent_prediction is None else
            float(_weighted_total(_endpoint_losses(
                parent_prediction, target, valid, exposure,
                readout.log_dispersion, inner_rows,
            )))
        )
    best, best_step, stale = parent, 0, 0
    best_readout = (
        {k: v.detach().cpu().clone() for k, v in readout.state_dict().items()}
        if parent_readout is None else parent_readout
    )
    best_state = None if state_computer is None else {
        k: v.detach().cpu().clone() for k, v in state_computer.model.state_dict().items()
    }
    parameter_start = [value.detach().cpu().clone() for value in parameters]
    active_ids = {id(value) for value in parameters}
    named = [(f'readout.{name}', value) for name, value in readout.named_parameters()
             if id(value) in active_ids]
    if stage == 'state':
        named += [(f'observer.{name}', value) for name, value in state_computer.model.named_parameters()
                  if id(value) in active_ids]
    initial_by_name = {name: value.detach().cpu().clone() for name, value in named}
    gradient_audit = {name: {'shape': list(value.shape), 'parameters': value.numel(),
                            'first_nonzero_step': None, 'nonzero_steps': 0,
                            'gradient_norm_sum': 0.0, 'gradient_norm_max': 0.0}
                      for name, value in named}
    first_step_gradient_norm = None
    peak_parameter_delta = 0.0
    history = [{
        "step": 0,
        "inner_loss": parent,
        "parent_inner_loss": parent,
        "optimiser_initial_inner_loss": optimiser_initial,
    }]
    for step in range(1, max_steps + 1):
        if stage == "state" and config.warmup_steps_state > 0:
            fraction = min(1.0, step / float(config.warmup_steps_state))
            optimizer.param_groups[0]["lr"] = config.lr_state_head * fraction
            optimizer.param_groups[1]["lr"] = config.lr_state * fraction
        if stage == "bmark" and config.warmup_steps_bmark > 0:
            fraction = min(1.0, step / float(config.warmup_steps_bmark))
            optimizer.param_groups[0]["lr"] = config.lr_bmark * fraction
        optimizer.zero_grad(set_to_none=True)
        prediction = predict()
        fit_losses = _endpoint_losses(prediction, target, valid, exposure, readout.log_dispersion, fit_rows)
        loss = _weighted_total(fit_losses)
        if not torch.isfinite(loss): raise FloatingPointError(f"{stage}: non-finite fit loss")
        loss.backward()
        gradient_norms = torch.stack([
            q.new_zeros(()) if value.grad is None else value.grad.detach().norm()
            for _name, value in named
        ]).cpu().tolist()
        for (name, _value), norm in zip(named, gradient_norms):
            if not np.isfinite(norm):
                raise FloatingPointError(f'{stage}: non-finite gradient in {name}')
            audit = gradient_audit[name]
            audit['gradient_norm_sum'] += norm
            audit['gradient_norm_max'] = max(audit['gradient_norm_max'], norm)
            if norm > 0:
                audit['nonzero_steps'] += 1
                if audit['first_nonzero_step'] is None:
                    audit['first_nonzero_step'] = step
        if step == 1:
            first_step_gradient_norm = float(torch.sqrt(sum(
                parameter.grad.detach().float().square().sum()
                for parameter in parameters if parameter.grad is not None
            )))
        torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
        optimizer.step()
        if step % config.validate_every == 0 or step == max_steps:
            peak_parameter_delta = max(
                peak_parameter_delta,
                max(
                    float((value.detach().cpu() - start).abs().max())
                    for value, start in zip(parameters, parameter_start)
                ),
            )
            with torch.no_grad():
                inner_losses = _endpoint_losses(predict(), target, valid, exposure, readout.log_dispersion, inner_rows)
                value = float(_weighted_total(inner_losses))
            history.append({
                "step": step,
                "fit_loss": float(loss.detach()),
                "fit_endpoints": {k: float(v.detach()) for k, v in fit_losses.items()},
                "inner_loss": value,
                "inner_endpoints": {k: float(v) for k, v in inner_losses.items()},
            })
            if np.isfinite(value) and value < best - 1e-5:
                best, best_step, stale = value, step, 0
                best_readout = {k: v.detach().cpu().clone() for k, v in readout.state_dict().items()}
                if state_computer is not None:
                    best_state = {k: v.detach().cpu().clone() for k, v in state_computer.model.state_dict().items()}
            else:
                stale += 1
            if stale >= config.patience_checks:
                break
    readout.load_state_dict({k: v.to(q.device) for k, v in best_readout.items()})
    if state_computer is not None and best_state is not None:
        state_computer.model.load_state_dict({k: v.to(q.device) for k, v in best_state.items()})
    for name, value in named:
        change = value.detach().cpu() - initial_by_name[name]
        gradient_audit[name].update(
            selected_delta_max_abs=float(change.abs().max()), selected_delta_l2=float(change.norm()),
            gradient_norm_mean=gradient_audit[name]['gradient_norm_sum'] / max(history[-1]['step'], 1),
        )
    return {
        "stage": stage,
        "parameter_audit": gradient_audit,
        "optimizer_audit": {'name': 'AdamW', 'betas': [config.adam_beta1, config.adam_beta2],
                            'epsilon': config.adam_epsilon,
                            'final_group_learning_rates': [group['lr'] for group in optimizer.param_groups],
                            'group_weight_decays': [group['weight_decay'] for group in optimizer.param_groups],
                            'gradient_clip': config.gradient_clip,
                            'batch_contract': 'full FIT anchors; complete causal carry segments for observer'},
        # Backward-compatible name, now with the scientifically correct
        # meaning: the loss of the fitted parent arm, not a random child start.
        "initial_inner_loss": parent,
        "parent_inner_loss": parent,
        "optimiser_initial_inner_loss": optimiser_initial,
        "best_inner_loss": best,
        "gain_over_parent": parent - best,
        "selected_step": best_step,
        "steps_run": history[-1]["step"],
        "selected_at_init": best_step == 0,
        "selected_parent_parity": best_step == 0,
        "first_step_gradient_norm": first_step_gradient_norm,
        "peak_parameter_delta_from_stage_start": peak_parameter_delta,
        "selected_at_budget_edge": best_step == max_steps,
        "training_budget_exhausted": bool(
            history[-1]["step"] == max_steps and stale < config.patience_checks
        ),
        "terminated_by_patience": bool(stale >= config.patience_checks),
        "history": history,
    }


def _block_shift(state: Tensor, data: H1SubjectData, rows: np.ndarray, minimum_seconds: float) -> tuple[Tensor, Tensor]:
    shifted = state.clone()
    valid = torch.zeros(state.shape[0], dtype=torch.bool, device=state.device)
    for segment in np.unique(data.rate.segment[rows]):
        rr = rows[data.rate.segment[rows] == segment]
        if rr.size < 4: continue
        donor = np.roll(rr, max(1, rr.size // 2))
        ok = np.abs(data.rate.anchor_time[donor] - data.rate.anchor_time[rr]) >= float(minimum_seconds)
        if np.any(ok):
            target_t = torch.as_tensor(rr[ok], dtype=torch.long, device=state.device)
            donor_t = torch.as_tensor(donor[ok], dtype=torch.long, device=state.device)
            shifted[target_t] = state[donor_t]
            valid[target_t] = True
    return shifted, valid


def train_h1_subject(
    data: H1SubjectData,
    config: H1TrainConfig,
    *,
    device: torch.device,
    out_dir: Path,
    state_mode: str = "event",
    grid_seconds: float = 300.0,
    stages_only: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    started = time.time()
    torch.manual_seed(int(config.seed)); np.random.seed(int(config.seed))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fit_np = np.flatnonzero(data.rate.phase == "FIT")
    inner_np = np.flatnonzero(data.rate.phase == "INNER")
    selection_np = np.flatnonzero(data.rate.phase == "SELECTION")
    fit = torch.as_tensor(fit_np, dtype=torch.long, device=device)
    inner = torch.as_tensor(inner_np, dtype=torch.long, device=device)
    selection = torch.as_tensor(selection_np, dtype=torch.long, device=device)
    q, q_centre, q_scale = _standardise_features(data.rate.q_raw, fit_np, device)
    with torch.no_grad(): fixed_raw = _fixed_features(data, device, config.taus_seconds)
    bmark, b_centre, b_scale = _standardise_features(fixed_raw, fit_np, device)
    target, valid, target_scales = _target_bundle(data, device)
    exposure = torch.as_tensor(data.rate.target_exposure_seconds, dtype=torch.float32, device=device)
    widths = {
        "count": 1,
        "burden": data.burden_mark.shape[1],
        "community": data.grammar_dictionary.n_communities,
        "coupling": data.grammar_dictionary.n_communities ** 2,
        "mixture": data.grammar_dictionary.n_repertoires,
        "embedding": data.grammar_dictionary.event_repertoire_embedding.shape[1],
        "mark": data.grammar_mark.shape[1],
    }
    observer = DualStreamEventCTSSM(
        data.burden_mark.shape[1], data.grammar_mark.shape[1],
        taus_seconds=config.taus_seconds,
        burden_channels_per_tau=config.burden_channels_per_tau,
        grammar_channels_per_tau=config.grammar_channels_per_tau,
    ).to(device)
    if state_mode == "event":
        state_computer = EventStateComputer(data, observer, device)
        state_label = "S_event"
    elif state_mode == "grid":
        state_computer = GridEventStateComputer(
            data, observer, device, grid_seconds=grid_seconds
        )
        state_label = "S_grid"
    else:
        raise ValueError("state_mode must be event or grid")
    random_observer = DualStreamEventCTSSM(
        data.burden_mark.shape[1], data.grammar_mark.shape[1],
        taus_seconds=config.taus_seconds,
        burden_channels_per_tau=config.burden_channels_per_tau,
        grammar_channels_per_tau=config.grammar_channels_per_tau,
    ).to(device)
    for parameter in random_observer.parameters(): parameter.requires_grad_(False)
    random_computer = (
        EventStateComputer(data, random_observer, device)
        if state_mode == "event" else
        GridEventStateComputer(data, random_observer, device, grid_seconds=grid_seconds)
    )
    with torch.no_grad(): random_raw = random_computer()
    random_state, random_centre, random_scale = _standardise_features(random_raw, fit_np, device)
    bmark_burden_dim = len(config.taus_seconds) * data.burden_mark.shape[1]
    state_burden_dim = len(config.taus_seconds) * config.burden_channels_per_tau
    readout = NestedH1Readout(
        q.shape[1],
        bmark_burden_dim,
        bmark.shape[1] - bmark_burden_dim,
        state_burden_dim,
        observer.state_dim - state_burden_dim,
        widths,
        len(data.rate.horizons_seconds),
        state_readout_init_std=config.state_readout_init_std,
        bmark_readout_init_std=config.bmark_readout_init_std,
    ).to(device)
    stages = {}
    stages["q"] = _train_stage(
        stage="q", readout=readout, q=q, bmark=bmark, state_computer=None, fixed_random=None,
        target=target, valid=valid, exposure=exposure, fit_rows=fit, inner_rows=inner, config=config,
    )
    stages["bmark"] = _train_stage(
        stage="bmark", readout=readout, q=q, bmark=bmark, state_computer=None, fixed_random=None,
        target=target, valid=valid, exposure=exposure, fit_rows=fit, inner_rows=inner, config=config,
    )
    # A control-only sweep must not spend most of its time fitting a random
    # observer that cannot affect either q or B_mark.  Keep the random stage in
    # every scientific H1 run, and run it only when explicitly requested by a
    # stage-only diagnostic.
    if stages_only is None or "random" in stages_only:
        stages["random"] = _train_stage(
            stage="random", readout=readout, q=q, bmark=bmark, state_computer=None, fixed_random=random_state,
            target=target, valid=valid, exposure=exposure, fit_rows=fit, inner_rows=inner, config=config,
        )
    if stages_only is not None:
        # Baseline-trainability sweep: the transparent-baseline stage is fitted
        # after the rate stage and touches only its own parameters, so its
        # outcome does not depend on any state-side hyperparameter.  Running it
        # alone gives the control arm the same recipe search as the model arm
        # without paying for a state fit that is then discarded.
        missing = set(stages_only) - set(stages)
        if missing:
            raise ValueError(f"stages_only requested unfitted stages: {sorted(missing)}")
        card = {
            "format": "group_event_state_v0_3_7_h1_stage_sweep_card_v1",
            "subject": data.subject, "seed": int(config.seed), "config": asdict(config),
            "observer_mode": state_mode, "stages": {k: stages[k] for k in stages_only},
            "code_provenance": _code_provenance(),
            "selection_scores_used": False, "selection_partition": "INNER",
            "development_targets_read": False, "seizure_targets_read": False,
            "sealed_partition_opened": False,
            "elapsed_seconds": time.time() - started,
        }
        atomic_json(out_dir / "card.json", card)
        return card
    stages["state"] = _train_stage(
        stage="state", readout=readout, q=q, bmark=bmark, state_computer=state_computer, fixed_random=None,
        target=target, valid=valid, exposure=exposure, fit_rows=fit, inner_rows=inner, config=config,
    )
    with torch.no_grad():
        learned = state_computer()
        fit_mean = learned[fit].mean(0, keepdim=True)
        constant = fit_mean.expand_as(learned)
        shifted, shift_valid = _block_shift(
            learned, data, selection_np, max(data.rate.horizons_seconds)
        )
        arms = {
            "B_rate": readout.predict(q),
            "B_mark": readout.predict(q, bmark=bmark),
            "random_frozen": readout.predict(q, bmark=bmark, random_state=random_state),
            state_label: readout.predict(q, bmark=bmark, state=learned),
            f"{state_label}_constant": readout.predict(q, bmark=bmark, state=constant),
            f"{state_label}_block_shift": readout.predict(q, bmark=bmark, state=shifted),
        }
        selection_scores = {}
        for name, prediction in arms.items():
            rows = selection
            local_valid = valid
            if name == f"{state_label}_block_shift":
                local_valid = {k: v & shift_valid[:, None] for k, v in valid.items()}
            if name == f"{state_label}_block_shift" and not bool(local_valid["count"][selection].any()):
                selection_scores[name] = None
                continue
            selection_scores[name] = _selection_score(
                prediction, target, local_valid, exposure, readout.log_dispersion,
                selection, selection_np, tuple(data.rate.horizons_seconds),
            )
        paired_valid = {k: v & shift_valid[:, None] for k, v in valid.items()}
        if bool(paired_valid["count"][selection].any()):
            selection_scores[f"{state_label}_paired"] = _selection_score(
                arms[state_label], target, paired_valid, exposure,
                readout.log_dispersion, selection, selection_np,
                tuple(data.rate.horizons_seconds),
            )
        else:
            selection_scores[f"{state_label}_paired"] = None

        # A horizon-specific shift is the estimable time-null.  Requiring every
        # donor to be farther than the maximum (8 h) made even the 0.5 h test
        # structurally empty in some patients.
        time_shift_by_horizon: dict[str, Any] = {}
        for h, seconds in enumerate(data.rate.horizons_seconds):
            shifted_h, valid_h = _block_shift(learned, data, selection_np, float(seconds))
            endpoint_valid = {
                name: mask & valid_h[:, None] & (
                    torch.arange(mask.shape[1], device=mask.device)[None, :] == h
                )
                for name, mask in valid.items()
            }
            key = str(int(seconds))
            if not bool(endpoint_valid["count"][selection].any()):
                time_shift_by_horizon[key] = None
                continue
            correct_h = _selection_score(
                arms[state_label], target, endpoint_valid, exposure,
                readout.log_dispersion, selection, selection_np,
                tuple(data.rate.horizons_seconds),
            )
            shifted_prediction = readout.predict(q, bmark=bmark, state=shifted_h)
            shifted_h_score = _selection_score(
                shifted_prediction, target, endpoint_valid, exposure,
                readout.log_dispersion, selection, selection_np,
                tuple(data.rate.horizons_seconds),
            )
            time_shift_by_horizon[key] = {
                "correct": correct_h["by_horizon"][key],
                "shifted": shifted_h_score["by_horizon"][key],
                "gain": shifted_h_score["by_horizon"][key]["total"] - correct_h["by_horizon"][key]["total"],
            }
    checkpoint = out_dir / "checkpoint.pt"
    torch.save(data, out_dir / 'training_input_bundle.pt')
    torch.save({
        "observer": observer.state_dict(), "readout": readout.state_dict(),
        "observer_mode": state_mode, "grid_seconds": float(grid_seconds),
        "config": asdict(config), "widths": widths,
        "q_centre": q_centre, "q_scale": q_scale,
        "bmark_centre": b_centre, "bmark_scale": b_scale,
        "random_centre": random_centre, "random_scale": random_scale,
        "target_scales": target_scales,
        'training_input_bundle': str(out_dir / 'training_input_bundle.pt'),
        'training_input_bundle_sha256': sha256_file(out_dir / 'training_input_bundle.pt'),
    }, checkpoint)
    np.savez_compressed(
        out_dir / "trajectory_and_targets.npz",
        anchor_time=data.rate.anchor_time,
        phase=data.rate.phase,
        horizons_seconds=np.asarray(data.rate.horizons_seconds),
        learned_state=learned.cpu().numpy(),
        fixed_mark_state=bmark.cpu().numpy(),
        target_count=data.rate.target_count,
        target_valid=data.rate.target_valid.astype(np.uint8),
        target_exposure_seconds=data.rate.target_exposure_seconds,
        block_shift_valid=shift_valid.cpu().numpy().astype(np.uint8),
    )
    b = selection_scores["B_mark"]["total"]
    s = selection_scores[state_label]["total"]

    # Held-out evidence is counted in non-overlapping physical windows, not in
    # the overlapping five-minute anchors that share almost all of their span.
    selection_time = data.rate.anchor_time[selection_np]
    selection_segment = data.rate.segment[selection_np]
    independent_windows = {}
    for h, seconds in enumerate(data.rate.horizons_seconds):
        estimable = valid["count"][selection, h].detach().cpu().numpy().astype(bool)
        independent_windows[str(int(seconds))] = {
            "anchor_rows": int(estimable.sum()),
            "independent_windows": independent_window_count(
                selection_time[estimable], float(seconds),
                segment=selection_segment[estimable],
            ),
        }
    selection_span = (
        float(selection_time.max() - selection_time.min()) if selection_time.size else 0.0
    )
    slowest_tau = float(max(config.taus_seconds))
    selection_window_audit = {
        "selection_span_seconds": selection_span,
        "slowest_memory_tau_seconds": slowest_tau,
        "slowest_memory_exceeds_selection_span": bool(slowest_tau > selection_span),
        "note": (
            "a memory channel slower than the held-out span cannot be separated from a "
            "constant inside that span; compare against the constant-state arm before "
            "reading any gain at the longest horizons"
        ),
    }
    shift_gains = [entry["gain"] for entry in time_shift_by_horizon.values() if entry is not None]
    card = {
        "format": "group_event_state_v0_3_7_h1_shared_observer_equal_horizon_card_v4",
        "subject": data.subject,
        "seed": int(config.seed),
        "config": asdict(config),
        "horizons_seconds": list(data.rate.horizons_seconds),
        "shared_producer_across_horizons": True,
        "horizon_weighting": "equal weight across estimable physical horizons",
        "state_semantics": "S_obs_predictive_observer_not_Z_phys",
        "observer_mode": state_mode,
        "grid_seconds": float(grid_seconds) if state_mode == "grid" else None,
        "representation_provenance": data.representation_provenance,
        "widths": widths,
        "stages": stages,
        "selection_scores": selection_scores,
        "time_shift_by_horizon": time_shift_by_horizon,
        "independent_windows_by_horizon": independent_windows,
        "selection_window_audit": selection_window_audit,
        "code_provenance": _code_provenance(),
        "primary_contrasts": {
            "B_mark_gain_over_B_rate": (
                selection_scores["B_rate"]["total"] - selection_scores["B_mark"]["total"]
            ),
            f"{state_label}_gain_over_B_mark": b - s,
            f"{state_label}_gain_over_random": selection_scores["random_frozen"]["total"] - s,
            "constant_unexplained_gain": selection_scores[f"{state_label}_constant"]["total"] - s,
            "correct_time_gain_on_shift_eligible": (
                float(np.mean(shift_gains)) if shift_gains else None
            ),
        },
        "checkpoint_path": str(checkpoint),
        "trajectory_path": str(out_dir / "trajectory_and_targets.npz"),
        "maximum_training_time": float(data.rate.phase_boundaries["60pct"]),
        "elapsed_seconds": time.time() - started,
        "development_targets_read": False,
        "seizure_targets_read": False,
        "sealed_partition_opened": False,
    }
    atomic_json(out_dir / "card.json", card)
    return card
