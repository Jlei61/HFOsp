"""Transparent physical-time mark-EWMA baselines for v0.3.7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from .ctssm import affine_associative_scan, event_time_deltas


@dataclass(frozen=True)
class MarkEWMAOutput:
    burden_sum: Tensor
    grammar_composition: Tensor
    grammar_effective_mass: Tensor
    state_features: Tensor


def mark_ewma_features_at_queries(
    output: MarkEWMAOutput,
    event_time: Tensor,
    query_time: Tensor,
    *,
    taus_seconds: Sequence[float],
) -> Tensor:
    """Causal fixed-EWMA features at arbitrary anchors for one chain."""

    if event_time.ndim != 1 or query_time.ndim != 1:
        raise ValueError("query helper currently accepts one unbatched chain")
    if output.burden_sum.ndim != 3:
        raise ValueError("query helper requires unbatched EWMA output")
    predecessor = torch.searchsorted(event_time, query_time, right=False) - 1
    valid = predecessor >= 0
    safe = predecessor.clamp_min(0)
    tau = torch.as_tensor(tuple(float(v) for v in taus_seconds), device=query_time.device, dtype=torch.float64)
    lag = (query_time - event_time[safe]).clamp_min(0.0)
    phi = torch.exp(-lag.to(torch.float64)[..., None] / tau).to(output.burden_sum.dtype)
    burden = phi[..., None] * output.burden_sum[safe]
    mass = phi[..., None] * output.grammar_effective_mass[safe]
    grammar = output.grammar_composition[safe]
    burden = torch.where(valid[..., None, None], burden, torch.zeros_like(burden))
    mass = torch.where(valid[..., None, None], mass, torch.zeros_like(mass))
    grammar = torch.where(valid[..., None, None], grammar, torch.zeros_like(grammar))
    return torch.cat(
        (burden.flatten(-2), grammar.flatten(-2), torch.log1p(mass).flatten(-2)), dim=-1
    )


def _expand_time_feature(values: Tensor, n_tau: int) -> Tensor:
    return values[..., None, :].expand(*values.shape[:-1], n_tau, values.shape[-1])


def fixed_mark_ewma(
    event_time: Tensor,
    burden_mark: Tensor,
    grammar_mark: Tensor,
    *,
    taus_seconds: Sequence[float],
    initial_time: Tensor | float | None = None,
    epsilon: float = 1e-6,
) -> MarkEWMAOutput:
    """Compute additive burden and rate-normalised conditional grammar history.

    The output is deterministic and parameter free.  Grammar composition is a
    decayed pseudo-count mean; log effective mass is supplied separately so a
    predictor can represent uncertainty without turning event density into the
    grammar vector's norm.
    """

    squeeze_batch = burden_mark.ndim == 2
    if squeeze_batch:
        burden_mark = burden_mark[:, None, :]
        grammar_mark = grammar_mark[:, None, :]
    if burden_mark.ndim != 3 or grammar_mark.ndim != 3:
        raise ValueError("marks must have [time, feature] or [time, batch, feature] shape")
    if burden_mark.shape[:2] != grammar_mark.shape[:2]:
        raise ValueError("burden and grammar marks must share time/batch axes")
    taus = torch.as_tensor(tuple(float(v) for v in taus_seconds), dtype=torch.float64, device=burden_mark.device)
    if taus.ndim != 1 or taus.numel() == 0 or torch.any(taus <= 0):
        raise ValueError("taus_seconds must be positive")
    delta = event_time_deltas(event_time, burden_mark.shape[1], initial_time)
    phi = torch.exp(-delta[..., None] / taus).to(burden_mark.dtype)

    burden_q = _expand_time_feature(burden_mark, taus.numel())
    burden_phi = phi[..., None].expand_as(burden_q)
    _p_burden, burden_sum = affine_associative_scan(burden_phi, burden_q)

    grammar_q = _expand_time_feature(grammar_mark, taus.numel())
    grammar_phi = phi[..., None].expand_as(grammar_q)
    _p_grammar, grammar_sum = affine_associative_scan(grammar_phi, grammar_q)
    mass_q = torch.ones(*grammar_mark.shape[:2], taus.numel(), 1, dtype=grammar_mark.dtype, device=grammar_mark.device)
    mass_phi = phi[..., None]
    _p_mass, mass = affine_associative_scan(mass_phi, mass_q)
    composition = grammar_sum / mass.clamp_min(float(epsilon))
    features = torch.cat(
        (
            burden_sum.flatten(-2),
            composition.flatten(-2),
            torch.log1p(mass).flatten(-2),
        ),
        dim=-1,
    )
    if squeeze_batch:
        burden_sum, composition, mass, features = (
            value[:, 0] for value in (burden_sum, composition, mass, features)
        )
    return MarkEWMAOutput(
        burden_sum=burden_sum,
        grammar_composition=composition,
        grammar_effective_mass=mass,
        state_features=features,
    )
