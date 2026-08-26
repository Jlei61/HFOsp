"""Recorded-time point-process likelihood and calibration diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


LogIntensity = Callable[[torch.Tensor], torch.Tensor]


def gauss_legendre_rule(order: int = 8, *, dtype=torch.float64,
                        device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    if int(order) < 2:
        raise ValueError("quadrature order must be at least two")
    node, weight = np.polynomial.legendre.leggauss(int(order))
    return (
        torch.as_tensor(node, dtype=dtype, device=device),
        torch.as_tensor(weight, dtype=dtype, device=device),
    )


def integrate_intensity(log_intensity: LogIntensity,
                        segment_start: torch.Tensor,
                        segment_stop: torch.Tensor,
                        *, order: int = 8,
                        max_log_intensity: float = 30.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate intensity on each supplied recorded segment."""
    start = torch.as_tensor(segment_start)
    stop = torch.as_tensor(segment_stop, dtype=start.dtype, device=start.device)
    if start.ndim != 1 or stop.shape != start.shape or len(start) == 0:
        raise ValueError("recorded segments must be equal non-empty 1-D tensors")
    if not bool(torch.isfinite(start).all() and torch.isfinite(stop).all()):
        raise ValueError("non-finite recorded boundary")
    if not bool((stop > start).all()):
        raise ValueError("non-positive recorded segment")
    node, weight = gauss_legendre_rule(order, dtype=start.dtype, device=start.device)
    midpoint = 0.5 * (start + stop)
    half_width = 0.5 * (stop - start)
    time = midpoint[:, None] + half_width[:, None] * node[None, :]
    log_rate = log_intensity(time.reshape(-1)).reshape(len(start), int(order))
    if log_rate.shape != time.shape or not bool(torch.isfinite(log_rate).all()):
        raise ValueError("log-intensity callback returned invalid values")
    rate = torch.exp(torch.clamp(log_rate, max=float(max_log_intensity)))
    per_segment = half_width * torch.sum(rate * weight[None, :], dim=1)
    return per_segment.sum(), per_segment


@dataclass(frozen=True)
class PointProcessTerms:
    event_log_intensity: torch.Tensor
    survival_integral: torch.Tensor
    log_likelihood: torch.Tensor
    n_events: int
    recorded_seconds: float

    @property
    def nll_per_event(self) -> torch.Tensor:
        return -self.log_likelihood / max(self.n_events, 1)


def point_process_log_likelihood(event_time: torch.Tensor,
                                 segment_start: torch.Tensor,
                                 segment_stop: torch.Tensor,
                                 log_intensity: LogIntensity,
                                 *, quadrature_order: int = 8) -> PointProcessTerms:
    event = torch.as_tensor(event_time)
    start = torch.as_tensor(segment_start, dtype=event.dtype, device=event.device)
    stop = torch.as_tensor(segment_stop, dtype=event.dtype, device=event.device)
    if event.ndim != 1 or len(event) == 0:
        raise ValueError("event times must be a non-empty 1-D tensor")
    if not bool(torch.isfinite(event).all()) or not bool((event[1:] >= event[:-1]).all()):
        raise ValueError("event times are non-finite or non-chronological")
    inside = torch.zeros(len(event), dtype=torch.bool, device=event.device)
    for left, right in zip(start, stop):
        inside |= (event >= left) & (event < right)
    if not bool(inside.all()):
        raise ValueError("an event lies outside recorded coverage")
    event_term = log_intensity(event)
    if event_term.shape != event.shape or not bool(torch.isfinite(event_term).all()):
        raise ValueError("invalid event log-intensity")
    integral, _ = integrate_intensity(
        log_intensity, start, stop, order=quadrature_order
    )
    summed = event_term.sum()
    return PointProcessTerms(
        event_log_intensity=summed,
        survival_integral=integral,
        log_likelihood=summed - integral,
        n_events=int(len(event)),
        recorded_seconds=float((stop - start).sum().detach().cpu()),
    )


def rescaled_interevent_integrals(event_time: torch.Tensor,
                                  segment_start: torch.Tensor,
                                  segment_stop: torch.Tensor,
                                  log_intensity: LogIntensity,
                                  *, quadrature_order: int = 8) -> torch.Tensor:
    """Integrated hazards between consecutive events over recorded portions only."""
    event = torch.as_tensor(event_time)
    start = torch.as_tensor(segment_start, dtype=event.dtype, device=event.device)
    stop = torch.as_tensor(segment_stop, dtype=event.dtype, device=event.device)
    values = []
    for left, right in zip(event[:-1], event[1:]):
        seg_left = torch.maximum(start, left)
        seg_right = torch.minimum(stop, right)
        keep = seg_right > seg_left
        if not bool(keep.any()):
            values.append(event.new_zeros(()))
            continue
        value, _ = integrate_intensity(
            log_intensity, seg_left[keep], seg_right[keep], order=quadrature_order
        )
        values.append(value)
    return torch.stack(values) if values else event.new_empty((0,))
