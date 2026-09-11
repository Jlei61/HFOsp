"""Event-time likelihood for the v0.3 marked temporal point process."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PointProcessTerms:
    """Per-interval terms of ``integral(lambda) - log lambda(event)``."""

    event_nll: Tensor
    survival_integral: Tensor
    event_log_intensity: Tensor
    observed_seconds: Tensor


def interval_point_process_terms(
    lambda_start: Tensor,
    lambda_event: Tensor,
    dt_seconds: Tensor,
    observed: Tensor | None = None,
) -> PointProcessTerms:
    """Trapezoidal point-process NLL for observed inter-event intervals.

    ``lambda_start`` is the intensity immediately after the previous event (or
    at a coverage-segment start), and ``lambda_event`` is the intensity just
    before the current event.  Both must be positive.  ``observed=False`` makes
    an interval contribute exactly zero: a recording gap is not evidence that
    no IED happened.

    The approximation is intentionally local to a real inter-event interval.
    It is substantially finer than a one-minute Bernoulli grid for this cohort,
    where group events commonly occur every few seconds.
    """

    if lambda_start.shape != lambda_event.shape or lambda_start.shape != dt_seconds.shape:
        raise ValueError("lambda and dt tensors must have identical shapes")
    if observed is None:
        observed = torch.ones_like(dt_seconds, dtype=torch.bool)
    if observed.shape != dt_seconds.shape:
        raise ValueError("observed mask must match dt")
    observed = observed.to(torch.bool)
    dt = dt_seconds.to(torch.float32)
    if bool((dt[observed] < 0).any()):
        raise ValueError("observed intervals must have non-negative duration")
    start = lambda_start.to(torch.float32).clamp_min(1e-8)
    event = lambda_event.to(torch.float32).clamp_min(1e-8)
    integral = 0.5 * (start + event) * dt.clamp_min(0.0)
    log_event = torch.log(event)
    integral = torch.where(observed, integral, torch.zeros_like(integral))
    log_event = torch.where(observed, log_event, torch.zeros_like(log_event))
    return PointProcessTerms(
        event_nll=integral - log_event,
        survival_integral=integral,
        event_log_intensity=log_event,
        observed_seconds=torch.where(observed, dt, torch.zeros_like(dt)),
    )


def censored_interval_integral(
    lambda_start: Tensor,
    lambda_stop: Tensor,
    dt_seconds: Tensor,
    observed: Tensor | None = None,
) -> Tensor:
    """Survival-only contribution at an observed segment or split tail."""

    if observed is None:
        observed = torch.ones_like(dt_seconds, dtype=torch.bool)
    dt = dt_seconds.to(torch.float32)
    value = 0.5 * (
        lambda_start.to(torch.float32).clamp_min(1e-8)
        + lambda_stop.to(torch.float32).clamp_min(1e-8)
    ) * dt.clamp_min(0.0)
    return torch.where(observed.to(torch.bool), value, torch.zeros_like(value))
