"""Stable diagonal continuous-time observers for irregular group-event streams.

An event is an observation impulse.  Long observer memory is not evidence that
the event physically changed the brain; H3 uses a separate generative model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn


def _validate_affine_inputs(phi: Tensor, q: Tensor) -> None:
    if phi.shape != q.shape:
        raise ValueError(f"phi and q must have identical shapes, got {phi.shape} and {q.shape}")
    if phi.ndim < 2:
        raise ValueError("affine scans require [time, ..., state] tensors")
    if phi.shape[0] == 0:
        raise ValueError("affine scan needs at least one time step")
    if not torch.isfinite(phi).all() or not torch.isfinite(q).all():
        raise ValueError("affine scan inputs must be finite")


def compose_affine(earlier: tuple[Tensor, Tensor], later: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    """Compose diagonal affine transitions in chronological order.

    ``earlier`` maps x0 -> p1*x0+q1 and ``later`` maps x1 -> p2*x1+q2.
    """

    p1, q1 = earlier
    p2, q2 = later
    return p2 * p1, p2 * q1 + q2


def affine_sequential_scan(phi: Tensor, q: Tensor) -> tuple[Tensor, Tensor]:
    """Reference inclusive scan, retained as the numerical oracle."""

    _validate_affine_inputs(phi, q)
    p = phi[0]
    value = q[0]
    out_p = [p]
    out_q = [value]
    for index in range(1, phi.shape[0]):
        p, value = compose_affine((p, value), (phi[index], q[index]))
        out_p.append(p)
        out_q.append(value)
    return torch.stack(out_p, 0), torch.stack(out_q, 0)


def affine_associative_scan(phi: Tensor, q: Tensor) -> tuple[Tensor, Tensor]:
    """Autograd-safe O(log T)-depth Hillis-Steele inclusive scan.

    PyTorch 2.5's prototype associative scan does not support autograd.  This
    implementation uses only ordinary tensor operations and is therefore the
    v0.3.7 training path.  A future fused kernel must retain parity with the
    sequential oracle before replacing it.
    """

    _validate_affine_inputs(phi, q)
    prefix_phi, prefix_q = phi, q
    offset = 1
    while offset < phi.shape[0]:
        p_left = prefix_phi[:-offset]
        q_left = prefix_q[:-offset]
        p_right = prefix_phi[offset:]
        q_right = prefix_q[offset:]
        combined_phi = p_right * p_left
        combined_q = p_right * q_left + q_right
        prefix_phi = torch.cat((prefix_phi[:offset], combined_phi), dim=0)
        prefix_q = torch.cat((prefix_q[:offset], combined_q), dim=0)
        offset *= 2
    return prefix_phi, prefix_q


def _as_time_by_batch(event_time: Tensor, batch: int) -> Tensor:
    if event_time.ndim == 1:
        return event_time[:, None].expand(-1, batch)
    if event_time.ndim == 2 and event_time.shape[1] in {1, batch}:
        return event_time.expand(-1, batch)
    raise ValueError("event_time must have shape [time] or [time, batch]")


def event_time_deltas(event_time: Tensor, batch: int, initial_time: Tensor | float | None = None) -> Tensor:
    times = _as_time_by_batch(event_time.to(torch.float64), batch)
    if not torch.isfinite(times).all():
        raise ValueError("event times must be finite")
    if initial_time is None:
        first = times[0]
    else:
        first = torch.as_tensor(initial_time, dtype=times.dtype, device=times.device)
        if first.ndim == 0:
            first = first.expand(batch)
        if first.shape != (batch,):
            raise ValueError("initial_time must be scalar or [batch]")
    delta = torch.cat(((times[0] - first)[None], times[1:] - times[:-1]), dim=0)
    if torch.any(delta < 0):
        raise ValueError("event times must be non-decreasing after initial_time")
    return delta


@dataclass(frozen=True)
class CTSSMOutput:
    pre_state: Tensor
    post_state: Tensor
    transition: Tensor
    impulse: Tensor
    tau_seconds: Tensor


@dataclass(frozen=True)
class DualStreamCTSSMOutput:
    """States whose burden magnitude and conditional grammar are separable."""

    pre_burden: Tensor
    post_burden: Tensor
    pre_grammar_composition: Tensor
    post_grammar_composition: Tensor
    pre_grammar_mass: Tensor
    post_grammar_mass: Tensor
    pre_features: Tensor
    post_features: Tensor
    tau_seconds: Tensor


@dataclass(frozen=True)
class BackgroundCTSSMOutput:
    """Rate-normalised continuous background estimate and its reliability."""

    composition: Tensor
    mass: Tensor
    features: Tensor
    transition: Tensor
    tau_seconds: Tensor


class GridBackgroundCTSSM(nn.Module):
    """Stable physical-time observer for fixed-clock non-event background EEG.

    A background frame is a noisy observation of common drive, not an event
    impulse. Numerator and observation mass use the same zero-order-hold EWMA
    update. Missing frames only decay reliability.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        taus_seconds: Sequence[float],
        channels_per_tau: int = 2,
        input_scale: float = 0.02,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        taus = torch.as_tensor(tuple(float(v) for v in taus_seconds), dtype=torch.float32)
        if taus.ndim != 1 or taus.numel() == 0 or torch.any(taus <= 0):
            raise ValueError("taus_seconds must be a non-empty positive sequence")
        if int(channels_per_tau) <= 0:
            raise ValueError("channels_per_tau must be positive")
        self.input_dim = int(input_dim)
        self.n_tau = int(taus.numel())
        self.channels_per_tau = int(channels_per_tau)
        self.register_buffer("log_tau", taus.log())
        self.input = nn.Linear(self.input_dim, self.n_tau * self.channels_per_tau, bias=False)
        nn.init.normal_(self.input.weight, mean=0.0, std=float(input_scale))
        self.epsilon = float(epsilon)

    @property
    def state_dim(self) -> int:
        return self.n_tau * (self.channels_per_tau + 1)

    def tau_seconds(self) -> Tensor:
        return self.log_tau.exp()

    def forward(
        self,
        frame_time: Tensor,
        frame_value: Tensor,
        available: Tensor,
        *,
        initial_time: Tensor | float | None = None,
        scan: str = "associative",
    ) -> BackgroundCTSSMOutput:
        squeeze_batch = frame_value.ndim == 2
        if squeeze_batch:
            frame_value = frame_value[:, None, :]
        if frame_value.ndim != 3 or frame_value.shape[-1] != self.input_dim:
            raise ValueError("frame_value must have shape [time,input] or [time,batch,input]")
        time, batch = frame_value.shape[:2]
        if frame_time.shape[0] != time:
            raise ValueError("frame_time and frame_value length mismatch")
        weight = available
        if weight.ndim == 1:
            weight = weight[:, None]
        if weight.shape != (time, batch):
            raise ValueError("available must have shape [time] or [time,batch]")
        weight = weight.to(device=frame_value.device, dtype=frame_value.dtype)
        if torch.any((weight < 0) | (weight > 1)) or not torch.isfinite(weight).all():
            raise ValueError("available must be finite in [0,1]")
        delta = event_time_deltas(frame_time, batch, initial_time)
        tau = self.tau_seconds().to(device=frame_value.device, dtype=torch.float64)
        phi = torch.exp(-delta[..., None] / tau).to(dtype=frame_value.dtype)
        alpha = 1.0 - phi
        encoded = self.input(frame_value).reshape(time, batch, self.n_tau, self.channels_per_tau)
        numerator_q = alpha[..., None] * encoded * weight[..., None, None]
        mass_q = alpha * weight[..., None]
        if scan == "associative":
            scan_fn = affine_associative_scan
        elif scan == "sequential":
            scan_fn = affine_sequential_scan
        else:
            raise ValueError("scan must be associative or sequential")
        numerator = scan_fn(phi[..., None].expand_as(numerator_q), numerator_q)[1]
        mass = scan_fn(phi, mass_q)[1]
        composition = numerator / mass[..., None].clamp_min(self.epsilon)
        features = torch.cat((composition.flatten(-2), mass), dim=-1)
        if squeeze_batch:
            composition, mass, features, phi = (
                value[:, 0] for value in (composition, mass, features, phi)
            )
        return BackgroundCTSSMOutput(
            composition=composition,
            mass=mass,
            features=features,
            transition=phi,
            tau_seconds=self.tau_seconds(),
        )


class DiagonalEventCTSSM(nn.Module):
    """Event-impulse observer with exact irregular-time diagonal transitions."""

    def __init__(
        self,
        burden_dim: int,
        grammar_dim: int,
        *,
        taus_seconds: Sequence[float],
        channels_per_tau: int = 4,
        learnable_tau: bool = False,
        minimum_tau_seconds: float = 60.0,
        maximum_tau_seconds: float = 72.0 * 3600.0,
        input_scale: float = 0.02,
    ) -> None:
        super().__init__()
        taus = torch.as_tensor(tuple(float(v) for v in taus_seconds), dtype=torch.float32)
        if taus.ndim != 1 or taus.numel() == 0 or torch.any(taus <= 0):
            raise ValueError("taus_seconds must be a non-empty positive sequence")
        if int(channels_per_tau) <= 0:
            raise ValueError("channels_per_tau must be positive")
        self.burden_dim = int(burden_dim)
        self.grammar_dim = int(grammar_dim)
        self.channels_per_tau = int(channels_per_tau)
        expanded = taus.repeat_interleave(self.channels_per_tau)
        self.minimum_tau_seconds = float(minimum_tau_seconds)
        self.maximum_tau_seconds = float(maximum_tau_seconds)
        if not self.minimum_tau_seconds <= float(expanded.min()):
            raise ValueError("minimum_tau_seconds exceeds a requested tau")
        if not float(expanded.max()) <= self.maximum_tau_seconds:
            raise ValueError("maximum_tau_seconds is below a requested tau")
        if learnable_tau:
            self.log_tau = nn.Parameter(expanded.log())
        else:
            self.register_buffer("log_tau", expanded.log())
        self.burden_input = nn.Linear(self.burden_dim, expanded.numel(), bias=False)
        self.grammar_input = nn.Linear(self.grammar_dim, expanded.numel(), bias=False)
        nn.init.normal_(self.burden_input.weight, mean=0.0, std=float(input_scale))
        nn.init.normal_(self.grammar_input.weight, mean=0.0, std=float(input_scale))

    @property
    def state_dim(self) -> int:
        return int(self.log_tau.numel())

    def tau_seconds(self) -> Tensor:
        return self.log_tau.exp().clamp(self.minimum_tau_seconds, self.maximum_tau_seconds)

    def transitions(self, delta_seconds: Tensor, *, dtype: torch.dtype) -> Tensor:
        tau = self.tau_seconds().to(device=delta_seconds.device, dtype=torch.float64)
        phi = torch.exp(-delta_seconds[..., None].to(torch.float64) / tau)
        return phi.to(dtype=dtype)

    def event_impulse(self, burden_mark: Tensor, grammar_mark: Tensor) -> Tensor:
        if burden_mark.shape[:-1] != grammar_mark.shape[:-1]:
            raise ValueError("burden and grammar marks must share time/batch axes")
        if burden_mark.shape[-1] != self.burden_dim or grammar_mark.shape[-1] != self.grammar_dim:
            raise ValueError("mark feature width does not match the CTSSM contract")
        return self.burden_input(burden_mark) + self.grammar_input(grammar_mark)

    def forward(
        self,
        event_time: Tensor,
        burden_mark: Tensor,
        grammar_mark: Tensor,
        *,
        initial_state: Tensor | None = None,
        initial_time: Tensor | float | None = None,
        scan: str = "associative",
    ) -> CTSSMOutput:
        squeeze_batch = burden_mark.ndim == 2
        if squeeze_batch:
            burden_mark = burden_mark[:, None, :]
            grammar_mark = grammar_mark[:, None, :]
        if burden_mark.ndim != 3 or grammar_mark.ndim != 3:
            raise ValueError("marks must have shape [time, feature] or [time, batch, feature]")
        time, batch, _ = burden_mark.shape
        if event_time.shape[0] != time:
            raise ValueError("event_time and marks have different time lengths")
        delta = event_time_deltas(event_time, batch, initial_time)
        impulse = self.event_impulse(burden_mark, grammar_mark)
        phi = self.transitions(delta, dtype=impulse.dtype)
        if scan == "associative":
            prefix_phi, prefix_q = affine_associative_scan(phi, impulse)
        elif scan == "sequential":
            prefix_phi, prefix_q = affine_sequential_scan(phi, impulse)
        else:
            raise ValueError("scan must be associative or sequential")
        if initial_state is None:
            initial = torch.zeros(batch, self.state_dim, dtype=impulse.dtype, device=impulse.device)
        else:
            initial = initial_state
            if initial.ndim == 1:
                initial = initial[None]
            if initial.shape != (batch, self.state_dim):
                raise ValueError("initial_state must have shape [state] or [batch, state]")
        post = prefix_phi * initial[None] + prefix_q
        previous = torch.cat((initial[None], post[:-1]), dim=0)
        pre = phi * previous
        if squeeze_batch:
            pre, post, phi, impulse = (value[:, 0] for value in (pre, post, phi, impulse))
        return CTSSMOutput(
            pre_state=pre,
            post_state=post,
            transition=phi,
            impulse=impulse,
            tau_seconds=self.tau_seconds(),
        )


class DualStreamEventCTSSM(nn.Module):
    """Rate-sensitive burden plus rate-normalised conditional grammar memory.

    The two streams share a physical-time bank but not their state magnitude.
    Burden is an additive shot-noise state.  Grammar is stored as a decayed
    numerator divided by a decayed event mass.  Repeating the same grammar mark
    at the same physical time therefore changes confidence, not composition.
    This is an observer; its event update must not be interpreted as an H3
    physiological jump.
    """

    def __init__(
        self,
        burden_dim: int,
        grammar_dim: int,
        *,
        taus_seconds: Sequence[float],
        burden_channels_per_tau: int = 2,
        grammar_channels_per_tau: int = 3,
        learnable_tau: bool = False,
        minimum_tau_seconds: float = 60.0,
        maximum_tau_seconds: float = 72.0 * 3600.0,
        input_scale: float = 0.02,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        taus = torch.as_tensor(tuple(float(v) for v in taus_seconds), dtype=torch.float32)
        if taus.ndim != 1 or taus.numel() == 0 or torch.any(taus <= 0):
            raise ValueError("taus_seconds must be a non-empty positive sequence")
        if burden_channels_per_tau <= 0 or grammar_channels_per_tau <= 0:
            raise ValueError("channels per tau must be positive")
        self.burden_dim = int(burden_dim)
        self.grammar_dim = int(grammar_dim)
        self.n_tau = int(taus.numel())
        self.burden_channels_per_tau = int(burden_channels_per_tau)
        self.grammar_channels_per_tau = int(grammar_channels_per_tau)
        self.minimum_tau_seconds = float(minimum_tau_seconds)
        self.maximum_tau_seconds = float(maximum_tau_seconds)
        if not self.minimum_tau_seconds <= float(taus.min()):
            raise ValueError("minimum_tau_seconds exceeds a requested tau")
        if not float(taus.max()) <= self.maximum_tau_seconds:
            raise ValueError("maximum_tau_seconds is below a requested tau")
        if learnable_tau:
            self.log_tau = nn.Parameter(taus.log())
        else:
            self.register_buffer("log_tau", taus.log())
        self.burden_input = nn.Linear(
            self.burden_dim, self.n_tau * self.burden_channels_per_tau, bias=False
        )
        self.grammar_input = nn.Linear(
            self.grammar_dim, self.n_tau * self.grammar_channels_per_tau, bias=False
        )
        nn.init.normal_(self.burden_input.weight, mean=0.0, std=float(input_scale))
        nn.init.normal_(self.grammar_input.weight, mean=0.0, std=float(input_scale))
        self.epsilon = float(epsilon)

    def tau_seconds(self) -> Tensor:
        return self.log_tau.exp().clamp(self.minimum_tau_seconds, self.maximum_tau_seconds)

    @property
    def state_dim(self) -> int:
        # Grammar mass is explicit because it is the state reliability signal.
        return self.n_tau * (
            self.burden_channels_per_tau + self.grammar_channels_per_tau + 1
        )

    def _transition(self, delta: Tensor, *, dtype: torch.dtype) -> Tensor:
        tau = self.tau_seconds().to(device=delta.device, dtype=torch.float64)
        return torch.exp(-delta[..., None] / tau).to(dtype=dtype)

    @staticmethod
    def _scan(phi: Tensor, q: Tensor, scan: str) -> Tensor:
        if scan == "associative":
            return affine_associative_scan(phi, q)[1]
        if scan == "sequential":
            return affine_sequential_scan(phi, q)[1]
        raise ValueError("scan must be associative or sequential")

    def forward(
        self,
        event_time: Tensor,
        burden_mark: Tensor,
        grammar_mark: Tensor,
        *,
        event_weight: Tensor | None = None,
        initial_time: Tensor | float | None = None,
        scan: str = "associative",
    ) -> DualStreamCTSSMOutput:
        squeeze_batch = burden_mark.ndim == 2
        if squeeze_batch:
            burden_mark = burden_mark[:, None, :]
            grammar_mark = grammar_mark[:, None, :]
        if burden_mark.ndim != 3 or grammar_mark.ndim != 3:
            raise ValueError("marks must have shape [time, feature] or [time, batch, feature]")
        if burden_mark.shape[:2] != grammar_mark.shape[:2]:
            raise ValueError("burden and grammar marks must share time/batch axes")
        if burden_mark.shape[-1] != self.burden_dim or grammar_mark.shape[-1] != self.grammar_dim:
            raise ValueError("mark feature width does not match the dual-stream contract")
        time, batch = burden_mark.shape[:2]
        if event_time.shape[0] != time:
            raise ValueError("event_time and marks have different time lengths")
        delta = event_time_deltas(event_time, batch, initial_time)
        phi = self._transition(delta, dtype=burden_mark.dtype)
        if event_weight is None:
            weight = torch.ones(time, batch, dtype=burden_mark.dtype, device=burden_mark.device)
        else:
            weight = event_weight
            if weight.ndim == 1:
                weight = weight[:, None]
            if weight.shape != (time, batch):
                raise ValueError("event_weight must have shape [time] or [time, batch]")
            weight = weight.to(device=burden_mark.device, dtype=burden_mark.dtype)
            if torch.any(weight < 0) or not torch.isfinite(weight).all():
                raise ValueError("event_weight must be finite and non-negative")

        burden_q = self.burden_input(burden_mark).reshape(
            time, batch, self.n_tau, self.burden_channels_per_tau
        ) * weight[..., None, None]
        burden = self._scan(phi[..., None].expand_as(burden_q), burden_q, scan)

        grammar_q = self.grammar_input(grammar_mark).reshape(
            time, batch, self.n_tau, self.grammar_channels_per_tau
        ) * weight[..., None, None]
        grammar_num = self._scan(phi[..., None].expand_as(grammar_q), grammar_q, scan)
        mass_q = weight[..., None, None].expand(time, batch, self.n_tau, 1)
        mass = self._scan(phi[..., None], mass_q, scan)
        grammar = grammar_num / mass.clamp_min(self.epsilon)

        zero_burden = torch.zeros_like(burden[:1])
        zero_grammar = torch.zeros_like(grammar[:1])
        zero_mass = torch.zeros_like(mass[:1])
        previous_burden = torch.cat((zero_burden, burden[:-1]), dim=0)
        previous_grammar_num = torch.cat((torch.zeros_like(grammar_num[:1]), grammar_num[:-1]), dim=0)
        previous_mass = torch.cat((zero_mass, mass[:-1]), dim=0)
        pre_burden = phi[..., None] * previous_burden
        pre_grammar_num = phi[..., None] * previous_grammar_num
        pre_mass = phi[..., None] * previous_mass
        pre_grammar = torch.where(
            pre_mass > self.epsilon,
            pre_grammar_num / pre_mass.clamp_min(self.epsilon),
            zero_grammar.expand_as(pre_grammar_num),
        )
        pre_features = torch.cat(
            (pre_burden.flatten(-2), pre_grammar.flatten(-2), torch.log1p(pre_mass).flatten(-2)),
            dim=-1,
        )
        post_features = torch.cat(
            (burden.flatten(-2), grammar.flatten(-2), torch.log1p(mass).flatten(-2)),
            dim=-1,
        )
        values = (
            pre_burden, burden, pre_grammar, grammar, pre_mass, mass, pre_features, post_features
        )
        if squeeze_batch:
            values = tuple(value[:, 0] for value in values)
        return DualStreamCTSSMOutput(
            pre_burden=values[0],
            post_burden=values[1],
            pre_grammar_composition=values[2],
            post_grammar_composition=values[3],
            pre_grammar_mass=values[4],
            post_grammar_mass=values[5],
            pre_features=values[6],
            post_features=values[7],
            tau_seconds=self.tau_seconds(),
        )


def dual_stream_features_at_queries(
    output: DualStreamCTSSMOutput,
    event_time: Tensor,
    query_time: Tensor,
) -> Tensor:
    """Causal state at arbitrary physical-time anchors for one event chain.

    Events exactly at a query are excluded (left search).  The helper is fully
    differentiable with respect to the event-state tensors; only the discrete
    predecessor lookup is non-differentiable, as intended.
    """

    if event_time.ndim != 1 or query_time.ndim != 1:
        raise ValueError("query helper currently accepts one unbatched chain")
    if output.post_burden.ndim != 3:
        raise ValueError("query helper requires unbatched dual-stream output")
    if event_time.numel() != output.post_burden.shape[0]:
        raise ValueError("event_time and output length differ")
    if torch.any(event_time[1:] < event_time[:-1]) or torch.any(query_time[1:] < query_time[:-1]):
        raise ValueError("event and query times must be sorted")
    predecessor = torch.searchsorted(event_time, query_time, right=False) - 1
    valid = predecessor >= 0
    safe = predecessor.clamp_min(0)
    lag = (query_time - event_time[safe]).clamp_min(0.0)
    tau = output.tau_seconds.to(device=query_time.device, dtype=torch.float64)
    phi = torch.exp(-lag.to(torch.float64)[..., None] / tau).to(output.post_burden.dtype)
    burden = phi[..., None] * output.post_burden[safe]
    mass = phi[..., None] * output.post_grammar_mass[safe]
    grammar = output.post_grammar_composition[safe]
    burden = torch.where(valid[..., None, None], burden, torch.zeros_like(burden))
    mass = torch.where(valid[..., None, None], mass, torch.zeros_like(mass))
    grammar = torch.where(valid[..., None, None], grammar, torch.zeros_like(grammar))
    return torch.cat(
        (burden.flatten(-2), grammar.flatten(-2), torch.log1p(mass).flatten(-2)), dim=-1
    )


def zoh_diagonal_step(state: Tensor, value: Tensor, delta_seconds: Tensor | float, tau_seconds: Tensor) -> Tensor:
    """Exact zero-order-hold step for dx/dt=-x/tau+value."""

    delta = torch.as_tensor(delta_seconds, dtype=torch.float64, device=state.device)
    tau = tau_seconds.to(device=state.device, dtype=torch.float64)
    phi = torch.exp(-delta[..., None] / tau)
    gain = tau * (1.0 - phi)
    return (phi.to(state.dtype) * state + gain.to(value.dtype) * value).to(state.dtype)
