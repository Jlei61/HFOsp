"""Same linear event write, different autonomous transitions.

Time is in hours. An input row is consumed only at its availability time.
RK4 advances the *pre-event* state before the linear jump. Padding must use
zero dt and zero input. No normalisation, posterior update or real future
observation is hidden inside autonomous rollout.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


TAUS_HOURS = (1/6, .5, 1., 2., 4., 8., 16.)


class EventTransition(nn.Module):
    def __init__(self, input_dim, family, width=16, rank=8, seed=0):
        super().__init__()
        if family not in ('F', 'L', 'N', 'C'):
            raise ValueError(family)
        self.family = family
        self.input_dim = int(input_dim)
        self.width = len(TAUS_HOURS) * input_dim if family == 'F' else width
        generator = torch.Generator().manual_seed(seed)
        if family == 'F':
            # Full, uncompressed marked-history bank, not a rank-losing random null.
            self.register_buffer('decay', torch.tensor(TAUS_HOURS).reciprocal().repeat_interleave(input_dim))
            self.register_buffer('write', torch.eye(input_dim).repeat(len(TAUS_HOURS), 1))
        elif family == 'C':
            self.register_buffer('write', torch.zeros(width, input_dim))
        else:
            self.write = nn.Parameter(torch.randn(width, input_dim, generator=generator) * (.1 / math.sqrt(input_dim)))
            self.skew = nn.Parameter(torch.zeros(width, width))
            rates = torch.logspace(math.log10(1/16), math.log10(4), width)
            self.log_decay = nn.Parameter(torch.log(torch.expm1(rates)))
            if family == 'N':
                self.u = nn.Parameter(torch.randn(width, rank, generator=generator) * .05)
                self.v = nn.Parameter(torch.randn(rank, width, generator=generator) * .05)
                self.bias = nn.Parameter(torch.zeros(rank))

    def generator_matrix(self):
        if self.family not in ('L', 'N'):
            raise ValueError('Only learned transitions have A')
        return .5 * (self.skew - self.skew.T) - torch.diag(nn.functional.softplus(self.log_decay) + 1e-4)

    def _drift(self, state, matrix):
        result = state @ matrix.T
        if self.family == 'N':
            result = result + torch.tanh(state @ self.v.T + self.bias) @ self.u.T
        return result

    def _step(self, state, dt, matrix):
        if self.family == 'C':
            return state
        if self.family == 'F':
            return state * torch.exp(-dt * self.decay)
        k1 = self._drift(state, matrix)
        k2 = self._drift(state + dt * k1 / 2, matrix)
        k3 = self._drift(state + dt * k2 / 2, matrix)
        k4 = self._drift(state + dt * k3, matrix)
        return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

    def advance(self, state, hours, max_step_hours=1/12):
        """Autonomous forecast: this API cannot receive future observations."""
        if hours < 0 or max_step_hours <= 0:
            raise ValueError('Nonnegative elapsed time and positive step required')
        matrix = self.generator_matrix() if self.family in ('L', 'N') else None
        steps = max(1, math.ceil(hours / max_step_hours))
        for _ in range(steps):
            state = self._step(state, hours / steps, matrix)
        return state

    def scan(self, inputs, dt_hours, *, checkpoint_chunk=32, return_pre_event=False):
        """Inputs (batch,time,feature), dt (batch,time), bounded dt <= 5 min.

        Callers must insert zero-input time steps across longer intervals.
        Chunking recomputes activations and never detaches the history graph.
        """
        if inputs.ndim != 3 or dt_hours.shape != inputs.shape[:2] or inputs.shape[-1] != self.input_dim:
            raise ValueError('Expected aligned batch,time,feature and batch,time')
        if not torch.isfinite(inputs).all() or not torch.isfinite(dt_hours).all():
            raise ValueError('Nonfinite input or elapsed time')
        if (dt_hours < 0).any() or (dt_hours > 1/12 + 1e-7).any():
            raise ValueError('Insert zero-input integration steps; elapsed time must be in [0, 5 min]')
        state = inputs.new_zeros((len(inputs), self.width))
        matrix = self.generator_matrix() if self.family in ('L', 'N') else None
        pre_event = []
        def chunk(state, x, dt):
            for j in range(x.shape[1]):
                state = self._step(state, dt[:, j:j+1], matrix)
                state = state + x[:, j] @ self.write.T
            return state
        if return_pre_event:
            for j in range(inputs.shape[1]):
                state = self._step(state, dt_hours[:, j:j+1], matrix)
                pre_event.append(state)
                state = state + inputs[:, j] @ self.write.T
            return state, torch.stack(pre_event, dim=1)
        size = checkpoint_chunk or max(1, inputs.shape[1])
        for start in range(0, inputs.shape[1], size):
            x = inputs[:, start:start+size]; dt = dt_hours[:, start:start+size]
            if checkpoint_chunk and torch.is_grad_enabled() and (inputs.requires_grad or self.family in ('L', 'N')):
                state = checkpoint(chunk, state, x, dt, use_reentrant=False)
            else:
                state = chunk(state, x, dt)
        return state


class FutureReadout(nn.Module):
    """Identical 32-unit nonlinear readout grammar for all transition families."""
    def __init__(self, state_dim, n_contacts, context_dim=0, hidden=32):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(state_dim + context_dim + 1, hidden), nn.Tanh(),
                                    nn.Linear(hidden, 1 + n_contacts))
        self.log_dispersion = nn.Parameter(torch.zeros(()))

    def forward(self, state, context, lead_hours):
        lead = state.new_full((len(state), 1), float(lead_hours) / 8)
        logits = self.layers(torch.cat((state, context, lead), dim=-1))
        return logits[:, 0], logits[:, 1:]


def endpoint_loss(log_mean, logits, counts, recruitment, log_dispersion, view='joint'):
    """NB count score plus anchor-normalised marginal recruitment score.

    Recruitment is a per-contact indicator for the future window. Averaging
    contacts prevents high event counts from overweighting the spatial task.
    """
    mean = log_mean.clamp(-12, 12).exp()
    r = nn.functional.softplus(log_dispersion) + 1e-4
    nb = -(torch.lgamma(counts+r) - torch.lgamma(r) - torch.lgamma(counts+1)
           + r*(torch.log(r)-torch.log(r+mean)) + counts*(torch.log(mean)-torch.log(r+mean)))
    spatial = nn.functional.binary_cross_entropy_with_logits(logits, recruitment, reduction='none').mean(-1)
    if view not in ('count', 'recruitment', 'joint'):
        raise ValueError(view)
    total = nb if view == 'count' else spatial if view == 'recruitment' else nb + spatial
    return total, nb, spatial
