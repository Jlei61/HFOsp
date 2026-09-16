"""Readout and arms conditioned on the common measurement conditions C.

The event encoder, slow evidence head and every probability model are the
repaired v0312 objects.  What changes here is that the readout g and the
history references both receive the same C vector (C10), so the background
baseline is not silently the state arm's private advantage.
"""
from __future__ import annotations
import math
import torch
from torch import nn

from ..v0312.model import (EventEncoder, SlowState, RICH_DIM, SLOW_HIDDEN, READOUT_HIDDEN,
                           count_log_prob, composition_log_prob, normal_log_prob,
                           zero_inflated_normal_log_prob, load_log_prob, LOG_RATE_LIMITS)
from ..v0312.numerics import Dynamics, LATENT
from .conditions import COND_DIM

__all__ = ['Readout', 'StateModel', 'HistoryReference', 'initialize_readout', 'RICH_DIM',
           'count_log_prob', 'composition_log_prob', 'normal_log_prob',
           'zero_inflated_normal_log_prob', 'load_log_prob', 'LOG_RATE_LIMITS', 'LATENT', 'COND_DIM']


class Readout(nn.Module):
    """Natural parameters for count, coarse composition, morphology and load."""

    def __init__(self, n_shaft, n_ratio, n_xlag, cond_dim=COND_DIM, hidden=READOUT_HIDDEN, linear=False):
        super().__init__()
        self.n_shaft = n_shaft
        self.n_ratio = n_ratio
        self.n_xlag = n_xlag
        self.cond_dim = cond_dim
        out = 1 + n_shaft + 2 * n_ratio + 2 * n_xlag + 3 + 2
        self.linear = linear
        self.net = nn.Linear(LATENT + cond_dim, out) if linear else nn.Sequential(
            nn.Linear(LATENT + cond_dim, hidden), nn.GELU(), nn.Linear(hidden, out))
        self.log_nb_dispersion = nn.Parameter(torch.zeros(1))
        self.register_buffer('fit_offset', torch.zeros(out))
        self.use_conditions = True

    def forward(self, z, cond):
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f'expected {self.cond_dim} condition columns, received {cond.shape[-1]}')
        if not self.use_conditions:
            cond = torch.zeros_like(cond)
        y = self.net(torch.cat((z, cond), dim=-1)) + self.fit_offset
        i = 0
        log_rate = y[..., 0]
        i = 1
        comp = y[..., i:i + self.n_shaft]
        i += self.n_shaft
        br_mu = y[..., i:i + self.n_ratio]
        i += self.n_ratio
        br_ls = y[..., i:i + self.n_ratio]
        i += self.n_ratio
        xl_mu = y[..., i:i + self.n_xlag]
        i += self.n_xlag
        xl_ls = y[..., i:i + self.n_xlag]
        i += self.n_xlag
        iqr = y[..., i:i + 3]
        i += 3
        load = y[..., i:i + 2]
        return dict(log_rate=log_rate, composition=comp, band_ratio_mu=br_mu, band_ratio_logsd=br_ls,
                    xlag_mu=xl_mu, xlag_logsd=xl_ls, iqr=iqr, load=load)


class StateModel(nn.Module):
    """q -> f -> g: event encoder, continuous-time latent core, conditioned readout."""

    def __init__(self, n_contacts, n_token, n_group, n_event, n_shaft, n_packet_input, n_ratio, n_xlag,
                 rich=True, coupled=False, nonlinear=False, linear_readout=False, cond_dim=COND_DIM,
                 seed=20260906):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        self.dynamics = Dynamics(coupled=coupled, nonlinear=nonlinear, generator=g)
        # Separate RNG streams keep common parameters identical across inputs.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 101)
            self.encoder = EventEncoder(n_contacts, n_token, n_group, n_event, n_shaft) if rich else None
            torch.manual_seed(int(seed) + 102)
            self.slow = SlowState(n_packet_input, rich=rich)
            torch.manual_seed(int(seed) + 103)
            self.readout = Readout(n_shaft, n_ratio, n_xlag, cond_dim, linear=linear_readout)
        self.rich = rich
        self.register_buffer('m0', torch.zeros(LATENT))
        self.register_buffer('P0', torch.eye(LATENT))

    def initial(self, batch, device, dtype=torch.float32):
        m = self.m0.to(device=device, dtype=dtype).expand(batch, LATENT).clone()
        P = self.P0.to(device=device, dtype=dtype).expand(batch, LATENT, LATENT).clone()
        c = torch.zeros(batch, SLOW_HIDDEN, device=device, dtype=dtype)
        return m, P, c


class HistoryReference(nn.Module):
    """B arms: identical rich encoder and conditioned readout, fixed history kernels.

    ``stats_history`` reads the coarse packet statistics stream, ``marked_history``
    the full event encoder.  Both see the same C and the same targets, so the
    B_stats/B_marks difference isolates the rich content (C51).
    """

    def __init__(self, prep, mode='marked_history', cond_dim=COND_DIM, seed=20260906):
        super().__init__()
        if mode not in ('recent_rate', 'marked_history', 'stats_history', 'fixed_marked_history',
                        'constant_state', 'intercept'):
            raise ValueError(mode)
        self.mode = mode
        self.rich = mode == 'marked_history'
        self.fixed_rich = mode == 'fixed_marked_history'
        fixed_dim = prep.part.shape[1] + 2 * prep.tokens.shape[-1] + 2 * prep.event.shape[-1]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 101)
            self.encoder = EventEncoder(prep.part.shape[1], prep.tokens.shape[-1], prep.groups.shape[-1],
                                        prep.event.shape[-1], prep.n_shaft) if self.rich else None
            self.base = (1 if mode == 'recent_rate'
                         else prep.stats.shape[1] + (RICH_DIM if self.rich else fixed_dim if self.fixed_rich else 0))
            torch.manual_seed(seed + 104)
            # history kernels + target clock + query conditions + elapsed time to the
            # target: exactly the deterministic information the state arm gets through f.
            self.net = nn.Sequential(nn.Linear(self.base * 5 + cond_dim + 1, 64), nn.GELU(), nn.Linear(64, LATENT)) \
                if mode in ('recent_rate', 'marked_history', 'stats_history', 'fixed_marked_history') else None
            self.constant = nn.Parameter(torch.zeros(LATENT)) if mode == 'constant_state' else None
            torch.manual_seed(seed + 103)
            self.readout = Readout(prep.n_shaft, prep.band_ratio.shape[1], prep.xlag.shape[1], cond_dim)
            self.readout.use_conditions = mode != 'intercept'

    def latent(self, history, cond, elapsed_hours):
        """History kernels plus the same C the state arm's readout receives."""
        if self.net is not None:
            return self.net(torch.cat((history, cond, torch.log1p(elapsed_hours).unsqueeze(-1)), dim=-1))
        if self.constant is not None:
            return self.constant.expand(len(cond), LATENT)
        return cond.new_zeros(len(cond), LATENT)


@torch.no_grad()
def initialize_readout(readout, prep):
    """FIT MLE offsets plus small nonzero residual weights (upstream gradients survive)."""
    import numpy as np
    mask = prep.split['train_packet']
    ep = np.empty(len(prep.payload['event_time']), np.int64)
    for i, (a, b) in enumerate(zip(prep.payload['packets']['event_lo'], prep.payload['packets']['event_hi'])):
        ep[a:b] = i
    ev = torch.as_tensor(mask[ep], device=prep.device)
    pm = torch.as_tensor(mask, device=prep.device)

    def moments(y, valid):
        v = valid[ev]
        yy = y[ev]
        n = v.sum(0).clamp(min=1)
        mu = (yy * v).sum(0) / n
        var = (((yy - mu) ** 2) * v).sum(0) / n
        return mu, torch.log(var.sqrt().clamp(min=.05))

    br, brs = moments(prep.band_ratio, prep.band_ratio_valid)
    xl, xls = moments(prep.xlag, prep.xlag_valid)
    iq = prep.iqr[ev]
    valid = prep.iqr_valid[ev] > 0
    p0 = ((iq[valid] <= 0).float().mean() if bool(valid.any()) else iq.new_tensor(.5)).clamp(.001, .999)
    pos = iq[valid & (iq > 0)]
    coord = (torch.log(pos) - prep.log_iqr_center) / prep.log_iqr_scale
    iqmu = coord.mean() if len(coord) else iq.new_tensor(0.)
    iqsd = coord.std(unbiased=False).clamp(min=.05).log() if len(coord) else iq.new_tensor(0.)
    load = prep.total_load[pm & (prep.count > 0)]
    lc = (torch.log(load.clamp(min=1e-9)) - prep.log_load_center) / prep.log_load_scale
    lmu = lc.mean() if len(lc) else iq.new_tensor(0.)
    lsd = lc.std(unbiased=False).clamp(min=.05).log() if len(lc) else iq.new_tensor(0.)
    comp = (prep.shaft_count[pm].sum(0) + .5).log()
    off = torch.cat((iq.new_tensor([math.log(prep.scaling['base_rate_per_hour'])]), comp, br, brs, xl, xls,
                     torch.stack((torch.logit(p0), iqmu, iqsd, lmu, lsd))))
    readout.fit_offset.copy_(off)
    readout.log_nb_dispersion.fill_(-math.log(prep.scaling['nb_size']))
    final = readout.net if readout.linear else readout.net[-1]
    nn.init.normal_(final.weight, 0., 1e-3)
    nn.init.zeros_(final.bias)
