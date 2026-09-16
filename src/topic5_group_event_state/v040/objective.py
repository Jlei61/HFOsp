"""Window-equal-weight predictive scoring with per-component path marginalisation.

C12: an event/scalar score is ``-log mean_s p(y_bik | z_s)`` -- the mixture is
taken for that one scalar, then the losses are averaged.  This is a conditional
marginal predictive score, not a packet-joint likelihood.

C13 fixes the aggregation order: valid scalars inside a morphology family ->
available families -> events with morphology support inside the window ->
windows with at least one scorable event.

C16: the v0312 packet-joint mixture is preserved verbatim under the name
``legacy_packet_joint_component_score`` and is never divided by an event count
and renamed.
"""
from __future__ import annotations
import math
import torch

from .model import (count_log_prob, composition_log_prob, zero_inflated_normal_log_prob,
                    load_log_prob)

VIEWS = ('count', 'spatial', 'morphology')
SECONDARY = ('load',)
FAMILIES = ('band_ratio', 'signed_xlag', 'delay_iqr')
LOAD_WEIGHT = 0.25
HORIZON_WEIGHTS = {1: .625, 5: .125, 30: .125, 120: .125}
VIEW_WEIGHTS = {'count': 1 / 3, 'spatial': 1 / 3, 'morphology': 1 / 3, 'load': LOAD_WEIGHT}
SELECTION_HORIZON = 30
SELECTION_VIEWS = ('spatial', 'morphology')


def event_rows(prep, pidx):
    """Flat event indices for packets ``pidx`` plus the window each event belongs to."""
    lo = prep.event_lo[pidx]
    hi = prep.event_hi[pidx]
    counts = hi - lo
    total = int(counts.sum())
    if total == 0:
        return None, None
    offs = torch.cumsum(counts, 0) - counts
    pos = torch.arange(total, device=prep.device)
    row = torch.searchsorted(offs + counts, pos, right=True)
    return lo[row] + (pos - offs[row]), row


def _mix_nll(lp):
    """-log mean_s exp(lp) over the leading path axis."""
    S = lp.shape[0]
    return -(torch.logsumexp(lp, dim=0) - math.log(S))


def _normal_component_lp(y, mu, logsd):
    ls = logsd.clamp(-6., 4.)
    return -0.5 * ((y - mu) / torch.exp(ls)) ** 2 - ls - 0.5 * math.log(2 * math.pi)


def interval_log_mean_rate(readout, grid, cond, prep, pidx):
    """Trapezoid path integral on the common 5-second scoring grid, conditioned on C."""
    G, S, B, _ = grid.shape
    fractions = torch.linspace(0, 1, G, device=grid.device, dtype=torch.float64)
    start = torch.as_tensor(prep.packet_start[pidx.detach().cpu().numpy()], device=grid.device)
    times = start[None, :] + fractions[:, None] * 60.
    phase = times * (2 * math.pi / 86400.)
    clock = torch.stack((phase.sin(), phase.cos()), -1).to(grid.dtype).unsqueeze(1).expand(G, S, B, 2)
    cc = cond.unsqueeze(0).unsqueeze(0).expand(G, S, B, cond.shape[-1])
    log_rate = readout(grid, torch.cat((clock, cc), -1))['log_rate'].clamp(-20, 15)
    if G == 13:
        seconds = prep.exposure_grid[pidx]
        weights = grid.new_zeros(G, B)
        weights[:-1] += seconds.T / 2
        weights[1:] += seconds.T / 2
        den = seconds.sum(-1).clamp(min=1e-12)
        return torch.logsumexp(log_rate + weights.clamp(min=1e-30).log()[:, None, :], 0) - den.log()[None, :]
    if G != 2:
        raise ValueError('unregistered interval grid')
    return torch.logsumexp(log_rate, 0) - math.log(2.)


def morphology_terms(readout, prep, pidx, cond, state_grid, event_index=None):
    """Per-event, per-family morphology NLL with the registered aggregation order.

    Returns ``None`` when the windows carry no events at all; a window with
    events but no valid morphology scalar is reported as unsupported rather than
    filled with a zero (C17).
    """
    S = state_grid.shape[1]
    idx, row = event_index if event_index is not None else event_rows(prep, pidx)
    if idx is None or len(idx) == 0:
        return None
    start = torch.as_tensor(prep.packet_start[pidx.detach().cpu().numpy()], device=prep.device)
    frac = ((prep.event_times[idx] - start[row]) / 60.).clamp(0, 1)
    gi = (frac * (state_grid.shape[0] - 1)).floor().long().clamp(max=state_grid.shape[0] - 1)
    z_event = state_grid[gi, :, row].transpose(0, 1)
    phase = prep.event_times[idx] * (2 * math.pi / 86400.)
    ce = torch.stack((phase.sin(), phase.cos()), -1).to(z_event.dtype).unsqueeze(0).expand(S, len(idx), 2)
    cc = cond[row].unsqueeze(0).expand(S, len(idx), cond.shape[-1])
    out = readout(z_event, torch.cat((ce, cc), -1))
    br = prep.band_ratio[idx].unsqueeze(0)
    brv = prep.band_ratio_valid[idx]
    xl = prep.xlag[idx].unsqueeze(0)
    xlv = prep.xlag_valid[idx]
    iq = prep.iqr[idx].unsqueeze(0)
    iqv = prep.iqr_valid[idx]
    w = (torch.log(iq.clamp(min=1e-9)) - prep.log_iqr_center) / prep.log_iqr_scale
    families = {}
    # C12: mixture per scalar component, then mean over the valid components of
    # the family. An all-missing family contributes no weight at all.
    for name, lp_sk, valid in (
            ('band_ratio', _normal_component_lp(br, out['band_ratio_mu'], out['band_ratio_logsd']), brv),
            ('signed_xlag', _normal_component_lp(xl, out['xlag_mu'], out['xlag_logsd']), xlv),
            ('delay_iqr', zero_inflated_normal_log_prob(w, (iq <= 0).to(iq.dtype), out['iqr'],
                                                        torch.ones_like(iq)).unsqueeze(-1), iqv.unsqueeze(-1))):
        nll_k = _mix_nll(lp_sk)
        n = valid.sum(-1)
        fam = (nll_k * valid).sum(-1) / n.clamp(min=1)
        families[name] = dict(nll=fam, valid=n > 0, n_components=n)
    # C16: the v0312 rule -- sum every event/component log-probability inside the
    # packet, mix over paths once, normalise by the total valid component count --
    # is preserved verbatim for monitoring. It is a packet-joint component score
    # and is never divided by an event count and renamed as the new estimator.
    B0 = len(pidx)
    legacy_lp = torch.zeros(S, B0, device=prep.device, dtype=z_event.dtype)
    legacy_units = torch.zeros(B0, device=prep.device, dtype=z_event.dtype)
    for name, lp_sk, valid in (('band_ratio', _normal_component_lp(br, out['band_ratio_mu'], out['band_ratio_logsd']), brv),
                               ('signed_xlag', _normal_component_lp(xl, out['xlag_mu'], out['xlag_logsd']), xlv),
                               ('delay_iqr', zero_inflated_normal_log_prob(w, (iq <= 0).to(iq.dtype), out['iqr'],
                                                                          torch.ones_like(iq)).unsqueeze(-1),
                                iqv.unsqueeze(-1))):
        legacy_lp = legacy_lp.index_add(1, row, (lp_sk * valid.unsqueeze(0)).sum(-1))
        legacy_units = legacy_units.index_add(0, row, valid.sum(-1).to(legacy_units.dtype))

    stack = torch.stack([families[f]['nll'] for f in FAMILIES])
    ok = torch.stack([families[f]['valid'] for f in FAMILIES]).to(stack.dtype)
    n_fam = ok.sum(0)
    event_nll = (stack * ok).sum(0) / n_fam.clamp(min=1)      # C13b equal weight over families
    event_valid = n_fam > 0
    B = len(pidx)
    ev = event_valid.to(stack.dtype)
    per_window = torch.zeros(B, device=prep.device, dtype=stack.dtype).index_add(0, row, event_nll * ev)
    n_events = torch.zeros(B, device=prep.device, dtype=stack.dtype).index_add(0, row, ev)
    window_nll = per_window / n_events.clamp(min=1)           # C13c equal weight over events
    return dict(families=families, event_nll=event_nll, event_valid=event_valid, event_row=row,
                event_index=idx, window_nll=window_nll, window_valid=n_events > 0, n_scored_events=n_events,
                n_families_per_event=n_fam,
                legacy_packet_joint_logp=_mix_nll(legacy_lp).neg(), legacy_component_units=legacy_units)


def window_scores(readout, prep, pidx, cond, z_start, z_end, state_grid, event_index=None):
    """Per-window NLL for every registered view on its own physical support."""
    S, B, _ = z_end.shape
    clock = prep.clock[pidx].unsqueeze(0).expand(S, B, 2)
    cc = cond.unsqueeze(0).expand(S, B, cond.shape[-1])
    out_e = readout(z_end, torch.cat((clock, cc), -1))
    count = prep.count[pidx]
    expo = prep.exposure_hours[pidx]
    valid = prep.valid[pidx] > 0
    has = (count > 0) & valid
    lr = interval_log_mean_rate(readout, state_grid, cond, prep, pidx)
    scores = {}
    scores['count'] = dict(nll=_mix_nll(count_log_prob(lr, lr, expo.unsqueeze(0), count.unsqueeze(0),
                                                       readout.log_nb_dispersion)), valid=valid)
    scores['spatial'] = dict(nll=_mix_nll(composition_log_prob(out_e['composition'],
                                                               prep.shaft_count[pidx].unsqueeze(0))), valid=has)
    load = prep.total_load[pidx].unsqueeze(0)
    ly = (torch.log(load.clamp(min=1e-9)) - prep.log_load_center) / prep.log_load_scale
    scores['load'] = dict(nll=_mix_nll(load_log_prob(ly, out_e['load'], torch.ones(S, B, device=prep.device,
                                                                                  dtype=z_end.dtype))), valid=has)
    morph = morphology_terms(readout, prep, pidx, cond, state_grid, event_index)
    if morph is None:
        zeros = torch.zeros(B, device=prep.device, dtype=z_end.dtype)
        scores['morphology'] = dict(nll=zeros, valid=torch.zeros(B, dtype=torch.bool, device=prep.device))
    else:
        scores['morphology'] = dict(nll=morph['window_nll'], valid=morph['window_valid'])
    return scores, morph


def legacy_packet_joint_component_score(scores, morph, prep, pidx):
    """C16: the preserved v0312 denominator convention, for monitoring only."""
    count = prep.count[pidx]
    valid = (prep.valid[pidx] > 0).to(count.dtype)
    has = ((count > 0) & (prep.valid[pidx] > 0)).to(count.dtype)
    units = dict(count=valid, spatial=has, load=has,
                 morphology=(morph['legacy_component_units'] if morph is not None
                             else torch.zeros_like(valid)))
    logp = dict(count=-scores['count']['nll'], spatial=-scores['spatial']['nll'], load=-scores['load']['nll'],
                morphology=(morph['legacy_packet_joint_logp'] if morph is not None else torch.zeros_like(valid)))
    return {k: dict(logp_sum=float((logp[k] * (units[k] > 0)).sum()), units=float(units[k].sum()))
            for k in ('count', 'spatial', 'morphology', 'load')}


def training_loss(scores, horizon, normalizers, views=VIEWS):
    """Registered horizon and view weights on microbatch-invariant denominators (C20)."""
    w = HORIZON_WEIGHTS[int(horizon)]
    loss = None
    for view in tuple(views) + SECONDARY:
        den = normalizers.get(view, 0.)
        if den <= 0:
            continue
        s = scores[view]
        term = w * VIEW_WEIGHTS[view] * (s['nll'] * s['valid'].to(s['nll'].dtype)).sum() / den
        loss = term if loss is None else loss + term
    return loss


def aggregate(rows):
    """Window-equal-weight main effect with the event-weighted companion (C14).

    ``rows`` are per-(horizon, view) records carrying window NLLs and, for
    morphology, the per-event terms needed for the companion estimator.
    """
    out = {}
    for (h, view), items in rows.items():
        nll = torch.cat([r['nll'] for r in items])
        valid = torch.cat([r['valid'] for r in items])
        n = int(valid.sum())
        window = float((nll * valid).sum() / n) if n else None
        record = dict(horizon=int(h), view=view, window_equal_weight=window, n_windows=n,
                      n_windows_without_support=int((~valid).sum()))
        if view == 'morphology':
            ev = [r for r in items if r.get('event_nll') is not None]
            if ev:
                e_nll = torch.cat([r['event_nll'] for r in ev])
                e_ok = torch.cat([r['event_valid'] for r in ev])
                m = int(e_ok.sum())
                record['event_equal_weight'] = float((e_nll * e_ok).sum() / m) if m else None
                record['n_events'] = m
                record['n_events_without_family'] = int((~e_ok).sum())
                fam = {}
                for f in FAMILIES:
                    fn = torch.cat([r['families'][f]['nll'] for r in ev])
                    fo = torch.cat([r['families'][f]['valid'] for r in ev])
                    k = int(fo.sum())
                    fam[f] = dict(event_equal_weight=float((fn * fo).sum() / k) if k else None,
                                  n_events_with_family=k, n_events_missing_family=int((~fo).sum()))
                record['families'] = fam
        out[f'{view}@{int(h)}'] = record
    return out


def selection_score(summary):
    """J_inner = 0.5 R_spatial,30 + 0.5 R_morphology,30 on window-equal weights (C19)."""
    parts = []
    for view in SELECTION_VIEWS:
        rec = summary.get(f'{view}@{SELECTION_HORIZON}')
        if rec is None or rec['window_equal_weight'] is None:
            return None, dict(status='NOT_ESTIMABLE', missing=view)
        parts.append(rec['window_equal_weight'])
    return .5 * parts[0] + .5 * parts[1], dict(status='COMPLETE', spatial=parts[0], morphology=parts[1],
                                               rule='0.5*spatial + 0.5*morphology at the 30-minute horizon, '
                                                    'both window-equal-weight; count and load are monitored only')
