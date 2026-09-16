"""One availability-aware query, prediction and scoring path for every arm.

The sole exported online object is ``QueryState`` = (m 24, P 24x24, the common
conditions C and the legal time metadata).  The inference GRU ``c`` and the
within-minute event memory are never handed to a consumer (C22).
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from . import data as D
from . import conditions as CD
from .model import StateModel, HistoryReference, initialize_readout, RICH_DIM
from ..v0312.numerics import LATENT, propagate_moments, propagate_samples, evidence_update, cholesky_psd
from .prepare import HISTORY_TAU_HOURS
from .objective import window_scores, legacy_packet_joint_component_score, VIEWS, SECONDARY

STATE_FIELDS = ('query_packet', 'query_time', 'source_time', 'release_time', 'information_age_minutes',
                'available_exposure_seconds', 'readable_events', 'prefix_start')


@dataclass
class QueryState:
    patient: str
    split_id: str
    transform_id: str
    query_packet: np.ndarray
    query_time: np.ndarray
    source_time: np.ndarray
    release_time: np.ndarray
    information_age_minutes: np.ndarray
    available_exposure_seconds: np.ndarray
    readable_events: np.ndarray
    prefix_start: np.ndarray
    input_digest: tuple
    m: torch.Tensor | None
    P: torch.Tensor | None
    history: torch.Tensor | None
    cond: torch.Tensor | None
    producer_hash: str = 'in_memory'
    donor_query_time: np.ndarray | None = None

    def subset(self, indices):
        ix = np.asarray(indices, int)
        anchor = self.m if self.m is not None else (self.history if self.history is not None else self.cond)
        tx = torch.as_tensor(ix, device=anchor.device)
        return replace(self, **{k: getattr(self, k)[ix] for k in STATE_FIELDS},
                       input_digest=tuple(self.input_digest[i] for i in ix),
                       m=None if self.m is None else self.m[tx], P=None if self.P is None else self.P[tx],
                       history=None if self.history is None else self.history[tx],
                       cond=None if self.cond is None else self.cond[tx],
                       donor_query_time=None if self.donor_query_time is None else self.donor_query_time[ix])

    def metadata(self):
        out = {k: (v.tolist() if isinstance(v, np.ndarray) else list(v) if isinstance(v, tuple) else v)
               for k, v in vars(self).items() if k not in ('m', 'P', 'history', 'cond')}
        observed = np.isfinite(self.release_time)
        out['has_observation'] = observed.tolist()
        out['last_observed_source_time'] = [float(t) if ok else None for t, ok in zip(self.source_time, observed)]
        out['observed_information_age_minutes'] = [float(t) if ok else None
                                                   for t, ok in zip(self.information_age_minutes, observed)]
        out['condition_names'] = list(CD.NAMES)
        out['conditions'] = None if self.cond is None else self.cond.detach().cpu().tolist()
        out['source_semantics'] = ('source_time is the prior origin when has_observation=false; '
                                   'it is not a last observation')
        return out

    def export(self):
        """C22/C23: the only object a consumer may read."""
        return dict(m=None if self.m is None else self.m.detach().cpu(),
                    P=None if self.P is None else self.P.detach().cpu(),
                    cond=None if self.cond is None else self.cond.detach().cpu(),
                    metadata=self.metadata(),
                    withheld=['inference_gru_c64', 'within_minute_event_memory', 'packet_encoder_activations'])


def build_model(prep, inputs='P_marks', family='I-L-G1', arm='state', seed=20260906):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 103)
        if arm == 'state':
            model = StateModel(prep.part.shape[1], prep.tokens.shape[2], prep.groups.shape[2],
                               prep.event.shape[1], prep.n_shaft, prep.stats.shape[1],
                               prep.band_ratio.shape[1], prep.xlag.shape[1],
                               rich=inputs == 'P_marks', coupled=family.startswith('C'),
                               nonlinear=family.split('-')[1] == 'N', linear_readout=family.endswith('G0'),
                               seed=seed).to(prep.device)
        else:
            model = HistoryReference(prep, arm, seed=seed).to(prep.device)
        torch.manual_seed(seed + 105)
        initialize_readout(model.readout, prep)
    return model


def infer_asof(model, prep, query_packets, role='outer', history_hours=None, training=False,
               grad_hours=2., activation_checkpoint=True, producer_hash='in_memory',
               grad_start_by_episode=None, _cache=True):
    """Replay each shared episode once, then propagate its last readable posterior to t.

    C25: ``grad_hours`` bounds only where gradient is credited.  The forward
    history always runs from the legal episode origin unless ``history_hours``
    explicitly re-trains a short arm.
    """
    qs = np.asarray(query_packets, int)
    if _cache and not training and hasattr(prep, 'frozen_query_cache'):
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('query caching requires a frozen producer')
        cache = prep.frozen_query_cache
        keys = [(producer_hash, prep.split['split_id'], prep.scaling['transform_id'], role, history_hours, int(q))
                for q in qs]
        missing = [i for i, k in enumerate(keys) if k not in cache]
        if missing:
            fresh = infer_asof(model, prep, qs[missing], role, history_hours, producer_hash=producer_hash, _cache=False)
            for j, i in enumerate(missing):
                cache[keys[i]] = fresh.subset([j])
        states = [cache[k] for k in keys]
        first = states[0]
        return replace(first, **{k: np.concatenate([getattr(s, k) for s in states]) for k in STATE_FIELDS},
                       input_digest=tuple(s.input_digest[0] for s in states),
                       m=None if first.m is None else torch.cat([s.m for s in states]),
                       P=None if first.P is None else torch.cat([s.P for s in states]),
                       history=None if first.history is None else torch.cat([s.history for s in states]),
                       cond=None if first.cond is None else torch.cat([s.cond for s in states]))
    if not len(qs):
        raise ValueError('empty query batch')
    if (qs < 0).any() or (qs >= prep.n_packets).any():
        raise IndexError('query outside packet grid')
    if prep.split['seizure_mask'][qs].any():
        raise ValueError('query inside ictal/postictal exclusion')
    allowed = D.input_mask(prep.split, role)
    pk = prep.payload['packets']
    ends = prep.packet_end
    release = prep.packet_release
    starts = prep.split['episode_start'][qs].copy()
    if history_hours is not None:
        if history_hours <= 0:
            raise ValueError('strict history must be positive')
        starts = np.maximum(starts, np.searchsorted(pk['start'], ends[qs] - history_hours * 3600 - 1e-6))
    readable = []
    groups = {}
    finite_release = release[allowed & np.isfinite(release)]
    ordered = bool(np.all(finite_release[1:] >= finite_release[:-1]))
    for j, (start, q) in enumerate(zip(starts, qs)):
        ix = np.arange(start, q + 1)
        ix = ix[allowed[ix] & (release[ix] <= ends[q])]
        readable.append(ix)
        key = (int(start), None if ordered else D.digest(ix))
        groups.setdefault(key, []).append(j)
    ms = [None] * len(qs)
    Ps = ms.copy()
    hs = ms.copy()
    source = np.empty(len(qs))
    rel = np.full(len(qs), np.nan)
    expo = np.zeros(len(qs))
    ne = np.zeros(len(qs), int)
    input_ids = []
    for ix, q, start in zip(readable, qs, starts):
        input_ids.append(D.digest(dict(packets=ix, role=role, start=int(start),
                                       transform=prep.scaling['transform_id'])))
    caches = {False: {}, True: {}}
    for (start, _), members in groups.items():
        longest = max(members, key=lambda j: len(readable[j]))
        stream = readable[longest]
        snapshots = {}
        is_state = isinstance(model, StateModel)
        if is_state:
            m, P, c = model.initial(1, prep.device, prep.stats.dtype)
        else:
            ema = prep.stats.new_zeros(5, model.base)
            mass = prep.stats.new_zeros(5, 1)
        prev_time = float(pk['start'][start]) if start < prep.n_packets else float(ends[qs[members[0]]])
        grad_start = int(min(qs[members]) - round(grad_hours * 60)) if history_hours is None else start
        if history_hours is None and grad_start_by_episode is not None:
            grad_start = int(grad_start_by_episode[start])
        rich_map = {}
        prev_event = None
        if model.encoder is not None:
            partitions = [stream[stream < grad_start], stream[stream >= grad_start]]
            for ids in partitions:
                for a in range(0, len(ids), 60):
                    batch = ids[a:a + 60]
                    on = training and int(batch[0]) >= grad_start
                    ti = torch.as_tensor(batch, device=prep.device)
                    previous = prev_event
                    with torch.set_grad_enabled(on):
                        fn = lambda idx, previous=previous: prep.rich_summary(model.encoder, idx, previous)
                        rich = checkpoint(fn, ti, use_reentrant=False, preserve_rng_state=True) \
                            if on and activation_checkpoint else fn(ti)
                    for ii, k in enumerate(batch):
                        rich_map[int(k)] = rich[ii:ii + 1]
                    last_events = [k for k in batch if pk['event_hi'][k] > pk['event_lo'][k]]
                    if last_events:
                        prev_event = float(prep.payload['event_time'][pk['event_hi'][last_events[-1]] - 1])
        wanted = {int(readable[j][-1]) for j in members if len(readable[j])}
        for k in stream:
            on = training and int(k) >= grad_start
            with torch.set_grad_enabled(on):
                dt = (float(ends[k]) - prev_time) / 3600.
                x = prep.stats[k:k + 1]
                rich = rich_map.get(int(k), x.new_zeros(1, RICH_DIM))
                if is_state:
                    m, P = propagate_moments(model.dynamics, m, P, dt, cache=caches[on])
                    c = model.slow.gru(model.slow.packet_input(x, rich), c)
                    a, R, _ = model.slow.evidence(c, m, P)
                    m, P, _, _ = evidence_update(m.double(), P.double(), a.double(), R.double())
                    m = m.to(x.dtype)
                    P = P.to(x.dtype)
                    if int(k) in wanted:
                        snapshots[int(k)] = (m, P)
                else:
                    xx = (x[:, :1] if model.mode == 'recent_rate'
                          else torch.cat((x, rich), -1) if model.rich
                          else torch.cat((x, prep.fixed_rich_summary(k)), -1) if model.fixed_rich else x)
                    tau = x.new_tensor(HISTORY_TAU_HOURS).reshape(-1, 1)
                    decay = torch.exp(-dt / tau)
                    w = 1 - torch.exp(-(float(pk['exposure'][k]) / 3600.) / tau)
                    ema = ema * decay + w * xx
                    mass = mass * decay + w
                    if int(k) in wanted:
                        snapshots[int(k)] = (ema / mass.clamp(min=1e-12)).reshape(1, -1)
            prev_time = float(ends[k])
        for j in members:
            ix = readable[j]
            q = qs[j]
            source[j] = float(ends[ix[-1]]) if len(ix) else float(pk['start'][start])
            if len(ix):
                rel[j] = float(np.max(release[ix]))
            expo[j] = float(pk['exposure'][ix].sum())
            ne[j] = int((pk['event_hi'][ix] - pk['event_lo'][ix]).sum())
            dt = max(0., float(ends[q] - source[j])) / 3600.
            if is_state:
                if len(ix):
                    m1, P1 = snapshots[int(ix[-1])]
                else:
                    m1, P1, _ = model.initial(1, prep.device, prep.stats.dtype)
                with torch.set_grad_enabled(training):
                    m1, P1 = propagate_moments(model.dynamics, m1, P1, dt, cache=caches[training])
                ms[j] = m1
                Ps[j] = P1
            else:
                hs[j] = snapshots[int(ix[-1])] if len(ix) else prep.stats.new_zeros(1, 5 * model.base)
    return QueryState(prep.split['subject'], prep.split['split_id'], prep.scaling['transform_id'],
                      qs, ends[qs], source, rel, (ends[qs] - source) / 60., expo, ne, starts, tuple(input_ids),
                      torch.cat(ms) if ms[0] is not None else None,
                      torch.cat(Ps) if Ps[0] is not None else None,
                      torch.cat(hs) if hs[0] is not None else None,
                      prep.query_conditions(qs, role), producer_hash)


def sample_posterior(m, P, paths, generator=None, noise=None):
    L, _ = cholesky_psd(P.double(), 'query covariance')
    e = noise if noise is not None else torch.randn(paths, *m.shape, device=m.device, dtype=m.dtype,
                                                    generator=generator)
    return m.unsqueeze(0) + torch.einsum('bij,sbj->sbi', L.to(m.dtype), e)


def query_noise(state, seed, steps, paths, device, dtype):
    """C29: physical-query keyed noise; microbatching cannot change a sample's paths."""
    arrays = []
    for t in state.query_time:
        key = int(D.digest((int(seed), float(t)))[:16], 16)
        arrays.append(np.random.default_rng(key).standard_normal((steps, paths, LATENT), dtype=np.float32))
    return torch.as_tensor(np.stack(arrays, axis=2), device=device, dtype=dtype)


@dataclass
class IntervalPrediction:
    z_start: torch.Tensor
    z_end: torch.Tensor
    grid: torch.Tensor


def predict(model, prep, state, horizons=(1, 5, 30, 120), paths=64, seed=20260906, rule='EVOLVE', reference=None):
    """C28: one read-only copy of Q per horizon, on one common noise trajectory."""
    if rule not in ('HOLD', 'EVOLVE', 'RELAX', 'RESET'):
        raise ValueError(rule)
    if state.m is not None:
        if rule in ('RELAX', 'RESET') and reference is None:
            raise ValueError('FIT reference distribution required')
        m, P = ((reference['m0'].to(state.m).expand_as(state.m), reference['P0'].to(state.P).expand_as(state.P))
                if rule == 'RESET' else (state.m, state.P))
        refined = set(horizons) | {1, 5, 30, 120}
        steps = 1 + 2 * sum(12 if h in refined else 1 for h in range(1, max(horizons) + 1))
        noise = query_noise(state, seed, steps, paths, prep.device, state.m.dtype)
        ni = 1
        z = sample_posterior(m, P, paths, noise=noise[0])
        cache = {}
        pred = {}
        if rule == 'RELAX':
            L0, _ = cholesky_psd(reference['P0'].double().unsqueeze(0), 'RELAX stationary covariance')
            L0 = L0[0].to(z.dtype)
        for h in range(1, max(horizons) + 1):
            zp = z
            grid = [z]
            n = 12 if h in refined else 1
            for _ in range(n):
                dt = 1 / (60. * n)
                if rule in ('EVOLVE', 'RESET'):
                    z = propagate_samples(model.dynamics, z, dt, cache=cache, noise=[noise[ni], noise[ni + 1]])
                elif rule == 'RELAX':
                    a = math.exp(-dt / reference['tau_hours'])
                    m0 = reference['m0'].to(z)
                    z = m0 + a * (z - m0) + math.sqrt(1 - a * a) * (noise[ni] @ L0.T)
                ni += 2
                grid.append(z)
            if h in horizons:
                pred[h] = IntervalPrediction(zp, z, torch.stack(grid))
    else:
        if rule != 'EVOLVE':
            raise ValueError('state rules are not defined for a history reference')
        pred = {}
        for h in horizons:
            ix = torch.as_tensor(np.minimum(state.query_packet + h, prep.n_packets - 1), device=prep.device)
            cond = torch.cat((prep.clock[ix], state.cond), -1)
            elapsed = state.cond.new_tensor(state.information_age_minutes / 60. + h / 60.)
            z = model.latent(state.history, cond, elapsed).unsqueeze(0)
            pred[h] = IntervalPrediction(z, z, z.unsqueeze(0).expand(13, *z.shape))
    return pred


def score_predictions(model, prep, state, predictions, role, target_ids=None, keep_events=False):
    """Same distribution, same physical targets; per-window supports carried in every row."""
    rows = []
    for h, pred in predictions.items():
        targets = state.query_packet + h
        keep = targets < prep.n_packets
        if target_ids is not None:
            keep &= np.isin(targets, target_ids)
        ids = np.flatnonzero(keep)
        if not len(ids):
            continue
        tgt = targets[ids]
        allowed = D.role_mask(prep.split, role)[tgt].copy()
        for j, (q, t) in enumerate(zip(state.query_packet[ids], tgt)):
            allowed[j] &= not prep.split['seizure_mask'][q + 1:t + 1].any()
        ti = torch.as_tensor(tgt, device=prep.device)
        ii = torch.as_tensor(ids, device=prep.device)
        scores, morph = window_scores(model.readout, prep, ti, state.cond[ii],
                                      pred.z_start[:, ii], pred.z_end[:, ii], pred.grid[:, :, ii])
        legacy = legacy_packet_joint_component_score(scores, morph, prep, ti)
        mask = torch.as_tensor(allowed, device=prep.device)
        row = dict(horizon=int(h), packet=tgt, query_packet=state.query_packet[ids],
                   information_age_minutes=state.information_age_minutes[ids],
                   legacy_packet_joint_component_score=legacy)
        for view in VIEWS + SECONDARY:
            row[view] = dict(nll=scores[view]['nll'], valid=scores[view]['valid'] & mask)
        if morph is not None:
            ev_ok = morph['event_valid'] & mask[morph['event_row']]
            row['morphology_events'] = dict(
                event_nll=morph['event_nll'], event_valid=ev_ok, event_row=morph['event_row'],
                event_index=morph['event_index'] if keep_events else None,
                families={f: dict(nll=morph['families'][f]['nll'],
                                  valid=morph['families'][f]['valid'] & mask[morph['event_row']],
                                  n_components=morph['families'][f]['n_components'])
                          for f in morph['families']})
        rows.append(row)
    return rows


DONOR_RULE = dict(min_lead_hours=4., max_clock_seconds=7200., max_support_age_minutes=10.,
                  max_exposure_fraction_difference=.2, max_log_segment_age_difference=math.log(2.))


def donor_match(state, reference, rule=DONOR_RULE):
    """C44: same-patient FIT donor chosen by time distance under fixed legality only.

    No target, seizure label or predictive score is read, and no criterion is
    relaxed when a recipient finds no donor -- that recipient is simply not
    estimable.
    """
    keep = []
    donors = []
    dt = reference['donor_time']
    for i, t in enumerate(state.query_time):
        clock = np.abs((dt - t + 43200) % 86400 - 43200)
        legal = ((dt < t - rule['min_lead_hours'] * 3600)
                 & (clock <= rule['max_clock_seconds'])
                 & (np.abs(reference['donor_age'] - state.information_age_minutes[i])
                    <= rule['max_support_age_minutes'])
                 & (np.abs(reference['donor_exposure_fraction'] - float(state.cond[i, CD.NAMES.index(
                     'short_window_published_coverage')])) <= rule['max_exposure_fraction_difference'])
                 & (np.abs(reference['donor_log_segment_age'] - float(state.cond[i, CD.NAMES.index(
                     'log_episode_age_hours')])) <= rule['max_log_segment_age_difference']))
        ix = np.flatnonzero(legal)
        if not len(ix):
            continue
        j = int(ix[np.argmin(np.abs(dt[ix] - t))])
        keep.append(i)
        donors.append(j)
    return keep, donors


def wrong_time_control(state, reference, rule=DONOR_RULE):
    """Donor state at the recipient's own time; recipient C, clock and exposure stay fixed."""
    keep, donors = donor_match(state, reference, rule)
    if not keep:
        return None, dict(status='NOT_ESTIMABLE', reason='no legal FIT donor for any recipient query',
                          matched=0, n_recipients=len(state.query_time))
    shifted = state.subset(keep)
    swapped = replace(shifted, m=reference['donor_m'][donors].to(state.m),
                      P=reference['donor_P'][donors].to(state.P),
                      donor_query_time=reference['donor_time'][donors],
                      input_digest=tuple(D.digest((d, 'WRONGTIME', float(t)))
                                         for d, t in zip(shifted.input_digest, reference['donor_time'][donors])))
    counts = np.bincount(np.asarray(donors), minlength=len(reference['donor_time']))
    lead = (shifted.query_time - reference['donor_time'][donors]) / 3600.
    report = dict(status='COMPLETE', matched=len(keep), n_recipients=len(state.query_time),
                  matched_fraction=len(keep) / max(len(state.query_time), 1),
                  distinct_donors=int((counts > 0).sum()), max_donor_reuse=int(counts.max()),
                  donor_lead_hours=dict(min=float(lead.min()), median=float(np.median(lead)), max=float(lead.max())),
                  rule=rule,
                  interpretation=('tests time correspondence only; it cannot on its own establish history '
                                  'integration or a causal role, and it is not a global admission gate'))
    return swapped, report
