"""Frozen-state consumers: H2a-A, the wrong-time diagnostic, H2a-B, S-A and S-B.

The producer is selected by the interictal task alone.  No seizure label, no
spatial agreement and no consumer score ever reaches back into the producer.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr

from . import data as D
from . import conditions as CD
from .engine import (infer_asof, build_model, donor_match, wrong_time_control, DONOR_RULE)
from .prepare import Prepared
from .train import RunConfig, evaluate, tensor_hash, atomic_json, atomic_torch, source_digest, file_hash
from ..v0312.frozen import conditional_set_lp, identity_units
from ..v0312.seizure import controls_mask, available_recent

SPATIAL_SOURCE = Path('/home/honglab/leijiaxin/HFOsp/results')
DECODER_ROOT = Path('/data/hfosp_group_event_state_v0_3_4/we_decoder')
CLUSTER_GAP_HOURS = 4.
S_A_WINDOWS = (('primary', -7200., -1800.), ('secondary', -1800., -300.))
CONDITION_SETS = ('C', 'C+H', 'C+S', 'C+H+S')
FEATURES = {'C': (), 'C+H': ('history',), 'C+S': ('state',), 'C+H+S': ('state', 'history')}


def load_selected(path, device='cpu'):
    if device.startswith('cuda'):
        torch.cuda.set_device(device)
    record = torch.load(path, weights_only=False, map_location='cpu')
    if record['source_digest'] != source_digest()[0]:
        raise ValueError('frozen source mismatch')
    cfg = RunConfig(**(record['config'] | {'device': device}))
    packet_path = Path(cfg.packets_root) / f'{cfg.subject}.pt'
    if record.get('packets_sha256') != file_hash(packet_path):
        raise ValueError('frozen measurement packets changed')
    payload = torch.load(packet_path, weights_only=False, map_location='cpu')
    prep = Prepared(payload, record['split'], record['scaling'], torch.device(device), record['cond_scaling'])
    model = build_model(prep, cfg.inputs, cfg.family, cfg.arm, cfg.seed)
    model.load_state_dict(record['state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, prep, cfg, record


@torch.no_grad()
def fit_reference(model, prep, cfg, stride=60):
    """FIT state distribution and the donor bank used by the wrong-time diagnostic."""
    queries = np.flatnonzero(prep.split['train_packet'])[::stride]
    states = []
    for a in range(0, len(queries), cfg.eval_chunk):
        states.append(infer_asof(model, prep, queries[a:a + cfg.eval_chunk], 'fit', cfg.history_hours,
                                 producer_hash=tensor_hash(model.state_dict())))
    if not states or states[0].m is None:
        raise ValueError('a state producer is required for the FIT reference distribution')
    m = torch.cat([s.m for s in states])
    P = torch.cat([s.P for s in states])
    cond = torch.cat([s.cond for s in states])
    m0 = m.mean(0)
    cov = (m - m0).T @ (m - m0) / max(1, len(m) - 1) + P.mean(0)
    exposure_col = CD.NAMES.index('short_window_published_coverage')
    segment_col = CD.NAMES.index('log_episode_age_hours')
    return dict(m0=m0, P0=cov, fit_query_count=int(len(m)), fit_query_digest=D.digest(queries),
                donor_m=m, donor_P=P, donor_cond=cond,
                donor_time=np.concatenate([np.where(np.isfinite(s.release_time), s.query_time, np.nan)
                                           for s in states]),
                donor_age=np.concatenate([s.information_age_minutes for s in states]),
                donor_exposure_fraction=cond[:, exposure_col].cpu().numpy(),
                donor_log_segment_age=cond[:, segment_col].cpu().numpy())


# ------------------------------------------------------------------ H2a-A
@torch.no_grad()
def adapter_data(model, prep, cfg, role, horizon=1):
    """C42: state at the last registered minute query strictly before the event."""
    table = D.target_table(prep.payload, prep.split, role, cfg.eval_stride, (horizon,))
    if not len(table):
        return None
    pk = prep.payload['packets']
    allowed = D.input_mask(prep.split, role)
    rows = []
    for a in range(0, len(table), cfg.eval_chunk):
        qq = table[a:a + cfg.eval_chunk, 2]
        st = infer_asof(model, prep, qq, role, cfg.history_hours, producer_hash=tensor_hash(model.state_dict()))
        if st.m is None:
            raise ValueError('the conditional identity consumer expects a state checkpoint')
        for j, q in enumerate(st.query_packet):
            t = int(q + horizon)
            lo, hi = int(pk['event_lo'][t]), int(pk['event_hi'][t])
            if lo == hi:
                continue
            # H: the explicit readable recent contact history over the short window.
            window = float(cfg.short_history_minutes) * 60.
            hix = np.flatnonzero(allowed & (pk['end'] <= st.query_time[j]) & (pk['end'] > st.query_time[j] - window)
                                 & (pk['release'] <= st.query_time[j])
                                 & (np.arange(prep.n_packets) >= st.prefix_start[j]))
            ei = np.concatenate([np.arange(pk['event_lo'][i], pk['event_hi'][i]) for i in hix]) if len(hix) else np.empty(0, int)
            count = prep.part[ei].sum(0) if len(ei) else prep.part.new_zeros(prep.part.shape[1])
            hist = torch.logit(((count + .5) / (len(ei) + 1.)).clamp(1e-4, 1 - 1e-4))
            n = hi - lo
            ids = torch.arange(lo, hi, device=prep.device)
            cond = torch.cat((prep.clock[t], st.cond[j]))
            rows.append(dict(packet=t, query=int(q), query_time=float(st.query_time[j]),
                             event_time=prep.event_times[ids].cpu().numpy(),
                             support_age_minutes=float(st.information_age_minutes[j]),
                             state=st.m[j].expand(n, -1), history=hist.expand(n, -1), cond=cond.expand(n, -1),
                             identity=prep.identity_target[ids], event_ids=ids))
    if not rows:
        return None
    out = {k: torch.cat([r[k] for r in rows]) for k in ('state', 'history', 'cond', 'identity', 'event_ids')}
    out['packet'] = np.concatenate([np.full(len(r['identity']), r['packet']) for r in rows])
    out['query'] = np.concatenate([np.full(len(r['identity']), r['query']) for r in rows])
    out['query_time'] = np.concatenate([np.full(len(r['identity']), r['query_time']) for r in rows])
    out['event_time'] = np.concatenate([r['event_time'] for r in rows])
    out['support_age_minutes'] = np.concatenate([np.full(len(r['identity']), r['support_age_minutes'])
                                                 for r in rows])
    out['query_to_event_seconds'] = out['event_time'] - out['query_time']
    return out


def subset_adapter(data, mask):
    if data is None or not np.any(mask):
        return None
    tx = torch.as_tensor(np.flatnonzero(mask), device=data['state'].device)
    return {k: (v[tx] if isinstance(v, torch.Tensor) else v[mask]) for k, v in data.items()}


def fit_identity_head(train, validation, test, community, condition_set, steps=400, seed=20260906,
                      state_override=None):
    """C42/C43: one conditional identity head per registered condition set."""
    if any(d is None for d in (train, validation, test)):
        return dict(status='NOT_ESTIMABLE', reason='an adapter split is empty', condition_set=condition_set)
    features = FEATURES[condition_set]

    def get(d, override=None):
        parts = []
        for k in features:
            parts.append(override if (k == 'state' and override is not None) else d[k])
        return torch.cat(parts + [d['cond']], -1)

    x, xv, xt = get(train), get(validation), get(test)
    center = x.mean(0)
    scale = x.std(0, unbiased=False).clamp(min=.1)
    counts = [int(identity_units(d['identity'], community).sum()) for d in (train, validation, test)]
    if min(counts) <= 0:
        return dict(status='NOT_ESTIMABLE', reason='no informative conditional identity unit in a split',
                    condition_set=condition_set, split_units=counts)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = nn.Linear(x.shape[1], train['identity'].shape[1]).to(x.device)
    with torch.no_grad():
        net.weight.zero_()
        pr = (train['identity'].sum(0) + .5) / (len(train['identity']) + 1)
        net.bias.copy_(torch.logit(pr))
        initial = {k: v.clone() for k, v in net.state_dict().items()}

    def loss(d, xx):
        u = identity_units(d['identity'], community)
        return -conditional_set_lp(net(xx), d['identity'], community)[u].sum() / u.sum().clamp(min=1)

    optimizer = torch.optim.AdamW(net.parameters(), lr=.01, weight_decay=1e-3)
    with torch.no_grad():
        step_zero_validation = float(loss(validation, (xv - center) / scale))
        step_zero_test = float(loss(test, (xt - center) / scale))
    xs, xvs, xts = (x - center) / scale, (xv - center) / scale, (xt - center) / scale
    best = step_zero_validation
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    best_step = 0
    bad = 0
    step = 0
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        v = loss(train, xs)
        v.backward()
        optimizer.step()
        if step % 20 == 0:
            with torch.no_grad():
                score = float(loss(validation, xvs))
            if score < best - 1e-5:
                best = score
                best_step = step
                bad = 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
            if bad >= 6:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        u = identity_units(test['identity'], community)
        per_event = -conditional_set_lp(net(xts), test['identity'], community)
        score = float(per_event[u].sum() / u.sum())
        if not np.isfinite(score):
            raise FloatingPointError('non-finite frozen identity head score')
        weight_change = float(sum(((best_state[k] - initial[k]) ** 2).sum() for k in initial).sqrt())
        k_per_community = {}
        for c in torch.unique(community):
            k = test['identity'][:, community == c].sum(-1)
            k_per_community[int(c)] = dict(size=int((community == c).sum()),
                                           informative_events=int(((k > 0) & (k < int((community == c).sum()))).sum()),
                                           mean_K=float(k.float().mean()))
    return dict(status='COMPLETE', condition_set=condition_set, score=score,
                step_zero_validation=step_zero_validation, step_zero_test=step_zero_test,
                selected_step=best_step, executed_steps=step, validation_score=best,
                parameter_change_l2=weight_change,
                learned=best_step > 0 and weight_change > 0,
                metric='conditional_set_nll', units=int(u.sum()), n_events=int(len(per_event)),
                split_units=counts, k_per_community=k_per_community,
                budget=dict(max_steps=steps, validate_every=20, patience=6, lr=.01, weight_decay=1e-3),
                rows=dict(event_ids=test['event_ids'].cpu().numpy(), packet=test['packet'],
                          nll=per_event.cpu().numpy(), unit=u.cpu().numpy()),
                fitted=dict(state_dict=best_state, center=center, scale=scale,
                            feature_names=[*features, 'cond']))


def h2a_a(model, prep, cfg, role, community, steps=400):
    """Four matched-budget condition sets on the event-equal-weight identity task."""
    train = adapter_data(model, prep, cfg, 'fit', 1)
    test = adapter_data(model, prep, cfg, role, 1)
    if train is None or test is None:
        return dict(status='NOT_ESTIMABLE', reason='no one-minute identity target in FIT or the scored role')
    unique = np.unique(train['packet'])
    cut = unique[max(0, int(.8 * len(unique)) - 1)]
    tr = subset_adapter(train, train['packet'] <= cut)
    va = subset_adapter(train, train['packet'] > cut)
    heads = {}
    for condition_set in CONDITION_SETS:
        heads[condition_set] = fit_identity_head(tr, va, test, community, condition_set, steps, cfg.seed)
    complete = {k: v for k, v in heads.items() if v['status'] == 'COMPLETE'}
    contrasts = {}
    if 'C+H' in complete and 'C+H+S' in complete:
        contrasts['R(C+H)-R(C+H+S)'] = paired_event_difference(complete['C+H'], complete['C+H+S'])
    if 'C' in complete and 'C+S' in complete:
        contrasts['R(C)-R(C+S)'] = paired_event_difference(complete['C'], complete['C+S'])
    if 'C' in complete and 'C+H' in complete:
        contrasts['R(C)-R(C+H)'] = paired_event_difference(complete['C'], complete['C+H'])
    return dict(status='COMPLETE' if complete else 'NOT_ESTIMABLE', heads=heads, contrasts=contrasts,
                query_interface=dict(
                    horizon_minutes=1, n_events=int(len(test['identity'])),
                    query_to_event_seconds=dict(
                        min=float(test['query_to_event_seconds'].min()),
                        median=float(np.median(test['query_to_event_seconds'])),
                        max=float(test['query_to_event_seconds'].max())),
                    support_age_minutes=dict(median=float(np.median(test['support_age_minutes'])),
                                             max=float(test['support_age_minutes'].max())),
                    note=('the conservative one-minute interface; this is not a state that uses every bit of '
                          'information available exactly at the event onset')),
                task_name='fine identity given the event size or coarse composition',
                caveat=('K_c and the community sizes are supplied at scoring time; this is not an '
                        'unconditional forecast of the full contact set'),
                data=dict(train=tr, validation=va, test=test))


def paired_event_difference(a, b, block_hours=2.):
    """Event-equal-weight paired difference with time-block resampling."""
    ka = {int(e): (float(n), int(p)) for e, n, u, p in zip(a['rows']['event_ids'], a['rows']['nll'],
                                                           a['rows']['unit'], a['rows']['packet']) if u}
    kb = {int(e): (float(n), int(p)) for e, n, u, p in zip(b['rows']['event_ids'], b['rows']['nll'],
                                                           b['rows']['unit'], b['rows']['packet']) if u}
    common = sorted(set(ka) & set(kb))
    if not common:
        return dict(status='NOT_ESTIMABLE', reason='no shared informative identity unit')
    d = np.array([ka[e][0] - kb[e][0] for e in common])
    blocks = np.array([ka[e][1] // int(block_hours * 60) for e in common])
    uniq = np.unique(blocks)
    ci = None
    if len(uniq) >= 3:
        rng = np.random.default_rng(827)
        sums = np.array([[d[blocks == g].sum(), (blocks == g).sum()] for g in uniq])
        draws = sums[rng.integers(len(uniq), size=(2000, len(uniq)))].sum(1)
        ci = np.quantile(draws[:, 0] / draws[:, 1], [.025, .975]).tolist()
    return dict(status='DEVELOPMENT', mean_difference=float(d.mean()), n_events=len(common),
                n_time_blocks=int(len(uniq)), block_hours=block_hours, block_bootstrap_ci=ci,
                lost_a=len(ka) - len(common), lost_b=len(kb) - len(common),
                note='positive means the right-hand model is better; time blocks handle dependence, '
                     'they are not independent patients')


@torch.no_grad()
def wrong_time(model, prep, cfg, bundle, reference, community):
    """C44: swap only the state slot of the fixed C+H+S head."""
    head = bundle['heads'].get('C+H+S')
    if head is None or head['status'] != 'COMPLETE':
        return dict(status='NOT_ESTIMABLE', reason='the C+H+S head is not available')
    test = bundle['data']['test']
    queries = np.unique(test['query'])
    order = {int(q): i for i, q in enumerate(queries)}
    states = []
    for a in range(0, len(queries), cfg.eval_chunk):
        states.append(infer_asof(model, prep, queries[a:a + cfg.eval_chunk], 'outer' if cfg.stage == 'outer'
                                 else 'inner', cfg.history_hours, producer_hash=tensor_hash(model.state_dict())))
    from dataclasses import replace as _replace
    merged = states[0]
    for k in ('query_packet', 'query_time', 'source_time', 'release_time', 'information_age_minutes',
              'available_exposure_seconds', 'readable_events', 'prefix_start'):
        merged = _replace(merged, **{k: np.concatenate([getattr(s, k) for s in states])})
    merged = _replace(merged, m=torch.cat([s.m for s in states]), P=torch.cat([s.P for s in states]),
                      cond=torch.cat([s.cond for s in states]),
                      input_digest=tuple(d for s in states for d in s.input_digest))
    swapped, report = wrong_time_control(merged, reference)
    if swapped is None:
        return dict(report)
    matched = {int(q): i for i, q in enumerate(swapped.query_packet)}
    keep = np.array([int(q) in matched for q in test['query']])
    if not keep.any():
        return dict(status='NOT_ESTIMABLE', reason='no scored event has a matched donor', donor_matching=report)
    donor_state = torch.stack([swapped.m[matched[int(q)]] for q in test['query'][keep]])
    sub = subset_adapter(test, keep)
    f = head['fitted']
    net = nn.Linear(f['center'].shape[0], sub['identity'].shape[1]).to(prep.device)
    net.load_state_dict(f['state_dict'])

    def score(state_tensor):
        x = torch.cat((state_tensor, sub['history'], sub['cond']), -1)
        x = (x - f['center']) / f['scale']
        u = identity_units(sub['identity'], community)
        per = -conditional_set_lp(net(x), sub['identity'], community)
        return float(per[u].sum() / u.sum()), per, u

    correct, per_c, u = score(sub['state'])
    wrong, per_w, _ = score(donor_state)
    d = (per_w - per_c)[u].cpu().numpy()
    ood = float((donor_state - sub['state']).norm(dim=-1).median())
    return dict(status='COMPLETE', correct_time_score=correct, wrong_time_score=wrong,
                difference=wrong - correct, n_events=int(u.sum()),
                mean_event_difference=float(d.mean()),
                state_distance_median=ood,
                incoherence=('the donor state is paired with the recipient explicit history; a large distance '
                             'means the swap is out of distribution rather than merely mistimed'),
                donor_matching=report)


# ------------------------------------------------------------------ H2a-B
def rollout_with_h0(model, starts, n_contacts, max_steps, device, h0=None):
    """``topic5_wiring_economy_rnn.rollout`` with a supplied initial tissue state.

    Every decision rule is copied verbatim from the frozen helper: one contact
    per step by argmax over the still-available contacts, and STOP decided by
    its own head.  Only the initial hidden state differs, which is exactly the
    with-state / without-state contrast; ``h0=None`` reproduces the original.
    """
    from src.topic5_wiring_economy_rnn import NEG_INF
    model.eval()
    out = []
    for i, start in enumerate(starts):
        h = (torch.zeros(1, model.n_nodes * model.state_dim, device=device) if h0 is None
             else h0[i:i + 1].to(device))
        recruited = torch.zeros(1, n_contacts, device=device)
        x = torch.zeros(1, n_contacts, device=device)
        x[0, list(start)] = 1.0
        recruited[0, list(start)] = 1.0
        sequence = [list(map(int, start))]
        denom = max(1, n_contacts - 1)
        for t in range(max_steps):
            h = model._step(h, x)
            logits = model._readout(h)
            t_norm = torch.full((1,), t / denom, device=device)
            if torch.sigmoid(model._stop(h, t_norm, recruited.mean(-1))).item() > 0.5:
                break
            logits = logits.masked_fill(recruited > 0, NEG_INF)
            if bool((recruited > 0).all()):
                break
            pick = int(logits.argmax(-1).item())
            sequence.append([pick])
            x = torch.zeros(1, n_contacts, device=device)
            x[0, pick] = 1.0
            recruited[0, pick] = 1.0
        out.append(sequence)
    return out


def h2a_b(model, prep, cfg, role, observed_prefix_groups=2, steps=400, unit_seed='seed0',
          max_fit_events=3000, max_scored_events=3000, max_rollout_events=300):
    """C45: identical genuinely observed prefix, with and without the state."""
    unit = DECODER_ROOT / 'formal_units' / f'{cfg.subject}__own_a' / 'L3_LOCAL_PLUS_LEARNED_LR' / unit_seed
    cache = DECODER_ROOT / 'cache' / f'{cfg.subject}__own_a'
    if not unit.exists() or not cache.exists():
        return dict(status='NOT_RUNNABLE', reason='no qualified frozen continuation decoder for this subject',
                    looked_for=[str(unit), str(cache)])
    from src.topic5_group_event_state.v034_spatial_state.we_decoder import (
        load_frozen_decoder, align_events, decoder_tensors, event_batch, WEStateScorer, per_event_scores,
        forward_with_h0)
    device = torch.device(cfg.device)
    bundle = load_frozen_decoder(unit, cache, device=device)
    tensors = decoder_tensors(bundle, device)
    ours = prep.payload['event_time']
    index = align_events(np.asarray(ours, float), bundle.event_abs_time)
    aligned = np.flatnonzero(index >= 0)
    if not len(aligned):
        return dict(status='NOT_RUNNABLE', reason='no event of this producer aligns with the decoder cache')
    ep = D.event_packets(prep.payload)
    role_mask = D.role_mask(prep.split, role)
    fit_mask = prep.split['train_packet']
    # Only events whose own packet is a registered target of this role, and whose
    # prefix genuinely exists in the cache, may be scored.
    lengths = (bundle.ranks[index[aligned]] >= 0).sum(-1)
    ok = (lengths >= observed_prefix_groups + 1)
    scored = aligned[ok & role_mask[ep[aligned]]]
    fitting = aligned[ok & fit_mask[ep[aligned]]]
    # Deterministic time-ordered thinning keeps the consumer within its registered
    # budget. The stride is fixed before any score is read.
    n_fit_all, n_scored_all = int(len(fitting)), int(len(scored))
    if len(fitting) > max_fit_events:
        fitting = fitting[:: int(np.ceil(len(fitting) / max_fit_events))]
    if len(scored) > max_scored_events:
        scored = scored[:: int(np.ceil(len(scored) / max_scored_events))]
    if not len(scored) or not len(fitting):
        return dict(status='NOT_RUNNABLE',
                    reason='no aligned event with an observable prefix in both the FIT and the scored role',
                    n_aligned=int(len(aligned)), n_with_prefix=int(ok.sum()))

    @torch.no_grad()
    def states_for(events, mask_role):
        qs = np.unique(np.maximum(ep[events] - 1, 0))
        qs = qs[~prep.split['seizure_mask'][qs]]
        pos = {int(q): i for i, q in enumerate(qs)}
        chunks = []
        for a in range(0, len(qs), cfg.eval_chunk):
            chunks.append(infer_asof(model, prep, qs[a:a + cfg.eval_chunk], mask_role, cfg.history_hours,
                                     producer_hash=tensor_hash(model.state_dict())))
        m = torch.cat([c.m for c in chunks])
        keep = np.array([int(ep[e] - 1) in pos for e in events])
        rows = torch.stack([m[pos[int(ep[e] - 1)]] for e in events[keep]]) if keep.any() else None
        return rows, keep

    fit_state, fit_keep = states_for(fitting, 'fit')
    test_state, test_keep = states_for(scored, role)
    if fit_state is None or test_state is None:
        return dict(status='NOT_RUNNABLE', reason='no legal minute query precedes the aligned events')
    fitting = fitting[fit_keep]
    scored = scored[test_keep]
    mu = fit_state.mean(0)
    sd = fit_state.std(0, unbiased=False).clamp(min=.1)
    results = {}
    for use_state in (True, False):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed)
            scorer = WEStateScorer(bundle, state_dim=fit_state.shape[1], rank=8).to(device)
        opt = torch.optim.AdamW([p for n, p in scorer.named_parameters() if not n.startswith('decoder.')],
                                lr=.01, weight_decay=1e-3)
        n_val = max(1, int(.2 * len(fitting)))
        tr_idx = np.arange(len(fitting) - n_val)
        va_idx = np.arange(len(fitting) - n_val, len(fitting))
        cache_tr = torch.as_tensor(index[fitting[tr_idx]], device=device)
        cache_va = torch.as_tensor(index[fitting[va_idx]], device=device)
        s_tr = ((fit_state[tr_idx] - mu) / sd)
        s_va = ((fit_state[va_idx] - mu) / sd)

        def objective(batch_index, state):
            batch = event_batch(tensors, batch_index)
            h0 = scorer.initial_state(state, use_bias=True, use_state=use_state)
            if h0 is not None and h0.shape[0] == 1:
                h0 = h0.expand(batch['x'].shape[0], -1)
            logits, stops = forward_with_h0(bundle.model, batch['x'], batch['recruited'], batch['valid'], h0)
            return per_event_scores(logits, stops, batch, 1.0, observed_prefix_groups=observed_prefix_groups)

        best = float('inf')
        best_state = {k: v.clone() for k, v in scorer.state_dict().items() if not k.startswith('decoder.')}
        best_step = 0
        bad = 0
        for step in range(1, steps + 1):
            opt.zero_grad()
            out = objective(cache_tr, s_tr)
            loss = out['grammar'].mean()
            loss.backward()
            opt.step()
            if step % 20 == 0:
                with torch.no_grad():
                    v = float(objective(cache_va, s_va)['grammar'].mean())
                if v < best - 1e-5:
                    best, best_step, bad = v, step, 0
                    best_state = {k: q.clone() for k, q in scorer.state_dict().items()
                                  if not k.startswith('decoder.')}
                else:
                    bad += 1
                if bad >= 6:
                    break
        scorer.load_state_dict(best_state, strict=False)
        with torch.no_grad():
            s_te = (test_state - mu) / sd
            out = objective(torch.as_tensor(index[scored], device=device), s_te)
            teacher = {k: float(v.mean()) for k, v in out.items() if k != 'n_predict'}
            teacher_rows = {k: v.detach().cpu().numpy() for k, v in out.items()}
            # Free continuation from the same genuinely observed prefix. Generation
            # is a per-event Python loop with a device synchronisation at every
            # step, so it runs on a fixed, smaller stride of the same scored set.
            roll = np.arange(len(scored))
            if len(roll) > max_rollout_events:
                roll = roll[:: int(np.ceil(len(roll) / max_rollout_events))]
            ranks = bundle.ranks[index[scored[roll]]]
            starts = [np.flatnonzero((r >= 0) & (r < observed_prefix_groups)) for r in ranks]
            h0 = scorer.initial_state(s_te[roll], use_bias=True, use_state=use_state)
            if h0 is not None and h0.shape[0] == 1:
                h0 = h0.expand(len(starts), -1)
            gen = rollout_with_h0(bundle.model, starts, len(bundle.contact_names),
                                  len(bundle.contact_names), device, h0)
            truth = [list(np.argsort(np.where(r >= 0, r, 10 ** 6))[:int((r >= 0).sum())]) for r in ranks]
            hits, lens = [], []
            for g, t, s in zip(gen, truth, starts):
                produced = [c for grp in g[1:] for c in grp]
                remaining = [c for c in t if c not in set(s.tolist())]
                lens.append(abs(len(produced) - len(remaining)))
                hits.append(len(set(produced) & set(remaining)) / max(len(remaining), 1))
        results['with_state' if use_state else 'without_state'] = dict(
            teacher_forced=teacher, selected_step=best_step, validation=best,
            free_continuation=dict(mean_recall_of_remaining=float(np.mean(hits)),
                                   mean_absolute_length_error=float(np.mean(lens)),
                                   n_events=int(len(starts))),
            rows=teacher_rows)
    a, b = results['without_state'], results['with_state']
    return dict(status='COMPLETE', observed_prefix_groups=observed_prefix_groups,
                n_scored_events=int(len(scored)), n_fit_events=int(len(fitting)),
                thinning=dict(aligned_fit_events=n_fit_all, aligned_scored_events=n_scored_all,
                              max_fit_events=max_fit_events, max_scored_events=max_scored_events,
                              max_rollout_events=max_rollout_events,
                              rule='fixed time-ordered stride chosen before any score is read'),
                decoder=dict(unit=str(unit), cache=str(cache), contacts=list(bundle.contact_names),
                             arm=bundle.metrics['arm'], contract=bundle.metrics.get('contract')),
                arms=results,
                endpoints={k: dict(without_state=a['teacher_forced'][k], with_state=b['teacher_forced'][k],
                                   difference=a['teacher_forced'][k] - b['teacher_forced'][k])
                           for k in ('next_bce', 'stop_bce', 'contact_nll', 'grammar')},
                free_continuation={k: dict(without_state=a['free_continuation'][k],
                                           with_state=b['free_continuation'][k])
                                   for k in a['free_continuation']},
                interface_note=('the free-continuation arm needed an initial-state argument that the frozen '
                                'rollout helper does not expose; rollout_with_h0 copies every decision rule '
                                'verbatim and adds only that argument'),
                forbidden_inputs_excluded=['final K', 'normalised rank', 'whole-event waveform',
                                           'suffix statistics', 'event end time'])


# --------------------------------------------------------------------- S-A
def coordinate_layout(prep):
    names, family = [], []
    for k in range(prep.n_shaft):
        names.append(f'shaft_probability_{k}')
        family.append('spatial')
    for k in range(prep.band_ratio.shape[1]):
        names.append(f'band_ratio_mean_{k}')
        family.append('band_ratio')
    for k in range(prep.xlag.shape[1]):
        names.append(f'signed_xlag_mean_{k}')
        family.append('signed_xlag')
    names += ['delay_iqr_zero_probability', 'delay_iqr_positive_log_mean']
    family += ['delay_iqr', 'delay_iqr']
    return names, np.asarray(family)


@torch.no_grad()
def functional_coordinates(model, prep, cfg, state, reference, paths=None, seed=None):
    """C46: fixed functional coordinates of the frozen readout, and the same-noise reference.

    Only the current Q and the legal conditions are read.  No future real event
    count, participation or morphology enters this map.
    """
    from .engine import query_noise, sample_posterior
    paths = paths or cfg.eval_paths
    seed = cfg.eval_seed if seed is None else seed
    noise = query_noise(state, seed, 1, paths, prep.device, state.m.dtype)[0]
    ti = torch.as_tensor(state.query_packet, device=prep.device)
    cond = torch.cat((prep.clock_end[ti], state.cond), -1).unsqueeze(0).expand(paths, len(ti), -1)

    def coords(z):
        out = model.readout(z, cond)
        shaft = torch.softmax(out['composition'], -1).mean(0)
        br = out['band_ratio_mu'].mean(0)
        xl = out['xlag_mu'].mean(0)
        p0 = torch.sigmoid(out['iqr'][..., 0]).clamp(1e-6, 1 - 1e-6)
        # The positive-branch mean is a mixture of conditional means weighted by
        # the per-path probability of being positive, not an average of the
        # per-path conditional means.
        w = 1. - p0
        pos = (w * out['iqr'][..., 1]).sum(0) / w.sum(0).clamp(min=1e-9)
        return torch.cat((shaft, br, xl, p0.mean(0).unsqueeze(-1), pos.unsqueeze(-1)), -1)

    z = sample_posterior(state.m, state.P, paths, noise=noise)
    rz = sample_posterior(reference['m0'].expand_as(state.m), reference['P0'].expand_as(state.P),
                          paths, noise=noise)
    return coords(z) - coords(rz)


@torch.no_grad()
def s_a(model, prep, cfg, reference, stride=30, quick=False):
    pk = prep.payload['packets']
    names, family = coordinate_layout(prep)
    # FIT-frozen centre, scale and constant-dimension mask for the standardised summary.
    fit_q = np.flatnonzero(prep.split['train_packet'])[::120]
    fit_q = fit_q[~prep.split['seizure_mask'][fit_q]]
    fit_rows = []
    for a in range(0, len(fit_q), cfg.eval_chunk):
        st = infer_asof(model, prep, fit_q[a:a + cfg.eval_chunk], 'fit', cfg.history_hours,
                        producer_hash=tensor_hash(model.state_dict()))
        fit_rows.append(functional_coordinates(model, prep, cfg, st, reference))
    fit_delta = torch.cat(fit_rows).cpu().numpy()
    scale = np.percentile(fit_delta, 75, axis=0) - np.percentile(fit_delta, 25, axis=0)
    constant = scale <= 1e-9
    scale = np.where(constant, 1., scale)
    qs = np.flatnonzero(prep.split['valid_packet'])[::(stride if not quick else 120)]
    qs = qs[(pk['end'][qs] <= prep.split['support_end']) & ~prep.split['seizure_mask'][qs]]
    rows = []
    for a in range(0, len(qs), cfg.eval_chunk):
        st = infer_asof(model, prep, qs[a:a + cfg.eval_chunk], 'descriptive', cfg.history_hours,
                        producer_hash=tensor_hash(model.state_dict()))
        delta = functional_coordinates(model, prep, cfg, st, reference).cpu().numpy()
        for j, q in enumerate(st.query_packet):
            recent = available_recent(prep, int(q), int(cfg.short_history_minutes))
            rows.append(dict(query=int(q), time=float(st.query_time[j]), delta=delta[j],
                             has_observation=bool(np.isfinite(st.release_time[j])),
                             age=float(st.information_age_minutes[j]),
                             clock=int(D.clock_stratum([st.query_time[j]])[0]),
                             exposure_fraction=float(st.cond[j, CD.NAMES.index('short_window_published_coverage')]),
                             log_segment_age=float(st.cond[j, CD.NAMES.index('log_episode_age_hours')]),
                             recent_rate=recent['rate_per_hour'], recent_coverage=recent['coverage'],
                             log_rate=None, variance=float(st.P[j].diagonal().mean())))
    seizures = prep.split['seizures']
    safe = controls_mask(np.array([r['time'] for r in rows]), seizures)
    onsets = sorted(float(s['onset_epoch']) for s in seizures)
    cluster_of = {}
    cid = 0
    for i, o in enumerate(onsets):
        if i and o - onsets[i - 1] > CLUSTER_GAP_HOURS * 3600:
            cid += 1
        cluster_of[o] = cid

    def match(case_rows, use_rate):
        used = set()
        pairs = []
        for r in case_rows:
            pool = [j for j, s in enumerate(rows)
                    if safe[j] and j not in used and s['has_observation'] and s['clock'] == r['clock']
                    and abs(s['age'] - r['age']) <= DONOR_RULE['max_support_age_minutes']
                    and abs(s['exposure_fraction'] - r['exposure_fraction'])
                    <= DONOR_RULE['max_exposure_fraction_difference']
                    and abs(s['log_segment_age'] - r['log_segment_age'])
                    <= DONOR_RULE['max_log_segment_age_difference']]
            if use_rate:
                if r['recent_rate'] is None:
                    continue
                pool = [j for j in pool if rows[j]['recent_rate'] is not None
                        and abs(np.log1p(rows[j]['recent_rate']) - np.log1p(r['recent_rate'])) <= .5]
            if not pool:
                continue
            j = min(pool, key=lambda j: abs(rows[j]['time'] - r['time']))
            used.add(j)
            pairs.append((r, rows[j]))
        return pairs

    cases = {}
    for seizure in seizures:
        onset = float(seizure['onset_epoch'])
        for window, lo, hi in S_A_WINDOWS:
            pre = [r for r in rows if onset + lo <= r['time'] < onset + hi and r['has_observation']]
            entry = dict(seizure_id=seizure.get('seizure_id'), onset_epoch=onset, window=window,
                         cluster=cluster_of[onset], relative_seconds=[lo, hi], n_pre_queries=len(pre),
                         after_producer_fit=onset > prep.split['fit_end'])
            for label, use_rate in (('support_matched', False), ('support_and_recent_rate_matched', True)):
                pairs = match(pre, use_rate)
                if not pairs:
                    entry[label] = dict(status='NOT_ESTIMABLE', n_pairs=0)
                    continue
                d = np.stack([a['delta'] - b['delta'] for a, b in pairs]).mean(0)
                z = d / scale
                # A FIT-constant coordinate carries no information and is dropped from
                # its family mean rather than diluting it with a zero.
                live = ~constant
                def family_mean(name):
                    sel = (family == name) & live
                    return float(np.mean(z[sel] ** 2)) if sel.any() else None
                spatial = family_mean('spatial')
                morph_parts = [v for v in (family_mean(f) for f in
                                           ('band_ratio', 'signed_xlag', 'delay_iqr')) if v is not None]
                morph = float(np.mean(morph_parts)) if morph_parts else None
                if spatial is None and morph is None:
                    overall = None
                elif spatial is None:
                    overall = float(np.sqrt(morph))
                elif morph is None:
                    overall = float(np.sqrt(spatial))
                else:
                    overall = float(np.sqrt(.5 * spatial + .5 * morph))
                entry[label] = dict(status='DESCRIPTIVE', n_pairs=len(pairs),
                                    n_live_coordinates=int(live.sum()),
                                    morphology_families_used=len(morph_parts),
                                    signed_coordinate_difference={n: float(v) for n, v in zip(names, d)},
                                    standardised={n: float(v) for n, v in zip(names, z)},
                                    overall_magnitude=overall,
                                    control_queries=[int(b['query']) for _, b in pairs],
                                    variance_difference=float(np.mean([a['variance'] - b['variance']
                                                                       for a, b in pairs])))
            cases[f"{seizure.get('seizure_id')}__{window}"] = entry
    common = sorted(k for k, v in cases.items()
                    if all(v.get(l, {}).get('status') == 'DESCRIPTIVE'
                           for l in ('support_matched', 'support_and_recent_rate_matched')))
    return dict(status='DESCRIPTIVE' if cases else 'NOT_ESTIMABLE', coordinate_names=names,
                coordinate_family=family.tolist(), constant_coordinates=[n for n, c in zip(names, constant) if c],
                fit_scale={n: float(s) for n, s in zip(names, scale)},
                n_reference_queries=int(len(fit_delta)), n_descriptive_queries=len(rows),
                cases=cases, cases_estimable_under_both_matchings=common,
                clusters=sorted(set(cluster_of.values())),
                interpretation=('per-coordinate signed differences against matched controls; the overall '
                                'magnitude is non-negative by construction and is not a positivity verdict. '
                                'Within-case queries are equally weighted and seizures are grouped into '
                                'clusters; reused controls are not independent seizures and rare seizures '
                                'do not manufacture cohort significance.'),
                matching=dict(support_only=list(DONOR_RULE), plus_recent_rate='log1p recent rate within 0.5',
                              rate_window_minutes=int(cfg.short_history_minutes)))


# --------------------------------------------------------------------- S-B
@torch.no_grad()
def s_b(model, prep, cfg, bundle, source_root=SPATIAL_SOURCE):
    """C48: within-community Spearman between the frozen interictal head and clinical onset."""
    from ..v0312.spatial_transfer import inventory_crosswalk
    root = Path(source_root)
    cache = root / 'topic5_ictal_recruitment/t0_feature_cache_bb150_1_150'
    inventory = root / 'epilepsiae_seizure_inventory.csv'
    if not inventory.exists():
        inventory = root / 'dataset_inventory/epilepsiae_seizure_inventory.csv'
    npz = cache / f'{cfg.subject}.npz'
    side = cache / f'{cfg.subject}.json'
    missing = [str(p) for p in (inventory, npz, side) if not p.exists()]
    if missing:
        return dict(status='NOT_AVAILABLE', missing=missing)
    meta = json.loads(side.read_text())
    if meta['t_window'] != [0., 10.] or meta['band_broad_1_150'] != [1., 150.]:
        raise ValueError('the ictal measurement contract changed')
    heads = {k: v for k, v in bundle['heads'].items() if v['status'] == 'COMPLETE'}
    if not heads:
        return dict(status='NOT_ESTIMABLE', reason='no qualified frozen interictal identity head')
    with open(inventory) as f:
        cross = inventory_crosswalk(cfg.subject, list(csv.DictReader(f)), prep.split['seizures'])
    cross = [r for r in cross if prep.split['support_start'] <= r['eeg_onset']
             and r['clinical_onset'] + 10 <= prep.split['support_end']]
    pk = prep.payload['packets']
    allowed = D.input_mask(prep.split, 'descriptive')
    community = np.asarray(prep.payload['shaft_index'])
    rows = []
    with np.load(npz, allow_pickle=False) as zz:
        names = zz['channels'].astype(str).tolist()
        lookup = {n: i for i, n in enumerate(names)}
        join = np.array([lookup.get(n, -1) for n in prep.payload['selected_contacts']])
        ok = join >= 0
        for r in cross:
            key = f"bb150_auc__{r['source_index']}"
            if key not in zz:
                continue
            q = int(np.searchsorted(pk['end'], r['eeg_onset'] - 300, side='right') - 1)
            if q < 0 or prep.split['seizure_mask'][q]:
                continue
            st = infer_asof(model, prep, [q], 'descriptive', cfg.history_hours,
                            producer_hash=tensor_hash(model.state_dict()))
            if not np.isfinite(st.release_time[0]):
                continue
            tt = pk['end'][q]
            window = float(cfg.short_history_minutes) * 60.
            ix = np.flatnonzero(allowed & (pk['release'] <= tt) & (pk['end'] <= tt) & (pk['end'] > tt - window)
                                & (np.arange(prep.n_packets) >= st.prefix_start[0]))
            ei = np.concatenate([np.arange(pk['event_lo'][i], pk['event_hi'][i]) for i in ix]) if len(ix) else np.empty(0, int)
            cnt = prep.part[ei].sum(0) if len(ei) else prep.part.new_zeros(prep.part.shape[1])
            hist = torch.logit(((cnt + .5) / (len(ei) + 1.)).clamp(1e-4, 1 - 1e-4))
            cond = torch.cat((prep.clock[min(q + 1, prep.n_packets - 1)], st.cond[0]))
            target = np.full(len(join), np.nan)
            target[ok] = np.asarray(zz[key])[join[ok]]
            valid = ok & np.isfinite(target)
            scores = {}
            for label, head in heads.items():
                f = head['fitted']
                parts = []
                for k in FEATURES[label]:
                    parts.append(st.m[0] if k == 'state' else hist)
                x = (torch.cat(parts + [cond]) - f['center']) / f['scale']
                logits = (x @ f['state_dict']['weight'].T + f['state_dict']['bias']).cpu().numpy()
                keep = valid & np.isfinite(logits)
                within = []
                for c in np.unique(community):
                    sel = keep & (community == c)
                    if sel.sum() < 3 or np.std(logits[sel]) < 1e-9 or np.std(target[sel]) < 1e-9:
                        continue
                    within.append(dict(community=int(c), n=int(sel.sum()),
                                       spearman=float(spearmanr(logits[sel], target[sel]).statistic)))
                overall = (float(spearmanr(logits[keep], target[keep]).statistic)
                           if keep.sum() >= 3 and np.std(logits[keep]) > 1e-9 and np.std(target[keep]) > 1e-9
                           else None)
                scores[label] = dict(
                    within_community=within,
                    within_community_mean=float(np.mean([w['spearman'] for w in within])) if within else None,
                    n_qualified_communities=len(within),
                    all_contact_spearman_auxiliary=overall)
            rows.append(dict(**r, query=q, query_time=float(tt), n_contacts=int(valid.sum()),
                             seconds_before_eeg_onset=float(r['eeg_onset'] - tt), scores=scores,
                             after_producer_fit=r['eeg_onset'] > prep.split['fit_end']))
    sizes = {int(c): int((community == c).sum()) for c in np.unique(community)}
    paired = {}
    for label in heads:
        vals = [r['scores'][label]['within_community_mean'] for r in rows
                if r['scores'][label]['within_community_mean'] is not None]
        paired[label] = dict(n_cases=len(vals), mean=float(np.mean(vals)) if vals else None,
                             values=[float(v) for v in vals])
    return dict(status='DESCRIPTIVE' if rows else 'NOT_ESTIMABLE', rows=rows, per_condition_set=paired,
                community_sizes=sizes,
                qualified_communities=[c for c, s in sizes.items() if s >= 3],
                measurement='1-150 Hz baseline robust-z activation, clinical onset [0,10] seconds, CAR',
                primary='within-community Spearman on identical valid contacts, equal weight over qualified communities',
                limitation=('only communities with at least three valid contacts are defined; the all-contact '
                            'Spearman is retained as the legacy convention and carries a cross-community '
                            'offset bias that the conditional likelihood does not identify'),
                interpretation=('a frozen interictal contact readout compared with a separately measured '
                                'ictal spatial field; it is not EEG-onset forecasting, not an ictally trained '
                                'decoder and not propagation'),
                sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (inventory, npz, side)})


def run_consumers(selected, out_dir, device='cpu', quick=False):
    """G2: everything downstream of one frozen S_marks producer."""
    model, prep, cfg, record = load_selected(selected, device)
    if quick:
        cfg = replace(cfg, eval_stride=180, eval_paths=8, eval_chunk=8)
    role = 'outer' if cfg.stage == 'outer' else 'inner'
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prep.frozen_query_cache = {}
    before = tensor_hash(model.state_dict())
    community = torch.as_tensor(prep.payload['shaft_index'], device=prep.device)
    result = dict(status='COMPLETE', producer_hash=before, source_digest=record['source_digest'],
                  config=vars(cfg), selected=str(selected), score_role=role,
                  scope='engineering_smoke' if quick else 'development',
                  producer_selected_by='the interictal task alone; no seizure or consumer score reached back')
    import time as _time
    timings = {}

    def _stage(name, fn):
        t0 = _time.time()
        try:
            return fn()
        finally:
            timings[name] = round(_time.time() - t0, 1)
            print(f'[consumers] {name} {timings[name]}s', flush=True)

    reference = _stage('fit_reference', lambda: fit_reference(model, prep, cfg))
    atomic_torch(reference, out / 'fit_reference.pt')
    bundle = _stage('H2a_A', lambda: h2a_a(model, prep, cfg, role, community, steps=20 if quick else 400))
    atomic_torch({k: v for k, v in bundle.items() if k == 'data'}, out / 'h2a_a_data.pt')
    result['H2a_A'] = dict(status=bundle['status'],
                           heads={k: {kk: vv for kk, vv in v.items() if kk not in ('rows', 'fitted')}
                                  for k, v in bundle['heads'].items()},
                           contrasts=bundle.get('contrasts', {}),
                           query_interface=bundle.get('query_interface'),
                           task_name=bundle.get('task_name'), caveat=bundle.get('caveat'))
    if bundle['status'] == 'COMPLETE':
        result['wrong_time'] = _stage('wrong_time', lambda: wrong_time(model, prep, cfg, bundle, reference, community))
    else:
        result['wrong_time'] = dict(status='NOT_ESTIMABLE', reason='no qualified C+H+S head')
    try:
        # H2a-B carries its own registered budget: the 400-step cap in the work
        # package governs the conditional identity heads. The frozen tissue
        # decoder is evaluated over thousands of events per step, so its h0
        # adapter (a bias plus a rank-8 map, under a thousand parameters) is
        # fitted on a smaller registered event budget.
        result['H2a_B'] = _stage('H2a_B', lambda: h2a_b(model, prep, cfg, role,
                                                        steps=20 if quick else 200,
                                                        max_fit_events=1200))
    except Exception as exc:
        result['H2a_B'] = dict(status='NOT_RUNNABLE', reason=f'{type(exc).__name__}: {exc}')
    result['S_A'] = _stage('S_A', lambda: s_a(model, prep, cfg, reference, quick=quick))
    if bundle['status'] == 'COMPLETE':
        result['S_B'] = _stage('S_B', lambda: s_b(model, prep, cfg, bundle))
    else:
        result['S_B'] = dict(status='NOT_ESTIMABLE', reason='no qualified frozen interictal identity head')
    result['S_C'] = dict(status='NOT_RUN',
                         reason=('no complete prospective observable person-hour denominator, event '
                                 'annotation and legal time validation are registered for this window'),
                         blocks_only='S-C; S-A and S-B are unaffected')
    result['stage_seconds'] = timings
    result['producer_unchanged'] = before == tensor_hash(model.state_dict())
    if not result['producer_unchanged']:
        raise RuntimeError('the frozen producer changed during consumer evaluation')
    atomic_json(result, out / 'consumers.json')
    return result
