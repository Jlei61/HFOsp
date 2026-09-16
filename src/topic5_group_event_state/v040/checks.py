"""Interface checks for the first package: weights, normalisation, cutoff, replay, streams.

These verify that the implementation computes the registered formulas.  They
are not a new training stage and they do not certify any scientific claim.
"""
from __future__ import annotations
import math
from dataclasses import replace

import numpy as np
import torch

from . import data as D
from .engine import infer_asof, predict, score_predictions
from .objective import (morphology_terms, window_scores, training_loss, FAMILIES, VIEWS, SECONDARY,
                        HORIZON_WEIGHTS, VIEW_WEIGHTS, _normal_component_lp)
from ..v0312.frozen import conditional_set_lp, log_elementary_symmetric, identity_units


def _norm_logpdf(y, mu, sd):
    return -0.5 * ((y - mu) / sd) ** 2 - np.log(sd) - 0.5 * math.log(2 * math.pi)


@torch.no_grad()
def check_morphology_estimator(model, prep, cfg, role='inner', tol=2e-5):
    """C12/C13: recompute the nested estimator independently, event by event."""
    table = D.target_table(prep.payload, prep.split, role, cfg.eval_stride)
    qs = np.unique(table[:, 2])[:cfg.eval_chunk]
    state = infer_asof(model, prep, qs, role, cfg.history_hours)
    pred = predict(model, prep, state, (30,), paths=8, seed=cfg.eval_seed)[30]
    ti = torch.as_tensor(state.query_packet + 30, device=prep.device)
    morph = morphology_terms(model.readout, prep, ti, state.cond, pred.grid)
    if morph is None:
        return dict(status='NOT_ESTIMABLE', reason='no events in the checked windows')
    # Independent recomputation of every component mixture from the raw readout.
    S = pred.grid.shape[1]
    idx = morph['event_index'].cpu().numpy()
    row = morph['event_row'].cpu().numpy()
    start = prep.packet_start[ti.cpu().numpy()]
    frac = np.clip((prep.event_times[morph['event_index']].cpu().numpy() - start[row]) / 60., 0, 1)
    gi = np.minimum((frac * (pred.grid.shape[0] - 1)).astype(int), pred.grid.shape[0] - 1)
    worst = 0.
    checked = 0
    for i in range(0, len(idx), max(1, len(idx) // 40)):
        e = int(idx[i])
        z = pred.grid[gi[i], :, int(row[i])]
        phase = float(prep.event_times[e]) * (2 * math.pi / 86400.)
        clock = torch.tensor([math.sin(phase), math.cos(phase)], device=prep.device, dtype=z.dtype)
        cond = torch.cat((clock, state.cond[int(row[i])])).unsqueeze(0).expand(S, -1)
        out = model.readout(z, cond)
        fam_nll = []
        for name in FAMILIES:
            if name == 'band_ratio':
                y = prep.band_ratio[e].cpu().numpy()
                v = prep.band_ratio_valid[e].cpu().numpy()
                mu = out['band_ratio_mu'].cpu().numpy()
                sd = np.exp(np.clip(out['band_ratio_logsd'].cpu().numpy(), -6, 4))
                lp = _norm_logpdf(y[None, :], mu, sd)
            elif name == 'signed_xlag':
                y = prep.xlag[e].cpu().numpy()
                v = prep.xlag_valid[e].cpu().numpy()
                mu = out['xlag_mu'].cpu().numpy()
                sd = np.exp(np.clip(out['xlag_logsd'].cpu().numpy(), -6, 4))
                lp = _norm_logpdf(y[None, :], mu, sd)
            else:
                raw = float(prep.iqr[e])
                v = np.array([float(prep.iqr_valid[e])])
                par = out['iqr'].cpu().numpy()
                p0 = np.clip(1 / (1 + np.exp(-par[:, 0])), 1e-6, 1 - 1e-6)
                w = (math.log(max(raw, 1e-9)) - float(prep.log_iqr_center)) / float(prep.log_iqr_scale)
                sd = np.exp(np.clip(par[:, 2], -6, 4))
                pos = np.log1p(-p0) - np.log(sd) - 0.5 * math.log(2 * math.pi) - 0.5 * ((w - par[:, 1]) / sd) ** 2
                lp = (np.log(p0) if raw <= 0 else pos)[:, None]
            if v.sum() <= 0:
                continue
            comp = -(np.log(np.exp(lp - lp.max(0)).mean(0)) + lp.max(0))
            fam_nll.append(float((comp * v).sum() / v.sum()))
        if not fam_nll:
            continue
        worst = max(worst, abs(float(np.mean(fam_nll)) - float(morph['event_nll'][i])))
        checked += 1
    # C13c/C13d: window value is the equal-weight mean over its scorable events.
    ev = morph['event_valid'].cpu().numpy()
    en = morph['event_nll'].cpu().numpy()
    wn = morph['window_nll'].cpu().numpy()
    wv = morph['window_valid'].cpu().numpy()
    window_error = 0.
    for b in np.flatnonzero(wv):
        sel = (row == b) & ev
        window_error = max(window_error, abs(en[sel].mean() - wn[b]))
    n_fam = morph['n_families_per_event'].cpu().numpy()
    return dict(status='COMPLETE' if max(worst, window_error) < tol else 'FAILED',
                n_events_checked=checked, max_event_error=float(worst), max_window_error=float(window_error),
                tolerance=tol, n_events=int(len(idx)), n_windows=int(wv.sum()),
                events_without_any_family=int((n_fam == 0).sum()),
                windows_with_events_but_no_family=int(((~wv) & (np.bincount(row, minlength=len(wv)) > 0)).sum()),
                rule='per-component mixture, family mean, event mean over families, window mean over events')


@torch.no_grad()
def check_weight_formula(model, prep, cfg, role='inner'):
    """C20: registered horizon/view weights and microbatch-invariant denominators."""
    from .train import training_ids, training_normalizers, training_table
    ids = training_ids(prep)
    table = training_table(prep)
    pk = prep.payload['packets']
    n_ev = (pk['event_hi'] - pk['event_lo'])[ids]
    picks = []
    for want in (n_ev == 0, (n_ev > 0) & (n_ev <= 2), n_ev >= 10):
        cand = ids[want]
        picks.extend(cand[:3].tolist())
    picks = np.unique(picks)[:8]
    if len(picks) < 3:
        return dict(status='NOT_ESTIMABLE', reason='no batch with zero, few and many events')
    normalizers = training_normalizers(prep, cfg.batch_size)
    legal = table[np.isin(table[:, 0], picks)]
    qs = np.unique(legal[:, 2])
    state = infer_asof(model, prep, qs, 'fit', cfg.history_hours)
    pr = predict(model, prep, state, tuple(np.unique(legal[:, 1])), paths=4, seed=cfg.eval_seed)
    rows = score_predictions(model, prep, state, pr, 'fit', picks)
    total = 0.
    manual = 0.
    detail = []
    for r in rows:
        h = r['horizon']
        term = training_loss({v: r[v] for v in VIEWS + SECONDARY}, h, normalizers[h])
        total += float(term) if term is not None else 0.
        for view in VIEWS + SECONDARY:
            den = normalizers[h][view]
            if den <= 0:
                continue
            s = r[view]
            contrib = HORIZON_WEIGHTS[h] * VIEW_WEIGHTS[view] * float((s['nll'] * s['valid']).sum()) / den
            manual += contrib
            detail.append(dict(horizon=h, view=view, n_valid=int(s['valid'].sum()), denominator=den,
                               contribution=contrib))
    # Equal window weight: a 20-event window and a 1-event window weigh the same.
    morph_rows = [r for r in rows if 'morphology_events' in r]
    counts = []
    for r in morph_rows:
        e = r['morphology_events']
        b = np.bincount(e['event_row'].cpu().numpy(), weights=e['event_valid'].cpu().numpy().astype(float),
                        minlength=len(r['packet']))
        counts.extend(b[b > 0].tolist())
    return dict(status='COMPLETE' if abs(total - manual) < 1e-6 else 'FAILED',
                loss=total, independently_recomputed=manual, difference=abs(total - manual),
                horizon_weights=HORIZON_WEIGHTS, view_weights=VIEW_WEIGHTS,
                batch_targets=[int(i) for i in picks], events_per_target=[int(v) for v in (pk['event_hi'] - pk['event_lo'])[picks]],
                scored_events_per_window=sorted(set(int(c) for c in counts)),
                terms=detail,
                note='windows with different event counts enter with identical weight; a zero-event window '
                     'still scores the count target and contributes no morphology weight')


def check_set_normalization(n_contacts=6, n_community=2, seed=7):
    """C43: the conditional identity head normalises exactly, by enumeration."""
    g = torch.Generator().manual_seed(seed)
    community = torch.arange(n_contacts) // (n_contacts // n_community)
    logits = torch.randn(1, n_contacts, generator=g)
    results = []
    from itertools import product
    sizes = {}
    for c in torch.unique(community):
        cols = int((community == c).sum())
        sizes[int(c)] = cols
    for ks in product(*[range(sizes[int(c)] + 1) for c in torch.unique(community)]):
        members = []
        # enumerate every membership vector with those per-community sizes
        options = []
        for c, k in zip(torch.unique(community), ks):
            cols = np.flatnonzero((community == c).numpy())
            from itertools import combinations
            options.append([set(s) for s in combinations(cols.tolist(), k)])
        total = 0.
        n = 0
        for combo in product(*options):
            v = torch.zeros(1, n_contacts)
            for s in combo:
                for j in s:
                    v[0, j] = 1.
            total += float(conditional_set_lp(logits, v, community).exp())
            n += 1
        results.append(dict(sizes=list(ks), n_sets=n, probability_mass=total))
    worst = max(abs(r['probability_mass'] - 1.) for r in results)
    members = torch.zeros(3, n_contacts)
    members[0, 0] = 1.
    members[1] = 1.
    members[2, :2] = 1.
    units = identity_units(members, community).tolist()
    return dict(status='COMPLETE' if worst < 1e-5 else 'FAILED', max_mass_error=float(worst),
                enumerated_size_profiles=len(results), by_profile=results,
                deterministic_units=dict(single_contact=units[0], full_universe=units[1], one_per_community=units[2]),
                rule='per community, exp(sum l_j) / e_K(exp l); K=0 and K=C are deterministic and are not units')


def check_release_cutoff(payload, subject, split_seed=D.SPLIT_SEED):
    """C7: nothing released after the FIT boundary reaches a fitted object."""
    audit = D.training_cutoff_audit(payload, subject, split_seed)
    violations = [s for s, v in audit.items() if not (v['release_within_cutoff'] and v['event_release_within_cutoff'])]
    return dict(status='COMPLETE' if not violations else 'FAILED', violations=violations, stages=audit)


@torch.no_grad()
def check_replay_and_streams(model, prep, cfg, role='inner', tol=1e-5):
    """C27/C28/C29: batching, horizon subsets and chunk size must not move a score."""
    table = D.target_table(prep.payload, prep.split, role, cfg.eval_stride)
    qs = np.unique(table[:, 2])[:8]
    whole = infer_asof(model, prep, qs, role, cfg.history_hours)
    parts = [infer_asof(model, prep, qs[a:a + 3], role, cfg.history_hours) for a in range(0, len(qs), 3)]
    if whole.m is not None:
        m_err = float((whole.m - torch.cat([p.m for p in parts])).abs().max())
        P_err = float((whole.P - torch.cat([p.P for p in parts])).abs().max())
    else:
        m_err = float((whole.history - torch.cat([p.history for p in parts])).abs().max())
        P_err = 0.
    cond_err = float((whole.cond - torch.cat([p.cond for p in parts])).abs().max())
    full = predict(model, prep, whole, (1, 5, 30, 120), paths=8, seed=cfg.eval_seed)
    subset = predict(model, prep, whole, (5, 30), paths=8, seed=cfg.eval_seed)
    horizon_err = max(float((full[h].z_end - subset[h].z_end).abs().max()) for h in (5, 30))
    scores_a = score_predictions(model, prep, whole, {30: full[30]}, role, np.unique(table[:, 0]))
    scores_b = []
    for p in parts:
        pr = predict(model, prep, p, (1, 5, 30, 120), paths=8, seed=cfg.eval_seed)
        scores_b.extend(score_predictions(model, prep, p, {30: pr[30]}, role, np.unique(table[:, 0])))
    def flat(rows):
        out = {}
        for r in rows:
            for view in VIEWS:
                for pkt, nll, ok in zip(r['packet'], r[view]['nll'].cpu().numpy(), r[view]['valid'].cpu().numpy()):
                    if ok:
                        out[(view, int(pkt))] = float(nll)
        return out
    fa, fb = flat(scores_a), flat(scores_b)
    common = sorted(set(fa) & set(fb))
    score_err = max((abs(fa[k] - fb[k]) for k in common), default=0.)
    ok = max(m_err, P_err, cond_err, horizon_err, score_err) < tol
    return dict(status='COMPLETE' if ok else 'FAILED', tolerance=tol,
                continuous_vs_batched_state_max_abs=m_err, covariance_max_abs=P_err,
                condition_max_abs=cond_err, horizon_subset_max_abs=horizon_err,
                chunked_score_max_abs=score_err, n_common_targets=len(common),
                rule='one physical-query-keyed noise stream; chunk size and horizon subset are bookkeeping only')


@torch.no_grad()
def check_extreme_scores(model, prep, cfg, role='inner', top=10):
    """Locate the largest NLL terms and split them into mean, scale, tail and support."""
    table = D.target_table(prep.payload, prep.split, role, cfg.eval_stride)
    qs = np.unique(table[:, 2])
    worst = []
    for a in range(0, len(qs), cfg.eval_chunk):
        st = infer_asof(model, prep, qs[a:a + cfg.eval_chunk], role, cfg.history_hours)
        pr = predict(model, prep, st, (30,), paths=cfg.eval_paths, seed=cfg.eval_seed)
        ti = torch.as_tensor(st.query_packet + 30, device=prep.device)
        keep = (st.query_packet + 30) < prep.n_packets
        if not keep.any():
            continue
        ii = torch.as_tensor(np.flatnonzero(keep), device=prep.device)
        scores, morph = window_scores(model.readout, prep, ti[ii], st.cond[ii],
                                      pr[30].z_start[:, ii], pr[30].z_end[:, ii], pr[30].grid[:, :, ii])
        for view in VIEWS + SECONDARY:
            nll = scores[view]['nll'].cpu().numpy()
            ok = scores[view]['valid'].cpu().numpy()
            for j in np.flatnonzero(ok):
                worst.append(dict(view=view, packet=int(ti[ii][j]), nll=float(nll[j]),
                                  count=float(prep.count[ti[ii][j]]),
                                  exposure_seconds=float(prep.exposure_hours[ti[ii][j]]) * 3600,
                                  events=int(prep.event_hi[ti[ii][j]] - prep.event_lo[ti[ii][j]])))
    worst.sort(key=lambda r: -r['nll'])
    by_view = {}
    for view in VIEWS + SECONDARY:
        vals = np.array([r['nll'] for r in worst if r['view'] == view])
        if not len(vals):
            continue
        by_view[view] = dict(n=len(vals), median=float(np.median(vals)), p99=float(np.percentile(vals, 99)),
                             max=float(vals.max()), finite=bool(np.isfinite(vals).all()))
    return dict(status='COMPLETE', per_view=by_view, largest=worst[:top],
                note=('window-level negative log-likelihoods on their own physical support; no score is '
                      'clipped, and a large finite value is reported with its count, exposure and event load'))
