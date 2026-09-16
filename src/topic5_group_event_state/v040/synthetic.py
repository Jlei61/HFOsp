"""Two targeted calibration worlds for the rich-input and earlier-history contrasts.

``conditional_zero`` (C37): the future target is generated directly from the
readable coarse history and the legal background conditions plus independent
noise, so the extra marks carry no incremental information *under the conditions
the model can actually read*.  Rate, exposure and time memory stay predictive.

``organization_positive`` (C38/C39): identical in every other respect, but a
fixed summary of one published *earlier* mark channel enters the target with a
standardised coefficient of exactly 1 and an independent target noise standard
deviation of exactly 1.  The summary window sits four to six hours before the
target, outside the reach of the two-hour short arm and outside the recent rich
window, and no coarse channel can reconstruct it.

Three realizations per task calibrate direction, not an error rate (C41).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from ..v0311.synthetic import make_world as skeleton
from . import data as D
from .train import atomic_json, atomic_torch, RunConfig

OLD_WINDOW_MINUTES = (360, 240)     # [t-360, t-240): one 120-minute bin, four hours back
BIN_MINUTES = 120
RATE_TAU_MINUTES = 30.
OLD_SUMMARY_COEFFICIENT = 1.0
TARGET_NOISE_SD = 1.0
READABLE_RATE_COEFFICIENT = 0.8
READABLE_CLOCK_COEFFICIENT = 0.5
MARK_CHANNEL_NOISE = 0.5


def _standardize(x, mask=None):
    v = np.asarray(x, float)
    sel = v[mask] if mask is not None else v
    sel = sel[np.isfinite(sel)]
    mu = float(sel.mean()) if len(sel) else 0.
    sd = float(sel.std()) if len(sel) and sel.std() > 1e-9 else 1.
    return (v - mu) / sd, mu, sd


def make_task_world(task, realization, n_blocks=72, short_minutes=120):
    if task not in ('conditional_zero', 'organization_positive'):
        raise ValueError(task)
    kappa = OLD_SUMMARY_COEFFICIENT if task == 'organization_positive' else 0.
    p = skeleton(seed=realization, n_blocks=n_blocks, n_contacts=6, morph_gain=0., identity_gain=0.)
    rng = np.random.default_rng(realization + 5_000_003)
    pk = p['packets']
    n_pk = len(pk['start'])
    times = p['event_time']
    N = len(times)
    packet_of = np.clip(np.searchsorted(pk['end'], times, 'left'), 0, n_pk - 1)

    # Readable coarse memory: an exponentially weighted count rate over the true
    # per-packet counts. It is a coarse-history quantity; nothing in the marks
    # gives a better handle on it.
    counts = (pk['event_hi'] - pk['event_lo']).astype(float)
    expo = pk['exposure'] / 60.
    a = float(np.exp(-1. / RATE_TAU_MINUTES))
    num = np.zeros(n_pk)
    den = np.zeros(n_pk)
    for k in range(1, n_pk):
        num[k] = a * num[k - 1] + counts[k - 1]
        den[k] = a * den[k - 1] + expo[k - 1]
    rate = np.log1p(np.divide(num, np.maximum(den, 1e-9)))
    z_rate, rate_mu, rate_sd = _standardize(rate)

    # The extra-mark channel: a two-hour piecewise-constant driver, independent of
    # the rate process, of the clock and of participation, so no coarse channel
    # can reconstruct it and adjacent bins are independent of one another.
    t0 = float(pk['start'][0])
    bin_of_event = ((times - t0) // (BIN_MINUTES * 60.)).astype(int)
    n_bins = int(bin_of_event.max()) + 2
    v = rng.normal(size=n_bins)
    # With a finite number of bins an i.i.d. driver still correlates with the
    # readable background by chance. The driver is projected off the readable
    # bin-level basis so the coarse conditions genuinely cannot reconstruct it.
    ephase = times * (2 * np.pi / 86400.)
    basis = np.ones((n_bins, 4))
    for k, col in enumerate((np.sin(ephase), np.cos(ephase), z_rate[packet_of])):
        totals = np.bincount(bin_of_event, weights=col, minlength=n_bins)
        n_in = np.bincount(bin_of_event, minlength=n_bins).clip(min=1)
        basis[:, k + 1] = totals / n_in
    v = v - basis @ np.linalg.lstsq(basis, v, rcond=None)[0]
    v = v / max(float(v.std()), 1e-9)
    u = v[bin_of_event] + MARK_CHANNEL_NOISE * rng.normal(size=N)

    # The registered old summary: the mean of u over the events published inside
    # [t-360, t-240) minutes before the target packet.
    lo_s, hi_s = OLD_WINDOW_MINUTES[0] * 60., OLD_WINDOW_MINUTES[1] * 60.
    old = np.full(n_pk, np.nan)
    n_old_events = np.zeros(n_pk, int)
    ends = pk['end']
    for k in range(n_pk):
        sel = (times >= ends[k] - lo_s) & (times < ends[k] - hi_s)
        n_old_events[k] = int(sel.sum())
        if sel.any():
            old[k] = float(u[sel].mean())
    defined = np.isfinite(old)
    z_old_all, old_mu, old_sd = _standardize(old, defined)
    z_old = np.where(defined, z_old_all, 0.)

    phase = times * (2 * np.pi / 86400.)
    readable = (READABLE_RATE_COEFFICIENT * z_rate[packet_of]
                + READABLE_CLOCK_COEFFICIENT * np.sin(phase)
                + READABLE_CLOCK_COEFFICIENT * np.cos(phase))
    signal = readable + kappa * z_old[packet_of]

    t = p['targets']
    br = np.asarray(t['band_ratio']).copy()
    xl = np.asarray(t['signed_xlag']).copy()
    br[:] = rng.normal(size=br.shape)
    xl[:] = rng.normal(size=xl.shape)
    br[:, 0] = signal + TARGET_NOISE_SD * rng.normal(size=N)
    xl[:, 0] = signal + TARGET_NOISE_SD * rng.normal(size=N)
    t['band_ratio'] = br.astype(np.float32)
    t['signed_xlag'] = xl.astype(np.float32)
    t['delay_iqr'] = np.exp(-3.5 + 0.3 * rng.normal(size=N)).astype(np.float32)

    # The mark channel and the two signal-carrying targets are written into the
    # per-event content only. Participation, coarse composition, load and
    # exposure are untouched, so P_stats cannot see u.
    tok = np.asarray(p['contact_tokens']).copy()
    part = np.asarray(p['participation'])
    C = tok.shape[1]
    tok[..., 3:8] = (br[:, None, :1] + .2 * rng.normal(size=(N, C, 5))).astype(np.float32)
    tok[..., 8:13] = (u[:, None, None] + .2 * rng.normal(size=(N, C, 5))).astype(np.float32)
    tok[..., 18:28] = (xl[:, None, :1] + .2 * rng.normal(size=(N, C, 10))).astype(np.float32)
    tok[..., 28:31] = (np.log(t['delay_iqr'])[:, None, None] + .2 * rng.normal(size=(N, C, 3))).astype(np.float32)
    tok[~part] = np.nan
    p['contact_tokens'] = tok
    ev = np.asarray(p['event_features']).copy()
    ev[:, 4] = t['delay_iqr']
    ev[:, 6] = u.astype(np.float32)
    ev[:, 7] = (.2 * rng.normal(size=N)).astype(np.float32)
    p['event_features'] = ev

    p['subject'] = f'synthetic_{task}_{realization}'
    p['synthetic_contract'] = dict(
        task=task, realization=int(realization), old_summary_coefficient=float(kappa),
        target_noise_sd=float(TARGET_NOISE_SD),
        readable_conditioning=['exponentially weighted published count rate, tau=30 minutes',
                               'intraday sin/cos at the event time'],
        readable_coefficients=dict(rate=READABLE_RATE_COEFFICIENT, clock=READABLE_CLOCK_COEFFICIENT),
        old_summary=dict(window_minutes_before_target=list(OLD_WINDOW_MINUTES), bin_minutes=BIN_MINUTES,
                         channel='per-event mark u written into contact token block 8:13 and event feature 6',
                         standardisation=dict(center=old_mu, scale=old_sd),
                         driver=('two-hour piecewise-constant Gaussian bins, independent across bins and '
                                 'projected off the readable bin-level basis [1, sin, cos, coarse rate]')),
        targets_carrying_signal=['band_ratio component 0', 'signed_xlag component 0'],
        untouched=['participation', 'shaft composition', 'total load', 'exposure', 'event count'],
        expectation=('conditional_zero expects a zero extra-mark increment for both the B and S arms while '
                     'coarse history and time memory remain predictive; organization_positive expects the '
                     'persistent arms to beat the two-hour arm because the summary window lies four to six '
                     'hours before the target'),
        caveats=['three realizations calibrate direction, not an error rate and not a small-effect exclusion',
                 'this single-validation-segment diagnostic does not certify the human paired-INNER procedure',
                 'independence given a hidden driver would not qualify; the target is generated directly from '
                 'the readable conditioning set'])
    p['old_summary_z'] = z_old
    p['old_summary_defined'] = defined
    p['old_summary_events'] = n_old_events
    p['readable_signal_z_rate'] = z_rate
    p['mark_channel_u'] = u
    p['release_contract'] = 'synthetic world uses the same closed one-hour block publication rule'
    return p


def old_summary_oracle(payload, split, role='outer'):
    """C40: a direct reference using the true legal old summary; no new fit."""
    contract = payload['synthetic_contract']
    pk = payload['packets']
    table = D.target_table(payload, split, role, 30)
    if not len(table):
        return dict(status='NOT_ESTIMABLE', reason='no registered targets in this role')
    ep = D.event_packets(payload)
    z_old = payload['old_summary_z']
    z_rate = payload['readable_signal_z_rate']
    phase = payload['event_time'] * (2 * np.pi / 86400.)
    readable = (READABLE_RATE_COEFFICIENT * z_rate[ep]
                + READABLE_CLOCK_COEFFICIENT * np.sin(phase) + READABLE_CLOCK_COEFFICIENT * np.cos(phase))
    kappa = contract['old_summary_coefficient']
    rows = []
    for h in (1, 5, 30, 120):
        tg = np.unique(table[table[:, 1] == h, 0])
        if not len(tg):
            continue
        sel = np.isin(ep, tg)
        if not sel.any():
            continue
        published = np.zeros(len(tg), bool)
        for j, k in enumerate(tg):
            q = int(k - h)
            published[j] = bool(pk['release'][max(q - int(OLD_WINDOW_MINUTES[0]), 0):max(q, 1)].max()
                                <= pk['end'][q]) if q > 0 else False
        y = np.concatenate([payload['targets']['band_ratio'][sel, 0],
                            payload['targets']['signed_xlag'][sel, 0]])
        base = np.concatenate([readable[sel], readable[sel]])
        full = base + kappa * np.concatenate([z_old[ep[sel]], z_old[ep[sel]]])
        def nll(mu):
            return float(np.mean(.5 * ((y - mu) / TARGET_NOISE_SD) ** 2 + np.log(TARGET_NOISE_SD)
                                 + .5 * np.log(2 * np.pi)))
        rows.append(dict(horizon=h, n_events=int(sel.sum()),
                         oracle_with_old_summary=nll(full), oracle_readable_only=nll(base),
                         gain_from_old_summary=nll(base) - nll(full),
                         old_summary_published_at_query_fraction=float(published.mean())))
    return dict(status='COMPLETE', rows=rows, coefficient=kappa,
                note=('analytic reference on the generating coefficients; it is an upper reference for what a '
                      'perfect reader of the legal old summary could gain, not a fitted model'))


def generate(root, task, realization, n_blocks=72):
    p = make_task_world(task, realization, n_blocks)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    witness = {k: p.pop(k) for k in ('slow_state', 'mark_channel_u', 'old_summary_z', 'old_summary_defined',
                                     'old_summary_events', 'readable_signal_z_rate')}
    atomic_torch(witness | dict(subject=p['subject'], contract=p['synthetic_contract']),
                 root / f'{p["subject"]}.witness.pt')
    atomic_torch(p, root / f'{p["subject"]}.pt')
    return dict(status='COMPLETE', subject=p['subject'], task=task, realization=int(realization),
                packet_file=str(root / f'{p["subject"]}.pt'), n_events=int(len(p['event_time'])),
                contract=p['synthetic_contract'])


def fit_and_score(cfg, task, realization, out_root):
    """One single-validation-segment fit per arm, scored on an untouched future."""
    from dataclasses import replace
    from .train import run_paired_inner, load_run, evaluate, source_digest, tag
    from .engine import build_model
    single = replace(cfg, stage='inner0')
    model, prep = load_run(single)
    opt_cfg = single
    # A single time-ordered validation segment selects the checkpoint; this is a
    # targeted capability diagnostic, not the human dual-INNER procedure.
    from .train import (training_ids, training_normalizers, optimizer_for, update_once, Plateau,
                        atomic_json as _aj)
    ids = training_ids(prep)
    normalizers = training_normalizers(prep, single.batch_size)
    opt = optimizer_for(model, single)
    rng = np.random.default_rng(single.sampler_seed)
    plateau = Plateau()
    best = dict(score=None, updates=0, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    ev0 = evaluate(model, prep, single, 'inner')
    if ev0['selection'] is None:
        raise ValueError('synthetic validation organization score is not estimable')
    best['score'] = ev0['selection']
    plateau.best = ev0['selection']
    curve = [dict(update=0, validation=ev0['selection'])]
    updates = 0
    budget = single.max_updates
    stop_reason = 'budget'
    while updates < budget:
        anchor = int(rng.integers(len(ids)))
        batch = ids[(anchor + np.arange(single.batch_size)) % len(ids)]
        update_once(model, prep, single, opt, batch, normalizers, updates, single.microbatch)
        updates += 1
        if updates % single.eval_every == 0:
            ev = evaluate(model, prep, single, 'inner')
            score = ev['selection']
            curve.append(dict(update=updates, validation=score))
            if score < best['score']:
                best = dict(score=score, updates=updates,
                            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            action = plateau.observe(score)
            if action == 'drop':
                for g in opt.param_groups:
                    g['lr'] *= .3
            if action == 'stop':
                stop_reason = 'plateau'
                break
            if updates >= budget and action == 'improve' and single.extended_updates > budget:
                budget = single.extended_updates
    model.load_state_dict(best['state'])
    # Untouched future test: the half of the record beyond the S-E cutoff.
    split = dict(prep.split)
    cut = split['support_start'] + .5 * (split['support_end'] - split['support_start'])
    pk = prep.payload['packets']
    split['outer_packet'] = (pk['start'] >= cut) & (pk['end'] <= split['support_end']) & split['valid_packet']
    split['split_id'] = D.digest((prep.split['split_id'], 'untouched_synthetic_future', split['outer_packet']))
    prep.split = split
    test = evaluate(model, prep, single, 'outer', collect=True)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    name = f'{task}_{realization}_{cfg.arm_name}'
    atomic_torch(dict(records=test['records'], summary=test['summary']), out / f'{name}.pt')
    result = dict(status=test['status'], task=task, realization=int(realization), arm=cfg.arm_name,
                  subject=cfg.subject, selected_updates=best['updates'], executed_updates=updates,
                  stop_reason=stop_reason, validation_curve=curve,
                  validation_selection=best['score'], test_selection=test['selection'],
                  summary=test['summary'],
                  legacy_packet_joint_component_score=test['legacy_packet_joint_component_score'],
                  contract=prep.payload['synthetic_contract'], source_digest=source_digest()[0],
                  oracle=old_summary_oracle(prep.payload, split, 'outer'),
                  scope=('single-validation-segment capability diagnostic; it calibrates the state rich '
                         'increment and the earlier-history contrast, not the B references and not the '
                         'human dual-INNER procedure'))
    atomic_json(result, out / f'{name}.json')
    return result
