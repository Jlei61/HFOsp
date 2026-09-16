"""Frozen masks, nested temporal validation, and FIT-only measurement coordinates."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from ..v0311 import data as legacy

PACKET_SECONDS = 60.
HORIZON_SECONDS = (60., 300., 1800., 7200.)
SPLIT_SEED = 20260906
clock_features = legacy.clock_features
packet_tables = legacy.packet_tables
encode_events = legacy.encode_events
seizure_intervals = legacy.seizure_intervals


def interval_mask(start, end, intervals):
    out = np.zeros(len(start), bool)
    for a, b in np.asarray(intervals, float).reshape(-1, 2):
        out |= (np.asarray(end) > a) & (np.asarray(start) < b)
    return out


def clock_stratum(t):
    return np.array([datetime.fromtimestamp(float(v), ZoneInfo('Europe/Berlin')).hour // 6
                     for v in np.asarray(t).reshape(-1)], int)


def calendar_day(t):
    return np.array([datetime.fromtimestamp(float(v), ZoneInfo('Europe/Berlin')).date().toordinal()
                     for v in np.asarray(t).reshape(-1)], int)


def digest(value):
    def encode(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=encode,
                                    separators=(',', ':')).encode()).hexdigest()


def build_split(payload, subject, split_seed=SPLIT_SEED, stage='inner0', protocol='S-E'):
    """Two nested temporal INNER fits precede a fixed-recipe outer refit.

    The detector/contact-definition prefix is never an S-ID held-out core.
    Optimizer seeds are deliberately absent from this API and its manifest.
    """
    pk = payload['packets']; pe = np.asarray(pk['end'])
    start = float(payload['observed_support'][0, 0])
    end = float(payload['phase_boundaries']['80pct'])
    cutoff = start + .5 * (end-start)
    excluded, seizures = (np.empty((0,2)),[]) if payload.get('synthetic_contract') else seizure_intervals(subject)
    bad = interval_mask(pk['start'], pe, excluded)
    valid = (pk['exposure'] > 1e-6) & ~bad
    measured_until = float(payload['phase_boundaries'].get('20pct', start))
    train = np.zeros(len(pe), bool); inner = train.copy(); outer = train.copy()
    cores = []; inner_cores = []
    if protocol == 'S-E':
        if stage not in ('inner0', 'inner1', 'outer'): raise ValueError(stage)
        fractions = {'inner0': (.6, .8), 'inner1': (.8, 1.), 'outer': (1., 2.)}
        a,b = fractions[stage]
        train_end = start + a*(cutoff-start)
        score_end = start + b*(cutoff-start)
        if train_end < measured_until:
            raise ValueError('nested FIT ends before the frozen measurement-definition prefix')
        train = (pe <= train_end) & valid
        selected = (pk['start'] >= train_end) & (pe <= score_end) & valid
        if stage == 'outer': outer = selected
        else: inner = selected
    elif protocol == 'S-ID':
        if stage != 'sid': raise ValueError('S-ID requires stage=sid')
        rng = np.random.default_rng(split_seed)
        grid = np.arange(np.ceil(max(start, measured_until)/1800)*1800, end-1800, 1800)
        candidates = []
        for g in grid:
            mask = (pk['start'] >= g) & (pe <= g+1800)
            if pk['exposure'][mask].sum() >= .9*1800 and not bad[mask].any(): candidates.append(g)
        candidates = np.asarray(candidates)
        days = calendar_day(candidates); clocks = clock_stratum(candidates)
        picked=[]
        for key in sorted(set(zip(days, clocks))):
            eligible = candidates[(days == key[0]) & (clocks == key[1])]
            picked.extend(rng.choice(eligible, max(1, round(.25*len(eligible))), replace=False))
        for g in sorted(picked):
            if not cores or g-cores[-1][0] >= 7200: cores.append((g,g+1800))
        outer = interval_mask(pk['start'],pe,cores) & valid
        rest = [g for g in candidates if not interval_mask(np.array([g]),np.array([g+1800]),cores)[0]]
        for g in rest[::8]: inner_cores.append((g,g+1800))
        inner = interval_mask(pk['start'],pe,inner_cores) & valid & ~outer
        train = valid & ~outer & ~inner & (pe <= end)
        train_end = end
    else: raise ValueError(protocol)
    # A physical grid gap also begins a new episode. Missing observations alone do not.
    reset = bad.copy()
    reset_start = np.zeros(len(pe), np.int64)
    last = 0
    for i in range(len(pe)):
        if i and abs(pk['start'][i]-pe[i-1]) > 1e-6: last=i
        if reset[i]: last=i+1
        reset_start[i]=last
    result = dict(subject=subject, protocol=protocol, stage=stage, seed=int(split_seed),
                  split_seed=int(split_seed), support_start=start,support_end=end,
                  se_cutoff=cutoff if protocol=='S-E' else None, fit_end=train_end,
                  measurement_definition_end=measured_until, train_packet=train,
                  inner_packet=inner, outer_packet=outer, valid_packet=valid,
                  seizure_mask=bad, episode_start=reset_start,
                  excluded_intervals=excluded,seizures=seizures,
                  core_intervals=np.asarray(cores).reshape(-1,2),
                  inner_core_intervals=np.asarray(inner_cores).reshape(-1,2),
                  contract='v0312: fixed split RNG; FIT-only transforms; explicit physical target IDs')
    result['split_id']=digest({k:v for k,v in result.items() if k!='seizures'})
    return result


def role_mask(split, role):
    if role=='fit': return split['train_packet']
    if role=='inner': return split['inner_packet']
    if role=='outer': return split['outer_packet']
    raise ValueError(role)


def input_mask(split, role):
    if role=='fit': return split['train_packet']
    if role=='inner' and split['protocol']=='S-ID':
        return split['valid_packet'] & ~split['outer_packet']
    if role in ('inner','outer','descriptive'): return split['valid_packet']
    raise ValueError(role)


def event_packets(payload):
    pk=payload['packets']
    out=np.full(len(payload['event_time']),-1,np.int64)
    for i,(a,b) in enumerate(zip(pk['event_lo'],pk['event_hi'])): out[a:b]=i
    if (out<0).any(): raise ValueError('events not assigned to a physical packet')
    return out


def nb_nll(par, y, exposure_hours):
    mu=np.exp(par[0])*exposure_hours; r=np.exp(par[1])
    return -(gammaln(y+r)-gammaln(r)-gammaln(y+1)+r*(np.log(r)-np.log(r+mu))
             +y*(np.log(np.maximum(mu,1e-100))-np.log(r+mu)))


def fit_scaling(payload,split,packet_x,packet_t):
    mask=split['train_packet']
    ep=event_packets(payload); ev_mask=mask[ep]
    if not mask.any() or not ev_mask.any(): raise ValueError('no FIT observations for transforms')
    # Reuse audited coordinate formulas, explicitly hiding forbidden data from each fit.
    pp=dict(payload); pp['event_time']=np.where(ev_mask,payload['event_time'],np.inf)
    pp['event_features']=payload['event_features'].copy()
    legal=np.flatnonzero(ev_mask)
    gap=np.full(len(legal),np.nan)
    if len(legal)>1:
        same=split['episode_start'][ep[legal[1:]]]==split['episode_start'][ep[legal[:-1]]]
        gap[1:]=np.where(same,np.log1p(np.diff(payload['event_time'][legal])),np.nan)
    pp['event_features'][legal,-1]=gap
    pt=dict(packet_t);pt['valid']=mask
    sc=legacy.fit_scaling(pp,split,packet_x,pt)
    y=packet_t['count'][mask]; expo=packet_t['exposure'][mask]/3600.
    rate=max(y.sum()/expo.sum(),1e-8)
    fits=[minimize(lambda p:nb_nll(p,y,expo).mean(),[np.log(rate),a],
                   method='L-BFGS-B',bounds=[(-20.,20.),(-12.,15.)]) for a in (-3.,0.,3.)]
    best=min(fits,key=lambda f:f.fun)
    if not best.success or not np.isfinite(best.fun): raise FloatingPointError('FIT NB MLE failed')
    sc.update(base_rate_per_hour=float(np.exp(best.x[0])),nb_size=float(np.exp(best.x[1])),
              nb_fit_nll=float(best.fun),fit_hours=float(expo.sum()),
              fit_packet_count=int(mask.sum()),fit_event_count=int(ev_mask.sum()),
              fit_mask_digest=digest(mask),note='v0312: full permission mask, not a calendar cutoff alone')
    sc['transform_id']=digest(sc)
    return sc


def target_table(payload,split,role,stride=30,horizons=(1,5,30,120)):
    """Fixed physical targets, with origins derived separately for each horizon."""
    pk=payload['packets']; allowed=role_mask(split,role)
    ids=np.flatnonzero(allowed & (np.arange(len(allowed))%stride==0))
    rows=[]
    for i in ids:
        for h in horizons:
            q=i-int(h)
            if q<0 or split['seizure_mask'][q]: continue
            # Conditional interictal prediction: no future seizure/reset in the rollout.
            if split['seizure_mask'][q+1:i+1].any(): continue
            if abs((pk['end'][i]-pk['end'][q])-h*60)>1e-6: continue
            rows.append((int(i),int(h),int(q)))
    return np.asarray(rows,np.int64).reshape(-1,3)
