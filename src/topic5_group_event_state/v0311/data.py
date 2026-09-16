"""Split contract, FIT-only scaling and episode assembly for v0.3.11.

Every scaler, vocabulary and target transform is estimated on FIT support only.
Query points carry their real information lag: minute packets publish with the
closed block that produced them, so a query on a block boundary sees a
zero-lag history and a query 30 minutes later does not.
"""
from __future__ import annotations
import json,math
from dataclasses import dataclass,field
import numpy as np
import torch

from ..v039.human_data import exposure,merge_intervals,subtract_intervals

PACKET_SECONDS=60.0
HORIZON_SECONDS=(60.,300.,1800.,7200.)
START_GRID_SECONDS=1800.
INNER_BLOCK_SECONDS=1800.
INNER_FRACTION=0.25
INNER_PREFIX_MARGIN_SECONDS=3*3600.
CLOCK_STRATA=4


def clock_features(t):
    a=2*math.pi*np.asarray(t,float)/86400.
    return np.stack((np.sin(a),np.cos(a)),axis=-1)


def clock_stratum(t,tz_offset=7200.):
    return (((np.asarray(t,float)+tz_offset)%86400.)//(86400./CLOCK_STRATA)).astype(int)


def calendar_day(t,tz_offset=7200.):
    return np.floor((np.asarray(t,float)+tz_offset)/86400.).astype(int)


def _robust(values,mask=None):
    v=np.asarray(values,float)
    if mask is not None:v=v[mask]
    v=v[np.isfinite(v)]
    if not len(v):return 0.,1.
    c=float(np.median(v));s=float(np.subtract(*np.percentile(v,[75,25])))
    if not np.isfinite(s) or s<=1e-9:
        s=float(np.std(v))
        if not np.isfinite(s) or s<=1e-9:s=1.
    return c,s


@dataclass
class Bundle:
    payload:dict
    split:dict
    scaling:dict
    packet_inputs:np.ndarray=field(default=None)
    packet_targets:dict=field(default=None)


def seizure_intervals(subject,dataset_root='/data/hfosp_group_event_state_v0_1/dataset',
                      post_ictal_seconds=3600.,pre_seconds=0.):
    index=json.loads(open(f'{dataset_root}/{subject}/index.json').read())
    rows=[[r['onset_epoch']-pre_seconds,r['offset_epoch']+post_ictal_seconds] for r in index['seizures']]
    return merge_intervals(rows) if rows else np.empty((0,2)),index['seizures']


def build_split(payload,subject,seed=20260906):
    """S-E forward cutoff at 50% of the authorised calendar span; INNER scattered."""
    support=payload['observed_support']
    packets=payload['packets']
    start=float(support[0,0]);end=float(payload['phase_boundaries']['80pct'])
    cutoff=start+0.5*(end-start)
    excluded,seizures=seizure_intervals(subject)
    blocks=payload['blocks']
    releases=np.array([b['release'] for b in blocks])
    gap=float(releases[releases>cutoff].min()) if (releases>cutoff).any() else cutoff
    fit_end=start+0.6*(cutoff-start)
    rng=np.random.default_rng(seed)
    pe=packets['end']
    train=pe<=cutoff
    grid=np.arange(math.ceil(start/START_GRID_SECONDS)*START_GRID_SECONDS,end,START_GRID_SECONDS)
    margin=4*3600.
    pool=grid[(grid>=start+margin)&(grid+max(HORIZON_SECONDS)<=cutoff)]
    pool=_drop_excluded(pool,excluded,before=0.)
    inner=_scatter_blocks(pool,rng)
    forward=grid[(grid>=gap)&(grid+max(HORIZON_SECONDS)<=end)]
    return dict(support_start=start,support_end=end,se_cutoff=cutoff,se_gap_start=gap,
                fit_end=fit_end,train_packet=train,inner_starts=inner,forward_starts=forward,
                inner_target_times=np.concatenate([inner+h for h in HORIZON_SECONDS]) if len(inner) else np.empty(0),
                excluded_intervals=excluded,seizures=seizures,seed=int(seed),
                contract=('S-E: parameters and measurement choices frozen before se_cutoff; '
                          'canonical forward segment runs from the first released block after the cutoff '
                          'to the authorised end'))


def _drop_excluded(candidates,excluded,before=INNER_PREFIX_MARGIN_SECONDS,after=max(HORIZON_SECONDS)):
    """A selection query needs a clean prefix and clean scored targets."""
    if not len(candidates) or not len(excluded):return candidates
    keep=np.ones(len(candidates),bool)
    for a,b in np.asarray(excluded,float).reshape(-1,2):
        keep&=~((candidates+after>a)&(candidates-before<b))
    return candidates[keep]


def _scatter_blocks(candidates,rng,fraction=INNER_FRACTION):
    """Pick ~20% of 30-minute blocks spread over calendar day x 6h clock stratum."""
    if not len(candidates):return np.empty(0)
    day=calendar_day(candidates);stratum=clock_stratum(candidates)
    keys=sorted(set(zip(day.tolist(),stratum.tolist())))
    picked=[]
    for k in keys:
        sel=candidates[(day==k[0])&(stratum==k[1])]
        n=max(1,int(round(fraction*len(sel))))
        picked.extend(rng.choice(sel,size=min(n,len(sel)),replace=False).tolist())
    return np.sort(np.asarray(picked))


def build_split_id(payload,subject,seed=20260906):
    """S-ID: scattered 30-minute cores held out on quality and clock only."""
    support=payload['observed_support'];packets=payload['packets']
    start=float(support[0,0]);end=float(payload['phase_boundaries']['80pct'])
    excluded,seizures=seizure_intervals(subject)
    rng=np.random.default_rng(seed)
    grid=np.arange(math.ceil(start/INNER_BLOCK_SECONDS)*INNER_BLOCK_SECONDS,end-INNER_BLOCK_SECONDS,INNER_BLOCK_SECONDS)
    ok=[]
    for g in grid:
        if exposure(support,g,g+INNER_BLOCK_SECONDS)<0.9*INNER_BLOCK_SECONDS:continue
        if len(excluded) and np.any((excluded[:,0]<g+INNER_BLOCK_SECONDS)&(excluded[:,1]>g)):continue
        ok.append(g)
    ok=np.asarray(ok,float)
    core=_scatter_blocks(ok,rng)
    keep=[]
    for c in core:
        if not keep or c-keep[-1]>=7200.:keep.append(float(c))
    core=np.asarray(keep)
    core_iv=np.stack((core,core+INNER_BLOCK_SECONDS),axis=-1) if len(core) else np.empty((0,2))
    # Only the held-out cores are removed from the readable inputs. Extending the
    # mask back by the longest horizon merged the windows into one blob and left a
    # third of the record trainable, which is a masking bug, not an S-ID property.
    # Targets that fall inside a core are excluded through target_region instead.
    masked=merge_intervals([[a,b] for a,b in core_iv]) if len(core_iv) else np.empty((0,2))
    pe=packets['end']
    train=np.ones(len(pe),bool)
    for a,b in masked:train&=~((pe>a)&(packets['start']<b))
    rest=np.asarray([g for g in ok if not any(a<=g<b for a,b in core_iv)],float)
    inner=_scatter_blocks(_drop_excluded(rest,excluded,before=0.),np.random.default_rng(seed+1))
    fit_end=start+0.6*(end-start)
    return dict(support_start=start,support_end=end,se_cutoff=None,se_gap_start=None,fit_end=fit_end,
                train_packet=train,inner_starts=inner,forward_starts=core,core_intervals=core_iv,
                target_region=core_iv,
                inner_target_times=np.concatenate([inner+h for h in HORIZON_SECONDS]) if len(inner) else np.empty(0),
                masked_intervals=masked,excluded_intervals=excluded,seizures=seizures,seed=int(seed),
                contract=('S-ID: scattered 30-minute cores chosen on time and exposure only; inputs and '
                          'targets masked back by the longest horizon; later support may fit parameters, '
                          'so this is retrospective identification'))


def packet_tables(payload,split):
    """Per-packet inputs (P_stats layout) and frozen targets."""
    pk=payload['packets'];n=len(pk['start'])
    lo=pk['event_lo'];hi=pk['event_hi']
    part=payload['participation'];tgt=payload['targets']
    shaft_count=tgt['shaft_count'];n_shaft=shaft_count.shape[1]
    count=(hi-lo).astype(np.float64)
    idx=np.arange(len(payload['event_time']))
    csum=np.concatenate((np.zeros((1,n_shaft)),np.cumsum(shaft_count,axis=0)))
    shaft=csum[hi]-csum[lo]
    load=shaft.sum(axis=1)
    expo=pk['exposure']
    valid=expo>1e-6
    frac=np.divide(shaft,np.maximum(load,1)[:,None],out=np.zeros_like(shaft),where=load[:,None]>0)
    mean_size=np.divide(load,np.maximum(count,1),out=np.zeros_like(load),where=count>0)
    clock=clock_features(0.5*(pk['start']+pk['end']))
    x=np.concatenate((np.log1p(count)[:,None],np.log1p(load)[:,None],frac,
                      np.log1p(mean_size)[:,None],(expo/PACKET_SECONDS)[:,None],clock,
                      valid.astype(float)[:,None],(count>0).astype(float)[:,None]),axis=-1)
    targets=dict(count=count,total_load=load,shaft_count=shaft,exposure=expo,valid=valid,
                 event_lo=lo,event_hi=hi)
    return x.astype(np.float32),targets,n_shaft


def fit_scaling(payload,split,packet_x,packet_t):
    """All scalers and target transforms from FIT support only."""
    pk=payload['packets']
    fit=(pk['end']<=split['fit_end'])&packet_t['valid']
    ev_t=payload['event_time']
    fit_ev=ev_t<=split['fit_end']
    tok=payload['contact_tokens'];evf=payload['event_features']
    part=payload['participation']
    tok_c=[];tok_s=[]
    for k in range(tok.shape[-1]):
        c,s=_robust(tok[fit_ev,:,k][part[fit_ev]])
        tok_c.append(c);tok_s.append(s)
    ev_c=[];ev_s=[]
    for k in range(evf.shape[-1]):
        c,s=_robust(evf[fit_ev,k])
        ev_c.append(c);ev_s.append(s)
    px_c=[];px_s=[]
    for k in range(packet_x.shape[-1]):
        c,s=_robust(packet_x[fit,k])
        px_c.append(c);px_s.append(s)
    t=payload['targets']
    br_c=[];br_s=[]
    for k in range(t['band_ratio'].shape[-1]):
        c,s=_robust(t['band_ratio'][fit_ev,k]);br_c.append(c);br_s.append(s)
    xl_c=[];xl_s=[]
    for k in range(t['signed_xlag'].shape[-1]):
        c,s=_robust(t['signed_xlag'][fit_ev,k]);xl_c.append(c);xl_s.append(s)
    iqr=t['delay_iqr'][fit_ev]
    pos=iqr[np.isfinite(iqr)&(iqr>0)]
    iqr_c,iqr_s=_robust(np.log(pos)) if len(pos)>8 else (0.,1.)
    load=packet_t['total_load'][fit&(packet_t['count']>0)]
    ld_c,ld_s=_robust(np.log(np.maximum(load,1e-6)))
    hours=exposure(payload['observed_support'],split['support_start'],split['fit_end'])/3600.
    base_rate=float(packet_t['count'][fit].sum()/max(hours,1e-6))
    return dict(contact_center=np.array(tok_c,np.float32),contact_scale=np.array(tok_s,np.float32),
                event_center=np.array(ev_c,np.float32),event_scale=np.array(ev_s,np.float32),
                packet_center=np.array(px_c,np.float32),packet_scale=np.array(px_s,np.float32),
                band_ratio_center=np.array(br_c,np.float32),band_ratio_scale=np.array(br_s,np.float32),
                xlag_center=np.array(xl_c,np.float32),xlag_scale=np.array(xl_s,np.float32),
                log_iqr_center=float(iqr_c),log_iqr_scale=float(iqr_s),
                log_load_center=float(ld_c),log_load_scale=float(ld_s),
                base_rate_per_hour=base_rate,fit_hours=float(hours),
                note='FIT-only medians and IQRs; degenerate IQR falls back to FIT std then to unit scale')


TOKEN_VALID_GROUPS=((0,3),(3,18),(18,28),(28,31))


def encode_events(payload,scaling):
    """Contact tokens, validity groups and event descriptors in frozen coordinates."""
    tok=payload['contact_tokens'];part=payload['participation']
    c=scaling['contact_center'];s=scaling['contact_scale']
    z=np.clip((tok-c)/s,-8,8)
    finite=np.isfinite(tok)
    z=np.where(finite,z,0.).astype(np.float32)
    groups=np.stack([finite[...,a:b].all(axis=-1) for a,b in TOKEN_VALID_GROUPS],axis=-1).astype(np.float32)
    z=z*part[...,None]
    groups=groups*part[...,None]
    evf=payload['event_features']
    ez=np.clip((evf-scaling['event_center'])/scaling['event_scale'],-8,8)
    ev_valid=np.isfinite(evf).astype(np.float32)
    ez=np.where(np.isfinite(evf),ez,0.).astype(np.float32)
    return z,groups,ez,ev_valid.astype(np.float32)
