"""Per-view predictive scoring on a common support.

Each view mixes over sampled state paths with ``log(mean_s p_s)``; the count,
coarse-spatial, conditional-morphology and secondary-load blocks are reported
separately so a single total can never fill in a missing endpoint.
"""
from __future__ import annotations
import math
import torch

from .model import (count_log_prob,composition_log_prob,normal_log_prob,
                    zero_inflated_normal_log_prob,load_log_prob)

VIEWS=('count','spatial','morphology')
SECONDARY=('load',)
LOAD_WEIGHT=0.25


def _event_rows(prep,pidx):
    lo=prep.event_lo[pidx];hi=prep.event_hi[pidx];counts=hi-lo
    total=int(counts.sum())
    if total==0:return None,None
    offs=torch.cumsum(counts,0)-counts
    pos=torch.arange(total,device=prep.device)
    row=torch.searchsorted(offs+counts,pos,right=True)
    return lo[row]+(pos-offs[row]),row


def view_log_probs(readout,z_start,z_end,prep,pidx,event_index=None):
    """Mixed per-view log-probability and effective unit counts for packets ``pidx``."""
    S,B,_=z_end.shape
    clock=prep.clock[pidx].unsqueeze(0).expand(S,B,2)
    out_s=readout(z_start,clock);out_e=readout(z_end,clock)
    count=prep.count[pidx];expo=prep.exposure_hours[pidx];valid=prep.valid[pidx]
    has=(count>0).to(count.dtype)*valid
    lp_count=count_log_prob(out_s['log_rate'],out_e['log_rate'],expo.unsqueeze(0),
                            count.unsqueeze(0),readout.log_nb_dispersion)
    lp_spatial=composition_log_prob(out_e['composition'],prep.shaft_count[pidx].unsqueeze(0))
    load=prep.total_load[pidx].unsqueeze(0)
    ly=(torch.log(load.clamp(min=1e-9))-prep.log_load_center)/prep.log_load_scale
    lp_load=load_log_prob(ly,out_e['load'],torch.ones_like(lp_spatial))
    if event_index is None:idx,row=_event_rows(prep,pidx)
    else:idx,row=event_index
    if idx is None or len(idx)==0:
        lp_morph=torch.zeros(S,B,device=prep.device,dtype=z_end.dtype)
        units=torch.zeros(B,device=prep.device,dtype=z_end.dtype)
    else:
        br=prep.band_ratio[idx].unsqueeze(0);brv=prep.band_ratio_valid[idx].unsqueeze(0)
        xl=prep.xlag[idx].unsqueeze(0);xlv=prep.xlag_valid[idx].unsqueeze(0)
        iq=prep.iqr[idx].unsqueeze(0);iqv=prep.iqr_valid[idx].unsqueeze(0)
        mu=out_e['band_ratio_mu'][:,row];ls=out_e['band_ratio_logsd'][:,row]
        a,_=normal_log_prob(br,mu,ls,brv)
        mu=out_e['xlag_mu'][:,row];ls=out_e['xlag_logsd'][:,row]
        b,_=normal_log_prob(xl,mu,ls,xlv)
        is_zero=(iq<=0).to(iq.dtype)
        w=(torch.log(iq.clamp(min=1e-9))-prep.log_iqr_center)/prep.log_iqr_scale
        c=zero_inflated_normal_log_prob(w,is_zero,out_e['iqr'][:,row],iqv)
        per=a+b+c
        lp_morph=torch.zeros(S,B,device=prep.device,dtype=z_end.dtype).index_add(1,row,per)
        n=(brv+0.).sum(-1)+(xlv+0.).sum(-1)+iqv
        units=torch.zeros(B,device=prep.device,dtype=z_end.dtype).index_add(0,row,n[0])
    def mix(x):return torch.logsumexp(x,dim=0)-math.log(S)
    return dict(count=mix(lp_count),spatial=mix(lp_spatial),morphology=mix(lp_morph),
                load=mix(lp_load)),dict(count=valid,spatial=has,morphology=units,load=has)


def combine(logp,units,weights=None):
    """Equal-weight main views, secondary load down-weighted; masked denominators."""
    terms={}
    for k in VIEWS+SECONDARY:
        u=units[k]
        tot=u.sum()
        terms[k]=-(logp[k]*(u>0).to(u.dtype)).sum()/tot.clamp(min=1e-6) if float(tot)>0 else None
    main=[terms[k] for k in VIEWS if terms[k] is not None]
    if not main:return None,terms
    loss=sum(main)/len(main)
    if terms['load'] is not None:loss=loss+LOAD_WEIGHT*terms['load']
    return loss,terms
