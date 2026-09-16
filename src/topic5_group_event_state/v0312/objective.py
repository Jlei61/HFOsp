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


def interval_log_mean_rate(readout,grid,prep,pidx):
    """Trapezoid path integral on the common 5-second scoring grid."""
    G,S,B,_=grid.shape
    fractions=torch.linspace(0,1,G,device=grid.device,dtype=torch.float64)
    start=torch.as_tensor(prep.packet_start[pidx.detach().cpu().numpy()],device=grid.device)
    times=start[None,:]+fractions[:,None]*60.
    phase=times*(2*math.pi/86400.)
    clock=torch.stack((phase.sin(),phase.cos()),-1).to(grid.dtype).unsqueeze(1).expand(G,S,B,2)
    log_rate=readout(grid,clock)['log_rate'].clamp(-20,15)
    if G==13:
        seconds=prep.exposure_grid[pidx]
        weights=grid.new_zeros(G,B)
        weights[:-1]+=seconds.T/2;weights[1:]+=seconds.T/2
        den=seconds.sum(-1).clamp(min=1e-12)
        return torch.logsumexp(log_rate+weights.clamp(min=1e-30).log()[:,None,:],0)-den.log()[None,:]
    if G!=2:raise ValueError('unregistered interval grid')
    return torch.logsumexp(log_rate,0)-math.log(2.)


def view_log_probs(readout,z_start,z_end,prep,pidx,event_index=None,state_grid=None,diagnostics=False):
    """Mixed per-view log-probability and effective unit counts for packets ``pidx``."""
    S,B,_=z_end.shape
    clock=prep.clock[pidx].unsqueeze(0).expand(S,B,2)
    out_s=readout(z_start,clock);out_e=readout(z_end,clock)
    count=prep.count[pidx];expo=prep.exposure_hours[pidx];valid=prep.valid[pidx]
    has=(count>0).to(count.dtype)*valid
    if state_grid is None:state_grid=torch.stack((z_start,z_end))
    lr=interval_log_mean_rate(readout,state_grid,prep,pidx)
    lp_count=count_log_prob(lr,lr,expo.unsqueeze(0),
                            count.unsqueeze(0),readout.log_nb_dispersion)
    lp_spatial=composition_log_prob(out_e['composition'],prep.shaft_count[pidx].unsqueeze(0))
    load=prep.total_load[pidx].unsqueeze(0)
    ly=(torch.log(load.clamp(min=1e-9))-prep.log_load_center)/prep.log_load_scale
    lp_load=load_log_prob(ly,out_e['load'],torch.ones_like(lp_spatial))
    if event_index is None:idx,row=_event_rows(prep,pidx)
    else:idx,row=event_index
    components={}
    if idx is None or len(idx)==0:
        lp_morph=torch.zeros(S,B,device=prep.device,dtype=z_end.dtype)
        units=torch.zeros(B,device=prep.device,dtype=z_end.dtype)
    else:
        br=prep.band_ratio[idx].unsqueeze(0);brv=prep.band_ratio_valid[idx].unsqueeze(0)
        xl=prep.xlag[idx].unsqueeze(0);xlv=prep.xlag_valid[idx].unsqueeze(0)
        iq=prep.iqr[idx].unsqueeze(0);iqv=prep.iqr_valid[idx].unsqueeze(0)
        # Event-time readout from the forecast path, before any target assimilation.
        # A 5-second left grid gives an explicit, refinable time approximation.
        frac=((prep.event_times[idx]-torch.as_tensor(prep.packet_start[pidx.detach().cpu().numpy()],device=prep.device)[row])/60.).clamp(0,1)
        gi=(frac*(len(state_grid)-1)).floor().long().clamp(max=len(state_grid)-1)
        z_event=state_grid[gi,:,row].transpose(0,1)
        phase=prep.event_times[idx]*(2*math.pi/86400.)
        ce=torch.stack((phase.sin(),phase.cos()),-1).to(z_end.dtype).unsqueeze(0).expand(S,len(idx),2)
        event_out=readout(z_event,ce)
        mu=event_out['band_ratio_mu'];ls=event_out['band_ratio_logsd']
        a,_=normal_log_prob(br,mu,ls,brv)
        mu=event_out['xlag_mu'];ls=event_out['xlag_logsd']
        b,_=normal_log_prob(xl,mu,ls,xlv)
        is_zero=(iq<=0).to(iq.dtype)
        w=(torch.log(iq.clamp(min=1e-9))-prep.log_iqr_center)/prep.log_iqr_scale
        c=zero_inflated_normal_log_prob(w,is_zero,event_out['iqr'],iqv)
        if diagnostics:
            for key,lp0,uu in (('band_ratio',a,brv.sum(-1)),('signed_xlag',b,xlv.sum(-1)),('delay_iqr',c,iqv)):
                components[key]=(torch.zeros(S,B,device=prep.device,dtype=z_end.dtype).index_add(1,row,lp0),
                                 torch.zeros(B,device=prep.device,dtype=z_end.dtype).index_add(0,row,uu[0]))
        per=a+b+c
        lp_morph=torch.zeros(S,B,device=prep.device,dtype=z_end.dtype).index_add(1,row,per)
        n=(brv+0.).sum(-1)+(xlv+0.).sum(-1)+iqv
        units=torch.zeros(B,device=prep.device,dtype=z_end.dtype).index_add(0,row,n[0])
    def mix(x):return torch.logsumexp(x,dim=0)-math.log(S)
    result=dict(count=mix(lp_count),spatial=mix(lp_spatial),morphology=mix(lp_morph),
                load=mix(lp_load)),dict(count=valid,spatial=has,morphology=units,load=has)
    if diagnostics:
        for k,(lp0,u) in components.items():result[0][k]=mix(lp0);result[1][k]=u
    return result


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
