"""Episode filtering, autonomous rollout and the v0.3.11 training loop.

Loss = 0.5 x per-packet prior prediction + 0.5 x multi-horizon autonomous
prediction. One gradient-carrying filter prefix serves many start points; the
warm-up prefix runs without gradient so the real back-propagation range is the
declared one.
"""
from __future__ import annotations
import copy,json,math,time
from dataclasses import dataclass,field,asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn

from .numerics import LATENT,propagate_moments,propagate_samples,evidence_update,cholesky_psd
from .model import StateModel,Readout,EventEncoder,SlowState,orthogonal_gru_,RICH_DIM,SLOW_HIDDEN
from .objective import view_log_probs,combine,VIEWS,SECONDARY
from .prepare import Prepared,HISTORY_TAU_HOURS
from ..v0310.trainer import PlateauSchedule,build_param_groups,per_parameter_hash,update_ratio,tensor_hash

PACKET_HOURS=1./60.
HORIZON_PACKETS=(1,5,30,120)
REFERENCE_ARMS=('intercept','clock','recent_rate','marked_history','constant_state')


@dataclass
class RunConfig:
    subject:str
    inputs:str='P_marks'
    family:str='I-L-G1'
    split:str='S-E'
    arm:str='state'
    seed:int=20260906
    warm_packets:int=60
    grad_packets:int=120
    n_start:int=16
    stride:int=7
    q_first:int=0  # 0 keeps the default warm+stride-1 placement
    batch_episodes:int=8
    train_paths:int=4
    eval_paths:int=64
    lr:float=1e-3
    max_updates:int=800
    extended_updates:int=6400
    eval_every:int=50
    target_ablation:bool=False
    crossview_ablation:bool=False
    shuffle_marks:bool=False
    packets_root:str='/data/hfosp_group_event_state_rich_event_identification_v0311/packets'
    out_dir:str='.'
    device:str='cuda:0'

    @property
    def coupled(self):return self.family.startswith('C')
    @property
    def nonlinear(self):return self.family.split('-')[1]=='N'
    @property
    def linear_readout(self):return self.family.endswith('G0')
    @property
    def rich(self):return self.inputs=='P_marks'


class HistoryReference(nn.Module):
    """Fixed-kernel causal history at 1/5/30 min and 2/8 h, then the shared readout."""

    def __init__(self,input_dim,n_shaft,n_ratio,n_xlag,mode,rich,n_contacts=None,
                 encoder_kwargs=None,hidden=64):
        super().__init__()
        self.mode=mode;self.rich=rich and mode=='marked_history'
        self.encoder=EventEncoder(**encoder_kwargs) if self.rich else None
        self.taus=torch.tensor(HISTORY_TAU_HOURS)
        base=input_dim+(RICH_DIM if self.rich else 0)
        if mode=='recent_rate':base=1
        self.base=base
        n_hist=0 if mode in ('intercept','clock','constant_state') else base*len(HISTORY_TAU_HOURS)
        extra=0 if mode=='intercept' else 4
        self.constant=nn.Parameter(torch.zeros(LATENT)) if mode=='constant_state' else None
        self.net=None if mode=='constant_state' else nn.Sequential(
            nn.Linear(max(n_hist+extra,1),hidden),nn.GELU(),nn.Linear(hidden,LATENT))
        self.readout=Readout(n_shaft,n_ratio,n_xlag)

    def latent(self,hist,clock,lead_hours,exposure,batch):
        if self.constant is not None:return self.constant.expand(batch,LATENT)
        pieces=[]
        if hist is not None:pieces.append(hist)
        if self.mode!='intercept':
            pieces.append(torch.cat((clock,torch.log1p(lead_hours).unsqueeze(-1),
                                     exposure.unsqueeze(-1)),dim=-1))
        if not pieces:pieces=[torch.ones(batch,1,device=clock.device,dtype=clock.dtype)]
        return self.net(torch.cat(pieces,dim=-1))


def _sample_paths(m,P,n_paths,generator):
    L,_=cholesky_psd(P.double(),'prior covariance')
    L=L.to(m.dtype)
    e=torch.randn(n_paths,*m.shape,device=m.device,dtype=m.dtype,generator=generator)
    return m.unsqueeze(0)+torch.einsum('nij,snj->sni',L,e)


def episode_plan(prep,split,cfg):
    """Legal episodes and, inside each, release-aware query points.

    A query at time tau may only use packets already published. Marks publish
    with the closed one-hour block, so the state used at tau is the one after
    the last packet of the last closed block, propagated forward by the real
    information lag.
    """
    L=cfg.warm_packets+cfg.grad_packets
    need=L+max(HORIZON_PACKETS)
    n=prep.n_packets
    allow=split['train_packet']
    valid=prep.valid.detach().cpu().numpy()>0
    rel=prep.packet_release;pe=prep.packet_end
    ok_allow=np.convolve(allow.astype(np.int32),np.ones(need,np.int32),'valid')==need
    cov=np.convolve(valid.astype(np.int32),np.ones(need,np.int32),'valid')/need
    cand=np.flatnonzero(ok_allow&(cov>=0.7))
    cand=cand[cand+need<=n]
    first=cfg.q_first or (cfg.warm_packets+cfg.stride-1)
    q=np.array([first+cfg.stride*j for j in range(cfg.n_start)])
    if q.max()>=L:raise ValueError('query offsets exceed the filtered prefix')
    if q.min()<60:
        # An hour block must have closed inside the prefix, or nothing is published.
        raise ValueError('query offsets below 60 packets cannot see any published packet')
    return cand,q


def query_indices(prep,starts,q):
    """Local index of the last released packet for each query, and the lag in packets."""
    rel=prep.packet_release;pe=prep.packet_end
    B=len(starts);J=len(q)
    kstar=np.zeros((B,J),np.int64);lag=np.zeros((B,J),np.int64)
    ok=np.ones((B,J),bool)
    for b,s in enumerate(starts):
        r=rel[s:s+int(q.max())+1]
        for j,off in enumerate(q):
            tau=pe[s+off]
            k=int(np.searchsorted(r[:off+1],tau,'right')-1)
            if k<0:
                # No hour block has closed inside this prefix, so nothing is published yet.
                ok[b,j]=False;k=0
            kstar[b,j]=k;lag[b,j]=off-k
    return kstar,lag,ok


def filter_episodes(model,prep,starts,q,cfg,generator,training=True,reference=None):
    """Causal filter over one batch of episodes; returns prior terms and query states."""
    device=prep.device
    B=len(starts);L=cfg.warm_packets+cfg.grad_packets
    idx=torch.as_tensor(starts,device=device).reshape(-1,1)+torch.arange(L,device=device).reshape(1,-1)
    flat=idx.reshape(-1)
    stats=prep.stats[flat]
    enc=model.encoder if (model is not None and model.rich) else (
        reference.encoder if (reference is not None and reference.rich) else None)
    if enc is not None:rich=prep.rich_summary(enc,flat)
    else:rich=torch.zeros(len(flat),RICH_DIM,device=device,dtype=stats.dtype)
    stats=stats.reshape(B,L,-1);rich=rich.reshape(B,L,-1)
    kstar,lag,q_ok=query_indices(prep,starts,q)
    kstar_t=torch.as_tensor(kstar,device=device)
    matches={}
    for j in range(kstar.shape[1]):
        for k in np.unique(kstar[:,j]):matches.setdefault(int(k),[]).append(j)
    grad_from=max(0,L-cfg.grad_packets)
    cache={}
    logp={k:0. for k in VIEWS+SECONDARY};units={k:0. for k in VIEWS+SECONDARY}
    prior_batch=[]
    S=cfg.train_paths if training else cfg.eval_paths
    if model is not None:
        m,P,c=model.initial(B,device,stats.dtype)
        snap_m=[None]*len(q);snap_P=[None]*len(q)
    else:
        ema=None;snap_h=[None]*len(q)
    n_grad_states=0
    for k in range(L):
        pidx=idx[:,k]
        on=training and k>=grad_from
        with (torch.enable_grad() if on else torch.no_grad()):
            if model is not None:
                if k>=grad_from and training:
                    z_s=_sample_paths(m,P,S,generator)
                    z_e=propagate_samples(model.dynamics,z_s,PACKET_HOURS,generator=generator,cache=cache)
                    prior_batch.append((z_s,z_e,pidx))
                m_e,P_e=propagate_moments(model.dynamics,m,P,PACKET_HOURS,cache=cache)
                # Inputs: during training only the fitting region may be read; at scoring
                # time every already-published packet is legitimately observed, which is
                # what a forward prediction means.
                readable=(prep.valid[pidx]*(prep.train_packet[pidx] if training else 1.)).reshape(-1,1)
                x=model.slow.packet_input(stats[:,k],rich[:,k])
                c=readable*model.slow.gru(x,c)+(1-readable)*c
                a,R,_=model.slow.evidence(c,m_e,P_e)
                m_p,P_p,_,_=evidence_update(m_e.double(),P_e.double(),a.double(),R.double())
                m=readable*m_p.to(m.dtype)+(1-readable)*m_e
                P=readable.unsqueeze(-1)*P_p.to(P.dtype)+(1-readable).unsqueeze(-1)*P_e
                keep=(1.-prep.seizure_masked[pidx]).reshape(-1,1)
                m0,P0,c0=model.initial(B,device,stats.dtype)
                m=keep*m+(1-keep)*m0
                P=keep.unsqueeze(-1)*P+(1-keep).unsqueeze(-1)*P0
                c=keep*c+(1-keep)*c0
            else:
                x=torch.cat((stats[:,k],rich[:,k]),dim=-1) if reference.rich else stats[:,k]
                if reference.mode=='recent_rate':x=stats[:,k][:,:1]
                hist=None
                if reference.mode not in ('intercept','clock','constant_state'):
                    if ema is None:
                        ema=torch.zeros(B,len(HISTORY_TAU_HOURS),x.shape[-1],device=device,dtype=x.dtype)
                        decay=torch.exp(-torch.as_tensor(PACKET_HOURS/np.asarray(HISTORY_TAU_HOURS),
                                                         device=device,dtype=x.dtype))
                    hist=ema.reshape(B,-1)
                    # Normalised weighted average: an unnormalised sum makes the 8-hour
                    # kernel ~480x the 1-minute kernel and cripples the reference at init.
                if k>=grad_from and training:
                    z=reference.latent(hist,prep.clock[pidx],
                                       torch.full((B,),PACKET_HOURS,device=device,dtype=x.dtype),
                                       prep.exposure_hours[pidx],B).unsqueeze(0)
                    prior_batch.append((z,z,pidx))
                if hist is not None:
                    readable=(prep.valid[pidx]*(prep.train_packet[pidx] if training else 1.)).reshape(-1,1,1)
                    w=(1.-decay).reshape(1,-1,1)
                    ema=ema*decay.reshape(1,-1,1)+readable*w*x.unsqueeze(1)
                    ema=ema*(1.-prep.seizure_masked[pidx]).reshape(-1,1,1)
        for j in matches.get(k,()):
            sel=(kstar_t[:,j]==k)
            if model is not None:
                snap_m[j]=m if snap_m[j] is None else torch.where(sel.reshape(-1,1),m,snap_m[j])
                snap_P[j]=P if snap_P[j] is None else torch.where(sel.reshape(-1,1,1),P,snap_P[j])
            else:
                h=hist if hist is not None else None
                snap_h[j]=h if (snap_h[j] is None or h is None) else torch.where(sel.reshape(-1,1),h,snap_h[j])
            n_grad_states+=int(sel.sum())*int(k>=grad_from)
    if prior_batch:
        head=model.readout if model is not None else reference.readout
        zs=torch.cat([r[0] for r in prior_batch],dim=1)
        ze=torch.cat([r[1] for r in prior_batch],dim=1)
        pid=torch.cat([r[2] for r in prior_batch])
        lp,un=view_log_probs(head,zs,ze,prep,pid)
        ok=prep.target_ok[pid]
        for key in lp:
            u=un[key]*ok
            logp[key]=(lp[key]*(u>0)).sum();units[key]=u.sum()
    if model is not None:
        return logp,units,dict(m=snap_m,P=snap_P,lag=lag,q=q,starts=starts,q_ok=q_ok,
                               grad_state_fraction=n_grad_states/max(1,B*len(q)))
    return logp,units,dict(hist=snap_h,lag=lag,q=q,starts=starts,q_ok=q_ok,grad_state_fraction=1.)


def rollout_from_queries(model,prep,snap,cfg,generator,training=True,reference=None,
                         rule='EVOLVE',relax=None,collect_scores=False):
    """Autonomous prediction 1/5/30/120 minutes past each query time."""
    device=prep.device
    q=snap['q'];starts=np.asarray(snap['starts']);lag=snap['lag']
    B=len(starts);J=len(q)
    base=torch.as_tensor((starts.reshape(-1,1)+q.reshape(1,-1)).reshape(-1),device=device)
    lag_flat=torch.as_tensor(lag.reshape(-1),device=device)
    q_ok=torch.as_tensor(snap['q_ok'].reshape(-1).astype(np.float32),device=device,dtype=prep.stats.dtype)
    S=cfg.train_paths if training else cfg.eval_paths
    logp={h:{k:0. for k in VIEWS+SECONDARY} for h in HORIZON_PACKETS}
    units={h:{k:0. for k in VIEWS+SECONDARY} for h in HORIZON_PACKETS}
    detail=[] if collect_scores else None
    if model is not None:
        m=torch.cat([snap['m'][j] for j in range(J)],0) if J else None
        P=torch.cat([snap['P'][j] for j in range(J)],0)
        order=torch.as_tensor(np.concatenate([np.arange(B)*J+j for j in range(J)]),device=device)
        inv=torch.argsort(order)
        m=m[inv];P=P[inv]
        if rule=='RELAX' and relax is not None:m,P=relax_state(m,P,relax,0.)
        z=_sample_paths(m,P,S,generator)
        cache={}
        max_lag=int(lag_flat.max())
        lag_np=lag_flat.detach().cpu().numpy()
        plan={}
        for h in HORIZON_PACKETS:
            for st in np.unique(lag_np+h):plan.setdefault(int(st),[]).append(h)
        buckets={h:[] for h in HORIZON_PACKETS}
        for step in range(1,max_lag+max(HORIZON_PACKETS)+1):
            z_prev=z
            if rule!='HOLD':
                z=propagate_samples(model.dynamics,z,PACKET_HOURS,generator=generator,cache=cache)
            for h in plan.get(step,()):
                sel=torch.as_tensor(np.flatnonzero((lag_np+h)==step),device=device)
                tgt=base[sel]+h
                keep=tgt<prep.n_packets
                if not bool(keep.any()):continue
                idx_sel=sel[keep]
                buckets[h].append((z_prev[:,idx_sel],z[:,idx_sel],tgt[keep],lag_flat[idx_sel],idx_sel))
        for h in HORIZON_PACKETS:
            if not buckets[h]:continue
            zp=torch.cat([r[0] for r in buckets[h]],dim=1);zn=torch.cat([r[1] for r in buckets[h]],dim=1)
            t=torch.cat([r[2] for r in buckets[h]]);lg=torch.cat([r[3] for r in buckets[h]])
            ok_sel=torch.cat([q_ok[r[4]] for r in buckets[h]])
            lp,un=view_log_probs(model.readout,zp,zn,prep,t)
            allow=prep.valid[t]*(1.-prep.seizure_masked[t])*ok_sel
            if prep.target_region is not None:allow=allow*prep.target_region[t]
            if training:allow=allow*prep.target_ok[t]
            for key in lp:
                u=un[key]*allow
                logp[h][key]=logp[h][key]+(lp[key]*(u>0)).sum()
                units[h][key]=units[h][key]+u.sum()
            if collect_scores:
                detail.append(dict(horizon=h,packet=t.detach().cpu().numpy(),
                                   lag_minutes=lg.detach().cpu().numpy(),
                                   **{f'logp_{key}':lp[key].detach().cpu().numpy() for key in lp},
                                   **{f'units_{key}':(un[key]*allow).detach().cpu().numpy() for key in lp}))
    else:
        hist=snap['hist']
        h0=hist[0]
        stacked=None if h0 is None else torch.cat([hist[j] for j in range(J)],0)
        if stacked is not None:
            order=torch.as_tensor(np.concatenate([np.arange(B)*J+j for j in range(J)]),device=device)
            stacked=stacked[torch.argsort(order)]
        for h in HORIZON_PACKETS:
            tgt=base+h
            keep=tgt<prep.n_packets
            t=tgt[keep]
            lead=(lag_flat[keep]+h).to(prep.stats.dtype)*PACKET_HOURS
            hh=None if stacked is None else stacked[keep]
            z=reference.latent(hh,prep.clock[t],lead,prep.exposure_hours[t],len(t)).unsqueeze(0)
            lp,un=view_log_probs(reference.readout,z,z,prep,t)
            allow=prep.valid[t]*(1.-prep.seizure_masked[t])*q_ok[keep]
            if prep.target_region is not None:allow=allow*prep.target_region[t]
            if training:allow=allow*prep.target_ok[t]
            for key in lp:
                u=un[key]*allow
                logp[h][key]=logp[h][key]+(lp[key]*(u>0)).sum()
                units[h][key]=units[h][key]+u.sum()
            if collect_scores:
                detail.append(dict(horizon=h,packet=t.detach().cpu().numpy(),
                                   lag_minutes=lag_flat[keep].detach().cpu().numpy(),
                                   **{f'logp_{key}':lp[key].detach().cpu().numpy() for key in lp},
                                   **{f'units_{key}':(un[key]*allow).detach().cpu().numpy() for key in lp}))
    return logp,units,detail


def relax_state(m,P,relax,hours):
    """RELAX: pull the state back to the FIT stationary distribution."""
    a=math.exp(-hours/relax['tau_hours'])
    m0=relax['m0'].to(m.device,m.dtype);P0=relax['P0'].to(P.device,P.dtype)
    return m0+a*(m-m0),a*a*P+(1-a*a)*P0


def _scalar(x):
    return float(x.detach()) if torch.is_tensor(x) else float(x)


def combined_loss(prior_logp,prior_units,roll_logp,roll_units,views=VIEWS):
    """0.5 x per-packet prior + 0.5 x multi-horizon, equal-weight main views."""
    def block(lp,un):
        main=[]
        for k in views:
            if _scalar(un[k])<=0:continue
            main.append(-lp[k]/un[k])
        if not main:return None
        out=sum(main)/len(main)
        if _scalar(un['load'])>0:out=out+0.25*(-lp['load']/un['load'])
        return out
    a=block(prior_logp,prior_units)
    hs=[block(roll_logp[h],roll_units[h]) for h in HORIZON_PACKETS]
    hs=[h for h in hs if h is not None]
    b=sum(hs)/len(hs) if hs else None
    if a is None and b is None:return None
    if a is None:return b
    if b is None:return a
    return 0.5*a+0.5*b


def per_view_nats(logp,units):
    return {k:(None if _scalar(units[k])<=0 else _scalar(logp[k])/_scalar(units[k])*-1.)
            for k in VIEWS+SECONDARY}


def evaluate_starts(model,prep,split,cfg,generator,query_times,reference=None,rule='EVOLVE',
                    relax=None,chunk=24,paths=None,collect=False):
    """Score a list of query times with the same prefix rule used in training."""
    L=cfg.warm_packets+cfg.grad_packets
    pe=prep.packet_end
    idx=np.searchsorted(pe,np.asarray(query_times,float)-1e-6)
    eligible=[];reasons={'prefix_before_support':0,'target_beyond_support':0,
                         'query_inside_exclusion':0,'all_targets_masked':0}
    history=[]
    for i in idx:
        if i-L+1<0:reasons['prefix_before_support']+=1;continue
        if i+max(HORIZON_PACKETS)>=prep.n_packets:reasons['target_beyond_support']+=1;continue
        if prep.seizure_masked_np[i]:reasons['query_inside_exclusion']+=1;continue
        tg=[i+h for h in HORIZON_PACKETS]
        ok_t=prep.valid_np[tg]&(~prep.seizure_masked_np[tg])&prep.target_region_np[tg]
        if not ok_t.any():
            reasons['all_targets_masked']+=1;continue
        pre=prep.seizure_masked_np[i-L+1:i+1]
        last=int(np.flatnonzero(pre)[-1])+1 if pre.any() else 0
        history.append((L-last)/60.)
        eligible.append(int(i))
    logp={h:{k:0. for k in VIEWS+SECONDARY} for h in HORIZON_PACKETS}
    units={h:{k:0. for k in VIEWS+SECONDARY} for h in HORIZON_PACKETS}
    detail=[]
    cfg_eval=copy.copy(cfg);cfg_eval.n_start=1;cfg_eval.stride=1
    q=np.array([L-1])
    for a in range(0,len(eligible),chunk):
        starts=np.array([e-L+1 for e in eligible[a:a+chunk]],np.int64)
        _,_,snap=filter_episodes(model,prep,starts,q,cfg_eval,generator,training=False,reference=reference)
        lp,un,det=rollout_from_queries(model,prep,snap,cfg_eval,generator,training=False,
                                       reference=reference,rule=rule,relax=relax,collect_scores=collect)
        for h in HORIZON_PACKETS:
            for k in VIEWS+SECONDARY:
                logp[h][k]=logp[h][k]+_scalar(lp[h][k]);units[h][k]=units[h][k]+_scalar(un[h][k])
        if collect and det:detail.extend(det)
    per={h:{k:(None if units[h][k]<=0 else -logp[h][k]/units[h][k]) for k in VIEWS+SECONDARY}
         for h in HORIZON_PACKETS}
    main=[per[h][k] for h in per for k in VIEWS if per[h][k] is not None]
    return dict(per_horizon=per,logp=logp,units=units,selection=(float(np.mean(main)) if main else None),
                n_eligible=len(eligible),n_requested=len(idx),not_estimable=reasons,detail=detail,
                effective_history_hours=dict(median=float(np.median(history)) if history else None,
                                             min=float(np.min(history)) if history else None,
                                             full_prefix_fraction=float(np.mean(np.asarray(history)>=(L-1)/60.))
                                             if history else None))


def build_run(cfg,payload,split,scaling,prep):
    torch.manual_seed(cfg.seed)
    n_token=prep.tokens.shape[-1];n_group=prep.groups.shape[-1]
    n_event=prep.event.shape[-1];n_shaft=prep.n_shaft
    n_ratio=prep.band_ratio.shape[-1];n_xlag=prep.xlag.shape[-1]
    n_stats=prep.stats.shape[-1]
    enc_kwargs=dict(n_contacts=prep.part.shape[-1],n_token=n_token,n_group=n_group,
                    n_event=n_event,n_shaft=n_shaft)
    if cfg.arm=='state':
        model=StateModel(prep.part.shape[-1],n_token,n_group,n_event,n_shaft,n_stats,n_ratio,n_xlag,
                         rich=cfg.rich,coupled=cfg.coupled,nonlinear=cfg.nonlinear,
                         linear_readout=cfg.linear_readout,seed=cfg.seed).to(cfg.device)
        return model,None
    ref=HistoryReference(n_stats,n_shaft,n_ratio,n_xlag,cfg.arm,cfg.rich,
                         encoder_kwargs=enc_kwargs).to(cfg.device)
    return None,ref


def apply_ablations(prep,cfg):
    """Targeted ablations from the rev3 plan, applied to inputs and targets together."""
    notes=[]
    if cfg.shuffle_marks:
        # Capacity control: the encoder keeps every parameter and every marginal
        # distribution, but each event is handed another event's marks, so the
        # event-specific information is gone. Targets are untouched.
        g=torch.Generator(device=prep.device).manual_seed(int(cfg.seed)+991)
        n=prep.tokens.shape[0]
        perm=torch.randperm(n,generator=g,device=prep.device)
        prep.tokens=prep.tokens[perm].contiguous()
        prep.groups=prep.groups[perm].contiguous()
        prep.event=prep.event[perm].contiguous()
        prep.event_valid=prep.event_valid[perm].contiguous()
        notes.append('mark-shuffle capacity control: per-event contact tokens and event descriptors '
                     'permuted across events with a fixed seed; participation, targets and the '
                     'per-minute summary layout are untouched')
    if cfg.crossview_ablation:
        prep.tokens[:,:,0:3]=0.;prep.groups[:,:,0]=0.
        prep.event[:,4]=0.;prep.event[:,5]=0.
        prep.event_valid[:,4]=0.;prep.event_valid[:,5]=0.
        prep.iqr_valid[:]=0.
        notes.append('cross-view ablation: relative delay, delay rank, tied-lead flag, delay IQR and '
                     'delay span removed from inputs; delay-IQR target removed')
    return notes


def main_views_for(cfg):
    if cfg.target_ablation:return ('count','spatial')
    return VIEWS


def run_cell(cfg,progress=None):
    """One fit: build, train with plateau schedule, select on INNER, score outer."""
    t0=time.time()
    out=Path(cfg.out_dir);out.mkdir(parents=True,exist_ok=True)
    payload=torch.load(f'{cfg.packets_root}/{cfg.subject}.pt',weights_only=False)
    from . import data as D
    split=(D.build_split(payload,cfg.subject,cfg.seed) if cfg.split=='S-E'
           else D.build_split_id(payload,cfg.subject,cfg.seed))
    px,pt,_=D.packet_tables(payload,split)
    scaling=D.fit_scaling(payload,split,px,pt)
    device=torch.device(cfg.device)
    prep=Prepared(payload,split,scaling,device)
    notes=apply_ablations(prep,cfg)
    model,reference=build_run(cfg,payload,split,scaling,prep)
    module=model if model is not None else reference
    initial={n:p.detach().cpu().clone() for n,p in module.named_parameters()}
    groups,group_record=build_param_groups([('model',module)],weight_decay=1e-4)
    opt=torch.optim.AdamW(groups,lr=cfg.lr,betas=(0.9,0.999),eps=1e-8)
    plateau=PlateauSchedule(cfg.lr,eval_every=cfg.eval_every)
    cand,q=episode_plan(prep,split,cfg)
    if len(cand)<cfg.batch_episodes:
        return dict(status='NOT_ESTIMABLE',reason=f'only {len(cand)} legal episodes',config=asdict(cfg))
    rng=np.random.default_rng(cfg.seed)
    gen=torch.Generator(device=device).manual_seed(int(cfg.seed))
    inner=split['inner_starts']
    curve=[];best=dict(score=math.inf,updates=-1,state=None)
    updates=0;budget=cfg.max_updates;stop_reason='budget'
    peak=0.;grad_fraction=[];skipped=0;non_finite_params=[]
    last_good={k:v.detach().clone() for k,v in module.state_dict().items()}
    mv=main_views_for(cfg)
    while updates<budget:
        starts=rng.choice(cand,size=cfg.batch_episodes,replace=False)
        opt.zero_grad(set_to_none=True)
        plog,punits,snap=filter_episodes(model,prep,starts,q,cfg,gen,training=True,reference=reference)
        rlog,runits,_=rollout_from_queries(model,prep,snap,cfg,gen,training=True,reference=reference)
        grad_fraction.append(snap['grad_state_fraction'])
        loss=combined_loss(plog,punits,rlog,runits,views=mv)
        if loss is None:
            stop_reason='no_estimable_target';break
        if not torch.isfinite(loss):
            skipped+=1;updates+=1
            if skipped>50:stop_reason='non_finite_loss';break
            continue
        loss.backward()
        gnorm=torch.nn.utils.clip_grad_norm_(module.parameters(),2.0)
        if not torch.isfinite(gnorm):
            # A non-finite gradient would write NaN into the dynamics and only surface
            # later as a singular Lyapunov solve; drop the step and record it.
            skipped+=1;updates+=1
            if skipped>50:stop_reason='non_finite_gradient';break
            continue
        for g in opt.param_groups:g['lr']=plateau.lr
        opt.step()
        bad=[n for n,q in module.named_parameters() if not torch.isfinite(q).all()]
        if bad:
            # A non-finite parameter would only surface later as a singular Lyapunov
            # solve; roll back to the last finite state and record the event.
            module.load_state_dict(last_good)
            opt=torch.optim.AdamW(build_param_groups([('model',module)],weight_decay=1e-4)[0],
                                  lr=plateau.lr,betas=(0.9,0.999),eps=1e-8)
            non_finite_params.append(dict(update=updates,tensors=bad[:5],
                                          loss=float(loss.detach()),grad_norm=float(gnorm)))
            skipped+=1;updates+=1
            if len(non_finite_params)>20:stop_reason='non_finite_parameters';break
            continue
        last_good={k:v.detach().clone() for k,v in module.state_dict().items()}
        updates+=1
        if device.type=='cuda':peak=max(peak,torch.cuda.max_memory_allocated(device)/2**30)
        if updates%cfg.eval_every==0:
            with torch.no_grad():
                ev=evaluate_starts(model,prep,split,cfg,gen,inner,reference=reference)
            score=ev['selection']
            curve.append(dict(updates=updates,train_loss=float(loss.detach()),inner=score,lr=plateau.lr,
                              per_horizon=ev['per_horizon'],n_eligible=ev['n_eligible']))
            if score is not None and score<best['score']:
                best=dict(score=score,updates=updates,
                          state={k:v.detach().cpu().clone() for k,v in module.state_dict().items()})
            action=plateau.observe(updates,score if score is not None else math.inf)
            if progress is not None:progress(updates,score,plateau.lr,action)
            if action=='stop':
                stop_reason='plateau';break
            if (updates>=cfg.max_updates and cfg.extended_updates>cfg.max_updates and score is not None
                    and curve[-1]['inner']<curve[max(0,len(curve)-9)]['inner']):
                budget=cfg.extended_updates;stop_reason='extended_budget'
    if best['state'] is not None:module.load_state_dict(best['state'])
    with torch.no_grad():
        inner_final=evaluate_starts(model,prep,split,cfg,gen,inner,reference=reference,paths=cfg.eval_paths)
        outer=evaluate_starts(model,prep,split,cfg,gen,split['forward_starts'],reference=reference,
                              paths=cfg.eval_paths,collect=True)
    card=dict(status='COMPLETE',config=asdict(cfg),seconds=round(time.time()-t0,1),
              updates=updates,stop_reason=stop_reason,selected_updates=best['updates'],
              inner_selection=best['score'],inner_final=inner_final['per_horizon'],
              outer=outer['per_horizon'],outer_units=outer['units'],
              outer_eligible=outer['n_eligible'],outer_requested=outer['n_requested'],
              outer_not_estimable=outer['not_estimable'],
              inner_eligible=inner_final['n_eligible'],
              curve=curve,plateau=plateau.state(),param_groups=group_record,
              parameter_update=update_ratio(initial,module),
              parameter_hash=per_parameter_hash(module),
              gradient_state_fraction=float(np.mean(grad_fraction)) if grad_fraction else None,
              skipped_non_finite_updates=int(skipped),
              non_finite_parameter_events=non_finite_params,
              effective_start_points=int(cfg.batch_episodes*cfg.n_start),
              peak_gpu_gib=round(peak,2),ablation_notes=notes,
              split=dict(se_cutoff=split.get('se_cutoff'),gap_start=split.get('se_gap_start'),
                         fit_end=split['fit_end'],n_inner=len(inner),
                         n_forward=len(split['forward_starts']),seed=split['seed'],
                         contract=split['contract']),
              scaling={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in scaling.items()},
              coarse_community_rule=payload.get('coarse_community_rule','physical shaft'),
              release_contract=payload['release_contract'],
              main_views=list(mv))
    tag=f"{cfg.subject}__{cfg.split}__{cfg.inputs}__{cfg.family}__{cfg.arm}__seed{cfg.seed}"
    if cfg.target_ablation:tag+='__oldtargets'
    if cfg.shuffle_marks:
        # Capacity control: the encoder keeps every parameter and every marginal
        # distribution, but each event is handed another event's marks, so the
        # event-specific information is gone. Targets are untouched.
        g=torch.Generator(device=prep.device).manual_seed(int(cfg.seed)+991)
        n=prep.tokens.shape[0]
        perm=torch.randperm(n,generator=g,device=prep.device)
        prep.tokens=prep.tokens[perm].contiguous()
        prep.groups=prep.groups[perm].contiguous()
        prep.event=prep.event[perm].contiguous()
        prep.event_valid=prep.event_valid[perm].contiguous()
        notes.append('mark-shuffle capacity control: per-event contact tokens and event descriptors '
                     'permuted across events with a fixed seed; participation, targets and the '
                     'per-minute summary layout are untouched')
    if cfg.crossview_ablation:tag+='__nodelay'
    if cfg.shuffle_marks:tag+='__shuffledmarks'
    (out/f'{tag}.card.json').write_text(json.dumps(card,indent=1,default=str))
    torch.save(dict(state_dict=module.state_dict(),config=asdict(cfg),scaling=scaling,
                    split_summary=card['split'],inner_selection=best['score']),out/f'{tag}.ckpt.pt')
    np.savez_compressed(out/f'{tag}.outer_detail.npz',
                        **{f"{d['horizon']}_{i}_{k}":v for i,d in enumerate(outer['detail'])
                           for k,v in d.items() if isinstance(v,np.ndarray)})
    return card
