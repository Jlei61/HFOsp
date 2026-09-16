"""Frozen-checkpoint readouts: HOLD / EVOLVE / RELAX and fine contact identity.

Nothing here retrains the producer. The identity adapter is a low-capacity
linear map fitted on producer FIT/INNER support only, so a fine-resolution
result can never have selected the producer.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn

from .numerics import LATENT,propagate_samples,propagate_moments
from .train import (RunConfig,filter_episodes,HORIZON_PACKETS,PACKET_HOURS,_sample_paths,
                    query_indices,relax_state)
from .objective import VIEWS,SECONDARY,view_log_probs

RELAX_TAU_GRID=(0.5,2.,8.,24.)


def log_elementary_symmetric(w,k):
    """log e_k(exp(w)) by the standard DP, in log space for stability."""
    n=w.shape[-1]
    e=[torch.full(w.shape[:-1],-math.inf,device=w.device,dtype=w.dtype) for _ in range(k+1)]
    e[0]=torch.zeros(w.shape[:-1],device=w.device,dtype=w.dtype)
    for i in range(n):
        for j in range(min(k,i+1),0,-1):
            e[j]=torch.logaddexp(e[j],e[j-1]+w[...,i])
    return e[k]


def fixed_k_set_log_prob(logits,members,k):
    """Normalised probability of an exact K-subset under a conditional Bernoulli."""
    chosen=(logits*members).sum(-1)
    return chosen-log_elementary_symmetric(logits,int(k))


def set_log_prob_batch(logits,members):
    """Per-event exact-set log-probability, grouped by the event's own K."""
    out=torch.zeros(members.shape[:-1],device=logits.device,dtype=logits.dtype)
    ks=members.sum(-1).long()
    for k in torch.unique(ks):
        if int(k)==0:continue
        sel=ks==k
        out[sel]=fixed_k_set_log_prob(logits[sel],members[sel],int(k))
    return out


def within_community_set_log_prob(logits,members,community):
    """Identity given the coarse community counts, so coarse position is controlled."""
    out=torch.zeros(members.shape[:-1],device=logits.device,dtype=logits.dtype)
    for c in torch.unique(community):
        cols=torch.nonzero(community==c).reshape(-1)
        if len(cols)<2:continue
        out=out+set_log_prob_batch(logits[...,cols],members[...,cols])
    return out


class IdentityAdapter(nn.Module):
    """Low-capacity linear map from a frozen feature vector to per-contact logits."""

    def __init__(self,in_dim,n_contacts,use_trait=True):
        super().__init__()
        self.n_contacts=int(n_contacts)
        self.trait=nn.Parameter(torch.zeros(n_contacts)) if use_trait else None
        self.linear=nn.Linear(in_dim,n_contacts) if in_dim>0 else None
        if self.linear is not None:
            nn.init.zeros_(self.linear.bias);nn.init.normal_(self.linear.weight,0.,1e-3)

    def forward(self,x):
        out=x.new_zeros(x.shape[0],self.n_contacts)
        if self.trait is not None:out=out+self.trait
        if self.linear is not None:out=out+self.linear(x)
        return out


@torch.no_grad()
def states_at_queries(model,prep,cfg,query_packets,reference=None,chunk=24):
    """Filtered state at each query, using the same prefix and release rule."""
    L=cfg.warm_packets+cfg.grad_packets
    import copy
    c=copy.copy(cfg);c.n_start=1;c.stride=1
    q=np.array([L-1])
    ms=[];Ps=[];keep=[]
    gen=torch.Generator(device=prep.device).manual_seed(int(cfg.seed))
    for a in range(0,len(query_packets),chunk):
        idx=np.asarray(query_packets[a:a+chunk],np.int64)
        starts=idx-L+1
        ok=starts>=0
        if not ok.any():continue
        starts=starts[ok]
        _,_,snap=filter_episodes(model,prep,starts,q,c,gen,training=False,reference=reference)
        ms.append(snap['m'][0]);Ps.append(snap['P'][0]);keep.append(idx[ok])
    if not ms:return None,None,np.empty(0,np.int64)
    return torch.cat(ms),torch.cat(Ps),np.concatenate(keep)


@torch.no_grad()
def score_rules(model,prep,cfg,query_packets,relax,paths=64):
    """HOLD / EVOLVE / RELAX on identical targets, clocks, exposure and masks."""
    m,P,kept=states_at_queries(model,prep,cfg,query_packets)
    if m is None:return {},kept
    gen=torch.Generator(device=prep.device).manual_seed(int(cfg.seed)+7)
    base=torch.as_tensor(kept,device=prep.device)
    out={}
    for rule in ('HOLD','EVOLVE','RELAX'):
        mm,PP=(relax_state(m,P,relax,0.) if rule=='RELAX' else (m,P))
        g=torch.Generator(device=prep.device).manual_seed(int(cfg.seed)+7)
        z=_sample_paths(mm,PP,paths,g)
        cache={}
        acc={h:{k:[0.,0.] for k in VIEWS+SECONDARY} for h in HORIZON_PACKETS}
        zp=z
        for step in range(1,max(HORIZON_PACKETS)+1):
            zp=z
            if rule=='EVOLVE':z=propagate_samples(model.dynamics,z,PACKET_HOURS,generator=g,cache=cache)
            elif rule=='RELAX':
                mm2,PP2=relax_state(m,P,relax,step*PACKET_HOURS)
                g2=torch.Generator(device=prep.device).manual_seed(int(cfg.seed)+7)
                z=_sample_paths(mm2,PP2,paths,g2)
            if step not in HORIZON_PACKETS:continue
            t=base+step
            sel=t<prep.n_packets
            lp,un=view_log_probs(model.readout,zp[:,sel],z[:,sel],prep,t[sel])
            allow=prep.valid[t[sel]]*(1.-prep.seizure_masked[t[sel]])
            for k in lp:
                u=un[k]*allow
                acc[step][k][0]+=float((lp[k]*(u>0)).sum());acc[step][k][1]+=float(u.sum())
        out[rule]={h:{k:(None if v[1]<=0 else -v[0]/v[1]) for k,v in acc[h].items()} for h in acc}
    return out,kept


def fit_relax(model,prep,cfg,fit_packets,inner_packets):
    """m0/P0 and the relaxation time from FIT and INNER support only."""
    m,P,_=states_at_queries(model,prep,cfg,fit_packets)
    if m is None:return None
    m0=m.mean(0)
    P0=torch.from_numpy(np.cov(m.detach().cpu().numpy().T)).to(m.dtype).to(m.device)
    P0=P0+1e-3*torch.eye(LATENT,device=m.device,dtype=m.dtype)
    best=None
    for tau in RELAX_TAU_GRID:
        relax=dict(m0=m0,P0=P0,tau_hours=tau)
        s,_=score_rules(model,prep,cfg,inner_packets,relax,paths=16)
        if not s:continue
        vals=[s['RELAX'][h][k] for h in s['RELAX'] for k in VIEWS if s['RELAX'][h][k] is not None]
        score=float(np.mean(vals)) if vals else math.inf
        if best is None or score<best[0]:best=(score,tau)
    tau=best[1] if best else RELAX_TAU_GRID[1]
    return dict(m0=m0,P0=P0,tau_hours=tau,inner_selected_score=best[0] if best else None,
                grid=list(RELAX_TAU_GRID))
