"""Rich event encoder, slow evidence head and future readout.

P_stats and P_marks share the slow input layout; P_stats leaves the rich
positions missing so the two arms differ only in whether per-event structure
reaches the encoder.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

from .numerics import Dynamics,LATENT,propagate_moments,propagate_samples,evidence_update

RICH_DIM=32
SLOW_HIDDEN=64
TOKEN_HIDDEN=32
EVENT_DIM=32
READOUT_HIDDEN=64


def orthogonal_gru_(cell,generator=None):
    h=cell.hidden_size
    with torch.no_grad():
        for k in range(3):
            w=torch.empty(h,h)
            nn.init.orthogonal_(w)
            cell.weight_hh[k*h:(k+1)*h].copy_(w)
        nn.init.xavier_uniform_(cell.weight_ih)
        cell.bias_ih.zero_();cell.bias_hh.zero_()


class EventEncoder(nn.Module):
    """Participating-contact tokens -> masked pool -> event -> within-minute GRU."""

    def __init__(self,n_contacts,n_token,n_group,n_event,n_shaft,embed=8):
        super().__init__()
        self.embed=nn.Embedding(n_contacts,embed)
        nn.init.normal_(self.embed.weight,0.,0.1)
        self.token=nn.Sequential(nn.Linear(n_token+n_group+embed,TOKEN_HIDDEN),nn.GELU(),
                                 nn.Linear(TOKEN_HIDDEN,TOKEN_HIDDEN))
        self.event=nn.Sequential(nn.Linear(2*TOKEN_HIDDEN+n_event*2+n_shaft+1,EVENT_DIM),nn.GELU())
        self.gru=nn.GRUCell(EVENT_DIM+1,RICH_DIM)
        orthogonal_gru_(self.gru)
        self.n_contacts=n_contacts

    def forward(self,tokens,groups,part,event,event_valid,shaft_frac,size,gap,segment,n_packets):
        """One flat run of events; ``segment`` maps each event to its packet row."""
        if tokens.shape[0]==0:
            return tokens.new_zeros(n_packets,RICH_DIM)
        n,c,_=tokens.shape
        ids=torch.arange(c,device=tokens.device).expand(n,c)
        x=torch.cat((tokens,groups,self.embed(ids)),dim=-1)
        h=self.token(x)*part.unsqueeze(-1)
        denom=part.sum(-1,keepdim=True).clamp(min=1.)
        mean=h.sum(1)/denom
        mx=torch.where(part.unsqueeze(-1).bool(),h,torch.full_like(h,-1e9)).max(1).values
        mx=torch.where(part.sum(-1,keepdim=True)>0,mx,torch.zeros_like(mx))
        e=self.event(torch.cat((mean,mx,event,event_valid,shaft_frac,size),dim=-1))
        state=tokens.new_zeros(n_packets,RICH_DIM)
        cur=tokens.new_zeros(n_packets,RICH_DIM)
        order=torch.argsort(segment,stable=True)
        e=e[order];gap=gap[order];seg=segment[order]
        step=torch.cat((e,gap.unsqueeze(-1)),dim=-1)
        # Events arrive in real order inside their packet; packets advance together.
        counts=torch.bincount(seg,minlength=n_packets)
        pos=torch.arange(len(seg),device=seg.device)-torch.cumsum(
            torch.cat((counts.new_zeros(1),counts[:-1])),0)[seg]
        for k in range(int(counts.max())):
            sel=pos==k
            rows=seg[sel]
            cur=cur.index_copy(0,rows,self.gru(step[sel],cur.index_select(0,rows)))
        return cur


class SlowState(nn.Module):
    """Packet-level GRU memory, evidence location and learned evidence noise."""

    def __init__(self,n_packet_input,rich=True):
        super().__init__()
        self.rich=rich
        self.input_dim=n_packet_input+RICH_DIM+1
        self.gru=nn.GRUCell(self.input_dim,SLOW_HIDDEN)
        orthogonal_gru_(self.gru)
        self.a_head=nn.Linear(SLOW_HIDDEN+2*LATENT,LATENT)
        self.eta_head=nn.Linear(SLOW_HIDDEN+2*LATENT,LATENT)
        nn.init.normal_(self.a_head.weight,0.,1e-3);nn.init.zeros_(self.a_head.bias)
        nn.init.zeros_(self.eta_head.weight);nn.init.zeros_(self.eta_head.bias)

    def packet_input(self,stats,rich):
        if self.rich:
            flag=torch.ones(stats.shape[0],1,dtype=stats.dtype,device=stats.device)
            return torch.cat((stats,rich,flag),dim=-1)
        zero=torch.zeros(stats.shape[0],RICH_DIM,dtype=stats.dtype,device=stats.device)
        flag=torch.zeros(stats.shape[0],1,dtype=stats.dtype,device=stats.device)
        return torch.cat((stats,zero,flag),dim=-1)

    def evidence(self,c,m,P):
        v=torch.diagonal(P,dim1=-2,dim2=-1)
        # Bound the prior features used by the innovation network. The identity
        # path m -> posterior mean remains outside this normalization, so this
        # prevents an unbounded positive feedback term without clipping history.
        h=torch.cat((c,torch.tanh(m),torch.log1p(v.clamp(min=0))),dim=-1)
        # A zero innovation must preserve the predicted location, rather than
        # repeatedly attracting every latent coordinate toward arbitrary zero.
        # Dependence on m/P remains learnable; Joseph covariance contraction is
        # unchanged and is not replaced by translating Monte Carlo particles.
        a=m+self.a_head(h)
        rho=torch.exp(math.log(10.)+6.*torch.tanh(self.eta_head(h)))
        R=torch.diag_embed(v*rho)
        return a,R,rho


class Readout(nn.Module):
    """Natural parameters for count, coarse composition, morphology and load."""

    def __init__(self,n_shaft,n_ratio,n_xlag,hidden=READOUT_HIDDEN,linear=False):
        super().__init__()
        self.n_shaft=n_shaft;self.n_ratio=n_ratio;self.n_xlag=n_xlag
        out=1+n_shaft+2*n_ratio+2*n_xlag+3+2
        self.linear=linear
        self.net=nn.Linear(LATENT+2,out) if linear else nn.Sequential(
            nn.Linear(LATENT+2,hidden),nn.GELU(),nn.Linear(hidden,out))
        self.log_nb_dispersion=nn.Parameter(torch.zeros(1))
        self.register_buffer('fit_offset',torch.zeros(out))
        self.contact_head=None
        self.use_clock=True

    def add_identity_head(self,n_contacts):
        self.contact_head=nn.Linear(LATENT+2,n_contacts)
        nn.init.zeros_(self.contact_head.bias);nn.init.normal_(self.contact_head.weight,0.,1e-3)

    def forward(self,z,clock):
        if not self.use_clock:clock=torch.zeros_like(clock)
        y=self.net(torch.cat((z,clock),dim=-1))+self.fit_offset
        i=0
        log_rate=y[...,0];i=1
        comp=y[...,i:i+self.n_shaft];i+=self.n_shaft
        br_mu=y[...,i:i+self.n_ratio];i+=self.n_ratio
        br_ls=y[...,i:i+self.n_ratio];i+=self.n_ratio
        xl_mu=y[...,i:i+self.n_xlag];i+=self.n_xlag
        xl_ls=y[...,i:i+self.n_xlag];i+=self.n_xlag
        iqr=y[...,i:i+3];i+=3
        load=y[...,i:i+2]
        return dict(log_rate=log_rate,composition=comp,band_ratio_mu=br_mu,band_ratio_logsd=br_ls,
                    xlag_mu=xl_mu,xlag_logsd=xl_ls,iqr=iqr,load=load)


class StateModel(nn.Module):
    def __init__(self,n_contacts,n_token,n_group,n_event,n_shaft,n_packet_input,n_ratio,n_xlag,
                 rich=True,coupled=False,nonlinear=False,linear_readout=False,seed=20260906):
        super().__init__()
        g=torch.Generator().manual_seed(int(seed))
        self.dynamics=Dynamics(coupled=coupled,nonlinear=nonlinear,generator=g)
        # Separate RNG streams keep common parameters identical across inputs.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed)+101)
            self.encoder=EventEncoder(n_contacts,n_token,n_group,n_event,n_shaft) if rich else None
            torch.manual_seed(int(seed)+102)
            self.slow=SlowState(n_packet_input,rich=rich)
            torch.manual_seed(int(seed)+103)
            self.readout=Readout(n_shaft,n_ratio,n_xlag,linear=linear_readout)
        self.rich=rich
        self.register_buffer('m0',torch.zeros(LATENT))
        self.register_buffer('P0',torch.eye(LATENT))

    def initial(self,batch,device,dtype=torch.float32):
        m=self.m0.to(device=device,dtype=dtype).expand(batch,LATENT).clone()
        P=self.P0.to(device=device,dtype=dtype).expand(batch,LATENT,LATENT).clone()
        c=torch.zeros(batch,SLOW_HIDDEN,device=device,dtype=dtype)
        return m,P,c


LOG_RATE_LIMITS=(-20.,15.)


def count_log_prob(log_rate_start,log_rate_end,exposure_hours,count,log_dispersion):
    """Trapezoid integral of exposure * rate over the packet, then NB.

    The log rate is clamped to e^15 events per hour, far above any observed
    value; without it a single large step overflows float32 and writes NaN into
    the dynamics, which only surfaces later as a singular Lyapunov solve.
    """
    lo,hi=LOG_RATE_LIMITS
    log_rate_start=log_rate_start.clamp(lo,hi);log_rate_end=log_rate_end.clamp(lo,hi)
    lam=0.5*(torch.exp(log_rate_start)+torch.exp(log_rate_end))*exposure_hours
    lam=lam.clamp(min=1e-9)
    r=torch.exp((-log_dispersion).clamp(math.log(1e-3),math.log(1e6)))
    return (torch.lgamma(count+r)-torch.lgamma(r)-torch.lgamma(count+1.)
            +r*(torch.log(r)-torch.log(r+lam))+count*(torch.log(lam)-torch.log(r+lam)))


def composition_log_prob(logits,counts):
    """Multinomial over coarse shafts given the total participating contacts."""
    total=counts.sum(-1)
    logp=F.log_softmax(logits,dim=-1)
    coef=torch.lgamma(total+1.)-torch.lgamma(counts+1.).sum(-1)
    return coef+(counts*logp).sum(-1)


def normal_log_prob(y,mu,logsd,valid):
    sd=torch.exp(logsd.clamp(-6.,4.))
    lp=-0.5*((y-mu)/sd)**2-logsd.clamp(-6.,4.)-0.5*math.log(2*math.pi)
    return (lp*valid).sum(-1),valid.sum(-1)


def zero_inflated_normal_log_prob(w,is_zero,params,valid):
    """Point mass at exact zero plus a Normal on the frozen log coordinate.

    A Gaussian truncated at zero after the fact would put the zero mass in the
    wrong place; the delay IQR really is exactly zero when all leads tie.
    """
    p0=torch.sigmoid(params[...,0]).clamp(1e-6,1-1e-6)
    mu=params[...,1];logsd=params[...,2].clamp(-6.,4.)
    pos=torch.log1p(-p0)-logsd-0.5*math.log(2*math.pi)-0.5*((w-mu)/torch.exp(logsd))**2
    lp=torch.where(is_zero>0,torch.log(p0),pos)
    return lp*valid


def load_log_prob(w,params,has_event):
    """Secondary total load on the frozen standardised log coordinate."""
    mu=params[...,0];logsd=params[...,1].clamp(-6.,4.)
    lp=-logsd-0.5*math.log(2*math.pi)-0.5*((w-mu)/torch.exp(logsd))**2
    return lp*has_event


class HistoryReference(nn.Module):
    """Identical rich encoder and readout, but fixed, normalized history kernels."""
    def __init__(self,prep,mode='marked_history',seed=20260906):
        super().__init__()
        self.mode=mode;self.rich=mode=='marked_history';self.fixed_rich=mode=='fixed_marked_history'
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+101)
            self.encoder=EventEncoder(prep.part.shape[1],prep.tokens.shape[-1],prep.groups.shape[-1],
                                      prep.event.shape[-1],prep.n_shaft) if self.rich else None
            self.base=1 if mode=='recent_rate' else prep.stats.shape[1]+(RICH_DIM if self.rich else prep.part.shape[1]+2*prep.tokens.shape[-1]+2*prep.event.shape[-1] if self.fixed_rich else 0)
            dim=self.base*5+3
            torch.manual_seed(seed+104)
            self.net=nn.Sequential(nn.Linear(dim,64),nn.GELU(),nn.Linear(64,LATENT)) if mode in (
                'recent_rate','marked_history','stats_history','fixed_marked_history') else None
            self.constant=nn.Parameter(torch.zeros(LATENT)) if mode=='constant_state' else None
            torch.manual_seed(seed+103)
            self.readout=Readout(prep.n_shaft,prep.band_ratio.shape[1],prep.xlag.shape[1])
            self.readout.use_clock=mode!='intercept'

    def latent(self,history,clock,age_hours):
        if self.net is not None:
            return self.net(torch.cat((history,clock,torch.log1p(age_hours).unsqueeze(-1)),dim=-1))
        if self.constant is not None:return self.constant.expand(len(clock),LATENT)
        return clock.new_zeros(len(clock),LATENT)


@torch.no_grad()
def initialize_readout(readout,prep):
    """FIT MLE offsets plus small nonzero residual weights (upstream gradients survive)."""
    import numpy as np
    mask=prep.split['train_packet']
    ep=np.empty(len(prep.payload['event_time']),np.int64)
    for i,(a,b) in enumerate(zip(prep.payload['packets']['event_lo'],prep.payload['packets']['event_hi'])):ep[a:b]=i
    ev=torch.as_tensor(mask[ep],device=prep.device)
    pm=torch.as_tensor(mask,device=prep.device)
    def moments(y,valid):
        v=valid[ev];yy=y[ev]
        n=v.sum(0).clamp(min=1)
        mu=(yy*v).sum(0)/n
        var=(((yy-mu)**2)*v).sum(0)/n
        return mu,torch.log(var.sqrt().clamp(min=.05))
    br,brs=moments(prep.band_ratio,prep.band_ratio_valid)
    xl,xls=moments(prep.xlag,prep.xlag_valid)
    iq=prep.iqr[ev];valid=prep.iqr_valid[ev]>0
    p0=((iq[valid]<=0).float().mean() if bool(valid.any()) else iq.new_tensor(.5)).clamp(.001,.999)
    pos=iq[valid & (iq>0)]
    coord=(torch.log(pos)-prep.log_iqr_center)/prep.log_iqr_scale
    iqmu=coord.mean() if len(coord) else iq.new_tensor(0.)
    iqsd=coord.std(unbiased=False).clamp(min=.05).log() if len(coord) else iq.new_tensor(0.)
    load=prep.total_load[pm & (prep.count>0)]
    lc=(torch.log(load.clamp(min=1e-9))-prep.log_load_center)/prep.log_load_scale
    lmu=lc.mean() if len(lc) else iq.new_tensor(0.)
    lsd=lc.std(unbiased=False).clamp(min=.05).log() if len(lc) else iq.new_tensor(0.)
    comp=(prep.shaft_count[pm].sum(0)+.5).log()
    off=torch.cat((iq.new_tensor([math.log(prep.scaling['base_rate_per_hour'])]),comp,br,brs,xl,xls,
                   torch.stack((torch.logit(p0),iqmu,iqsd,lmu,lsd))))
    readout.fit_offset.copy_(off)
    readout.log_nb_dispersion.fill_(-math.log(prep.scaling['nb_size']))
    final=readout.net if readout.linear else readout.net[-1]
    nn.init.normal_(final.weight,0.,1e-3);nn.init.zeros_(final.bias)
