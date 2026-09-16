"""Low-capacity probes of a frozen observer and frozen contact decoder."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn


def exact_subset_nll(logits,target,available):
    """Exact unordered next-set likelihood, conditional on observed set size.

    The sample space contains every size-k subset of still available contacts.
    This differs from averaging singleton softmax probabilities when k > 1.
    Small patient vocabularies make complete enumeration practical.
    """
    c=logits.shape[-1]
    if c>12:raise ValueError('Exact subset probe is registered for <=12 contacts')
    masks=((torch.arange(2**c,device=logits.device)[:,None]>>torch.arange(c,device=logits.device))&1).to(logits.dtype)
    k=target.sum(-1);sizes=masks.sum(-1)
    valid=(sizes[None]==k[:,None])&((~available).to(logits.dtype)@masks.T==0)
    scores=logits@masks.T
    logz=torch.logsumexp(scores.masked_fill(~valid,-torch.inf),dim=-1)
    return logz-(logits*target).sum(-1)


def first_unknown_group(ranks):
    ranks=np.asarray(ranks);target=(ranks==2).astype(np.float32);available=(ranks<0)|(ranks>=2)
    stop=~np.any(ranks>=2,axis=1);keys=[];suffix=[]
    for r,y in zip(ranks,target):
        prefix=tuple(tuple(np.flatnonzero(r==group).tolist()) for group in (0,1))
        keys.append(repr((prefix,int(y.sum()))));suffix.append(tuple(np.flatnonzero(y).tolist()))
    return target,available,stop,np.array(keys),suffix


def branching_strata(ranks,phases,minimum_fit_events=5):
    target,available,stop,keys,suffix=first_unknown_group(ranks)
    eligible=set()
    for key in np.unique(keys[np.asarray(phases)=='FIT']):
        rows=np.flatnonzero((keys==key)&(np.asarray(phases)=='FIT')&~stop)
        if len(rows)>=minimum_fit_events and len({suffix[i] for i in rows})>=2:eligible.add(key)
    mask=np.array([key in eligible for key in keys])&~stop
    return mask,sorted(eligible),keys


class LogitAdapter(nn.Module):
    """Eight-unit residual readout; zero output preserves its fitted parent."""
    def __init__(self,context_dim,n_contacts,rank=8):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(context_dim,rank),nn.GELU(),nn.Linear(rank,n_contacts+1))
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)

    def forward(self,x):return self.layers(x)


def standardise_anchor_features(values,fit_rows):
    x=np.asarray(values,float);center=x[fit_rows].mean(0);scale=x[fit_rows].std(0)
    scale=np.where(scale>1e-6,scale,1.)
    return np.clip((x-center)/scale,-8,8).astype(np.float32),dict(center=center,scale=scale)


def equal_anchor_mean(values,anchor):
    """One event once; average events inside each anchor, then anchors."""
    _,index=np.unique(anchor,return_inverse=True);count=np.bincount(index)
    sums=np.zeros((len(count),)+np.shape(values)[1:]);np.add.at(sums,index,values)
    return sums/count.reshape((-1,)+(1,)*(sums.ndim-1))
