"""Morphology information world: rate and coarse composition are fixed by
construction, and a slow variable modulates only shape / propagation.

This is the discrimination instrument for the input and target contrasts. If
P_marks cannot beat P_stats here, a human null cannot be read as biology.
"""
from __future__ import annotations
import numpy as np

from .packets import CONTACT_FEATURE_NAMES,EVENT_FEATURE_NAMES,RATIO_BANDS,XLAG_PAIRS

BLOCK_SECONDS=3600.
PACKET_SECONDS=60.


def make_world(seed=0,n_blocks=90,n_contacts=6,rate_per_minute=8.0,slow_tau_hours=2.0,
               morph_gain=1.2,identity_gain=0.0,zero_effect=False,t0=1.0e9):
    """Constant rate and constant coarse composition; only shape follows the slow state."""
    rng=np.random.default_rng(seed)
    n_token=len(CONTACT_FEATURE_NAMES);n_event=len(EVENT_FEATURE_NAMES)+1
    shafts=['A']*(n_contacts//2)+['B']*(n_contacts-n_contacts//2)
    unique=sorted(set(shafts));shaft_index=np.array([unique.index(s) for s in shafts])
    contacts=[f'{s}{i+1}' for i,s in enumerate(shafts)]
    blocks=[];times=[];rel=[];blk=[]
    dt=PACKET_SECONDS/3600.
    a=np.exp(-dt/slow_tau_hours)
    s=0.
    slow=[]
    for b in range(n_blocks):
        start=t0+b*BLOCK_SECONDS
        blocks.append(dict(block=b,start=start,end=start+BLOCK_SECONDS,release=start+BLOCK_SECONDS,n_events=0))
        for k in range(int(BLOCK_SECONDS//PACKET_SECONDS)):
            s=a*s+np.sqrt(1-a*a)*rng.normal()
            slow.append(s)
            n=rng.poisson(rate_per_minute)
            if n:
                t=np.sort(rng.uniform(0,PACKET_SECONDS,n))+start+k*PACKET_SECONDS
                times.append(t);rel.append(np.full(n,start+BLOCK_SECONDS));blk.append(np.full(n,b))
    slow=np.asarray(slow)
    times=np.concatenate(times);rel=np.concatenate(rel);blk=np.concatenate(blk).astype(np.int32)
    N=len(times)
    packet_of=((times-t0)//PACKET_SECONDS).astype(int)
    sv=slow[packet_of]
    if zero_effect:sv=rng.permutation(sv)
    # Fixed participation law: coarse composition cannot carry the slow state.
    base=np.array([0.55,0.5,0.45]*4)[:n_contacts]
    logits=np.log(base/(1-base))[None,:]+identity_gain*sv[:,None]*np.linspace(-1,1,n_contacts)[None,:]
    part=rng.uniform(size=(N,n_contacts))<1/(1+np.exp(-logits))
    empty=~part.any(1)
    part[empty,rng.integers(0,n_contacts,empty.sum())]=True
    # Morphology carries the slow state; per-contact tokens expose it.
    band_ratio=np.zeros((N,len(RATIO_BANDS)),np.float32)
    band_ratio[:,0]=morph_gain*sv+0.4*rng.normal(size=N)
    band_ratio[:,1:]=0.4*rng.normal(size=(N,len(RATIO_BANDS)-1))
    xlag=np.zeros((N,len(XLAG_PAIRS)),np.float32)
    xlag[:,0]=0.8*morph_gain*sv+0.4*rng.normal(size=N)
    xlag[:,1:]=0.4*rng.normal(size=(N,len(XLAG_PAIRS)-1))
    iqr=np.exp(-3.5+0.5*morph_gain*sv+0.3*rng.normal(size=N)).astype(np.float32)
    tok=np.full((N,n_contacts,n_token),np.nan,np.float32)
    lead=rng.normal(size=(N,n_contacts))*0.01
    tok[...,0]=lead
    tok[...,1]=rng.uniform(size=(N,n_contacts))
    tok[...,2]=(lead==lead.min(1,keepdims=True)).astype(np.float32)
    tok[...,3:8]=(band_ratio[:,None,:1]+0.2*rng.normal(size=(N,n_contacts,5))).astype(np.float32)
    tok[...,8:13]=0.2*rng.normal(size=(N,n_contacts,5))
    tok[...,13:18]=0.2*rng.normal(size=(N,n_contacts,5))
    tok[...,18:28]=(xlag[:,None,:1]+0.2*rng.normal(size=(N,n_contacts,10))).astype(np.float32)
    tok[...,28:31]=(np.log(iqr)[:,None,None]+0.2*rng.normal(size=(N,n_contacts,3))).astype(np.float32)
    tok[~part]=np.nan
    size=part.sum(1).astype(np.float32)
    ev=np.zeros((N,n_event),np.float32)
    ev[:,0]=np.log1p(size);ev[:,1]=part.mean(1);ev[:,2]=0.15;ev[:,3]=1.
    ev[:,4]=iqr;ev[:,5]=lead.max(1)-lead.min(1)
    ev[:,6]=0.2*rng.normal(size=N);ev[:,7]=0.2*rng.normal(size=N)
    ev[:,8]=np.log1p(np.r_[0.,np.diff(times)])
    shaft_count=np.stack([part[:,shaft_index==k].sum(1) for k in range(len(unique))],-1).astype(np.float32)
    edges=np.arange(t0,t0+n_blocks*BLOCK_SECONDS+PACKET_SECONDS,PACKET_SECONDS)
    lo=np.searchsorted(times,edges[:-1],'left');hi=np.searchsorted(times,edges[1:],'left')
    prel=np.array([blocks[min(int((e-t0)//BLOCK_SECONDS),n_blocks-1)]['release'] for e in edges[:-1]])
    prel=np.maximum(prel,edges[1:])
    support=np.array([[t0,t0+n_blocks*BLOCK_SECONDS]])
    return dict(subject=f'synthetic_morphology_seed{seed}',selected_contacts=contacts,shafts=unique,
                shaft_index=shaft_index,n_contacts=n_contacts,bands=list(RATIO_BANDS)+['ied_low'],
                band_edges_hz={},cross_band_pairs=[list(p) for p in XLAG_PAIRS],
                contact_feature_names=list(CONTACT_FEATURE_NAMES),
                event_feature_names=list(EVENT_FEATURE_NAMES)+['log1p_inter_event_gap_s'],
                ratio_bands=list(RATIO_BANDS),xlag_pairs=[list(p) for p in XLAG_PAIRS],
                event_time=times,event_end=times+0.15,event_block=blk,event_release=rel,
                contact_tokens=tok,event_features=ev,participation=part,
                targets=dict(band_ratio=band_ratio,signed_xlag=xlag,delay_iqr=iqr,size=size,
                             shaft_count=shaft_count),
                blocks=blocks,block_context=np.zeros((n_blocks,1)),
                packets=dict(start=edges[:-1],end=edges[1:],release=prel,
                             block=((edges[:-1]-t0)//BLOCK_SECONDS).astype(np.int32),
                             event_lo=lo.astype(np.int64),event_hi=hi.astype(np.int64),
                             exposure=np.full(len(lo),PACKET_SECONDS),
                             ambiguous_intervals=np.empty((0,2))),
                observed_support=support,ambiguous_intervals=np.empty((0,2)),
                phase_boundaries={'20pct':t0+0.2*n_blocks*BLOCK_SECONDS,'60pct':t0+0.6*n_blocks*BLOCK_SECONDS,
                                  '70pct':t0+0.7*n_blocks*BLOCK_SECONDS,'80pct':t0+n_blocks*BLOCK_SECONDS},
                clock_restorations=[],background_repairs=[],source_cards=[],
                coarse_community_rule='synthetic two-group implant',
                slow_state=slow,slow_tau_hours=slow_tau_hours,morph_gain=morph_gain,
                identity_gain=identity_gain,zero_effect=bool(zero_effect),
                release_contract='synthetic world uses the same closed-block publication rule')
