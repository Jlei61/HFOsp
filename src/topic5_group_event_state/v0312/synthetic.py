"""Measurement-consistent morphology world; generated targets never bypass producer."""
import itertools
import numpy as np
from ..v0311.synthetic import make_world as skeleton
from ..v0311.packets import block_event_tables,RATIO_BANDS


def make_world(seed=0,n_blocks=24,morph_gain=1.2,identity_gain=0.,zero_effect=False,release_mode='delayed'):
    p=skeleton(seed,n_blocks=n_blocks,n_contacts=8,morph_gain=morph_gain,identity_gain=0.,zero_effect=False)
    rng=np.random.default_rng(seed+101);N=len(p['event_time']);C=p['n_contacts']
    packet=((p['event_time']-p['observed_support'][0,0])//60).astype(int)
    signal=p['slow_state'][packet].copy()
    if zero_effect:signal=rng.normal(size=N)
    bands=['ied_low']+list(RATIO_BANDS);pairs=list(itertools.combinations(bands,2))
    # Two contacts from each community: count and coarse composition cannot encode s.
    part=np.zeros((N,C),bool)
    for i,s in enumerate(signal):
        for cols in (np.arange(4),np.arange(4,8)):
            weights=np.exp(np.clip(identity_gain*s*np.array([-1.,-.3,.3,1.]),-5,5));weights/=weights.sum()
            part[i,rng.choice(cols,2,replace=False,p=weights)]=True
    delay=rng.normal(size=(N,C))*.01*np.exp(.3*morph_gain*signal[:,None])
    energy=rng.normal(size=(N,C,5))*.2
    energy[:,:,1]+=morph_gain*signal[:,None]  # Only gamma relative to low IED changes.
    xlag=rng.normal(size=(N,C,10))*.005
    xlag[:,:,pairs.index(('ied_low','ripple'))]+=.02*morph_gain*signal[:,None]
    bf=np.zeros((N,C,5,5),np.float32);bf[:,:,:,2]=energy;bf[:,:,:,0]=0.;bf[:,:,:,1]=0.;bf[:,:,:,4]=0.
    wave=rng.normal(size=(N,C,16)).astype(np.float32)
    raw=dict(participation=part,relative_delay_s=delay,tied_group_id=np.where(delay==np.where(part,delay,np.inf).min(1,keepdims=True),0,1),
        band_features=bf,cross_band_lag_s=xlag,has_waveform=np.ones(N,bool),core_seconds_raw=np.full(N,.15,np.float32),
        waveform_detector=wave,waveform_bipolar=wave,waveform_shaft_car=wave)
    manifest=dict(bands=bands,cross_band_pairs=pairs)
    tokens,event,target,part=block_event_tables(raw,manifest,C)
    event=np.column_stack((event,np.log1p(np.r_[0,np.diff(p['event_time'])]))).astype(np.float32)
    target['shaft_count']=np.column_stack((part[:,:4].sum(1),part[:,4:].sum(1))).astype(np.float32)
    p.update(contact_tokens=tokens,event_features=event,targets=target,participation=part,bands=bands,cross_band_pairs=pairs,
        synthetic_contract='v0312 raw block_event_tables; fixed count/coarse composition; no target-only latent channel',
        raw_measurement=raw,raw_manifest=manifest,zero_effect=zero_effect,identity_gain=identity_gain)
    if release_mode=='local':
        p['packets']['release']=p['packets']['end'].copy();p['event_release']=p['event_end'].copy()
    elif release_mode!='delayed':raise ValueError(release_mode)
    p['measurement_mode']=release_mode
    return p
