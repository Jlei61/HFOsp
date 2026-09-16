"""Single availability-aware query and prediction engine used by every consumer."""
from __future__ import annotations
from dataclasses import dataclass,replace
import math
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from . import data as D
from .model import StateModel,HistoryReference,initialize_readout,RICH_DIM
from .numerics import LATENT,propagate_moments,propagate_samples,evidence_update,cholesky_psd
from .prepare import HISTORY_TAU_HOURS
from .objective import view_log_probs,VIEWS,SECONDARY


@dataclass
class QueryState:
    patient:str
    split_id:str
    transform_id:str
    query_packet:np.ndarray
    query_time:np.ndarray
    source_time:np.ndarray
    release_time:np.ndarray
    information_age_minutes:np.ndarray
    available_exposure_seconds:np.ndarray
    readable_events:np.ndarray
    prefix_start:np.ndarray
    input_digest:tuple
    m:torch.Tensor|None
    P:torch.Tensor|None
    history:torch.Tensor|None
    producer_hash:str='in_memory'
    donor_query_time:np.ndarray|None=None

    def subset(self,indices):
        ix=np.asarray(indices,int);tx=torch.as_tensor(ix,device=(self.m if self.m is not None else self.history).device)
        return replace(self,**{k:getattr(self,k)[ix] for k in (
            'query_packet','query_time','source_time','release_time','information_age_minutes',
            'available_exposure_seconds','readable_events','prefix_start')},
            input_digest=tuple(self.input_digest[i] for i in ix),
            m=None if self.m is None else self.m[tx],P=None if self.P is None else self.P[tx],
            history=None if self.history is None else self.history[tx],donor_query_time=None if self.donor_query_time is None else self.donor_query_time[ix])

    def metadata(self):
        out={k:(v.tolist() if isinstance(v,np.ndarray) else list(v) if isinstance(v,tuple) else v)
                for k,v in vars(self).items() if k not in ('m','P','history')}
        observed=np.isfinite(self.release_time)
        out['has_observation']=observed.tolist()
        out['last_observed_source_time']=[float(t) if ok else None for t,ok in zip(self.source_time,observed)]
        out['observed_information_age_minutes']=[float(t) if ok else None for t,ok in zip(self.information_age_minutes,observed)]
        out['source_semantics']='source_time is prior origin when has_observation=false; this is not a last observation'
        return out


def build_model(prep,inputs='P_marks',family='I-L-G1',arm='state',seed=20260906):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+103)
        if arm=='state':
            model=StateModel(prep.part.shape[1],prep.tokens.shape[2],prep.groups.shape[2],
                prep.event.shape[1],prep.n_shaft,prep.stats.shape[1],prep.band_ratio.shape[1],prep.xlag.shape[1],
                rich=inputs=='P_marks',coupled=family.startswith('C'),nonlinear=family.split('-')[1]=='N',
                linear_readout=family.endswith('G0'),seed=seed).to(prep.device)
        else:model=HistoryReference(prep,arm,seed).to(prep.device)
        torch.manual_seed(seed+105)
        initialize_readout(model.readout,prep)
    return model


def apply_ablations(prep,crossview=False,shuffle_marks=False):
    if shuffle_marks:
        raise ValueError('global mark shuffle is invalid; use the registered empty-channel capacity control')
    if crossview:
        for a,b in ((0,3),(8,13),(18,28)):prep.tokens[:,:,a:b]=0.
        prep.groups[:,:,0]=0.;prep.groups[:,:,2]=0.
        prep.event[:,4:6]=0.;prep.event_valid[:,4:6]=0.
        prep.xlag_valid[:]=0.;prep.iqr_valid[:]=0.
    return 'no_timing_view' if crossview else 'full_input'


def infer_asof(model,prep,query_packets,role='outer',history_hours=None,training=False,
               grad_hours=2.,activation_checkpoint=True,producer_hash='in_memory',grad_start_by_episode=None,_cache=True):
    """Replay each shared episode once, then propagate its last readable posterior to t.

    Cached transitions are scoped to one forward call and one grad context;
    no parameter update can reuse a stale or detached transition matrix.
    """
    qs=np.asarray(query_packets,int)
    if _cache and not training and hasattr(prep,'frozen_query_cache'):
        if any(p.requires_grad for p in model.parameters()):raise ValueError('query caching requires a frozen producer')
        cache=prep.frozen_query_cache
        keys=[(producer_hash,prep.split['split_id'],prep.scaling['transform_id'],role,history_hours,int(q)) for q in qs]
        missing=[i for i,k in enumerate(keys) if k not in cache]
        if missing:
            fresh=infer_asof(model,prep,qs[missing],role,history_hours,producer_hash=producer_hash,_cache=False)
            for j,i in enumerate(missing):cache[keys[i]]=fresh.subset([j])
        states=[cache[k] for k in keys];first=states[0]
        return replace(first,**{k:np.concatenate([getattr(s,k) for s in states]) for k in (
            'query_packet','query_time','source_time','release_time','information_age_minutes',
            'available_exposure_seconds','readable_events','prefix_start')},
            input_digest=tuple(s.input_digest[0] for s in states),
            m=None if first.m is None else torch.cat([s.m for s in states]),
            P=None if first.P is None else torch.cat([s.P for s in states]),
            history=None if first.history is None else torch.cat([s.history for s in states]))
    if not len(qs):raise ValueError('empty query batch')
    if (qs<0).any() or (qs>=prep.n_packets).any():raise IndexError('query outside packet grid')
    if prep.split['seizure_mask'][qs].any():raise ValueError('query inside ictal/postictal exclusion')
    allowed=D.input_mask(prep.split,role)
    pk=prep.payload['packets'];ends=prep.packet_end;release=prep.packet_release
    starts=prep.split['episode_start'][qs].copy()
    if history_hours is not None:
        if history_hours<=0:raise ValueError('strict history must be positive')
        starts=np.maximum(starts,np.searchsorted(pk['start'],ends[qs]-history_hours*3600-1e-6))
    readable=[];groups={}
    finite_release=release[allowed & np.isfinite(release)]
    ordered=bool(np.all(finite_release[1:]>=finite_release[:-1]))
    for j,(start,q) in enumerate(zip(starts,qs)):
        ix=np.arange(start,q+1)
        ix=ix[allowed[ix] & (release[ix]<=ends[q])]
        readable.append(ix)
        # Nonmonotone release streams need separate filtered histories, not searchsorted.
        key=(int(start),None if ordered else D.digest(ix))
        groups.setdefault(key,[]).append(j)
    ms=[None]*len(qs);Ps=ms.copy();hs=ms.copy()
    source=np.empty(len(qs));rel=np.full(len(qs),np.nan);expo=np.zeros(len(qs));ne=np.zeros(len(qs),int)
    input_ids=[]
    for ix,q,start in zip(readable,qs,starts):
        input_ids.append(D.digest(dict(packets=ix,role=role,start=int(start),transform=prep.scaling['transform_id'])))
    caches={False:{},True:{}}
    for (start,_),members in groups.items():
        longest=max(members,key=lambda j:len(readable[j]))
        stream=readable[longest]
        snapshots={};is_state=isinstance(model,StateModel)
        if is_state:m,P,c=model.initial(1,prep.device,prep.stats.dtype)
        else:
            ema=prep.stats.new_zeros(5,model.base);mass=prep.stats.new_zeros(5,1)
        prev_time=float(pk['start'][start]) if start<prep.n_packets else float(ends[qs[members[0]]])
        grad_start=int(min(qs[members])-round(grad_hours*60)) if history_hours is None else start
        if history_hours is None and grad_start_by_episode is not None:
            grad_start=int(grad_start_by_episode[start])
        # Encoding blocks are shared across all query prefixes. Recompute activations,
        # while retaining the complete registered history and event order.
        rich_map={};prev_event=None
        if model.encoder is not None:
            partitions=[stream[stream<grad_start],stream[stream>=grad_start]]
            for ids in partitions:
                for a in range(0,len(ids),60):
                    batch=ids[a:a+60];on=training and int(batch[0])>=grad_start
                    ti=torch.as_tensor(batch,device=prep.device)
                    previous=prev_event
                    with torch.set_grad_enabled(on):
                        fn=lambda idx,previous=previous:prep.rich_summary(model.encoder,idx,previous)
                        rich=checkpoint(fn,ti,use_reentrant=False,preserve_rng_state=True) if on and activation_checkpoint else fn(ti)
                    for ii,k in enumerate(batch):rich_map[int(k)]=rich[ii:ii+1]
                    last_events=[k for k in batch if pk['event_hi'][k]>pk['event_lo'][k]]
                    if last_events:prev_event=float(prep.payload['event_time'][pk['event_hi'][last_events[-1]]-1])
        wanted={int(readable[j][-1]) for j in members if len(readable[j])}
        for k in stream:
            on=training and int(k)>=grad_start
            with torch.set_grad_enabled(on):
                dt=(float(ends[k])-prev_time)/3600.
                x=prep.stats[k:k+1]
                rich=rich_map.get(int(k),x.new_zeros(1,RICH_DIM))
                if is_state:
                    m,P=propagate_moments(model.dynamics,m,P,dt,cache=caches[on])
                    c=model.slow.gru(model.slow.packet_input(x,rich),c)
                    a,R,_=model.slow.evidence(c,m,P)
                    m,P,_,_=evidence_update(m.double(),P.double(),a.double(),R.double())
                    m=m.to(x.dtype);P=P.to(x.dtype)
                    if int(k) in wanted:snapshots[int(k)]=(m,P)
                else:
                    xx=x[:,:1] if model.mode=='recent_rate' else torch.cat((x,rich),-1) if model.rich else torch.cat((x,prep.fixed_rich_summary(k)),-1) if model.fixed_rich else x
                    # Decay across missingness, add only the observed interval's weight.
                    tau=x.new_tensor(HISTORY_TAU_HOURS).reshape(-1,1)
                    decay=torch.exp(-dt/tau);w=1-torch.exp(-(float(pk['exposure'][k])/3600.)/tau)
                    ema=ema*decay+w*xx;mass=mass*decay+w
                    if int(k) in wanted:snapshots[int(k)]=(ema/mass.clamp(min=1e-12)).reshape(1,-1)
            prev_time=float(ends[k])
        for j in members:
            ix=readable[j];q=qs[j]
            source[j]=float(ends[ix[-1]]) if len(ix) else float(pk['start'][start])
            if len(ix):rel[j]=float(np.max(release[ix]))
            expo[j]=float(pk['exposure'][ix].sum());ne[j]=int((pk['event_hi'][ix]-pk['event_lo'][ix]).sum())
            dt=max(0.,float(ends[q]-source[j]))/3600.
            if is_state:
                if len(ix):m1,P1=snapshots[int(ix[-1])]
                else:m1,P1,_=model.initial(1,prep.device,prep.stats.dtype)
                with torch.set_grad_enabled(training):
                    m1,P1=propagate_moments(model.dynamics,m1,P1,dt,cache=caches[training])
                ms[j]=m1;Ps[j]=P1
            else:
                hs[j]=snapshots[int(ix[-1])] if len(ix) else prep.stats.new_zeros(1,5*model.base)
    return QueryState(prep.split['subject'],prep.split['split_id'],prep.scaling['transform_id'],
        qs,ends[qs],source,rel,(ends[qs]-source)/60.,expo,ne,starts,tuple(input_ids),
        torch.cat(ms) if ms[0] is not None else None,torch.cat(Ps) if Ps[0] is not None else None,
        torch.cat(hs) if hs[0] is not None else None,producer_hash)


def sample_posterior(m,P,paths,generator=None,noise=None):
    L,_=cholesky_psd(P.double(),'query covariance')
    e=noise if noise is not None else torch.randn(paths,*m.shape,device=m.device,dtype=m.dtype,generator=generator)
    return m.unsqueeze(0)+torch.einsum('bij,sbj->sbi',L.to(m.dtype),e)


def relax_moments(state,reference,hours):
    a=math.exp(-hours/reference['tau_hours'])
    m0=reference['m0'].to(state.m);P0=reference['P0'].to(state.P)
    return m0+a*(state.m-m0),a*a*state.P+(1-a*a)*P0


@dataclass
class IntervalPrediction:
    z_start:torch.Tensor
    z_end:torch.Tensor
    grid:torch.Tensor


def query_noise(state,seed,steps,paths,device,dtype):
    """Physical-query keyed noise: microbatching cannot change a sample's paths."""
    arrays=[]
    for t in state.query_time:
        key=int(D.digest((int(seed),float(t)))[:16],16)
        arrays.append(np.random.default_rng(key).standard_normal((steps,paths,LATENT),dtype=np.float32))
    return torch.as_tensor(np.stack(arrays,axis=2),device=device,dtype=dtype)


def predict(model,prep,state,horizons=(1,5,30,120),paths=64,seed=20260906,rule='EVOLVE',reference=None):
    """Every rule starts at query t; observation time and masks are never donor-specific."""
    if rule not in ('HOLD','EVOLVE','RELAX','RESET'):raise ValueError(rule)
    if state.m is not None:
        if rule in ('RELAX','RESET') and reference is None:raise ValueError('FIT reference distribution required')
        m,P=(reference['m0'].to(state.m).expand_as(state.m),reference['P0'].to(state.P).expand_as(state.P)) if rule=='RESET' else (state.m,state.P)
        refined=set(horizons)|{1,5,30,120}
        steps=1+2*sum(12 if h in refined else 1 for h in range(1,max(horizons)+1))
        noise=query_noise(state,seed,steps,paths,prep.device,state.m.dtype);ni=1
        z=sample_posterior(m,P,paths,noise=noise[0]);cache={};pred={}
        if rule=='RELAX':
            L0,_=cholesky_psd(reference['P0'].double().unsqueeze(0),'RELAX stationary covariance')
            L0=L0[0].to(z.dtype)
        for h in range(1,max(horizons)+1):
            zp=z;grid=[z];n=12 if h in refined else 1
            for _ in range(n):
                dt=1/(60.*n)
                if rule in ('EVOLVE','RESET'):
                    z=propagate_samples(model.dynamics,z,dt,cache=cache,noise=[noise[ni],noise[ni+1]])
                elif rule=='RELAX':
                    a=math.exp(-dt/reference['tau_hours']);m0=reference['m0'].to(z)
                    z=m0+a*(z-m0)+math.sqrt(1-a*a)*(noise[ni]@L0.T)
                ni+=2;grid.append(z)
            if h in horizons:pred[h]=IntervalPrediction(zp,z,torch.stack(grid))
    else:
        if rule!='EVOLVE':raise ValueError('state rules are not defined for a history reference')
        pred={}
        for h in horizons:
            ix=torch.as_tensor(np.minimum(state.query_packet+h,prep.n_packets-1),device=prep.device)
            clock=prep.clock[ix];age=clock.new_tensor(state.information_age_minutes/60.+h/60.)
            z=model.latent(state.history,clock,age).unsqueeze(0)
            pred[h]=IntervalPrediction(z,z,z.unsqueeze(0).expand(13,*z.shape))
    return pred


def score_predictions(model,prep,state,predictions,role,target_ids=None):
    """Same distribution, same physical targets; valid denominators carried in every row."""
    rows=[]
    for h,pred in predictions.items():
        zs,ze=pred.z_start,pred.z_end
        targets=state.query_packet+h
        keep=targets<prep.n_packets
        if target_ids is not None:keep &= np.isin(targets,target_ids)
        ids=np.flatnonzero(keep)
        if not len(ids):continue
        tgt=targets[ids]
        allowed=D.role_mask(prep.split,role)[tgt].copy()
        for j,(q,t) in enumerate(zip(state.query_packet[ids],tgt)):
            allowed[j] &= not prep.split['seizure_mask'][q+1:t+1].any()
        ti=torch.as_tensor(tgt,device=prep.device);ii=torch.as_tensor(ids,device=prep.device)
        lp,un=view_log_probs(model.readout,zs[:,ii],ze[:,ii],prep,ti,state_grid=pred.grid[:,:,ii])
        mask=prep.stats.new_tensor(allowed)
        for key in un:un[key]=un[key]*mask
        rows.append(dict(horizon=h,packet=tgt,query_packet=state.query_packet[ids],
                         information_age_minutes=state.information_age_minutes[ids],
                         logp=lp,units=un))
    return rows


def wrong_time_control(state,reference):
    """FIT donor value at recipient t. Recipient time, clock and exposure stay fixed."""
    keep=[];donors=[]
    for i,t in enumerate(state.query_time):
        dt=reference['donor_time']
        clock=np.abs((dt-t+43200)%86400-43200)
        legal=(dt<t-4*3600)&(clock<=7200)&(np.abs(reference['donor_age']-state.information_age_minutes[i])<=10)
        ix=np.flatnonzero(legal)
        if not len(ix):continue
        j=int(ix[np.argmin(np.abs(dt[ix]-t))]);keep.append(i);donors.append(j)
    if not keep:return None
    shifted=state.subset(keep)
    return replace(shifted,m=reference['donor_m'][donors].to(state.m),P=reference['donor_P'][donors].to(state.P),
        donor_query_time=reference['donor_time'][donors],input_digest=tuple(D.digest((d,'WRONGTIME',float(t))) for d,t in zip(shifted.input_digest,reference['donor_time'][donors])))
