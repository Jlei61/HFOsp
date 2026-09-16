"""Device-resident tensors for one subject and split."""
from __future__ import annotations
import numpy as np
import torch

from . import data as D

HISTORY_TAU_HOURS=(1/60.,5/60.,0.5,2.,8.)


class Prepared:
    def __init__(self,payload,split,scaling,device,dtype=torch.float32):
        self.payload=payload;self.split=split;self.scaling=scaling;self.device=device
        px,pt,n_shaft=D.packet_tables(payload,split)
        self.n_shaft=n_shaft
        c=scaling['packet_center'];s=scaling['packet_scale']
        self.stats=torch.as_tensor(np.clip((px-c)/s,-8,8),dtype=dtype,device=device)
        tok,grp,ev,evv=D.encode_events(payload,scaling)
        part=payload['participation']
        self.tokens=torch.as_tensor(tok,dtype=dtype,device=device)
        self.groups=torch.as_tensor(grp,dtype=dtype,device=device)
        self.part=torch.as_tensor(part.astype(np.float32),dtype=dtype,device=device)
        self.event=torch.as_tensor(ev,dtype=dtype,device=device)
        self.event_valid=torch.as_tensor(evv,dtype=dtype,device=device)
        sc=payload['targets']['shaft_count']
        size=sc.sum(axis=1,keepdims=True)
        self.shaft_frac=torch.as_tensor(np.divide(sc,np.maximum(size,1),where=size>0,
                                                  out=np.zeros_like(sc)),dtype=dtype,device=device)
        self.size=torch.as_tensor(np.log1p(size),dtype=dtype,device=device)
        gap=payload['event_features'][:,-1]
        self.gap=torch.as_tensor(np.where(np.isfinite(gap),gap,0.),dtype=dtype,device=device)
        pk=payload['packets']
        lo=pk['event_lo'];hi=pk['event_hi']
        seg=np.zeros(len(payload['event_time']),np.int64)
        for i,(a,b) in enumerate(zip(lo,hi)):seg[a:b]=i
        self.segment=torch.as_tensor(seg,dtype=torch.long,device=device)
        self.packet_start=pk['start'];self.packet_end=pk['end'];self.packet_release=pk['release']
        self.exposure_hours=torch.as_tensor(pt['exposure']/3600.,dtype=dtype,device=device)
        self.count=torch.as_tensor(pt['count'],dtype=dtype,device=device)
        self.shaft_count=torch.as_tensor(pt['shaft_count'],dtype=dtype,device=device)
        self.total_load=torch.as_tensor(pt['total_load'],dtype=dtype,device=device)
        self.valid=torch.as_tensor(pt['valid'].astype(np.float32),dtype=dtype,device=device)
        self.event_lo=torch.as_tensor(lo,dtype=torch.long,device=device)
        self.event_hi=torch.as_tensor(hi,dtype=torch.long,device=device)
        t=payload['targets']
        br=(t['band_ratio']-scaling['band_ratio_center'])/scaling['band_ratio_scale']
        xl=(t['signed_xlag']-scaling['xlag_center'])/scaling['xlag_scale']
        self.band_ratio=torch.as_tensor(np.nan_to_num(br),dtype=dtype,device=device)
        self.band_ratio_valid=torch.as_tensor(np.isfinite(br).astype(np.float32),dtype=dtype,device=device)
        self.xlag=torch.as_tensor(np.nan_to_num(xl),dtype=dtype,device=device)
        self.xlag_valid=torch.as_tensor(np.isfinite(xl).astype(np.float32),dtype=dtype,device=device)
        iqr=t['delay_iqr']
        self.iqr=torch.as_tensor(np.nan_to_num(iqr),dtype=dtype,device=device)
        self.iqr_valid=torch.as_tensor(np.isfinite(iqr).astype(np.float32),dtype=dtype,device=device)
        mid=0.5*(pk['start']+pk['end'])
        self.clock=torch.as_tensor(D.clock_features(mid),dtype=dtype,device=device)
        self.clock_end=torch.as_tensor(D.clock_features(pk['end']),dtype=dtype,device=device)
        self.log_load_center=scaling['log_load_center'];self.log_load_scale=scaling['log_load_scale']
        self.log_iqr_center=scaling['log_iqr_center'];self.log_iqr_scale=scaling['log_iqr_scale']
        self.train_packet=torch.as_tensor(split['train_packet'].astype(np.float32),dtype=dtype,device=device)
        excl=split['excluded_intervals']
        bad=np.zeros(len(pk['start']),bool)
        for a,b in np.asarray(excl,float).reshape(-1,2):
            bad|=(pk['end']>a)&(pk['start']<b)
        self.seizure_masked=torch.as_tensor(bad.astype(np.float32),dtype=dtype,device=device)
        self.seizure_masked_np=bad
        self.valid_np=pt['valid']
        held=np.zeros(len(pk['start']),bool)
        for t in np.asarray(split.get('inner_target_times',np.empty(0)),float):
            k=int(np.searchsorted(pk['end'],t-1e-6))
            if 0<=k<len(held):held[k]=True
        self.inner_target_held=held
        region=split.get('target_region')
        if region is None or not len(region):
            self.target_region=None;self.target_region_np=np.ones(len(pk['start']),bool)
        else:
            inside=np.zeros(len(pk['start']),bool)
            for a,b in np.asarray(region,float).reshape(-1,2):
                inside|=(pk['start']>=a-1e-6)&(pk['end']<=b+1e-6)
            self.target_region_np=inside
            self.target_region=torch.as_tensor(inside.astype(np.float32),dtype=dtype,device=device)
        ok=split['train_packet']&(~bad)&(~held)&(pt['valid'])
        self.target_ok=torch.as_tensor(ok.astype(np.float32),dtype=dtype,device=device)
        self.target_ok_np=ok
        self.n_packets=len(pk['start'])
        self.contiguous=self._contiguous()
        self.event_times=torch.as_tensor(payload['event_time'],dtype=torch.float64,device=device)
        self.clock_start=torch.as_tensor(D.clock_features(pk['start']),dtype=dtype,device=device)
        self.identity_target=torch.as_tensor(part.astype(np.float32),dtype=dtype,device=device)
        # Exposure by five-second interval, from observed support rather than a
        # fictitious uniform fraction. Count quadrature consumes these weights.
        left=pk['start'][:,None]+np.arange(12)[None,:]*5.
        seconds=np.zeros_like(left)
        for a,b in payload['observed_support']:
            seconds+=np.maximum(0.,np.minimum(left+5.,b)-np.maximum(left,a))
        seconds=np.clip(seconds,0.,5.)
        ok=pk['exposure']>1e-6
        if not np.allclose(seconds.sum(1)[ok],pk['exposure'][ok],atol=1e-3):
            raise ValueError('exposure grid disagrees with declared packet support')
        self.exposure_grid=torch.as_tensor(seconds,dtype=dtype,device=device)

    def _contiguous(self):
        """Index of the run each packet belongs to; runs never cross a recording gap."""
        gap=np.abs(self.packet_start[1:]-self.packet_end[:-1])>1e-6
        run=np.concatenate(([0],np.cumsum(gap)))
        return run

    def rich_summary(self,encoder,packet_index,previous_event_time=None):
        """Encode every event of the requested packets in one call."""
        lo=self.event_lo[packet_index];hi=self.event_hi[packet_index]
        counts=(hi-lo)
        total=int(counts.sum())
        if total==0:
            return torch.zeros(len(packet_index),encoder.gru.hidden_size,
                               device=self.device,dtype=self.stats.dtype)
        offs=torch.cumsum(counts,0)-counts
        pos=torch.arange(total,device=self.device)
        row=torch.searchsorted(offs+counts,pos,right=True)
        idx=lo[row]+(pos-offs[row])
        # Recompute gaps over the supplied readable stream. Hidden events cannot
        # survive indirectly as the predecessor in the next event's gap feature.
        times=self.event_times[idx]
        prev=torch.cat((times.new_tensor([float('nan') if previous_event_time is None else previous_event_time]),times[:-1]))
        gap_valid=torch.isfinite(prev)
        gap=torch.where(gap_valid,torch.log1p((times-prev).clamp(min=0)),torch.zeros_like(times)).to(self.stats.dtype)
        event=self.event[idx].clone();evv=self.event_valid[idx].clone()
        event[:,-1]=torch.where(gap_valid,((gap-float(self.scaling['event_center'][-1]))
                                  /float(self.scaling['event_scale'][-1])).clamp(-8,8),0.)
        evv[:,-1]=gap_valid.to(evv.dtype)
        return encoder(self.tokens[idx],self.groups[idx],self.part[idx],event,
                       evv,self.shaft_frac[idx],self.size[idx],gap,row,len(packet_index))

    def fixed_rich_summary(self,k):
        a,b=int(self.event_lo[k]),int(self.event_hi[k])
        dim=self.part.shape[1]+2*self.tokens.shape[-1]+2*self.event.shape[-1]
        if a==b:return self.stats.new_zeros(1,dim)
        part=self.part[a:b];tok=self.tokens[a:b];finite=torch.as_tensor(
            np.isfinite(self.payload['contact_tokens'][a:b]),device=self.device).to(tok.dtype)
        weight=finite*part.unsqueeze(-1)
        avg=(tok*weight).sum((0,1))/weight.sum((0,1)).clamp(min=1)
        valid=weight.sum((0,1))/part.sum().clamp(min=1)
        # The inter-event gap coordinate is excluded here: its raw stored value
        # may refer to an unreadable event. Count kernels already encode timing.
        ev=self.event[a:b].clone();evv=self.event_valid[a:b].clone();ev[:,-1]=0.;evv[:,-1]=0.
        e=(ev*evv).sum(0)/evv.sum(0).clamp(min=1)
        return torch.cat((part.mean(0),avg,valid,e,evv.mean(0))).reshape(1,-1)
