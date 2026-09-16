"""FIT-normalised, delayed-availability human tokens for the design pilot.

All event marks in a block become available together at its closed-block
release time. Additive simultaneous writes are pooled exactly, without
inventing intermediate observations. History length refers to real event
times; available-time delay is stored separately.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def merge_intervals(intervals):
    result=[]
    for lo,hi in sorted(intervals):
        if hi<=lo:continue
        if result and lo<=result[-1][1]+1e-6:result[-1][1]=max(result[-1][1],hi)
        else:result.append([float(lo),float(hi)])
    return np.asarray(result,float).reshape(-1,2)


def exposure(support,lo,hi):
    return float(np.maximum(0,np.minimum(support[:,1],hi)-np.maximum(support[:,0],lo)).sum())


def subtract_intervals(support,excluded):
    out=[]
    for lo,hi in support:
        cursor=lo
        for left,right in excluded:
            if right<=cursor or left>=hi:continue
            if left>cursor:out.append([cursor,min(left,hi)])
            cursor=max(cursor,right)
            if cursor>=hi:break
        if cursor<hi:out.append([cursor,hi])
    return merge_intervals(out)


def contained_events(times,ends,support):
    good=np.zeros(len(times),bool)
    for lo,hi in support:good|=(times>=lo)&(ends<=hi)
    return good


def phase(times,bounds):
    out=np.full(np.shape(times),'CLOSED',dtype='<U12')
    for lo,hi,name in ((-np.inf,bounds['20pct'],'CALIBRATION'),(bounds['20pct'],bounds['60pct'],'FIT'),
                       (bounds['60pct'],bounds['70pct'],'INNER'),(bounds['70pct'],bounds['80pct'],'SELECTION')):
        out[(times>=lo)&(times<hi)]=name
    return out


def _participant_mean(values,part):
    mask=part.reshape(part.shape+(1,)*(values.ndim-2)) & np.isfinite(values)
    count=mask.sum(axis=1)
    return np.divide(np.where(mask,values,0).sum(axis=1),count,out=np.full(count.shape,np.nan,dtype=float),where=count>0)


def block_features(z,views):
    part=z['participation'].astype(bool);n,c=part.shape
    parts=[part.astype(float)];names=[f'participation_{i}' for i in range(c)]
    delay=z['relative_delay_s'].astype(float)
    parts += [np.where(part,delay,np.nan)];names += [f'delay_{i}' for i in range(c)]
    group=z['tied_group_id']; first=(group==0)&part
    # Undefined centroids do not establish that a contact was recruited late.
    first=np.where(part & ~np.isfinite(delay),np.nan,first.astype(float))
    parts += [first];names += [f'first_group_{i}' for i in range(c)]
    for column,label in ((1,'band_centroid'),(2,'band_log_energy'),(4,'band_log_peak')):
        values=_participant_mean(z['band_features'][...,column],part)
        parts.append(values);names += [f'{label}_{i}' for i in range(values.shape[1])]
    cross=_participant_mean(z['cross_band_lag_s'],part);parts.append(cross);names += [f'cross_band_lag_{i}' for i in range(cross.shape[1])]
    for view in views:
        w=z['waveform_'+view].astype(np.float32)
        statistics=np.stack((np.sqrt(np.mean(w*w,axis=-1)),np.max(np.abs(w),axis=-1),np.mean(np.abs(np.diff(w,axis=-1)),axis=-1)),axis=-1)
        values=_participant_mean(np.log1p(statistics),part)
        values[~z['has_waveform'].astype(bool)]=np.nan
        parts.append(values);names += [view+'_'+v for v in ('log_rms','log_peak','log_line_length')]
    a,b=np.triu_indices(c,k=1);parts.append((part[:,a]&part[:,b]).astype(float));names += [f'coupled_{i}_{j}' for i,j in zip(a,b)]
    marks=np.concatenate(parts,axis=-1)
    return marks,names


def background_core_mask(times,packed_cores,window_seconds=2.):
    """A missing fine mark never makes a known event core into background."""
    times=np.asarray(times,float);cores=np.asarray(packed_cores,float).reshape(-1,2)
    if not len(cores):return np.ones(len(times),bool)
    return ~np.any((times[:,None]<cores[None,:,1])&(times[:,None]+window_seconds>cores[None,:,0]),axis=1)


def background_summary(bg):
    bg=np.asarray(bg,float)
    if not len(bg):return None
    spatial=np.nanmean(bg,axis=1)
    return np.concatenate((spatial[-1],np.nanmean(spatial,axis=0),np.nanstd(spatial,axis=0)))


def restore_clock_only_events(card,times,ends,marks,participation,names):
    """Preserve known detections when only stitched centroid measurement fails.

    A 200-second processing boundary is not a recording gap. The packed
    window and per-contact detector overlaps remain observed. Fine marks are
    explicitly missing; no centroid, order, band or waveform is imputed.
    """
    excluded=np.asarray(card['segment_crossing_exclusions'],float).reshape(-1,2)
    if not len(excluded):return times,ends,marks,participation
    paths=[p for p in card['input_hashes'] if p.endswith('_gpu.npz')]
    if len(paths)!=1:raise ValueError('Ambiguous block-local detector lineage')
    gpu=Path(paths[0])
    if hashlib.sha256(gpu.read_bytes()).hexdigest()!=card['input_hashes'][str(gpu)]:
        raise ValueError('Block-local detector changed')
    with np.load(gpu,allow_pickle=True) as z:
        lookup={str(n):i for i,n in enumerate(z['chns_names'])}
        part=np.zeros((len(excluded),len(card['selected_contacts'])),float)
        for ci,name in enumerate(card['selected_contacts']):
            det=np.asarray(z['whole_dets'][lookup[name]],float).reshape(-1,2)
            for ei,(lo,hi) in enumerate(excluded):part[ei,ci]=np.any((det[:,0]<hi)&(det[:,1]>lo))
    missing=np.full((len(excluded),len(names)),np.nan)
    for ci in range(part.shape[1]):missing[:,names.index(f'participation_{ci}')]=part[:,ci]
    a,b=np.triu_indices(part.shape[1],k=1)
    for i,j in zip(a,b):missing[:,names.index(f'coupled_{i}_{j}')]=part[:,i]*part[:,j]
    times=np.r_[times,card['block_start']+excluded[:,0]]
    ends=np.r_[ends,card['block_start']+excluded[:,1]]
    marks=np.concatenate((marks,missing));participation=np.concatenate((participation,part))
    order=np.argsort(times,kind='stable')
    if len(times)!=card['n_group_windows_before_raw_segments'] or len(np.unique(times))!=len(times):
        raise ValueError('Restored event clock is incomplete or duplicated')
    return times[order],ends[order],marks[order],participation[order]


def make_human_data(subject,root,output):
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError(output)
    boundary=json.loads((root/'input_boundary'/subject/'card.json').read_text())
    with np.load(boundary['release_table']) as z:support=z['support'].copy()
    cards=[json.loads(p.read_text())|{'card_path':str(p),'card_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
           for p in sorted((root/'measurements'/subject).glob('*/card.json'))]
    expected=[r for r in boundary['block_sources'] if r['fully_inside_pre80']]
    if {r['block'] for r in cards}!={r['block'] for r in expected}:raise ValueError('Human subject measurement queue is incomplete')
    selected=cards[0]['selected_contacts']
    if any(r['selected_contacts']!=selected for r in cards):raise ValueError('Mixed contact dictionaries')
    selection_path=root/'input_boundary'/subject/'fit_only_interictal_channel_selection.json'
    interictal_selection=json.loads(selection_path.read_text())
    if interictal_selection['fit_refined_selection']!=selected:
        raise ValueError('Pure interictal FIT sensor selection differs; rebuild affected measurements')
    raw_support=merge_intervals([[r['block_start'],r['available_time']] for r in cards])
    observed=merge_intervals([[max(a,c),min(b,d)] for a,b in support for c,d in raw_support if max(a,c)<min(b,d)])
    missing=merge_intervals([[r['block_start']+lo,r['block_start']+hi] for r in cards for lo,hi in r['segment_crossing_exclusions']])
    # These are missing fine marks, not missing recording exposure.
    bounds=boundary['phase_boundaries'];event_parts=[];blocks=[]
    reference=next(r for r in cards if r['cache_path'])
    reference_manifest=json.loads(Path(reference['cache_path']).with_suffix('.manifest.json').read_text())
    with np.load(reference['cache_path']) as z:_,all_names=block_features(z,reference_manifest['stored_views'])
    clock_restorations=[];background_repairs=[]
    for card in sorted(cards,key=lambda r:r['block_start']):
        if not card['cache_path']:
            time,end,marks,part=restore_clock_only_events(card,np.empty(0),np.empty(0),
                np.empty((0,len(all_names))),np.empty((0,len(selected))),all_names)
            good=contained_events(time,end,observed)
            event_parts.append((time[good],part[good]))
            overlay=root/'background_overlays'/subject/f"block_{card['block']:04d}.npz"
            overlay_meta=json.loads(overlay.with_suffix('.json').read_text())
            if hashlib.sha256(overlay.read_bytes()).hexdigest()!=overlay_meta['sha256']:raise ValueError('Changed empty-block background overlay')
            with np.load(overlay) as z:context=background_summary(z['background_features'])
            blocks.append(dict(release=card['available_time'],times=time[good],marks=marks[good],context=context))
            background_repairs.append(dict(block=card['block'],empty_block_background_restored=True,path=str(overlay),sha256=overlay_meta['sha256']))
            clock_restorations.append(dict(block=card['block'],restored=len(time),retained=int(good.sum())))
            continue
        cache=Path(card['cache_path'])
        if hashlib.sha256(cache.read_bytes()).hexdigest()!=card['cache_sha256']:raise ValueError('Changed measurement cache')
        manifest=json.loads(cache.with_suffix('.manifest.json').read_text())
        with np.load(cache) as z:
            time=z['event_abs_time'].copy();end=time+z['core_seconds_raw']
            marks,names=block_features(z,manifest['stored_views'])
            if names!=all_names:raise ValueError('Feature order changes across blocks')
            time,end,marks,part=restore_clock_only_events(card,time,end,marks,z['participation'].astype(float),names)
            good=contained_events(time,end,observed)
            event_parts.append((time[good],part[good]))
            clock_restorations.append(dict(block=card['block'],restored=len(card['segment_crossing_exclusions']),retained=int(good.sum())))
            bg=np.asarray(z['background_features'],float)
            background_keep=background_core_mask(z['background_time_s'],card['segment_crossing_exclusions'])
            background_repairs.append(dict(block=card['block'],n_omitted_core_contaminated_windows=int((~background_keep).sum())))
            bg=bg[background_keep]
            # Background uses a completed fixed raw window, with block-level
            # publication delay just like the event stream.
            # Completed-window last value, within-block mean and variation
            # strengthen the background-only reference without event marks.
            context=background_summary(bg)
            blocks.append(dict(release=card['available_time'],times=time[good],marks=marks[good],context=context))
    fit=np.concatenate([b['marks'] for b in blocks if b['marks'] is not None and b['release']<=bounds['60pct']],axis=0)
    center=np.nanmedian(fit,axis=0);center=np.where(np.isfinite(center),center,0.)
    scale=np.nanmedian(np.abs(fit-center),axis=0)*1.4826
    scale=np.where(np.isfinite(scale)&(scale>1e-6),scale,1.)
    fit_hours=exposure(observed,observed[:,0].min(),bounds['60pct'])/3600
    count_scale=max(1.,len(fit)/fit_hours)
    context_dim=next(len(b['context']) for b in blocks if b['context'] is not None)
    bg_fit=np.array([b['context'] for b in blocks if b['release']<=bounds['60pct'] and b['context'] is not None])
    bg_center=np.nanmedian(bg_fit,axis=0);bg_center=np.nan_to_num(bg_center)
    bg_scale=np.nanmedian(np.abs(bg_fit-bg_center),axis=0)*1.4826;bg_scale=np.where(np.isfinite(bg_scale)&(bg_scale>1e-6),bg_scale,1.)
    input_dim=1+len(center)*2
    for b in blocks:
        if b['marks'] is None:b['cumsum']=np.zeros((1,input_dim))
        else:
            valid=np.isfinite(b['marks']);values=np.clip((np.where(valid,b['marks'],center)-center)/scale,-8,8)
            x=np.concatenate((np.ones((len(values),1)),values,valid.astype(float)),axis=-1)/count_scale
            b['cumsum']=np.concatenate((np.zeros((1,input_dim)),np.cumsum(x,axis=0)),axis=0)
        if b['context'] is None:b['normal_context']=np.zeros(context_dim*2)
        else:
            valid=np.isfinite(b['context']);b['normal_context']=np.concatenate((np.clip((np.where(valid,b['context'],bg_center)-bg_center)/bg_scale,-8,8),valid))
    et=np.concatenate([r[0] for r in event_parts]);part=np.concatenate([r[1] for r in event_parts]);order=np.argsort(et,kind='stable');et=et[order];part=part[order]
    # Coarse spatial groups are physical shafts, frozen with the sensor set.
    import re
    shafts=[re.sub(r"[0-9]+$",'',name) for name in selected];unique=sorted(set(shafts))
    coarse=np.stack([part[:,np.array(shafts)==name].mean(axis=1) for name in unique],axis=-1)
    cs=np.concatenate((np.zeros((1,len(unique))),np.cumsum(coarse,axis=0)),axis=0)
    anchors=np.arange(np.ceil(bounds['20pct']/300)*300,bounds['80pct'],300.)
    index=json.loads((Path('/data/hfosp_group_event_state_v0_1/dataset')/subject/'index.json').read_text())
    seizures=np.array([[r['onset_epoch'],r['offset_epoch']] for r in index['seizures']],float).reshape(-1,2)
    samples=[]
    for anchor in anchors:
        phase_name=str(phase(np.array([anchor]),bounds)[0])
        if exposure(observed,anchor-28800,anchor)<.9*28800:continue
        if np.any((seizures[:,0]<anchor)&(seizures[:,1]>anchor-28800)):continue
        targets=[]
        for lead in (0.,7200.,21600.):
            lo=anchor+lead;hi=lo+1800
            good=exposure(observed,lo,hi)>=1800-1e-5 and str(phase(np.array([np.nextafter(hi,-np.inf)]),bounds)[0])==phase_name
            if np.any((seizures[:,0]<hi)&(seizures[:,1]>anchor)):good=False
            left,right=np.searchsorted(et,[lo,hi]);count=int(right-left)
            recruitment=(cs[right]-cs[left])/count if count else np.zeros(len(unique))
            targets.append((float(count),recruitment,good,count>0 and good))
        if not any(t[2] for t in targets):continue
        context=np.zeros(context_dim*2);released=[b for b in blocks if b['release']<=anchor]
        if released:context=released[-1]['normal_context']
        background_bank=[]
        recent=[b for b in released if b['release']>anchor-28800]
        for tau in (1/6,.5,1.,2.,4.,8.,16.):
            numerator=np.zeros(context_dim);denominator=np.zeros(context_dim)
            for b in recent:
                weight=np.exp(-(anchor-b['release'])/3600/tau)
                valid=b['normal_context'][context_dim:]
                numerator+=weight*b['normal_context'][:context_dim]*valid;denominator+=weight*valid
            background_bank.extend(np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0))
            background_bank.extend(np.log1p(denominator))
        context=np.concatenate((context,background_bank))
        context=np.concatenate((context,[np.sin(2*np.pi*anchor/86400),np.cos(2*np.pi*anchor/86400),
            (anchor-released[-1]['release'])/3600 if released else 8.,
            exposure(observed,anchor-28800,anchor)/28800,
            np.log1p(max(0,anchor-observed[0,0])/28800)]))
        histories={}
        for hours in (.5,8.):
            left=anchor-hours*3600;last=left;xs=[];dts=[]
            for b in released:
                if b['release']<=left:continue
                cursor=np.searchsorted(b['times'],left)
                delta=(b['release']-last)/3600
                pieces=max(1,int(np.ceil(delta/(1/12))))
                for _ in range(pieces-1):xs.append(np.zeros(input_dim));dts.append(delta/pieces)
                xs.append(b['cumsum'][-1]-b['cumsum'][cursor]);dts.append(delta/pieces);last=b['release']
            delta=(anchor-last)/3600;pieces=max(1,int(np.ceil(delta/(1/12))))
            for _ in range(pieces):xs.append(np.zeros(input_dim));dts.append(delta/pieces)
            histories[str(hours)]=(np.asarray(xs,np.float32),np.asarray(dts,np.float32))
        samples.append(dict(anchor=anchor,phase=phase_name,context=context,targets=targets,histories=histories))
    if not samples:raise ValueError('No estimable anchors after measurement support restrictions')
    output.parent.mkdir(parents=True,exist_ok=True)
    # A portable torch bundle keeps exact variable-length elapsed-time rows.
    import torch
    payload=dict(subject=subject,samples=samples,input_dim=input_dim,context_dim=context_dim*16+5,n_recruitment=len(unique),
                 selected_contacts=selected,coarse_shafts=unique,feature_names=['count']+all_names+['available_'+name for name in all_names],
                 normalization=dict(center=center,scale=scale,count_scale=count_scale,bg_center=bg_center,bg_scale=bg_scale),
                 observed_support=observed,fine_mark_missing_intervals=missing,missing_measurement_intervals=np.empty((0,2)),phase_boundaries=bounds,
                 clock_restorations=clock_restorations,
                 background_repairs=background_repairs,
                 missing_fine_mark_contract='Processing-boundary events retain clock/count/detector participation/coupling; all other marks are missing. No recording exposure is removed for unavailable centroids.',
                 interictal_sensor_selection=dict(path=str(selection_path),sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),same_sensor_operator=True),
                 background_contract='last/mean/std of released raw background windows plus a fixed 7-timescale 8-hour bank; identical across short/long event histories',
                 event_replay_blocks=[dict(release=b['release'],times=b['times'],cumsum=b['cumsum']) for b in blocks],
                 source_cards=[{k:r[k] for k in ('card_path','card_sha256','cache_path','cache_sha256')} for r in cards],
                 interpretation='Closed-block receiver-time observer; physical event lookback is bounded separately. No claim of immediate event-driven physiology.')
    torch.save(payload,output)
    return payload
