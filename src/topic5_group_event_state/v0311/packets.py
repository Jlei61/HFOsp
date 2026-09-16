"""Rich per-event packets on a 1-minute slow clock.

One group event keeps its real time, participation mask, per-contact relative
delay / tied-lead flag, per-band energy descriptors, cross-band lags, waveform
descriptors and the real inter-event gap. A minute packet is the slow update
unit; it never publishes earlier than the closed block that produced its marks,
so ``release`` is the honest availability time and the query-time information
lag is reported rather than assumed away.
"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
import numpy as np

from ..v039.human_data import (merge_intervals,exposure,contained_events,subtract_intervals,
                               background_core_mask,background_summary)

PACKET_SECONDS=60.0
WAVE_VIEW='detector'
MEAN_VIEWS=('bipolar','shaft_car')
# Frozen target coordinates: log-energy of each fast band relative to ied_low.
RATIO_BANDS=('gamma','low_ripple','ripple','fast_ripple')
XLAG_PAIRS=(('ied_low','gamma'),('ied_low','ripple'),('gamma','ripple'),('ripple','fast_ripple'))
BAND_FEAT=dict(peak_time_s=0,centroid_time_s=1,log_integrated_energy=2,width_s=3,log_peak_amplitude=4)
MIN_DELAY_FOR_IQR=4

CONTACT_FEATURE_NAMES=(['relative_delay_s','delay_rank_fraction','tied_lead_flag']
                       +[f'band_log_energy_{i}' for i in range(5)]
                       +[f'band_centroid_s_{i}' for i in range(5)]
                       +[f'band_log_peak_{i}' for i in range(5)]
                       +[f'cross_band_lag_s_{i}' for i in range(10)]
                       +['wave_log_rms','wave_log_peak','wave_log_line_length'])


def _wave_stats(w):
    w=np.asarray(w,np.float32)
    return np.log1p(np.stack((np.sqrt(np.mean(w*w,axis=-1)),np.max(np.abs(w),axis=-1),
                              np.mean(np.abs(np.diff(w,axis=-1)),axis=-1)),axis=-1))


def _participant_mean(values,part):
    mask=part.reshape(part.shape+(1,)*(values.ndim-2))&np.isfinite(values)
    count=mask.sum(axis=1)
    return np.divide(np.where(mask,values,0).sum(axis=1),count,
                     out=np.full(count.shape,np.nan,float),where=count>0)


def _iqr(delay,part):
    out=np.full(len(delay),np.nan)
    for i in range(len(delay)):
        v=delay[i][part[i]&np.isfinite(delay[i])]
        if len(v)>=MIN_DELAY_FOR_IQR:out[i]=np.subtract(*np.percentile(v,[75,25]))
    return out


def block_event_tables(z,manifest,n_contacts):
    """Per-event contact tokens and frozen target coordinates for one closed block."""
    part=z['participation'].astype(bool)
    n=part.shape[0]
    delay=z['relative_delay_s'].astype(np.float32)
    delay=np.where(part,delay,np.nan)
    order=np.where(np.isfinite(delay),delay,np.inf).argsort(axis=1).argsort(axis=1).astype(np.float32)
    denom=np.maximum(part.sum(axis=1,keepdims=True)-1,1)
    rank=np.where(part&np.isfinite(delay),order/denom,np.nan)
    lead=np.where(part&np.isfinite(delay),(z['tied_group_id']==0).astype(np.float32),np.nan)
    bands=list(manifest['bands'])
    bf=z['band_features'].astype(np.float32)
    energy=bf[...,BAND_FEAT['log_integrated_energy']]
    centroid=bf[...,BAND_FEAT['centroid_time_s']]
    peak=bf[...,BAND_FEAT['log_peak_amplitude']]
    xlag=z['cross_band_lag_s'].astype(np.float32)
    wave=_wave_stats(z['waveform_'+WAVE_VIEW])
    has=z['has_waveform'].astype(bool)
    wave[~has]=np.nan
    tokens=np.concatenate((delay[...,None],rank[...,None],lead[...,None],
                           energy,centroid,peak,xlag,wave),axis=-1).astype(np.float32)
    tokens[~part]=np.nan
    pairs=[tuple(p) for p in manifest['cross_band_pairs']]
    ratio_idx=[bands.index(b) for b in RATIO_BANDS];base=bands.index('ied_low')
    mean_energy=_participant_mean(energy,part)
    band_ratio=(mean_energy[:,ratio_idx]-mean_energy[:,[base]]).astype(np.float32)
    mean_xlag=_participant_mean(xlag,part)
    xlag_idx=[pairs.index(p) for p in XLAG_PAIRS]
    signed_xlag=mean_xlag[:,xlag_idx].astype(np.float32)
    iqr=_iqr(delay,part).astype(np.float32)
    extra=[]
    for view in MEAN_VIEWS:
        v=_participant_mean(_wave_stats(z['waveform_'+view]),part)
        v[~has]=np.nan
        extra.append(v[:,0])
    targets=dict(band_ratio=band_ratio,signed_xlag=signed_xlag,delay_iqr=iqr,
                 size=part.sum(axis=1).astype(np.float32))
    event=np.stack((np.log1p(part.sum(axis=1)).astype(np.float32),
                    part.mean(axis=1).astype(np.float32),
                    z['core_seconds_raw'].astype(np.float32),
                    has.astype(np.float32),
                    np.where(np.isfinite(iqr),iqr,np.nan),
                    np.nanmax(np.where(part,delay,np.nan),axis=1)-np.nanmin(np.where(part,delay,np.nan),axis=1),
                    extra[0],extra[1]),axis=-1).astype(np.float32)
    return tokens,event,targets,part


EVENT_FEATURE_NAMES=('log1p_n_participants','participation_fraction','core_seconds_raw',
                     'has_waveform','delay_iqr_s','delay_span_s','bipolar_log_rms','shaft_car_log_rms')


def _restore_clock_only(card,n_contacts,names):
    """Processing-boundary events keep clock and detector participation only."""
    excluded=np.asarray(card['segment_crossing_exclusions'],float).reshape(-1,2)
    if not len(excluded):return np.empty(0),np.empty(0),np.zeros((0,n_contacts),bool)
    paths=[p for p in card['input_hashes'] if p.endswith('_gpu.npz')]
    if len(paths)!=1:raise ValueError('Ambiguous block-local detector lineage')
    gpu=Path(paths[0])
    if hashlib.sha256(gpu.read_bytes()).hexdigest()!=card['input_hashes'][str(gpu)]:
        raise ValueError('Block-local detector changed')
    with np.load(gpu,allow_pickle=True) as z:
        lookup={str(n):i for i,n in enumerate(z['chns_names'])}
        part=np.zeros((len(excluded),n_contacts),bool)
        for ci,name in enumerate(names):
            det=np.asarray(z['whole_dets'][lookup[name]],float).reshape(-1,2)
            for ei,(lo,hi) in enumerate(excluded):part[ei,ci]=np.any((det[:,0]<hi)&(det[:,1]>lo))
    return card['block_start']+excluded[:,0],card['block_start']+excluded[:,1],part


def build_subject(subject,source_root,out_dir,dataset_root='/data/hfosp_group_event_state_v0_1/dataset'):
    source_root=Path(source_root);out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    boundary=json.loads((source_root/'input_boundary'/subject/'card.json').read_text())
    with np.load(boundary['release_table']) as z:support=z['support'].copy()
    cards=[json.loads(p.read_text())|{'card_path':str(p),'card_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
           for p in sorted((source_root/'measurements'/subject).glob('*/card.json'))]
    expected={r['block'] for r in boundary['block_sources'] if r['fully_inside_pre80']}
    if {r['block'] for r in cards}!=expected:raise ValueError('Measurement queue incomplete')
    selected=cards[0]['selected_contacts']
    if any(r['selected_contacts']!=selected for r in cards):raise ValueError('Mixed contact dictionaries')
    sel_path=source_root/'input_boundary'/subject/'fit_only_interictal_channel_selection.json'
    if json.loads(sel_path.read_text())['fit_refined_selection']!=selected:
        raise ValueError('Interictal FIT sensor selection differs')
    n_contacts=len(selected)
    shafts=[re.sub(r'[0-9]+$','',n) for n in selected]
    numbers=[int(re.search(r'([0-9]+)$',n).group(1)) for n in selected]
    unique=sorted(set(shafts))
    coarse_rule='physical shaft'
    if len(unique)<2:
        # A single-shaft implant has no shaft composition; split each shaft at its
        # median contact number so the coarse-spatial view stays estimable.
        labels=[]
        for name in selected:
            sh=re.sub(r'[0-9]+$','',name);num=int(re.search(r'([0-9]+)$',name).group(1))
            med=np.median([n for n,s2 in zip(numbers,shafts) if s2==sh])
            labels.append(f'{sh}_{"distal" if num>med else "proximal"}')
        shafts=labels;unique=sorted(set(labels))
        coarse_rule='single-shaft implant split at the median contact number'
    shaft_index=np.array([unique.index(s) for s in shafts])
    raw_support=merge_intervals([[r['block_start'],r['available_time']] for r in cards])
    observed=merge_intervals([[max(a,c),min(b,d)] for a,b in support for c,d in raw_support if max(a,c)<min(b,d)])
    spans=np.array(sorted([[r['block_start'],r['block_start']+3600.] for r in cards]))
    ambiguous=merge_intervals([[max(spans[i,0],spans[j,0]),min(spans[i,1],spans[j,1])]
                               for i in range(len(spans)) for j in range(i+1,len(spans))
                               if max(spans[i,0],spans[j,0])<min(spans[i,1],spans[j,1])])
    # Two records covering the same wall clock cannot be pooled into one rate.
    if len(ambiguous):observed=subtract_intervals(observed,ambiguous)
    reference=next(r for r in cards if r['cache_path'])
    ref_manifest=json.loads(Path(reference['cache_path']).with_suffix('.manifest.json').read_text())
    n_token=len(CONTACT_FEATURE_NAMES);n_event=len(EVENT_FEATURE_NAMES)
    times=[];ends=[];tok=[];ev=[];parts=[];blk=[];rel=[];tg={k:[] for k in ('band_ratio','signed_xlag','delay_iqr','size')}
    blocks=[];restorations=[];background=[]
    for card in sorted(cards,key=lambda r:r['block_start']):
        bt,be,bpart=_restore_clock_only(card,n_contacts,selected)
        if card['cache_path']:
            cache=Path(card['cache_path'])
            if hashlib.sha256(cache.read_bytes()).hexdigest()!=card['cache_sha256']:raise ValueError('Changed measurement cache')
            manifest=json.loads(cache.with_suffix('.manifest.json').read_text())
            if list(manifest['bands'])!=list(ref_manifest['bands']) or manifest['cross_band_pairs']!=ref_manifest['cross_band_pairs']:
                raise ValueError('Band coordinates change across blocks')
            with np.load(cache,allow_pickle=True) as z:
                t=z['event_abs_time'].copy();e=t+z['core_seconds_raw']
                tk,evf,tgt,part=block_event_tables(z,manifest,n_contacts)
                bg=np.asarray(z['background_features'],float)
                keep=background_core_mask(z['background_time_s'],card['segment_crossing_exclusions'])
                context=background_summary(bg[keep])
            background.append(dict(block=card['block'],n_omitted_core_contaminated_windows=int((~keep).sum())))
        else:
            t=np.empty(0);e=np.empty(0);tk=np.empty((0,n_contacts,n_token),np.float32)
            evf=np.empty((0,n_event),np.float32);part=np.empty((0,n_contacts),bool)
            tgt={'band_ratio':np.empty((0,len(RATIO_BANDS)),np.float32),
                 'signed_xlag':np.empty((0,len(XLAG_PAIRS)),np.float32),
                 'delay_iqr':np.empty(0,np.float32),'size':np.empty(0,np.float32)}
            overlay=source_root/'background_overlays'/subject/f"block_{card['block']:04d}.npz"
            meta=json.loads(overlay.with_suffix('.json').read_text())
            if hashlib.sha256(overlay.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Changed background overlay')
            with np.load(overlay) as z:context=background_summary(z['background_features'])
            background.append(dict(block=card['block'],empty_block_background_restored=True))
        if len(bt):
            t=np.r_[t,bt];e=np.r_[e,be]
            tk=np.concatenate((tk,np.full((len(bt),n_contacts,n_token),np.nan,np.float32)))
            evf=np.concatenate((evf,np.full((len(bt),n_event),np.nan,np.float32)))
            evf[-len(bt):,0]=np.log1p(bpart.sum(axis=1));evf[-len(bt):,1]=bpart.mean(axis=1)
            part=np.concatenate((part,bpart))
            tgt={'band_ratio':np.concatenate((tgt['band_ratio'],np.full((len(bt),len(RATIO_BANDS)),np.nan,np.float32))),
                 'signed_xlag':np.concatenate((tgt['signed_xlag'],np.full((len(bt),len(XLAG_PAIRS)),np.nan,np.float32))),
                 'delay_iqr':np.r_[tgt['delay_iqr'],np.full(len(bt),np.nan,np.float32)],
                 'size':np.r_[tgt['size'],bpart.sum(axis=1).astype(np.float32)]}
        o=np.argsort(t,kind='stable')
        if len(t)!=card['n_group_windows_before_raw_segments'] or len(np.unique(t))!=len(t):
            raise ValueError('Restored event clock is incomplete or duplicated')
        t,e,tk,evf,part=t[o],e[o],tk[o],evf[o],part[o];tgt={k:v[o] for k,v in tgt.items()}
        good=contained_events(t,e,observed)
        restorations.append(dict(block=card['block'],restored=int(len(bt)),retained=int(good.sum())))
        bi=len(blocks)
        blocks.append(dict(block=int(card['block']),start=float(card['block_start']),
                           end=float(card['block_start']+3600.),release=float(card['available_time']),
                           context=context,n_events=int(good.sum())))
        times.append(t[good]);ends.append(e[good]);tok.append(tk[good]);ev.append(evf[good])
        parts.append(part[good]);blk.append(np.full(int(good.sum()),bi,np.int32))
        rel.append(np.full(int(good.sum()),card['available_time'],float))
        for k in tg:tg[k].append(tgt[k][good])
    ctx_dim=next(len(b['context']) for b in blocks if b['context'] is not None)
    for b in blocks:
        if b['context'] is None:
            b['context']=np.full(ctx_dim,np.nan)
            b['background_unavailable']=True
    times=np.concatenate(times);ends=np.concatenate(ends);tok=np.concatenate(tok)
    ev=np.concatenate(ev);part=np.concatenate(parts);blk=np.concatenate(blk);rel=np.concatenate(rel)
    tg={k:np.concatenate(v) for k,v in tg.items()}
    o=np.argsort(times,kind='stable')
    times,ends,tok,ev,part,blk,rel=times[o],ends[o],tok[o],ev[o],part[o],blk[o],rel[o]
    tg={k:v[o] for k,v in tg.items()}
    gap=np.r_[np.nan,np.diff(times)]
    ev=np.concatenate((ev,np.log1p(np.where(np.isfinite(gap),gap,np.nan))[:,None].astype(np.float32)),axis=-1)
    shaft_count=np.stack([part[:,shaft_index==s].sum(axis=1) for s in range(len(unique))],axis=-1).astype(np.float32)
    packets=build_packets(times,blocks,observed,ambiguous)
    payload=dict(subject=subject,selected_contacts=selected,shafts=unique,shaft_index=shaft_index,
                 n_contacts=n_contacts,coarse_community_rule=coarse_rule,bands=list(ref_manifest['bands']),
                 band_edges_hz=ref_manifest['band_edges_hz'],
                 cross_band_pairs=[tuple(p) for p in ref_manifest['cross_band_pairs']],
                 contact_feature_names=list(CONTACT_FEATURE_NAMES),
                 event_feature_names=list(EVENT_FEATURE_NAMES)+['log1p_inter_event_gap_s'],
                 ratio_bands=list(RATIO_BANDS),xlag_pairs=[list(p) for p in XLAG_PAIRS],
                 event_time=times,event_end=ends,event_block=blk,event_release=rel,
                 contact_tokens=tok,event_features=ev,participation=part,
                 targets=tg|dict(shaft_count=shaft_count),
                 blocks=[{k:v for k,v in b.items() if k!='context'} for b in blocks],
                 background_context_missing=[b['block'] for b in blocks if b.get('background_unavailable')],
                 block_context=np.stack([b['context'] for b in blocks]),
                 packets=packets,observed_support=observed,ambiguous_intervals=ambiguous,
                 phase_boundaries=boundary['phase_boundaries'],
                 clock_restorations=restorations,background_repairs=background,
                 source_cards=[{k:r[k] for k in ('card_path','card_sha256','cache_path','cache_sha256')} for r in cards],
                 release_contract=('minute packets publish no earlier than the closed 1-hour block that '
                                   'produced their marks; query-time information lag is reported, not removed'))
    import torch
    torch.save(payload,out_dir/f'{subject}.pt')
    return payload


def build_packets(times,blocks,observed,ambiguous):
    """Uniform global 60-second lattice; a packet publishes with its latest block."""
    start=min(b['start'] for b in blocks);stop=max(b['end'] for b in blocks)
    edges=np.arange(np.floor(start/PACKET_SECONDS)*PACKET_SECONDS,stop+PACKET_SECONDS,PACKET_SECONDS)
    lo_t=edges[:-1];hi_t=edges[1:]
    rel=np.full(len(lo_t),np.inf)
    bidx=np.full(len(lo_t),-1,np.int32)
    for bi,b in enumerate(blocks):
        a=int(np.searchsorted(lo_t,b['start'],'right')-1);z=int(np.searchsorted(lo_t,b['end'],'left'))
        a=max(a,0)
        sel=slice(a,z)
        rel[sel]=np.maximum(np.where(np.isfinite(rel[sel]),rel[sel],-np.inf),b['release'])
        bidx[sel]=bi
    lo=np.searchsorted(times,lo_t,'left');hi=np.searchsorted(times,hi_t,'left')
    expo=np.array([exposure(observed,a,b) for a,b in zip(lo_t,hi_t)])
    return dict(start=lo_t,end=hi_t,release=rel,block=bidx,
                event_lo=lo.astype(np.int64),event_hi=hi.astype(np.int64),exposure=expo,
                ambiguous_intervals=ambiguous)
