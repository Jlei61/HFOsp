#!/usr/bin/env python3
"""Repack one raw block using FIT-only sensors and rebuild multimodal tokens.

The scalar detector asset is block-local and is retained by hash. Group-event
selection and waveform/centroid measurement are rebuilt. This does not claim
bit parity with the historical cusignal detector or its population-selected
group-event dictionary.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.signal import iirnotch, filtfilt, firwin
from src.group_event_analysis import (EventWindow, _legacy_return_time_ranges,
    _legacy_pick_no_overlap_time_ranges, refine_packed_windows_by_all_bool,
    compute_stitched_spectrogram_centroids_legacy, lag_rank_from_centroids)
from src.topic5_group_event_state.cache import BlockSpec, build_block_shard
from src.topic5_group_event_state.raw_views import (EpilepsiaeBlockReader, build_contact_universe,
    build_view_plan, build_event_views, clean_contact)
from src.topic5_group_event_state.v035.contracts import atomic_json


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pack(detections,selected,duration,width):
    fs=500.;timeline=np.zeros(int((duration+1)*fs)+2,dtype=np.int32)
    for name in selected:
        intervals=detections[name]
        if not len(intervals):continue
        lo=np.clip(((intervals[:,0]-.03)*fs).astype(int),0,len(timeline)-1)
        hi=np.clip(((intervals[:,1]+.03)*fs).astype(int)+1,0,len(timeline)-1)
        np.add.at(timeline,lo,1);np.add.at(timeline,hi,-1)
    runs=_legacy_return_time_ranges(np.cumsum(timeline)>=.5*len(selected),fs)
    centers=runs.mean(1)
    windows=[EventWindow(float(c-width/2),float(c+width/2),i) for i,c in enumerate(centers) if c-width/2>=0 and c+width/2<=duration]
    # Preserve the registered order: all-channel noise rejection, then remove
    # both members of overlapping group-event pairs.
    windows=refine_packed_windows_by_all_bool(windows,detections,fs=fs,thresh=.7)
    return _legacy_pick_no_overlap_time_ranges(np.array([[w.start,w.end] for w in windows]).reshape(-1,2),2.)


def measure(reader,plan,universe,packed,selected,detections):
    fs=reader.native_rate_hz;duration=reader.n_samples/fs
    keep=[];centroids=[]
    for left in np.arange(0.,duration,200.):
        right=min(left+200.,duration)
        # A packed event that crosses the raw processing segment is preserved
        # in the measurement exclusions, not silently called event-free.
        rows=np.flatnonzero((packed[:,0]>=left)&(packed[:,1]<right-1/fs))
        if not len(rows):continue
        first=int(round(left*fs));stop=min(reader.n_samples,int(round(right*fs)))
        data=build_event_views(reader,universe,first,stop,plan=plan)['detector'].astype(np.float64)
        for freq in np.arange(50.,min(250.,fs/2-1)+1,50.):
            b,a=iirnotch(freq/(fs/2),30);data=filtfilt(b,a,data,axis=-1)
        b=firwin(201,[80/(fs/2),min(250.,fs/2-1)/(fs/2)],pass_zero=False)
        data=filtfilt(b,[1.],data,axis=-1)
        pieces=[];sizes=[]
        for row in rows:
            lo=max(0,int(np.ceil((packed[row,0]-left)*fs)));hi=min(data.shape[1],int(np.floor((packed[row,1]-left)*fs))+1)
            pieces.append(data[:,lo:hi]);sizes.append(hi-lo)
        stitched=np.concatenate(pieces,axis=1)
        centers=compute_stitched_spectrogram_centroids_legacy(stitched,np.cumsum(sizes)/fs,sfreq=fs)
        keep.extend(rows.tolist());centroids.append(centers)
    keep=np.asarray(keep,dtype=int)
    if not len(keep):return packed[:0],np.zeros((len(selected),0),bool),np.zeros((len(selected),0)),keep
    chosen=packed[keep];raw=np.concatenate(centroids,axis=1)
    participation=np.zeros((len(selected),len(chosen)),bool)
    for ci,name in enumerate(selected):
        intervals=detections[name]
        for ei,(lo,hi) in enumerate(chosen):
            participation[ci,ei]=np.any((intervals[:,0]<hi)&(intervals[:,1]>lo))
    # Nonparticipants never receive a usable rank or synthetic delay.
    raw[~participation]=np.nan
    return chosen,participation,raw,keep


def run(args):
    start=time.time();output=args.output
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    selection_path=args.root/'input_boundary'/args.subject/'fit_only_channel_selection.json'
    selection=json.loads(selection_path.read_text());selected=selection['fit_refined_selection']
    boundary=json.loads((selection_path.parent/'card.json').read_text())
    row=next(r for r in boundary['block_sources'] if r['block']==args.block)
    if row['end']>boundary['phase_boundaries']['80pct']:raise PermissionError('Do not rebuild blocks crossing unopened partition')
    original=json.loads(Path(row['manifest']).read_text());src=original['source'];gpu=Path(src['gpu_path'])
    with np.load(gpu,allow_pickle=True) as z:
        all_names=[str(n) for n in z['chns_names']]
        detections={name:np.asarray(z['whole_dets'][i],float).reshape(-1,2) for i,name in enumerate(all_names)}
    if all_names!=selection['all_channels']:raise ValueError('FIT and current detector channel order mismatch')
    reader=EpilepsiaeBlockReader(Path(src['raw_path']),Path(src['head_path']))
    universe=build_contact_universe('epilepsiae',args.subject,selected,all_names,'car_global_intracranial_from_producer',{})
    plan=build_view_plan(reader,universe,[reader.index[clean_contact(n)] for n in all_names])
    width={'epilepsiae_1096':.18,'epilepsiae_1125':.15,'epilepsiae_253':.3}[args.subject]
    packed=pack(detections,selected,reader.n_samples/reader.native_rate_hz,width)
    before=packed.copy();packed,part,raw,keep=measure(reader,plan,universe,packed,selected,detections)
    lag,rank=lag_rank_from_centroids(raw,part)
    parent=output.parent;stem=original['record_name'];pfile=parent/(stem+'_packedTimes_withFreqCent.npy');lfile=parent/(stem+'_lagPat_withFreqCent.npz')
    np.save(pfile,packed);np.savez_compressed(lfile,eventsBool=part,lagPatRaw=lag,lagPatRank=rank,chnNames=np.array(selected))
    cache=None
    if len(packed):
        spec=BlockSpec('epilepsiae',args.subject,stem,src['raw_path'],src['head_path'],str(gpu),str(lfile),str(pfile),row['start'],reader.native_rate_hz)
        manifest=build_block_shard(spec,parent/'cache')
        cache=parent/'cache'/(stem+'.npz')
    excluded=np.setdiff1d(np.arange(len(before)),keep)
    card=dict(status='COMPLETE',subject=args.subject,block=args.block,record_name=stem,block_start=row['start'],
              available_time=row['end'],fit_measurement_end=selection['fit_end'],selected_contacts=selected,
              n_group_windows_before_raw_segments=len(before),n_events=len(packed),
              segment_crossing_exclusions=before[excluded].tolist(),
              centroid_definition='participant-masked stitched spectrogram centroid on each closed 200-second processing segment',
              cache_path=str(cache) if cache else None,cache_sha256=sha(cache) if cache else None,
              packed_path=str(pfile),packed_sha256=sha(pfile),lagpat_path=str(lfile),lagpat_sha256=sha(lfile),
              input_hashes={str(selection_path):sha(selection_path),str(gpu):sha(gpu),src['head_path']:sha(src['head_path'])},
              source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/group_event_analysis.py',ROOT/'src/topic5_group_event_state/cache.py',ROOT/'src/topic5_group_event_state/raw_views.py']},
              historical_detector_rebuilt=False,detector_dependency='fixed block-local gpu asset; parameters and channel reference inherited; no population refine counts read',
              missing_waveform_is_not_silence=True,empty_block_background='unavailable when no event shard is built',
              development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False,
              elapsed_seconds=time.time()-start)
    atomic_json(output,card)
    print(json.dumps({k:card[k] for k in ('status','subject','record_name','n_events','elapsed_seconds')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',required=True);p.add_argument('--block',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('/data/hfosp_group_event_state_v0_3_9_transition_transfer'))
    run(p.parse_args())
