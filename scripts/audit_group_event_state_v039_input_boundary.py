#!/usr/bin/env python3
"""Block-level release audit before any v0.3.9 human fitting.

Legacy event identities depend on a complete detector/packer block. A local
waveform shoulder alone is therefore not accepted as full token availability.
This audit distinguishes raw-view cutoff replay from unproven bit-for-bit
reconstruction of the historical detector; it never silently equates them.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.topic5_group_event_state.cache import BlockSpec, load_universe, FILTER_PAD_SECONDS
from src.topic5_group_event_state.raw_views import EpilepsiaeBlockReader, build_event_views, build_view_plan, clean_contact
from src.topic5_group_event_state.v035.contracts import atomic_json


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phase_at(times,bounds):
    t=np.asarray(times);out=np.full(t.shape,'CLOSED',dtype='<U12')
    for left,right,label in ((-np.inf,bounds['20pct'],'CALIBRATION'),(bounds['20pct'],bounds['60pct'],'FIT'),
                              (bounds['60pct'],bounds['70pct'],'INNER'),(bounds['70pct'],bounds['80pct'],'SELECTION')):
        out[(t>=left)&(t<right)]=label
    return out


def overlap(intervals,lo,hi):
    return float(np.maximum(0,np.minimum(intervals[:,1],hi)-np.maximum(intervals[:,0],lo)).sum())


class CutoffReader:
    """Reject any actual raw sample request beyond the simulated cutoff."""
    def __init__(self,reader,cutoff):
        self.reader=reader; self.cutoff=int(cutoff); self.requests=[]
    def read(self,start,stop,channels=None):
        if stop>self.cutoff: raise PermissionError('raw request crosses prediction cutoff')
        self.requests.append((int(start),int(stop)))
        return self.reader.read(start,stop,channels)


def audit(subject,output):
    dataset=Path('/data/hfosp_group_event_state_v0_1/dataset')/subject
    index=json.loads((dataset/'index.json').read_text())
    manifest_path=Path('/data/hfosp_group_event_state_v0_3_3/agent_c/human_inputs')/subject/'manifest_v3.json'
    manifest=json.loads(manifest_path.read_text()); bounds=manifest['report']['phase_boundaries_epoch']
    with np.load(manifest['input_path']) as z:
        support=np.asarray(z['target_segment_bounds'],float)
        source_times=np.asarray(z['event_time'],float)
    with np.load(dataset/'scalars.npz') as z:
        times=z['t_abs'].copy();block=z['block_of_event'].copy();core=z['core_seconds'].copy();wave_ok=z['has_waveform'].copy()
    rows=np.flatnonzero(times<bounds['80pct'])
    position=np.searchsorted(times,source_times[source_times<bounds['80pct']])
    if not np.array_equal(times[position],source_times[source_times<bounds['80pct']]): raise ValueError('event identity/order mismatch')
    release=np.full(len(times),np.inf); block_rows=[];raw_checks=[]
    eligible_blocks=[]
    for bi,shard_path in enumerate(index['source_shards']):
        shard=Path(shard_path); mp=shard.with_suffix('.manifest.json'); m=json.loads(mp.read_text())
        start=float(m['block_start_epoch'])
        if start>=bounds['80pct']: continue
        hpath=Path(m['source']['head_path'])
        head=dict(line.split('=',1) for line in hpath.read_text().strip().splitlines() if '=' in line)
        n_samples=int(head['num_samples']); fs=float(head['sample_freq']);end=start+n_samples/fs
        block_events=np.flatnonzero(block==bi)
        release[block_events]=end
        src=m['source']; raw=Path(src['raw_path'])
        expected=n_samples*int(head['num_channels'])*2
        if raw.stat().st_size!=expected: raise ValueError(f'raw size/header mismatch: {raw}')
        identity={key:dict(path=value,sha256=sha(value)) for key,value in src.items() if key in ('head_path','packed_path','lagpat_path')}
        # Whole-block release is conservative even for subsecond waveform features.
        row=dict(block=bi,start=start,end=end,n_events=int(len(block_events)),manifest=str(mp),manifest_sha256=sha(mp),
                 sources=identity,raw_path=str(raw),raw_size_bytes=expected,raw_mtime_ns=raw.stat().st_mtime_ns,
                 fully_inside_pre80=bool(end<=bounds['80pct']))
        block_rows.append(row)
        if end<bounds['60pct'] and np.any(wave_ok[block_events]): eligible_blocks.append((bi,shard,m))
    # Three real FIT blocks per patient, distributed across the prefix.
    selected=np.linspace(0,len(eligible_blocks)-1,min(3,len(eligible_blocks))).astype(int)
    for bi,shard,m in [eligible_blocks[j] for j in selected]:
        src=m['source'];reader=EpilepsiaeBlockReader(Path(src['raw_path']),Path(src['head_path']))
        spec=BlockSpec(index['dataset'],subject,m['record_name'],src['raw_path'],src['head_path'],src['gpu_path'],src['lagpat_path'],src['packed_path'],m['block_start_epoch'],reader.native_rate_hz)
        universe,_,labels=load_universe(spec)
        car=[reader.index[clean_contact(label)] for label in labels]
        plan=build_view_plan(reader,universe,car)
        with np.load(shard) as z:
            candidates=np.flatnonzero(z['has_waveform']); event=int(candidates[len(candidates)//2])
            onset=float(z['core_start_seconds'][event]);fs=reader.native_rate_hz
            first=int(np.rint(onset*fs))-int(m['core_offset_samples'])-int(round(FILTER_PAD_SECONDS*fs))
            stop=first+int(m['n_context_samples'])+2*int(round(FILTER_PAD_SECONDS*fs))
            full=build_event_views(reader,universe,first,stop,plan=plan)
            restricted=CutoffReader(reader,stop)
            limited=build_event_views(restricted,universe,first,stop,plan=plan)
            exact=all(np.array_equal(full[key],limited[key],equal_nan=True) for key in full)
            pad=int(round(FILTER_PAD_SECONDS*fs));n_ctx=int(m['n_context_samples'])
            cache_matches={view:bool(np.array_equal(full[view][:,pad:pad+n_ctx].astype(np.float16),z['waveform_'+view][event],equal_nan=True)) for view in m['stored_views']}
            try: restricted.read(first,stop+1,plan.picks)
            except PermissionError: rejection=True
            else: rejection=False
            raw_checks.append(dict(block=bi,local_event_row=event,raw_cutoff_sample=stop,requests=restricted.requests,
                                   exact_full_vs_truncated_raw_views=exact,stored_waveform_matches_raw=cache_matches,
                                   future_sample_request_rejected=rejection,
                                   raw_window_sha256=hashlib.sha256(reader.read(first,stop,plan.picks).tobytes()).hexdigest()))
    # Count full target support independently of history length and lead.
    seismic=np.array([[r['onset_epoch'],r['offset_epoch']] for r in index.get('seizures',[])],float).reshape(-1,2)
    anchors=np.arange(np.ceil(bounds['20pct']/300)*300,bounds['80pct'],300.)
    support_rows=[]
    for lead in (0.,7200.,21600.):
        by_phase={p:[] for p in ('FIT','INNER','SELECTION')}
        for anchor in anchors:
            phase=str(phase_at(np.array([anchor]),bounds)[0]);end=anchor+lead+1800
            if phase not in by_phase or str(phase_at(np.array([np.nextafter(end,-np.inf)]),bounds)[0])!=phase: continue
            exposure=overlap(support,anchor+lead,end)
            if exposure<1800-1e-5: continue
            if np.any((seismic[:,0]<end)&(seismic[:,1]>anchor-28800)): continue
            history=overlap(support,anchor-28800,anchor)
            if history<28800*.9: continue
            if not np.any((release[position]<=anchor)&(release[position]>anchor-28800)): continue
            by_phase[phase].append(float(anchor))
        for phase,valid in by_phase.items():
            picked=[]
            for t in valid:
                if not picked or t>=picked[-1]+1800: picked.append(t)
            support_rows.append(dict(lead_seconds=lead,target_width_seconds=1800,history_seconds=28800,phase=phase,
                                     n_anchors=len(valid),n_nonoverlapping_target_windows=len(picked),anchor_times=valid,
                                     independent_anchor_times=picked))
    output.parent.mkdir(parents=True,exist_ok=True)
    npz=output.with_suffix('.npz')
    np.savez_compressed(npz,raw_rows=position,event_time=times[position],available_time=release[position],support=support)
    card=dict(status='COMPLETE',subject=subject,phase_boundaries=bounds,n_prefix_event_rows=len(position),
              whole_block_release_delay_seconds={key:float(fun(release[position]-times[position])) for key,fun in [('median',np.median),('max',np.max)]},
              local_waveform_min_delay_seconds=float(index['core_seconds_nominal']+.25+FILTER_PAD_SECONDS),
              event_identity_order_exact=True,raw_view_cutoff_replay_pass=all(r['exact_full_vs_truncated_raw_views'] and r['future_sample_request_rejected'] for r in raw_checks),
              raw_waveform_replays=raw_checks,block_sources=block_rows,future_window_support=support_rows,
              input_manifest=str(manifest_path),input_manifest_sha256=sha(manifest_path),index_sha256=sha(dataset/'index.json'),
              release_table=str(npz),release_table_sha256=sha(npz),
              historical_detector_bit_rebuild_performed=False,
              boundary='Whole-block event release pending block-local detector dependency verification; raw waveform replay is narrower than complete detector/packer rebuild.',
              human_training_authorized_by_this_card=False,
              development_targets_read=False,sealed_partition_opened=False,seizure_outcomes_used_for_model_selection=False,
              seizure_annotation_use='existing coverage and reset exclusions only',
              source_sha256=sha(__file__))
    atomic_json(output,card)
    print(json.dumps({key:card[key] for key in ('status','subject','n_prefix_event_rows','whole_block_release_delay_seconds','raw_view_cutoff_replay_pass')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();audit(args.subject,args.output)
