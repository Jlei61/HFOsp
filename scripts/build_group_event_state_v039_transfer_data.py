#!/usr/bin/env python3
"""Pair new event expressions with an available pre-event state anchor."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.topic5_group_event_state.v039.human_data import block_features,contained_events,phase
from src.topic5_group_event_state.v035.contracts import atomic_json
from build_group_event_state_v037_strict_decoder_cache import _densify


def build(subject,root,output):
    if output.exists():raise FileExistsError(output)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    data_path=root/'human_data_v2'/f'{subject}.pt';data=torch.load(data_path,map_location='cpu',weights_only=False)
    decoder_provenance=root/'decoder_rebuilt'/subject/'cache'/f'{subject}__anatomy'/'provenance.json'
    decoder=json.loads(decoder_provenance.read_text());mapped=decoder['joint_contacts'];cols=np.array([data['selected_contacts'].index(n) for n in mapped])
    anchor=np.array([s['anchor'] for s in data['samples']]);phases=np.array([s['phase'] for s in data['samples']]);out=[];source=[];excluded={}
    for card_ref in data['source_cards']:
        card_path=Path(card_ref['card_path']);c=json.loads(card_path.read_text())
        if sha(card_path)!=card_ref['card_sha256']:raise ValueError('Changed frozen measurement card')
        if not c['cache_path'] or c['available_time']<=data['phase_boundaries']['20pct']:continue
        if sha(c['cache_path'])!=c['cache_sha256']:raise ValueError('Changed frozen cache')
        manifest=json.loads(Path(c['cache_path']).with_suffix('.manifest.json').read_text())
        with np.load(c['cache_path']) as z:
            t=z['event_abs_time'];position=np.searchsorted(anchor,t,side='right')-1;safe=np.maximum(position,0)
            raw_phase=phase(t,data['phase_boundaries']);part=z['participation'][:,cols];delay=z['relative_delay_s'][:,cols]
            ranks=_densify(z['tied_group_id'][:,cols]);groups=np.array([len(np.unique(v[v>=0])) for v in ranks])
            complete=~np.any(part&~np.isfinite(delay),axis=1)
            keep=(position>=0)&(t-anchor[safe]<=300+1e-6)&(raw_phase==phases[safe])&complete&(groups>=2)&(part.sum(-1)>=3)
            keep &= contained_events(t,t+z['core_seconds_raw'],data['observed_support'])
            if not np.any(keep):continue
            if np.any(c['available_time']<=anchor[safe[keep]]):raise ValueError('Current event could already be in its pre-event state')
            marks,names=block_features(z,manifest['stored_views'])
            targets={}
            for prefix in ('band_centroid','band_log_energy','band_log_peak','cross_band_lag'):
                columns=[i for i,name in enumerate(names) if name.startswith(prefix)]
                targets[prefix]=marks[:,columns][keep]
            columns=[i for i,name in enumerate(names) if name.startswith(('detector_','bipolar_','shaft_car_'))]
            targets['waveform_statistics']=marks[:,columns][keep]
            with np.errstate(invalid='ignore'):
                span=np.max(np.where(part,delay,-np.inf),axis=1)-np.min(np.where(part,delay,np.inf),axis=1)
                second=np.divide(np.where(ranks==1,delay,0).sum(-1),(ranks==1).sum(-1),out=np.full(len(t),np.nan),where=(ranks==1).sum(-1)>0)
                third=np.divide(np.where(ranks==2,delay,0).sum(-1),(ranks==2).sum(-1),out=np.full(len(t),np.nan),where=(ranks==2).sum(-1)>0)
            targets['propagation_span_s']=span[keep,None];targets['next_group_delay_s']=(third-second)[keep,None]
            a,b=np.triu_indices(len(cols),k=1);targets['contact_coupling']=(part[:,a]&part[:,b])[keep].astype(float)
            out.append(dict(time=t[keep],anchor_position=position[keep],ranks=ranks[keep],targets=targets,block=np.full(keep.sum(),c['block'],int)))
            source.append(dict(card_path=str(card_path),card_sha256=sha(card_path)))
            excluded[str(c['block'])]=dict(clock_only_events_not_scored_for_fine=len(c['segment_crossing_exclusions']),undefined_fine=int((~complete).sum()))
    if not out:raise ValueError('No eligible pre-event expression pairs')
    times=np.concatenate([r['time'] for r in out]);order=np.argsort(times,kind='stable');positions=np.concatenate([r['anchor_position'] for r in out])[order]
    arrays=dict(event_time=times[order],anchor_position=positions,phase=phases[positions],ranks=np.concatenate([r['ranks'] for r in out])[order],
        block=np.concatenate([r['block'] for r in out])[order],anchor_time=anchor[positions],contact_names=np.array(mapped))
    for key in out[0]['targets']:arrays['target_'+key]=np.concatenate([r['targets'][key] for r in out])[order]
    output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**arrays)
    record=dict(status='COMPLETE',subject=subject,data_path=str(output),data_sha256=sha(output),human_data_path=str(data_path),human_data_sha256=sha(data_path),
        decoder_provenance=str(decoder_provenance),decoder_provenance_sha256=sha(decoder_provenance),contact_names=mapped,
        support={p:dict(events=int((arrays['phase']==p).sum()),anchors=len(np.unique(positions[arrays['phase']==p])),raw_blocks=len(np.unique(arrays['block'][arrays['phase']==p]))) for p in ['FIT','INNER','SELECTION']},
        current_event_excluded_from_state=True,source_cards=source,excluded_fine=excluded,
        target_contract='Event expression conditioned on a retrospectively measured contact prefix; this is a structural conditional task, not a claim that a live raw-waveform decoder can expose the prefix immediately.',
        pairing='Each event scored once, latest eligible anchor at or before onset within 5 minutes; current raw block unreleased. Fit/evaluation phase shared with upstream.',
        development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False,source_sha256=sha(__file__))
    atomic_json(output.with_suffix('.json'),record);print(json.dumps({k:record[k] for k in ['status','subject','support']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.subject,a.root,a.output)
