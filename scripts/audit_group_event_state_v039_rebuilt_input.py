#!/usr/bin/env python3
"""Validate the new FIT sensor operator and held-out raw availability boundary."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from audit_group_event_state_v039_input_boundary import CutoffReader,sha
from build_group_event_state_v039_fit_measurement import pack
from src.topic5_group_event_state.cache import BlockSpec,load_universe,FILTER_PAD_SECONDS
from src.topic5_group_event_state.raw_views import EpilepsiaeBlockReader,build_event_views,build_view_plan,clean_contact
from src.topic5_group_event_state.v035.contracts import atomic_json


def run(subject,root,output):
    boundary=json.loads((root/'input_boundary'/subject/'card.json').read_text())
    sensor_path=root/'input_boundary'/subject/'fit_only_interictal_channel_selection.json'
    sensor=json.loads(sensor_path.read_text());bounds=boundary['phase_boundaries']
    cards=[json.loads(p.read_text())|{'card_path':str(p)} for p in sorted((root/'measurements'/subject).glob('*/card.json'))]
    candidates=[c for c in cards if c['block_start']>=bounds['60pct'] and c['available_time']<=bounds['70pct'] and c['cache_path']]
    if not candidates:raise ValueError('No new-measurement INNER block available')
    checks=[]
    for c in [candidates[0],candidates[-1]] if len(candidates)>1 else candidates:
        if c['selected_contacts']!=sensor['fit_refined_selection']:raise ValueError('FIT sensor mismatch')
        row=next(b for b in boundary['block_sources'] if b['block']==c['block'])
        src=json.loads(Path(row['manifest']).read_text())['source']
        if sha(c['cache_path'])!=c['cache_sha256']:raise ValueError('Changed new cache')
        reader=EpilepsiaeBlockReader(Path(src['raw_path']),Path(src['head_path']))
        spec=BlockSpec('epilepsiae',subject,c['record_name'],src['raw_path'],src['head_path'],src['gpu_path'],c['lagpat_path'],c['packed_path'],c['block_start'],reader.native_rate_hz)
        universe,_,labels=load_universe(spec);plan=build_view_plan(reader,universe,[reader.index[clean_contact(n)] for n in labels])
        manifest=json.loads(Path(c['cache_path']).with_suffix('.manifest.json').read_text())
        with np.load(src['gpu_path'],allow_pickle=True) as z:
            detections={str(n):np.asarray(z['whole_dets'][i],float).reshape(-1,2) for i,n in enumerate(z['chns_names'])}
        width={'epilepsiae_1096':.18,'epilepsiae_1125':.15,'epilepsiae_253':.3}[subject]
        rebuilt=pack(detections,c['selected_contacts'],reader.n_samples/reader.native_rate_hz,width)
        stored=np.concatenate((np.load(c['packed_path']),np.asarray(c['segment_crossing_exclusions'],float).reshape(-1,2)))
        stored=stored[np.argsort(stored[:,0])]
        if not np.array_equal(rebuilt,stored):raise ValueError('Full packed event clock replay failed')
        with np.load(c['cache_path']) as z:
            valid=np.flatnonzero(z['has_waveform'])
            for event in valid[np.linspace(0,len(valid)-1,min(2,len(valid))).astype(int)]:
                fs=reader.native_rate_hz;pad=int(round(FILTER_PAD_SECONDS*fs));nctx=manifest['n_context_samples']
                first=int(np.rint(float(z['core_start_seconds'][event])*fs))-manifest['core_offset_samples']-pad
                stop=first+nctx+2*pad
                limited=CutoffReader(reader,reader.n_samples)
                raw=build_event_views(limited,universe,first,stop,plan=plan)
                matched={view:bool(np.array_equal(raw[view][:,pad:pad+nctx].astype(np.float16),z['waveform_'+view][event],equal_nan=True)) for view in manifest['stored_views']}
                if not all(matched.values()):raise ValueError('New measurement raw cutoff replay failed')
                try:limited.read(reader.n_samples-1,reader.n_samples+1,plan.picks)
                except PermissionError:rejected=True
                else:raise AssertionError('Future raw read accepted')
                checks.append(dict(block=c['block'],phase='INNER',event=int(event),prediction_cutoff=c['available_time'],
                    raw_window_sha256=hashlib.sha256(reader.read(first,stop,plan.picks).tobytes()).hexdigest(),
                    all_raw_requests_at_or_before_cutoff=all(b<=reader.n_samples for a,b in limited.requests),
                    stored_waveform_matches=matched,full_packed_clock_exact=True,future_read_rejected=rejected,
                    source_card=c['card_path'],source_card_sha256=sha(c['card_path'])))
    atomic_json(output,dict(status='COMPLETE',subject=subject,checks=checks,new_fit_sensor_operator=True,
        sensor_path=str(sensor_path),sensor_sha256=sha(sensor_path),
        human_design_pilot_input_qualified=True,
        qualification_scope='FIT-selected sensor operator, exact repacked clock and new INNER waveforms with enforced block-end raw cutoff; every token released only after its entire source block closes.',
        historical_detector_bit_rebuild=False,detector_boundary='Historical fixed block-local detector assets retained by hash; no claim of reproducing the historical cusignal stack.',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/build_group_event_state_v039_fit_measurement.py',ROOT/'scripts/audit_group_event_state_v039_input_boundary.py']},
        development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False))
    print(json.dumps(dict(status='COMPLETE',subject=subject,raw_checks=len(checks))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.subject,a.root,a.output)
