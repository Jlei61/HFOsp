#!/usr/bin/env python3
"""Measure background even when a raw block has no packed event windows."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.topic5_group_event_state.cache import BlockSpec,load_universe,_background_anchors
from src.topic5_group_event_state.contract import ANALYSIS_BANDS_HZ,supported_band_mask
from src.topic5_group_event_state.raw_views import EpilepsiaeBlockReader,build_view_plan,clean_contact
from src.topic5_group_event_state.v035.contracts import atomic_json


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',required=True);p.add_argument('--block',type=int,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    card_path=a.root/'measurements'/a.subject/f'block_{a.block:04d}'/'card.json';card=json.loads(card_path.read_text())
    if card['n_group_windows_before_raw_segments']!=0:raise ValueError('This overlay is for truly zero-event blocks')
    boundary=json.loads((a.root/'input_boundary'/a.subject/'card.json').read_text());row=next(b for b in boundary['block_sources'] if b['block']==a.block)
    src=json.loads(Path(row['manifest']).read_text())['source'];reader=EpilepsiaeBlockReader(Path(src['raw_path']),Path(src['head_path']))
    spec=BlockSpec('epilepsiae',a.subject,card['record_name'],src['raw_path'],src['head_path'],src['gpu_path'],card['lagpat_path'],card['packed_path'],card['block_start'],reader.native_rate_hz)
    universe,_,labels=load_universe(spec);plan=build_view_plan(reader,universe,[reader.index[clean_contact(n)] for n in labels])
    bands=tuple(ANALYSIS_BANDS_HZ);support=supported_band_mask(reader.native_rate_hz);available=np.array([support[b] for b in bands])
    background=_background_anchors(reader,universe,plan,reader.native_rate_hz,np.empty((0,2)),bands,available)
    output=a.root/'background_overlays'/a.subject/f'block_{a.block:04d}.npz';output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    np.savez_compressed(output,background_time_s=background['time_s'],background_features=background['features'])
    atomic_json(output.with_suffix('.json'),dict(status='COMPLETE',sha256=sha(output),source_card=str(card_path),source_card_sha256=sha(card_path),
        feature_names=background['feature_names'],n_background_windows=len(background['time_s']),source_sha256=sha(__file__),
        raw_path=src['raw_path'],head_sha256=sha(src['head_path']),available_time=card['available_time'],
        development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False))
    print(json.dumps(dict(status='COMPLETE',subject=a.subject,block=a.block,background_windows=len(background['time_s']))))
