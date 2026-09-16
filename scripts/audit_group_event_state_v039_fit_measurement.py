#!/usr/bin/env python3
"""Re-fit the legacy contact-selection procedure using only closed FIT blocks."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.group_event_analysis import legacy_refine_counts_from_gpu_npz_paths, select_core_channels_by_event_count
from src.topic5_group_event_state.v035.contracts import atomic_json


def main(subject,root,interictal_only=False):
    directory=root/'input_boundary'/subject
    boundary=json.loads((directory/'card.json').read_text());paths=[]
    with np.load(boundary['release_table']) as z: support=z['support'].copy()
    for row in boundary['block_sources']:
        if row['end']<=boundary['phase_boundaries']['60pct']:
            exposure=np.maximum(0,np.minimum(support[:,1],row['end'])-np.maximum(support[:,0],row['start'])).sum()
            if interictal_only and exposure<row['end']-row['start']-1e-5: continue
            m=json.loads(Path(row['manifest']).read_text());paths.append(m['source']['gpu_path'])
    suffix='_interictal' if interictal_only else ''
    report_path=directory/f'fit_only{suffix}_channel_selection.json'
    start=time.time();counts_path=directory/f'fit{suffix}_refine_counts.npz'
    if counts_path.exists():
        previous_path=report_path
        if not previous_path.exists(): raise ValueError('Unbound count cache; recompute in a new output directory')
        previous=json.loads(previous_path.read_text())
        if previous.get('counts_sha256')!=hashlib.sha256(counts_path.read_bytes()).hexdigest(): raise ValueError('Count cache hash changed')
        if previous.get('fit_input_hashes')!={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}: raise ValueError('FIT input set or bytes changed')
        with np.load(counts_path) as z:
            if z['fit_end'].item()!=boundary['phase_boundaries']['60pct']: raise ValueError('cached FIT boundary changed')
            names=z['all_channels'].astype(str).tolist();counts=z['refined_counts'].copy();initial=z['initial_selected'].astype(str).tolist()
    else:
        out=legacy_refine_counts_from_gpu_npz_paths(paths,pick_k=1.,refine_window_sec=.4,refine_all_bool_thresh=.7)
        names=out['all_channels'];counts=out['refined_counts'];initial=out['initial_selected_channels']
        np.savez_compressed(counts_path,all_channels=np.array(names),refined_counts=counts,initial_selected=np.array(initial),fit_end=boundary['phase_boundaries']['60pct'])
    k=.2 if subject=='epilepsiae_253' else 1.
    selected=select_core_channels_by_event_count(events_count=counts,ch_names=names,method='mean_std',k=k)
    idx=json.loads((Path('/data/hfosp_group_event_state_v0_1/dataset')/subject/'index.json').read_text())
    old=[x['lagpat_label'] for x in idx['contacts']]
    report=dict(subject=subject,status='COMPLETE',fit_end=boundary['phase_boundaries']['60pct'],fit_source_count=len(paths),
                initial_selected=initial,fit_refined_selection=selected,old_contact_universe=old,same_contact_set=set(selected)==set(old),
                elapsed_seconds=time.time()-start,fit_input_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
                all_channels=names,refined_counts=counts.tolist(),definition={'initial_pick_k':1.,'refine_window_sec':.4,'refine_all_bool_thresh':.7,'final_pick_k':k},
                counts_path=str(counts_path),counts_sha256=hashlib.sha256(counts_path.read_bytes()).hexdigest(),
                source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/group_event_analysis.py']},
                original_population_selection_causal=False,interictal_complete_blocks_only=interictal_only,
                development_targets_read=False,sealed_partition_opened=False)
    atomic_json(report_path,report)
    print(json.dumps({key:report[key] for key in ['subject','fit_source_count','fit_refined_selection','old_contact_universe','same_contact_set','elapsed_seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',required=True);p.add_argument('--root',type=Path,default=Path('/data/hfosp_group_event_state_v0_3_9_transition_transfer'))
    p.add_argument('--interictal-only',action='store_true')
    args=p.parse_args();main(args.subject,args.root,args.interictal_only)
