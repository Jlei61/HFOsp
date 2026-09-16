#!/usr/bin/env python3
"""Rebuild the mature anatomy decoder's calibration cache from new sensors."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from build_group_event_state_v037_strict_decoder_cache import _anatomical_plane,_anatomy_tissue_layout,_densify,_split_prefix
from src.seeg_coord_loader import load_subject_coords
from src.topic5_group_event_state.v039.human_data import contained_events
from src.topic5_group_event_state.v035.contracts import atomic_json
from src.topic5_lbss_rnn_v0_2 import build_pool_contract,strong_component_audit


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(subject,root):
    boundary=json.loads((root/'input_boundary'/subject/'card.json').read_text());end=boundary['phase_boundaries']['20pct']
    selection=root/'input_boundary'/subject/'fit_only_interictal_channel_selection.json'
    names=json.loads(selection.read_text())['fit_refined_selection']
    geometry=load_subject_coords('epilepsiae',subject.split('_')[1],names,allow_voxel_fallback=False)
    if geometry.coord_units!='mm':raise ValueError('Decoder distance model requires documented millimetres')
    coords=geometry.coords_array_in_requested_order;columns=np.flatnonzero(np.isfinite(coords).all(-1))
    if len(columns)<6:raise ValueError('Fewer than six anatomically mapped contacts')
    mapped_names=np.array(names)[columns];coords=coords[columns]
    with np.load(boundary['release_table']) as z:support=z['support']
    expected=[r for r in boundary['block_sources'] if r['end']<=end]
    rows=[];sources=[];n_missing_fine=0
    for row in expected:
        p=root/'measurements'/subject/f"block_{row['block']:04d}"/'card.json'
        c=json.loads(p.read_text());sources.append(dict(path=str(p),sha256=sha(p)))
        if c['selected_contacts']!=names:raise ValueError('Mixed sensor universe')
        if not c['cache_path']:continue
        if sha(c['cache_path'])!=c['cache_sha256']:raise ValueError('Changed calibration measurement')
        with np.load(c['cache_path']) as z:
            time=z['event_abs_time'];part=z['participation'][:,columns];delay=z['relative_delay_s'][:,columns]
            complete=~np.any(part&~np.isfinite(delay),axis=1)
            groups=_densify(z['tied_group_id'][:,columns]);count=np.array([len(np.unique(v[v>=0])) for v in groups])
            keep=contained_events(time,time+z['core_seconds_raw'],support)&complete&(part.sum(-1)>=3)&(count>=2)&(time<end)
            rows.append((time[keep],groups[keep],delay[keep],count[keep]));n_missing_fine+=int((~complete).sum())+len(c['segment_crossing_exclusions'])
    times=np.concatenate([r[0] for r in rows]);order=np.argsort(times,kind='stable');times=times[order]
    ranks=np.concatenate([r[1] for r in rows])[order];lag=np.concatenate([r[2] for r in rows])[order];count=np.concatenate([r[3] for r in rows])[order]
    split=_split_prefix(len(times));xy,center,axes=_anatomical_plane(coords);sigma,nodes,H=_anatomy_tissue_layout(xy,seed=20260904)
    distance=np.linalg.norm(nodes[:,None]-nodes[None],axis=-1).astype(np.float32);pools=build_pool_contract(distance)
    graph=strong_component_audit(pools.local_mask,supported=np.abs(H).sum(0)>0)
    if not graph['all_nodes_one_strong_component']:raise ValueError('Disconnected anatomical decoder graph')
    target=root/'decoder_rebuilt'/subject/'cache'/f'{subject}__anatomy';target.mkdir(parents=True)
    np.savez_compressed(target/'plane.npz',contacts_xy_mm=xy,contacts_xyz_mm=coords.astype(np.float32),anatomical_pca_center=center,
        anatomical_pca_axes=axes,nodes_xy_mm=nodes,H=H,D_mm=distance,sigma_mm=np.array([sigma],np.float32),scale_mm=np.array([1.],np.float32),
        latent_domain_version=np.array(['ANATOMY_ONLY_FIT_SENSOR_V039']))
    source_index=np.arange(len(times));neutral=np.full(len(times),-1,np.int8)
    np.savez_compressed(target/'events_raw.npz',ranks=ranks,base_split=split,event_group_count=count,event_lag_raw=lag,
        event_abs_time=times,event_source_index=source_index,event_dataset_index=source_index,contact_names=mapped_names)
    np.savez_compressed(target/'events.npz',ranks=ranks,split=split,mode=neutral,full_train_mode=neutral,
        prefix_posterior=np.ones((len(times),1),np.float32),prefix_entropy=np.zeros(len(times),np.float32),event_abs_time=times,
        event_source_index=source_index,event_dataset_index=source_index)
    provenance=dict(status='COMPLETE',subject=subject,fit_id=f'{subject}__anatomy',scope='anatomy_only_new_fit_sensors',n_contacts=len(columns),
        n_nodes=len(nodes),n_events=len(times),n_train=int((split==0).sum()),n_validation=int((split==1).sum()),n_test=int((split==2).sum()),
        selected_contacts=names,joint_contacts=mapped_names.tolist(),excluded_unmapped_contacts=[n for i,n in enumerate(names) if i not in columns],
        contact_vocabulary_event_selected=True,contact_vocabulary_selection_scope='pure interictal FIT only, frozen before held-out measurement',
        decoder_max_used_time=float(times.max()),state_fit_start_20pct=end,strictly_pre_state_fit=bool(times.max()<end),
        spatial_target_scope='anatomically mapped subset of new sensor universe, no synthetic coordinates for missing contacts',
        geometry_source=geometry.provenance,geometry_source_sha256=sha(geometry.provenance['source_path']),geometry_uses_event_values=False,
        sensor_selection=str(selection),sensor_selection_sha256=sha(selection),source_cards=sources,n_missing_fine_events=n_missing_fine,
        plane_sha256=sha(target/'plane.npz'),events_sha256=sha(target/'events.npz'),events_raw_sha256=sha(target/'events_raw.npz'),
        development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False,target_values_read=False,
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/build_group_event_state_v037_strict_decoder_cache.py',ROOT/'src/seeg_coord_loader.py']})
    atomic_json(target/'provenance.json',provenance)
    parent=target.parents[1]
    atomic_json(parent/'INPUT_CACHE_MANIFEST.json',dict(fits={provenance['fit_id']:provenance},split_contract='80/10/10 chronological inside pre20 calibration only'))
    atomic_json(parent/'RUN_CONTRACT.json',dict(trainer='scripts/train_topic5_lbss_unit_v0_2.py',arm='L3_LOCAL_PLUS_LEARNED_LR',
        seeds=[0,1,2],epochs_warmup=10,epochs_rewire=40,epochs_freeze_cap=300,lr=.006,decoder_selection='calibration validation only; never select by H2a/SELECTION',
        budget_interpretation='Ceiling is an optimization limit, not a biological negative',development_targets_read=False,sealed_partition_opened=False,seizure_targets_read=False))
    print(json.dumps({k:provenance[k] for k in ['status','subject','n_contacts','n_events','n_train','n_validation','n_test','excluded_unmapped_contacts']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();build(a.subject,a.root)
