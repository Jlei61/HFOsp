#!/usr/bin/env python3
"""Same-site PRE-onset assay; preserve matched sham and every stimulation site."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
import json
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_topic4_rev21_fig5_checkpoints import load_selected_contract, build_selected_substrate, sha256
from scripts.run_topic4_zm_perturbation_worker import _continue, _json_safe
from scripts.run_topic4_dual_core_spatial_z_bifurcation import atomic_json, atomic_npz
from src.snn_engine import checkpoint as ckpt
from src.topic4_zm_perturbation import response_metrics, select_packet, in_window_ignition, _descendant

BASE=Path('/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z/perturbation')
WORK={}


def probe_site(i):
    sub,transition,state,sham,sites=(WORK[k] for k in ('sub','transition','state','sham','sites'))
    packet=select_packet(sub.positions_e,sites[i],n_cells=16,
                         radius_mm=float(transition['perturbation']['packet_radius_mm']))
    probe,_=_continue(sub,transition,state,duration_ms=200.,packet=packet)
    dt=float(sub.engine['dt'])
    cm=sub.extras['cmrun']
    pa,adt=cm.active_fraction(np.asarray(probe['E_spk_bool'],bool),dt,cm.BIN_MS)
    m=response_metrics(probe,sham,dt_ms=dt,positions_e=sub.positions_e,
                       packet_mask=packet,packet_xy=sites[i],
                       envelope_probe=np.zeros((15,1)),envelope_sham=np.zeros((15,1)),
                       envelope_dt_ms=2.,inject_step=0,split_ms=50.,window_ms=200.)
    flags=in_window_ignition(pa,WORK['sham_active'],active_dt_ms=float(adt),
                            detector_threshold=sub.detector_threshold,inject_ms=0.,window_ms=200.,
                            probe_rate_hz=np.asarray(probe['rate_E'],float),dt_ms=dt,
                            es_thresh_hz=float(transition['simulation']['es_thresh_hz']),
                            es_dur_ms=float(transition['simulation']['es_dur_ms']))
    descendant=_descendant(probe,sham,packet,0)
    excess_time=descendant.sum(axis=1).astype(float)-np.asarray(sham['E_spk_bool'],bool).sum(axis=1)
    arrays={'early_field':m['excess_per_neuron_early'],'full_field':m['excess_per_neuron'],
            'excess_spikes_per_step':excess_time.astype(np.float32)}
    record={'site_index':i,'site_xy_mm':sites[i].tolist(),'dose_cells':int(packet.sum()),
            **{k:v for k,v in m.items() if not isinstance(v,np.ndarray)},**flags}
    path=WORK['out']/'per_site'/f'site_{i:02d}'
    atomic_npz(path.with_suffix('.npz'),**arrays)
    atomic_json(_json_safe(record),path.with_suffix('.json'))
    return record,arrays


def main():
    start=time.time()
    art=Path('/home/honglab/leijiaxin/HFOsp')
    base=art/'results/topic4_sef_hfo/data_driven_dual_core_zm_transition'
    source=base/'timescale/workers/rev21_ts_tz3000_ta500_topology_2542_dynamics_2642.json'
    config,candidate,meta,npz=load_selected_contract(
        ROOT/'config/topic4_rev21_dual_core_zm_transition.json',
        base/'timescale/candidate_manifest.json',source,art)
    manifest_path=BASE/'preonset_20260905/checkpoints/checkpoint_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    cp=manifest['checkpoints']['pre_onset']
    if not manifest['pre_onset_replay_exact'] or sha256(Path(cp['path']))!=cp['sha256']:
        raise RuntimeError('pre-onset checkpoint failed provenance')
    lowpath=BASE/'dualcore_rev21_low_activity_random_sites.json'
    lowmeta=json.loads(lowpath.read_text())
    if sha256(lowpath.with_suffix('.npz'))!=lowmeta['npz']['sha256']:
        raise RuntimeError('low-state reference arrays changed')
    if lowmeta['checkpoint_manifest']['sha256']!=manifest['parent_manifest']['sha256']:
        raise RuntimeError('low and pre-onset states have different source trajectories')
    if lowmeta['dose_cells']!=16 or lowmeta['window_ms']!=200. or not lowmeta['resumed_sham_exact']:
        raise RuntimeError('low-state probe contract mismatch')
    with np.load(lowpath.with_suffix('.npz'),allow_pickle=False) as a:
        low={k:a[k] for k in a.files}
    transition,sub=build_selected_substrate(config,candidate,topology_seed=2542,dynamics_seed=2642,artifact_root=art)
    if not np.array_equal(np.asarray(sub.positions_e,np.float32),low['positions_E']):
        raise RuntimeError('realized substrate neuron ordering mismatch')
    state=ckpt.load(cp['path'])
    dt=float(sub.engine['dt'])
    if cp['time_ms']+200.>=float(meta['model_ictal_rev21']['scientific_onset_ms']):
        raise RuntimeError('response window must finish strictly before onset')
    out=BASE/'preonset_20260905'
    (out/'per_site').mkdir(parents=True,exist_ok=True)
    frozen={'status':'FROZEN_BEFORE_PROBES','checkpoint':cp,'dose_cells':16,'window_ms':200.,
            'early_response_window_ms':[0.,50.],
            'sites':low['site_xy_mm'].tolist(),'state_contrast':['low_activity','pre_onset'],
            'selection':'pre-existing stratified sites and pre-existing 16-cell dose; all sites retained',
            'scope':'one trajectory, one continuation per site, matched future noise; sites are not independent patients',
            'sources':{str(p):sha256(p) for p in [source,npz,lowpath,lowpath.with_suffix('.npz'),manifest_path,Path(__file__)]}}
    atomic_json(frozen,out/'frozen_probe_contract.json')
    sham,_=_continue(sub,transition,state,duration_ms=200.)
    offset=int(round(cp['time_ms']/dt))
    with np.load(npz,allow_pickle=False) as a: ref=a['transition_rate_E_hz_raw'][offset:offset+len(sham['rate_E'])]
    if not np.array_equal(np.asarray(sham['rate_E'],np.float32),ref):
        raise RuntimeError('pre-onset sham is not exact source continuation')
    cm=sub.extras['cmrun']
    sa,_=cm.active_fraction(np.asarray(sham['E_spk_bool'],bool),dt,cm.BIN_MS)
    WORK.update(sub=sub,transition=transition,state=state,sham=sham,
                sites=low['site_xy_mm'].astype(float),sham_active=sa,out=out)
    results=[]
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('fork')) as pool:
        for f in as_completed([pool.submit(probe_site,int(i)) for i in low['site_index']]):
            record,arrays=f.result();results.append((record,arrays))
            print(json.dumps({'done':len(results),'site':record['site_index'],'early_excess':record['excess_spikes_early'],'seconds':round(time.time()-start)}),flush=True)
    results.sort(key=lambda x:x[0]['site_index'])
    rows=[r for r,a in results]
    arrays={'positions_E':low['positions_E'],'contact_xy_mm':low['contact_xy_mm'],
            'site_index':low['site_index'],'site_xy_mm':low['site_xy_mm'],
            'low_early_field':low['excess_per_neuron_early'],
            'pre_early_field':np.stack([a['early_field'] for r,a in results]),
            'low_full_field':low['excess_per_neuron_full'],
            'pre_full_field':np.stack([a['full_field'] for r,a in results]),
            'pre_excess_spikes_per_step':np.stack([a['excess_spikes_per_step'] for r,a in results]),
            'time_after_pulse_ms':np.arange(len(sham['rate_E']))*dt,
            'sham_rate_hz':np.asarray(sham['rate_E'],np.float32),
            'low_evaluable':low['e1_evaluable'],'pre_evaluable':np.array([r['e1_evaluable'] for r in rows])}
    target=out/'state_contrast.npz'
    atomic_npz(target,**arrays)
    summary={**frozen,'status':'PRE_ONSET_PAIRED_PERTURBATION_COMPLETE',
             'state_times_ms':{'low_activity':lowmeta['checkpoint']['time_ms'],'pre_onset':cp['time_ms']},
             'pre_sham_exact':True,'all_sites_retained':len(rows)==16,'rows':rows,
             'low_rows':lowmeta['rows'],'npz':{'path':str(target),'sha256':sha256(target)},
             'wall_seconds':time.time()-start}
    atomic_json(_json_safe(summary),out/'state_contrast.json')


if __name__=='__main__': main()
