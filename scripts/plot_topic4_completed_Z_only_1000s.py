#!/usr/bin/env python3
"""Full Fig5 layout for the actual completed1000s Z-only restoration control."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import json
import numpy as np
import plot_topic4_m_parameter_modes as f
import run_topic4_m_parameter_modes as core

ROOT=f.ROOT
RESET=ROOT/'results/topic4_sef_hfo/reset_state_diagnosis_20260911'
PREFIX=ROOT/'results/topic4_sef_hfo/fig5_manual_core_release_v1/weak_fast_z_refill_recurrence_v2/runs/weak_fast_z_refill_recurrence.npz'
OUT=ROOT/'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/completed_Z_only_1000s_fig5'


def main():
    result=f.read(RESET/'runs/z_only_long/result.json')
    assert result['status']=='COMPLETE' and result['end_s']==1000 and result['recurrence_onset_s'] is None
    with np.load(f.OUT/'geometry.npz') as geo:geometry={k:geo[k] for k in geo.files}
    with np.load(RESET/'geometry.npz') as geo:
        for key in ['sample_ids','sample_source_indices','positions_e','region_counts']:
            assert np.array_equal(geo[key],geometry[key]),key
    ns=10000000;cut=2380000;nch=len(geometry['contact_names'])
    a=dict(spikes_1ms=np.empty((ns//10,2),np.uint16),
        regions_1ms=np.empty((ns//10,6),np.uint16),raster=np.empty((ns,80),np.bool_),
        Z=np.empty((ns//50,9),float),M=np.empty((ns//50,4),float),currents=np.empty((ns//50,3),float),
        lfp_raw=np.empty((ns//5,nch),float),field_5ms=np.empty((ns//50,400),np.uint32),
        slow_time_ms=np.arange(ns//50)*5.,lfp_time_ms=np.arange(ns//5)*.5,
        time_ms=np.arange(ns//10)+.5)
    with np.load(PREFIX) as source:
        a['spikes_1ms'][:cut//10,0]=np.rint(source['rate_e_hz'][:cut].reshape(-1,10).mean(1)*32).astype(np.uint16)
        a['spikes_1ms'][:cut//10,1]=np.rint(source['rate_i_hz'][:cut].reshape(-1,10).mean(1)*8).astype(np.uint16)
        a['regions_1ms'][:cut//10]=source['region_spikes_1ms'][:cut//10]
        a['field_1ms']=source['field_e_count_1ms'][:cut//10]
        a['field_5ms'][:cut//50]=a['field_1ms'].reshape(-1,5,400).sum(1,dtype=np.uint32)
        a['raster'][:cut]=source['sample_spikes'][:cut,geometry['sample_source_indices']]
        a['Z'][:cut//50]=source['z_stats'][:cut//50,:9]
        a['M'][:cut//50]=source['m_stats'][:cut//50][:,[0,5,6,7]]
        a['currents'][:cut//50]=source['currents_5ms'][:cut//50,:3]
        a['lfp_raw'][:cut//5]=source['lfp_raw'][:cut//5]
        assert np.array_equal(a['slow_time_ms'][:cut//50],source['z_time_ms'][:cut//50])
        assert np.array_equal(a['lfp_time_ms'][:cut//5],source['lfp_time_ms'][:cut//5])
    expected=cut;paths=[]
    for path in sorted((RESET/'runs/z_only_long/chunks').glob('*.npz')):
        with np.load(path) as source:
            lo,hi=int(source['start_step']),int(source['end_step']);assert lo==expected
            expected=hi
            for key,scale in [('spikes_1ms',10),('regions_1ms',10),('raster',1),('Z',50),('M',50),('currents',50),('lfp_raw',5)]:
                a[key][lo//scale:hi//scale]=source[key]
            a['field_5ms'][lo//50:hi//50]=source['field_5ms']
            assert np.array_equal(source['slow_time_ms'],a['slow_time_ms'][lo//50:hi//50])
            assert np.array_equal(source['lfp_time_ms'],a['lfp_time_ms'][lo//5:hi//5])
            assert np.array_equal(source['time_ms'],a['time_ms'][lo//10:hi//10])
            assert np.allclose(source['field_time_ms'],np.arange(lo//50,hi//50)*5+2.5,atol=1e-9)
        paths.append(str(path))
    assert expected==ns
    a.update(geometry,native_spatial_mixed_sampling=True,
        native_spatial_sampling_segments=[[0,238,1],[238,1000,5]])
    # Recompute first entry and recovery with the current explicit Fig5 readout
    # definitions. The actual historical intervention remains75.5–76.5s.
    tracker=core.fresh_tracker();tracker.update(restore_s=75.5,release_s=76.5)
    rates=a['spikes_1ms'].reshape(-1,10,2).sum(1)[:,0]/32000/.01
    for i,rate in enumerate(rates):core.tracker_step(tracker,rate,(i+1)*.01,rescue=False)
    assert len(tracker['entries'])==1 and len(tracker['recoveries'])==1
    assert abs(tracker['entries'][0]['confirmation_s']-73.68)<1e-9
    assert tracker['recoveries'][0]['mechanism']=='EXTERNAL_Z'
    for key in ['rates','wall_s','stop_s','last_entry_s','high_bins']:tracker.pop(key,None)
    r=dict(status='COMPLETE_INHERITED_CONTROL',job=dict(name='completed_Z_only_1000s',eta_m=.02,tau_M_s=2,seed=9108401),
        tracker=tracker,end_s=1000)
    metrics=f.analyze(a,r);assert not metrics['full_1_to_5_observed']
    OUT.mkdir(exist_ok=True)
    f.write(OUT/'source_qualification.json',dict(source_result=str(RESET/'runs/z_only_long/result.json'),
        original_prefix=str(PREFIX),prefix_s=[0,238],closed_continuation_chunks=paths,
        continuous_state_and_RNG=True,prefix_238_240_replay_counted_once=True,
        result_status='COMPLETE1000S_WITHOUT_SECOND_HIGH',trajectory=r,
        native_spatial_sampling_segments=a['native_spatial_sampling_segments'],
        no_invented_1ms_field_after238s=True,early_power_uses_actual_original1ms_fields=True,
        new_M40_sample=False,extra_F_samples=0,
        first_recovery_recomputed_from_saved_counts=True,
        tracker_used_for_observation_only=True,inherited_observation_horizon_s=1000,
        full_layout_not_full_scientific_acceptance=True,human_review='PENDING',agent_visual_review='PENDING'))
    f.render(a,r,metrics,OUT/'figures',f.grid_summary())
    f.write(OUT/'metrics.json',metrics)
    for name in ['fig5_metadata.json','fig5_transition_zoom_metadata.json']:
        path=OUT/name
        if path.exists():
            metadata=f.read(path);metadata['source_qualification']=str(OUT/'source_qualification.json');f.write(path,metadata)
    readme=OUT/'figures/README.md'
    readme.write_text(readme.read_text()+'\n此组是已完成的1000秒Z-only继承对照，不是新增M40样本。0–238秒空间场为1ms，后续实际保存为5ms；全部早期1–150Hz计算使用原始1ms场，未上采样后段。⑤因实际未再进入而留空，F沿用当前M40首次进入扫描且不加此对照。\n')
    print(json.dumps({k:metrics[k] for k in ['mode','duration_s','finite_events_before_first_high','finite_events_after_return','full_1_to_5_observed','recoveries']}))


if __name__=='__main__':main()
