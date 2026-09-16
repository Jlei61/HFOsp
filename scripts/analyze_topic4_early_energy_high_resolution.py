#!/usr/bin/env python3
"""Rebuild E2 from integration-step spatial counts after identical replay QA."""
from pathlib import Path
import argparse
import importlib
import os
import time
import numpy as np
import psutil
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import plot_topic4_m_parameter_modes as f
from run_topic4_m_parameter_modes import sha

WINDOW=f.ROOT/'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913'
OUT=WINDOW/'early_energy_high_resolution_replay'


def wait_for_replay():
    while not (OUT/'qa.json').exists():
        runtime=f.read(OUT/'runtime.json') if (OUT/'runtime.json').exists() else None
        if runtime:
            try:
                p=psutil.Process(runtime['pid'])
                live=p.status()!=psutil.STATUS_ZOMBIE and any(
                    Path(x).name=='replay_topic4_early_energy_high_resolution.py' for x in p.cmdline())
            except psutil.Error:live=False
            if not live:
                if (OUT/'qa.json').exists():break
                f.write(OUT/'analysis_status.json',dict(status='REPLAY_TERMINATED_WITHOUT_QA',pid=os.getpid()))
                raise RuntimeError('Measurement replay ended without its required observation QA')
        time.sleep(20)


def main():
    global f
    f=importlib.reload(f)
    qa=f.read(OUT/'qa.json');assert qa['status']=='PASS'
    assert sha(qa['field_path'])==qa['field_sha256']
    assert qa['raw_to_1ms_field_exact'] and qa['raw_to_1ms_population_exact']
    source=WINDOW/'early_Z_lookup_dense_figures/runs/early_z_refill_s9108401'
    r=f.read(source/'result.json');assert r['end_s']==37.5
    a=f.load(source,end_step=375000);metrics=f.analyze(a,r)
    old=f.early(a,metrics)
    raw=np.load(qa['field_path'],mmap_mode='r')
    assert raw.shape==(120000,400)
    assert np.array_equal(raw.reshape(12000,10,400).sum(1),a['field_1ms'][:12000])
    a['early_field_0p1ms']=raw
    a['early_field_source']=dict(path=qa['field_path'],sha256=qa['field_sha256'],
        recording_window_s=[0,12],QA=str(OUT/'qa.json'),same_full_trajectory=True)
    corrected=f.early(a,metrics)
    assert old['baseline_s']==corrected['baseline_s'] and old['target_s']==corrected['target_s']
    assert old['contact_robust_z']==corrected['contact_robust_z']
    old_db=np.asarray(old['native_bandpower_change_db'],float)
    new_db=np.asarray(corrected['native_bandpower_change_db'],float)
    valid=np.isfinite(old_db)&np.isfinite(new_db)
    report=dict(status='COMPLETE_SAME_TRAJECTORY_OBSERVATION_COMPARISON',
        baseline_s=old['baseline_s'],target_s=old['target_s'],
        before=old,after=corrected,
        native_db_change_quantiles=np.quantile(new_db[valid]-old_db[valid],[0,.1,.5,.9,1]).tolist(),
        contact_result_unchanged=True,patient_Fig3C_unchanged=True,
        statistical_unit='Same one noise trajectory at two observation resolutions.',
        new_F_samples=0,full_trajectory_end_s=37.5,high_resolution_recording_end_s=12.,
        original_intervention_and_state_classification_unchanged=True,
        source_QA=str(OUT/'qa.json'),agent_visual_review='PENDING',human_review='PENDING')
    f.write(OUT/'energy_comparison.json',report)
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,4.4))
    for ax,values,label in zip(axes,[old_db,new_db,new_db-old_db],
            ['1-ms spike counts','0.1-ms spike counts','0.1 ms minus 1 ms']):
        limit=20 if label!='0.1 ms minus 1 ms' else 10
        im=ax.imshow(np.ma.masked_invalid(values.reshape(20,20)),origin='lower',extent=[0,20,0,20],
            cmap='RdBu',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),interpolation='nearest')
        ax.set(title=label,xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20])
        if ax is axes[0]:ax.set_ylabel('y (mm)')
        fig.colorbar(im,ax=ax,pad=.03,shrink=.85,extend='both').set_label('Power change (dB)' if limit==20 else 'Difference (dB)')
    fig.tight_layout();fig.savefig(folder/'native_energy_resolution.png',dpi=180,bbox_inches='tight')
    fig.savefig(folder/'native_energy_resolution.pdf',bbox_inches='tight');plt.close(fig)
    (folder/'README.md').write_text('### native_energy_resolution.png / .pdf\n'
        '同一个已验证的SNN轨迹、同一基线与早期窗口，比较1ms计数和原生0.1ms计数的1–150Hz空间功率变化。第三幅为两种观察结果的差，不改变模型、患者参考或电极读出。\n'
        '**关注点**：功率增强和降低都保留；这项重放只解决时间分辨率的观测问题，不增加独立样本。\n')
    f.render(a,r,metrics,OUT/'candidate/figures',f.grid_summary())
    for p in (OUT/'candidate').glob('*_metadata.json'):
        d=f.read(p);d['high_resolution_observation_producer']=str(Path(__file__).resolve())
        d['high_resolution_observation_producer_sha256']=sha(__file__)
        d['physical_trajectory_complete']=True
        d['full_scientific_acceptance']='PENDING_ENERGY_AND_OSCILLATION_REVIEW'
        f.write(p,d)
    f.write(OUT/'analysis_status.json',dict(status='COMPLETE_PENDING_VISUAL_REVIEW',pid=os.getpid(),
        comparison=str(OUT/'energy_comparison.json'),figure=str(OUT/'candidate/figures/fig5.png')))
    print(f.safe(dict(native_increased_cells_1ms=old['native_cells_with_increased_power'],
        native_increased_cells_0p1ms=corrected['native_cells_with_increased_power'],
        contacts_increased=corrected['n_model_contacts_above_baseline'],
        native_db_change_quantiles=report['native_db_change_quantiles'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    if args.wait:wait_for_replay()
    main()
