#!/usr/bin/env python3
"""Figure only: committed late-Z-refill prefix, with the verified native E2 observer."""
import hashlib
import json
import argparse
import os
import pickle
import time
from pathlib import Path
import numpy as np
import psutil
import plot_topic4_m_parameter_modes as f

WINDOW=f.ROOT/'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913'
SOURCE=f.OUT/'runs/e0_t0_s9108401'
OUT=WINDOW/'late_Z_refill_fig5'


def main():
    blob=(SOURCE/'checkpoint.pkl').read_bytes()
    saved=pickle.loads(blob)
    end=int(saved['engine']['step']);sec=end*.0001
    tr=saved['tracker']
    if tr['release_s'] is None or sec < tr['release_s']+2 or not tr['recoveries']:
        print(json.dumps(dict(status='WAITING_FOR_COMMITTED_RETURN',committed_s=sec,
            restore_s=tr['restore_s'],release_s=tr['release_s'])))
        return False
    assert saved['identity']==f.read(f.OUT/'protocol.json')['identity']
    r=dict(status='COMMITTED_PREFIX_FOLLOWUP_RUNNING',job=saved['job'],tracker=tr,
        end_s=sec,identity=saved['identity'])
    del saved
    a=f.load(SOURCE,end_step=end)
    high=WINDOW/'early_energy_high_resolution_replay'
    qa=f.read(high/'qa.json');assert qa['status']=='PASS'
    raw=np.load(qa['field_path'],mmap_mode='r')
    assert raw.shape==(120000,400)
    assert np.array_equal(raw.reshape(12000,10,400).sum(1),a['field_1ms'][:12000])
    assert tr['restore_s']>12 and r['job']['eta_m']==.005 and r['job']['tau_M_s']==1
    a['early_field_0p1ms']=raw
    a['stagger_close_stage_labels']=True
    a['early_field_source']=dict(path=qa['field_path'],sha256=qa['field_sha256'],
        recording_window_s=[0,12],QA=str(high/'qa.json'),
        same_prefix_through_12s_verified=True,
        original_late_refill_trajectory_unchanged=True)
    metrics=f.analyze(a,r)
    destination=OUT/f'through_{sec:.1f}s'
    f.render(a,r,metrics,destination/'figures',f.grid_summary())
    f.render(a,r,metrics,destination/'figures',f.grid_summary(),
        stem='fig5_entry_zoom',time_window=[max(0,tr['entries'][0]['onset_s']-3),
            tr['entries'][0]['onset_s']+3])
    f.render(a,r,metrics,destination/'figures',f.grid_summary(),
        stem='fig5_refill_zoom',time_window=[tr['restore_s']-3,sec])
    qualification=dict(status='COMMITTED_RETURN_PREFIX_FOLLOWUP_RUNNING',
        source_run=str(SOURCE),source_duration_s=sec,source_checkpoint_step=end,
        source_checkpoint_sha256=hashlib.sha256(blob).hexdigest(),
        physical_trajectory_complete=False,new_simulations=0,new_F_samples=0,
        complete_layout_not_complete_followup=True,
        early_observer_same_prefix_0_to_12s_exact=True,
        entries=tr['entries'],recoveries=tr['recoveries'],restore_s=tr['restore_s'],
        release_s=tr['release_s'],metrics=f.safe(metrics),
        figure=str(destination/'figures/fig5.png'),
        agent_visual_review='PENDING',human_review='PENDING',
        full_scientific_acceptance='NOT_ESTABLISHED_ENERGY_AND_NATIVE_BURST_GAPS')
    for path in destination.glob('*_metadata.json'):
        d=f.read(path)
        d.update(physical_trajectory_complete=False,
            high_resolution_observation_producer=str(Path(__file__).resolve()),
            high_resolution_observation_producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            full_scientific_acceptance=qualification['full_scientific_acceptance'])
        f.write(path,d)
    f.write(destination/'prefix_qualification.json',qualification)
    f.write(OUT/'latest.json',qualification)
    p=destination/'figures/README.md'
    p.write_text(p.read_text()+f'\n本图只使用原M网格的已保存{sec:.1f}秒前缀。首次高态后先保留60秒原生演化，再按原协议只补Z；恢复后继续原定随访，未观察到的第五状态不补画。E2原生0.1ms计数仅复用已核查完全一致的0–12秒共同前缀，不新增仿真或独立样本。\n')
    print(json.dumps(qualification,ensure_ascii=False))
    return True


def wait_then_plot():
    f.write(OUT/'analysis_status.json',dict(status='WAITING_FOR_COMMITTED_RETURN',
        pid=os.getpid(),source=str(SOURCE),new_simulations=0,
        purpose='One finite postprocessing job for an already-running original M-grid trajectory.'))
    last_stamp=None
    while True:
        progress=f.read(SOURCE/'progress.json')
        stamp=(SOURCE/'checkpoint.pkl').stat().st_mtime_ns
        if progress.get('recoveries') and stamp!=last_stamp:
            last_stamp=stamp
            if main():
                f.write(OUT/'analysis_status.json',dict(status='COMPLETE_PENDING_VISUAL_REVIEW',
                    pid=os.getpid(),source=str(SOURCE),new_simulations=0))
                return
        p=psutil.Process(progress['pid'])
        if p.status()==psutil.STATUS_ZOMBIE or not any(
                Path(arg).name==SOURCE.name+'.json' for arg in p.cmdline()):
            raise RuntimeError('Source worker stopped before committed return was available.')
        time.sleep(20)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    if args.wait:wait_then_plot()
    else:main()
