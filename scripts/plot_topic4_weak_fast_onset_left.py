#!/usr/bin/env python3
"""Original Fig5 left-panel layout from complete native observations around onset."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import pickle
import time
import numpy as np
from scipy.signal import find_peaks
import plot_topic4_fig5_m_return as drawing

ROOT=Path(__file__).resolve().parents[1]
BASE=drawing.OUT
OUT=BASE/'weak_fast_onset_preview'
ORDER=['SCL9','SCL8','SCL7','SCL6','ICL11','ICL10','ICL9','ICL8','ICL7','ICL6','ICL5','ICL4','ICL3','ICL2','ICL1']
STATIC=['sample_ids','sample_groups','region_counts','cell_e_counts','positions_e','cell_e','centers_mm',
        'contact_names','contact_xy','valid_contacts','shaft_ids','dt_ms']

def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}

def write(p,r):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def extract(checkpoint_path):
    # The writer atomically replaces this file, so an opened descriptor is a complete snapshot.
    with checkpoint_path.open('rb') as stream:pack=pickle.load(stream)
    expected=read(ROOT/'results/topic4_sef_hfo/historical_manual_hard_native_z_v1/substrate.json')['identity']
    assert pack['identity']==expected
    assert pack['job']['eta_m']==.02 and pack['job']['tau_adp_ms']==2000 and not pack['job']['refill']
    o=pack['observations'];clock=float(pack['engine']['absolute_time_ms'])
    with np.load(drawing.BASE/'runs/continuous_refill_release.npz') as f:
        a={k:f[k] for k in STATIC};reference={k:f[k] for k in ['lfp_time_ms','lfp_effective','contact_names','shaft_ids','valid_contacts']}
    a.update(rate_e_hz=o['rates'][:,0],rate_i_hz=o['rates'][:,1],sample_spikes=o['raster'],
        field_e_count_1ms=o['fields'],region_spikes_1ms=o['region_counts'])
    names={'zt':'z_time_ms','zs':'z_stats','zrhs':'z_rhs','inputs':'input_summary','mstats':'m_stats',
        'zf':'z_field_5ms','currents':'currents_5ms','lfp_time':'lfp_time_ms','lfp_raw':'lfp_raw','lfp_effective':'lfp_effective'}
    for source,target in names.items():a[target]=np.asarray(o[source])
    assert len(a['rate_e_hz'])*float(a['dt_ms'])==clock
    assert len(a['field_e_count_1ms'])==round(clock)
    assert a['sample_spikes'].shape[0]==len(a['rate_e_hz'])
    t=pack['tracker'];r=dict(job=pack['job'],eta_m=.02,tau_adp_ms=2000.,
        first_trigger_ms=t['triggers'][0] if t['triggers'] else None,restore_start_ms=None,
        external_input_prefix_matches_M_off=t['prefix_qa'] is not None)
    return a,r,reference,clock

def render(checkpoint_path,start=68.,end=74.,qa=False):
    a,r,reference,clock=extract(checkpoint_path)
    assert clock>=end*1000
    if not qa:
        assert r['first_trigger_ms']==73680.,'Replay onset differs from the archived original.'
    metric=drawing.analyze(a,r)
    rate=a['rate_e_hz'].reshape(-1,100).mean(1);t=(np.arange(len(rate))+.5)*.01
    peaks=find_peaks(rate,height=20,distance=5)[0]
    if qa:anchors=np.linspace(start+.3,end-.15,5).tolist()
    else:anchors=[68.5,70.8,72.8,73.48,73.90]
    snaps=[]
    for k,target in enumerate(anchors):
        if k<2:
            candidates=[v for v in peaks if abs(t[v]-target)<.4 and start+.025<=t[v]<=end-.025]
            if candidates:target=float(t[min(candidates,key=lambda v:abs(t[v]-target))])
        snaps.append(dict(time_s=float(target),label='Native50ms window'))
    assert all(start+.025<=s['time_s']<=end-.025 for s in snaps)
    assert np.all(np.diff([s['time_s'] for s in snaps])>0)
    # Fixed shaft order; preserve every channel slot and the original physical coordinates.
    ids=np.array([int(np.flatnonzero(a['contact_names']==n)[0]) for n in ORDER])
    ref_t=reference['lfp_time_ms']/1000;raw=reference['lfp_effective'][:,ids]
    x=raw-np.median(raw[(ref_t>=.5)&(ref_t<1)],axis=0)
    gain=float(np.max(np.quantile(x,.995,axis=0)-np.quantile(x,.005,axis=0)))
    target=OUT/'layout_qa' if qa else OUT
    drawing.FIG=target/'figures'
    metric['name']='weak_fast_onset' if not qa else 'weak_fast_layout_canary'
    layout=drawing.left(a,r,metric,reference,gain,time_limits=(start,end),snapshot_override=snaps,contact_indices=ids)
    low,high=round(start*10000),round(end*10000)
    selected=np.r_[np.arange(0,60,3),np.arange(60,120,3),np.arange(120,240,6),np.arange(240,300,3)]
    assert layout['raster_spikes']==int(a['sample_spikes'][low:high,selected].sum())
    global_spikes=np.rint(a['rate_e_hz'][low:high]*32000*.0001).reshape(-1,10).sum(1)
    assert np.array_equal(global_spikes,a['field_e_count_1ms'][round(start*1000):round(end*1000)].sum(1))
    meta=dict(status='COMPLETE_NATIVE_WINDOW_PENDING_VISUAL_REVIEW',source=str(checkpoint_path),
        checkpoint_time_ms=clock,time_limits_s=[start,end],first_high_confirmed_s=None if r['first_trigger_ms'] is None else r['first_trigger_ms']/1000,
        eta_m=.02,tau_M_s=2.,external_reset=False,layout=layout,complete90s_run=False,canary_only=qa,
        raw_observations=True,summary_interpolation=False,spatial_spike_conservation=True,
        readout_sampling_ms=.5,raster_sampling_ms=.1,slow_state_sampling_ms=5.,spatial_count_sampling_ms=1.,
        snapshot_selection='First2 nearest population E peaks within0.4s of fixed anchors; remaining fixed times; no spatial pattern selection.',
        producer=str(Path(__file__).resolve()),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(target/'metadata.json',meta)
    # Save the actual plotted window alongside metadata, so later checkpoint replacement cannot remove its source.
    zsel=(a['z_time_ms']>=start*1000)&(a['z_time_ms']<end*1000)
    lsel=(a['lfp_time_ms']>=start*1000)&(a['lfp_time_ms']<end*1000)
    np.savez_compressed(target/'native_window.npz',time_start_s=start,time_end_s=end,
        rate_e_hz=a['rate_e_hz'][low:high],rate_i_hz=a['rate_i_hz'][low:high],sample_spikes=a['sample_spikes'][low:high],
        z_time_ms=a['z_time_ms'][zsel],z_stats=a['z_stats'][zsel],m_stats=a['m_stats'][zsel],
        field_e_count_1ms=a['field_e_count_1ms'][round(start*1000):round(end*1000)],
        lfp_time_ms=a['lfp_time_ms'][lsel],lfp_effective=a['lfp_effective'][lsel],lfp_raw=a['lfp_raw'][lsel],
        **{k:a[k] for k in STATIC})
    file=metric['name']+'_left.png'
    (drawing.FIG/'README.md').write_text(f'### {file}\n原左侧四层布局，仅显示弱快M（ηM=0.02、τM=2秒）的{start:g}–{end:g}秒连续原始记录：A电极电流读出，B连续80神经元raster，C全E与双核Z/M电流，D五个50ms原生二维活动窗。各时间轴与采样标记对齐，电极按固定杆内顺序保留15个槽位；无人工reset，无汇总曲线补造原始读出。'+('本图仅作早期窗口的绘图自查，不是onset结果。' if qa else '红虚线为原10ms在线判据确认的73.68秒，尚未代表完整90秒结果。')+'\n**关注点**：转变前的有限事件、起始阶段的原生空间招募与Z/M变化；不能由该有限窗口宣称自主终止或分岔类型。\n')
    if not qa:write(OUT/'status.json',dict(status='FIGURE_READY_PENDING_VISUAL_REVIEW',figure=str(drawing.FIG/file),window=[start,end]))
    print(json.dumps(meta,indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--qa',action='store_true')
    parser.add_argument('--start',type=float,default=68.);parser.add_argument('--end',type=float,default=74.);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    checkpoint=BASE/'checkpoints/weak_fast.pkl'
    if args.watch:
        while True:
            r=read(checkpoint.with_suffix('.json'))
            if r.get('time_s',0)>=args.end:break
            progress=read(BASE/'progress/weak_fast.json')
            write(OUT/'status.json',dict(status='WAITING_FOR_NATIVE_ONSET_WINDOW',target_s=args.end,
                saved_until_s=r.get('time_s',0),worker_status=progress.get('status'),watcher_pid=os.getpid()))
            if progress.get('status')=='FAILED' or read(BASE/'status.json').get('status')=='RECOVERY_FAILED':
                raise RuntimeError('Native replay failed; see its progress and supervisor status files')
            time.sleep(30)
    render(checkpoint,args.start,args.end,args.qa)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'status.json',dict(status='FAILED',error=repr(exc)));raise
