#!/usr/bin/env python3
"""Native recruitment order and early-window E fields, with separate labels."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'paper_figures'))
import build_fig6_unified_adaptation as base
from src.topic5_template_axis_field import score_field
import matplotlib.pyplot as plt

RAW=Path('/data/hfosp/topic4_sef_hfo/unified_adaptation_entry_switch_20261030')
OUT=base.MAIN/'results/topic4_sef_hfo/unified_adaptation_entry_switch_20261030'
CURRENT=base.OUT.with_name('unified_current_adaptation_readout14_20261030')

def load_counts(folder):
    keys=['spikes_1ms','regions_1ms']
    d=base.analysis.load(folder,'chunks',keys)
    assert d['spikes_1ms'][:,0].sum()==d['regions_1ms'][:,:3].sum()
    return d

def recruitment(folder,geo):
    d=load_counts(folder);n=len(d['spikes_1ms'])//10
    rates=np.c_[d['spikes_1ms'][:n*10,0],d['regions_1ms'][:n*10,:3]].reshape(n,10,4).sum(1)/np.r_[32000,geo['region_counts'][:3]]/.01
    temporal=base.analysis.audit.temporal_audit(rates)
    episodes=[]
    for e in temporal['entries']:
        end=round(e['onset_s']*100)
        # Last completed 50-ms global-quiet run before sustained high activity.
        q=np.flatnonzero(np.convolve((rates[:end,0]<5).astype(int),np.ones(5,int),'valid')==5)
        start=int(q[-1]+5) if len(q) else 0
        onset=[]
        for j in range(4):
            v=(rates[start:min(end+200,n),j]>100).astype(int)
            hits=np.flatnonzero(np.convolve(v,np.ones(5,int),'valid')==5)
            onset.append((start+int(hits[0]))*.01 if len(hits) else None)
        a,b=onset[1:3]
        lead=None if a is None or b is None else ('A' if a<b else 'B' if b<a else 'tie')
        persistent=[]
        for j in range(4):
            v=(rates[start:min(end+200,n),j]>100).astype(int)
            hits=np.flatnonzero(np.convolve(v,np.ones(20,int),'valid')==20)
            persistent.append((start+int(hits[0]))*.01 if len(hits) else None)
        pa,pb=persistent[1:3]
        persistent_lead=None if pa is None or pb is None else 'A' if pa<pb else 'B' if pb<pa else 'tie'
        margin=None if a is None or b is None else (b-a)*1000
        core_min=min(x for x in [a,b] if x is not None) if any(x is not None for x in [a,b]) else None
        first_region=min(range(1,4),key=lambda j:float('inf') if onset[j] is None else onset[j])
        episodes.append(dict(high_threshold_onset_s=e['onset_s'],last_quiet_end_s=start*.01,
            onset_all_A_B_surround_s=onset,core_leader=lead,B_minus_A_onset_ms=margin,
            sustained_200ms_onset_all_A_B_surround_s=persistent,sustained_200ms_core_leader=persistent_lead,
            sustained_200ms_B_minus_A_ms=None if pa is None or pb is None else (pb-pa)*1000,
            first_region=['All E','Core A','Core B','Surround'][first_region],
            core_precedes_surround=bool(core_min is not None and onset[3] is not None and core_min<onset[3]),
            core_precedes_global_100=bool(core_min is not None and onset[0] is not None and core_min<onset[0])))
    return rates,dict(observed_s=n*.01,episodes=episodes,
        core_order=''.join(e['core_leader'] or '?' for e in episodes),
        sustained_200ms_core_order=''.join(e['sustained_200ms_core_leader'] or '?' for e in episodes),
        definition='Per region, first >100 Hz continuously for50ms in10ms bins after the last50ms of all-E<5Hz before high onset. High onset remains all-E>=200Hz for200ms.',
        sustained_definition='Same100Hz threshold and10ms bins, sustained200ms to distinguish persistent recruitment from an initial brief burst.',
        independent_unit='One fixed-network noise realization; overlapping frames are not independent.')

def scan():
    OUT.mkdir(parents=True,exist_ok=True);geo=dict(np.load(RAW/'geometry.npz'));rows=[]
    for folder in sorted((RAW/'runs').glob('seed*')):
        if not list((folder/'chunks').glob('*.npz')):continue
        rates,row=recruitment(folder,geo);row.update(seed=int(folder.name[4:]),folder=str(folder),complete=(folder/'result.json').exists())
        screening=folder/'screening_status.json'
        row['screening_status']=json.loads(screening.read_text()) if screening.exists() else None
        row['search_complete']=row['complete'] or screening.exists()
        row['has_A_then_B_core_order']=row['core_order'][:2]=='AB'
        row['has_A_then_B_earliest_sustained_region']=bool(row['has_A_then_B_core_order'] and
            [e['first_region'] for e in row['episodes'][:2]]==['Core A','Core B'])
        if row['complete']:
            r=json.loads((folder/'result.json').read_text());row['network_identity']=r['identity']
        rows.append(row)
    base.write(OUT/'seed_search.json',dict(seeds=rows,selection_target='First two high episodes have A then B core recruitment; report surround-first separately. No sign-based field relabeling.'))
    print(json.dumps([{k:r[k] for k in ['seed','observed_s','core_order','episodes','complete']} for r in rows]),flush=True)

def details(seed):
    """Keep core timing, field identity and threshold sensitivity separate."""
    folder=RAW/'runs'/f'seed{seed}';geo=dict(np.load(RAW/'geometry.npz'))
    d=load_counts(folder);_,route=recruitment(folder,geo)
    counts=np.r_[32000,geo['region_counts'][:3]]
    sensitivity=[]
    for episode in route['episodes']:
        rows=[]
        for bin_ms in [5,10]:
            n=len(d['spikes_1ms'])//bin_ms
            rates=np.c_[d['spikes_1ms'][:n*bin_ms,0],d['regions_1ms'][:n*bin_ms,:3]].reshape(n,bin_ms,4).sum(1)/counts/(bin_ms/1000)
            start=round(episode['last_quiet_end_s']*1000/bin_ms)
            end=min(n,round((episode['high_threshold_onset_s']+2)*1000/bin_ms))
            for duration_ms,threshold in [(dur,thr) for dur in [50,100,200] for thr in [50,100,150]]:
                width=duration_ms//bin_ms
                onsets=[]
                for j in range(1,4):
                    hits=np.flatnonzero(np.convolve((rates[start:end,j]>threshold).astype(int),np.ones(width,int),'valid')==width)
                    onsets.append((start+int(hits[0]))*bin_ms/1000 if len(hits) else None)
                a,b=onsets[:2]
                rows.append(dict(bin_ms=bin_ms,threshold_Hz=threshold,duration_ms=duration_ms,
                    onset_A_B_surround_s=onsets,core_leader=None if a is None or b is None else 'A' if a<b else 'B' if b<a else 'tie',
                    B_minus_A_onset_ms=None if a is None or b is None else (b-a)*1000))
        sensitivity.append(rows)
    energy=base.native.energy
    record,_=energy.clinical._load_record('epilepsiae_1146')
    frozen=energy.clinical.load_frozen('epilepsiae_1146');scorers=energy.scorers_from_interictal_record(record)
    names=list(geo['contact_names']);ids=[names.index(n) for n in frozen['names']]
    raw=base.analysis.load(folder,'dense_readout_chunks',['time_ms','contact_current'])
    signal=raw['contact_current'][::5];time=raw['time_ms'][::5]/1000
    signal=(signal-signal.mean(1,keepdims=True))[:,ids]
    bp,bt=energy.segment_power(signal,time,2000,(.5,3.5));fields=[]
    for e in route['episodes']:
        lo=e['last_quiet_end_s']
        if time[-1]<lo+1:continue
        ep,et=energy.segment_power(signal,time,2000,(lo,lo+1))
        robust=energy.clinical.distal_baseline_robust_z(np.c_[bp,ep],np.r_[bt,et],(.5,3.5),min_frames=5)
        z=robust['delta'][:,-1];sa,sb=[score_field(scorers[k],z) for k in ['shared_a','shared_b']]
        fields.append(dict(window_s=[lo,lo+1],robust_z=z,TA=sa,TB=sb,
            best_abs_template='TA' if sa['abs_r']>=sb['abs_r'] else 'TB'))
    result=dict(seed=seed,route=route,onset_sensitivity=sensitivity,energy_fields=fields,
        contract='Core order uses native rates. Energy field uses1s after local process onset, own0.5-3.5s interictal baseline, frozen clinical templates, allowed transverse mirror, max absolute r with sign retained.')
    base.write(OUT/f'seed{seed}_details.json',result)
    print(json.dumps(dict(seed=seed,core_order=route['core_order'],field_templates=[x['best_abs_template'] for x in fields],
        sensitivity_leaders=[[x['core_leader'] for x in rows] for rows in sensitivity])),flush=True)

def report():
    scan();search=json.loads((OUT/'seed_search.json').read_text());rows=[]
    for r in search['seeds']:
        e=r['episodes'];p=OUT/f"seed{r['seed']}_details.json"
        detail=json.loads(p.read_text()) if p.exists() else {};fields=detail.get('energy_fields',[])
        rows.append(dict(seed=r['seed'],observed_s=r['observed_s'],
            status='COMPLETE' if r['complete'] else r['screening_status']['status'] if r['screening_status'] else 'RUNNING',
            core_order=r['core_order'],first_global_high_s=e[0]['high_threshold_onset_s'] if e else '',
            sustained_200ms_core_order=r['sustained_200ms_core_order'],
            second_global_high_s=e[1]['high_threshold_onset_s'] if len(e)>1 else '',
            first_region=e[0]['first_region'] if e else '',second_region=e[1]['first_region'] if len(e)>1 else '',
            first_B_minus_A_ms=e[0]['B_minus_A_onset_ms'] if e else '',
            second_B_minus_A_ms=e[1]['B_minus_A_onset_ms'] if len(e)>1 else '',
            energy_template_order=' / '.join(f['best_abs_template'] for f in fields),
            A_then_B_core_order=r['has_A_then_B_core_order'],A_then_B_earliest_sustained_region=r['has_A_then_B_earliest_sustained_region']))
    with (OUT/'seed_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    table='\n'.join(f"| {r['seed']} | {r['observed_s']:g} | {r['status']} | {r['core_order']} | {r['sustained_200ms_core_order']} | {r['energy_template_order'] or '未计算'} |" for r in rows)
    (OUT/'README.md').write_text(f'''# 局部招募起点与连续 A/B 种子搜索

科学问题：固定网络、统一适应方程和全部参数，从 t=0 连续运行、不中断或重置状态，仅更换输入噪声种子，能否得到第一轮 A 核先招募、第二轮 B 核先招募；另外核查第二轮是否确由 B 核最早持续激活。

初始核招募用原生分区率定义：最近 50 ms 全局低于 5 Hz 之后，分区率超过 100 Hz 持续 50 ms 的首时刻，10 ms 分箱。另用持续 200 ms 的同阈值读出区分短促先发 burst 与后续持续招募。全局高活动仍是 200 Hz 持续 200 ms。核外区域独立计时。`seed*_details.json` 保留 5/10 ms 分箱、50/100/150 Hz 阈值及 50/100/200 ms 持续时间的敏感性和一秒能量场分类，不能将两个观察量互相替代。

当前原图 seed9108405 的 B 核在 8.99 s 先启动，A 核在 9.14 s，9.56 s 才满足全局高活动阈值。状态 1 的 9.10 s 已属于进入过程。局部启动窗 8.99–9.99 s 的 TA/TB 相关为 +0.837/−0.917，按原 max-|r| 规则偏 TB；从状态 1 严格取 9.10–10.10 s，则为 +0.781/−0.520，偏 TA。两者使用同一基线、模板和原允许镜像的规则，窗口内是否包含启动瞬态改变了场读出。原图第二轮是核外先起，不能称为 B 核引发。

共尝试 8 个新噪声种子，先筛第一轮，B 先起的轨迹保留完整检查点后停止跟踪。A 先起者跟踪到第二轮进入确认后约 2 s 或最长 60 s。首次 A 最清楚的 seed9108402 在约 52 s 才返回短事件，因此在保留原 60 s 结果后，从完整检查点继续到第二次进入或最长 80 s，全部动力学状态及随机状态保持不变。这是寻找示例的两阶段搜索，不估计发生率；停止于第一轮的 B 种子不能计为“无第二次发作”。

| seed | 保存时长 s | 状态 | 初始招募顺序，50 ms | 持续招募顺序，200 ms | 能量模板顺序 |
|---|---:|---|---|---|---|
{table}

首次 A 招募并不一定对应 TA 场：seed9108407 是一个反例。seed9108409 的首次顺序对低阈值/分箱敏感。seed9108406 的首次 A 是短促先发 burst，持续招募则由 B 在先，因此不能把初始 50 ms 的 AA 顺序直接当成持续 AA。

选中 seed9108402：第一轮持续招募 A=11.60 s、B=11.81 s；第二轮 B=61.91 s、A=62.14 s，即 A 领先 210 ms 后转为 B 领先 230 ms。持续 200 ms 时，两轮的顺序均在 5/10 ms 分箱和 50/100/150 Hz 阈值下保持一致。50 ms 的第二轮初始瞬态有一组较高阈值设置给出不同先后，故主要图注明持续招募的定义。两轮全局进入分别在 12.06 和 62.43 s，最终保存到 66 s。该轨迹的两次一秒能量场仍都偏 TA（TA/TB 相关分别 −0.774/+0.468、−0.739/+0.444），因此验证的是核心持续招募顺序 A→B，未验证能量场类型 TA→TB。

所有新种子第一轮前均为 B 的平均 Z 较低，但 A/B 均可先启动。因此核平均 Z 的排序不足以单独决定启动次序。此为描述性比较，不是对 Z 或适应电流的因果消融。

[原图两次招募](figures/current-core-recruitment.png) / [E 窗敏感性](figures/entry-window-sensitivity.png)。重排图在 `../../paper-ready-figure/fig6/candidates/unified_current_adaptation_rearranged_20261030/`，顶部大 A、中排左 D 右 C、底排左 E 右 F，原 B 放 supplementary。F 仍是历史模型 13 人参考，不构成新模型队列验证。

复现：运行 `scripts/analyze_topic4_unified_adaptation_entry_switch.py --details SEED` 更新该种子的双读出；`--report` 更新扫描和本报告。原生仿真目录：`{RAW}`。完整结果以 JSON/CSV 为准；人工作图验收待作者目视检查。
''')
    print(json.dumps(dict(report=str(OUT/'README.md'),all_searches_resolved=all(r['search_complete'] for r in search['seeds']))),flush=True)

def first_entry_examples(pair_seed=None):
    """Show native timing across seeds or across two episodes of one seed."""
    geo=dict(np.load(RAW/'geometry.npz'));fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    paths=[base.ROOT/'runs/unified_ref48_s9108405',RAW/'runs/seed9108402'] if pair_seed is None else [RAW/'runs'/f'seed{pair_seed}']*2
    seeds=[9108405,9108402] if pair_seed is None else [pair_seed]*2
    for i,(ax,path,seed) in enumerate(zip(axes,paths,seeds)):
        rates,route=recruitment(path,geo);e=route['episodes'][0 if pair_seed is None else i]
        lo=e['last_quiet_end_s']-.1;hi=e['high_threshold_onset_s']+.15
        t=(np.arange(len(rates))+.5)*.01;take=(t>=lo)&(t<=hi)
        for j,label,color in [(1,'Core A','#b2182b'),(2,'Core B','#2166ac'),(3,'Surround','#888888')]:
            ax.plot(t[take],rates[take,j],color=color,label=label,lw=1.4)
        for j,color in [(1,'#b2182b'),(2,'#2166ac')]:ax.axvline(e['onset_all_A_B_surround_s'][j],color=color,lw=.8,ls=':')
        title=f"Seed {seed}: {e['core_leader']} earlier" if pair_seed is None else f"Entry {i+1}: Core {e['core_leader']} earlier"
        ax.set(xlim=(lo,hi),xlabel='Time (s)',title=title)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('E rate (Hz)');axes[1].legend(frameon=False,fontsize=10);fig.tight_layout()
    stem='first-entry-noise-examples' if pair_seed is None else f'seed{pair_seed}-two-entry-rates'
    for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'{stem}.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    p=OUT/'figures/README.md';text=p.read_text()
    if pair_seed is not None:
        if f'### {stem}' not in text:p.write_text(text+f'\n### {stem}.png / .pdf\n同一连续seed{pair_seed}的两次进入，显示两核与核外区域的原生放电率。阈值虚线、绝对时间和核身份均沿用固定规则。\n**关注点**：此图直接核验两核间先后；能量场类型另见该种子的details.json。\n')
    elif '### first-entry-noise-examples' not in text:
        p.write_text(text+'\n### first-entry-noise-examples.png / .pdf\n相同网络和参数、不同噪声种子的首次进入：原seed9108405为B先起，seed9108402为A先起。虚线为100 Hz持续50 ms的核起点。\n**关注点**：这是两个独立噪声实现，不能作为同一轨迹A→B切换的证据。\n')

def current():
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    a=base.load('unified_ref48_s9108405');energy=base.native.energy
    rec,_=energy.clinical._load_record('epilepsiae_1146');fz=energy.clinical.load_frozen('epilepsiae_1146')
    scorers=energy.scorers_from_interictal_record(rec)
    c=json.loads((CURRENT/'field_comparison.json').read_text())
    assert list(fz['names'])==c['contact_names']
    names=list(a['contact_names']);ids=[names.index(n) for n in c['contact_names']]
    raw=a['lfp_native'][::5];t=a['lfp_native_time_ms'][::5]/1000
    sig=(raw-raw.mean(1,keepdims=True))[:,ids]
    median=np.asarray(c['model_baseline_log_power_median']);scale=np.asarray(c['model_baseline_log_power_mad']);center=np.asarray(c['model_baseline_z_median'])
    rows=[]
    for lo in [8.99,9.0,9.1,9.3,9.56]:
        lp,_=energy.segment_power(sig,t,2000,(lo,lo+1));z=(lp[:,0]-median)/scale-center
        sa,sb=[score_field(scorers[k],z) for k in ['shared_a','shared_b']]
        mf=energy._smooth_from_weights(z,scorers['shared_a']['weight_id']);pf=energy._smooth_from_weights(np.asarray(c['patient_robust_z']),scorers['shared_a']['weight_id'])
        rows.append(dict(window_s=[lo,lo+1],robust_z=z,log_power=lp,TA=sa,TB=sb,
            best_abs_template='TA' if sa['abs_r']>=sb['abs_r'] else 'TB',patient_field_r=float(np.corrcoef(mf,pf)[0,1])))
    rates,native=recruitment(base.ROOT/'runs/unified_ref48_s9108405',a)
    native['state_core_rates']=[]
    for t0 in [9.1,9.3,9.56,13.505]:
        j=round(t0*1000);rr=a['regions_1ms'][j-10:j+10,:3].sum(0)/a['region_counts'][:3]/.02
        native['state_core_rates'].append(dict(time_s=t0,core_A_B_surround_rate_Hz=rr))
    base.write(OUT/'current_entry_audit.json',dict(native=native,fields=rows,
        corrected_display_entry_s=9.1,high_threshold_onset_s=9.56,
        model_window_duration_s=1,patient_reselected=False))
    np.savez_compressed(OUT/'current_native_rates.npz',time_s=(np.arange(len(rates))+.5)*.01,rates_Hz=rates)
    plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,(lo,hi) in zip(axes,[(8.85,9.8),(16.8,18.3)]):
        times=(np.arange(len(rates))+.5)*.01;sel=(times>=lo)&(times<=hi)
        for j,label,color in [(1,'Core A','#b2182b'),(2,'Core B','#2166ac'),(3,'Surround','#888888')]:
            ax.plot(times[sel],rates[sel,j],label=label,color=color,lw=1.5)
        ax.set(xlim=(lo,hi),xlabel='Time (s)',title='First recruitment' if lo<10 else 'Second recruitment')
        ax.spines[['top','right']].set_visible(False)
    axes[0].axvline(9.1,color='#555',ls=':',lw=1);axes[0].axvspan(9.1,9.56,color='#777',alpha=.10)
    axes[0].set_ylabel('E rate (Hz)');axes[1].legend(frameon=False,fontsize=10)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'current-core-recruitment.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    fig=plt.figure(figsize=(12,4.5));gs=fig.add_gridspec(1,3,left=.06,right=.96,bottom=.22,top=.83,wspace=.30)
    for j,start in enumerate([8.99,9.1,9.56]):
        row=next(r for r in rows if abs(r['window_s'][0]-start)<1e-10)
        v=np.asarray(row['robust_z']);ax=fig.add_subplot(gs[j])
        energy.clinical._draw_field(ax,fz,energy.clinical._normalize_minmax(v),np.asarray(fz['support_a']),
            cmap='Blues',colorbar_values=v,title=f"{start:.2f}–{start+1:.2f} s",title_color='black',show_y=j==0)
        ax.set(xticks=[-20,0,20],yticks=[-10,0,10,20])
        ax.text(.5,-.34,f"TA {row['TA']['signed_r']:+.3f}   TB {row['TB']['signed_r']:+.3f}",transform=ax.transAxes,ha='center',fontsize=11)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'entry-window-sensitivity.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    (OUT/'figures/README.md').write_text('''### current-core-recruitment.png / .pdf
当前seed9108405两次进入时的Core A、Core B和核外区域10 ms率。第一次由B核先激活；第二次核外先升高，随后两核中仍是B较早。
**关注点**：核先被招募与核引发整个过程须分开，不能把核外反弹称为局灶B启动。

### entry-window-sensitivity.png / .pdf
同一条轨迹在局部启动8.99 s、状态1的9.10 s与全局阈值9.56 s之后各取1 s计算E场。全部使用原冻结基线和模板，场图独立归一化；下方相关保留符号，按原允许横向镜像的规则计算。
**关注点**：窗口是否包含启动瞬态会影响能量场形态，核招募类型不能仅由这一秒的能量相关替代。
''')
    print(json.dumps(dict(native=native,fields=[{k:r[k] for k in ['window_s','best_abs_template','patient_field_r','TA','TB']} for r in rows]),default=lambda x:x.tolist()),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--scan',action='store_true');ap.add_argument('--report',action='store_true');ap.add_argument('--first-entry-examples',action='store_true');ap.add_argument('--pair-seed',type=int);ap.add_argument('--details',type=int);args=ap.parse_args()
    first_entry_examples(args.pair_seed) if args.first_entry_examples or args.pair_seed else details(args.details) if args.details else report() if args.report else scan() if args.scan else current()
