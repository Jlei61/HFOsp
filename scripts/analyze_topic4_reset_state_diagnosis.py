#!/usr/bin/env python3
"""Analyze completed reset-state arms and maintain an automatic review package."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/reset_state_diagnosis_20260911'
PREV=ROOT/'results/topic4_sef_hfo/fig5_manual_core_release_v1/weak_fast_z_refill_recurrence_v2'


def read(p):return json.loads(p.read_text())


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');temp.replace(p)


def load_series(folder):
    collections={k:[] for k in ['time_s','Z','M','rate_E','region_E']}
    digests={}
    nreg=np.load(OUT/'geometry.npz')['region_counts'][:3]
    paths=sorted((folder/'chunks').glob('*.npz'))
    if folder.name=='z_only_long':
        # The 238--240 s exact replay is counted once, not as a replicate.
        with np.load(PREV/'runs/weak_fast_z_refill_recurrence.npz') as a:
            keep=a['z_time_ms']<238000
            collections['time_s'].append(a['z_time_ms'][keep]/1000)
            collections['Z'].append(a['z_stats'][keep,:9])
            collections['M'].append(a['m_stats'][keep][:,[0,5,6,7]]*.02)
            collections['rate_E'].append(a['rate_e_hz'][:2380000].reshape(-1,50).mean(1))
            collections['region_E'].append(a['region_spikes_1ms'][:238000,:3].reshape(-1,5,3).sum(1)/nreg/.005)
    for path in paths:
        with np.load(path) as a:
            collections['time_s'].append(a['slow_time_ms']/1000)
            collections['Z'].append(a['Z'])
            collections['M'].append(a['M']*.02)
            collections['rate_E'].append(a['spikes_1ms'][:,0].reshape(-1,5).sum(1)/32000/.005)
            collections['region_E'].append(a['regions_1ms'][:,:3].reshape(-1,5,3).sum(1)/nreg/.005)
            digests[path.name]=str(a['input_digest'])
    series={k:np.concatenate(v) for k,v in collections.items()}
    assert all(len(v)==len(series['time_s']) for v in series.values())
    return series,digests


def summarize(series,result):
    t=series['time_s'];start=0. if result['job'].get('fresh') else 76.5
    onset=result['recurrence_onset_s'];end=result['end_s']
    rows=[]
    windows=[('post_release',start,min(start+10,end)),('late',max(start,end-30),end)]
    if onset is not None:windows += [('before_entry',max(start,onset-5),onset),('entry',onset,min(onset+2,end))]
    for label,lo,hi in windows:
        q=(t>=lo)&(t<hi)
        if not q.any():continue
        rows.append(dict(window=label,time_s=[lo,hi],mean_Z=float(series['Z'][q,0].mean()),
            mean_Z_core_AB=series['Z'][q][:,[5,6]].mean(0).tolist(),
            mean_adaptation_current_E_AB=series['M'][q,:3].mean(0).tolist(),
            mean_E_hz=float(series['rate_E'][q].mean()),
            regional_E_hz=series['region_E'][q].mean(0).tolist()))
    return dict(name=result['job']['name'],observed=result['recurrence_observed'],
                recovered=result['recovered'],observation_status=result['interpretation'],
                latency_since_release_s=None if onset is None else onset-start,
                censor_time_since_release_s=end-start,windows=rows)


def original_replay_qa(folder):
    checks=[]
    with np.load(PREV/'runs/weak_fast_z_refill_recurrence.npz') as source:
        rates=source['rate_e_hz'];raster=source['sample_spikes']
        selected=np.load(OUT/'geometry.npz')['sample_source_indices']
    for path in sorted((folder/'chunks').glob('*.npz')):
        with np.load(path) as a:
            lo=int(a['start_step']);hi=min(int(a['end_step']),755000)
            if hi<=lo:continue
            expected=np.rint(rates[lo:hi].reshape(-1,10).mean(1)*32).astype(np.uint16)
            checks.append(bool(np.array_equal(a['spikes_1ms'][:(hi-lo)//10,0],expected)))
            checks.append(bool(np.array_equal(a['raster'][:hi-lo],raster[lo:hi,selected])))
    assert checks and all(checks),'Original-initialization replay differs from original pre-reset trajectory'
    return dict(status='PASS',scope='E spike counts and sampled raster before original 75.5-s intervention',checks=len(checks))


def report():
    completed=[p for p in sorted((OUT/'runs').glob('*/result.json')) if not p.parent.name.startswith('qa_')]
    result={p.parent.name:read(p) for p in completed}
    series={};digests={};stats=[]
    for name,r in result.items():
        series[name],digests[name]=load_series(OUT/'runs'/name)
        stats.append(summarize(series[name],r))
    pair_checks=[]
    for seed in [9109401,9109402,9109403]:
        a=f'paired_{seed}_Z';b=f'paired_{seed}_ZM'
        if a in digests and b in digests:
            common=sorted(set(digests[a])&set(digests[b]))
            match=bool(common) and all(digests[a][c]==digests[b][c] for c in common)
            pair_checks.append(dict(seed=seed,common_blocks=len(common),input_matches=match))
            assert match,'Paired input mismatch'
    replay=None
    if 'original_initialization_replay' in result:
        replay=original_replay_qa(OUT/'runs/original_initialization_replay')
    lines=['# Z 与 M reset 后再进入：科学分析','',
           '所有条件固定手放双核和原有模型参数；未再拟合位置、连接或患者模式。'
           '统计单位是一条初始状态与后续噪声实现的组合，原轨迹重复前缀只计一次。','']
    a=result.get('z_only_long');b=result.get('z_m_reset_long')
    if a and a['recurrence_observed']:
        latency=a['recurrence_onset_s']-76.5
        lines += [f'原Z-only轨迹在绝对时间{a["recurrence_onset_s"]:.2f}秒再次进入高态，距释放Z为{latency:.2f}秒。'
                  '这直接排除了本次reset使网络再也不能进入高态的解释；此前240秒阴性是观察窗内未见再次进入。'
                  '它尚不能说明每次reset都会以相同等待时间复发。按预定条件无需启动M清零分支。','']
    elif a and b and b['recurrence_observed']:
        lines += ['原Z-only长轨迹右删失，而相同起始快状态及后续噪声的M清零轨迹再次进入。'
                  '这支持M初值干预改变本条轨迹的演化；是否系统改变进入机会与等待时间，读取下方新噪声配对结果。'
                  '不能把单条反事实差异解释为M建立了永久不发作状态。','']
    elif a and b and not b['recovered']:
        lines += ['M清零条件尚未建立预定义的低活动返回，不能把未终止的第一段高活动记作一次新的runaway。'
                  '此结果定位的是返回过程，而不是恢复后的等待时间；快状态清零对照将进一步区分残留突触/延迟与膜状态的作用。','']
    elif a and b:
        lines += ['Z-only与Z+M条件均未在既定观察窗内再次进入。'
                  'M清零不足以在该噪声实现和时间窗内恢复所需行为；继续用快状态清零与原始初始化回放定位。'
                  '这仍不是永不进入的证明。','']
    elif a:
        lines += ['原Z-only轨迹到1000秒仍未再次进入；这是右删失结果。M清零对照正在执行或等待完成。','']
    else:lines += ['正式长程结果尚未完成。','']
    lines+=['| 条件 | 是否再进入 | 释放/初始化后等待或观察(s) |','|---|---|---:|']
    for row in stats:
        value=row['latency_since_release_s'] if row['observed'] else row['censor_time_since_release_s']
        label='观察到' if row['observed'] else ('观察窗内未见' if row['recovered'] else '未建立低活动返回')
        lines.append(f'| {row["name"]} | {label} | {value:.2f} |')
    lines+=['','M清零仅在76.5秒执行一次，之后立即恢复原生方程；并非一直关闭M。'
            '快状态all组另外清除膜电位/不应期、AMPA/GABA及延迟缓冲，仍保留当时OU状态和后续随机输入。'
            '原始初始化回放则从原始随机序列起点开始，其阳性不能作为独立样本。',
            '若全部状态干预仍无再次进入，剩余差别包括OU状态/输入历史与未来噪声序列；'
            '不能把固定参数未改变的状态差异称作某个参数已永久改变。',
            '终止/恢复由外部Z补充造成，M本轮保持开启；高率判据不等同于已证明的临床发作或Hopf分岔。']
    lines+=['','## Figure 5验收仍按原标准','',
            '本报告是机制定位结果；接受标准见[完整Figure 5验收合同](../figure_acceptance.md)。'
            '同一轨迹的①–⑤、对齐的电极读出/raster/Z-M/空间场、可读的E1轨迹及以正式Fig3C为参考的模型早期场比较须分别完成。'
            '增加M清零或快状态清零的结果属于该干预条件，不能替代Z-only条件通过。'
            '计算完成或检测到再次高态均不自动等于整图通过；F缺项也需保留，最终仍需用户目视验收。']
    write(OUT/'analysis/state_comparison.json',dict(stats=stats,input_pair_checks=pair_checks,
          original_replay_qa=replay,human_review='PENDING',
          figure_acceptance=dict(contract=str(OUT/'figure_acceptance.md'),
            status='FULL_FIGURE_ACCEPTANCE_NOT_ESTABLISHED',
            native_recurrence_observed=bool(a and a['recurrence_observed']),
            diagnostic_results_do_not_replace_full_figure=True)))
    (OUT/'analysis/long_run_scientific_review.md').write_text('\n'.join(lines)+'\n')
    if not series:return
    plt.rcParams.update({'font.size':13,'axes.labelsize':15,'xtick.labelsize':12,'ytick.labelsize':12,
                         'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    names=[n for n in ['z_only_long','z_m_reset_long','reset_ZM_all_fast','original_initialization_replay',
                       'reset_ZM_voltage','reset_ZM_synapse_delay'] if n in series]
    fig,axes=plt.subplots(len(names),3,figsize=(17,3.5*len(names)),squeeze=False)
    for row,name in enumerate(names):
        s=series[name];r=result[name];zero=0 if r['job'].get('fresh') else 76.5
        t=s['time_s']-zero;keep=t>=(-8 if zero else 0)
        axes[row,0].plot(t[keep],s['rate_E'][keep],c='#364e68',lw=.6,rasterized=True)
        axes[row,0].set_ylabel(name+'\nE rate (Hz)',fontsize=12)
        for idx,c,label in [(0,'#775294','All E'),(5,'#b34567','Core A'),(6,'#2182a1','Core B')]:
            axes[row,1].plot(t[keep],s['Z'][keep,idx],c=c,lw=1,label=label)
        for idx,c in [(0,'#775294'),(1,'#b34567'),(2,'#2182a1')]:
            axes[row,2].plot(t[keep],s['M'][keep,idx],c=c,lw=1)
        axes[row,1].set_ylim(0,1.05)
        for ax in axes[row]:
            ax.axvline(0,color='#329b84',ls='--',lw=1)
            ax.set_xlabel('Time since '+('initialization' if not zero else 'release')+' (s)')
            if r['recurrence_onset_s'] is not None:ax.axvline(r['recurrence_onset_s']-zero,c='#b5233b',ls=':',lw=1.4)
    axes[0,0].set_title('Native population activity')
    axes[0,1].set_title('Inhibitory resource Z');axes[0,1].legend(frameon=False)
    axes[0,2].set_title('Adaptation current ηM × M')
    fig.tight_layout()
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'reset_state_comparison.png',dpi=170,bbox_inches='tight')
    fig.savefig(folder/'reset_state_comparison.pdf',bbox_inches='tight');plt.close(fig)
    (folder/'README.md').write_text('### reset_state_comparison.png / .pdf\n'
        '按已完成条件逐行显示原生E群体放电率、全E及两核Z、对应适应电流。'
        '横轴为Z释放后的时间；完整初始化回放单独标为初始化后的时间，红虚线标出实际再进入。\n'
        '**关注点**：未出现再次高态的条件保留真实观察终点，不把普通事件或高率平台自动称为振荡发作；'
        '本图随已完成结果更新，仍待Agent及用户目视检查。\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    last=None
    while True:
        paths=[p for p in sorted((OUT/'runs').glob('*/result.json')) if not p.parent.name.startswith('qa_')]
        fingerprint=[(str(p),p.stat().st_mtime_ns) for p in paths]
        status=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
        if fingerprint!=last or status.get('status','').startswith(('COMPLETE','FAILED')):
            try:
                report();last=fingerprint
                write(OUT/'analysis/report_status.json',dict(status='UPDATED',completed_runs=len(paths),pid=os.getpid()))
            except Exception as exc:
                write(OUT/'analysis/report_status.json',dict(status='FAILED',error=repr(exc)));raise
        if not args.watch or status.get('status','').startswith(('COMPLETE','FAILED')):break
        time.sleep(30)


if __name__=='__main__':main()
