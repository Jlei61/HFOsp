#!/usr/bin/env python3
"""Close the prespecified full-fast90s pilot with equally followed controls."""
import json
import numpy as np
import matplotlib.pyplot as plt
import analyze_topic4_reset_matched_prefix as prior

START, END = 76.5, 166.5
OUT = prior.WINDOW / 'reset_matched_90s'


def main():
    pilot = prior.WINDOW / 'fast_state_pilot/runs/all_fast_90s'
    completed = prior.read(pilot / 'result.json')
    prior.END = END
    with np.load(prior.BASE / 'reset_state_diagnosis_20260911/geometry.npz') as geometry:
        indices = geometry['sample_source_indices']
    arms = [prior.control(indices),
            prior.continuation(prior.BASE / 'reset_state_diagnosis_20260911/runs/z_m_reset_long'),
            prior.continuation(pilot)]
    names = ['Z only', 'Z + M clear', 'Z + M + fast-state clear']
    keys = ['Z_only', 'Z_M_clear', 'Z_M_fast_clear']
    for data in arms:
        assert data['raster'].shape == (900000, 80)
        for key in ['time_ms', 'slow_time_ms', 'inputs']:
            assert np.array_equal(data[key], arms[0][key]), key
    assert arms[1]['input_digests'] == arms[2]['input_digests']
    rows = {key: prior.summary(data) for key, data in zip(keys, arms)}
    for key, data in zip(keys, arms):
        rate = data['spikes_1ms'].reshape(-1, 10, 2).sum(1) / [.01*32000, .01*8000]
        t = data['time_ms'].reshape(-1, 10).mean(1)/1000 - START
        ts = data['slow_time_ms']/1000 - START
        mask = (t >= 50) & (t < 90); slow = (ts >= 50) & (ts < 90)
        rows[key]['windows'].append(dict(after_release_s=[50, 90],
            mean_E_Hz=float(rate[mask, 0].mean()), mean_I_Hz=float(rate[mask, 1].mean()),
            quiet_E_10ms_fraction=float((rate[mask, 0] < 5).mean()),
            mean_Z=float(data['Z'][slow, 0].mean()),
            mean_applied_M=float((.02*data['M'][slow, 0]).mean()),
            mean_GABA_above_depletion_fraction=float(data['Z'][slow, 8].mean())))
    OUT.mkdir(exist_ok=True)
    report = dict(status='COMPLETE_PRESPECIFIED_FAST90S_PILOT', absolute_window_s=[START, END],
        metrics=rows, pilot_result=completed, full_fast_pilot_complete=True,
        longer_M_clear_followup_complete=(prior.BASE/'reset_state_diagnosis_20260911/runs/z_m_reset_long/result.json').exists(),
        external_input_summary_bitwise_equal_all_arms=True,
        full_input_vector100ms_digests_equal_M_and_fast_clear=True,
        sources={key: data['source'] for key, data in zip(keys, arms)},
        parameter=dict(eta_M=.02, tau_M_s=2., tau_Z_s=5., seed=9108401),
        statistical_unit='One paired parent and future-noise realization under three state interventions.',
        original_long_followups_unchanged=True, permanent_nonentry_claim=False,
        pilot_subarm_dispatch_allowed=rows['Z_M_fast_clear']['global_high200Hz_for200ms_observed'],
        agent_visual_review='PENDING', human_review='PENDING')
    (OUT/'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    plt.rcParams.update({'font.size':13, 'axes.labelsize':15, 'axes.titlesize':16, 'pdf.fonttype':42})
    fig, axes = plt.subplots(4, 3, figsize=(18,12), sharex=True, sharey='row',
        gridspec_kw={'height_ratios':[1,1.8,1,1], 'hspace':.14, 'wspace':.14})
    for col, (name, data) in enumerate(zip(names, arms)):
        t = data['time_ms'].reshape(-1,5).mean(1)/1000-START
        rate = data['spikes_1ms'].reshape(-1,5,2).sum(1)/[32000*.005,8000*.005]
        for k, color in [(1,'#d88431'),(0,'#287aaf')]:
            axes[0,col].plot(t,rate[:,k],c=color,lw=.45,label=['E','I'][k])
        axes[0,col].set_title(name)
        step, cell = np.nonzero(data['raster'])
        for mask, color in [(cell<60,'#287aaf'),(cell>=60,'#d88431')]:
            axes[1,col].scatter(step[mask]*.0001,cell[mask],s=.18,c=color,marker='.',rasterized=True)
        axes[1,col].set(ylim=(-1,80),yticks=[9.5,29.5,49.5,69.5])
        for y in [19.5,39.5,59.5]:axes[1,col].axhline(y,c='#cccccc',lw=.5)
        ts=data['slow_time_ms']/1000-START
        axes[2,col].plot(ts,data['Z'][:,0],c='#78468d',lw=1.2)
        axes[2,col].fill_between(ts,data['Z'][:,2],data['Z'][:,4],color='#78468d',alpha=.15)
        axes[2,col].set_ylim(.65,1.02)
        axes[3,col].plot(ts,.02*data['M'][:,0],c='#aa6323',lw=1.2)
        axes[3,col].set(xlabel='Time after Z release (s)',xlim=(0,90),xticks=[0,30,60,90],ylim=(0,9))
    axes[0,0].set_ylabel('Population rate (Hz)');axes[0,0].legend(frameon=False,ncol=2)
    axes[1,0].set_yticklabels(['Core A E','Core B E','Other E','I'])
    axes[2,0].set_ylabel('Mean Z / 10–90%')
    axes[3,0].set_ylabel('Applied M current\n(mV equiv.)')
    fig.subplots_adjust(left=.095,right=.985,bottom=.08,top=.95)
    figs=OUT/'figures';figs.mkdir(exist_ok=True)
    for suffix in ['png','pdf']:fig.savefig(figs/f'matched_reset_90s.{suffix}',dpi=170,bbox_inches='tight')
    for col in range(3):
        axes[3,col].set(xlim=(50,52),xticks=[50,51,52],ylim=(0,1.2))
    for suffix in ['png','pdf']:fig.savefig(figs/f'matched_reset_late_zoom.{suffix}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    (figs/'README.md').write_text('### matched_reset_90s.png / .pdf\n'
        '同一释放状态及后续噪声下，比较仅补Z、再清M以及再清完整细胞快状态的90秒轨迹。全快状态90秒试验已经结束，另外两个长程随访按原定终点独立记录。\n'
        '**关注点**：既有M影响初期恢复，但清除M及快状态是否足以再次进入，要看实际持续高态终点；有限90秒阴性不能解释为永久不可能。\n\n'
        '### matched_reset_late_zoom.png / .pdf\n'
        '同一90秒数据的固定释放后50–52秒放大，显示压缩全程图中不易辨认的单次放电与静默间隔。三个干预使用同一绝对时间窗和坐标。\n'
        '**关注点**：放大用于检查有限事件，未重新选择进入终点或改变原完整随访。\n')
    lines=['# 全快状态90秒试验：完成后的配对判读','',
        '| 条件 | 时间窗(s) | E均率(Hz) | 平均Z | M电流 |', '|---|---|---:|---:|---:|']
    for name,row in rows.items():
        for v in row['windows']:
            lines.append(f'| {name} | {v["after_release_s"]} | {v["mean_E_Hz"]:.3f} | {v["mean_Z"]:.5f} | {v["mean_applied_M"]:.5f} |')
    seen=rows['Z_M_fast_clear']['global_high200Hz_for200ms_observed']
    lines += ['', ('全快状态干预后观察到再次进入，可以按预案分解电压/不应期与突触/延迟状态。' if seen else
        '这90秒里，全快状态清零仍未产生再次进入，因此不启动原先有条件的90秒电压/突触拆分试验。'),
        '这不是初始化不再有效的证明：本试验保留释放时的OU状态与后续随机历史，只将细胞快状态、Z、M置于初始值。三臂是同一噪声实现的干预配对，不是三个独立样本。',
        'M清零后原有M不可能直接永久保留；新放电会再次建立M，其与Z及空间活动的联合分布仍可改变再进入概率。90秒阴性和1000秒随访应分别陈述，不能互相替代。']
    (OUT/'scientific_review.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v['global_high200Hz_for200ms_observed'] for k,v in rows.items()}))


if __name__ == '__main__':
    main()
