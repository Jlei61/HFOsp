#!/usr/bin/env python3
"""Fixed20s descriptive prefix: equal steady M gain, different memory times."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_topic4_m_parameter_modes import OUT as SOURCE, ROOT, read, write

OUT=ROOT/'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/matched_gain_prefix'
GROUP=[('e3_t0',.04,1.),('e2_t1',.02,2.),('e1_t2',.01,4.),('e0_t3',.005,8.)]
SEEDS=[9108401,9108402]


def prefix(name):
    keys=['time_ms','spikes_1ms','regions_1ms','slow_time_ms','Z','M','currents','inputs']
    items={key:[] for key in keys};end=0;files=[]
    for path in sorted((SOURCE/'runs'/name/'chunks').glob('*.npz')):
        if '.tmp.' in path.name:continue
        with np.load(path) as a:
            assert int(a['start_step'])==end
            end=int(a['end_step']);files.append(str(path))
            for key in keys:items[key].append(a[key])
        if end>=200000:break
    if end<200000:return None
    result={key:np.concatenate(v) for key,v in items.items()}
    for key in ['time_ms','spikes_1ms','regions_1ms']:result[key]=result[key][:20000]
    for key in ['slow_time_ms','Z','M','currents']:result[key]=result[key][:4000]
    result['inputs']=result['inputs'][result['inputs'][:,0]<20000]
    assert len(result['spikes_1ms'])==20000 and len(result['Z'])==4000
    assert np.array_equal(result['spikes_1ms'][:,0],result['regions_1ms'][:,:3].sum(1))
    result['source_files']=files
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[];data={};pending=[];noise_checks=[]
    for seed in SEEDS:
        reference=None
        for stem,eta,tau in GROUP:
            name=f'{stem}_s{seed}';a=prefix(name)
            if a is None:pending.append(name);continue
            if reference is None:reference=a['inputs']
            else:assert np.array_equal(a['inputs'],reference),name
            counts=a['spikes_1ms'];ne,ni=32000,8000
            data[(seed,tau)]=a
            late=a['slow_time_ms']>=10000
            rows.append(dict(name=name,seed=seed,eta_M=eta,tau_M_s=tau,steady_gain=eta*tau,
                window_s=[0,20],first_20s_E_Hz=float(counts[:,0].sum()/ne/20),
                last_10s_E_Hz=float(counts[10000:,0].sum()/ne/10),
                last_10s_I_Hz=float(counts[10000:,1].sum()/ni/10),
                last_recorded_Z=float(a['Z'][-1,0]),last_recorded_time_s=float(a['slow_time_ms'][-1]/1000),
                last10_mean_Z=float(a['Z'][late,0].mean()),
                last10_mean_depletion_duty=float(a['Z'][late,8].mean()),
                last10_mean_M_current=float((eta*a['M'][late,0]).mean()),source_files=a['source_files']))
        noise_checks.append(dict(seed=seed,input_samples=0 if reference is None else len(reference),
            saved_global_OU_and_EI_input_rate_means_identical=True))
    if pending:
        write(OUT/'analysis.json',dict(status='WAITING_FOR_FIXED20S_PREFIX',pending=pending,records=rows));return
    plt.rcParams.update({'font.size':13,'axes.labelsize':15,'axes.titlesize':16,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    colors=['#c34d48','#3177ae','#279274','#8a62a6']
    fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True,sharey='row',gridspec_kw={'hspace':.13,'wspace':.15})
    for col,seed in enumerate(SEEDS):
        for (stem,eta,tau),color in zip(GROUP,colors):
            a=data[(seed,tau)];t=a['time_ms']/1000;st=a['slow_time_ms']/1000
            axes[0,col].plot(t,np.cumsum(a['spikes_1ms'][:,0])/32000,color=color,lw=1.5,label=f'τM={tau:g}s, ηM={eta:g}')
            axes[1,col].plot(st,a['Z'][:,0],color=color,lw=1.5)
            axes[2,col].plot(st,eta*a['M'][:,0],color=color,lw=1.1)
        axes[0,col].set_title(f'Noise seed {seed}')
        axes[2,col].set(xlabel='Time (s)',xlim=(0,20),xticks=[0,5,10,15,20])
    axes[0,0].set_ylabel('Cumulative E spikes / cell')
    axes[1,0].set_ylabel('Mean Z')
    axes[2,0].set_ylabel('Adaptation current\n(mV equiv.)')
    axes[0,0].legend(frameon=False,fontsize=10,loc='upper left')
    fig.subplots_adjust(left=.11,right=.98,bottom=.08,top=.94)
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'equal_gain_different_memory.png',dpi=170,bbox_inches='tight');fig.savefig(folder/'equal_gain_different_memory.pdf',bbox_inches='tight');plt.close(fig)
    write(OUT/'analysis.json',dict(status='COMPLETE_FIXED_PREFIX_ONLY',records=rows,
        statistical_unit='Two paired noise realizations, each repeated at four M conditions; time bins are not replicates.',
        window_s=[0,20],steady_gain=.04,noise_checks=noise_checks,
        first_entry_or_native_return_conclusion=False,independent_extra_simulations=0,
        agent_visual_review='PENDING',human_review='PENDING'))
    (folder/'README.md').write_text('### equal_gain_different_memory.png / .pdf\n'
        '固定前20秒，对照ηM×τM相同的四种原生M条件；每列为一个配对噪声种子，显示累积E放电、Z和实际适应电流。'
        '所有数据来自原40条轨迹的闭合前缀，未新增模拟；已核对所保存全局OU和E/I输入率均值逐样本相同。\n'
        '**关注点**：长期平均增益相同不意味着瞬态或放电历史相同；这里只描述早期动力学，不预判最终进入、返回或患者匹配。\n')
    lines=['# 相同M长期增益，不同记忆时间：固定前20秒','',
        'ηM×τM=0.04 (mV equiv./Hz)。在给定恒定率下的稳态适应电流相同，并不保证非稳态脉冲/网络反馈相同。'
        '以两个配对噪声为实验重复；实际保存的全局OU和E/I外部输入率均值跨参数逐点一致，神经活动未用于改变外部噪声。',
        '这里不把20秒窗口当作全任务完成，不据此判断是否最终进入高态或原生返回。','',
        '| seed | ηM | τM(s) | 后10秒E均率(Hz) | 后10秒平均Z | 后10秒耗竭占比 | 后10秒适应电流 |',
        '|---|---|---|---|---|---|---|']
    for r in rows:lines.append(f'| {r["seed"]} | {r["eta_M"]} | {r["tau_M_s"]} | {r["last_10s_E_Hz"]:.3f} | {r["last10_mean_Z"]:.4f} | {r["last10_mean_depletion_duty"]:.4f} | {r["last10_mean_M_current"]:.4f} |')
    (OUT/'scientific_review.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([{k:r[k] for k in ['name','last_10s_E_Hz','last10_mean_Z','last10_mean_M_current']} for r in rows]))


if __name__=='__main__':main()
