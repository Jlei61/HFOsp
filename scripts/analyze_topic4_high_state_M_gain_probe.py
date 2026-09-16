#!/usr/bin/env python3
"""Actual spike and slow-variable response to eta_M steps at one shared state."""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_topic4_high_state_M_gain_probe import OUT, ROOT, SOURCE, read, write


def load_control():
    geometry = np.load(ROOT/'results/topic4_sef_hfo/m_parameter_modes_fig5_20260913/geometry.npz')
    with np.load(SOURCE) as a:
        lo, hi = 750000, 805000
        return dict(name='Unchanged', eta_m=.02, time_ms=np.arange(lo,hi)*.1,
            spikes=np.rint(np.c_[a['rate_e_hz'][lo:hi]*32000*.0001,
                a['rate_i_hz'][lo:hi]*8000*.0001]).astype(int),
            raster=a['sample_spikes'][lo:hi,geometry['sample_source_indices']],
            slow_time_ms=a['z_time_ms'][lo//50:hi//50],
            Z=a['z_stats'][lo//50:hi//50,0],M=a['m_stats'][lo//50:hi//50,0])


def main(matched_prefix=False):
    assert read(OUT/'qa.json')['status']=='PASS'
    control=load_control(); conditions=[control]
    for name in ['eta_0p2','eta_2']:
        folder=OUT/'runs'/name
        if not (folder/'result.json').exists() and not matched_prefix:continue
        if not (folder/'observations.npz').exists():continue
        r=read(folder/'result.json') if (folder/'result.json').exists() else None
        with np.load(folder/'observations.npz') as a:
            eta=float(a['eta_m'])
            data={}
            for key in ['time_ms','spikes','raster','slow_time_ms','Z','M']:
                prefix=5000 if key in ['time_ms','spikes','raster'] else 100
                data[key]=np.concatenate([control[key][:prefix],a[key]])
            conditions.append(dict(name=name,eta_m=eta,result=r,**data))
    end_s=80.5
    if matched_prefix:
        assert len(conditions)==3
        end_s=min((a['time_ms'][-1]+.1)/1000 for a in conditions)
        count=round((end_s-75)*10000);assert count%50==0 and count>5000
        for a in conditions:
            for key in ['time_ms','spikes','raster']:a[key]=a[key][:count]
            for key in ['slow_time_ms','Z','M']:a[key]=a[key][:count//50]
    plt.rcParams.update({'font.size':13,'axes.labelsize':15,'axes.titlesize':16,
        'xtick.labelsize':12,'ytick.labelsize':12,'pdf.fonttype':42,
        'axes.spines.top':False})
    fig,axes=plt.subplots(3,len(conditions),figsize=(6*len(conditions),10.8),
        sharex=True,squeeze=False,gridspec_kw={'height_ratios':[1,1.6,1],'hspace':.13,'wspace':.42})
    records=[]
    for col,a in enumerate(conditions):
        eta=a['eta_m'];time=a['time_ms']/1000
        rate=a['spikes'].reshape(-1,50,2).sum(1)/[32000,8000]/.005
        rate_time=time.reshape(-1,50).mean(1)
        axes[0,col].plot(rate_time,rate[:,0],color='#287aaf',lw=1,label='E')
        axes[0,col].plot(rate_time,rate[:,1],color='#dc8730',lw=1,label='I')
        axes[0,col].set(ylim=(0,720),title=f'ηM = {eta:g}'+(' · unchanged' if col==0 else ' · step at 75.5 s'))
        if col==0:axes[0,col].set_ylabel('Population rate (Hz)');axes[0,col].legend(frameon=False,loc='upper left')
        sp=a['raster'];row,neuron=np.nonzero(sp)
        for mask,color in [(neuron<60,'#287aaf'),(neuron>=60,'#dc8730')]:
            axes[1,col].scatter(time[row[mask]],neuron[mask],s=.6,c=color,marker='.',rasterized=True)
        axes[1,col].set(ylim=(-1,80),yticks=[9.5,29.5,49.5,69.5],
            yticklabels=['Core A E','Core B E','Other E','I'] if col==0 else [])
        for y in [19.5,39.5,59.5]:axes[1,col].axhline(y,color='#aaaaaa',lw=.6)
        zs=axes[2,col];ms=zs.twinx();slow_time=a['slow_time_ms']/1000
        zs.plot(slow_time,a['Z'],c='#78468d',lw=2,label='Z')
        applied_eta=np.where(slow_time<75.5,.02,eta)
        ms.plot(slow_time,applied_eta*a['M'],c='#aa6323',lw=1.8,label='ηM M')
        zs.set(ylim=(0,1.03),xlim=(75.,end_s),xlabel='Time (s)')
        if not matched_prefix:zs.set_xticks([76,78,80])
        ms.set(yscale='log',ylim=(1,1200),yticks=[1,10,100,1000]);ms.tick_params(axis='y',colors='#aa6323')
        zs.tick_params(axis='y',colors='#78468d')
        if col==0:zs.set_ylabel('Mean Z',color='#78468d')
        ms.set_ylabel('Adaptation current (mV equiv.)',color='#aa6323',fontsize=12)
        raw_rates=a['spikes'][5000:].reshape(-1,100,2).sum(1)/[32000,8000]/.01
        for ax in axes[:,col]:ax.axvline(75.5,c='#666666',ls=':',lw=1)
        recovery=None
        for hi in range(200,len(raw_rates)+1):
            rr=raw_rates[hi-200:hi,0].reshape(2,100)
            if np.all(rr.mean(1)<50) and np.all((rr<5).mean(1)>=.2):
                recovery=75.5+hi*.01;break
        if recovery is not None:
            for ax in axes[:,col]:ax.axvline(recovery,c='#34876d',ls='--',lw=1)
        records.append(dict(name=a['name'],eta_M=eta,tau_M_s=2.,end_s=end_s,
            recovery_confirmation_s=recovery,late_E_Hz=float(raw_rates[-100:,0].mean()),
            initial_mean_Z=float(a['Z'][100]),last_mean_Z=float(a['Z'][-1]),
            initial_M_current=float(eta*a['M'][100]),last_M_current=float(eta*a['M'][-1]),
            external_parameter_step=col!=0,Z_reset=False,M_reset=False))
        assert len(sp)==round((end_s-75)*10000) and len(a['spikes'])==len(sp)
    fig.subplots_adjust(left=.09,right=.94,bottom=.09,top=.94)
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    suffix='_matched_prefix' if matched_prefix else ''
    stem='same_high_state_M_gain'+suffix
    fig.savefig(folder/(stem+'.png'),dpi=170,bbox_inches='tight')
    fig.savefig(folder/(stem+'.pdf'),bbox_inches='tight');plt.close(fig)
    write(OUT/('analysis'+suffix+'.json'),dict(status='MATCHED_COMMITTED_PREFIX' if matched_prefix else ('COMPLETE' if len(conditions)==3 else 'PARTIAL'),records=records,
        source_native_control=str(SOURCE),same_full_parent_and_OU_history=True,
        formal_grid_sample_increment=0,constant_parameter_native_cycle_demonstrated=False,
        post_intervention_observation_s=end_s-75.5,full_5s_interventions_complete=not matched_prefix and len(conditions)==3,
        figure=str(folder/(stem+'.png')),agent_visual_review='PENDING',human_review='PENDING'))
    readme=folder/'README.md'
    description='### '+stem+'.png / .pdf\n'+(
        '同一75.5秒高活动状态，保持逐细胞Z、M和所有快状态及随机历史，只改变适应电流系数ηM。'
        '每列显示实际E/I群体率、固定80神经元raster和Z/适应电流，75–75.5秒为共同的真实原生前缀；灰点线为参数步骤，绿虚线为低活动返回标准的确认时刻。\n'
        '**关注点**：参数步骤为外部干预，不能宣称恒参数自主终止；不填入原M扫描F。\n')
    if matched_prefix:description+=f'本版本统一截到{end_s:.2f}秒，仅含干预后{end_s-75.5:.2f}秒已保存数据；原5秒续跑仍在进行。返回判据需要连续2秒，短于2秒的对齐窗不能提前确认返回。\n'
    previous=readme.read_text() if readme.exists() else ''
    heading='### '+stem+'.png / .pdf'
    if heading not in previous:readme.write_text(previous+'\n'+description)
    text=['# 同一高态的M增益干预','',
        '目标是判断既有逐细胞M电流能否将实际高活动拉回；所有干预从同一个完整状态和随机历史出发，Z与M均未清零。',
        '这里改变ηM属于外部参数干预，不能将返回称为固定参数模型的自主发作终止，也不作为原40条搜索的新样本。','',
        '| ηM | 返回确认时间(s) | 末1秒E均率(Hz) | 末Z |', '|---|---|---|---|']
    for r in records:text.append(f'| {r["eta_M"]} | {r["recovery_confirmation_s"]} | {r["late_E_Hz"]:.6g} | {r["last_mean_Z"]:.4f} |')
    text+=['',f'本图实际干预后观察{end_s-75.5:.2f}秒。返回也只回答该外部参数步骤的响应。强适应能使放电停止时，原生Z可恢复；尚不能由此认定适应强度随意增加便能兼容最初的自限事件和后续高态进入。']
    if matched_prefix:text+=['原5秒续跑并未被截短或终止，此图只比较三种条件都已保存的同长度前缀。已完成的η=2完整5秒结果另在77.5秒确认返回；若当前匹配窗尚未到77.5秒，本图不提前标记该确认。']
    (OUT/('scientific_review'+suffix+'.md')).write_text('\n'.join(text)+'\n')
    print(json.dumps(records))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--matched-prefix',action='store_true');args=parser.parse_args()
    main(args.matched_prefix)
