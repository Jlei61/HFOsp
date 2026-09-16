#!/usr/bin/env python3
"""Analyze the current autonomous screen without treating spectral peaks as Hopf evidence."""
from screen_topic4_corrected_high_activity import OUT, BASE, read, write, diagnostics, load_patient_coarse_model
import numpy as np
from scipy.signal import find_peaks, periodogram
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    rows=[];models={g:load_patient_coarse_model(BASE/f'coarse_{g}/model.npz') for g in (10,20)}
    for path in sorted((OUT/'runs').glob('*.json')):
        row=read(path);a=np.load(path.with_suffix('.npz'));row['diagnostics']=diagnostics(a['fields_hz'],models[row['grid']]);rows.append(row)
    snn=[]
    for path in sorted((OUT/'snn').glob('*.json')):
        row=read(path);a=np.load(path.with_suffix('.npz'));e=a['field_e_hz'];fields=np.stack([e,np.zeros_like(e)],axis=1)
        # Finite-population spike rates are smoothed by a declared 5-ms window for peak detection.
        fields=uniform_filter1d(fields,5,axis=0,mode='nearest');row['diagnostics']=diagnostics(fields,models[20]);snn.append(row)
        native=a['rate_e_hz'][len(a['rate_e_hz'])//2:];dt=float(a['dt_ms']);f,p=periodogram(native,fs=1000/dt,window='hann',detrend='linear')
        peaks,_=find_peaks(p);top=sorted(peaks,key=lambda i:p[i],reverse=True)[:8]
        row['native_unsmoothed_global']={'dt_ms':dt,'mean_hz':float(native.mean()),'std_hz':float(native.std()),
            'min_hz':float(native.min()),'max_hz':float(native.max()),
            'largest_spectral_peaks':[{'frequency_hz':float(f[i]),'power_density':float(p[i])} for i in top],
            'interpretation':'No frequency band restriction; listed peaks do not by themselves establish an oscillatory attractor.'}
    report={'status':'DEVELOPMENT_HIGH_ACTIVITY_ASSAY','screen_status':read(OUT/'status.json')['status'],
        'confirmation_status':read(OUT/'confirmation_status.json') if (OUT/'confirmation_status.json').exists() else None,
        'rate_results':rows,'snn_results':snn,
        'scope':'One fixed substrate; final patient interictal point remains undecided. User explicitly released this capability test independently of V1/V2.',
        'scientific_boundary':'Constant-input sustained periodic activity is evidence of autonomous oscillation in this reduction, not a proof of Hopf, patient seizure fidelity, or geometry invariance.',
        'SNN_limit':'SNN cold starts whereas primary rate screen starts from high history. Compare the dedicated low-start rate confirmation before attributing failure to closure rather than accessibility.',
        'q_definition':'GABA jumps scaled in both I-to-E and I-to-I; q is not spatial or time-varying Z.',
        'local_evaluation':'All local cells retained, largest oscillating cell and population average reported separately.'}
    if (OUT/'half_step_area_audit.json').exists():report['half_step_area_audit']=read(OUT/'half_step_area_audit.json')
    native=next((r for r in rows if r['name']=='q0.5_gaba20.6116_grid10_dt0.1_high_6000ms'),None)
    half=next((r for r in rows if r.get('preserve_native_impulse_area')),None)
    if native and half:
        a=native['diagnostics']['signals']['global_E'];b=half['diagnostics']['signals']['global_E']
        report['native_area_step_check']={'frequency_relative_change':b['peak_interval_frequency_hz']/a['peak_interval_frequency_hz']-1,
            'amplitude_relative_change':b['peak_to_peak_hz']/a['peak_to_peak_hz']-1,
            'supports':'Persistent autonomous bursting in this corrected reduction at the native calibrated synaptic dose; not a Hopf classification.'}
    write(OUT/'analysis.json',report)
    lines=['# corrected rate 高活动端能力测试','',report['scope'],'',
        '目前结论：corrected rate存在持续自主周期爆发，能与衰减振铃和近饱和高平台区分。该结论不依赖最终患者间期工作点；也不代表已经证明患者发作、Hopf或不同双核几何上的普遍性。', '',
        '## 已完成的rate运行','',
        '|q|GABA ms|格数|时长s|dt ms|初值/面积约定|平均E Hz|峰谷Hz|峰间频率Hz|低于1Hz占比|诊断|','|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        d=r['diagnostics']['signals']['global_E'];freq=d['peak_interval_frequency_hz']
        initial=r['initial']+('/native-area' if r.get('preserve_native_impulse_area') else '/literal-dt')
        lines.append(f"|{r['q']:g}|{r['tau_gaba_ms']:.3f}|{r['grid']}|{r['duration_ms']/1000:g}|{r['dt_ms']}|{initial}|{d['mean_hz']:.2f}|{d['min_hz']:.2f}–{d['max_hz']:.2f}|{f'{freq:.2f}' if freq else '—'}|{d['fraction_below_1hz']:.3f}|{d['label']}|")
    lines += ['', '## 对应SNN','', '|GABA ms|平均E Hz|5ms平滑峰谷Hz|诊断|','|---|---|---|---|']
    for r in snn:
        d=r['diagnostics']['signals']['global_E'];lines.append(f"|{r['tau_gaba_ms']:.3f}|{d['mean_hz']:.2f}|{d['min_hz']:.2f}–{d['max_hz']:.2f}|{d['label']}|")
    if len(snn)==2:
        lines += ['', '两种GABA设置的SNN都出现大幅反复爆发；原20.61ms设置后半段均值约57.25Hz、5ms平滑峰值约257.2Hz，幅度未呈明显衰减。峰间周期比deterministic rate更不规则；筛选器的UNRESOLVED标签反映短窗和周期不规则，不是没有爆发或没有动力学能力。尚不能把有限Poisson网络的3秒轨迹称为严格周期轨道。', '']
    lines += ['', '## 解释边界','',
        '直接把literal离散突触步长从0.1减至0.05ms，会同时把AMPA/GABA电流面积降为原来的0.9655/0.9756；该条件下周期活动消失。因此这不是保持同一工作点的纯步长检验。另行保留native脉冲面积的半步长对照，按明确校正jump_dt运行，两类结果均保留，不相互覆盖。', '',
        'q=0.5附近出现的非衰减周期活动包含明显的低谷。应称自主周期爆发/振荡候选；数学上持续周期轨道与生理上的发作样高活动并非同一验收，不能仅凭它持续重复就称为患者发作。q=0.25附近的近饱和高平台单独报告。', '',
        '局部格和群体平均均保留。频谱最大峰可能是谐波，因此频率同时报告峰间周期估计；近零振幅的固定状态不赋予有意义的谱频率。', '',
        '当前没有进行分岔归类。延长、半步长与细网格通过只支持轨迹持续性和数值稳健；Hopf需要新方程的相关固定点分支与共轭特征值穿越等证据。SNN若只得到tonic而rate得到强周期爆发，应优先判为这项混合抑制近似尚未保真，不能把rate轨道直接转成SNN机制。', '']
    if 'native_area_step_check' in report:
        c=report['native_area_step_check'];lines += ['## 数值确认结果','',
            f"保持原脉冲面积后，dt从0.1减到0.05ms，周期爆发仍存在；峰间频率相对变化{100*c['frequency_relative_change']:.3f}%，峰谷振幅相对变化{100*c['amplitude_relative_change']:.3f}%。因此原半步长失败主要提示工作点对有效输入剂量敏感，不能直接归为积分伪振荡。", '',
            '空间从2mm加密到1mm也保留周期爆发，频率约2.89→3.10Hz，群体振幅328→225Hz。行为类别保留，但幅度尚未空间收敛；最终分岔边界和振幅不得直接把2mm格当成精确网络结果。', '']
    (OUT/'ANALYSIS.md').write_text('\n'.join(lines))
    plot(rows,snn,models)
    print(len(rows),'rate',len(snn),'SNN',flush=True)


def plot(rows,snn,models):
    tau=20.611550480127335;fig,axes=plt.subplots(2,3,figsize=(13,7.4),layout='constrained')
    for ax,q,title in zip(axes[0],[1.,.5,.25],['Low stationary state','Repeated autonomous bursts','High tonic plateau']):
        rs=[r for r in rows if r['q']==q and r['tau_gaba_ms']==tau and r['grid']==10 and r['dt_ms']==.1 and r['initial']=='high']
        if not rs:continue
        r=max(rs,key=lambda r:r['duration_ms']);a=np.load(OUT/'runs'/f"{r['name']}.npz");f=a['fields_hz'];e=np.average(f[:,0],axis=1,weights=models[10].count_e)
        t=np.arange(len(e))/1000;ix=t>=max(0,t[-1]-1.5);ax.plot(t[ix],e[ix],color='#cc3311',lw=1.3)
        ax.set(title=f'{title}\nq={q:g}, GABA={tau:.2f} ms',xlabel='Time (s)',ylabel='Population E rate (Hz)')
        if q==1:ax.set_ylim(0,1)
        if q==.25:ax.set_ylim(0,500)
    ax=axes[1,0];rs=sorted([r for r in rows if r['tau_gaba_ms']==tau and r['grid']==10 and r['dt_ms']==.1 and r['initial']=='high' and r['duration_ms']==3000],key=lambda r:r['q'])
    qs=[r['q'] for r in rs];mins=[r['diagnostics']['signals']['global_E']['min_hz'] for r in rs];maxs=[r['diagnostics']['signals']['global_E']['max_hz'] for r in rs]
    ax.plot(qs,mins,'o-',label='Late minimum');ax.plot(qs,maxs,'o-',label='Late maximum');ax.set(xlabel='Global GABA jump multiplier q',ylabel='Population E rate (Hz)',title='Sampled activity envelope');ax.legend(fontsize=8)
    ax=axes[1,1]
    for g,dt,init,color in [(10,.1,'high','#cc3311'),(10,.05,'high','#4477aa'),(10,.1,'low','#228833'),(20,.1,'high','#aa3377')]:
        rs=[r for r in rows if r['q']==.5 and r['tau_gaba_ms']==tau and r['grid']==g and r['dt_ms']==dt and r['initial']==init and r['duration_ms']==6000]
        if dt==.05:rs=[r for r in rs if r.get('preserve_native_impulse_area')]
        if not rs:continue
        r=rs[0];a=np.load(OUT/'runs'/f"{r['name']}.npz");e=np.average(a['fields_hz'][:,0],axis=1,weights=models[g].count_e);ax.plot(np.arange(1000)/1000,e[-1000:],color=color,label=f'{g}×{g}, dt={dt}, {init}',lw=1)
    ax.set(title='Checks at fixed native current area',xlabel='Last second (s; phases unaligned)',ylabel='Population E rate (Hz)')
    if ax.lines:
        ax.set_ylim(-5,ax.get_ylim()[1]*1.3);ax.legend(fontsize=7,loc='upper right')
    ax=axes[1,2]
    for r in snn:
        a=np.load(OUT/'snn'/f"q{r['q']:g}_gaba{r['tau_gaba_ms']:g}_seed{r['seed']}_{r['duration_ms']:g}ms.npz");e=uniform_filter1d(a['rate_e_hz'],50);t=np.arange(len(e))*.0001;ix=t>=t[-1]-1
        line,=ax.plot(t[ix]-t[ix][0],e[ix],lw=1,label=f"SNN GABA={r['tau_gaba_ms']:.2f}")
        ax.plot(t[ix]-t[ix][0],a['rate_e_hz'][ix],color=line.get_color(),alpha=.12,lw=.4)
    ax.set(title='Native SNN, q=0.5',xlabel='Last second (s)',ylabel='Population E rate (Hz)')
    if ax.lines:ax.legend(fontsize=8)
    else:ax.text(.5,.5,'SNN comparisons running',ha='center',transform=ax.transAxes)
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Corrected rate: autonomous high-activity capability\nConstant input; OU off; one fixed substrate; not a Hopf classification',fontsize=14)
    dest=OUT/'figures';dest.mkdir(exist_ok=True);fig.savefig(dest/'high_activity_capability.png',dpi=160);fig.savefig(dest/'high_activity_capability.pdf');plt.close(fig)
    (dest/'README.md').write_text('### high_activity_capability.png\n恒定背景输入下比较corrected rate的低固定状态、重复周期爆发和高平台，下排显示采样峰谷包络、数值确认及同控制量SNN。所有原始局部E/I场保留；图中群体均值不替代局部检查，曲线相位没有强行对齐。\n**关注点**：持续周期爆发与高平台是否被正确区分，SNN是否支持rate现象；这不是Hopf证明或最终患者工作点验收。\n')


if __name__=='__main__':main()
