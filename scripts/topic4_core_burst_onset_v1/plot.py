"""Native core burst-onset figures, plus a separately labeled literature benchmark."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Patch
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from scipy.signal import welch
from PIL import Image
from analyze import OUT,OLD,read,write,acf

FIG=OUT/'figures'
COLORS={'background':'#dce3e8','sparse_bursts':'#a5aab2','irregular_bursts':'#268b91',
 'variable_bursts':'#e2a340','regular_bursts':'#8855a3','patterned_bursts':'#bf657b',
 'sustained_activity':'#bd373b','active_without_bursts':'#87919d'}
NAMES={'background':'Low activity','sparse_bursts':'Sparse bursts','irregular_bursts':'Irregular bursts',
 'variable_bursts':'Intermediate','regular_bursts':'Regular bursts','patterned_bursts':'Patterned bursts'}
EES=[.5,.7,.85,1.,1.2];DEPTH=[0.,.4,.7,1.,1.3]
STATES=[(.5,'Low activity','#657b8c'),(.7,'Irregular bursts',COLORS['irregular_bursts']),
 (.85,'Intermediate',COLORS['variable_bursts']),(1.,'Regular bursts',COLORS['regular_bursts'])]
MANIFEST=[];README=[];BOOK=None

def save(fig,stem,description,focus,meta=None):
    for ext in ('png','pdf'):fig.savefig(FIG/f'{stem}.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
    BOOK.savefig(fig,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    with Image.open(FIG/f'{stem}.png') as im: im.load();size=list(im.size)
    MANIFEST.append(dict(stem=stem,pixels=size,metadata=meta or {},files={e:str(FIG/f'{stem}.{e}') for e in ('png','pdf')}))
    README.append(f'### {stem}.png / .pdf\n{description}\n**关注点**：{focus}\n')

def setup_axes(axes):
    for a in np.asarray(axes).ravel():
        a.spines[['top','right']].set_visible(False);a.tick_params(length=4)

def phase_map(base):
    fig,axes=plt.subplots(1,2,figsize=(13,6.4),sharex=True,sharey=True)
    for a,g,name in zip(axes,('coreAE','coreBE'),('Core A','Core B')):
        for x,ee in enumerate(EES):
            for y,d in enumerate(DEPTH):
                rr=sorted([r for r in base if r['group']==g and r['ee']==ee and r['depth']==d],key=lambda r:r['seed'])
                assert len(rr)==2
                verts=[[(x-.48,y-.48),(x-.48,y+.48),(x+.48,y+.48)],[(x-.48,y-.48),(x+.48,y-.48),(x+.48,y+.48)]]
                for r,v in zip(rr,verts):a.add_patch(Polygon(v,facecolor=COLORS[r['metric']['label']],edgecolor='white',lw=1.1))
        a.axhline(3,color='#252525',ls='--',lw=1.1,alpha=.65)
        a.set(xlim=(-.6,4.6),ylim=(-.6,4.6),xticks=range(5),xticklabels=[f'{x:g}' for x in EES],
            yticks=range(5),yticklabels=[f'{x:g}' for x in DEPTH],xlabel='Within-core E→E multiplier')
        a.set_title(name,loc='left',fontweight='bold');a.set_aspect('equal')
    axes[0].set_ylabel('Threshold-lowering amplitude')
    handles=[Patch(facecolor=COLORS[k],label=v) for k,v in NAMES.items()]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,fontsize=12,bbox_to_anchor=(.5,-.01))
    fig.suptitle('Native SNN activity regions',fontsize=20,y=.99)
    fig.text(.5,.92,'Each square: two noise seeds  ·  dashed row: depth = 1  ·  finite-time labels',ha='center',fontsize=12)
    fig.subplots_adjust(bottom=.23,top=.83,wspace=.2)
    save(fig,'01_native_core_regime_map','复用同一拓扑的25参数格×2噪声、每条20秒原生SNN；两个core分别显示。每格上左/下右三角依次为848101/848102，虚线标出配套一维切片。','颜色表示18秒活动表型；未插值边界，未将CV分类线命名为Hopf。',dict(n_native_runs=50,topology=2511,analysis_s=[2,20]))

def onset(base):
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True)
    for c,g in enumerate(('coreAE','coreBE')):
        a,b=axes[:,c]
        for ee in EES:
            rows=sorted([r for r in base if r['group']==g and r['ee']==ee and r['depth']==1],key=lambda r:r['seed'])
            for q,r in enumerate(rows):
                x=ee+(-.009 if q==0 else .009);color=COLORS[r['metric']['label']]
                peaks=np.array([e['peak_rate_hz'] for e in r['events']]);iei=np.array(r['intervals_s'])*1000
                for ax,vals in ((a,peaks),(b,iei)):
                    if len(vals):
                        jitter=np.linspace(-.005,.005,len(vals))
                        ax.scatter(x+jitter,vals,s=16,c=color,alpha=.55,edgecolors='none',zorder=2)
                        low,med,high=np.quantile(vals,[.1,.5,.9])
                        ax.plot([x,x],[low,high],color='#333333',lw=1.2,zorder=3)
                        ax.plot(x,med,'_',color='#111111',ms=9,zorder=4)
                a.plot(x,r['metric']['mean_rate_hz'],'s',color='#172536',ms=5,zorder=5)
        a.set_title('Core '+('A' if c==0 else 'B'),loc='left',fontweight='bold')
        a.set_ylabel('Rate (Hz / E cell)');b.set_ylabel('Inter-burst interval (ms)')
        b.set_xlabel('Within-core E→E multiplier')
        for ax in (a,b):ax.set(xlim=(.46,1.24),xticks=EES);ax.set_ylim(bottom=0)
    fig.suptitle('Burst onset and event-to-event variability',fontsize=20,y=.98)
    fig.text(.5,.927,'Dots: native event peaks / intervals  ·  bars: within-run 10–90%  ·  squares: mean firing rate',ha='center',fontsize=11.5)
    fig.subplots_adjust(top=.85,bottom=.1,hspace=.24,wspace=.28)
    save(fig,'02_native_burst_onset_curves','固定降阈值幅度1，显示核内EE增强时实际burst峰率、群体均率和相邻事件间隔。每个参数下两噪声略左右错开，散点是运行内事件，细线是该运行事件的10–90%范围。','这是原生参数响应图；峰值散点不是固定点分支或经过延拓的极限环。')

def regularization(base):
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for q,(g,clr,mk) in enumerate([('coreAE','#247985','o'),('coreBE','#a15387','s')]):
        for seed,ls in [(848101,'-'),(848102,'--')]:
            rows=sorted([r for r in base if r['group']==g and r['depth']==1 and r['seed']==seed],key=lambda r:r['ee'])
            for ax,key in [(axes[0,0],'cv'),(axes[0,1],'median_peak_active_fraction_2ms')]:
                vals=[r['metric'].get(key) if r['metric']['n_bursts']>=8 else np.nan for r in rows]
                ax.plot([r['ee'] for r in rows],vals,marker=mk,color=clr,ls=ls,ms=7,alpha=.8,lw=1.5)
    axes[0,0].set(ylabel='Burst-interval CV',xlabel='Within-core E→E multiplier',ylim=(0,1.05),xticks=EES)
    axes[0,1].set(ylabel='Peak 2 ms active fraction',xlabel='Within-core E→E multiplier',ylim=(0,1),xticks=EES)
    for ee,title,color in STATES:
        path=OLD/'per_run'/f'ee{ee:g}_d1_n1_t2511_s848101'/'trajectory.npz'
        with np.load(path) as z:
            rate=z['spike_counts_2ms'][1000:,0]/int(z['group_sizes'][0])/.002
            ac=acf(rate.copy());f,p=welch(rate,fs=500,nperseg=2000,noverlap=1000)
        axes[1,0].plot(np.arange(len(ac))*.002,ac,color=color,lw=1.5,label=title)
        sel=(f>=.5)&(f<=15);total=np.trapezoid(p[sel],f[sel]) if hasattr(np,'trapezoid') else np.trapz(p[sel],f[sel])
        axes[1,1].plot(f[sel],p[sel]/max(total,1e-30),color=color,lw=1.5)
    axes[1,0].set(xlabel='Lag (s)',ylabel='Rate autocorrelation',xlim=(0,1.2),ylim=(-.15,1.03))
    axes[1,1].set(xlabel='Frequency (Hz)',ylabel='Normalized rate PSD',xlim=(.5,15),ylim=(0,None))
    axes[0,0].legend(handles=[Line2D([],[],marker='o',color='#247985',label='Core A'),Line2D([],[],marker='s',color='#a15387',label='Core B')],frameon=False,fontsize=12)
    axes[1,0].legend(frameon=False,fontsize=10,loc='upper right')
    fig.suptitle('Timing, recruitment and population rhythmicity',fontsize=20,y=.99)
    fig.text(.5,.94,'Top: both cores × two noise seeds  ·  bottom: fixed Core A, seed 848101',ha='center',fontsize=12)
    fig.subplots_adjust(top=.87,bottom=.09,hspace=.45,wspace=.32)
    save(fig,'03_timing_recruitment_and_rhythm','上排分别显示事件间隔CV与峰值2ms同步参与比例，两个core、两个噪声保留身份。下排为同一Core A代表轨迹的群体率自相关和0.5–15Hz归一化功率谱，四种状态使用相同算法。','群体谱峰、单细胞irregularity和burst间隔CV是不同读出；自相关衰减不能单独确定Hopf两侧。')

def intermediate(base):
    fig,axes=plt.subplots(3,2,figsize=(14,9),gridspec_kw={'width_ratios':[3.5,1.3]})
    for axs,(ee,title,color) in zip(axes,STATES[1:]):
        r=next(r for r in base if r['group']=='coreAE' and r['depth']==1 and r['seed']==848101 and r['ee']==ee)
        t=np.array([e['onset_s'] for e in r['events']]);p=np.array([e['peak_active_fraction_10ms'] for e in r['events']])
        axs[0].vlines(t,0,p,color=color,lw=1.5);axs[0].scatter(t,p,s=25,color=color)
        axs[0].set_title(f'{title}  |  E→E = {ee:g}',loc='left',fontsize=15)
        axs[0].set(xlim=(2,20),ylim=(0,1.06),ylabel='Peak 10 ms\nactive fraction',xticks=[2,6,10,14,18,20])
        axs[1].step(np.sort(p),np.arange(1,len(p)+1)/len(p),where='post',color=color,lw=2.2)
        axs[1].set(xlim=(0,1.03),ylim=(0,1.03),ylabel='Cumulative fraction',xticks=[0,.5,1])
    axes[-1,0].set_xlabel('Simulation time (s)');axes[-1,1].set_xlabel('Peak active fraction')
    fig.suptitle('Intermediate bursts: variable recruitment',fontsize=20,y=.99)
    fig.subplots_adjust(top=.91,bottom=.09,hspace=.47,wspace=.3)
    save(fig,'04_intermediate_recruitment','同一Core A、噪声和18秒窗口内，将每个burst的峰值10ms参与比例按真实发生时刻画出，并在右侧显示其经验分布。该读出与2ms同步指标区分。','中间条件是否混合小范围与大范围招募；这是事件幅度分布，不预设第三个吸引子。')

def state_waveforms(fine=False):
    meta=read(OLD/'figures/native_burst_four_states_waveform_raster.json')
    sample=np.array(meta['raster_neuron_ids'],int)
    if fine:sample=sample[np.linspace(0,len(sample)-1,30,dtype=int)]
    assert np.all(np.diff(sample)>0)
    for ee,title,color in STATES:
        path=OLD/'per_run'/f'ee{ee:g}_d1_n1_t2511_s848101'/'trajectory.npz'
        with np.load(path) as z:
            rate=z['spike_counts_2ms'][:,0]/int(z['group_sizes'][0])/.002
            t=(np.arange(len(rate))+.5)*.002
            keep=np.isin(z['raster_cell'],sample);rt=z['raster_time_ms'][keep]/1000-.001
            ri=np.searchsorted(sample,z['raster_cell'][keep])+1
        fig,axes=plt.subplots(2,1,figsize=(12.5,7.5),sharex=True,gridspec_kw={'height_ratios':[1.3,3]})
        lo,hi=(5.,5.6) if fine else (4.,7.)
        sel=(t>=lo)&(t<hi);rsel=(rt>=lo)&(rt<hi)
        axes[0].plot(t[sel],rate[sel],color=color,lw=1.4)
        axes[0].set(ylabel='Rate\n(Hz / E cell)',ylim=(0,max(2,rate[sel].max()*1.15)))
        axes[0].yaxis.set_major_locator(MaxNLocator(3))
        axes[1].vlines(rt[rsel],ri[rsel]-.34,ri[rsel]+.34,color='#111111',lw=1.3 if fine else .95)
        axes[1].set(ylabel=f'Core A E neuron\n(fixed {len(sample)} cells)',xlabel='Simulation time (s)',ylim=(.4,len(sample)+.6),yticks=[1,10,20,30] if fine else [1,25,50,75,100],xlim=(lo,hi))
        fig.suptitle(f'{title}  |  E→E = {ee:g}',fontsize=20,y=.98)
        fig.text(.5,.915,'Rate: all 720 Core A E cells  ·  raster: occupied 2 ms bins  ·  same time and cells across states',ha='center',fontsize=11.5)
        fig.subplots_adjust(top=.84,bottom=.11,left=.12,hspace=.23)
        suffix='_closeup' if fine else ''
        save(fig,f'05_waveform_raster_ee{ee:g}{suffix}',f'固定同一Core A和噪声，展示EE={ee:g}的{lo:g}–{hi:g}秒群体率与固定{len(sample)}细胞raster。群体率来自全部720个E细胞，raster为原记录的2ms占用标记，四图使用相同细胞与窗口。','看清零散发放、大小burst和稳定重复招募；各图率轴明确标注，图形本身不证明分岔。',dict(source=str(path),raster_ids=sample.tolist(),window_s=[lo,hi]))

def noise_summary(comps):
    rows=[r for r in comps if r['contrast']=='ou_removed']
    if not rows:return
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True)
    for c,g in enumerate(('coreAE','coreBE')):
        for context,clr,lab in [('baseline','#547586','OU + Poisson'),('intervention','#d48333','Poisson only')]:
            for seed,ls in [(848101,'-'),(848102,'--')]:
                rr=sorted([r for r in rows if r['group']==g and r['context']==context and r['seed']==seed],key=lambda r:r['ee'])
                if not rr:continue
                x=[r['ee'] for r in rr]
                axes[0,c].plot(x,[r['burst_rate_hz'] for r in rr],marker='o',color=clr,ls=ls,ms=7,label=lab if seed==848101 else None)
                axes[1,c].plot(x,[r['metric']['cv'] if r['metric']['n_bursts']>=8 else np.nan for r in rr],marker='o',color=clr,ls=ls,ms=7)
        axes[0,c].set_title('Core '+('A' if c==0 else 'B'),loc='left',fontweight='bold')
        axes[0,c].set_ylabel('Burst occurrence rate (1/s)');axes[1,c].set_ylabel('Burst-interval CV')
        axes[1,c].set_xlabel('Within-core E→E multiplier')
        for a in axes[:,c]:a.set(xticks=EES[1:]);a.set_ylim(bottom=0)
    axes[0,0].legend(frameon=False,fontsize=12)
    fig.suptitle('Does rhythmic bursting require slow core-wide OU input?',fontsize=19,y=.99)
    fig.text(.5,.94,'OU removed at 6 s; private Poisson retained  ·  matched comparison window: 8–20 s',ha='center',fontsize=12)
    fig.subplots_adjust(top=.87,bottom=.11,hspace=.32,wspace=.3)
    save(fig,'06_private_poisson_control','原生连续轨迹在6秒去掉core共享OU但保留私有Poisson；统计8–20秒，与对应原轨迹同窗比较。两个core和两噪声分别保留，少于8事件不画CV；EE=0.7的次数对检测阈值敏感，不能直接推断潜在触发次数减少。','群体burst能否在恒定Poisson强度下出现；保留Poisson的结果不等于无噪声极限环。')

def intervention_rasters():
    oldmeta=read(OLD/'figures/native_burst_four_states_waveform_raster.json');sample=np.array(oldmeta['raster_neuron_ids'],int)
    for ee in (.85,1.2):
        for arm in ('ou_off','all_off_probe'):
            folder=OUT/'per_run'/f'ee{ee:g}_s848101_{arm}'
            if not (folder/'result.json').exists():continue
            with np.load(folder/'trajectory.npz') as z:
                rate=z['spike_counts_2ms'][:,0]/int(z['group_sizes'][0])/.002
                t=(np.arange(len(rate))+.5)*.002
                cells=z['exact_spike_cell'];keep=np.isin(cells,sample)
                rt=z['exact_spike_time_ms'][keep]/1000;ri=np.searchsorted(sample,cells[keep])+1
            fig,axes=plt.subplots(2,1,figsize=(14,7.5),sharex=True,gridspec_kw={'height_ratios':[1.4,3]})
            axes[0].plot(t,rate,color='#287b85',lw=.9);axes[0].set(ylabel='Rate\n(Hz / E cell)',ylim=(0,max(rate.max()*1.06,2)))
            lo,hi=4.,20.;m=(rt>=lo)&(rt<hi)
            axes[1].vlines(rt[m],ri[m]-.32,ri[m]+.32,color='#141414',lw=.65)
            axes[1].set(xlim=(lo,hi),ylim=(.4,100.6),yticks=[1,25,50,75,100],ylabel='Core A E neuron\n(fixed 100 cells)',xlabel='Simulation time (s)',xticks=[4,6,8,12,16,20])
            for a in axes:a.axvline(6,color='#be6e2c',lw=1.6,ls='--')
            if arm=='all_off_probe':
                for a in axes:a.axvline(12,color='#93468e',lw=1.6,ls=':')
            title='OU removed; private Poisson retained' if arm=='ou_off' else 'All input fluctuations removed; finite pulse at 12 s'
            fig.suptitle(f'E→E = {ee:g}  |  {title}',fontsize=17,y=.985)
            fig.text(.5,.927,'Continuous native state  ·  actual 0.1 ms spike times  ·  no Z/M dynamics',ha='center',fontsize=12)
            fig.subplots_adjust(top=.855,bottom=.105,left=.1,hspace=.2)
            save(fig,f'07_continuation_ee{ee:g}_{arm}',f'EE={ee:g}的真实连续轨迹与精确spike raster，6秒时改变外部随机输入。完全去噪协议另于12秒让core A各E细胞发一个强制spike，后续活动由完整网络演化。','分别检查去噪后的活动和单次有限扰动后的返回/维持；脉冲本身不算自发burst。')

def probe_summary(comps):
    rows=[r for r in comps if r['contrast'] in ('all_removed','probe_late') and r['context']=='intervention']
    if not rows:return
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.5))
    for c,g in enumerate(('coreAE','coreBE')):
        for contrast,color,mk,label in [('all_removed','#547586','o','After removal: 8–12 s'),('probe_late','#93468e','s','After pulse: 14–20 s')]:
            for seed,ls in [(848101,'-'),(848102,'--')]:
                rr=sorted([r for r in rows if r['group']==g and r['contrast']==contrast and r['seed']==seed],key=lambda r:r['ee'])
                shift=(-.018 if contrast=='all_removed' else .018)+(-.005 if seed==848101 else .005)
                axes[c].plot([r['ee']+shift for r in rr],[r['burst_rate_hz'] for r in rr],marker=mk,ls='None',color=color,ms=7,mfc=color if seed==848101 else 'white',label=label if seed==848101 else None)
        axes[c].set(xlabel='Within-core E→E multiplier',ylabel='Burst occurrence rate (1/s)',xticks=EES[1:],ylim=(-.04,max(.5,max(r['burst_rate_hz'] for r in rows)*1.1)))
        axes[c].set_title('Core '+('A' if c==0 else 'B'),loc='left',fontweight='bold')
        if all(r['burst_rate_hz']==0 for r in rows if r['group']==g):
            axes[c].text(.5,.5,'No spontaneous bursts\nin either late window',ha='center',va='center',transform=axes[c].transAxes,fontsize=16,color='#4c5962')
    axes[0].legend(frameon=False,fontsize=11)
    fig.suptitle('Finite-time persistence under deterministic input',fontsize=19,y=1.01)
    fig.tight_layout()
    save(fig,'08_deterministic_persistence','完全去除外部随机性后，分别报告干预后8–12秒和单次强制spike后14–20秒的自发burst出现率。两种时间窗用途不同，不将两者当成等长效应量比较；协议/种子沿横轴略错开避免零值重叠，实心/空心依次为848101/848102。','有界初态和脉冲下能否维持活动；阴性结果不证明不存在其他吸引子，也不排除噪声背景下的群体Hopf。')

def pulse_closeup():
    sample=np.array(read(OLD/'figures/native_burst_four_states_waveform_raster.json')['raster_neuron_ids'],int)
    for ee in (.85,1.2):
        path=OUT/'per_run'/f'ee{ee:g}_s848101_all_off_probe'/'trajectory.npz'
        if not path.exists():continue
        with np.load(path) as z:
            t=(np.arange(len(z['spike_counts_2ms']))+.5)*2-12000
            er=z['spike_counts_2ms'][:,0]/int(z['group_sizes'][0])/.002
            ir=z['core_i_counts_2ms'][:,0]/int(z['core_i_sizes'][0])/.002
            aux=z['core_current_voltage_2ms'][:,:2]
            keep=np.isin(z['exact_spike_cell'],sample)
            rt=z['exact_spike_time_ms'][keep]-12000
            ri=np.searchsorted(sample,z['exact_spike_cell'][keep])+1
        fig,axes=plt.subplots(3,1,figsize=(12.5,10),sharex=True,gridspec_kw={'height_ratios':[1.2,1.2,3]})
        sel=(t>=-20)&(t<150);rsel=(rt>=-20)&(rt<150)
        axes[0].plot(t[sel],er[sel],color='#247985',lw=1.8,label='Core A E')
        axes[0].plot(t[sel],ir[sel],color='#a15387',lw=1.8,label='Local I')
        axes[0].set_ylabel('Rate\n(Hz / cell)');axes[0].legend(frameon=False,fontsize=11,loc='upper right')
        for j,color,label in [(0,'#247985','Excitatory input'),(1,'#a15387','Inhibitory input')]:
            axes[1].plot(t[sel]+1,aux[sel,j],color=color,lw=1.8,label=label)
        axes[1].set_ylabel('Mean input\n(mV equivalent)');axes[1].legend(frameon=False,fontsize=11,loc='upper right')
        axes[2].vlines(rt[rsel],ri[rsel]-.34,ri[rsel]+.34,color='#111111',lw=.95)
        axes[2].set(ylabel='Core A E neuron\n(fixed 100 cells)',xlabel='Time from forced spike (ms)',xlim=(-20,150),ylim=(.4,100.6),yticks=[1,25,50,75,100])
        for a in axes:a.axvline(0,color='#777',lw=1,ls=':')
        fig.suptitle(f'Finite pulse evokes a transient burst  |  E→E = {ee:g}',fontsize=20,y=.985)
        fig.text(.5,.938,'One forced spike per Core A E cell at 12 s  ·  constant external input thereafter',ha='center',fontsize=12)
        fig.subplots_adjust(top=.89,bottom=.08,left=.13,hspace=.25)
        save(fig,f'13_finite_pulse_response_ee{ee:g}','在恒定输入下放大单次强制spike后的150ms原生活动，分别显示Core A E/局部I群体率、E细胞接收的平均兴奋/抑制输入和精确spike raster。输入电流采用执行器内等效膜电位单位，不是pA；这一脉冲让全部720个Core A E细胞各发一个spike。','直接查看有限扰动后短暂招募及返回；输入时序是机制线索，不能单凭此图确定Hopf类型或证明抑制的因果必要性。')

def recovery_traces():
    for ee in (.85,1.2):
        path=OUT/'per_run'/f'ee{ee:g}_s848101_ou_off'/'trajectory.npz'
        if not path.exists():continue
        with np.load(path) as z:
            t=(np.arange(len(z['spike_counts_2ms']))+.5)*.002
            rate=z['spike_counts_2ms'][:,0]/z['group_sizes'][0]/.002
            aux=z['core_current_voltage_2ms'][:,:3]
        fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True)
        sel=(t>=14)&(t<15.2)
        axes[0].plot(t[sel],rate[sel],color='#247985',lw=1.4);axes[0].set_ylabel('Rate\n(Hz / E cell)')
        axes[1].plot(t[sel]+.001,aux[sel,2],color='#4c638c',lw=1.7)
        axes[1].axhline(11,color='#777',ls='--',lw=1,label='Reset = 11 mV')
        axes[1].set_ylabel('Mean membrane\nvoltage (mV)');axes[1].legend(frameon=False,fontsize=11,loc='lower right')
        for j,color,label in [(0,'#247985','Excitatory input'),(1,'#a15387','Inhibitory input')]:
            axes[2].plot(t[sel]+.001,aux[sel,j],color=color,lw=1.6,label=label)
        axes[2].set(ylabel='Mean input\n(mV equivalent)',xlabel='Simulation time (s)',xlim=(14,15.2))
        axes[2].legend(frameon=False,fontsize=11,loc='upper right')
        fig.suptitle(f'Native burst and recovery trajectory  |  E→E = {ee:g}',fontsize=20,y=.985)
        fig.text(.5,.932,'Core A, Poisson only  ·  current-based LIF  ·  model voltage reference',ha='center',fontsize=12)
        fig.subplots_adjust(top=.87,bottom=.085,left=.13,hspace=.27)
        save(fig,f'14_native_recovery_ee{ee:g}','在去掉共享OU、保留私有Poisson的连续轨迹中，放大同一Core A的群体率、平均膜电位及兴奋/抑制输入。电位和输入均直接取执行器记录，虚线仅表示11mV reset参考；没有做额外滤波或单位重标。','当前电流型LIF允许很深的burst后超极化；恢复时间可能影响重复周期。必须先核查该特征的物理稳健性，再将当前规则burst认作可推广的生理振荡机制。')

def benchmark():
    path=OUT/'literature_hopf_benchmark.npz'
    if not path.exists():return
    d=read(OUT/'literature_hopf_benchmark.json')
    with np.load(path) as z:
        fig,axes=plt.subplots(1,3,figsize=(15,4.9))
        axes[0].plot(z['K'],z['eigen'][:,0],color='#445e86',lw=2);axes[0].axhline(0,color='#555',lw=1)
        axes[0].axvline(d['critical_K'],color='#ac5285',ls='--',lw=1.2)
        axes[0].set(xlabel='Effective feedback gain K',ylabel='Re(leading eigenvalue) (1/s)')
        for ax,k,color in zip(axes[1:],(8.4,8.8),('#547586','#ac5285')):
            y=z[f'trace_{k:g}'];t=np.arange(len(y))*.02
            ax.plot(t,y,color=color,lw=1.1);ax.axhline(d['r0'],color='#777',lw=.8,ls='--')
            ax.set(xlabel='Time (ms)',ylabel='Rate (model units)',xlim=(0,600))
            ax.set_title(f'K = {k:g}',loc='left',fontweight='bold')
    fig.suptitle('Literature benchmark: Brunel–Hakim 2008 delay-rate model',fontsize=19,y=1.01)
    fig.text(.5,.015,f'Kc = {d["critical_K"]:.4f}  ·  onset frequency = {d["critical_frequency_hz"]:.2f} Hz  ·  not a fitted reduction of our SNN',ha='center',fontsize=12)
    fig.subplots_adjust(bottom=.22,top=.82,wspace=.4)
    save(fig,'09_literature_hopf_benchmark','按Brunel–Hakim 2008图2给出的标量时延率模型复算特征根及临界两侧时间序列。该模型随耦合改变外部均值以固定工作点，时间常数10ms、时延2ms；临界值由特征方程直接计算。','这是独立文献基准，不是当前空间SNN的降阶结果；K不能替代核内EE倍率，其134Hz也不能当作当前burst频率。',d)

def cell_irregularity():
    rows=[r for r in read(OUT/'single_cell_irregularity.json') if r['window']=='after']
    if not rows:return
    fig,axes=plt.subplots(1,2,figsize=(13,5.7),sharey=True)
    for a,g,title in zip(axes,('coreAE','coreBE'),('Core A','Core B')):
        for ee in EES[1:]:
            rr=sorted([r for r in rows if r['group']==g and r['ee']==ee],key=lambda r:r['seed'])
            for i,r in enumerate(rr):
                x=ee+(-.008 if i==0 else .008)
                if r['cell_isi_cv']:
                    low,med,high=np.quantile(r['cell_isi_cv'],[.1,.5,.9])
                    a.plot([x,x],[low,high],color='#ad8550',lw=2)
                    a.plot(x,med,'o',color='#ad8550',ms=7)
                if r['group_burst_iei_cv'] is not None:
                    a.plot(x,r['group_burst_iei_cv'],'s',color='#277b8b',ms=7)
        a.set_title(title,loc='left',fontweight='bold');a.set(xlabel='Within-core E→E multiplier',xticks=EES[1:],ylim=(0,None))
    axes[0].set_ylabel('Coefficient of variation')
    handles=[Line2D([],[],marker='o',color='#ad8550',label='Cell spike ISI: median / 10–90%'),Line2D([],[],marker='s',color='#277b8b',label='Population burst IEI')]
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,fontsize=12)
    fig.suptitle('Cell irregularity and burst timing are different observables',fontsize=18,y=.99)
    fig.text(.5,.91,'Poisson only, 8–20 s  ·  cell CV requires ≥21 spikes  ·  cell bars describe eligible recorded cells',ha='center',fontsize=11.5)
    fig.subplots_adjust(top=.81,bottom=.23,wspace=.2)
    save(fig,'10_cell_isi_vs_population_iei','使用新轨迹的精确0.1ms spike时刻，在去OU后8–20秒分别计算细胞spike ISI-CV与群体burst IEI-CV。细胞至少21个spike才纳入，图中范围是合格记录细胞的10–90%而非网络层置信区间；合格细胞数保存在single_cell_irregularity.json。','不能把Brunel文中的单细胞irregular标签直接对应到我们的burst间隔标签；低率条件的细胞筛选偏差需结合资格数阅读。')

def unthresholded_response(base):
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.7))
    for g,color,mk in [('coreAE','#247985','o'),('coreBE','#a15387','s')]:
        for seed,ls in [(848101,'-'),(848102,'--')]:
            rr=sorted([r for r in base if r['group']==g and r['depth']==1 and r['seed']==seed],key=lambda r:r['ee'])
            for a,key in zip(axes,('unthresholded_rate_mean_hz','unthresholded_rate_std_hz')):
                a.plot([r['ee'] for r in rr],[r[key] for r in rr],color=color,marker=mk,ms=7,ls=ls,label=g.replace('core','Core ').replace('E','') if seed==848101 else None)
    for a in axes:a.set(xlabel='Within-core E→E multiplier',xticks=EES,ylim=(0,None))
    axes[0].set_ylabel('Mean population rate (Hz / E cell)');axes[1].set_ylabel('SD of population rate (Hz / E cell)')
    axes[0].legend(frameon=False,fontsize=12)
    fig.suptitle('Population response without burst-detection thresholds',fontsize=19,y=1.0)
    fig.text(.5,.035,'All native 2 ms bins, 2–20 s  ·  depth = 1  ·  two cores × two noise seeds',ha='center',fontsize=12)
    fig.subplots_adjust(bottom=.23,top=.86,wspace=.3)
    save(fig,'11_unthresholded_population_response','使用完整2–20秒的全部原生2ms群体率，计算均值与时间标准差；不经过burst起止阈值或事件筛选。两core、两噪声的身份分别保留。','检查可见活动起始是否只是检测阈值造成；这里仍是有限噪声网络的参数响应，不把采样点之间的连线当成稳定性边界。')

def quiet_branch():
    path=OUT/'quiescent_branch.json'
    if not path.exists():return
    d=read(path)
    if d['status']!='VALID_SILENT_EQUILIBRIUM':return
    fig,axes=plt.subplots(1,2,figsize=(13,5.7))
    x=np.array(EES)
    axes[0].plot(x,np.full(len(x),d['leading_growth_per_s']),'o-',color='#327981',ms=7,lw=2)
    axes[0].axhline(0,color='#555',lw=1,ls='--')
    axes[0].set(xlabel='Within-core E→E multiplier',ylabel='Leading neuronal growth rate (1/s)',ylim=(-60,5),xticks=EES)
    groups=['coreAE','coreBE','surroundE','allI'];vals=[d['region_summaries'][g]['margin_min_mV'] for g in groups]
    axes[1].bar(range(4),vals,color=['#247985','#a15387','#9faab1','#8c9f66'],width=.6)
    axes[1].set(xticks=range(4),xticklabels=['Core A E','Core B E','Surround E','All I'],ylabel='Minimum threshold margin (mV)',ylim=(0,None))
    axes[1].tick_params(axis='x',labelsize=11)
    fig.suptitle('Exact silent branch of the native deterministic network',fontsize=19,y=1.0)
    fig.text(.5,.035,'Positive threshold margin: no local spike feedback  ·  this is not the noisy active population state',ha='center',fontsize=12)
    fig.subplots_adjust(bottom=.22,top=.85,wspace=.31)
    save(fig,'12_native_quiescent_branch_stability','直接从当前原生离散方程计算完全确定输入下的无spike平衡点，核对全部细胞的阈值裕量和一步更新残差。该分段内递归spike映射导数为零，EE不进入雅可比；连续神经元/突触状态的特征值来自膜和滤波衰减，延迟队列移位为幂零。','这条无噪声静息分支在所扫EE上保持局部稳定；该结论不排除非零噪声背景的群体Hopf、有限扰动吸引子或其他分岔。',d)

def main():
    global BOOK
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.labelsize':15,'axes.titlesize':16,
        'xtick.labelsize':12,'ytick.labelsize':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    FIG.mkdir(exist_ok=True)
    base=read(OUT/'baseline_observables.json');comps=read(OUT/'noise_comparisons.json')
    with PdfPages(FIG/'core_burst_onset_v1_booklet.pdf') as book:
        BOOK=book
        phase_map(base);onset(base);unthresholded_response(base);regularization(base);intermediate(base);state_waveforms();state_waveforms(fine=True)
        noise_summary(comps);intervention_rasters();probe_summary(comps);pulse_closeup();recovery_traces();cell_irregularity();quiet_branch();benchmark()
    README.append('### core_burst_onset_v1_booklet.pdf\n全部候选图的多页矢量审阅包，每页保留完整尺寸。原生SNN与文献Hopf基准分开成页。\n**关注点**：按参数区域、连续指标、代表raster、噪声干预及文献基准的顺序审阅；作者目视验收待定。\n')
    (FIG/'README.md').write_text('# Core burst起始第一版图\n\n'+'\n'.join(README))
    write(OUT/'figure_manifest.json',dict(status='GENERATED_REVIEW_PENDING',figures=MANIFEST))
    print(json.dumps(dict(figures=len(MANIFEST),booklet=str(FIG/'core_burst_onset_v1_booklet.pdf'))))

if __name__=='__main__':main()
