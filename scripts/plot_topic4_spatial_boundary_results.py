#!/usr/bin/env python3
"""Render completed boundary diagnostics; never declare scientific acceptance."""
from topic4_spatial_boundary_common import OUT, OLD, REFERENCE, read, write, observables
from analyze_topic4_spatial_boundary_factorial import analyze
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import argparse
import os
from scipy.signal import welch

FIG=OUT/'figures'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})


def save(fig,name,description,focus):
    FIG.mkdir(exist_ok=True);fig.savefig(FIG/f'{name}.png',bbox_inches='tight');fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    path=FIG/'README.md';previous=path.read_text() if path.exists() else ''
    sections=['### '+p for p in previous.split('### ') if p.strip() and not p.startswith(name+'\n')]
    path.write_text(''.join(sections)+f'\n### {name}\n\n{description}\n\n**关注点**：{focus} PNG/PDF 待用户目视审阅。\n')


def raster(ax,a,duration=None):
    ids=np.r_[np.arange(0,240,4),np.arange(240,300,4)]
    tt,nn=np.where(a['sample_spikes'][:,ids]);ax.scatter(tt*.0001,nn,s=.25,
        c=np.where(ids[nn]<240,'#262626','#3e7faf'),rasterized=True)
    ax.set(ylim=(-1,len(ids)),xlabel='Time after Z fixation (s)',ylabel='Sampled neuron')
    if duration is not None:ax.set_xlim(0,duration)


def native_histories():
    profiles=[8000,8400,8800,9200,9400];fig,axs=plt.subplots(5,4,figsize=(17,13),layout='constrained')
    for row,z in enumerate(profiles):
        pair=[np.load(OUT/'native'/f'z{z}_history{h}.npz') for h in (8000,9400)]
        for col,a in enumerate(pair):
            raster(axs[row,col],a,2.)
            axs[row,col].set_title(f'Z from {z/1000:g} s | history {8 if col==0 else 9.4:g} s')
            e=a['rate_e_hz'].reshape(-1,50).mean(1)
            axs[row,2].plot((np.arange(len(e))+.5)*.005,e,lw=.9,c=('#aa6434','#326d9c')[col],label=('History 8.0 s','History 9.4 s')[col])
        axs[row,2].set(xlabel='Time after Z fixation (s)',ylabel='Mean E rate (Hz)',ylim=(0,320));axs[row,2].legend(fontsize=8)
        duty=[]
        for a in pair:
            rates=a['field_e_count_1ms'][-1000:].astype(float).reshape(100,10,400).sum(1)/a['cell_e_counts']/ .01
            duty.append(np.mean(rates>50,axis=0))
        im=axs[row,3].imshow((duty[0]-duty[1]).reshape(20,20),origin='lower',extent=[0,20,0,20],vmin=-1,vmax=1,cmap='RdBu_r')
        axs[row,3].set(xlabel='x (mm)',ylabel='y (mm)',title='Spatial duty difference')
    fig.colorbar(im,ax=axs[:,3],label='High-rate duty: history 8.0 minus 9.4',shrink=.65)
    fig.suptitle('Native SNN | same Z and future input, different complete activity histories',fontsize=15)
    save(fig,'native_paired_history_rasters',
        '五个冻结 Z 条件分别展示两种初始历史的真实放电 raster、全 E 群体率和最后一秒的空间占据差异。每组使用完全相同的未来 OU 和私有输入随机创新；raster 的 75 个神经元沿用原参考抽样，群体和空间指标使用全部 E 神经元。',
        '红蓝空间图表示高率时间占据差，而非两个传播方向；注意较高平均率、失去安静间隔和持续空间活动可以不同步出现。')


def adaptive():
    jobs=read(OUT/'native_adaptive_protocol.json')['jobs'];fig,axs=plt.subplots(4,3,figsize=(15,11),layout='constrained');rows=[]
    for row,job in enumerate(jobs):
        a=np.load(OUT/'native'/f"{job['name']}.npz")
        if job.get('resume_endpoint'):
            first=np.load(OUT/'native'/f"z8800_history{job['history_ms']}.npz")
            data={k:np.concatenate([first[k],a[k]]) for k in ('sample_spikes','rate_e_hz','field_e_count_1ms')}
            label=f"History {job['history_ms']/1000:g} s | uninterrupted 6 s"
        else:
            data={k:a[k] for k in ('sample_spikes','rate_e_hz','field_e_count_1ms')}
            label=f"History 9.4 s | future seed {job['future_seed']}"
        duration=len(data['rate_e_hz'])*.0001;raster(axs[row,0],data,6.)
        axs[row,0].set_title(label,fontsize=10)
        rate=data['rate_e_hz'].reshape(-1,50).mean(1)
        axs[row,1].plot((np.arange(len(rate))+.5)*.005,rate,color='#262626',lw=.8)
        axs[row,1].set(xlim=(0,6),ylim=(0,320),xlabel='Time after Z fixation (s)',ylabel='Mean E rate (Hz)')
        if job.get('resume_endpoint'):
            for col in (0,1):axs[row,col].axvline(2,color='#348675',ls='--',lw=.8)
        else:
            for col in (0,1):
                axs[row,col].axvspan(duration,6,color='#ececec',zorder=-1)
                axs[row,col].text(.83,.85,'Outside\nobservation',transform=axs[row,col].transAxes,
                    ha='center',va='center',fontsize=9,color='#666666')
        field=data['field_e_count_1ms'];count=a['cell_e_counts']
        duty=np.mean(field[-1000:].astype(float).reshape(100,10,400).sum(1)/count/.01>50,axis=0)
        im=axs[row,2].imshow(duty.reshape(20,20),origin='lower',extent=[0,20,0,20],vmin=0,vmax=1,cmap='magma')
        axs[row,2].set(xlabel='x (mm)',ylabel='y (mm)',title='Last 1 s: spatial duty')
        windows=[observables(field[k:k+1000],count) for k in range(0,len(field),1000)]
        rows.append({'job':job,'total_duration_ms':len(field),'windows':windows})
    fig.colorbar(im,ax=axs[:,2],label='Fraction of 10-ms bins >50 Hz',shrink=.7)
    fig.suptitle('History and future-noise controls at the same 8.8-s Z field',fontsize=15)
    save(fig,'native_boundary_extension_and_noise',
        '同一 8.8 秒 Z 场下，将两种历史的 2 秒轨迹不重置地延长至 6 秒，并从 9.4 秒历史更换两组未来噪声各观察 4 秒。右列是每条轨迹最后一秒的空间高率时间占据。',
        '虚线只标记保存点的继续运行，没有刺激或状态重置；持续的历史差异仍须区分长瞬态与渐近多稳态，三个未来种子不支持精确转移概率。')
    write(OUT/'adaptive_metrics.json',rows)


def resolution():
    ref=np.load(REFERENCE/'trajectory.npz');fig,axs=plt.subplots(2,2,figsize=(14,8),layout='constrained');rows=[]
    native_e=ref['rate_e_hz'].reshape(-1,50).mean(1);native_t=(np.arange(len(native_e))+.5)*.005
    for col,mode in enumerate(('native_replay','autonomous_gaussian')):
        axs[0,col].plot(native_t,native_e,c='k',lw=.8,label='Native SNN')
        axs[1,col].plot(ref['z_time_ms']/1000,ref['z_stats'][:,0],c='k',label='Native SNN')
        for grid,color in [(10,'#ae3a6b'),(20,'#3b7e9d')]:
            path=OLD/'rate'/f'{mode}_expected.npz' if grid==10 else OUT/'resolution'/f'grid20_{mode}.npz'
            a=np.load(path);e=np.average(a['fields_hz'][:,0],axis=1,weights=a['count_e']);z=np.average(a['z'],axis=1,weights=a['count_e'])
            sm=e.reshape(-1,5).mean(1);axs[0,col].plot((np.arange(len(sm))+.5)*.005,sm,c=color,lw=.8,label=f'Rate {grid} x {grid}')
            axs[1,col].plot((np.arange(len(z))+1)*.001,z,c=color,label=f'Rate {grid} x {grid}')
            e10=e.reshape(-1,10).mean(1);hit=np.flatnonzero(np.convolve((e10>=200).astype(int),np.ones(20,dtype=int),'valid')==20)
            rows.append({'mode':mode,'grid':grid,'trigger_ms':float((hit[0]+20)*10) if len(hit) else None,
                'high_E_hz':float(e[10180:10680].mean()),'final_E_hz':float(e[-1000:].mean()),
                'spatial_Z_rmse_at_8s':float(np.sqrt(np.average((a['z'][7999]-native_z_for_grid(ref,grid))**2,weights=a['count_e'])))})
        axs[0,col].set(title=('Prescribed native Z','Autonomous original-Z closure')[col],ylabel='Mean E rate (Hz)',xlabel='Time (s)')
        axs[1,col].set(ylabel='Mean E-target Z',xlabel='Time (s)')
        for row in (0,1):
            axs[row,col].legend(fontsize=9);axs[row,col].axvspan(10.68,11.68,color='#58ac8a',alpha=.15);axs[row,col].set_xlim(0,13.68)
    fig.suptitle('Spatial resolution sensitivity | all biological parameters unchanged',fontsize=15)
    save(fig,'native_rate_spatial_resolution_comparison',
        '将同一模型从 10×10 细化到 20×20，分别使用原 SNN 的 Z 回放和原 Z 方程的自主平均化；背景随机过程完全重放后重新投影。比较完整放电率和 Z 轨迹，绿色区域为同一外部恢复协议。',
        '细化后的潜伏期改善不等于整个动力学已经匹配，还需核对固定 Z 下的自限事件、持续空间范围和局部振荡。')
    write(OUT/'resolution_comparison.json',{'rows':rows,'native_trigger_ms':10180,'scope':'Fixed C substrate; 20x20 input RNG QA passed, biological parameters not fitted.'})


def native_z_for_grid(ref,grid):
    if grid==20:return ref['z_field_10ms'][800]
    source=np.load(OLD/'external_input.npz');ce10=source['cell_e'];ce20=ref['cell_e']
    mapping=np.bincount(ce10*400+ce20,minlength=100*400).reshape(100,400)/source['count_e'][:,None]
    return mapping@ref['z_field_10ms'][800]


def local_activity():
    name='z9400_history9400';native=np.load(OUT/'native'/f'{name}.npz');rate=np.load(OUT/'rate'/f'{name}.npz')
    source=np.load(OLD/'external_input.npz');ce10=source['cell_e'];ci10=source['cell_i']
    ae=np.zeros((100,400));ae[ce10,native['cell_e']]=1
    ai=np.zeros((100,400));ai[ci10,native['cell_i']]=1
    e_native=(native['field_e_count_1ms']@ae.T)/source['count_e']*1000
    i_native=(native['field_i_count_1ms']@ai.T)/source['count_i']*1000
    tail=e_native[1000:].reshape(200,5,100).mean(1)
    persistent=int(np.argmax(tail.mean(0)));variable=int(np.argmax(tail.std(0)))
    weights=source['count_e'];fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained');rows=[]
    for row,(model,e,i) in enumerate([('Native SNN',e_native,i_native),('Rate 10 x 10',rate['fields_hz'][:,0],rate['fields_hz'][:,1])]):
        global_e=np.average(e,axis=1,weights=weights)
        traces=[global_e,e[:,persistent],e[:,variable]];labels=['All E',f'Highest-mean cell {persistent}',f'Most-variable cell {variable}']
        colors=['#333333','#aa6434','#3679a3']
        spectra=[]
        for values,label,color in zip(traces,labels,colors):
            y=values.reshape(-1,5).mean(1);tt=(np.arange(len(y))+.5)*.005
            axs[row,0].plot(tt,y,c=color,lw=.9,label=label)
            f,p=welch(values[-1000:],fs=1000,nperseg=500,noverlap=250,detrend='constant')
            mask=(f>=2)&(f<=200);axs[row,2].semilogy(f[mask],np.maximum(p[mask],1e-12),c=color,label=label)
            spectra.append({'trace':label,'largest_PSD_bin_2_to_200Hz':float(f[mask][np.argmax(p[mask])]),
                'late_mean_Hz':float(values[-1000:].mean()),'late_SD_5ms_Hz':float(y[-200:].std())})
        axs[row,0].set(xlabel='Time after Z fixation (s)',ylabel='E rate (Hz)',ylim=(0,500),title=f'{model}: local and population activity');axs[row,0].legend(fontsize=8)
        ex=e[1000:,variable].reshape(-1,5).mean(1);iy=i[1000:,variable].reshape(-1,5).mean(1)
        axs[row,1].plot(ex,iy,c='#87949b',lw=.7,alpha=.6)
        sc=axs[row,1].scatter(ex,iy,c=np.arange(len(ex))*.005+1,cmap='viridis',s=9,zorder=3)
        for k in range(0,len(ex)-1,20):
            axs[row,1].annotate('',xy=(ex[k+1],iy[k+1]),xytext=(ex[k],iy[k]),arrowprops={'arrowstyle':'->','color':'#384b5b','lw':.8})
        axs[row,1].set(xlabel=f'Cell {variable} E rate (Hz)',ylabel=f'Cell {variable} I rate (Hz)',xlim=(0,500),ylim=(0,1000),title='Observed E-I phase projection')
        axs[row,2].set(xlabel='Frequency (Hz)',ylabel='PSD (Hz squared / Hz)',ylim=(1e-9,1e5),title='Last 1 s; no predefined oscillation band');axs[row,2].legend(fontsize=8)
        rows.append({'model':model,'spectra':spectra})
    fig.colorbar(sc,ax=axs[:,1],label='Time after fixation (s)',shrink=.6)
    fig.suptitle('Local activity near sustained recruitment | frozen 9.4-s Z, common history',fontsize=15)
    save(fig,'local_activity_and_ei_phase_projection',
        '在固定 9.4 秒 Z 场及相同历史条件下，比较原 SNN 和 rate model 的群体、最高均值格及最高波动格的真实时间序列和频谱；两格按原 SNN 最后一秒选定，在 rate model 中保持相同空间位置。中列为最高波动格实际 E-I 轨迹，颜色及箭头表示时间方向。',
        '相轨迹投影的箭头是实际运动方向，不能当成封闭二维系统的方向场；这里没有伪造 nullcline。频谱峰与局部波动本身不构成持续极限环或 Hopf 的证明。')
    write(OUT/'local_activity_metrics.json',{'condition':name,'highest_mean_native_cell':persistent,
        'most_variable_native_cell':variable,'selection':'Exploratory native-cell selection, held fixed in rate comparison.',
        'frequency_resolution_Hz':2,'rows':rows,'bifurcation_type':'NOT_ESTABLISHED'})


def main(wait_for_results=False):
    if wait_for_results:
        for name in ('native_batch','native_adaptive','resolution'):
            while True:
                p=OUT/f'{name}_status.json';status=read(p) if p.exists() else {}
                if status.get('status')=='COMPLETE':break
                if status.get('status')=='FAILED':raise RuntimeError(status)
                os.kill(read(OUT/f'{name}_process.json')['pid'],0);time.sleep(15)
    analyze();native_histories();adaptive();resolution();local_activity()
    write(OUT/'render_status.json',{'status':'COMPLETE_PENDING_AGENT_REVIEW',
        'files':sorted(p.name for p in FIG.glob('*.png')),'human_visual_acceptance':False,
        'meaning':'Rendering and observables complete; inspect numerical results, figures and limitations before closeout.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    try:main(args.wait)
    except Exception as exc:
        write(OUT/'render_status.json',{'status':'FAILED','error':repr(exc)});raise
