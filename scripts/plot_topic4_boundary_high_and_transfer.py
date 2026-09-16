#!/usr/bin/env python3
"""Native long-duration state comparison and independently measured transfer bias."""
from plot_topic4_spatial_boundary_results import save,raster
from topic4_spatial_boundary_common import OUT,read,write,observables
import numpy as np
import matplotlib.pyplot as plt


def high_boundary():
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained');rows=[]
    for row,z in enumerate((8800,9400)):
        a=np.load(OUT/'native'/f'z{z}_history8000.npz');b=np.load(OUT/'native'/f'z{z}_history8000_extend4s.npz')
        data={k:np.concatenate([a[k],b[k]]) for k in ('sample_spikes','rate_e_hz','field_e_count_1ms')}
        raster(axs[row,0],data,6.)
        axs[row,0].set_title(f"Native Z field from {z/1000:g} s | mean Z = {a['initial_z_e'].mean():.4f}")
        e=data['rate_e_hz'].reshape(-1,50).mean(1)
        axs[row,1].plot((np.arange(len(e))+.5)*.005,e,color='#222222',lw=.8)
        axs[row,1].set(xlabel='Time after Z fixation (s)',ylabel='Mean E rate (Hz)',ylim=(0,320),title='All E neurons; identical future noise')
        for col in (0,1):axs[row,col].axvline(2,color='#3c8c7a',ls='--',lw=.8)
        rate10=data['field_e_count_1ms'][-1000:].astype(float).reshape(100,10,400).sum(1)/a['cell_e_counts']/.01
        duty=np.mean(rate10>50,axis=0)
        im=axs[row,2].imshow(duty.reshape(20,20),origin='lower',extent=[0,20,0,20],vmin=0,vmax=1,cmap='magma')
        axs[row,2].set(xlabel='x (mm)',ylabel='y (mm)',title='Last 1 s: high-rate spatial duty')
        rows.append({'z_profile_ms':z,'history_ms':8000,'mean_Z':float(a['initial_z_e'].mean()),
            'late_metrics':observables(data['field_e_count_1ms'],a['cell_e_counts']),
            'duration_ms':6000,'source_files':[str(OUT/'native'/f'z{z}_history8000.npz'),str(OUT/'native'/f'z{z}_history8000_extend4s.npz')]})
    fig.colorbar(im,ax=axs[:,2],label='Fraction of 10-ms bins >50 Hz',shrink=.8)
    fig.suptitle('Original SNN: fixed Z controls self-limited events versus sustained local recruitment',fontsize=15)
    save(fig,'native_six_second_state_boundary',
        '从相同的完整 8 秒快速状态开始，使用同一未来输入，分别冻结原轨迹 8.8 秒与 9.4 秒的逐神经元 Z 场，连续观察 6 秒。上排恢复有间隔的自限事件，下排在最后一秒仍有局部持续高率活动；Z 在全程均保持原值。',
        '虚线为完整状态保存/续算点，不是刺激或复位；本图支持有限时间内的 Z 场控制状态差别，还没有识别经典分岔类型，也未证明全空间均进入同一种振荡。')
    write(OUT/'native_six_second_boundary.json',{'rows':rows,'scientific_status':'FINITE_TIME_STATE_SEPARATION_SUPPORTED','bifurcation_type':'NOT_ESTABLISHED'})


def transfer():
    candidate=read(OUT/'mixed_transfer_candidate_diagnostic.json')['rows'];fig,axs=plt.subplots(1,2,figsize=(12,5.8),layout='constrained');summary=[]
    for col,pop in enumerate(('E','I')):
        cases=read(OUT/'mixed_transfer'/f'{pop}.json')['cases'];chosen=[r for r in candidate if r['population']==pop]
        x=np.array([r['native_LIF_rate_hz'] for r in cases]);old=np.array([r['existing_Phi_hz'] for r in cases])
        new=np.array([r['mixed_phi_GH5_9_21'][-1] for r in chosen])
        assert [r['case'] for r in chosen]==[r['name'] for r in cases]
        axs[col].plot([.01,1000],[.01,1000],c='#777777',ls='--',lw=.8)
        mixed=np.array([r['name'].startswith('t') for r in cases])
        for y,label,color in [(old,'Previous transfer','#ae3a6b'),(new,'Fast AMPA + slow GABA candidate','#337e9e')]:
            axs[col].scatter(x[mixed],y[mixed],color=color,s=45,label=label)
            axs[col].scatter(x[~mixed],y[~mixed],edgecolor=color,facecolor='none',s=45)
        axs[col].set(xscale='symlog',yscale='symlog',xlim=(0,1000),ylim=(0,1000),
            xlabel='Simulated isolated native-LIF rate (Hz)',ylabel='Approximate transfer rate (Hz)',title=f'{pop} population: no parameter fitting')
        axs[col].legend(fontsize=9);axs[col].text(.04,.95,'Filled: mixed input\nOpen: excitation only',transform=axs[col].transAxes,va='top',fontsize=9)
        high=mixed&(x>100)
        summary.append({'population':pop,'high_context_old_relative_errors':(old[high]/x[high]-1).tolist(),
            'high_context_candidate_relative_errors':(new[high]/x[high]-1).tolist(),
            'scope':'Stationary isolated moment-matched assays; candidate values use 21-node quadrature on these points. Full network validation uses 33 nodes, whose integration is separately checked.'})
    fig.suptitle('The transfer approximation underestimates I more strongly than E',fontsize=15)
    save(fig,'mixed_input_native_lif_transfer_audit',
        '保留原 LIF 膜、复位和双指数突触方程，在隔离仿真及Phi中配对使用现有8节点E阈值支持；I阈值保持原值。独立Poisson输入的一二矩来自旧rate原Z回放的边界上下文，并非原SNN电流直接回放；填充点为混合输入，空心点为仅兴奋性输入，蓝色为未拟合的快AMPA/慢GABA候选。',
        '旧近似对高驱动 I 细胞的低估明显大于 E；隔离静态响应改善不能直接视为完整网络的自主 Z 动力学或频率响应已经通过。')
    write(OUT/'mixed_transfer_comparison.json',summary)


if __name__=='__main__':high_boundary();transfer()
