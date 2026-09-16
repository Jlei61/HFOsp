#!/usr/bin/env python3
"""Continuous phase projections of the user's exact native-Z reference run."""
from analyze_topic4_prescribed_z_phase import OUT,REFERENCE,read,write
from plot_topic4_prescribed_z_phase import save
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    a=np.load(REFERENCE/'trajectory.npz');z=a['z_stats'];zt=a['z_time_ms']/1000
    rates=np.c_[a['rate_e_hz'],a['rate_i_hz']].reshape(-1,50,2).mean(1);tt=(np.arange(len(rates))+.5)*.005
    zm=np.interp(tt,zt,z[:,0]);e,i=rates.T
    stages=[(0,10.68,'#357aa1','Native Z evolution'),(10.68,11.68,'#238662','External Z refill'),(11.68,13.68,'#a05c8f','Z held at 1')]
    landmarks=[(8.8,'1'),(10.5,'2'),(11.1,'3'),(12.5,'4')]
    fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    axs[0,0].fill_between(zt,z[:,2],z[:,4],color='#c8b6d8',alpha=.45,label='E-neuron 10th–90th percentile')
    axs[0,0].plot(zt,z[:,0],c='#70388d',lw=1.6,label='Mean E-target Z')
    axs[0,0].set(xlim=(0,13.68),ylim=(.45,1.025),xlabel='Time (s)',ylabel='Z',title='The supplied native SNN reference')
    axs[0,0].legend(fontsize=9,loc='lower left')
    axs[0,1].plot(tt,e,c='#357aa1',lw=.8,label='All E');axs[0,1].plot(tt,i,c='#da872d',lw=.8,label='All I')
    axs[0,1].set(xlim=(0,13.68),ylim=(0,500),xlabel='Time (s)',ylabel='Population rate (Hz; 5-ms bins)',title='Actual spiking activity')
    axs[0,1].legend(fontsize=10,loc='upper left')
    for ax in axs[0]:ax.axvspan(10.68,11.68,color='#238662',alpha=.10)
    for lo,hi,color,label in stages:
        ix=(tt>=lo)&(tt<=hi);xx=zm[ix];ee=e[ix];ii=i[ix]
        axs[1,0].plot(xx,ee,c=color,lw=.9,alpha=.9,label=label)
        axs[1,1].plot(ee,ii,c=color,lw=.8,alpha=.8)
        indices=np.flatnonzero(ix)
        # Time arrows sampled evenly in each protocol stage; omit tiny moves.
        for k in np.linspace(0,len(indices)-5,8,dtype=int):
            j=indices[k];jj=j+3
            for ax,xxall,yyall in ((axs[1,0],zm,e),(axs[1,1],e,i)):
                if np.hypot((xxall[jj]-xxall[j])/(.4 if ax is axs[1,0] else 400),(yyall[jj]-yyall[j])/500)>.015:
                    ax.annotate('',xy=(xxall[jj],yyall[jj]),xytext=(xxall[j],yyall[j]),arrowprops={'arrowstyle':'->','color':color,'lw':1.2})
    rows=[]
    for time_s,label in landmarks:
        ix=int(np.argmin(abs(tt-time_s)));zz=float(zm[ix]);ee=float(e[ix]);ii=float(i[ix]);rows.append({'label':label,'requested_time_s':time_s,'rate_bin_center_s':float(tt[ix]),'mean_Z':zz,'E_hz':ee,'I_hz':ii})
        for ax,x,y in ((axs[0,0],time_s,np.interp(time_s,zt,z[:,0])),(axs[1,0],zz,ee),(axs[1,1],ee,ii)):
            ax.scatter(x,y,s=34,c='#e3a731',edgecolor='black',zorder=5)
            ax.annotate(label,(x,y),xytext=(7,8),textcoords='offset points',fontsize=10,fontweight='bold',bbox={'facecolor':'white','alpha':.75,'edgecolor':'none','pad':1})
    axs[1,0].set(xlim=(.64,1.01),ylim=(0,370),xlabel='Mean E-target Z',ylabel='Mean E rate (Hz)',title='One continuous Z–activity trajectory')
    axs[1,1].set(xlim=(0,370),ylim=(0,500),xlabel='Mean E rate (Hz)',ylabel='Mean I rate (Hz)',title='The same trajectory in E–I coordinates')
    handles=[Line2D([],[],color=c,lw=2,label=l) for _,_,c,l in stages]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False,fontsize=11)
    fig.suptitle('Native SNN: follow the same Z field through recruitment, external refill and return',fontsize=15)
    save(fig,'native_z_cycle_continuous_phase',
        '完全使用用户所指原图的同一个SNN trajectory.npz，连续展示0–13.68秒的Z、实际E/I群体率及两种相空间投影。蓝色为原Z自主演化，绿色为10.68–11.68秒外部补回，紫色为随后Z固定1；数字1–4对应8.8、10.5、11.1、12.5秒。',
        '相轨迹来自全部真实E/I神经元5毫秒计数，不由rate model生成。Z轴为全E均值的展示投影；模拟和关键快照仍保留整个空间场。不同时间经过相同平均Z不证明多稳态或分岔，返回是外部补回的结果。')
    fields=a['z_field_10ms'];fig,axs=plt.subplots(1,4,figsize=(14,4),layout='constrained')
    for ax,(tm,label) in zip(axs,landmarks):
        index=int(np.argmin(abs(zt-tm)));field=fields[index]
        im=ax.imshow(field.reshape(20,20),origin='lower',extent=[0,20,0,20],vmin=0,vmax=1,cmap='viridis')
        ax.set(title=f'{label}: {tm:g} s\nmean Z = {z[index,0]:.3f}',xlabel='x (mm)',ylabel='y (mm)')
    fig.colorbar(im,ax=axs,label='E-target Z',shrink=.8)
    save(fig,'native_z_cycle_key_spatial_fields',
        '从用户原SNN轨迹直接提取8.8、10.5、11.1、12.5秒的20×20空间Z场，对应连续相图的1–4号标记。四图保留同一空间坐标与0–1色标，展示耗竭至外部补回后的真实空间变化。',
        '这些是实际空间场，不能用平均Z替代模拟；最后的均匀Z=1由外部恢复和夹持产生。')
    write(OUT/'native_cycle_phase_metadata.json',{'source':str(REFERENCE/'trajectory.npz'),'reference_figure':str(REFERENCE/'figures/autonomous_z_manual_restore.png'),
        'rate_bin_ms':5,'Z_plot_interpolation':'Native mean Z linearly interpolated to rate-bin centres; markers use nearest 5-ms rate bin.',
        'Z_mean_reconstruction_error':float(np.max(abs(np.average(fields,axis=1,weights=a['cell_e_counts'])-z[:,0]))),
        'landmarks':rows,'manual_refill_ms':[10680,11680],'full_trajectory_duration_ms':13680,
        'interpretation':'Observed native trajectories only; no closed vector field inferred from path geometry.'})


if __name__=='__main__':main()
