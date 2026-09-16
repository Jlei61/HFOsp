"""Measured Z/adaptation-current phase trajectories; no inferred nullclines."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/topic4_sef_hfo'
OUT=ROOT/'results/paper-ready-figure/fig5_E_z_adaptation_phase_preview/figures'
IDS=['support_rank__vth_low','old_joint__tau_d_GABA_ms_high']


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    series=[];meta=[]
    for cid in IDS:
        p=json.loads((BASE/'fig5_two_gif_bases_recovery'/cid/'protocol.json').read_text())
        path=BASE/'fig5_two_gif_bases_recovery'/cid/'trajectory.npz'
        if not path.exists():path=BASE/'fig5_two_gif_bases'/cid/'trajectory.npz'
        with np.load(path) as a:arr={k:a[k] for k in a.files}
        onset=p['states_ms']['pre_onset']+250
        t=arr['slow_time_ms'];eta=p['job']['config']['eta_m']
        keep=(t>=200)&(t<=onset+2000)
        z=arr['slow_z_core_mean'][keep];adapt=eta*arr['slow_m_core_mean'][keep];times=t[keep]
        rates=np.interp(times,arr['time_ms'],arr['rates_hz'][:,0])
        series.append((z,adapt,times,rates,onset))
        meta.append(dict(candidate_id=cid,source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),eta_m=eta,
            start_ms=float(times[0]),end_ms=float(times[-1]),recruitment_onset_ms=onset,
            onset_Z=float(np.interp(onset,times,z)),onset_adaptation=float(np.interp(onset,times,adapt)),
            aggregation='Neuron-weighted union-core mean Z and eta_M*M. This projection does not assert equality of core states.',
            color='Population E rate from 20-ms bins, linearly interpolated to 10-ms slow recording times.'))
    fig,axes=plt.subplots(2,2,figsize=(11.8,8.0),gridspec_kw={'height_ratios':[1.35,1]},layout='constrained')
    norm=PowerNorm(.5,vmin=0,vmax=500)
    ymax=max(float(a.max()) for z,a,t,r,onset in series)*1.08
    zoommax=max(float(a[t<onset].max()) for z,a,t,r,onset in series)*1.15
    zmin=min(float(z.min()) for z,a,t,r,onset in series)-.02
    zoomxmin=min(float(z[t<onset].min()) for z,a,t,r,onset in series)-.015
    for col,(z,adapt,t,rates,onset) in enumerate(series):
        for row in range(2):
            ax=axes[row,col]
            keep=np.ones(len(t),bool) if row==0 else t<=onset
            points=np.c_[z[keep],adapt[keep]]
            lines=LineCollection(np.stack([points[:-1],points[1:]],axis=1),cmap='viridis',norm=norm,linewidths=1.8 if row==0 else 1.25)
            lines.set_array((rates[keep][:-1]+rates[keep][1:])/2);ax.add_collection(lines)
            ax.set_xlim(1.,zmin if row==0 else zoomxmin);ax.set_ylim(0,ymax if row==0 else zoommax)
            ax.set_xlabel('Inhibitory efficacy Z   (weaker inhibition →)')
            if col==0:ax.set_ylabel('Adaptation current $A = \\eta_M M$ (a.u.)')
            x=float(np.interp(onset,t,z));y=float(np.interp(onset,t,adapt))
            ax.scatter([z[0]],[adapt[0]],s=40,facecolor='white',edgecolor='black',zorder=6)
            ax.scatter([x],[y],s=110,marker='*',facecolor='#ed7730',edgecolor='black',lw=.65,zorder=7)
            if row==0:
                ax.annotate(f'Recruitment begins\n{onset/1000:.2f} s',xy=(x,y),xytext=(.27,.36),textcoords='axes fraction',fontsize=9,
                    arrowprops=dict(arrowstyle='->',color='#555555',lw=.8))
                # Direction markers on actual trajectory, not a vector field.
                for target in [onset+600,onset+1400]:
                    i=int(np.searchsorted(t,target));j=min(len(t)-1,i+5)
                    if i<len(t)-1:ax.annotate('',xy=(z[j],adapt[j]),xytext=(z[i],adapt[i]),arrowprops=dict(arrowstyle='->',lw=1,color='#555555'))
            else:ax.set_title('Interictal excursions · enlarged',fontsize=10)
        label='Base 1 · threshold gain 0.7 · GABA 18 ms' if col==0 else 'Base 2 · threshold gain 1.0 · GABA 24 ms'
        axes[0,col].set_title(label,fontsize=11,pad=10)
    fig.colorbar(lines,ax=axes.ravel().tolist(),shrink=.7,pad=.02,label='Population E firing rate (Hz)',ticks=[0,25,100,250,500])
    fig.suptitle('E candidate: the measured trajectory in Z–adaptation space\nTrajectory projection only · stability boundary and nullclines not yet computed',fontsize=14)
    handles=[Line2D([],[],marker='o',markerfacecolor='white',color='black',linestyle='none',label='Start after 200-ms transient'),Line2D([],[],marker='*',markerfacecolor='#ed7730',color='black',markersize=10,linestyle='none',label='Earliest sustained regional recruitment')]
    axes[0,0].legend(handles=handles,loc='upper left',fontsize=8,frameon=False)
    stem=OUT/'fig5-E-z-adaptation-phase-preview'
    fig.savefig(stem.with_suffix('.png'),dpi=180);fig.savefig(stem.with_suffix('.pdf'));plt.close(fig)
    stem.with_suffix('.json').write_text(json.dumps(dict(status='MEASURED_PHASE_PROJECTION_PREVIEW',nullclines_computed=False,stability_boundary_computed=False,author_accepted=False,series=meta),indent=2)+'\n')
    (OUT/'README.md').write_text('### fig5-E-z-adaptation-phase-preview.png\n两个固定 GIF 基底的实际 SNN 轨迹投影到双核合并 Z 与适应电流坐标；颜色为群体 E 放电率，下排放大转变前事件。星号是从实际区域招募得到的时刻，不是已证明的分叉点；没有绘制尚未计算的 nullcline 或稳定性边界。\n**关注点**：这张图检验新的 E 坐标是否能显示慢抑制变化与快速适应起伏；双核均值投影不等于闭合二维动力系统。\n')
    print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
