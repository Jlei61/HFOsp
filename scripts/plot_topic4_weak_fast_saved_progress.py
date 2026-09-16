#!/usr/bin/env python3
"""Plot actual archived weak-fast M summaries; never reconstruct missing spikes."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/topic4_sef_hfo/fig5_manual_core_release_v1/m_runaway_return_v1'
SOURCE=BASE/'interrupted_attempt_20260911/weak_fast.json'
OUT=BASE/'weak_fast_interim';FIG=OUT/'figures'


def main():
    record=json.loads(SOURCE.read_text());history=record['history'];job=record['job']
    assert job['eta_m']==.02 and job['tau_adp_ms']==2000 and not job['refill']
    t=np.array([v['time_s'] for v in history]);rate=np.array([v['E_mean_hz'] for v in history])
    z=np.array([v['mean_Z'] for v in history]);m=np.array([v['mean_M'] for v in history])
    current=np.array([v['mean_adaptation_current'] for v in history])
    assert np.allclose(np.diff(np.r_[0,t]),.5)
    assert np.max(abs(current-.02*m))<1e-12
    confirmed=record['first_trigger_ms']/1000
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':17,'axes.labelsize':19,
        'xtick.labelsize':17,'ytick.labelsize':17,'axes.titlesize':19,'pdf.fonttype':42,
        'axes.spines.right':False,'axes.spines.top':False})
    fig,axes=plt.subplots(3,2,figsize=(15.5,10),sharex='col',sharey='row',
        gridspec_kw={'width_ratios':[1.7,1],'hspace':.16,'wspace':.12})
    for col in range(2):
        axes[0,col].stairs(rate,np.r_[0,t],color='#2b5675',linewidth=1.5,baseline=None)
        axes[1,col].plot(np.r_[0,t],np.r_[1,z],color='#6c3f8c',lw=1.9)
        axes[2,col].plot(np.r_[0,t],np.r_[0,current],color='#b06b23',lw=1.9)
        if col==1:
            for row,values,color in [(1,z,'#6c3f8c'),(2,current,'#b06b23')]:
                sel=t>=68;axes[row,col].plot(t[sel],values[sel],'o',ms=3.5,c=color)
        for row in range(3):
            ax=axes[row,col];ax.axvspan(confirmed,t[-1],color='#bc3e46',alpha=.085,lw=0)
            ax.axvline(confirmed,c='#a73443',ls='--',lw=1.35)
            ax.set_xlim((0,t[-1]) if col==0 else (68,t[-1]))
            ax.grid(axis='y',color='#e2e2e2',lw=.7)
        axes[0,col].set_ylim(0,520);axes[0,col].set_yticks([0,200,400])
        axes[1,col].set_ylim(0,1.04);axes[1,col].set_yticks([0,.5,1])
        axes[2,col].set_ylim(0,20.5);axes[2,col].set_yticks([0,10,20])
        axes[2,col].set_xlabel('Time (s)')
    axes[0,0].set_title(r'$\eta_M=0.02,\quad\tau_M=2\ \mathrm{s}$',loc='left',pad=15)
    axes[0,1].set_title('Transition detail',loc='left',pad=15)
    axes[0,1].text(confirmed-.25,500,'73.68 s',ha='right',va='top',color='#a73443',fontsize=17)
    axes[0,0].set_ylabel('Mean E rate (Hz)\n0.5-s bins')
    axes[1,0].set_ylabel('Mean resource Z')
    axes[2,0].set_ylabel('Adaptation current\nηM × mean M (mV-eq.)')
    axes[2,0].set_xticks([0,20,40,60,80]);axes[2,1].set_xticks([68,72,76,80])
    fig.subplots_adjust(left=.13,right=.985,top=.94,bottom=.09)
    FIG.mkdir(parents=True,exist_ok=True)
    path=FIG/'weak_fast_saved_trajectory_80p5s'
    fig.savefig(path.with_suffix('.png'),dpi=170,bbox_inches='tight',pad_inches=.15)
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.15);plt.close(fig)
    metadata=dict(status='PARTIAL_OBSERVATION_NOT_COMPLETE_RUN',source=str(SOURCE),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),job=job,
        observed_until_s=float(t[-1]),number_of_summary_intervals=len(t),summary_interval_s=.5,
        E_rate='Mean full-E firing rate over preceding0.5s interval; step rendering; fast bursts are not resolved.',
        Z_and_M='Actual E-population means at each interval endpoint; points joined for display; initial Z=1,M=0.',
        high_confirmation_s=confirmed,
        high_criterion='Original online10ms E-rate>=200Hz for200 consecutive ms; not inferred from displayed0.5s bins.',
        final_summary=history[-1],M_current_identity_pass=True,
        no_extrapolation=True,no_synthetic_spikes_or_SEEG=True,formal_Fig5=False,
        producer=str(Path(__file__).resolve()))
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (FIG/'README.md').write_text('### weak_fast_saved_trajectory_80p5s.png\n展示ηM=0.02、τM=2秒条件前次真实运行保留的0–80.5秒汇总轨迹，右列放大68–80.5秒。放电率为0.5秒均值；Z与M电流为区间末实际全E均值，虚线73.68秒来自原10ms在线判据。该运行未完成90秒，完整SEEG/raster正在同种子重演，不能从本图读出逐事件快速振荡。\n**关注点**：M电流上升的同时Z仍下降，高放电态截至80.5秒尚未返回；图没有补造丢失的原始spike或空间场。\n')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
