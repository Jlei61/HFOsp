#!/usr/bin/env python3
"""Plot measured conditional responses, with no invented equilibrium branches."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from campaign import ROOT,NATIVE,read,write

COLORS=['#74398f','#d34e99','#249ac1']
HISTORY=dict(high=('-', 'o'),interictal=('--','s'),recovery=(':','^'))
TITLES=dict(entry='Entry: K = 0.0002',exit='Termination: Z = 0.21',
            return_='Brief-event return: Z = 0.995')


def main():
    data=read(NATIVE/'extended_analysis_summary.json');rows=data['rows']
    if not rows:raise RuntimeError('No complete trajectories to plot.')
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
                         'axes.spines.right':False,'axes.spines.top':False})
    fig,axes=plt.subplots(2,3,figsize=(13.5,7),sharex='col',layout='constrained')
    drift,da=plt.subplots(2,3,figsize=(13.5,7),sharex='col',layout='constrained')
    subsets=[]
    max_frequency=max([r['tail_brief_events']/10 for r in rows]+[1.])
    for col,cut in enumerate(['entry','exit','return']):
        coordinate='target_Z' if cut=='entry' else 'target_K'
        subset=[r for r in rows if r['job'].get('local_cut')==cut]
        subsets.append(subset)
        title=TITLES['return_' if cut=='return' else cut]
        axes[0,col].set_title(title);da[0,col].set_title(title)
        freq_ax=axes[1,col].twinx();freq_ax.spines['right'].set_visible(True)
        for history,(ls,marker) in HISTORY.items():
            items=sorted([r for r in subset if r['job']['source_history']==history],
                         key=lambda r:r['job'][coordinate])
            if not items:continue
            x=np.array([r['job'][coordinate] for r in items])
            means=np.array([r['tail_mean_Hz'][:3] for r in items])
            quant=np.array([r['tail_500ms_mean_quantiles_allE_A_B_other_Hz'] for r in items])
            for j,color in enumerate(COLORS):
                axes[0,col].plot(x,means[:,j],color=color,ls=ls,marker=marker,ms=4,lw=1.1)
                zz=np.array([r['counterfactual_drift_mean_allE_A_B_other'][j][0] for r in items])
                da[0,col].plot(x,zz,color=color,ls=ls,marker=marker,ms=4,lw=1.1)
                kk=np.array([r['counterfactual_drift_mean_allE_A_B_other'][j][1] for r in items])
                da[1,col].plot(x,kk,color=color,ls=ls,marker=marker,ms=4,lw=1.1)
            # These intervals are temporal variability, not confidence intervals.
            axes[0,col].vlines(x,quant[:,0,0],quant[:,2,0],color=COLORS[0],lw=2.5,alpha=.28)
            axes[1,col].plot(x,[r['tail_joint_quiet_fraction'] for r in items],
                             color='#5d6168',ls=ls,marker=marker,ms=4)
            freq_ax.plot(x,[r['tail_brief_events']/10 for r in items],
                         color='#237d65',ls=ls,marker=marker,ms=4)
            for xx,r in zip(x,items):
                cens=r['censoring']['final_right_censored']
                if cens is not None:
                    axes[0,col].scatter([xx],[r['tail_mean_Hz'][0]],marker='x',s=65,
                                        c='#333333',linewidths=.75,zorder=5)
        for ax in [*axes[:,col],*da[:,col]]:
            ax.grid(axis='y',alpha=.15);ax.margins(x=.08)
            if cut=='return':ax.set_xscale('log')
        axes[0,col].set_ylim((-1,50) if cut=='return' else (-8,510))
        axes[1,col].set_ylim(-.03,1.03);freq_ax.set_ylim(-.03*max_frequency,1.1*max_frequency)
        axes[1,col].set_ylabel('Joint-quiet fraction',color='#5d6168')
        freq_ax.set_ylabel('Complete brief events / s',color='#237d65')
        for ax in da[:,col]:ax.axhline(0,color='#777777',lw=.7,zorder=0)
        label='Held mean Z' if cut=='entry' else 'Held mean K (gK/gL)'
        axes[1,col].set_xlabel(label);da[1,col].set_xlabel(label)
    axes[0,0].set_ylabel('Native E rate (Hz)')
    da[0,0].set_ylabel('Mean natural dZ/dt (s$^{-1}$)')
    da[1,0].set_ylabel('Mean natural dK/dt (s$^{-1}$)')
    handles=[Line2D([],[],color=c,label=l) for c,l in zip(COLORS,['All E','Core A','Core B'])]
    handles += [Line2D([],[],color='#333333',ls=ls,marker=m,label=h+' history')
                for h,(ls,m) in HISTORY.items()]
    fig.legend(handles=handles,loc='outside upper center',ncol=6,frameon=False)
    drift.legend(handles=handles,loc='outside upper center',ncol=6,frameon=False)
    status=f"{data['completed']}/{data['total']} complete; final 10 s of each 30-s branch"
    fig.supxlabel(status+'\nConditional responses; lines guide the eye. Bars: temporal 10–90%; ×: unfinished final activity.',fontsize=9)
    drift.supxlabel(status+'\nZ/K are held; natural drift is read without releasing the clamps. G and M remain dynamic.',fontsize=9)
    for f,name in [(fig,'native_transition_cuts'),(drift,'native_recovery_drift_cuts')]:
        f.savefig(out/f'{name}.png',dpi=180);f.savefig(out/f'{name}.svg');plt.close(f)
    write(out/'native_cuts_metadata.json',dict(source=str(NATIVE/'extended_analysis_summary.json'),
        completed=data['completed'],total=data['total'],human_review='PENDING',
        certified_bifurcation=False,conditional_responses=True,
        interpretation='Markers are measured finite-window native responses. No equilibrium/periodic branch, convergence, stability or fold/Hopf is certified. Within-trajectory quantiles are not seed uncertainty.',
        source_histories='12s high,30s recovery,50s interictal; common t20 spatial Z/K template and paired future noise.'))
    (out/'README.md').write_text('''### native_transition_cuts.png / native_transition_cuts.svg
三条条件切片分别围绕进入、终止和短事件返回，展示末10秒全E/双核平均率、共同低活动占比和完整短事件频率。线型表示不同内源初值；竖条是同一轨迹500ms均值的10–90%分位，叉号表示最后活动尚未结束。图中点是原生有限窗条件响应，尚不是认证的平衡/周期分支，完成数见图下方。
**关注点**：进入与终止边界是否不同；安静占比高时是否仍保留短事件；不同初值的差异是否需要延长与额外状态解释。

### native_recovery_drift_cuts.png / native_recovery_drift_cuts.svg
同一条件切片下，读取原方程在被固定Z/K场上产生的自然漂移。G/M仍动态；Z/K本身不会按这些漂移移动，所以此图不计自主恢复或闭环。双核与全E分别显示，避免均值恢复掩盖核内继续消耗。
**关注点**：Z净恢复与K净消退是否处于不同区域，以及这些漂移能否解释真实自主轨迹的返回方向。
''')
    print(status,flush=True)


if __name__=='__main__':main()
