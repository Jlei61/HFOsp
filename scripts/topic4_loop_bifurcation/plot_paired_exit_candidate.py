#!/usr/bin/env python3
"""Compact measured response diagram before independent native review finishes."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import shutil
import numpy as np
from campaign import ROOT,read,write,sha
from native_mean_exit_interval import MODEL
import analyze_mean_exit_interval as a

OUT=ROOT/'paired_exit_candidate'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'result.json').exists()
    assert read(MODEL/'runs/high_K9p5/result.json')['both_RNGs_paired_with_four_probes']
    sizes,masks,counts,proj=a.projection();old=a.OUT;a.OUT=MODEL
    try:upper=a.load_case('high_K9p5',sizes,masks,counts,proj)
    finally:a.OUT=old
    points=[]
    for row in read(ROOT/'mean_exit_interval/analysis/result.json')['rows']:
        with np.load(ROOT/'mean_exit_interval/analysis'/f"{row['name']}_readouts.npz") as d:
            points.append(dict(name=row['name'],K=row['held_mean_K'],history=row['history'],
                rate=d['rate'][5000:].mean(0).tolist(),Zdot=d['drift'][5000:].mean(0).tolist()))
    points.append(dict(name='high_K9p5',K=9.5,history='high',rate=upper['rate'][5000:].mean(0).tolist(),Zdot=upper['drift'][5000:].mean(0).tolist()))
    assert min(points[2]['rate'][1:3])>100 and max(points[-1]['rate'][:3])<5
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,ax=plt.subplots(1,2,figsize=(10.3,4.8),layout='constrained')
    colors=['#8a63b4','#d63378','#008eb3'];labels=['All E','Core A','Core B']
    for p in points:
        marker='o' if p['history']=='high' else 's'
        for j,(c,s) in enumerate(zip(colors,[90,54,24])):
            for axis,key in zip(ax,['rate','Zdot']):axis.scatter(p['K'],p[key][j],edgecolors=c,facecolors='none',marker=marker,s=s,lw=1.1,zorder=4)
    for axis in ax:
        axis.axvspan(9.4625,9.5,color='.94',zorder=0)
        axis.set(xlabel=r'Held mean $K$ ($g_K/g_L$)',xlim=(9.337,9.513),xticks=[9.35,9.40,9.45,9.50])
    ax[0].set(ylabel='E rate, 5–10 s (Hz)',ylim=(-15,505));ax[1].set(ylabel='dZ/dt if released (1/s)',ylim=(-.05,.18));ax[1].axhline(0,color='.6',lw=.7)
    ax[0].legend([Line2D([],[],color=c,lw=2) for c in colors],labels,frameon=False,loc='center left')
    ax[1].legend([Line2D([],[],color='.3',marker=x,markerfacecolor='none',ls='none') for x in ['o','s']],['High history','Quiet history'],frameon=False,loc='center left')
    fig.suptitle(r'Conditional exit candidate: same spatial $Z$ field, $\bar Z=0.21$',fontsize=12)
    fig.text(.5,-.025,'Paired future input; 10 s copied-network responses. Shading: transition interval, not a certified bifurcation.',ha='center',fontsize=9)
    for ext in ['png','svg','pdf']:fig.savefig(ROOT/f'figures/paired_exit_candidate.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    write(OUT/'result.json',dict(status='COMPLETE_MODEL_CANDIDATE_NATIVE_REVIEW_PENDING',points=points,
        observation_s=10,tail_s=[5,10],paired_future_numerical_streams=True,measured_transition_interval=[9.4625,9.5],
        limitations='Actual heldK is a conductance state, not the k100 increment parameter. FixedZ does not recover; drift is counterfactual. The high-state source and fixedinputlaws are identical across the fourhighKresponses, and thequiet source has pairedfuturenumericalstreams. No infinite-time equilibrium, stable/unstable branch, criticaltype or newautonomousloop is certified.',
        agent_visual='PENDING',human_visual='PENDING',formal_bifurcation_allowed=False,producer_sha256=sha(__file__)))
    shutil.copy2(__file__,OUT/'producer.py')
    with (ROOT/'figures/README.md').open('a') as f:f.write('\n\n### paired_exit_candidate.png / paired_exit_candidate.svg / paired_exit_candidate.pdf\n五个候选模型十秒条件响应共用未来数值随机流；四个高历史点共享完整起态，静默史另用同Z场的已完成静默状态。显示后五秒All E与双核率、若释放Z的资源收支，采用原Fig5群体语义颜色；零率重合点以同心标记保留。\n**关注点**：9.4625–9.5阴影是模型在此协议下的转换区间，原生核对仍在进行；不是已认证Fold/Hopf，固定Z不发生实际恢复，人工待审。\n')


if __name__=='__main__':main()
