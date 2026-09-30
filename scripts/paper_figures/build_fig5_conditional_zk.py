#!/usr/bin/env python3
"""Measured conditional points and unclamped native drift; no interpolated basins."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO=Path(__file__).resolve().parents[2]
SOURCE=Path('/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924')
OUT=REPO/'results/paper-ready-figure/fig5/candidates/autonomous_loop_20260924'
STATES={'sustained_high':('#c03d52','Sustained high'),
        'recurrent_brief':('#248aab','Brief events'),
        'quiet':('#8c979e','Mostly quiet'),
        'mixed_or_transient':('#dc9a35','Mixed / transient')}


def main():
    path=SOURCE/'conditional_summary.json';summary=json.loads(path.read_text())
    lookup={r['name']:r for r in summary['rows']}
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.7),sharex=True,sharey=True)
    plt.rcParams.update({'font.size':10})
    arrows=[]
    for ax,history,title in zip(axes,['high','interictal'],['High-state history','Interictal history']):
        for z in [.25,.75,.95]:
            for k in [.02,2.,8.]:
                name=f'z{z:g}_k{k:g}_{history}';row=lookup.get(name);x=np.log10(k)
                if row is None:
                    ax.plot(x,z,marker='x',color='#c3c8cc',ms=7,mew=1)
                    continue
                color,label=STATES[row['finite_window_state']]
                ax.scatter([x],[z],s=125,c=[color],edgecolor='white',lw=1,zorder=3)
                rates=row['tail_mean_Hz'];brief=row['tail_brief_events']
                text=f'{rates[0]:.1f} Hz' if rates[0]<100 else f'{rates[0]:.0f} Hz'
                if brief:text+=f'\n{brief} brief'
                ax.annotate(text,(x,z),xytext=(0,12),textcoords='offset points',ha='center',va='bottom',fontsize=8)
                dz,dk=row['counterfactual_drift_mean_allE_A_B_other'][0]
                vector=np.array([dk/(k*np.log(10))/3.2,dz/.95])
                norm=np.linalg.norm(vector)
                if norm>1e-10:
                    # Normalize in screen-coordinate units: direction is retained,
                    # magnitude is deliberately not encoded by arrow length.
                    delta=.082*vector/norm*np.array([3.2,.95])
                    ax.annotate('',(x+delta[0],z+delta[1]),(x,z),
                        arrowprops=dict(arrowstyle='->',color='#292f33',lw=.9),zorder=4)
                arrows.append(dict(name=name,Z=z,K=k,dZ_per_s=dz,dK_per_s=dk,
                    plotting_coordinates=['log10K','meanZ'],direction_only=True))
        ax.set(xlim=(-1.98,1.20),ylim=(.08,1.065),xticks=np.log10([.02,2,8]),
               xticklabels=['0.02','2','8'],yticks=[.25,.75,.95],xlabel='Prescribed mean K / gL',title=title)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Prescribed mean Z')
    handles=[Line2D([],[],marker='o',linestyle='none',color=c,label=l,markersize=7) for c,l in STATES.values()]
    handles.append(Line2D([],[],marker='x',linestyle='none',color='#c3c8cc',label='Not completed',markersize=7))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.10),ncol=5,frameon=False,fontsize=8)
    fig.text(.5,.055,'Arrows: mean native drift if released; core drift can have opposite signs (G/M dynamic).',ha='center',fontsize=9)
    fig.text(.5,.018,'Sparse range excludes parts of the observed entry/exit path • no bifurcation type assigned',ha='center',fontsize=9)
    prefix='Completed' if summary['completed']==summary['total'] else 'Running snapshot'
    fig.suptitle(f'{prefix}: {summary["completed"]}/{summary["total"]} conditional trajectories',fontsize=11,y=.98)
    fig.subplots_adjust(left=.08,right=.98,top=.86,bottom=.28,wspace=.18)
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    files=[]
    for ext in ['png','pdf','svg']:
        f=dest/f'fig5-zk-conditional.{ext}';fig.savefig(f,dpi=220,facecolor='white')
        files.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    plt.close(fig)
    metadata=dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        completed=summary['completed'],total=summary['total'],arrows=arrows,files=files,
        source_quantity='Prescribed full spatial Z/K fields, endogenous G/M; last10s finite-window responses of two carried histories.',
        regional_drift_limit='All-E mean drift may hide the opposite sign in an active core; see regional rates/drift in conditional_table.md and conditional_summary.json.',
        trajectory_coverage_audit=str(SOURCE/'trajectory_grid_coverage.json'),
        grid_coverage='Does not enclose the observed first loop: exit K exceeds8 and Z falls below0.25; entry K is below0.02. No claim of locating the complete transition boundary.',
        not_claimed=['Attractor convergence','Complete state boundary','Autonomous loop from a clamp','Certified bifurcation','Planar autonomous vector field'],
        human_review='PENDING',agent_render_review='PENDING_LATEST_RENDER')
    (OUT/'conditional_figure_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    readme=dest/'README.md';old=readme.read_text() if readme.exists() else ''
    title='### fig5-zk-conditional.png'
    section='''### fig5-zk-conditional.png

两种完整初始历史在同一空间Z/K模板、同一未来输入下的30秒条件响应；颜色是末10秒的描述性状态，数值是全E平均率，未完成点用灰叉。Mostly quiet指至少95%的10ms采样中全E及两核均低于5Hz，仍可能有少量短事件，图中保留其数量；这不等于完全静默。箭头仅保留解除钳制时原方程给出的平均漂移方向，长度不表示大小，不能视为二维自主矢量场；平均Z向上也可能同时伴随活跃核心Z向下，须查看区域漂移表。真实轨迹的进入、退出均有部分超出当前网格范围，不能据此定位完整转换边界；G和M仍动态，brief标签的空间形态须另验。

**关注点**：两历史是否趋向不同的有限窗状态，以及恢复/消耗方向；这是测得的稀疏条件点，没有插值出的吸引域。图待人工审阅。
'''
    if title in old:
        before,after=old.split(title,1);end=after.find('\n### ')
        old=before+section+(after[end:] if end>=0 else '')
    else:old=old.rstrip()+'\n\n'+section
    readme.write_text(old)
    print(json.dumps(dict(completed=summary['completed'],total=summary['total'],figure=str(dest/'fig5-zk-conditional.png'))),flush=True)


if __name__=='__main__':main()
