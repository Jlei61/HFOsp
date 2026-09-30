#!/usr/bin/env python3
"""Four matched conditional coordinates per graph, retaining both histories."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from paper_figures.build_fig5_conditional_zk import STATES

ROOT=Path('/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/axis_controls/conditional_runs')
POINTS=[(.25,.02),(.95,.02),(.75,2.),(.75,8.)]


def main():
    path=ROOT/'comparison.json';data=json.loads(path.read_text())
    lookup={(r['axis_condition'],r['name']):r for r in data['rows']}
    plt.rcParams.update({'font.size':9,'axes.labelsize':10,'axes.titlesize':11,
                         'xtick.labelsize':9,'ytick.labelsize':9})
    fig,axes=plt.subplots(2,3,figsize=(10.8,7.0),sharex=True,sharey=True)
    displayed=[]
    for j,condition in enumerate(['current','rotated','isotropic']):
        for i,history in enumerate(['high','interictal']):
            ax=axes[i,j]
            for z,k in POINTS:
                name=f'z{z:g}_k{k:g}_{history}';row=lookup.get((condition,name));x=np.log10(k)
                if row is None:
                    ax.plot(x,z,marker='x',color='#c3c8cc',ms=7,mew=1);continue
                color,_=STATES[row['finite_window_state']]
                ax.scatter(x,z,s=100,c=color,edgecolor='white',lw=.8,zorder=3)
                rate=row['tail_mean_Hz'][0];brief=row['tail_brief_events']
                label=f'{rate:.1f} Hz' if rate<100 else f'{rate:.0f} Hz'
                if brief:label+=f'\n{brief} brief'
                ax.annotate(label,(x,z),xytext=(0,11),textcoords='offset points',ha='center',fontsize=8)
                dz,dk=row['counterfactual_drift_mean_allE_A_B_other'][0]
                vector=np.array([dk/(k*np.log(10))/3.2,dz/.95]);norm=np.linalg.norm(vector)
                if norm>1e-10:
                    delta=.085*vector/norm*np.array([3.2,.95])
                    ax.annotate('',(x+delta[0],z+delta[1]),(x,z),
                        arrowprops={'arrowstyle':'->','color':'#292f33','lw':.9},zorder=4)
                displayed.append(dict(condition=condition,name=name,state=row['finite_window_state'],
                    Z=z,K=k,dZ_per_s=dz,dK_per_s=dk,mean_Hz=rate,brief=brief))
            ax.set(xlim=(-1.98,1.22),ylim=(.08,1.09),xticks=np.log10([.02,2,8]),
                xticklabels=['0.02','2','8'],yticks=[.25,.75,.95])
            ax.spines[['top','right']].set_visible(False)
            if i==0:ax.set_title({'current':'Current axis','rotated':'Rotated structure','isotropic':'Isotropic structure'}[condition])
            if i==1:ax.set_xlabel('Prescribed mean K / gL')
            if j==0:ax.set_ylabel(('High-state history' if i==0 else 'Interictal history')+'\nPrescribed mean Z')
    handles=[Line2D([],[],marker='o',linestyle='none',color=c,label=l,markersize=7) for c,l in STATES.values()]
    handles.append(Line2D([],[],marker='x',linestyle='none',color='#c3c8cc',label='Not completed',markersize=7))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.088),ncol=5,frameon=False,fontsize=8)
    fig.text(.5,.06,'Arrows: native drift direction if released. G and M remain dynamic; Z and K are prescribed.',ha='center',fontsize=9)
    fig.text(.5,.029,'Common spatial fields, carried histories and future input • four coordinates, no interpolated boundaries',ha='center',fontsize=9)
    fig.suptitle(f'Finite-window structural responses: {len(displayed)}/24 completed',fontsize=11,y=.98)
    fig.subplots_adjust(left=.10,right=.98,bottom=.23,top=.90,hspace=.20,wspace=.18)
    dest=ROOT/'figures';dest.mkdir(exist_ok=True);files=[]
    for ext in ['png','pdf','svg']:
        p=dest/f'axis_conditional_responses.{ext}';fig.savefig(p,dpi=200,facecolor='white')
        files.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    plt.close(fig)
    metadata=dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        completed=len(displayed),total=24,new_completed=data['new_completed'],new_total=16,
        rows=displayed,files=files,finite_window_s=30,tail_window_s=10,
        interpretation='Finite-window responses at four coordinates. A difference brackets conditional responses; agreement does not establish equal boundaries. Drift arrows retain direction only and do not define a planar autonomous vector field.',
        structural_limits='Rotation also attenuates anisotropy; outgoing degree and low-threshold source output strength change.',
        formal_bifurcation='NOT_ESTABLISHED',human_review='PENDING',agent_latest_render_review='PENDING')
    (ROOT/'figure_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (dest/'README.md').write_text('### axis_conditional_responses.png\n\n当前轴、旋转结构、各向同性结构在四个共同完整Z/K空间场下的30秒条件响应，两行保留高态与间期两种内源历史，未来输入配对。颜色是末10秒状态，数字是平均全E率及短事件数；未完成点为灰叉，箭头只显示解除钳制时原方程的平均漂移方向。当前图复用8条，另两种图各新增8条；四个点不构成完整边界，G和M在钳制期间继续演化。\n\n**关注点**：同坐标、同历史下结构改变是否伴随状态变化；输出度、低阈值来源强度和旋转后轴比未完全匹配，不能叫纯方向效应或已认证分岔。图待人工审阅。\n')
    print(json.dumps(dict(available=len(displayed),total=24,figure=str(dest/'axis_conditional_responses.png'))),flush=True)


if __name__=='__main__':main()
