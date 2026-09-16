#!/usr/bin/env python3
"""Diagnostic figure for the frozen direction target, plus completed confirmation."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts import run_topic4_xy_research as base


def save(fig, stem):
    for ext in ('png','pdf','svg'): fig.savefig(stem.with_suffix('.'+ext),dpi=200)
    plt.close(fig)


def main():
    out=base.OUT; contract_path=out/'direction_objective_v2.json'; c=base.read(contract_path)
    directory=out/'direction_review/figures'; directory.mkdir(parents=True,exist_ok=True)
    xy=np.array(c['contact_xy_mm']); angle=c['patient_direction_summary']['axial_angle_deg']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig, axes=plt.subplots(1,3,figsize=(13.5,4.6),gridspec_kw={'width_ratios':[1,1.35,1]})
    fig.subplots_adjust(left=.065,right=.98,bottom=.20,top=.80,wspace=.38)
    fig.suptitle('XY search: directly fit the patient propagation directions',fontsize=16,weight='bold',y=.96)
    ax=axes[0]
    for shaft,col in [('ICL','#E58C32'),('SCL','#30A3AF')]:
        mask=np.array([n.startswith(shaft) for n in c['contact_names']])
        ax.scatter(xy[mask,0],xy[mask,1],s=22,color=col,zorder=3)
    ax.text(16.4,2.7,'ICL',color='#E58C32',fontsize=9)
    ax.text(11.2,13.5,'SCL',color='#30A3AF',fontsize=9)
    for theta,col,style,label in [(c['structural_ee_axis_deg'],'#888888','--','EE kernel (fixed)'),
                                 (angle,'#245C3F','-','Patient behavior (training)')]:
        delta=7*np.array([np.cos(np.radians(theta)),np.sin(np.radians(theta))])
        ax.plot([10-delta[0],10+delta[0]],[10-delta[1],10+delta[1]],color=col,ls=style,lw=2,label=label)
    ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',ylabel='y (mm)',title='A   Input axis and observed axis')
    ax.legend(frameon=False,fontsize=8,loc='upper right')
    hist=np.asarray(c['patient_direction_histogram']); degrees=np.arange(24)*15.
    signed=(degrees+180)%360-180; order=np.argsort(signed)
    axes[1].bar(signed[order],hist[:-1][order],width=13,color='#245C3F',alpha=.8)
    axes[1].set(xlim=(-187,187),ylim=(0,.14),xticks=[-180,-90,0,90,180],xlabel='Earlier-to-later direction (degrees)',
                ylabel='Coherence-weighted mass / all events',title='B   Signed direction distribution')
    axes[1].text(.03,.93,f"Unresolved / nonplanar mass: {hist[-1]:.1%}",transform=axes[1].transAxes,va='top',fontsize=9)
    splits=np.array(c['training_block_split_direction_distances'])
    vals=[np.median(splits),c['normalization']['direction_90_degree_rotation_distance'],
          c['normalization']['direction_180_degree_rotation_distance']]
    axes[2].bar([0,1,2],vals,color=['#999999','#D5843C','#864477'],width=.58)
    axes[2].scatter(np.linspace(-.16,.16,len(splits)),splits,s=8,color='#555555',alpha=.5)
    axes[2].set(xticks=[0,1,2],xticklabels=['Block\nsplits','Rotate\n90°','Reverse\n180°'],
                ylabel='Direction distribution distance',title='C   Training-only sensitivity')
    for ax in axes[1:]: ax.spines[['top','right']].set_visible(False)
    fig.text(.065,.045,'Primary: event-distribution fit + signed-direction fit. Core alignment is a separate weak-prior comparison.',fontsize=10)
    save(fig,directory/'xy_direction_training_target')
    readme='''### xy_direction_training_target.png / .pdf / .svg
A 将固定的结构 EE 轴与训练事件测得的传播轴画在同一触点坐标中；线段只表示方向，中心不是拟合出的病灶位置。B 保留正反向差异，方向质量由平面拟合的 adjusted R² 加权，不能可靠表达成平面传播的部分仍计入分母和目标函数。C 用训练记录块拆分、90°旋转和180°反向检查方向距离是否有分辨力，不是候选拟合结果。
**关注点**：结构轴是已有输入，行为轴来自起始时间；主搜索不强制 core 沿轴或放在两端。
'''
    cp=out/'confirmation/aggregate.json'
    if cp.exists() and base.read(cp).get('objective_version')=='xy_direction_v2':
        report=base.read(cp); rows=report['candidates']
        fig, axes=plt.subplots(2,len(rows),figsize=(4*len(rows),7.7),squeeze=False)
        fig.subplots_adjust(left=.045,right=.985,bottom=.12,top=.83,wspace=.35,hspace=.42)
        fig.suptitle('XY nominees on six new networks: geometry and measured direction',fontsize=16,weight='bold',y=.97)
        for j,row in enumerate(rows):
            ax=axes[0,j]; centers=np.array(row['node_field']['centers_mm'])
            radius=np.mean([u['geometry']['distance_cutoff_mm'] for u in row['units']])
            ax.scatter(xy[:,0],xy[:,1],s=10,color='#888888')
            for center in centers: ax.add_patch(Circle(center,radius,color='#245C3F',alpha=.22))
            ax.plot(centers[:,0],centers[:,1],color='#245C3F',ls='--')
            value='NA' if row['J_direction'] is None else f"{row['J_direction']:.2f}"
            ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',ylabel='y (mm)',
                   title=f"{row['candidate_id']}\nJ = {value}; returned n = {row['n_returned']}")
            ax=axes[1,j]; mh=row['direction']['histogram']
            ax.plot(signed[order],hist[:-1][order],color='#245C3F',label='Patient training')
            if mh is not None: ax.plot(signed[order],np.array(mh[:-1])[order],color='#864477',label='Model, pooled')
            ax.set(xlim=(-180,180),xticks=[-180,0,180],xlabel='Signed direction (degrees)',ylabel='Mass / all events')
            ax.spines[['top','right']].set_visible(False)
            if j==0: ax.legend(frameon=False,fontsize=8)
        fig.text(.045,.045,'All nominated candidates are retained. Patient-training target; independent network seeds. No final substrate freeze.',fontsize=10)
        save(fig,directory/'xy_direction_confirmation')
        readme+='''
### xy_direction_confirmation.png / .pdf / .svg
上排展示筛选后指定候选在六个新网络上的几何和主目标距离，下排直接对比模拟与患者训练数据的有符号方向分布。无先验与弱先验提名均保留，不利用确认结果重新选优；不可估计质量、逐网络结果和全片行为轴保存在同一 aggregate JSON 中。
**关注点**：方向匹配必须由实际事件验证，不能从 core 连线或结构长轴直接推断。
'''
    (directory/'README.md').write_text(readme)
    base.write(out/'direction_review/figure_metadata.json',{'producer_sha256':base.sha(Path(__file__)),
        'objective_sha256':base.sha(contract_path),'confirmation_sha256':base.sha(cp) if cp.exists() else None,
        'output_hashes':{str(p):base.sha(p) for p in sorted(directory.iterdir()) if p.suffix in ('.png','.pdf','.svg')}})


if __name__=='__main__': main()
