#!/usr/bin/env python3
"""Plot proposed XY coverage and completed search results, without inventing scores."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle,Rectangle

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/topic4_sef_hfo/vth_dual_core_xy_research'


def main():
    design_path=OUT/'search_design.json';design=json.loads(design_path.read_text())
    rows=design['candidates'];figures=OUT/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(13,5.8))
    fig.subplots_adjust(left=.055,right=.97,top=.80,bottom=.24,wspace=.3)
    fig.suptitle('XY search reopened: both core centers vary across the sheet',fontsize=17,weight='bold',y=.96)
    fig.text(.055,.88,'VTH-only perturbation. Learned connectivity and Z/M off. Fixed primary core budget: 1499 E cells.',fontsize=10)
    for ax,domain,title in zip(axes[:2],('whole_sheet','interior'),('64 whole-sheet proposals','64 interior proposals')):
        selected=[r for r in rows if r['domain']==domain]
        segments=[r['node_field']['centers_mm'] for r in selected]
        ax.add_collection(LineCollection(segments,colors='#245C3F',alpha=.13,lw=.7))
        centers=np.asarray(segments).reshape(-1,2)
        ax.scatter(centers[:,0],centers[:,1],s=10,color='#245C3F',alpha=.55)
        ax.set_title(title)
    contacts_path=Path('/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/data_driven_core_field_rev10_sa/shaft_aware_target/contact_shaft_contract.json')
    contacts=np.array([r['sheet_xy_mm'] for r in json.loads(contacts_path.read_text())['contacts']])
    axes[2].scatter(contacts[:,0],contacts[:,1],s=14,facecolors='none',edgecolors='#777777')
    for cid,color,label in [('control_old_edge','#A63A79','Old edge reference'),('control_historical_matched','#245C3F','Historical, budget matched')]:
        row=next(r for r in rows if r['candidate_id']==cid);centers=np.array(row['node_field']['centers_mm'])
        axes[2].plot(centers[:,0],centers[:,1],color=color,ls='--',label=label)
        for c in centers:axes[2].add_patch(Circle(c,row['geometry']['distance_cutoff_mm'],facecolor=color,edgecolor=color,alpha=.22))
    axes[2].set_title('Paired geometry controls');axes[2].legend(frameon=False,fontsize=8,loc='upper center')
    for ax in axes:
        ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xticks=[0,5,10,15,20],yticks=[0,5,10,15,20],xlabel='x (mm)')
    axes[0].set_ylabel('y (mm)')
    fig.text(.055,.07,'Lines link the two centers within each proposal. No axis-alignment constraint.\nThese are proposed locations; neither a training score nor a selected optimum is implied.',fontsize=10)
    stem=figures/'xy_search_coverage'
    for ext in ('png','pdf','svg'):fig.savefig(stem.with_suffix('.'+ext),dpi=220)
    plt.close(fig)
    readme='### xy_search_coverage.png / .pdf / .svg\n左侧为64个全平面XY候选，中间为64个满足完整核边界余量的内部候选；线段连接同一候选的两个中心，没有强迫沿长轴排列。右侧显示旧边缘几何与预算匹配的历史手放几何及固定触点，所有主比较均为1499个核内E细胞。\n**关注点**：本图展示搜索覆盖，不包含优化结果；阈值生成规则固定，学习连接与Z/M关闭。\n'
    cp=OUT/'confirmation/aggregate.json'
    if cp.exists():
        report=json.loads(cp.read_text());rr=report['candidates']
        fig,axes=plt.subplots(1,len(rr),figsize=(4*len(rr),4.8),squeeze=False)
        fig.subplots_adjust(top=.75,bottom=.18,wspace=.35)
        fig.suptitle('Independent-network confirmation of XY nominees',fontsize=16,weight='bold')
        for ax,row in zip(axes[0],rr):
            centers=np.array(row['node_field']['centers_mm'])
            ax.scatter(contacts[:,0],contacts[:,1],s=14,facecolors='none',edgecolors='#888888')
            radius=float(np.mean([u['geometry']['distance_cutoff_mm'] for u in row['units']]))
            for c in centers:ax.add_patch(Circle(c,radius,color='#245C3F',alpha=.25))
            ax.plot(centers[:,0],centers[:,1],color='#245C3F',ls='--')
            score='NA' if row['D_cloud'] is None else f"{row['D_cloud']:.3f}"
            ax.set_title(f"{row['candidate_id']}\nFull-distribution distance: {score}",fontsize=9)
            ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',ylabel='y (mm)')
        fig.text(.08,.065,'Six confirmation networks; original patient-training target. No final substrate freeze or patient-held-out claim.',fontsize=10)
        stem=figures/'xy_confirmation_geometries'
        for ext in ('png','pdf','svg'):fig.savefig(stem.with_suffix('.'+ext),dpi=220)
        plt.close(fig)
        readme+='\n### xy_confirmation_geometries.png / .pdf / .svg\n对筛选后预先指定的候选及历史对照，在6个新网络上重新计算完整事件分布距离。圆盘半径为确认网络均值，具体逐网络成员和阈值差异保留在JSON中。\n**关注点**：这是训练患者目标上的独立网络确认，不是患者留出结论，不会自动冻结Fig.5基底。\n'
    (figures/'README.md').write_text(readme)
    outputs=sorted(p for p in figures.iterdir() if p.suffix in ('.png','.pdf','.svg'))
    (OUT/'figure_metadata.json').write_text(json.dumps({'source_design_sha256':hashlib.sha256(design_path.read_bytes()).hexdigest(),
        'confirmation_present':cp.exists(),'producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'output_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}},indent=2)+'\n')


if __name__=='__main__':main()
