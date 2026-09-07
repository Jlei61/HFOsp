#!/usr/bin/env python3
"""Show all round-1 nominees with both data fit and relative-position score."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts import run_topic4_xy_research as base


def main():
    out=base.OUT; report_path=out/'confirmation/aggregate.json'; report=base.read(report_path)
    if report.get('round_primary_score')!='J_round1':raise RuntimeError('not a round-1 confirmation report')
    contract=base.read(out/'direction_objective_v2.json'); xy=np.asarray(contract['contact_xy_mm'])
    target=np.asarray(contract['patient_direction_histogram']);angles=(np.arange(24)*15+180)%360-180;order=np.argsort(angles)
    rows=report['candidates'];fig,axes=plt.subplots(2,len(rows),figsize=(4*len(rows),8.2),squeeze=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig.subplots_adjust(left=.045,right=.985,bottom=.14,top=.81,wspace=.38,hspace=.45)
    fig.suptitle('Round 1: relative core position prior, with independent-network confirmation',fontsize=16,weight='bold',y=.97)
    for j,row in enumerate(rows):
        ax=axes[0,j];centers=np.asarray(row['node_field']['centers_mm'])
        radius=np.mean([u['geometry']['distance_cutoff_mm'] for u in row['units']])
        ax.scatter(xy[:,0],xy[:,1],s=12,color='#888888')
        for center in centers:ax.add_patch(Circle(center,radius,color='#245C3F',alpha=.25))
        ax.plot(centers[:,0],centers[:,1],color='#245C3F',ls='--')
        value=lambda v:'NA' if v is None else f'{v:.3f}'
        ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',ylabel='y (mm)',
               title=f"{row['candidate_id']}\nData fit: {value(row['J_direction'])}; with prior: {value(row['J_round1'])}")
        ax=axes[1,j];ax.plot(angles[order],target[:-1][order],color='#245C3F',label='Patient training')
        h=row['direction']['histogram']
        if h is not None:ax.plot(angles[order],np.asarray(h[:-1])[order],color='#864477',label='Model')
        ax.set(xlim=(-180,180),xticks=[-180,0,180],xlabel='Earlier-to-later direction (degrees)',ylabel='Mass / all returned events')
        ax.spines[['top','right']].set_visible(False)
        if j==0:ax.legend(frameon=False,fontsize=8)
    fig.text(.045,.055,'Prior: 0.1 × sin²(core-line angle − patient training axis). Midpoint and separation remain free.\nConstrained and unconstrained nominees are retained; no final substrate freeze.',fontsize=10)
    directory=out/'round1/figures';directory.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','svg'):fig.savefig(directory/f'xy_relative_position_confirmation.{ext}',dpi=200)
    plt.close(fig)
    (directory/'README.md').write_text('''### xy_relative_position_confirmation.png / .pdf / .svg
第一轮把两核连线与患者训练传播轴的软对齐惩罚纳入主排序，图中同时列出纯数据拟合分数和加入位置先验的分数，避免把位置先验改善误当成传播拟合改善。上排是全部预先提名的几何，下排是六个新网络上的实际传播方向与患者训练目标；无先验提名和历史对照均保留。未解析／非平面质量和逐网络不确定性见同一确认 JSON，图不代表最终基底已冻结。
**关注点**：相对位置限制是否改善了实际行为，需要看纯数据拟合、方向分布和新网络配对差值，不能只看加入惩罚后的主分数。
''')
    base.write(out/'round1/figure_metadata.json',{'producer_sha256':base.sha(Path(__file__)),
        'confirmation_sha256':base.sha(report_path),'round_contract_sha256':base.sha(out/'round1_relative_position_contract.json'),
        'output_hashes':{str(p):base.sha(p) for p in directory.iterdir() if p.suffix in ('.png','.pdf','.svg')}})


if __name__=='__main__':main()
