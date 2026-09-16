#!/usr/bin/env python3
"""Render the E1146 retrospective template-share audit without refitting labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_e1146_preseizure_template_share import OUT, coverage

COLORS={'A':'#B2182B','B':'#2166AC'}


def save(fig,root,stem):
    fig.savefig(root/f'{stem}.png',dpi=200,bbox_inches='tight',facecolor='white')
    fig.savefig(root/f'{stem}.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def render(root):
    labels=pd.read_csv(root/'seizure_labels.csv')
    counts=pd.read_csv(root/'interval_template_shares.csv')
    counts=counts[counts.exclude_post_minutes.eq(0)]
    block=pd.read_csv(root/'block_coverage_audit.csv')
    z=np.load(root/'event_index.npz')
    times=z['event_abs_time'];lab=z['template_label'];keep=z['strict_eligible']
    baseline=float(np.mean(lab[keep]==0))
    figures=root/'figures';figures.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    mode=json.loads((root/'analysis_contract.json').read_text()).get('boundary_mode','strict_parent_block')
    fig,axes=plt.subplots(1,3,figsize=(15,12),gridspec_kw={'width_ratios':[1,1.3,1.3]},sharey=True)
    y=np.arange(26)
    for ax in axes:
        for j in range(26):
            if j%2==0:ax.axhspan(j-.48,j+.48,color='#f3f3f3',zorder=0)
        ax.set_ylim(25.7,-.8)
        ax.set_yticks(y)
    ax=axes[0]
    for i,r in labels.iterrows():
        if r.status!='ok':
            ax.text(-.95,i,'Unclassifiable: baseline unavailable',fontsize=7,va='center');continue
        a,b=r.fig3c_r_a,r.fig3c_r_b
        ax.plot([a,b],[i,i],color='#bbbbbb',lw=.8)
        ax.scatter([a,b],[i,i],c=[COLORS['A'],COLORS['B']],s=22,zorder=3)
        ax.text(1.08,i,f'{r.fig3c_abs_label} / {r.fig3c_signed_label}',va='center',fontsize=8)
    ax.axvline(0,color='#aaaaaa',lw=.7)
    ax.set_xlim(-1.05,1.6);ax.set_xticks([-1,-.5,0,.5,1]);ax.set_yticklabels([f'SZ {i+1}' for i in y])
    ax.set_xlabel('Signed field similarity r')
    ax.set_title('Early ictal energy (clinical 0–10 s)\nRight: Fig3 |r| label / signed-r label',fontsize=11)
    for ax,window,title in zip(axes[1:],('whole','pre60min'),('Previous seizure offset → next onset','Last 60 min before onset')):
        sub=counts[counts.window.eq(window)].set_index('seizure_idx')
        for i,r in sub.iterrows():
            if r.n_events:
                alpha=.9 if r.inference_eligible else .35
                ax.barh(i,r.ta_share,color=COLORS['A'],height=.58,alpha=alpha)
                ax.barh(i,1-r.ta_share,left=r.ta_share,color=COLORS['B'],height=.58,alpha=alpha)
                ax.text(1.05,i,f'{r.ta_share:.0%}   {r.n_events:,.0f}   {r.strict_coverage_fraction:.0%}',fontsize=8,va='center')
            else:
                ax.text(.04,i,'No admissible observations',fontsize=8,color='#777777',va='center')
        ax.axvline(.5,color='black',lw=.7,ls='--')
        ax.axvline(baseline,color='#666666',lw=.8,ls=':')
        ax.set_xlim(0,1.86);ax.set_xticks([0,.25,.5,.75,1]);ax.set_xticklabels(['0','25','50','75','100'])
        ax.set_xlabel('TA share (%) | remaining share = TB')
        ax.set_title(title+'\nRight: TA share · n events · coverage',fontsize=11)
    fig.suptitle(f'E1146 | seizure energy labels and preceding event composition\n{mode}',fontsize=14,y=.992)
    fig.legend(handles=[Line2D([],[],color=COLORS['A'],lw=5,label='TA'),Line2D([],[],color=COLORS['B'],lw=5,label='TB')],
               loc='lower center',bbox_to_anchor=(.5,.014),ncol=2,frameon=False)
    fig.text(.05,.004,'Faded bars: descriptive only (coverage <50%, n<20, truncated 60-min window, missing label, or no previous seizure).  Dotted line: observed patient TA baseline.',fontsize=8)
    fig.subplots_adjust(left=.055,right=.995,bottom=.075,top=.91,wspace=.09)
    save(fig,figures,'seizure_labels_and_template_shares')

    whole=counts[counts.window.eq('whole')].set_index('seizure_idx')
    ranges=list(zip(block.loc[block.strict_eligible,'start'],block.loc[block.strict_eligible,'end']))
    # Event-level mode writes an exact union of admissible recording spans.
    range_file=root/'admissible_ranges.json'
    if range_file.exists():ranges=json.loads(range_file.read_text())
    fig,axes=plt.subplots(5,5,figsize=(16,11),sharex=True,sharey=True)
    trajectories=[]
    for idx,ax in enumerate(axes.flat,start=1):
        row=whole.loc[idx];r=labels.iloc[idx]
        lo,hi=row.start_epoch,row.end_epoch
        edges=np.linspace(lo,hi,11)
        fractions=[]
        for j,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
            mask=keep&(times>=a)&(times<b)
            if 'event_abs_end_time' in z:mask&=z['event_abs_end_time']<=b
            n=int(mask.sum());p=np.mean(lab[mask]==0) if n else np.nan
            cov=coverage(ranges,a,b)/(b-a) if b>a else 0.
            fractions.append(p)
            trajectories.append(dict(sz=idx+1,bin=j+1,start_epoch=a,end_epoch=b,
                                     n_events=n,ta_share=p,coverage_fraction=cov))
            if n:ax.scatter(5+j*10,p,s=8+min(n,100)/5,color=COLORS.get(r.fig3c_abs_label,'#999999'),alpha=1 if n>=20 and cov>=.5 else .25,zorder=3)
        ax.plot(np.arange(5,100,10),fractions,color='#777777',lw=.65,alpha=.6)
        ax.axhline(.5,color='#999999',ls='--',lw=.6)
        if hi>lo:ax.axvspan(max(0,100*(1-3600/(hi-lo))),100,color='#f5ebd5',zorder=0)
        name=r.fig3c_abs_label if r.status=='ok' else '?'
        signed=r.fig3c_signed_label if r.status=='ok' else '?'
        ax.set_title(f'SZ {idx+1} | {name} (signed {signed}) | {row.interval_hours:.2f} h',fontsize=9)
        if not np.isfinite(fractions).any():ax.text(.5,.5,'No admissible events',transform=ax.transAxes,ha='center',fontsize=8,color='#777777')
        ax.set_xlim(0,100);ax.set_ylim(0,1);ax.set_xticks([0,50,100]);ax.set_yticks([0,.5,1])
    fig.suptitle('TA event share throughout each inter-seizure interval\n10 equal-duration bins; tan = final 60 min; blank bins remain unobserved',fontsize=14)
    fig.supxlabel('Elapsed fraction of interval (%)');fig.supylabel('TA share (TB = 1 − TA)')
    fig.tight_layout(rect=[.02,.02,1,.94])
    save(fig,figures,'interseizure_template_share_trajectories')
    pd.DataFrame(trajectories).to_csv(root/'trajectory_bins.csv',index=False)
    (figures/'README.md').write_text(
        '# E1146 发作前 TA/TB 占比：探索性候选图\n\n'
        '### seizure_labels_and_template_shares.png / .pdf\n\n'
        '逐次展示临床起始后 0–10 秒的 TA/TB 空间相关，以及上一发作结束后整段和发作前 60 分钟的事件比例。'
        '左侧保留正负号并同时列出 Fig3 绝对相关标签和方向敏感标签；右侧列出样本数与真实可用覆盖率，浅色条只作描述。'
        '\n**关注点**：无数据不等于零比例，首发没有已知前次发作；候选图尚待用户目视检查。\n\n'
        '### interseizure_template_share_trajectories.png / .pdf\n\n'
        '每格是一次发作前的完整间隔，横轴按该间隔长度归一化，均分为 10 个时间格。'
        '纵轴为 TA 占比，TB 为其补数，浅棕色表示末 60 分钟；缺数据时间格保留为空。'
        '\n**关注点**：不同格子的真实时间长度不同，不能把归一化时间当作统一小时尺度；低计数或低覆盖点用浅色显示。\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input-dir',type=Path,default=OUT)
    render(parser.parse_args().input_dir)
