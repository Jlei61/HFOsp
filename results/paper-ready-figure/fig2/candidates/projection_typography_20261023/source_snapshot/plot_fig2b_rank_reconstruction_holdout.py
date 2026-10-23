#!/usr/bin/env python3
"""Figure 2B candidate: rank reconstruction errors on the left, axis roses right."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import gaussian_kde

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.paper_figures import plot_interictal_spatial_information_gain as prior
from scripts.paper_figures.patient_public_labels import public_patient_label
from scripts.run_fig2b_rank_reconstruction_holdout import OUT as ANALYSIS, FROZEN, SEED, write_json
from src.interictal_rank_holdout import METHODS
from src.topic5_interictal_direction_rose import axis_pair_display_basis, project_directions_to_angles

CURRENT=ROOT/'results/paper-ready-figure/fig2/candidates/wider_b_score_axes_20261012'
OUT=ROOT/'results/paper-ready-figure/fig2/candidates/rank_reconstruction_holdout_20261022'
EXAMPLES=('epilepsiae_1146','epilepsiae_548')
COLORS=(prior.TEMPORAL_COLOR,prior.HYBRID_COLOR)


def label(sid):
    return public_patient_label(*sid.split('_',1))


def rose_data(sid):
    record=json.loads((FROZEN/'per_subject'/f'{sid}.json').read_text())
    fold=record['folds'][0]
    basis=axis_pair_display_basis(*np.asarray(fold['models'][METHODS[1]]['axes']))
    with np.load(FROZEN/'per_subject'/f'{sid}_heldout.npz') as cache:
        directions=cache['f0_test_directions'][cache['f0_finite_mask']]
    projection=project_directions_to_angles(directions,basis['axis_a'],basis['transverse'])
    angles=np.asarray(projection['angles'])
    edges=np.linspace(0,2*np.pi,19)
    assert np.histogram(angles[np.isfinite(angles)],edges)[0].sum()==np.isfinite(angles).sum()
    methods={}
    for m,key in zip(METHODS,(prior.METHOD_TEMPORAL,prior.METHOD_HYBRID)):
        p=project_directions_to_angles(np.asarray(fold['models'][m]['axes']),basis['axis_a'],basis['transverse'])
        methods[key]={'axis_angles_rad':p['angles']}
    return dict(subject_id=sid,all_event_angles=angles[np.isfinite(angles)],methods=methods,
                edges=edges,n_projection_undefined=int((~np.isfinite(angles)).sum()),
                train_events=fold['n_train'],test_events=fold['n_direction_test'])


def draw_errors(ax, records, summary, *, xlabel='Held-out rank error (MAE)',
                xtick_step=.1, min_xmax=.36, xpad=.018, kde_pad=.002,
                show_density_axis=False):
    """Shared paired-error renderer; callers may adapt a different loss and units."""
    ordered=sorted(records,key=lambda r:r['rank_trajectory_mae'])
    a=np.asarray([r['rank_sequence_mae'] for r in ordered])
    b=np.asarray([r['rank_trajectory_mae'] for r in ordered])
    y=np.arange(len(ordered))+14
    for x1,x2,height in zip(a,b,y):
        ax.plot([x1,x2],[height,height],color='#C9C9C9' if x2<x1 else '#C4867A',lw=.85,zorder=1)
    ax.scatter(a,y,s=23,facecolor='white',edgecolor=COLORS[0],lw=.85,zorder=3)
    ax.scatter(b,y,s=27,facecolor=COLORS[1],edgecolor='white',lw=.5,zorder=4)
    for i,r in enumerate(ordered):
        if r['subject_id'] in EXAMPLES:
            ax.scatter([b[i]],[y[i]],s=55,facecolor='none',edgecolor='#333333',lw=.9,zorder=5)
            ax.annotate(label(r['subject_id']),(max(a[i],b[i]),y[i]),xytext=(6,0),textcoords='offset points',
                        fontsize=11,fontweight='bold',va='center')
    # Paired patient bootstrap: identical patient resamples for all four curves.
    rng=np.random.default_rng(SEED)
    sample=rng.integers(0,len(records),(10000,len(records)))
    values=[np.asarray([r[f'{m}_mae'] for r in records]) for m in METHODS]
    values += [np.asarray([r[f'{m}_null_mae'] for r in records]) for m in METHODS]
    draws=[np.median(v[sample],axis=1) for v in values]
    curve_colors=[*COLORS,'#999999','#666666']
    curve_styles=['-','-','--','-']
    medians=[float(np.median(v)) for v in values]
    for index in (2,3,0,1):
        d,color,style=draws[index],curve_colors[index],curve_styles[index]
        grid=np.linspace(d.min()-kde_pad,d.max()+kde_pad,350)
        density=gaussian_kde(d)(grid);density*=3.1/density.max()
        ax.fill_between(grid,.8,.8+density,color=color,alpha=.14,lw=0,zorder=2)
        ax.plot(grid,.8+density,color=color,ls=style,lw=.85,zorder=3)
        lo,hi=np.quantile(d,[.025,.975])
        baseline=.8+(index-1.5)*.13
        ax.errorbar(medians[index],baseline,xerr=[[medians[index]-lo],[hi-medians[index]]],
                    fmt='D',ms=4,mfc='white',mec=color,mew=.85,ecolor=color,elinewidth=1,capsize=2,zorder=6)
    brackets=[(0,1,5.0,'sequence_vs_trajectory','#333333'),
              (0,2,7.8,'rank_sequence_null_vs_real',COLORS[0]),
              (1,3,10.6,'rank_trajectory_null_vs_real',COLORS[1])]
    for left,right,height,key,color in brackets:
        lo,hi=sorted([medians[left],medians[right]])
        ax.plot([lo,lo,hi,hi],[height-.35,height,height,height-.35],color=color,lw=.85,zorder=6)
        ax.text((lo+hi)/2,height+.12,summary['comparisons'][key]['stars'],ha='center',va='bottom',
                fontsize=11,fontweight='bold',color=color,zorder=7)
    # Error has a natural zero; no axis truncation to magnify small differences.
    xmax=max(min_xmax,float(max(a.max(),b.max(),*(d.max() for d in draws)))+xpad)
    ax.set_xlim(0,xmax)
    ax.set_xticks(np.arange(0,xmax+.001,xtick_step))
    ax.set_ylim(-.2,len(records)+15)
    ax.set_yticks([])
    ax.set_xlabel(xlabel,fontsize=16,labelpad=8)
    ax.set_ylabel('Patients (ordered)',fontsize=15,labelpad=9)
    ax.yaxis.set_label_coords(-.05,.68)
    ax.tick_params(axis='x',labelsize=13,pad=3,length=3)
    for spine in ('top','left','right'): ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#777777')
    if show_density_axis:
        # These curves show bootstrap cohort medians, each scaled to its own
        # peak. Their ordinate is distinct from the ordered patient rows.
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_bounds(.8,3.9)
        ax.spines['left'].set_color('#777777')
        ax.set_yticks([.8,3.9],labels=['0','1'])
        ax.tick_params(axis='y',labelsize=11.5,pad=3,length=3)
        ax.yaxis.label.set_verticalalignment('center')
        ax.yaxis.set_label_coords(-.10,.68)
        ax.text(-.10,2.35,'Relative density',transform=ax.get_yaxis_transform(),
                rotation=90,ha='center',va='center',fontsize=13,clip_on=False)
    handles=[Line2D([],[],ls='',marker='o',mfc='white',mec=COLORS[0],ms=5,label='Rank SEQ'),
             Line2D([],[],ls='',marker='o',mfc=COLORS[1],mec='white',ms=5,label='Spatial TRAJ.'),
             Line2D([],[],color='#999999',ls='--',lw=1.2,label='SEQ null'),
             Line2D([],[],color='#666666',lw=1.2,label='TRAJ. null')]
    legend=ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(.005,.995),frameon=False,
                     fontsize=10.5,handlelength=1.4,handletextpad=.6,labelspacing=.5,borderpad=0)
    ax.figure.canvas.draw()
    box=legend.get_window_extent(ax.figure.canvas.get_renderer())
    points=ax.transData.transform(np.c_[np.r_[a,b],np.r_[y,y]])
    assert not any(box.contains(*point) for point in points), 'legend covers patient markers'
    return dict(patient_order=[r['subject_id'] for r in ordered],xlim=list(ax.get_xlim()),
                n_pairs=len(records),bootstrap_draws=10000,bootstrap_seed=SEED,
                displayed_comparisons=summary['comparisons'],legend_patient_overlap=False,
                curve_order=['Rank SEQ','Spatial TRAJ.','SEQ null','TRAJ. null'],
                curve_medians=medians,curve_ci95=[np.quantile(d,[.025,.975]) for d in draws])


def build(analysis,out,dpi):
    summary=json.loads((analysis/'summary.json').read_text())
    subjects=json.loads((analysis/'run_manifest.json').read_text())['subjects']
    records=[json.loads((analysis/'per_subject'/f'{sid}.json').read_text()) for sid in subjects]
    assert summary['status']=='complete' and len(records)==summary['n_requested']
    figs=out/'figures';figs.mkdir(parents=True,exist_ok=True)
    roses=[rose_data(sid) for sid in EXAMPLES]
    rmax=float(np.ceil(max(prior._histogram_proportions(d['all_event_angles'],d['edges']).max() for d in roses)/.05)*.05)
    with plt.rc_context({'font.family':'Liberation Sans','pdf.fonttype':42,'axes.linewidth':.8}):
        fig=plt.figure(figsize=(583/72,430/72),facecolor='white')
        # Reserve the lower left margin for the unchanged D panel letter in the
        # complete composition; the statistics moved into that horizontal span.
        ax=fig.add_axes([42/583,85/430,300/583,323/430])
        plot_meta=draw_errors(ax,records,summary)
        rose_meta=[]
        for data,bottom in zip(roses,(257,30)):
            polar=fig.add_axes([399/583,bottom/430,150/583,150/430],projection='polar')
            prior.draw_combined_probability_rose(polar,data,edges=data['edges'],rmax=rmax,row_label=label(data['subject_id']))
            polar.tick_params(labelsize=12,pad=1)
            polar.set_rlabel_position(132)
            for text in polar.texts: text.set_fontsize(17)
            rays=[dict(theta=float(line.get_xdata()[0]),color=line.get_color(),style=line.get_linestyle()) for line in polar.lines]
            assert len(rays)==4
            assert all(np.ptp(line.get_xdata())==0 for line in polar.lines)
            rose_meta.append(dict(subject=label(data['subject_id']),train_events=data['train_events'],
                                  heldout_events=data['test_events'],projection_undefined=data['n_projection_undefined'],
                                  n_template_rays=len(rays),rays=rays))
        fig.legend(handles=[Patch(facecolor='#DDDDDD',edgecolor='#777777',label='Event directions'),
                            Line2D([],[],color='#333333',ls='--',lw=1.2,label='Rank SEQ'),
                            Line2D([],[],color='#333333',lw=1.8,label='Spatial TRAJ.')],
                   loc='center',bbox_to_anchor=(474/583,220/430),frameon=False,
                   fontsize=10.5,handlelength=1.5,labelspacing=.35)
        fig.savefig(figs/'fig2-panelb.pdf')
        plt.close(fig)
    import pymupdf as pdf
    # PNGs are rasterized from the actual delivered PDFs, keeping both states identical.
    def export_pdf(stem,preview_width):
        with pdf.open(figs/f'{stem}.pdf') as doc:
            page=doc[0]
            page.get_pixmap(matrix=pdf.Matrix(dpi/72,dpi/72),alpha=False).save(figs/f'{stem}.png')
            scale=preview_width/page.rect.width
            page.get_pixmap(matrix=pdf.Matrix(scale,scale),alpha=False).save(figs/f'{stem}-preview.png')
    export_pdf('fig2-panelb',1600)
    previous=json.loads((CURRENT/'figure2_layout_metadata.json').read_text())
    preserved={}
    for letter in 'acd':
        for path in (CURRENT/'figures').glob(f'fig2-panel{letter}.*'):
            shutil.copy2(path,figs/path.name)
            preserved[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    doc=pdf.open();page=doc.new_page(width=1240,height=860)
    for letter in 'abcd':
        placement=previous['complete_layout'][letter]
        with pdf.open(figs/f'fig2-panel{letter}.pdf') as part:
            page.show_pdf_page(pdf.Rect(placement['placed_rect_pt']),part,0,clip=pdf.Rect(placement['source_clip_pt']))
    for letter,x,y in [('A',12,32),('B',667,32),('C',12,443),('D',824,443)]:
        page.insert_text((x,y),letter,fontsize=27,fontname='hebo')
    doc.save(figs/'fig2-complete-layout.pdf');doc.close()
    export_pdf('fig2-complete-layout',1800)
    (out/'source_snapshot').mkdir(exist_ok=True)
    shutil.copy2(Path(__file__),out/'source_snapshot'/Path(__file__).name)
    meta=dict(analysis_root=str(analysis),source_candidate=str(CURRENT),producer=str(Path(__file__)),
              status='PENDING_AUTHOR_VISUAL_REVIEW',cohort_summary=summary,plot=plot_meta,
              complete_layout=previous['complete_layout'],preserved_acd_sha256=preserved,roses=rose_meta,
              left='rank reconstruction error and matched rank-shuffle nulls',right='two held-out direction roses with straight training-template rays',
              rose_semantics='illustrative direction space, not the rank-error evaluation target',
              significance='two-sided patient Wilcoxon, Holm across three comparisons; stars only',
              png_source='delivered same-state PDF',dpi=dpi,agent_visual_review='PENDING')
    write_json(out/'metadata.json',meta)
    (figs/'README.md').write_text(
        '# Figure 2B 留出 rank 重建候选\n\n状态：PENDING_AUTHOR_VISUAL_REVIEW；正式入口未替换。\n\n'
        '### fig2-panelb.pdf / fig2-panelb.png\n'
        '左侧每行一位患者：空心蓝灰点为 rank sequence，实心橙红点为 rank trajectory，横轴是两折等权的归一化 rank MAE，越低越好。底部分布为患者 bootstrap 的队列中位数与95%区间；灰色虚/实线分别为两种方法的 rank-channel shuffle null。黑色括号比较两种模型，蓝灰／橙红括号分别比较该模型与自己的 null，星号使用三个双侧配对检验的 Holm 校正 P 值。右侧玫瑰图显示 E10/E14 留出方向，TA/TB 用红/蓝直线表示，sequence 虚线、trajectory 实线。\n'
        '**关注点**：rank 误差与方向示意图是不同读出；直接模型比较能否支持增加空间信息后重建改善。\n\n'
        '### fig2-complete-layout.pdf / fig2-complete-layout.png\n'
        '在当前四面板候选中替换 B；A/C/D 的源文件及摆放保持相同。PNG 从交付 PDF 直接渲染，完整图与单图同一数值和图形状态。\n'
        '**关注点**：统计左置后的阅读顺序、直线玫瑰图、星号与灰色 null 的可辨识性，待作者目视验收。\n')
    return meta


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,default=ANALYSIS)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--dpi',type=int,default=600)
    args=parser.parse_args();build(args.analysis,args.output,args.dpi)
    print(args.output)
