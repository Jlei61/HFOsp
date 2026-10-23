#!/usr/bin/env python3
"""Two matched Figure 2B variants: held-out rank and held-out direction error."""
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

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.paper_figures import plot_fig2b_rank_reconstruction_holdout as prior
from scripts.run_fig2b_two_target_holdout import OUT as ANALYSIS
from scripts.run_fig2b_rank_reconstruction_holdout import write_json
from src.interictal_rank_holdout import METHODS

OUT=ROOT/'results/paper-ready-figure/fig2/candidates/rank_direction_compact_20261023'
WIDTH,HEIGHT=583,350
TARGETS={'rank_mae':('rank-target','Held-out error (MAE)'),
         'angular_mae_deg':('direction-target','Held-out error (°)')}


def build(analysis,out,dpi,*,rose_event_label=None,rose_legend_inside=False,rose_bottoms=(208,20)):
    summary=json.loads((analysis/'summary.json').read_text())
    manifest=json.loads((analysis/'run_manifest.json').read_text())
    nearest_direction=manifest.get('main_direction_assignment')=='nearest_3d_angle'
    rose_event_label=rose_event_label or manifest.get('rose_event_label','Event directions')
    assert summary['status']=='complete' and summary['n_complete']==26
    records=[json.loads((analysis/'per_subject'/f'{sid}.json').read_text()) for sid in manifest['subjects']]
    figs=out/'figures';figs.mkdir(parents=True,exist_ok=True)
    roses=[prior.rose_data(sid) for sid in prior.EXAMPLES]
    rmax=float(np.ceil(max(prior.prior._histogram_proportions(d['all_event_angles'],d['edges']).max() for d in roses)/.05)*.05)
    plots={}
    with plt.rc_context({'font.family':'Liberation Sans','pdf.fonttype':42,'axes.linewidth':.8}):
        for target,(name,xlabel) in TARGETS.items():
            # The established renderer receives generic paired losses via its
            # legacy column names; metadata retains the actual metric and unit.
            adapted=[dict(subject_id=r['subject_id'],
                          **{f'{m}_mae':r['combined'][m][target] for m in METHODS},
                          **{f'{m}_null_mae':r['null'][m][target] for m in METHODS}) for r in records]
            fig=plt.figure(figsize=(WIDTH/72,HEIGHT/72),facecolor='white')
            ax=fig.add_axes([42/WIDTH,53/HEIGHT,300/WIDTH,279/HEIGHT])
            options=dict(xlabel=xlabel,show_density_axis=True)
            if target=='angular_mae_deg':
                options.update(xtick_step=30,min_xmax=100,xpad=5,kde_pad=.3)
            plots[target]=prior.draw_errors(ax,adapted,summary['primary'][target],**options)
            ax.yaxis.label.set_text('Patients')
            plots[target].update(metric=target,unit='degrees' if target=='angular_mae_deg' else 'normalized rank',
                                 null=(manifest.get('direction_null_description','within-block direction-to-assignment permutation')
                                       if target=='angular_mae_deg' else 'within-event rank-channel permutation'),
                                 assignment=('nearest 3D angle' if nearest_direction and target=='angular_mae_deg' else 'nearest normalized rank MAE'))
            rose_axes=[]
            for data,bottom in zip(roses,rose_bottoms):
                polar=fig.add_axes([410/WIDTH,bottom/HEIGHT,120/WIDTH,120/HEIGHT],projection='polar')
                rose_axes.append(polar)
                prior.prior.draw_combined_probability_rose(polar,data,edges=data['edges'],rmax=rmax,row_label=prior.label(data['subject_id']))
                for line in list(polar.lines):
                    if line.is_dashed():
                        line.remove()
                polar.tick_params(labelsize=12,pad=1);polar.set_rlabel_position(132)
                for text in polar.texts: text.set_fontsize(17)
                assert len(polar.lines)==2 and all(not line.is_dashed() and np.ptp(line.get_xdata())==0 for line in polar.lines)
            handles=[Patch(facecolor='#DDDDDD',edgecolor='#777777',label=rose_event_label),
                     Line2D([],[],color='#333333',lw=1.8,label='Spatial TRAJ.')]
            if rose_legend_inside:
                legend=rose_axes[0].legend(handles=handles,loc='center',bbox_to_anchor=(.5,.23),
                                    fontsize=8.5,handlelength=1.3,labelspacing=.35,
                                    frameon=True,facecolor='white',edgecolor='none',framealpha=.95,
                                    borderpad=.25,handletextpad=.55)
                fig.canvas.draw()
                box=legend.get_window_extent(fig.canvas.get_renderer())
                corners=rose_axes[0].transAxes.inverted().transform(box.corners())
                assert np.all(np.linalg.norm(corners-.5,axis=1)<.5),'Legend must fit inside rose circle'
                assert not any(patch.get_path().transformed(patch.get_transform()).intersects_bbox(box,filled=True)
                               for patch in rose_axes[0].patches),'Legend overlaps rose bars'
            else:
                fig.legend(handles=handles,loc='center',bbox_to_anchor=(470/WIDTH,174/HEIGHT),
                           frameon=False,fontsize=10.5,handlelength=1.5,labelspacing=.35)
            fig.savefig(figs/f'fig2b-{name}.pdf');plt.close(fig)
    import pymupdf as pdf
    def raster(stem,preview_width):
        with pdf.open(figs/f'{stem}.pdf') as doc:
            page=doc[0];page.get_pixmap(matrix=pdf.Matrix(dpi/72,dpi/72),alpha=False).save(figs/f'{stem}.png')
            scale=preview_width/page.rect.width
            page.get_pixmap(matrix=pdf.Matrix(scale,scale),alpha=False).save(figs/f'{stem}-preview.png')
    previous=json.loads((prior.CURRENT/'figure2_layout_metadata.json').read_text())
    # Keep A/C/D and B's physical scale; insert the shorter B from its top edge.
    layout=previous['complete_layout']
    placement=layout['b']
    placement['source_clip_pt']=[0,0,WIDTH,HEIGHT]
    placement['placed_rect_pt'][3]=placement['placed_rect_pt'][1]+HEIGHT*placement['scale']
    preserved={}
    for letter in 'acd':
        for path in (prior.CURRENT/'figures').glob(f'fig2-panel{letter}.*'):
            shutil.copy2(path,figs/path.name);preserved[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    for _,(name,_) in TARGETS.items():
        raster(f'fig2b-{name}',1600)
        doc=pdf.open();page=doc.new_page(width=1240,height=860)
        for letter in 'abcd':
            placement=layout[letter]
            stem=f'fig2b-{name}' if letter=='b' else f'fig2-panel{letter}'
            with pdf.open(figs/f'{stem}.pdf') as part:
                page.show_pdf_page(pdf.Rect(placement['placed_rect_pt']),part,0,clip=pdf.Rect(placement['source_clip_pt']))
        for letter,x,y in [('A',12,32),('B',667,32),('C',12,443),('D',824,443)]:
            page.insert_text((x,y),letter,fontsize=27,fontname='hebo')
        doc.save(figs/f'fig2-complete-{name}.pdf');doc.close()
        raster(f'fig2-complete-{name}',1800)
    snap=out/'source_snapshot';snap.mkdir(exist_ok=True)
    for source in (Path(__file__),Path(prior.__file__)):
        shutil.copy2(source,snap/source.name)
    write_json(out/'metadata.json',dict(status='PENDING_AUTHOR_VISUAL_REVIEW',analysis_root=str(analysis),
               source_candidate=str(prior.CURRENT),producer=str(Path(__file__)),plots=plots,
               cohort_summary=summary,preserved_acd_sha256=preserved,complete_layout=layout,
               panel_size_pt=[WIDTH,HEIGHT],previous_panel_size_pt=[583,430],
               axis_labels={target:dict(xlabel=xlabel,ylabel='Patients') for target,(_,xlabel) in TARGETS.items()},
               same_rank_assignment_for_both_targets=not nearest_direction,
               rose_semantics='first held-out fold, 2D projection of actual 3D event directions; illustrative, not the 3D loss distribution',
               rose_rays='two solid Spatial TRAJ. rays per rose; Rank SEQ dashed rays removed at author request',
               rose_legend=dict(event_label=rose_event_label,
                                location='inside upper rose' if rose_legend_inside else 'between roses'),
               rose_layout=dict(bottoms_pt=list(rose_bottoms),diameter_pt=120,
                                circle_gap_pt=rose_bottoms[0]-rose_bottoms[1]-120),
               density_axis=dict(label='Relative density',ticks=[0,1],
                                 observable='patient-bootstrap distribution of cohort median error',
                                 normalization='each KDE divided by its own peak density'),
               png_source='same-state delivered PDFs',dpi=dpi,agent_visual_review='PENDING'))
    (figs/'README.md').write_text(
        '# Figure 2B：两个留出目标，紧凑版\n\n状态：PENDING_AUTHOR_VISUAL_REVIEW；正式入口未覆盖。B画布由583×430 pt缩短为583×350 pt，高度减少18.6%，文字保持原字号，rose保持圆形。横轴简写为Held-out error，方向版保留度数单位，rank版保留MAE；患者纵轴简写为Patients。下方Relative density仍为患者bootstrap中位误差分布，各曲线峰值归一化到1；rose仅显示Spatial TRAJ.实线，具体读出定义见各面板说明。\n\n'
        '### fig2b-rank-target.pdf / fig2b-rank-target.png\n'
        '左侧为两种训练模板对完整留出rank序列的归一化MAE，右侧为E10/E14的留出方向玫瑰图与训练模板直线。两个方法各自保留事件内rank-channel shuffle null；星号对两张图共六个患者配对检验作Holm校正。\n'
        '**关注点**：逐通道rank重建是否改善，不能由方向图或另一张图的显著性替代。\n\n'
        '### fig2b-direction-target.pdf / fig2b-direction-target.png\n'+
        ('左侧为每个留出事件到两个训练模板的最小三维夹角的均值，单位为度；不使用rank选择模板。灰色null在固定患者坐标与支持下置换训练模板rank，重拟合三维轴，再重新选择最近方向。玫瑰图与rank版本相同，明确为第一折二维投影示意，不能直接作为三维误差分布。\n' if nearest_direction else
         '左侧为同一rank分配下，留出事件方向与对应训练模板三维轴的平均夹角，单位为度。灰色null打乱同一记录块内方向与模板身份的配对，保留患者及记录块原有方向分布；不是球面均匀方向。玫瑰图与rank版本完全相同。\n')+
        '**关注点**：增加空间信息能否提高方向表达精度；此方向来自rank与电极坐标拟合，不是独立测得的传播真值。\n\n'
        '### fig2-complete-rank-target.pdf / fig2-complete-rank-target.png\n'
        '完整A–D中的B使用rank误差版本，A/C/D导出及摆放沿用当前候选。PNG直接从交付PDF渲染。\n'
        '**关注点**：完整图的阅读顺序与统计口径。\n\n'
        '### fig2-complete-direction-target.pdf / fig2-complete-direction-target.png\n'
        '完整A–D中的B使用方向误差版本，A/C/D与rank版本相同。此版与rank版回答不同的留出目标，需分别解释。\n'
        '**关注点**：方向改善与rank阴性结果同时保留，待作者目视验收。\n')
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,default=ANALYSIS);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--dpi',type=int,default=600);args=p.parse_args()
    print(build(args.analysis,args.output,args.dpi))
