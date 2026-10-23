#!/usr/bin/env python3
"""Render nearest-3D holdout panels and exact exemplar error-distribution checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures import plot_fig2b_two_target_holdout as shared
from scripts.paper_figures import plot_fig2b_rank_reconstruction_holdout as prior
from scripts.run_fig2b_nearest_direction_holdout import OUT as ANALYSIS
from scripts.run_fig2b_rank_reconstruction_holdout import write_json
from src.interictal_rank_holdout import METHODS

OUT=ROOT/'results/paper-ready-figure/fig2/candidates/nearest_direction_compact_roses_20261023'


def build(analysis, out, dpi):
    shared.build(analysis, out, dpi,rose_event_label='Event directions',rose_legend_inside=True,
                 rose_bottoms=(208,48))
    figs=out/'figures';checks=[]
    with plt.rc_context({'font.family':'Liberation Sans','pdf.fonttype':42,'axes.linewidth':.8}):
        fig, axes=plt.subplots(1,2,figsize=(660/72,260/72))
        fig.subplots_adjust(left=.08,right=.985,bottom=.22,top=.84,wspace=.27)
        for ax,sid in zip(axes,prior.EXAMPLES):
            record=json.loads((analysis/'per_subject'/f'{sid}.json').read_text())
            row=dict(subject=prior.label(sid),subjects_source=sid,methods={})
            with np.load(analysis/'per_subject'/f'{sid}_heldout.npz') as cache:
                for m,color,label in zip(METHODS,prior.COLORS,('Rank SEQ','Spatial TRAJ.')):
                    losses=[cache[f'f{f}_{m}_angular_mae_deg'] for f in (0,1)]
                    values=np.concatenate(losses)
                    weights=np.concatenate([np.full(len(v),.5/len(v)) for v in losses])
                    mean=float(weights@values)
                    np.testing.assert_allclose(mean,record['combined'][m]['angular_mae_deg'],atol=1e-12)
                    edges=np.arange(0,181,10)
                    mass=np.histogram(values,edges,weights=weights)[0]
                    np.testing.assert_allclose(mass.sum(),1,atol=1e-12)
                    ax.stairs(100*mass,edges,color=color,lw=1.4,label=f'{label}: {mean:.2f}°')
                    ax.axvline(mean,color=color,lw=1.1,alpha=.8)
                    row['methods'][m]=dict(fold_n=list(map(len,losses)),fold_weights=[.5,.5],
                        weighted_mean_deg=mean,left_panel_value_deg=record['combined'][m]['angular_mae_deg'],
                        histogram_edges_deg=edges,histogram_probability=mass)
            ax.set_xlim(0,180);ax.set_ylim(bottom=0)
            ax.set_xticks([0,45,90,135,180]);ax.tick_params(labelsize=11)
            ax.set_xlabel('Held-out error (°)',fontsize=14)
            ax.set_title(prior.label(sid),fontsize=16,fontweight='bold',loc='left')
            ax.spines[['top','right']].set_visible(False)
            ax.legend(frameon=False,fontsize=10,loc='upper right')
            checks.append(row)
        axes[0].set_ylabel('Weighted events (%)',fontsize=13)
        fig.text(.5,.04,'Both folds · 3D nearest-template errors · vertical lines = means',ha='center',fontsize=11)
        fig.savefig(figs/'fig2b-3d-error-check.pdf');plt.close(fig)
    import pymupdf as pdf
    with pdf.open(figs/'fig2b-3d-error-check.pdf') as doc:
        page=doc[0]
        page.get_pixmap(matrix=pdf.Matrix(dpi/72,dpi/72),alpha=False).save(figs/'fig2b-3d-error-check.png')
        scale=1700/page.rect.width
        page.get_pixmap(matrix=pdf.Matrix(scale,scale),alpha=False).save(figs/'fig2b-3d-error-check-preview.png')
    write_json(out/'exemplar_3d_consistency.json',checks)
    meta=json.loads((out/'metadata.json').read_text())
    meta.update(producer=str(Path(__file__)),shared_renderer=str(Path(shared.__file__)),
                direction_assignment='nearest_3d_angle',exemplar_consistency='exemplar_3d_consistency.json',
                old_rank_selected_direction_candidate=str(shared.OUT))
    write_json(out/'metadata.json',meta)
    shutil.copy2(Path(__file__),out/'source_snapshot'/Path(__file__).name)
    with (figs/'README.md').open('a') as handle:
        handle.write('\n右侧保留原始方向分布与两条模板实线；图内不显示fold信息，图例恢复为Event directions并放在上方rose内。两个rose直径仍为120 pt；下方rose上移28 pt，圆周之间间距由68 pt缩至40 pt，上方rose位置与字号不变。上述第一折投影定义保留在说明与metadata中，数值和统计均沿用nearest_direction_holdout_20261023。\n'
            '\n### fig2b-3d-error-check.pdf / fig2b-3d-error-check.png\n'
            'E10/E14两个互补留出折的真实三维最近模板角误差分布；每折权重为1/2，每折内事件等权，柱区间宽10°。竖线为原始逐事件角误差的加权均值，未使用分箱中心近似，与主图左侧对应患者的点精确一致。\n'
            '**关注点**：此图可以直接核对左侧误差；主图右侧rose为第一折二维方向示意，不能替代本图。\n')
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,default=ANALYSIS)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--dpi',type=int,default=600)
    args=parser.parse_args();print(build(args.analysis,args.output,args.dpi))
