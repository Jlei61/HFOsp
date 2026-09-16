#!/usr/bin/env python3
"""Contact residuals in six selected geometries; diagnostic, not cohort inference."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/topic4_sef_hfo/joint_rank_space_dual_core_search_v2/replicated_residual_audit'


def main():
    source=OUT/'summary.json';d=json.loads(source.read_text());models=d['models']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(14,5.4));fig.subplots_adjust(left=.17,right=.98,top=.8,bottom=.22,wspace=.46)
    fig.suptitle('Contact-pattern residuals after eight common training networks',fontsize=16,weight='bold',y=.97)
    fig.text(.17,.87,'Six selected dual-core geometries; each compared with the same patient training distribution.',fontsize=11)
    labels=[r['candidate_id'].replace('control_historical_matched','Historical manual').replace('xy_','').replace('joint_r001_','r1_')+f' (N={r["n_events"]})' for r in models]
    for ax,key,title,bar_label,lim in zip(axes,['participation_residual','rank_residual'],
        ['A  Participation probability','B  Mean participating-contact rank'],
        ['Model − patient probability','Model − patient normalized rank'],[.6,.3]):
        a=np.array([r[key] for r in models]);im=ax.imshow(a,aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim)
        ax.set(title=title,xticks=range(len(d['contact_names'])),xticklabels=d['contact_names'],yticks=range(6),yticklabels=labels)
        ax.tick_params(axis='x',rotation=60,labelsize=8);ax.tick_params(axis='y',labelsize=8)
        c=fig.colorbar(im,ax=ax,orientation='horizontal',pad=.33,fraction=.08,shrink=.8,extend='both');c.set_label(bar_label)
    fig.text(.17,.025,'Descriptive residuals only: geometries, seeds and contacts are dependent. Red = model higher/later; blue = lower/earlier.\n'
        'Mean ranks summarize the pattern; the fitted loss still compares complete event distributions. No acceptance threshold was changed.',fontsize=9)
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'replicated_contact_residuals.{ext}',dpi=180)
    plt.close(fig)
    (folder/'README.md').write_text('''### replicated_contact_residuals.png / .pdf / .svg
比较六个已完成八个共同训练网络的双核几何与同一患者训练分布：左图为 contact 参与率残差，右图为参与条件下平均 rank 残差；红色代表模型偏高或偏晚，蓝色代表偏低或偏早。这里的平均值仅用于定位误差，优化仍比较完整事件分布。这不是六个独立患者，也不能证明所有 XY 都存在同样的模型容量限制。
**关注点**：SCL9 等 contact 的顺序偏晚是否跨已测几何存在，以及下一轮新位置能否缩小这种偏差；不能只看结构轴是否对齐。
''')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'figure_metadata.json').write_text(json.dumps({'input_hashes':{str(p):sha(p) for p in [source,Path(__file__)]},
        'outputs':{p.name:sha(p) for p in folder.glob('replicated_contact_residuals.*')},'author_acceptance':False},indent=2)+'\n')


if __name__=='__main__':main()
