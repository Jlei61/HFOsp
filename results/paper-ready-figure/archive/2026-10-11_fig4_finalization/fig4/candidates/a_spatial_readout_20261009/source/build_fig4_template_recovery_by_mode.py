#!/usr/bin/env python3
"""Build current Fig4 with separate full-template TA/MTA and TB/MTB scores."""
from pathlib import Path
import json
import shutil
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
from scripts.paper_figures import build_fig4_dual_mode_recovery as layout
from scripts.analyze_topic4_template_recovery_by_mode import OUT as DATA
from scripts.analyze_topic4_dual_mode_recovery import sha, read, write

OUT=ROOT/'results/paper-ready-figure/fig4/candidates/template_recovery_by_mode_20260928'
FIG=OUT/'figures'
COLORS=['#C63D3A','#287FA1']
LIGHT=['#EAB7B5','#B0D3E1']
PARENT=ROOT/'results/paper-ready-figure/fig4/candidates/complete_readability_20260923'


def draw_modes(ax,summary,*,legend_loc=None):
    values=np.array([r['matched_rhos'] for r in summary['subjects']],float)
    rng=np.random.default_rng(20260928)
    jitter=rng.uniform(-.12,.12,len(values))
    metadata=[]
    ax.axhline(0,color='.72',lw=.6,ls='--',zorder=0)
    for mode in [0,1]:
        finite=np.isfinite(values[:,mode]);y=values[finite,mode]
        ax.scatter(mode-.11+jitter[finite],y,s=13,c=LIGHT[mode],edgecolors=COLORS[mode],
                   linewidths=.4,alpha=.95,zorder=3)
        q25,median,q75=np.quantile(y,[.25,.5,.75])
        np.testing.assert_allclose([q25,median,q75],
                                  [summary['statistics'][mode]['q25_q75'][0],summary['statistics'][mode]['median'],
                                   summary['statistics'][mode]['q25_q75'][1]])
        x=mode+.24
        ax.plot([x,x],[q25,q75],color=COLORS[mode],lw=2.6,solid_capstyle='butt')
        ax.plot([x-.10,x+.10],[median,median],color=COLORS[mode],lw=1.8)
        for yy in [q25,q75]:ax.plot([x-.065,x+.065],[yy,yy],color=COLORS[mode],lw=.8)
        metadata.append(dict(mode=summary['mode_order'][mode],n=int(finite.sum()),
                             subjects=[r['subject'] for r,ok in zip(summary['subjects'],finite) if ok],
                             points=y.tolist(),q25_q50_q75=[q25,median,q75]))
    ax.set_xlim(-.5,1.5);ax.set_ylim(-1,1)
    ax.set_xticks([0,1],['TA–MTA','TB–MTB'])
    ax.tick_params(axis='x',length=0,pad=5,labelsize=8)
    ax.set_yticks([-1,0,1]);ax.set_ylabel('Template similarity (ρ)',fontsize=10,labelpad=3)
    for t,color in zip(ax.get_xticklabels(),COLORS):t.set_color(color);t.set_fontweight('bold')
    if legend_loc is not None:
        # Keep the legend above the largest observation within the same axis box.
        ax.set_ylim(-1,1.55)
        ax.legend([Line2D([],[],marker='o',ls='',color='#C8C8C8',
                          markeredgecolor='#666666',markeredgewidth=.4,ms=3.5),
                   Line2D([],[],color='#666666',lw=1.6)],
                  ['Subject','Median'],loc=legend_loc,ncol=1,fontsize=7,
                  handlelength=.9,handletextpad=.3,columnspacing=.7,
                  borderpad=.2,borderaxespad=.35,frameon=True,
                  facecolor='white',edgecolor='none',framealpha=1)
    return dict(groups=metadata,legend_present=legend_loc is not None,legend_location=legend_loc,
                legend_columns=1 if legend_loc is not None else 0,
                summary_interval='interquartile range, not confidence interval',
                metric='two separate signed Spearman correlations; no minimum or cross-match penalty')


def subject_table(summary):
    rows=summary['subjects'];values=np.array([r['matched_rhos'] for r in rows],float)
    fig,ax=plt.subplots(figsize=(5.7,9.2));fig.subplots_adjust(left=.41,right=.93,bottom=.10,top=.94)
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#E9E9E9')
    im=ax.imshow(values,vmin=-1,vmax=1,cmap=cmap,aspect='auto')
    ax.set_xticks([0,1],['TA–MTA','TB–MTB'])
    ax.tick_params(axis='x',top=True,labeltop=True,bottom=False,labelbottom=False,labelsize=11,length=0,pad=7)
    labels=[r['subject'].replace('epilepsiae_','E').replace('yuquan_','Y: ') for r in rows]
    ax.set_yticks(np.arange(len(rows)),labels);ax.tick_params(axis='y',labelsize=10,length=0)
    for t,c in zip(ax.get_xticklabels(),COLORS):t.set_color(c)
    for i in range(len(rows)):
        for j in [0,1]:
            v=values[i,j]
            ax.text(j,i,'NA' if not np.isfinite(v) else f'{v:+.2f}',ha='center',va='center',fontsize=10,
                    color='white' if np.isfinite(v) and abs(v)>.6 else '#333333')
    ax.axvline(.5,color='white',lw=3)
    for spine in ax.spines.values():spine.set_visible(False)
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.04,pad=.025,aspect=22)
    cb.set_ticks([-1,0,1]);cb.set_label('Template similarity (Spearman ρ)',fontsize=11)
    for ext in ['png','pdf','svg']:fig.savefig(FIG/f'cohort_template_similarity.{ext}',dpi=220)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    summary=read(DATA/'summary.json')
    previous=read(layout.DATA/'summary.json')
    assert summary['selection_sha256']==sha(layout.DATA/'selection.json')
    assert [(r['subject'],r['candidate']) for r in summary['subjects']]==[(r['subject'],r['candidate']) for r in previous['subjects']]
    with patch.object(layout,'OUT',OUT),patch.object(layout,'FIG',FIG),patch.object(layout,'draw_recovery',draw_modes):
        checks=layout.complete(summary)
    subject_table(summary)
    current=PARENT/'figures/fig4-complete-layout.png'
    before=np.asarray(Image.open(current).convert('RGB'));after=np.asarray(Image.open(FIG/'fig4-complete-layout.png').convert('RGB'))
    assert before.shape==after.shape
    h,w=before.shape[:2];region=np.ones((h,w),bool)
    region[int((layout.original.H-63)/layout.original.H*h):,int(228/layout.original.W*w):]=False
    changed=np.any(before!=after,axis=2)&region
    checks['outside_panel_i_changed_pixels']=int(changed.sum());assert not changed.any()
    source=OUT/'source'
    for name in ['summary.json','selection.json','per_subject.csv']:shutil.copy2(DATA/name,source/name)
    for letter in 'abcdefgh':
        for ext in ['png','pdf','svg']:
            name=f'fig4-panel{letter}.{ext}'
            shutil.copy2(PARENT/'figures'/name,FIG/name)
            assert sha(PARENT/'figures'/name)==sha(FIG/name)
    registry=OUT/'figure4_candidate_registry.json'
    record=read(registry) if registry.exists() else {}
    record.update(dict(
        status='CURRENT_AUTHOR_DESIGNATED',author_designated_on='2026-09-29',
        producer=str(Path(__file__).relative_to(ROOT)),
        source=str(DATA/'summary.json'),source_sha256=sha(DATA/'summary.json'),checks=checks,
        palette=dict(TA_MTA=COLORS[0],TB_MTB=COLORS[1]),statistics=summary['statistics'],
        working_points='Same 25 subject-specific J minima in the previously frozen 417-condition snapshot',
        new_simulations=0,statistical_unit='subject',source_denominator=34,eligible_subjects=25,
        changes='I now shows two independent matched-template distributions; model-only K=2 before one-to-one naming; no minimum statistic',
        H_versus_I='H retains its original contact-crossfit example; I is now full-template descriptive similarity at individual cohort workpoints',
        canonical_assets_overwritten=False,current_version_updated=True,legend_present=False,
        scope_decision='2026-09-29: user designated the separate A/B similarity panel as new Fig4I without a legend; the null branch remains discontinued.',
        interpretation='Descriptive full-template similarities at patient FIT-selected workpoints, not a significance test or held-out recovery. Patient FIT data enter training/workpoint selection; model KMeans alone excludes patient profiles.',
        source_metadata_correction='The frozen source summary incorrectly states that patient data do not enter workpoint selection; this registry interpretation supersedes that sentence without changing source arrays or scores.',
        null_or_p_values_used=False,
        output_sha256={p.name:sha(p) for p in FIG.iterdir() if p.suffix in ['.png','.pdf','.svg']}))
    write(registry,record)
    (FIG/'README.md').write_text(
        '### fig4-paneli.png\n分别展示每位患者的TA–MTA和TB–MTB完整平均rank模板相似性，红色对应TA、蓝色对应TB；每个点是一位患者，粗横线是中位数，竖段与端帽表示四分位区间。模型事件先独立做固定K=2聚类，再以两项有符号Spearman相关之和最大的单次一一配对确定名称；交叉格只作为命名备选，不作误差惩罚或额外分数。工作点、primary事件和患者模板均沿用上一版冻结快照，本版不再取双模式最小值，25位患者两项均可估计。\n**关注点**：这是完整模板的描述性匹配，不是上一版跨触点验证分数；固定K=2不等于证明自然双模态，负相关仍如实保留。\n\n'
        '### fig4-complete-layout.png\n复用9月28日版A–H的冻结绘图函数，I分别展示A/B两项相似度且无图例；替换区域外像素与上一版完全一致。整图和独立I使用同次绘图对象，均提供PNG/PDF/SVG；本版于2026-09-29由用户指定为新的Fig4I，current_version.json同步指向本完整图。\n**关注点**：H保留原示例的触点分折检验，I采用本次完整模板描述性读出；二者不应混称为同一种验证。\n\n'
        '### cohort_template_similarity.png\n按原队列顺序列出25位可运行患者的两项对应相似性，统一使用−1至1色标。A/B已按每人自己的固定模型模板完成最佳一一命名；不按结果删患者，不显示交叉格或合成的Both列。\n**关注点**：逐患者模型模式事件数、共同有效触点数和原始匹配矩阵保存在分析目录；原34人中的9位执行资格缺口继续保留。\n',encoding='utf-8')
    parent_notes=(PARENT/'figures/README.md').read_text()
    with (FIG/'README.md').open('a') as handle:
        for block in parent_notes.split('### ')[1:]:
            if any(block.startswith(f'fig4-panel{letter}.png\n') for letter in 'abcdefgh'):
                handle.write('\n### '+block)
    publish_current(record)
    print(json.dumps(summary['statistics']),flush=True)
    print(FIG,flush=True)


def publish_current(record):
    """Persist the user's 2026-09-29 designation without modifying prior assets."""
    path=ROOT/'results/paper-ready-figure/fig4/current_version.json'
    prior=read(path)
    allowed=[record['producer'],'scripts/paper_figures/build_fig4_readable_complete.py']
    if prior['producer'] not in allowed:
        return  # A later author designation takes precedence over this renderer.
    archive=OUT/'source/previous_current_version_20260928.json'
    if not archive.exists() and prior['producer']!=record['producer']:
        shutil.copy2(path,archive)
    pointer=dict(prior)
    pointer.update(author_designated_on='2026-09-29',package=str(OUT.relative_to(ROOT)),
        producer=record['producer'],registry=str((OUT/'figure4_candidate_registry.json').relative_to(ROOT)),
        visual_qa=str((OUT/'visual_qa.json').relative_to(ROOT)),
        complete_layout={ext:str((FIG/f'fig4-complete-layout.{ext}').relative_to(ROOT)) for ext in ['png','pdf','svg']},
        preview=str((FIG/'fig4-complete-layout-preview.png').relative_to(ROOT)),
        outputs_sha256={str((FIG/name).relative_to(ROOT)):digest for name,digest in record['output_sha256'].items()},
        parent_package=str(PARENT.relative_to(ROOT)),
        panel_i=dict(legend_present=False,n_subjects=25,working_points=record['working_points'],
                     statistical_unit='subject',source=record['source'],source_sha256=record['source_sha256']),
        scientific_claims=record['interpretation'])
    pointer['palette']=dict(prior['palette'],I=dict(record['palette'],legend_present=False))
    temporary=path.with_suffix('.json.tmp');write(temporary,pointer);temporary.replace(path)
    titles=['局部E/I回路与患者电极空间基底','参数搜索的三类误差','EE强度、方向与核位置的误差响应',
            '较高误差工作点传播示例','较低误差工作点传播示例','连续30–80 Hz虚拟接触活动',
            '模型与患者的平均传播rank','模型—患者触点分折交叉匹配','逐患者TA–MTA与TB–MTB模板相似度']
    table=['# 当前 Figure 4 A–I 对应关系','',
        'CURRENT_AUTHOR_DESIGNATED：2026-09-29用户指定无图例的A/B分别展示版为新Fig4I。',
        'A–H沿用9月28日版本，独立资产逐字节复制；完整拼图I区域外像素完全相同。','',
        '| 编号 | 内容 | PNG |','|---|---|---|']
    table += [f'| {letter} | {title} | [PNG](figures/fig4-panel{letter.lower()}.png) |'
              for letter,title in zip('ABCDEFGHI',titles)]
    table += ['', '各panel及完整拼图均提供同名PDF/SVG。A–H原始来源和版式见[上一版映射](../complete_readability_20260923/panel_map.md)。',
        '', 'I：红色TA–MTA，蓝色TB–MTB；点为患者，横线为中位数，竖段与端帽为四分位区间，无legend。25位患者沿用417条件冻结快照内各自训练J最低的工作点；原34人的9位执行资格缺口保留。',
        '', 'H保留原触点分折示例；I为患者FIT选点后的完整模板描述性相似度，不是独立验证。具体定义及统计见[科学说明](../../../../../docs/archive/topic4/cohort_template_recovery_by_mode_2026-09-28.md)。']
    (OUT/'panel_map.md').write_text('\n'.join(table)+'\n')
    publish_paper_ready(record,pointer)


def publish_paper_ready(record,pointer):
    """Publish the approved A-I exports to the stable manuscript directory."""
    dest=ROOT/'results/paper-ready-figure/fig4'
    archive=ROOT/'results/paper-ready-figure/archive/2026-09-29_pre_template_recovery_fig4/fig4'
    for name,digest in record['output_sha256'].items():
        assert sha(FIG/name)==digest,name
    if not archive.exists():
        archive.mkdir(parents=True)
        shutil.copytree(dest/'figures',archive/'figures')
        for path in dest.iterdir():
            if path.is_file():shutil.copy2(path,archive/path.name)
    for name in record['output_sha256']:
        shutil.copy2(FIG/name,dest/'figures'/name)
        assert sha(dest/'figures'/name)==record['output_sha256'][name]
    shutil.copy2(FIG/'README.md',dest/'figures/README.md')
    shutil.copytree(OUT/'source',dest/'source',dirs_exist_ok=True)
    # These metadata describe the archived A-G numbering, not the current panels.
    for name in ['fig4-panela-metadata.json','fig4-panelf-metadata.json','fig4-panelg-metadata.json']:
        path=dest/'figures'/name
        if path.exists():
            assert sha(path)==sha(archive/'figures'/name)
            path.unlink()
    old_stat=dest/'fig4_panele_pairwise_similarity_statistics.json'
    if old_stat.exists():
        assert sha(old_stat)==sha(archive/old_stat.name)
        old_stat.unlink()
    canonical=dict(record,schema_version='paper_figure4_a_i_template_recovery_v1',
        published_from=str(OUT.relative_to(ROOT)),historical_package=str(archive.relative_to(ROOT)),
        panels={s:[str((dest/'figures'/f'fig4-panel{s}.{ext}').relative_to(ROOT))
                   for ext in ['png','pdf','svg']] for s in 'abcdefghi'},
        panel_letters_in_individual_files=False,panel_letters_in_complete_layout=True)
    canonical['checks']=json.loads(json.dumps(record['checks']))
    sources=canonical['checks'].get('source_files',{})
    for name,digest in list(sources.items()):
        path=Path(name)
        if path.is_relative_to(dest/'figures'):
            frozen=archive/'figures'/path.name
            assert sha(frozen)==digest
            sources[str(frozen)]=sources.pop(name)
    write(dest/'figure4_panel_registry.json',canonical)
    write(dest/'figure4_candidate_registry.json',canonical)
    qa=read(OUT/'visual_qa.json')
    qa.update(published_from=str(OUT.relative_to(ROOT)),publication_bytes_identical=True)
    qa['outputs_sha256']={str((dest/'figures'/name).relative_to(ROOT)):digest
                          for name,digest in record['output_sha256'].items()}
    write(dest/'visual_qa.json',qa)
    mapping=(OUT/'panel_map.md').read_text().replace(
        '../complete_readability_20260923/panel_map.md','candidates/complete_readability_20260923/panel_map.md').replace(
        '../../../../../docs/','../../../docs/')
    (dest/'panel_map.md').write_text(mapping)
    (dest/'README.md').write_text(
        '# 当前 Figure 4\n\n2026-09-29按用户要求，将已指定的A–I完整图正式放入本目录的`figures/`。'
        'I采用无图例的TA–MTA / TB–MTB分别展示版，替代旧训练loss图；A–H沿用已确认版本。\n\n'
        '[独立Fig4I](figures/fig4-paneli.png) · [完整PDF](figures/fig4-complete-layout.pdf) · '
        '[完整预览](figures/fig4-complete-layout-preview.png)\n\n'
        '读取[current_version.json](current_version.json)、[panel_map.md](panel_map.md)和'
        '[版本说明](../../../docs/current_figure4.md)。PNG/PDF/SVG与已检查的版本逐字节一致。\n\n'
        '旧A–G图包保留在[历史目录](../archive/2026-09-29_pre_template_recovery_fig4/fig4/figures/README.md)；'
        '源绘图包仍保留于`candidates/template_recovery_by_mode_20260928/`以便复现。\n')
    pointer.update(package=str(dest.relative_to(ROOT)),published_from=str(OUT.relative_to(ROOT)),
        registry=str((dest/'figure4_panel_registry.json').relative_to(ROOT)),
        visual_qa=str((dest/'visual_qa.json').relative_to(ROOT)),
        complete_layout={ext:str((dest/'figures'/f'fig4-complete-layout.{ext}').relative_to(ROOT))
                         for ext in ['png','pdf','svg']},
        preview=str((dest/'figures/fig4-complete-layout-preview.png').relative_to(ROOT)),
        outputs_sha256=qa['outputs_sha256'],legacy_package=str(archive.relative_to(ROOT)))
    path=dest/'current_version.json';temporary=path.with_suffix('.json.tmp')
    write(temporary,pointer);temporary.replace(path)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish-only',action='store_true',help='Publish existing verified exports without redrawing.')
    args=parser.parse_args()
    if args.publish_only:publish_current(read(OUT/'figure4_candidate_registry.json'))
    else:main()
