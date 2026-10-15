#!/usr/bin/env python3
"""Render and publish the author-designated A-J Fig4 with J's upper-right legend."""
from pathlib import Path
import argparse
import json
import shutil
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scripts.paper_figures import build_fig4_core_count_candidate as base

plate=base.plate
OUT=ROOT/'results/paper-ready-figure/fig4/candidates/complete_aj_legend_20260929'
DEST=ROOT/'results/paper-ready-figure/fig4'
PARENT=base.OUT
ARCHIVE=ROOT/'results/paper-ready-figure/archive/2026-09-29_pre_aj_legend_fig4/fig4'
PRODUCER=str(Path(__file__).relative_to(ROOT))
CORE_PREVIEW=None


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def render():
    current_record=plate.read(DEST/'figure4_panel_registry.json')
    baseline_path=DEST/'figures/fig4-complete-layout.png'
    baseline_sha=plate.sha(baseline_path)
    before=np.asarray(Image.open(baseline_path).convert('RGB'))
    draw=base.recovery.draw_modes
    validate=base.validate_new_panel
    frozen=base.frozen_case
    def selected_case():
        if CORE_PREVIEW is None:return frozen()
        return frozen(CORE_PREVIEW/'source/core_count_input_snapshot.json',
                      core_counts=(1,2,3,4),summary_style='mean_std')
    # A separate D candidate can advance independently of the published figure.
    d_metric=current_record.get('original_panel_checks',{}).get('D_candidate_right_axis',{}).get('metric')
    draw_d=base.panel_d if d_metric=='participation_error' else lambda fig,curves,geometry,workpoints:plate.panel_c(fig,curves,geometry)
    def with_legend(ax,summary,**kwargs):
        return draw(ax,summary,legend_loc='upper right')
    def validate_legend(fig,case):
        checks=validate(fig,case)
        ax=plate.GROUPS['I']['axes'][0]  # Source group I is displayed as J.
        legend=ax.get_legend();box=legend.get_window_extent()
        maximum=max(float(np.max(c.get_offsets()[:,1])) for c in ax.collections)
        top=ax.transData.transform((0,maximum))[1]
        assert box.y0>top+3, (box.y0,top)
        assert legend._loc==1
        subject,median=[t.get_window_extent() for t in legend.get_texts()]
        assert subject.y0>median.y1 and abs(subject.x0-median.x0)<1e-7
        checks.update(J_legend_labels=[t.get_text() for t in legend.get_texts()],
                      J_legend_location='upper right',J_legend_clear_of_points=True,
                      J_legend_columns=1,J_legend_rows=2)
        return checks
    titles=dict(base.TITLES,J='逐患者 TA–MTA 与 TB–MTB 模板相似度（右上角 Subject / Median 上下排列图例）')
    titles.update({key:value for key,value in current_record['titles'].items() if key!='J'})
    if CORE_PREVIEW is not None:
        titles['C']='E1146 一至四 core 的最佳训练 loss：均值±标准差'
    with patch.object(base,'OUT',OUT),patch.object(base,'TITLES',titles),\
         patch.object(base,'frozen_case',selected_case),\
         patch.object(base.recovery,'draw_modes',with_legend),\
         patch.object(base,'panel_d',draw_d),\
         patch.object(base,'validate_new_panel',validate_legend):
        base.main()
    after=np.asarray(Image.open(OUT/'figures/fig4-complete-layout.png').convert('RGB'))
    assert before.shape==after.shape
    h,w=before.shape[:2];outside=np.ones((h,w),bool)
    if CORE_PREVIEW is None:
        outside[int((plate.H-63)/plate.H*h):,int(228/plate.W*w):]=False
    else:
        outside[:int((plate.H-195)/plate.H*h)+1,int(208/plate.W*w):]=False
    changed=int(np.count_nonzero(np.any(before!=after,axis=2)&outside))
    assert changed==0,changed
    record=plate.read(OUT/'figure4_candidate_registry.json')
    parent=plate.read(PARENT/'figure4_candidate_registry.json')
    assert record['candidate_J']['groups']==parent['candidate_J']['groups']
    if CORE_PREVIEW is None:
        assert record['core_count_summary']==parent['core_count_summary']
    else:
        # Preserve every other standalone asset byte-for-byte after checking the
        # native rebuild reproduces their rendered pixels.
        for letter in 'abdefghij':
            stem=f'fig4-panel{letter}'
            np.testing.assert_array_equal(np.asarray(Image.open(DEST/'figures'/f'{stem}.png')),
                                          np.asarray(Image.open(OUT/'figures'/f'{stem}.png')))
            for ext in ('png','pdf','svg'):
                shutil.copy2(DEST/'figures'/f'{stem}.{ext}',OUT/'figures'/f'{stem}.{ext}')
        record['outputs']={str(p.relative_to(OUT)):plate.sha(p) for p in (OUT/'figures').iterdir()
                           if p.suffix in ('.png','.pdf','.svg')}
    record.update(status='CURRENT_AUTHOR_DESIGNATED',author_designated_on='2026-09-29',
                  producer=PRODUCER,changes='Use the A-J layout; J has Subject above Median in one legend column at upper right.',
                  parent_package=str(PARENT.relative_to(ROOT)),outside_panel_j_changed_pixels=changed,
                  panel_j=record.pop('candidate_J'))
    record['base_published_complete_sha256']=baseline_sha
    record['source_files'][str(Path(__file__))]=plate.sha(__file__)
    record['checks']['matched_mode_values_unchanged']=True
    recovery_record=plate.read(base.recovery.OUT/'figure4_candidate_registry.json')
    record['panel_j_interpretation']=recovery_record['interpretation']
    record['source_metadata_correction']=recovery_record['source_metadata_correction']
    if CORE_PREVIEW is not None:
        preview=plate.read(CORE_PREVIEW/'preview_registry.json')
        row=next(r for r in preview['comparison'] if r['core_counts']=='1,2,3,4')
        record.update(changes='Update C to four core counts with thin mean lines, small points and sample SD bands.',
                      updated_on='2026-10-08',outside_panel_c_changed_pixels=changed,
                      other_standalone_panels_pixel_identical=True,
                      scientific_scope='E1146 best-so-far training loss over 31 shared parameter evaluations; four optimization restarts per arm; independent-noise confirmation deferred.',
                      panel_c=dict(preview_package=str(CORE_PREVIEW.relative_to(ROOT)),
                          core_counts=[1,2,3,4],summary_style='mean_std',std_ddof=1,
                          statistical_unit='optimization_restart_within_E1146',restarts=4,common_epoch=31,
                          comparison=row,author_accepted_on='2026-10-08',
                          snapshot_sha256=plate.sha(CORE_PREVIEW/'source/core_count_input_snapshot.json')))
        record['checks'].update(C_summary_style='mean_std',C_sample_std_ddof=1,C_restarts=4,
                                C_shading_artist_values_exact=True,C_individual_trajectories=False)
        record.pop('outside_panel_j_changed_pixels',None)
        for name in ('comparison.csv','mean_std_curves.csv'):
            shutil.copy2(CORE_PREVIEW/name,OUT/'source'/f'core_count_{name}')
        record['packaged_rendering_sources']={}
        for source in (Path(__file__),Path(base.__file__),Path(plate.__file__),Path(base.progress.__file__)):
            target=OUT/'source'/source.name
            shutil.copy2(source,target)
            record['packaged_rendering_sources'][str(target.relative_to(OUT))]=plate.sha(target)
    write(OUT/'figure4_candidate_registry.json',record)
    qa=plate.read(OUT/'visual_qa.json')
    qa.update(status='LAYOUT_AND_VALUE_CHECKS_PASS_VISUAL_PENDING',outside_panel_j_changed_pixels=changed,
              matched_mode_values_unchanged=True,author_designated_on='2026-09-29')
    if CORE_PREVIEW is not None:
        qa.pop('outside_panel_j_changed_pixels',None)
        qa.update(outside_panel_c_changed_pixels=changed,other_standalone_panels_pixel_identical=True,
                  panel_c_author_accepted_on='2026-10-08',panel_c_summary='mean +/- sample SD; n=4; ddof=1')
        qa['checks']=record['checks']
    write(OUT/'visual_qa.json',qa)
    mapping=(OUT/'panel_map.md').read_text().replace('# Figure 4：新 C 候选（A–J）','# 当前 Figure 4：A–J')
    mapping=mapping.replace('状态：候选，待作者目视检查；当前指定 A–I 的文件与机器入口保持不变。',
        '状态：CURRENT_AUTHOR_DESIGNATED。用户确认当前编号为A–J；J在图内右上角以一列两行显示Subject / Median图例。')
    mapping=mapping.replace('候选编号','当前编号').replace('build_fig4_core_count_candidate.py','build_fig4_current_aj.py')
    if (DEST/'panel_map.md').exists():
        mapping=(DEST/'panel_map.md').read_text().replace('右上角加入Subject / Median图例','右上角以一列两行显示Subject / Median图例')
        mapping='\n'.join(f'| J | {titles["J"]} |' if line.startswith('| J |') else line for line in mapping.split('\n'))
    if d_metric!='participation_error':
        mapping='\n'.join('D沿用当前正式图：上排左轴为Mean rank error、右轴为Order error；下排空间地图为rank/order误差。'
                          if line.startswith('D 上排两张参数曲线的右轴改为 Participation error') else line
                          for line in mapping.split('\n'))
    mapping+='\nJ每个点代表一位患者，横线为中位数，竖段及端帽为四分位区间；25位患者的两项相似度及工作点均未改变。I为原触点分折交叉匹配，J为完整模板描述性相似度。\n'
    if CORE_PREVIEW is not None:
        mapping='\n'.join(f'| C | {titles["C"]} |' if line.startswith('| C |') else
            'C 的轴框为51 × 45 mm，与B纵轴上下边界一致；使用Epochs／Loss标签、四种颜色及小点细线，无标题或图注。' if line.startswith('C 的轴框') else
            'C 使用2026-10-08作者认可的数据快照：每组4次优化重复，共同前31个epoch。线为各次最佳训练loss的算术均值，阴影为均值±样本标准差（ddof=1）。3／4 core的两轮反馈优化阶段已完成；独立噪声复测暂缓。' if line.startswith('数据冻结于') else line
            for line in mapping.split('\n'))
    (OUT/'panel_map.md').write_text(mapping)
    p=OUT/'figures/README.md'
    notes=p.read_text().replace('候选待作者目视检查。','本版按用户指定作为当前A–J版；J图例在右上角。').replace('候选编号','当前编号')
    if (DEST/'figures/README.md').exists():
        baseline_notes=(DEST/'figures/README.md').read_text()
        j_section=next(block for block in notes.split('### ') if block.startswith('fig4-panelj.png'))
        notes='### '.join(j_section if block.startswith('fig4-panelj.png') else block
                         for block in baseline_notes.split('### '))
    p.write_text(notes)
    if CORE_PREVIEW is not None:
        block=('fig4-panelc.png\nE1146一至四core的最佳训练loss；每组4次优化重复，展示共同前31个epoch。'
               '细线和小点为均值，色带为均值±样本标准差（ddof=1），另有同源PDF/SVG。\n'
               '**关注点**：用户已确认该四组预览并要求放入C；训练loss比较不等于独立噪声确认。\n\n')
        notes='### '.join(block if part.startswith('fig4-panelc.png') else part for part in notes.split('### '))
        notes=notes.replace('新 C 单核／双核 loss 曲线','C 一至四 core 均值±标准差 loss 曲线')
        p.write_text(notes)


def publish():
    record=plate.read(OUT/'figure4_candidate_registry.json')
    current=plate.read(DEST/'current_version.json')
    assert current['producer'] in [PRODUCER,'scripts/paper_figures/build_fig4_template_recovery_by_mode.py']
    expected=record.get('base_published_complete_sha256')
    current_image=plate.sha(DEST/'figures/fig4-complete-layout.png')
    assert expected is None or current_image in [expected,record['outputs']['figures/fig4-complete-layout.png']], 'Published figure changed during rendering; rebuild against the new version.'
    for name,digest in record['outputs'].items():assert plate.sha(OUT/name)==digest,name
    if not ARCHIVE.exists():
        ARCHIVE.mkdir(parents=True)
        for directory in ['figures','source']:
            shutil.copytree(DEST/directory,ARCHIVE/directory)
        for path in DEST.iterdir():
            if path.is_file():shutil.copy2(path,ARCHIVE/path.name)
    shutil.copytree(OUT/'figures',DEST/'figures',dirs_exist_ok=True)
    shutil.copytree(OUT/'source',DEST/'source',dirs_exist_ok=True)
    shutil.copy2(OUT/'panel_map.md',DEST/'panel_map.md')
    record.update(published_from=str(OUT.relative_to(ROOT)),historical_package=str(ARCHIVE.relative_to(ROOT)),
        current_package_and_pointer_unchanged=False,
        panels={s:[str((DEST/'figures'/f'fig4-panel{s}.{ext}').relative_to(ROOT))
                   for ext in ['png','pdf','svg']] for s in 'abcdefghij'})
    for name in ['figure4_panel_registry.json','figure4_candidate_registry.json']:write(DEST/name,record)
    qa=plate.read(OUT/'visual_qa.json')
    hashes={str((DEST/name).relative_to(ROOT)):digest for name,digest in record['outputs'].items()}
    for name,digest in hashes.items():assert plate.sha(ROOT/name)==digest
    qa.update(publication_bytes_identical=True,outputs_sha256=hashes,current_package_and_pointer_unchanged=False)
    write(DEST/'visual_qa.json',qa)
    (DEST/'README.md').write_text(
        '# 当前 Figure 4：A–J\n\n当前图已正式放入`figures/`。C为E1146单核／双核训练loss曲线；'
        'I为交叉匹配矩阵，J为25位患者的TA–MTA / TB–MTB相似度，图内右上角为一列两行的Subject / Median图例。\n\n'
        '[Fig4J](figures/fig4-panelj.png) · [完整PDF](figures/fig4-complete-layout.pdf) · '
        '[完整预览](figures/fig4-complete-layout-preview.png) · [编号说明](panel_map.md)\n\n'
        '点为患者，横线为中位数，竖段与端帽为四分位区间。PNG/PDF/SVG同源导出，科学数据保持冻结快照。'
        '生成入口：`python scripts/paper_figures/build_fig4_current_aj.py`。\n\n'
        '上一版A–I保留在[归档](../archive/2026-09-29_pre_aj_legend_fig4/fig4/README.md)。\n')
    if CORE_PREVIEW is not None:
        p=DEST/'README.md'
        text=p.read_text().replace('C为E1146单核／双核训练loss曲线',
            'C为E1146一至四core训练loss曲线，细线小点为4次优化重复的均值，阴影为±样本标准差，共同前31个epoch')
        text+=f'\n2026-10-08按用户要求更新C；更新前完整包保留在 `{ARCHIVE.relative_to(ROOT)}`。\n'
        p.write_text(text)
    panel_j=current.pop('panel_i',current.get('panel_j',{}))
    panel_j.update(legend_present=True,legend_location='upper right',legend_labels=['Subject','Median'],
                   legend_columns=1,legend_rows=2)
    palette=current['palette'];palette.pop('I',None)
    palette['J']=dict(TA_MTA='#C63D3A',TB_MTB='#287FA1',legend_present=True)
    current.update(panel_letters=list('ABCDEFGHIJ'),asset_id='patient_geometry_prior_snn_complete_a_j',
        producer=PRODUCER,published_from=str(OUT.relative_to(ROOT)),outputs_sha256=hashes,
        parent_package=str(PARENT.relative_to(ROOT)),panel_j=panel_j,palette=palette,
        previous_version_archive=str(ARCHIVE.relative_to(ROOT)))
    if CORE_PREVIEW is not None:
        current.update(updated_on='2026-10-08',panel_c=record['panel_c'])
        current['palette']['C_core_counts']={str(k):v for k,v in base.progress.COLORS.items()}
    path=DEST/'current_version.json';temporary=path.with_suffix('.json.tmp')
    write(temporary,current);temporary.replace(path)
    print(DEST/'figures'/('fig4-panelc.png' if CORE_PREVIEW is not None else 'fig4-panelj.png'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--render-only',action='store_true')
    mode.add_argument('--publish-only',action='store_true')
    parser.add_argument('--core-count-preview',type=Path,
                        help='Author-selected four-core mean/SD preview package to use in C.')
    args=parser.parse_args()
    pointer=plate.read(DEST/'current_version.json')
    selected=args.core_count_preview or pointer.get('panel_c',{}).get('preview_package')
    if selected:
        CORE_PREVIEW=(ROOT/selected).resolve()
        OUT=ROOT/'results/paper-ready-figure/fig4/candidates/complete_aj_core_mean_sd_20261008'
        ARCHIVE=ROOT/'results/paper-ready-figure/archive/2026-10-08_pre_core_mean_sd_fig4/fig4'
    if not args.publish_only:render()
    if not args.render_only:publish()
