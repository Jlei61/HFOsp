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


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def render():
    current_record=plate.read(DEST/'figure4_panel_registry.json')
    baseline_path=DEST/'figures/fig4-complete-layout.png'
    baseline_sha=plate.sha(baseline_path)
    before=np.asarray(Image.open(baseline_path).convert('RGB'))
    draw=base.recovery.draw_modes
    validate=base.validate_new_panel
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
    with patch.object(base,'OUT',OUT),patch.object(base,'TITLES',titles),\
         patch.object(base.recovery,'draw_modes',with_legend),\
         patch.object(base,'panel_d',draw_d),\
         patch.object(base,'validate_new_panel',validate_legend):
        base.main()
    after=np.asarray(Image.open(OUT/'figures/fig4-complete-layout.png').convert('RGB'))
    assert before.shape==after.shape
    h,w=before.shape[:2];outside=np.ones((h,w),bool)
    outside[int((plate.H-63)/plate.H*h):,int(228/plate.W*w):]=False
    changed=int(np.count_nonzero(np.any(before!=after,axis=2)&outside))
    assert changed==0,changed
    record=plate.read(OUT/'figure4_candidate_registry.json')
    parent=plate.read(PARENT/'figure4_candidate_registry.json')
    assert record['candidate_J']['groups']==parent['candidate_J']['groups']
    assert record['core_count_summary']==parent['core_count_summary']
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
    write(OUT/'figure4_candidate_registry.json',record)
    qa=plate.read(OUT/'visual_qa.json')
    qa.update(status='LAYOUT_AND_VALUE_CHECKS_PASS_VISUAL_PENDING',outside_panel_j_changed_pixels=changed,
              matched_mode_values_unchanged=True,author_designated_on='2026-09-29')
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
    (OUT/'panel_map.md').write_text(mapping)
    p=OUT/'figures/README.md'
    notes=p.read_text().replace('候选待作者目视检查。','本版按用户指定作为当前A–J版；J图例在右上角。').replace('候选编号','当前编号')
    if (DEST/'figures/README.md').exists():
        baseline_notes=(DEST/'figures/README.md').read_text()
        j_section=next(block for block in notes.split('### ') if block.startswith('fig4-panelj.png'))
        notes='### '.join(j_section if block.startswith('fig4-panelj.png') else block
                         for block in baseline_notes.split('### '))
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
    panel_j=current.pop('panel_i',current.get('panel_j',{}))
    panel_j.update(legend_present=True,legend_location='upper right',legend_labels=['Subject','Median'],
                   legend_columns=1,legend_rows=2)
    palette=current['palette'];palette.pop('I',None)
    palette['J']=dict(TA_MTA='#C63D3A',TB_MTB='#287FA1',legend_present=True)
    current.update(panel_letters=list('ABCDEFGHIJ'),asset_id='patient_geometry_prior_snn_complete_a_j',
        producer=PRODUCER,published_from=str(OUT.relative_to(ROOT)),outputs_sha256=hashes,
        parent_package=str(PARENT.relative_to(ROOT)),panel_j=panel_j,palette=palette,
        previous_version_archive=str(ARCHIVE.relative_to(ROOT)))
    path=DEST/'current_version.json';temporary=path.with_suffix('.json.tmp')
    write(temporary,current);temporary.replace(path)
    print(DEST/'figures/fig4-panelj.png',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--render-only',action='store_true')
    mode.add_argument('--publish-only',action='store_true')
    args=parser.parse_args()
    if not args.publish_only:render()
    if not args.render_only:publish()
