from pathlib import Path
from copy import deepcopy
import hashlib
import json
import shutil
import numpy as np
from PIL import Image

ROOT=Path('/home/honglab/leijiaxin/HFOsp')
REL=Path('results/paper-ready-figure/fig4')
BASE=ROOT/REL
ARCHREL=Path('results/paper-ready-figure/archive/2026-10-11_fig4_finalization')
ARCH=ROOT/ARCHREL
STAGE=BASE.parent/'.fig4-final-staging-20261011'
CANDIDATE='a_core_search_central_shading_20261011'
SELECTED=BASE/'candidates'/CANDIDATE
VERSION='core_search_final_20261011'
LAYOUT='compact_ai_v4_a_core_search_central_shading_v15'
PRODUCER='scripts/paper_figures/build_fig4_current.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def relocate(x):
 if isinstance(x,dict):return {relocate(k):relocate(v) for k,v in x.items()}
 if isinstance(x,list):return [relocate(v) for v in x]
 if isinstance(x,str):return x.replace(str(REL)+'/candidates/',str(ARCHREL/'fig4/candidates')+'/')
 return x
def accept(x):
 if isinstance(x,dict):
  return {('previous_formal_unchanged_during_candidate_generation' if k=='canonical_output_hashes_unchanged' else k):
    ('AUTHOR_ACCEPTED_FINAL' if k=='human_visual_acceptance' else accept(v)) for k,v in x.items()}
 if isinstance(x,list):return [accept(v) for v in x]
 return x
assert not ARCH.exists() and not STAGE.exists()
pointer=read(BASE/'current_version.json')
record=read(SELECTED/'figure4_candidate_registry.json')
assert pointer['pending_panel_a_revision']['version']=='a_core_search_central_shading_v15'
assert record['layout_version']==LAYOUT
for name,value in pointer['outputs_sha256'].items():assert sha(ROOT/name)==value,name
for name,value in record['outputs'].items():assert sha(SELECTED/name)==value,name
for letter in 'bcdefghi':
 name=f'figures/fig4-panel{letter}.png'
 with Image.open(BASE/name) as a,Image.open(SELECTED/name) as b:np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
for key in ('panel_b','panel_c','panel_d','panel_e','panel_i'):assert pointer[key]==record[key],key
old_files={str(p.relative_to(BASE)):sha(p) for p in BASE.rglob('*') if p.is_file()}
candidates=[]
for p in sorted((BASE/'candidates').iterdir()):
 if not p.is_dir():continue
 metadata={}
 for n in ('figure4_candidate_registry.json','figure_registry.json','current_version.json','metadata.json'):
  if (p/n).exists():
   metadata=read(p/n);break
 candidates.append(dict(id=p.name,status='PROMOTED_SOURCE_ARCHIVED' if p.name==CANDIDATE else 'HISTORICAL_SUPERSEDED',
  original_status=metadata.get('status','UNREGISTERED_HISTORICAL'),
  layout_version=metadata.get('layout_version',metadata.get('version')),
  original_path=str(REL/'candidates'/p.name),path=str(ARCHREL/'fig4/candidates'/p.name)))
(STAGE/'figures').mkdir(parents=True)
shutil.copytree(SELECTED/'source',STAGE/'source')
for name in record['outputs']:
 source=SELECTED/name if Path(name).name.startswith(('fig4-panela.','fig4-complete-layout')) else BASE/name
 shutil.copy2(source,STAGE/name)
outputs={name:sha(STAGE/name) for name in record['outputs']}
(STAGE/'source/native_exports').mkdir()
for name in outputs:shutil.copy2(STAGE/name,STAGE/'source/native_exports'/Path(name).name)
shutil.copy2(ROOT/PRODUCER,STAGE/'source/build_fig4_current.py')
accepted=relocate(deepcopy(pointer['pending_panel_a_revision']))
accepted.update(status='AUTHOR_ACCEPTED_FINAL',human_visual_acceptance='AUTHOR_ACCEPTED_FINAL',
 author_accepted_on='2026-10-11',canonical_figure_replaced=True,package=str(REL),producer=PRODUCER,
 preview=str(REL/'figures/fig4-complete-layout-preview.png'),registry=str(REL/'figure4_panel_registry.json'),
 archived_source_package=str(ARCHREL/'fig4/candidates'/CANDIDATE))
record=accept(relocate(record))
record.update(status='AUTHOR_ACCEPTED_FINAL',version=VERSION,layout_version=LAYOUT,
 human_visual_acceptance='AUTHOR_ACCEPTED_FINAL',author_accepted_on='2026-10-11',updated_on='2026-10-11',
 producer=PRODUCER,original_render_producer='source/build_fig4_a_core_optimization_candidate.py',
 canonical_figure_replaced=True,published_from=accepted['archived_source_package'],
 previous_version_archive=str(ARCHREL/'fig4'),accepted_panel_a_revision=accepted,outputs=outputs)
record.pop('pending_panel_a_revision',None)
record['panel_a'].update(status='AUTHOR_ACCEPTED_FINAL',author_accepted_on='2026-10-11')
record['checks']['A_revision']=record['panel_a']
record['titles']['A']='局部E/I回路、二维多core优化示意、局部采样与SEEG读出'
record['source_files_role']='Historical provenance; current package verification and export restoration use only bundled files.'
write(STAGE/'figure4_panel_registry.json',record)
current=accept(relocate(deepcopy(pointer)))
current.pop('pending_panel_a_revision',None)
for k in ('status','version','layout_version','human_visual_acceptance','updated_on','author_accepted_on','producer','published_from','previous_version_archive','accepted_panel_a_revision','panel_a','panel_b','panel_c','panel_d','panel_e','panel_i'):
 current[k]=record[k]
current.update(author_designated_on='2026-10-11',outputs_sha256={str(REL/name):value for name,value in outputs.items()},
 history_registry=str(REL/'historical_versions.json'),release_manifest=str(REL/'release_manifest.json'))
write(STAGE/'current_version.json',current)
qa=accept(relocate(read(SELECTED/'visual_qa.json')))
qa.update(status='PASS',human_visual_acceptance='AUTHOR_ACCEPTED_FINAL',author_accepted_on='2026-10-11',version=VERSION)
write(STAGE/'visual_qa.json',qa)
validation=accept(relocate(read(SELECTED/'mechanism_validation.json')))
validation.update(status='PASS',human_visual_acceptance='AUTHOR_ACCEPTED_FINAL',author_accepted_on='2026-10-11')
write(STAGE/'mechanism_validation.json',validation)
unchanged={name:sha(BASE/name) for name in outputs if Path(name).name.startswith(tuple('fig4-panel'+l+'.' for l in 'bcdefghi'))}
assert len(unchanged)==24
for name,value in unchanged.items():assert outputs[name]==value
approved={name:value for name,value in record['outputs'].items() if Path(name).name.startswith(('fig4-panela.','fig4-complete-layout'))}
for name,value in approved.items():assert sha(SELECTED/name)==value
write(STAGE/'publication_validation.json',dict(status='PASS',version=VERSION,author_acceptance='CONFIRMED',
 formal_A_and_full_plate_match_author_accepted_v15=approved,B_through_I_byte_identical_to_previous_formal=unchanged,
 prior_package_files_archived=len(old_files),candidate_packages_archived=len(candidates),scientific_inputs_and_results_changed=False))
history=dict(schema_version=1,current_version=VERSION,archive_root=str(ARCHREL),
 policy='All archived packages are historical; only current_version.json selects the formal Figure 4. Original registrations remain intact as provenance.',
 previous_formal=dict(status='HISTORICAL_SUPERSEDED',layout_version=pointer['layout_version'],path=str(ARCHREL/'fig4')),
 candidates=candidates,old_animations=str(ARCHREL/'fig4/animations'),
 older_archives=sorted(str(p.relative_to(ROOT)) for p in (BASE.parent/'archive').glob('*/fig4') if p.is_dir()))
write(STAGE/'historical_versions.json',history)
summary=('2026-10-11作者确认的Paper Ready Figure 4，状态为AUTHOR_ACCEPTED_FINAL，版本为core_search_final_20261011。'
 '三行A–I，画布300×232 mm；B/C/D/E/H/I数据轴为50×50 mm。A采用v15：三个淡紫色问号区域、两个与E图密度峰对齐的core、'
 '上方两条OPTM箭头，下方不加箭头或文字；浅绿底纹仅在中央取样框，右侧保留局部高斯采样权重。')
(STAGE/'README.md').write_text('# Figure 4 正式定稿\n\n'+summary+'\n\n'
 '[完整预览](figures/fig4-complete-layout-preview.png) · [PDF](figures/fig4-complete-layout.pdf) · [A–I单图](figures/README.md) · [历史版本登记](historical_versions.json)\n\n'
 '正式入口：`python scripts/paper_figures/build_fig4_current.py`。添加`--output-dir /tmp/fig4-export`可从随包冻结导出恢复31个PNG/PDF/SVG文件，逐字节一致；这是图文件恢复，不是参数重绘或重跑SNN。'
 '原绘图代码、数值输入及原生导出保存在source，来源解释见source/README.md。\n\n'
 '旧正式v10、58个候选包、旧动画及重复图均已迁入`../archive/2026-10-11_fig4_finalization/`；此前更早归档仍原地保留。'
 '后续修改先输出独立修订，不能用旧工作树或旧producer覆盖正式图。\n')
figread=['### fig4-complete-layout.png / .pdf / .svg / -preview.png','',summary,
 '**关注点**：已获作者目视确认；保持三行对齐与现有字号。','']
for letter,title in record['titles'].items():
 focus=('三个问号区域、两处最终core及仅中央浅绿底纹。OPTM为示意优化方向，SEEG读出为SNN发放密度代理。' if letter=='A' else
 '保留原冻结数值、来源和统计口径；本次定稿未重新拟合模型。')
 figread += [f'### fig4-panel{letter.lower()}.png / .pdf / .svg','',title+'。独立面板不带角标，完整拼版带A–I角标。',f'**关注点**：{focus}','']
(STAGE/'figures/README.md').write_text('\n'.join(figread))
(STAGE/'panel_map.md').write_text('# 正式 Figure 4：A–I\n\n| Panel | 内容 |\n|---|---|\n'+
 '\n'.join(f'| {k} | {v} |' for k,v in record['titles'].items())+'\n\n'+summary+'\n')
(STAGE/'source/README.md').write_text('# 定稿来源与恢复方式\n\n'
 '`native_exports/`冻结已获作者确认的整图及A–I单图，保留原PNG/PDF/SVG，不重新栅格化或改变版式。正式入口build_fig4_current.py只验证与恢复这些原文件，无需数据盘、旧worktree或历史候选目录。\n\n'
 '本目录其余数值快照与绘图脚本保留原始来源，用于后续科学修改和追溯。旧脚本的路径及版本条件是当时的绘图环境记录，不是当前正式发布命令；重新按参数绘制仍需按其原依赖进行适配。'
 '历史candidate路径的迁移映射见../historical_versions.json，完整旧包在归档。\n\n'
 'A的两个core位置取E中两个边际密度峰，仅作为位置示意，不等于联合最优解或临床病灶。SEEG readout为SNN发放密度代理，不是新增电位前向模型；B–I数据与先前正式版保持。\n')
manifest=dict(status='AUTHOR_ACCEPTED_FINAL',version=VERSION,layout_version=LAYOUT,producer=PRODUCER,
 author_accepted_on='2026-10-11',scope='Accepted figure exports, frozen numerical snapshots, rendering provenance, and current metadata.',
 restore_method='Byte-identical restoration from bundled native_exports; no raw-data or parameter rerender claim.',
 previous_version_archive=str(ARCHREL/'fig4'),files={str(p.relative_to(STAGE)):dict(sha256=sha(p),bytes=p.stat().st_size)
  for p in sorted(STAGE.rglob('*')) if p.is_file()})
write(STAGE/'release_manifest.json',manifest)
ARCH.mkdir(parents=True)
(ARCH/'documentation').mkdir()
for n in ['AGENTS.md','docs/current_figure4.md','docs/paper_figure_registry.md','docs/figure_style_guide.md','results/paper-ready-figure/README.md','results/FIGURE_INDEX.md']:
 dest=ARCH/'documentation'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
(ARCH/'code').mkdir()
for n in ['build_main_figure_4.py','build_fig4_current_aj.py','build_fig4_compact_ai.py','build_fig4_a_mechanism_candidate.py','build_fig4_a_core_optimization_candidate.py','publish_fig4_a_mechanism.py']:
 shutil.copy2(ROOT/'scripts/paper_figures'/n,ARCH/'code'/n)
shutil.copy2(__file__,ARCH/'finalize_fig4_20261011.py')
BASE.rename(ARCH/'fig4')
for name,value in old_files.items():assert sha(ARCH/'fig4'/name)==value,name
STAGE.rename(BASE)
write(ARCH/'archive_manifest.json',dict(status='HISTORICAL_ARCHIVE',archived_on='2026-10-11',
 original_package=str(REL),superseded_by=str(REL/'current_version.json'),file_count=len(old_files),files=old_files))
write(ARCH/'history_registry.json',history)
(ARCH/'README.md').write_text('# Figure 4 历史版本归档\n\n'
 '此目录仅供历史追溯，当前正式版本为[core_search_final_20261011](../../fig4/current_version.json)。'
 'fig4保留归档前的整个目录，含原正式v10、58个候选、旧动画与重复导出；原记录未改写，历史身份以history_registry.json为准。'
 'a_core_search_central_shading_20261011为获作者确认后提升至正式的来源，其余候选均登记为历史。\n\n'
 f'迁移前后{len(old_files)}个文件逐一核对SHA256一致。archive_manifest.json保留完整清单；documentation与code保留当时说明和命令入口。\n')
print(json.dumps(dict(status='PASS',current_version=VERSION,archived_files=len(old_files),candidate_packages=len(candidates),export_files=len(outputs)),indent=2))
