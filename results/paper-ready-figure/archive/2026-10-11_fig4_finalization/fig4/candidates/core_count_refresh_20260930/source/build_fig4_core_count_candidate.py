#!/usr/bin/env python3
"""Assemble the A–J candidate: B profile and new C core-count loss panel.

Reuse the author-designated Fig4 renderers and a fixed progress snapshot.
Never advance the live experiment or change current_version.json.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.paper_figures import build_fig4_readable_complete as plate
from scripts.paper_figures import plot_core_count_optimization_progress as progress
from scripts.paper_figures import build_fig4_template_recovery_by_mode as recovery
from scripts.paper_figures import fig4_parameter_response_assets as parameters
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np

OUT=ROOT/'results/paper-ready-figure/fig4/candidates/core_count_panel_c_20260929'
SNAPSHOT=progress.DEFAULT_OUT/'snapshots/20260929_162510/input_snapshot.json'
LETTERS={'A':'A','B':'B','core_count':'C',**dict(zip('CDEFGHI','DEFGHIJ'))}
TITLES={
    'A':'局部 E/I 回路与患者电极空间基底',
    'B':'参数搜索中的三类误差（仅原左侧 panel）',
    'C':'E1146 单核与双核的最佳训练 loss',
    'D':'向外 EE／方向响应、低误差位置密度及核内 EE 三误差响应',
    'E':'较高误差工作点的传播示例（原 D；lf_bo04_07）',
    'F':'较低误差工作点的传播示例（原 E；xy_left_20）',
    'G':'连续 30–80 Hz 虚拟接触活动（原 F）',
    'H':'模型与患者的平均传播 rank（原 G）',
    'I':'模型—患者触点交叉匹配（原 H）',
    'J':'逐患者 TA–MTA 与 TB–MTB 模板相似度（沿用当前图例）',
}


def frozen_case():
    """Restore the exact attached curves, without reading later live scores."""
    path=plate.freeze(SNAPSHOT)
    shutil.copyfile(path,plate.SOURCE/'core_count_input_snapshot.json')
    snapshot=plate.read(path)
    reducer_path=plate.freeze(snapshot['reducer'])
    assert plate.sha(reducer_path)==snapshot['reducer_sha256']
    reduce=progress.original_reducer()
    case=next(c for c in snapshot['cases'] if c['summary']['subject']=='epilepsiae_1146')
    summary=case['summary'];arrays={}
    for k in (1,2):
        reduced=[reduce(rows,summary['planned_epochs']) for rows in case['records'][str(k)]]
        assert [n for _,n in reduced]==summary['prefix_by_restart'][str(k)]
        arrays[k]=np.asarray([a for a,_ in reduced])
        # Independent selection from raw records checks the plotted best-so-far values.
        for r,rows in enumerate(case['records'][str(k)]):
            for epoch in range(1,summary['prefix_by_restart'][str(k)][r]+1):
                scores=[x['J'] for x in rows if x['epoch']<=epoch and x['J'] is not None]
                if scores:np.testing.assert_allclose(arrays[k][r,epoch-1,0],min(scores),atol=0,rtol=0)
        np.testing.assert_allclose(np.median(arrays[k][:,summary['common_epoch']-1,0]),
                                   summary['median_best_loss'][str(k)],atol=1e-12,rtol=0)
    np.savez_compressed(plate.SOURCE/'core_count_plot_arrays.npz',**{f'core{k}':a for k,a in arrays.items()})
    return dict(summary=summary,records=case['records'],arrays=arrays),snapshot


def validate_new_panel(fig,case):
    fig.canvas.draw()
    b=plate.GROUPS['B']['axes'][0];c=plate.GROUPS['core_count']['axes'][0]
    bb,cb=b.get_window_extent(),c.get_window_extent()
    np.testing.assert_allclose([bb.y0,bb.y1],[cb.y0,cb.y1],atol=1e-7,rtol=0)
    assert not any(a.name=='3d' for a in fig.axes)
    assert c.get_title()=='' and not plate.GROUPS['core_count']['texts']
    legend=c.legend_.get_window_extent()
    assert cb.x0<=legend.x0<legend.x1<=cb.x1 and cb.y0<=legend.y0<legend.y1<=cb.y1
    assert [t.get_text() for t in c.legend_.get_texts()]==['1 core','2 cores']
    assert c.xaxis.label.get_fontsize()==c.yaxis.label.get_fontsize()==plate.FONT['label']
    assert all(t.get_fontsize()==plate.FONT['legend'] for t in c.legend_.get_texts())
    # Matplotlib artists must retain the exact original data despite new point sizes.
    for k,offset in [(1,0),(2,5)]:
        block=case['arrays'][k][:,:,0]
        for line,row in zip(c.lines[offset:offset+4],block):
            finite=np.flatnonzero(np.isfinite(row))
            np.testing.assert_array_equal(line.get_xdata(),finite+1)
            np.testing.assert_array_equal(line.get_ydata(),row[finite])
        np.testing.assert_array_equal(c.lines[offset+4].get_xdata(),np.arange(1,8))
        np.testing.assert_array_equal(c.lines[offset+4].get_ydata(),np.median(block[:,:7],axis=0))
    renderer=fig.canvas.get_renderer()
    b_right=max(a.get_tightbbox(renderer).x1 for a in plate.GROUPS['B']['axes'])
    c_left=c.get_tightbbox(renderer).x0
    assert c_left>b_right
    corner=b.transAxes.transform((0,0))
    for spine in ('left','bottom'):
        s=b.spines[spine]
        np.testing.assert_allclose(s.get_transform().transform(s.get_path().vertices)[0],corner,atol=1e-6)
    outside=[]
    for t in plate.visible_texts(fig):
        box=t.get_window_extent(renderer)
        if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.x1+.5 or box.y1>fig.bbox.y1+.5:
            outside.append(t.get_text())
    assert not outside,outside
    return dict(BC_equal_y_edges=True,C_axis_size_mm=[51,45],C_no_title_or_caption=True,
                C_legend_inside=True,C_artist_data_exact=True,C_original_reducer_independently_checked=True,
                BC_visible_gap_mm=(c_left-b_right)/fig.dpi*25.4,B_axis_corner_gap_px=0,
                no_3d_axes=True,text_outside_canvas=outside)


def panel_d(fig,curves,geometry,workpoints):
    """Replace the upper right-axis observations with participation error.

    Each upper curve uses the same 13 conditions as before, joined by candidate
    ID. The lower row uses the requested position density and recurrent EE scan.
    """
    start=len(fig.axes);text_start=len(fig.texts)
    plate.panel_c(fig,curves,geometry)
    axes=fig.axes[start:]
    lookup={r['candidate']:r for r in workpoints}
    assert len(lookup)==len(workpoints)
    color=plate.ERROR_COLORS['participation_error']
    plotted=[]
    for axis,right in [(4,axes[1]),(5,axes[3])]:
        selected=sorted((r for r in curves if int(r['axis'])==axis and r['observable']=='local_order'),
                        key=lambda r:float(r['parameter_value']))
        assert len(selected)==13
        values=[]
        for r in selected:
            source=lookup[r['candidate']]
            x=float(r['parameter_value']);y=source['participation_error']
            assert int(r['N'])==source['N']
            np.testing.assert_allclose(json.loads(source['vector'])[axis],x,atol=1e-12,rtol=0)
            plotted.append(dict(axis=axis,parameter_value=x,candidate=r['candidate'],
                                participation_error=y,N=source['N'],source=source['source'],
                                source_sha256=source['source_sha256']))
            values.append(y)
        np.testing.assert_array_equal(right.lines[0].get_xdata(),[float(r['parameter_value']) for r in selected])
        right.lines[0].set_ydata(values);right.lines[0].set_color(color)
        right.spines['right'].set_color(color)
        right.tick_params(axis='y',colors=color)
        np.testing.assert_array_equal(right.lines[0].get_ydata(),values)
        assert all(right.get_ylim()[0]<=y<=right.get_ylim()[1] for y in values)
    axes[3].set_ylabel(r'Participation error $\varepsilon_p$',color=color,
                      fontsize=plate.MIDDLE_LABEL_PT,labelpad=2)
    with (plate.SOURCE/'D_participation_response.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(plotted[0]));writer.writeheader();writer.writerows(plotted)
    plate.CHECKS['C_palette']['curve_right']='participation_error'
    plate.CHECKS['C_palette']['curve_right_color']=color
    plate.CHECKS['C_palette'].pop('map_gradient',None)
    plate.CHECKS['C_palette'].pop('map_limits',None)
    plate.CHECKS.pop('C_position_conditions',None)
    plate.CHECKS['D_candidate_right_axis']=dict(metric='participation_error',symbol='epsilon_p',color=color,
        rows=plotted,condition_count=26,rendered_values_equal_frozen_workpoints=True,
        shared_y_limits=list(axes[3].get_ylim()))
    parameters.draw_lower(plate,fig,axes,list(fig.texts[text_start:]))


def export(fig,letters):
    positions={a:a.get_position().frozen() for a in fig.axes}
    def save(stem,box=None,dpi=320,extensions=('png','pdf','svg')):
        for ext in extensions:
            fig.savefig(plate.FIG/f'{stem}.{ext}',dpi=dpi,bbox_inches=box)
            for a,p in positions.items():a.set_position(p)
    save('fig4-complete-layout')
    save('fig4-complete-layout-preview',dpi=135,extensions=('png',))
    visibility={a:a.get_visible() for a in fig.axes}
    for t in letters:t.set_visible(False)
    for key,display in LETTERS.items():
        for k,g in plate.GROUPS.items():
            for a in g['axes']:a.set_visible(k==key and visibility[a])
            for a in g['texts']+g['artists']:a.set_visible(k==key)
        restored=[]
        if key=='D':  # Former D / candidate E shares units with the lower row.
            for a in plate.GROUPS[key]['axes']:
                if a.images and a.get_visible() and any(t.get_visible() for t in a.get_xticklabels()):
                    restored.append((a,a.get_xlabel()))
                    a.set_xlabel('Time (ms)' if a.get_xlim()[0]<0 else 'x (mm)',fontsize=plate.MIDDLE_LABEL_PT)
        fig.canvas.draw();renderer=fig.canvas.get_renderer();g=plate.GROUPS[key]
        boxes=[a.get_tightbbox(renderer) for a in g['axes'] if a.get_visible()]
        boxes.extend(t.get_window_extent(renderer) for t in g['texts'] if t.get_visible())
        box=Bbox.union([b for b in boxes if b is not None]).transformed(fig.dpi_scale_trans.inverted()).padded(1.2*plate.MM)
        save(f'fig4-panel{display.lower()}',box=box)
        for a,old in restored:a.set_xlabel(old)


def main():
    pointer=ROOT/'results/paper-ready-figure/fig4/current_version.json'
    current=plate.read(pointer)
    # Use the fixed A-I source package even after the current pointer advances.
    base_package=recovery.OUT
    # A promoted base package can be fig4/ itself, which also contains this
    # candidate. Protect its published files without including our own outputs.
    protected=[pointer,*[ROOT/p for p in current['outputs_sha256']],
               ROOT/current['registry'],ROOT/current['visual_qa'],
               *[p for p in (base_package/'source').rglob('*') if p.is_file()]]
    before={str(p):plate.sha(p) for p in protected}
    plate.OUT=OUT;plate.FIG=OUT/'figures';plate.SOURCE=OUT/'source';plate.base.OUT=plate.SOURCE
    plate.FIG.mkdir(parents=True,exist_ok=True);plate.SOURCE.mkdir(exist_ok=True)
    plate.GROUPS.clear();plate.CHECKS.clear()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,
        'xtick.labelsize':9,'ytick.labelsize':9,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False})
    case,snapshot=frozen_case()
    rows=list(csv.DictReader(plate.freeze(plate.OLD/'source/all_workpoints_source.csv').open()))
    for r in rows:
        for k in plate.base.KEYS:r[k]=float(r[k])
        r['stage']=int(r['stage']);r['N']=int(r['N'])
    cases=plate.read(plate.freeze(plate.PREVIOUS/'source/selected_workpoints.json'))['cases']
    curves=list(csv.DictReader(plate.freeze(plate.OLD/'source/connection_response_source.csv').open()))
    geometry=plate.read(plate.freeze(plate.EXPANDED/'source/position_map_source.json'))
    assert geometry['conditions']==83
    fig=plt.figure(figsize=(plate.W*plate.MM,plate.H*plate.MM),dpi=160,facecolor='white')
    # Only B's cross-references change; the propagation renderers retain source identities.
    b_cases=[dict(c,panel=LETTERS[c['panel']]) for c in cases]
    for key,fn in [('A',lambda:plate.panel_a(fig)),
                   ('B',lambda:plate.panel_b(fig,rows,b_cases,profile_only=True)),
                   ('C',lambda:panel_d(fig,curves,geometry,rows))]:
        with plate.group(fig,key):fn()
        print('Rendered',LETTERS[key],flush=True)
    with plate.group(fig,'core_count'):
        ax=fig.add_axes(plate.rect(223,178+plate.TOP_SHIFT,51,45))
        progress.draw(ax,case,show_title=False,paper_style=plate.FONT,show_legend=True)
        ax.set_yticks([0,2,4,6,8])
    plate.panels_de(fig,cases);plate.panels_fgh(fig)
    with plate.group(fig,'I'):
        recovery_source=plate.freeze(base_package/'source/summary.json')
        recovery_summary=plate.read(recovery_source)
        shutil.copyfile(recovery_source,plate.SOURCE/'template_recovery_summary.json')
        ax=fig.add_axes(plate.rect(242,plate.LOWER_Y,32,plate.ROW_HEIGHT))
        plate.axis_type(ax)
        recovery_metadata=recovery.draw_modes(ax,recovery_summary,
            legend_loc=current.get('panel_j',{}).get('legend_location'))
    letters=[];lower_letters=[]
    for key,x,y in [('A',2,248),('B',137,248),('core_count',212,248),('C',2,179),
                    ('D',122,178),('E',122,119),('F',2,63),('G',120,63),('H',170,63),('I',230,63)]:
        t=plate.label(fig,x,y,LETTERS[key],fontsize=plate.FONT['letter'],weight='bold',va='top')
        letters.append(t)
        if key in 'FGHI':lower_letters.append(t)
    lower_axes={a for k in 'FGHI' for a in plate.GROUPS[k]['axes']}
    lower_texts={t for k in 'FGHI' for t in plate.GROUPS[k]['texts']}|set(lower_letters)
    before_shift={a:a.get_position().frozen() for a in fig.axes}
    for a,b in before_shift.items():
        x,y,w,h=b.bounds;shift=5 if a in lower_axes else 5+plate.LOWER_ROW_EXTRA_GAP_MM
        a.set_position([x,y+shift/plate.H,w,h])
    for t in fig.texts:
        shift=5 if t in lower_texts else 5+plate.LOWER_ROW_EXTRA_GAP_MM
        x,y=t.get_position();t.set_position((x,y+shift/plate.H))
    for a in plate.GROUPS['A']['artists']:
        a.set_ydata(np.array(a.get_ydata())+(5+plate.LOWER_ROW_EXTRA_GAP_MM)/plate.H)
    try:
        plate.verify_alignment(fig,panel_letters=LETTERS,extra_top_groups=('core_count',))
    except AssertionError:
        (OUT/'qa').mkdir(exist_ok=True)
        fig.savefig(OUT/'qa/layout-draft.png',dpi=135)
        raise
    checks=validate_new_panel(fig,case)
    export(fig,letters);plt.close(fig)
    changed=[p for p,digest in before.items() if plate.sha(p)!=digest]
    assert not changed,changed
    for mod in (plate,progress,recovery,parameters,plate.compare,plate.combined):plate.freeze(mod.__file__)
    plate.freeze(__file__)
    record=dict(status='CANDIDATE_PENDING_AUTHOR_VISUAL_REVIEW',producer=str(Path(__file__).relative_to(ROOT)),
        base_current_version=str(pointer.relative_to(ROOT)),base_package=str(base_package.relative_to(ROOT)),
        panel_letters=list('ABCDEFGHIJ'),base_group_to_candidate_letter=LETTERS,
        titles=TITLES,whole_canvas_mm=[plate.W,plate.H],font_pt=plate.FONT,
        changes='B keeps only the original profile; C replaces the former B 3D panel. Original C–I become D–J. D upper right axes plot participation error at the same 26 conditions. Its lower row shows the requested top-20-percent position density and the seven-point within-core EE response.',
        source_files=plate.SOURCES,source_snapshot_time=snapshot['updated_at'],core_count_summary=case['summary'],
        curve_semantics=snapshot['semantics'],current_package_and_pointer_unchanged=True,
        scientific_scope='E1146 initial-sampling training progress, not completed optimization or independent validation.',
        original_panel_checks=plate.CHECKS,checks=checks,new_simulations=0,
        candidate_J=recovery_metadata,
        outputs={str(p.relative_to(OUT)):plate.sha(p) for p in plate.FIG.iterdir() if p.suffix in ('.png','.pdf','.svg')})
    progress.write(OUT/'figure4_candidate_registry.json',record)
    progress.write(OUT/'visual_qa.json',dict(status='NUMERICAL_AND_LAYOUT_PASS_VISUAL_PENDING',
        checks=checks,current_package_and_pointer_unchanged=True,human_visual_acceptance='PENDING'))
    lines=['# Figure 4：新 C 候选（A–J）','',
        '状态：D 修订候选，待作者目视检查；当前指定 A–J 的文件与机器入口保持不变。',
        '新 C 放在 B 右侧，替换原 B 的三维 panel；B 保留左侧误差曲线及紫色迭代色条，原 C–I 顺延为 D–J。B 中工作点引用已同步为 E/F。','',
        '| 候选编号 | 内容 |','|---|---|']
    lines += [f'| {letter} | {title} |' for letter,title in TITLES.items()]
    lines += ['', 'C 的轴框为 51 × 45 mm，与 B 纵轴上下边界一致；轴标签 10 pt、刻度 9 pt、图例 8.8 pt。图内仅保留 Loss、Epochs、1 core / 2 cores，取消 E1146 子标题和进度图注。', '',
        'D 上排两张参数曲线的右轴改为 Participation error ε_p，曲线、刻度、轴线及标签使用与 B 相同的语义色 #6B7546。从冻结 all_workpoints_source.csv 按 candidate ID 提取参与误差，保留每条曲线原来的13个参数条件和0–0.42范围；实际绘图数据见 source/D_participation_response.csv。左轴保持 Mean rank error ε_r。', '',
        'D 左下复用用户参考图：257个可评分配置中 J_joint 最低20%的52个配置，核中心以0.35 mm高斯核构建等权位置密度；高度和紫色色条为 Position density (mm⁻²)，不是误差值、参数后验或独立重复的不确定性。52个配置均来自兼容历史池；快照、阈值、实际坐标及密度数组均保留。', '',
        'D 右下为已完成的 E1146 核内 E→E 权重单参数扫描：J_EE,core = 0.60、0.70、0.80、0.85、0.95、1.10、1.25；七条60秒轨迹固定位置、拓扑2511、噪声847401及其余物理。三条曲线分别为 ε_r 平均rank、ε_o 杆内顺序、ε_p 参与误差，按B语义配色；从原primary事件时刻重算并验证，非训练kernel loss。背景是队列上一轮E1146训练提名点，与上排的历史单患者扫描背景不同，不能将三张曲线拼成同一工作点的全参数切片。实际数据及审计见 source/D_recurrent_EE_response.csv 和 D_recurrent_EE_audit.json；仅为FIT描述性误差，不增加独立验证或机制验收结论。', '',
        '数据冻结于 2026-09-29 16:25:10 CST，与用户截图一致：E1146 每组四次随机搜索，共同完成前 7 个 epoch；较快的重复细线显示至第 8 个 epoch。粗线和色带分别为共同前缀的最佳训练 loss 中位数及四分位范围；第 7 个 epoch 中位数为 4.1394 和 1.9557。仍属于初始采样，未完成 96 epoch 计划；不表示独立验证。', '',
        '可复现命令：`python scripts/paper_figures/build_fig4_core_count_candidate.py`。只读取冻结数据；不重跑仿真，也不跟随后台训练自动更新输入。', '',
        '除 B/C 的布局更新、D 内容修订及编号顺延外，其他内容与排版保持；保留此前整图等高、行间距和图例位置。详情沿用基底版本 panel_map.md。']
    (OUT/'panel_map.md').write_text('\n'.join(lines)+'\n')
    notes=['### fig4-complete-layout.png\n新 C 单核／双核 loss 曲线放入 B 右侧，B 仅保留原左侧误差 panel 和色条，原 C–I 顺延为 D–J。另有同源 PDF/SVG 和整图预览，所有编号和 B 内工作点引用同步。\n**关注点**：按整图检查字号与布局；候选待作者目视检查。']
    for letter,title in TITLES.items():
        detail=('E1146 每组四次搜索，共同前7个epoch的中位数及四分位范围；个别细线延伸至第8个epoch。' if letter=='C' else
                '上排右轴为 Participation error ε_p；下排左侧为参考图的低J_joint位置密度，右侧为核内EE强度对三项误差的七点响应，各曲线与B语义色一致。数据来源及不同背景工作点见上级 panel_map.md。' if letter=='D' else
                '与完整拼图使用相同绘图对象；独立导出不带 panel 角标。')
        notes.append(f'### fig4-panel{letter.lower()}.png\n{title}。{detail}另有同源 PDF/SVG。\n**关注点**：'+('仅为冻结初始采样进度，尚非完整优化或独立验证。' if letter=='C' else '沿用原数据合同；候选编号见上级 panel_map.md。'))
    (plate.FIG/'README.md').write_text('\n\n'.join(notes)+'\n')
    print(plate.FIG,flush=True)


if __name__=='__main__':main()
