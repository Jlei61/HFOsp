#!/usr/bin/env python3
"""Align Figure 2 A/D and B+C/E columns using the existing renderers."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pymupdf as pdf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures import plot_fig2b_rank_reconstruction_holdout as errors
from scripts.paper_figures.build_fig2_split_holdout_panels import raster, sha, write_json
from src.interictal_rank_holdout import METHODS

SOURCE = ROOT / 'results/paper-ready-figure/fig2/candidates/split_holdout_panels_20261023'
OUT = SOURCE.parent / 'aligned_columns_20261023'
FIELD_RENDERER = ROOT / 'scripts/paper_figures/plot_fig2f_all_event_shared_field_reversal.py'
if not FIELD_RENDERER.is_file():
    FIELD_RENDERER = ROOT / '.worktrees/main-cherry/scripts/paper_figures/plot_fig2f_all_event_shared_field_reversal.py'
FIELD_META = ROOT / 'results/paper-ready-figure/fig2/candidates/sequence_to_spatial_field_20261009/source_snapshot/fig2_panelf_metadata.json'
LEFT_X, LEFT_W, RIGHT_X, RIGHT_W = 24., 700., 748., 583.
TOP, TOP_H, BOTTOM = 44., 430., 504.
BOTTOM_H = 350. * LEFT_W / 710.
CANVAS = [1355., 870.]


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    return value


def load_field_renderer():
    import scripts.paper_figures as package
    # Reuse the exact producer of the frozen current E, including its registry.
    path = str(FIELD_RENDERER.parent)
    if path not in package.__path__: package.__path__.append(path)
    spec = importlib.util.spec_from_file_location('fig2_current_field_renderer', FIELD_RENDERER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_b(old, figures, dpi, *, width=360., height=TOP_H,
            axes_rect=(42., 34., 342., 384.), type_scale=1.):
    analysis = Path(old['analysis_root'])
    manifest = json.loads((analysis / 'run_manifest.json').read_text())
    records = [json.loads((analysis / 'per_subject' / f'{sid}.json').read_text())
               for sid in manifest['subjects']]
    target = 'angular_mae_deg'
    adapted = [dict(subject_id=r['subject_id'],
                    **{f'{m}_mae': r['combined'][m][target] for m in METHODS},
                    **{f'{m}_null_mae': r['null'][m][target] for m in METHODS}) for r in records]
    with plt.rc_context({'font.family': 'Liberation Sans', 'pdf.fonttype': 42, 'axes.linewidth': .8}):
        fig = plt.figure(figsize=(width/72, height/72), facecolor='white')
        x0, y0, x1, y1 = axes_rect
        ax = fig.add_axes([x0/width, (height-y1)/height,
                           (x1-x0)/width, (y1-y0)/height])
        plotted = errors.draw_errors(ax, adapted, old['cohort_summary']['primary'][target],
            xlabel='Held-out error (°)', show_density_axis=True,
            xtick_step=30, min_xmax=100, xpad=5, kde_pad=.3)
        ax.yaxis.label.set_text('Patients')
        if type_scale != 1.:
            from matplotlib.text import Text
            for text in fig.findobj(Text):
                text.set_fontsize(text.get_fontsize() * type_scale)
        fig.savefig(figures / 'fig2-panelb.pdf')
        plt.close(fig)
    plotted = plain(plotted)
    for key, value in plotted.items():
        assert value == old['panel_b'][key], key
    raster(figures / 'fig2-panelb.pdf', dpi, 1000)
    return {'axes_rect_pt': list(axes_rect), 'type_scale': type_scale, 'plot_statistics': plotted,
            'renderer': str(Path(errors.__file__)), 'same_patient_values_and_statistics': True}


def build_e(figures, out, dpi, *, width=RIGHT_W, height=BOTTOM_H, type_scale=1., axis_alignment=None):
    renderer = load_field_renderer()
    frozen = json.loads(FIELD_META.read_text())
    root = Path(frozen['input_root'])
    bundled = FIELD_META.parent / 'field_inputs'
    if bundled.is_dir():
        root = bundled
        # These are the 18 already displayed labels, not the private crosswalk.
        display_rows = json.loads((bundled / 'e_patient_rows.json').read_text())
        display_ids = {row['subject_id']: row['display_id'] for row in display_rows}
        renderer._manuscript_id = display_ids.__getitem__
    rows = renderer._load_rows(root / 'shared_field_similarity_subjects.csv')
    nulls = renderer._load_channel_nulls(root / 'shared_field_similarity_null_draws.npz', rows)
    raw_summary, null_r = renderer._build_raw_cohort_median_null(
        rows, nulls, n_draws=100000, base_seed=20260721)
    for key, value in frozen['display_raw_cohort_median_null'].items():
        if isinstance(value, (int, float)):
            np.testing.assert_allclose(raw_summary[key], value, rtol=0, atol=1e-12)
    sign_p = frozen['reported_statistics']['channel']['sign_test_p_more_negative_than_half']
    # Retain E's previous on-page font and marker sizes while widening its axes.
    old_scale = 1.6427015250544663 * type_scale
    axes_top = 12. if axis_alignment is None else axis_alignment['axes_top_pt']
    axes_bottom = height-48. if axis_alignment is None else axis_alignment['axes_bottom_pt']
    with plt.rc_context({'font.family': 'Liberation Sans', 'pdf.fonttype': 42, 'axes.linewidth': .8}):
        fig = plt.figure(figsize=(width/72, height/72), facecolor='white')
        grid = fig.add_gridspec(2, 1, height_ratios=(1.75, .75),
            left=42/width, right=(width-12)/width,
            top=1-axes_top/height, bottom=1-axes_bottom/height, hspace=.025)
        distribution_ax = fig.add_subplot(grid[0, 0])
        null_ax = fig.add_subplot(grid[1, 0])
        distribution = renderer._draw_distribution(distribution_ax, rows, sign_test_p=sign_p)
        renderer._draw_null(null_ax, null_r, distribution['median'], p_negative=raw_summary['p_negative'])
        for ax in (distribution_ax, null_ax):
            for text in ax.texts: text.set_fontsize(text.get_fontsize() * old_scale)
            for line in ax.lines: line.set_linewidth(line.get_linewidth() * old_scale)
            for collection in ax.collections:
                if hasattr(collection, 'get_sizes'):
                    collection.set_sizes(collection.get_sizes() * old_scale**2)
                collection.set_linewidths(collection.get_linewidths() * old_scale)
        null_ax.tick_params(axis='x', labelsize=9*old_scale, length=2.6*old_scale, pad=3)
        null_ax.set_xlabel('Cohort median r vs spatial null', fontsize=16*type_scale, labelpad=9)
        if axis_alignment is not None:
            null_ax.tick_params(axis='x', labelsize=axis_alignment['xtick_fontsize'],
                                length=axis_alignment['xtick_length_pt'], pad=axis_alignment['xtick_pad_pt'])
            null_ax.xaxis.label.set_fontsize(axis_alignment['xlabel_fontsize'])
            null_ax.xaxis.label.set_verticalalignment('baseline')
            null_ax.xaxis.set_label_coords((42.+width-12.)/(2*width),
                1-axis_alignment['xlabel_baseline_pt']/height, transform=fig.transFigure)
        # Spread labels horizontally in the wider axes; keep every data point
        # and its vertical ordering unchanged.
        fig.canvas.draw()
        canvas_renderer = fig.canvas.get_renderer()
        ordered = sorted(rows, key=lambda row: row['r'])
        markers = distribution_ax.transData.transform(
            np.c_[[row['r'] for row in ordered], np.arange(len(ordered))])
        patient_text = {text.get_text(): text for text in distribution_ax.texts}
        occupied = []; label_positions = []
        for index, row in enumerate(ordered):
            text = patient_text[row['display_id']]
            preferred = 1 if text.get_ha() == 'left' else -1
            chosen = None
            for step in (1, 2, 3, 4):
                for sign in (preferred, -preferred):
                    text.set_position((row['r'] + sign*.035*step, index))
                    text.set_ha('left' if sign == 1 else 'right')
                    box = text.get_window_extent(canvas_renderer)
                    padded = box.padded(1.0*fig.dpi/72)
                    if box.x0 < 8*fig.dpi/72 or box.x1 > (width-8)*fig.dpi/72:
                        continue
                    if any(padded.overlaps(other) for other in occupied):
                        continue
                    if any(box.padded(3*fig.dpi/72).contains(*point) for point in markers):
                        continue
                    chosen = box
                    break
                if chosen is not None: break
            assert chosen is not None, ('No clear label position', row['display_id'])
            occupied.append(chosen)
            label_positions.append(dict(label=row['display_id'], xy=list(text.get_position()), ha=text.get_ha()))
        fig.savefig(figures / 'fig2-panele.pdf')
        plt.close(fig)
    for key, value in frozen['distribution'].items():
        np.testing.assert_allclose(distribution[key], value, rtol=0, atol=1e-12)
    data = out / 'data'; data.mkdir(exist_ok=True)
    write_json(data / 'e_patient_rows.json', rows)
    np.savez_compressed(data / 'e_display_null.npz', raw_cohort_median_r=null_r)
    raster(figures / 'fig2-panele.pdf', dpi, 1400)
    return {'renderer': str(FIELD_RENDERER), 'source_metadata': str(FIELD_META),
            'distribution': distribution, 'display_raw_cohort_median_null': raw_summary,
            'significance_display': frozen['significance_display'],
            'label_positions': label_positions, 'patient_label_overlaps': 0,
            'patient_order': [r['subject_id'] for r in sorted(rows, key=lambda r: r['r'])],
            'input_root': str(root), 'frozen_statistics_match': True,
            'axis_alignment': axis_alignment}


def build(source, out, dpi):
    figures = out / 'figures'; figures.mkdir(parents=True, exist_ok=True)
    snap = out / 'source_snapshot'; snap.mkdir(exist_ok=True)
    old = json.loads((source / 'figure2_layout_metadata.json').read_text())
    frozen = {}
    for letter in 'acd':
        for path in (source / 'figures').glob(f'fig2-panel{letter}.*'):
            shutil.copy2(path, figures / path.name)
            frozen[path.name] = sha(path)
    panel_b = build_b(old, figures, dpi)
    panel_e = build_e(figures, out, dpi)
    layout = {
        'a': {'source_clip_pt': [0, 0, 700, 430], 'placed_rect_pt': [LEFT_X, TOP, LEFT_X+LEFT_W, TOP+TOP_H], 'scale': 1.},
        'b': {'source_clip_pt': [0, 0, 360, 430], 'placed_rect_pt': [RIGHT_X, TOP, RIGHT_X+360, TOP+TOP_H], 'scale': 1.},
        'c': {'source_clip_pt': [0, 0, 223, 350], 'placed_rect_pt': [RIGHT_X+360, TOP, RIGHT_X+RIGHT_W, TOP+350], 'scale': 1.},
        'd': {'source_clip_pt': [0, 0, 710, 350], 'placed_rect_pt': [LEFT_X, BOTTOM, LEFT_X+LEFT_W, BOTTOM+BOTTOM_H], 'scale': LEFT_W/710},
        'e': {'source_clip_pt': [0, 0, RIGHT_W, BOTTOM_H], 'placed_rect_pt': [RIGHT_X, BOTTOM, RIGHT_X+RIGHT_W, BOTTOM+BOTTOM_H], 'scale': 1.},
    }
    labels = [('A', LEFT_X-12, 32), ('B', RIGHT_X-12, 32), ('C', RIGHT_X+360, 32),
              ('D', LEFT_X-12, BOTTOM-12), ('E', RIGHT_X-12, BOTTOM-12)]
    doc = pdf.open(); page = doc.new_page(width=CANVAS[0], height=CANVAS[1])
    for letter in 'abcde':
        with pdf.open(figures / f'fig2-panel{letter}.pdf') as part:
            p = layout[letter]
            page.show_pdf_page(pdf.Rect(p['placed_rect_pt']), part, 0, clip=pdf.Rect(p['source_clip_pt']))
    for label, x, y in labels: page.insert_text((x, y), label, fontsize=27, fontname='hebo')
    complete = figures / 'fig2-complete-layout.pdf'; doc.save(complete); doc.close()
    raster(complete, dpi, 1900)
    meta = copy.deepcopy(old)
    meta.update(schema_version='fig2_aligned_columns_v1', producer=str(Path(__file__)), source_candidate=str(source),
        complete_layout=layout, canvas_pt=CANVAS, panel_labels=labels, frozen_panel_exports=frozen,
        panel_b_layout=panel_b, panel_e=panel_e, dpi=dpi, agent_visual_review='PENDING',
        status='PENDING_AUTHOR_VISUAL_REVIEW',
        alignment={'A_D_width_pt': LEFT_W, 'BC_E_width_pt': RIGHT_W, 'column_gap_pt': RIGHT_X-LEFT_X-LEFT_W,
                   'A_B_top_pt': TOP, 'A_B_bottom_pt': TOP+TOP_H,
                   'A_B_axes_top_pt': TOP+34, 'A_B_axes_bottom_pt': TOP+384,
                   'D_E_top_pt': BOTTOM, 'D_E_bottom_pt': BOTTOM+BOTTOM_H})
    write_json(out / 'figure2_layout_metadata.json', plain(meta))
    for path in (Path(__file__), Path(errors.__file__), FIELD_RENDERER): shutil.copy2(path, snap / path.name)
    shutil.copy2(FIELD_META, snap / 'field_statistics_metadata.json')
    shutil.copy2(source / 'figure2_layout_metadata.json', snap / 'previous_layout_metadata.json')
    (figures / 'README.md').write_text(
        '# Figure 2：A/D与B+C/E两列对齐\n\n状态：PENDING_AUTHOR_VISUAL_REVIEW。'
        '左列A/D等宽700 pt，右列B+C/E等宽583 pt；A/B顶底与数据轴上下边缘对齐，D/E顶底对齐。'
        'B与E复用原绘图函数调整轴框，不非等比拉伸文字或圆点；A/C/D原始单图保留。'
        '复现：`PYTHONPATH=/tmp/hfosp_fig5_pdf_tools python scripts/paper_figures/build_fig2_aligned_columns.py --dpi 600`。\n\n'
        '### fig2-complete-layout.pdf / fig2-complete-layout.png\n'
        '完整A–E采用两列对齐布局，画布1355×870 pt。PDF为原生矢量与原MRI图层，PNG直接由同一PDF生成。\n'
        '**关注点**：A/B上下对齐、A/D同宽、B+C与E左右边界一致。\n\n'
        '### fig2-panela.pdf / fig2-panela.png\n原A的序列、解剖投影和空间轨迹保持，单图文件未改。\n**关注点**：与下方D的整体宽度对齐。\n\n'
        '### fig2-panelb.pdf / fig2-panelb.png\n沿用原方向误差函数、26人顺序、null与Holm星号，仅调整纵向轴框以对齐A。测试方向选择最近模板，因此是留出方向匹配／重建。\n**关注点**：患者误差、bootstrap及P值保持。\n\n'
        '### fig2-panelc.pdf / fig2-panelc.png\n两个rose及圆内图例原样保留，圆周间距40 pt。灰柱是第一折方向二维投影，红蓝线为同折空间模板方向，不直接等同B的三维角误差。\n**关注点**：圆形和紧凑间距保持。\n\n'
        '### fig2-paneld.pdf / fig2-paneld.png\n四患者TA/TB场与轨迹原样保留，在完整图中等比缩放至与A同宽。\n**关注点**：颜色、坐标与轨迹未重算。\n\n'
        '### fig2-panele.pdf / fig2-panele.png\n复用当前18人场相关统计的原绘图代码，横向轴框扩展到B+C宽度；患者值、中位数、IQR、原始r空间null与两项星号均与冻结版本核对。患者名称左右避让，点的位置与顺序不变。\n**关注点**：上下两个菱形仍表示同一个原始相关中位数。\n'
    )
    return meta


def validate(out):
    meta = json.loads((out / 'figure2_layout_metadata.json').read_text())
    rects = {k: pdf.Rect(v['placed_rect_pt']) for k, v in meta['complete_layout'].items()}
    assert rects['a'].x0 == rects['d'].x0 and rects['a'].x1 == rects['d'].x1
    assert rects['b'].x0 == rects['e'].x0 and rects['c'].x1 == rects['e'].x1
    assert rects['a'].y0 == rects['b'].y0 and rects['a'].y1 == rects['b'].y1
    assert rects['d'].y0 == rects['e'].y0 and rects['d'].y1 == rects['e'].y1
    for k, v in meta['complete_layout'].items():
        source = pdf.Rect(v['source_clip_pt']); target = rects[k]
        np.testing.assert_allclose(target.width/source.width, target.height/source.height, atol=1e-12)
    for name, value in meta['frozen_panel_exports'].items(): assert sha(out/'figures'/name) == value
    for letter in 'be':
        with pdf.open(out / f'figures/fig2-panel{letter}.pdf') as doc:
            page = doc[0]
            for block in page.get_text('dict', flags=pdf.TEXTFLAGS_TEXT)['blocks']:
                for line in block.get('lines', []):
                    for span in line['spans']: assert page.rect.contains(pdf.Rect(span['bbox'])), (letter, span['text'])
            text = page.get_text()
            if letter == 'b': assert text.count('***') == 3 and 'Patients (ordered)' not in text
            else: assert 'negative: **' in text and '15/18 patients' in text
    report = {'status': 'PASS', 'alignment': meta['alignment'], 'A_C_D_native_exports_unchanged': True,
              'B_patient_values_bootstrap_and_statistics_unchanged': True,
              'E_18_patient_statistics_and_raw_null_match_frozen_source': True,
              'no_nonuniform_scaling': True, 'B_E_text_inside_canvas': True,
              'human_visual_acceptance': 'PENDING_AUTHOR_VISUAL_REVIEW'}
    write_json(out / 'validation.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    build(args.source, args.output, args.dpi)
    print(json.dumps(validate(args.output), indent=2), flush=True)
    print(args.output, flush=True)
