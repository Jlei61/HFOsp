"""Refine Figure 2 spacing, geometry flow, score densities, and two-cohort examples."""
from __future__ import annotations
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT = INPUT_ROOT
pass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
import numpy as np
import pymupdf as pdf
from scipy.ndimage import label
from plotting.vendor.scripts.paper_figures import plot_interictal_spatial_information_gain as scores
from plotting.vendor.scripts.paper_figures.build_fig2_complete_trajectories import compute_pair, plain
from plotting.vendor.scripts.paper_figures.build_fig2_sequence_field_layout import export, place
from plotting.vendor.scripts.paper_figures.plot_fig2_exemplar_trajectory import mpl_path, TEMPLATE_COLORS
from plotting.vendor.scripts.paper_figures.plot_fig3f_ab_dominance_cohort import _pretty
from plotting.vendor.scripts.plot_topic5_interictal_template_ab_fields import build_interictal_ab_panel_payloads, draw_interictal_rank_field_panel
from plotting.vendor.scripts.plot_contact_plane_static import _smooth_rank_field_mm
PREVIOUS = ROOT / 'results/paper-ready-figure/fig2/candidates/complete_spatial_trajectories_20261012'
FIELD_ROOT = ROOT / 'results/interictal_propagation_masked/template_gradient_fields_all_events_timing_plus_space'
OUT = ROOT / 'results/paper-ready-figure/fig2/candidates/refined_layout_two_cohorts_20261012'
EXAMPLES = ('E1', 'E5', 'Y9', 'Y14')
ZOOM_WINDOWS = {'E1': ([-13.0, 13.0], [-11.6, 14.4]), 'E5': ([-34.0, 22.0], [-17.0, 39.0]), 'Y9': ([-13.0, 15.0], [-15.5, 12.5]), 'Y14': ([-10.0, 14.0], [-11.0, 13.0])}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def freeze(out):
    snap = out / 'source_snapshot'
    snap.mkdir(parents=True, exist_ok=True)
    manifest_path = snap / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, entry in manifest.items():
            assert sha(snap / name) == entry['sha256'], name
        return snap
    bmeta = json.loads((PREVIOUS / 'source_snapshot/panelb_source_metadata.json').read_text())
    files = {'original_panelb.pdf': PREVIOUS / 'figures/fig2-panelb.pdf', 'panelb_source_metadata.json': PREVIOUS / 'source_snapshot/panelb_source_metadata.json', 'original_paneld.pdf': PREVIOUS / 'figures/fig2-paneld.pdf', 'original_paneld.png': PREVIOUS / 'figures/fig2-paneld.png', 'paneld_source_metadata.json': PREVIOUS / 'source_snapshot/paneld_source_metadata.json', 'field_cohort_metrics.csv': FIELD_ROOT / 'shared_field_similarity_subjects.csv'}
    for key, dest in [('subject_csv', 'B_scores.csv'), ('summary_json', 'B_summary.json'), ('cohort_null_npz', 'B_null.npz')]:
        path = ROOT / bmeta['source'][key]
        assert sha(path) == bmeta['source'][key + '_sha256']
        files[dest] = path
    for path in (FIELD_ROOT / 'per_subject').glob('*.json'):
        record = json.loads(path.read_text())
        try:
            display_id = _pretty(record['subject_id'])
        except ValueError:
            continue
        if display_id in (*EXAMPLES, 'Y15'):
            files[f'{display_id}_fields.json'] = path
    assert all((f'{name}_fields.json' in files for name in (*EXAMPLES, 'Y15')))
    manifest = {}
    for name, path in files.items():
        shutil.copy2(path, snap / name)
        manifest[name] = dict(source=str(path), sha256=sha(snap / name))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return snap

def support_zoom(record, sigma):
    a, b, mode = build_interictal_ab_panel_payloads(record, display_sigma_mm=sigma)
    assert mode == 'shared'
    points = np.c_[a['xs'], a['ys']]
    low, high = (points.min(0) - 3 * sigma, points.max(0) + 3 * sigma)
    grids = [_smooth_rank_field_mm(*points.T, p['vals'], p['sup'], low[[0, 0]] + [0, high[0] - low[0]], low[[1, 1]] + [0, high[1] - low[1]], sigma) for p in (a, b)]
    X, Y = grids[0][:2]
    mask = grids[0][4] | grids[1][4]
    included = np.c_[X[mask], Y[mask]]
    low, high = (included.min(0) - 1.5, included.max(0) + 1.5)
    middle = (low + high) / 2
    side = float(max(high - low))
    return ([float(middle[0] - side / 2), float(middle[0] + side / 2)], [float(middle[1] - side / 2), float(middle[1] + side / 2)])

def build_c(out, snap, dpi):
    with (snap / 'field_cohort_metrics.csv').open() as f:
        metrics = {r['subject_id']: r for r in csv.DictReader(f)}
    results = []
    width, height = (660.0, 322.0)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.linewidth': 0.7, 'pdf.fonttype': 42}):
        fig = plt.figure(figsize=(width / 72, height / 72), facecolor='white')
        for col, name in enumerate(EXAMPLES):
            record = json.loads((snap / f'{name}_fields.json').read_text())
            sigma = 12.0 if name == 'E5' else 6.0
            limits = ZOOM_WINDOWS[name]
            integration_limits = None if name == 'E5' else support_zoom(record, sigma)
            payloads, paths, meta = compute_pair(name, record, out, sigma_mm=sigma, display_limits=integration_limits)
            meta['integration_xlim_mm'] = meta['xlim_mm']
            meta['integration_ylim_mm'] = meta['ylim_mm']
            meta.update(xlim_mm=limits[0], ylim_mm=limits[1])
            for payload in payloads:
                payload['frame'].update(xlim=limits[0], ylim=limits[1])
            assert all((len(paths[lab]) == 1 for lab in ('TA', 'TB')))
            metric = metrics[record['subject_id']]
            meta.update(frozen_shared_field_r=float(metric['observed_shared_field_r']), channel_p_negative=float(metric['channel_p_negative']), channel_q_bh=float(metric['channel_q_bh']), crop_rule='local_square_zoom_inside_common_display_support; full displayed TA/TB paths retained; all contacts used in calculation', selection_role='author_requested_two_Epilepsiae_two_Yuquan_illustrative_morphology')
            for row, (template, payload) in enumerate(zip(('TA', 'TB'), payloads)):
                left, top, side = (54.0 + 143 * col, 20.0 + 145 * row, 120.0)
                ax = fig.add_axes([left / width, 1 - (top + side) / height, side / width, side / height])
                draw_interictal_rank_field_panel(ax, payload, template, compact=True, contact_size=20, contact_outline_lw=0.75, show_template_tag=False)
                title = name if row == 0 else ''
                ax.set_title(title, fontsize=10.5, fontweight='bold', pad=3)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
                ax.tick_params(axis='both', labelsize=7.5, length=2.5, width=0.6, pad=1.5)
                if col == 0:
                    ax.set_ylabel('y (mm)', fontsize=8.5, labelpad=2)
                    ax.text(-0.36, 0.5, f'{template} trajectory', transform=ax.transAxes, rotation=90, ha='center', va='center', color=TEMPLATE_COLORS[template], fontsize=9, fontweight='bold', clip_on=False)
                mpl_path(ax, paths[template][0], TEMPLATE_COLORS[template], scale=0.85)
                for collection in ax.collections:
                    collection.set_zorder(9)
                p = np.c_[payload['xs'], payload['ys']]
                grid = _smooth_rank_field_mm(*p.T, payload['vals'], payload['sup'], payload['frame']['xlim'], payload['frame']['ylim'], sigma)
                components = label(grid[4])[1]
                assert components == 1
                assert grid[4].all(), 'Local zoom must have no unsupported white margins'
                for xy in paths[template][0]:
                    assert limits[0][0] < xy[0] < limits[0][1] and limits[1][0] < xy[1] < limits[1][1]
                visible = (p[:, 0] > limits[0][0]) & (p[:, 0] < limits[0][1]) & (p[:, 1] > limits[1][0]) & (p[:, 1] < limits[1][1])
                meta.setdefault('visible_contact_names', {})[template] = np.asarray(payload['names'])[visible].tolist()
                meta.setdefault('viewport_support_fraction', {})[template] = float(grid[4].mean())
                meta.setdefault('display_support_components', {})[template] = components
                meta.setdefault('axes_rect_pt', {})[template] = [left, top, left + side, top + side]
            results.append(meta)
        cax = fig.add_axes([620 / width, (height - 285) / height, 6 / width, 265 / height])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='viridis'), cax=cax, ticks=[0, 1])
        cb.ax.set_yticklabels(['early', 'late'])
        cb.ax.tick_params(labelsize=8, length=2, pad=2)
        cb.ax.set_title('rank', fontsize=8.5, pad=4)
        cb.outline.set_linewidth(0.7)
        fig.text(323 / width, 1 - 316 / height, 'Shared TA axis (mm)', ha='center', va='bottom', fontsize=10)
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(out / f'figures/fig2-panelc.{ext}', dpi=dpi, facecolor='white')
        plt.close(fig)
    return results

def build_b(out, snap, dpi):
    rows = scores._load_rows(snap / 'B_scores.csv')
    summary = json.loads((snap / 'B_summary.json').read_text())['cohort_statistics']
    oldmeta = json.loads((snap / 'panelb_source_metadata.json').read_text())
    examples = {r['subject_id']: r['display_label'] for r in oldmeta['rose_examples']}
    with np.load(snap / 'B_null.npz') as cache:
        assert cache['subject_ids'].tolist() == [r['subject_id'] for r in rows]
        null = np.median(cache['patient_null_timing_plus_space_score'], axis=0)
    absolute = summary['absolute_direction_scores']
    np.testing.assert_allclose(np.median(null), absolute['hybrid_direction_shuffle_null_median'], atol=1e-12)
    np.testing.assert_allclose(np.percentile(null, [2.5, 97.5]), absolute['hybrid_direction_shuffle_null_ci95'], atol=1e-12)
    width, height = (520.0, 390.0)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42, 'axes.linewidth': 0.7}):
        fig = plt.figure(figsize=(width / 72, height / 72), facecolor='white')
        specs = {'shared_score_axis': [251, 42, 258, 304]}
        left, top, w, h = specs['shared_score_axis']
        ax = fig.add_axes([left / width, 1 - (top + h) / height, w / width, h / height])
        scores.draw_absolute_scores_and_null(ax, rows, summary, null, example_labels=examples, bootstrap_in_source_order=True)

        def expand_summary(y):
            y = np.asarray(y)
            return np.where(y <= 2.4, (y + 1.65) * 114 / 4.05, 114 + (y - 2.4) * 190 / 27.6)

        def inverse_summary(y):
            y = np.asarray(y)
            return np.where(y <= 114, y * 4.05 / 114 - 1.65, (y - 114) * 27.6 / 190 + 2.4)
        ax.set_yscale('function', functions=(expand_summary, inverse_summary))
        ax.set_ylim(-1.65, len(rows) + 4.0)
        ax.set_yticks([])
        for artist in list(ax.texts):
            text = artist.get_text()
            if text in ['Direction-\nshuffle null', 'Timing-only', 'Timing + 3D\ndirection']:
                artist.remove()
            elif 'higher with' in text:
                artist.set_position((0.025, 0.975))
                artist.set_fontsize(8)
        ax.set_xlim(-0.1, 0.98)
        ax.tick_params(axis='x', labelsize=8, length=2.5, pad=2)
        ax.set_ylabel('Patients (ordered)', fontsize=10, labelpad=7)
        ax.yaxis.set_label_coords(-0.06, 209 / 304)
        ax.set_xlabel('Held-out direction score', fontsize=10.5, labelpad=9)
        ax.legend(handles=[Line2D([], [], marker='o', ms=4, mfc='white', mec=scores.TEMPORAL_COLOR, lw=0, label='Timing only'), Line2D([], [], marker='o', ms=4, mfc=scores.HYBRID_COLOR, mec='white', lw=0, label='Timing + 3D'), Patch(facecolor='#BDBDBD', edgecolor='#777777', label='Direction-shuffle null')], loc='lower left', bbox_to_anchor=(-0.04, 1.035), ncol=2, frameon=False, fontsize=7.8, handlelength=1.25, columnspacing=1.2, labelspacing=0.5, borderaxespad=0)
        fig.legend(handles=[Patch(facecolor=(0.55, 0.55, 0.55, 0.24), edgecolor='#6F6F6F'), Line2D([], [], color='#202020', lw=1.5, ls=(0, (3, 2))), (Line2D([], [], color=TEMPLATE_COLORS['TA'], lw=1.7), Line2D([], [], color=TEMPLATE_COLORS['TB'], lw=1.7)), Line2D([], [], color='#202020', lw=2)], labels=['Event directions', 'Timing-only axis', 'Template A / B', 'Timing + 3D axis'], loc='upper center', bbox_to_anchor=(0.225, 0.994), ncol=2, frameon=False, fontsize=7.8, handlelength=1.35, columnspacing=1.05, labelspacing=0.55, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})
        fig.savefig(snap / 'B-right-native.pdf', facecolor='white')
        plt.close(fig)
    with pdf.open(snap / 'B-right-native.pdf') as result, pdf.open(snap / 'original_panelb.pdf') as original:
        page = result[0]
        rose_layout = []
        for i, (y0, y1, top, bottom) in enumerate([(56, 178, 54, 184), (196, 318, 220, 350)]):
            info = place(page, original, (13, top, 231, bottom), (30, y0, 194, y1))
            rose_layout.append(info)
            center = info['placed_rect_pt'][0] + (114.0093 - 30) * info['scale']
            for text, baseline in [('90°', top - 7), ('270°', bottom + 13)]:
                fontsize = 8.0
                tw = pdf.get_text_length(text, fontname='helv', fontsize=fontsize)
                page.insert_text((center - tw / 2, baseline), text, fontname='helv', fontsize=fontsize, color=(0.12, 0.12, 0.12))
        export(result, out / 'figures/fig2-panelb', dpi)
    return dict(source_metadata_sha256=sha(snap / 'panelb_source_metadata.json'), n_patients=len(rows), cohort_statistics=summary, rose_source_reused=True, rose_layout=rose_layout, axes_bounds_pt=specs, distribution_reserved_height_pt=114, common_xlim=[-0.1, 0.98], bootstrap_and_null_calculation='original analysis CSV order, 10000 draws; exact stored CI verified by draw_absolute_scores_and_null', null_reconstruction='median across the same 26 frozen patient-null arrays, per draw', shared_axis_alignment=True, n_score_axes=1, separate_distribution_axis=False, legend_locations='separate reserved top area of each B column')

def write_readme(out):
    (out / 'figures/README.md').write_text('# Figure 2：紧凑布局与双队列轨迹候选\n\n### fig2-complete-layout.png / .pdf / .svg\n\n完整 A–D 版式，上下两排缩短 45 pt；A 改投影连线和短转换箭头，B 放大分布并对齐横轴，C 为 E1/E5/Y9/Y14，D 沿用原图。全图 10 张二维 rank 场各有一条 early→late 轨迹，TA 深红、TB 蓝色并带白色外沿。**关注点**：本版供作者目视检查，尚未替换正式版本。\n\n### fig2-panela.png / .pdf / .svg\n\n投影虚线连续连接 MRI 中的切面与下方投影框上沿；转换箭头缩短并移到虚线右侧。MRI 复用作者选定的 az −110°、elev 28°、SP 沿 R 移 12 mm 版本，序列、场和轨迹数值保持一致。**关注点**：虚线不被箭头遮挡，左右两列同尺寸、底部横轴标题对齐。\n\n### fig2-panelb.png / .pdf / .svg\n\n患者配对得分与底部队列中位数分布画在同一个坐标轴中，给分布预留更大的纵向空间，只有一条底部横轴，图例置于两列顶部预留区。玫瑰图复用冻结图层；按原分析 CSV 顺序和随机种子恢复 10000 次患者自助抽样，与保存的置信区间一致，方向打乱零分布和所有检验不变。**关注点**：灰色为方向打乱零分布，蓝色和陶红色为两种方法的患者自助中位数分布；没有单独的分布纵轴。\n\n### fig2-panelc.png / .pdf / .svg\n\nEpilepsiae 选择 E1、E5，Yuquan 选择 Y9、Y14；各患者 TA/TB 共用等比例毫米坐标的局部放大范围，彩色场铺满轴框，完整轨迹均在视野内，标题仅显示患者编号。局部放大裁去 E5 和 Y9 的部分远端触点，但所有触点继续参与场与轨迹计算，支持掩膜没有放宽；E5 显示 σ 为 12 mm，其余为 6 mm，仅在计算记录说明。**关注点**：E5 的显示场与轨迹随 σ 改变，原 rank、支持权重、拟合轴、分析核和队列统计不变；曲线是插值摘要而非实测组织路径，病例按形态展示选取，Y14 不作为患者级显著性主张。\n\n### fig2-paneld.png / .pdf / .svg\n\n保留原 18 名患者共享场相关、15/18 负相关及空间零模型。PNG/PDF 与冻结来源逐字节一致。**关注点**：C 的病例更换及显示平滑不用于重算或替换 D 的统计。\n\n### fig2-complete-layout-preview.png\n\n完整 PDF 的 1800 像素宽预览，与高分辨率导出对应。**关注点**：检查两排间距、投影虚线、B 分布与 C 各箭头方向的可读性。\n')

def build(out, dpi=600):
    out.mkdir(parents=True, exist_ok=True)
    for directory in ('figures', 'trajectories'):
        (out / directory).mkdir(exist_ok=True)
    snap = freeze(out)
    aout = out / 'panela_revision'
    subprocess.run([sys.executable, str(ROOT / 'scripts/paper_figures/build_fig2_rank_sequence_labels.py'), '--compact-geometry-flow', '--output', str(aout), '--anatomy-source', str(ROOT / 'results/paper-ready-figure/fig2/candidates/balanced_mri_view_20261012')], check=True, stdout=subprocess.DEVNULL)
    for ext in ('png', 'pdf', 'svg'):
        shutil.copy2(aout / f'figures/fig2-panela-trajectories.{ext}', out / f'figures/fig2-panela.{ext}')
    bmeta = build_b(out, snap, dpi)
    cmeta = build_c(out, snap, dpi)
    for ext in ('png', 'pdf'):
        shutil.copy2(snap / f'original_paneld.{ext}', out / f'figures/fig2-paneld.{ext}')
    with pdf.open(snap / 'original_paneld.pdf') as doc:
        (out / 'figures/fig2-paneld.svg').write_text(doc[0].get_svg_image(text_as_path=True))
    complete = pdf.open()
    page = complete.new_page(width=1240, height=825)
    docs = {k: pdf.open(out / f'figures/fig2-panel{k}.pdf') for k in 'abcd'}
    layout = {'a': place(page, docs['a'], (28, 52, 680, 438), docs['a'][0].rect), 'b': place(page, docs['b'], (710, 52, 1221, 438), docs['b'][0].rect), 'c': place(page, docs['c'], (25, 445, 811, 807), docs['c'][0].rect), 'd': place(page, docs['d'], (839, 453, 1216, 804))}
    page.insert_font(fontname='FigureTwoPanelBold', fontfile=str(INPUT_ROOT / 'fonts' / 'DejaVuSans-Bold.ttf'))
    for letter, x, y in [('A', 12, 33), ('B', 695, 33), ('C', 12, 429), ('D', 823, 429)]:
        page.insert_text((x, y), letter, fontname='FigureTwoPanelBold', fontsize=24, color=(0, 0, 0))
    export(complete, out / 'figures/fig2-complete-layout', dpi)
    page.get_pixmap(matrix=pdf.Matrix(1800 / 1240, 1800 / 1240), alpha=False).save(out / 'figures/fig2-complete-layout-preview.png')
    for doc in [complete, *docs.values()]:
        doc.close()
    metadata = dict(schema_version='fig2_refined_layout_two_cohorts_v1', human_visual_acceptance='PENDING', producer=str(Path(__file__).relative_to(CODE_ROOT)), previous_candidate=str(PREVIOUS), examples=list(EXAMPLES), n_Epilepsiae=2, n_Yuquan=2, complete_layout=layout, canvas_pt=[1240, 825], lower_row_shift_up_pt=45, panel_b=bmeta, panel_c=cmeta, display_bandwidth_mm={'E1': 6, 'E5': 12, 'Y9': 6, 'Y14': 6}, frozen_scientific_analysis='Original ranks, support, axes, analysis kernels and all cohort statistics unchanged; E5 display smoothing and the derived display trajectory change together.', example_selection='Author requested E1/E5 plus Y9 and either Y14/Y15; Y14 selected for clearer opposite local directions in the same supported shared plane. Examples are descriptive, not selected patient-level significance claims.', reviewed_alternative='Y15 is geometry eligible; comparison retained in source_snapshot. Its local trajectories are less simply opposed.', claim_boundary='Rank-gradient display streamlines are interpolation summaries, not measured tissue tracks or a unique physical path.', validation=dict(agent_png_review='PENDING', agent_pdf_review='PENDING', canonical_assets_replaced=False))
    (out / 'figure2_layout_metadata.json').write_text(json.dumps(metadata, default=plain, ensure_ascii=False, indent=2) + '\n')
    for path in [Path(__file__), ROOT / 'scripts/paper_figures/build_fig2_complete_trajectories.py', ROOT / 'scripts/paper_figures/plot_fig2_exemplar_trajectory.py', ROOT / 'scripts/paper_figures/build_fig2_rank_sequence_labels.py', ROOT / 'scripts/paper_figures/plot_interictal_spatial_information_gain.py', ROOT / 'src/topic5_field_trajectory.py']:
        shutil.copy2(path, snap / path.name)
    write_readme(out)
    return metadata
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    result = build(args.output.resolve(), args.dpi)
    print(json.dumps(dict(output=str(args.output), examples=result['examples'], display_sigma=result['display_bandwidth_mm']), indent=2))
