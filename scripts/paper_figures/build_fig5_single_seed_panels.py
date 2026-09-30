#!/usr/bin/env python3
"""Export the accepted Fig5 painter into a frozen composite and native panels."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import psutil
import plot_topic4_fig5_clean_panels as painter
import analyze_topic4_fig5_boundary_refinement as refinement


def write(path, data):
    path.write_text(json.dumps(painter.f.safe(data), ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(fig, groups, metadata, destination):
    figures = destination / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
    records = {}

    def save(stem, bbox):
        paths = []
        for extension in ('png', 'pdf', 'svg'):
            path = figures / f'{stem}.{extension}'
            fig.savefig(path, dpi=300, bbox_inches=bbox, pad_inches=.12, facecolor='white')
            paths.append(dict(path=str(path.relative_to(destination)), sha256=sha(path), bytes=path.stat().st_size))
        return paths

    records['complete'] = dict(files=save('fig5-complete-layout', 'tight'), panel_letters=True)
    visibility = [(artist, artist.get_visible()) for artist in list(fig.axes) + list(fig.texts) + list(fig.artists)]
    titles = [(ax, ax.get_title(loc='left')) for ax in fig.axes]
    raster = groups['A'][0]
    original_xlabel = raster.get_xlabel()
    original_tick_labels = [tick.label1.get_visible() for tick in raster.xaxis.get_major_ticks()]
    try:
        for letter, axes in groups.items():
            for artist, _ in visibility:
                artist.set_visible(artist in axes)
            for ax, title in titles:
                ax.set_title('' if title in tuple('ABCDEF') else title, loc='left')
            if letter == 'A':
                # A shared its main time axis with B in the composite.
                raster.tick_params(axis='x', labelbottom=True)
                raster.set_xlabel('Time (s)')
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            included = axes + [child for ax in axes for child in ax.child_axes if child.get_visible()]
            bounds = [ax.get_tightbbox(renderer) for ax in included]
            # Axes3D.get_tightbbox omits projected axis-label extents.
            # Include them explicitly so long oblique labels remain complete.
            for ax in axes:
                if hasattr(ax, 'zaxis'):
                    bounds.extend(axis.label.get_window_extent(renderer)
                                  for axis in (ax.xaxis, ax.yaxis, ax.zaxis))
            tight = Bbox.union([b for b in bounds if b is not None]).transformed(fig.dpi_scale_trans.inverted())
            padding = .14
            bbox = Bbox.from_extents(tight.x0-padding, tight.y0-padding, tight.x1+padding, tight.y1+padding)
            records[letter] = dict(files=save(f'fig5-panel{letter.lower()}', bbox),
                panel_letters=False, axes=len(axes), child_axes=len(included)-len(axes),
                bbox_inches=bbox.bounds, native_artist_export=True, cropped_bitmap=False,
                standalone_time_axis_added=letter == 'A')
            raster.set_xlabel(original_xlabel)
            for tick, visible in zip(raster.xaxis.get_major_ticks(), original_tick_labels):
                tick.label1.set_visible(visible)
    finally:
        for artist, visible in visibility:
            artist.set_visible(visible)
        for ax, title in titles:
            ax.set_title(title, loc='left')
        raster.set_xlabel(original_xlabel)
        for tick, visible in zip(raster.xaxis.get_major_ticks(), original_tick_labels):
            tick.label1.set_visible(visible)
    assert metadata['E_grid_layout']['colorbar_limits'] == [1, 1000]
    assert metadata['E_grid_layout']['colorbar_ticks'] == [1, 10, 100, 1000]
    assert metadata['E_grid_layout']['colormap'] == 'viridis'
    assert metadata['E_grid_layout']['square']
    e_axis = groups['E'][0]
    metadata['E_display_artists'] = dict(
        collections=[type(artist).__name__ for artist in e_axis.collections],
        lines=len(e_axis.lines), legend=e_axis.get_legend() is not None)
    if not metadata['F_grid'].get('display_overlays', True):
        assert metadata['E_display_artists'] == dict(collections=['QuadMesh'], lines=0, legend=False)
    records['E']['display_artists'] = metadata['E_display_artists']
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--continuous-boundary', action='store_true')
    parser.add_argument('--surface-only', action='store_true',
                        help='Hide Fig5E sampling points, boundary, hatching, working point and legend.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.snapshot:
        snapshot = json.loads(args.snapshot.read_text())
    else:
        if args.continuous_boundary:
            import plot_topic4_fig5_continuous_boundary as continuous
            grid = continuous.collect()
        else:
            grid = refinement.collect()
        status = json.loads((refinement.OUT / 'status.json').read_text())
        live = {}
        for name, record in status['running'].items():
            try:
                process = psutil.Process(record['pid'])
                live[name] = dict(pid=process.pid, verified=name in process.cmdline() and 'worker' in process.cmdline())
            except psutil.NoSuchProcess:
                live[name] = dict(pid=record['pid'], verified=False)
        snapshot = dict(grid=painter.f.safe(grid), simulation_status=status, live_processes=live,
                        frozen_at_unix_s=time.time(), seed=9108401,
                        source_figure=str(painter.audit.OUT / 'clean_panels_v6_boundary_20260917/eta0.0005_s9108401/figures/fig5.png'))
    if args.surface_only:
        import plot_topic4_fig5_continuous_boundary as continuous
        continuous.surface_only(snapshot['grid'])
    snapshot_path = args.output / 'source_snapshot.json'
    write(snapshot_path, snapshot)
    grid = snapshot['grid']
    for key in ('value', 'entered', 'followup', 'stages'):
        grid['base_grid'][key] = np.asarray(grid['base_grid'][key])
    rows = [row for row in painter.audit.sources() if row['eta_m'] == .0005 and row['seed'] == 9108401]
    assert len(rows) == 1
    exported = {}
    # Keep the recomputed clinical/model comparison beside this export,
    # including when native inputs are read from a retained external workspace.
    import analyze_topic4_fig5_onset_z_field as onset_field
    onset_field.OUT = args.output / 'analysis'
    def hook(fig, groups, metadata):
        exported.update(export(fig, groups, metadata, args.output))
    metadata = painter.render(rows[0], grid, export_hook=hook, save_default=False)
    metadata['E_display_artists'] = exported['E']['display_artists']
    metadata.update(paper_slot='Fig5-A–F', asset_id='single_seed_zm_entry_and_onset_field',
                    publication_status='CANDIDATE', author_selected_canonical=True,
                    author_selection='2026-09-18: 这个图作为新的paper-ready的fig5，给我单panel的图',
                    frozen_snapshot=True, automatically_updated=False,
                    export_producer=str(Path(__file__).resolve()), export_producer_sha256=sha(Path(__file__)),
                    source_snapshot_sha256=sha(snapshot_path))
    write(args.output / 'fig5_metadata.json', metadata)
    write(args.output / 'panel_manifest.json', dict(asset_id=metadata['asset_id'], files=exported,
          source_snapshot_sha256=sha(snapshot_path), simulation_status=snapshot['simulation_status'],
          native_matplotlib_export=True, png_dpi=300, svg_text_editable=True,
          human_review='PENDING', agent_visual_review='PENDING'))
    descriptions = {
        'A': '固定80个神经元的原始放电栅格，以及状态2/3的300 ms放大窗。Core A/B及zoom仅画E细胞，I样本单独在上方；独立图补回主时间轴Time (s)。',
        'B': '同一次模型轨迹的资源Z及有效ηM×M；All E、Core A与Core B共享时间轴。实线/虚线及右上竖排图例沿用整图。',
        'C': '四个原生50 ms空间发放率快照：Rest、Interictal、Pre-ictal、Onset。保留1–4、小标题、坐标及第四幅右侧竖直E rate色条。',
        'D': '同一模型轨迹在平均Z、施加的抑制H_E和E发放率空间中的实际演化。色条表示真实时间，1–4对应C中的状态时刻。',
        'E': painter.grid_caption(grid),
        'F': '模型首次高态起点后1秒的1–150 Hz log-power robust-z场，与E10 | SZ4早期发作场。调用Fig3C右图原生画法、同一冻结投影及真实有符号robust-z色条；患者按该患者25次合格发作中场相关最高选择。',
    }
    cautions = {
        'A': '放电及放大窗来自同一次轨迹，状态名是该模型示例的操作性标记。',
        'B': 'ηM×M右轴保持0–0.5 mV equiv.，不改变仿真状态。',
        'C': '模型Onset是操作性高态进入，不自动等同于临床发作起始。',
        'D': '这是原生随机SNN的有限时长轨迹，不是独立验证的低维动力系统。',
        'E': ('固定单种子9108401，59点均已到终点；连续插值不是概率或严格分岔，未进入点的1000秒为下界。' if grid.get('display_mode') else '固定单种子9108401；空心点或随访下界不能当作完整1000秒终点，colorbar保持log 1–1000秒。'),
        'F': '模型基线0.5–3.5秒仅5个重叠PSD窗；患者基线−120至−90秒、早期窗0–10秒。该短窗适配及最大相似病例不构成时间尺度等价或独立临床验证。',
    }
    if grid.get('long_followup'):
        cautions['E'] = '70个唯一参数点、固定单种子；顶端颜色包含晚进入及不同随访时限的未进入。长时结果见图注与逐点表，不把未延长点当作3000秒未进入。'
    readme = []
    for letter in 'ABCDEF':
        readme.append(f'### fig5-panel{letter.lower()}.png / .pdf / .svg\n{descriptions[letter]}\n**关注点**：{cautions[letter]}\n')
    readme.append('### fig5-complete-layout.png / .pdf / .svg\n当前作者指定的完整A–F Fig5，保留panel字母；独立panel均去角标。整图和单panel来自同一冻结数据快照，PNG为300 dpi，PDF/SVG由Matplotlib原生导出。\n**关注点**：本次导出待作者目视检查；E的完整扫描状态见上文及同级metadata。\n')
    (args.output / 'figures/README.md').write_text('\n'.join(readme))
    p = grid['progress_summary']
    scan_state = ('全部既定轨迹已结束；本次无新增仿真，paper-ready保存本次冻结快照。' if grid['all_complete'] else '后台继续原24点任务，paper-ready不会自动变动。')
    scan_intro = f'本次冻结E的原35点完整结果及24点加密进度：{p["new_complete"]}/24完成，{p["new_observed"]}进入，{p["new_censored"]}在1000秒未进入。固定seed9108401，colorbar仍为viridis、log 1–1000秒。'+scan_state
    if grid.get('long_followup'):
        scan_intro = '2026-09-28纳入已完成的19条长时补充（8条续跑、11个新参数组合），E更新为70个唯一参数点的冻结连续色面。所有既定任务已结束；colorbar仍为viridis、log 1–1000秒。'
    (args.output / 'README.md').write_text(
        '# Figure 5：单种子Z/M进入与早期能量场\n\n'
        '2026-09-18按作者指示设为新的paper-ready Fig5；替换旧seed1801 A–D版本。独立A–F不含左上角字母，完整拼版保留字母；见[文件与说明](figures/README.md)。\n\n'
        +scan_intro+'\n\n'+painter.grid_caption(grid)+'\n\n'
        'F保留Fig3C算法和真实robust-z色条；模型采用0.5–3.5秒短基线与1秒早期窗，患者为30秒远端基线与0–10秒窗。模型短基线不确立模型/临床时间尺度等价，最近似患者病例只是示例。\n\n'
        '来源和可重建快照：`source_snapshot.json`、`fig5_metadata.json`、`panel_manifest.json`。作者已指定本图为新的Fig5，当前单panel导出仍待人工目视检查，科学状态维持CANDIDATE。\n\n'
        '重建（写入另一个目录）：\n\n```bash\n'
        'LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python scripts/paper_figures/build_fig5_single_seed_panels.py --snapshot results/paper-ready-figure/fig5/source_snapshot.json --output /tmp/fig5-rebuild\n```\n')
    print(json.dumps(dict(output=str(args.output), panels=list(exported), progress=p), ensure_ascii=False))


if __name__ == '__main__':
    main()
