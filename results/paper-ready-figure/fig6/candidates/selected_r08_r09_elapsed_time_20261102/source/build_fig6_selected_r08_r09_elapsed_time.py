#!/usr/bin/env python3
"""Restore continuous simulation-time colors throughout Figure 6B."""
import json
from pathlib import Path
import shutil

import build_fig6_selected_r08_r09_clean_trajectory as clean

base, np, plt = clean.base, clean.np, clean.plt
trajectory = clean.trajectory
SOURCE = clean.OUT.with_name('selected_r08_r09_black_interictal_20261102')
OUT = SOURCE.with_name('selected_r08_r09_elapsed_time_20261102')
NATIVE_EXPORT = trajectory.NATIVE_EXPORT
HORIZON = 120.
RECORDS = {}


def export_time(fig, stem):
    axes = [ax for ax in fig.axes if getattr(ax, 'name', '') == '3d']
    counts = []
    with np.load(SOURCE / 'trajectory_data.npz') as data:
        time = data['time_s']
    for ax in axes:
        curve = next(c for c in ax.collections if isinstance(c, base.Line3DCollection))
        values = np.asarray(curve.get_array())
        selected = time[time < HORIZON] if len(values) > 1000 else time[(time >= 5.90) & (time < 9.24)]
        np.testing.assert_array_equal(values, (selected[:-1] + selected[1:]) / 2)
        assert curve.norm.vmin == 0 and curve.norm.vmax == HORIZON
        assert curve.cmap.name == 'viridis'
        counts.append(len(values))
    RECORDS[stem] = dict(segment_counts=counts, color_quantity='simulation time',
                         color_units='s', color_limits=[0, HORIZON], manual_black_override=False)
    if stem == 'fig6-panelb':
        main, inset, cbax = fig.axes
        assert counts == [23999, 667]
        assert main.collections[0].norm is inset.collections[0].norm is cbax._colorbar.norm
        assert cbax.get_title() == 'Time (s)'
        NATIVE_EXPORT(fig, stem)
        fig.delaxes(inset)
        for item in list(fig.texts):
            if item.get_text() == 'Interictal':
                item.remove()
        NATIVE_EXPORT(fig, 'fig6-panelb-full-trajectory')
    else:
        assert stem == 'fig6-panelb-interictal' and counts == [667]
        NATIVE_EXPORT(fig, stem)


def main():
    pointer = base.MAIN / 'results/paper-ready-figure/fig6/current_version.json'
    frozen = pointer.read_bytes()
    shutil.copytree(SOURCE, OUT, dirs_exist_ok=True, ignore=shutil.ignore_patterns(
        'validation.json', 'visual_review.json', 'cycle_color_definition.json',
        'cycle_color_values.npz', 'interictal_color_definition.json'))
    meta = json.loads((SOURCE / 'figure_metadata.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'path.simplify': False})
    trajectory.recovery.HORIZON = HORIZON
    trajectory.SOURCE, trajectory.OUT, clean.OUT = SOURCE, OUT, OUT
    trajectory.export, trajectory.NATIVE_EXPORT = clean.export_clean, export_time
    try:
        display = trajectory.draw_b(meta)
    finally:
        trajectory.NATIVE_EXPORT = NATIVE_EXPORT
    display.update(clean.DISPLAY)
    definition = dict(quantity='elapsed simulation time', units='s', range=[0, HORIZON],
        colormap='viridis', segment_value='Mean of the two native sample times bounding each segment.',
        cycle_reset=False, manual_color_override=False,
        scope='Main trajectory, interictal inset, direction arrows and standalone trajectory exports.')
    display.update(color_definition=definition, display_time_s=[0, HORIZON])
    for key in ['inset', 'standalone_interictal']:
        display[key].update(color_quantity='elapsed simulation time', color_limits=[0, HORIZON],
                            color_definition=definition)
    display['standalone_interictal'].update(grid=False, background='transparent panes')
    meta['B_display'] = display
    meta['trajectory'].update(colorbar_quantity='elapsed simulation time', colorbar_units='s',
                              color_definition=definition, display_revision=display)
    meta['trajectory']['outputs'].update({'fig6-panelb': display,
        'fig6-panelb-interictal': display['standalone_interictal'],
        'fig6-panelb-full-trajectory': dict(segment_count=23999, time_s=[0, HORIZON],
            title=None, grid=False, background='transparent panes', colorbar_title='Time (s)',
            color_definition=definition)})
    meta.update(producer=str(Path(__file__).resolve()), previous_candidate=str(SOURCE),
        revision_scope='Restore simulation-time coloring everywhere in B; remove cycle normalization and black overrides.',
        status='PENDING_AUTHOR_VISUAL_REVIEW')
    base.write(OUT / 'figure_metadata.json', meta)
    base.write(OUT / 'interictal_display_selection.json', display['inset'])
    base.write(OUT / 'time_color_definition.json', definition)
    compositor = clean.previous.compositor
    compositor.OUT, compositor.SOURCE = OUT, SOURCE
    compositor.C_DISPLAY.update(meta['C_display'])
    compositor.compose()
    copied = []
    for stem in ['fig6-panela', 'fig6-panelc', 'fig6-paneld', 'fig6-panele']:
        for ext in ['png', 'pdf', 'svg']:
            name = f'{stem}.{ext}'
            assert (OUT / 'figures' / name).read_bytes() == (SOURCE / 'figures' / name).read_bytes()
            copied.append(name)
    names = ['trajectory_data.npz', 'stage_readouts.npz', 'normalized_propagation_fields.npz',
             'model_selected_patient_arrays.npz', 'cohort_statistics_selected.json']
    for name in names:
        assert (OUT / name).read_bytes() == (SOURCE / name).read_bytes()
    checks = {}
    for stem in ['fig6-panelb', 'fig6-panelb-interictal', 'fig6-panelb-full-trajectory', 'fig6-complete-layout']:
        with base.fitz.open(OUT / 'figures' / f'{stem}.pdf') as doc:
            checks[stem] = all(page.rect.contains(base.fitz.Rect(s['bbox'])) for page in doc
                for b in page.get_text('dict')['blocks'] if b['type'] == 0
                for line in b['lines'] for s in line['spans'])
            assert 'Cycle progress' not in doc[0].get_text()
    assert all(checks.values()) and pointer.read_bytes() == frozen
    base.write(OUT / 'validation.json', dict(unchanged_panel_files=copied, unchanged_data_files=names,
        absolute_sample_times_verified=RECORDS, PDF_text_inside_canvas=checks,
        formal_pointer_unchanged=True, human_visual_acceptance='PENDING'))
    caption = (clean.SOURCE / 'caption_zh.md').read_text()
    paragraphs = caption.split('\n\n')
    paragraphs = [p for p in paragraphs if not p.startswith('本次仅调整 B')]
    for i, p in enumerate(paragraphs):
        if p.startswith('B：'):
            paragraphs[i] = ('B：显示 0–120 秒原生轨迹及 Interictal 小窗，1–3 在小窗、4 在主轨迹。'
                '所有轨迹线段按原生仿真时间连续着色，主图和小窗共享 0–120 秒色条，顶部标 Time (s)。'
                '不按发作周期重置或归一化，也不单独覆盖间期颜色；箭头沿用相同时间配色。'
                '主图包括三次进入和前两次完整恢复，第三次终止未在记录内观察到。')
    (OUT / 'caption_zh.md').write_text('\n\n'.join(paragraphs))
    readme = (SOURCE / 'figures/README.md').read_text()
    for stem, desc in [
        ('fig6-panelb', '主图及 Interictal 小窗均恢复按 0–120 秒仿真时间连续着色，色条顶部标 Time (s)。布局、坐标、阶段标记与数据保持。'),
        ('fig6-panelb-interictal', '单独展示相同间期至进入轨迹，所有线段及方向箭头使用仿真时间配色。与主图共用 0–120 秒时间范围。'),
        ('fig6-panelb-full-trajectory', '独立展示同一 0–120 秒全局轨迹，不叠加间期小窗。色条表示连续仿真时间。')]:
        start = readme.index('### ' + stem + '.png')
        end = readme.find('\n### ', start + 1)
        if end < 0:
            end = len(readme)
        readme = readme[:start] + f'### {stem}.png / .pdf / .svg\n\n{desc}\n**关注点**：颜色对应绝对仿真时间，不按发作周期重置。\n' + readme[end:]
    (OUT / 'figures/README.md').write_text(readme)
    (OUT / 'README.md').write_text('# Figure 6：恢复仿真时间着色\n\n'
        '[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf) · [B](figures/fig6-panelb.png)\n\n'
        'B 主图、小窗及箭头统一按原生仿真时间着色，色条为 Time (s)，范围 0–120 秒。'
        '保留现有完整记录、布局及其他面板，不再使用周期归一化或黑色覆盖。\n\n'
        f'复现：`python {Path(__file__).resolve()}`。\n')
    shutil.copy2(__file__, OUT / 'source' / Path(__file__).name)
    print(json.dumps(dict(output=str(OUT), records=RECORDS, checks=checks), indent=2))


if __name__ == '__main__':
    main()
