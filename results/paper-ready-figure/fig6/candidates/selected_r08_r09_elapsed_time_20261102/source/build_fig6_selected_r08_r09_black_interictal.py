#!/usr/bin/env python3
"""Separate the inset's interictal trajectory from its cycle-colored entry."""
import json
from pathlib import Path
import shutil

import build_fig6_selected_r08_r09_cycle_colors as previous

base, np, plt = previous.base, previous.np, previous.plt
SOURCE = previous.OUT
OUT = SOURCE.with_name('selected_r08_r09_black_interictal_20261102')
NATIVE_EXPORT = previous.NATIVE_EXPORT
RECORDS = {}


def export_black_interictal(fig, stem):
    if stem in ['fig6-panelb', 'fig6-panelb-interictal']:
        ax = fig.axes[1] if stem == 'fig6-panelb' else fig.axes[0]
        detail = previous.META['B_display']['inset']
        boundary = detail['entry_highlight_start_s']
        with np.load(SOURCE / 'trajectory_data.npz') as data:
            times = data['time_s']
            times = times[(times >= detail['time_s'][0]) & (times < detail['time_s'][1])]
        midpoint = (times[:-1] + times[1:]) / 2
        black = midpoint < boundary
        curve = next(c for c in ax.collections if isinstance(c, base.Line3DCollection))
        phase = np.asarray(curve.get_array()).copy()
        np.testing.assert_array_equal(phase, previous.progress(midpoint)[0])
        rgba = plt.colormaps['viridis'](phase)
        rgba[black] = [0, 0, 0, 1]
        curve.set_array(None)
        curve.set_color(rgba)
        arrows = [item.arrow_patch for item in ax.texts
                  if item.get_text() == '' and getattr(item, 'arrow_patch', None) is not None]
        black_arrows = 0
        for arrow, record in zip(arrows, detail['direction_arrows']):
            if (record['from_time_s'] + record['to_time_s']) / 2 < boundary:
                arrow.set_color('black')
                black_arrows += 1
        assert len(arrows) == 4 and black_arrows == 3
        assert len(midpoint) == detail['segment_count'] == 667
        np.testing.assert_array_equal(curve.get_colors()[black], rgba[black])
        np.testing.assert_array_equal(curve.get_colors()[~black], plt.colormaps['viridis'](phase[~black]))
        RECORDS[stem] = dict(segment_count=len(midpoint), black_segments=int(black.sum()),
            cycle_colored_entry_segments=int((~black).sum()), black_direction_arrows=black_arrows,
            cycle_colored_direction_arrows=1, display_boundary_s=boundary,
            geometry_and_numbered_markers_unchanged=True)
    NATIVE_EXPORT(fig, stem)


def main():
    previous.SOURCE, previous.OUT = SOURCE, OUT
    previous.NATIVE_EXPORT = export_black_interictal
    try:
        previous.main()
    finally:
        previous.NATIVE_EXPORT = NATIVE_EXPORT
        previous.trajectory.NATIVE_EXPORT = NATIVE_EXPORT
    meta = json.loads((OUT / 'figure_metadata.json').read_text())
    display = meta['B_display']
    definition = dict(
        boundary_s=display['inset']['entry_highlight_start_s'],
        boundary_definition='Existing A/C Local state 1; display boundary, not a newly detected onset.',
        before_boundary='Opaque black trajectory and direction arrows in the inset only.',
        from_boundary='Existing viridis cycle progress colors; shared with the main trajectory.',
        numbered_markers='Original 1–3 stage colors and positions retained.',
        colorbar_scope='Main trajectory and colored entry segment only; black is not a colorbar value.')
    for key in ['inset', 'standalone_interictal']:
        display[key].update(color_quantity='black before state 1; cycle progress from state 1',
                            color_definition=definition)
        display[key]['selection'] = ('Original continuous interval, with no reordering or coordinate changes; '
                                     'black before the existing Local state 1, then cycle-colored entry.')
    display['inset_color_definition'] = definition
    meta['trajectory']['display_revision'] = display
    meta['trajectory']['outputs']['fig6-panelb'] = display
    meta['trajectory']['outputs']['fig6-panelb-interictal'] = display['standalone_interictal']
    meta.update(producer=str(Path(__file__).resolve()), previous_candidate=str(SOURCE),
                revision_scope='Only the inset trajectory and arrows before state 1 become black.',
                status='PENDING_AUTHOR_VISUAL_REVIEW')
    base.write(OUT / 'figure_metadata.json', meta)
    base.write(OUT / 'interictal_display_selection.json', display['inset'])
    base.write(OUT / 'interictal_color_definition.json', definition)
    # The separately exported main trajectory must retain every rendered pixel.
    main_file = 'figures/fig6-panelb-full-trajectory.png'
    assert (OUT / main_file).read_bytes() == (SOURCE / main_file).read_bytes()
    with np.load(OUT / 'cycle_color_values.npz') as new, np.load(SOURCE / 'cycle_color_values.npz') as old:
        for key in old.files:
            np.testing.assert_array_equal(new[key], old[key])
    validation = json.loads((OUT / 'validation.json').read_text())
    validation.update(black_interictal=RECORDS, main_trajectory_png_byte_identical=True,
                      cycle_values_unchanged=True)
    base.write(OUT / 'validation.json', validation)
    caption = (SOURCE / 'caption_zh.md').read_text()
    old = next(p for p in caption.split('\n\n') if p.startswith('B：'))
    caption = caption.replace(old,
        'B：显示 0–120 秒原生轨迹及 Interictal 小窗，1–3 在小窗、4 在主轨迹。'
        '小窗在既定 Local 状态 1（8.85 秒）之前用黑色显示连续轨迹及方向箭头；'
        '从 1 起的进入段沿用主图的 0–1 周期进度色条，黑色不对应色条数值。'
        '这只是沿用原四阶段标记的显示分界，没有重新检测发作起点。'
        '主图包括三次进入和前两次完整恢复；第三次终止尚未观察到。')
    (OUT / 'caption_zh.md').write_text(caption)
    readme = (SOURCE / 'figures/README.md').read_text()
    for stem, desc in [
        ('fig6-panelb', '主图保留原周期进度着色。Interictal 小窗在状态 1 之前显示黑色轨迹及方向箭头，之后保留彩色进入段及 1–3 标记。'),
        ('fig6-panelb-interictal', '单独展示相同连续间期至进入轨迹。状态 1 之前为黑色，之后沿用周期进度着色，坐标、样本及标记位置保持。')]:
        start = readme.index('### ' + stem + '.png')
        end = readme.find('\n### ', start + 1)
        if end < 0:
            end = len(readme)
        readme = readme[:start] + f'### {stem}.png / .pdf / .svg\n\n{desc}\n**关注点**：黑色用于突出间期起伏，不代表周期进度为零。\n' + readme[end:]
    (OUT / 'figures/README.md').write_text(readme)
    (OUT / 'README.md').write_text('# Figure 6：黑色间期轨迹\n\n'
        '[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf) · [B](figures/fig6-panelb.png)\n\n'
        '仅将 Interictal 小窗中状态 1 之前的轨迹及方向箭头改为黑色；进入段保留周期色。'
        '主轨迹、其他面板及所有原始数据保持。黑色不对应色条数值。\n\n'
        f'复现：`python {Path(__file__).resolve()}`。\n')
    shutil.copy2(__file__, OUT / 'source' / Path(__file__).name)
    print(json.dumps(dict(output=str(OUT), black_interictal=RECORDS), indent=2))


if __name__ == '__main__':
    main()
