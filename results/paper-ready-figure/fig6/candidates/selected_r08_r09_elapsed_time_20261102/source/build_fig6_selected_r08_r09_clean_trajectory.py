#!/usr/bin/env python3
"""Clear Fig6B grids, shift the interictal inset left, and title its time bar."""
import json
from pathlib import Path
import shutil

import build_fig6_selected_r08_r09_without_n as previous
import build_fig6_selected_r08_r09_compact_rows as trajectory

base, plt = previous.base, previous.plt
np = trajectory.np
SOURCE = previous.OUT
OUT = SOURCE.with_name('selected_r08_r09_clean_trajectory_20261102')
INSET_SHIFT = .06
DISPLAY = {}


def export_clean(fig, stem):
    for ax in fig.axes:
        if getattr(ax,'name','') != '3d':
            continue
        ax.grid(False)
        # Preserve the three coordinate axes and pale box edges, without grid lines.
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
            axis.pane.set_alpha(None)
            axis.pane.set_facecolor((1,1,1,0))
            axis.pane.set_edgecolor((.7,.7,.7,.55))
    if stem == 'fig6-panelb':
        main, inset, cbax = fig.axes
        old_position = inset.get_position(original=True).bounds
        new_position = (old_position[0]-INSET_SHIFT,*old_position[1:])
        inset.set_position(new_position)
        inset.set_ylabel('')
        inset.text2D(.25,.27,r'$\eta_M M$',transform=inset.transAxes,
                     fontsize=trajectory.layout.TITLE,ha='center',va='center',rotation=-35)
        # A light, ungridded inset backing separates its local trajectory from the main view.
        for axis in [inset.xaxis,inset.yaxis,inset.zaxis]:
            axis.pane.set_facecolor('#f2f5fa')
            axis.pane.set_edgecolor((.7,.7,.7,.55))
        for text in fig.texts:
            if text.get_text() == 'Interictal':
                x,y = text.get_position()
                text.set_position((x-INSET_SHIFT,y))
        assert cbax.get_ylabel() == 'Time (s)'
        cbax.set_ylabel('')
        cbax.set_title('Time (s)',fontsize=trajectory.layout.LABEL,pad=7)
        main_curve, inset_curve = main.collections[0], inset.collections[0]
        assert main_curve.norm is inset_curve.norm
        assert main_curve.norm.vmin == 0 and main_curve.norm.vmax == trajectory.recovery.HORIZON
        assert all(not ax._draw_grid for ax in [main,inset])
        DISPLAY.update(inset_axes_position=list(new_position),
            inset_shift_left_mm=INSET_SHIFT*139, main_background='transparent panes',
            inset_background='pale gray panes', main_grid=False, inset_grid=False,
            inset_adaptation_label_axes_xy=[.25,.27],
            colorbar_title='Time (s)', colorbar_title_position='above colorbar',
            colorbar_side_label='', colorbar_quantity='elapsed time', colorbar_units='s',
            colorbar_limits=[0,trajectory.recovery.HORIZON],
            rate_quantity='all-E firing rate on the vertical spatial axis (Hz)')
    trajectory.matched.OUT = OUT
    trajectory.NATIVE_EXPORT(fig,stem)


def main():
    pointer=base.MAIN/'results/paper-ready-figure/fig6/current_version.json'
    frozen=pointer.read_bytes()
    shutil.copytree(SOURCE,OUT,dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('validation.json','visual_review.json'))
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans',
                         'pdf.fonttype':42,'svg.fonttype':'none','path.simplify':False})
    meta=json.loads((SOURCE/'figure_metadata.json').read_text())
    trajectory.SOURCE,trajectory.OUT,trajectory.export = SOURCE,OUT,export_clean
    updated=trajectory.draw_b(meta)
    assert updated['segment_count'] == meta['B_display']['segment_count']
    assert updated['inset']['segment_count'] == meta['B_display']['inset']['segment_count']
    np.testing.assert_array_equal(updated['limits'],meta['B_display']['limits'])
    np.testing.assert_array_equal(updated['inset']['limits'],meta['B_display']['inset']['limits'])
    updated.update(DISPLAY)
    updated['standalone_interictal'].update(grid=False,background='transparent panes')
    meta['B_display']=updated
    meta['trajectory']['display_revision']=updated
    meta['trajectory']['outputs']['fig6-panelb']=updated
    meta['trajectory']['outputs']['fig6-panelb-interictal']=updated['standalone_interictal']
    meta.update(producer=str(Path(__file__).resolve()),previous_candidate=str(SOURCE),
        revision_scope='B: remove grids, make main panes transparent, shift inset left 8.34 mm, title the time colorbar.',
        status='PENDING_AUTHOR_VISUAL_REVIEW')
    base.write(OUT/'figure_metadata.json',meta)
    compositor=previous.compositor
    compositor.OUT,compositor.SOURCE=OUT,SOURCE
    compositor.C_DISPLAY.update(meta['C_display'])
    compositor.compose()
    copied=[]
    for stem in ['fig6-panela','fig6-panelc','fig6-paneld','fig6-panele','fig6-panelb-full-trajectory']:
        for ext in ['png','pdf','svg']:
            name=f'{stem}.{ext}'
            assert (OUT/'figures'/name).read_bytes()==(SOURCE/'figures'/name).read_bytes()
            copied.append(name)
    data_names=['trajectory_data.npz','stage_readouts.npz','normalized_propagation_fields.npz',
                'model_selected_patient_arrays.npz','cohort_statistics_selected.json']
    for name in data_names:
        assert (OUT/name).read_bytes()==(SOURCE/name).read_bytes()
    checks={}
    for stem in ['fig6-panelb','fig6-panelb-interictal','fig6-complete-layout']:
        with base.fitz.open(OUT/'figures'/f'{stem}.pdf') as doc:
            checks[stem]=all(page.rect.contains(base.fitz.Rect(s['bbox'])) for page in doc
                for b in page.get_text('dict')['blocks'] if b['type']==0 for line in b['lines'] for s in line['spans'])
            if stem=='fig6-panelb':
                assert 'Time (s)' in doc[0].get_text()
    assert all(checks.values()),checks
    assert pointer.read_bytes()==frozen
    base.write(OUT/'validation.json',dict(unchanged_panel_files=copied,unchanged_data_files=data_names,
        full_trajectory_and_inset_ranges_and_segment_counts_unchanged=True, **DISPLAY,
        PDF_text_inside_canvas=checks,formal_pointer_unchanged=True,human_visual_acceptance='PENDING'))
    caption=(SOURCE/'caption_zh.md').read_text()
    caption+='\n\n本次仅调整 B 的展示：主三维坐标面透明，主图与 Interictal 小窗均无网格线；小窗左移 8.34 mm 并保留浅灰底。色条仍为 0–118 s 时间，Time (s) 移至色条顶部；放电率仍由三维竖轴表示，未改变颜色编码、坐标范围或轨迹。\n'
    (OUT/'caption_zh.md').write_text(caption)
    readme=(SOURCE/'figures/README.md').read_text()
    for stem,description in [
        ('fig6-panelb','主三维背景透明且去掉网格，小窗左移 8.34 mm、保留浅灰底并去掉网格。Time (s) 放到时间色条顶部，原轨迹、阶段标记与显示范围保持。'),
        ('fig6-panelb-interictal','独立间期至进入轨迹去掉网格，背景透明。轨迹数据和 1–3 标记保持。')]:
        start=readme.index('### '+stem+'.png')
        end=readme.find('\n### ',start+1)
        if end<0:end=len(readme)
        readme=readme[:start]+f'### {stem}.png / .pdf / .svg\n\n{description}\n**关注点**：色条表示时间，放电率由竖轴表示。\n'+readme[end:]
    (OUT/'figures/README.md').write_text(readme)
    (OUT/'README.md').write_text('# Figure 6：简化 B 三维背景\n\n'
        '[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf) · [B](figures/fig6-panelb.png)\n\n'
        'B 主图透明背景，无网格；Interictal 小窗左移，色条顶部注明 Time (s)。其他面板保持。\n\n'
        f'复现：`python {Path(__file__).resolve()}`。\n')
    shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
    print(json.dumps(dict(output=str(OUT),checks=checks),indent=2))


if __name__=='__main__':
    main()
