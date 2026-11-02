#!/usr/bin/env python3
"""Color each post-exit-to-onset interval by its own normalized time."""
import json
from pathlib import Path
import shutil

import build_fig6_selected_r08_r09_clean_trajectory as previous

base, np, plt = previous.base, previous.np, previous.plt
trajectory = previous.trajectory
SOURCE = previous.OUT
OUT = SOURCE.with_name('selected_r08_r09_cycle_colors_20261102')
HORIZON = 120.
NATIVE_EXPORT = trajectory.NATIVE_EXPORT
CYCLES = []
RECORDS = {}
META = {}


def progress(times):
    times=np.asarray(times,dtype=float)
    values=np.full(times.shape,np.nan)
    ids=np.zeros(times.shape,dtype=int)
    for cycle in CYCLES:
        use=(times>=cycle['start_s']) & (times<cycle['stop_s'])
        values[use]=np.minimum((times[use]-cycle['start_s'])/
                              (cycle['onset_s']-cycle['start_s']),1.)
        ids[use]=cycle['episode']
    assert np.isfinite(values).all() and np.all((values>=0)&(values<=1))
    return values,ids


def color_at(sec):
    return plt.colormaps['viridis'](progress([sec])[0][0])


def recolor_arrows(ax):
    arrows=[item.arrow_patch for item in ax.texts
            if item.get_text()=='' and getattr(item,'arrow_patch',None) is not None]
    records=META['B_display']['inset']['direction_arrows']
    assert len(arrows)==len(records)==4
    for arrow,record in zip(arrows,records):
        arrow.set_color(color_at((record['from_time_s']+record['to_time_s'])/2))


def export_cycle(fig,stem):
    norm=base.Normalize(0,1)
    axes=[ax for ax in fig.axes if getattr(ax,'name','')=='3d']
    counts=[]
    for ax in axes:
        curves=[c for c in ax.collections if isinstance(c,base.Line3DCollection)]
        assert len(curves)==1
        curve=curves[0]
        times=np.asarray(curve.get_array()).copy()
        values,ids=progress(times)
        curve.set_array(values)
        curve.set_norm(norm)
        counts.append(len(values))
    if stem=='fig6-panelb':
        main,inset,cbax=fig.axes
        bar=cbax._colorbar
        bar.update_normal(main.collections[0])
        bar.set_ticks([0,.5,1])
        bar.set_ticklabels(['0','0.5','1'])
        cbax.set_title('Cycle progress',fontsize=trajectory.layout.LABEL,pad=7)
        for collection,sec in zip(main.collections[1:3],META['trajectory']['first_returned_brief_s']):
            collection.set_facecolor(color_at(sec))
        recolor_arrows(inset)
        assert main.collections[0].norm is inset.collections[0].norm is bar.norm
        assert counts==[23999,667]
        RECORDS[stem]=dict(segment_counts=counts,shared_cycle_normalization=True)
        previous.DISPLAY.update(colorbar_title='Cycle progress',colorbar_quantity='normalized time to onset',
            colorbar_units='dimensionless',colorbar_limits=[0,1])
        trajectory.matched.OUT=OUT
        NATIVE_EXPORT(fig,stem)
        # Also supply the same full trajectory without the inset; no stale absolute-time export.
        fig.delaxes(inset)
        for item in list(fig.texts):
            if item.get_text()=='Interictal':item.remove()
        NATIVE_EXPORT(fig,'fig6-panelb-full-trajectory')
        RECORDS['fig6-panelb-full-trajectory']=dict(segment_counts=[counts[0]],time_s=[0,HORIZON],
            title=None,grid=False,background='transparent panes',colorbar_title='Cycle progress')
    else:
        assert stem=='fig6-panelb-interictal' and counts==[667]
        recolor_arrows(axes[0])
        RECORDS[stem]=dict(segment_counts=counts,shared_cycle_normalization=True)
        trajectory.matched.OUT=OUT
        NATIVE_EXPORT(fig,stem)


def main():
    global META
    pointer=base.MAIN/'results/paper-ready-figure/fig6/current_version.json'
    frozen=pointer.read_bytes()
    shutil.copytree(SOURCE,OUT,dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('validation.json','visual_review.json'))
    META=json.loads((SOURCE/'figure_metadata.json').read_text())
    history=trajectory.recovery.history()
    start=0.
    for episode in history['episodes']:
        stop=episode['strict_exit_s'] if episode['strict_exit_s'] is not None else HORIZON
        CYCLES.append(dict(episode=episode['episode'],start_s=start,onset_s=episode['global_high_s'],
            onset_definition='existing global_high_s',stop_s=stop,termination_s=episode['strict_exit_s'],
            preceding_termination_observed=episode['episode']>1,right_censored_termination=episode['right_censored']))
        start=stop
    assert len(CYCLES)==3 and CYCLES[-1]['right_censored_termination']
    for c in CYCLES:
        samples=np.array([c['start_s'],(c['start_s']+c['onset_s'])/2,c['onset_s'],c['stop_s']-.001])
        np.testing.assert_allclose(progress(samples)[0],[0,.5,1,1],atol=1e-12)
    with np.load(SOURCE/'trajectory_data.npz') as data:
        times=data['time_s']
        values,ids=progress(times)
        assert len(times)==24000 and times.max()<HORIZON
    np.savez_compressed(OUT/'cycle_color_values.npz',time_s=times,cycle_progress=values,episode=ids)
    color_definition=dict(cycles=CYCLES,
        formula='min((t - previous_termination) / (next_global_high_onset - previous_termination), 1)',
        first_cycle_start='simulation start t=0; no preceding termination observed',
        ictal_coloring='Hold progress at 1 from global_high_s until strict_exit_s; reset at the exit.',
        no_time_warp=True,coordinates_unchanged=True,colormap='viridis',range=[0,1],
        interpretation='Retrospective event-aligned display; uses observed future onset, not a forecast or autonomous phase estimate.',
        history_source=str(base.MAIN/'results/topic4_sef_hfo/adaptation_history_20261031/long120s/results.json'))
    base.write(OUT/'cycle_color_definition.json',color_definition)
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans',
                         'pdf.fonttype':42,'svg.fonttype':'none','path.simplify':False})
    trajectory.recovery.HORIZON=HORIZON
    trajectory.SOURCE,trajectory.OUT=SOURCE,OUT
    previous.OUT=OUT
    trajectory.export=previous.export_clean
    trajectory.NATIVE_EXPORT=export_cycle
    try:
        updated=trajectory.draw_b(META)
    finally:
        trajectory.NATIVE_EXPORT=NATIVE_EXPORT
    updated.update(previous.DISPLAY)
    updated.update(color_definition=color_definition,display_time_s=[0,HORIZON])
    for detail_key in ['inset','standalone_interictal']:
        updated[detail_key].update(color_quantity='cycle progress',color_limits=[0,1])
        updated[detail_key].pop('time_color_scale_s',None)
    updated['standalone_interictal'].update(grid=False,background='transparent panes')
    base.write(OUT/'interictal_display_selection.json',updated['inset'])
    old_end=META['trajectory']['time_s'][1]
    META['B_display']=updated
    META['trajectory'].update(time_s=[0,HORIZON],colorbar_units='dimensionless',
        colorbar_quantity='cycle progress',color_definition=color_definition,display_revision=updated,
        view_end_reason='Include all three observed onsets from the existing 120 s record; third termination is not observed.')
    META['trajectory']['outputs'].update({'fig6-panelb':updated,
        'fig6-panelb-interictal':updated['standalone_interictal'],
        'fig6-panelb-full-trajectory':RECORDS['fig6-panelb-full-trajectory']})
    META.update(producer=str(Path(__file__).resolve()),previous_candidate=str(SOURCE),
        revision_scope='B line colors reset each cycle, align each high-activity onset to 1, and include the existing 118–120 s tail.',
        status='PENDING_AUTHOR_VISUAL_REVIEW')
    base.write(OUT/'figure_metadata.json',META)
    compositor=previous.previous.compositor
    compositor.OUT,compositor.SOURCE=OUT,SOURCE
    compositor.C_DISPLAY.update(META['C_display'])
    compositor.compose()
    copied=[]
    for stem in ['fig6-panela','fig6-panelc','fig6-paneld','fig6-panele']:
        for ext in ['png','pdf','svg']:
            name=f'{stem}.{ext}'
            assert (OUT/'figures'/name).read_bytes()==(SOURCE/'figures'/name).read_bytes()
            copied.append(name)
    data_names=['trajectory_data.npz','stage_readouts.npz','normalized_propagation_fields.npz',
                'model_selected_patient_arrays.npz','cohort_statistics_selected.json']
    for name in data_names:
        assert (OUT/name).read_bytes()==(SOURCE/name).read_bytes()
    checks={}
    for stem in ['fig6-panelb','fig6-panelb-interictal','fig6-panelb-full-trajectory','fig6-complete-layout']:
        with base.fitz.open(OUT/'figures'/f'{stem}.pdf') as doc:
            checks[stem]=all(page.rect.contains(base.fitz.Rect(s['bbox'])) for page in doc
                for b in page.get_text('dict')['blocks'] if b['type']==0 for line in b['lines'] for s in line['spans'])
    assert all(checks.values()),checks
    assert pointer.read_bytes()==frozen
    base.write(OUT/'validation.json',dict(unchanged_panel_files=copied,unchanged_data_files=data_names,
        all_three_cycle_starts_zero_and_onsets_one=True,phase_held_one_during_ictal=True,
        original_118_seconds_plus_native_two_second_tail=True,previous_display_end_s=old_end,
        shared_colormap='viridis',colorbar_range=[0,1],record_counts=RECORDS,
        PDF_text_inside_canvas=checks,formal_pointer_unchanged=True,human_visual_acceptance='PENDING'))
    caption=(SOURCE/'caption_zh.md').read_text()
    b_paragraph=next(p for p in caption.split('\n\n') if p.startswith('B：'))
    caption=caption.replace(b_paragraph,'B：显示 0–120 秒原生轨迹及 Interictal 小窗，1–3 在小窗、4 在主轨迹；主图与小窗共享 0–1 周期进度色条，色条与三维坐标框等高。小窗包含间期事件及连续进入段，主图包括三次进入和前两次完整恢复；第三次终止尚未观察到。')
    caption=caption.replace('色条仍为 0–118 s 时间，Time (s) 移至色条顶部；放电率仍由三维竖轴表示，未改变颜色编码、坐标范围或轨迹。',
        '色条改为 Cycle progress（0–1），坐标与轨迹数值保持；放电率仍由竖轴表示。')
    caption+='\n\nB 的周期进度：第一段从仿真开始计时，后续从前一次 strict_exit_s 计时，到下一次 global_high_s 线性归一化为 1；发作期间保持 1，终止后重置为 0。三次进入为 9.39、60.22、119.27 s，前两次终止为 17.2、75.9 s；第三次终止未在 120 s 记录内观察到。B 补上已有记录的最后 2 s，不重跑仿真。此着色使用已观察到的起止时刻，仅用于回顾性对齐，并非预测指标或动力学相位估计。\n'
    (OUT/'caption_zh.md').write_text(caption)
    readme=(SOURCE/'figures/README.md').read_text()
    for stem,desc in [
        ('fig6-panelb','主图与小窗共用 Cycle progress 色条，0 为本周期起点，1 为进入发作；发作期间维持终点色，终止后重新计时。主图沿用完整 120 s 原始记录，包括两次完整发作及第三次进入，背景与小窗布局保持。'),
        ('fig6-panelb-interictal','保留同一连续间期至进入轨迹和 1–3 标记，改用与主图一致的周期进度着色。窗口、坐标和方向箭头对应的样本保持。'),
        ('fig6-panelb-full-trajectory','独立展示同一 0–120 s 全局轨迹，使用相同周期进度色条。此图不叠加间期小窗。')]:
        start=readme.index('### '+stem+'.png');end=readme.find('\n### ',start+1)
        if end<0:end=len(readme)
        readme=readme[:start]+f'### {stem}.png / .pdf / .svg\n\n{desc}\n**关注点**：颜色依据已知起止时刻归一化，第三次仅观察到进入。\n'+readme[end:]
    (OUT/'figures/README.md').write_text(readme)
    (OUT/'README.md').write_text('# Figure 6：按每次发作周期着色\n\n'
        '[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf) · [B](figures/fig6-panelb.png)\n\n'
        'B 的颜色在每次终止后重新计时，到下一次全局高活动进入时为 1。颜色为回顾性周期进度，定义见 cycle_color_definition.json。\n\n'
        f'复现：`python {Path(__file__).resolve()}`。\n')
    shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
    print(json.dumps(dict(output=str(OUT),cycles=CYCLES,checks=checks),indent=2))


if __name__=='__main__':
    main()
