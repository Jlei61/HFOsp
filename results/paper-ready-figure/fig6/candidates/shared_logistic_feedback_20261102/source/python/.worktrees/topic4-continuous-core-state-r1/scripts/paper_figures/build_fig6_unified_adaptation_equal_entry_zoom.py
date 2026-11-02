#!/usr/bin/env python3
"""Equal onset windows and one linear Z-focused trajectory, with A–E labels."""
import copy
import json
from pathlib import Path
import shutil

import build_fig6_unified_adaptation_entry_zoom as prior

base = prior.base
rows = prior.previous
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

PARENT = prior.OUT
OUT = PARENT.with_name('unified_current_adaptation_equal_entry_zoom_s9108406_20261031')
WINDOWS = [(7.80, 9.20), (53.61, 55.01)]
GAP = .065


def draw_a(a, states, meta):
    native = base.native
    native.RECRUIT = 8.84
    native.ETA = .0005
    native.MAIN_BAND_HZ = (80., 1000.)
    native.GAIN_PERCENTILE = 99.5
    native.DISPLAY_GAIN = .65
    native.HIDDEN_STATE_NUMBERS = set(range(1, 9))

    def marks(ax, ss, onset):
        for start, end, color in [(8.4, meta['termination_s'], '#2166ac'), (54., 58., '#b2182b')]:
            ax.axvspan(start, end, fc=color, alpha=.065, lw=0)
        for s in ss:
            ax.axvline(s['time_s'], ls=':', color=s['color'], lw=.85)
        ax.set_xlim(native.WINDOW)

    native.marks = marks
    fig = plt.figure(figsize=(15.7, 5.45))
    width = (.905-GAP)/2
    axes = [fig.add_axes([.075, .15, width, .77]),
            fig.add_axes([.075+width+GAP, .15, width, .77])]
    records = []
    for k, (ax, window) in enumerate(zip(axes, WINDOWS)):
        native.WINDOW = window
        ss = states[k*4:k*4+4]
        wm = native.draw_a(ax, a, ss, 8.4)
        assert wm['contact_order'][-2:] == prior.REMOVED
        traces = list(ax.lines[:15])
        separator = ax.lines[15]
        for line in traces[-2:]:
            line.remove()
        for line in traces[:13]:
            line.set_ydata(line.get_ydata()-2)
        separator.set_ydata(np.asarray(separator.get_ydata())-2)
        plotted = np.array([line.get_ydata() for line in traces[:13]])
        ax.set(yticks=np.arange(12, -1, -1), yticklabels=wm['contact_order'][:13],
               ylim=(min(-.65, plotted.min()-.12), max(13., plotted.max()+.12)), xlabel='')
        ax.tick_params(axis='x', labelbottom=True, labelsize=13)
        ax.set_xticks([7.8, 8.2, 8.6, 9.0] if k == 0 else [53.8, 54.2, 54.6, 55.0])
        ax.spines[['top', 'right']].set_visible(False)
        if k:
            ax.set_ylabel('')
            ax.tick_params(axis='y', left=False, labelleft=False)
            ax.spines['left'].set_visible(False)
        for s in ss:
            ax.annotate(str(s['number']), xy=(s['time_s'], 1), xycoords=ax.get_xaxis_transform(),
                        xytext=(s['time_s'], 1.025), textcoords=ax.get_xaxis_transform(),
                        ha='center', va='bottom', color=s['color'], fontsize=15, weight='bold',
                        annotation_clip=False)
        assert np.array_equal(wm['gain_per_contact'], meta['waveform']['segments'][0]['gain_per_contact'])
        wm.update(display_contact_order=wm['contact_order'][:13], display_removed=prior.REMOVED,
                  displayed_channel_count=13, visible_state_numbers=[s['number'] for s in ss],
                  titles_removed=True, scalar_vertical_offset_for_remaining_contacts=-2,
                  analysis_contact_order_unchanged=True)
        records.append(wm)
    ylim = (min(ax.get_ylim()[0] for ax in axes), max(ax.get_ylim()[1] for ax in axes))
    for ax in axes:
        ax.set_ylim(ylim)
    for ax, x in [(axes[0], 1.), (axes[1], 0.)]:
        for y in [0., 1.]:
            ax.plot([x-.009, x+.009], [y-.012, y+.012], transform=ax.transAxes,
                    color='#555', lw=1.1, clip_on=False)
    fig.text(.5275, .065, '9.20–53.61 s omitted', ha='center', fontsize=11, color='#555')
    fig.text(.53, .012, 'Time (s)', ha='center', va='bottom', fontsize=16)
    assert np.isclose(np.ptp(WINDOWS[0]), np.ptp(WINDOWS[1]))
    assert np.isclose(axes[0].get_position().width, axes[1].get_position().width)
    base.save(fig, 'fig6-panela')
    return dict(segments=records, visible_windows_s=WINDOWS, omitted_s=[9.20, 53.61],
                window_duration_s=1.4, same_y_limits=ylim, same_contact_gains=True,
                filter_before_windowing=True, seconds_per_width_equal=True,
                axes_gap_figure_fraction=GAP, previous_gap_figure_fraction=.04,
                display_removed=prior.REMOVED, analysis_contacts_removed=[],
                titles_removed=True, continuous_simulation=True,
                state_markers='1–4 and 5–8: Local, Spread, Entry, Ictal',
                termination_visible=False)


def draw_b(states):
    raw = np.load(PARENT/'trajectory_data.npz')
    t, xyz = raw['time_s'], raw['Z_etaM_rate']
    bounds = np.array([[.7, 1.], [0., 130.], [0., 250.]])
    inside = (xyz[:, 0] >= .7) & (xyz[:, 0] <= 1.)
    assert np.all((xyz[inside] >= bounds[:, 0]) & (xyz[inside] <= bounds[:, 1]))
    keep = inside[:-1] & inside[1:]
    segments = np.stack([xyz[:-1], xyz[1:]], axis=1)
    fig = plt.figure(figsize=(4.7, 4.9))
    ax = fig.add_axes([.13, .23, .80, .69], projection='3d')
    ax.computed_zorder = False
    ax.add_collection3d(Line3DCollection(segments[keep], cmap='viridis', norm=Normalize(0, 1),
        array=((t[:-1]+t[1:])/(2*58.))[keep], linewidths=.8, zorder=6, clip_on=False))
    ax.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2], xlabel='$Z$',
           ylabel=r'$\eta_M M$ (mV)', xticks=[.7, .8, .9, 1.], yticks=[0, 60, 120], zticks=[0, 125, 250])
    ax.set_yticklabels(['', '60', '120'])
    ax.view_init(elev=26, azim=-125)
    ax.set_box_aspect((1.65, 1, 1.25))
    ax.set_proj_type('ortho')
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_fontsize(11)
        axis.labelpad = 5
        axis.pane.set_facecolor('#f2f5fa')
    ax.tick_params(labelsize=9, pad=0)
    fig.text(.015, .64, r'$r_E$ (Hz)', rotation=90, va='center', ha='left', fontsize=11)
    records = prior.label_states(ax, states, t, xyz, fontsize=10)
    assert all(np.all((np.array(s['Z_etaM_rate']) >= bounds[:, 0]) &
                      (np.array(s['Z_etaM_rate']) <= bounds[:, 1])) for s in records)
    vertices = np.unique(segments[keep].reshape(-1, 3), axis=0)
    px, py, _ = proj3d.proj_transform(*vertices.T, ax.get_proj())
    screen = ax.transData.transform(np.c_[px, py])
    fb = fig.bbox
    assert np.all((screen[:, 0] >= fb.x0) & (screen[:, 0] <= fb.x1) &
                  (screen[:, 1] >= fb.y0) & (screen[:, 1] <= fb.y1))
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='viridis'),
                     cax=fig.add_axes([.23, .15, .55, .02]), orientation='horizontal', ticks=[0, .5, 1])
    cb.set_label('Normalized time', fontsize=10, labelpad=2)
    cb.ax.tick_params(labelsize=9, pad=2, length=2)
    legend = fig.legend(handles=[Line2D([], [], marker='o', ls='none', color=rows.COLORS[i],
        label=f'{i+1}, {i+5}  {rows.PHASES[i]}', markersize=4) for i in range(4)],
        loc='lower center', bbox_to_anchor=(.51, -.04), ncol=2, frameon=False, fontsize=9,
        handletextpad=.15, columnspacing=.8)
    fig.canvas.draw()
    assert not legend.get_window_extent().overlaps(cb.ax.xaxis.label.get_window_extent())
    # Tight export keeps the projected 3D axis labels; data clipping is applied
    # explicitly in Z above, rather than using Axes3D's square 2D clipping box.
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT/'figures'/f'fig6-panelb.{ext}', dpi=220, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)
    shutil.copy2(PARENT/'trajectory_data.npz', OUT/'trajectory_data.npz')
    return dict(coordinates=['mean Z', 'eta_M mean M (mV)', 'all-E rate (Hz)'],
                time_s=[0, 58], rate_bin_ms=5, display_smoothing_sigma_ms=5,
                normalized_time='t / full duration', view_count=1, limits=bounds,
                linear_axes=True, orthographic=True, elevation_deg=26, azimuth_deg=-125,
                box_aspect=[1.65, 1, 1.25], segment_count=int(keep.sum()), total_segments=len(segments),
                axes_clipping_disabled=True, all_displayed_vertices_inside_canvas=True,
                no_bridging_across_hidden_segments=True, only_Z_controls_segment_selection=True,
                full_raw_trajectory_preserved=True, states=records,
                interpretation='One Z=0.7–1 view of the native trajectory; no coordinate warp or second view.')


def compose():
    mm = 72/25.4
    doc = base.fitz.open()
    page = doc.new_page(width=320*mm, height=330*mm)
    boxes = {'A': (3, 8, 317, 117), 'B': (3, 125, 110, 243), 'C': (112, 125, 317, 243),
             'D': (3, 253, 218, 327), 'E': (225, 253, 317, 327)}
    letters = {'A': (3, 7), 'B': (3, 125), 'C': (111, 125), 'D': (3, 251), 'E': (224, 251)}
    records = {}
    for label, box in boxes.items():
        path = OUT/'figures'/f'fig6-panel{label.lower()}.pdf'
        with base.fitz.open(path) as src:
            page.show_pdf_page(base.fitz.Rect(*(v*mm for v in box)), src, 0, keep_proportion=True)
        page.insert_text(tuple(v*mm for v in letters[label]), label, fontsize=16, fontname='hebo')
        records[label] = dict(source=str(path), box_mm=box)
    path = OUT/'figures/fig6-complete-layout.pdf'
    doc.save(path, garbage=4, deflate=True)
    doc.close()
    rows.layout.export_pdf(path)
    base.write(OUT/'layout_manifest.json', dict(canvas_mm=[320, 330], panels=records,
        relabel_from_parent={'A': 'A', 'D': 'B', 'C': 'C', 'E': 'D', 'F': 'E'},
        A='equal1.4s onset windows; wider gap;13 display contacts; no titles',
        B='one linear Z=0.7–1 trajectory', C='two horizontal rows of four states',
        D='two model onset fields only', E='unchanged historical cohort'))


def write_docs():
    (OUT/'caption_zh.md').write_text('''A–D来自seed9108406的同一条连续58 s统一适应仿真。A左右分别显示7.80–9.20 s与53.61–55.01 s，均为1.40 s、相同绘图宽度与时间比例，中间断轴省略9.20–53.61 s。两个窗口都展示进入前的短暂事件及随后进入发作；第一轮终止与恢复位于省略区。两段使用相同80–1000 Hz滤波、固定触点增益与纵坐标；隐藏ICL2和ICL1仅影响显示，13触点保留原顺序，顶部无标题。滤波在完整10 kHz原生信号上进行，15触点增益规则和下游分析均保持不变。

B为原D的单个三维轨迹，横轴线性放大到Z=0.70–1.00。另两轴为适应电流eta_M M（0–130 mV）与全体E率（0–250 Hz），足以容纳该Z区间内的全部原生数据；只按Z筛选连续线段，跨出显示范围后不连接不相邻点。颜色仍为完整58 s的归一化时间，1–8号与A/C同源。图显示局部Z范围，完整轨迹保留在trajectory_data.npz，完整慢变量过程见补图。

C两行四帧分别对应两次进入，1–4与5–8均保持Local、Spread、Entry、Ictal语义。4号从13.220 s移到9.100 s，为第一轮高活动确认9.040 s之后的早期Ictal；8号保持54.800 s，为第二轮确认54.700 s之后的早期Ictal。A/B/C同步使用这八个时点，4号空间场从原生20 ms窗口重新计算，其他七帧数值保持原样。

D为原E，只显示两个模型onset能量场（8.40–9.40 s、54.00–55.00 s）。患者参考场不绘制，两次相关仍以原E1146 SZ26场为参考，r分别为−0.435、+0.767；15触点、CAR、基线0.5–3.5 s、1–150 Hz能量算子、平滑和逐场min–max显示保持不变。E为原F，历史13人模型队列的统计及去小脚注版本原样保留，未用本次连续仿真重算。

两次持续招募的领先核依次为B和A；首次进入前的短暂A核放电仍保留。核的领先顺序与能量场TA/TB分类是不同观测量，本例两个能量场仍均偏TA。当前修改仅为显示修订，正式入口未替换，待作者目视检查。
''')
    (OUT/'README.md').write_text(f'''# Figure 6：对称进入放大与单幅Z轨迹

[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf) · [图注](caption_zh.md)。

A两侧同为1.40 s，中央间隔由画布宽度4%增至6.5%。原D改为单幅B，Z轴0.7–1；原E/F依次改为D/E，图层原样保留。4号改为9.100 s早期Ictal，A/B/C同步，其他七帧未变。正式入口未替换。

复现：`python {Path(__file__).resolve()}`。原始连续仿真、全部触点分析及能量场统计均复用上一版；验证见validation.json。
''')
    descriptions = {
        'fig6-complete-layout': ('顶部A为相同时间比例的两个进入窗口；中排为单幅B与两行C，底排D/E。所有面板按A–E重新编号。', '原始仿真及底排分析数值保持不变。'),
        'fig6-panela': ('左右均为1.40 s进入放大，中央断轴间隔加宽。13触点、共同增益及无标题样式保留。', '1–4和5–8的四阶段含义相同；4号为9.100 s早期Ictal。'),
        'fig6-panelb': ('原D合为单个三维轨迹，Z轴显示0.7–1。另两轴容纳该区间内全部原生数据，出界线段不跨接。', '这是局部Z范围，颜色仍表示完整58 s的归一化时间。'),
        'fig6-panelc': ('两行均按Local、Spread、Entry、Ictal展开。4号从9.100 s原生20 ms窗口重新计算，其他七帧原样保留。', '与A/B的数字及时点一致。'),
        'fig6-paneld': ('原E的两个模型onset场及相关原样保留。患者参考场继续隐藏。', '相似度的参考对象仍是固定E1146 SZ26场。'),
        'fig6-panele': ('原F统计图原样保留并改标为E。图内小脚注继续去除。', '历史模型队列，不是新适应模型的队列检验。')}
    (OUT/'figures/README.md').write_text('\n\n'.join(
        f'### {name}.png / .pdf\n{body}\n**关注点**：{focus}'
        for name, (body, focus) in descriptions.items())+'\n')
    (OUT/'supplementary_candidates/figures/README.md').write_text('''### slow-state-traces.png / .pdf
完整58 s全体E及双核Z与适应电流曲线。4号时点同步至9.100 s，8号保持54.800 s。
**关注点**：A省略的第一轮终止和中间恢复期仍保留。

### two-entry-core-recruitment.png / .pdf
两次进入的原生raster及三分区放电率。首次短暂A核放电与随后B核持续招募均可见。
**关注点**：持续招募领先与首次spike不是同一判据。
''')


def main():
    (OUT/'figures').mkdir(parents=True, exist_ok=True)
    pointer = base.MAIN/'results/paper-ready-figure/fig6/current_version.json'
    pointer_before = pointer.read_bytes()
    meta = json.loads((PARENT/'figure_metadata.json').read_text())
    route = json.loads((PARENT/'core_recruitment.json').read_text())
    states = copy.deepcopy(meta['states'])
    states[3]['time_s'] = 9.10
    for idx in [3, 7]:
        assert states[idx]['label'] == 'Ictal'
        assert states[idx]['time_s'] > route['episodes'][idx//4]['high_threshold_onset_s']+.2
    base.ROOT = base.analysis.ROOT = rows.audit.RAW
    base.RUN_NAME = 'seed9108406'
    base.OUT = OUT
    a = base.load(base.RUN_NAME)
    plt.rcParams.update({'font.size': 13, 'axes.labelsize': 15, 'axes.titlesize': 14,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'path.simplify': False})
    wave = draw_a(a, states, meta)
    maps = rows.draw_c(a, states)
    for idx in [0, 1, 2, 4, 5, 6, 7]:
        assert np.array_equal(meta['spatial_maps'][idx]['E_rate_Hz'], maps[idx]['E_rate_Hz'])
    trajectory = draw_b(states)
    for old, new in [('e', 'd'), ('f', 'e')]:
        for ext in ['png', 'pdf', 'svg']:
            source = PARENT/'figures'/f'fig6-panel{old}.{ext}'
            target = OUT/'figures'/f'fig6-panel{new}.{ext}'
            shutil.copy2(source, target)
            assert source.read_bytes() == target.read_bytes()
    rates, _ = rows.audit.recruitment(Path(meta['source_run']), a)
    rows.supplements(a, rates, route, states, meta['termination_s'])
    compose()
    for name in ['core_recruitment.json', 'scientific_result.json', 'field_comparisons.json']:
        shutil.copy2(PARENT/name, OUT/name)
    newmeta = copy.deepcopy(meta)
    newmeta.pop('E')
    newmeta.pop('F_scope')
    newmeta.update(parent=str(PARENT), same_run_panels=['A', 'B', 'C', 'D'], states=states,
        waveform=wave, spatial_maps=maps, trajectory=trajectory, D=meta['E'], E_scope=meta['F_scope'],
        state4_revision=dict(previous_time_s=13.22, time_s=9.10,
            meaning='Ictal unchanged; early post-confirmation sample for the first onset zoom'),
        source_simulation_unchanged=True, human_visual_acceptance='PENDING',
        producer=str(Path(__file__).resolve()), panel_relabel={'D': 'B', 'E': 'D', 'F': 'E'})
    base.write(OUT/'figure_metadata.json', newmeta)
    checks = {}
    for path in (OUT/'figures').glob('*.pdf'):
        with base.fitz.open(path) as doc:
            checks[path.name] = all(pg.rect.contains(base.fitz.Rect(s['bbox'])) for pg in doc
                for block in pg.get_text('dict')['blocks'] if block['type'] == 0
                for line in block['lines'] for s in line['spans'])
    assert all(checks.values()), checks
    with base.fitz.open(OUT/'figures/fig6-panela.pdf') as doc:
        text = doc[0].get_text()
        assert all(word not in text for word in ['ICL1\n', 'ICL2\n', 'Interictal', 'Core B', 'Core A'])
    from PIL import Image
    with base.fitz.open(OUT/'figures/fig6-complete-layout.pdf') as doc:
        pix = doc[0].get_pixmap(matrix=base.fitz.Matrix(2.6, 2.6), alpha=False)
        pdf_image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    assert np.array_equal(np.asarray(Image.open(OUT/'figures/fig6-complete-layout.png')), pdf_image)
    for seq in [maps, trajectory['states']]:
        assert [(s['number'], s['time_s']) for s in seq] == [(s['number'], s['time_s']) for s in states]
    assert pointer.read_bytes() == pointer_before
    assert (OUT/'trajectory_data.npz').read_bytes() == (PARENT/'trajectory_data.npz').read_bytes()
    base.write(OUT/'validation.json', dict(status='PASS', same_native_simulation=True,
        A_equal_duration_and_physical_width=True, A_gap_increased=True,
        A_ICL1_ICL2_display_only_removed=True, A_has_no_titles=True, A_parent_gains_identical=True,
        B_single_linear_Z_zoom=True, B_all_selected_vertices_inside_canvas=True,
        B_only_Z_controls_selection=True, B_all_eight_states_in_view=True,
        unchanged_seven_maps=True, state4_native20ms_count_conserved=True,
        state4_and8_follow_respective_entry_confirmations=True, A_B_C_states_synchronized=True,
        full_trajectory_byte_identical=True, D_identical_to_previous_E=True, E_identical_to_previous_F=True,
        main_PNG_matches_PDF_render_exactly=True, PDF_text_inside=checks,
        formal_pointer_unchanged=True, human_visual_acceptance='PENDING'))
    (OUT/'source').mkdir(exist_ok=True)
    for p in [Path(__file__), Path(prior.__file__), Path(rows.__file__), Path(base.__file__)]:
        shutil.copy2(p, OUT/'source'/p.name)
    write_docs()
    print(json.dumps(dict(output=str(OUT), windows_s=WINDOWS, state4_s=9.10,
                          trajectory_views=1, trajectory_limits=trajectory['limits'].tolist())), flush=True)


if __name__ == '__main__':
    main()
