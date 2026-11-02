#!/usr/bin/env python3
"""Focus Fig6 A-C on one interictal event, first entry and termination.

Reuse the completed unified-adaptation simulation and the previous candidate's
E/F. Preserve the complete 60-s trajectory in D. No new simulation or fitting.
"""
import json
from pathlib import Path
import shutil
import hashlib

import build_fig6_unified_adaptation as base
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

PARENT = base.OUT
PRIOR_FOCUS = PARENT.with_name('unified_current_adaptation_focus_20261030')
OUT = PARENT.with_name('unified_current_adaptation_focus14_20261030')
RUN = 'unified_ref48_s9108405'
WINDOW = (8.65, 14.0)
INTER_COLOR = '#267ba8'
STATE_COLORS = ['#9b8d32', '#bd6c35', '#ab4255', '#ba263c']
STATE_LABELS = ['Local', 'Spread', 'Entry', 'Ictal']
INTER_EVENT = None
TERMINATION = None


def marks(ax, states, onset):
    ax.axvspan(onset, TERMINATION, fc='#ba263c', alpha=.075, lw=0)
    ax.axvspan(TERMINATION, WINDOW[1], fc='#547785', alpha=.045, lw=0)
    ax.axvline(INTER_EVENT['peak_s'], color=INTER_COLOR, ls='--', lw=.8, alpha=.8)
    for s in states:
        ax.axvline(s['time_s'], color=s['color'], ls=':', lw=.85, alpha=.85)
    ax.set_xlim(WINDOW)
    ax.xaxis.set_major_locator(FixedLocator([9, 10, 11, 12, 13, 14]))


def select_states(a, result):
    global INTER_EVENT, TERMINATION
    entry = result['temporal']['entries'][0]['onset_s']
    episode = result['episodes'][0]
    INTER_EVENT = result['initial_brief_events'][-1]
    assert INTER_EVENT['start_s'] > WINDOW[0]
    assert result['initial_brief_events'][-2]['end_s'] < WINDOW[0]
    assert WINDOW[1] < result['temporal']['entries'][1]['onset_s']
    # Locate the start of the first 100-ms interval with all E and both cores
    # below 5 Hz. Keep the independently defined observer exit in the metadata.
    counts = np.c_[a['spikes_1ms'][:, 0], a['regions_1ms'][:, :2]]
    rates = counts.reshape(-1, 10, 3).sum(1) / np.r_[32000, a['region_counts'][:2]] / .01
    quiet = np.all(rates < 5, axis=1)
    starts = np.flatnonzero(np.convolve(quiet.astype(int), np.ones(10, int), 'valid') == 10) * .01
    starts = starts[(starts >= episode['exit_start_s'] - .5) & (starts <= episode['exit_confirmed_s'])]
    TERMINATION = float(starts[0])
    # The first two representative windows show local reactivation after the
    # initial transient wave and its subsequent spread. Entry uses the locked
    # sustained-rate criterion. Ictal is the peak 20-ms window of this episode.
    centers = np.arange(round(entry/.005), round(TERMINATION/.005))
    totals = np.array([a['field_5ms'][i-2:i+2].sum() for i in centers])
    ictal = float(centers[np.argmax(totals)] * .005)
    times = [9.10, 9.30, entry, ictal]
    states = [dict(number=i+1, time_s=t, color=STATE_COLORS[i], label=STATE_LABELS[i])
              for i, t in enumerate(times)]
    assert np.all(np.diff(times) > 0)
    assert states == json.loads((PRIOR_FOCUS/'display_selection.json').read_text())['states'][:4]
    return states, entry, episode


def draw_waveform(a, states, entry):
    fig = plt.figure(figsize=(12, 6.5))
    ax = fig.add_axes([.12, .10, .83, .74])
    # Hide only the legacy automatic labels, then place leaders explicitly so
    # the closely spaced recruitment states remain legible on this wider axis.
    base.native.HIDDEN_STATE_NUMBERS = {s['number'] for s in states}
    meta = base.native.draw_a(ax, a, states, entry)
    positions = [9.05, 9.43, 9.81, states[3]['time_s']-.20]
    for s, x in zip(states, positions):
        label = str(s['number'])
        ax.annotate(label, xy=(s['time_s'], 1), xycoords=ax.get_xaxis_transform(),
                    xytext=(x, 1.04),
                    textcoords=ax.get_xaxis_transform(), ha='center', va='bottom',
                    color=s['color'], fontsize=16, weight='bold', annotation_clip=False,
                    arrowprops=dict(arrowstyle='-', color=s['color'], lw=.7))
    ax.annotate('Interictal', xy=(INTER_EVENT['peak_s'], 1), xycoords=ax.get_xaxis_transform(),
                xytext=(WINDOW[0], 1.14), textcoords=ax.get_xaxis_transform(),
                ha='left', va='bottom', fontsize=14, color=INTER_COLOR,
                annotation_clip=False, arrowprops=dict(arrowstyle='-', color=INTER_COLOR, lw=.7))
    ax.set_xlabel('Time (s)')
    ax.tick_params(axis='x', labelbottom=True)
    meta.update(visible_state_numbers=[1, 2, 3, 4],
                state_markers='Interictal dashed, states 1-4 dotted; no termination marker.',
                state_label_offset_points=None, interictal_event=INTER_EVENT,
                display_window_is_contiguous=True, time_warp=False, omitted_internal_intervals=[])
    base.save(fig, 'fig6-panela')
    return meta


def crop_abc():
    fitz = base.fitz
    mm = 72/25.4
    clip = fitz.Rect(0, 0, 194*mm, 238*mm)
    with fitz.open(OUT/'figures/fig6-complete-layout.pdf') as source:
        doc = fitz.open()
        page = doc.new_page(width=clip.width, height=clip.height)
        page.show_pdf_page(page.rect, source, 0, clip=clip)
        doc.save(OUT/'figures/fig6-abc-focus.pdf', garbage=4, deflate=True)
    with fitz.open(OUT/'figures/fig6-abc-focus.pdf') as doc:
        doc[0].get_pixmap(matrix=fitz.Matrix(2.6, 2.6), alpha=False).save(OUT/'figures/fig6-abc-focus.png')
        (OUT/'figures/fig6-abc-focus.svg').write_text(doc[0].get_svg_image())


def main():
    base.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    previous = json.loads((PARENT/'figure_metadata.json').read_text())
    result = json.loads((PARENT/'scientific_result.json').read_text())
    a = base.load(RUN)
    states, entry, episode = select_states(a, result)
    base.native.ETA = .0005
    base.native.WINDOW = WINDOW
    # Freeze the previous full-run filtering and gain reference; only crop the
    # display. In particular do not normalize separately for each event/state.
    base.native.RECRUIT = entry
    base.native.MAIN_BAND_HZ = (80., 1000.)
    base.native.GAIN_PERCENTILE = 99.5
    base.native.DISPLAY_GAIN = .68/1.48
    base.native.marks = marks
    plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 17,
                        'xtick.labelsize': 14, 'ytick.labelsize': 14, 'pdf.fonttype': 42,
                        'svg.fonttype': 'none', 'path.simplify': False})
    waveform = draw_waveform(a, states, entry)
    fig = plt.figure(figsize=(12, 3.9))
    gs = fig.add_gridspec(2, 1, left=.12, right=.95, bottom=.18, top=.84, hspace=.22)
    za = fig.add_subplot(gs[0])
    ma = fig.add_subplot(gs[1], sharex=za)
    slow = base.native.draw_slow(za, ma, a, states, entry)
    base.save(fig, 'fig6-panelb')
    maps = json.loads((PRIOR_FOCUS/'figure_metadata.json').read_text())['spatial_maps']
    for ext in ['png', 'pdf', 'svg']:
        shutil.copy2(PRIOR_FOCUS/'figures'/f'fig6-panelc.{ext}', OUT/'figures'/f'fig6-panelc.{ext}')
    assert [m['state'] for m in maps] == [1, 2, 3, 4]
    trajectory = base.draw_trajectory(a, states, result['observed_s'])
    for panel in 'ef':
        for ext in ['png', 'pdf', 'svg']:
            shutil.copy2(PARENT/'figures'/f'fig6-panel{panel}.{ext}', OUT/'figures'/f'fig6-panel{panel}.{ext}')
    for name in ['scientific_result.json', 'field_comparison.json']:
        shutil.copy2(PARENT/name, OUT/name)
    base.compose()
    crop_abc()
    selection = dict(window_s=WINDOW, continuous_time=True, interictal_event=INTER_EVENT,
                     states=states, map_state_numbers=[1, 2, 3, 4],
                     previous_map_semantics=['Local', 'Spread', 'Entry', 'Ictal'],
                     previous_map_number_to_new={'2': 1, '3': 2, '4': 3, '5': 4},
                     interictal_numbering='Unnumbered event label; no interictal spatial map.',
                     termination_display='No numbered state, text label or vertical line.',
                     first_low_onset_s=TERMINATION, exit_observer=episode,
                     next_high_entry_s=result['temporal']['entries'][1]['onset_s'], trajectory_window_s=[0, result['observed_s']],
                     state_selection='Local9.10s and Spread9.30s after the initial transient wave; Entry is the locked first high-entry onset; Ictal is the highest20ms all-E rate before the first termination.',
                     scientific_scope='First autonomous termination is temporary: next high entry17.54s. Full interictal return occurs after the second exit, retained in D and the parent full-run diagnostics.',
                     new_simulations=0, internal_splicing=False, human_visual_acceptance='PENDING')
    base.write(OUT/'display_selection.json', selection)
    previous.update(source_layout=str(PARENT), states=states, waveform=waveform, slow=slow,
                    spatial_maps=maps, trajectory=trajectory, marked_episode=episode,
                    state_selection=selection['state_selection'],
                    shaded_intervals=[dict(state='high', start_s=entry, end_s=TERMINATION),
                                      dict(state='post_high', start_s=TERMINATION, end_s=WINDOW[1])],
                    display_selection=selection, current_pointer_replaced=False)
    base.write(OUT/'figure_metadata.json', previous)
    source = OUT/'source'
    source.mkdir(exist_ok=True)
    for p in [Path(__file__), Path(base.__file__), Path(base.analysis.__file__)]:
        shutil.copy2(p, source/p.name)
    for name in ['protocol.json', 'effective_model.json']:
        shutil.copy2(PARENT/'source'/name, source/name)
    provenance = dict(parent=str(PARENT), raw_run=str(base.ROOT/'runs'/RUN),
                      source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()},
                      unchanged_panels=['C', 'E', 'F'], unchanged_model=True, new_simulations=0)
    base.write(source/'provenance.json', provenance)
    # Validate scientific identities and the publication files actually saved.
    oldtraj = np.load(PARENT/'trajectory_data.npz')
    newtraj = np.load(OUT/'trajectory_data.npz')
    assert all(np.array_equal(oldtraj[k], newtraj[k]) for k in oldtraj.files)
    previous_gain = json.loads((PARENT/'figure_metadata.json').read_text())['waveform']['gain_per_contact']
    assert np.array_equal(waveform['gain_per_contact'], previous_gain)
    unchanged = {'c': all((PRIOR_FOCUS/'figures'/f'fig6-panelc.{ext}').read_bytes() == (OUT/'figures'/f'fig6-panelc.{ext}').read_bytes() for ext in ['png', 'pdf', 'svg'])}
    for panel in 'ef':
        unchanged[panel] = all((PARENT/'figures'/f'fig6-panel{panel}.{ext}').read_bytes() ==
                               (OUT/'figures'/f'fig6-panel{panel}.{ext}').read_bytes() for ext in ['png', 'pdf', 'svg'])
    assert all(unchanged.values())
    inside = {}
    for pdf in sorted((OUT/'figures').glob('*.pdf')):
        with base.fitz.open(pdf) as doc:
            inside[pdf.name] = all(page.rect.contains(base.fitz.Rect(span['bbox']))
                for page in doc for block in page.get_text('dict')['blocks'] if block['type'] == 0
                for line in block['lines'] for span in line['spans'])
    assert all(inside.values()), inside
    from PIL import Image
    with base.fitz.open(OUT/'figures/fig6-complete-layout.pdf') as doc:
        pix = doc[0].get_pixmap(matrix=base.fitz.Matrix(2.6, 2.6), alpha=False)
        identity = np.array_equal(np.asarray(Image.open(OUT/'figures/fig6-complete-layout.png')),
                                 np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n))
    assert identity
    base.write(OUT/'validation.json', dict(status='PASS', new_simulations=0,
               A_B_time_axis_alignment_error_mm=0, one_complete_preentry_event=True,
               no_internal_time_splicing=True, C_semantics=STATE_LABELS[:4], C_has_termination=False,
               raw_spike_count_conservation=True, fixed_waveform_gain_unchanged=True,
               complete_D_trajectory_values_unchanged=True, unchanged_panels=unchanged,
               saved_PDF_PNG_pixel_identity=identity, PDF_text_inside_canvas=inside,
               first_four_states_unchanged=True, termination_annotation=False,
               human_visual_acceptance='PENDING'))
    descriptions = {
        'fig6-complete-layout': '沿用两列A–F，A/B聚焦8.65–14.00 s。C恢复Local、Spread、Entry、Ictal四种空间状态；D保留完整60 s轨迹，E/F沿用上一候选。',
        'fig6-abc-focus': '完整图左列的同版裁切，便于查看一次间期事件、首次高活动进入及终止。A/B为同一连续时间窗，C只显示进入过程的四幅空间场。',
        'fig6-panela': '只保留进入前最后一次完整间期事件8.70–8.84 s，随后展示高活动及约13.75 s终止。在14.00 s截止，终止后仅保留0.25 s，整段不拼接、没有改变各段增益。',
        'fig6-panelb': '蓝色Z与橙色单变量适应电流，和A使用完全相同的时间轴。整体E为实线，Core A为虚线；终止后Z回升不等于已经完成间期返回。',
        'fig6-panelc': '1 Local、2 Spread、3 Entry、4 Ictal，均来自同条轨迹的20 ms空间放电率。保留统一0–500 Hz色标，termination不占用C的空间图位置。',
        'fig6-paneld': '完整60 s的真实Z、适应电流和总体E率轨迹，坐标数值与上一候选完全相同。保留与A/C对应的1–4标记并移除5，因此仍可看到第一次退出后的再进入及后续恢复。',
        'fig6-panele': '逐字节保留上一候选的新模型与固定SZ26场对照。没有重新选择患者或重新计算匹配。',
        'fig6-panelf': '逐字节保留上一候选的旧模型队列参考。它不代表新适应模型的队列验证。'}
    (OUT/'figures/README.md').write_text('\n\n'.join(f'### {k}.png / .pdf\n{v}\n**关注点**：连续观察窗及状态对应关系见display_selection.json；候选待作者目视检查。' for k,v in descriptions.items())+'\n')
    (OUT/'caption_zh.md').write_text(f'''# Fig6 一次进入与终止的聚焦版

A/B显示同一次已完成仿真中的连续{WINDOW[0]:.2f}–{WINDOW[1]:.2f} s：最后一次完整间期事件8.70–8.84 s、9.56 s持续高活动进入、{TERMINATION:.2f} s低活动起点，以及终止后至14.00 s的短暂低活动。低活动起点定义为总体E和两核10 ms分箱率同时低于5 Hz且持续100 ms的首时刻；既有独立观察器的退出窗仍为13.60–15.60 s。A先对完整60 s原始10 kHz代理读出作80–1000 Hz四阶零相位滤波，再按原先进入前P99.5定标截取该窗口，所有触点增益与上一版完全相同；没有时间拼接或状态间调幅。

C保持旧版四个空间含义，旧编号2–5顺次改成1–4：Local（9.100 s）、Spread（9.300 s）、Entry（9.560 s）、Ictal（{states[3]['time_s']:.3f} s）。前两幅显示初始瞬态波回落后的局部再次激活及扩展，Entry锚定200 Hz持续200 ms的起点，Ictal取本次进入至终止前总体E最高的20 ms率窗。四幅并不声称两帧之间招募始终单调；保留同一20 ms计数窗、20×20空间网格、真实双核位置和0–500 Hz色标。间期事件单独标注；不添加Termination文字、编号5或对应竖线。1–4的时刻和含义与上一聚焦版完全一致，C直接复用上一版文件。

D保留完整60 s轨迹及原来的t/60时间色标，仅同步状态编号。第一次终止后17.54 s再次进入高活动，故本版A/B展示一次自主终止及其后的低活动段，不能据此宣称已经完成间期恢复。第二次终止与53.78 s开始的返回事件仍保留在D；完整raster和原始科学读出见上一候选。

E/F、模型方程、患者示例与统计结果均未改变。E为新模型进入场与固定SZ26，F为旧模型队列参考；本版没有新增仿真、拟合或患者选择。统计单位仍为同一固定网络的一条噪声实现；新图待作者目视检查。
''')
    (OUT/'README.md').write_text(f'''# Fig6 一次进入与终止的聚焦版

已把A/B改为8.65–14.00 s的连续时间窗，只保留进入前最后一次完整间期事件，随后显示首次高活动及13.75 s终止。C恢复Local→Spread→Entry→Ictal四种含义，编号1–4的时刻及含义保持不变；移除5 Termination标记。

- [ABC放大](figures/fig6-abc-focus.png) / [PDF](figures/fig6-abc-focus.pdf)
- [完整A–F](figures/fig6-complete-layout.png) / [PDF](figures/fig6-complete-layout.pdf)
- [图注](caption_zh.md) / [采样与窗口](display_selection.json) / [验证](validation.json)

D仍是完整60 s轨迹；本次13.75 s终止后17.54 s再进入，因此不把第一次退出称为完整间期恢复。完整仿真及恢复诊断见[上一候选]({PARENT}/README.md)。E/F及数据处理增益不变，没有新仿真，未替换正式Fig6，待作者目视检查。

复现：`python {Path(__file__).resolve()}`。
''')
    print(json.dumps(dict(output=str(OUT), states=states, window_s=WINDOW, termination_s=TERMINATION), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
