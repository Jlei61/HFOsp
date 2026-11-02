#!/usr/bin/env python3
"""Author-selected R08/R09 panel A, with the requested tighter time windows.

This is a display revision of the existing continuous native simulation. The
author's revised onset shading is stored separately from the statistical entry
times and original E/F field labels.
"""
import copy
import json
from pathlib import Path
import shutil
from matplotlib.lines import Line2D

import build_fig6_history_seizure_selection as previous

np = previous.np
plt = previous.plt
MAIN = previous.MAIN
SOURCE = previous.OUT
OUT = SOURCE.with_name('selected_r08_r09_panela_termination_left_20261031')
SEED = 9108404
WINDOWS = [(8.10, 13.80), (58.50, 60.40)]
DISPLAY_ONSETS = [8.85, 59.30]
PHASES = ['Local', 'Spread', 'Entry', 'Ictal']
COLORS = ['#9b8d32', '#bd6c35', '#ab4255', '#ba263c']
STATE_TIMES = [8.85, 8.90, 8.96, 9.41, 59.30, 59.45, 59.67, 60.24]
# Separate only the three very close numerals; their dotted lines remain at
# exact sample times. Descriptive words never displace a numbered state.
NUMBER_LABEL_TIMES = [8.81, 8.90, 8.99, 9.41, 59.30, 59.45, 59.67, 60.24]


def stage_readouts():
    root = previous.RAW
    with np.load(root / 'geometry.npz') as geo:
        regional_counts = geo['region_counts'][:3]
        cell_counts = geo['cell_e_counts']
    times, regions, field_times, fields = [], [], [], []
    for path in sorted((root / f'runs/seed{SEED}/chunks').glob('*.npz')):
        lo, hi = [int(v) / 10000 for v in path.stem.split('_')]
        if not any(hi > sec - .01 and lo < sec + .01 for sec in STATE_TIMES):
            continue
        with np.load(path) as data:
            times.append(data['time_ms'] / 1000)
            regions.append(data['regions_1ms'][:, :3])
            field_times.append(data['field_time_ms'] / 1000)
            fields.append(data['field_5ms'])
    t, counts = np.concatenate(times), np.concatenate(regions)
    ft, field = np.concatenate(field_times), np.concatenate(fields)
    states, maps = [], []
    for j, sec in enumerate(STATE_TIMES):
        window = [sec - .01, sec + .01]
        count = counts[(t >= window[0]) & (t < window[1])].sum(0)
        spatial_count = field[(ft >= window[0]) & (ft < window[1])].sum(0)
        assert spatial_count.sum() == count.sum()
        spatial_rate = spatial_count / cell_counts / .02
        states.append(dict(number=j+1, label=PHASES[j % 4], episode=j//4+1,
            time_s=sec, label_anchor_time_s=NUMBER_LABEL_TIMES[j], color=COLORS[j % 4],
            sample_window_s=window, all_E_rate20ms_Hz=float(count.sum()/32000/.02),
            core_A_B_surround_rate20ms_Hz=count/regional_counts/.02,
            active_grid_fraction_above20Hz=float((spatial_rate>20).mean())))
        maps.append(spatial_rate)
    # Keep the illustrated frame separate from an inferred causal origin.
    assert states[0]['core_A_B_surround_rate20ms_Hz'][0] == 0
    assert states[0]['core_A_B_surround_rate20ms_Hz'][1] > 100
    if STATE_TIMES[4] == 59.30:
        assert np.all(states[4]['core_A_B_surround_rate20ms_Hz'][:2] == 0)
        assert states[4]['active_grid_fraction_above20Hz'] < .2
    assert states[3]['all_E_rate20ms_Hz'] > 200 and states[7]['all_E_rate20ms_Hz'] > 200
    np.savez_compressed(OUT / 'stage_readouts.npz', time_s=STATE_TIMES,
        field_rate20ms_Hz=maps, cell_e_counts=cell_counts)
    return states


def main():
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    (OUT / 'source').mkdir(exist_ok=True)
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    chosen = [next(c for c in manifest['candidates'] if c['id'] == name) for name in ('R08', 'R09')]
    episodes = [c['episode'] for c in chosen]
    assert [e['seed'] for e in episodes] == [SEED, SEED]
    assert [e['cycle'] for e in episodes] == [1, 2]
    assert episodes[0]['strict_exit_s'] == 17.2
    assert not episodes[0]['right_censored']
    states = stage_readouts()

    print('Filtering the original complete 120-s seed 9108404 readout.', flush=True)
    t_ms, raw, names, paths = previous.load_seed(SEED)
    ids = [names.index(n) for n in previous.ORDER]
    signal = previous.sosfiltfilt(
        previous.butter(4, previous.BAND, btype='bandpass', fs=10000, output='sos'),
        np.asarray(raw[:, ids], float), axis=0)
    t = t_ms / 1000
    cutoff = episodes[0]['global_high_s']
    scales = np.percentile(abs(signal[t < cutoff]), 99.5, axis=0)
    gains = np.maximum(scales, .15 * np.median(scales[scales > 0]))
    seed_record = next(s for s in manifest['seeds'] if s['seed'] == SEED)
    np.testing.assert_array_equal(gains, seed_record['gain_per_contact'])
    # Reproduce both previously shown candidates exactly before extending them.
    for candidate in chosen:
        with np.load(SOURCE / 'waveform_windows' / f"{candidate['id']}.npz") as old:
            index = np.searchsorted(t, old['time_s'])
            np.testing.assert_array_equal(t[index], old['time_s'])
            np.testing.assert_array_equal(signal[index], old['filtered_readout'])
            np.testing.assert_array_equal(gains, old['gain_per_contact'])

    fig = plt.figure(figsize=(17., 5.6))
    durations = np.array([hi - lo for lo, hi in WINDOWS])
    widths = .840 * durations / durations.sum()
    axes = [fig.add_axes([.075, .145, widths[0], .775]),
            fig.add_axes([.075 + widths[0] + .065, .145, widths[1], .775])]
    records = []
    limits = []
    for k, (ax, candidate, window, onset) in enumerate(zip(axes, chosen, WINDOWS, DISPLAY_ONSETS)):
        ep = candidate['episode']
        selected = (t >= window[0]) & (t <= window[1])
        values = (signal[selected, :13] / gains[:13] * previous.GAIN + np.arange(14, 1, -1)) - 2
        drawing_episode = copy.deepcopy(ep)
        drawing_episode['process_start_s'] = onset
        item = dict(episode=drawing_episode, t=t[selected], values=values, window_s=window)
        limits.append(previous.draw(ax, item, left=k == 0))
        # Remove the old gray onset line; colored stages retain their exact
        # times even when the author moves a stage later than the shaded onset.
        assert len(ax.lines) == 15
        ax.lines[-1].remove()
        for state in states[k*4:k*4+4]:
            ax.axvline(state['time_s'], color=state['color'], ls=':', lw=.85, alpha=.9)
        # Both requested windows end during the ongoing episode. The viewport
        # boundary must not be interpreted as the observed termination time.
        for patch in list(ax.patches):
            patch.remove()
        shade_end = min(ep['strict_exit_s'], window[1])
        ax.axvspan(onset, shade_end, fc='#ba263c', alpha=.065, lw=0, zorder=0)
        if k == 0:
            ax.set_xticks([9, 10, 11, 12, 13])
            ax.set_xticklabels(['9', '10', '11', '12', '13'])
        else:
            ax.set_xticks([59, 60])
            ax.set_xticklabels(['59', '60'])
        ax.set_xlim(window)
        np.testing.assert_array_equal(ax.get_xlim(), window)
        ax.tick_params(axis='x', labelsize=13, pad=3)
        ax.tick_params(axis='y', labelsize=13)
        records.append(dict(candidate_id=candidate['id'], seed=SEED, cycle=ep['cycle'],
            visible_window_s=window, displayed_onset_s=onset, shaded_window_s=[onset, shade_end],
            original_statistical_process_start_s=ep['process_start_s'],
            global_high_threshold_s=ep['global_high_s'], observed_termination_s=ep['strict_exit_s'],
            termination_visible=window[0] <= ep['strict_exit_s'] <= window[1],
            original_energy_classification=ep['energy'], samples_plotted=int(selected.sum()),
            gain_per_contact=gains, display_contact_order=previous.ORDER[:13],
            axes_fraction=ax.get_position().bounds, trace_min=float(values.min()), trace_max=float(values.max())))
        np.savez_compressed(OUT / f"{candidate['id']}_display_data.npz", time_s=t[selected],
            filtered_readout=signal[selected], gain_per_contact=gains,
            contact_names=np.array(previous.ORDER), display_onset_s=onset)
    shared_ylim = (min(v[0] for v in limits), max(v[1] for v in limits))
    for ax in axes:
        ax.set_ylim(shared_ylim)
    # Join the two bottom spines with one continuous line. A smooth wave in
    # the middle marks omitted time without adding ticks or interpolated data.
    left_pos, right_pos = [ax.get_position() for ax in axes]
    np.testing.assert_allclose(left_pos.y0, right_pos.y0, rtol=0, atol=1e-14)
    middle = (left_pos.x1 + right_pos.x0) / 2
    wave_width = .025
    u = np.linspace(0., 1., 201)
    wave_x = middle + (u - .5) * wave_width
    wave_y = left_pos.y0 + .010 * np.sin(2 * np.pi * u) * np.sin(np.pi * u)**2
    connector_x = np.r_[left_pos.x1, wave_x, right_pos.x0]
    connector_y = np.r_[left_pos.y0, wave_y, right_pos.y0]
    spine = axes[0].spines['bottom']
    connector = Line2D(connector_x, connector_y, transform=fig.transFigure,
        color=spine.get_edgecolor(), linewidth=spine.get_linewidth(),
        solid_capstyle='butt', clip_on=False)
    fig.add_artist(connector)
    fig.text(.017, .982, 'A', fontsize=20, weight='bold', va='top')
    fig.text(.5275, .087, 'Time (s)', ha='center', va='top', fontsize=16)
    header_y = .945
    annotations = []
    for state in states:
        ax = axes[state['episode'] - 1]
        pos = ax.get_position()
        lo, hi = WINDOWS[state['episode'] - 1]
        label_x = pos.x0 + (state['label_anchor_time_s'] - lo) / (hi - lo) * pos.width
        annotation = ax.annotate(str(state['number']), xy=(state['time_s'], 1.),
            xycoords=ax.get_xaxis_transform(), xytext=(label_x, header_y),
            textcoords='figure fraction', va='baseline', ha='center',
            fontsize=14, weight='bold', color=state['color'], annotation_clip=False,
            arrowprops=dict(arrowstyle='-', color=state['color'], lw=.5, shrinkA=1, shrinkB=0))
        annotations.append(annotation)
    descriptions = [
        fig.text(left_pos.x0, header_y, 'First seizure', ha='left', va='baseline', fontsize=13),
        fig.text(left_pos.x1-.045, header_y, 'Termination', ha='right', va='baseline', fontsize=13),
        fig.text(right_pos.x0, header_y, 'Second seizure', ha='left', va='baseline', fontsize=13),
    ]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Check text extents only, since annotation extents also include leaders.
    from matplotlib.text import Text
    header_boxes = [Text.get_window_extent(a, renderer) for a in annotations] + [
        a.get_window_extent(renderer) for a in descriptions]
    for i, box in enumerate(header_boxes):
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)
        for other in header_boxes[i+1:]:
            assert not box.overlaps(other), (box.bounds, other.bounds)
    assert header_boxes[-1].x0 - header_boxes[-2].x1 >= 8
    one_second_pixels = [float(np.diff(ax.transData.transform(
        [(window[0], 0), (window[0] + 1., 0)])[:, 0])[0])
        for ax, window in zip(axes, WINDOWS)]
    np.testing.assert_allclose(one_second_pixels[0], one_second_pixels[1], rtol=0, atol=1e-10)
    inches_per_second = [v / fig.dpi for v in one_second_pixels]
    fig.savefig(OUT / 'figures/fig6-panela.svg', dpi=220, facecolor='white')
    previous.OUT = OUT
    files = previous.export(fig, 'fig6-panela')

    meta = dict(status='PENDING_AUTHOR_VISUAL_REVIEW', selected_candidates=['R08', 'R09'],
        source_manifest=str(SOURCE / 'manifest.json'), native_run=str(previous.RAW / f'runs/seed{SEED}'),
        producer=str(Path(__file__).resolve()), same_continuous_run=True,
        filter=dict(band_Hz=previous.BAND, order=4, zero_phase=True, sampling_Hz=10000,
            scope='Full 120-s native readout before windowing', common_average_reference=False,
            baseline_subtraction=False),
        normalization=dict(percentile=99.5, cutoff_s=cutoff, gain_floor_fraction=.15,
            display_gain=previous.GAIN, same_fixed_gain_for_both_segments=True),
        channels=dict(analysis=previous.ORDER, display=previous.ORDER[:13], hidden_display_only=['ICL2', 'ICL1']),
        segments=records, shared_ylim=shared_ylim,
        unequal_time_scales=False, inches_per_second=inches_per_second,
        png_pixels_per_second=[v * 220 for v in inches_per_second],
        time_axes='Same linear time scale in both windows; widths proportional to 5.70 s and 1.90 s; 1-s tick spacing.',
        annotation='Numbered stages and First seizure / Termination / Second seizure share one text baseline; connected X axis with a central smooth wave.',
        header=dict(baseline_figure_y=header_y, descriptive_words_displace_numbers=False,
            text_overlap=False, termination_actual_time_s=17.2,
            termination_label_location='Schematic header near the right end of the first segment, left of the gap; actual 17.2 s lies in the omitted interval, not at the viewport edge 13.8 s.',
            right_window_extended_to_include_Ictal=True),
        time_axis_connector=dict(style='Straight connector with a smooth central wave',
            omitted_time_s=[WINDOWS[0][1], WINDOWS[1][0]],
            endpoints_figure=[[float(connector_x[0]), float(connector_y[0])],
                              [float(connector_x[-1]), float(connector_y[-1])]],
            wave_width_figure=wave_width, diagonal_corner_marks=False,
            connector_contains_data=False),
        onset_source='Author waveform review: exclude the transient near 8.7-8.8 s and near 59.2 s.',
        onset_effect='Display shading only. The original event statistics and E/F labels were not recomputed.',
        original_energy_labels=['TB', 'TA'], no_energy_label_printed=True,
        four_stage_semantics=['Local', 'Spread', 'Entry', 'Ictal'],
        states=states,
        stage_selection='Illustrative native 20-ms frames after the author-selected onset, retaining Local/Spread/Entry/Ictal semantics.',
        stage_5_note=('Local reactivation is in the surround; both cores have zero spikes in this 20-ms frame.'
            if STATE_TIMES[4] == 59.30 else
            'Author moved stage 5 into early core recruitment; it is an illustrative sample, not a redefinition of first onset. Native core rates are retained.'),
        numbered_stage_markers='1-4 and 5-8 mark the two entries; no numbered termination state.',
        old_candidate_waveform_and_gain_exact_match=True, pdf_and_png_same_state=True,
        formal_figure_pointer_modified=False, files=files)
    previous.write_json(OUT / 'figure_metadata.json', meta)
    # Check that all requested observations are actually visible, and that no
    # descriptive title or provenance text remains on the figure canvas.
    with previous.fitz.open(files['pdf']) as doc:
        assert len(doc) == 1
        page = doc[0]
        text = page.get_text()
        for forbidden in ('Seed', 'Seizure', 'Field', 'Core:', 'R08', 'R09', 'sensitive'):
            assert forbidden not in text
        for b in page.get_text('blocks'):
            assert 0 <= b[0] and 0 <= b[1] and b[2] <= page.rect.width and b[3] <= page.rect.height
    assert WINDOWS[0][0] < DISPLAY_ONSETS[0] < WINDOWS[0][1] == 13.8 < episodes[0]['strict_exit_s']
    assert WINDOWS[1][0] == 58.5 and WINDOWS[1][1] == 60.4
    assert DISPLAY_ONSETS[0] > 8.80 and DISPLAY_ONSETS[1] == 59.30
    shutil.copy2(__file__, OUT / 'source' / Path(__file__).name)
    shutil.copy2(Path(previous.__file__), OUT / 'source' / Path(previous.__file__).name)
    previous.write_json(OUT / 'source/selected_original_records.json', chosen)
    (OUT / 'figures/README.md').write_text(
        '# Figure 6A：作者选定 R08/R09\n\n'
        '### fig6-panela.png\n\n'
        '同一连续仿真（种子 9108404）的两次发作：左侧 R08 为 8.10–13.80 秒，右侧 R09 为 58.50–60.40 秒；右端略延至 60.40 秒以显示第 8 个 Ictal 采样点（60.24 秒），两侧每秒等宽。'
        '1–4 与 5–8 均保留 Local、Spread、Entry、Ictal 四阶段语义，数字和 First seizure、Termination、Second seizure 共用同一行基线；前三个接近的数字用短引线区分，竖线仍在真实采样时间，文字说明不移动编号。'
        'Termination 文字向左移至第一段右上方，与 gap 分开；该文字为阶段示意，实际终止在省略区内的 17.20 秒，并非左窗终点 13.80 秒，文字不作为时间采样点。X 轴以弯曲线连接，onset 底纹仍从 8.85 秒和 59.30 秒开始，波形通道、滤波、增益与前版一致。\n'
        '**关注点**：阶段编号表示选定采样帧，不单独用于推断哪个核最早启动；每帧原生核率及空间活动保存在 stage_readouts.npz 与 figure_metadata.json。\n\n'
        '### fig6-panela.pdf\n\n'
        '上述 Figure 6A 的原始 PDF 导出；PNG 直接由此 PDF 渲染，显示内容一致。SVG 同时保留便于排版。\n'
        '**关注点**：放大检查各通道波形、每秒等宽的时间轴和中间表示省略时间的弯曲连接线。\n')
    print(json.dumps(dict(output=str(OUT), segments=[{k:r[k] for k in ('candidate_id', 'visible_window_s',
        'displayed_onset_s', 'observed_termination_s', 'trace_min', 'trace_max')} for r in records],
        shared_ylim=shared_ylim, exact_old_waveform_match=True), indent=2), flush=True)


if __name__ == '__main__':
    main()
