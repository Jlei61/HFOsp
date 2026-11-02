#!/usr/bin/env python3
"""Integrate the author-selected R08/R09 waveforms and their exact spatial frames."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import build_fig6_selected_r08_r09_panela as panel_a
import build_fig6_unified_adaptation_origin_review as reference

base = reference.base
np, plt = base.np, base.plt
OUT = panel_a.SOURCE.with_name('selected_r08_r09_complete_20261031')
HISTORICAL = reference.OUT
SEED = panel_a.SEED
HORIZON = panel_a.WINDOWS[-1][-1]


def export(fig, name):
    """Render the preview from the exact vector PDF being delivered."""
    folder = base.OUT / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{name}.pdf'
    bounds = dict(bbox_inches='tight', pad_inches=.08) if name in ['fig6-paneld', 'fig6-panele', 'onset-field-signed-z'] else {}
    fig.savefig(path, dpi=220, facecolor='white', **bounds)
    fig.savefig(path.with_suffix('.svg'), dpi=220, facecolor='white', **bounds)
    plt.close(fig)
    with base.fitz.open(path) as doc:
        doc[0].get_pixmap(matrix=base.fitz.Matrix(220/72, 220/72), alpha=False).save(path.with_suffix('.png'))


def load_native():
    run = panel_a.previous.RAW / f'runs/seed{SEED}'
    a = base.analysis.load(run, 'chunks', ['time_ms', 'spikes_1ms', 'regions_1ms',
        'field_time_ms', 'field_5ms', 'slow_time_ms', 'Z', 'M'])
    with np.load(panel_a.previous.RAW / 'geometry.npz') as data:
        a.update(dict(data))
    raw = base.analysis.load(run, 'dense_readout_chunks', ['time_ms', 'contact_current'])
    a.update(lfp_native_time_ms=raw['time_ms'], lfp_native=raw['contact_current'])
    assert a['spikes_1ms'][:, 0].sum() == a['field_5ms'].sum() == a['regions_1ms'][:, :3].sum()
    return a


def draw_b(a, states):
    rate = a['spikes_1ms'][:, 0].reshape(-1, 5).sum(1) / 32000 / .005
    t = (np.arange(len(rate)) + .5) * .005
    st = a['slow_time_ms'] / 1000
    xyz = np.c_[np.interp(t, st, a['Z'][:, 0]),
        .0005 * np.interp(t, st, a['M'][:, 0]), base.gaussian_filter1d(rate, 1)]
    np.savez_compressed(OUT / 'trajectory_data.npz', time_s=t, Z_etaM_rate=xyz)
    visible = (t <= HORIZON) & (xyz[:, 0] >= .5) & (xyz[:, 0] <= 1.)
    ymax = max(300., float(np.ceil(xyz[visible, 1].max() / 50) * 50))
    zmax = max(350., float(np.ceil(xyz[visible, 2].max() / 50) * 50))
    bounds = np.array([[.5, 1.], [0., ymax], [0., zmax]])
    assert np.all((xyz[visible] >= bounds[:, 0]) & (xyz[visible] <= bounds[:, 1]))
    keep = visible[:-1] & visible[1:]
    segments = np.stack([xyz[:-1], xyz[1:]], axis=1)[keep]
    fig = plt.figure(figsize=(6.7, 6.1))
    ax = fig.add_axes([.06, .09, .76, .87], projection='3d', computed_zorder=False)
    ax.add_collection3d(base.Line3DCollection(segments, cmap='viridis', norm=base.Normalize(0, 1),
        array=((t[:-1]+t[1:])/(2*HORIZON))[keep], linewidths=.85, zorder=6, clip_on=False))
    ax.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2], xlabel='$Z$',
        ylabel=r'$\eta_M M$ (mV)', xticks=[.5, .75, 1.],
        yticks=[0, ymax/2, ymax], zticks=[0, zmax/2, zmax])
    ax.set_yticklabels(['', f'{ymax/2:g}', f'{ymax:g}'])
    ax.view_init(elev=26, azim=-125)
    ax.set_box_aspect((1.65, 1., 1.25), zoom=1.09)
    ax.set_proj_type('ortho')
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_fontsize(13)
        axis.labelpad = 7
        axis.pane.set_facecolor('#f2f5fa')
    ax.tick_params(labelsize=11, pad=1)
    fig.text(.009, .61, r'$r_E$ (Hz)', rotation=90, va='center', fontsize=13)
    labels = reference.prior.label_states(ax, states[:4], t, xyz, fontsize=12)
    cb = fig.colorbar(base.ScalarMappable(norm=base.Normalize(0, 1), cmap='viridis'),
        cax=fig.add_axes([.87, .30, .022, .48]), ticks=[0, .5, 1])
    cb.set_label('Normalized time', fontsize=12, labelpad=5)
    cb.ax.tick_params(labelsize=11, pad=3, length=3)
    fig.canvas.draw()
    vertices = np.unique(segments.reshape(-1, 3), axis=0)
    px, py, _ = base.proj3d.proj_transform(*vertices.T, ax.get_proj())
    xy = ax.transData.transform(np.c_[px, py])
    assert np.all((xy[:, 0] >= 0) & (xy[:, 0] <= fig.bbox.width)
        & (xy[:, 1] >= 0) & (xy[:, 1] <= fig.bbox.height))
    export(fig, 'fig6-panelb')
    return dict(seed=SEED, time_s=[0, HORIZON], full_saved_time_s=[0, 120], limits=bounds,
        coordinates=['mean Z', 'eta_M mean M (mV)', 'all-E rate (Hz)'],
        rate_bin_ms=5, display_smoothing_sigma_ms=5, visible_state_numbers=[1, 2, 3, 4],
        states=labels, colorbar_orientation='vertical', legend=False,
        normalized_time=f't / {HORIZON} s', no_bridging_across_hidden_segments=True,
        only_Z_and_time_control_segment_selection=True, segment_count=int(keep.sum()))


def draw_c(a, states):
    with np.load(OUT / 'stage_readouts.npz') as source:
        np.testing.assert_array_equal(source['time_s'], [s['time_s'] for s in states])
        maps = source['field_rate20ms_Hz'].copy()
    fig = plt.figure(figsize=(9.3, 5.65))
    gs = fig.add_gridspec(2, 4, left=.085, right=.875, bottom=.11, top=.90,
        wspace=.15, hspace=.66)
    records, row_axes = [], [[], []]
    ft, rt = a['field_time_ms']/1000, a['time_ms']/1000
    for j, state in enumerate(states):
        row, col = divmod(j, 4)
        lo, hi = state['sample_window_s']
        count = a['field_5ms'][(ft >= lo) & (ft < hi)].sum(0)
        regional = a['regions_1ms'][(rt >= lo) & (rt < hi), :3].sum(0)
        assert count.sum() == regional.sum()
        np.testing.assert_array_equal(maps[j], count / a['cell_e_counts'] / .02)
        ax = fig.add_subplot(gs[row, col])
        row_axes[row].append(ax)
        im = ax.imshow(maps[j].reshape(20, 20), origin='lower', extent=[0, 20, 0, 20],
            cmap='magma', norm=base.PowerNorm(.6, 0, 500), interpolation='nearest')
        for k, xy in enumerate(a['centers_mm']):
            ax.add_patch(base.Circle(xy, float(a['core_radius_mm']), ec='#2dd4cd', fc='none', lw=1.1))
            ax.text(xy[0], xy[1]+2.1, 'AB'[k], color='#10656a', fontsize=10, ha='center',
                weight='bold', bbox=dict(fc='white', ec='none', pad=.15, alpha=.85))
        ax.set(xticks=[0, 10, 20], yticks=[0, 10, 20])
        ax.tick_params(labelsize=10, length=2, pad=2, labelbottom=row==1, labelleft=col==0)
        if row == 1:
            ax.set_xlabel('x (mm)', fontsize=11, labelpad=3)
        if col == 0:
            ax.set_ylabel('y (mm)', fontsize=11, labelpad=3)
        ax.set_title(f"{state['number']}  {state['label']}\n{state['time_s']:.2f} s",
            color=state['color'], fontsize=12, pad=6)
        records.append(dict(**state, native_spike_count=int(count.sum()),
            field_matches_A_stage_readout=True, E_rate_Hz=maps[j]))
    fig.canvas.draw()
    colorbars = []
    for row, axs in enumerate(row_axes):
        pos = axs[-1].get_position()
        cb = fig.colorbar(im, cax=fig.add_axes([.903, pos.y0, .015, pos.height]), ticks=[0, 250, 500])
        cb.set_label('E rate (Hz)', fontsize=11, labelpad=4)
        cb.ax.tick_params(labelsize=10, length=2, pad=2)
        fig.text(.012, pos.y0+pos.height/2, f'Entry {row+1}', rotation=90,
            ha='center', va='center', fontsize=12)
        np.testing.assert_allclose(cb.ax.get_position().extents[[1, 3]], pos.extents[[1, 3]])
        colorbars.append(dict(row=row+1, bounds=cb.ax.get_position().bounds, limits=[0, 500]))
    export(fig, 'fig6-panelc')
    return dict(frames=records, rows=2, columns=4, separate_row_colorbars=colorbars,
        both_colorbars_same_normalization=True, sample_width_ms=20,
        A_C_times_and_fields_exact_match=True)


def compose():
    mm = 72/25.4
    doc = base.fitz.open()
    page = doc.new_page(width=320*mm, height=330*mm)
    boxes = {'A': (0, 1, 320, 110), 'B': (3, 119, 132, 249),
        'C': (135, 119, 317, 249), 'D': (3, 258, 218, 327), 'E': (225, 258, 317, 327)}
    letters = {'B': (3, 119), 'C': (135, 119), 'D': (3, 257), 'E': (224, 257)}
    records = {}
    for letter, box in boxes.items():
        path = OUT / 'figures' / f'fig6-panel{letter.lower()}.pdf'
        with base.fitz.open(path) as src:
            page.show_pdf_page(base.fitz.Rect(*(v*mm for v in box)), src, 0, keep_proportion=True)
        if letter in letters:
            page.insert_text(tuple(v*mm for v in letters[letter]), letter, fontsize=16, fontname='hebo')
        records[letter] = dict(source=str(path), box_mm=box)
    path = OUT / 'figures/fig6-complete-layout.pdf'
    doc.save(path, garbage=4, deflate=True)
    doc.close()
    with base.fitz.open(path) as doc:
        doc[0].get_pixmap(matrix=base.fitz.Matrix(220/72, 220/72), alpha=False).save(path.with_suffix('.png'))
        path.with_suffix('.svg').write_text(doc[0].get_svg_image(text_as_path=False))
    base.write(OUT / 'layout_manifest.json', dict(canvas_mm=[320, 330], panels=records,
        A='Selected continuous R08/R09; shared seconds per width; left-shifted Termination',
        B='Enlarged single global mean trajectory, Z 0.5-1; states 1-4 only',
        C='Two rows of four exact A frames, one equally scaled colorbar per row',
        D='Same-run onset fields at author-selected 8.85 s and 59.30 s', E='Unchanged historical cohort'))


def documents(meta):
    times = ', '.join(f'{s["time_s"]:.2f}' for s in meta['states'])
    (OUT / 'caption_zh.md').write_text(f'''Figure 6：同一条统一适应模型连续仿真的两次发作。

A：种子 {SEED}，左侧 8.10–13.80 s，右侧 58.50–60.40 s，时间与通道增益均使用相同比例；80–1000 Hz 滤波在完整 120 s 原生信号上进行。1–4 和 5–8 分别为 Local、Spread、Entry、Ictal，实际采样时刻为 {times} s。Termination 文字向左移并与 gap 分开，其位置是阶段示意，不是时间采样点：实际终止为 17.20 s，在断轴省略区内。

B：同一仿真 0–{HORIZON:.2f} s 的全体 E 平均 Z、适应电流 eta_M M 和放电率轨迹，线性显示 Z=0.5–1；只标 1–4。竖色条为该时间范围的归一化时间。超出 Z 范围的段落不跨接；完整 120 s 轨迹另存。

C：每个编号读取 A 的同一个原生时间，以该时刻为中心的 20 ms 窗口计算 20×20 空间网格放电率；两行各有一个 0–500 Hz 色条，采用相同显示映射。1/5 都是最后一次间歇事件之后的局部再激活；第 5 帧主要在核外，两核该 20 ms 均无放电，不能据此称为 core A 起始。

D：同一仿真在 8.85–9.85 s 和 59.30–60.30 s 的模型 onset 能量场。复用固定 0.5–3.5 s 基线、全部 15 触点 CAR、1–150 Hz 谱功率和既定空间平滑方法；r 对比固定患者 E1146 SZ26 的发作能量场，患者参考场不显示。旧候选 TB/TA 标签来自另两个统计起点，本图未直接沿用这些类型标签。

E：历史 13 人模型队列的统计图原样保留，不使用此次两例重新计算。所有模型 A–D 同源，当前仅为待作者目视检查的候选，正式 Figure 6 入口未替换。
''')
    descriptions = {
        'fig6-complete-layout': '完整 A–E：两次发作、单个平均状态轨迹、八个空间状态、两个 onset 能量场以及历史队列统计。A–D 使用相同连续仿真。',
        'fig6-panela': '保留已选两次发作的信号与编号，Termination 向左移。两段时间每秒等宽，中间保留弯曲断轴连接。',
        'fig6-panelb': '放大单个三维状态轨迹，Z 为 0.5–1。仅标第一轮 1–4，使用竖色条并去掉图例。',
        'fig6-panelc': '两行各四幅原生空间放电率图，编号和中心时刻严格对应 A。每行均有同尺度、与格子等高的独立色条。',
        'fig6-paneld': '显示两次作者选定 onset 后一秒的模型能量场。色条与实际场图等高，不显示患者参考场。',
        'fig6-panele': '原样复制历史模型队列统计。此次示例不改变该图的数据和检验。'}
    (OUT / 'figures/README.md').write_text('\n\n'.join(f'### {name}.png / .pdf / .svg\n\n{desc}\n'
        '**关注点**：' + ('逐一对应 A 与 C 的 1–8；第 5 帧为核外局部活动。' if name in ['fig6-complete-layout', 'fig6-panelc']
            else 'Termination 实际时刻在省略区内，文字位置不是采样点。' if name == 'fig6-panela'
            else '保留原生数据与显示范围的区别，待作者目视检查。') for name, desc in descriptions.items()) + '\n')
    (OUT / 'README.md').write_text(f'# Figure 6：R08/R09 完整候选\n\n'
        '[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf) · [图注](caption_zh.md)\n\n'
        'A/C 逐帧对应表见 A_C_stage_correspondence.csv，数据见 stage_readouts.npz；旧候选及正式入口保留。\n\n'
        f'复现：`python {Path(__file__).resolve()}`。\n')


def main():
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    pointer = base.MAIN / 'results/paper-ready-figure/fig6/current_version.json'
    pointer_bytes = pointer.read_bytes()
    panel_a.OUT = OUT
    panel_a.main()
    wave = json.loads((OUT / 'figure_metadata.json').read_text())
    shutil.copy2(OUT / 'figure_metadata.json', OUT / 'panela_metadata.json')
    states = wave['states']
    base.OUT = reference.OUT = OUT
    base.ROOT = base.analysis.ROOT = panel_a.previous.RAW
    base.RUN_NAME = f'seed{SEED}'
    base.save = export
    plt.rcParams.update({'font.size': 13, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'path.simplify': False})
    print('Loading native counts, slow states and current readout.', flush=True)
    a = load_native()
    trajectory = draw_b(a, states)
    spatial = draw_c(a, states)
    comparisons = []
    for j, onset in enumerate(panel_a.DISPLAY_ONSETS):
        print(f'Calculating onset field {j+1}: {onset:.2f}–{onset+1:.2f} s.', flush=True)
        base.OUT = OUT / 'onset_comparisons' / f'entry{j+1}'
        base.OUT.mkdir(parents=True, exist_ok=True)
        c = base.onset_comparison(a, onset)
        c['contract'].update(model_source=f'Same continuous 120-s unified-adaptation run; seed {SEED}',
            model_onset='Author-selected local reactivation after last brief; same as A states 1/5; fixed one-second window.')
        c.update(episode=j+1, candidate_id=['R08', 'R09'][j], producer=str(Path(__file__).resolve()))
        c['producer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        base.write(base.OUT / 'field_comparison.json', c)
        comparisons.append(c)
    base.OUT = OUT
    assert comparisons[0]['patient_source_sha256'] == comparisons[1]['patient_source_sha256']
    np.testing.assert_array_equal(comparisons[0]['model_baseline_log_power'], comparisons[1]['model_baseline_log_power'])
    dmeta = reference.draw_d(comparisons)
    for ext in ['png', 'pdf', 'svg']:
        src = HISTORICAL / 'figures' / f'fig6-panele.{ext}'
        dst = OUT / 'figures' / src.name
        shutil.copy2(src, dst)
        assert src.read_bytes() == dst.read_bytes()
    compose()
    with (OUT / 'A_C_stage_correspondence.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['number', 'episode', 'phase', 'A_time_s', 'C_time_s', 'window_start_s', 'window_end_s', 'native_spike_count'])
        for state, frame in zip(states, spatial['frames']):
            assert state['number'] == frame['number'] and state['time_s'] == frame['time_s']
            writer.writerow([state['number'], state['episode'], state['label'], state['time_s'], frame['time_s'],
                *state['sample_window_s'], frame['native_spike_count']])
    metadata = dict(status='PENDING_AUTHOR_VISUAL_REVIEW', seed=SEED,
        selected_candidates=['R08', 'R09'], source_run=str(base.ROOT / 'runs' / base.RUN_NAME),
        same_run_panels=['A', 'B', 'C', 'D'], source_simulation_rerun=False, continuous_no_reset=True,
        waveform=wave, states=states, trajectory=trajectory, spatial_maps=spatial, D=dmeta,
        E_source=str(HISTORICAL / 'figures/fig6-panele.pdf'), E_scope='Unchanged historical 13-patient cohort',
        producer=str(Path(__file__).resolve()), formal_figure_pointer_modified=False)
    base.write(OUT / 'figure_metadata.json', metadata)
    base.write(OUT / 'field_comparisons.json', comparisons)
    checks = {}
    for path in (OUT / 'figures').glob('*.pdf'):
        with base.fitz.open(path) as doc:
            checks[path.name] = all(page.rect.contains(base.fitz.Rect(span['bbox'])) for page in doc
                for block in page.get_text('dict')['blocks'] if block['type'] == 0
                for line in block['lines'] for span in line['spans'])
    assert all(checks.values()), checks
    assert pointer.read_bytes() == pointer_bytes
    base.write(OUT / 'validation.json', dict(A_C_all_eight_times_and_fields_exact_match=True,
        spatial_regional_global_spike_counts_conserved=True, C_individual_row_colorbars=True,
        E_byte_identical=True, all_PDF_text_inside_canvas=checks, formal_pointer_unchanged=True,
        new_PNGs_rendered_from_delivered_PDF=True, human_visual_acceptance='PENDING'))
    for module in [panel_a, panel_a.previous, reference, base, reference.prior]:
        shutil.copy2(module.__file__, OUT / 'source' / Path(module.__file__).name)
    shutil.copy2(__file__, OUT / 'source' / Path(__file__).name)
    documents(metadata)
    print(json.dumps(dict(output=str(OUT), A_C_times=[s['time_s'] for s in states],
        patient_field_r=dmeta['correlations'], checks=checks), indent=2), flush=True)


if __name__ == '__main__':
    main()
