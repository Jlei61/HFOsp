#!/usr/bin/env python3
"""Show all native long-run seizure entries in the current Fig6A readout style.

No simulation, event reclassification, or formal figure-pointer changes. Filter
each complete 120-s readout once; verify the adapter against the original Fig6A
producer. Candidate identifiers do not imply a seizure class.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.signal import butter, sosfiltfilt

MAIN = Path('/home/honglab/leijiaxin/HFOsp')
RAW = Path('/data/hfosp/topic4_sef_hfo/adaptation_history_20261031')
SOURCE = MAIN / 'results/topic4_sef_hfo/adaptation_history_20261031/long120s/results.json'
STYLE = MAIN / 'results/paper-ready-figure/fig6/candidates/unified_current_adaptation_origin_review_s9108402_20261031'
OUT = MAIN / 'results/paper-ready-figure/fig6/candidates/seizure_selection_history_20261031'
sys.path.insert(0, str(MAIN / 'scripts/paper_figures'))
import build_fig5_fig4f_historical_zm as native
try:
    import pymupdf as fitz
except ImportError:
    sys.path.insert(0, '/tmp/hfosp_fig5_pdf_tools')
    import pymupdf as fitz

ORDER = list(native.layout.previous.f.ORDER)
BAND = (80., 1000.)
GAIN = .65
DURATION = 2.0
COUNTS = 13


def write_json(path, value):
    def default(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x))
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=default) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_seed(seed):
    files = sorted((RAW / f'runs/seed{seed}/dense_readout_chunks').glob('*.npz'))
    times, signals = [], []
    for path in files:
        with np.load(path) as a:
            times.append(a['time_ms'])
            signals.append(a['contact_current'])
    t = np.concatenate(times)
    raw = np.concatenate(signals)
    assert raw.shape == (1200000, 15)
    assert np.allclose(np.diff(t), .1, rtol=0, atol=1e-7)
    with np.load(RAW / 'geometry.npz') as a:
        names = a['contact_names'].tolist()
    return t, raw, names, files


def event_window(ep, briefs):
    # Keep at least the last two detected IEDs and the same time scale in every
    # candidate. Start no later than 0.60 s before local reactivation.
    preceding = [b for b in briefs if b['end_s'] <= ep['process_start_s']]
    assert len(preceding) >= 2
    start = min(ep['process_start_s'] - .60, preceding[-2]['start_s'] - .06)
    start = min(np.floor(start * 100) / 100, 120. - DURATION)
    assert ep['global_high_s'] + .15 <= start + DURATION
    assert start + DURATION <= 120.
    return [float(start), float(start + DURATION)], [
        {key: b[key] for key in ('start_s', 'end_s', 'core_order')}
        for b in preceding[-2:]
    ]


def short_metadata(e):
    energy = e['energy']
    sensitivity = []
    if not energy['stable_offsets']:
        sensitivity.append('window-sensitive')
    if not energy['preentry_baseline_sensitivity']['agrees_with_primary']:
        sensitivity.append('baseline-sensitive')
    extra = ' (' + ', '.join(sensitivity) + ')' if sensitivity else ''
    return (f"Core: early {e['early_core_leader']} / recruit {e['core_leader']} / "
            f"sustained {e['sustained_core_leader']}   |   Field: {energy['type']}{extra}")


def draw(ax, item, left=True):
    e = item['episode']
    t, values = item['t'], item['values']
    window = item['window_s']
    for i in range(COUNTS):
        ax.plot(t, values[:, i], color='#27929d' if i < 4 else '#c67b35',
                lw=.6, rasterized=True)
    ax.axhline(8.5, c='#ccc', lw=.8)
    ax.axvspan(e['process_start_s'], window[1], fc='#ba263c', alpha=.065, lw=0, zorder=0)
    ax.axvline(e['process_start_s'], color='#888', ls=':', lw=.85, zorder=1)
    # Core-specific times remain in metadata rather than drawing potentially
    # confusing whole-channel vertical lines across the electrode waveforms.
    ax.set(xlim=window, yticks=np.arange(12, -1, -1), yticklabels=ORDER[:COUNTS],
           ylabel='SEG readout' if left else '', xlabel='')
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(axis='x', labelsize=12, pad=3)
    ax.tick_params(axis='y', labelsize=12, length=3)
    ax.spines[['top', 'right']].set_visible(False)
    if not left:
        ax.tick_params(axis='y', left=False, labelleft=False)
        ax.spines['left'].set_visible(False)
    return [min(-.65, float(values.min()) - .12), max(13., float(values.max()) + .12)]


def export(fig, stem):
    pdf = OUT / 'figures' / f'{stem}.pdf'
    png = pdf.with_suffix('.png')
    fig.savefig(pdf, dpi=220, facecolor='white')
    plt.close(fig)
    # The preview is rendered from the delivered PDF, so both have the same
    # geometry, labels, waveforms, and viewport.
    with fitz.open(pdf) as doc:
        assert len(doc) == 1
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(220/72, 220/72), alpha=False)
        pix.save(png)
    return {'pdf': str(pdf), 'png': str(png)}


def single(item):
    e = item['episode']
    fig = plt.figure(figsize=(12.4, 5.6))
    ax = fig.add_axes([.085, .125, .89, .71])
    ax.set_ylim(draw(ax, item))
    end_note = '   |   Ongoing at 120 s' if e['right_censored'] else ''
    fig.text(.085, .953, f"{item['id']}   |   Seed {e['seed']}   |   Seizure {e['cycle']}" + end_note,
             fontsize=13, weight='bold')
    fig.text(.085, .905, short_metadata(e), fontsize=11, color='#444')
    fig.text(.53, .055, 'Time (s)', ha='center', fontsize=14)
    return export(fig, item['id'])


def pair(items, stem):
    assert len(items) == 2
    assert items[0]['episode']['seed'] == items[1]['episode']['seed']
    assert items[0]['episode']['cycle'] + 1 == items[1]['episode']['cycle']
    fig = plt.figure(figsize=(15.7, 5.65))
    gap = .065
    width = (.905 - gap) / 2
    axes = [fig.add_axes([.075 + k * (width + gap), .15, width, .66]) for k in range(2)]
    limits = []
    for ax, item, k in zip(axes, items, range(2)):
        limits.append(draw(ax, item, left=k == 0))
        x = ax.get_position().x0
        e = item['episode']
        fig.text(x, .94, f"{item['id']}   |   Seizure {e['cycle']}   |   Field {e['energy']['type']}",
                 fontsize=13, weight='bold')
        fig.text(x, .893, f"Core: early {e['early_core_leader']} / recruit {e['core_leader']} / "
                 f"sustained {e['sustained_core_leader']}", fontsize=11, color='#444')
        notes = []
        if not e['energy']['stable_offsets']: notes.append('Field label: window-sensitive')
        if not e['energy']['preentry_baseline_sensitivity']['agrees_with_primary']:
            notes.append('Field label: baseline-sensitive')
        if e['right_censored']:
            notes.append('Ongoing at 120 s')
        if notes: fig.text(x, .851, '; '.join(notes), fontsize=10, color='#777')
    for ax in axes:
        ax.set_ylim(min(v[0] for v in limits), max(v[1] for v in limits))
    for ax, x in ((axes[0], 1.), (axes[1], 0.)):
        for y in (0., 1.):
            ax.plot([x-.009, x+.009], [y-.012, y+.012], transform=ax.transAxes,
                    color='#555', lw=1.1, clip_on=False)
    fig.text(.5275, .09, 'Time (s)', ha='center', va='top', fontsize=16)
    fig.text(.975, .025, f"Seed {items[0]['episode']['seed']}  |  same continuous simulation",
             ha='right', fontsize=10, color='#777')
    return export(fig, stem)


def validate_native(t_ms, raw, names, signal, gains, episode, window):
    native.MAIN_BAND_HZ = BAND
    native.GAIN_PERCENTILE = 99.5
    native.DISPLAY_GAIN = GAIN
    native.RECRUIT = episode['global_high_s']
    native.WINDOW = tuple(window)
    fig, ax = plt.subplots()
    record = native.draw_a(ax, {'contact_names': names, 'lfp_native_time_ms': t_ms,
                               'lfp_native': raw}, [], episode['process_start_s'])
    t = t_ms / 1000
    select = (t >= window[0]) & (t <= window[1])
    expected = (signal[select, :COUNTS] / gains[:COUNTS] * GAIN + np.arange(14, 1, -1)) - 2
    original = np.column_stack([line.get_ydata() - 2 for line in ax.lines[:COUNTS]])
    np.testing.assert_array_equal(expected, original)
    np.testing.assert_array_equal(gains, record['gain_per_contact'])
    plt.close(fig)
    old = json.loads((STYLE / 'figure_metadata.json').read_text())['waveform']['segments'][0]
    relative_error = float(np.max(np.abs(gains / old['gain_per_contact'] - 1)))
    assert relative_error < 1e-10
    return {'native_renderer_max_abs_error': float(np.max(abs(expected - original))),
            'gain_max_relative_difference_from_prior_fig6A': relative_error,
            'native_renderer': str(Path(native.__file__)), 'prior_fig6A': str(STYLE)}


def main():
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    (OUT / 'source').mkdir(exist_ok=True)
    (OUT / 'waveform_windows').mkdir(exist_ok=True)
    data = json.loads(SOURCE.read_text())
    items = []
    seeds = []
    validation = None
    idx = 0
    for run in data:
        seed = run['seed']
        print(f'Loading and filtering seed {seed}', flush=True)
        t_ms, raw, names, paths = load_seed(seed)
        order_indices = [names.index(n) for n in ORDER]
        signal = sosfiltfilt(butter(4, BAND, btype='bandpass', fs=10000, output='sos'),
                            np.asarray(raw[:, order_indices], float), axis=0)
        t = t_ms / 1000
        recruit = run['episodes'][0]['global_high_s']
        scales = np.percentile(abs(signal[t < recruit]), 99.5, axis=0)
        gains = np.maximum(scales, .15 * np.median(scales[scales > 0]))
        entries = [e for e in run['episodes'] if e['episode_role'] != 'rebound_without_interictal_return']
        for e in entries:
            idx += 1
            window, preceding = event_window(e, run['briefs'])
            select = (t >= window[0]) & (t <= window[1])
            if seed == 9108402 and e['cycle'] == 1:
                validation = validate_native(t_ms, raw, names, signal, gains, e, window)
            item = {'id': f'R{idx:02d}', 'episode': e, 'window_s': window,
                    'preceding_IEDs': preceding, 't': t[select].copy(),
                    'values': (signal[select, :COUNTS] / gains[:COUNTS] * GAIN + np.arange(14, 1, -1)) - 2,
                    'gains': gains.copy()}
            np.savez_compressed(OUT / 'waveform_windows' / f"{item['id']}.npz",
                                time_s=item['t'], filtered_readout=signal[select],
                                gain_per_contact=gains, contact_names=np.array(ORDER))
            item['files'] = single(item)
            items.append(item)
            print(f"  {item['id']} cycle {e['cycle']}: core {e['core_leader']}, field {e['energy']['type']}", flush=True)
        seeds.append({'seed': seed, 'raw_files': [str(p) for p in paths], 'samples': len(t),
                      'observed_s': run['observed_s'], 'normalization_cutoff_s': recruit,
                      'gain_per_contact': gains, 'full_trajectory_filtering': True})
        del raw, signal, t, t_ms
    assert len(items) == 21 and validation is not None
    # Every adjacent true cycle is available as a paired page, even when the
    # two field labels or leading cores agree. Never splice different seeds.
    pairs = []
    for run in data:
        selected = [it for it in items if it['episode']['seed'] == run['seed']]
        for first, second in zip(selected[:-1], selected[1:]):
            stem = f"pair_{first['id']}_{second['id']}"
            np.testing.assert_array_equal(first['gains'], second['gains'])
            files = pair([first, second], stem)
            pairs.append({'ids': [first['id'], second['id']], 'seed': run['seed'], 'files': files})
    atlas = fitz.open()
    for item in items:
        with fitz.open(item['files']['pdf']) as doc:
            atlas.insert_pdf(doc)
        atlas.set_page_labels([{'startpage': 0, 'prefix': 'R', 'style': 'D', 'firstpagenum': 1}])
    atlas.set_metadata({'title': 'Fig6A waveform candidates: all 21 native seizure entries',
                        'subject': 'E1146; eight continuous 120 s runs; author selection pending'})
    atlas.save(OUT / 'seizure_candidates_all21.pdf', garbage=4, deflate=True)
    atlas.close()
    manifest_items = [{k: v for k, v in it.items() if k not in ('t', 'values', 'gains')} for it in items]
    manifest = {'status': 'PENDING_AUTHOR_SELECTION', 'source': str(SOURCE), 'source_sha256': sha(SOURCE),
                'style_parent': str(STYLE), 'sampling_Hz': 10000, 'band_Hz': BAND,
                'zero_phase': True, 'filter_order': 4, 'common_average_reference': False,
                'baseline_subtraction': False, 'gain_percentile': 99.5, 'display_gain': GAIN,
                'gain_fixed_within_seed': True, 'contact_order': ORDER, 'display_contacts': ORDER[:COUNTS],
                'display_only_removed': ORDER[COUNTS:], 'duration_each_panel_s': DURATION,
                'entry_marker': 'Last quiet end before local reactivation; not a four-stage state marker.',
                'four_stage_semantics_unchanged': ['Local', 'Spread', 'Entry', 'Ictal'],
                'numbered_stage_markers': 'Deferred until author selects a candidate.',
                'core_classification': {'early': '20 Hz for 10 ms', 'recruit': '100 Hz for 50 ms',
                                        'sustained': '100 Hz for 200 ms'},
                'field_classification': 'Existing E/F energy-field classification, copied without modification.',
                'event_selection': 'All true seizure-cycle entries; exclude short rebounds without interictal return.',
                'seeds': seeds, 'candidates': manifest_items, 'pairs': pairs, 'validation': validation,
                'pdf_png_same_state': 'Every PNG directly rendered from its delivered PDF.',
                'formal_figure_pointers_modified': False}
    write_json(OUT / 'manifest.json', manifest)
    with (OUT / 'candidate_index.csv').open('w', newline='') as handle:
        fields = ['id', 'seed', 'cycle', 'entry_s', 'window_start_s', 'window_end_s', 'early_core',
                  'recruit_core', 'sustained_core', 'field', 'field_window_stable', 'field_local_baseline_agrees']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for it in items:
            e = it['episode']
            writer.writerow(dict(zip(fields, [it['id'], e['seed'], e['cycle'], e['process_start_s'],
                *it['window_s'], e['early_core_leader'], e['core_leader'], e['sustained_core_leader'],
                e['energy']['type'], e['energy']['stable_offsets'],
                e['energy']['preentry_baseline_sensitivity']['agrees_with_primary']])))
    shutil.copy2(__file__, OUT / 'source' / Path(__file__).name)
    shutil.copy2(SOURCE, OUT / 'source' / 'results.json')
    readme = [
        '# Fig6A 发作候选，待作者选择', '',
        '复用 8 个种子的 120 秒连续仿真，列出全部 21 次真正进入新发作周期的事件；5 次尚未返回间期的短暂反弹不混入。没有重跑仿真，也没有改动正式 Figure 6。', '',
        '波形严格沿用最近 Fig6A：80–1000 Hz、四阶零相位滤波，先对完整 120 秒滤波再截窗；不做 CAR 或基线相减。固定显示 SCL9–SCL6 与 ICL11–ICL3，ICL2/ICL1 仅在显示中隐藏。同一种子所有发作共用首次高活动之前的 P99.5 增益及 0.65 显示倍数，跨种子使用同一规则而不是同一数值增益。每窗 2 秒，保留末两次被检测的间期事件与局部再激活。', '',
        '灰色竖虚线与淡红色底纹从最后静默结束后的局部再激活开始。候选 R 编号只用于选择，没有更改 Local → Spread → Entry → Ictal 四阶段语义；具体 1–4/5–8 取帧待选定事件后复核。', '',
        'Core early/recruit/sustained 分别是两个 core 首次满足 20 Hz 持续 10 ms、100 Hz 持续 50 ms、100 Hz 持续 200 ms 的先后；这些是模型局部群体读出。Field TA/TB 是原 E/F 15 通道能量场分类，两者不等同。window-sensitive 指原分析的时间偏移会改变分类；baseline-sensitive 指用局部发作前基线替代原基线会改变分类。这里只展示既有结果，不重新分类。', '',
        '同种子相邻发作单独提供并排页，横轴是连续仿真的绝对时间，中间用断轴表示省略的时间。所有候选逐页 PDF 为 seizure_candidates_all21.pdf，精确时间与标签见 candidate_index.csv，逐样本截窗数据见 waveform_windows。', '',
        'R10、R16、R21 在 120 秒记录结束时尚未终止，因此适合选择进入过程，但这段记录不能提供其完整终止图。其他候选的退出时间保留在 manifest.json。', '',
        '|编号|种子|第几次发作|局部再激活(s)|early / recruit / sustained|Field|',
        '|---|---|---|---|---|---|',
    ]
    for it in items:
        e = it['episode']
        readme.append(f"|[{it['id']}](figures/{it['id']}.png)|{e['seed']}|{e['cycle']}|{e['process_start_s']:.3f}|"
                      f"{e['early_core_leader']} / {e['core_leader']} / {e['sustained_core_leader']}|{e['energy']['type']}|")
    (OUT / 'README.md').write_text('\n'.join(readme) + '\n')
    fig_readme = ['# 候选波形', '']
    for it in items:
        e = it['episode']
        fig_readme.extend([f"### {it['id']}.png", '',
            f"种子 {e['seed']} 的第 {e['cycle']} 次发作，展示 {it['window_s'][0]:.2f}–{it['window_s'][1]:.2f} 秒的 Fig6A 式电极波形；同名 PDF 与 PNG 完全同源。",
            f"最早活动、50 ms 招募、200 ms 持续招募的领先 core 分别为 {e['early_core_leader']}、{e['core_leader']}、{e['sustained_core_leader']}，E/F 能量场标签为 {e['energy']['type']}。",
            '**关注点**：末次间期事件结束后各电极的局部再激活顺序，以及它与 core 招募和场分类是否一致。', ''])
    for p in pairs:
        stem = Path(p['files']['png']).name
        fig_readme.extend([f'### {stem}', '',
            f"同一连续仿真（种子 {p['seed']}）的相邻发作 {p['ids'][0]} 与 {p['ids'][1]} 并排。两窗共用通道增益、纵轴范围和 2 秒时间尺度，未拼接不同种子；同名 PDF 为原始输出。",
            '**关注点**：比较实际波形，不把 core 先行标签自动等同于 TA/TB 能量场标签。', ''])
    (OUT / 'figures' / 'README.md').write_text('\n'.join(fig_readme))
    print(json.dumps({'output': str(OUT), 'candidates': len(items), 'pairs': len(pairs),
                      'validation': validation}, indent=2), flush=True)


if __name__ == '__main__':
    main()
