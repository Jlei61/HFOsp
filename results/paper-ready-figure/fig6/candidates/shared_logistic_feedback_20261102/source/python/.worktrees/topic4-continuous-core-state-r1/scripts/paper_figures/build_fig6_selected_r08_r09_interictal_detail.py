#!/usr/bin/env python3
"""Adjust the interictal viewport using the unchanged saved state trajectory."""
import json
from pathlib import Path
import shutil

import build_fig6_selected_r08_r09_recovery_references as previous

base, parent = previous.base, previous.parent
np, plt = previous.np, previous.plt
SOURCE = previous.OUT
OUT = SOURCE.with_name('selected_r08_r09_interictal_detail_20261031')
BOUNDS = np.array([[.75, 1.], [0., 1.5], [0., 200.]])
INTERVALS = [(0., 8.71), (46.95, 59.109), (107.35, 118.)]


def interictal_ranges(t, xyz):
    records = []
    for lo, hi in INTERVALS:
        take = (t >= lo) & (t < hi)
        values = xyz[take]
        visible = np.all((values >= BOUNDS[:, 0]) & (values <= BOUNDS[:, 1]), axis=1)
        assert visible.all(), 'The chosen viewport must cover these entire interictal intervals.'
        records.append(dict(time_s=[lo, hi], samples=int(take.sum()),
            minimum_Z_etaM_rate=values.min(0), maximum_Z_etaM_rate=values.max(0),
            all_sample_rate_percentile95_Hz=float(np.percentile(values[:, 2], 95)),
            fraction_within_interictal_viewport=float(visible.mean())))
    return records


def draw_detail(t, xyz, states):
    take = t < previous.HORIZON
    t, xyz = t[take], xyz[take]
    fig = plt.figure(figsize=(7.6, 6.3))
    ax = fig.add_axes([.075, .10, .73, .80], projection='3d', computed_zorder=False)
    segments, keep = previous.add_curve(ax, t, xyz, BOUNDS)
    ax.view_init(elev=22, azim=-115)
    ax.set_box_aspect((1.55, 1., 1.25), zoom=1.06)
    ax.set(xlabel='$Z$', ylabel=r'$\eta_M M$ (mV)', xticks=[.75, .875, 1.],
        yticks=[0, .75, 1.5], zticks=[0, 100, 200])
    ax.set_yticklabels(['', '0.75', '1.5'])
    ax.tick_params(labelsize=12, pad=1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_fontsize(14)
        axis.labelpad = 6
    fig.text(.025, .57, r'$r_E$ (Hz)', fontsize=14, rotation=90, va='center')
    fig.text(.075, .975, 'Interictal', fontsize=17, va='top')
    labels = parent.reference.prior.label_states(ax, states[:3], t, xyz, fontsize=12)
    cb = fig.colorbar(base.ScalarMappable(norm=base.Normalize(0, previous.HORIZON), cmap='viridis'),
        cax=fig.add_axes([.89, .21, .024, .49]), ticks=[0, 60, previous.HORIZON])
    cb.set_label('Time (s)', fontsize=13, labelpad=5)
    cb.ax.tick_params(labelsize=12, length=3, pad=3)
    previous.check_projected(fig, ax, segments)
    parent.export(fig, 'fig6-panelb-interictal')
    return dict(limits=BOUNDS, title='Interictal', segment_count=int(keep.sum()),
        no_bridging_across_out_of_view_segments=True, states=labels,
        observation='Same all-E mean trajectory, 5-ms rate bins and 5-ms sigma smoothing; no changed data or observer.')


def main():
    pointer = base.MAIN/'results/paper-ready-figure/fig6/current_version.json'
    pointer_before = pointer.read_bytes()
    shutil.copytree(SOURCE, OUT, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns('visual_review.json', 'validation.json'))
    meta = json.loads((SOURCE/'figure_metadata.json').read_text())
    with np.load(SOURCE/'trajectory_data.npz') as data:
        t, xyz = data['time_s'].copy(), data['Z_etaM_rate'].copy()
    ranges = interictal_ranges(t, xyz)
    base.OUT = parent.OUT = previous.OUT = OUT
    plt.rcParams.update({'font.size': 13, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'path.simplify': False})
    trajectory = previous.draw_b_trajectory(t, xyz, meta['states'],
        inset_bounds=BOUNDS, inset_title='Interictal', export_full=False)
    trajectory['outputs']['fig6-panelb-full-trajectory'] = meta['trajectory']['outputs']['fig6-panelb-full-trajectory']
    trajectory['outputs']['fig6-panelb-interictal'] = draw_detail(t, xyz, meta['states'])
    trajectory.update(interictal_interval_ranges=ranges, inset_title='Interictal',
        inset_bounds_rationale='All-E interictal rate peaks are 166-183 Hz and early-return eta_M M reaches 1.39 mV; use 200 Hz and 1.5 mV to retain entire interictal intervals.')
    previous.compose()
    meta.update(producer=str(Path(__file__).resolve()), previous_candidate=str(SOURCE),
        trajectory=trajectory, status='PENDING_AUTHOR_VISUAL_REVIEW',
        revision_scope='B interictal viewport and title only; no simulation or observation recomputation.')
    base.write(OUT/'figure_metadata.json', meta)
    base.write(OUT/'interictal_range_summary.json', dict(
        observable='All-E mean Z, eta_M M, and 5-ms-binned rate smoothed with sigma=5 ms',
        statistical_unit='One continuous seed 9108404 realization; three descriptive interictal intervals',
        previous_viewport=[[.70, 1.], [0., .1], [0., 250.]], viewport=BOUNDS, intervals=ranges))
    copied = []
    for stem in ['fig6-panela', 'fig6-panelc', 'fig6-paneld', 'fig6-panele', 'fig6-panelb-full-trajectory']:
        for ext in ['png', 'pdf', 'svg']:
            filename = f'{stem}.{ext}'
            assert (OUT/'figures'/filename).read_bytes() == (SOURCE/'figures'/filename).read_bytes()
            copied.append(filename)
    with np.load(OUT/'trajectory_data.npz') as data:
        np.testing.assert_array_equal(data['time_s'], t)
        np.testing.assert_array_equal(data['Z_etaM_rate'], xyz)
    checks = {}
    for stem in ['fig6-panelb', 'fig6-panelb-interictal', 'fig6-complete-layout']:
        with base.fitz.open(OUT/'figures'/f'{stem}.pdf') as doc:
            checks[stem] = all(page.rect.contains(base.fitz.Rect(span['bbox'])) for page in doc
                for block in page.get_text('dict')['blocks'] if block['type']==0
                for line in block['lines'] for span in line['spans'])
            assert 'Interictal zoom' not in ''.join(page.get_text() for page in doc)
    assert all(checks.values()), checks
    assert pointer.read_bytes() == pointer_before
    validation = json.loads((SOURCE/'validation.json').read_text())
    validation.update(previous_candidate=str(SOURCE), same_trajectory_arrays=True,
        unchanged_panel_files=copied, all_interictal_samples_inside_new_view=True,
        new_PDF_text_inside_canvas=checks, human_visual_acceptance='PENDING')
    base.write(OUT/'validation.json', validation)
    caption = (SOURCE/'caption_zh.md').read_text()
    old = '小窗线性放大 Z=0.70–1、eta_M M=0–0.10 mV、r_E=0–250 Hz'
    assert old in caption
    caption = caption.replace(old, '小窗标题为 Interictal，线性显示 Z=0.75–1、eta_M M=0–1.5 mV、r_E=0–200 Hz')
    caption += '\n本次只调整间期小窗，原始轨迹与其他面板保持。三个间期窗口内的全体 E 平均率最大值约 183、166、178 Hz，返回后初期间期的 eta_M M 最大约 1.39 mV；新范围覆盖这三个窗口内的全部采样点。单独放大的同范围图见 fig6-panelb-interictal。\n'
    (OUT/'caption_zh.md').write_text(caption)
    readme = (SOURCE/'figures/README.md').read_text()
    readme += '\n\n### fig6-panelb-interictal.png / .pdf / .svg\n\n独立放大 B 中的 Interictal 小窗，Z=0.75–1、全体 E 平均放电率 0–200 Hz、eta_M M=0–1.5 mV。三个间期窗口内的全部采样点均在新范围内，色条仍为原始时间。\n**关注点**：间期峰值可超过 100 Hz，且刚返回时残余适应电流超过旧小窗的 0.1 mV；本次范围拓宽使这部分轨迹可见。\n'
    (OUT/'figures/README.md').write_text(readme)
    (OUT/'README.md').write_text(f'# Figure 6：Interictal 小窗范围修订\n\n'
        '[B 图](figures/fig6-panelb.png) · [间期细节](figures/fig6-panelb-interictal.png) · '
        '[完整 Figure 6](figures/fig6-complete-layout.pdf)\n\n'
        '只调整小窗标题和三轴范围，完整轨迹、原始数据、A/C/D/E、全范围 B 大图均保持。当前候选待作者目视检查。\n\n'
        f'复现：`python {Path(__file__).resolve()}`。\n')
    shutil.copy2(__file__, OUT/'source'/Path(__file__).name)
    shutil.copy2(previous.__file__, OUT/'source'/Path(previous.__file__).name)
    print(json.dumps(dict(output=str(OUT), bounds=BOUNDS.tolist(),
        interval_rate_max_Hz=[r['maximum_Z_etaM_rate'][2] for r in ranges], checks=checks), indent=2))


if __name__ == '__main__':
    main()
