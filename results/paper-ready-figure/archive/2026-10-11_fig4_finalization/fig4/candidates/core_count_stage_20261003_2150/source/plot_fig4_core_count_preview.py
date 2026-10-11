#!/usr/bin/env python3
"""Refresh standalone core-count previews with the current Fig4C renderer.

Read live scores once, or replay --snapshot; never publish the current figure.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from scripts.paper_figures import build_fig4_current_aj as current

progress = current.base.progress
plate = current.plate
DEFAULT_OUT = ROOT / 'results/paper-ready-figure/fig4/candidates/core_count_refresh_20260930'


def digest(path):
    return progress.sha_bytes(Path(path).read_bytes())


def restore_cases(snapshot):
    reduce = progress.original_reducer()
    assert digest(snapshot['reducer']) == snapshot['reducer_sha256']
    cases = []
    entries = list(snapshot['cases'])
    if 'three_core_case' in snapshot:
        entries.append(snapshot['three_core_case'])
    if 'four_core_case' in snapshot:
        entries.append(snapshot['four_core_case'])
    for entry in entries:
        summary = entry['summary']
        arrays = {}
        for k in sorted(map(int, entry['records'])):
            reduced = [reduce(rows, summary['planned_epochs']) for rows in entry['records'][str(k)]]
            assert [n for _, n in reduced] == summary['prefix_by_restart'][str(k)]
            arrays[k] = np.asarray([arr for arr, _ in reduced])
        cases.append(dict(summary=summary, records=entry['records'], arrays=arrays))
    return cases


def collect_snapshot():
    manifest = {}
    reduce = progress.original_reducer()
    cases = [progress.load_case('E1146', progress.DATA / 'e1146_core_count_optimization_20260928',
                                'epilepsiae_1146', reduce, manifest)]
    root = progress.DATA / 'representative_core_count_20260929'
    for row in progress.read(root / 'selection.json')['selected']:
        cases.append(progress.load_case(row['display_id'], root, row['subject'], reduce, manifest))
    triple = progress.load_case('E1146', progress.DATA / 'e1146_core_count_optimization_20260928',
                                'epilepsiae_1146', reduce, manifest, core_counts=(1, 2, 3))
    reducer = ROOT / 'scripts/topic4_core_count_optimization/report.py'
    snapshot = dict(updated_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                cases=[dict(summary=c['summary'], records=c['records']) for c in cases],
                three_core_case=dict(summary=triple['summary'], records=triple['records']),
                score_sources_sha256=manifest, reducer=str(reducer), reducer_sha256=digest(reducer),
                semantics='Original best-so-far loss; common-prefix median and IQR across all displayed arms and restarts. '
                          'Thin lines retain their own completed prefix. Unscorable evaluations consume budget; '
                          'no fabricated loss, arm-specific normalization, or extrapolation.')
    if 4 in progress.read(progress.DATA/'e1146_core_count_optimization_20260928/plan.json')['core_counts']:
        four = progress.load_case('E1146', progress.DATA/'e1146_core_count_optimization_20260928',
                                  'epilepsiae_1146', reduce, manifest, core_counts=(1, 2, 3, 4))
        # Do not invent a fourth curve before all four searches have valid support.
        if four['summary']['common_epoch'] > 0 and all(v is not None for v in four['summary']['median_best_loss'].values()):
            snapshot['four_core_case'] = dict(summary=four['summary'], records=four['records'])
        else:
            snapshot['four_core_pending'] = four['summary']
    return snapshot


def validate_data(case):
    """Check every plotted point independently against persisted evaluations."""
    summary = case['summary']
    for k in sorted(case['arrays']):
        for restart, rows in enumerate(case['records'][str(k)]):
            prefix = summary['prefix_by_restart'][str(k)][restart]
            for epoch in range(1, prefix + 1):
                eligible = [r for r in rows if r['epoch'] <= epoch and r['J'] is not None]
                if eligible:
                    np.testing.assert_allclose(case['arrays'][k][restart, epoch - 1, 0],
                                               min(r['J'] for r in eligible), rtol=0, atol=0)
        np.testing.assert_allclose(np.median(case['arrays'][k][:, summary['common_epoch'] - 1, 0]),
                                   summary['median_best_loss'][str(k)], rtol=0, atol=1e-12)


def render_case(case, folder):
    # Same 51 x 45 mm plotting area, draw function, font sizes and export padding as current C.
    fig = plt.figure(figsize=(68 * plate.MM, 60 * plate.MM), dpi=160, facecolor='white')
    ax = fig.add_axes([14 / 68, 12 / 60, 51 / 68, 45 / 60])
    progress.draw(ax, case, show_title=False, paper_style=plate.FONT, show_legend=True)
    if case['summary']['subject'] == 'epilepsiae_1146':
        ax.set_yticks([0, 2, 4, 6, 8])
    fig.canvas.draw()
    # Additional subjects can have high, flat curves under the upper-right legend.
    # Add only the necessary vertical room; keep the current C renderer and labels.
    for _ in range(20):
        legend = ax.legend_.get_window_extent()
        overlaps = any(line.get_transform().transform_path(line.get_path()).intersects_bbox(legend, filled=False)
                       for line in ax.lines)
        if not overlaps:
            break
        low, high = ax.get_ylim()
        ax.set_ylim(low, low + (high - low) * 1.08)
        fig.canvas.draw()
    assert not overlaps, 'Legend overlaps a data curve.'
    assert ax.get_title() == '' and not fig.texts and not ax.texts
    assert (ax.get_xlabel(), ax.get_ylabel()) == ('Epochs', 'Loss')
    assert ax.xaxis.label.get_fontsize() == ax.yaxis.label.get_fontsize() == plate.FONT['label']
    assert [t.get_text() for t in ax.legend_.get_texts()] == [f'{k} core' + ('s' if k > 1 else '') for k in sorted(case['arrays'])]
    box = ax.get_window_extent()
    legend = ax.legend_.get_window_extent()
    assert box.x0 <= legend.x0 < legend.x1 <= box.x1
    assert box.y0 <= legend.y0 < legend.y1 <= box.y1
    np.testing.assert_allclose([box.width, box.height], np.array([51, 45]) * plate.MM * fig.dpi)
    # Check actual artists, in addition to the raw-record reduction.
    offset = 0
    common = case['summary']['common_epoch']
    for k in sorted(case['arrays']):
        block = case['arrays'][k][:, :, 0]
        for row in block:
            finite = np.flatnonzero(np.isfinite(row))
            np.testing.assert_array_equal(ax.lines[offset].get_xdata(), finite + 1)
            np.testing.assert_array_equal(ax.lines[offset].get_ydata(), row[finite])
            offset += 1
        shared = np.flatnonzero(np.isfinite(block[:, :common]).all(0))
        if len(shared) == 1:
            offset += 2  # The shared renderer adds two error-bar cap lines.
        np.testing.assert_array_equal(ax.lines[offset].get_xdata(), shared + 1)
        np.testing.assert_allclose(ax.lines[offset].get_ydata(), np.median(block[:, shared], axis=0), rtol=0, atol=0)
        offset += 1
    tight = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).padded(1.2 * plate.MM)
    stem = 'fig4-panelc' if case['summary']['subject'] == 'epilepsiae_1146' else case['summary']['display_id'].lower() + '_single_vs_dual_loss'
    if 3 in case['arrays']:
        stem = 'e1146_one_two_three_core_loss'
    if 4 in case['arrays']:
        stem = 'e1146_one_two_three_four_core_loss'
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(folder / f'{stem}.{ext}', dpi=320, bbox_inches=tight)
    plt.close(fig)
    return stem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--snapshot', type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    published = ROOT / 'results/paper-ready-figure/fig4'
    assert out != published and out not in (published / 'figures', published / 'source')
    protected = [p for sub in ('figures', 'source') for p in (published / sub).rglob('*') if p.is_file()]
    protected += [p for p in published.iterdir() if p.is_file()]
    before = {str(p): digest(p) for p in protected}
    figdir = out / 'figures'
    figdir.mkdir(parents=True, exist_ok=True)
    source = out / 'source'
    source.mkdir(exist_ok=True)
    snapshot = progress.read(args.snapshot) if args.snapshot else collect_snapshot()
    progress.write(source / 'core_count_input_snapshot.json', snapshot)
    # Reload exactly the saved JSON that can later be reviewed and promoted.
    snapshot = progress.read(source / 'core_count_input_snapshot.json')
    cases = restore_cases(snapshot)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7, 'legend.frameon': False})
    notes = []
    rows = []
    for case in cases:
        validate_data(case)
        stem = render_case(case, figdir)
        s = case['summary']
        variant = '_cores1234' if 4 in case['arrays'] else '_cores123' if 3 in case['arrays'] else ''
        counts = '单核／双核／三核／四核' if 4 in case['arrays'] else '单核／双核／三核' if variant else '单核／双核'
        stage = '初始采样' if s['stage'] == 'initial_sampling' else '含后续搜索评估'
        np.savez_compressed(source / f'{s["subject"]}{variant}_plot_arrays.npz', **{f'core{k}': a for k, a in case['arrays'].items()})
        notes.append(f'### {stem}.png\n{s["display_id"]} 的{counts}最佳训练 loss，复用当前 Figure 4C 的绘图函数、字号、颜色和轴标签，另有同源 PDF/SVG。'
                     f'每组 {s["restarts"]} 次优化重复，粗线和四分位色带使用图中所有组共同完成的前 {s["common_epoch"]} 个 epoch；细线显示各自完整已评估前缀。\n'
                     f'**关注点**：无标题或进度图注；当前为{stage}，本候选待作者目视检查，尚未同步正式 Figure 4。')
        rows.append(dict(subject=s['subject'], display_id=s['display_id'], common_epoch=s['common_epoch'],
                         planned_epochs=s['planned_epochs'], restarts=s['restarts'],
                         single_core_median=s['median_best_loss']['1'], dual_core_median=s['median_best_loss']['2'],
                         three_core_median=s['median_best_loss'].get('3'), core_counts=','.join(map(str, sorted(case['arrays']))),
                         four_core_median=s['median_best_loss'].get('4'),
                         reduction_percent=s['reduction_percent'], stage=s['stage']))
    (figdir / 'README.md').write_text('\n\n'.join(notes) + '\n')
    with (out / 'comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert all(digest(p) == value for p, value in before.items()), 'Published Fig4 changed during preview generation.'
    assert all(digest(p) == value for p, value in snapshot['score_sources_sha256'].items())
    modules = [current, current.base, progress, plate]
    sources = {mod.__file__: digest(mod.__file__) for mod in modules}
    sources[str(Path(__file__))] = digest(__file__)
    for path in sources:
        (source / Path(path).name).write_bytes(Path(path).read_bytes())
    progress.write(out / 'preview_registry.json', dict(status='CANDIDATE_PENDING_AUTHOR_VISUAL_REVIEW',
        data_time=snapshot['updated_at'], producer=str(Path(__file__).relative_to(ROOT)),
        current_producer=str(Path(current.__file__).relative_to(ROOT)),
        drawing_function='scripts.paper_figures.plot_core_count_optimization_progress.draw',
        drawing_function_sha256=progress.sha_bytes(inspect.getsource(progress.draw).encode()),
        source_files_sha256=sources, panel_axis_size_mm=[51, 45], font_pt=plate.FONT,
        labels=dict(x='Epochs', y='Loss'), title=False, caption=False, comparison=rows,
        current_package_unchanged=True, protected_files_sha256=before,
        outputs_sha256={str(p.relative_to(out)): digest(p) for p in figdir.iterdir() if p.suffix in ('.png', '.pdf', '.svg')}))
    progress.write(out / 'visual_qa.json', dict(status='NUMERICAL_AND_LAYOUT_PASS_VISUAL_PENDING',
        actual_artist_values_checked=True, raw_record_minima_checked=True, original_reducer_used=True,
        current_panel_renderer_used=True, current_package_unchanged=True, human_visual_acceptance='PENDING'))
    print(snapshot['updated_at'])
    for row in rows:
        print(row)
    print(out)


if __name__ == '__main__':
    main()
