#!/usr/bin/env python3
"""Annotate event boundaries so nearby activity is not attributed to a brief."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import shutil
import numpy as np
from campaign import ROOT, read, write, sha
import analyze_constant_tau_return as analysis
import zoom_topic4_return_core_propagation as zoom

OUT = analysis.OUT/'spatial_sequence_v2'


def main():
    OUT.mkdir(exist_ok=True)
    r = read(analysis.DEST/'result.json');d = dict(np.load(analysis.DEST/'readouts.npz'))
    review = read(analysis.OUT/'scientific_review/result.json');geo = dict(np.load(analysis.OUT/'geometry.npz'))
    original = next(e for e in r['strict_pre']['brief_events'] if e['start_s'] <= .58+1e-9 and e['end_s'] > .58)
    events = r['episodes'][0]['after_reference_events']['brief_events']
    selected = [('Original interictal reference', .57, original)]
    selected += [(f'Returned event {i+1}', e['start_s'], e) for i, e in enumerate(events[:3])]
    weak = review['first_weak_events'][0];selected.append(('First weak returned event', weak['start_s'], weak))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(20, 13.8));grid = fig.add_gridspec(len(selected), 10,
        left=.06, right=.93, bottom=.07, top=.92, wspace=.4, hspace=.65,
        width_ratios=[1.55, 1.55, 1, 1, 1, 1, 1, 1, 1, 1])
    offsets = list(range(0, 200, 25));arrays = {}
    for row, (label, start, event) in enumerate(selected):
        ax = fig.add_subplot(grid[row, :2]);zoom.trace(ax, dict(rates=d['rates5_Hz']), start, start+.2, relative=True)
        onset = (event['start_s']-start)*1000;end = (event['end_s']-start)*1000
        if onset > 0:ax.axvspan(0, onset, color='.9', alpha=.6)
        ax.axvspan(end, 200, color='.9', alpha=.6);ax.axvline(end, color='.45', ls=':', lw=.9)
        ax.set_title(label+f'\nt = {start:.3f} s; event {event["duration_s"]*1000:.0f} ms', fontsize=10)
        if row == 0:ax.legend(frameon=False, fontsize=7, ncol=2, loc='upper right')
        if row == len(selected)-1:ax.set_xlabel('Time from window start (ms)')
        frames = []
        for col, offset in enumerate(offsets):
            a = fig.add_subplot(grid[row, col+2]);value = d['field5_Hz'][round(start*200)+offset//5]
            im = zoom.field(a, value, geo);frames.append(value)
            context = offset >= end-1e-8 or offset+5 <= onset+1e-8
            a.set_title(f'+{offset}–{offset+5} ms'+('\nContext' if context else ''), fontsize=8, color='.45' if context else 'black')
            if col:a.set_yticklabels([])
            else:a.set_ylabel('y (mm)', fontsize=9)
            if row == len(selected)-1:a.set_xlabel('x (mm)', fontsize=9)
        arrays[f'row{row}_fields'] = np.array(frames)
    fig.colorbar(im, cax=fig.add_axes([.947, .23, .012, .51]), label='Native E rate (Hz; 5-ms bins)')
    fig.suptitle('Returned brief events and surrounding activity: native spatial sequence', fontsize=14)
    fig.text(.5, .955, 'Common 200-ms windows; shaded trace and Context frames lie outside the detected event.', ha='center', fontsize=10)
    for ext in ['png', 'svg']:fig.savefig(ROOT/f'figures/constant_tau_return_spatial_sequence_v2.{ext}', dpi=170)
    plt.close(fig)
    np.savez_compressed(OUT/'display_data.npz', **arrays)
    write(OUT/'result.json', dict(status='COMPLETE_EVENT_BOUNDARY_ANNOTATED_SPATIAL_CANDIDATE', selected=selected,
        common_window_ms=200, frame_bin_ms=5, supersedes_display='constant_tau_return_spatial_sequence.png',
        event_definitions_unchanged=True, analysis_unchanged=True, agent_visual='PENDING', human_visual='PENDING', producer_sha256=sha(__file__)))
    shutil.copy2(__file__, OUT/'producer.py')
    title = '### constant_tau_return_spatial_sequence_v2.png / constant_tau_return_spatial_sequence_v2.svg';p = ROOT/'figures/README.md'
    if title not in p.read_text():
        with p.open('a') as f:f.write('\n\n'+title+'\n保留前版相同原示例和按时间选取的返回事件，统一显示200ms以覆盖180ms事件；8帧均为真实5ms空间计数。率图用灰色标出检测事件之外的时间，空间帧注明Context，避免将弱事件之后的新核活动算入该弱事件。\n**关注点**：核外起始、核招募与核主导起燃分别判断；此版取代前版150ms图的显示，但不改变任何事件定义、指标或验收结论，人工待审。\n')


if __name__ == '__main__':main()
