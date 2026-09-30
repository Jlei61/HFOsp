#!/usr/bin/env python3
"""Retain the native raster/resource and return-zoom semantics for this variant."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import shutil
import time
import numpy as np
from campaign import ROOT, read, write, sha
import analyze_constant_tau_return as analysis
import zoom_topic4_return_core_propagation as zoom

OUT = analysis.OUT/'figure_review'


def raster():
    parts = []
    for folder, lo, hi in [(analysis.BASE, 0, 16.8), (analysis.SOURCE, 16.8, 26.8),
                           (analysis.OUT/'runs'/analysis.NAME, 26.8, analysis.END)]:
        first, last = round(lo*10000), round(hi*10000);start = first
        for p in sorted((folder/'chunks').glob('*.npz')):
            if '.tmp.' in p.name:continue
            a, b = map(int, p.stem.split('_'))
            if b <= first or a >= last:continue
            with np.load(p) as z:
                assert z['raster'].shape == (b-a, 80)
                x, y = max(first, a), min(last, b);assert x == start
                parts.append(z['raster'][x-a:y-a]);start = y
        assert start == last
    result = np.concatenate(parts);assert result.shape == (568000, 80)
    with np.load(analysis.native.SOURCE/'references/native_s9108405.npz') as reference:
        assert np.array_equal(result[:80000], reference['raster'][:80000])
    return result


def raster_panel(ax, ras, lo, hi):
    a, b = round(lo*10000), round(hi*10000);t, ix = np.where(ras[a:b])
    for lower, upper, color in [(0, 20, zoom.COLORS[1]), (20, 40, zoom.COLORS[2]),
                                (40, 60, zoom.COLORS[3]), (60, 80, '#bd7736')]:
        m = (ix >= lower) & (ix < upper)
        ax.scatter((t[m]+a)*.0001, ix[m], s=1.8, c=color, lw=0, rasterized=True)
    ax.set(yticks=[9.5, 29.5, 49.5, 69.5], yticklabels=['Core A E', 'Core B E', 'Other E', 'I'],
        ylim=(-1, 81), xlim=(lo, hi))


def main(wait):
    OUT.mkdir(exist_ok=True)
    while not (analysis.DEST/'result.json').exists():
        if (analysis.OUT/'supervisor.json').exists() and read(analysis.OUT/'supervisor.json')['status'] == 'FAILED':
            write(OUT/'progress.json', dict(status='STOPPED_ON_NATIVE_FAILURE'));return
        write(OUT/'progress.json', dict(status='WAITING_COMPLETE_RETURN_ANALYSIS', pid=os.getpid(), updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    r = read(analysis.DEST/'result.json');d = dict(np.load(analysis.DEST/'readouts.npz'));ras = raster()
    geo = dict(np.load(analysis.OUT/'geometry.npz'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(4, 1, figsize=(15, 9), sharex=True, layout='constrained',
        gridspec_kw={'height_ratios': [2, .9, 1, .9]})
    raster_panel(ax[0], ras, 0, analysis.END)
    for j in range(4):ax[1].plot(d['time5_s'], d['rates5_Hz'][:, j], color=zoom.COLORS[j], lw=.7, label=zoom.LABELS[j])
    ax[1].set(ylabel='E rate (Hz)', ylim=(0, 510));ax[1].legend(frameon=False, ncol=4, fontsize=8)
    for i, c, label in [(0, zoom.COLORS[0], 'All E'), (5, zoom.COLORS[1], 'Core A'), (6, zoom.COLORS[2], 'Core B')]:
        ax[2].plot(d['slow_time_s'], d['Z'][:, i], color=c, lw=1, label=label)
    for ref, color in zip(r['reference_core_Z'], zoom.COLORS[1:3]):ax[2].axhline(ref, color=color, lw=.8, ls=':')
    ax[2].set(ylabel='Resource Z', ylim=(0, 1.02));ax[2].legend(frameon=False, ncol=3, fontsize=8)
    ax[3].plot(d['time1_s'], d['Graw'], color='#52684d', lw=1, label='Global G')
    ax[3].plot(d['time1_s'], d['K_mean'], color='#bc7736', lw=1, label='Mean K')
    ax[3].set(ylabel='Conductance / gL', xlabel='Time (s)');ax[3].legend(frameon=False, ncol=2, fontsize=8)
    for event in r['primary']['entries']:
        for a in ax:a.axvline(event['onset_s'], color='#c74259', ls='--', lw=.6)
    fig.suptitle('Constant 0.5 s K decay: complete observed parameter trajectory', weight='bold')
    for ext in ['png', 'svg']:fig.savefig(ROOT/f'figures/constant_tau_complete_trajectory.{ext}', dpi=180)
    plt.close(fig)
    episodes = [e for e in r['episodes'] if e['after_reference_events'] and e['after_reference_events']['brief_events']]
    selected = None
    if episodes:
        e = episodes[0];first = e['after_reference_events']['brief_events'][0]['start_s']
        selected = dict(first_brief_s=first, recovered_s=e['both_core_reference_s'], episode_end_s=e['window_end_s'])
        lo = max(e['both_core_reference_s'], first-.25);hi = min(lo+3, analysis.END)
        fig, ax = plt.subplots(3, 2, figsize=(13, 7.5), sharex='col', layout='constrained',
            gridspec_kw={'height_ratios': [1.6, 1, .8]})
        dd = dict(rates=d['rates5_Hz'])
        for col, (a, b) in enumerate([(.5, 3.5), (lo, hi)]):
            raster_panel(ax[0, col], ras, a, b);zoom.trace(ax[1, col], dd, a, b)
            m = (d['slow_time_s'] >= a) & (d['slow_time_s'] < b)
            for i, color, ref in zip([5, 6], zoom.COLORS[1:3], r['reference_core_Z']):
                ax[2, col].plot(d['slow_time_s'][m], d['Z'][m, i], color=color, lw=1)
                ax[2, col].axhline(ref, color=color, ls=':', lw=.7)
            ax[2, col].set(ylim=(.6, 1.02), ylabel='Core Z', xlabel='Time (s)', xlim=(a, b))
        ax[0, 0].set_title('Original interictal activity');ax[0, 1].set_title('First brief sequence after core Z reference')
        ax[1, 0].legend(frameon=False, ncol=2, fontsize=8)
        fig.suptitle('Same fixed-cell raster and native 5 ms population readout', fontsize=12)
        for ext in ['png', 'svg']:fig.savefig(ROOT/f'figures/constant_tau_return_zoom.{ext}', dpi=180)
        plt.close(fig)
    import xml.etree.ElementTree as ET
    stems = ['constant_tau_complete_trajectory']+(['constant_tau_return_zoom'] if selected else [])
    for stem in stems:ET.parse(ROOT/f'figures/{stem}.svg')
    write(OUT/'result.json', dict(status='COMPLETE_CANDIDATE_FIGURES', stems=stems,
        first8s_raster_bitwise=True, selected_first_return=selected,
        sustained_native_return_screen=r['sustained_native_return_screen'],
        scope='One paired parameter trajectory; first recovered brief is chosen chronologically, not by morphology. Whole observed trajectory and all episodes remain in analysis. A returned brief or Z crossing is not automatically sustained interictal return.',
        SVG_XML='PASS', agent_visual='PENDING', human_visual='PENDING', producer_sha256=sha(__file__)))
    shutil.copy2(__file__, OUT/'producer.py')
    path = ROOT/'figures/README.md'
    notes = [('constant_tau_complete_trajectory', '恒定0.5秒K衰减候选的完整56.8秒轨迹，保留固定细胞raster、原生区域率、Z及反馈电导。前16.8秒仅在已核验参数变更不影响原轨迹后复用，其后由完整状态连续演化；进入、退出及再次进入都保留。'),
             ('constant_tau_return_zoom', '沿用原有间期保留及返回放大格式，对照原间期0.5–3.5秒和两核Z达到原参考后的首批短事件。示例按时间选取，共同细胞、群体及色彩定义不变；持续返回的结论由整段验收决定。')]
    for stem, note in notes:
        if stem not in stems:continue
        title = f'### {stem}.png / {stem}.svg'
        if title not in path.read_text():
            with path.open('a') as f:f.write('\n\n'+title+'\n'+note+'\n**关注点**：分别检查Z恢复、短事件出现和足够长的原样间期返回，不能相互替代；人工待审。\n')
    write(OUT/'progress.json', dict(status='COMPLETE_PENDING_VISUAL', updated_epoch=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--wait', action='store_true');main(p.parse_args().wait)
