#!/usr/bin/env python3
"""Compare finite native count noise with the free coupled-density candidate."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import pickle
import time
import numpy as np
from campaign import ROOT, read, write, sha
from native_count_noise import OUT, SEEDS
from analyze_coupled_density_exit import native_data, candidate, metrics, ADAPTED


def load_native(seed, geo):
    folder = OUT / 'runs' / f'count{seed}'
    def stream(kind, keys):
        rows = {key: [] for key in keys}
        for path in sorted((folder / kind).glob('*.npz')):
            with np.load(path) as z:
                for key in keys:
                    rows[key].append(z[key])
        return {key: np.concatenate(value) for key, value in rows.items()}
    chunks = stream('chunks', ['time_ms', 'spikes_1ms', 'regions_1ms', 'field_time_ms', 'field_5ms'])
    glob = stream('mechanism_chunks', ['time_ms', 'global_E_rate_Hz', 'global_raw_conductance_ratio'])
    regional = stream('regional_chunks', ['time_ms', 'values'])
    sizes, pop, regions = geo['group_size'], geo['population'], geo['group_region']
    weights = np.array([sizes[(pop == 0) & (regions == q)].sum() for q in range(3)])
    count = np.bincount(geo['group_cell'][pop == 0], weights=sizes[pop == 0], minlength=400)
    rates = np.c_[chunks['spikes_1ms'][:, 0]/32000, chunks['regions_1ms'][:, :3]/weights]*1000
    assert rates.shape == (10000, 4) and len(glob['time_ms']) == 10000
    with (folder / 'checkpoint.pkl').open('rb') as handle:
        end = pickle.load(handle)['engine']
    assert end['step'] == 200000
    masks = [np.arange(32000)] + [np.flatnonzero(regions[geo['cell_group'][:32000]] == q) for q in range(3)]
    values = regional['values']
    all_z = (values[:, :, 8]*weights).sum(1)/32000
    all_k = (values[:, :, 0]*weights).sum(1)/32000
    return dict(name=f'Native count {seed}', rate_time_ms=chunks['time_ms'], rates=rates,
        global_time_ms=glob['time_ms'], R=glob['global_E_rate_Hz'], Graw=glob['global_raw_conductance_ratio'],
        slow_time_ms=regional['time_ms'], Z=np.c_[all_z, values[:, :, 8]], K=np.c_[all_k, values[:, :, 0]],
        field_time_ms=chunks['field_time_ms'], fields=chunks['field_5ms']/count/.005, cell_counts=count,
        final_Z=np.array([end['slow']['z'][mask].mean() for mask in masks]),
        final_K=np.array([end['termination_mechanism']['sahp_g'][mask].mean() for mask in masks]))


def plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharex='col', layout='constrained')
    colors = ['black', '#277c9e', '#ad5d89', '#cc8b38', '#5b966e']
    for d, color in zip(data, colors):
        numerical = d['name'].startswith('Density')
        for row, (t, y) in enumerate([(d['global_time_ms'], d['R']), (d['global_time_ms'], d['Graw']),
                                     (d['slow_time_ms'], d['K'][:, 0]), (d['slow_time_ms'], d['Z'][:, 0])]):
            for ax in axes[row]:
                ax.plot(t/1000, y, color=color, lw=1.1, ls='--' if numerical else '-', label=d['name'])
    for row, label in enumerate(['Causal E rate (Hz)', 'Raw global G / gL', 'Mean K / gL', 'Mean Z']):
        axes[row, 0].set_ylabel(label)
        axes[row, 0].set_xlim(10, 20)
        axes[row, 1].set_xlim(13, 14.3)
    axes[0, 0].set_title('Complete post-entry comparison', loc='left', weight='bold')
    axes[0, 1].set_title('The earlier activity dip', loc='left', weight='bold')
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    for ax in axes[0]:
        ax.axhline(5, color='.5', ls=':', lw=.7)
    for ax in axes[1]:
        ax.axhline(95.19851312666987/(18+17.662847938268442), color='.5', ls=':', lw=.7)
    for ax in axes[-1]:
        ax.set_xlabel('Native time (s)')
    fig.suptitle('Identical native state and OU input: vary only external Poisson counts', weight='bold')
    fig.text(.5, -.01, 'Two new count streams plus the original; density streams are numerical. No phase alignment or fitted parameters.', ha='center', fontsize=9)
    for ext in ['png', 'svg']:
        fig.savefig(ROOT / 'figures' / f'native_count_noise_exit.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    write(OUT / 'figure_metadata.json', dict(agent_visual='PENDING', human_review='PENDING', producer_sha256=sha(__file__)))


def main(wait):
    geo = dict(np.load(ADAPTED / 'geometry.npz'))
    original = native_data(geo)
    original['name'] = 'Native original 8405'
    data, rows, done = [original], [metrics(original, original)], []
    while len(done) < 2:
        for seed in SEEDS:
            gate = OUT / 'runs' / f'count{seed}' / 'paired_input_audit.json'
            if seed in done or not gate.exists():
                continue
            assert read(gate)['status'] == 'PASS'
            d = load_native(seed, geo)
            row = metrics(d, original)
            row['paired_input_gate'] = str(gate)
            data.append(d)
            rows.append(row)
            done.append(seed)
            print('NATIVE COUNT COMPARISON', seed, row['first_causal_R_le5_for100ms_s'], flush=True)
        write(OUT / 'comparison.json', dict(status='PARTIAL', completed=done, rows=rows,
              native_correspondence_certified=False, formal_bifurcation_allowed=False,
              statistical_unit='Three countrealizations conditionalon one nativeinitialstate andone identicalOU meantrajectory. No preciseensemble distributionestimated.'))
        if len(done) < 2:
            if not wait:
                return
            write(OUT / 'analysis_progress.json', dict(status='WAITING_NATIVE_INPUT_QA', pid=os.getpid(), completed=done, updated_epoch=time.time()))
            time.sleep(30)
    for seed in [927671, 927672]:
        assert read(ROOT / 'coupled_density_exit' / f'num{seed}' / 'result.json')['status'] == 'COMPLETE'
        d = candidate(seed, geo)
        data.append(d)
        rows.append(metrics(d, original))
    report = read(OUT / 'comparison.json')
    report.update(status='COMPLETE', rows=rows, producer_sha256=sha(__file__))
    write(OUT / 'comparison.json', report)
    plot(data)
    write(OUT / 'analysis_progress.json', dict(status='COMPLETE', completed=done, updated_epoch=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    main(parser.parse_args().wait)
