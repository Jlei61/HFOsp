#!/usr/bin/env python3
"""Common120s native graph trajectories; no new event classification."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_topic4_loop_axis_native as run
from plot_topic4_loop_axis_prefix import prefix


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    analysis_path = run.OUT / 'analysis.json'
    analysis = json.loads(analysis_path.read_text())
    assert analysis['status'] == 'COMPLETE'
    lookup = {r['condition']: r for r in analysis['rows']}
    sources = [('current', run.native.SOURCE, run.native.NAME)] + [
        (c, run.OUT / c, f'{c}_s9108405') for c in ['rotated', 'isotropic']]
    rows = []; traces = []; expected = None; saved = {}
    for condition, root, name in sources:
        folder = root / 'runs' / name
        result_path = folder / 'result.json'
        result = json.loads(result_path.read_text())
        assert result['status'] == 'COMPLETE'
        row = lookup[condition]
        assert row['full120s'] and row['paired_exogenous_records_exact']
        assert row['spike_field_integrity'] == 'PASS'
        d = prefix(folder, 'chunks', ['spikes_1ms', 'regions_1ms', 'raster',
                   'slow_time_ms', 'Z', 'inputs'], end=120.)
        fb = prefix(folder, 'feedback_chunks', ['time_ms', 'G_applied_mean',
                    'K_mean'], end=120.)
        assert len(d['spikes_1ms']) >= 120000 and len(d['raster']) >= 1200000
        assert d['raster'].shape[1] == 80
        with np.load(root / 'geometry.npz') as g:
            counts = np.r_[32000, g['region_counts'][:3]]
        raw = np.c_[d['spikes_1ms'][:120000, 0], d['regions_1ms'][:120000, :3]]
        assert np.array_equal(raw[:, 0], raw[:, 1:].sum(1))
        rates = raw.reshape(12000, 10, 4).sum(1) / counts / .01
        inputs = d['inputs'][:1200]
        assert len(inputs) == 1200
        if expected is not None:
            assert np.array_equal(inputs, expected)
        expected = inputs
        sel = d['slow_time_ms'] < 120000
        fsel = fb['time_ms'] < 120000
        quiet = np.all(rates[:, :3] < 5., axis=1)
        traces.append((d, fb, rates, quiet, sel, fsel))
        saved.update({f'{condition}_rates_10ms_Hz': rates,
                      f'{condition}_jointquiet_10ms': quiet,
                      f'{condition}_slow_time_ms': d['slow_time_ms'][sel],
                      f'{condition}_Z_all_A_B': d['Z'][sel][:, [0, 5, 6]],
                      f'{condition}_feedback_time_ms': fb['time_ms'][fsel],
                      f'{condition}_G_applied_mean': fb['G_applied_mean'][fsel],
                      f'{condition}_K_mean': fb['K_mean'][fsel]})
        rows.append(dict(condition=condition, source=str(folder),
                         result_sha256=sha(result_path), entries=row['entries'],
                         exits=row['exits'], initial_brief=row['initial_short_events']['brief_count'],
                         brief_recurrence_after_Z=row['brief_recurrence_episodes_after_common_Z'],
                         own_initial_comparison=row['temporal_return_comparison_status']))
    colors = ['#8952ab', '#cf3e87', '#159cbe', '#37698b']
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 11,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8})
    fig, axes = plt.subplots(5, 3, figsize=(15.3, 8.8), sharex=True,
                            gridspec_kw={'height_ratios': [2.5, 1.8, .20, 1, 1.1]})
    mapping = np.r_[np.linspace(0, 33, 20), np.linspace(36, 69, 20),
                    np.linspace(72, 84, 20), np.linspace(87, 99, 20)]
    t = (np.arange(12000) + .5) * .01
    feedback_max = max(float(np.max(fb[k][fsel])) for _, fb, _, _, _, fsel in traces
                       for k in ['G_applied_mean', 'K_mean'])
    for c, (row, (d, fb, rates, quiet, sel, fsel)) in enumerate(zip(rows, traces)):
        ax = axes[0, c]; it, ix = np.where(d['raster'][:1200000])
        for lo, hi, color in [(0, 20, colors[1]), (20, 40, colors[2]),
                              (40, 60, colors[3]), (60, 80, '#c17730')]:
            mask = (ix >= lo) & (ix < hi)
            ax.scatter(it[mask] * .0001, mapping[ix[mask]], s=.4, lw=0,
                       c=color, rasterized=True)
        ax.set(ylim=(-2, 101), yticks=[16.5, 52.5, 78, 93],
               yticklabels=['Core A E', 'Core B E', 'Other E', 'I'],
               title={'current': 'Current axis', 'rotated': 'Rotated structure',
                      'isotropic': 'Isotropic structure'}[row['condition']])
        ax = axes[1, c]
        for j, label in enumerate(['All E', 'Core A', 'Core B', 'Other E']):
            ax.plot(t, rates[:, j], lw=.45, c=colors[j], label=label)
        ax.set(ylim=(0, 510), ylabel='E rate (Hz)')
        ax = axes[2, c]
        ax.imshow(quiet[None, :], origin='lower', extent=(0, 120, 0, 1),
                  aspect='auto', interpolation='nearest', cmap='Greys', vmin=0, vmax=1)
        ax.set(yticks=[], ylabel='Quiet'); ax.spines[:].set_visible(False)
        ax = axes[3, c]
        for j, k in [(0, 0), (1, 5), (2, 6)]:
            ax.plot(d['slow_time_ms'][sel] / 1000, d['Z'][sel, k], lw=.85, c=colors[j])
        ax.set(ylim=(0, 1.03), ylabel='Resource Z')
        ax = axes[4, c]
        ax.plot(fb['time_ms'][fsel] / 1000, fb['G_applied_mean'][fsel], c='#385a34', label='Applied G')
        ax.plot(fb['time_ms'][fsel] / 1000, fb['K_mean'][fsel], c='#bc7837', label='K')
        ax.set(ylabel='Mean g / gL', xlabel='Time (s)', ylim=(0, feedback_max * 1.04))
        for r in [0, 1, 3, 4]:
            axes[r, c].spines[['top', 'right']].set_visible(False)
        for ax in axes[:, c]:
            ax.set_xlim(0, 120); ax.set_xticks([0, 30, 60, 90, 120])
    fig.subplots_adjust(left=.08, right=.98, top=.94, bottom=.10, hspace=.23, wspace=.25)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    feedback_handles, feedback_labels = axes[4, 0].get_legend_handles_labels()
    fig.legend(handles + feedback_handles, labels + feedback_labels, loc='upper center',
               bbox_to_anchor=(.5, 1.0), ncol=6, frameon=False, fontsize=9)
    fig.text(.5, .039, 'Same 120 s window and external input; original fixed 80-cell raster. Black strip: All E and both cores < 5 Hz.', ha='center', fontsize=9)
    fig.text(.5, .013, 'One trajectory per graph. Structure controls also change outgoing degree and low-threshold source strength.', ha='center', fontsize=9)
    dest = run.OUT / 'full_comparison'; figs = dest / 'figures'; figs.mkdir(parents=True, exist_ok=True)
    files = []
    for ext in ['png', 'pdf']:
        path = figs / f'axis_native_120s.{ext}'
        fig.savefig(path, dpi=180, facecolor='white')
        files.append(dict(path=str(path), sha256=sha(path)))
    plt.close(fig)
    np.savez_compressed(dest / 'trace_inputs.npz', time_10ms_s=t, **saved)
    metadata = dict(status='COMPLETE_CANDIDATE', rows=rows, files=files,
                    analysis=str(analysis_path), analysis_sha256=sha(analysis_path),
                    producer_sha256=sha(Path(__file__)), paired_future_input_records_exact=True,
                    raster='Original fixed80 cells, all0.1ms spike indicators over0–120s; no resampling or jitter.',
                    role='Diagnostic common-window comparison, not a replacement for the accepted main Figure5 layout.',
                    interpretation='No new detector or phase alignment. Use native analysis for entry/exit/return and per-event spatial views for propagation. Repeated bouts are not a certified periodic orbit.',
                    agent_visual_review='PENDING', human_review='PENDING')
    (dest / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (figs / 'README.md').write_text('### axis_native_120s.png\n\n三种结构在同一外源输入、共同0–120秒窗口下的自主轨迹。沿用固定80细胞raster、两核及全E/核外率、共同低活动条带、Z、实际施加G和K；各列共用坐标和反馈色标，没有按进入时刻重排时间。进入、退出、返回采用已有分析判据，本图没有新增状态定义。\n\n**关注点**：短事件、长活动段及资源恢复是否共同保留；输出度和低阈值来源强度亦有变化，不能单独归因于轴方向，也不能从重复形态认定极限环。PNG/PDF均待作者目视。\n')
    print(json.dumps(dict(figure=str(figs / 'axis_native_120s.png'), rows=rows)), flush=True)


if __name__ == '__main__':
    main()
