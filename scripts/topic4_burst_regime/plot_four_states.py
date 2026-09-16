"""Matched waveform/raster companion for the completed native burst regime map.

Only the within-core EE multiplier varies. This reads four existing runs and
does not launch a simulation, change a phenotype label, or synthesize EEG.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image

from run import OUT, read, write, name

STEM = 'native_burst_four_states_waveform_raster'
STATES = [
    ('background', 'Low-activity background', .5, '#59636a'),
    ('irregular_bursts', 'Irregular burst', .7, '#ce8051'),
    ('regular_bursts', 'Regular burst', 1., '#627da8'),
    ('variable_bursts', 'Intermediate state', .85, '#aa903f'),
]
WINDOW = (2., 20.)
GROUP = 'coreAE'
SEED = 848101
TOPOLOGY = 2511
DEPTH = 1.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    plan = read(OUT / 'plan.json')
    rows = []
    for label, title, ee, color in STATES:
        job = next(j for j in plan['jobs'] if j['family'] == 'grid'
                   and j['ee'] == ee and j['depth'] == DEPTH
                   and j['seed'] == SEED and j['topology'] == TOPOLOGY)
        folder = OUT / 'per_run' / name(job)
        result = read(folder / 'result.json')
        metric = read(folder / 'metrics_v2.json')[GROUP]
        assert result['status'] == 'COMPLETE'
        assert result['actual_duration_ms'] == 20000.
        assert metric['label'] == label and metric['threshold_stable']
        path = folder / 'trajectory.npz'
        assert sha(path) == result['arrays_sha256']
        with np.load(path) as z:
            arrays = {k: z[k] for k in [
                'group_names', 'group_sizes', 'core_index_E', 'positions_E',
                'spike_counts_2ms', 'raster_sample_ids', 'raster_cell',
                'raster_time_ms']}
        rows.append(dict(job=job, metric=metric, arrays=arrays, title=title,
                         color=color, folder=folder, result=result))

    reference = rows[0]['arrays']
    common = None
    for row in rows:
        z = row['arrays']
        assert np.array_equal(z['core_index_E'], reference['core_index_E'])
        assert np.array_equal(z['positions_E'], reference['positions_E'])
        recorded = z['raster_sample_ids']
        recorded = recorded[recorded < len(z['core_index_E'])]
        recorded = recorded[z['core_index_E'][recorded] == 0]
        common = recorded if common is None else np.intersect1d(common, recorded)
    assert len(common) >= 100
    sample = common[np.linspace(0, len(common) - 1, 100, dtype=int)]
    assert len(np.unique(sample)) == 100
    lookup = np.full(len(reference['core_index_E']), -1, dtype=int)
    lookup[sample] = np.arange(1, 101)

    fig, axes = plt.subplots(4, 2, figsize=(14.6, 10.1), sharex=True,
                             gridspec_kw={'width_ratios': [1.08, 1.]})
    metadata = []
    for i, row in enumerate(rows):
        z, metric = row['arrays'], row['metric']
        group_index = z['group_names'].tolist().index(GROUP)
        size = int(z['group_sizes'][group_index])
        time = (np.arange(len(z['spike_counts_2ms'])) + .5) * .002
        rate = z['spike_counts_2ms'][:, group_index] / size / .002
        visible = (time >= WINDOW[0]) & (time < WINDOW[1])
        assert visible.sum() == 9000
        assert np.isclose(rate[visible].mean(), metric['mean_rate_hz'])
        left, right = axes[i]
        left.plot(time[visible], rate[visible], color=row['color'], lw=.7)
        left.set_ylim(0, max(1., rate[visible].max() * 1.08))
        left.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        left.set_ylabel('Firing rate\n(Hz / E cell)')
        left.set_title(f'{row["title"]}  |  E→E = {row["job"]["ee"]:g}',
                       loc='left', fontweight='bold', fontsize=11,
                       color=row['color'], pad=24)
        cv = 'N/A' if metric['cv'] is None else f'{metric["cv"]:.2f}'
        left.text(.995, 1.035,
                  f'{metric["n_bursts"]} bursts   CV {cv}   Mean {metric["mean_rate_hz"]:.2f} Hz',
                  ha='right', va='bottom', transform=left.transAxes, fontsize=9)

        # The observer saved occupied 2-ms bins at their right edges. Display
        # their centers, matching the population-rate bin centers exactly.
        rt = z['raster_time_ms'].astype(float) / 1000. - .001
        cells = z['raster_cell']
        selected = (rt >= WINDOW[0]) & (rt < WINDOW[1]) & np.isin(cells, sample)
        displayed = lookup[cells[selected]]
        assert np.all((displayed >= 1) & (displayed <= 100))
        right.scatter(rt[selected], displayed, s=4., marker='.',
                      linewidths=0, color='#25282a', rasterized=True)
        right.set(ylim=(.5, 100.5), yticks=[1, 50, 100],
                  ylabel='Same 100 core E cells')
        right.set_title('Raster | occupied 2 ms bins', loc='left', fontsize=10, pad=24)
        for ax in (left, right):
            ax.set(xlim=WINDOW, xticks=np.arange(2, 21, 3))
            ax.tick_params(axis='x', labelbottom=True)
            ax.set_xlabel('Time (s)')
            ax.grid(axis='x', color='#e7e7e7', lw=.45)
            ax.set_axisbelow(True)
        metadata.append(dict(
            name=name(row['job']), expected_label=metric['label'],
            parameters=row['job'], group=GROUP, n_core_E=size,
            n_bursts=metric['n_bursts'], interval_cv=metric['cv'],
            mean_rate_hz=metric['mean_rate_hz'], threshold_stable=True,
            waveform_y_limits=list(left.get_ylim()),
            n_raster_occupied_bins=int(selected.sum()),
            trajectory_sha256=row['result']['arrays_sha256'],
            metrics_sha256=sha(row['folder'] / 'metrics_v2.json')))

    fig.suptitle('Waveforms and rasters across four activity states', fontsize=15, y=.991)
    fig.text(.5, .949,
             'Core A · one fixed network and noise realization · threshold-lowering amplitude = 1',
             ha='center', fontsize=10)
    fig.text(.5, .012,
             'Population rate: all 720 core E cells. Raster: the same 100 recorded cells. '
             'Same 2–20 s window; waveform y scales differ by row.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.075, right=.986, bottom=.075, top=.885,
                        hspace=.72, wspace=.19)
    folder = OUT / 'figures'
    for extension in ['png', 'pdf']:
        fig.savefig(folder / f'{STEM}.{extension}', dpi=220,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    with Image.open(folder / f'{STEM}.png') as im:
        im.load()
        dimensions = [im.width, im.height]
    write(folder / f'{STEM}.json', dict(
        status='GENERATED_AGENT_VISUAL_REVIEW_PENDING',
        producer=str(Path(__file__).resolve()), producer_sha256=sha(__file__),
        selection='Fixed Core A, topology 2511, seed 848101 and depth 1; only EE changes; user-requested state order.',
        waveform='Native population firing rate: spike count per 2 ms / number of core E cells / 0.002 s; no smoothing or EEG inference.',
        raster='Actual occupied 2 ms bins of the same 100 recorded E neurons, ordered by cell ID; not exact individual spike timestamps.',
        raster_neuron_ids=sample.tolist(), window_s=list(WINDOW),
        full_record_duration_s=20., analyzed_duration_s=18., rows=metadata,
        figure_dimensions_px=dimensions,
        checks=dict(exact_saved_source_hashes=True, expected_labels=True,
                    threshold_stable_selected_runs=True, same_native_geometry=True,
                    same_raster_neurons=True, same_waveform_raster_window=True,
                    displayed_mean_matches_original_metric=True),
        figure_human_review='PENDING', physical_simulations_added=0))
    readme = folder / 'README.md'
    heading = f'### {STEM}.png / .pdf'
    existing = readme.read_text()
    if heading not in existing:
        with readme.open('a') as f:
            f.write(f'\n{heading}\n'
                    '按低活动背景、不规则burst、规则burst、中间状态排列，每行左侧为原生2ms群体发放率，右侧为同一批100个E细胞的raster。'
                    '固定同一Core A、网络、噪声和降阈值幅度1，仅改变核内E→E倍率；两列都显示2–20秒，四行波形使用各自标明的真实发放率量程。'
                    'CV与类别来自该完整18秒窗口；raster是实际2ms占用记录，群体波形不代表模拟EEG电压。'
                    '**关注点**：逐行核对波形与raster的事件对应，区分时序规则性、招募强度与平均发放率。\n')
    print(json.dumps(dict(status='FIGURE_GENERATED',
                          png=str(folder / f'{STEM}.png'), rows=len(rows))))


if __name__ == '__main__':
    main()
