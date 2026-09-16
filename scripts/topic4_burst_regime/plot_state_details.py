"""Large separate state figures, preserving the previously shown cells and runs."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image

from run import OUT, read, write
from plot_four_states import STATES, STEM, GROUP

DEST = OUT / 'figures' / 'four_state_detail'
FULL = (2., 20.)
DETAIL = (4., 7.)
FINE = (5., 5.6)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows():
    previous = read(OUT / 'figures' / f'{STEM}.json')
    sample = np.array(previous['raster_neuron_ids'], dtype=int)
    assert len(sample) == len(np.unique(sample)) == 100
    rows = []
    for prior, (label, title, ee, color) in zip(previous['rows'], STATES):
        folder = OUT / 'per_run' / prior['name']
        result = read(folder / 'result.json')
        metric = read(folder / 'metrics_v2.json')[GROUP]
        assert prior['expected_label'] == metric['label'] == label
        assert prior['parameters']['ee'] == ee
        path = folder / 'trajectory.npz'
        assert sha(path) == prior['trajectory_sha256'] == result['arrays_sha256']
        with np.load(path) as z:
            j = z['group_names'].tolist().index(GROUP)
            size = int(z['group_sizes'][j])
            assert size == 720
            assert np.all(z['core_index_E'][sample] == 0)
            assert np.isin(sample, z['raster_sample_ids']).all()
            rate = z['spike_counts_2ms'][:, j].astype(float) / size / .002
            time = (np.arange(len(rate)) + .5) * .002
            cells = z['raster_cell']
            selected = np.isin(cells, sample)
            lookup = np.full(len(z['core_index_E']), -1, dtype=int)
            lookup[sample] = np.arange(1, 101)
            rt = z['raster_time_ms'][selected].astype(float) / 1000. - .001
            ri = lookup[cells[selected]]
            assert np.all((ri >= 1) & (ri <= 100))
        full_mask = (time >= FULL[0]) & (time < FULL[1])
        assert full_mask.sum() == 9000
        assert np.isclose(rate[full_mask].mean(), metric['mean_rate_hz'])
        onsets = np.array([e['start_s'] + 2. for e in metric['events'] if not e['left_censored']])
        assert len(onsets) == metric['n_bursts']
        rows.append(dict(label=label, title=title, ee=ee, color=color,
                         prior=prior, metric=metric, sample=sample,
                         time=time, rate=rate, rt=rt, ri=ri, onsets=onsets,
                         ymax=max(1., rate[full_mask].max() * 1.12)))
    return rows


def draw(row, window, overview, fine=False):
    fig, (wave, raster) = plt.subplots(
        2, 1, figsize=(13.2, 8.6), sharex=True,
        gridspec_kw={'height_ratios': [1.35, 4.0], 'hspace': .22})
    t, rate, rt, ri = (row[k] for k in ['time', 'rate', 'rt', 'ri'])
    displayed_indices = np.linspace(1, 100, 30, dtype=int) if fine else np.arange(1, 101)
    selected_cells = np.isin(ri, displayed_indices)
    rt, ri = rt[selected_cells], ri[selected_cells]
    ri = np.searchsorted(displayed_indices, ri) + 1
    n_cells = len(displayed_indices)
    mask = (t >= window[0]) & (t < window[1])
    raster_mask = (rt >= window[0]) & (rt < window[1])
    wave.plot(t[mask], rate[mask], color=row['color'], lw=1.0)
    wave.set(ylim=(0, row['ymax']), ylabel='Population rate\n(Hz / E cell)')
    wave.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    wave.set_title('All 720 excitatory neurons in Core A', loc='left', fontsize=12, pad=10)
    # A short black vertical stroke occupies 0.64 neuron rows. Keep these as
    # vector paths in PDF so enlarging the page does not shrink or blur points.
    raster.vlines(rt[raster_mask], ri[raster_mask] - .32,
                  ri[raster_mask] + .32, color='#111111', linewidth=1.2 if fine else .85)
    raster.set(ylim=(.4, n_cells+.6), yticks=[1, 10, 20, 30] if fine else [1, 25, 50, 75, 100],
               ylabel=f'Neuron index\n(same {n_cells} recorded Core A E cells)',
               xlabel='Simulation time (s)')
    raster.set_title('Raster | one black tick = one occupied 2 ms bin',
                     loc='left', fontsize=12, pad=10)
    if overview:
        for ax in (wave, raster):
            ax.axvspan(*DETAIL, facecolor='#dee6ee', alpha=.5, zorder=-1)
        wave.text(np.mean(DETAIL), .93, '4–7 s detail',
                  transform=wave.get_xaxis_transform(), ha='center', va='top',
                  fontsize=10, color='#536171')
    ticks = np.arange(2., 20.1, 2.) if overview else (np.arange(5., 5.601, .1) if fine else np.arange(4., 7.01, .5))
    for ax in (wave, raster):
        ax.set(xlim=window, xticks=ticks)
        ax.tick_params(axis='both', labelsize=11, length=4)
    wave.tick_params(axis='x', labelbottom=True)
    m = row['metric']
    cv = 'N/A' if m['cv'] is None else f'{m["cv"]:.2f}'
    view_name = 'Full 18 s record' if overview else ('5.0–5.6 s | 30-cell close-up' if fine else 'Common 4–7 s detail')
    fig.suptitle(f'{row["title"]}  |  {view_name}', fontsize=18,
                 fontweight='bold', color=row['color'], y=.985)
    fig.text(.5, .93,
             f'E→E = {row["ee"]:g} · threshold-lowering amplitude = 1 · same network, noise and neuron IDs',
             ha='center', fontsize=12)
    fig.text(.5, .895,
             f'Full 18 s: {m["n_bursts"]} detected bursts · interval CV = {cv} · mean rate = {m["mean_rate_hz"]:.2f} Hz / cell',
             ha='center', fontsize=11)
    fig.text(.5, .018,
             'Rate and raster share the time axis. 2 ms native records; no smoothing. '
             'Rate scale is fixed within each state and differs between states.',
             ha='center', fontsize=10)
    fig.subplots_adjust(left=.105, right=.976, bottom=.092, top=.81)
    shown_onsets = row['onsets'][(row['onsets'] >= window[0]) & (row['onsets'] < window[1])]
    return fig, dict(window_s=list(window), n_rate_bins=int(mask.sum()),
                     n_raster_neurons=n_cells,
                     displayed_original_100_cell_indices=displayed_indices.tolist(),
                     n_raster_occupied_bins=int(raster_mask.sum()),
                     n_detected_onsets_in_window=len(shown_onsets),
                     onset_times_s_in_window=shown_onsets.tolist(),
                     native_rate_y_limits=[0, row['ymax']])


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    DEST.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    manifest = []
    readme = []
    zh = {'background': '低活动背景', 'irregular_bursts': '不规则burst',
          'regular_bursts': '规则burst', 'variable_bursts': '中间状态'}
    booklet = DEST / 'four_states_large_rasters.pdf'
    with PdfPages(booklet) as pages:
        for row in rows:
            for kind, window in [('overview', FULL), ('detail', DETAIL), ('closeup', FINE)]:
                fig, display = draw(row, window, kind == 'overview', kind == 'closeup')
                stem = f'{row["label"]}_{kind}'
                for ext in ['png', 'pdf']:
                    fig.savefig(DEST / f'{stem}.{ext}', dpi=210,
                                bbox_inches='tight', facecolor='white')
                pages.savefig(fig, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                with Image.open(DEST / f'{stem}.png') as im:
                    im.load()
                    pixels = [im.width, im.height]
                manifest.append(dict(stem=stem, state=row['label'],
                    source_name=row['prior']['name'], source_sha256=row['prior']['trajectory_sha256'],
                    group=GROUP, n_population_E=720, n_raster_neurons=display['n_raster_neurons'],
                    raster_neuron_ids=row['sample'][np.array(display['displayed_original_100_cell_indices'])-1].tolist(),
                    classification_window_s=list(FULL), label=row['metric']['label'],
                    full_record_cv=row['metric']['cv'], full_record_n_bursts=row['metric']['n_bursts'],
                    raster_marker='vertical stroke spanning 0.64 neuron rows; 1.2 pt for closeup, 0.85 pt otherwise; vector PDF',
                    display=display, dimensions_px=pixels,
                    files={ext:dict(path=str((DEST / f'{stem}.{ext}').resolve()),
                                    sha256=sha(DEST / f'{stem}.{ext}')) for ext in ['png', 'pdf']}))
                kind_zh = {'overview':'2–20秒完整时段', 'detail':'固定4–7秒局部放大', 'closeup':'固定5.0–5.6秒、30细胞近景'}[kind]
                readme.append(f'### {stem}.png / .pdf\n'
                    f'展示{zh[row["label"]]}的{kind_zh}，上方为Core A内全部720个E细胞的原生2ms群体发放率，下方为固定{display["n_raster_neurons"]}个实际记录细胞的大幅raster。'
                    '黑色短竖线表示2ms格内发生过发放，四状态使用相同细胞和时间窗；状态标签及CV始终来自完整18秒记录。'
                    '**关注点**：看清零散发放、局部短暂招募与贯穿细胞群的重复竖列，不将标记大小或不同发放率量程解释成生物差异。\n')
    readme.append('### four_states_large_rasters.pdf\n'
                  '十二页审阅包按低活动背景、不规则burst、规则burst、中间状态排列，每个状态依次为全程、3秒局部和0.6秒近景。'
                  'raster采用矢量短竖线，便于放大查看；群体发放率与100/30细胞raster的统计对象明确区分，30细胞从原100细胞中按固定索引均匀抽取。'
                  '**关注点**：逐页比较相同时间窗与相同细胞，观察事件间隔、参与程度和重复性。\n')
    (DEST / 'README.md').write_text('\n'.join(readme))
    write(DEST / 'figure_manifest.json', dict(
        status='GENERATED_AGENT_VISUAL_REVIEW_PENDING',
        producer=str(Path(__file__).resolve()), producer_sha256=sha(__file__),
        previous_companion_metadata=str(OUT / 'figures' / f'{STEM}.json'),
        figures=manifest, booklet=dict(path=str(booklet.resolve()), pages=12, sha256=sha(booklet)),
        physical_simulations_added=0, new_run_selection=False,
        zoom_selection='Common 4–7 s with the original 100 cells; common 5.0–5.6 s with the same 30 fixed evenly indexed cells from those 100. Labels and CV use 2–20 s.',
        checks=dict(source_hashes_match=True, prior_labels_preserved=True,
                    same_100_native_cells=True, same_windows_all_states=True,
                    displayed_rate_mean_matches_full_record_metric=True,
                    rate_scale_identical_in_overview_and_detail=True),
        human_review='PENDING'))
    print(json.dumps(dict(status='FIGURES_GENERATED', separate_figures=len(manifest),
                          booklet_pages=12, directory=str(DEST))))


if __name__ == '__main__':
    main()
