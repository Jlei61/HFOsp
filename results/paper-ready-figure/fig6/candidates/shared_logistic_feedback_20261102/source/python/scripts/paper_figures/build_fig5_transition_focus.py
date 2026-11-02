#!/usr/bin/env python3
"""Reorganize the frozen Fig5 Z/M trajectory around its first high-state entry."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from scipy.ndimage import gaussian_filter1d
import plot_topic4_fig5_clean_panels as previous
import build_fig5_single_seed_panels as exporter

WINDOW = (8.0, 11.0)
COLORS = ['#267ba8', '#dd871c', '#bd6c35', '#ab4255', '#ba263c']
Z_COLOR = '#2878b5'
M_COLOR = '#d77d27'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(previous.f.safe(data), ensure_ascii=False, indent=2) + '\n')


def states(reference):
    _, pre, onset = reference['snapshots'][1:]
    # Late interictal burst stays inside the 8–11 s overview. Earlier native
    # frames expose recruitment before the previously displayed9.737-s state.
    times = [8.420, pre['time_s'], 9.480, 9.600, round(onset['time_s'], 3)]
    return [dict(number=i+1, time_s=float(t), color=COLORS[i],
                 label=['Interictal', 'Pre-ictal', '', '', 'Onset'][i],
                 previous_state={1:3, 4:4}.get(i))
            for i, t in enumerate(times)]


def markings(ax, samples, onset):
    ax.axvspan(onset, WINDOW[1], fc='#ba263c', alpha=.08, lw=0, zorder=0)
    for s in samples:
        ax.axvline(s['time_s'], color=s['color'], ls=':', lw=1.2, alpha=.85, zorder=1)
    ax.set_xlim(WINDOW)
    ax.set_xticks(np.arange(8, 11.01, .5))


def draw_readout(ax, a, samples, onset, recruitment_start):
    names = list(a['contact_names'])
    order = previous.f.ORDER
    ids = [names.index(n) for n in order]
    t = a['lfp_time_ms'] / 1000
    raw = np.asarray(a['lfp_raw'][:, ids], float)
    assert np.allclose(np.diff(t), .0005, atol=1e-10)
    # Display the actual recorded proxy, including its sustained level. No
    # temporal filtering, centering, detrending, CAR, or time-varying gain.
    baseline = t < recruitment_start
    scale_p95 = np.percentile(abs(raw[baseline]), 95, axis=0)
    positive = scale_p95[np.isfinite(scale_p95) & (scale_p95 > 1e-12)]
    assert positive.size
    floor = .15 * float(np.median(positive))
    gain = np.maximum(scale_p95, floor)
    display_gain = .68 / 1.48
    take = (t >= WINDOW[0]) & (t <= WINDOW[1])
    plotted = raw[take] / gain * display_gain + np.arange(14, -1, -1)
    for i in range(15):
        ax.plot(t[take], plotted[:, i],
                c='#27929d' if i < 4 else '#c67b35', lw=.65, rasterized=True)
    ax.axhline(10.5, color='#cccccc', lw=.8)
    ymin, ymax = min(-.65, float(plotted.min())-.12), max(15., float(plotted.max())+.12)
    ax.set(yticks=np.arange(15)[::-1], yticklabels=order, ylim=(ymin, ymax),
           ylabel='Readout (a.u.)')
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelbottom=False)
    markings(ax, samples, onset)
    for s in samples:
        # Separate the tightly spaced2/3 labels with short leader lines.
        offset = {2:(-8,7), 3:(8,24)}.get(s['number'], (0,7))
        ax.annotate(str(s['number']), xy=(s['time_s'], 1.), xycoords=ax.get_xaxis_transform(),
            xytext=offset, textcoords='offset points', color=s['color'], weight='bold',
            ha='center', va='bottom', fontsize=17,
            arrowprops=dict(arrowstyle='-', color=s['color'], lw=.8) if s['number'] in (2,3) else None)
    for s in samples[:2]:
        lo = s['time_s'] - .05
        ax.add_patch(Rectangle((lo, ymin+.1), .30, ymax-ymin-.2, fc='none', ec=s['color'], lw=1.5))
    ax.text(.01, 1.025, 'Unfiltered', transform=ax.transAxes, fontsize=13,
            ha='left', va='bottom')
    return dict(contact_order=order, band_hz=None, sample_rate_Hz=2000,
        filter='None; all stored 2000-Hz samples in the display window are plotted.',
        signal='Recorded pre-Z virtual synaptic-current proxy; no CAR in this historical waveform display.',
        gain_per_contact=gain, gain_definition='Per-contact P95 absolute unfiltered signal before the frozen9.420s entry-episode start, with a15%-of-median floor; fixed across time. No baseline subtraction.',
        normalization_window_s=[0., float(recruitment_start)], scale_percentile=95.,
        raw_scale_p95=scale_p95, scale_floor=floor,
        display_gain_per_row_spacing=display_gain,
        normalization_cutoff_source='reference.selection.entry_episode_start_s; last >=20ms global-E quiet interval before sustained-high entry ends at9.420s. This is the current trajectory marker, not the historical trajectory t_recruit.',
        visible_time_s=list(WINDOW), raw_lfp_sha256=hashlib.sha256(a['lfp_raw'].tobytes()).hexdigest(),
        unfiltered=True, dc_preserved=True, baseline_subtraction=False, detrending=False,
        common_average_reference=False, samples_plotted_per_contact=int(take.sum()),
        signed_waveform=False, envelope=False, smoothing=False, contact_dependent_gain=True,
        time_varying_gain=False, raw_mean_strip_displayed=False, waveform_clipped=False,
        y_limits=[ymin, ymax])


def draw_slow(za, ma, a, eta, samples, onset):
    t = a['slow_time_ms'] / 1000
    take = (t >= WINDOW[0]) & (t <= WINDOW[1])
    z = a['Z'][:, [0, 5]]
    m = eta * a['M'][:, [0, 1]]
    for j, (label, style) in enumerate([('All E', '-'), ('Core A', '--')]):
        za.plot(t[take], z[take, j], color=Z_COLOR, ls=style, lw=2.4, label=label, zorder=3)
        ma.plot(t[take], m[take, j], color=M_COLOR, ls=style, lw=2.4, label=label, zorder=3)
    za.set(ylabel='$Z$', ylim=(.5, .9), yticks=[.5, .7, .9])
    ma.set(ylabel=r'$\eta_M M$ (mV)', ylim=(0, .20),
           yticks=[0, .1, .2], xlabel='Time (s)')
    za.tick_params(axis='x', labelbottom=False)
    from matplotlib.lines import Line2D
    za.legend(handles=[Line2D([],[],color='#555555',ls=style,lw=2.4,label=label)
                       for label,style in [('All E','-'),('Core A','--')]],
              loc='upper right', frameon=False, fontsize=16, ncol=2, handlelength=2.6, columnspacing=1.2)
    for ax in (za, ma):
        markings(ax, samples, onset)
    assert z[take].min() > .5 and z[take].max() < .9
    assert m[take].max() < .2
    return dict(Z_columns=[0, 5], M_columns=[0, 1], populations=['All E', 'Core A'],
        Z_limits=[.5, .9], effective_M_limits=[0, .2], distinct_lines='All E solid; Core A dashed. Both Z curves blue; both adaptation curves orange.',
        variable_colors=dict(Z=Z_COLOR, effective_M=M_COLOR), population_styles={'All E':'solid','Core A':'dashed'},
        numerical_offsets=False, per_population_rescaling=False,
        max_abs_CoreA_minus_AllE_Z=float(np.max(abs(np.diff(z[take], axis=1)))),
        max_abs_CoreA_minus_AllE_effectiveM=float(np.max(abs(np.diff(m[take], axis=1)))))


def draw_zooms(fig, spec, a, samples):
    # Reuse the accepted E-only raster painter, with a dummy first item because
    # it historically consumed the second and third states.
    it, ix = np.where(a['raster'])
    prior = len(fig.axes)
    records = previous.zooms(fig, spec, a, [None, *samples[:2]], it*.0001, ix)
    return fig.axes[prior:], records


def draw_maps(fig, spec, a, samples):
    gs = spec.subgridspec(1, 4, wspace=.22)
    axes = []
    records = []
    for i, s in enumerate(samples[1:]):
        ax = fig.add_subplot(gs[i])
        lo = round(s['time_s']*1000) - 25
        count = a['field_1ms'][lo:lo+50].sum(0)
        hz = count / a['cell_e_counts'] / .05
        assert np.array_equal(count.sum(), a['spikes_1ms'][lo:lo+50, 0].sum())
        im = ax.imshow(hz.reshape(20, 20), origin='lower', extent=[0, 20, 0, 20],
                       cmap='magma', norm=PowerNorm(.6, 0, 500), interpolation='nearest')
        for j, xy in enumerate(a['centers_mm']):
            ax.add_patch(Circle(xy, float(a['core_radius_mm']), ec='#2dd4cd', fc='none', lw=1.7))
            ax.text(xy[0], xy[1]+2.1, 'AB'[j], color='#10656a', fontsize=12,
                    ha='center', weight='bold', bbox=dict(fc='white', ec='none', pad=.2, alpha=.8))
        ax.set(xticks=[0, 10, 20], yticks=[0, 10, 20], xlabel='x (mm)')
        if i == 0:
            ax.set_ylabel('y (mm)')
        else:
            ax.tick_params(labelleft=False)
        label = f'{s["number"]}' + (f'  {s["label"]}' if s['label'] else '')
        ax.set_title(f'{label}\n{s["time_s"]:.3f} s', color=s['color'], fontsize=16, pad=10)
        if i == 3:
            cb = fig.colorbar(im, cax=ax.inset_axes([1.10, 0, .06, 1]), ticks=[0, 250, 500])
            cb.set_label('E rate (Hz)', fontsize=17)
        axes.append(ax)
        records.append(dict(number=s['number'], time_s=s['time_s'],
            time_window_s=[lo/1000, (lo+50)/1000], rate_Hz=hz,
            total_E_spikes=int(count.sum()), sheet_fraction_above20Hz=float(np.mean(hz >= 20))))
    return axes, records


def draw_trajectory(fig, spec, a, samples):
    # Keep the original coordinates and5-ms rate/5-ms Gaussian smoothing.
    ax = fig.add_subplot(spec, projection='3d')
    ax.computed_zorder = False
    zt = a['slow_time_ms']/1000
    rate = a['spikes_1ms'][:, 0].reshape(-1, 5).sum(1)/32000/.005
    t = (np.arange(len(rate))+.5)*.005
    xyz = np.c_[np.interp(t, zt, a['Z'][:, 0]), np.interp(t, zt, a['currents'][:, 2]),
                gaussian_filter1d(rate, 1)]
    take = (t >= WINDOW[0]) & (t <= WINDOW[1])
    length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyz[take]/[.5, 1000, 500], axis=0), axis=1))]
    keep = np.unique(np.r_[0, np.flatnonzero(np.diff(np.floor(length/.035)) > 0)+1, take.sum()-1])
    points, times = xyz[take][keep], t[take][keep]
    norm = Normalize(*WINDOW)
    cmap = plt.get_cmap('viridis')
    ax.add_collection3d(Line3DCollection(np.stack([points[:-1], points[1:]], axis=1),
        colors=cmap(norm((times[:-1]+times[1:])/2)), linewidths=1.45, zorder=6))
    ax.set(xlim=(xyz[take, 0].min()-.025, xyz[take, 0].max()+.025),
        ylim=(-25, xyz[take, 1].max()*1.05), zlim=(-15, xyz[take, 2].max()*1.08),
        xlabel='$Z$', ylabel=r'$H_E$ (mV)', zlabel=r'$r_E$ (Hz)')
    ax.view_init(elev=26, azim=-125)
    ax.set_box_aspect((1.25, 1, 1.05))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.labelpad=18
        axis.pane.set_facecolor('#f2f5fa')
        axis.set_major_locator(MaxNLocator(4))
    fig.canvas.draw()
    boxes = []
    records = []
    for s in samples:
        pt = np.array([np.interp(s['time_s'], t, xyz[:, k]) for k in range(3)])
        ax.scatter(*pt, c=s['color'], s=40, edgecolor='white', depthshade=False, zorder=10)
        x, y, _ = proj3d.proj_transform(*pt, ax.get_proj())
        anchor = ax.transData.transform((x, y))
        unit = fig.dpi/72
        for offset in [(14, 14), (-24, 14), (22, -24), (-24, -24), (0, 40), (40, 0), (-40, 0), (0, -40)]:
            center = anchor+np.array(offset)*unit
            box = (center[0]-12*unit, center[1]-12*unit, center[0]+12*unit, center[1]+12*unit)
            if all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in boxes):
                boxes.append(box)
                break
        ax.annotate(str(s['number']), xy=(x, y), xytext=offset, textcoords='offset points',
            ha='center', va='center', fontsize=15, weight='bold',
            bbox=dict(boxstyle='circle,pad=.2', fc='white', ec='#777'),
            arrowprops=dict(arrowstyle='-', color='#777', lw=.6), zorder=20)
        records.append(dict(number=s['number'], time_s=s['time_s'], Z_H_E=pt))
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ax.inset_axes([1.02, .14, .035, .7]))
    cb.set_label('Time (s)', fontsize=18)
    cb.set_ticks([8, 9, 10, 11])
    return [ax], dict(time_window_s=list(WINDOW), coordinates='Mean Z / applied inhibition H_E / all-E rate',
        states=records, interpretation='Same native stochastic SNN trajectory; display cropped only.')


def build(reference, destination):
    row = dict(name='eta0.0005_s9108401', eta_m=.0005, seed=9108401, source=Path(reference['source']))
    a, result = previous.audit.load_small(row['source'], keys=(
        'spikes_1ms', 'regions_1ms', 'field_1ms', 'raster', 'slow_time_ms', 'Z', 'M',
        'currents', 'lfp_time_ms', 'lfp_raw', 'time_ms'))
    previous.old.layout.trim_display(a, reference['display_time_window_s'][1])
    first = result['tracker']['entries'][0]
    assert np.isclose(first['onset_s'], 9.87) and np.isclose(first['confirmation_s'], 10.07)
    assert hashlib.sha256(a['raster'].tobytes()).hexdigest() == reference['raster']['raster_sha256']
    assert np.array_equal(a['field_1ms'].sum(1), a['spikes_1ms'][:, 0])
    samples = states(reference)
    plt.rcParams.update({'font.size':18, 'axes.labelsize':20, 'axes.titlesize':20,
        'xtick.labelsize':17, 'ytick.labelsize':17, 'svg.fonttype':'none', 'pdf.fonttype':42})
    fig = plt.figure(figsize=(25, 18))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.12, 1], left=.072, right=.948,
                            top=.95, bottom=.085, wspace=.25)
    left = outer[0].subgridspec(7, 1, height_ratios=[3.25, .38, 1.38, .48, 1.35, .48, 1.7], hspace=.10)
    bgrid = left[2].subgridspec(2, 1, hspace=.25)
    right = outer[1].subgridspec(2, 1, height_ratios=[1.48, 1], hspace=.23)
    ax = fig.add_subplot(left[0])
    za = fig.add_subplot(bgrid[0], sharex=ax)
    ma = fig.add_subplot(bgrid[1], sharex=ax)
    readout = draw_readout(ax, a, samples, first['onset_s'], reference['selection']['entry_episode_start_s'])
    slow = draw_slow(za, ma, a, row['eta_m'], samples, first['onset_s'])
    zoom_axes, zooms = draw_zooms(fig, left[4], a, samples)
    map_axes, maps = draw_maps(fig, left[6], a, samples)
    traj_axes, trajectory = draw_trajectory(fig, right[0], a, samples)
    before = len(fig.axes)
    field_metadata, field_bars = previous.draw_f(fig, right[1], a, copy.deepcopy(reference['E2']))
    field_axes = [v for v in fig.axes[before:] if v.axison]
    for field_ax in fig.axes[before:]:
        if not field_ax.axison:
            for text in field_ax.texts:
                if text.get_text() == 'F':
                    text.set_visible(False)
            continue
        field_ax.tick_params(labelsize=16)
        field_ax.xaxis.label.set_fontsize(18)
        field_ax.yaxis.label.set_fontsize(18)
    for _, field_ax in field_bars:
        field_ax.set_xticks([-20, 0, 20])
        field_ax.set_yticks([-10, 0, 10, 20])
    for letter, spec in [('A', left[0]), ('B', left[2]), ('C', left[4]), ('D', left[6]), ('E', right[0]), ('F', right[1])]:
        pos = spec.get_position(fig)
        fig.text(pos.x0-.022, pos.y1+.010, letter, weight='bold', fontsize=25, va='bottom')
    fig.canvas.draw()
    for cbar, field in field_bars:
        cb, mp = cbar.get_position(), field.get_position()
        cbar.set_position([cb.x0, mp.y0, cb.width, mp.height])
    links = []
    for s in samples:
        px = [v.transData.transform((s['time_s'], 0))[0] for v in (ax, za, ma)]
        assert max(px)-min(px) < 1e-8
        links.append(dict(number=s['number'], time_s=s['time_s'], aligned_x_px=px))
    for old_map, new_map in [(reference['native_maps'][2], maps[0]), (reference['native_maps'][3], maps[-1])]:
        assert np.array_equal(old_map['rate_Hz'], new_map['rate_Hz'])
    assert field_metadata == reference['E2_display']
    metadata = dict(asset_id='single_seed_zm_entry_and_onset_field', paper_slot='Fig5-A–F',
        publication_status='CANDIDATE', source=str(row['source']), job=result['job'],
        display_time_window_s=list(WINDOW), source_loaded_time_window_s=reference['display_time_window_s'],
        first_entry=first, snapshots=samples, native_maps=maps, raster_zoom_windows=zooms,
        waveform=readout, slow_variables=slow, state_time_links=links, E1=trajectory,
        E2=reference['E2'], E2_display=field_metadata,
        panel_mapping=dict(A='15-contact unfiltered proxy including DC, with fixed pre-recruitment P95 normalization', B='Blue Z; orange adaptation; All E solid and Core A dashed',
            C='Late interictal8.420s and pre-ictal9.420s E-only raster windows',
            D='Native50ms spatial sequence at9.420,9.480,9.600,10.370s',
            E='Same native SNN state trajectory with matching1–5 markers', F='Unchanged model/patient early broadband fields'),
        removed_from_main_figure=['Rest snapshot', 'Interictal spatial snapshot', 'parameter scan'],
        no_new_simulations=True, no_simulation_state_change_in_plotting=True,
        old_spatial_endpoints_identical=True, F_quantities_unchanged=True,
        layout_revision='8–11s; narrower left column; flatter B axes; larger A/B gap and tighter B internal gap.',
        simulation_provenance='Historical manually placed hard cores on the audited C fast carrier; not Fig4 F2.',
        separate_F2_experiment=str(ROOT/'results/topic4_sef_hfo/fig5_f2_native_zm_20260930'),
        human_review='PENDING', agent_visual_review='PENDING',
        producer=str(Path(__file__).resolve()), producer_sha256=sha(__file__))
    groups = dict(A=[ax], B=[za, ma], C=zoom_axes, D=map_axes, E=traj_axes, F=field_axes)
    files = exporter.export(fig, groups, metadata, destination, validate_parameter_map=False)
    plt.close(fig)
    return metadata, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-metadata', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = json.loads(args.source_metadata.read_text())
    ref_path = args.output / 'reference_fig5_metadata.json'
    if args.source_metadata.resolve() != ref_path.resolve():
        shutil.copy2(args.source_metadata, ref_path)
    metadata, files = build(reference, args.output)
    metadata['reference_metadata_sha256'] = sha(ref_path)
    metadata['export_producer'] = str(Path(exporter.__file__).resolve())
    metadata['export_producer_sha256'] = sha(exporter.__file__)
    write(args.output / 'fig5_metadata.json', metadata)
    write(args.output / 'panel_manifest.json', dict(asset_id=metadata['asset_id'], files=files,
        native_matplotlib_export=True, png_dpi=300, svg_text_editable=True,
        human_review='PENDING', agent_visual_review='PENDING'))
    source = Path(metadata['source'])
    write(args.output / 'source_snapshot.json', dict(reference_metadata='reference_fig5_metadata.json',
        reference_metadata_sha256=sha(ref_path), source=str(source),
        source_result_sha256=sha(source / 'result.json'), display_window_s=list(WINDOW),
        snapshots=metadata['snapshots'], no_new_simulations=True))
    print(json.dumps(dict(output=str(args.output), window=WINDOW, samples=metadata['snapshots']), ensure_ascii=False))


if __name__ == '__main__':
    main()
