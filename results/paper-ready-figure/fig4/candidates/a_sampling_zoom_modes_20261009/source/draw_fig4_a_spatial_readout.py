#!/usr/bin/env python3
"""Preserve the original circuit; combine planar connectivity and sampling."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, ConnectionPatch, Ellipse, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/paper-ready-figure/fig4/candidates/a_sampling_zoom_modes_20261009'
SOURCE = OUT / 'source'
RED, BLUE, GREEN = '#D94748', '#3985B8', '#407661'
MM = 1 / 25.4


def draw(fig, *, width=300., height=232., offset_y=0.):
    meta = json.loads((SOURCE / 'mechanism_source.json').read_text())
    geometry = json.loads((SOURCE / 'legacy_a_geometry.json').read_text())

    def axbox(x, y, w, h):
        return fig.add_axes([x / width, (y - offset_y) / height, w / width, h / height])

    def text(x, y, label, **kw):
        return fig.text(x / width, (y - offset_y) / height, label,
                        fontsize=kw.pop('fontsize', 9), **kw)

    def arrow(ax, start, end, color, *, lw=.8, **kw):
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(
            arrowstyle='-|>', color=color, lw=lw, shrinkA=2, shrinkB=3,
            mutation_scale=8, **kw), zorder=8)

    # Frozen images and exact axis boxes restore both original left components.
    restored = []
    with np.load(SOURCE / 'legacy_a_components.npz') as z:
        for index, item in enumerate(geometry['axes']):
            x, y, w, h = np.array(item['position']) * [300, 232, 300, 232]
            ax = axbox(x, y, w, h)
            ax.imshow(z[f'image_{index}'], extent=item['image_extent'], interpolation='lanczos')
            ax.set(xlim=item['xlim'], ylim=item['ylim'], aspect=item['aspect'])
            restored.append(ax)
    local, spatial = restored
    local.set_axis_off()
    x, y, w, h = geometry['local_frame']
    local.add_patch(Rectangle((x, y), w, h, transform=local.transAxes, fill=False,
        ec='#4A4A4A', lw=.65, ls=(0, (2, 1.4)), clip_on=False, zorder=5))
    for item in geometry['texts']:
        style = {k:v for k,v in item.items() if k not in ('position', 'text')}
        text(item['position'][0] * 300, item['position'][1] * 232, item['text'], **style)
    spatial.set(xlabel='x (mm)', ylabel='y (mm)', xticks=[-10, 0, 10], yticks=[-10, 0, 10])
    spatial.tick_params(labelsize=9, length=2.4, pad=2)
    for axis in [spatial.xaxis, spatial.yaxis]:
        axis.label.set_fontsize(10); axis.labelpad = 3
    for spine in spatial.spines.values(): spine.set_visible(True)
    spatial.legend(handles=[Line2D([], [], marker=m, color=c, ls='', ms=3, label=s)
        for m, c, s in [('^', '#ef6868', 'E neuron'), ('o', '#72aadb', 'I neuron')]],
        loc='upper right', fontsize=8.8, frameon=True, framealpha=1,
        facecolor='white', edgecolor='#888888', handlelength=.7,
        handletextpad=.3, labelspacing=.12, borderpad=.25, borderaxespad=.25)
    center = np.array([-8.5, 0.]); fov = np.array(meta['display_contract']['zoom_fov_mm'])
    spatial.add_patch(Rectangle(center - fov / 2, *fov, fill=False, ec='#4a4a4a',
                                lw=.75, ls=(0, (3, 2)), zorder=6))
    connectors = []
    for item in geometry['connectors']:
        line = ConnectionPatch(item['xy1'], item['xy2'], coordsA=local.transAxes,
            coordsB=spatial.transData, arrowstyle='-', color='#4a4a4a', lw=.65,
            ls=(0, (3, 2)), clip_on=False, zorder=4)
        fig.add_artist(line); connectors.append(line)

    # A true coordinate zoom of the same frozen illustrative population.
    with np.load(SOURCE / 'spatial_reference.npz') as z:
        e, i = z['posE'] + 10., z['posI'] + 10.
        overview_contacts = z['contacts']
        names = z['names'].astype(str)
    selected = [list(names).index(f'ICL{n}') for n in range(5, 0, -1)]
    contacts = overview_contacts[selected] + 10.
    contact = contacts[3]  # ICL2, near the right end of the overview shaft.
    kernel = meta['combined_physical_kernel']
    theta = kernel['theta_deg']; angle = np.deg2rad(theta)
    along = np.array([np.cos(angle), np.sin(angle)])
    sigma = meta['readout']['sigma_mm']; r95 = sigma * np.sqrt(-2 * np.log(.05))
    sample = axbox(203, 190, 68, 34)
    sampling_fov = np.array(meta['sampling_zoom']['field_of_view_mm'])
    xlim = contact[0] + np.array([-sampling_fov[0], sampling_fov[0]]) / 2
    ylim = contact[1] + np.array([-sampling_fov[1], sampling_fov[1]]) / 2
    sample.set(xlim=xlim, ylim=ylim, aspect='equal'); sample.set_axis_off()
    rng = np.random.default_rng(106)
    for points, marker, color, count in [(e, '^', RED, 35), (i, 'o', BLUE, 10)]:
        mask = ((points[:, 0] > xlim[0] + .07) & (points[:, 0] < xlim[1] - .07)
                & (points[:, 1] > ylim[0] + .07) & (points[:, 1] < ylim[1] - .07))
        nearby = points[mask]
        shown = nearby[rng.choice(len(nearby), min(count, len(nearby)), replace=False)]
        sample.scatter(*shown.T, marker=marker, c=color, s=9, alpha=.40, lw=0, zorder=1)
    grid = np.linspace(-.85, .85, 160); xx, yy = np.meshgrid(grid, grid)
    weight = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    rgba = np.zeros((*weight.shape, 4)); rgba[:, :, :3] = to_rgb(GREEN); rgba[:, :, 3] = .45 * weight
    sample.imshow(rgba, extent=[contact[0] - .85, contact[0] + .85,
                               contact[1] - .85, contact[1] + .85], origin='lower', zorder=0)
    sample.add_patch(Circle(contact, r95, fill=False, ec=GREEN, lw=.65, alpha=.7))
    nearest = e[np.argsort(np.linalg.norm(e - contact, axis=1))[:5]]
    sample.scatter(*nearest.T, marker='^', c=RED, s=13, alpha=.85, lw=0, zorder=3)
    for point in nearest:
        sample.plot([point[0], contact[0]], [point[1], contact[1]], color=GREEN, lw=.6, alpha=.7)
    sample.plot(*contacts.T, color='#555555', lw=1.2, zorder=4)
    inside = np.all((contacts > [xlim[0], ylim[0]]) & (contacts < [xlim[1], ylim[1]]), axis=1)
    sample.scatter(*contacts[inside].T, s=28, fc='white', ec='#333333', lw=.8, zorder=6)
    sample.scatter(*contact, s=43, fc='white', ec='#333333', lw=1., zorder=7)
    sample.text(contact[0] + .10, contact[1] - .24, 'ICL2', fontsize=8)
    ellipse_centres = []
    # Multiple local circuits share one physical anisotropy orientation.
    # The outer contour is rho=0.7 of the same fitted exponential kernel.
    contour = float(meta['sampling_zoom']['ellipse_contour_level'])
    for index, offset in enumerate([[-1.15, .46], [-.15, .48], [.95, .46], [-.9, -.48], [.3, -.49]]):
        desired = contact + offset
        p = e[np.argmin(np.linalg.norm(e - desired, axis=1))]
        ellipse_centres.append(p.tolist())
        for level, alpha in [(contour, .12), (.55 * contour, .10)]:
            sample.add_patch(Ellipse(p, 2 * kernel['l_par'] * level,
                2 * kernel['l_perp'] * level, angle=theta, fc=RED, ec=RED,
                lw=.75, alpha=alpha, zorder=2))
        sample.plot(*np.array([p - kernel['l_par'] * contour * along,
                              p + kernel['l_par'] * contour * along]).T,
                    color=RED, lw=.7, ls=(0, (3, 2)), zorder=3)
        sample.scatter(*p, marker='^', c=RED, s=24, edgecolors='white', lw=.35, zorder=6)
        if index in [0, 2, 3]:
            target = p + along * kernel['l_par'] * contour * .8
            partner = e[np.argmin(np.linalg.norm(e - target, axis=1))]
            sample.scatter(*partner, marker='^', fc='white', ec=RED, s=16, lw=.65, zorder=5)
            arrow(sample, partner, p, RED, lw=.7, connectionstyle='arc3,rad=.25')
        if index in [1, 4]:
            inhibitory = i[np.argmin(np.linalg.norm(i - (p + [-.10, -.20]), axis=1))]
            sample.scatter(*inhibitory, marker='o', fc='white', ec=BLUE, s=16, lw=.7, zorder=5)
            sample.annotate('', xy=p, xytext=inhibitory, arrowprops=dict(
                arrowstyle='-[,widthB=.4,lengthB=0', color=BLUE, lw=.7,
                shrinkA=3, shrinkB=4, mutation_scale=7, connectionstyle='arc3,rad=.20'), zorder=5)
    sample.plot(contact[0] + np.array([1.04, 1.54]), contact[1] + np.array([-.65, -.65]), color='black', lw=1.1)
    sample.text(contact[0] + 1.29, contact[1] - .83, '0.5 mm', ha='center', fontsize=7.5)
    sample.add_patch(Rectangle((xlim[0], ylim[0]), *sampling_fov, fill=False,
        ec=GREEN, lw=.8, ls=(0, (3, 2)), clip_on=False, zorder=9))
    text(237, 227, 'Local sampling', ha='center', va='bottom', weight='bold', fontsize=11)

    # This new callout deliberately encloses contacts; the old left callout stays remote.
    sampling_center = contact - 10.
    sampling_lo, sampling_hi = sampling_center - sampling_fov / 2, sampling_center + sampling_fov / 2
    spatial.add_patch(Rectangle(sampling_lo, *sampling_fov, fill=False, ec=GREEN,
                                lw=.8, ls=(0, (3, 2)), zorder=9))
    sampling_connectors = []
    for a, b in [((sampling_hi[0], sampling_hi[1]), (0, 1)),
                 ((sampling_hi[0], sampling_lo[1]), (0, 0))]:
        line = ConnectionPatch(a, b, coordsA=spatial.transData, coordsB=sample.transAxes,
            arrowstyle='-', color=GREEN, lw=.65, ls=(0, (3, 2)), clip_on=False, zorder=4)
        fig.add_artist(line); sampling_connectors.append(line)
    np.testing.assert_allclose(np.array([xlim, ylim]).T - 10., [sampling_lo, sampling_hi])
    assert np.all((sampling_center > sampling_lo) & (sampling_center < sampling_hi))

    # Two short envelope sequences below the zoom, distinct from F's oscillatory traces.
    with np.load(SOURCE / 'mode_readout_arrays.npz') as z:
        t, traces, peak_t, wave_names = (z[k] for k in ['time_ms', 'envelopes', 'peak_times_ms', 'names'])
        scale = float(z['common_amplitude_scale'])
    for mode_index, (x, mode, color) in enumerate([(179, 'MTA', '#C63D3A'), (245, 'MTB', '#287FA1')]):
        wave = axbox(x, 155, 47, 22)
        peak_y = []
        for row, name in enumerate(wave_names):
            y = 2 - row + .65 * traces[mode_index, row] / scale
            wave.plot(t, y, color=color, lw=.8)
            value = float(np.interp(peak_t[mode_index, row], t, y)); peak_y.append(value)
            wave.scatter([peak_t[mode_index, row]], [value], s=10, c=color, edgecolors='white', lw=.25, zorder=5)
        wave.plot(peak_t[mode_index], peak_y, color=color, lw=.7, ls=(0, (2, 2)), zorder=2)
        wave.set(xlim=(0, 60), ylim=(-.2, 2.85), yticks=[2, 1, 0],
            yticklabels=wave_names.astype(str), xticks=[0, 30, 60])
        wave.tick_params(labelsize=8, length=2, pad=2)
        wave.spines[['left', 'top', 'right']].set_visible(False); wave.tick_params(axis='y', length=0)
        text(x + 23.5, 178.2, mode, color=color, fontsize=9, ha='center', va='bottom')
    text(237, 183, 'Model readout', ha='center', va='bottom', weight='bold', fontsize=10.5)
    text(237, 146, 'Time (ms)', ha='center', va='bottom', fontsize=9)
    flow = axbox(232, 186.8, 10, 2.6); flow.set(xlim=(0, 1), ylim=(0, 1)); flow.set_axis_off()
    arrow(flow, (.5, .98), (.5, .02), GREEN, lw=1.)

    fig.canvas.draw()
    delta = sample.transData.transform(along) - sample.transData.transform([0, 0])
    np.testing.assert_allclose(np.rad2deg(np.arctan2(delta[1], delta[0])), theta, atol=1e-8)
    assert np.all(np.diff(peak_t[0]) > 0)
    assert np.all(np.diff(peak_t[1]) < 0)
    zoom = json.loads((SOURCE / 'A_zoom_in.json').read_text())
    contact_px = spatial.transData.transform(zoom['source']['contact_xy_mm'])
    clearances = []
    for line in connectors:
        start, end = line.get_path().transformed(line.get_transform()).vertices[[0, -1]]
        step = end - start
        fractions = np.clip((contact_px - start) @ step / (step @ step), 0, 1)
        distance = np.linalg.norm(contact_px - (start + fractions[:, None] * step), axis=1).min()
        clearances.append(float(distance / spatial.bbox.width * 20))
    assert min(clearances) > r95
    return dict(status='CANDIDATE_PENDING_AUTHOR_VISUAL_REVIEW',
        original_left_circuit_restored=True, overview_background_restored=True,
        original_left_title_and_internal_labels_preserved=True,
        combined_third_component=True, formulas_and_lower_explanations_removed=True,
        zoom_center_mm=center.tolist(), zoom_field_of_view_mm=fov.tolist(),
        left_zoom_connector_minimum_contact_distance_mm=min(clearances),
        left_zoom_connectors_clear_of_electrode_footprints=True,
        actual_workpoint_kernel=kernel, kernel_centres_E_coordinates_mm=ellipse_centres,
        enlarged_sampling_contact='ICL2', local_circuit_count=len(ellipse_centres),
        sampling_zoom_center_overview_mm=sampling_center.tolist(),
        sampling_zoom_field_of_view_mm=sampling_fov.tolist(),
        sampling_frame_is_a_coordinate_exact_enlargement=True,
        two_distinct_zoom_roles=True, mode_readouts_below_sampling=True,
        physical_xy_equal_scale=True, kernel_arrows='Schematic partner-sampling arrows, not a graph-edge reconstruction.',
        readout=meta['readout'], mode_showcases=meta['mode_showcases'],
        readout_peak_times_relative_ms=peak_t.tolist(), two_opposite_peak_sequences=True,
        data_lineage='Overview and local sampling zoom share the same frozen illustrative neuron population. Two short contact-envelope examples reuse F events 10 and 8; no exact graph-recovery claim.')


def standalone():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'axes.labelsize':10,
        'xtick.labelsize':9, 'ytick.labelsize':9, 'svg.fonttype':'none', 'pdf.fonttype':42,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.7, 'legend.frameon':False})
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(300 * MM, 87 * MM), dpi=180, facecolor='white')
    record = draw(fig, height=87, offset_y=145)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / 'figures' / f'fig4-panela.{ext}', dpi=220)
    (OUT / 'mechanism_validation.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    standalone()
