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
OUT = ROOT / 'results/paper-ready-figure/fig4/candidates/a_circuit_sampling_sequence_20261009'
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

    # The third component combines actual physical dimensions, E/E/I positions,
    # the fitted elliptical kernel and local readout footprints in one plane.
    with np.load(SOURCE / 'sequence_arrays.npz') as z:
        data = {k:z[k] for k in z.files}
    contacts, e, i = (data[k] for k in ('contacts', 'positions_E', 'positions_I'))
    kernel = meta['combined_physical_kernel']
    theta = kernel['theta_deg']; angle = np.deg2rad(theta)
    along = np.array([np.cos(angle), np.sin(angle)])
    sigma = meta['readout']['sigma_mm']; r95 = sigma * np.sqrt(-2 * np.log(.05))
    sample = axbox(168, 161, 64, 51)
    # A single enlarged ICL3 neighbourhood makes the two spatial operators legible.
    contact = contacts[2]
    xlim = contact[0] + np.array([-1.25, 1.25])
    ylim = contact[1] + np.array([-1.0, -1.0 + 2.5 * 51 / 64])
    sample.set(xlim=xlim, ylim=ylim, aspect='equal'); sample.set_axis_off()
    rng = np.random.default_rng(106)
    for points, marker, color, count in [(e, '^', RED, 27), (i, 'o', BLUE, 8)]:
        mask = ((points[:, 0] > xlim[0] + .07) & (points[:, 0] < xlim[1] - .07)
                & (points[:, 1] > ylim[0] + .07) & (points[:, 1] < ylim[1] - .07))
        nearby = points[mask]
        shown = nearby[rng.choice(len(nearby), min(count, len(nearby)), replace=False)]
        sample.scatter(*shown.T, marker=marker, c=color, s=15, alpha=.40, lw=0, zorder=1)
    grid = np.linspace(-.85, .85, 160); xx, yy = np.meshgrid(grid, grid)
    weight = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    rgba = np.zeros((*weight.shape, 4)); rgba[:, :, :3] = to_rgb(GREEN); rgba[:, :, 3] = .45 * weight
    sample.imshow(rgba, extent=[contact[0] - .85, contact[0] + .85,
                               contact[1] - .85, contact[1] + .85], origin='lower', zorder=0)
    sample.add_patch(Circle(contact, r95, fill=False, ec=GREEN, lw=.65, alpha=.7))
    nearest = e[np.argsort(np.linalg.norm(e - contact, axis=1))[:5]]
    sample.scatter(*nearest.T, marker='^', c=RED, s=19, alpha=.85, lw=0, zorder=3)
    for point in nearest:
        sample.plot([point[0], contact[0]], [point[1], contact[1]], color=GREEN, lw=.7, alpha=.7)
    shaft = contacts[-1] - contacts[0]; shaft /= np.linalg.norm(shaft)
    sample.plot(*np.array([contact - 1.22 * shaft, contact + 1.22 * shaft]).T,
                color='#555555', lw=1.3, zorder=4)
    sample.scatter(*contact, s=70, fc='white', ec='#333333', lw=1.1, zorder=6)
    sample.text(contact[0] + .16, contact[1] - .22, 'ICL3', fontsize=8)
    desired = contact + [-.32, .34]
    p = e[np.argmin(np.linalg.norm(e - desired, axis=1))]
    ellipse_centres = [p.tolist()]
    for level, alpha in [(1., .11), (.65, .10), (.35, .10)]:
        sample.add_patch(Ellipse(p, 2 * kernel['l_par'] * level,
            2 * kernel['l_perp'] * level, angle=theta, fc=RED, ec=RED,
            lw=.85, alpha=alpha, zorder=2))
    sample.plot(*np.array([p - kernel['l_par'] * along, p + kernel['l_par'] * along]).T,
                color=RED, lw=.85, ls=(0, (3, 2)), zorder=3)
    sample.scatter(*p, marker='^', c=RED, s=49, edgecolors='white', lw=.4, zorder=6)
    for sign in [-1, 1]:
        target = p + sign * along * kernel['l_par'] * .8
        partner = e[np.argmin(np.linalg.norm(e - target, axis=1))]
        sample.scatter(*partner, marker='^', fc='white', ec=RED, s=28, lw=.7, zorder=5)
        arrow(sample, partner, p, RED, lw=.85, connectionstyle='arc3,rad=.25')
    # A nearby inhibitory neuron belongs to this same planar local circuit.
    inhibitory = i[np.argmin(np.linalg.norm(i - (p + [-.1, -.35]), axis=1))]
    sample.scatter(*inhibitory, marker='o', fc='white', ec=BLUE, s=29, lw=.8, zorder=5)
    sample.annotate('', xy=p, xytext=inhibitory, arrowprops=dict(
        arrowstyle='-[,widthB=.4,lengthB=0', color=BLUE, lw=.8,
        shrinkA=4, shrinkB=5, mutation_scale=8, connectionstyle='arc3,rad=.20'), zorder=5)
    sample.plot(contact[0] + np.array([.40, .90]), contact[1] + np.array([-.75, -.75]),
                color='black', lw=1.2)
    sample.text(contact[0] + .65, contact[1] - .90, '0.5 mm', ha='center', fontsize=8)
    title_y = geometry['texts'][0]['position'][1] * 232
    text(200, title_y, 'Local sampling', ha='center', va='bottom', weight='bold', fontsize=11)

    # Same frozen F event, same per-contact time coordinates and common scale.
    wave = axbox(249, 161, 43, 51)
    t, traces = data['time_ms'], data['filtered_activity']
    scale = float(data['common_amplitude_scale']); peak_t = data['peak_times_ms']
    peak_y = []
    for row, name in enumerate(data['names']):
        y = 4 - row + .38 * traces[row] / scale
        wave.plot(t, y, color='#555555', lw=.8)
        value = float(np.interp(peak_t[row], t, y)); peak_y.append(value)
        wave.scatter([peak_t[row]], [value], s=14, c='#222222', edgecolors='white', lw=.3, zorder=5)
    wave.plot(peak_t, peak_y, color='#555555', lw=.75, ls=(0, (2, 2)), zorder=2)
    wave.set(xlim=(0, 80), ylim=(-.65, 4.75), yticks=np.arange(4, -1, -1),
        yticklabels=data['names'].astype(str), xticks=[0, 40, 80], xlabel='Time (ms)')
    wave.tick_params(labelsize=8, length=2, pad=2)
    wave.xaxis.label.set_fontsize(9); wave.xaxis.labelpad = 3
    wave.spines[['left', 'top', 'right']].set_visible(False); wave.tick_params(axis='y', length=0)
    text(270.5, title_y, 'Model readout', ha='center', va='bottom', weight='bold', fontsize=11)
    flow = axbox(157.5, 177, 9, 40); flow.set(xlim=(0, 1), ylim=(0, 1)); flow.set_axis_off()
    arrow(flow, (.05, .5), (.95, .5), GREEN, lw=1.)
    flow2 = axbox(232.5, 177, 7, 40); flow2.set(xlim=(0, 1), ylim=(0, 1)); flow2.set_axis_off()
    arrow(flow2, (.05, .5), (.95, .5), GREEN, lw=1.)

    fig.canvas.draw()
    delta = sample.transData.transform(along) - sample.transData.transform([0, 0])
    np.testing.assert_allclose(np.rad2deg(np.arctan2(delta[1], delta[0])), theta, atol=1e-8)
    assert np.all(np.diff(peak_t) > 0)
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
        original_left_circuit_restored=True, first_two_components_restored=True,
        original_left_title_and_internal_labels_preserved=True,
        combined_third_component=True, formulas_and_lower_explanations_removed=True,
        zoom_center_mm=center.tolist(), zoom_field_of_view_mm=fov.tolist(),
        connector_minimum_contact_distance_mm=min(clearances),
        connectors_clear_of_electrode_footprints=True,
        actual_workpoint_kernel=kernel, kernel_centres_E_coordinates_mm=ellipse_centres,
        enlarged_sampling_contact='ICL3',
        physical_xy_equal_scale=True, kernel_arrows='Schematic partner-sampling arrows, not a graph-edge reconstruction.',
        readout=meta['readout'], sequence=meta['sequence'],
        readout_peak_times_relative_ms=peak_t.tolist(), readout_peaks_strictly_ordered=True,
        data_lineage='First two components retain the accepted illustrative substrate. Third and fourth use the actual F-event workpoint geometry, kernel and traces.')


def standalone():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'axes.labelsize':10,
        'xtick.labelsize':9, 'ytick.labelsize':9, 'svg.fonttype':'none', 'pdf.fonttype':42,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.7, 'legend.frameon':False})
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(300 * MM, 83 * MM), dpi=180, facecolor='white')
    record = draw(fig, height=83, offset_y=149)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / 'figures' / f'fig4-panela.{ext}', dpi=220)
    (OUT / 'mechanism_validation.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    standalone()
