#!/usr/bin/env python3
"""Draw a source-backed, planar Fig4A mechanism candidate from frozen arrays."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, ConnectionPatch, Ellipse, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/paper-ready-figure/fig4/candidates/a_spatial_readout_20261009'
SOURCE = OUT / 'source'
RED, BLUE, GREEN, PURPLE = '#D94748', '#3985B8', '#407661', '#855BA3'
MM = 1 / 25.4


def draw(fig, *, width=300., height=232., offset_y=0.):
    """Use physical mm positions; offset_y permits a standalone 83-mm top row."""
    meta = json.loads((SOURCE / 'mechanism_source.json').read_text())
    with np.load(SOURCE / 'spatial_reference.npz') as z:
        e, i, contacts = (z[k] for k in ('posE', 'posI', 'contacts'))
        names, theta = z['names'].astype(str), float(z['theta_deg'])
    theta_r = np.deg2rad(theta)
    along = np.array([np.cos(theta_r), np.sin(theta_r)])
    across = np.array([-along[1], along[0]])
    lp, lt = (meta['EE_kernel'][k] for k in ('l_parallel_mm', 'l_perpendicular_mm'))
    center = np.array([-8.5, 0.])
    fov = np.array(meta['display_contract']['zoom_fov_mm'])
    rng = np.random.default_rng(106)

    def axbox(x, y, w, h):
        return fig.add_axes([x / width, (y - offset_y) / height, w / width, h / height])

    def text(x, y, label, **kw):
        return fig.text(x / width, (y - offset_y) / height, label,
                        fontsize=kw.pop('fontsize', 9), **kw)

    def arrow(ax, start, end, color, *, style='-|>', lw=.8, **kw):
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(
            arrowstyle=style, color=color, lw=lw, shrinkA=2, shrinkB=3,
            mutation_scale=8, **kw), zorder=8)

    # The local frame is a top-down x/y view, with exactly the zoom box's aspect.
    local = axbox(11, 172, 64, 64 * fov[1] / fov[0])
    local.set(xlim=(-fov[0] / 2, fov[0] / 2), ylim=(-fov[1] / 2, fov[1] / 2))
    local.set_aspect('equal'); local.set_axis_off()
    local.add_patch(Rectangle(-fov / 2, *fov, fill=False, ec='#666666',
                              lw=.7, ls=(0, (3, 2)), clip_on=False))
    for points, marker, color, count in [(e, '^', RED, 30), (i, 'o', BLUE, 9)]:
        nearby = points[np.all(np.abs(points - center) < fov / 2 * .92, axis=1)] - center
        shown = nearby[rng.choice(len(nearby), min(count, len(nearby)), replace=False)]
        local.scatter(*shown.T, s=11, marker=marker, c=color, alpha=.38, lw=0, zorder=2)
    for level, alpha in [(1., .055), (.7, .075), (.4, .095)]:
        local.add_patch(Ellipse((0, 0), 2 * lp * level, 2 * lt * level,
                               angle=theta, fc=RED, ec=RED, alpha=alpha, lw=.8))
    # No directed arrow on the major axis: this symmetric kernel permits both signs.
    local.plot(*np.array([-lp * along, lp * along]).T, color=RED, lw=1., ls=(0, (4, 2)))
    local.plot(*np.array([-lt * across, lt * across]).T, color=RED, lw=.75, ls=':')
    local.plot([0, .63], [0, 0], color='#777777', lw=.65, ls=':')
    local.add_patch(Arc((0, 0), .43, .43, theta1=theta, theta2=0, color='#555555', lw=.7))
    local.text(.26, -.065, r'$\theta$', fontsize=10, color='#444444')
    local.text(.39, -.31, r'$\ell_{\parallel}$', color=RED, fontsize=10)
    local.text(.15, .25, r'$\ell_{\perp}$', color=RED, fontsize=10)
    local.scatter([0], [0], s=51, marker='^', c=RED, edgecolors='white', lw=.5, zorder=9)
    for displacement in [lp * along * .82, -lp * along * .84]:
        local.scatter(*displacement, s=27, marker='^', facecolors='white', edgecolors=RED, lw=.8, zorder=8)
        arrow(local, displacement, np.zeros(2), RED, lw=.9,
              connectionstyle='arc3,rad=.28')
    # Coordinate cue fixes the orientation in the plane, independently of the shafts.
    origin = np.array([-.69, -.46])
    arrow(local, origin, origin + [.28, 0], '#444444', lw=.65)
    arrow(local, origin, origin + [0, .24], '#444444', lw=.65)
    local.text(-.385, -.48, 'x', fontsize=8)
    local.text(-.72, -.18, 'y', fontsize=8)
    local.plot([.17, .67], [-.49, -.49], color='black', lw=1.3)
    local.text(.42, -.58, '0.5 mm', ha='center', fontsize=8)
    text(43, 227, 'Local E/I circuit', ha='center', weight='bold', fontsize=11)
    local.text(-.70, .50, 'E→E kernel', color=RED, fontsize=9, weight='bold')

    # Keep the E/I/adaptation loop separate from spatial placement of neurons.
    loop = axbox(11, 153, 64, 15)
    loop.set(xlim=(0, 1), ylim=(0, 1)); loop.set_axis_off()
    loop.scatter([.18], [.5], s=54, marker='^', c=RED)
    loop.scatter([.51], [.5], s=40, marker='o', facecolors='white', edgecolors=BLUE, lw=1.1)
    loop.text(.83, .5, 'm', color=PURPLE, fontsize=11, weight='bold', ha='center', va='center')
    arrow(loop, (.23, .66), (.45, .66), RED, connectionstyle='arc3,rad=-.30')
    arrow(loop, (.46, .38), (.23, .38), BLUE, style='-[,widthB=.4,lengthB=0', connectionstyle='arc3,rad=-.30')
    arrow(loop, (.20, .82), (.78, .70), PURPLE, connectionstyle='arc3,rad=-.16')
    arrow(loop, (.79, .31), (.20, .34), PURPLE, style='-[,widthB=.4,lengthB=0', connectionstyle='arc3,rad=-.16')
    loop.text(.18, .0, 'E', color=RED, fontsize=8, ha='center')
    loop.text(.51, .0, 'I', color=BLUE, fontsize=8, ha='center')
    loop.text(.83, .0, 'Adaptation', color=PURPLE, fontsize=8, ha='center')

    # The frozen network keeps all contacts and the left-side representative zoom.
    spatial = axbox(91, 161, 62, 62)
    for points, marker, color, count in [(e, '^', RED, 850), (i, 'o', BLUE, 235)]:
        shown = points[rng.choice(len(points), count, replace=False)]
        spatial.scatter(*shown.T, marker=marker, c=color, s=2.8, alpha=.42, lw=0, rasterized=True)
    sigma = meta['readout']['sigma_mm']
    r95 = sigma * np.sqrt(-2 * np.log(.05))
    for contact in contacts:
        spatial.add_patch(Circle(contact, r95, fc=GREEN, ec='none', alpha=.14))
    for shaft in ['SCL', 'ICL']:
        ids = np.flatnonzero(np.char.startswith(names, shaft))
        spatial.plot(*contacts[ids].T, color='#444444', lw=.8, zorder=4)
    spatial.scatter(*contacts.T, s=14, fc='white', ec='#333333', lw=.7, zorder=5)
    # Identical physical kernel orientation at several arbitrary network locations.
    for xy, level in [(center, 1.), ([-1.8, .4], 2.), ([5., 5.8], 2.), ([4., -1.], 2.)]:
        spatial.add_patch(Ellipse(xy, 2 * lp * level, 2 * lt * level,
                                 angle=theta, fc='white', ec=RED, lw=.9, alpha=.92, zorder=6))
        a, b = np.asarray(xy) - lp * level * along, np.asarray(xy) + lp * level * along
        spatial.plot(*np.array([a, b]).T, color=RED, lw=.65, zorder=7)
    lo, hi = center - fov / 2, center + fov / 2
    spatial.add_patch(Rectangle(lo, *fov, fill=False, ec='#555555', lw=.75, ls=(0, (3, 2)), zorder=8))
    connectors = []
    for a, b in [((1, 1), lo + [0, fov[1]]), ((1, 0), lo)]:
        line = ConnectionPatch(a, b, coordsA=local.transAxes, coordsB=spatial.transData,
            color='#666666', lw=.65, ls=(0, (3, 2)), clip_on=False)
        fig.add_artist(line); connectors.append(line)
    spatial.set(xlim=(-10, 10), ylim=(-10, 10), xticks=[-10, 0, 10], yticks=[-10, 0, 10],
                xlabel='x (mm)', ylabel='y (mm)')
    spatial.set_aspect('equal')
    spatial.tick_params(labelsize=9, length=2.4, pad=2)
    for axis in [spatial.xaxis, spatial.yaxis]:
        axis.label.set_fontsize(10); axis.labelpad = 3
    for spine in spatial.spines.values(): spine.set_visible(True)
    text(122, 227, 'Spatial network', ha='center', weight='bold', fontsize=11)
    spatial.legend(handles=[Line2D([], [], marker=m, color=c, ls='', ms=3, label=s)
                            for m, c, s in [('^', RED, 'E'), ('o', BLUE, 'I')]],
                   loc='upper left', ncol=2, fontsize=8, frameon=True, framealpha=.95,
                   handletextpad=.25, columnspacing=.65, borderpad=.3)

    # Gaussian readout is explicitly a sampling operator, not an electrical field.
    sample = axbox(173, 180, 47, 43)
    sample.set(xlim=(-.85, .85), ylim=(-.78, .78)); sample.set_aspect('equal'); sample.set_axis_off()
    grid = np.linspace(-.85, .85, 220)
    xx, yy = np.meshgrid(grid, grid)
    weight = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    from matplotlib.colors import to_rgb
    rgba = np.zeros((*weight.shape, 4)); rgba[:, :, :3] = to_rgb(GREEN); rgba[:, :, 3] = .40 * weight
    sample.imshow(rgba, extent=(-.85, .85, -.85, .85), origin='lower', zorder=0)
    sample.add_patch(Circle((0, 0), r95, fill=False, ec=GREEN, lw=.65, ls=(0, (3, 2)), alpha=.7))
    q = int(np.flatnonzero(names == 'ICL7')[0])
    nearby = e[np.all(np.abs(e - contacts[q]) < .74, axis=1)] - contacts[q]
    shown = nearby[rng.choice(len(nearby), 24, replace=False)]
    sample.scatter(*shown.T, marker='^', c=RED, s=13, alpha=.5, lw=0, zorder=2)
    for p in shown[np.argsort(np.linalg.norm(shown, axis=1))[:7]]:
        strength = np.exp(-np.dot(p, p) / (2 * sigma ** 2))
        sample.plot([p[0], 0], [p[1], 0], color=GREEN, alpha=float(.15 + .65 * strength), lw=.45 + strength, zorder=1)
    sample.plot([-.76, .76], [.20, -.20], color='#555555', lw=2, zorder=3)
    sample.scatter([0], [0], s=65, fc='white', ec='#333333', lw=1.2, zorder=5)
    sample.text(.20, .53, 'SEEG\ncontact', ha='center', fontsize=8.5)
    arrow(sample, (.19, .43), (.035, .065), '#555555', lw=.65)
    text(196.5, 227, 'Local sampling', ha='center', weight='bold', fontsize=11)
    text(196.5, 176, 'Nearer neurons weigh more', color=GREEN, ha='center', fontsize=8.5)
    text(196.5, 167, r'$w_{qi}\propto e^{-d_{qi}^{2}/(2\sigma^{2})}$', color=GREEN, ha='center', fontsize=10)
    text(196.5, 158, r'$\sigma=0.25$ mm', color=GREEN, ha='center', fontsize=8.5)

    flow = axbox(155, 184, 18, 30); flow.set(xlim=(0, 1), ylim=(0, 1)); flow.set_axis_off()
    arrow(flow, (.05, .5), (.94, .5), GREEN, lw=1.1)
    flow2 = axbox(222, 183, 16, 30); flow2.set(xlim=(0, 1), ylim=(0, 1)); flow2.set_axis_off()
    arrow(flow2, (.03, .5), (.99, .5), GREEN, lw=1.1)

    wave = axbox(248, 180, 42, 40)
    with np.load(SOURCE / 'waveform_arrays.npz') as z:
        t = z['time_ms']; traces = z['filtered_contact_activity']; order = z['contact_names'].astype(str)
        scale = float(z['common_amplitude_scale'])
    selected = (t >= 100) & (t <= 270)
    wave_names = ['ICL9', 'ICL6', 'ICL3']
    for row, name in enumerate(wave_names):
        j = int(np.flatnonzero(order == name)[0])
        wave.plot(t[selected] - 100, 2 - row + .43 * traces[j, selected] / scale,
                  color='#555555', lw=.8)
    wave.set(xlim=(0, 170), ylim=(-.6, 2.6), yticks=[2, 1, 0], yticklabels=wave_names,
             xticks=[0, 80, 160], xlabel='Time (ms)')
    wave.tick_params(labelsize=8, length=2, pad=2)
    wave.xaxis.label.set_fontsize(9); wave.xaxis.labelpad = 3
    wave.spines[['left', 'top', 'right']].set_visible(False)
    wave.tick_params(axis='y', length=0)
    text(269, 227, 'Model readout', ha='center', weight='bold', fontsize=11)
    text(269, 166, 'Weighted E activity', ha='center', fontsize=8.5)
    text(269, 158, '30–80 Hz activity', ha='center', fontsize=8.5)

    fig.canvas.draw()
    endpoints = [line.get_path().transformed(line.get_transform()).vertices[[0, -1]] for line in connectors]
    contact_px = spatial.transData.transform(contacts)
    clearances = []
    for start, end in endpoints:
        delta = end - start
        fractions = np.clip((contact_px - start) @ delta / (delta @ delta), 0, 1)
        distance = np.linalg.norm(contact_px - (start + fractions[:, None] * delta), axis=1).min()
        clearances.append(float(distance / spatial.bbox.width * 20))
    assert min(clearances) > r95
    # Orientation is checked in displayed coordinates, so unequal axis scaling cannot rotate it.
    for ax in [local, spatial]:
        delta = ax.transData.transform(along) - ax.transData.transform([0, 0])
        np.testing.assert_allclose(np.rad2deg(np.arctan2(delta[1], delta[0])), theta, atol=1e-8)
    return dict(status='CANDIDATE_PENDING_AUTHOR_VISUAL_REVIEW',
        zoom_center_mm=center.tolist(), zoom_field_of_view_mm=fov.tolist(),
        reference_kernel=meta['EE_kernel'], planar_angle_consistent=True,
        connector_minimum_contact_distance_mm=min(clearances),
        connectors_clear_of_electrode_footprints=True, connectors_attached_to_both_frames=True,
        readout=meta['readout'], waveform_contacts=wave_names, waveform_window_relative_to_F_ms=[100, 270],
        kernel_contours='Local and zoom: rho=0.4/0.7/1; three distant examples: rho=2, at physical scale.',
        data_lineage='A retains its frozen schematic substrate. Waveforms are exact F values, not a simulation of the illustrated substrate.')


def standalone():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'svg.fonttype':'none', 'pdf.fonttype':42,
                         'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.7})
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(300 * MM, 83 * MM), dpi=180, facecolor='white')
    record = draw(fig, height=83, offset_y=149)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / 'figures' / f'fig4-panela.{ext}', dpi=220)
    (OUT / 'mechanism_validation.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    standalone()
