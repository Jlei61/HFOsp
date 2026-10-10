#!/usr/bin/env python3
"""Preserve the original circuit; combine planar connectivity and sampling."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Ellipse, Rectangle
import matplotlib.patheffects as pe
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/paper-ready-figure/fig4/candidates/a_local_sampling_sigma_20261010'
SOURCE = OUT / 'source'
RED, BLUE, GREEN = '#D94748', '#3985B8', '#257653'
CALLOUT = '#666666'
MM = 1 / 25.4


def draw(fig, *, width=300., height=232., offset_y=0.):
    meta = json.loads((SOURCE / 'mechanism_source.json').read_text())
    geometry = json.loads((SOURCE / 'legacy_a_geometry.json').read_text())
    left_scale = meta['layout_refinement']['left_circuit_scale']
    old_left_box = np.array(geometry['axes'][0]['position']) * [300, 232, 300, 232]
    new_left_box = old_left_box.copy()
    new_left_box[1] -= old_left_box[3] * (left_scale - 1) / 2
    new_left_box[2:] *= left_scale

    def axbox(x, y, w, h):
        return fig.add_axes([x / width, (y - offset_y) / height, w / width, h / height])

    def text(x, y, label, **kw):
        return fig.text(x / width, (y - offset_y) / height, label,
                        fontsize=kw.pop('fontsize', 9), **kw)

    # Preserve the original left circuit; redraw the overview from frozen samples.
    restored = []
    with np.load(SOURCE / 'legacy_a_components.npz') as z:
        for index, item in enumerate(geometry['axes']):
            x, y, w, h = np.array(item['position']) * [300, 232, 300, 232]
            if index == 0:
                x, y, w, h = new_left_box
            else:
                x += meta['layout_refinement']['sheet_shift_x_mm']
                sheet_scale = meta['layout_refinement']['sheet_scale']
                x -= w * (sheet_scale - 1) / 2
                y -= h * (sheet_scale - 1) / 2
                w *= sheet_scale
                h *= sheet_scale
                sheet_box = [x, y, w, h]
            ax = axbox(x, y, w, h)
            if index == 0:
                ax.imshow(z['image_0'], extent=item['image_extent'], interpolation='lanczos')
            ax.set(xlim=item['xlim'], ylim=item['ylim'], aspect=item['aspect'])
            restored.append(ax)
    local, spatial = restored
    local.set_axis_off()
    x, y, w, h = geometry['local_frame']
    local.add_patch(Rectangle((x, y), w, h, transform=local.transAxes, fill=False,
        ec='#4A4A4A', lw=.65, ls=(0, (2, 1.4)), clip_on=False, zorder=5))
    for item in geometry['texts']:
        style = {k:v for k,v in item.items() if k not in ('position', 'text')}
        title_gap = item['position'][1] * 232 - old_left_box[1] - old_left_box[3]
        text(new_left_box[0] + new_left_box[2] / 2,
             new_left_box[1] + new_left_box[3] + title_gap, item['text'], **style)
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
        names = z['names'].astype(str)
    with np.load(SOURCE / 'rigid_contact_geometry.npz') as z:
        np.testing.assert_array_equal(names, z['names'])
        overview_contacts = z['rigid_xy_mm']
    sigma = meta['readout']['sigma_mm']; r95 = sigma * np.sqrt(-2 * np.log(.05))
    with np.load(SOURCE / 'overview_neuron_samples.npz') as z:
        for key, marker, color, size, alpha in [('I', 'o', '#1f77b4', 18, .46),
                                               ('E', '^', '#d62728', 24, .44)]:
            spatial.scatter(*z[key].T, marker=marker, c=color,
                s=size * (66.6414 / 127.) ** 2, alpha=alpha, lw=0, zorder=1)
    straightness = {}
    for shaft in ['SCL', 'ICL']:
        points = overview_contacts[np.char.startswith(names, shaft)]
        direction = points[-1] - points[0]
        normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
        residual = np.abs((points - points[0]) @ normal).max()
        assert residual < 1e-10
        straightness[shaft] = float(residual)
        spatial.plot(*points[[0, -1]].T, color='#4c4c4c', lw=1.25,
                     solid_capstyle='round', zorder=3)
        spatial.scatter(*points.T, s=19, fc='white', ec='#333333', lw=.75, zorder=4)
    selected = [list(names).index(f'ICL{n}') for n in range(11, 0, -1)]
    contacts = overview_contacts[selected] + 10.
    sampling_names = meta['sampling_zoom']['labeled_contacts']
    display_names = meta['sampling_zoom']['display_labels']
    sampling_contacts = overview_contacts[[list(names).index(n) for n in sampling_names]] + 10.
    contact = sampling_contacts[[0, -1]].mean(axis=0)
    kernel = meta['combined_physical_kernel']
    theta = kernel['theta_deg']; angle = np.deg2rad(theta)
    along = np.array([np.cos(angle), np.sin(angle)])
    sample = axbox(*meta['layout_refinement']['sampling_axes_mm'])
    right_center = 248.5
    sampling_fov = np.array(meta['sampling_zoom']['field_of_view_mm'])
    xlim = contact[0] + np.array([-sampling_fov[0], sampling_fov[0]]) / 2
    ylim = contact[1] + np.array([-sampling_fov[1], sampling_fov[1]]) / 2
    sample.set(xlim=xlim, ylim=ylim, aspect='equal'); sample.set_axis_off()
    rng = np.random.default_rng(106)
    population_spec = meta['local_population_display']
    symbol_scale = population_spec['symbol_scale']
    contour = float(meta['sampling_zoom']['ellipse_contour_level'])
    basis = np.column_stack([along, [-along[1], along[0]]])
    radii = contour * np.array([kernel['l_par'], kernel['l_perp']])
    ellipse_centres = [e[np.argmin(np.linalg.norm(e - (contact + offset), axis=1))]
                       for offset in population_spec['ellipse_offsets_mm']]

    def elliptical_radius(points, centre):
        return np.linalg.norm(((points - centre) @ basis) / radii, axis=1)

    # Display the existing Gaussian operator at its physical sigma, only here.
    # The zoom clips its tails; there is no artificial sampling-radius cutoff.
    footprint = meta['sampling_zoom']['footprint_display']
    np.testing.assert_allclose(footprint['sigma_mm'], sigma)
    extent = footprint['render_extent_sigma'] * sigma
    grid = np.linspace(-extent, extent, 257)
    xx, yy = np.meshgrid(grid, grid)
    weights = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    rgba = np.zeros((*weights.shape, 4))
    rgba[:, :, :3] = to_rgb(GREEN)
    rgba[:, :, 3] = footprint['maximum_alpha'] * weights
    for point in sampling_contacts:
        sample.imshow(rgba, extent=[point[0] - extent, point[0] + extent,
                                   point[1] - extent, point[1] + extent],
                      origin='lower', interpolation='bilinear', zorder=0)

    # The background uses the same hollow E symbols as the local populations.
    for points, marker, color, count in [(e, '^', RED, 22), (i, 'o', BLUE, 6)]:
        mask = ((points[:, 0] > xlim[0] + .07) & (points[:, 0] < xlim[1] - .07)
                & (points[:, 1] > ylim[0] + .07) & (points[:, 1] < ylim[1] - .07))
        for centre in ellipse_centres:
            mask &= elliptical_radius(points, centre) > 1.15
        nearby = points[mask]
        shown = nearby[rng.choice(len(nearby), min(count, len(nearby)), replace=False)]
        sample.scatter(*shown.T, marker=marker, fc='white', ec=color,
                       s=(12 if marker == '^' else 10) * symbol_scale ** 2,
                       alpha=.55, lw=.6, zorder=1)
    sample.plot(*contacts[[0, -1]].T, color='#4c4c4c', lw=1.6, zorder=4,
                solid_capstyle='round')
    inside = np.all((contacts > [xlim[0], ylim[0]]) & (contacts < [xlim[1], ylim[1]]), axis=1)
    assert inside.sum() == 3
    assert set(names[selected][inside]) == set(sampling_names)
    sample.scatter(*sampling_contacts.T, s=38, fc='white', ec='#333333', lw=1., zorder=7)
    sampling_labels = []
    for name, point in zip(display_names, sampling_contacts):
        sampling_labels.append(sample.annotate(name, point, xytext=(0, 8),
            textcoords='offset points', ha='center', va='bottom', fontsize=8,
            bbox=dict(fc='white', ec='none', alpha=.85, pad=.1), zorder=8))
    populations = []
    circuit_rng = np.random.default_rng(866)
    shaft_direction = contacts[-1] - contacts[0]
    shaft_normal = np.array([-shaft_direction[1], shaft_direction[0]]) / np.linalg.norm(shaft_direction)
    # Randomly spaced E/I symbols inside each neighbourhood replace neuron-like
    # arrowheads. All chosen coordinates still belong to the frozen population.
    for index, p in enumerate(ellipse_centres):
        for level, alpha in [(contour, .12), (.55 * contour, .10)]:
            sample.add_patch(Ellipse(p, 2 * kernel['l_par'] * level,
                2 * kernel['l_perp'] * level, angle=theta, fc=RED, ec=RED,
                lw=.75, alpha=alpha, zorder=2))
        for attempt in range(64):
            placed = [p]
            groups = {'E': [p], 'I': []}
            complete = True
            for key, points, count in [('I', i, population_spec['I_counts'][index]),
                                       ('E', e, population_spec['E_counts'][index] - 1)]:
                # Keep complete symbols inside, especially blue circles. Retry
                # the deterministic subsampling if one greedy order crowds E.
                mask = (elliptical_radius(points, p) < (.76 if key == 'I' else .84))
                mask &= np.abs((points - contact) @ shaft_normal) > .10
                candidates = points[mask]
                for point in candidates[circuit_rng.permutation(len(candidates))]:
                    if np.min(np.linalg.norm(np.array(placed) - point, axis=1)) < .10:
                        continue
                    groups[key].append(point); placed.append(point)
                    count -= 1
                    if count == 0:
                        break
                if count:
                    complete = False
                    break
            if complete:
                break
        assert complete, (index, key, count)
        for key, marker, color, size in [('E', '^', RED, 14), ('I', 'o', BLUE, 11)]:
            points = np.array(groups[key])
            assert np.all(elliptical_radius(points, p) < 1.)
            sample.scatter(*points.T, marker=marker, fc='white', ec=color,
                           s=size * symbol_scale ** 2, lw=.7, zorder=6)
        assert len(groups['E']) > len(groups['I']) > 0
        populations.append({key: np.array(value).tolist() for key, value in groups.items()})
    bar_x = xlim[1] - .48
    sample.plot([bar_x - .25, bar_x + .25], [ylim[0] + .23] * 2, color='black', lw=1.1)
    sample.text(bar_x, ylim[0] + .05, '0.5 mm', ha='center', fontsize=7.5)
    sample.add_patch(Rectangle((xlim[0], ylim[0]), *sampling_fov, fill=False,
        ec=CALLOUT, lw=1.1, ls=(0, (3, 1.7)), clip_on=False, zorder=9))
    text(right_center, 223, 'Local sampling', ha='center', va='bottom', weight='bold', fontsize=11)

    # This new callout deliberately encloses contacts; the old left callout stays remote.
    sampling_center = contact - 10.
    sampling_lo, sampling_hi = sampling_center - sampling_fov / 2, sampling_center + sampling_fov / 2
    sampling_box = Rectangle(sampling_lo, *sampling_fov, fill=False,
        ec=CALLOUT, lw=1.55, ls=(0, (3, 1.7)), zorder=9)
    sampling_box.set_path_effects([pe.Stroke(linewidth=3., foreground='white'), pe.Normal()])
    spatial.add_patch(sampling_box)
    sampling_connectors = []
    for a, b in [((sampling_hi[0], sampling_hi[1]), (0, 1)),
                 ((sampling_hi[0], sampling_lo[1]), (0, 0))]:
        line = ConnectionPatch(a, b, coordsA=spatial.transData, coordsB=sample.transAxes,
            arrowstyle='-', color=CALLOUT, lw=.85, ls=(0, (3, 2)), clip_on=False, zorder=4)
        fig.add_artist(line); sampling_connectors.append(line)
    np.testing.assert_allclose(np.array([xlim, ylim]).T - 10., [sampling_lo, sampling_hi])
    assert np.all((sampling_center > sampling_lo) & (sampling_center < sampling_hi))
    assert np.all((sampling_contacts - 10. > sampling_lo)
                  & (sampling_contacts - 10. < sampling_hi))
    assert np.all(np.diff([int(n.removeprefix('ICL')) for n in sampling_names]) == 1)

    # True short burst crops from F, with no nonlinear sharpening or time warping.
    with np.load(SOURCE / 'burst_readout_arrays.npz') as z:
        t, traces, peak_t, wave_names = (z[k] for k in ['time_ms', 'waveforms', 'peak_times_ms', 'names'])
        scale = float(z['common_amplitude_scale'])
        wave_labels = z['display_labels'].astype(str)
    np.testing.assert_array_equal(sampling_names, wave_names.astype(str))
    np.testing.assert_array_equal(display_names, wave_labels)
    for mode_index, (mode, color) in enumerate([('MTA', '#C63D3A'), ('MTB', '#287FA1')]):
        x, y, w, h = meta['layout_refinement']['readout_axes_mm'][mode_index]
        wave = axbox(x, y, w, h)
        peak_y = []
        for row, name in enumerate(wave_names):
            y = 2 - row + .47 * traces[mode_index, row] / scale
            wave.plot(t, y, color=color, lw=.8)
            value = float(np.interp(peak_t[mode_index, row], t, y)); peak_y.append(value)
            wave.scatter([peak_t[mode_index, row]], [value], s=10, c=color, edgecolors='white', lw=.25, zorder=5)
        wave.plot(peak_t[mode_index], peak_y, color=color, lw=.7, ls=(0, (2, 2)), zorder=2)
        wave.set(xlim=(0, t[-1]), ylim=(-.55, 2.65), yticks=[2, 1, 0],
            yticklabels=wave_labels, xticks=[0, t[-1] / 2, t[-1]])
        wave.tick_params(labelsize=8, length=2, pad=2)
        wave.spines[['left', 'top', 'right']].set_visible(False); wave.tick_params(axis='y', length=0)
        text(x + w / 2, 178.2, mode, color=color, fontsize=9, ha='center', va='bottom')
    text(right_center, 183, 'SEEG readout', ha='center', va='bottom', weight='bold', fontsize=10.5)
    text(right_center, 146, 'Time (ms)', ha='center', va='bottom', fontsize=9)

    fig.canvas.draw()
    for label in sampling_labels:
        box = label.get_window_extent(fig.canvas.get_renderer())
        assert sample.bbox.contains(box.x0, box.y0) and sample.bbox.contains(box.x1, box.y1)
    delta = sample.transData.transform(along) - sample.transData.transform([0, 0])
    np.testing.assert_allclose(np.rad2deg(np.arctan2(delta[1], delta[0])), theta, atol=1e-8)
    assert np.all(np.diff(peak_t[0]) < 0)
    assert np.all(np.diff(peak_t[1]) >= 0) and np.any(np.diff(peak_t[1]) > 0)
    contact_px = spatial.transData.transform(overview_contacts)
    clearances = []
    for line in connectors:
        start, end = line.get_path().transformed(line.get_transform()).vertices[[0, -1]]
        step = end - start
        fractions = np.clip((contact_px - start) @ step / (step @ step), 0, 1)
        distance = np.linalg.norm(contact_px - (start + fractions[:, None] * step), axis=1).min()
        clearances.append(float(distance / spatial.bbox.width * 20))
    assert min(clearances) > r95
    return dict(status='CANDIDATE_PENDING_AUTHOR_VISUAL_REVIEW',
        original_left_circuit_restored=True, overview_neuron_samples_preserved=True,
        left_circuit_artwork_scale=left_scale,
        left_circuit_axes_mm=new_left_box.tolist(),
        left_circuit_original_axes_mm=old_left_box.tolist(),
        sheet_axes_mm=sheet_box, sheet_scale_from_v6=sheet_scale,
        electrode_shafts_are_straight=True, shaft_collinearity_residual_mm=straightness,
        rigid_geometry_is_shared_with_E_and_sampling_zoom=True,
        electrode_display=meta['electrode_display'],
        original_left_title_and_internal_labels_preserved=True,
        combined_third_component=True, formulas_and_lower_explanations_removed=True,
        zoom_center_mm=center.tolist(), zoom_field_of_view_mm=fov.tolist(),
        left_zoom_connector_minimum_contact_distance_mm=min(clearances),
        left_zoom_connectors_clear_of_electrode_footprints=True,
        actual_workpoint_kernel=kernel, kernel_centres_E_coordinates_mm=np.array(ellipse_centres).tolist(),
        layout_refinement=meta['layout_refinement'],
        local_population_coordinates_mm=populations,
        local_population_counts=[{key: len(value) for key, value in pop.items()} for pop in populations],
        all_local_E_markers_hollow=True, all_ellipses_contain_fewer_I_than_E=True,
        local_synaptic_arrowheads_removed=True, ellipse_rows=[2, 3],
        labeled_sampling_contacts=sampling_names,
        displayed_sampling_labels=display_names,
        display_label_to_source_contact=dict(zip(display_names, sampling_names)),
        exactly_three_consecutive_contacts=True,
        sampling_weights_shown_only_in_right_zoom=True,
        sampling_weight_display=footprint,
        sampling_weight_sigma_mm=sigma,
        gaussian_95_percent_mass_radius_mm=float(r95),
        central_sampling_region_removed=True,
        sampling_footprint_boundary_rings_removed=True,
        sampling_contact_coordinates_overview_mm=(sampling_contacts - 10.).tolist(),
        local_sampling_labels_match_both_readouts=True,
        tiny_readout_arrow_removed=True, local_circuit_count=len(ellipse_centres),
        sampling_zoom_center_overview_mm=sampling_center.tolist(),
        sampling_zoom_field_of_view_mm=sampling_fov.tolist(),
        sampling_frame_is_a_coordinate_exact_enlargement=True,
        two_distinct_zoom_roles=True, mode_readouts_below_sampling=True,
        physical_xy_equal_scale=True, kernel_arrows='Removed to distinguish neuronal symbols from arrows; ellipse orientation conveys anisotropy.',
        readout=meta['readout'], mode_showcases=meta['mode_showcases'],
        readout_peak_times_relative_ms=peak_t.tolist(),
        readout_peak_order='MTA strictly descending; MTB ascending with an ICL3/ICL4 tie at the original 2-ms sampling resolution.',
        data_lineage='Overview and sampling zoom share frozen illustrative neurons and E-panel rigid rod display coordinates. SEEG readout means virtual-contact SNN activity; both bursts are exact 120-ms waveform crops from F events 10 and 8, not measured voltage or graph recovery.')


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
