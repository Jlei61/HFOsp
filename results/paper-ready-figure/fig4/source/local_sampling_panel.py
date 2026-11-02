"""Adapted local sampling artwork from the accepted frozen producer; no waveforms."""
from pathlib import Path
import json
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Ellipse,Rectangle
HERE=Path(__file__).parent
RED,BLUE,GREEN='#D94748','#3985B8','#257653'
CALLOUT='#666666'


def draw_sampling(sample):
    meta=json.loads((HERE/'mechanism_source.json').read_text())
    with np.load(HERE/'spatial_reference.npz') as z:e,i=z['posE']+10.,z['posI']+10.;names=z['names'].astype(str)
    with np.load(HERE/'rigid_contact_geometry.npz') as z:
        np.testing.assert_array_equal(names,z['names']);overview_contacts=z['rigid_xy_mm']
    sigma=meta['readout']['sigma_mm']
    selected = [list(names).index(f'ICL{n}') for n in range(11, 0, -1)]
    contacts = overview_contacts[selected] + 10.
    sampling_names = meta['sampling_zoom']['labeled_contacts']
    display_names = sampling_names  # use true names shared with the rest of this figure
    sampling_contacts = overview_contacts[[list(names).index(n) for n in sampling_names]] + 10.
    contact = sampling_contacts[[0, -1]].mean(axis=0)
    kernel = meta['combined_physical_kernel']
    theta = kernel['theta_deg']; angle = np.deg2rad(theta)
    along = np.array([np.cos(angle), np.sin(angle)])
    # sample is supplied by the new B panel.
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
        collection=sample.scatter(*shown.T, marker=marker, fc='white', ec=color,
                       s=(6 if marker == '^' else 5) * symbol_scale ** 2,
                       alpha=.55, lw=.6, zorder=1)
        collection.set_gid('sampling_background_'+('E' if marker=='^' else 'I'))
    sample.plot(*contacts[[0, -1]].T, color='#4c4c4c', lw=1., zorder=4,
                solid_capstyle='round')
    inside = np.all((contacts > [xlim[0], ylim[0]]) & (contacts < [xlim[1], ylim[1]]), axis=1)
    assert inside.sum() == 3
    assert set(names[selected][inside]) == set(sampling_names)
    sample.scatter(*sampling_contacts.T, s=18, fc='white', ec='#333333', lw=1., zorder=7)
    sampling_labels = []
    for name, point in zip(display_names, sampling_contacts):
        sampling_labels.append(sample.annotate(name, point, xytext=((-7,4) if name=='ICL3' else ((7,4) if name=='ICL5' else (0,4))),
            textcoords='offset points', ha='center', va='bottom', fontsize=7.3,
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
            collection=sample.scatter(*points.T, marker=marker, fc='white', ec=color,
                           s=size * .47 * symbol_scale ** 2, lw=.7, zorder=6)
            collection.set_gid(f'sampling_group_{index}_{key}')
        assert len(groups['E']) > len(groups['I']) > 0
        populations.append({key: np.array(value).tolist() for key, value in groups.items()})
    bar_x = xlim[1] - .48
    sample.plot([bar_x - .25, bar_x + .25], [ylim[0] + .23] * 2, color='black', lw=1.1)
    sample.text(bar_x, ylim[0] + .05, '0.5 mm', ha='center', fontsize=6.5)
    sample.add_patch(Rectangle((xlim[0], ylim[0]), *sampling_fov, fill=False,
        ec=CALLOUT, lw=.65, ls=(0, (3, 1.7)), clip_on=False, zorder=9))
    sample.set_title('Local sampling',fontsize=9.2,fontweight='bold',pad=3,bbox=dict(fc='white',ec='none',alpha=.95,pad=.5))
    # Thin only schematic glyphs using their actual printed size. Preserve the
    # physical neuron coordinates, electrode sites, kernels and Gaussian weights.
    sample.figure.canvas.draw()
    glyphs=[c for c in sample.collections if (c.get_gid() or '').startswith('sampling_')]
    glyphs.sort(key=lambda c: ('background' in c.get_gid(), not c.get_gid().endswith('_I')))
    placed=[];radii_pt=[];retained={}
    for c in glyphs:
        xy=np.asarray(c.get_offsets())
        centers=sample.transData.transform(xy)/sample.figure.dpi*72.
        radius=np.sqrt(c.get_sizes()[0])*np.linalg.norm(c.get_paths()[0].vertices,axis=1).max()+c.get_linewidths()[0]/2
        keep=[]
        for idx,point in enumerate(centers):
            if placed and np.any(np.linalg.norm(np.asarray(placed)-point,axis=1)<np.asarray(radii_pt)+radius+.6):
                continue
            keep.append(idx);placed.append(point);radii_pt.append(radius)
        c.set_offsets(xy[keep]);retained[c.get_gid()]=len(keep)
        if 'group' in c.get_gid():
            _,_,index,key=c.get_gid().split('_')
            populations[int(index)][key]=xy[keep].tolist()
    return dict(center_overview_mm=(contact-10.).tolist(),field_of_view_mm=sampling_fov.tolist(),contacts=sampling_names,
        sigma_mm=float(sigma),kernel=kernel,neuron_groups=populations,source='accepted draw_fig4_a_spatial_readout.py local sampling body',waveforms=False,
        minimum_glyph_gap_pt=.6,retained_schematic_glyphs=retained)
