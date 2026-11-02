"""One minimum-spacing subset of frozen schematic neurons shared by B, D and E.

Selection is for illustration only. Native simulation geometry and spikes are
unchanged. Original coordinates and E/I identities are retained exactly.
"""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).parent
MIN_SPACING_MM = .52
EDGE_MARGIN_MM = .25
STYLE = {'E': dict(color='#D84840', marker='^', size=4.4),
         'I': dict(color='#347FB3', marker='o', size=3.7)}


def spaced_samples():
    with np.load(HERE/'overview_neuron_samples.npz') as z:
        source = {k: z[k].copy() for k in ('E', 'I')}
    original = np.concatenate([source['E'], source['I']])
    eligible = np.flatnonzero(np.all(np.abs(original) <= 10-EDGE_MARGIN_MM, axis=1))
    xy = original[eligible]
    # Farthest-point selection gives broad coverage without moving any neuron.
    first = int(np.argmin(np.sum(xy**2, axis=1)))
    selected = [first]
    nearest_sq = np.sum((xy-xy[first])**2, axis=1)
    while nearest_sq.max() >= MIN_SPACING_MM**2:
        idx = int(np.argmax(nearest_sq))
        selected.append(idx)
        nearest_sq = np.minimum(nearest_sq, np.sum((xy-xy[idx])**2, axis=1))
    indices = np.sort(eligible[selected])
    n_e = len(source['E'])
    kept = {'E': indices[indices < n_e], 'I': indices[indices >= n_e]-n_e}
    points = {k: source[k][kept[k]] for k in ('E', 'I')}
    return points, kept


def draw_neurons(ax, shift_mm=0.):
    points, indices = spaced_samples()
    for kind in ('E', 'I'):
        style = STYLE[kind]
        artist = ax.scatter(*(points[kind]+shift_mm).T, s=style['size'],
                            marker=style['marker'], facecolors='none',
                            edgecolors=style['color'], lw=.32, alpha=.80,
                            rasterized=True, zorder=2)
        artist.set_gid(f'shared_overview_{kind}')
    return dict(E_points=len(points['E']), I_points=len(points['I']),
                min_center_spacing_mm=MIN_SPACING_MM,
                edge_margin_mm=EDGE_MARGIN_MM,
                original_indices={k: v.tolist() for k, v in indices.items()},
                coordinates='Original frozen overview coordinates; display subset only',
                shared_by=['B', 'D', 'E'])


if __name__ == '__main__':
    points, indices = spaced_samples()
    np.savez_compressed(HERE/'overview_neuron_samples_spaced.npz', **points,
                        E_indices=indices['E'], I_indices=indices['I'])
    (HERE/'overview_neuron_spacing.json').write_text(json.dumps(dict(
        min_spacing_mm=MIN_SPACING_MM, edge_margin_mm=EDGE_MARGIN_MM, counts={k: len(v) for k, v in points.items()},
        selection='Deterministic farthest-point subset; no coordinate displacement',
        source='overview_neuron_samples.npz', shared_panels=['B', 'D', 'E'],
        native_simulation_unchanged=True), indent=2)+'\n')
