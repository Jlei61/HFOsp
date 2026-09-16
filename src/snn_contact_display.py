"""Fixed E1146 contact rows for patient/model SNN comparisons (display only)."""
import numpy as np

CONTACT_ORDER = tuple([f'SCL{i}' for i in range(9, 5, -1)] + [f'ICL{i}' for i in range(11, 0, -1)])
SHAFT_COLORS = {'SCL': '#42a9b5', 'ICL': '#dc8722'}
YLIM = (14.5, -0.5)


def contact_indices(names):
    names = list(map(str, names))
    if len(names) != len(set(names)) or set(names) != set(CONTACT_ORDER):
        raise ValueError('E1146 comparison requires the same 15 named contacts')
    return [names.index(name) for name in CONTACT_ORDER]


def contact_axis(ax):
    ax.set(ylim=YLIM, yticks=np.arange(15), yticklabels=CONTACT_ORDER)
    ax.tick_params(axis='y', labelsize=7)
    for tick, name in zip(ax.get_yticklabels(), CONTACT_ORDER):
        tick.set_color(SHAFT_COLORS[name[:3]])
    ax.axhline(3.5, color='#dddddd', lw=1.0)
    ax.set_facecolor('#b6b6b6')


def centroid_lines(ax, times, ys=None, color='#1f77b4'):
    """Join within a shaft only; NaN rows break the line across absent contacts."""
    times = np.asarray(times, float)
    ys = np.arange(15) if ys is None else np.asarray(ys, float)
    for block in [slice(0, 4), slice(4, 15)]:
        ax.plot(times[block], ys[block], '-o', color=color, lw=.8, ms=3,
                markerfacecolor='#ffb000')


def patient_readout(ax, event, names, canonical, xlim, color):
    """Preserve frozen STFT cells, normalization and frequency-centroid positions."""
    order = contact_indices(names)
    n_freq = len(event['spec_freq_hz'])
    edges = canonical.full_extent_edges(np.asarray(event['spec_t_ms'], float),
        float(event['tile_lo_ms']), float(event['tile_hi_ms']))
    cmap = __import__('matplotlib').colormaps[canonical.FIG1A_CMAP].copy()
    cmap.set_bad('#777777')
    blocks = [event['spec'][c] if event['usable'][c]
              else np.full_like(event['spec'][c], np.nan) for c in order]
    shown = np.concatenate(blocks, axis=0)
    ax.pcolormesh(edges, np.arange(15*n_freq+1)/n_freq-.5, shown,
                  cmap=cmap, vmin=0, vmax=1, shading='flat', rasterized=True)
    times = np.array([event['centroid_ms'][c] if event['usable'][c] else np.nan for c in order])
    ys = np.array([i-.5+(event['centroid_freq_index'][c]+.5)/n_freq
                   if event['usable'][c] else np.nan for i,c in enumerate(order)])
    centroid_lines(ax, times, ys, color)
    for i, c in enumerate(order):
        if not event['usable'][c]:
            ax.text((xlim[0]+xlim[1])/2, i, '未参与 / 无有效质心', ha='center', va='center', fontsize=6, color='white')
    contact_axis(ax)
    ax.set_xlim(*xlim)
    return dict(contact_order=list(CONTACT_ORDER), ylim=list(YLIM),
                usable=[bool(event['usable'][c]) for c in order],
                centroid_ms=[float(t) if np.isfinite(t) else None for t in times])
