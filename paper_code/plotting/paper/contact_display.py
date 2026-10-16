"""E1146 contact display copied from src/snn_contact_display.py."""
import numpy as np
CONTACT_ORDER=tuple([f'SCL{i}' for i in range(9,5,-1)]+[f'ICL{i}' for i in range(11,0,-1)])
SHAFT_COLORS={'SCL':'#42a9b5','ICL':'#dc8722'}
YLIM=(14.5,-.5)

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
