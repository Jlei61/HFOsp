"""Original Fig5F patient-level statistical painter."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
FONT=SimpleNamespace(axis_label=18.,tick_label=16.,annotation=15.,significance=18.)

def swarm(values, y_separation=.043, x_separation=.065):
    """Separate adjacent patient markers deterministically; x has no data meaning."""
    offsets = np.zeros(len(values))
    placed = []
    candidates = [0.] + [sign * x_separation * k for k in range(1, 6) for sign in (-1, 1)]
    for i in np.argsort(values, kind='stable'):
        for candidate in candidates:
            if all(abs(values[i] - values[j]) >= y_separation or abs(candidate - offsets[j]) >= x_separation - 1e-12 for j in placed):
                offsets[i] = candidate
                placed.append(i)
                break
        else:
            raise ValueError('Patient points cannot fit without overlap')
    return offsets

def _add_violin_box_points(
    ax: plt.Axes,
    values: np.ndarray,
    x: float,
    *,
    facecolor: str,
    edgecolor: str,
    rng: np.random.Generator,
    point_face: str,
    point_edge: str,
    jitter: np.ndarray | None = None,
) -> np.ndarray:
    # A one-subject pre-existing stratum is a descriptive case.  Keep the
    # accepted box/point grammar but do not ask KDE-based violinplot to fit a
    # singular one-point distribution.
    if len(values) >= 2 and not np.allclose(values, values[0]):
        parts = ax.violinplot(
            [values],
            positions=[x],
            widths=0.58,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        body = parts["bodies"][0]
        body.set_facecolor(facecolor)
        body.set_edgecolor("none")
        body.set_alpha(0.72)

    ax.boxplot(
        [values],
        positions=[x],
        widths=0.34,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
        boxprops={"facecolor": facecolor, "edgecolor": edgecolor, "linewidth": 1.1, "alpha": 0.8},
        whiskerprops={"color": edgecolor, "linewidth": 1.0},
        capprops={"color": edgecolor, "linewidth": 1.0},
    )
    if jitter is None:
        jitter = rng.normal(0.0, 0.045, size=len(values))
    point_x = np.full(len(values), x) + jitter
    ax.scatter(
        point_x,
        values,
        s=25,
        facecolors=point_face,
        edgecolors=point_edge,
        linewidths=0.8,
        alpha=0.9,
        zorder=4,
    )
    return point_x

def draw_stat(fig,rect,stat,small=False):
    ax=fig.add_axes(rect);rows=stat['matches']
    values=np.array([v['best_abs_r'] for v in rows]);null=np.array([v['null_median'] for v in rows])
    xs=1+swarm(values,y_separation=.04,x_separation=.07);ys=2+swarm(null,y_separation=.04,x_separation=.07)
    for x,y,v,n in zip(xs,ys,values,null):ax.plot([x,y],[v,n],c='#b5b5b5',alpha=.75,lw=.9,zorder=1)
    for pos,v,x,fill in [(1,values,xs,'#ead8d0'),(2,null,ys,'#e0e0e0')]:
        _add_violin_box_points(ax,v,pos,facecolor=fill,edgecolor='#878787',rng=np.random.default_rng(20261009),
                              point_face='none',point_edge='none',jitter=x-pos)
    ax.scatter(xs,values,c='#b45b40',s=24 if small else 60,edgecolor='#333333',lw=.6,zorder=6,clip_on=False)
    ax.scatter(ys,null,c='#a2a2a2',s=24 if small else 60,edgecolor='#333333',lw=.6,zorder=6,clip_on=False)
    ax.plot([1,1,2,2],[1.015,1.035,1.035,1.015],c='black',lw=1.2)
    ax.text(1.5,1.055,f'$p = {stat["p"]:.4f}$',ha='center',fontsize=FONT.annotation if small else FONT.significance)
    ax.set(xlim=(.55,2.45),ylim=(0,1.16),yticks=[0,.25,.5,.75,1])
    ax.set_ylabel('Best field similarity |r|',fontsize=FONT.axis_label)
    ax.set_xticks([1,2],['Model–\nseizure','Full-contact\nnull'] if small else ['Model–seizure','Full-contact null'])
    ax.set_xlabel(f'$n = {len(rows)}$ patients',fontsize=FONT.axis_label,labelpad=7)
    ax.tick_params(labelsize=13 if small else FONT.tick_label);ax.spines[['top','right']].set_visible(False)
    return ax
