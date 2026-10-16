"""Paper producer functions copied locally; input/output adapters are separate.
Source: results/paper-ready-figure/fig4/source/build_fig4_template_recovery_by_mode.py
"""
from __future__ import annotations
from typing import Any, List, Mapping, Sequence
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import Normalize, PowerNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
COLORS=['#C63D3A','#287FA1']
LIGHT=['#EAB7B5','#B0D3E1']


def draw_modes(ax,summary,*,legend_loc=None):
    values=np.array([r['matched_rhos'] for r in summary['subjects']],float)
    rng=np.random.default_rng(20260928)
    jitter=rng.uniform(-.12,.12,len(values))
    metadata=[]
    ax.axhline(0,color='.72',lw=.6,ls='--',zorder=0)
    for mode in [0,1]:
        finite=np.isfinite(values[:,mode]);y=values[finite,mode]
        ax.scatter(mode-.11+jitter[finite],y,s=13,c=LIGHT[mode],edgecolors=COLORS[mode],
                   linewidths=.4,alpha=.95,zorder=3)
        q25,median,q75=np.quantile(y,[.25,.5,.75])
        np.testing.assert_allclose([q25,median,q75],
                                  [summary['statistics'][mode]['q25_q75'][0],summary['statistics'][mode]['median'],
                                   summary['statistics'][mode]['q25_q75'][1]])
        x=mode+.24
        ax.plot([x,x],[q25,q75],color=COLORS[mode],lw=2.6,solid_capstyle='butt')
        ax.plot([x-.10,x+.10],[median,median],color=COLORS[mode],lw=1.8)
        for yy in [q25,q75]:ax.plot([x-.065,x+.065],[yy,yy],color=COLORS[mode],lw=.8)
        metadata.append(dict(mode=summary['mode_order'][mode],n=int(finite.sum()),
                             subjects=[r['subject'] for r,ok in zip(summary['subjects'],finite) if ok],
                             points=y.tolist(),q25_q50_q75=[q25,median,q75]))
    ax.set_xlim(-.5,1.5);ax.set_ylim(-1,1)
    ax.set_xticks([0,1],['TA–MTA','TB–MTB'])
    ax.tick_params(axis='x',length=0,pad=5,labelsize=8)
    ax.set_yticks([-1,0,1]);ax.set_ylabel('Template similarity (ρ)',fontsize=10,labelpad=3)
    for t,color in zip(ax.get_xticklabels(),COLORS):t.set_color(color);t.set_fontweight('bold')
    if legend_loc is not None:
        # Keep the legend above the largest observation within the same axis box.
        ax.set_ylim(-1,1.55)
        ax.legend([Line2D([],[],marker='o',ls='',color='#C8C8C8',
                          markeredgecolor='#666666',markeredgewidth=.4,ms=3.5),
                   Line2D([],[],color='#666666',lw=1.6)],
                  ['Subject','Median'],loc=legend_loc,ncol=1,fontsize=7,
                  handlelength=.9,handletextpad=.3,columnspacing=.7,
                  borderpad=.2,borderaxespad=.35,frameon=True,
                  facecolor='white',edgecolor='none',framealpha=1)
    return dict(groups=metadata,legend_present=legend_loc is not None,legend_location=legend_loc,
                legend_columns=1 if legend_loc is not None else 0,
                summary_interval='interquartile range, not confidence interval',
                metric='two separate signed Spearman correlations; no minimum or cross-match penalty')
