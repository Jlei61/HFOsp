"""Paper producer functions copied locally; input/output adapters are separate.
Source: results/paper-ready-figure/fig4/source/fig4_parameter_response_assets.py
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
ERROR_LABELS=('Mean rank','Order','Participation')


def error_curves(plate,ax,x,values,xlabel,xticks,references):
    """Use one numeric Error scale for all three contact observations."""
    plate.axis_type(ax)
    for key,label,marker,ys in zip(plate.base.KEYS,ERROR_LABELS,['o','s','^'],values):
        line,=ax.plot(x,ys,color=plate.ERROR_COLORS[key],marker=marker,lw=1,ms=2.5,label=label)
        np.testing.assert_array_equal(line.get_xdata(),x)
        np.testing.assert_array_equal(line.get_ydata(),ys)
    for label,value in references:
        ax.axvline(value,color='#777777',ls='--' if label=='E' else ':',lw=.8,zorder=1)
        ax.text(value,.03,label,transform=ax.get_xaxis_transform(),ha='center',va='bottom',
                fontsize=plate.FONT['legend'],weight='bold',color='#444444',
                bbox=dict(facecolor='white',edgecolor='none',pad=.5),zorder=4)
    ax.set(ylim=(0,.42),yticks=[0,.2,.4],xticks=xticks,xlabel=xlabel,ylabel='Error')
    ax.tick_params(axis='y',left=True,labelleft=True,right=False,labelright=False,pad=1)
    ax.spines['right'].set_visible(False)
    ax.xaxis.label.set_fontsize(plate.MIDDLE_LABEL_PT)
    ax.yaxis.label.set_fontsize(plate.MIDDLE_LABEL_PT)
    ax.xaxis.labelpad=ax.yaxis.labelpad=2
    ax.legend(loc='upper right',fontsize=plate.FONT['legend'],frameon=True,framealpha=1,
              facecolor='white',edgecolor='none',borderpad=.2,labelspacing=.15,
              handlelength=1.1,handletextpad=.35,borderaxespad=.15)
