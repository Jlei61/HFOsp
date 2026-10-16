"""Paper producer functions copied locally; input/output adapters are separate.
Source: scripts/paper_figures/build_fig4_geometry_patient_comparison.py
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
MM=1/25.4
RED,BLUE='#C63D3A','#287FA1'
DISPLAY=['SCL9','SCL8','SCL7','SCL6']+[f'ICL{i}' for i in range(11,0,-1)]
SHAFT={'SCL':'#00A1B5','ICL':'#E78929'}
def save(fig, letter): pass


def rank_panel(order, model, patient):
    fig = plt.figure(figsize=(87*MM,98*MM))
    ax = fig.add_axes([.195,.225,.765,.745])
    for mode,color in [(1,RED),(0,BLUE)]:
        for values,ls,marker in [(model[mode],'-','o'),(patient[mode],'--',None)]:
            for sl in [slice(0,4),slice(4,15)]:
                pos = np.arange(15)[sl]
                ax.plot(values[order][sl]*14,pos,ls=ls,marker=marker,color=color,
                        markersize=2.8,lw=1.15)
    ax.axhline(3.5,color='#BABABA',lw=.6)
    ax.set(ylim=(14.65,-.65),xlim=(-.3,14.3),xticks=[0,4,8,12,14],
           yticks=np.arange(15),yticklabels=DISPLAY,xlabel='Mean rank')
    ax.grid(axis='x',color='#E1E3E5',lw=.6); ax.set_axisbelow(True)
    ax.tick_params(length=2.5,pad=3)
    for t in ax.get_yticklabels(): t.set_color(SHAFT[t.get_text()[:3]])
    handles = [Line2D([],[],color=c,ls=ls,marker=mk,lw=1.15,ms=3,label=label)
               for c,ls,mk,label in [(RED,'-','o','Model TA'),(RED,'--',None,'Patient TA'),
                                      (BLUE,'-','o','Model TB'),(BLUE,'--',None,'Patient TB')]]
    fig.legend(handles=handles,ncol=2,loc='lower center',bbox_to_anchor=(.57,.005),
               columnspacing=1.3,handlelength=2.1,handletextpad=.5)
    return fig

def matrix_panel(cf,tests):
    fig = plt.figure(figsize=(92*MM,69*MM))
    ax = fig.add_axes([15/92,12/69,49/92,49/69])
    matrix = cf['matrix'][np.ix_([1,0],[1,0])]
    im = ax.imshow(matrix,cmap='RdBu_r',vmin=-1,vmax=1,aspect='equal')
    for i in range(2):
        for j in range(2):
            color = 'white' if abs(matrix[i,j])>.45 else '#222222'
            ax.text(j,i-(.09 if i==j else 0),f'{matrix[i,j]:+.2f}',ha='center',va='center',
                    color=color,fontsize=12)
            if i==j:
                key = ['MTA_vs_TA','MTB_vs_TB'][i]
                ax.text(j,i+.24,tests[key]['stars'],ha='center',va='center',color=color,fontsize=9)
    ax.set(xticks=[0,1],xticklabels=['TA','TB'],yticks=[0,1],yticklabels=['MTA','MTB'])
    for t,c in zip(ax.get_xticklabels(),[RED,BLUE]): t.set_color(c)
    for t,c in zip(ax.get_yticklabels(),[RED,BLUE]): t.set_color(c)
    ax.tick_params(length=2.5,pad=4)
    cb = fig.colorbar(im,cax=fig.add_axes([68/92,12/69,2.5/92,49/69]),ticks=[-1,-.5,0,.5,1])
    cb.outline.set_visible(False); cb.set_label('Spearman ρ',labelpad=3)
    cb.ax.tick_params(length=2,pad=3)
    return fig
