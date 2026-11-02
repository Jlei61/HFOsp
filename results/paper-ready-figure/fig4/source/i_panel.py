"""Actual three training-loss components and their mean J on one common axis."""
from pathlib import Path
import csv
import numpy as np

SOURCE = Path(__file__).resolve().parent
GROUPS = ('rank_pattern', 'local_and_interrod_timing', 'participation')
COLORS = ('#3D5C6F', '#E47159', '#6B7546')


def render_i(fig, rect):
    rows = list(csv.DictReader((SOURCE/'i_angle_response.csv').open()))
    x = np.array([float(r['parameter_value']) for r in rows])
    components = np.array([[float(r['J_component_'+g]) for g in GROUPS] for r in rows])
    component_sd = np.sqrt([[float(r['J_component_'+g+'_bootstrap_variance']) for g in GROUPS] for r in rows])
    total = np.array([float(r['optimizer_J']) for r in rows])
    total_sd = np.array([float(r['J_bootstrap_sd']) for r in rows])
    np.testing.assert_allclose(components.mean(1), total, rtol=0, atol=1e-12)
    assert len(x) == 13 and np.all(np.diff(x)>0)
    ax = fig.add_axes(rect); ax.set_gid('fig4_i_training_components_and_total')
    labels = ('Rank', 'Order', 'Participation')
    for j,(label,color,marker) in enumerate(zip(labels,COLORS,('o','s','^'))):
        ax.fill_between(x,components[:,j]-component_sd[:,j],components[:,j]+component_sd[:,j],
                        color=color,alpha=.16,lw=0,zorder=1)
        line, = ax.plot(x,components[:,j],color=color,lw=.75,marker=marker,ms=2.1,mew=.35,
                        label=label,zorder=2)
        line.set_gid('i_L_'+('r','o','p')[j])
    ax.fill_between(x,total-total_sd,total+total_sd,color='#202020',alpha=.14,lw=0,zorder=1)
    line, = ax.plot(x,total,color='#171717',lw=1.25,marker='o',ms=2.4,label='Total loss',zorder=3)
    line.set_gid('i_total_J')
    ax.axvline(x[np.argmin(total)],ymax=.80,color='#444444',ls=(0,(3,2)),lw=.8,zorder=0)
    ax.set(xlim=(-21.5,21.5),ylim=(0,8),xticks=[-20,0,20],yticks=[0,2,4,6,8])
    assert np.min(components-component_sd)>=0 and np.max(components+component_sd)<8
    ax.set_xlabel('E→E angle offset (°)',fontsize=10.5,labelpad=2)
    ax.set_ylabel('Loss',fontsize=10.5,labelpad=2)
    ax.tick_params(axis='both',labelsize=9.5,width=.65,length=2.5,pad=2)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper right',ncol=1,fontsize=8.3,frameon=False,handlelength=1.05,
              handletextpad=.35,labelspacing=.2,borderaxespad=.3)
    return ax
