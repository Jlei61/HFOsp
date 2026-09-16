"""Readable labels and the full applied six-scalar values for scientific figures."""
from scripts import run_topic4_three_observable_bo as run
rt=run.rt;OUT=run.OUT

def point_label(cid):
    from scripts import analyze_topic4_three_observable_bo as a
    return a.point_label(cid)

def plot_label(cid):
    label=point_label(cid)
    if ' / ' not in label:return label
    if cid.startswith('g2_b'):
        return f"联合{int(cid.split('_')[1][1:])}-{int(cid.split('_')[2][1:])}"
    return '联合初始'+cid.split('_')[-1]

def parameter_table(fig,cids,*,height=.14):
    """Keep plot labels short while showing the actual six scalars on the figure."""
    rows=[]
    for cid in cids:
        x=run.vector(rt.read(OUT/'candidates'/f'{cid}.json'))
        rows.append([plot_label(cid),f'({x[0]:.3f}, {x[1]:.3f})',f'({x[2]:.3f}, {x[3]:.3f})',f'{x[4]:.3f}',f'{x[5]:+.2f}'])
    ax=fig.add_axes([.06,.012,.88,height]);ax.axis('off')
    tab=ax.table(cellText=rows,colLabels=['条件','左核(X,Y) mm','右核(X,Y) mm','核向外EE倍率','EE轴偏移 °'],cellLoc='center',loc='center',bbox=[0,0,1,1])
    tab.auto_set_font_size(False);tab.set_fontsize(8)
    for (row,col),cell in tab.get_celld().items():
        cell.set_edgecolor('#dddddd');cell.set_linewidth(.5)
        if row==0:cell.set_facecolor('#eeeeee')
