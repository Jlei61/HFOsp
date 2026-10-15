"""Numerical Fig4C renderer copied from the accepted producer; no live experiment access."""
import numpy as np
from matplotlib.lines import Line2D
COLORS={1:"#397DAA",2:"#8C4B20",3:"#8073AC",4:"#4F8A64"}
GROUPS=("rank_pattern","local_and_interrod_timing","participation")

def mean_std_trajectory(block, common):
    """Mean +/- sample SD across every repeat, at jointly observed epochs."""
    if block.shape[0] < 2:
        raise ValueError('Sample standard deviation requires at least two repeats.')
    shared = np.flatnonzero(np.isfinite(block[:, :common]).all(axis=0))
    values = block[:, shared]
    mean = np.mean(values, axis=0)
    sd = np.std(values, axis=0, ddof=1)
    return shared + 1, mean, sd

def draw(ax,case,component=0,*,show_title=True,paper_style=None,show_legend=False,
         summary_style='median_iqr'):
    """Draw the frozen trajectories directly at their final panel size.

    ``paper_style`` uses the complete figure's label/tick/legend point sizes;
    no title or figure-level progress caption is needed in an assembled panel.
    ``mean_std`` draws only means and sample SD bands on the common prefix;
    the legacy default preserves existing published-figure consumers.
    """
    if summary_style not in ('median_iqr', 'mean_std'):
        raise ValueError(f'Unknown summary style: {summary_style}')
    thin_lw,thin_ms,median_lw,median_ms=(.65,2.,1.45,3.2) if paper_style else (1.,3.8,2.3,5.5)
    s=case['summary'];common=s['common_epoch'];max_epoch=max(n for counts in s['prefix_by_restart'].values() for n in counts)
    mean_sd = summary_style == 'mean_std'
    summary_lw,summary_ms=((.85,1.5) if paper_style else (1.2,2.2)) if mean_sd else (median_lw,median_ms)
    lower_bounds=[]
    if mean_sd:
        max_epoch=common
    for k in sorted(case['arrays']):
        block=case['arrays'][k][:,:,component]
        if mean_sd:
            x,mean,sd=mean_std_trajectory(block,common)
            if len(x):
                lower_bounds.append(float(np.min(mean-sd)))
                band=ax.fill_between(x,mean-sd,mean+sd,color=COLORS[k],alpha=.14,linewidth=0,zorder=1)
                band.set_gid(f'core{k}_sample_sd')
                if len(x)==1:
                    ax.vlines(x,mean-sd,mean+sd,color=COLORS[k],alpha=.35,lw=.7,zorder=2)
                line,=ax.plot(x,mean,color=COLORS[k],lw=summary_lw,marker='o',ms=summary_ms,
                             markeredgewidth=0,zorder=3)
                line.set_gid(f'core{k}_mean')
            continue
        for row in block:
            finite=np.flatnonzero(np.isfinite(row))
            ax.plot(finite+1,row[finite],color=COLORS[k],alpha=.28,lw=thin_lw,marker='o',ms=thin_ms)
        # Compare all displayed arms only where every planned restart has finished.
        shared=np.isfinite(block[:,:common]).all(0);x=np.flatnonzero(shared)+1
        if len(x):
            q=np.quantile(block[:,x-1],[.25,.5,.75],axis=0)
            if len(x)>1:ax.fill_between(x,q[0],q[2],color=COLORS[k],alpha=.13,linewidth=0)
            else:ax.errorbar(x,q[1],yerr=np.stack([q[1]-q[0],q[2]-q[1]]),color=COLORS[k],fmt='none',capsize=5,lw=2)
            ax.plot(x,q[1],color=COLORS[k],lw=median_lw,marker='o',ms=median_ms)
    if max_epoch<=1:
        ax.set_xlim(.7,1.3);ax.set_xticks([1])
    else:
        ax.set_xlim(.85,max_epoch+.3)
        ax.set_xticks(np.unique(np.linspace(1,max_epoch,min(8,max_epoch),dtype=int)))
    ymin=min(lower_bounds,default=0.) if mean_sd else min((float(np.nanmin(a[:,:,component])) for a in case['arrays'].values() if np.isfinite(a[:,:,component]).any()),default=0.)
    if ymin>=0:ax.set_ylim(bottom=0)
    ax.set_xlabel('Epochs');ax.set_ylabel('Loss' if component==0 else 'Scaled error')
    ax.set_title(s['display_id'] if show_title else '',fontweight='bold',pad=11)
    ax.tick_params(direction='out',length=4,width=.8)
    if paper_style:
        ax.xaxis.label.set_fontsize(paper_style['label'])
        ax.yaxis.label.set_fontsize(paper_style['label'])
        ax.tick_params(labelsize=paper_style['tick'],length=2.4,width=.7,pad=2)
        ax.xaxis.labelpad=ax.yaxis.labelpad=3
        ax.spines[['top','right']].set_visible(False)
    if show_legend:
        handles=[Line2D([],[],color=COLORS[k],lw=summary_lw,marker='o',ms=summary_ms,
                       markeredgewidth=0 if mean_sd else plt.rcParams['lines.markeredgewidth'],
                       label=f'{k} core'+('s' if k>1 else '')) for k in sorted(case['arrays'])]
        ax.legend(handles=handles,loc='upper right',frameon=False,
                  fontsize=paper_style['legend'] if paper_style else 11,
                  handlelength=1.3,handletextpad=.45,borderaxespad=.35,labelspacing=.3)
