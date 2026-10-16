"""Figure 1 adapters; rank painters are copied from the accepted producer."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal,ndimage
from plotting.paper.fig1 import _plot_rank_heatmap,_plot_cluster_rank_fig4
from data_processing.raw_to_events import event_spectrogram
TA,TB='#B2182B','#2166AC'


def waveform(pre,detected):
    fs=pre['sfreq'];windows=detected['windows'][:3];pieces=[]
    for w in windows:
        center=(w.start+w.end)/2;start=max(0,round((center-.08)*fs));pieces.append(pre['band_V'][:12,start:start+round(.16*fs)])
    x=np.concatenate(pieces,axis=1)*1e6;gap=max(float(np.quantile(abs(x),.99))*3,1e-5);t=np.arange(x.shape[1])/fs
    fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
    for i,row in enumerate(x):ax.plot(t,row+i*gap,color='black',lw=.6)
    for bound in [.16,.32]:ax.axvline(bound,color='.7',ls='--',lw=.7)
    ax.set(xlim=(0,t[-1]),ylim=((len(x)-.2)*gap,-.8*gap),yticks=np.arange(len(x))*gap,yticklabels=pre['names'][:len(x)],xlabel='Time (s)')
    ax.text(1,1.02,'80–250 Hz',ha='right',transform=ax.transAxes);ax.text(0,1.02,'E1146',transform=ax.transAxes,weight='bold')
    return fig


def spectra(pre,features,detected):
    from plotting.paper.fig1_spectrum import draw_spectrum,draw_hfo
    # Keep one shaft and one contact order across all three new group events.
    counts=features['participating'].sum(0);ids=np.argsort(-counts,kind='stable')[:3]
    names=list(features['channel_names'])
    common=features['participating'][:,ids].all(1)
    selected=sorted([i for i,n in enumerate(names) if n.startswith('ICL') and common[i]],key=lambda i:int(names[i][3:]))
    if len(selected)<3:selected=list(np.flatnonzero(common))
    events=[]
    for ei in ids:
        t,f,S,centers=event_spectrogram(pre,features,int(ei));nf=len(f)
        mid=(features['windows_sec'][ei,1]-features['windows_sec'][ei,0])*500
        frequency=[np.sum(np.arange(nf)[:,None]*S[i]**3)/np.sum(S[i]**3) for i in selected]
        events.append(dict(specs=S[selected],times=(t-mid)/1000,
            centers=np.column_stack([(centers[selected]-mid)/1000,frequency])))
    # The separate HFO ensemble uses 180-ms Hann windows, 160-ms overlap and
    # pre-event normalization. This is distinct from group-event S^3 centroids.
    ci=int(np.argmax(detected['counts']));fs=pre['sfreq'];half=round(.3*fs)
    centers=np.rint(np.mean(detected['detections'][names[ci]],axis=1)*fs).astype(int)
    snippets=np.array([pre['broadband_V'][ci,c-half:c+half] for c in centers if half<=c<pre['broadband_V'].shape[1]-half])
    fig=plt.figure(figsize=(9,4.8));axes=[fig.add_axes([.08,.75-i*.3,.17,.18]) for i in range(3)]
    draw_hfo(fig,snippets,fs,axes)
    draw_spectrum(fig,events,[[3.10+i*1.65,.55,1.45,3.70] for i in range(3)],
        [8.2,.55,.08,3.70],4.45,channel_names=[names[i] for i in selected])
    return fig


def rank_heatmap(events,clustered=None):
    ranks=np.where(events['participating'],events['ranks'],np.nan);mask=events['participating'];names=list(events['channel_names']);n=len(names)
    # One fixed channel order is shared by the chronological and clustered panels.
    means=np.nanmean(ranks,axis=1);order=np.argsort(means);valid=np.flatnonzero(mask.sum(0)>=3)
    ids=valid if clustered is None else valid[np.argsort(clustered['labels'][valid],kind='stable')]
    fig=plt.figure(figsize=(11,4));heat=fig.add_axes([.09,.16,.65,.75]);bar=fig.add_axes([.755,.16,.012,.75]);profile=fig.add_axes([.85,.16,.14,.75])
    im=_plot_rank_heatmap(heat,ranks[order][:,ids],[names[i] for i in order],'',display_bools=mask[order][:,ids],ytick_fontsize=9,xtick_fontsize=10);im.set_clim(0,n-1)
    fig.colorbar(im,cax=bar,ticks=[0,n-1]);bar.set_title('Rank',fontsize=10)
    heat.set(xlim=(0,len(ids)),ylim=(0,n),xlabel='Population events (time-ordered)' if clustered is None else 'Population events (clustered)')
    if clustered is None:
        for row,ci in enumerate(order):
            vals=ranks[ci,valid];vals=vals[np.isfinite(vals)];hist=np.histogram(vals,bins=np.arange(n+1)-.5)[0]
            h=.8*hist/max(hist.max(),1);base=row+.08;color=plt.cm.viridis(row/max(n-1,1))
            profile.stairs(base+h,np.arange(n+1)-.5,baseline=base,fill=True,facecolor=color,edgecolor=np.asarray(color)*[.65,.65,.65,1],alpha=.85,lw=.85)
            peaks=np.flatnonzero(hist==hist.max());profile.scatter(peaks,np.full(len(peaks),base+.8),s=5,c='.2')
        profile.set(ylim=(0,n),yticks=[],xlabel='Rank',xticks=[0,(n-1)//2,n-1],xlim=(0,n-1))
    else:
        labels=clustered['labels'][valid];boundary=int(np.sum(labels==0));heat.axvline(boundary,color='white',lw=1.5)
        for k,(lo,hi) in enumerate([(0,boundary),(boundary,len(ids))]):heat.text((lo+hi)/2,n*1.035,f"T{'AB'[k]} (n={hi-lo})",color=[TA,TB][k],ha='center',weight='bold')
        _plot_cluster_rank_fig4(profile,ranks,mask,valid,labels,order,names,'',show_ylabels=False,show_legend=False,invert_yaxis=False,line_colors=[TA,TB],label_names=['TA','TB'],marker_size=2,label_fontsize=10,xtick_fontsize=10)
        profile.set_xlim(0,n-1);profile.set_xticks([0,(n-1)//2,n-1])
    # No day/night strip is inferred from the demo's segment-relative clock.
    return fig


def statistics(stats,panel='D'):
    fig,ax=plt.subplots(figsize=(4.6,4),layout='constrained');color='#B07A74'
    if panel=='D':
        o=stats['overall'];values=[o['mi_mean'],o['permuted_mean_median']]
        ax.scatter([0,1],values,c=[color,'.6'],s=30,zorder=3,clip_on=False);ax.set(xticks=[0,1],xticklabels=['Data','Null'],ylabel='MI',xlim=(-.55,1.55),ylim=(0,max(.58,max(values)*1.1)))
        ax.text(.5,-.13,'Epilepsiae · E1146',transform=ax.transAxes,ha='center')
    else:
        x=stats['pairwise']['overall'];y=stats['pairwise']['within'];hi=max(.85,x+.05,y+.05)
        ax.fill_between([0,hi],[0,hi],[0,0],color='#F0F0F0');ax.plot([0,hi],[0,hi],'--',lw=.9,color='.55');ax.scatter(x,y,c=color,s=35,edgecolors='white')
        ax.set(xlim=(0,hi),ylim=(0,hi),xlabel='Overall MI',ylabel='Within-template MI',aspect='equal')
        inset=ax.inset_axes([.58,.13,.35,.34]);inset.plot([0,1],[x,y],'o-',c=color,ms=3,lw=.7);inset.bar([0,1],[x,y],color='#B8C9CF',width=.65,alpha=.4);inset.set(xticks=[0,1],xticklabels=['Single','Multi'],ylim=(0,hi),ylabel='MI');inset.tick_params(labelsize=8)
    # The demo has one patient: do not invent a cohort violin, interval or significance test.
    return fig
