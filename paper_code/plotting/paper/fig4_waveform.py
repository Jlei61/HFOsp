"""Paper Fig4F painter; returns the figure instead of writing paper artifacts."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import butter,sosfiltfilt
from plotting.paper.fig4_matching import DISPLAY,SHAFT,RED,BLUE
MM=1/25.4

def waveform_panel(run,a,ids,order):
    # Legacy pair rule adapted to current primary events; no patient-distance choice.
    events = run['events']; labels = a['event_mode']
    def center(i): return float(np.mean(events[i]['qualifying_interval_ms']))
    choices = []
    for i in ids[labels[ids]==1]:
        for j in ids[labels[ids]==0]:
            n1,n2 = [int(np.isfinite(a['recruitment_ms'][k]).sum()) for k in [i,j]]
            gap = abs(center(i)-center(j))
            choices.append(dict(ta=int(i),tb=int(j),gap=gap,n1=n1,n2=n2))
    pool = [r for r in choices if 250 <= r['gap'] <= 1200] or choices
    pair = min(pool,key=lambda r:(-min(r['n1'],r['n2']),-(r['n1']+r['n2']),abs(r['gap']-550),r['ta'],r['tb']))
    i,j = pair['ta'],pair['tb']
    width = max(760,pair['gap']+280)
    dt = float(a['contact_envelope_dt_ms']); duration = run['actual_duration_ms']
    center_pair = .5*(center(i)+center(j))
    start = max(1500,min(duration-width,center_pair-width/2)); stop = start+width
    t = np.arange(a['contact_envelope'].shape[1])*dt
    filt = sosfiltfilt(butter(4,[30,80],btype='bandpass',fs=1000/dt,output='sos'),
                      a['contact_envelope'],axis=1)
    select = (t>=start)&(t<=stop); trace = filt[order][:,select]
    scale = max(float(np.quantile(abs(trace),.99)),1e-9)
    fig = plt.figure(figsize=(180*MM,67*MM))
    ax = fig.add_axes([.094,.17,.895,.73])
    plotted = (14-np.arange(15))[:,None]*1.25+.72*trace/scale
    for row,name in enumerate(DISPLAY):
        ax.plot(t[select]-start,plotted[row],
                color=SHAFT[name[:3]],lw=.7)
    spans = []
    for k,c in [(i,RED),(j,BLUE)]:
        valid = a['recruitment_ms'][k]; valid = valid[np.isfinite(valid)]
        lo,hi = float(valid.min()-12),float(valid.max()+12)
        ax.axvspan(lo-start,hi-start,color=c,alpha=.14,lw=0,zorder=-2)
        spans.append(dict(event=k,mode='TA' if labels[k]==1 else 'TB',
                          recruitment_onset_span_ms=[lo+12,hi-12],shaded_span_ms=[lo,hi]))
    ax.axhline((14-3.5)*1.25,color='#C8C8C8',ls=':',lw=.55,zorder=-1)
    ax.set(xlim=(0,width),ylim=(float(plotted.min()-.3),float(plotted.max()+.3)),yticks=(14-np.arange(15))*1.25,yticklabels=DISPLAY,
           xlabel='Time (ms)',ylabel='30–80 Hz activity',xticks=np.arange(0,width+1,100))
    ax.tick_params(length=2.5,pad=2)
    for text in ax.get_yticklabels(): text.set_color(SHAFT[text.get_text()[:3]])
    ax.legend(handles=[Patch(facecolor=RED,alpha=.22,label='MTA'),Patch(facecolor=BLUE,alpha=.22,label='MTB')],
              ncol=2,loc='lower right',bbox_to_anchor=(1,1.01),borderaxespad=0,
              handlelength=1.5,columnspacing=1.4)
    return fig
