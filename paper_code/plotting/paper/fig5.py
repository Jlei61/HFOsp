"""Copied numerical painters behind Fig5 reference_layout_20261011.
Source: scripts/paper_figures/build_fig5_fig4f_historical_zm.py.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter,sosfiltfilt
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Circle
from matplotlib.colors import Normalize,PowerNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import proj3d
from types import SimpleNamespace
from plotting.paper.fig4_matching import DISPLAY
layout=SimpleNamespace(previous=SimpleNamespace(f=SimpleNamespace(ORDER=DISPLAY)),Z_COLOR='#2878b5',M_COLOR='#d77d27',PowerNorm=PowerNorm)
HIDDEN_STATE_NUMBERS=set()
# Runtime settings recorded with the six-panel reference_layout_20261011.
MAIN_BAND_HZ=(80.,1000.)
COMPARISON_BAND_HZ=(30.,1000.)
GAIN_PERCENTILE=99.5
DISPLAY_GAIN=.65
MAP_STATE_NUMBERS=[2,3,4,5]
MAP_WINDOW_MS=20
TRAJECTORY_WINDOW=None
WINDOW=None
ETA=None
RECRUIT=None

def marks(ax,samples,onset):
    ax.axvspan(onset,WINDOW[1],fc='#ba263c',alpha=.08,lw=0,zorder=0)
    for s in samples:
        if s['number'] in HIDDEN_STATE_NUMBERS:continue
        ax.axvline(s['time_s'],color=s['color'],ls='--' if s['number']==1 else ':',lw=1.2,alpha=.85,zorder=1)
    ax.set_xlim(WINDOW);ax.xaxis.set_major_locator(MaxNLocator(5))

def draw_slow(za,ma,a,samples,onset):
    t=a['slow_time_ms']/1000;take=(t>=WINDOW[0])&(t<=WINDOW[1])
    z=a['Z'][:,[0,5]];m=ETA*a['M'][:,[0,1]]
    for j,(label,style) in enumerate([('All E','-'),('Core A','--')]):
        za.plot(t[take],z[take,j],c=layout.Z_COLOR,ls=style,lw=2.4,label=label)
        ma.plot(t[take],m[take,j],c=layout.M_COLOR,ls=style,lw=2.4,label=label)
    for ax in (za,ma):
        marks(ax,samples,onset);ax.yaxis.set_major_locator(MaxNLocator(3))
    za.set_ylabel('$Z$');za.tick_params(axis='x',labelbottom=False)
    za.set_ylim(float(z[take].min())-.002,float(z[take].max())+.002)
    ma.set(ylabel=r'$\eta_M M$ (mV)',xlabel='Time (s)',ylim=(0,m[take].max()*1.12))
    from matplotlib.lines import Line2D
    legend=za.legend(handles=[Line2D([],[],color='#555',ls=s,lw=2.4,label=l) for l,s in [('All E','-'),('Core A','--')]],
        loc='lower right',bbox_to_anchor=(1,1.03),borderaxespad=0,frameon=True,ncol=1,fontsize=14,handlelength=2.6,
        facecolor='white',edgecolor='#888',framealpha=1,fancybox=False,borderpad=.35,labelspacing=.25)
    # Reserve only the headroom needed to keep both data curves below the legend.
    for _ in range(20):
        za.figure.canvas.draw()
        box=legend.get_window_extent().transformed(za.transData.inverted())
        under=(t>=box.x0)&(t<=box.x1)&take
        if not under.any() or z[under].max()<box.y0-.002:break
        low,high=za.get_ylim();za.set_ylim(low,high+.008)
    else:raise RuntimeError('B legend overlaps a data curve.')
    return dict(Z_columns=[0,5],M_columns=[0,1],eta=ETA,Z_limits=za.get_ylim(),effective_M_limits=ma.get_ylim(),
        variable_colors={'Z':layout.Z_COLOR,'effective_M':layout.M_COLOR},population_styles={'All E':'solid','Core A':'dashed'},
        legend=dict(location='above upper right',columns=1,frame=True,data_curve_overlap=False),
        axis_zoom='Tight Z limits; shared A/B time axis preserved.',
        entry_Z_all_E=float(np.interp(onset,t,z[:,0])),entry_Z_core_A=float(np.interp(onset,t,z[:,1])))

def draw_a(ax,a,samples,onset,filtered=False):
    names=list(a['contact_names']);order=layout.previous.f.ORDER;ids=[names.index(n) for n in order]
    t=a['lfp_native_time_ms']/1000
    raw=np.asarray(a['lfp_native'][:,ids],float)
    band=COMPARISON_BAND_HZ if filtered else MAIN_BAND_HZ
    is_filtered=band is not None
    signal=sosfiltfilt(butter(4,band,btype='bandpass',fs=10000,output='sos'),raw,axis=0) if is_filtered else raw
    scales=np.percentile(abs(signal[t<RECRUIT]),GAIN_PERCENTILE,axis=0)
    floor=.15*np.median(scales[scales>0]);gains=np.maximum(scales,floor)
    take=(t>=WINDOW[0])&(t<=WINDOW[1]);display_gain=DISPLAY_GAIN
    plotted=signal[take]/gains*display_gain+np.arange(14,-1,-1)
    for i in range(15):
        ax.plot(t[take],plotted[:,i],c='#27929d' if i<4 else '#c67b35',lw=.6,rasterized=True)
    ymin=min(-.65,float(plotted.min())-.12);ymax=max(15.,float(plotted.max())+.12)
    ax.axhline(10.5,c='#ccc',lw=.8)
    ax.set(yticks=np.arange(14,-1,-1),yticklabels=order,ylim=(ymin,ymax),ylabel='SEG readout')
    ax.tick_params(axis='y',labelsize=14);ax.tick_params(axis='x',labelbottom=False)
    marks(ax,samples,onset)
    for s in samples:
        if s['number'] in HIDDEN_STATE_NUMBERS:continue
        ax.annotate(str(s['number']),xy=(s['time_s'],1),xycoords=ax.get_xaxis_transform(),xytext=(0,7),
            textcoords='offset points',ha='center',va='bottom',fontsize=17,weight='bold',color=s['color'])
    return dict(band_hz=band,sampling_Hz=10000,dc_preserved=not is_filtered,
        filtered=is_filtered,filter_order=4 if is_filtered else None,zero_phase=is_filtered,baseline_subtraction=False,
        filter_scope='Full native trajectory before extracting the display window.',
        filter_implementation='scipy.signal.butter(4, band, bandpass, fs=10000, output=sos); sosfiltfilt' if is_filtered else None,
        signal='Distance-weighted abs(I_E)+abs(I_I) over E cells, observed before Z/M current modulation.',
        common_average_reference=False,contact_order=order,gain_per_contact=gains,
        normalization=f'P{GAIN_PERCENTILE:g} absolute signal before the measured entry episode;15% median floor;fixed {DISPLAY_GAIN:.12g} display gain.',
        gain_percentile=GAIN_PERCENTILE,display_gain=DISPLAY_GAIN,frequency_equalization=False,
        zoom_rectangles=False,band_title_shown=False,state_label_offset_points=[0,7],
        state_markers='State1 dashed; subsequent map states dotted.',
        visible_state_numbers=[s['number'] for s in samples if s['number'] not in HIDDEN_STATE_NUMBERS],ylabel='SEG readout',
        normalization_cutoff_s=RECRUIT,time_varying_gain=False,visible_time_s=WINDOW,samples_plotted=int(take.sum()))

def draw_native_maps(fig,spec,a,samples):
    selected=[s for s in samples if s['number'] in MAP_STATE_NUMBERS]
    assert len(selected)==4
    gs=spec.subgridspec(1,4,wspace=.22);axes=[];records=[]
    for i,s in enumerate(selected):
        ax=fig.add_subplot(gs[i]);lo=round(s['time_s']*1000)-MAP_WINDOW_MS//2
        count=a['field_1ms'][lo:lo+MAP_WINDOW_MS].sum(0)
        hz=count/a['cell_e_counts']/(MAP_WINDOW_MS/1000)
        assert count.sum()==a['spikes_1ms'][lo:lo+MAP_WINDOW_MS,0].sum()
        im=ax.imshow(hz.reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',
            norm=layout.PowerNorm(.6,0,500),interpolation='nearest')
        for j,xy in enumerate(a['centers_mm']):
            ax.add_patch(Circle(xy,float(a['core_radius_mm']),ec='#2dd4cd',fc='none',lw=1.7))
            ax.text(xy[0],xy[1]+2.1,'AB'[j],color='#10656a',fontsize=12,ha='center',weight='bold',
                bbox=dict(fc='white',ec='none',pad=.2,alpha=.8))
        ax.set(xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)')
        if i==0:ax.set_ylabel('y (mm)')
        else:ax.tick_params(labelleft=False)
        label=str(s['number'])+(f"  {s['label']}" if s['label'] else '')
        ax.set_title(f"{label}\n{s['time_s']:.3f} s",color=s['color'],fontsize=16,pad=10)
        if i==3:
            cb=fig.colorbar(im,cax=ax.inset_axes([1.10,0,.06,1]),ticks=[0,250,500])
            cb.set_label('E rate (Hz)',fontsize=17)
        axes.append(ax)
        records.append(dict(number=s['number'],time_s=s['time_s'],time_window_s=[lo/1000,(lo+MAP_WINDOW_MS)/1000],
            window_ms=MAP_WINDOW_MS,rate_Hz=hz,total_E_spikes=int(count.sum()),
            sheet_fraction_above20Hz=float((hz>=20).mean()),sheet_fraction_above200Hz=float((hz>=200).mean())))
    return axes,records

def clip_trajectory_segments(points,times,bounds):
    """Clip the drawn polyline to the 3D view, retaining all source samples separately."""
    p=points[:-1];delta=np.diff(points,axis=0)
    start=np.zeros(len(p));stop=np.ones(len(p));valid=np.ones(len(p),bool)
    for k,(low,high) in enumerate(bounds):
        moving=delta[:,k]!=0
        valid&=moving|((p[:,k]>=low)&(p[:,k]<=high))
        with np.errstate(divide='ignore',invalid='ignore'):
            a=np.where(moving,(low-p[:,k])/delta[:,k],-np.inf)
            b=np.where(moving,(high-p[:,k])/delta[:,k],np.inf)
        start=np.maximum(start,np.minimum(a,b));stop=np.minimum(stop,np.maximum(a,b))
    valid&=start<=stop
    ids=np.flatnonzero(valid);fractions=np.c_[start[valid],stop[valid]]
    segments=p[ids,None,:]+fractions[:,:,None]*delta[ids,None,:]
    segment_times=times[ids,None]+fractions*np.diff(times)[ids,None]
    return segments,segment_times,ids

def draw_zm_trajectory(fig,spec,a,samples,energy_window):
    """Plot actual Z, effective adaptation and rate; the middle axis is not inhibition."""
    duration=len(a['spikes_1ms'])/1000
    window=TRAJECTORY_WINDOW or (0.,duration)
    assert window[0]<=energy_window[0]<energy_window[1]<=window[1]
    ax=fig.add_subplot(spec,projection='3d');ax.computed_zorder=False
    st=a['slow_time_ms']/1000
    rate=a['spikes_1ms'][:,0].reshape(-1,5).sum(1)/32000/.005
    t=(np.arange(len(rate))+.5)*.005
    xyz=np.c_[np.interp(t,st,a['Z'][:,0]),ETA*np.interp(t,st,a['M'][:,0]),gaussian_filter1d(rate,1)]
    take=(t>=window[0])&(t<=window[1]);points=xyz[take];times=t[take]
    visible=(points[:,0]>=.7)&(points[:,2]>=0)&(points[:,2]<=400)
    bounds=np.array([[.7,1.02],[-.02,points[visible,1].max()*1.10],[0,400]])
    segments,segment_times,segment_ids=clip_trajectory_segments(points,times,bounds)
    norm=Normalize(0,1);cmap=plt.get_cmap('viridis')
    energy_take=(segment_times.mean(1)>=energy_window[0])&(segment_times.mean(1)<=energy_window[1])
    ax.add_collection3d(Line3DCollection(segments[energy_take],colors='#555555',linewidths=4,alpha=.5,zorder=5))
    ax.add_collection3d(Line3DCollection(segments,
        colors=cmap(norm(segment_times.mean(1)/duration)),linewidths=1.5,zorder=6))
    ax.set(xlim=bounds[0],ylim=bounds[1],zlim=bounds[2],
        xlabel='$Z$',ylabel=r'$\eta_M M$ (mV)',zlabel=r'$r_E$ (Hz)')
    ax.view_init(elev=26,azim=-125);ax.set_box_aspect((1.25,1,1.05));ax.set_anchor('W')
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.labelpad=18;axis.pane.set_facecolor('#f2f5fa');axis.set_major_locator(MaxNLocator(4))
    fig.canvas.draw();boxes=[];records=[]
    for s in [s for s in samples if s['number']<=5 and s['number'] not in HIDDEN_STATE_NUMBERS]:
        assert window[0]<=s['time_s']<=window[1]
        pt=np.array([np.interp(s['time_s'],t,xyz[:,k]) for k in range(3)])
        ax.scatter(*pt,c=s['color'],s=40,edgecolor='white',depthshade=False,zorder=10)
        x,y,_=proj3d.proj_transform(*pt,ax.get_proj());anchor=ax.transData.transform((x,y));unit=fig.dpi/72
        for offset in [(14,14),(-24,14),(22,-24),(-24,-24),(0,40),(40,0),(-40,0),(0,-40)]:
            center=anchor+np.array(offset)*unit
            box=(center[0]-12*unit,center[1]-12*unit,center[0]+12*unit,center[1]+12*unit)
            if all(box[2]<b[0] or box[0]>b[2] or box[3]<b[1] or box[1]>b[3] for b in boxes):
                boxes.append(box);break
        ax.annotate(str(s['number']),xy=(x,y),xytext=offset,textcoords='offset points',
            ha='center',va='center',fontsize=15,weight='bold',
            bbox=dict(boxstyle='circle,pad=.2',fc='white',ec='#777'),
            arrowprops=dict(arrowstyle='-',color='#777',lw=.6),zorder=20)
        records.append(dict(number=s['number'],time_s=s['time_s'],Z_etaM_M_rate=pt))
    cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=ax.inset_axes([1.06,.14,.035,.7]))
    ax.set_xticks([.7,.8,.9,1.]);ax.set_yticks([.2,.4,.6]);ax.set_zticks([0,100,200,300,400])
    cb.set_label('Normalized time',fontsize=16)
    cb.set_ticks([0,.25,.5,.75,1]);cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    return [ax],dict(time_window_s=list(window),coordinates='Mean Z / eta_M times mean M / all-E rate',
        middle_axis_source='eta_M * observations.M[:,0]',middle_axis_unit='mV',eta_M=ETA,
        states=records,time_s=times,trajectory_Z_etaM_M_rate=points,
        rate_bin_ms=5,rate_gaussian_sigma_ms=5,time_downsampling=False,energy_field_window_s=energy_window,
        color_time_normalization=dict(formula='t / simulation_duration_s',simulation_duration_s=duration,limits=[0,1]),
        normalized_time=times/duration,view_limits_Z_etaM_M_rate=bounds,
        visible_segments=segments,visible_segment_times_s=segment_times,visible_segment_source_indices=segment_ids,
        annotation_state_numbers=[r['number'] for r in records],energy_window_legend_shown=False,
        interpretation='Full simulation from0s to termination; colors use t/T. Only the displayed polyline is clipped at Z>=0.7 and rate<=400Hz. All source samples are preserved in metadata. The visible portion of F energy interval retains its grey underlay without a legend.')
