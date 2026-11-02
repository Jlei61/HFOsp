#!/usr/bin/env python3
"""Current Fig5 A-F, Fig4F network and historical Z/M parameters."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import copy
import argparse
import shutil
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(Path(__file__).resolve().parent)]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.signal import butter,sosfiltfilt,welch,find_peaks
from scipy.ndimage import uniform_filter1d,gaussian_filter1d
import build_fig5_transition_focus as layout
import analyze_topic4_fig5_onset_z_field as energy

OUT=ROOT/'results/topic4_sef_hfo/fig5_fig4f_historical_zm_20260930'
SOURCE=OUT
MAIN_BAND_HZ=None
COMPARISON_BAND_HZ=(30.,80.)
GAIN_PERCENTILE=95.
DISPLAY_GAIN=.68/1.48
WINDOW=(11.8,13.2)
TRAJECTORY_WINDOW=None
MAP_STATE_NUMBERS=None
MAP_WINDOW_MS=50
HIDDEN_STATE_NUMBERS=set()
ZOOM_STATE_NUMBERS=[1,2]
FIELD_DISPLAY_MODE='raw_z'
ETA=.007451594355587098
RECRUIT=12.7805


def write(path,obj):
    layout.write(path,obj)


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


def draw_onset_fields(fig,spec,a,comparison,mode):
    before=len(fig.axes)
    meta,bars=layout.previous.draw_f(fig,spec,a,comparison)
    axes=[v for v in fig.axes[before:] if v.axison]
    raw_limits=copy.deepcopy(meta['colorbar_limits'])
    for v in fig.axes[before:]:
        if not v.axison:
            for text in v.texts:
                if text.get_text()=='F':text.set_visible(False)
        else:
            v.tick_params(labelsize=16);v.xaxis.label.set_fontsize(18);v.yaxis.label.set_fontsize(18)
    for i,(cb,v) in enumerate(bars):
        v.set_xticks([-20,0,20]);v.set_yticks([-10,0,10,20])
        if mode=='minmax_z':
            cb._colorbar.update_normal(ScalarMappable(norm=Normalize(0,1),cmap='Blues'))
            cb.set_yticks([0,.5,1]);cb.set_title('Normalized\nz field',fontsize=16,pad=12)
        elif mode=='raw_z':
            cb.set_yticks(np.linspace(*cb.get_ylim(),3))
        else:raise ValueError(f'Unknown field display: {mode}')
        cb.yaxis.set_major_formatter(FormatStrFormatter('%g' if mode=='minmax_z' else '%.1f'))
        v.set_title('Model onset' if i==0 else f'Seizure onset {comparison["selected"]["seizure_idx"]+1}',fontsize=17,weight='bold')
    meta.update(display_mode=mode,raw_z_colorbar_limits=raw_limits,
        display_formula='(z - min(z)) / (max(z) - min(z)), separately for model and patient' if mode=='minmax_z' else 'signed baseline robust z',
        displayed_contact_values=[energy.clinical._normalize_minmax(np.asarray(comparison[k])).tolist() if mode=='minmax_z' else comparison[k]
            for k in ['model_robust_z','patient_robust_z']],
        enhancement_direction_visible=mode=='raw_z',raw_values_preserved=True)
    if mode=='minmax_z':
        meta['quantity']='Within-field min-max normalized baseline robust z; spatial shape only'
        meta['colorbar_limits']=[[0,1],[0,1]]
    return meta,bars,axes


def align_field_row(fig,bars,trajectory_axes,b_spec):
    """Match F's visible width to E, with F map tops/bottoms aligned to B."""
    from matplotlib.transforms import Bbox
    def tight(axes):
        renderer=fig.canvas.get_renderer();boxes=[]
        for ax in axes:
            boxes.append(ax.get_tightbbox(renderer))
            boxes.extend(child.get_tightbbox(renderer) for child in ax.child_axes)
            if hasattr(ax,'zaxis'):
                boxes.extend(axis.label.get_window_extent(renderer) for axis in (ax.xaxis,ax.yaxis,ax.zaxis))
        return Bbox.union([b for b in boxes if b is not None]).transformed(fig.dpi_scale_trans.inverted())
    fig.canvas.draw();target=tight(trajectory_axes)
    bb=b_spec.get_position(fig);fw,fh=fig.get_size_inches();height=bb.height*fh
    field_axes=[]
    for i,(cb,ax) in enumerate(bars):
        width=height*abs(np.diff(ax.get_xlim())[0]/np.diff(ax.get_ylim())[0])
        x=target.x0+.60+i*target.width/2
        ax.set_position([x/fw,bb.y0,width/fw,bb.height])
        cb.set_position([(x+width+.18)/fw,bb.y0,.10/fw,bb.height])
        ax.tick_params(labelsize=12);ax.xaxis.label.set_fontsize(14);ax.yaxis.label.set_fontsize(14)
        ax.title.set_fontsize(15);cb.tick_params(labelsize=12);cb.title.set_fontsize(12)
        for artist in ax.collections:
            if hasattr(artist,'set_sizes'):artist.set_sizes([26])
        field_axes.extend([ax,cb])
    fig.canvas.draw();current=tight(field_axes)
    dx=(target.x0-current.x0)/fw
    for ax in field_axes:
        pos=ax.get_position();ax.set_position([pos.x0+dx,pos.y0,pos.width,pos.height])
    fig.canvas.draw();current=tight(field_axes)
    dx=(target.x1-current.x1)/fw
    for ax in (bars[1][1],bars[1][0]):
        pos=ax.get_position();ax.set_position([pos.x0+dx,pos.y0,pos.width,pos.height])
    fig.canvas.draw();actual=tight(field_axes)
    assert np.allclose([actual.x0,actual.x1],[target.x0,target.x1],atol=1e-8)
    return dict(E_visible_bounds_inches=target.bounds,F_visible_bounds_inches=actual.bounds,
        F_map_bounds=[ax.get_position().bounds for _,ax in bars],B_bounds=bb.bounds,
        E_F_visible_width_equal=True,F_map_row_aligned_with_B=True)


def choose_states(a,onset):
    """Find the last quiet interval and an earlier event; retain actual sample times."""
    hz=a['spikes_1ms'][:,0]/32000/.001
    sm=uniform_filter1d(hz,20,mode='nearest')
    # A quiet run ends before the sustained entry, and is at least20ms long.
    quiet=sm<5
    starts=np.flatnonzero(np.diff(np.r_[False,quiet,False].astype(int))==1)
    ends=np.flatnonzero(np.diff(np.r_[False,quiet,False].astype(int))==-1)
    good=(ends-starts>=20)&(ends<onset*1000)&(ends>(onset-2)*1000)
    recruit=float(ends[good][-1]/1000) if good.any() else onset-.3
    lower=max(500,round((recruit-1.2)*1000));upper=round((recruit-.20)*1000)
    peaks,_=find_peaks(sm[lower:upper],distance=100,prominence=2)
    if len(peaks):
        heights=sm[lower+peaks]
        use=peaks[heights>=max(5.,heights.max()*.2)]
        inter=(lower+int(use[-1] if len(use) else peaks[np.argmax(heights)]))/1000
    else:inter=(lower+int(np.argmax(sm[lower:upper])))/1000
    # Three increasingly recruited native states before the operational entry.
    gap=max(.12,onset-recruit)
    times=[inter,recruit+min(.025,gap*.08),recruit+gap*.25,recruit+gap*.60,onset+.10]
    times=[round(t,3) for t in times]
    window=(max(0.,round((inter-.35)*10)/10),round((onset+.35)*10)/10)
    return times,recruit,window


def diagnostics(a,onset):
    t=a['lfp_native_time_ms']/1000;raw=a['lfp_native'];rate=a['rate_E']
    bp=sosfiltfilt(butter(4,[30,80],btype='bandpass',fs=10000,output='sos'),raw,axis=0)
    rows=[]
    duration=len(rate)/10000
    for label,lo,hi in [('pre_entry',onset-1,onset),('post_entry_early',onset,onset+1),('late',duration-1.25,duration-.25)]:
        mask=(t>=lo)&(t<hi)
        f,p=welch(rate[mask],fs=10000,nperseg=min(mask.sum(),5000))
        peaks={}
        for band in [(1,150),(150,1000)]:
            take=(f>=band[0])&(f<=band[1])
            peaks[str(band)]=dict(peak_Hz=float(f[take][np.argmax(p[take])]),rate_RMS_Hz=float(np.sqrt(np.sum(p[take])*(f[1]-f[0]))))
        rows.append(dict(label=label,time_s=[lo,hi],rate_mean_Hz=float(rate[mask].mean()),rate_std_Hz=float(rate[mask].std()),
            native_rate_spectrum=peaks,raw_mean=raw[mask].mean(0),raw_std=raw[mask].std(0),band30_80_rms=np.sqrt(np.mean(bp[mask]**2,axis=0))))
    write(OUT/'waveform_diagnostics.json',dict(windows=rows,contact_names=a['contact_names'],
        interpretation='Mean firing rate and population oscillation frequency are separate observables; all spectra use native10kHz samples.'))
    # A narrow raw view makes high-frequency cycles inspectable without changing main A.
    fig,axs=plt.subplots(3,2,figsize=(12,8))
    idx=list(a['contact_names']).index('ICL1')
    for j,(label,center) in enumerate([('Entry',onset+.15),('Late',duration-.75)]):
        mask=(t>=center-.05)&(t<center+.05)
        axs[0,j].plot(t[mask],rate[mask],lw=.6,color='#333333')
        axs[0,j].set(title=label,ylabel='All E rate (Hz)')
        axs[1,j].plot(t[mask],raw[mask,idx],lw=.7,color='#c67b35')
        axs[1,j].set(ylabel='ICL1 current proxy',xlabel='Time (s)')
        lo,hi=(onset,onset+1) if j==0 else (duration-1.25,duration-.25)
        broad=(t>=lo)&(t<hi)
        f,p=welch(rate[broad],fs=10000,nperseg=min(broad.sum(),5000))
        axs[2,j].semilogy(f,p,color='#333333',lw=1)
        axs[2,j].set(xlim=(1,1000),xlabel='Frequency (Hz)',ylabel='Rate PSD')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'figures/native-oscillation-check.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    folder=OUT/'visual_qa';folder.mkdir(exist_ok=True)
    fig,axs=plt.subplots(3,1,figsize=(12,7),sharex=True)
    rate10=a['spikes_1ms'][:,0].reshape(-1,10).sum(1)/32000/.01
    t10=(np.arange(len(rate10))+.5)*.01
    axs[0].plot(t10,rate10,color='#333333',lw=.65,label='All E (10 ms bins)')
    axs[0].axhline(200,c='#ba263c',ls=':',lw=.8)
    axs[0].set_ylabel('E rate (Hz)')
    st=a['slow_time_ms']/1000
    for j,style in enumerate(['-','--']):
        axs[1].plot(st,a['Z'][:,[0,5][j]],c=layout.Z_COLOR,ls=style,lw=1,label=['All E','Core A'][j])
        axs[2].plot(st,ETA*a['M'][:,[0,1][j]],c=layout.M_COLOR,ls=style,lw=1)
    axs[1].set_ylabel('Z');axs[1].legend(frameon=False)
    axs[2].set(ylabel='eta_M M (mV)',xlabel='Time (s)')
    for ax in axs:ax.axvline(onset,c='#7043a5',ls='--',lw=.8)
    axs[0].legend(frameon=False);fig.tight_layout()
    fig.savefig(folder/'whole_run.png',dpi=150);plt.close(fig)


def spatial_selection_audit(a,onset,samples):
    folder=OUT/'visual_qa';folder.mkdir(exist_ok=True)
    fig,axs=plt.subplots(2,8,figsize=(20,6))
    rows=[]
    scan_times=np.r_[samples[1]['time_s'],onset+np.array([.025,.05,.075,.1,.15,.2,.3,.4,.5,.65,.8,1.,1.5,2.,3.])]
    for ax,ts in zip(axs.flat,scan_times):
        lo=round(ts*1000)-25
        count=a['field_1ms'][lo:lo+50].sum(0)
        hz=count/a['cell_e_counts']/.05
        image=ax.imshow(hz.reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',norm=layout.PowerNorm(.6,0,500))
        for j,xy in enumerate(a['centers_mm']):
            from matplotlib.patches import Circle
            ax.add_patch(Circle(xy,float(a['core_radius_mm']),ec='#2dd4cd',fc='none',lw=1))
            ax.text(*xy,'AB'[j],color='white',fontsize=8)
        rr=a['regions_1ms'][lo:lo+50].sum(0)/a['region_counts']/.05
        row=dict(time_s=float(ts),global_rate_Hz=float(count.sum()/32000/.05),coreA_Hz=float(rr[0]),coreB_Hz=float(rr[1]),fraction_sheet_above20Hz=float((hz>=20).mean()))
        rows.append(row)
        ax.set_title(f"{ts:.3f}s; E={row['global_rate_Hz']:.1f}\nA={rr[0]:.0f}, B={rr[1]:.0f}",fontsize=9)
    fig.tight_layout();fig.savefig(folder/'spatial_selection.png',dpi=140);plt.close(fig)
    write(folder/'spatial_selection.json',rows)
    (folder/'README.md').write_text('### spatial_selection.png\n从Local采样时刻及进入后3 s的16个原生50 ms空间率窗；标题同时列出总体及两核放电率。用于核对局部招募如何进入全场高活动及其后续。\n**关注点**：所有格来自同次已完成轨迹；新增显示帧不代表新增仿真。\n')


def package_notes(meta):
    selected=meta['E2']['selected'];times=[s['time_s'] for s in meta['snapshots']]
    map_samples=[s for s in meta['snapshots'] if s['number'] in MAP_STATE_NUMBERS]
    map_numbers='、'.join(str(s['number']) for s in map_samples)
    start,end=meta['display_time_window_s'];entry=meta['first_entry']['onset_s']
    band=meta['waveform']['band_hz']
    readout=f'{band[0]:g}–{band[1]:g} Hz带通电流代理' if band is not None else '未滤波电流代理（保留直流）'
    compare_name=f'{COMPARISON_BAND_HZ[0]:g}-{COMPARISON_BAND_HZ[1]:g}Hz'
    compare_label=f'{COMPARISON_BAND_HZ[0]:g}–{COMPARISON_BAND_HZ[1]:g} Hz'
    processing=(f'主图在完整18.1 s原生轨迹上使用四阶Butterworth {band[0]:g}–{band[1]:g} Hz带通，SOS前后向零相位滤波后才截取显示窗；保留10 kHz样本，不作CAR、频谱均衡或额外基线扣除。'
        if band is not None else '主图保留全部10 kHz原始样本及直流，不作时间滤波、CAR或基线扣除。')
    descriptions={
      'a':f'15触点{readout}读出；所有时间点与B–E同步，整段采用相同滤波及每触点固定增益。',
      'b':'Z为蓝色，有效adaptation为橙色；总体E用实线，Core A用虚线。',
      'c':'状态1和2的两个300 ms窗口内Core A/B的E-only raster；每核20个固定细胞，两窗使用同一组细胞。',
      'd':f'状态{map_numbers}各{MAP_WINDOW_MS} ms原生E放电率场，统一0–500 Hz色条放在第4幅右侧。',
      'e':'从0秒开始的同次完整仿真轨迹，以t/T归一化时间着色；视野限制为Z≥0.7、E率0–400 Hz，并标注状态1–5。',
      'f':f'模型进入后1 s的宽带robust-z场，与同患者选出的{selected["public_seizure"]}早期场对照。'}
    text=[]
    for k,v in descriptions.items():
        text.append(f'### fig5-panel{k}.png / .pdf / .svg\n{v}\n全部模型数据来自Fig4F网络加旧Z/M的同一条已保存轨迹。\n**关注点**：图件来自已保存的同一条仿真，候选待作者目视检查。\n')
    text.extend([
      f'### fig5-complete-layout.png / .pdf / .svg\n当前A–F布局、右下预留G，使用Fig4F的xy_left_20网络，接入旧版Z/M。A为{readout}。\n**关注点**：操作性高率进入与持续振荡须分别判断。\n',
      f'### fig5-complete-layout-{compare_name}.png / .pdf\n仅A改用{compare_label}带通读出，B–F完全相同。先滤完整轨迹再截窗，按进入前固定定标。\n**关注点**：用于比较频带对读出形态的影响。\n',
      f'### fig5-panela-{compare_name}.png / .pdf / .svg\n对照频带的A单panel。状态标记与主图一致。\n**关注点**：定标固定，无进入后增益。\n',
      '### native-oscillation-check.png / .pdf\n进入期和记录末期各100 ms的原生放电率与未滤波ICL1电流读出，并列给出原生率频谱。该图辅助辨别高平均率与群体周期性。\n**关注点**：保留10 kHz样本，没有30–80 Hz限制。\n'])
    text.append('### onset-field-signed-z.png / .pdf\nF的原始带符号robust-z对照；窗口、基线和空间投影与主图一致。\n**关注点**：模型15触点全部为负，不能由主图0–1色条推断能量增强。\n')
    (OUT/'figures/README.md').write_text('\n'.join(text))
    caption=f"""# Figure 5：Fig4F网络加旧Z/M

**同一患者拟合网络中，慢变量驱动的高活动进入与空间读出。** 保留Fig4F的xy_left_20全部快速网络参数：32,000 E、8,000 I；拓扑种子2511、输入种子847401。仅接入历史Z/M，tau_Z=5 s，tau_M=0.5 s，eta_M=0.007451594355587098，抑制电流阈值95.19851312666987，初始Z=1、M=0。没有外部kick、Z重置或新参数搜索。关闭Z/M时的首1 s已逐点复现Fig4F存档；CPU和GPU同一短轨迹的观测与最终Z/M一致。

**A，连续读出。** 显示{start:.3f}–{end:.3f} s，15触点按SCL9–SCL6、ICL11–ICL1固定排列。信号是E细胞突触电流abs(I_E)+abs(I_I)的距离加权平均，在Z/M调节电流之前观察；它不是膜电位或临床电压。{processing}每触点以{RECRUIT:g} s前绝对信号P{GAIN_PERCENTILE:g}固定定标，下限为跨触点中位数的15%，显示系数{DISPLAY_GAIN:.6g}，全程增益相同。另附A的四阶零相位{compare_label}对照；其余panel相同。状态编号连续为1–5，状态1用竖虚线标记近似起始，不再用矩形圈出C的窗口；其余编号竖线对应D和E的实际采样时刻。状态2（Local）采样此前事件回落后的Core A再次激活，不等同于持续高态进入阈值。图内省略频带说明，状态编号统一紧邻竖线。带通改变波形形态；间期间歇事件的时间与传播顺序可见，不应把滤波后的每个周期直接当作神经元同步振荡证据。

**B，抑制效能与adaptation。** 上图Z为蓝色；下图eta_M M为橙色，单位mV。两图均以实线表示总体E、虚线表示Core A内E，时间轴与A对齐。每次E放电令M增加1，M随后衰减；膜电流减去eta_M M。Z乘在E细胞收到的抑制电流上，I细胞的抑制与adaptation均不受此模块调节。

**C，细胞放电。** 状态1和2分别位于{times[0]:.3f}及{times[1]:.3f} s，显示各状态前50至后250 ms。两个核心分别固定抽取20个E细胞，两个窗口内细胞和顺序不变。

**D，空间招募。** 四幅仅显示状态{map_numbers}，对应{', '.join(f"{s['time_s']:.3f}" for s in map_samples)} s；首幅标题为Local。每幅统计中心前后{MAP_WINDOW_MS/2:g} ms，将20×20网格内E放电数除以该格E数量和{MAP_WINDOW_MS} ms得到Hz。所有状态采用相同0–500 Hz色标与幂归一化，不插值时间或空间数据。圆为Fig4F中的实际Core A/B位置与半径。{meta['map_description']}早期波前主要向外扩展，不应把它称为只沿长轴的传播；传播后核心活动曾回落并再激活，因此四幅图也不意味着单调累积招募。

**E，状态轨迹。** 同一次完整仿真的总体Z、有效adaptation eta_M mean(M_i)和总体E率构成三维轨迹；中间轴使用实际保存的M乘以eta_M，和B下图总体E实线同义。采用从{meta['E1']['time_window_s'][0]:.3f}秒至仿真结束{meta['E1']['time_window_s'][1]:.3f}秒的全部记录，颜色定义为t/T，其中T={meta['E1']['color_time_normalization']['simulation_duration_s']:.3f}秒，色条为0–1归一化时间，不按左列短窗重新定标。按作者要求，显示范围限制为Z≥0.7及E率0–400 Hz；超出范围的线段仅从图中裁掉，不能由图的终点推断仿真结束或高活动停止。完整源轨迹仍保存在metadata中。状态1–5按其实际时点标注，与A/B及D的对应状态一致。F实际使用的{meta['E2']['model_early_s'][0]:.3f}–{meta['E2']['model_early_s'][1]:.3f}秒区间仅保留视野内部分的灰色描边，取消图内F window说明。率以5 ms分箱并作仅用于显示的5 ms高斯平滑，慢变量插值到分箱中心；最后一个分箱中心比慢变量末个样本晚2.5 ms，按末值延拓。源记录覆盖0–T，率的分箱中心为0.0025至T−0.0025秒。这是原生轨迹，不是分岔或吸引子估计。

**F，进入后宽带能量场。** 本版进入标记沿用上一版历史工作点图的t120：总体E率作居中20 ms平均，取首个100 ms窗内至少80%样本达到120 Hz的位置，本次为{entry:.4f} s。严格的200 Hz连续200 ms指标另记为次要读出，不能与t120混称。模型从进入起点取1 s，基线0.5–3.5 s；使用Fig3C的CAR、1 s Tukey窗、0.5 s步长、1–150 Hz对数功率，以基线中位数及1.4826 MAD计算robust z。模型基线5个重叠谱窗，进入窗1个谱窗；较短基线仅用于本模型描述。患者为E10的{selected['public_seizure']}，沿用临床0–10 s与远端基线−120至−90 s；在25次完整发作中按正平均增强门槛和有符号场相关选出最近似的一次，场r={selected['field_r']:.3f}。绘图使用Fig3C冻结共享空间；主图对每幅触点z分别作(z−min(z))/(max(z)−min(z))，色条为0–1，表示空间相对高低，不保留相对基线的增强或减弱方向。另附onset-field-signed-z恢复实际带符号z色条，原始数值不变。图内标题为Model onset与Seizure onset 26，能量定义及时间窗写于图注。模型{meta['E2']['model_positive_contacts']}/15触点为正，平均z={meta['E2']['model_mean_z']:.3f}；实际增强方向须结合这些值，不能由形状相关替代。

**G，预留区域。** 仅预留空白及角标，待加入群体统计；本版不填入推测数据。

模型统计单位为一个固定网络的一条输入轨迹。未建立模型秒与患者秒的定量换算，观测算子也不同；该场比较是选择后的示例，不表示独立验证或完整发作机制。平均放电率不等于群体振荡频率；进入与后续振荡由native-oscillation-check另行检查。
"""
    (OUT/'fig5_caption_detailed_zh.md').write_text(caption)
    (OUT/'README.md').write_text(f"""# Fig5：Fig4F网络 + 历史Z/M

本版复用同一次18.1秒原生仿真，将五个可见状态连续编号为1–5；C为1/2，D为2–5，首幅空间图标题为Local。沿用紧凑双列布局，F与B同排且可见宽度与E一致；E的Z显示下限为0.7，右下预留G。F主图以0–1显示空间形状，另附真实正负z及算法核对。网络来自Fig4F xy_left_20，Z/M来自历史tau500设置；没有新增实验。

- [完整图](figures/fig5-complete-layout.png) / [PDF](figures/fig5-complete-layout.pdf)
- [单panel包](fig5-single-panels-png-pdf-svg.zip)
- [原生振荡诊断](figures/native-oscillation-check.png)
- [详细图注](fig5_caption_detailed_zh.md)
- [实验参数](protocol.json) / [结果](result.json)

仿真脚本：scripts/run_topic4_fig5_fig4f_historical_zm.py。绘图脚本：scripts/paper_figures/build_fig5_fig4f_historical_zm.py。原始观测在{SOURCE}/zm/observations.npz；无需重新仿真即可重绘。A为{readout}，对照为{compare_label}；F继续使用原Fig3C的CAR及1–150 Hz能量定义。数据验证与图件自查写入visual_qa.json；作者目视检查待完成，未覆盖正式图。
""")


def main():
    global WINDOW,TRAJECTORY_WINDOW,MAP_STATE_NUMBERS,MAP_WINDOW_MS,RECRUIT,OUT,MAIN_BAND_HZ,COMPARISON_BAND_HZ,GAIN_PERCENTILE,DISPLAY_GAIN,HIDDEN_STATE_NUMBERS,ZOOM_STATE_NUMBERS,FIELD_DISPLAY_MODE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=SOURCE)
    parser.add_argument('--readout-band',type=float,nargs=2,default=None,metavar=('LOW_HZ','HIGH_HZ'))
    parser.add_argument('--comparison-band',type=float,nargs=2,default=COMPARISON_BAND_HZ,metavar=('LOW_HZ','HIGH_HZ'))
    parser.add_argument('--readout-percentile',type=float,default=GAIN_PERCENTILE)
    parser.add_argument('--readout-gain',type=float,default=DISPLAY_GAIN)
    parser.add_argument('--selection-file',type=Path)
    parser.add_argument('--trajectory-window',type=float,nargs=2,metavar=('START_S','END_S'))
    args=parser.parse_args()
    OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
    MAIN_BAND_HZ=args.readout_band;COMPARISON_BAND_HZ=args.comparison_band
    GAIN_PERCENTILE=args.readout_percentile;DISPLAY_GAIN=args.readout_gain
    TRAJECTORY_WINDOW=args.trajectory_window
    for band in [MAIN_BAND_HZ,COMPARISON_BAND_HZ]:
        if band is not None and not (0<band[0]<band[1]<5000):parser.error('Readout cutoffs must satisfy 0 < low < high < 5000 Hz.')
    if not (0<GAIN_PERCENTILE<=100 and DISPLAY_GAIN>0):parser.error('Invalid display normalization.')
    if OUT!=SOURCE:
        for name in ['protocol.json','result.json','prefix_gate.json','static_reconstruction.json','acceleration_gate.json']:
            shutil.copy2(SOURCE/name,OUT/name)
        selection_source=args.selection_file or SOURCE/'display_selection.json'
        if selection_source.resolve()!=(OUT/'display_selection.json').resolve():
            shutil.copy2(selection_source,OUT/'display_selection.json')
    result=json.loads((SOURCE/'result.json').read_text());assert result['status']=='COMPLETE'
    for gate in ['prefix_gate.json','static_reconstruction.json','acceleration_gate.json']:
        assert json.loads((SOURCE/gate).read_text())['status']=='PASS'
    (OUT/'figures').mkdir(exist_ok=True)
    a={}
    for path in [SOURCE/'geometry.npz',SOURCE/'zm/observations.npz']:
        with np.load(path) as x:a.update({k:x[k] for k in x.files})
    assert np.allclose(np.diff(a['lfp_native_time_ms']),.1,rtol=0,atol=1e-8)
    assert np.array_equal(a['field_1ms'].sum(1),a['spikes_1ms'][:,0])
    smooth=np.convolve(np.asarray(a['rate_E'],float),np.ones(200)/200,mode='same')
    crossings=np.flatnonzero(np.convolve((smooth>=120).astype(float),np.ones(1000),mode='valid')>=800)
    if not len(crossings): raise RuntimeError('No historical t120 entry; produce a no-entry report rather than an onset figure.')
    onset=float(crossings[0]/10000)
    rates10=a['spikes_1ms'][:,0].reshape(-1,10).sum(1)/32000/.01
    windows200=np.flatnonzero(np.convolve((rates10>=200).astype(int),np.ones(20,dtype=int),'valid')==20)
    onset200=None if not len(windows200) else float(windows200[0]*.01)
    times,RECRUIT,WINDOW=choose_states(a,onset)
    if onset200 is not None:
        times[-1]=round(max(times[-1],onset200+.10),3)
        WINDOW=(WINDOW[0],float(np.ceil(max(WINDOW[1],onset200+.30)*10)/10))
    selection=OUT/'display_selection.json'
    selected={}
    if selection.exists():
        selected=json.loads(selection.read_text());times=selected['times_s'];RECRUIT=selected['recruitment_start_s'];WINDOW=tuple(selected['window_s'])
    else: write(selection,dict(times_s=times,recruitment_start_s=RECRUIT,window_s=WINDOW,method='Last20ms quiet interval (<5Hz centered20ms rate) and preceding prominent event; increasing native times within the entry episode. Final state and view extend beyond the stricter200Hz entry so the high state is visible.'))
    labels=selected.get('labels',['Interictal','Pre-ictal','','','Onset'])
    HIDDEN_STATE_NUMBERS=set(selected.get('hidden_state_numbers',[]))
    ZOOM_STATE_NUMBERS=selected.get('raster_zoom_state_numbers',[1,2])
    FIELD_DISPLAY_MODE=selected.get('field_display_mode','raw_z')
    colors=selected.get('colors',layout.COLORS)
    assert len(times)==len(labels)==len(colors) and np.all(np.diff(times)>0)
    samples=[dict(number=i+1,time_s=t,color=colors[i],label=labels[i]) for i,t in enumerate(times)]
    MAP_STATE_NUMBERS=selected.get('map_state_numbers',[s['number'] for s in samples[-4:]])
    MAP_WINDOW_MS=int(selected.get('map_window_ms',50))
    assert MAP_WINDOW_MS>0 and MAP_WINDOW_MS%2==0
    assert 0<=WINDOW[0]<min(times)<max(times)<WINDOW[1]<=result['duration_s']
    if TRAJECTORY_WINDOW is not None:
        assert 0<=TRAJECTORY_WINDOW[0]<min(times)<max(times)<TRAJECTORY_WINDOW[1]<=result['duration_s']
    layout.WINDOW=WINDOW;layout.markings=marks
    energy.OUT=OUT/'onset_field'
    def protocol():
        energy.OUT.mkdir(exist_ok=True)
        # Reuse the established estimator while naming this trajectory's actual onset observer.
        value=json.loads((ROOT/'results/topic4_sef_hfo/fig5_preentry_event_audit_20260914/onset_broadband_z_20260915/analysis_contract.json').read_text())
        value['model_onset']='Historical t120 retained from preceding figure: centered20ms all-E rate; earliest100ms window with >=80% samples >=120Hz. Rounded to nearest0.5ms only for PSD alignment.'
        value['model_source']='Fig4F xy_left_20 network, historicalZ/M, topology2511 and dynamics847401; all native panels share this source.'
        write(energy.OUT/'analysis_contract.json',value)
        return value
    energy.protocol=protocol
    onset_sample=round(onset*2000)/2000
    comparison=energy.compute(a,onset_sample,'fig4f_historical_zm')
    if onset200 is not None:
        primary_protocol=energy.protocol
        energy.protocol=lambda: dict(primary_protocol(),model_onset='Secondary observer: all-E10ms rate >=200Hz continuously200ms; start of episode.')
        secondary=energy.compute(a,round(onset200*2000)/2000,'secondary_200Hz_entry')
        write(OUT/'secondary_entry_field.json',secondary)
        energy.protocol=primary_protocol
    comparison['exact_entry_s']=onset;comparison['PSD_alignment_error_ms']=(onset_sample-onset)*1000
    write(OUT/'field_comparison.json',comparison)
    diagnostics(a,onset)
    spatial_selection_audit(a,onset,samples)
    plt.rcParams.update({'font.size':18,'axes.labelsize':20,'axes.titlesize':20,'xtick.labelsize':17,'ytick.labelsize':17,'svg.fonttype':'none','pdf.fonttype':42,'path.simplify':False})
    # Keep panel widths in inches while removing excess space between columns.
    fig=plt.figure(figsize=(25*2.10/2.25,18))
    outer=fig.add_gridspec(1,2,width_ratios=[1.12,1],left=.072,right=.948,top=.95,bottom=.085,wspace=.10)
    left=outer[0].subgridspec(7,1,height_ratios=[3.25,.38,1.38,.48,1.35,.48,1.7],hspace=.10)
    bgrid=left[2].subgridspec(2,1,hspace=.25)
    # Translate the entire right column without changing any panel's dimensions.
    right_column_shift_inches=.60
    right_box=outer[1].get_position(fig)
    right_dx=right_column_shift_inches/fig.get_figwidth()
    right=fig.add_gridspec(7,1,left=right_box.x0+right_dx,right=right_box.x1+right_dx,
        bottom=right_box.y0,top=right_box.y1,
        height_ratios=[3.25,.38,1.38,.48,1.35,.48,1.7],hspace=.10)
    e_spec=right[0];f_spec=right[2];g_spec=right[4:]
    ax=fig.add_subplot(left[0]);za=fig.add_subplot(bgrid[0],sharex=ax);ma=fig.add_subplot(bgrid[1],sharex=ax)
    waveform=draw_a(ax,a,samples,onset)
    slow=draw_slow(za,ma,a,samples,onset)
    zoom_samples=[s for s in samples if s['number'] in ZOOM_STATE_NUMBERS]
    assert len(zoom_samples)==2
    zoom_axes,zooms=layout.draw_zooms(fig,left[4],a,zoom_samples)
    map_axes,maps=draw_native_maps(fig,left[6],a,samples)
    trajectory_axes,trajectory=draw_zm_trajectory(fig,e_spec,a,samples,comparison['model_early_s'])
    field_meta,bars,field_axes=draw_onset_fields(fig,f_spec,a,comparison,FIELD_DISPLAY_MODE)
    field_alignment=align_field_row(fig,bars,trajectory_axes,left[2])
    for letter,spec in [('A',left[0]),('B',left[2]),('C',left[4]),('D',left[6]),('E',e_spec),('F',f_spec),('G',g_spec)]:
        p=spec.get_position(fig);fig.text(p.x0-.022,p.y1+(-.02 if letter=='G' else .01),letter,weight='bold',fontsize=25,va='bottom')
    fig.canvas.draw()
    for cb,v in bars:
        cp,p=cb.get_position(),v.get_position();cb.set_position([cp.x0,p.y0,cp.width,p.height])
    for s in samples:
        px=[v.transData.transform((s['time_s'],0))[0] for v in [ax,za,ma]]
        assert max(px)-min(px)<1e-8
    meta=dict(asset_id='fig4f_historical_zm_current_fig5_layout',source=str(SOURCE),output=str(OUT),result=result,
        fast_network='Fig4F xy_left_20',slow_source_commit='7951052',topology_seed=2511,dynamics_seed=847401,tau_Z_s=5.,tau_M_s=.5,eta_M=ETA,display_time_window_s=WINDOW,
        snapshots=samples,first_entry=dict(onset_s=onset,t120_s=onset,criterion_window_end_s=onset+.1,definition='Centered20ms all-E rate >=120Hz in >=80% of100ms; same observer as preceding historical figure.'),
        entry_check_from_saved_arrays=dict(onset_s=onset200,agrees=bool((onset200 is None and result['first_entry'] is None) or (onset200 is not None and result['first_entry'] is not None and np.isclose(onset200,result['first_entry']['onset_s'])))),
        waveform=waveform,slow_variables=slow,raster_zoom_windows=zooms,native_maps=maps,map_state_numbers=MAP_STATE_NUMBERS,map_window_ms=MAP_WINDOW_MS,
        map_description=selected.get('map_description','前三幅加密展示Core A局部活动、向外传播和到达远端Core B附近的阶段，最后一幅显示全场高活动。'),E1=trajectory,E2=comparison,E2_display=field_meta,
        all_native_panels_same_trajectory=True,F_model_recomputed=True,patient_reselected=True,
        F_title_lines=['Model onset',f'Seizure onset {comparison["selected"]["seizure_idx"]+1}'],
        hidden_state_numbers=sorted(HIDDEN_STATE_NUMBERS),raster_zoom_state_numbers=ZOOM_STATE_NUMBERS,
        panel_layout=dict(E_aligned_with_A=True,F_at_B_C=False,F_aligned_with_B=True,G_reserved_blank=True,E_anchor='W',
            **field_alignment,
            column_wspace=.10,right_column_shift_inches=right_column_shift_inches,figure_size_inches=fig.get_size_inches(),
            column_gap_inches=(e_spec.get_position(fig).x0-outer[0].get_position(fig).x1)*fig.get_figwidth(),
            A_bounds=left[0].get_position(fig).bounds,E_bounds=e_spec.get_position(fig).bounds,
            F_bounds=f_spec.get_position(fig).bounds,G_bounds=g_spec.get_position(fig).bounds),
        human_review='PENDING',agent_visual_review='PENDING',publication_status='CANDIDATE',producer=str(Path(__file__).resolve()),
        producer_sha256=layout.sha(__file__),source_result_sha256=layout.sha(SOURCE/'result.json'),
        rendering_command=[sys.executable,*sys.argv],
        plotting_dependencies={str(Path(m.__file__).resolve()):layout.sha(m.__file__) for m in [layout,layout.previous,layout.exporter,energy,energy.clinical]})
    groups=dict(A=[ax],B=[za,ma],C=zoom_axes,D=map_axes,E=trajectory_axes,F=field_axes)
    # Same composition and same raw trajectory, only the explicitly labeled A filter changes.
    # Save both composites before the shared panel exporter resets title padding.
    ax.clear();filtered=draw_a(ax,a,samples,onset,filtered=True)
    fig.canvas.draw()
    compare_name=f'{COMPARISON_BAND_HZ[0]:g}-{COMPARISON_BAND_HZ[1]:g}Hz'
    for ext in ['png','pdf']:
        fig.savefig(OUT/f'figures/fig5-complete-layout-{compare_name}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.12)
    ax.clear();draw_a(ax,a,samples,onset,filtered=False)
    fig.canvas.draw()
    files=layout.exporter.export(fig,groups,meta,OUT,validate_parameter_map=False)
    write(OUT/'fig5_metadata.json',meta);write(OUT/'panel_manifest.json',dict(files=files))
    plt.close(fig)
    af,aa=plt.subplots(figsize=(12,5));draw_a(aa,a,samples,onset,filtered=True)
    aa.tick_params(axis='x',labelbottom=True);aa.set_xlabel('Time (s)')
    af.tight_layout()
    for ext in ['png','pdf','svg']:af.savefig(OUT/f'figures/fig5-panela-{compare_name}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(af);write(OUT/'filtered_readout_metadata.json',filtered)
    with zipfile.ZipFile(OUT/'fig5-single-panels-png-pdf-svg.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted((OUT/'figures').glob('fig5-panel*')):z.write(path,path.name)
    if FIELD_DISPLAY_MODE=='minmax_z':
        rawfig=plt.figure(figsize=(12,5.5))
        rawspec=rawfig.add_gridspec(1,1,left=.085,right=.92,bottom=.15,top=.92)[0]
        rawmeta,rawbars,_=draw_onset_fields(rawfig,rawspec,a,comparison,'raw_z')
        rawfig.canvas.draw()
        for cb,v in rawbars:
            cp,p=cb.get_position(),v.get_position();cb.set_position([cp.x0,p.y0,cp.width,p.height])
        for ext in ['png','pdf']:rawfig.savefig(OUT/f'figures/onset-field-signed-z.{ext}',dpi=300,bbox_inches='tight')
        plt.close(rawfig);write(OUT/'signed_field_display.json',rawmeta)
    package_notes(meta)
    print(json.dumps(dict(output=str(OUT),onset=onset,patient=comparison['selected'],model_mean_z=comparison['model_mean_z']),ensure_ascii=False))


if __name__=='__main__':main()
