#!/usr/bin/env python3
"""Full Fig5 candidates for every observed M condition; no invented states."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt, spectrogram, welch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import proj3d
from run_topic4_m_parameter_modes import ROOT, OUT, read, write, ETA, TAU, SEEDS

ORDER=['SCL9','SCL8','SCL7','SCL6','ICL11','ICL10','ICL9','ICL8','ICL7','ICL6','ICL5','ICL4','ICL3','ICL2','ICL1']
COLORS=['#267ba8','#dd871c','#ba263c','#248d78','#ba263c']
REGCOL=['#724c91','#b33c6c','#197f9e']
CANON=ROOT/'results/paper-ready-figure/fig3'
plt.rcParams.update({'font.size':15,'axes.labelsize':17,'axes.titlesize':18,
    'xtick.labelsize':15,'ytick.labelsize':15,'pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False})


def safe(value):
    if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if isinstance(value,np.ndarray):return safe(value.tolist())
    if isinstance(value,np.generic):return safe(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def spans(mask):
    v=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(v==1),np.flatnonzero(v==-1)))


def load(folder,end_step=None):
    data={};end=0
    for path in sorted((folder/'chunks').glob('*.npz')):
        if '.tmp.' in path.name:continue
        with np.load(path) as a:
            if end_step is not None and int(a['end_step'])>end_step:break
            assert int(a['start_step'])==end
            end=int(a['end_step'])
            for k in a.files:
                if k not in ['start_step','end_step']:data.setdefault(k,[]).append(a[k])
    data={k:np.concatenate(v) for k,v in data.items()}
    assert len(data['spikes_1ms'])*10==end and len(data['raster'])==end
    if end_step is not None:assert end==end_step
    with np.load(OUT/'geometry.npz') as g:data.update({k:g[k] for k in g.files})
    return data


def analyze(a,r):
    rate=a['spikes_1ms'].reshape(-1,10,2).sum(1)/np.array([32000,8000])/.01
    regional=a['regions_1ms'].reshape(-1,10,6).sum(1)/a['region_counts']/.01
    high=[(lo,hi) for lo,hi in spans(rate[:,0]>=200) if hi-lo>=20]
    quiet=[(lo,hi) for lo,hi in spans(rate[:,0]<5) if hi-lo>=2]
    hm=np.zeros(len(rate),bool)
    for lo,hi in high:hm[lo:hi]=True
    events=[]
    for (_,lo),(hi,_) in zip(quiet[:-1],quiet[1:]):
        if hi-lo>=2 and not hm[lo:hi].any() and rate[lo:hi,0].max()>=20:
            peak=lo+int(np.argmax(rate[lo:hi,0]));events.append(dict(start_s=lo*.01,end_s=hi*.01,peak_s=(peak+.5)*.01))
    tr=r['tracker'];entries=tr['entries'];rec=tr['recoveries'];last=rate[-min(1000,len(rate)):]
    if not entries:mode='QUIET_ARREST' if last[:,0].mean()<1 else 'NO_HIGH_ENTRY'
    elif not rec:mode='NO_RECOVERY_AFTER_Z_REFILL' if tr['restore_s'] is not None else 'HIGH_WITHOUT_RECOVERY'
    elif rec[0]['mechanism']=='NATIVE':mode='NATIVE_RECURRENT_HIGH' if len(entries)>1 else 'NATIVE_RETURN'
    else:mode='Z_REFILL_RECURRENT_HIGH' if len(entries)>1 else 'Z_REFILL_RETURN_WITHOUT_REENTRY'
    before=sum(v['end_s']<=entries[0]['onset_s'] for v in events) if entries else len(events)
    return_origin=max(rec[0]['confirmation_s'],tr['release_s'] or 0.) if rec else None
    return_end=entries[1]['onset_s'] if len(entries)>1 else len(rate)*.01
    # State4 must precede state5. Events after a second high/return cannot
    # retrospectively satisfy the first post-return interictal interval.
    after=sum(v['start_s']>=return_origin and v['end_s']<=return_end for v in events) if rec else 0
    # Describe high-state local modulation without a fixed-frequency/20% rule.
    hd=[]
    for lo,hi in high:
        if hi-lo<100:continue
        v=np.c_[rate[lo:hi,0],regional[lo:hi,:3]]
        hf,px=welch(v,fs=100,nperseg=min(len(v),200),axis=0,detrend='linear')
        use=(hf>=1)&(hf<=45)
        hd.append(dict(window_s=[lo*.01,hi*.01],mean_Hz=v.mean(0),sd_Hz=v.std(0),
            peak_psd_Hz=hf[use][np.argmax(px[use],axis=0)],quantity_order=['All E','Core A','Core B','Other E']))
    dense_field_end=len(a['field_1ms'])
    assert np.array_equal(a['spikes_1ms'][:dense_field_end,0],a['field_1ms'].sum(1))
    if dense_field_end<len(a['spikes_1ms']):
        # Some inherited long continuations stored native spatial counts at5ms.
        # Validate their complete coverage without inventing1ms spatial samples.
        assert a.get('native_spatial_mixed_sampling',False)
        assert len(a['field_5ms'])*5==len(a['spikes_1ms'])
        assert np.array_equal(a['spikes_1ms'][:,0].reshape(-1,5).sum(1),a['field_5ms'].sum(1))
    assert np.array_equal(a['spikes_1ms'][:,0],a['regions_1ms'][:,:3].sum(1))
    expected=[(hi)*.01 for lo,hi in high]
    for entry in entries:
        lo=round(entry['onset_s']*100);hi=round(entry['confirmation_s']*100)
        assert hi-lo==20 and np.all(rate[lo:hi,0]>=200)
    return safe(dict(mode=mode,entries=entries,recoveries=rec,duration_s=len(rate)*.01,
        high_intervals_s=[[l*.01,h*.01] for l,h in high],events=events,
        finite_events_before_first_high=before,finite_events_after_return=after,
        postreturn_event_count_window_s=[return_origin,return_end] if rec else None,
        late_E_Hz=float(last[:,0].mean()),late_quiet_fraction=float((last[:,0]<5).mean()),
        full_1_to_5_observed=bool(len(entries)>=2 and rec and before>=2 and after>=2),
        recurrence_window_observed_s=(len(rate)*.01-max(rec[0]['confirmation_s'],tr['release_s'] or 0)) if rec else None,
        high_state_native_modulation=hd,sustained_oscillation_status='NOT_ESTABLISHED_BY_HIGH_RATE_THRESHOLD',
        human_review='PENDING'))


def snapshots(m,r):
    events=m['events'];entry=m['entries'];rec=m['recoveries'];end=m['duration_s']
    pre=[v for v in events if not entry or v['end_s']<entry[0]['onset_s']]
    t1=(pre[-1]['peak_s'] if entry and pre else pre[0]['peak_s'] if pre else min(.5,end/2))
    ts=[t1,None,None,None,None];labels=['Self-limited' if pre else 'Activity','Entry','High rate','Return','Second high']
    if entry:
        lo=entry[0]['onset_s'];ts[1]=max(.025,lo-.1)
        first=next(v for v in m['high_intervals_s'] if abs(v[0]-lo)<.011)
        ts[2]=min(lo+.5,first[1]-.025)
    if rec:
        origin=max(rec[0]['confirmation_s'],r['tracker']['release_s'] or 0)
        phase_end=entry[1]['onset_s'] if len(entry)>1 else end
        post=[v for v in events if v['start_s']>=origin and v['end_s']<=phase_end]
        if phase_end-origin>=.05:
            ts[3]=post[0]['peak_s'] if post else min(origin+.5,phase_end-.025)
        labels[3]='Native return' if rec[0]['mechanism']=='NATIVE' else 'After Z refill'
    if len(entry)>1:ts[4]=min(entry[1]['confirmation_s']+.25,end-.025)
    return [dict(number=i+1,time_s=None if t is None else float(np.clip(t,.025,max(.025,end-.025))),
                 label=labels[i],color=COLORS[i]) for i,t in enumerate(ts)]


def bandframes(x,fs):
    f,t,p=spectrogram(x,fs=fs,window='hann',nperseg=fs,noverlap=fs//2,
                       detrend='constant',scaling='density',axis=0)
    keep=(f>=1)&(f<=150)
    return np.maximum(p[keep].sum(0).T*(f[1]-f[0]),1e-20)


def early(a,m):
    if not m['entries']:return dict(status='NO_HIGH_ENTRY')
    onset=m['entries'][0]['onset_s'];baseline=[1.,min(30.,onset-2.)];target=[onset,onset+1.]
    if baseline[1]-baseline[0]<3 or target[1]>m['duration_s']:
        return dict(status='INSUFFICIENT_BASELINE_OR_TARGET',baseline_s=baseline,target_s=target)
    field_fs=10000 if 'early_field_0p1ms' in a else 1000
    spike_field=a['early_field_0p1ms'] if field_fs==10000 else a['field_1ms']
    if round(target[1]*field_fs)>len(spike_field):
        return dict(status='NO_NATIVE_1MS_FIELD_FOR_1_150HZ',baseline_s=baseline,target_s=target)
    raw=a['lfp_raw'].astype(float);raw-=raw.mean(1,keepdims=True)
    b=bandframes(raw[round(baseline[0]*2000):round(baseline[1]*2000)],2000)
    t=bandframes(raw[round(target[0]*2000):round(target[1]*2000)],2000)
    b=np.log10(b);t=np.log10(t);med=np.median(b,0);mad=1.4826*np.median(abs(b-med),0)
    z=np.full(len(mad),np.nan);ok=mad>1e-12;z[ok]=(t.mean(0)[ok]-med[ok])/mad[ok]
    native=spike_field[round(target[0]*field_fs):round(target[1]*field_fs)]/a['cell_e_counts']*field_fs
    power=bandframes(native,field_fs).mean(0)
    native_base=spike_field[round(baseline[0]*field_fs):round(baseline[1]*field_fs)]/a['cell_e_counts']*field_fs
    native_base_power=bandframes(native_base,field_fs)
    native_baseline_log=np.median(np.log10(native_base_power),axis=0)
    native_valid=native_baseline_log>np.log10(1e-18)
    native_db=np.full(len(power),np.nan)
    native_db[native_valid]=10*(np.log10(power[native_valid])-native_baseline_log[native_valid])
    power/=max(power.max(),1e-20)
    ref=read(CANON/'fig3_panelc_metadata.json');names=list(a['contact_names'])
    ids=[names.index(n) for n in ref['contact_order']];y=np.array(ref['raw_ictal_robust_z_mean'])
    valid=np.isfinite(z[ids])&np.isfinite(y)
    rho=float(spearmanr(z[ids][valid],y[valid]).statistic) if valid.sum()>=3 else None
    return safe(dict(status='MEASURED',baseline_s=baseline,target_s=target,contact_robust_z=z,
        native_normalized_bandpower=power,native_bandpower_change_db=native_db,
        native_spatial_bin_ms=1000/field_fs,
        native_observation='Spike-count power at the stated temporal resolution; not a biophysical LFP.',
        native_temporal_sampling='FULL_INTEGRATION_TIMESTEP' if field_fs==10000 else 'BINNED_COUNTS_WITH_POSSIBLE_HIGH_RATE_ALIASING',
        native_highres_source=a.get('early_field_source'),
        native_valid_baseline_cells=int(native_valid.sum()),native_cells_with_increased_power=int(np.sum(native_db>0)),
        baseline_frames=len(b),model_patient_rho=rho,n_contacts=int(valid.sum()),
        n_model_contacts_above_baseline=int(np.sum(z>0)),
        spatial_correlation_does_not_imply_energy_increase=True,
        model_band_Hz=[1,150],model_reference='CAR applied to original current proxy',
        patient_source=str(CANON/'figures/fig3-panelc.png'),patient_metadata=str(CANON/'fig3_panelc_metadata.json'),
        scope='Descriptive selected-seizure contact concordance; different biological observables and time windows, not a new patient fit or validation.'))


def edges(v):
    x=np.log(v);return np.exp(np.r_[x[0]-(x[1]-x[0])/2,(x[:-1]+x[1:])/2,x[-1]+(x[-1]-x[-2])/2])


def first_entry_from_counts(spikes_1ms):
    """Same all-E 200 Hz / 200 ms endpoint, independently read from counts."""
    assert len(spikes_1ms)%10==0
    rate=spikes_1ms.reshape(-1,10,2).sum(1)[:,0]/32000/.01
    intervals=[(lo,hi) for lo,hi in spans(rate>=200) if hi-lo>=20]
    if not intervals:return None
    lo=intervals[0][0]
    return dict(onset_s=lo*.01,confirmation_s=(lo+20)*.01)


def committed_first_endpoint(folder,horizon_s=180.):
    """A closed prefix may establish first entry before full follow-up ends.

    No entry in an unfinished prefix is pending, never right-censoring at180.
    Only original formal-grid runs call this; sibling branches add no samples.
    """
    parts=[];end=0;files=[];entry=None
    for path in sorted((folder/'chunks').glob('*.npz')):
        if '.tmp.' in path.name:continue
        with np.load(path) as a:
            assert int(a['start_step'])==end,path
            hi=int(a['end_step']);spikes=a['spikes_1ms'];regions=a['regions_1ms']
            assert len(spikes)*10==hi-end and len(regions)==len(spikes)
            assert np.array_equal(spikes[:,0],regions[:,:3].sum(1)),path
            assert np.array_equal(spikes[:,1],regions[:,3:].sum(1)),path
            parts.append(spikes);end=hi;files.append(str(path))
        counts=np.concatenate(parts)[:round(horizon_s*1000)]
        entry=first_entry_from_counts(counts)
        if entry is not None or end*.0001>=horizon_s:break
    end_s=end*.0001
    established=entry is not None or end_s>=horizon_s
    return dict(status='ESTABLISHED' if established else 'PENDING',
        observed=None if not established else entry is not None,
        onset_s=None if entry is None else entry['onset_s'],
        confirmation_s=None if entry is None else entry['confirmation_s'],
        restricted_time_s=None if not established else (horizon_s if entry is None else entry['confirmation_s']),
        closed_prefix_end_s=end_s,source_chunks=files,
        endpoint_verified_from_saved_spikes=True,statistical_unit='one original parameter condition and noise seed')


def grid_summary():
    times=np.full((4,5,2),np.nan);fractions=times.copy();rows=[];endpoint_rows=[]
    for job in read(OUT/'protocol.json')['jobs']:
        folder=OUT/'runs'/job['name'];f=folder/'result.json'
        endpoint=committed_first_endpoint(folder)
        e,t,s=job['eta_index'],job['tau_index'],SEEDS.index(job['seed'])
        if endpoint['status']=='ESTABLISHED':
            times[e,t,s]=endpoint['restricted_time_s'];fractions[e,t,s]=endpoint['observed']
        endpoint_rows.append(dict(name=job['name'],eta_M=job['eta_m'],tau_M_s=job['tau_M_s'],seed=job['seed'],
            full_trajectory_complete=f.exists(),**endpoint))
        if not f.exists():continue
        r=read(f);tr=r['tracker'];entry=tr['entries']
        observed=bool(entry and entry[0]['confirmation_s']<=180)
        assert endpoint['status']=='ESTABLISHED' and observed==endpoint['observed'],job['name']
        if observed:assert abs(entry[0]['confirmation_s']-endpoint['confirmation_s'])<1e-8
        rows.append(dict(name=job['name'],eta_M=job['eta_m'],tau_M_s=job['tau_M_s'],seed=job['seed'],
            first_entry_confirmation_s=entry[0]['confirmation_s'] if entry else None,
            observed_by_180s=observed,end_s=r['end_s'],entries=len(entry),
            recovery_mechanisms=[x['mechanism'] for x in tr['recoveries']],Z_refill_s=tr['restore_s']))
    count=np.isfinite(times).sum(-1);full=count==2;mean=np.full((4,5),np.nan);prob=mean.copy()
    mean[full]=times[full].mean(-1);prob[full]=fractions[full].mean(-1)
    return dict(mean=mean,prob=prob,count=count,rows=rows,endpoint_rows=endpoint_rows,
        established_first_endpoints=int(np.isfinite(times).sum()),
        F_scope='Only first-entry endpoint needs to be complete; native/external return and recurrence follow-up may still be running.')


def f_panel(fig,spec,g,job):
    ax=fig.add_subplot(spec);cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#ededed')
    xe,ye=edges(TAU),edges(ETA)
    im=ax.pcolormesh(xe,ye,np.ma.masked_invalid(g['mean']),cmap=cmap,vmin=0,vmax=180,
                    shading='flat',edgecolors='#ffffff55',linewidth=.4)
    ax.set(xscale='log',yscale='log',xticks=TAU,xticklabels=['1','2','4','8','20'],
        yticks=ETA,yticklabels=['.005','.01','.02','.04'],xlabel=r'$\tau_M$ (s)',ylabel=r'$\eta_M$')
    ax.minorticks_off();ax.set_title('F  M kinetics',loc='left',weight='bold')
    for y,x in zip(*np.where((g['count']==2)&(g['prob']<1))):
        ax.add_patch(Rectangle((xe[x],ye[y]),xe[x+1]-xe[x],ye[y+1]-ye[y],fc='none',ec='#555555',lw=0,hatch='///'))
    for y,x in zip(*np.where(g['count']==2)):
        value=float(g['mean'][y,x])
        entered=int(round(2*g['prob'][y,x]))
        ax.text(np.sqrt(xe[x]*xe[x+1]),np.sqrt(ye[y]*ye[y+1]),
                f'{value:.1f}\n{entered}/2',ha='center',va='center',fontsize=11,
                color='white' if value<90 else '#111111')
    xi=int(np.argmin(abs(np.asarray(TAU)-job['tau_M_s'])))
    yi=int(np.argmin(abs(np.asarray(ETA)-job['eta_m'])))
    ax.add_patch(Rectangle((xe[xi],ye[yi]),xe[xi+1]-xe[xi],ye[yi+1]-ye[yi],
                           fc='none',ec='#222222',lw=2))
    cb=fig.colorbar(im,ax=ax,pad=.035);cb.set_label('Restricted mean entry time (s)')
    ax.text(1,1.025,'Entered by 180 s: n/2',ha='right',transform=ax.transAxes,fontsize=11)
    return ax


def render(a,r,m,folder,g,stem='fig5',time_window=None):
    folder.mkdir(parents=True,exist_ok=True);job=r['job'];snaps=a.get('display_snapshots') or snapshots(m,r);en=early(a,m)
    zooms=a.get('AB_zoom_windows',[])
    fig=plt.figure(figsize=(28,24 if zooms else 20));outer=fig.add_gridspec(1,2,width_ratios=[1.6,1],
        left=.062,right=.945,top=.95,bottom=.06,wspace=.28)
    if zooms:
        left=outer[0].subgridspec(6,1,height_ratios=[1.7,1.4,1.9,1.2,.16,1.05],hspace=.38)
        c_index,header_index,map_index=3,4,5
    else:
        left=outer[0].subgridspec(5,1,height_ratios=[2,1.55,1.13,.16,1.05],hspace=.32)
        c_index,header_index,map_index=2,3,4
    right=outer[1].subgridspec(3,1,height_ratios=[1.1,1.45,.72],hspace=.42)
    end=m['duration_s'];limits=[0,end] if time_window is None else time_window
    names=list(a['contact_names']);ids=[names.index(n) for n in ORDER]
    lt=a['lfp_time_ms']/1000;raw=a['lfp_raw'][:,ids]
    filtered=sosfiltfilt(butter(4,[30,80],fs=2000,btype='bandpass',output='sos'),raw,axis=0)
    pre_end=min(30,m['entries'][0]['onset_s']-1) if m['entries'] else min(30,end)
    reference=(lt>=min(1,pre_end/4))&(lt<pre_end)
    # One constant per contact for the entire trace; retain relative stage
    # amplitudes while keeping adjacent contact slots visibly separate.
    scale=np.maximum(np.max(abs(filtered),axis=0),1e-12)
    ax=fig.add_subplot(left[0]);axes=[ax]
    # Keep peaks when compressing the long timeline: consecutive min/max pairs.
    stride=max(1,len(lt)//50000);keep=np.arange(0,len(lt),stride)
    for i in range(15):
        value=filtered[:,i]/scale[i]*.40
        if stride>1:
            n=len(value)//stride;block=value[:n*stride].reshape(n,stride)
            j=np.c_[np.argmin(block,1),np.argmax(block,1)]+np.arange(n)[:,None]*stride
            take=np.sort(j,axis=1).ravel()
        else:take=keep
        ax.plot(lt[take],value[take]+14-i,c='#285f72' if i<4 else '#95612f',lw=.65,rasterized=True)
    ax.set(yticks=np.arange(15)[::-1],yticklabels=ORDER,ylim=(-.7,15.2),ylabel='Virtual SEEG\n30–80 Hz (a.u.)')
    ax.set_title('A  Electrode readout',loc='left',weight='bold',pad=28)
    ax.set_title(f'ηM = {job["eta_m"]:g}, τM = {job["tau_M_s"]:g} s',
                 loc='right',pad=28,fontsize=14,weight='normal')
    ra=fig.add_subplot(left[1]);axes.append(ra)
    it,ix=np.where(a['raster']);tt=it*.0001
    for lo,hi,col in [(0,60,'#286889'),(60,80,'#b2753a')]:
        mask=(ix>=lo)&(ix<hi);ra.scatter(tt[mask],ix[mask],s=1.0,marker='.',c=col,lw=0,rasterized=True)
    for y in [19.5,39.5,59.5]:ra.axhline(y,c='#bbbbbb',lw=.6)
    ra.set(ylim=(-1,80),yticks=[10,30,50,70],yticklabels=['Core A E','Core B E','Other E','I'])
    ra.set_title('B  Continuous spike raster',loc='left',weight='bold')
    za=fig.add_subplot(left[c_index]);ma=za.twinx();ma.spines['right'].set_visible(True);axes.append(za)
    zt=a['slow_time_ms']/1000;z=a['Z'];mm=a['M']*job['eta_m']
    za.fill_between(zt,z[:,2],z[:,4],color=REGCOL[0],alpha=.13,lw=0)
    for zi,mi,col in [(0,0,REGCOL[0]),(5,1,REGCOL[1]),(6,2,REGCOL[2])]:
        za.plot(zt,z[:,zi],c=col,lw=1.2);ma.plot(zt,mm[:,mi],c=col,ls='--',lw=1.2)
    za.set(ylabel='Resource Z',ylim=(0,1.05),xlabel='Time, t (s)');ma.set_ylabel('ηM × M (mV equiv.)',labelpad=10)
    if job['eta_m']==0:
        ma.set(ylim=(0,1),yticks=[0])
    za.set_title('C  Inhibition resource and adaptation',loc='left',weight='bold')
    handles=[Line2D([],[],color=c,label=n) for c,n in zip(REGCOL,['All E','Core A','Core B'])]
    handles += [Line2D([],[],color='black',label='Z'),Line2D([],[],color='black',ls='--',label='ηM × M')]
    header=fig.add_subplot(left[header_index]);header.axis('off')
    header.legend(handles=handles,ncol=5,loc='upper center',bbox_to_anchor=(.5,1.95),frameon=False,fontsize=12)
    header.text(0,-.2,'D  Spatial activity · 50 ms',weight='bold',fontsize=18)
    restore=r['tracker']['restore_s'];release=r['tracker']['release_s']
    for axis in axes:
        axis.set_xlim(*limits)
        for l,h in m['high_intervals_s']:axis.axvspan(l,h,color='#b42c40',alpha=.11,lw=0)
        if restore is not None:
            axis.axvspan(restore,release,color='#258c77',alpha=.15,lw=0)
            axis.axvline(restore,c='#258c77',ls='--',lw=1);axis.axvline(release,c='#258c77',ls='--',lw=1)
        for v in snaps:
            if v['time_s'] is not None:axis.axvline(v['time_s'],c=v['color'],ls=':',lw=.9)
        if axis is not za:axis.tick_params(labelbottom=False)
    if zooms:
        ra.tick_params(labelbottom=True)
        ra.set_xlabel('Time, t (s)')
    if restore is not None:
        if limits[0]<=restore<=limits[1]:
            ax.annotate('Refill Z',xy=(restore,1.025),xycoords=ax.get_xaxis_transform(),xytext=(-10,0),textcoords='offset points',ha='right',color='#258c77',fontsize=13)
        if limits[0]<=release<=limits[1]:
            ax.annotate('Release Z',xy=(release,1.025),xycoords=ax.get_xaxis_transform(),xytext=(10,0),textcoords='offset points',ha='left',color='#258c77',fontsize=13)
    previous_label_time=-np.inf;label_row=0
    for v in snaps:
        if v['time_s'] is not None and limits[0]<=v['time_s']<=limits[1]:
            if a.get('stagger_close_stage_labels',False):
                label_row=(label_row+1)%3 if v['time_s']-previous_label_time < .02*(limits[1]-limits[0]) else 0
            ax.text(v['time_s'],.99-.09*label_row,str(v['number']),transform=ax.get_xaxis_transform(),ha='center',va='top',weight='bold',color=v['color'],bbox=dict(fc='white',ec='none',alpha=.8,pad=.3))
            previous_label_time=v['time_s']
    if zooms:
        # Same filtered samples, fixed whole-trace scales and sampled neurons.
        # The overview above remains continuous; these are magnifications only.
        detail=left[2].subgridspec(1,len(zooms),wspace=.35)
        detail_axes=[]
        subset=[0,3,4,14]
        for column,window in enumerate(zooms):
            sub=detail[column].subgridspec(2,1,height_ratios=[1,1.1],hspace=.08)
            ua=fig.add_subplot(sub[0]);ub=fig.add_subplot(sub[1],sharex=ua)
            detail_axes.extend([ua,ub])
            lo,hi=window['window_s'];use=(lt>=lo)&(lt<=hi)
            for row,j in enumerate(subset):
                ua.plot(lt[use],filtered[use,j]/scale[j]*.40+3-row,
                    c='#285f72' if j<4 else '#95612f',lw=.8)
            ua.set(yticks=[3,2,1,0],yticklabels=[ORDER[j] for j in subset],ylim=(-.6,3.7))
            if column==0:ua.set_ylabel('30–80 Hz\n(a.u.)',fontsize=13)
            ua.set_title(window['label'],fontsize=17,loc='left');ua.tick_params(labelbottom=False,labelsize=13)
            use=(tt>=lo)&(tt<=hi)
            for low,high,col in [(0,60,'#286889'),(60,80,'#b2753a')]:
                take=use&(ix>=low)&(ix<high)
                ub.scatter(tt[take],ix[take],s=5,c=col,marker='|',lw=.5,rasterized=True)
            for y in [19.5,39.5,59.5]:ub.axhline(y,c='#bbbbbb',lw=.5)
            ub.set(ylim=(-1,80),yticks=[10,30,50,70],yticklabels=['Core A','Core B','Other E','I'],
                   xlim=(lo,hi),xlabel='Time, t (s)')
            ub.xaxis.set_major_locator(MaxNLocator(nbins=4));ub.tick_params(labelsize=13)
            for axis in [ua,ub]:
                for l,h in m['high_intervals_s']:
                    if l<hi and h>lo:axis.axvspan(l,h,color='#b42c40',alpha=.11,lw=0)
            for overview in [ax,ra]:
                overview.axvspan(lo,hi,fc=window['color'],alpha=.055,lw=0)
                overview.axvline(lo,c=window['color'],lw=.55,ls='--')
                overview.axvline(hi,c=window['color'],lw=.55,ls='--')
        a['AB_zoom_observer']=dict(windows=zooms,contacts=[ORDER[j] for j in subset],
            same_filter_and_full_trace_scales=True,same_raster_neurons=True,overview_continuous=True)
    maps=left[map_index].subgridspec(1,5,wspace=.26)
    for i,v in enumerate(snaps):
        q=fig.add_subplot(maps[i]);q.set(xlim=(0,20),ylim=(0,20),xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)')
        if i==0:q.set_ylabel('y (mm)')
        else:q.set_yticklabels([])
        if v['time_s'] is None:
            q.set_facecolor('#f0f0f0');q.text(.5,.5,'Not observed',ha='center',transform=q.transAxes,fontsize=12)
            q.set_title(f'{v["number"]}  {v["label"]}',fontsize=13,color=v['color']);continue
        lo=round(v['time_s']*1000)-25
        if lo+50<=len(a['field_1ms']):
            counts=a['field_1ms'][lo:lo+50].sum(0);v['native_spatial_bin_ms']=1
        else:
            assert a.get('native_spatial_mixed_sampling',False) and lo%5==0
            counts=a['field_5ms'][lo//5:(lo+50)//5].sum(0);v['native_spatial_bin_ms']=5
        field=counts/a['cell_e_counts']/.05
        im=q.imshow(field.reshape(20,20),extent=[0,20,0,20],origin='lower',cmap='magma',norm=PowerNorm(.6,0,500),interpolation='nearest')
        for xy in a['centers_mm']:q.add_patch(Circle(xy,float(a['core_radius_mm']),ec='#2dd4cd',fc='none',lw=1.4))
        q.set_title(f'{v["number"]}  {v["label"]}\n{v["time_s"]:.2f} s',fontsize=13,color=v['color'])
        if i==0:
            ca=q.inset_axes([0,-.48,1,.06]);cb=fig.colorbar(im,cax=ca,orientation='horizontal',ticks=[0,250,500]);cb.set_label('E rate (Hz)',fontsize=12);ca.tick_params(labelsize=11)
    # Projected native trajectory, never called an autonomous vector field.
    trax=fig.add_subplot(right[0],projection='3d');trax.computed_zorder=False
    er=a['spikes_1ms'][:,0].reshape(-1,5).sum(1)/32000/.005;t=(np.arange(len(er))+.5)*.005
    xyz=np.c_[np.interp(t,zt,z[:,0]),np.interp(t,zt,a['currents'][:,2]),gaussian_filter1d(er,1)]
    key_times=[v['time_s'] for v in snaps if v['time_s'] is not None]
    trajectory_window=[max(0,min(key_times)-2),min(end,max(key_times)+5)]
    if not m['entries']:trajectory_window=[0,end]
    elif not m['recoveries']:trajectory_window[1]=end
    trajectory_window=a.get('trajectory_time_window_s',trajectory_window)
    visible=(t>=trajectory_window[0])&(t<=trajectory_window[1])
    view_t=t[visible];view_xyz=xyz[visible]
    distance=np.linalg.norm(np.diff(view_xyz/np.array([.5,1000,500]),axis=0),axis=1)
    cumulative=np.r_[0,np.cumsum(distance)];keep=np.r_[0,np.flatnonzero(np.diff(np.floor(cumulative/.035))>0)+1,len(view_t)-1]
    # Keep dense points through the forced return to expose the Z-directed path.
    if restore is not None:keep=np.union1d(keep,np.flatnonzero((view_t>=restore)&(view_t<=release)))
    keep=np.unique(keep);pts=view_xyz[keep];ts=view_t[keep];norm=Normalize(*trajectory_window);cmap=plt.get_cmap('viridis')
    if restore is not None:
        sel=(ts>=restore)&(ts<=release);trax.plot(*pts[sel].T,c='#555d65',lw=4,alpha=.7,zorder=5)
    trax.add_collection3d(Line3DCollection(np.stack([pts[:-1],pts[1:]],axis=1),colors=cmap(norm((ts[:-1]+ts[1:])/2)),linewidths=1.1,zorder=6))
    ymax=max(1,float(xyz[:,1].max()));zmax=max(30,float(xyz[:,2].max()))
    trax.set(xlim=(max(0,xyz[:,0].min()-.03),1.025),ylim=(-.03*ymax,1.05*ymax),zlim=(-.03*zmax,1.06*zmax),
        xlabel='Mean Z',ylabel=r'$H_E$ (mV equiv.)',zlabel=r'E rate, $r_E$ (Hz)')
    trax.view_init(elev=26,azim=-125);trax.set_box_aspect((1.25,1,1.05));trax.grid(True)
    for axis in [trax.xaxis,trax.yaxis,trax.zaxis]:axis.labelpad=11;axis.label.set_fontsize(15);axis.pane.set_facecolor('#f2f5fa')
    for axis in [trax.xaxis,trax.yaxis,trax.zaxis]:axis.set_major_locator(MaxNLocator(nbins=4))
    trax.tick_params(labelsize=12);trax.set_title('E1  State trajectory',loc='left',weight='bold')
    separate_labels=a.get('separate_trajectory_labels',False)
    if separate_labels:
        fig.canvas.draw()
    occupied_labels=[]
    for v in snaps:
        if v['time_s'] is None:continue
        pt=np.array([np.interp(v['time_s'],t,xyz[:,k]) for k in range(3)])
        trax.scatter(*pt,s=27,c=v['color'],edgecolors='white',depthshade=False,zorder=10)
        x,y,_=proj3d.proj_transform(*pt,trax.get_proj())
        offset=(8,8)
        if separate_labels:
            anchor=trax.transData.transform((x,y));unit=fig.dpi/72
            choices=[(12,12),(-20,12),(20,-20),(-20,-20),(0,34),(34,0),(-34,0),(0,-34),
                     (40,30),(-40,30),(40,-30),(-40,-30)]
            for candidate in choices:
                center=anchor+np.array(candidate)*unit
                box=(center[0]-12*unit,center[1]-12*unit,center[0]+12*unit,center[1]+12*unit)
                if all(box[2]<b[0] or box[0]>b[2] or box[3]<b[1] or box[1]>b[3] for b in occupied_labels):
                    offset=candidate;occupied_labels.append(box);break
            else:
                raise RuntimeError('Could not separate trajectory state labels')
        trax.annotate(str(v['number']),xy=(x,y),xytext=offset,textcoords='offset points',weight='bold',
            ha='center' if separate_labels else 'left',va='center' if separate_labels else 'baseline',fontsize=13,
            bbox=dict(boxstyle='circle,pad=.2',fc='white',ec='#777777'),
            arrowprops=dict(arrowstyle='-',color='#666666',lw=.7) if offset!=(8,8) else None,zorder=20)
    ca=trax.inset_axes([1.04,.15,.03,.67]);cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=ca);cb.set_label('Time (s)',labelpad=8)
    if callable(a.get('E2_renderer')):
        en=a['E2_renderer'](fig,right[1],a,m,en)
        refpath=CANON/'figures/fig3-panelc.png'
    else:
        # Measured native model maps plus the actual frozen Fig3C image.
        eg=right[1].subgridspec(3,1,height_ratios=[.09,1,1.25],hspace=.52)
        h=fig.add_subplot(eg[0]);h.axis('off');h.text(0,.2,'E2  Early energy · 1–150 Hz',weight='bold',fontsize=18)
        mg=eg[1].subgridspec(1,2,wspace=.48)
        for i in range(2):
            q=fig.add_subplot(mg[i]);q.set(xlim=(0,20),ylim=(0,20),xticks=[0,10,20],yticks=[0,10,20],xlabel='x (model mm)')
            if i==0:q.set_ylabel('y (model mm)')
            else:q.set_yticklabels([])
            if en['status']!='MEASURED':
                q.set_facecolor('#f0f0f0');q.text(.5,.5,'No early-high map',transform=q.transAxes,ha='center',fontsize=12);continue
            xy=a['contact_xy']
            if i==0:
                values=np.array(en['native_bandpower_change_db'],float)
                # A field with zero median baseline power has no estimable fold
                # change. Never turn the numerical PSD floor into enhancement.
                norm_native=TwoSlopeNorm(vmin=-20,vcenter=0,vmax=20)
                im=q.imshow(np.ma.masked_invalid(values.reshape(20,20)),origin='lower',extent=[0,20,0,20],cmap='RdBu',norm=norm_native,interpolation='nearest')
                title=f'Native spikes · {en.get("native_spatial_bin_ms",1):g} ms';quantity='Power change (dB)'
            else:
                values=np.array(en['contact_robust_z'],float);valid=np.isfinite(values)
                if not valid.any():
                    q.set_facecolor('#eeeeee');q.set_title('Model contact readout',fontsize=12)
                    q.text(.5,.5,'Undefined baseline z',transform=q.transAxes,ha='center',fontsize=12)
                    continue
                grid=np.linspace(0,20,151);xx,yy=np.meshgrid(grid,grid)
                weights=np.exp(-((xx[...,None]-xy[valid,0])**2+(yy[...,None]-xy[valid,1])**2)/(2*2.5**2))
                support=weights.sum(-1);field=(weights*values[valid]).sum(-1)/np.maximum(support,1e-12)
                limit=max(1.,float(np.ceil(np.max(np.abs(values[valid])))))
                norm2=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
                im=q.imshow(field,origin='lower',extent=[0,20,0,20],cmap='RdBu',norm=norm2,alpha=np.clip(support/(.3*support.max()),0,1),interpolation='nearest')
                q.scatter(xy[:,0],xy[:,1],c=values,cmap='RdBu',norm=norm2,s=22,edgecolors='#222222',lw=.5)
                title='Model contact readout';quantity='Log-power robust z'
            for center in a['centers_mm']:q.add_patch(Circle(center,float(a['core_radius_mm']),fc='none',ec='#21c9c1' if i==0 else '#d77937',lw=1.1))
            q.set_title(title+'\n'+f'{en["target_s"][0]:.2f}–{en["target_s"][1]:.2f} s',fontsize=12)
            ca=q.inset_axes([1.04,0,.045,1]);cb=fig.colorbar(im,cax=ca,extend='both' if i==0 else 'neither');cb.set_label(quantity,fontsize=12);ca.tick_params(labelsize=11)
        refpath=CANON/'figures/fig3-panelc.png';refmeta=read(CANON/'fig3_panelc_metadata.json')
        assert refmeta['seizure_idx']==2 and refmeta['ictal_extraction']['clinical_window_sec']==[0.,10.]
        refax=fig.add_subplot(eg[2]);refax.imshow(plt.imread(refpath),interpolation='none');refax.axis('off')
        refax.set_title('Fig. 3C · E1146 / SZ3 · onset +0–10 s',fontsize=14,pad=4)
    if callable(a.get('F_renderer')):a['F_renderer'](fig,right[2],g,job)
    else:f_panel(fig,right[2],g,job)
    fig.canvas.draw();bounds=np.array([[x.get_position().x0,x.get_position().x1] for x in axes])
    assert np.allclose(bounds,bounds[0])
    if zooms:
        box=Bbox.union([q.get_tightbbox(fig.canvas.get_renderer()) for q in detail_axes])
        box=box.transformed(fig.dpi_scale_trans.inverted()).padded(.08)
        fig.savefig(folder/(stem+'_AB_magnified.png'),dpi=180,bbox_inches=box)
        fig.savefig(folder/(stem+'_AB_magnified.pdf'),bbox_inches=box)
    fig.savefig(folder/(stem+'.png'),dpi=145,bbox_inches='tight');fig.savefig(folder/(stem+'.pdf'),bbox_inches='tight');plt.close(fig)
    metadata=safe(dict(job=job,source_duration_s=end,display_time_window_s=limits,metrics=m,snapshots=snaps,E2=en,
        readout=dict(source='Original |AMPA|+|GABA| current proxy, before Z/M application',band_Hz=[30,80],scale_reference='Per-contact full-trace absolute maximum; one unchanged scale for all stages',scales=scale,
                     is_biophysical_SEEG=False,filter_does_not_establish_oscillation=True),
        native_spatial_sampling_ms=1 if not a.get('native_spatial_mixed_sampling',False) else None,
        native_spatial_sampling_segments=a.get('native_spatial_sampling_segments',[[0,end,1]]),
        raster_fixed_neurons=80,ABC_time_alignment_checked=True,
        E2_display=a.get('E2_display',dict(native='1–150 Hz power change in dB from same model baseline; unestimable zero-power baseline cells masked',
            native_spatial_bin_ms=en.get('native_spatial_bin_ms'),
            native_display_limits_db=[-20,20],contact='Original CAR robust z; zero-centered diverging scale',
            colors='Blue means increase, red means decrease; fixed patient Fig3C left untouched',
            primary_contact_endpoint_changed=False)),
        E1='Actual Z / applied GABA / E rate projection; no closed vector-field or bifurcation inference.',E1_time_window_s=trajectory_window,
        producer=str(Path(__file__).resolve()),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        patient_reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),
        display_contract=a.get('display_contract'),E2_semantics=a.get('E2_semantics'),AB_zoom=a.get('AB_zoom_observer'),
        F_semantics=a.get('F_semantics'),
        complete_F_cells=a.get('F_complete_cells',int((g['count']==2).sum())),full_figure_acceptance='PENDING_SCIENTIFIC_AND_USER_REVIEW',agent_visual_review='PENDING'))
    write(folder.parent/(stem+'_metadata.json'),metadata)
    readme=folder/'README.md'
    entry=f'### {stem}.png / .pdf\n'
    content=entry+(
        '同一条连续原生SNN轨迹的电极读出、固定80神经元raster、双轴Z/M、对应50毫秒原生二维场和三维轨迹。'
        'E2给出实际模型早期读出与原生功率场，并原样引用正式Fig3C；F为本轮M参数扫描，灰格尚未完成。\n'
        '**关注点**：未观测阶段明确留空，人工Z补充与原生返回分开标注；带通波包不证明持续振荡，等待目视验收。\n')
    previous=readme.read_text() if readme.exists() else ''
    if entry not in previous:readme.write_text(previous+'\n'+content)
    # Preserve the full observation horizon, with an aligned close view so
    # stages separated by milliseconds remain readable on long trajectories.
    if stem=='fig5' and m['entries'] and trajectory_window[1]-trajectory_window[0]<.8*end:
        render(a,r,m,folder,g,stem='fig5_transition_zoom',time_window=trajectory_window)


def update(final=False):
    g=grid_summary();records=[]
    analysis=OUT/'analysis';analysis.mkdir(exist_ok=True)
    for row in g['rows']:
        folder=OUT/'runs'/row['name'];r=read(folder/'result.json');dest=OUT/'candidates'/row['name']
        meta=dest/'fig5_metadata.json'
        if meta.exists() and not final:m=read(meta)['metrics']
        else:
            a=load(folder);m=analyze(a,r);render(a,r,m,dest/'figures',g);del a
        energy=read(meta)['E2']
        records.append(dict(**row,mode=m['mode'],full_1_to_5_observed=m['full_1_to_5_observed'],
            E2_status=energy['status'],E2_contacts_above_baseline=energy.get('n_model_contacts_above_baseline'),
            E2_model_patient_rho=energy.get('model_patient_rho'),figure=str(dest/'figures/fig5.png')))
    selected={}
    for mode in sorted({r['mode'] for r in records}):
        candidates=[r for r in records if r['mode']==mode]
        pick=min(candidates,key=lambda r:(abs(np.log(r['eta_M']/.02))+abs(np.log(r['tau_M_s']/2)),r['seed'],r['name']))
        selected[mode]=pick
        if final:
            folder=OUT/'runs'/pick['name'];a=load(folder);r=read(folder/'result.json');m=analyze(a,r)
            render(a,r,m,OUT/'mode_gallery'/mode/'figures',g);del a
    if records:
        with (OUT/'mode_results.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    write(OUT/'analysis_summary.json',safe(dict(completed=len(records),total=40,records=records,mode_representatives=selected,
        complete_F_cells=int((g['count']==2).sum()),restricted_mean_first_entry_s=g['mean'],entry_fraction_180s=g['prob'],
        established_first_endpoints=g['established_first_endpoints'],first_endpoint_records=g['endpoint_rows'],
        F_scope=g['F_scope'],
        status='COMPLETE_PENDING_REVIEW' if len(records)==40 else 'PARTIAL',human_review='PENDING')))
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));xe,ye=edges(TAU),edges(ETA)
    for ax,values,label,limit in zip(axes,[g['mean'],g['prob']],['Restricted entry time (s)','Entry fraction by 180 s'],[180,1]):
        cm=plt.get_cmap('viridis').copy();cm.set_bad('#eeeeee')
        im=ax.pcolormesh(xe,ye,np.ma.masked_invalid(values),vmin=0,vmax=limit,cmap=cm,edgecolors='white',linewidth=.3)
        ax.set(xscale='log',yscale='log',xticks=TAU,xticklabels=[str(x) for x in TAU],yticks=ETA,yticklabels=[str(x) for x in ETA],xlabel='τM (s)',ylabel='ηM')
        ax.minorticks_off();fig.colorbar(im,ax=ax).set_label(label)
        for y,x in zip(*np.where(g['count']==2)):
            value=float(values[y,x])
            ax.text(np.sqrt(xe[x]*xe[x+1]),np.sqrt(ye[y]*ye[y+1]),
                    f'{value:.1f}' if limit==180 else f'{round(value*2):d}/2',
                    ha='center',va='center',fontsize=12,
                    color='white' if value<limit/2 else '#111111')
    fig.tight_layout();gf=OUT/'figures';gf.mkdir(exist_ok=True)
    fig.savefig(gf/'M_entry_time_and_fraction.png',dpi=170);fig.savefig(gf/'M_entry_time_and_fraction.pdf');plt.close(fig)
    (gf/'README.md').write_text('### M_entry_time_and_fraction.png / .pdf\n固定手放双核和Z参数，显示M强度与时间常数对首次进入时间及180秒内进入比例的影响。每格两个配对噪声种子，首次终点从已闭合原始放电计数独立核验；后续返回/再进入可以仍在运行，灰色表示该格首次终点尚未齐备。\n**关注点**：未跑满180秒且未进入的条件继续留待定，不能提前填为删失或永久稳定；待目视验收。\n')
    lines=['# M参数组合：模式与完整Fig5候选','',f'完整轨迹 {len(records)}/40 条；首次终点已确定 {g["established_first_endpoints"]}/40；完整F格点 {int((g["count"]==2).sum())}/20。','',
        '每条轨迹连续携带Z、M、膜、突触、延迟和随机历史。原生恢复与60秒未恢复后的一次Z人工补充分别记录；从不清零M。',
        '模式依据实际进入、恢复和再次进入，不把高率平台直接命名为持续振荡。每个完成条件都有完整布局；未观测到的①–⑤阶段不能用普通事件冒充。','',
        '| 模式 | 代表条件 | 完整①–⑤ | 早期接触功率高于基线数 | 图 |','|---|---|---|---|---|']
    for mode,r in selected.items():lines.append(f'| {mode} | ηM={r["eta_M"]}, τM={r["tau_M_s"]}s, seed={r["seed"]} | {r["full_1_to_5_observed"]} | {r["E2_contacts_above_baseline"]} | [Fig5]({r["figure"]}) |')
    lines+=['','代表选择只按距原M工作点的参数距离和固定噪声顺序，不按患者匹配挑选。正式患者参考仍为Fig3C的E1146/SZ3；模型接触能量对应关系独立计算，尚不能以图像相似宣布恢复。',
        '所有图均为候选，最终仍需原完整Fig5的科学和人工验收。']
    (OUT/'scientific_review.md').write_text('\n'.join(lines)+'\n')


def legacy_preview():
    """Use an existing real 240-s trajectory only to inspect the new layout."""
    src=ROOT/'results/topic4_sef_hfo/fig5_manual_core_release_v1/weak_fast_z_refill_recurrence_v2/runs/weak_fast_z_refill_recurrence'
    with np.load(src.with_suffix('.npz')) as f, np.load(OUT/'geometry.npz') as geo:
        dt=float(f['dt_ms']);a=dict(time_ms=np.arange(len(f['field_e_count_1ms']))+.5,
            spikes_1ms=np.rint(np.c_[f['rate_e_hz'].reshape(-1,10).mean(1)*32,f['rate_i_hz'].reshape(-1,10).mean(1)*8]).astype(np.uint16),
            regions_1ms=f['region_spikes_1ms'],field_1ms=f['field_e_count_1ms'],
            raster=f['sample_spikes'][:,geo['sample_source_indices']],slow_time_ms=f['z_time_ms'],Z=f['z_stats'][:,:9],
            M=f['m_stats'][:,[0,5,6,7]],currents=f['currents_5ms'][:,:3],lfp_time_ms=f['lfp_time_ms'],lfp_raw=f['lfp_raw'])
        a.update({k:geo[k] for k in geo.files})
    tr=dict(entries=[dict(onset_s=73.48,confirmation_s=73.68)],recoveries=[dict(start_s=76.1,confirmation_s=78.1,mechanism='EXTERNAL_Z')],restore_s=75.5,release_s=76.5)
    r=dict(job=dict(name='existing_Z_only_240s_layout_qa',eta_m=.02,tau_M_s=2,seed=9108401),tracker=tr)
    m=analyze(a,r);render(a,r,m,OUT/'layout_qa/figures',grid_summary(),stem='existing_trajectory_full_layout')
    meta=read(OUT/'layout_qa/existing_trajectory_full_layout_metadata.json')
    render(a,r,m,OUT/'layout_qa/figures',grid_summary(),stem='existing_trajectory_transition_zoom',time_window=meta['E1_time_window_s'])
    write(OUT/'layout_qa/source.json',dict(source=str(src.with_suffix('.npz')),new_search_result=False,
        note='Existing observed trajectory used for renderer QA; earlier Z refill at75.5s, not new60s-native-return protocol.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--update',action='store_true');p.add_argument('--final',action='store_true');p.add_argument('--legacy-preview',action='store_true');a=p.parse_args()
    if a.legacy_preview:legacy_preview()
    else:update(a.final)
