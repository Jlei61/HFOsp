#!/usr/bin/env python3
"""Render native-SNN observations without inventing a closed E-I-Z system."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import PowerNorm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import binned_statistic_2d

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/fig5_manual_core_release_v1'
FIG=OUT/'figures'
COL=['#487b9e','#b3a37a','#bd6c48','#a93659','#2e8c78','#895698']
E_COL='#357ca8';I_COL='#db873c';Z_COL='#7c418c'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                     'axes.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})


class TrajectoryArrow3D(FancyArrowPatch):
    """Point-sized heads after projection; Hz and dimensionless Z must not mix."""
    def __init__(self,start,end,color):
        super().__init__((0,0),(0,0),arrowstyle='-|>',mutation_scale=8,lw=.75,color=color)
        self.xyz=np.column_stack([start,end])
    def do_3d_projection(self,renderer=None):
        x,y,z=proj3d.proj_transform(*self.xyz,self.axes.get_proj())
        self.set_positions((x[0],y[0]),(x[1],y[1]))
        return float(np.min(z))


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def load_main():
    a=np.load(OUT/'runs/continuous_refill_release.npz')
    r=read(OUT/'runs/continuous_refill_release.json')
    e=a['rate_e_hz'].reshape(-1,50).mean(1);i=a['rate_i_hz'].reshape(-1,50).mean(1)
    t=(np.arange(len(e))+.5)*.005
    z=np.interp(t,a['z_time_ms']/1000,a['z_stats'][:,0])
    return a,r,t,e,i,z


def windows(t,e,r):
    # Selection uses only timing and population activity, never a desired spatial route.
    onset=r['first_trigger_ms']/1000;release=r['release_ms']/1000
    idx=np.flatnonzero((t>1)&(t<min(4,onset-1)))
    smooth=gaussian_filter1d(e,1)
    pp,_=find_peaks(smooth[idx],prominence=20,distance=30)
    event=idx[pp[0]] if len(pp) else idx[np.argmax(smooth[idx])]
    qidx=np.flatnonzero((t>t[event]+.16)&(t<min(t[event]+1,onset-1)))
    qs=gaussian_filter1d(e,8)
    quiet=qidx[np.argmin(qs[qidx])]
    after=np.flatnonzero((t>release+.5)&(t<min(release+3,t[-1]-.2)))
    pp,_=find_peaks(smooth[after],prominence=15,distance=30)
    returned=after[pp[0]] if len(pp) else after[np.argmax(smooth[after])]
    late=r['duration_ms']/1000-.5
    high_start=onset-.2
    last_quiet=np.flatnonzero((t>=onset-2)&(t<high_start)&(e<1.))
    quiet_end=float(t[last_quiet[-1]]+.0025) if len(last_quiet) else onset-.6
    entry=.5*(quiet_end+high_start)
    result=[dict(time=float(tm),label=label,color=co,halfwidth=.125) for tm,label,co in zip(
        [t[event],t[quiet],entry,onset+.3,t[returned],late],
        ['Self-limited event','Quiet interval','High-state entry','Sustained high','After release','Later native Z'],COL)]
    for w in result:
        lo=int(round((w['time']-.025)*1000))
        w['spatial_window_ms']=[lo,lo+50]
        w['spatial_center_s']=(lo+25)/1000
    return result


def decorate_time(ax,r,win,end):
    if r['restore_start_ms'] is not None:
        lo=r['restore_start_ms']/1000;hi=r['release_ms']/1000
        ax.axvspan(lo,hi,color='#2e8c78',alpha=.13,lw=0)
        ax.axvline(hi,c='#2e8c78',lw=.8,ls='--')
    ax.set_xlim(0,end)
    for j,w in enumerate(win):
        ax.axvspan(w['time']-w['halfwidth'],w['time']+w['halfwidth'],color=w['color'],alpha=.1,lw=0)


def choose_contacts(a,n=8):
    valid=np.flatnonzero(a['valid_contacts'])
    ids=[]
    for shaft in np.unique(a['shaft_ids'][valid]):
        group=valid[a['shaft_ids'][valid]==shaft]
        ids.extend(group[np.unique(np.round(np.linspace(0,len(group)-1,n//2)).astype(int))].tolist())
    return np.array(ids[:n],int)


def left_panels(fig,subspec,a,r,t,e,i,z,win,letters=True):
    gs=subspec.subgridspec(4,1,height_ratios=[1.25,.85,1.05,.85],hspace=.45)
    ax=fig.add_subplot(gs[0]);lt=a['lfp_time_ms']/1000;contacts=choose_contacts(a)
    raw=a['lfp_effective'][:,contacts]
    base=np.median(raw[(lt>=.5)&(lt<1)],axis=0);x=raw-base
    scale=float(np.max(np.quantile(x,.995,axis=0)-np.quantile(x,.005,axis=0)))
    scale=max(scale,1e-12)
    offsets=np.arange(len(contacts))[::-1]
    for k,j in enumerate(contacts):
        shaft=str(a['shaft_ids'][j]);color='#d27943' if shaft in ('ICL','0','A') else '#27929d'
        ax.plot(lt,x[:,k]/scale*.82+offsets[k],lw=.4,c=color,rasterized=True)
    ax.set_yticks(offsets);ax.set_yticklabels(a['contact_names'][contacts],fontsize=8)
    ax.set_ylim(-.25,len(contacts)+.25);ax.set_ylabel('Virtual SEEG\ncurrent proxy (a.u.)')
    ax.set_title(('A  ' if letters else '')+'Unfiltered contact readout',loc='left',fontweight='bold')
    decorate_time(ax,r,win,r['duration_ms']/1000);ax.set_xlabel('Time (s)')
    for j,w in enumerate(win):
        ax.text(w['time'],.97,str(j+1),transform=ax.get_xaxis_transform(),ha='center',va='top',color=w['color'],weight='bold',bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.4))
    start=r['restore_start_ms']/1000;rel=r['release_ms']/1000
    ax.annotate('Refill Z',xy=((start+rel)/2,.99),xycoords=('data','axes fraction'),xytext=(0,16),textcoords='offset points',ha='center',color='#2e8c78',fontsize=9)
    ax.annotate('Release: native Z resumes',xy=(rel,.98),xycoords=('data','axes fraction'),xytext=(25,4),textcoords='offset points',color='#2e8c78',fontsize=9)
    rastergs=gs[1].subgridspec(1,len(win),wspace=.12)
    selected=np.r_[np.arange(0,60,3),np.arange(60,120,3),np.arange(120,240,6),np.arange(240,300,3)]
    for j,w in enumerate(win):
        ra=fig.add_subplot(rastergs[j]);lo=w['time']-.125;hi=w['time']+.125
        k0=int(round(lo*10000));k1=int(round(hi*10000))
        st,sn=np.where(a['sample_spikes'][k0:k1,selected])
        ra.scatter(lo+st*.0001,sn,s=.6,c=np.where(selected[sn]<240,E_COL,I_COL),rasterized=True,linewidths=0)
        ra.axvspan(w['spatial_window_ms'][0]/1000,w['spatial_window_ms'][1]/1000,color=w['color'],alpha=.1,lw=0)
        ra.set(xlim=(lo,hi),ylim=(-1,80),xticks=[w['time']],xlabel=f'{w["time"]:.2f} s')
        ra.set_xticklabels([]);ra.tick_params(axis='x',length=0)
        for y in (19.5,39.5,59.5):ra.axhline(y,c='#d6d6d6',lw=.5)
        ra.set_yticks([10,30,50,70]);ra.set_yticklabels(['Core A E','Core B E','Other E','I'] if j==0 else [],fontsize=8)
        ra.set_title(f'{j+1}  {w["label"]}',fontsize=8,color=w['color'])
        if j==0:ra.text(0,1.23,('B  ' if letters else '')+'Spike raster · 250-ms windows',transform=ra.transAxes,fontsize=11,fontweight='bold')
    slowgs=gs[2].subgridspec(2,1,height_ratios=[2,1],hspace=.08)
    za=fig.add_subplot(slowgs[0]);zt=a['z_time_ms']/1000;zs=a['z_stats']
    za.fill_between(zt,zs[:,2],zs[:,4],color=Z_COL,alpha=.15,lw=0)
    za.plot(zt,zs[:,0],c=Z_COL,lw=1.7,label='E mean Z')
    za.plot(zt,zs[:,5],c='#ca548f',lw=.7,label='Core A')
    za.plot(zt,zs[:,6],c='#3e9ec3',lw=.7,label='Core B')
    za.set(ylabel='Inhibitory resource Z',ylim=(max(0,float(zs[:,2].min())-.06),1.03))
    za.tick_params(labelbottom=False)
    decorate_time(za,r,win,r['duration_ms']/1000);za.set_title(('C  ' if letters else '')+'Slow inhibitory resource',loc='left',fontweight='bold')
    za.legend(loc='lower left',ncol=3,frameon=False,fontsize=8)
    za.text(.99,.97,'M adaptation off',transform=za.transAxes,ha='right',va='top',fontsize=8,color='#666666')
    ga=fig.add_subplot(slowgs[1],sharex=za)
    ga.plot(zt,zs[:,8],c='#a96b32',lw=.65)
    ga.set(xlabel='Time (s)',ylabel='Fraction E with\nGABA ≥ threshold',ylim=(-.04,1.04),yticks=[0,1])
    ga.yaxis.label.set_size(8);ga.tick_params(axis='y',labelsize=8)
    decorate_time(ga,r,win,r['duration_ms']/1000)
    fg=gs[3].subgridspec(1,len(win)+1,width_ratios=[1]*len(win)+[.065],wspace=.11)
    fieldmax=0.;fm=[]
    for w in win:
        lo,hi=w['spatial_window_ms']
        f=a['field_e_count_1ms'][lo:hi].sum(0)/a['cell_e_counts']/.05
        fm.append(f);fieldmax=max(fieldmax,float(f.max()))
    vmax=np.ceil(fieldmax/50)*50
    for j,(w,f) in enumerate(zip(win,fm)):
        fa=fig.add_subplot(fg[j]);im=fa.imshow(f.reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',norm=PowerNorm(.6,vmin=0,vmax=vmax),interpolation='nearest')
        xy=a['centers_mm'];fa.scatter(xy[:,0],xy[:,1],s=19,facecolors='none',edgecolors='#42d5d5',lw=.9)
        fa.set(xticks=[0,20],yticks=[0,20],xlabel='x (mm)');fa.tick_params(labelsize=8)
        if j:fa.set_yticklabels([])
        else:
            fa.set_ylabel('y (mm)');fa.text(0,1.23,('D  ' if letters else '')+'Spatial activity · central 50 ms marked in each raster',transform=fa.transAxes,fontsize=11,fontweight='bold')
        fa.set_title(f'{j+1}  {w["time"]:.2f} s',fontsize=9,color=w['color'])
    cb=fig.colorbar(im,cax=fig.add_subplot(fg[-1]));cb.set_label('Local E rate (Hz)',fontsize=9);cb.ax.tick_params(labelsize=8)
    return dict(selected_contacts=a['contact_names'][contacts].tolist(),readout_gain=scale,
                readout='Applied-current proxy: weighted |I_AMPA|+|Z I_GABA| at the original virtual contacts. Fixed per-contact baseline subtraction and one common gain; no temporal filter or amplitude clipping.',
                raster_sample_count=len(selected),raster_indices=selected.tolist(),spatial_window_ms=50,spatial_color_max_hz=float(vmax),
                entry_selection='Midpoint between last quiet 5-ms bin within 2 s before trigger and the start of its qualifying 200-ms high interval; no spatial selection.')


def stages(r,t):
    start=r['restore_start_ms']/1000;rel=r['release_ms']/1000
    return [(0,start,'#407ca0','Native depletion'),(start,rel,'#2e8c78','External refill'),(rel,t[-1]+.01,'#97558d','Native Z after release')]


def plot_3d(ax,t,e,i,z,r,title='E  Native E–I–Z trajectory',landmarks=None):
    ee=gaussian_filter1d(e,1);ii=gaussian_filter1d(i,1)
    for lo,hi,col,label in stages(r,t):
        ix=np.flatnonzero((t>=lo)&(t<hi))
        ax.plot(ee[ix],ii[ix],z[ix],c=col,lw=.8,alpha=.85,label=label)
        for p in np.linspace(0,max(0,len(ix)-5),10,dtype=int):
            k=ix[p];j=min(k+3,ix[-1])
            ax.add_artist(TrajectoryArrow3D([ee[k],ii[k],z[k]],[ee[j],ii[j],z[j]],col))
    ax.set(xlabel='$r_E$ (Hz)',ylabel='$r_I$ (Hz)',zlabel='Mean E-target Z')
    for number,w in enumerate(landmarks or [],1):
        k=int(np.argmin(abs(t-w['time'])))
        ax.scatter([ee[k]],[ii[k]],[z[k]],s=15,facecolor='white',edgecolor=w['color'],lw=.9,depthshade=False)
        ax.text(ee[k]+5,ii[k],z[k]+.012,str(number),color=w['color'],fontsize=8,weight='bold')
    ax.set_title(title,loc='left',fontweight='bold',pad=8)
    ax.view_init(elev=24,azim=-61);ax.tick_params(labelsize=8,pad=0)
    ax.xaxis.labelpad=1;ax.yaxis.labelpad=1;ax.zaxis.labelpad=1
    ax.legend(loc='upper left',bbox_to_anchor=(-.05,1.02),fontsize=8,frameon=False)
    return ee,ii


def latency_rows():
    rows=[]
    for f in sorted((OUT/'runs').glob('*.json')):
        r=read(f)
        if r.get('status')=='COMPLETE':rows.append(r)
    return rows


def heatmaps(fig,subspec,rows,letter=True):
    taus=[2500.,5000.,10000.];ths=[75.,95.19851312666987,120.]
    probability=np.full((3,3),np.nan);rmst=probability.copy();n=probability.copy()
    for y,h in enumerate(ths):
        for x,tau in enumerate(taus):
            rr=[r for r in rows if r['job']['tau_z_ms']==tau and abs(r['job']['threshold']-h)<1e-6]
            if rr:
                observed=[r['first_trigger_ms'] is not None and r['first_trigger_ms']<=24000 for r in rr]
                probability[y,x]=np.mean(observed);n[y,x]=len(rr)
                rmst[y,x]=np.mean([(r['first_trigger_ms']/1000 if o else 24.) for r,o in zip(rr,observed)])
    gs=subspec.subgridspec(1,2,wspace=.38)
    for j,(data,cmap,vmin,vmax,title) in enumerate([(probability,'Blues',0,1,'High-rate criterion by 24 s'),(rmst,'viridis',0,24,'Time before high-rate criterion')]):
        ax=fig.add_subplot(gs[j]);im=ax.imshow(data,origin='lower',cmap=cmap,vmin=vmin,vmax=vmax,aspect='equal')
        ax.set(xticks=range(3),xticklabels=['2.5','5','10'],yticks=range(3),yticklabels=['75','95.2','120'],xlabel=r'$\tau_Z$ (s)')
        if j==0:ax.set_ylabel('Depletion threshold $I_{th}$\n(model current units)')
        else:ax.set_yticklabels([])
        ax.set_title((('F  ' if letter else '') if j==0 else '')+title,fontsize=10,loc='left')
        for y in range(3):
            for x in range(3):
                if not np.isfinite(data[y,x]):continue
                text=f'{round(probability[y,x]*n[y,x])}/{int(n[y,x])}' if j==0 else f'{data[y,x]:.1f}'
                ax.text(x,y,text,ha='center',va='center',fontsize=11,color='white' if (j==0 and data[y,x]>.6) or (j==1 and data[y,x]<12) else '#222222')
        cb=fig.colorbar(im,ax=ax,orientation='horizontal',pad=.22,fraction=.07)
        cb.set_label('Run fraction' if j==0 else 's; restricted to 24 s',fontsize=8);cb.ax.tick_params(labelsize=8)
    return dict(tau_z_ms=taus,thresholds=ths,n=n.tolist(),transition_probability=probability.tolist(),restricted_mean_transition_free_time_s=rmst.tolist())


def supported_drift(a,r,t,e,i,z):
    """Conditional average of observed derivatives, only at occupied E/I cells.

    This is not a closed ODE. Split by Z and explicitly exclude external refill.
    Refuse off-support interpolation and suppress cancellation-dominated arrows.
    """
    ee=gaussian_filter1d(e,1);ii=gaussian_filter1d(i,1)
    u=np.gradient(ee,.005);v=np.gradient(ii,.005)
    start=r['restore_start_ms']/1000;end=r['release_ms']/1000
    native=(t<start-.025)|(t>end+.025)
    bins=np.linspace(0,max(ee.max(),ii.max())*1.04,19)
    slabs=[(.9,1.001),(.78,.9),(.62,.78)]
    fig=plt.figure(figsize=(15,10));gs=fig.add_gridspec(2,3,height_ratios=[1.1,1],hspace=.4,wspace=.3)
    ax=fig.add_subplot(gs[0,:2],projection='3d');plot_3d(ax,t,e,i,z,r,title='Observed native SNN trajectory',landmarks=windows(t,e,r))
    side=fig.add_subplot(gs[0,2]);side.axis('off')
    side.text(0,.95,'Coordinates',weight='bold',fontsize=13)
    side.text(0,.84,'All-E and all-I firing rates\n5-ms bins, Gaussian smoothing σ = 5 ms\n\nZ: actual mean of 32,000 E targets\nSimulation retains every neuron’s Z.\n\nArrows below: conditional mean drift\nwithin occupied E–I bins and Z bands.\n\nBlank regions are not sampled.\nNo fixed points or nullclines inferred.\n\nExternal refill is shown in green\nand excluded from drift estimates.',va='top',fontsize=11,linespacing=1.55)
    metadata=[];arrays={}
    for k,(zl,zh) in enumerate(slabs):
        ax=fig.add_subplot(gs[1,k]);sel=native&(z>=zl)&(z<zh)
        x=ee[sel];y=ii[sel]
        def stat(values,kind):return binned_statistic_2d(x,y,values,statistic=kind,bins=[bins,bins])[0].T
        count=stat(u[sel],'count');U=stat(u[sel],'mean');V=stat(v[sel],'mean');speed=stat(np.hypot(u[sel],v[sel]),'mean')
        coh=np.hypot(U,V)/np.maximum(speed,1e-9)
        # At least 20 ms total support; arrow alignment is disclosed, not significance.
        valid=(count>=4)&(coh>=.35)&np.isfinite(U)&(speed>1)
        c=(bins[:-1]+bins[1:])/2;X,Y=np.meshgrid(c,c)
        for lo,hi,col,label in stages(r,t):
            rr=sel&(t>=lo)&(t<hi)
            xx=np.where(rr,ee,np.nan);yy=np.where(rr,ii,np.nan)
            ax.plot(xx,yy,c=col,lw=.55,alpha=.5)
        norm=np.hypot(U,V)
        ax.quiver(X[valid],Y[valid],U[valid]/norm[valid],V[valid]/norm[valid],coh[valid],
                  cmap='Greens',clim=(.35,1),angles='xy',scale_units='xy',scale=1/(bins[1]*.65),width=.006,zorder=5)
        ax.set(xlabel='$r_E$ (Hz)',ylabel='$r_I$ (Hz)',xlim=(0,bins[-1]),ylim=(0,bins[-1]),aspect='equal')
        ax.set_title(f'{zl:.2f} ≤ mean Z < {min(zh,1):.2f}\nObserved conditional drift',fontsize=11)
        metadata.append(dict(z_bounds=[zl,zh],sample_count=int(sel.sum()),shown_arrows=int(valid.sum()),minimum_bin_samples=4,minimum_direction_consistency=.35,arrow_length='normalized: direction only'))
        for key,value in dict(X=X,Y=Y,U=U,V=V,count=count,consistency=coh,shown=valid).items():arrays[f'band{k}_{key}']=value
    save(fig,'native_ei_z_trajectory_and_supported_drift')
    np.savez_compressed(OUT/'phase_drift_data.npz',**arrays,time_s=t,e_smooth_hz=ee,i_smooth_hz=ii,z=z)
    return metadata


def regional_phase(a,r,t,e,i,z):
    rr=a['region_spikes_1ms'].reshape(-1,5,6).sum(1)/a['region_counts'][None,:]/.005
    zt=a['z_time_ms']/1000;zs=a['z_stats']
    fig=plt.figure(figsize=(16,5.7))
    data=[(e,i,z,'Whole network'),(rr[:,0],rr[:,3],np.interp(t,zt,zs[:,5]),'Core A neighbourhood'),
          (rr[:,1],rr[:,4],np.interp(t,zt,zs[:,6]),'Core B neighbourhood')]
    for k,(ee,ii,zz,title) in enumerate(data):
        ax=fig.add_subplot(1,3,k+1,projection='3d');plot_3d(ax,t,ee,ii,zz,r,title=title)
        if k:ax.get_legend().remove()
        ax.set_zlabel('Mean E-target Z')
    fig.suptitle('The same continuous run in global and local E–I–Z coordinates',fontsize=15)
    fig.text(.08,.045,'Neighbourhood radius 1.75 mm; threshold-core radius 1.5 mm. Rates include every neuron in each defined region.\nThese are trajectory projections; neither recurrent loops nor a small mean rate identify a limit cycle.',fontsize=10)
    save(fig,'global_and_core_ei_z_trajectories')


def readout_audit(a,r,win):
    tt=a['lfp_time_ms']/1000;ids=choose_contacts(a)
    legacy=a['lfp_raw'][:,ids];applied=a['lfp_effective'][:,ids]
    base=(tt>=.5)&(tt<1);offset=np.median(legacy[base],axis=0)
    gain=float(np.max(np.ptp(np.quantile(np.r_[legacy,applied],[.005,.995],axis=0),axis=0)))
    fig,axs=plt.subplots(2,1,figsize=(15,8),sharex=True,layout='constrained')
    for ax,x,title in zip(axs,[legacy,applied],['Historical proxy: |AMPA| + |GABA| before applying Z','Primary readout: |AMPA| + |Z × GABA| actually applied to E targets']):
        y=np.arange(len(ids))[::-1]
        for k in range(len(ids)):ax.plot(tt,(x[:,k]-offset[k])/gain*.8+y[k],lw=.4,c='#344d60',rasterized=True)
        ax.set_yticks(y);ax.set_yticklabels(a['contact_names'][ids],fontsize=8)
        ax.set(ylabel='Virtual SEEG proxy (a.u.)',title=title)
        decorate_time(ax,r,win,r['duration_ms']/1000)
    axs[-1].set_xlabel('Time (s)')
    fig.suptitle('Readout definition check — same contacts, baseline offsets and amplitude gain',fontsize=14)
    save(fig,'readout_definition_audit')


def paired_controls(a,r,t,e,i,z):
    oldroot=ROOT/'results/topic4_sef_hfo/historical_manual_hard_native_z_v1'
    old=np.load(oldroot/'trajectory.npz');frozen=np.load(oldroot/'native/frozen_t8000.npz')
    frozen94=np.load(oldroot/'native/frozen_t9400.npz')
    fig,axs=plt.subplots(2,3,figsize=(16,7),layout='constrained')
    release=r['release_ms']/1000
    pairs=[(8.,10.,frozen,'Frozen at the 8-s checkpoint'),
           (9.4,r['restore_start_ms']/1000,frozen94,'Frozen at the 9.4-s checkpoint'),
           (release,14.18,old,'Held at Z = 1 after refill')]
    for col,(lo,hi,b,label) in enumerate(pairs):
        ix=(t>=lo)&(t<hi);ta=t[ix]
        if col<2:
            bt=(np.arange(len(b['rate_e_hz'])//50)+.5)*.005+lo
            bz_t=np.arange(len(b['z_summary_1ms']))*.001+lo;bz=b['z_summary_1ms'][:,0]
        else:
            bt=(np.arange(len(b['rate_e_hz'])//50)+.5)*.005
            bz_t=b['z_time_ms']/1000;bz=b['z_stats'][:,0]
        er=b['rate_e_hz'].reshape(-1,50).mean(1);ir=b['rate_i_hz'].reshape(-1,50).mean(1)
        bi=(bt>=lo)&(bt<hi);zi=(bz_t>=lo)&(bz_t<hi)
        axs[0,col].plot(ta,z[ix],color=Z_COL,label='Native Z evolves')
        axs[0,col].plot(bz_t[zi],bz[zi],color='#666666',ls='--',label=label)
        axs[0,col].set(xlim=(lo,hi),ylabel='Mean E-target Z',title=f'From the {lo:g}-s checkpoint' if col<2 else 'After the same external refill')
        axs[0,col].legend(fontsize=9,frameon=False)
        for yy,byy,color,name in [(e,er,E_COL,'E'),(i,ir,I_COL,'I')]:
            axs[1,col].plot(ta,yy[ix],color=color,lw=.9,label=f'{name}: native Z')
            axs[1,col].plot(bt[bi],byy[bi],color=color,lw=.7,ls='--',alpha=.7,label=f'{name}: fixed Z')
        axs[1,col].set(xlim=(lo,hi),xlabel='Time (s)',ylabel='Population rate (Hz; 5-ms bins)')
        axs[1,col].legend(fontsize=8,ncol=2,frameon=False)
    fig.suptitle('Matched native-SNN controls: hold Z or let its original dynamics continue',fontsize=14)
    save(fig,'paired_z_controls_ei')


def main():
    a,r,t,e,i,z=load_main();win=windows(t,e,r);rows=latency_rows()
    fig=plt.figure(figsize=(16,12));gs=fig.add_gridspec(1,1,left=.08,right=.96,top=.94,bottom=.06)
    lm=left_panels(fig,gs[0],a,r,t,e,i,z,win)
    fig.suptitle('Hand-placed dual-core SNN: native depletion, one refill, then native Z resumes',fontsize=15)
    fig.text(.08,.018,'Same continuous trajectory throughout. Raster: fixed sampled neurons. Spatial fields: all E cells. Refill changes only Z; M off.',fontsize=9)
    save(fig,'continuous_readout_raster_z_spatial')
    dm=supported_drift(a,r,t,e,i,z)
    regional_phase(a,r,t,e,i,z)
    readout_audit(a,r,win)
    paired_controls(a,r,t,e,i,z)
    fig=plt.figure(figsize=(10,5));gs=fig.add_gridspec(1,1,left=.1,right=.96,bottom=.2,top=.86)
    hm=heatmaps(fig,gs[0],rows,letter=False)
    fig.suptitle('Transition timing on the same manual dual-core substrate',fontsize=14)
    fig.text(.09,.035,'3 noise seeds per cell; one topology. τZ affects depletion and recovery; I_th is the current threshold for depletion.\nHigh-state trigger: all-E rate ≥200 Hz for 200 ms. Runs without a transition contribute 24 s to the restricted mean.',fontsize=10)
    save(fig,'parameter_transition_time')
    fig=plt.figure(figsize=(22,13));gs=fig.add_gridspec(1,2,width_ratios=[1.95,1],left=.055,right=.98,top=.94,bottom=.075,wspace=.2)
    left_panels(fig,gs[0],a,r,t,e,i,z,win)
    right=gs[1].subgridspec(3,1,height_ratios=[1.25,.65,.9],hspace=.3)
    ax=fig.add_subplot(right[0],projection='3d');plot_3d(ax,t,e,i,z,r,landmarks=win)
    ax=fig.add_subplot(right[1]);ee=gaussian_filter1d(e,1);ii=gaussian_filter1d(i,1)
    reference=(t>=.5)&(t<5.)
    beta=float(np.dot(ee[reference],ii[reference])/np.dot(ee[reference],ee[reference]))
    balance=ii-beta*ee
    for lo,hi,col,label in stages(r,t):
        ix=(t>=lo)&(t<hi);ax.plot(ee[ix],balance[ix],lw=.6,c=col,alpha=.65)
    ax.axhline(0,c='#aaaaaa',lw=.5)
    ax.set(xlabel='$r_E$ (Hz)',ylabel=r'$r_I-\beta r_E$ (Hz)',title='E–I balance view of the same trajectory')
    ax.text(.02,.06,f'β = {beta:.2f}, fixed from 0.5–5 s; coordinate change only',transform=ax.transAxes,va='bottom',fontsize=8)
    heatmaps(fig,right[2],rows)
    fig.suptitle('Slow inhibitory depletion and spatial recruitment in a dual-core SNN',fontsize=17)
    fig.text(.055,.024,'Native SNN observations · one external refill followed by release · same substrate in the parameter scan · M adaptation off',fontsize=11)
    save(fig,'fig5_manual_core_release_candidate')
    meta=dict(main_run=r,windows=win,left=lm,phase_drift=dm,latency=hm,
              EI_balance_projection=dict(beta=beta,reference_s=[.5,5.],definition='r_I - beta*r_E; invertible linear coordinate transform, not a new biological state or inhibitory-current estimate'),
              scope='Observed native-SNN trajectories and conditional drift, not a validated reduced autonomous E-I-Z model or a proven bifurcation.',
              human_acceptance='PENDING_USER_REVIEW')
    write(OUT/'figure_metadata.json',meta)
    (FIG/'README.md').write_text('''### continuous_readout_raster_z_spatial.png / .pdf
同一条手放双核 SNN 连续轨迹：未滤波的虚拟电极电流读出、六个250毫秒固定采样raster、实际Z场统计和raster内标出的中心50毫秒全E空间活动。一次外部补回后重新开启原Z方程，神经元、突触、延迟和噪声历史不重置；M关闭。
**关注点**：主图电流proxy取实际施加的|AMPA|+|Z×GABA|，不是临床原始电压；空间活动共享绝对Hz尺度，所有编号对应同一条时间轴，不是冻结后的续跑。

### native_ei_z_trajectory_and_supported_drift.png / .pdf
全E率、全I率与平均Z形成实际三维轨迹，下方按Z分层显示有采样支持的E–I条件平均流。绿色轨迹为外部补回，估计方向场时排除该阶段。
**关注点**：保留空间及突触隐藏状态的SNN不能自动闭合成三个均值的ODE；空白区域不外推，箭头不能当作固定点、nullcline或Hopf证据。

### parameter_transition_time.png / .pdf
固定同一底物，对三种tauZ与三种耗竭电流阈值分别运行三条噪声，显示24秒内进入持续高活动的比例及24秒限制平均未转变时间。后者是所有运行min(转变时间,24秒)的均值，包含未转变者。
**关注点**：tauZ同时影响恢复和耗竭，阈值不是每事件耗竭量；单位是运行，不是患者或事件。未达到200Hz判据不等于没有较低率的持续活动；CSV另列安静间隔消失的辅助时刻，当前为三种子的有界诊断。

### parameter_first_entry_endpoints.png / .pdf
参数首入时间已齐全、代表连续轨迹仍在记录时的阶段图。数值端点与最终parameter_transition_time相同，标题和图注保留当时的计算状态。
**关注点**：正式本轮参数候选以parameter_transition_time为准，阶段图不替代完整数值QA。

### paired_z_controls_ei.png / .pdf
前两列从同一8秒或9.4秒检查点比较自然演化与冻结Z的延续，9.4秒对照截到外部补回前；右侧比较同一次补回后把Z维持在1或重新释放原Z方程。对照保留相同快速初态和未来输入，E/I率使用相同5毫秒分箱。
**关注点**：右侧两条完整SNN轨迹在释放之前逐步相同；左侧检查点已有精确重放QA。有限时长对照解释Z干预的作用，不等同于长期吸引子或分岔证明。

### global_and_core_ei_z_trajectories.png / .pdf
同一条连续SNN分别投影到全网、Core A邻域和Core B邻域的E率/I率/Z坐标；局部率包含各邻域全部E或I神经元。用于核对全网平均是否掩盖局部活动的往返过程。
**关注点**：邻域半径1.75毫米与实际阈值核心1.5毫米区分；回环是观测轨迹，不能据此认定自治极限环。

### readout_definition_audit.png / .pdf
对照历史Z作用前的电流proxy与本版主图使用的实际施加电流proxy，电极权重、基线偏移和增益完全相同。未滤波指没有时间滤波，不等于忽略Z对抑制电流的调节。
**关注点**：这是观测量定义的检查，不改变神经元动力学；两种信号均保存于同一轨迹文件。

### fig5_manual_core_release_candidate.png / .pdf
按用户新布局组合连续观测、空间招募、E–I–Z轨迹和参数图的Fig5候选；保留独立面板以便再排版。
**关注点**：本版基于历史手放Node场和现有C快速连接/噪声，未切换到另一个任务的core输入版本；待用户科学和目视审阅，不替换正式Figure5。
''',encoding='utf-8')


if __name__=='__main__':main()
