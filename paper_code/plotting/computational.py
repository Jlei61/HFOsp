"""Draw computed arrays only. No image/panel/result-cache input is accepted."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import gaussian_filter1d
from data_processing.raw_to_events import event_spectrogram


def save(fig, directory, name, description):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    fig.savefig(directory/(name+'.png'),dpi=150,bbox_inches='tight')
    fig.savefig(directory/(name+'.pdf'),bbox_inches='tight')
    readme=directory/'README.md'
    text=readme.read_text() if readme.exists() else '# 本次计算生成的图\n'
    title='### '+name+'.png'
    if title not in text:readme.write_text(text+'\n'+title+'\n'+description+'\n**关注点**：以本次输入、参数和计算规模解释结果；不替换论文定稿。\n')
    return fig


def waveform(pre,detected,index=0):
    w=detected['windows'][index];fs=pre['sfreq'];a,b=round(w.start*fs),round(w.end*fs)
    x=pre['band_V'][:12,a:b]*1e6; t=(np.arange(a,b)/fs-w.start)*1000
    scale=max(np.percentile(abs(x),99)*2,1e-6)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    ax.plot(t,(x/scale+np.arange(len(x))[:,None]).T,lw=.6)
    ax.set(yticks=np.arange(len(x)),yticklabels=pre['names'][:len(x)],xlabel='Time within new event (ms)',title='Fig1A: raw-derived 80–250 Hz waveform')
    return fig


def spectra(pre,features,index=0):
    t,f,s,centers=event_spectrogram(pre,features,index)
    keep=np.flatnonzero(features['participating'][:,index]);keep=keep[:8]
    fig,axes=plt.subplots(len(keep),1,figsize=(8,max(3,len(keep)*.8)),sharex=True,layout='constrained',squeeze=False)
    for ax,ci in zip(axes[:,0],keep):
        ax.pcolormesh(t,f,s[ci],cmap='magma',vmin=0,vmax=1,shading='auto')
        ax.axvline(centers[ci],color='cyan',lw=1);ax.set(ylabel=str(pre['names'][ci]),yticks=[80,250])
    axes[0,0].set_title('Fig1B: newly detected event / full S³ time centroid')
    axes[-1,0].set_xlabel('Time in group window (ms)')
    return fig


def rank_heatmap(events,clustered=None):
    r=np.where(events['participating'],events['ranks']+1,np.nan)
    order=np.arange(r.shape[1]) if clustered is None else np.argsort(clustered['labels'],kind='stable')
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained');im=ax.imshow(r[:,order],aspect='auto',interpolation='nearest',cmap='viridis')
    ax.set(xlabel='Event (chronological blocks)' if clustered is None else 'Event (grouped by new K-means labels)',ylabel='Contact',yticks=np.arange(len(r)),yticklabels=events['channel_names'],title='Fig1C: participating-channel ranks' if clustered is None else 'Fig1E: masked-rank K-means')
    ax.tick_params(axis='y',labelsize=7);fig.colorbar(im,ax=ax,label='Within-event rank (1 = earliest)')
    if clustered is not None:
        counts=np.bincount(clustered['labels'][clustered['labels']>=0]);ax.axvline(np.sum(clustered['labels']<0)+counts[0]-.5,color='red')
    return fig


def statistics(stats,panel="D/F"):
    fig,ax=plt.subplots(figsize=(5,4),layout='constrained')
    o=stats['overall'];values=[o['permuted_mean_median'],o['mi_mean'],*[s['mi_mean'] for s in stats['within']]]
    labels=['Null','All','TA','TB'];colors=['gray','#4477aa','#228833','#aa3377']
    take=[0,1] if panel=='D' else ([1,2,3] if panel=='F' else list(range(4)))
    ax.bar(range(len(take)),[values[i] for i in take],color=[colors[i] for i in take])
    ax.set(xticks=range(len(take)),xticklabels=[labels[i] for i in take],ylabel='Matching index',title=f'Fig1{panel}: ONE patient excerpt')
    return fig


def contact_geometry(xyz):
    fig=plt.figure(figsize=(6,4),layout='constrained')
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter(*xyz.T,c=np.arange(len(xyz)),cmap='viridis')
    ax.set(xlabel='MRI x (mm)',ylabel='y (mm)',zlabel='z (mm)',title='Fig2A: contact geometry')
    return fig


def rank_directions(directions,clustered):
    fig,ax=plt.subplots(figsize=(5,4),subplot_kw={'projection':'polar'},layout='constrained')
    d=directions['directions'];angle=np.arctan2(d[:,1],d[:,0])
    for k,color in enumerate(['#4477aa','#cc6677']):
        use=(clustered['labels']==k)&np.isfinite(angle)
        ax.hist(angle[use],bins=np.linspace(-np.pi,np.pi,19),alpha=.5,label=['TA','TB'][k],color=color)
    ax.legend();ax.set_title('Fig2B: early-to-late directions\n3D rank gradient shown in MRI x/y')
    return fig


def shared_field_null(result):
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    if result['status']=='OK':
        ax.hist(result['channel_null'],bins=25,color='gray')
        ax.axvline(result['observed_shared_field_r'],color='red',label='observed')
        ax.legend()
    else:
        ax.text(.5,.5,result['reason'],ha='center',transform=ax.transAxes)
    ax.set(xlabel='TA/TB shared-field Pearson r',ylabel='Permutations',
           title='Fig2D: one-patient field comparison')
    return fig


def noise_confirmation(baseline,confirmation):
    fig,ax=plt.subplots(figsize=(5,4),layout='constrained')
    for i,run in enumerate([baseline,confirmation]):
        score=run['score']
        if score['J'] is None:
            ax.text(i,0,score['status'],ha='center',fontsize=8)
        else:
            ax.bar(i,score['J'],color=['#4477aa','#228833'][i])
    ax.set(xticks=[0,1],xticklabels=['Training noise','New noise'],ylabel='Calibrated J',
           title='Fig4I method: one patient, fixed configuration')
    return fig


def selected_propagation_movie(runs,selection,path,radius):
    """Render the selected new events, or explain why the comparison is unavailable."""
    if selection['status']!='READY':
        print('两个工作点尚未各产生两类合格事件，无法生成四事件比较。')
        return
    propagation_movie(runs,path,radius,event_indices=selection['events'])
    selection['status']='GENERATED_FROM_NEW_SPIKES'
    from IPython.display import Image,display
    display(Image(filename=str(path)))


def crossfit_scores(result):
    fig,axs=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for row in result['folds']:axs[0].plot([0,1],[row['timing_only_score'],row['timing_plus_space_score']],'o-',label=f"fold {row['fold']}")
    axs[0].set(xticks=[0,1],xticklabels=['Timing','Timing + Space'],ylabel='Held-out signed cosine',title='Fig2B: common held-out events');axs[0].legend()
    axs[1].hist(result['direction_shuffle_null_gain'],bins=20,color='gray',alpha=.6)
    axs[1].axvline(result['spatial_information_gain'],color='red',label='observed');axs[1].legend();axs[1].set(xlabel='Spatial information gain',title='Blockwise direction shuffle null')
    return fig


def field_panels(fields,qualification):
    fig,axs=plt.subplots(1,len(fields),figsize=(9,3.8),layout='constrained',squeeze=False)
    for ax,item,label in zip(axs[0],fields,['TA','TB']):
        grid=item['grid'];f,s=item['field'];f=np.where(s>=grid['s_thresh'],f,np.nan)
        im=ax.pcolormesh(grid['X'],grid['Y'],f,cmap='RdBu_r',vmin=-2,vmax=2,shading='auto');ax.scatter(*item['points'].T,c='k',s=9)
        ax.set(aspect='equal',xlabel='Along TA direction / span',ylabel='Transverse / span',title=f'Fig2C: {label}, shared TA plane')
    fig.colorbar(im,ax=list(axs[0]),label='Smoothed earliness = −z(rank)')
    fig.suptitle('Axis qualification: '+str(qualification))
    return fig


def substrate_panel(run,radius):
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    ne=len(run['ec']);im=axs[0].scatter(*run['positions'][:ne].T,c=run['vtheta'][:ne],s=1,cmap='viridis',vmin=11,vmax=18)
    for center in run['centers']:axs[0].add_patch(Circle(center,radius,fill=False,color='red'))
    axs[0].scatter(*run['contact_xy'].T,c='cyan',s=12);axs[0].set(aspect='equal',xlabel='x (mm)',ylabel='y (mm)',title='Fig4A: core thresholds + fixed readout');fig.colorbar(im,ax=axs[0],label='V threshold (mV)')
    t=run['time_ms'];axs[1].plot(t/1000,gaussian_filter1d(run['rate_E'],50),label='E');axs[1].plot(t/1000,gaussian_filter1d(run['rate_I'],50),label='I');axs[1].legend();axs[1].set(xlabel='Simulation time (s)',ylabel='Population rate (Hz)',title='New native SNN dynamics')
    return fig


def model_waveforms(run,title='Fig4F/G: newly simulated contact envelopes'):
    env=run['envelope'];t=(np.arange(env.shape[1])+.5)*run['envelope_dt_ms']
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained');im=ax.imshow(env,aspect='auto',origin='lower',extent=[t[0]/1000,t[-1]/1000,0,len(env)],cmap='magma');ax.set(xlabel='Simulation time (s)',ylabel='Contact',title=title);fig.colorbar(im,ax=ax,label='Gaussian weighted E spikes / 2 ms (smoothed)')
    return fig


def optimization_curves(searches):
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for search in searches:
        rows=search['rows'];best=np.inf;curve=[]
        for r in rows:
            if r['J'] is not None:best=min(best,r['J'])
            curve.append(best if np.isfinite(best) else np.nan)
        k=rows[0]['k'];axs[0].plot(range(1,len(curve)+1),curve,'o-',label=f"K={k} r={rows[0]['restart']}")
        for r in rows:
            if r['J'] is not None:axs[1].scatter(k,r['J'],s=25)
    axs[0].set(xlabel='New simulation evaluation',ylabel='Best training J',title='Fig4C: equal-budget core search');axs[0].legend()
    axs[1].set(xlabel='Number of cores',ylabel='Training J',xticks=[1,2,3,4],title='Core count and training discrepancy')
    return fig


def objective_components(rows,title='Fig4B: new candidate objectives'):
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    names=['rank_pattern','local_and_interrod_timing','participation']
    for name in names:ax.plot(range(len(rows)),[r.get('groups',{}).get(name,{}).get('scaled',np.nan) for r in rows],'o-',label=name)
    ax.set(xlabel='New configuration',ylabel='Calibrated discrepancy',title=title);ax.legend(fontsize=8)
    return fig


def core_density(rows,radius):
    good=[r for r in rows if r['J'] is not None]
    fig,ax=plt.subplots(figsize=(5,4),layout='constrained')
    if good:
        cut=np.quantile([r['J'] for r in good],.2);xy=np.concatenate([np.asarray(r['vector'][:2*r['k']]).reshape(-1,2) for r in good if r['J']<=cut])
        ax.hist2d(*xy.T,bins=20,range=[[0,20],[0,20]],cmap='viridis')
    else:ax.text(.5,.5,'No scorable candidates',transform=ax.transAxes,ha='center')
    ax.set(xlim=(0,20),ylim=(0,20),xlabel='Core x (mm)',ylabel='Core y (mm)',title='Fig4E: lowest training-J quintile positions',aspect='equal')
    return fig


def zm_traces(run):
    tr=run['slow'];fig,axs=plt.subplots(3,1,figsize=(10,6),sharex=True,layout='constrained')
    axs[0].plot(run['time_ms']/1000,gaussian_filter1d(run['rate_E'],50));axs[0].set(ylabel='E rate (Hz)',title='Fig5A: same graph, Z/M enabled')
    axs[1].plot(tr['time_ms']/1000,tr['z_mean']);axs[1].set(ylabel='z mean')
    axs[2].plot(tr['time_ms']/1000,tr['m_mean']);axs[2].set(ylabel='m mean',xlabel='Time (s)')
    return fig


def propagation_movie(runs,path,radius,*,event_indices=None):
    """Each row uses two independently selected NEW events, aligned by centroid.

    Deterministic rule: first eligible event of each new rank-cluster if possible;
    caller provides indices. No cached raster/image and no event-time forcing.
    """
    fig,axs=plt.subplots(len(runs),4,figsize=(12,3*len(runs)),squeeze=False,layout='constrained')
    images=[];cursors=[];selected=[]
    for row,run in enumerate(runs):
        ids=event_indices[row];selected.append(ids)
        for j,ei in enumerate(ids):
            e=run['observed']['events'][int(ei)]
            zero=float(np.nanmin(run['observed']['centroid_ms'][int(ei)]))
            # Common -100..148ms phase grid, nearest native 2ms activity bin.
            start=round((zero-101)/2);stop=start+125
            if start<0 or stop>len(run['movie']) or stop>run['envelope'].shape[1]:raise ValueError('Selected event lacks a complete aligned movie window')
            env=run['envelope'][:,start:stop];env=env/np.maximum(env.max(1,keepdims=True),1e-30)
            t=(np.arange(start,stop)+.5)*2-zero
            ax=axs[row,j];ax.imshow(env,aspect='auto',origin='lower',extent=[t[0],t[-1],0,len(env)],cmap='magma',vmin=0,vmax=1);ax.set(xlabel='Relative time (ms)',ylabel='Contact',title=f'New event {ei}');cursor=ax.axvline(t[0],color='cyan');cursors.append((cursor,-100.));ax.set_xlim(-100,150)
            ax=axs[row,j+2];im=ax.imshow(run['movie'][start],origin='lower',extent=[0,20,0,20],cmap='inferno',vmin=0,vmax=75);images.append((im,run,start));ax.scatter(*run['contact_xy'].T,facecolors='none',edgecolors='cyan',s=12)
            for center in run['centers']:ax.add_patch(Circle(center,radius,fill=False,color='white',lw=.7))
            ax.set(xlabel='x (mm)',ylabel='y (mm)',title=f"Field / J={run['score']['J']:.3g}")
    fig.colorbar(im,ax=list(axs[:,2:].ravel()),label='Active E cells / mm² / 2 ms',shrink=.7)
    def update(frame):
        for im,run,start in images:im.set_data(run['movie'][start+frame])
        for cursor,offset in cursors:cursor.set_xdata([offset+2*frame]*2)
        return [x[0] for x in images]+[x[0] for x in cursors]
    animation=FuncAnimation(fig,update,frames=125,interval=50,blit=False)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);animation.save(path,writer=PillowWriter(fps=20),dpi=90)
    update(45);fig.savefig(path.with_suffix('.png'),dpi=120,bbox_inches='tight');plt.close(fig)
    readme=path.parent/'README.md'
    if not readme.exists() or ('### '+path.name) not in readme.read_text():
        with readme.open('a') as stream:stream.write('\n### '+path.name+'\n两个新模拟工作点各取两个合格事件，显示触点包络和活动E细胞空间分布。每个事件按自身最早质心对齐。\n**关注点**：工作点按本次J排序；不同事件不代表同时发生。\n')
    return selected


def field_dynamics(run,energy,*,indices=None):
    t=energy['time_sec'];z=energy['robust_z'];xy=run['contact_xy'];indices=indices if indices is not None else np.linspace(0,len(t)-1,4).astype(int)
    fig,axs=plt.subplots(1,len(indices),figsize=(12,3),layout='constrained',squeeze=False)
    for ax,i in zip(axs[0],indices):
        im=ax.scatter(*xy.T,c=z[:,i],cmap='RdBu_r',vmin=-5,vmax=5,s=80)
        ax.set(title=f'Fig5B/C: t={t[i]:.2f}s',xlabel='x (mm)',ylabel='y (mm)',aspect='equal')
    fig.colorbar(im,ax=list(axs[0]),label='Band log-power, baseline robust z')
    return fig


def energy_alignment(energy):
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained');t=energy['time_sec']
    scores=energy['scores']
    for name,values in scores.items():
        a=np.asarray(values)
        if a.shape==t.shape and name in ('signed_r','signed_projection_z'):axs[0].plot(t,a,label=name)
    axs[0].set(xlabel='Simulation time (s)',ylabel='Signed field score',title='Supplement: energy vs fixed interictal field');axs[0].legend(fontsize=7)
    axs[1].plot(t,np.nanmean(energy['robust_z'],axis=0));axs[1].set(xlabel='Simulation time (s)',ylabel='Mean energy robust z',title='Supplement: mean energy trajectory')
    return fig


def raw_errors(rows,*,angle=False):
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for name,label in [('mean_rank_error','Mean rank'),('within_shaft_order_error','Order'),('participation_error','Participation')]:
        x=[r['angle'] for r in rows] if angle else range(len(rows))
        y=[r.get('descriptive',{}).get(name) if r.get('descriptive',{}).get('full_contact_pair_support',False) else np.nan for r in rows]
        y=[np.nan if v is None else v for v in y]
        ax.plot(x,y,'o-',label=label)
    ax.set(xlabel='E→E angle offset (deg)' if angle else 'New simulated configuration',ylabel='Descriptive error (0–1)',
           title='Fig4D: fixed-background angle response' if angle else 'Fig4B: three observable errors')
    if not any(r.get('descriptive',{}).get('full_contact_pair_support',False) for r in rows):
        ax.text(.5,.5,'Insufficient contact/pair support',ha='center',transform=ax.transAxes)
    ax.legend();return fig


def model_rank_profiles(result,names):
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for k,ax in enumerate(axes):
        ax.plot(result['patient_profiles'][k],'o-',label='Patient FIT');ax.plot(result['model_profiles'][k],'o-',label='New model')
        ax.set(xticks=np.arange(len(names)),xticklabels=names,ylim=(-.05,1.05),ylabel='Mean normalized participating rank',title=f'Fig4G: {"TA/MTA" if k==0 else "TB/MTB"}')
        ax.tick_params(axis='x',rotation=90);ax.legend()
    return fig


def model_crossfit_matrix(result):
    fig,ax=plt.subplots(figsize=(5,4),layout='constrained');m=np.asarray(result['matrix'])
    im=ax.imshow(m,vmin=-1,vmax=1,cmap='RdBu_r')
    for i in range(2):
        for j in range(2):ax.text(j,i,'NE' if not np.isfinite(m[i,j]) else f'{m[i,j]:+.2f}',ha='center',va='center')
    ax.set(xticks=[0,1],xticklabels=['Patient TA','Patient TB'],yticks=[0,1],yticklabels=['Model MTA','Model MTB'],title='Fig4H: held-out contact correlation')
    fig.colorbar(im,ax=ax,label='Cross-fitted Spearman ρ');return fig


def bandpassed_model_waveform(run,band=(30,80)):
    from scipy.signal import butter,sosfiltfilt
    fs=1000/run['envelope_dt_ms'];x=sosfiltfilt(butter(3,band,btype='bandpass',fs=fs,output='sos'),run['envelope'],axis=1)
    step=np.maximum(np.percentile(abs(x),99,axis=1),1e-12);t=(np.arange(x.shape[1])+.5)/fs
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained');ax.plot(t,(x/step[:,None]+np.arange(len(x))[:,None]*3).T,lw=.5)
    for i in run.get('observed',{}).get('primary_event_indices',[]):
        lo,hi=run['observed']['events'][i]['window_ms'];ax.axvspan(lo/1000,hi/1000,color='gray',alpha=.08)
    ax.set(xlabel='New simulation time (s)',ylabel='Contact (per-contact display scale)',title='Fig4F: 30–80 Hz filtered spike-readout proxy / detected windows')
    return fig


def raw_current_waveforms(run,*,baseline_sec=(1.5,3.),onset=None):
    t=run['current_time_ms']/1000;x=run['contact_current'];base=(t>=baseline_sec[0])&(t<=baseline_sec[1])
    scale=np.maximum(np.percentile(abs(x[:,base]),95,axis=1),1e-12)
    window=(max(0,onset-1),min(t[-1],onset+1)) if onset is not None else (max(0,t[-1]-3),t[-1])
    use=(t>=window[0])&(t<=window[1]);scaled=x[:,use]/scale[:,None]
    gap=max(2.,float(np.nanmax(scaled))*1.1);offset=np.arange(len(x))*gap
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained');ax.plot(t[use],(scaled+offset[:,None]).T,lw=.6)
    if onset is not None:ax.axvline(onset,color='red',ls='--',label='t120 entry');ax.legend()
    ax.set(yticks=offset,yticklabels=np.arange(1,len(x)+1),xlabel='Simulation time (s)',ylabel='Virtual contact / fixed baseline P95 scale',title='Fig5A: new unfiltered current proxy (DC retained)')
    return fig


def slow_variables(run,eta):
    s=run['slow'];t=s['time_ms']/1000
    fig,axs=plt.subplots(2,1,figsize=(9,4),sharex=True,layout='constrained')
    axs[0].plot(t,s['z_mean'],color='#4477aa');axs[0].set(ylabel='Mean E: Z',title='Fig5B: native slow variables, same new trajectory')
    axs[1].plot(t,eta*s['m_mean'],color='#ee7733');axs[1].set(ylabel='Mean E: ηM (mV)',xlabel='Simulation time (s)')
    return fig


def activity_snapshots(run,radius,*,onset=None):
    frames=(np.arange(len(run['movie']))+.5)*float(run['frame_ms'])/1000;times=np.linspace(1.5,frames[-1]-.03,4) if onset is None else np.clip(onset+np.array([-.3,-.1,.1,.3]),frames[0]+.025,frames[-1]-.025)
    fields=np.asarray([run['movie'][abs(frames-t)<=.025].mean(0) for t in times]);vmax=max(float(np.max(fields)),1.)
    fig,axs=plt.subplots(1,4,figsize=(12,3),layout='constrained')
    for ax,t,field in zip(axs,times,fields):
        im=ax.imshow(field,origin='lower',extent=[0,20,0,20],vmin=0,vmax=vmax,cmap='inferno')
        for center in run['centers']:ax.add_patch(Circle(center,radius,fill=False,color='cyan',lw=.7))
        ax.set(title=f'Fig5C: {t:.3f}s',xlabel='x (mm)',ylabel='y (mm)')
    fig.colorbar(im,ax=list(axs),label='Active E / mm² / 2ms; averaged over 50ms')
    return fig


def state_trajectory(run,eta):
    s=run['slow'];dt=float(run['parameters']['dt']);width=round(5/dt);n=len(run['rate_E'])//width
    rate=run['rate_E'][:n*width].reshape(n,width).mean(1);t=(np.arange(n)+.5)*.005
    rate=gaussian_filter1d(rate,1);z=np.interp(t,s['time_ms']/1000,s['z_mean']);m=eta*np.interp(t,s['time_ms']/1000,s['m_mean'])
    fig=plt.figure(figsize=(8,5));ax=fig.add_axes([.02,.06,.72,.9],projection='3d')
    ax.plot(z,m,rate,color='gray',alpha=.5,lw=.6);im=ax.scatter(z,m,rate,c=t,s=3,cmap='viridis')
    ax.set(xlabel='Mean E: Z',ylabel='Mean E: ηM (mV)',zlabel='E rate (Hz)',title='Fig5D: one new trajectory (not a bifurcation diagram)')
    fig.colorbar(im,cax=fig.add_axes([.88,.2,.025,.6]),label='Simulation time (s)');return fig


def onset_energy_field(run,result,geometry,*,label="model"):
    from analysis.src.topic5_gradient_grid_field import make_adaptive_grid,build_grid_field
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    if result['status']=='OK':
        scorer=geometry['scorer'];points=scorer['points'];weight=scorer['support'];sigma=scorer['sigma']
        grid=make_adaptive_grid(points,sigma,float(weight.sum()))
        field,support=build_grid_field(grid['X'],grid['Y'],points,weight,np.asarray(result['robust_z'])[geometry['contact_mask']],sigma)
        field=np.where(support>=grid['s_thresh'],field,np.nan)
        lim=max(float(np.nanmax(abs(field))),1e-8)
        im=ax.pcolormesh(grid['X'],grid['Y'],field,cmap='RdBu_r',vmin=-lim,vmax=lim,shading='auto')
        ax.scatter(*points.T,c='k',s=10);fig.colorbar(im,ax=ax,label='Broadband log-power / baseline robust z')
    else:ax.text(.5,.5,result['reason'],ha='center',transform=ax.transAxes)
    ax.set(aspect='equal',xlabel='Along fixed TA axis / span',ylabel='Transverse / span',title=f'Fig5E {label}: newly computed 1–150 Hz onset field')
    return fig
