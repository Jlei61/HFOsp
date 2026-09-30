"""Same-space native/source-only comparisons, with all propagation readouts."""
from shared_source import *
import importlib.util
spec=importlib.util.spec_from_file_location('population_figure_helpers',ROOT/'scripts/topic4_spatial_population_dynamics/figures.py')
oldplot=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldplot)
plt=oldplot.plt;Circle=oldplot.Circle;gaussian_filter1d=oldplot.gaussian_filter1d
DISPLAY=oldplot.DISPLAY;ORDER=oldplot.ORDER;F=OUT/'figures'
OLD='mean_adaptive1_J1.355_s848101';NEW='source_only_cuda_J1.355_s848101'
REFINED=False

def save(fig,name):
    F.mkdir(exist_ok=True);fig.savefig(F/(name+'.png'),dpi=170,bbox_inches='tight');fig.savefig(F/(name+'.svg'),bbox_inches='tight');plt.close(fig)

def load(name):
    if name=='native':return np.load(PRIOR/'native/848101/trajectory.npz')
    return np.load((BASE if name.startswith('mean_') else OUT)/'runs'/name/'trajectory.npz')

def summary(name):
    if name=='native':return read(BASE/'summaries/native_s848101.json')
    return read((BASE if name.startswith('mean_') else OUT)/'summaries'/f'{name}_population.json')

def data():
    natives=[read(BASE/'summaries'/f'native_s{s}.json') for s in (848101,848102,848103)]
    pattern='source_only_half_cuda*_population.json' if REFINED else 'source_only_cuda*_population.json'
    candidates=[q for f in sorted((OUT/'summaries').glob(pattern)) if (q:=read(f)).get('readout')=='population']
    return natives,candidates

def waveform():
    sizes=np.bincount(model()['region'],minlength=6)
    fig,axs=plt.subplots(3,2,figsize=(14,9),layout='constrained')
    for i,(name,title) in enumerate([('native','Native SNN'),(OLD,'Source only: coarse grid' if REFINED else 'Previous: both endpoints averaged'),(NEW,'Source only: refined grid' if REFINED else 'New: source averaging only')]):
        z=load(name);oldplot.core_lines(axs[i,0],z,sizes);oldplot.envelope(axs[i,1],z)
        axs[i,0].set_title(title,loc='left')
        if i<2:
            for ax in axs[i]:ax.tick_params(labelbottom=False)
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.3,1),ncol=3)
    axs[0,1].set_title('SEEG firing envelope (normalized)',loc='left')
    for ax in axs[-1]:ax.set_xlabel('Time (s)')
    save(fig,'01_waveforms_and_seeg')

def raster():
    m=model();sizes=np.bincount(m['region'],minlength=6);original=np.load(V10/'native/a/trajectory.npz')
    sample=original['raster_sample_ids'];reg=m['region'];sample=sample[np.lexsort((sample,reg[sample]))]
    inv=np.full(len(reg),-1);inv[sample]=np.arange(len(sample));n=np.bincount(reg[sample],minlength=6);edge=np.r_[0,np.cumsum(n)]
    fig,axs=plt.subplots(2,2,figsize=(14,7),gridspec_kw={'height_ratios':[1,1.5]},layout='constrained')
    for col,(name,title) in enumerate([('native','Native SNN'),(NEW,'Source-only / refined grid' if REFINED else 'Source-only / coarse grid')]):
        z=load(name);oldplot.core_lines(axs[0,col],z,sizes);axs[0,col].set_title(title)
        if name=='native':time=original['exact_spike_time_ms'];cells=original['exact_spike_cell']
        else:
            time=z['raster_times_ms'];cells=z['raster_neuron_ids'];assert np.array_equal(np.sort(z['raster_sample_ids']),np.sort(sample))
        keep=(time>=2500)&(time<4000);ax=axs[1,col];colors=np.array(['#0072B2','#D55E00','#555555','#56B4E9','#E69F00','#999999'])
        ax.scatter(time[keep]/1000,inv[cells[keep]],s=2.8,c=colors[reg[cells[keep]]],marker='|',linewidths=.55)
        for e in edge[1:-1]:ax.axhline(e-.5,color='#bbbbbb',lw=.65)
        ax.set(xlim=(2.5,4),ylim=(len(sample),0),yticks=(edge[:-1]+edge[1:])/2,yticklabels=[f'{s} ({v})' for s,v in zip(['AE','BE','SE','AI','BI','SI'],n)],xlabel='Time (s)',ylabel='Same sampled neurons')
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,1),ncol=3)
    save(fig,'02_matched_neuron_rasters')

def observables():
    natives,candidates=data();suffix='seed' if len(candidates)==1 else 'seeds'
    groups=[(f'Native ({len(natives)} seeds)',natives,'black'),('Coarse sources (1 seed)' if REFINED else 'Previous mean input (1 seed)',[summary(OLD)],'#E69F00'),(f'{"Refined sources" if REFINED else "Source only"} ({len(candidates)} {suffix})',candidates,'#0072B2')]
    fig,axs=plt.subplots(1,3,figsize=(15,6),layout='constrained',gridspec_kw={'width_ratios':[1,1,1.3]})
    for j,metric in enumerate(['rank','participation']):
        ax=axs[j];yy=np.arange(15)
        for label,qs,color in groups:
            arr=np.array([np.array(q['summary']['mean_rank'])[ORDER] if metric=='rank' else [q['summary']['contacts'][n]['participation'] for n in DISPLAY] for q in qs],float)
            for ids in [np.arange(4),np.arange(4,15)]:
                if len(qs)>1:ax.fill_betweenx(yy[ids],arr.min(0)[ids],arr.max(0)[ids],color=color,alpha=.1)
                ax.plot(arr.mean(0)[ids],yy[ids],'o-',color=color,ms=4,label=label if ids[0]==0 else None)
        ax.axhline(3.5,color='#bbbbbb',lw=.8);ax.set(xlim=(-.03,1.03),ylim=(14.5,-.5),yticks=yy,yticklabels=DISPLAY,
            xlabel='Mean normalized rank' if metric=='rank' else 'Participation probability',title='Mean propagation order' if metric=='rank' else 'Contact participation')
    delta=np.full((15,15),np.nan)
    for key in natives[0]['summary']['pairs']:
        a,b=key.split('→');i,j=DISPLAY.index(a),DISPLAY.index(b)
        v=[q['summary']['pairs'][key]['order_probability'] for q in candidates];r=[q['summary']['pairs'][key]['order_probability'] for q in natives]
        if all(x is not None for x in v+r):delta[i,j]=np.mean(v)-np.mean(r);delta[j,i]=-delta[i,j]
    ax=axs[2];im=ax.imshow(delta,cmap='RdBu_r',vmin=-1,vmax=1);ax.set(xticks=np.arange(15),xticklabels=DISPLAY,yticks=np.arange(15),yticklabels=DISPLAY,title='Within-shaft order difference')
    ax.tick_params(axis='x',labelrotation=90,labelsize=9);ax.tick_params(axis='y',labelsize=9);ax.axhline(3.5,color='#bbbbbb',lw=.8);ax.axvline(3.5,color='#bbbbbb',lw=.8)
    fig.colorbar(im,ax=ax,fraction=.047,pad=.02,label=('Refined sources' if REFINED else 'Source only')+' − native mean')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,1),ncol=3)
    save(fig,'03_three_propagation_observables')

def snapshots(sign):
    m=model();sizes=np.bincount(m['region'],minlength=6);cfg=read(PRIOR/'model_config.json');selected=[]
    for name,title in [('native','Native SNN'),(NEW,'Source-only / refined grid' if REFINED else 'Source-only / coarse grid')]:
        rows=[r for r in summary(name)['spatial_events'] if r['core_B_minus_A_ms'] is not None and r['core_B_minus_A_ms']*sign>0]
        if not rows:raise ValueError('No event of requested onset sign; cannot fabricate a snapshot')
        median=np.median([r['core_B_minus_A_ms'] for r in rows]);row=min(rows,key=lambda r:abs(r['core_B_minus_A_ms']-median))
        z=load(name);core=gaussian_filter1d(z['six_counts'][:,:2]/sizes[:2]/.002,2.5,axis=0)
        lo=int(np.ceil(row['window_ms'][0]/2));hi=int(np.floor(row['window_ms'][1]/2))
        t0=min((lo+np.flatnonzero(core[lo:hi,j]>20)[0])*2+1 for j in range(2));inds=[round((t0+t-1)/2) for t in [0,20,40,60]]
        selected.append((name,title,z,row,t0,inds))
    vmax=max(float(z['field_counts'][i].max()) for _,_,z,_,_,inds in selected for i in inds)
    fig,axs=plt.subplots(2,5,figsize=(18,7),layout='constrained',gridspec_kw={'width_ratios':[1.5,1,1,1,1]});meta=[]
    for r,(name,title,z,event,t0,inds) in enumerate(selected):
        key='contact_envelope' if name=='native' else 'group_contact_envelope';env=z[key][:,ORDER]
        start=max(0,int((t0-30)//2));end=min(len(env),int((t0+120)//2));e=env[start:end];e=e/np.maximum(e.max(0),1e-12)
        ax=axs[r,0];ax.imshow(e.T,aspect='auto',extent=(start*2-t0,end*2-t0,14.5,-.5),cmap='magma',vmin=0,vmax=1,interpolation='nearest')
        ax.set(yticks=np.arange(15),yticklabels=DISPLAY,xlabel='Time from first core onset (ms)',title=title);ax.tick_params(axis='y',labelsize=9);ax.axhline(3.5,color='white',lw=.7)
        for col,i in enumerate(inds,1):
            ax=axs[r,col];im=ax.imshow(z['field_counts'][i],origin='lower',extent=(0,20,0,20),cmap='inferno',vmin=0,vmax=vmax,interpolation='nearest')
            for center,radius in zip(cfg['core_centers'],cfg['core_radii']):ax.add_patch(Circle(center,radius,fill=False,ec='white',lw=1.3))
            xy=m['contact_xy'];ax.scatter(xy[:,0],xy[:,1],s=14,facecolors='none',edgecolors='#29d2dd',lw=.8)
            ax.set(title=f'+{[0,20,40,60][col-1]} ms',xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)')
            if col==1:ax.set_ylabel('y (mm)')
        meta.append(dict(model=name,event=event,t0_ms=t0,snapshot_times_ms=np.array(inds)*2+1))
    fig.colorbar(im,ax=axs[:,1:],fraction=.02,pad=.01,label='E spikes / 2 ms / 1 mm cell')
    label='a_leads' if sign==1 else 'b_leads';save(fig,'04_spatial_'+label)
    write(F/f'04_spatial_{label}_selection.json',dict(events=meta,selection='Valid events; requested sign of first-core-onset difference; closest to its median; no matching by appearance.',color_max=vmax))

def quantitative():
    natives,candidates=data();before=summary(OLD);fig,axs=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    specs=[('Core A mean rate (Hz)',lambda q:q['dynamics']['AE']['mean_hz']),('Core A interval CV',lambda q:q['dynamics']['AE']['peak_interval_CV']),
        ('Core A low-activity fraction',lambda q:q['dynamics']['AE']['low_rate_fraction']),
        ('Recruited sheet fraction',lambda q:np.median([e['area_fraction'] for e in q['spatial_events']])),
        ('Field onset span (ms)',lambda q:np.median([e['field_onset_span_ms'] for e in q['spatial_events']])),
        ('Participation error / native pair maximum',lambda q:np.array(q['errors_to_native'])[int(q['config']['seed'])-848101,2]/q['native_pair_max'][2])]
    for j,(label,fn) in enumerate(specs):
        ax=axs.flat[j]
        if j<5:
            for x,qs,color in [(0,natives,'black'),(1,[before],'#E69F00'),(2,candidates,'#0072B2')]:
                values=[fn(q) for q in qs];xx=x+np.linspace(-.08,.08,len(values)) if len(values)>1 else [x];ax.plot(xx,values,'o',color=color,ms=6)
            ax.set(xticks=[0,1,2],xticklabels=['Native','Coarse\nsources','Refined\nsources'] if REFINED else ['Native','Previous','Source only'])
        else:
            ax.axhline(1,ls='--',color='black',lw=1)
            for x,qs,color in [(0,[before],'#E69F00'),(1,candidates,'#0072B2')]:
                xx=x+np.linspace(-.08,.08,len(qs)) if len(qs)>1 else [x];ax.plot(xx,[fn(q) for q in qs],'o',color=color,ms=6)
            ax.set(xticks=[0,1],xticklabels=['Coarse\nsources','Refined\nsources'] if REFINED else ['Previous','Source only'],ylim=(0,None))
        ax.set_ylabel(label);ax.margins(x=.3)
    save(fig,'05_noise_repeat_diagnostics')

def amplitude():
    import csv
    with (OUT/'contact_amplitude_audit.csv').open() as f:rows=list(csv.DictReader(f))
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    series=[('Native 848101','native_s848101','black'),('Coarse source grid','source_only_cuda_J1.355_s848101','#E69F00')]
    if REFINED:series.append(('Refined source grid',NEW,'#0072B2'))
    for ax,name in zip(axs,['SCL9','ICL10']):
        for label,run,color in series:
            values=np.sort([float(r['peak_over_threshold']) for r in rows if r['model']==run and r['contact']==name])
            ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',color=color,lw=2,label=label)
        ax.axvline(1,color='black',ls='--',lw=1);ax.set(xscale='log',ylim=(0,1.02),title=name,xlabel='Envelope peak / fixed threshold',ylabel='Fraction of all detected windows')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,1),ncol=len(series))
    save(fig,'06_contact_amplitude_audit')

def readout_effect():
    z=load(NEW);q=read(OUT/'summaries'/f'{NEW}_neuron.json');ob=read(OUT/'observations'/f'{NEW}_neuron.json')
    j=0;bar=ob['threshold'][j];mu=np.array(ob['centroid_ms'],float)
    ids=[i for i in q['event_ids'] if not np.isfinite(mu[i,j])][:2];fig,axs=plt.subplots(1,len(ids),figsize=(6*len(ids),4),layout='constrained',squeeze=False)
    meta=[]
    for ax,event in zip(axs[0],ids):
        a,b=(np.array(ob['events'][event]['window_ms'])/2).astype(int);peak=a+np.argmax(z['contact_envelope'][a:b,j]);lo=max(a,peak-15);hi=min(b,peak+16)
        for key,label,color in [('contact_envelope','Exact neuron weights','black'),('group_contact_envelope','Refined group readout','#0072B2'),('common_group_contact_envelope','Common coarse readout','#E69F00')]:
            ax.plot((np.arange(lo,hi)-peak)*2,z[key][lo:hi,j]/bar,'o-',color=color,ms=3,label=label)
        ax.axhline(1,color='black',ls='--',lw=1);ax.set(title=f'SCL9: event {event}',xlabel='Time from local peak (ms)',ylabel='Envelope / fixed threshold')
        meta.append(dict(event=event,window_ms=ob['events'][event]['window_ms'],peak_ms=peak*2+1))
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,1),ncol=3)
    save(fig,'07_readout_resolution_effect');write(F/'07_readout_resolution_effect.json',dict(events=meta,selection='First two valid-event windows without SCL9 participation in the exact neuron readout; all curves on the same exact-readout windows.'))

def main():
    waveform();raster();observables();snapshots(1);snapshots(-1);quantitative();amplitude()
    if REFINED:readout_effect()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--refined',action='store_true');a=p.parse_args()
    if a.refined:
        REFINED=True;OLD=NEW;NEW='source_only_half_cuda_J1.355_s848101';F=OUT/'figures/refined_sources';F.mkdir(exist_ok=True,parents=True)
    main()
