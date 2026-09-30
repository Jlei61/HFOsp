"""Diagnostic scientific figures from autonomous simulations; no synthetic raster."""
from shared import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter1d

F=OUT/'figures'
DISPLAY=['SCL9','SCL8','SCL7','SCL6']+[f'ICL{i}' for i in range(11,0,-1)]
NAMES=read(V10/'native/a/observation_contract.json')['contact_names']
ORDER=[NAMES.index(n) for n in DISPLAY]
COLORS=['#0072B2','#D55E00','#555555']
LABELS=['Core A','Core B','Surround E']
REFINED='composition_adaptive1_J1.355_s848101'
BASE='mean_grid20_J1.355_s848101'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':12,
    'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})

def save(fig,name):
    F.mkdir(exist_ok=True);fig.savefig(F/(name+'.png'),dpi=170,bbox_inches='tight');fig.savefig(F/(name+'.svg'),bbox_inches='tight');plt.close(fig)

def load(name):
    return np.load(PRIOR/'native/848101/trajectory.npz') if name=='native' else np.load(OUT/'runs'/name/'trajectory.npz')

def summary(name):return read(OUT/'summaries'/('native_s848101.json' if name=='native' else name+'_population.json'))

def core_lines(ax,z,sizes,lo=2500,hi=4000):
    time=np.arange(len(z['six_counts']))*2+1;r=gaussian_filter1d(z['six_counts']/sizes/.002,2.5,axis=0)
    for j in range(3):ax.plot(time/1000,r[:,j],color=COLORS[j],lw=1.2,label=LABELS[j])
    ax.set(xlim=(lo/1000,hi/1000),ylim=(0,440),ylabel='Rate (Hz)');ax.set_yticks([0,200,400])

def envelope(ax,z,lo=2500,hi=4000):
    key='group_contact_envelope' if 'group_contact_envelope' in z.files else 'contact_envelope'
    env=z[key][:,ORDER];den=env[1000:].max(axis=0);env=env/np.maximum(den,1e-12)
    ax.imshow(env.T,aspect='auto',origin='upper',extent=(0,len(env)*.002,14.5,-.5),cmap='magma',vmin=0,vmax=1,interpolation='nearest')
    ax.axhline(3.5,color='white',lw=.8);ax.set(xlim=(lo/1000,hi/1000),yticks=np.arange(15),yticklabels=DISPLAY)
    ax.tick_params(axis='y',labelsize=9)

def dynamics_figure(sizes):
    runs=[('Native SNN',load('native')),('Previous rate / same 1 mm graph',np.load(PRIOR/'rate/grid20_seed848101.npz')),
        ('Population particles / 1 mm',load(BASE)),('Population particles / source composition',load(REFINED))]
    fig,axs=plt.subplots(4,2,figsize=(14,11),gridspec_kw={'width_ratios':[1.15,1]},layout='constrained')
    for i,(label,z) in enumerate(runs):
        core_lines(axs[i,0],z,sizes);axs[i,0].set_title(label,loc='left');envelope(axs[i,1],z)
        if i<3:
            axs[i,0].tick_params(labelbottom=False);axs[i,1].tick_params(labelbottom=False)
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.3,1),ncol=3,fontsize=10)
    axs[0,1].set_title('SEEG contact firing envelope',loc='left')
    for ax in axs[-1]:ax.set_xlabel('Time (s)')
    save(fig,'01_dynamics_and_seeg')

def raster_figure(model,sizes):
    fig,axs=plt.subplots(2,2,figsize=(14,7),gridspec_kw={'height_ratios':[1,1.5]},layout='constrained')
    original=np.load(V10/'native/a/trajectory.npz')
    for col,name in enumerate(['native',REFINED]):
        z=load(name);core_lines(axs[0,col],z,sizes,2500,4000)
        axs[0,col].set_title('Native SNN' if name=='native' else 'Population particles / source composition')
        if name=='native':tt=original['exact_spike_time_ms'];cells=original['exact_spike_cell'];sample=original['raster_sample_ids']
        else:tt=z['raster_times_ms'];cells=z['raster_neuron_ids'];sample=z['raster_sample_ids']
        reg=model['region'];sample=sample[np.lexsort((sample,reg[sample]))]
        sample=np.concatenate([sample[reg[sample]==r][np.linspace(0,np.sum(reg[sample]==r)-1,100,dtype=int)] for r in range(6)])
        inverse=np.full(len(reg),-1);inverse[sample]=np.arange(len(sample))
        ind=(tt>=2500)&(tt<4000)&(inverse[cells]>=0);y=inverse[cells[ind]];assert (y>=0).all()
        ax=axs[1,col];ax.scatter(tt[ind]/1000,y,s=2.7,c=np.array(['#0072B2','#D55E00','#555555','#56B4E9','#E69F00','#999999'])[reg[cells[ind]]],marker='|',linewidths=.55)
        counts=np.bincount(reg[sample],minlength=6);edge=np.r_[0,np.cumsum(counts)]
        for pos in edge[1:-1]:ax.axhline(pos-.5,color='#bbbbbb',lw=.65)
        ax.set(xlim=(2.5,4),ylim=(len(sample),0),yticks=(edge[:-1]+edge[1:])/2,
            yticklabels=[f'{s} ({n})' for s,n in zip(['AE','BE','SE','AI','BI','SI'],counts)],xlabel='Time (s)',ylabel='Sampled neurons')
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,fontsize=10,loc='lower center',bbox_to_anchor=(.5,1))
    save(fig,'02_true_spike_rasters')

def metrics_figure():
    natives=[read(OUT/'summaries'/f'native_s{s}.json')['summary'] for s in (848101,848102,848103)]
    curves=[('1 mm mean input',summary(BASE)['summary'],'#E69F00'),('Source composition',summary(REFINED)['summary'],'#009E73')]
    fig,axs=plt.subplots(1,3,figsize=(15,6),layout='constrained',gridspec_kw={'width_ratios':[1,1,1.3]})
    for panel,metric in enumerate(['rank','participation']):
        values=lambda q:np.array(q['mean_rank'])[ORDER] if metric=='rank' else np.array([q['contacts'][n]['participation'] for n in DISPLAY])
        nv=np.array([values(q) for q in natives]);ax=axs[panel];yy=np.arange(15)
        for ids in [np.arange(4),np.arange(4,15)]:
            ax.fill_betweenx(yy[ids],nv.min(0)[ids],nv.max(0)[ids],color='black',alpha=.12)
            ax.plot(nv.mean(0)[ids],yy[ids],'o-',color='black',ms=4,label='Native: 3 seeds' if ids[0]==0 else None)
            for label,q,color in curves:ax.plot(values(q)[ids],yy[ids],'o-',color=color,ms=4,label=label if ids[0]==0 else None)
        ax.axhline(3.5,color='#bbbbbb',lw=.8);ax.set(xlim=(-.03,1.03),ylim=(14.5,-.5),yticks=yy,yticklabels=DISPLAY,
            xlabel='Mean normalized rank' if metric=='rank' else 'Participation probability',title='Mean propagation order' if metric=='rank' else 'Contact participation')
    q=curves[-1][1];delta=np.full((15,15),np.nan)
    for key,v in q['pairs'].items():
        a,b=key.split('→');i,j=DISPLAY.index(a),DISPLAY.index(b)
        native=np.array([r['pairs'][key]['order_probability'] for r in natives],float)
        if v['order_probability'] is not None and np.isfinite(native).all():
            delta[i,j]=v['order_probability']-native.mean();delta[j,i]=-delta[i,j]
    ax=axs[2];im=ax.imshow(delta,cmap='RdBu_r',vmin=-1,vmax=1);ax.set(xticks=np.arange(15),xticklabels=DISPLAY,yticks=np.arange(15),yticklabels=DISPLAY,title='Within-shaft order difference')
    ax.tick_params(axis='x',labelrotation=90,labelsize=9);ax.tick_params(axis='y',labelsize=9)
    for d in [3.5]:ax.axhline(d,color='#bbbbbb',lw=.8);ax.axvline(d,color='#bbbbbb',lw=.8)
    fig.colorbar(im,ax=ax,fraction=.047,pad=.02,label='Composition − native mean')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,1),ncol=3,fontsize=10)
    save(fig,'03_propagation_observables')

def choose_event(name,sign):
    q=summary(name);rows=[r for r in q['spatial_events'] if r['core_B_minus_A_ms'] is not None and r['core_B_minus_A_ms']*sign>0]
    if not rows:return None
    median=np.median([r['core_B_minus_A_ms'] for r in rows]);return min(rows,key=lambda r:abs(r['core_B_minus_A_ms']-median))

def propagation_figure(model,sizes,sign):
    selected=[];cfg=read(PRIOR/'model_config.json')
    for name in ['native',REFINED]:
        row=choose_event(name,sign)
        if row is None:continue
        z=load(name);core=gaussian_filter1d(z['six_counts'][:,:2]/sizes[:2]/.002,2.5,axis=0)
        a,b=row['window_ms'];lo=int(np.ceil(a/2));hi=int(np.floor(b/2))
        t0=min((lo+np.flatnonzero(core[lo:hi,j]>20)[0])*2+1 for j in range(2))
        indices=[round((t0+offset-1)/2) for offset in [0,20,40,60]]
        selected.append((name,z,row,t0,indices))
    if len(selected)!=2:raise ValueError('Missing A- or B-leading events; do not invent examples')
    vmax=max(float(z['field_counts'][ii].max()) for _,z,_,_,inds in selected for ii in inds)
    fig,axs=plt.subplots(2,5,figsize=(18,7),layout='constrained',gridspec_kw={'width_ratios':[1.5,1,1,1,1]})
    eventmeta=[]
    for row,(name,z,event,t0,inds) in enumerate(selected):
        envkey='group_contact_envelope' if 'group_contact_envelope' in z.files else 'contact_envelope'
        env=z[envkey][:,ORDER];start=max(0,int((t0-30)//2));end=min(len(env),int((t0+120)//2));e=env[start:end]
        e=e/np.maximum(e.max(0),1e-12);ax=axs[row,0]
        ax.imshow(e.T,aspect='auto',extent=(start*2-t0,end*2-t0,14.5,-.5),cmap='magma',vmin=0,vmax=1,interpolation='nearest')
        ax.set(yticks=np.arange(15),yticklabels=DISPLAY,xlabel='Time from first core onset (ms)',title='Native SNN' if name=='native' else 'Source composition')
        ax.tick_params(axis='y',labelsize=9);ax.axhline(3.5,color='white',lw=.7)
        for col,ii in enumerate(inds,1):
            ax=axs[row,col];im=ax.imshow(z['field_counts'][ii],origin='lower',extent=(0,20,0,20),cmap='inferno',vmin=0,vmax=vmax,interpolation='nearest')
            for center,radius in zip(cfg['core_centers'],cfg['core_radii']):ax.add_patch(Circle(center,radius,fill=False,ec='white',lw=1.3))
            xy=model['contact_xy'];ax.scatter(xy[:,0],xy[:,1],s=14,facecolors='none',edgecolors='#29d2dd',lw=.8)
            ax.set(title=f'+{[0,20,40,60][col-1]} ms',xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)')
            if col==1:ax.set_ylabel('y (mm)')
        eventmeta.append(dict(model=name,event=event,t0_ms=t0,snapshot_times_ms=(np.array(inds)*2+1).tolist()))
    fig.colorbar(im,ax=axs[:,1:],fraction=.02,pad=.01,label='E spikes / 2 ms / 1 mm cell')
    label='a_leads' if sign==1 else 'b_leads';save(fig,'04_spatial_'+label)
    write(F/f'04_spatial_{label}_selection.json',dict(events=eventmeta,selection='Closest to median B-minus-A first-20Hz-onset lag of the requested sign, among valid events; not matched trials or TA/TB labels.',color_max=vmax))

def diagnostic_figure():
    models=[('2 mm / mean','mean_J1.355_s848101'),('1 mm / mean',BASE),('1 + 0.5 mm','mean_adaptive1_J1.355_s848101'),('E/I total gain','gain_adaptive1_J1.355_s848101'),('Source composition',REFINED)]
    ns=[read(OUT/'summaries'/f'native_s{s}.json') for s in (848101,848102,848103)]
    qs=[summary(name) for _,name in models];fig,axs=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    for i,(field,label) in enumerate([('area_fraction','Recruited sheet fraction'),('field_onset_span_ms','Field onset span (ms)')]):
        nv=[np.median([e[field] for e in n['spatial_events']]) for n in ns]
        ax=axs[i];ax.axvspan(min(nv),max(nv),color='black',alpha=.12,label='Native seed medians')
        for y,q in enumerate(qs):
            x=np.array([e[field] for e in q['spatial_events']]);ax.plot(np.quantile(x,[.05,.95]),[y,y],color='#009E73',lw=2);ax.plot(np.median(x),y,'o',color='#009E73')
        ax.set(yticks=np.arange(5),yticklabels=[x[0] for x in models] if i==0 else [],ylim=(4.5,-.5),xlabel=label)
    ax=axs[2];spread=np.array(read(OUT/'comparison.json')['native_pair_max'])
    for j,(lab,col) in enumerate(zip(['Mean rank','Within-shaft order','Participation'],['#0072B2','#E69F00','#D55E00'])):
        xx=[np.array(q['errors_to_native'])[0,j]/spread[j] for q in qs]
        ax.plot(xx,np.arange(5)+(j-1)*.16,'o',color=col,label=lab)
    ax.axvline(1,color='black',ls='--');ax.set(yticks=np.arange(5),yticklabels=[],ylim=(4.5,-.5),xlabel='Error / maximum native pair difference')
    handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,fontsize=10,loc='lower center',bbox_to_anchor=(.75,1),ncol=3)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,fontsize=10,loc='lower center',bbox_to_anchor=(.2,1))
    save(fig,'05_remaining_mismatch')

def main():
    model=np.load(OUT/'model_adaptive1.npz');sizes=np.bincount(model['region'],minlength=6)
    dynamics_figure(sizes);raster_figure(model,sizes);metrics_figure()
    propagation_figure(model,sizes,1);propagation_figure(model,sizes,-1);diagnostic_figure()

if __name__=='__main__':main()
