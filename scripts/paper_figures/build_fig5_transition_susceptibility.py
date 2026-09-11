#!/usr/bin/env python3
"""Figure 5: same-trajectory transition, pre-onset probes, measured Z/M boundary."""
from pathlib import Path
import sys
import json
import hashlib
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import PowerNorm, SymLogNorm, ListedColormap, BoundaryNorm
from matplotlib.patches import Circle, Patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.paper_figures.plot_fig5_dual_core_transition_story import (
    _select_interictal_event, _stage_contract, _spatial_rate_maps, _fit_plane_gradient, _candidate_eta_m)

ART=Path('/home/honglab/leijiaxin/HFOsp')
BASE=Path('/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_dual_core_spatial_z')
COL={'event':'#D5964B','pre':'#9272B5','run':'#BB315B','z':'#326CA5',
     'm':'#D56A31','pop':'#713E96','A':'#EB923E','B':'#2FA5B5',
     'bounded':'#9BB9D1','dependent':'#BDA7D5','unresolved':'#DDDDDD'}


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_npz(path):
    with np.load(path,allow_pickle=False) as a: return {k:a[k] for k in a.files}


def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=8,length=3,width=.7)


def title(fig,box,letter,text):
    fig.text(box[0]-.035,box[1],letter,fontsize=16,fontweight='bold',va='top')
    fig.text(box[0],box[1]-.002,text,fontsize=11,fontweight='bold',va='top')


def shade(ax,stages,labels=False):
    for a,b in stages['returned_interictal_ms']:
        ax.axvspan(a/1000,b/1000,color=COL['event'],alpha=.10,lw=0)
    a,b=np.array(stages['pre_onset_ms'])/1000
    ax.axvspan(a,b,color=COL['pre'],alpha=.10,lw=0)
    onset=stages['onset_ms']/1000
    end=stages['display_ms'][1]/1000
    ax.axvspan(onset,end,color=COL['run'],alpha=.055,lw=0)
    ax.axvline(onset,color=COL['run'],ls='--',lw=.9)
    ax.set_xlim(0,end)
    if labels:
        for x,t,c in [(0.83,'Interictal events','event'),((a+b)/2,'Pre-onset','pre'),
                      ((onset+end)/2,'Tonic runaway','run')]:
            ax.text(x,1.02,t,transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=9,color=COL[c])


def plot_a(ax,arrays,stages):
    raw=np.asarray(arrays['transition_lfp_trace'],float)
    dt=float(arrays['transition_lfp_dt_ms'])
    time=np.arange(len(raw))*dt/1000
    names=arrays['contact_names'].astype(str)
    chosen=['ICL11','ICL8','ICL5','ICL2','SCL9','SCL8','SCL7','SCL6']
    order=[list(names).index(k) for k in chosen]
    baseline=(time>=.5)&(time<1.)
    centered=raw-np.median(raw[baseline],axis=0)
    scale=float(np.percentile(np.abs(centered[time<stages['onset_ms']/1000]),99.5))
    shade(ax,stages,True)
    onset=stages['onset_ms']/1000
    for i,k in enumerate(order):
        y=(len(order)-i-1)*1.1+.65*centered[:,k]/scale
        ax.plot(time[::4],y[::4],color='#252525',lw=.45,rasterized=True)
        keep=time>=onset
        ax.plot(time[keep][::4],y[keep][::4],color=COL['run'],lw=.45,rasterized=True)
    ax.set_yticks(np.arange(len(order))*1.1,chosen[::-1])
    for tick in ax.get_yticklabels(): tick.set_color(COL['A'] if tick.get_text().startswith('ICL') else COL['B'])
    ax.set_ylim(-.4,(len(order)-1)*1.1+1.0)
    ax.set_ylabel('Virtual SEEG · raw LFP proxy',fontsize=9)
    ax.tick_params(axis='x',labelbottom=False)
    style(ax)
    return {'display_contacts':chosen,'selection':'four fixed ICL contacts spanning the shaft plus all four SCL contacts',
            'scaling':'per-contact baseline median subtraction, one shared pre-onset p99.5 absolute amplitude scale',
            'scale_au':scale,'filter':'none','unit':'model current-based LFP proxy, arbitrary units'}


def plot_b(axes,a,stages,eta):
    t=np.asarray(a['transition_spatial_frame_time_ms'])/1000
    axes[0].plot(t,a['transition_rate_E_hz_20ms'],color=COL['pop'],lw=1.1)
    axes[0].set_ylabel('E rate\n(Hz)',color=COL['pop'],fontsize=9)
    axes[0].set_ylim(0,370)
    t=np.asarray(a['slow_time_ms'])/1000
    axes[1].plot(t,a['slow_z_core_mean'],color=COL['z'],lw=1.1,label='Core mean')
    axes[1].plot(t,a['slow_z_surround_mean'],color='#88ABC5',lw=1.1,label='Surround')
    axes[1].set_ylim(.38,1.04);axes[1].set_yticks([.5,1.])
    axes[1].set_ylabel('$Z$',color=COL['z'],fontsize=10)
    axes[1].legend(frameon=False,ncol=2,fontsize=7.5,loc='lower left',borderaxespad=.2)
    axes[2].plot(t,eta*a['slow_m_core_mean'],color=COL['m'],lw=1.1)
    axes[2].set_ylabel('$\\eta_M M$\n(a.u.)',color=COL['m'],fontsize=9)
    axes[2].set_ylim(0,float(np.max(eta*a['slow_m_core_mean']))*1.12)
    for ax in axes: shade(ax,stages);style(ax)
    for ax in axes[:-1]: ax.tick_params(axis='x',labelbottom=False)
    axes[-1].set_xlabel('Time (s)',fontsize=9)


def geometry(ax,contacts,centers,axis_unit=None):
    ax.scatter(contacts[:,0],contacts[:,1],s=13,facecolor='none',edgecolor='#626262',lw=.5,zorder=4)
    for c,color in zip(centers,(COL['A'],COL['B'])):
        ax.add_patch(Circle(c,1.8,fill=False,edgecolor=color,lw=1.0,zorder=5))
    if axis_unit is not None:
        d=7*np.asarray(axis_unit);center=np.array([10.,10.])
        ax.plot([center[0]-d[0],center[0]+d[0]],[center[1]-d[1],center[1]+d[1]],
                color='#777777',lw=.7,ls='--',alpha=.7,zorder=3)
    ax.set(xlim=(0,20),ylim=(0,20),xticks=[0,10,20],yticks=[0,10,20],aspect='equal')
    ax.set_anchor('N')
    ax.set_xlabel('x (mm)',fontsize=8)
    style(ax)


def plot_c(axes,cax,a,stages,selected,centers):
    maps,coords,occupancy=_spatial_rate_maps(a,stages)
    # Both source producers use (y,x); _spatial_rate_maps converts to (x,y).
    onset=a['source_onset_maps_ms'][int(selected['event_index'])].T
    oc=(np.arange(onset.shape[0])+.5)*float(a['source_bin_mm'])
    ox,oy=np.meshgrid(oc,oc,indexing='ij')
    ref=_fit_plane_gradient(onset,ox,oy)
    xx,yy=np.meshgrid(coords,coords,indexing='ij')
    align=[]
    conservation=[]
    plane_r2=[]
    windows=[stages[k] for k in ['interictal_ms','pre_onset_ms','early_ictal_ms']]
    vmax=max(float(np.max(r)) for r in maps)
    for i,(ax,r,name) in enumerate(zip(axes,maps,['Interictal','Pre-onset','Early runaway'])):
        im=ax.imshow(r.T,origin='lower',extent=(0,20,0,20),cmap='magma',
                     norm=PowerNorm(.5,vmin=0,vmax=vmax),interpolation='nearest',rasterized=True)
        geometry(ax,a['contact_xy_mm'],centers)
        d=6*ref;center=np.array([10.,10.])
        ax.annotate('',xy=center+d,xytext=center-d,
                    arrowprops={'arrowstyle':'->','color':'white','lw':1.,'linestyle':'--'})
        align.append(float(abs(np.dot(ref,_fit_plane_gradient(np.log1p(r),xx,yy)))))
        keep=(a['transition_spatial_frame_time_ms']>=windows[i][0])&(a['transition_spatial_frame_time_ms']<windows[i][1])
        spatial_mean=float(np.sum(r*occupancy)/np.sum(occupancy))
        original_mean=float(np.mean(a['transition_rate_E_hz_20ms'][keep]))
        if not np.isclose(spatial_mean,original_mean,atol=1e-3,rtol=1e-5):
            raise RuntimeError('spatial rate no longer conserves the original population readout')
        conservation.append({'spatial_mean_hz':spatial_mean,'source_mean_hz':original_mean})
        design=np.c_[np.ones(r.size),xx.ravel(),yy.ravel()]
        v=np.log1p(r.ravel());coef=np.linalg.lstsq(design,v,rcond=None)[0]
        plane_r2.append(float(1-np.sum((v-design@coef)**2)/np.sum((v-v.mean())**2)))
        ax.set_title(name,fontsize=9,pad=8)
        if i: ax.tick_params(axis='y',labelleft=False)
    axes[0].set_ylabel('y (mm)',fontsize=8)
    cb=axes[0].figure.colorbar(im,cax=cax)
    cb.ax.set_title('E rate\n(Hz)',fontsize=7,pad=7);cb.ax.tick_params(labelsize=7)
    return {'axis_order_fix':'producer (time,y,x) converted to (time,x,y) before occupancy division; imshow transpose only at rendering',
            'shared_rate_vmax_hz':vmax,'color_normalization':'PowerNorm gamma=0.5',
            'reference_axis_unit_xy':ref.tolist(),'absolute_gradient_cosines':align,
            'log_rate_plane_r2':plane_r2,'population_rate_conservation':conservation,
            'reference_axis':'selected returned interictal event recruitment-onset gradient, fixed across all three maps',
            'core_outline':'nominal 1.8-mm circles around frozen centers; membership is budget matched',
            'frame_center_selection_windows_ms':[stages[k] for k in ['interictal_ms','pre_onset_ms','early_ictal_ms']]}


def plot_d(axes,cax,a,meta,centers):
    low=np.asarray(a['low_early_field']).sum(axis=1)
    pre=np.asarray(a['pre_early_field']).sum(axis=1)
    vmax=max(1.,float(np.max(np.abs(np.r_[low,pre]))))
    norm=SymLogNorm(linthresh=1.,vmin=-vmax,vmax=vmax,base=10)
    xy=a['site_xy_mm']
    for ax,v,label in zip(axes[:2],[low,pre],['Low activity','Pre-onset']):
        geometry(ax,a['contact_xy_mm'],centers)
        im=ax.scatter(xy[:,0],xy[:,1],c=v,cmap='RdBu_r',norm=norm,s=67,
                      edgecolor='#666666',lw=.5,zorder=6)
        ax.set_title(label,fontsize=9,pad=8)
    axes[0].set_ylabel('y (mm)',fontsize=8)
    axes[1].tick_params(axis='y',labelleft=False)
    cb=axes[0].figure.colorbar(im,cax=cax,orientation='horizontal',ticks=[-100,0,100,10000])
    cb.set_label('Extra spikes · 0–50 ms',fontsize=8,labelpad=2);cb.ax.tick_params(labelsize=7)
    ax=axes[2]
    for u,v in zip(low,pre): ax.plot([0,1],[u,v],color='#BBBBBB',lw=.6,zorder=1)
    ax.scatter(np.zeros(16),low,s=14,color=COL['z'],zorder=2)
    ax.scatter(np.ones(16),pre,s=14,color=COL['pre'],zorder=2)
    ax.set_xticks([0,1],['Low','Pre']);ax.set_xlim(-.3,1.3)
    ax.set_yscale('symlog',linthresh=1.)
    ax.axhline(0,color='.75',lw=.6);style(ax)
    ax.set_ylabel('Extra spikes',fontsize=8,labelpad=1)
    ax.set_yticks([-100,0,100,10000])
    ax.set_title('Same 16 sites',fontsize=9,pad=8)
    return {'measurement':'signed descendant E spikes: pulse minus exact sham; injected spikes excluded',
            'window_ms':[0,50],'dose_cells':16,'all_sites_retained':True,
            'low_mean':float(low.mean()),'pre_mean':float(pre.mean()),
            'low_median':float(np.median(low)),'pre_median':float(np.median(pre)),
            'sites_with_increased_response':int(np.sum(pre>low)),
            'low_site_responses':low.tolist(),'pre_site_responses':pre.tolist(),
            'window_sensitivity':{
                'low_0_200_mean':float(a['low_full_field'].sum(axis=1).mean()),
                'pre_0_200_mean':float(a['pre_full_field'].sum(axis=1).mean()),
                'low_0_200_median':float(np.median(a['low_full_field'].sum(axis=1))),
                'pre_0_200_median':float(np.median(a['pre_full_field'].sum(axis=1))),
                'sites_increased_0_200':int(np.sum(a['pre_full_field'].sum(axis=1)>a['low_full_field'].sum(axis=1)))},
            'state_times_ms':meta['state_times_ms'],
            'causal_scope':'state-dependent response on one frozen trajectory; no anisotropy rotation/isotropic intervention',
            'future_noise_pairing':'probe and sham share the future within each state; low and pre-onset use their own trajectory futures, not matched independent noise ensembles across states',
            'color_normalization':'signed symlog, linear within +/-1 extra spike, shared symmetric range',
            'color_vmax':vmax}


def plot_e(ax,summary,fold):
    rows=sorted([r for r in summary['rows'] if r['m_gain_scale']==1. and r['preparation']=='pre_fold'],key=lambda r:r['s'])
    x=np.array([r['s'] for r in rows])
    for key,color,label in [('core_a',COL['A'],'Core A'),('core_b',COL['B'],'Core B'),('surround','#527C61','Surround')]:
        ax.plot(x,[r['regional_hz'][key] for r in rows],'-o',ms=3,lw=1.,color=color,label=label)
    ax.plot(x,[r['population_hz'] for r in rows],'-o',ms=4,lw=1.6,color=COL['pop'],label='Population')
    high=sorted([r for r in summary['rows'] if r['m_gain_scale']==1. and r['preparation']=='recruited'],key=lambda r:r['s'])
    ax.plot([r['s'] for r in high],[r['population_hz'] for r in high],'--',color='.45',lw=1.,label='Recruited start')
    ax.axhline(300,color='.7',lw=.6,ls=':')
    ax.axvspan(.428,.429,color=COL['run'],alpha=.20,lw=0)
    ax.annotate('Runaway escape',xy=(.429,310),xytext=(.448,180),fontsize=9,color=COL['run'],
                arrowprops={'arrowstyle':'->','lw':.8,'color':COL['run']})
    ax.set(xlim=(.352,.503),ylim=(0,480),xlabel='Core disinhibition, s = 1 − Zcore',ylabel='Terminal E rate (Hz)')
    ax.legend(frameon=False,ncol=5,fontsize=7.5,loc='lower left',handlelength=1.5,columnspacing=.9)
    style(ax)
    return {'gain_scale':1.,'tail_ms':1000.,'duration_ms':10000.,'same_gate_as_summary':True,
            'population_definition':'unweighted mean of the 100 coarse E-population rates, as in the frozen boundary gate',
            'meaning':'10-second outcomes from two defined initial preparations; connecting lines guide the eye, not equilibrium continuation'}


def combined_code(left,right):
    if 'unresolved' in (left,right): return 3
    if left==right=='bounded': return 0
    if left==right=='tonic_runaway': return 2
    return 1


def plot_f(ax,summary):
    s=np.array(summary['config']['s_values']);g=summary['config']['m_gain_scales']
    z=np.zeros((len(g),len(s)),int)
    for i,gain in enumerate(g):
        for j,value in enumerate(s):
            r=[r for r in summary['rows'] if r['m_gain_scale']==gain and r['s']==value]
            if len(r)!=2: raise RuntimeError('map cell lacks both preparations')
            z[i,j]=combined_code(r[0]['state'],r[1]['state'])
    # Midpoint cell edges retain nonuniform spacing around the measured bracket.
    e=np.r_[s[0]-(s[1]-s[0])/2,(s[1:]+s[:-1])/2,s[-1]+(s[-1]-s[-2])/2]
    colors=[COL['bounded'],COL['dependent'],COL['run'],COL['unresolved']]
    ax.pcolormesh(e,np.arange(len(g)+1)-.5,z,cmap=ListedColormap(colors),
                  norm=BoundaryNorm(np.arange(5)-.5,4),shading='flat',edgecolors='white',linewidth=.4)
    ax.set_yticks(range(len(g)),[f'{v:g}×' for v in g])
    ax.set_ylabel('Adaptation gain',fontsize=9)
    ax.set_xlabel('Core disinhibition, s = 1 − Zcore',fontsize=9)
    ax.set_xlim(e[0],e[-1]);ax.set_xticks([.36,.40,.44,.48,.5]);style(ax)
    legend_labels=['Both bounded','Preparation dependent','Both runaway','Unresolved']
    ax.legend(handles=[Patch(color=colors[i],label=legend_labels[i]) for i in np.unique(z)],
              frameon=False,ncol=2,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.23),columnspacing=1.4)
    return {'codes':z.tolist(),'labels':['both bounded','preparation dependent','both runaway','unresolved'],
            'meaning':'two tested preparations at each sampled point, not exhaustive attractor counting',
            's_values':s.tolist(),'m_gain_scales':g,'interpolation':'none; midpoint rectangles around sampled s values'}


def save_all(fig,stem):
    for ext in ['png','pdf','svg']:
        fig.savefig(stem.with_suffix('.'+ext),dpi=350,facecolor='white')
    return {ext:{'path':str(stem.with_suffix('.'+ext)),'sha256':digest(stem.with_suffix('.'+ext))} for ext in ['png','pdf','svg']}


def response_fields(a,centers,out):
    """Response DESTINATION fields, distinct from the stimulation-site map."""
    pos=a['positions_E'];edges=np.linspace(0,20,41)
    occupancy=np.histogram2d(pos[:,0],pos[:,1],bins=(edges,edges))[0]
    fields=[]
    for key in ['low_early_field','pre_early_field','low_full_field','pre_full_field']:
        mean=a[key].mean(axis=0)
        total=np.histogram2d(pos[:,0],pos[:,1],bins=(edges,edges),weights=mean)[0]
        numerator=gaussian_filter(total,.7)
        denominator=gaussian_filter(occupancy,.7)
        fields.append(np.divide(numerator,denominator,out=np.zeros_like(total),where=denominator>0))
    fig,axes=plt.subplots(2,2,figsize=(7.8,6.5))
    for row in range(2):
        vmax=max(float(np.max(np.abs(x))) for x in fields[row*2:row*2+2])
        for col in range(2):
            ax=axes[row,col]
            im=ax.imshow(fields[row*2+col].T,origin='lower',extent=(0,20,0,20),
                         cmap='RdBu_r',vmin=-vmax,vmax=vmax,interpolation='nearest',rasterized=True)
            geometry(ax,a['contact_xy_mm'],centers)
            ax.set_title(('Low activity' if col==0 else 'Pre-onset')+f' · 0–{50 if row==0 else 200} ms',fontsize=10)
            if col==0: ax.set_ylabel('y (mm)')
        cb=fig.colorbar(im,ax=axes[row].tolist(),fraction=.035,pad=.03)
        cb.set_label('Extra spikes per local E cell',fontsize=9)
    fig.subplots_adjust(left=.09,right=.83,bottom=.09,top=.94,hspace=.34,wspace=.35)
    outputs=save_all(fig,out/'fig5-perturbation-response-fields')
    plt.close(fig)
    return outputs,{'mean_over':16,'spatial_bin_mm':.5,'display_gaussian_sigma_bins':.7,
                    'smoothing':'smooth signed spike numerator and cell-count denominator separately, then divide',
                    'windows_ms':[[0,50],[0,200]],'scale':'shared symmetric linear scale within each row',
                    'meaning':'where extra response spikes occur after averaging all 16 stimulation sites; not a map of injection-site susceptibility'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out-dir',type=Path,default=ROOT/'results/paper-ready-figure/fig5_transition_susceptibility_v3/figures')
    args=parser.parse_args()
    worker=ART/'results/topic4_sef_hfo/data_driven_dual_core_zm_transition/timescale/workers/rev21_ts_tz3000_ta500_topology_2542_dynamics_2642.json'
    wm=json.loads(worker.read_text());a=load_npz(worker.with_suffix('.npz'))
    if digest(worker.with_suffix('.npz'))!=wm['arrays']['sha256']: raise RuntimeError('worker arrays changed')
    configpath=ROOT/'config/topic4_rev21_dual_core_zm_transition.json'
    manifestpath=worker.parent.parent/'candidate_manifest.json'
    eta=_candidate_eta_m(wm,json.loads(manifestpath.read_text()))
    bp=BASE/'bifurcation/dualcore_spatial_z_bifurcation.json';b=json.loads(bp.read_text())
    centers=np.asarray(b['substrate']['centers_mm'])
    pp=BASE/'perturbation/preonset_20260905/state_contrast.json';pm=json.loads(pp.read_text())
    if pm['status']!='PRE_ONSET_PAIRED_PERTURBATION_COMPLETE' or digest(pp.with_suffix('.npz'))!=pm['npz']['sha256']:
        raise RuntimeError('pre-onset probe package failed')
    p=load_npz(pp.with_suffix('.npz'))
    sp=BASE/'upper_runaway_map_20260905/summary.json';summary=json.loads(sp.read_text())
    if len(summary['rows'])!=80: raise RuntimeError('incomplete upper map')
    foldpath=BASE/'runaway_boundary/localized_to_global_fold_m1.json';fold=json.loads(foldpath.read_text())
    selected=_select_interictal_event(wm,a);stages=_stage_contract(wm,selected)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.7,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=(14.2,10.4),facecolor='white')
    fig.text(.06,.984,'From recurrent interictal events to tonic runaway',fontsize=16,fontweight='bold',va='top')
    # A/B share exact horizontal limits and physical axis width.
    title(fig,(.065,.934),'A','Spontaneous transition in the two-core SNN')
    axa=fig.add_axes([.065,.650,.485,.247]);ma=plot_a(axa,a,stages)
    title(fig,(.065,.604),'B','Population activity and inhibitory state')
    bax=[fig.add_axes([.065,y,.485,.058]) for y in [.509,.439,.369]]
    plot_b(bax,a,stages,eta)
    title(fig,(.62,.934),'C','Spatial recruitment across the same trajectory')
    caxs=[fig.add_axes([.62+i*.115,.673,.095,.207]) for i in range(3)]
    cb=fig.add_axes([.957,.750,.009,.130]);mc=plot_c(caxs,cb,a,stages,selected,centers)
    title(fig,(.62,.604),'D','Susceptibility at the same stimulation sites')
    dax=[fig.add_axes([.62+i*.128,.394,.106,.16]) for i in range(2)]
    dax.append(fig.add_axes([.901,.394,.059,.16]))
    dcb=fig.add_axes([.637,.353,.198,.008]);md=plot_d(dax,dcb,p,pm,centers)
    title(fig,(.065,.293),'E','Global recruitment jumps as inhibition weakens')
    axe=fig.add_axes([.065,.075,.485,.176]);me=plot_e(axe,summary,fold)
    title(fig,(.62,.293),'F','Adaptation and initial state shift the boundary')
    axf=fig.add_axes([.62,.095,.34,.157]);mf=plot_f(axf,summary)
    stem=out/'fig5-transition-susceptibility-v3'
    outputs=save_all(fig,stem)
    # Source-based supplement: carried-state transition and response time course.
    edgepath=BASE/'runaway_boundary/native_edge_tracking_0p428_to_0p429.json'
    edge=load_npz(edgepath.with_suffix('.npz'))
    supp,axs=plt.subplots(1,2,figsize=(10,3.4))
    t=edge['time_ms']/1000
    axs[0].plot(t,edge['source_trace_hz'],color=COL['z'],lw=.8,label='s = 0.428')
    axs[0].plot(t,edge['target_trace_hz'],color=COL['run'],lw=.8,label='s = 0.429 · carried state')
    axs[0].set(xlabel='Time after setting s (s)',ylabel='Population E rate (Hz)',title='Full hidden state carried across the jump')
    axs[0].legend(frameon=False,fontsize=8)
    dt=float(np.diff(p['time_after_pulse_ms'])[0]);width=int(round(10/dt))
    per=np.asarray(p['pre_excess_spikes_per_step']);n=per.shape[1]//width
    binned=per[:,:n*width].reshape(16,n,width).sum(axis=2)
    for row in binned: axs[1].plot((np.arange(n)+.5)*10,row,color=COL['pre'],alpha=.25,lw=.8)
    axs[1].plot((np.arange(n)+.5)*10,binned.mean(axis=0),color=COL['pop'],lw=1.5,label='Mean across 16 sites')
    axs[1].axhline(0,color='.6',lw=.6)
    axs[1].set(xlabel='Time after pre-onset pulse (ms)',ylabel='Extra spikes / 10 ms',title='Probe minus exact sham')
    axs[1].legend(frameon=False,fontsize=8)
    for ax in axs: style(ax)
    supp.tight_layout();supp_outputs=save_all(supp,out/'fig5-transition-susceptibility-diagnostics')
    response_outputs,response_meta=response_fields(p,centers,out)
    sources=[worker,worker.with_suffix('.npz'),configpath,manifestpath,bp,pp,pp.with_suffix('.npz'),sp,foldpath,edgepath,edgepath.with_suffix('.npz'),Path(__file__)]
    sources.append(ROOT/'scripts/paper_figures/plot_fig5_dual_core_transition_story.py')
    metadata={'status':'FIG5_TRANSITION_SUSCEPTIBILITY_V3_CANDIDATE','author_acceptance':False,
              'substrate':'dualcore_s39 + Joint=1.25; predecessor of rev22, not the rev22 optimum',
              'topology_seed':2542,'dynamics_seed':2642,'stages':stages,
              'A_B_same_time_axis':True,'panel_A':ma,'panel_C':mc,'panel_D':md,'panel_E':me,'panel_F':mf,
              'panel_B':{'eta_m':eta,'Z':'pooled union-core mean and surround mean; does not assert core A = core B',
                         'M':'pooled core mean adaptation current eta_M*M'},
              'sources':{str(q):digest(q) for q in sources},'outputs':outputs,'diagnostic_outputs':supp_outputs,
              'response_field_outputs':response_outputs,'response_field_contract':response_meta,
              'claim_boundary':'single development substrate; full SNN A-D and deterministic frozen-Z dynamic-M coarse E-F. Operational tonic runaway only. No specific local bifurcation type, clinical seizure reproduction, cohort prediction, or causal anisotropy susceptibility established.'}
    stem.with_name(stem.name+'-metadata').with_suffix('.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    # README is written after both real figures have been generated.
    (out/'README.md').write_text(
        '### fig5-transition-susceptibility-v3.png\n'
        'A/B 是同一条 two-core、连续 OU、Z/M 开启的 40,000-cell 轨迹，保留间期事件、转变前和 early runaway。A 是未经带通的电流型 LFP proxy，8 个固定触点；B 的 Z/M 为双 core 合并均值，不能解释为两核始终相等。C 修复旧版 y/x 误置后展示同轨迹的空间招募；D 使用同 16 个位置、同 16-cell 脉冲与精确 sham，比的是刺激位置的额外后继放电，不是自然活动热图。E/F 使用 realized delays 和 0.1 ms 步长，展示 10 s 有限初值响应，不能把区域边界称为已定型的局部分叉。\n\n'
        '**关注点**：D 的 pre-onset 在 2015.4 ms，整个 200 ms 响应窗都早于 onset；E/F 同时保留未定与初值依赖。此版继承 Joint=1.25，不是 rev22 新最优点，尚未作者接受。\n\n'
        '### fig5-transition-susceptibility-diagnostics.png\n'
        '左图展示 s=0.428 状态完整传到 0.429 后的群体率。右图展示 pre-onset 扰动减 sham 的逐时间响应，每条浅线是一个刺激位置。\n\n'
        '**关注点**：位置是同一网络内重复测量，不作为独立患者或多 seed 检验；短窗响应不能替代长时转变概率。\n\n'
        '### fig5-perturbation-response-fields.png\n'
        '每行比较相同窗口中 low 与 pre-onset 的 probe-minus-sham 响应场，分别为 0–50 ms 与 0–200 ms；每张图平均全部 16 个刺激位置。每个格点是当地 E 神经元的额外后继放电数，同一行共用 signed 线性色标。\n\n'
        '**关注点**：这里画响应放电发生在哪里；主图 D 画从哪里施加刺激更易产生响应，两者不能混为一谈。负值和所有位置均被保留。\n',encoding='utf-8')
    print(json.dumps({'outputs':outputs,'spatial_cosines':mc['absolute_gradient_cosines'],
                      'probe_summary':md,'map_codes':mf['codes']},indent=2))


if __name__=='__main__': main()
