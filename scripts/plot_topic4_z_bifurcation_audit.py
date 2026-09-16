#!/usr/bin/env python3
"""Scientific diagnostics of the fixed C E-only Z transition, using actual results."""
from validate_topic4_fixed_rate_base import ROOT,read,write
from topic4_e_only_z_rate import EOnlySystem
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import argparse

OUT=ROOT/'results/topic4_sef_hfo/z_transition_bifurcation_audit_v1'
REF=ROOT/'results/topic4_sef_hfo/autonomous_z_manual_restore_v1'
FIG=OUT/'figures'
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':180})
COL={'native':'#222222','replay':'#527bb0','auto':'#b94272','fixed':'#858585','z':'#295986'}


def save(fig,name):
    FIG.mkdir(exist_ok=True);fig.savefig(FIG/f'{name}.png',bbox_inches='tight');fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight');plt.close(fig)


def reduced(name):
    a=np.load(OUT/'rate'/f'{name}.npz');f=a['fields_hz'];return a,np.average(f[:,0],axis=1,weights=a['count_e']),np.average(f[:,1],axis=1,weights=a['count_i'])


def smooth(y,binms):
    n=len(y)//binms;return np.asarray(y[:n*binms]).reshape(n,binms).mean(1)


def validation():
    a=np.load(REF/'trajectory.npz');native=smooth(a['rate_e_hz'],50);time=(np.arange(len(native))+.5)*.005
    fig,axes=plt.subplots(4,1,figsize=(13,10.8),sharex=True,gridspec_kw={'height_ratios':[1.1,1.4,1,1]})
    groups=a['sample_groups'];ids=np.r_[np.arange(0,240,4),np.arange(240,300,4)]
    spikes=a['sample_spikes'][:,ids];tt,nn=np.where(spikes)
    axes[0].scatter(tt*.0001,nn,s=.35,c=np.where(ids[nn]<240,'#292929','#447ea8'),rasterized=True)
    axes[0].set(ylim=(-1,len(ids)),ylabel='Sampled neuron',title='A  Native SNN: original neuron-wise Z, unchanged OU input')
    axes[0].axhline(59.5,c='0.7',lw=.6);axes[0].text(.1,65,'E: black; I: blue',fontsize=10,color='#447ea8')
    axes[1].plot(time,native,c=COL['native'],lw=.9,label='Native SNN')
    vals={}
    for name,label,col in [('native_replay_expected','Rate: native Z replay',COL['replay']),
                           ('autonomous_gaussian_expected','Rate: autonomous Z closure',COL['auto']),
                           ('fixed_expected','Rate: Z fixed at 1',COL['fixed'])]:
        b,e,i=reduced(name);vals[name]=(b,e,i);v=smooth(e,5)
        axes[1].plot((np.arange(len(v))+.5)*.005,v,c=col,lw=.85,label=label)
    axes[1].set(ylabel='Mean E rate (Hz)',title='B  Fast response transfers; autonomous depletion is too fast',ylim=(0,450));axes[1].legend(ncol=2,loc='upper left',fontsize=10)
    zn=np.average(a['z_field_10ms'],axis=1,weights=a['cell_e_counts']);zt=a['z_time_ms']/1000
    axes[2].plot(zt,zn,c=COL['native'],lw=1.8,label='Native mean Z')
    b,e,i=vals['autonomous_gaussian_expected'];axes[2].plot(np.arange(len(e))*.001,np.average(b['z'],axis=1,weights=b['count_e']),c=COL['auto'],lw=1.5,label='Autonomous rate closure')
    axes[2].set(ylabel='Mean E-target Z',ylim=(.48,1.03),title='C  Original Z law retained; cell averaging changes its kinetics');axes[2].legend(loc='lower left',fontsize=10)
    fields=a['field_e_count_1ms'].astype(float)/a['cell_e_counts'][None,:]*1000
    n=len(fields)//10;f10=fields[:n*10].reshape(n,10,400).mean(1)
    # Fraction of neurons residing in cells above a displayed rate threshold; not a new propagation metric.
    for threshold,color in [(50,'#bf742e'),(200,'#913168')]:
        y=np.average(f10>threshold,axis=1,weights=a['cell_e_counts'])
        axes[3].plot((np.arange(n)+.5)*.01,y,c=color,label=f'Cells > {threshold} Hz')
    axes[3].set(ylabel='Spatial recruitment\n(neuron-weighted fraction)',xlabel='Time (s)',ylim=(-.03,1.03),title='D  Sustained activity recruits the sheet, rather than only increasing a core average');axes[3].legend(loc='upper left',fontsize=10)
    for ax in axes:
        ax.axvspan(10.68,11.68,color='#53b491',alpha=.16);ax.axvline(10.68,c='#298d6c',ls='--',lw=1);ax.set_xlim(0,13.68)
    axes[0].text(10.75,8,'External Z refill',color='#22755a',fontsize=10)
    fig.suptitle('Fixed C substrate | Z evolves endogenously, then is externally restored',fontsize=16,y=1.01)
    fig.tight_layout(h_pad=1.2);save(fig,'native_and_reduced_z_transition')
    rows=[]
    for name,(b,e,i) in vals.items():
        e10=smooth(e,10);good=e10>=200;conv=np.convolve(good.astype(int),np.ones(20,dtype=int),'valid');ix=np.flatnonzero(conv==20)
        rows.append({'name':name,'first_200ms_above_200Hz_end_ms':float((ix[0]+20)*10) if len(ix) else None,
                     'mean_Z_at_10680ms':float(np.average(b['z'][10679],weights=b['count_e'])),
                     'high_window_hz':float(e[10180:10680].mean())})
    write(OUT/'validation_comparison.json',{'native_trigger_end_ms':10180,'native_mean_Z_at_refill':float(zn[np.argmin(abs(zt-10.68))]),'rate_rows':rows,
          'recruitment_definition':'10-ms cell-average E rate above 50 or 200 Hz, weighted by number of E neurons; sensitivity display, not patient propagation recovery.'})


def bounded_mean(r,weights,target,cap):
    """Unique common additive shift, clipped to the physical refractory-rate interval."""
    if abs(np.average(r,weights=weights)-target)<1e-14:return r.copy()
    lo=-cap;hi=cap
    for _ in range(48):
        mid=(lo+hi)/2
        if np.average(np.clip(r+mid,0,cap),weights=weights)<target:lo=mid
        else:hi=mid
    return np.clip(r+(lo+hi)/2,0,cap)


def conditional_planes():
    assert read(OUT/'phase_replay_status.json')['status']=='COMPLETE'
    s=EOnlySystem();m=s.m;n=s.n;a=np.load(OUT/'rate/native_replay_expected_phase_phase_slices.npz')
    fig,axes=plt.subplots(1,3,figsize=(15.8,5.2));rows=[]
    for ax,tm,label in zip(axes,[9000,10300,11100],['Event / recruitment','Sustained recruitment','External restoration']):
        key=f't{tm}';r=a[key+'_r'];c=a[key+'_current'];z=a[key+'_z'];ne=a[key+'_expected_e'];ni=a[key+'_expected_i']
        def drift(rr):
            e,i=rr[:n],rr[n:];te,ti=m.tau_mem_e_ms,m.tau_mem_i_ms
            ex=np.r_[te*(m.v_ee@e+m.j_ext_e_mv**2*ne),ti*(m.v_ie@e+m.j_ext_i_mv**2*ni)]
            inh=np.r_[te*z*z*(m.v_ei@i),ti*(m.v_ii@i)]
            mu=np.r_[c[0]-z*c[1]+c[4],c[2]-c[3]+c[5]]
            d=(s.phi(mu,ex,inh)-rr)/s.tr
            return np.array([np.average(d[:n],weights=m.count_e),np.average(d[n:],weights=m.count_i)])*1e6
        xy=np.array([np.average(r[:n],weights=m.count_e),np.average(r[n:],weights=m.count_i)])*1000
        d0=drift(r);actual=(a[key+'_next_r']-r)/s.dt
        observed=np.array([np.average(actual[:n],weights=m.count_e),np.average(actual[n:],weights=m.count_i)])*1e6
        error=float(np.max(abs(d0-observed)));assert error<1e-6,error
        micro=a['microtrajectory'];sel=abs(micro[:,0]-tm)<=50;trail=micro[sel]
        xx=np.average(trail[:,1:101],axis=1,weights=m.count_e)*1000;yy=np.average(trail[:,101:],axis=1,weights=m.count_i)*1000
        endpoint=xy+d0*np.array([s.tr[0],s.tr[100]])*.001
        xmax=min(500,max(xy[0],xx.max(),endpoint[0])*1.22+5);ymax=min(1000,max(xy[1],yy.max(),endpoint[1])*1.22+5)
        xs=np.linspace(0,xmax,51);ys=np.linspace(0,ymax,51);U=np.zeros((51,51));V=U.copy()
        er=[bounded_mean(r[:n],m.count_e,x*.001,.5) for x in xs]
        ir=[bounded_mean(r[n:],m.count_i,y*.001,1.) for y in ys]
        for j in range(51):
            for k in range(51):U[j,k],V[j,k]=drift(np.r_[er[k],ir[j]])
        X,Y=np.meshgrid(xs,ys);speed=np.sqrt((U/xmax)**2+(V/ymax)**2)
        ax.quiver(X[::4,::4],Y[::4,::4],(U/np.maximum(speed,1e-12))[::4,::4],(V/np.maximum(speed,1e-12))[::4,::4],
                  color='0.65',angles='xy',scale_units='xy',scale=22,width=.003,headwidth=3.5)
        for field,color in [(U,'#b13c76'),(V,'#2a7d88')]:
            if field.min()<0<field.max():ax.contour(X,Y,field,levels=[0],colors=[color],linewidths=2)
        ax.plot(xx,yy,c='#2e2e2e',lw=1.1,zorder=4);ax.scatter(xx[0],yy[0],marker='s',s=16,c='0.45',zorder=5)
        for k in [100,350,650,850]:
            if k+8<len(xx):ax.annotate('',xy=(xx[k+8],yy[k+8]),xytext=(xx[k],yy[k]),arrowprops=dict(arrowstyle='->',color='k',lw=1.2))
        ax.scatter(*xy,c='#eeaa32',edgecolor='black',s=60,zorder=7)
        ax.set(xlim=(0,xmax),ylim=(0,ymax),xlabel='Mean E rate (Hz)',title=f'{label}\nt = {tm/1000:.2f} s; mean Z = {np.average(z,weights=m.count_e):.3f}')
        rows.append({'time_ms':tm,'mean_rates_hz':xy.tolist(),'exact_native_step_drift_hz_per_s':d0.tolist(),'center_identity_absolute_error':error,
                     'zero_contours_present':{'E':bool(U.min()<0<U.max()),'I':bool(V.min()<0<V.max())}})
        np.savez_compressed(OUT/f'conditional_plane_{tm}ms.npz',X=X,Y=Y,U=U,V=V,trajectory_hz=np.c_[xx,yy],trajectory_time_ms=trail[:,0],center_hz=xy)
    axes[0].set_ylabel('Mean I rate (Hz)')
    handles=[Line2D([0],[0],c='#b13c76',lw=2,label='d mean E / dt = 0'),Line2D([0],[0],c='#2a7d88',lw=2,label='d mean I / dt = 0'),
             Line2D([0],[0],c='k',lw=1.3,label='Actual delayed-model trajectory: +/-50 ms'),Line2D([0],[0],marker='o',c='none',markerfacecolor='#eeaa32',markeredgecolor='k',label='Exact matched state')]
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.085),fontsize=11)
    fig.suptitle('Instantaneous conditional phase planes: filters, delay history, input and Z held at each marked state',fontsize=14,y=1.04)
    fig.text(.5,-.13,'Nullclines are projected zero-drift curves, not full-network equilibrium branches. Hidden states evolve along the black trajectory.',ha='center',fontsize=11)
    fig.tight_layout();save(fig,'conditional_nullclines_vector_field_trajectory')
    write(OUT/'conditional_phase_qa.json',{'status':'PASS','rows':rows,
          'plane_definition':'Vary cell rates by a common additive E or I offset followed by refractory clipping, choosing offset to match requested neuron-weighted mean. Other hidden states fixed at the exact update.',
          'time_convention':'Currents after the native-step filter update, rates before the rate update. Projected finite step equals the displayed instantaneous drift at the gold point.',
          'limitation':'Not an invariant 2D subsystem. Full delayed trajectory need not follow this fixed vector field away from the snapshot; zero curves need not locate full equilibria.'})


def native_controls(first_only=False):
    status=read(OUT/'native_batch_status.json')
    if not first_only:assert status['status']=='COMPLETE'
    times=[8000,9400,9800,10180,10680];fig,axs=plt.subplots(3,5,figsize=(16,9.5),gridspec_kw={'height_ratios':[1,1,1.35]})
    centers=np.array(read(ROOT/'config/topic4_rate_model_dynamics_validation_v1.json')['candidate']['node_field']['centers_mm'])
    metrics=[]
    for col,tm in enumerate(times):
        name=f'frozen_t{tm}';a=np.load(OUT/'native'/f'{name}.npz');row=read(OUT/'native'/f'{name}.json')
        e=smooth(a['rate_e_hz'],50);t=(np.arange(len(e))+.5)*.005
        axs[0,col].plot(t,e,c='#262626',lw=.8);axs[0,col].set(xlim=(0,2),ylim=(0,460),title=f'Freeze at {tm/1000:g} s\nmean Z = {row["initial_Z_mean"]:.3f}')
        axs[0,col].text(.06,.95,f'Late E: {row["late_E_mean_hz"]:.0f} Hz\nQuiet bins: {row["late_E_quiet_fraction"]:.0%}',transform=axs[0,col].transAxes,va='top',fontsize=10)
        ids=np.arange(0,300,4);sp=a['sample_spikes'][-5000:,ids];st,sn=np.where(sp)
        axs[1,col].scatter(st*.0001+1.5,sn,s=.4,c=np.where(ids[sn]<240,'#222222','#477eaa'),rasterized=True)
        axs[1,col].set(xlim=(1.5,2),ylim=(-1,len(ids)),xlabel='Continuation time (s)')
        cells=a['field_e_count_1ms'][-1000:].mean(0)/a['cell_e_counts']*1000
        im=axs[2,col].imshow(cells.reshape(20,20),origin='lower',extent=(0,20,0,20),vmin=0,vmax=460,cmap='magma',interpolation='nearest')
        axs[2,col].scatter(centers[:,0],centers[:,1],s=100,facecolors='none',edgecolors='#65dfdf',lw=1.5)
        axs[2,col].set(xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20])
        if col:
            for rr in range(3):axs[rr,col].tick_params(labelleft=False)
        metrics.append(dict(name=name,late_spatial_fraction_above_50Hz=float(np.average(cells>50,weights=a['cell_e_counts'])),late_spatial_fraction_above_200Hz=float(np.average(cells>200,weights=a['cell_e_counts']))))
    axs[0,0].set_ylabel('Mean E rate (Hz)');axs[1,0].set_ylabel('Sampled neuron');axs[2,0].set_ylabel('y (mm)')
    fig.suptitle('Native SNN with each actual neuron-wise Z field frozen | full fast state and OU history carried forward',fontsize=15,y=1.01)
    fig.tight_layout();fig.colorbar(im,ax=axs[2,:],shrink=.75,label='Late E rate (Hz)',pad=.015);save(fig,'native_frozen_z_raster_and_recruitment')
    write(OUT/'native_spatial_metrics.json',metrics)
    if first_only:return
    fig,axs=plt.subplots(3,1,figsize=(12,9),sharex=False)
    original=np.load(REF/'trajectory.npz');noref=np.load(OUT/'native/autonomous_no_refill_t10680.npz')
    for arr,label,color in [(original['rate_e_hz'][106800:126800],'External Z refill','#247e69'),(noref['rate_e_hz'],'Continue original Z ODE','#b34574')]:
        e=smooth(arr,50);axs[0].plot((np.arange(len(e))+.5)*.005+10.68,e,label=label,c=color,lw=1.2)
    axs[0].axvspan(10.68,11.68,color='#6bad92',alpha=.12);axs[0].set(ylabel='Mean E rate (Hz)',xlabel='Absolute time (s)',title='A  Same checkpoint and future innovations; only the Z intervention differs');axs[0].legend(fontsize=10)
    axs[1].plot(noref['z_summary_1ms'][:,0],c='#b34574',label='Continue Z ODE')
    zz=np.average(original['z_field_10ms'],axis=1,weights=original['cell_e_counts']);sel=(original['z_time_ms']>=10680)&(original['z_time_ms']<12680)
    axs[1].plot(original['z_time_ms'][sel]-10680,zz[sel],c='#247e69',label='External refill');axs[1].set(xlabel='Time since shared checkpoint (ms)',ylabel='Mean Z',ylim=(.35,1.03))
    names=['frozen_t9800','frozen_t9800_V-0.05','frozen_t9800_V+0.05','frozen_t9800_future9108502','frozen_t9800_future9108503']
    labels=['Original','E V -0.05 mV','E V +0.05 mV','Future noise 1','Future noise 2']
    for name,label,col in zip(names,labels,['k','#996f36','#b54d51','#478291','#795ba4']):
        a=np.load(OUT/'native'/f'{name}.npz');e=smooth(a['rate_e_hz'],500)
        axs[2].plot((np.arange(len(e))+.5)*.05,e,c=col,lw=1.1,label=label)
    axs[2].set(xlabel='Continuation time (s)',ylabel='Mean E rate (Hz)',title='B  Frozen 9.8-s Z field: small voltage perturbations and changed future noise');axs[2].legend(ncol=3,fontsize=10)
    fig.tight_layout(h_pad=1.5);save(fig,'native_counterfactual_and_sensitivity')
    write(OUT/'native_spatial_metrics.json',metrics)


def critical_planes():
    assert read(OUT/'critical_plane_status.json')['status']=='COMPLETE'
    fig,axs=plt.subplots(1,3,figsize=(15,5.2),gridspec_kw={'width_ratios':[1,1,1.25]});checks=[]
    for ax,side,title,color in zip(axs[:2],[1,-1],['Stable side','Unstable side'],['#3972a2','#b34574']):
        a=np.load(OUT/'reduced_bifurcation'/f'critical_plane_{side:+d}.npz');row=read(OUT/'reduced_bifurcation'/f'critical_plane_{side:+d}.json');A=a['A_per_s'];t=a['records'][:,0]/1000;xy=a['records'][:,1:3]
        # Canonical complex-mode coordinates remove arbitrary ellipse distortion from the QR basis.
        ev,vectors=np.linalg.eig(A);v=vectors[:,np.argmax(ev.imag)];K=np.c_[v.real,-v.imag];T=np.linalg.inv(K)
        initial=T@np.array([1.,0.]);norm0=np.linalg.norm(initial);initial/=norm0;xy=xy@T.T/norm0;A=T@A@K
        limit=max(1.15,np.max(abs(xy))*1.15);v=np.linspace(-limit,limit,31);X,Y=np.meshgrid(v,v);U=A[0,0]*X+A[0,1]*Y;V=A[1,0]*X+A[1,1]*Y
        speed=np.hypot(U,V);ax.quiver(X[::3,::3],Y[::3,::3],(U/np.maximum(speed,1e-12))[::3,::3],(V/np.maximum(speed,1e-12))[::3,::3],color='0.7',angles='xy',scale_units='xy',scale=7/limit,width=.003)
        ax.contour(X,Y,U,[0],colors=['#b13c76'],linewidths=1.5);ax.contour(X,Y,V,[0],colors=['#2a7d88'],linewidths=1.5)
        ax.plot(xy[:,0],xy[:,1],c=color,lw=1.1);ax.scatter(*xy[0],marker='s',c=color,s=24);ax.scatter(*xy[-1],c=color,s=24)
        ax.set(xlabel='Critical-mode coordinate a',ylabel='Critical-mode coordinate b',xlim=(-limit,limit),ylim=(-limit,limit),title=f'{title}: mean Z = {row["q"]:.6f}\nRe lambda = {row["lambda_real_per_s"]:+.3f} /s');ax.set_aspect('equal')
        norm=np.linalg.norm(xy,axis=1);axs[2].plot(t,norm,c=color,label=f'{title}: full nonlinear trajectory')
        # Compare the actual discrete linear map, avoiding an ellipse-dependent envelope convention.
        Alin=np.eye(2)+A*.0001;ab=initial.copy();values=[]
        for k in range(60000):
            ab=Alin@ab
            if (k+1)%10==0:values.append(np.linalg.norm(ab))
        axs[2].plot(t,values,c=color,ls='--',lw=1,label=f'{title}: local linear prediction')
        checks.append({'side':side,'maximum_relative_radius_error':float(np.max(abs(norm-np.array(values))/np.maximum(values,1e-12))),
                       'observed_end_radius':float(norm[-1]),'linear_end_radius':float(values[-1]),'coordinate_transform_condition':float(np.linalg.cond(K))})
    axs[2].set(xlabel='Time (s)',ylabel='Distance in critical-mode plane',yscale='log',title='Growth/decay of the same small perturbation');axs[2].legend(fontsize=9,loc='upper left')
    fig.suptitle('Low-activity instability: critical eigenspace nullclines and vector field with full delayed-model trajectories',fontsize=14,y=1.03)
    fig.text(.5,-.025,'Pink: da/dt = 0; teal: db/dt = 0. a and b mix rates, synaptic filters and delay history; this is a local mode plane, not a global E-I closure.',ha='center',fontsize=11)
    fig.tight_layout();save(fig,'hopf_critical_mode_nullclines_and_trajectories')
    write(OUT/'critical_plane_plot_qa.json',{'status':'PASS','checks':checks,'coordinates':'Canonical real/imaginary complex eigenmode coordinates, with initial radius one; a linear coordinate change of the saved QR-plane projection.'})


def diagrams():
    folder=OUT/'reduced_bifurcation';fig,axs=plt.subplots(2,2,figsize=(14.5,10));s=EOnlySystem()
    colors={'uniform':'#527bb0','native_path':'#b34574'};labels={'uniform':'Uniform E-target Z','native_path':'Checkpoint-interpolated Z field'}
    for kind in colors:
        a=np.load(folder/f'{kind}_tau20.6116_branches.npz');col=colors[kind]
        for branch in ['low','high']:
            ids=a['branch']==branch;y=np.average(a['r'][ids,:100],axis=1,weights=s.m.count_e)*1000
            axs[0,0].plot(a['q'][ids],y,c=col,lw=1.6,ls='-' if branch=='low' else '--',label=labels[kind] if branch=='low' else None)
            row=read(folder/f'{kind}_tau20.6116_{branch}_fold.json');axs[0,0].scatter(row['q'],row['E_mean_hz'],c=col,marker='s',s=45)
        h=read(folder/f'hopf_{kind}_tau20.6116_dt0.1.json');axs[0,0].scatter(h['q'],h['E_mean_hz'],marker='*',s=170,c=col,zorder=6)
        track=h['track'];axs[1,0].plot([x['q']-h['q'] for x in track],[x['real_per_s'] for x in track],'-o',c=col,label=labels[kind])
        # Newly traced neighboring tau points replace failed uncontinued guesses.
        curves=[]
        for file in folder.glob(f'hopf_{kind}_tau*_dt0.1.json'):
            row=read(file)
            if 'q' in row:curves.append((row['tau_ms'],row['q'],row['frequency_hz']))
        curves=np.array(sorted(curves));axs[0,1].plot(curves[:,0],curves[:,1],c=col,label=labels[kind]);axs[0,1].scatter([h['tau_ms']],[h['q']],c=col,s=40)
    fixed=read(OUT/'fixed_rate_status.json')['rows']
    for k,row in enumerate(fixed):
        axs[0,0].plot([row['q']]*2,[max(.004,row['min_E_hz']),row['max_E_hz']],c='#b98233',lw=1.3,alpha=.8)
        axs[0,0].scatter(row['q'],row['mean_E_hz'],c='#b98233',marker='D',s=25,label='Fixed-field trajectory: late mean / range' if k==0 else None)
    axs[0,0].set(yscale='log',xlabel='E-target Z or field mean Z',ylabel='Mean E rate (Hz)',title='A  Tracked equilibria and finite-time activity ranges',xlim=(.48,1.02),ylim=(.004,550));axs[0,0].legend(fontsize=9,loc='center left')
    axs[0,0].text(.98,.03,'Star: complex crossing; square: fold.\nDashed high branch is not necessarily stable.',transform=axs[0,0].transAxes,ha='right',fontsize=9,bbox=dict(fc='white',ec='none',alpha=.8))
    axs[0,1].set(xlabel='GABA decay time (ms)',ylabel='Z at the low-branch complex crossing',title='B  Tracked complex-crossing curve in Z / GABA decay');axs[0,1].legend(fontsize=9)
    axs[1,0].axhline(0,c='0.6',lw=.7);axs[1,0].set(xlabel='Z - Z at complex crossing',ylabel='Real part of critical eigenvalue (1/s)',title='C  Pair crosses from decay to growth as Z decreases');axs[1,0].ticklabel_format(axis='x',style='sci',scilimits=(0,0));axs[1,0].legend(fontsize=9)
    for name,label,col in [('stability_uniform_high_q0.665.json','Uniform Z = 0.665','#527bb0'),('stability_native_path_high_q0.665.json','Z field mean = 0.665','#b34574')]:
        row=read(folder/name);roots=row['spectrum']['roots'];axs[1,1].scatter([x['real_per_s'] for x in roots],[x['frequency_hz'] for x in roots],c=col,label=label,s=25)
    axs[1,1].axvline(0,c='k',lw=.8);axs[1,1].set(xlabel='Real eigenvalue part (1/s)',ylabel='Frequency (Hz)',title='D  High-state sampled modes near the stability boundary',xlim=(-70,40),ylim=(-2,65));axs[1,1].legend(fontsize=9)
    fig.suptitle('E-only inhibition: delayed-rate stability analysis on the fixed C substrate',fontsize=16,y=1.01)
    fig.text(.5,-.02,'Reference constant external input for equilibria. These are reduced-model boundaries, not a classification of the noisy hybrid SNN transition.',ha='center',fontsize=11)
    fig.tight_layout(h_pad=1.8);save(fig,'reduced_z_gaba_stability_diagram')


def fixed_regimes():
    from scipy.signal import welch
    status=read(OUT/'fixed_rate_status.json');assert status['status']=='COMPLETE'
    fig,axs=plt.subplots(3,5,figsize=(16,9.5),gridspec_kw={'height_ratios':[1,1,1.2]});metrics=[]
    for col,row in enumerate(status['rows']):
        a=np.load(OUT/'fixed_rate'/f"{row['name']}.npz");fields=a['fields_hz'][:,0].astype(float);weights=a['count_e'];e=np.average(fields,axis=1,weights=weights);t=(np.arange(len(e))+1)*.001
        late=fields[len(e)//2:];late_e=e[len(e)//2:];std=late.std(0);cell=int(np.argmax(std));lo=len(e)-1000
        axs[0,col].plot(t,e,c='#252525',lw=1);axs[0,col].set(title=f'mean Z = {row["q"]:.3f}',xlabel='Time (s)',ylim=(-5,360))
        axs[0,col].text(.03,.92,f'Late mean {late_e.mean():.2f} Hz',transform=axs[0,col].transAxes,fontsize=10,va='top')
        axs[1,col].plot(t[lo:]-t[lo],e[lo:],c='#252525',label='Mean E',lw=1)
        axs[1,col].plot(t[lo:]-t[lo],fields[lo:,cell],c='#b34574',label='Most variable cell',lw=1)
        axs[1,col].set(xlabel='Last 1 s',title=f'Cell {cell}: SD {std[cell]:.2f} Hz',ylim=(-5,500))
        im=axs[2,col].imshow(std.reshape(10,10),origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0,vmax=150,interpolation='nearest')
        axs[2,col].set(xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20])
        freq,power=welch(late_e,fs=1000,nperseg=min(len(late_e),2000));freqc,pc=welch(late[:,cell],fs=1000,nperseg=min(len(late),2000));sel=(freq>=.5)&(freq<=200)
        metrics.append({'name':row['name'],'q':row['q'],'mean_E_hz':float(late_e.mean()),'global_std_hz':float(late_e.std()),
                        'quiet_fraction_5ms':float(np.mean(smooth(late_e,5)<1)),'maximum_cell_std_hz':float(std.max()),'selected_cell':cell,
                        'global_peak_frequency_hz':float(freq[sel][np.argmax(power[sel])]) if late_e.std()>1e-5 else None,
                        'selected_cell_peak_frequency_hz':float(freqc[sel][np.argmax(pc[sel])]) if std.max()>1e-5 else None,
                        'population_averaging_variance_ratio':float(late_e.var()/np.average(std**2,weights=weights)) if np.average(std**2,weights=weights)>1e-10 else None})
        if col:
            for rr in range(3):axs[rr,col].tick_params(labelleft=False)
    axs[0,0].set_ylabel('Mean E rate (Hz)');axs[1,0].set_ylabel('E rate (Hz)');axs[2,0].set_ylabel('y (mm)');axs[1,1].legend(loc='upper left',fontsize=8)
    fig.suptitle('Frozen Z, constant input: the same delayed rate equations produce bursts and continuous spatial activity',fontsize=15,y=1.01)
    fig.tight_layout();fig.colorbar(im,ax=axs[2,:],shrink=.8,pad=.015,label='Temporal SD of cell E rate (Hz)');save(fig,'deterministic_rate_regimes_and_local_oscillations')
    write(OUT/'fixed_rate_regime_metrics.json',{'rows':metrics,'scope':'Late half of each run. Most variable cell chosen descriptively; spectra are finite-window diagnostics, not proof of stable limit cycles.'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['validation','planes','native','native_field','critical','diagram','regimes'],required=True);a=p.parse_args()
    {'validation':validation,'planes':conditional_planes,'native':native_controls,'native_field':lambda:native_controls(True),'critical':critical_planes,'diagram':diagrams,'regimes':fixed_regimes}[a.part]()


if __name__=='__main__':main()
