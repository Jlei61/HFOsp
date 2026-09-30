#!/usr/bin/env python3
"""Collect the fixed pair of autonomous target-projection comparisons."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from scipy import sparse
from campaign import ROOT,read,write,sha
from target_density_exit import OUT,NAME
from conditional_density_inputs import OPS
from coupled_density_exit import ADAPTED
from analyze_exit_branch_density import native_prefix
from analyze_native import original


def load_candidate(mode,geo):
    with np.load(OUT/mode/'trajectory.npz') as z:
        value=z['group_output'];R=z['global_R_Hz'];G=30*z['global_s'];tm=z['elapsed_time_ms']/1000
    sizes=geo['group_size'];E=geo['population']==0;mask=[E]+[E&(geo['group_region']==q) for q in range(3)]
    region=np.stack([np.average(value[:,0,m],weights=sizes[m],axis=1) for m in mask],axis=1)
    drift=np.stack([np.average((value[:,8,m]-value[:,1,m])/5,weights=sizes[m],axis=1) for m in mask],axis=1)
    counts=np.bincount(geo['group_cell'][E],weights=sizes[E],minlength=400)
    projection=sparse.coo_matrix((sizes[E]/np.maximum(counts[geo['group_cell'][E]],1),
             (geo['group_cell'][E],np.flatnonzero(E))),shape=(400,len(sizes))).tocsr()
    field=np.asarray((projection@value[:,0].T).T)
    assert np.allclose(field@counts/counts.sum(),region[:,0],atol=2e-4)
    return dict(mode=mode,rate=region,drift=drift,R=R,G=G,time=tm,field=field)


def main(wait):
    geo=dict(np.load(ADAPTED/'geometry.npz'));counts=np.bincount(geo['group_cell'][geo['population']==0],weights=geo['group_size'][geo['population']==0],minlength=400)
    native=native_prefix(NAME);native_rate=native['rate5'];data={};rows=[]
    drift=original.load(ROOT/'exit_return_probes/runs'/NAME/'conditional_drift_chunks',['time_ms','values'])
    start=read(ROOT/'exit_return_probes/jobs'/f'{NAME}.json')['branch_start_s']
    nt=drift['time_ms']/1000-start;keep=(nt>0)&(nt<=10)
    native.update(drift_time=nt[keep],drift=drift['values'][keep,:,0])
    while len(rows)<2:
        for mode in ['homogeneous','individual']:
            if mode in data or not (OUT/mode/'result.json').exists():continue
            assert read(OUT/mode/'result.json')['status']=='COMPLETE'
            d=load_candidate(mode,geo);data[mode]=d;windows=[]
            for lo,hi in [(0,5),(5,10)]:
                s=slice(lo*1000,hi*1000);ns=slice(lo*200,hi*200);field=d['field'][s].mean(0);nfield=native['fields'][ns].mean(0)
                nd=(native['drift_time']>lo)&(native['drift_time']<=hi)
                windows.append(dict(interval_s=[lo,hi],native_rates_Hz=native_rate[ns].mean(0).tolist(),density_rates_Hz=d['rate'][s].mean(0).tolist(),
                    density_Graw_mean=float(d['G'][s].mean()),native_Graw_mean=float(native['Graw'][s].mean()),
                    density_counterfactual_dZ_per_s=d['drift'][s].mean(0).tolist(),native_counterfactual_dZ_per_s=native['drift'][nd].mean(0).tolist(),
                    drift_statistics='Density1msinstantaneousperparticleeligibility versus native20msfullstepbudgets; both averaged over named window.',
                    weighted_field_RMS_Hz=float(np.sqrt(np.average((field-nfield)**2,weights=counts))),
                    weighted_field_MAE_Hz=float(np.average(abs(field-nfield),weights=counts))))
            row=dict(mode=mode,windows=windows);rows.append(row);write(OUT/mode/'comparison.json',row)
            print('TARGET COMPARISON',mode,windows[-1],flush=True)
        write(OUT/'comparison.json',dict(status='COMPLETE' if len(rows)==2 else 'PARTIAL',rows=rows,
             formal_bifurcation_allowed=False,native_correspondence_certified=False,producer_sha256=sha(__file__)))
        if len(rows)<2:
            if not wait:return
            write(OUT/'analysis_progress.json',dict(status='WAITING_FIXED_PAIR',pid=os.getpid(),completed=list(data),updated_epoch=time.time()));time.sleep(20)
    plot(native,data,geo)
    write(OUT/'analysis_progress.json',dict(status='COMPLETE',updated_epoch=time.time()))


def plot(native,data,geo):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained',gridspec_kw={'height_ratios':[.85,1]})
    tn=native['global_time_ms']/1000;colors={'homogeneous':'#287c8e','individual':'#cc7722'}
    axes[0,0].plot(tn,native['R'],'k',lw=1.4,label='Native');axes[0,1].plot(tn,native['Graw'],'k',lw=1.4)
    axes[0,2].plot(native['drift_time'],native['drift'][:,0],'k',lw=1.4)
    for mode,d in data.items():
        label='Group inputs' if mode=='homogeneous' else 'Individual target inputs'
        for j,key in enumerate(['R','G','drift']):axes[0,j].plot(d['time'],d[key][:,0] if key=='drift' else d[key],color=colors[mode],lw=1.0,label=label)
    axes[0,0].legend(frameon=False,fontsize=8)
    for ax,ylabel in zip(axes[0],['Causal E rate (Hz)','Global G / gL','Counterfactual dZ/dt (1/s)']):ax.set(xlim=(0,10),xlabel='Time since clamp (s)',ylabel=ylabel)
    axes[0,2].axhline(0,color='.6',ls=':',lw=.8)
    fields=[native['fields'][1000:2000].mean(0)]+[data[m]['field'][5000:10000].mean(0) for m in ['homogeneous','individual']]
    centers=np.load(ROOT/'native_slices/geometry.npz')['centers_mm']
    for ax,title,field in zip(axes[1],['Native','Group inputs','Individual target inputs'],fields):
        im=ax.imshow(field.reshape(20,20),origin='lower',extent=(0,20,0,20),vmin=0,vmax=500,cmap='magma',interpolation='nearest')
        for xy in centers:ax.add_patch(Circle(xy,1.5,fill=False,edgecolor='#00c3c5',lw=.9))
        ax.set(title=title+' (5–10 s)',xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20])
    axes[1,0].set_ylabel('y (mm)');fig.colorbar(im,ax=axes[1].tolist(),label='E rate (Hz)',shrink=.85)
    fig.suptitle('Actual exit fields: target heterogeneity under autonomous coupling',weight='bold')
    fig.text(.5,-.01,'Held Z=.21/K=9 field family; recurrent input, M/R/G free. Two paired numerical runs; Z drift is the unclamped equation evaluated at held Z.',ha='center',fontsize=8)
    for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'target_density_exit.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig);write(OUT/'figure_metadata.json',dict(agent_visual='PENDING',human_review='PENDING',producer_sha256=sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)
