#!/usr/bin/env python3
"""Complete-only collection of the five target-resolved field checks."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from campaign import ROOT,read,write,sha
from target_density_field_family import OUT,JOBS
from coupled_density_exit import ADAPTED
import analyze_target_density_exit as comparison
from analyze_native import original


def native_data(root,name):
    job=read(root/'jobs'/f'{name}.json');start=job['branch_start_s']*1000
    with np.load(root/'extended_analysis'/f'{name}_readouts.npz') as z:
        rate=z['rate_5ms_Hz'][:2000];field=z['field_rate_5ms_Hz'][:2000]
    g=original.load(root/'runs'/name/'mechanism_chunks',['time_ms','global_E_rate_Hz','global_raw_conductance_ratio'])
    mask=(g['time_ms']>=start)&(g['time_ms']<start+10000);assert mask.sum()==10000
    drift=original.load(root/'runs'/name/'conditional_drift_chunks',['time_ms','values'])
    dt=(drift['time_ms']-start)/1000;keep=(dt>0)&(dt<=10)
    return dict(rate=rate,field=field,R=g['global_E_rate_Hz'][mask],G=g['global_raw_conductance_ratio'][mask],
         time=(g['time_ms'][mask]-start)/1000,drift=drift['values'][keep,:,0],drift_time=dt[keep])


def main(wait):
    geo=dict(np.load(ADAPTED/'geometry.npz'));E=geo['population']==0
    counts=np.bincount(geo['group_cell'][E],weights=geo['group_size'][E],minlength=400)
    rows=[];data={}
    while len(rows)<len(JOBS):
        for source,name in JOBS:
            root=ROOT/source
            if name in data or not (OUT/name/'result.json').exists() or not (root/'extended_analysis'/f'{name}_readouts.npz').exists():continue
            assert read(OUT/name/'result.json')['status']=='COMPLETE'
            assert read(root/'runs'/name/'result.json')['status']=='COMPLETE'
            old=comparison.OUT;comparison.OUT=OUT
            try:d=comparison.load_candidate(name,geo)
            finally:comparison.OUT=old
            n=native_data(root,name);data[name]=(n,d);windows=[]
            for lo,hi in [(0,5),(5,10)]:
                s=slice(lo*1000,hi*1000);ns=slice(lo*200,hi*200);mask=(n['drift_time']>lo)&(n['drift_time']<=hi)
                delta=d['field'][s].mean(0)-n['field'][ns].mean(0)
                windows.append(dict(interval_s=[lo,hi],native_rates_Hz=n['rate'][ns].mean(0).tolist(),density_rates_Hz=d['rate'][s].mean(0).tolist(),
                     native_Graw=float(n['G'][s].mean()),density_Graw=float(d['G'][s].mean()),
                     native_dZ_per_s=n['drift'][mask].mean(0).tolist(),density_dZ_per_s=d['drift'][s].mean(0).tolist(),
                     weighted_field_RMS_Hz=float(np.sqrt(np.average(delta**2,weights=counts)))))
            row=dict(name=name,native_root=str(root),windows=windows);rows.append(row);write(OUT/name/'comparison.json',row)
            print('TARGET FIELD CHECK',name,windows[-1],flush=True)
        result=dict(status='COMPLETE' if len(rows)==len(JOBS) else 'PARTIAL',completed=list(data),rows=rows,
             native_correspondence_certified=False,formal_bifurcation_allowed=False,
             drift_statistics='Native20msfullstep budgets vs density1msinstantaneousparticleeligibility; means over namedwindows.',producer_sha256=sha(__file__))
        write(OUT/'comparison.json',result)
        if len(rows)<len(JOBS):
            if not wait:return
            write(OUT/'analysis_progress.json',dict(status='WAITING_FIXED_CONDITIONS_AND_NATIVE',pid=os.getpid(),completed=list(data),updated_epoch=time.time()));time.sleep(20)
    plot(data)
    write(OUT/'analysis_progress.json',dict(status='COMPLETE',completed=list(data),updated_epoch=time.time()))


def plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(3,len(JOBS),figsize=(15,9),layout='constrained',gridspec_kw={'height_ratios':[.75,1,1]})
    centers=np.load(ROOT/'native_slices/geometry.npz')['centers_mm']
    for col,(source,name) in enumerate(JOBS):
        n,d=data[name];job=read(ROOT/source/'jobs'/f'{name}.json')
        axes[0,col].plot(n['time'],n['R'],'k',lw=1,label='Native')
        axes[0,col].plot(d['time'],d['R'],color='#cc7722',lw=1,label='Individual target inputs')
        axes[0,col].set(title=f"K={job['target_K']:g}, {job['source_history']}",xlabel='Time since clamp (s)',xlim=(0,10),ylim=(-3,300),yticks=[0,100,200,300])
        if col==0:axes[0,col].set_ylabel('Causal E rate (Hz)');axes[0,col].legend(frameon=False,fontsize=7)
        for row,field in [(1,n['field'][1000:2000].mean(0)),(2,d['field'][5000:10000].mean(0))]:
            ax=axes[row,col];im=ax.imshow(field.reshape(20,20),origin='lower',extent=(0,20,0,20),vmin=0,vmax=500,cmap='magma',interpolation='nearest')
            for xy in centers:ax.add_patch(Circle(xy,1.5,fill=False,edgecolor='#00c3c5',lw=.8))
            ax.set(xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)')
            if col==0:ax.set_ylabel(('Native' if row==1 else 'Density')+' 5–10 s\ny (mm)')
    fig.colorbar(im,ax=axes[1:].ravel().tolist(),label='E rate (Hz)',shrink=.65)
    fig.suptitle('Actual exit field family: neighboring conditions and recovery history',weight='bold')
    fig.text(.5,-.01,'Z/K held, M/R/G and recurrent input free. Fixed conditional correspondence checks; no stability or bifurcation certification.',ha='center',fontsize=8)
    for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'target_density_field_family.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig);write(OUT/'figure_metadata.json',dict(agent_visual='PENDING',human_review='PENDING',producer_sha256=sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)
