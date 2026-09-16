#!/usr/bin/env python3
"""Native unsmoothed spatial counts, for distinguishing a wave from global rate."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,sys
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation,PillowWriter
import analyze_topic4_fixed_zm_termination as a

def main(folder,end,gif,times=None):
    root=folder.parent.parent;data=a.load(folder,keys=['spikes_1ms','regions_1ms','field_5ms','slow_time_ms','Z','M'])
    intrinsic=a.load(folder,'intrinsic_adaptation_chunks');geo=dict(np.load(root/'geometry.npz'))
    end=min(end,len(data['spikes_1ms'])/1000);N=int(end/.05)
    fields=data['field_5ms'][:N*10].reshape(N,10,400).sum(1)/geo['cell_e_counts']/.05
    t=(np.arange(N)+.5)*.05
    plt.rcParams.update({'font.size':13,'axes.spines.right':False,'axes.spines.top':False})
    dest=root/'figures';dest.mkdir(exist_ok=True)
    if times is None:times=[.125,.375,.775,1.025,1.525,2.025,3.025,min(end-.025,5.025)]
    assert len(times)==8
    fig,axs=plt.subplots(2,4,figsize=(14,7),layout='constrained')
    for ax,tm in zip(axs.flat,times):
        k=min(int(tm/.05),N-1)
        im=ax.imshow(fields[k].reshape(20,20),origin='lower',extent=(0,20,0,20),vmin=0,vmax=500,cmap='magma',interpolation='nearest')
        for name,xy in zip('AB',geo['centers_mm']):ax.add_patch(Circle(xy,1.5,fill=False,ec='#55dddd'));ax.text(*xy,name,c='cyan',ha='center')
        ax.set(title=f'{t[k]:.3f} s',xlabel='x (mm)',ylabel='y (mm)')
    fig.colorbar(im,ax=axs,label='E rate in 50 ms (Hz)',shrink=.65)
    fig.savefig(dest/f"native_storyboard_{folder.name}.png",dpi=150);plt.close(fig)
    if not gif:a.figure_readme(dest);return
    fig=plt.figure(figsize=(12,5));gs=fig.add_gridspec(2,2,width_ratios=[1,1.5]);sp=fig.add_subplot(gs[:,0]);rate_ax=fig.add_subplot(gs[0,1]);z_ax=fig.add_subplot(gs[1,1])
    im=sp.imshow(fields[0].reshape(20,20),origin='lower',extent=(0,20,0,20),vmin=0,vmax=500,cmap='magma',interpolation='nearest')
    for name,xy in zip('AB',geo['centers_mm']):sp.add_patch(Circle(xy,1.5,fill=False,ec='#55dddd'));sp.text(*xy,name,c='cyan',ha='center')
    sp.set(xlabel='x (mm)',ylabel='y (mm)');fig.colorbar(im,ax=sp,label='E rate (Hz)',shrink=.65)
    n=int(end*100);pop=data['spikes_1ms'][:n*10,0].reshape(n,10).sum(1)/320
    regions=data['regions_1ms'][:n*10,:2].reshape(n,10,2).sum(1)/geo['region_counts'][:2]/.01
    rt=(np.arange(n)+.5)*.01
    rate_ax.plot(rt,pop,c='black',label='All E',lw=1.2)
    for i,c in enumerate(['#237ca9','#d24b99']):rate_ax.plot(rt,regions[:,i],c=c,label='Core '+str('AB'[i]),lw=.7,alpha=.8)
    rate_ax.set(xlim=(0,end),ylabel='Rate (Hz)',ylim=(0,520));rate_ax.legend(fontsize=9,ncol=3,loc='upper right')
    st=data['slow_time_ms']/1000;z_ax.plot(st,data['Z'][:,0],c='#713399',label='Mean Z');z_ax.set(xlim=(0,end),ylim=(0,1.05),xlabel='Time (s)',ylabel='Mean Z')
    if intrinsic:
        k_ax=z_ax.twinx();k_ax.plot(intrinsic['time_ms']/1000,intrinsic['sahp_mean_conductance_ratio'],c='#b56e22',lw=1);k_ax.set_ylabel('Mean gK/gL',c='#b56e22')
    cursor=[rate_ax.axvline(0,c='red',lw=1),z_ax.axvline(0,c='red',lw=1)];fig.tight_layout()
    def update(k):
        im.set_data(fields[k].reshape(20,20));sp.set_title(f'Time {t[k]:.2f} s')
        for line in cursor:line.set_xdata([t[k],t[k]])
        return [im,*cursor]
    FuncAnimation(fig,update,frames=N,interval=80,blit=False).save(dest/f'native_{folder.name}.gif',writer=PillowWriter(fps=12),dpi=90)
    plt.close(fig)
    a.figure_readme(dest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--end',type=float,default=8.);p.add_argument('--gif',action='store_true');p.add_argument('--times',type=float,nargs=8);x=p.parse_args();main(x.folder,x.end,x.gif,x.times)
