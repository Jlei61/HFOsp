#!/usr/bin/env python3
"""Native regional recordings distinguish global mean suppression from core quenching."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import analyze_topic4_autonomous_recovery as a

def draw(root,name):
    folder=root/'runs'/name;job=a.read(root/'jobs'/(name+'.json'))
    d=a.load(folder,['time_ms','spikes_1ms','regions_1ms','slow_time_ms','Z','M','currents','regional_currents'])
    t=d['time_ms']/1000;ts=d['slow_time_ms']/1000
    geo=np.load(root/'geometry.npz');nr=geo['region_counts'][:3]
    rates=np.column_stack([d['spikes_1ms'][:,0]*1000/32000,d['regions_1ms'][:,:3]*1000/nr])
    currents=np.empty((len(ts),4,3))
    currents[:,0,:]=np.column_stack([d['currents'][:,0],d['currents'][:,2],job['eta_m']*d['M'][:,0]])
    currents[:,1:,:]=d['regional_currents'][:,:,[0,3,4]]
    z=d['Z'][:,[0,5,6,7]]
    plt.rcParams.update({'axes.labelsize':18,'xtick.labelsize':14,'ytick.labelsize':14,'font.size':16})
    fig,axes=plt.subplots(4,4,figsize=(22,15),sharex=True,sharey='row',gridspec_kw={'hspace':.20,'wspace':.13})
    for c,label in enumerate(['All E','Core A E','Core B E','Surround E']):
        axes[0,c].set_title(label,fontsize=19,pad=12)
        axes[0,c].plot(t,gaussian_filter1d(rates[:,c],3),c='#353535',lw=.7)
        axes[0,c].set_ylim(-5,505)
        axes[1,c].plot(ts,currents[:,c,0],c='#cf6454',lw=.9,label='Excitation')
        axes[1,c].plot(ts,currents[:,c,1],c='#3575a1',lw=.9,label='Applied inhibition')
        axes[2,c].plot(ts,currents[:,c,0]-currents[:,c,1]-currents[:,c,2],c='#6c3c79',lw=1.)
        axes[2,c].axhline(0,color='.4',ls=':',lw=.9)
        axes[3,c].plot(ts,z[:,c],c='#7e399d',lw=1.4)
        axes[3,c].set_ylim(0,1.04);axes[3,c].set_xlabel('Time (s)')
        for ax in axes[:,c]:
            ax.set_xlim(0,t[-1]);ax.spines[['right','top']].set_visible(False)
    for i,label in enumerate(['Rate (Hz)','Current (mV equiv.)','Net current (mV equiv.)','Resource Z']):
        axes[i,0].set_ylabel(label)
        axes[i,0].text(-.24,1.02,'ABCD'[i],transform=axes[i,0].transAxes,fontsize=23,weight='bold')
    axes[1,0].legend(loc='upper left',fontsize=12,framealpha=.95,facecolor='white',edgecolor='none')
    fig.subplots_adjust(left=.085,right=.98,top=.95,bottom=.08)
    out=folder/'figures';out.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(out/f'regional_current_balance.{ext}',dpi=170)
    plt.close(fig)
    a.write(folder/'regional_current_balance_plot.json',dict(source=str(folder),window_s=[0,float(t[-1])],
        current_observation_step_s=.020,spike_count_step_s=.001,rate_display_gaussian_sigma_s=.003,
        current_columns=['IE','actual Z times delivered II','eta_M times M'],regions=['all E','Core A E','Core B E','surround E'],
        interpretation='Spatially averaged currents are descriptive recordings; positive or negative regional mean is not a single-neuron stability calculation. Both native local and added global inhibition are included after Zi.',
        human_review='PENDING'))
    with (out/'README.md').open('a') as f:f.write('\n### regional_current_balance.png / .pdf\n同一连续轨迹中并列全E、Core A、Core B与外围的实际放电、兴奋/有效抑制电流、扣除M后的净电流及Z。放电计数为1ms，电流和Z为实际20ms采样，各列共用坐标尺度。**关注点**：全网平均净抑制可与核内持续放电并存；平均电流不是单细胞固定点或终止判据。\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=a.OUT/'activity_global_pool_round3');p.add_argument('--name',required=True);v=p.parse_args();draw(v.root,v.name)
