#!/usr/bin/env python3
"""Read-only state context and spatial pulse-response panels for both GIF bases."""
import json
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'results/topic4_sef_hfo/fig5_two_gif_bases'
OUT=ROOT/'results/paper-ready-figure/fig5_two_gif_bases/figures'
IDS=['support_rank__vth_low','old_joint__tau_d_GABA_ms_high']
NAMES=['Two-core base 1: threshold gain 0.7, GABA decay 18 ms',
       'Two-core base 2: threshold gain 1.0, GABA decay 24 ms']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})


def read(p): return json.loads(p.read_text())
def load(p):
    with np.load(p,allow_pickle=False) as a: return {k:a[k] for k in a.files}
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180)
    fig.savefig(OUT/(name+'.pdf'))
    plt.close(fig)


def geometry(ax, protocol, a):
    centers=np.asarray(protocol['candidate']['node_field']['centers_mm'])
    for index,center in enumerate(centers):
        points=a['positions_E'][a['region_E']==index]
        ax.scatter(points[:,0],points[:,1],s=.6,color='#d5d5d5',rasterized=True,zorder=0)
        ax.text(*center,str(index+1),ha='center',va='center',fontsize=9)
    ax.scatter(a['contact_xy'][:,0],a['contact_xy'][:,1],marker='+',s=18,color='#555555')
    ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',ylabel='y (mm)')
    ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])


def context():
    ready=[(cid,name) for cid,name in zip(IDS,NAMES) if (DATA/cid/'trajectory.json').exists()]
    if not ready: return []
    fig,axs=plt.subplots(len(ready),3,figsize=(13,3.3*len(ready)),squeeze=False,layout='constrained')
    for row,(cid,name) in enumerate(ready):
        a=load(DATA/cid/'trajectory.npz');j=read(DATA/cid/'trajectory.json')
        t=a['time_ms']/1000; ax=axs[row,0]
        for column,(label,col) in enumerate(zip(['Population','Core 1','Core 2'],['black','#d95f02','#1b9e77'])):
            ax.plot(t,a['rates_hz'][:,column],label=label,color=col,lw=.9)
        ax.set(xlabel='Time (s)',ylabel='Firing rate (Hz)',title=name)
        ax.legend(handles=[ax.lines[0]],fontsize=8,loc='upper left',bbox_to_anchor=(0,.87))
        ts=a['slow_time_ms']/1000
        for col,(label,color) in enumerate(zip(['Core 1','Core 2','Surround'],['#d95f02','#1b9e77','#777777'])):
            axs[row,1].plot(ts,a['region_z'][:,col],label=label,color=color,lw=1)
            axs[row,2].plot(ts,a['region_m'][:,col],label=label,color=color,lw=1)
        axs[row,1].legend(fontsize=8,loc='lower left')
        axs[row,1].set(xlabel='Time (s)',ylabel='Inhibitory efficacy z',ylim=(0,1.03),title='Z: inhibitory efficacy')
        axs[row,2].set(xlabel='Time (s)',ylabel='Adaptation m',title='M: adaptation')
        onset=j['trajectory']['onset_ms']
        for ax in axs[row]:
            if onset is not None: ax.axvline(onset/1000,color='#aa0066',ls='--',lw=.8)
            ax.set_xlim(0,max(t))
        label=j['trajectory']['classification'].replace('_',' ').lower()
        if j['trajectory']['classification']=='EARLY_STOP_UNRESOLVED':
            label='Early stop: plateau check pending'
        axs[row,0].text(.02,.97,label,
                       va='top',transform=axs[row,0].transAxes,fontsize=8)
    fig.suptitle('Identical Z/M reference applied to two fixed GIF substrates',fontsize=14)
    save(fig,'fig5-two-bases-state-context')
    return [cid for cid,_ in ready]


def probes():
    ready=[(cid,name) for cid,name in zip(IDS,NAMES) if (DATA/cid/'probe.json').exists()]
    if not ready: return []
    # Shared physical scales across every state and both substrates.
    totals=[]; spatial=[]; bundles=[]
    for cid,name in ready:
        a=load(DATA/cid/'probe.npz');j=read(DATA/cid/'probe.json');p=read(DATA/cid/'protocol.json')
        dest={}
        pos=a['positions_E'];edges=np.linspace(0,20,21)
        occupancy=np.histogram2d(pos[:,0],pos[:,1],bins=(edges,edges))[0]
        for state in j['states_ms']:
            fields=a[state+'_full_field']
            totals.extend(fields.sum(axis=1).tolist())
            # Fixed central pulse, selected before responses are inspected. Never pool origins.
            h=np.histogram2d(pos[:,0],pos[:,1],bins=(edges,edges),weights=fields[4])[0]
            dest[state]=np.divide(h,occupancy,out=np.zeros_like(h),where=occupancy>0).T
            spatial.extend(dest[state].ravel().tolist())
        bundles.append((cid,name,a,j,p,dest))
    nt=SymLogNorm(linthresh=1,vmin=-max(1,max(abs(np.asarray(totals)))),vmax=max(1,max(abs(np.asarray(totals)))))
    nf=SymLogNorm(linthresh=.01,vmin=-max(.01,max(abs(np.asarray(spatial)))),vmax=max(.01,max(abs(np.asarray(spatial)))))
    outputs=[]
    for cid,name,a,j,p,dest in bundles:
        states=list(j['states_ms']);fig,axs=plt.subplots(2,3,figsize=(12.2,7.1),layout='constrained',gridspec_kw={'width_ratios':[1,1,1.35]})
        for row,state in enumerate(states):
            label={'reference':'Early reference','pre_onset':'Pre-onset','later_reference':'Later reference (no qualified transition)'}[state]
            label+=f" · {j['states_ms'][state]/1000:g} s"
            ax=axs[row,0];geometry(ax,p,a)
            im=ax.scatter(a['sites_mm'][:,0],a['sites_mm'][:,1],c=a[state+'_full_field'].sum(axis=1),cmap='RdBu_r',norm=nt,s=90,edgecolors='#555555',lw=.6)
            ax.set_title(label+'\nWhich stimulation sites are effective?',fontsize=9)
            ax=axs[row,1];geometry(ax,p,a)
            field=ax.imshow(dest[state],origin='lower',extent=(0,20,0,20),cmap='RdBu_r',norm=nf,interpolation='none',alpha=.9,zorder=-1)
            ax.scatter([10],[10],marker='*',s=110,facecolor='gold',edgecolor='black',lw=.6,zorder=8)
            ax.set_title('Where does the extra activity occur?\nFixed pulse at (10, 10) mm',fontsize=9)
            ax=axs[row,2];curves=np.cumsum(a[state+'_extra_spikes_per_ms'],axis=1)
            for curve in curves:ax.plot(np.arange(1,201),curve,color='#999999',alpha=.55,lw=.7)
            ax.plot(np.arange(1,201),curves.mean(axis=0),color='#713080',lw=2,label='Mean of nine sites')
            ax.axhline(0,color='black',lw=.6);ax.set(xlabel='Time after pulse (ms)',ylabel='Cumulative extra E spikes',title='Does the response grow or settle?',xlim=(0,200))
            ax.set_yscale('symlog',linthresh=1);ax.legend(fontsize=8)
        fig.colorbar(im,ax=axs[:,0].tolist(),shrink=.65,label='Extra E spikes, 0–200 ms')
        fig.colorbar(field,ax=axs[:,1].tolist(),shrink=.65,label='Extra spikes / local E neuron, 0–200 ms')
        fig.suptitle(name+'\n16-cell pulse minus exact sham; injected spikes excluded',fontsize=13)
        filename='fig5-D-'+cid
        save(fig,filename);outputs.append(filename)
        j['plot_contract']=dict(shared_across_both_bases=True,negative_responses_retained=True,
            destination_field='Unsmooth 1 mm bins, prespecified central pulse (site 4) divided by local E count; not probability.',
            independent_units='One topology and one noise trajectory per base; sites are paired interventions, not patients.',
            stimulus_color_range=[float(nt.vmin),float(nt.vmax)],destination_color_range=[float(nf.vmin),float(nf.vmax)])
        (OUT/(filename+'-metadata.json')).write_text(json.dumps(j,indent=2)+'\n')
    return outputs


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    trajectories=context();panels=probes()
    if not trajectories and not panels: return
    text=''
    if trajectories:
        text+='### fig5-two-bases-state-context.png\n两组 GIF 的完整底物分别施加相同的 Z/M 参考参数，展示全局及双核放电率、Z 和 M。虚线仅在实测满足 tonic runaway 判据时标注，不代表临床发作或分叉类型。\n**关注点**：不同基底能否保留足够长的转变前活动；没有跳变同样保留。\n\n'
    for name in panels:
        text+=f'### {name}.png\n左右分别展示刺激位置的效应、额外响应的空间位置，以及其累积时间过程；均为同状态 exact sham 的有符号差值并扣除注入帧。两行状态按实际轨迹命名，没有合格转变时仅称早期/后期参考；两组图使用共同色标。\n**关注点**：红色为额外放电，蓝色为放电减少；九个位置不构成九个独立患者或模型重复。\n\n'
    (OUT/'README.md').write_text(text)
    (OUT/'render_status.json').write_text(json.dumps(dict(trajectories=trajectories,panels=panels,author_accepted=False),indent=2)+'\n')
    print(json.dumps(dict(trajectories=trajectories,panels=panels)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args();DATA=args.data;OUT=args.out
    main()
