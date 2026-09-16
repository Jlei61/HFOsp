#!/usr/bin/env python3
"""Same-state long continuations: transient returns versus retained termination.

The conditional Z solution is checked against saved states, not used to infer
a whole-network fixed point, bifurcation, or independent biological replicate.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import analyze_topic4_autonomous_recovery as common
import analyze_topic4_autonomous_events as event

BASE=common.OUT
PAIRS=[('nativeZ_continuation_to60','nativeZ_k200_tau10_continuation60_s9108401','Original Z'),
       ('revisedZ_continuation_to60','revisedZ_k200_tau10_continuation60_s9108401','Additional recovery ρ = 0.25')]

def load_pool(folder):
    keys=['time_ms','rate_Hz','raw_global_current','effective_global_current']
    records={key:[] for key in keys}
    for path in sorted((folder/'pool_chunks').glob('*.npz')):
        if '.tmp.' in path.name:continue
        with np.load(path) as q:
            for key in keys:records[key].append(q[key])
    return {key:np.concatenate(value) for key,value in records.items()}

def bound(job,a,pool):
    t=pool['time_ms'];rg=pool['rate_Hz'];rho=job['recovery_ratio']
    assert np.allclose(np.diff(t),5.)
    assert job['pool_gain']>0 and job['mode']=='native' and job['gamma']==0
    lower=job['pool_gain']*np.maximum(rg[:-1]*np.exp(-5/(job['pool_tau_s']*1000))-job['pool_threshold_Hz'],0)
    forced=lower>=job['threshold']
    edge=np.diff(np.r_[False,forced,False].astype(int))
    intervals=[dict(start_s=float(t[lo]/1000),end_s=float(t[hi]/1000),duration_s=float((t[hi]-t[lo])/1000))
        for lo,hi in zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1)) if t[hi]-t[lo]>=100]
    zstar=rho/(1+rho);factor=1-.1/(job['tau_Z_s']*1000)*(1+rho)
    z=a['Z'][:,[0,5,6]];ts=a['slow_time_ms'];errors=[]
    for i in range(len(ts)-1):
        lo,hi=int(round(ts[i]/5)),int(round(ts[i+1]/5))
        if hi>=len(t) or not forced[lo:hi].all():continue
        assert np.isclose(t[lo],ts[i]) and np.isclose(t[hi],ts[i+1])
        expected=zstar+(z[i]-zstar)*factor**round((ts[i+1]-ts[i])/.1)
        errors.append(float(np.max(np.abs(expected-z[i+1]))))
    assert errors and max(errors)<1e-10
    return dict(guaranteed_strong_global_input_intervals=intervals,
        strong_input_Z_equilibrium=zstar,strong_input_Z_relaxation_s=job['tau_Z_s']/(1+rho),
        actual_20ms_intervals_checked=len(errors),max_mean_A_B_Z_conditional_Euler_error=max(errors),
        sufficient_Rg_Hz=job['pool_threshold_Hz']+job['threshold']/job['pool_gain'],
        interpretation='Within the guaranteed intervals, the global current alone keeps every E cell above the Z-depletion threshold. The conditional Euler solution is measured against actual saved mean/core Z. It does not prove a whole-network equilibrium or that future recovery is impossible.')

def main(require_complete=False):
    collected=[]
    for subdir,name,label in PAIRS:
        root=BASE/subdir;folder=root/'runs'/name
        if require_complete:assert (folder/'continuation_complete.json').exists() or (root/'continuation_complete.json').exists()
        job=common.read(root/'jobs'/(name+'.json'));common.OUT=root
        a=common.load(folder,['time_ms','spikes_1ms','regions_1ms','slow_time_ms','Z','M','currents'])
        stats=common.classify(folder);pool=load_pool(folder);nr=np.load(root/'geometry.npz')['region_counts']
        n=len(a['spikes_1ms'])//10
        rr=a['regions_1ms'].astype(float).reshape(n,10,6).sum(1)/nr/.01
        rate=a['spikes_1ms'][:,0].reshape(n,10).sum(1)/32000/.01
        timing=event.temporal_audit(rate,rr[:,:3],stats['entries'],stats['recoveries'])
        proof=bound(job,a,pool)
        collected.append(dict(label=label,job=job,a=a,pool=pool,stats=stats,timing=timing,proof=proof,nr=nr))
    same=['seed','mode','gamma','eta_m','tau_M_s','tau_Z_s','threshold','pool_gain','pool_threshold_Hz','pool_tau_s']
    for key in same:assert collected[0]['job'][key]==collected[1]['job'][key],key
    plt.rcParams.update({'font.size':15,'axes.labelsize':18,'xtick.labelsize':15,'ytick.labelsize':15})
    fig,axes=plt.subplots(4,2,figsize=(19,13),sharex='col',sharey='row',gridspec_kw={'hspace':.18,'wspace':.16})
    for column,c in enumerate(collected):
        a,pool,job=c['a'],c['pool'],c['job'];t=a['time_ms']/1000;ts=a['slow_time_ms']/1000
        rate=np.column_stack([a['spikes_1ms'][:,0]/32,a['regions_1ms'][:,:2].astype(float)*1000/c['nr'][:2]])
        for i,(color,label) in enumerate([('#30343b','All E'),('#bc74a5','Core A E'),('#408abd','Core B E')]):
            axes[0,column].plot(t,gaussian_filter1d(rate[:,i],3),c=color,lw=.9,label=label)
        axes[0,column].set_title(c['label'],fontsize=20,pad=15)
        axes[0,column].set_ylim(-8,515)
        for i,color,label in [(0,'#763c92','All E'),(5,'#bc74a5','Core A'),(6,'#408abd','Core B')]:
            axes[1,column].plot(ts,a['Z'][:,i],c=color,lw=1.4,label=label)
        if job['recovery_ratio']:
            axes[1,column].axhline(c['proof']['strong_input_Z_equilibrium'],c='.5',ls=':',lw=1)
        axes[1,column].set_ylim(-.02,1.03)
        axes[2,column].plot(ts,a['currents'][:,0],c='#c76551',lw=1,label='Excitatory input')
        axes[2,column].plot(ts,a['currents'][:,2],c='#397ca0',lw=1,label='Applied inhibition')
        axes[3,column].plot(pool['time_ms']/1000,pool['rate_Hz'],c='#756196',lw=1.4)
        axes[3,column].axhline(job['pool_threshold_Hz'],c='#756196',ls=':',lw=1)
        axes[3,column].set_xlabel('Time (s)')
        for row in range(4):
            ax=axes[row,column];ax.set_xlim(0,c['stats']['observed_s'])
            ax.spines[['top','right']].set_visible(False)
            ax.text(-.12,1.02,'ABCDEFGH'[row*2+column],transform=ax.transAxes,weight='bold',fontsize=23)
        axes[0,column].legend(loc='upper left',fontsize=11,ncol=3,framealpha=.85)
        axes[2,column].legend(loc='upper left',fontsize=11,ncol=2,framealpha=.85)
    for ax,label in zip(axes[:,0],['E rate (Hz)','Resource Z','Current (mV equiv.)','Global rate filter (Hz)']):ax.set_ylabel(label)
    fig.subplots_adjust(left=.11,right=.98,bottom=.075,top=.95)
    output=BASE/'long_recovery_contrast';figures=output/'figures';figures.mkdir(parents=True,exist_ok=True)
    for extension in ['png','pdf']:fig.savefig(figures/f'long_recovery_contrast.{extension}',dpi=180)
    plt.close(fig)
    common.write(output/'comparison.json',dict(updated_at=time.time(),require_complete=require_complete,
        same_parameters_except_rho=same,rows=[dict(label=c['label'],job=c['job'],stats=c['stats'],
        actual_temporal_audit=c['timing'],conditional_resource_proof=c['proof']) for c in collected],
        source='Two separate30s simulations, each continued from its own full actual state to the originally planned60s. No noise reset or parameter change within a trajectory; not an independent extra replicate.',
        interpretation='Separate finite high/quiet episodes, late persistent activity and recovery of original interictal event repertoire. A repeated trace alone does not establish a stable limit cycle or Hopf bifurcation.',human_review='PENDING'))
    (figures/'README.md').write_text('### long_recovery_contrast.png / .pdf\n原生Z与rho=.25候选使用同一参数/噪声条件，各自从实际30秒状态连续延长；图示完整已保存时段的全局/双核率、Z、有效抑制与兴奋输入，以及全局活动滤波状态。没有reset或重启噪声，虚线为模型中预先固定的全局招募阈值与新方程强输入条件下的Z平衡值。**关注点**：前段短循环是否在后段失去终止能力；新增资源项是否改变这一结果，不能把条件Z平衡值当作全网固定点。\n')
    print([(c['label'],c['stats']['observed_s'],c['stats']['tail10s_mean_Hz'],c['proof']['max_mean_A_B_Z_conditional_Euler_error']) for c in collected])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');a=p.parse_args();main(a.require_complete)
