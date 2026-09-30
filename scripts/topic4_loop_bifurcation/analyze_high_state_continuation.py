#!/usr/bin/env python3
"""Collect complete finite K ramps without promoting them to bifurcations."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
from scipy import sparse
from campaign import ROOT,read,write,sha
from continue_target_high_state import OUT,JOBS,SOURCE
from coupled_density_exit import ADAPTED


def load(folder,geo):
    with np.load(folder/'trajectory.npz') as z:
        v=z['group_output'];t=z['elapsed_time_ms']/1000;R=z['global_R_Hz'];G=30*z['global_s']
    size=geo['group_size'];E=geo['population']==0;reg=geo['group_region'];masks=[E]+[E&(reg==j) for j in range(3)]
    rate=np.array([np.average(v[:,0,m],axis=1,weights=size[m]) for m in masks]).T
    drift=np.array([np.average((v[:,8,m]-v[:,1,m])/5,axis=1,weights=size[m]) for m in masks]).T
    K=np.average(v[:,3,E],axis=1,weights=size[E])
    count=np.bincount(geo['group_cell'][E],weights=size[E],minlength=400)
    S=sparse.coo_matrix((size[E]/np.maximum(count[geo['group_cell'][E]],1),
        (geo['group_cell'][E],np.flatnonzero(E))),shape=(400,len(size))).tocsr()
    field=np.asarray((S@v[:,0].T).T)
    assert np.allclose(field@count/count.sum(),rate[:,0],atol=3e-4)
    return dict(time_s=t,rate_Hz=rate,drift_per_s=drift,K=K,causal_R_Hz=R,Graw=G,field_Hz=field)


def analysis(d,name):
    # Non-overlapping 20ms population rates define a sustained low core state.
    # This interval is fixed before ramp data exist; raw1ms rate is retained.
    q=d['rate_Hz'].reshape(-1,20,4).mean(1)
    quiet=(q[:,1:3]<5).all(1)
    hits=np.flatnonzero(np.convolve(quiet.astype(int),np.ones(50,dtype=int),'valid')==50)
    transition=None
    if len(hits):
        j=int(hits[0]);a=j*20;b=(j+50)*20-1
        transition=dict(first_low_interval_s=[float(d['time_s'][a]-.001),float(d['time_s'][b])],
            K_start=float(d['K'][a]),K_confirmation=float(d['K'][b]),
            definition='Both core20msrates<5Hz throughout50consecutivebins; imposed parametertrajectory,not a bifurcation estimator.')
    tail=d['time_s']>d['time_s'][-1]-3
    return dict(name=name,complete_duration_s=float(d['time_s'][-1]),
        final3s_rates_allE_A_B_surround_Hz=d['rate_Hz'][tail].mean(0).tolist(),
        final3s_Graw=float(d['Graw'][tail].mean()),final3s_dZ_per_s=d['drift_per_s'][tail].mean(0).tolist(),
        both_cores_sustained_low=transition,low_transition_right_censored=transition is None,
        imposedK_range=[float(d['K'].min()),float(d['K'].max())])


def main(wait):
    dest=OUT/'analysis';dest.mkdir(exist_ok=True)
    assert not (dest/'result.json').exists()
    geo=dict(np.load(ADAPTED/'geometry.npz'));data={};rows=[]
    write(dest/'contract.json',dict(question='Does a carried high state lose both cores during the fixed K ramp?',
        definition='20msnonoverlappingcorepopulationrates; bothcoresbelow5Hzfor1s. Full raw readouts retained. Terminal3s summaries and fixed fourfield snapshots.',
        registered_before_ramp_completion=True,created_epoch=time.time(),producer_sha256=sha(__file__)))
    while True:
        for name in JOBS:
            if name in data or not (OUT/name/'result.json').exists():continue
            assert read(OUT/name/'result.json')['status']=='COMPLETE'
            d=load(OUT/name,geo);data[name]=d;row=analysis(d,name);rows.append(row)
            np.savez_compressed(dest/f'{name}.npz',**d);write(dest/f'{name}.json',row)
            print('CARRIED K ANALYSIS',row,flush=True)
        status=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
        if len(data)==3 or (status.get('status')=='FAILED' and not status.get('active')):break
        write(dest/'progress.json',dict(status='WAITING_COMPLETE_FIXED_RUNS',completed=list(data),pid=os.getpid(),updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    reference=load(SOURCE,geo);comparison=None
    if 'control_K9' in data:
        d=data['control_K9'];count=np.bincount(geo['group_cell'][geo['population']==0],weights=geo['group_size'][geo['population']==0],minlength=400)
        diff=d['field_Hz'][-2000:].mean(0)-reference['field_Hz'][-2000:].mean(0)
        comparison=dict(original_last2s_rates_Hz=reference['rate_Hz'][-2000:].mean(0).tolist(),
            constant_input_control_last2s_rates_Hz=d['rate_Hz'][-2000:].mean(0).tolist(),
            weighted_field_RMS_Hz=float(np.sqrt(np.average(diff**2,weights=count))),
            note='This exposes replacing the prescribed time-varying external mean by the stationary mean used for DirectDC. It is not an independent native validation.')
    result=dict(status='COMPLETE' if len(data)==3 else 'STOPPED_ON_CONTROL_OR_RUN_FAILURE',rows=rows,constant_drive_comparison=comparison,
        statistical_unit='One continued128replica density state, paired numerical streams across two forced ramps; no independent native seeds.',
        scope='HeldZ and prescribedK; finite trajectories do not certify stationary branches, fold, Hopf, autonomous exit or Zrecovery.',
        formal_bifurcation_allowed=False,producer_sha256=sha(__file__))
    write(dest/'result.json',result)
    if len(data)==3:plot(data)
    write(dest/'progress.json',dict(status=result['status'],completed=list(data),updated_epoch=time.time()))


def plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(13.5,8),layout='constrained')
    gs=fig.add_gridspec(3,4,height_ratios=[1,1,1.05])
    colors=['#8a63b4','#d63378','#008eb3','#647582'];labels=['All E','Core A','Core B','Surround E']
    threshold=95.19851312666987/(18+17.662847938268442)
    for column,name in enumerate(['ramp2s_K10p5','ramp10s_K10p5']):
        d=data[name];rate=d['rate_Hz'].reshape(-1,20,4).mean(1);t=d['time_s'].reshape(-1,20).mean(1)
        ax=fig.add_subplot(gs[0,column*2:column*2+2])
        for j in range(4):ax.plot(t,rate[:,j],color=colors[j],lw=1,label=labels[j])
        ax.set(title=f'{"AB"[column]}  {JOBS[name][1]:g} s K ramp, then 5 s hold',xlabel='Time since ramp start (s)',ylabel='Rate (Hz)',ylim=(-5,505))
        ax.axvline(JOBS[name][1],ls=':',lw=.9,color='.5')
        if column==0:ax.legend(frameon=False,ncol=4,fontsize=8,loc='upper right')
        ax=fig.add_subplot(gs[1,column*2]);k=d['K'].reshape(-1,20).mean(1)
        for j in range(4):ax.plot(k,rate[:,j],color=colors[j],lw=1)
        ax.set(title=f'{"CD"[column]}  Finite K trajectory',xlabel=r'Mean imposed $K$ ($g_K/g_L$)',ylabel='Rate (Hz)',xlim=(8.96,10.54),ylim=(-5,505))
        ax=fig.add_subplot(gs[1,column*2+1]);ax.plot(d['time_s'],d['Graw'],color='#b27228',lw=1)
        ax.axhline(threshold,ls=':',lw=.9,color='#637963')
        ax.set(title='Global feedback',xlabel='Time (s)',ylabel=r'$G_{\rm raw}$')
    # Use the same fixed elapsed windows for the slow ramp regardless of result.
    slow=data['ramp10s_K10p5'];windows=[(0,.25),(4.75,5.25),(9.5,10),(14,15)]
    centers=np.load(ROOT/'native_slices/geometry.npz')['centers_mm']
    for col,(lo,hi) in enumerate(windows):
        ax=fig.add_subplot(gs[2,col]);mask=(slow['time_s']>lo)&(slow['time_s']<=hi)
        im=ax.imshow(slow['field_Hz'][mask].mean(0).reshape(20,20),origin='lower',extent=(0,20,0,20),
            vmin=0,vmax=500,cmap='magma',interpolation='nearest')
        for xy in centers:ax.add_patch(Circle(xy,1.5,fill=False,edgecolor='#00c3c5',lw=.9))
        ax.set(title=f'Slow ramp: {lo:g}–{hi:g} s',xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20])
        if col==0:ax.set_ylabel('y (mm)')
    fig.colorbar(im,ax=fig.axes[-4:],label='E rate (Hz)',shrink=.8)
    fig.suptitle('High-state continuation under prescribed K',weight='bold')
    fig.text(.5,-.025,'Actual exit Z field held at mean 0.21; constant expected external input. Forced density trajectories, not equilibrium branches or autonomous recovery.',ha='center',fontsize=9)
    for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'carried_high_state_K_ramps.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    write(OUT/'analysis/figure_metadata.json',dict(agent_visual='PENDING',human_review='PENDING',formal_Fig5_replaced=False,producer_sha256=sha(__file__)))
    readme=ROOT/'figures/README.md'
    with readme.open('a') as handle:
        handle.write('\n### carried_high_state_K_ramps.png / .svg\n从已形成的高活动完整密度状态出发，在固定实际退出空间 Z 场下，比较 2 秒和 10 秒的 K 增长及随后 5 秒保持。上部显示四个群体和全局反馈，下部显示慢变化条件的固定时段空间招募；K 不变对照及其外源均值替换误差另存分析表。曲线属于人为参数变化下的有限轨迹，不能直接标为平衡分支、正式分岔或自主恢复；候选尚待人工目视检查。\n**关注点**：核外活动收缩是否进一步使两核退出，以及转换是否明显依赖参数变化速度。\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)
