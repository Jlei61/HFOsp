#!/usr/bin/env python3
"""Paired native verification and a measured conditional branch candidate."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import shutil
import time
import numpy as np
from campaign import ROOT,read,write,sha
from native_mean_exit_interval import OUT,MODEL,JOBS,native as implementation
from analyze_native import analyze,original
import analyze_mean_exit_interval as model_read
from analyze_mean_boundary import first_low


def ready():
    ok=all((OUT/'runs'/n/'fixed_background_result.json').exists() for n in JOBS)
    p=MODEL/'runs/high_K9p5/result.json'
    return ok and p.exists() and read(p).get('both_RNGs_paired_with_four_probes',False)


def main(wait):
    dest=OUT/'analysis';dest.mkdir(exist_ok=True)
    while not ready():
        failed=[n for n in JOBS if (OUT/'runs'/n/'progress.json').exists() and read(OUT/'runs'/n/'progress.json')['status']=='FAILED']
        if failed:
            write(dest/'progress.json',dict(status='STOPPED_ON_NATIVE_FAILURE',names=failed));return
        write(dest/'progress.json',dict(status='WAITING_TWO_NATIVE_AND_ONE_MODEL',pid=os.getpid(),updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    write(dest/'progress.json',dict(status='ANALYZING',pid=os.getpid(),updated_epoch=time.time()))
    data={};rows=[];native_inputs=None;native_rng=None
    sizes,masks,counts,proj=model_read.projection()
    for name,K in JOBS.items():
        row,inputs=analyze(OUT,name);folder=OUT/'runs'/name
        n=dict(np.load(OUT/'extended_analysis'/f'{name}_readouts.npz'))
        me=original.load(folder/'mechanism_chunks',['time_ms','global_E_rate_Hz','global_raw_conductance_ratio'])
        dr=original.load(folder/'conditional_drift_chunks',['time_ms','values'])
        nt=me['time_ms']/1000-82;dt=dr['time_ms']/1000-82
        use=(nt>=-1e-9)&(nt<10-1e-9);dd=(dt>1e-9)&(dt<=10+1e-9)
        n=dict(field=n['field_rate_5ms_Hz'],rate=n['rate_5ms_Hz'],time=nt[use],
            R=me['global_E_rate_Hz'][use],G=me['global_raw_conductance_ratio'][use],drift=dr['values'][dd,:,0],drift_time=dt[dd])
        assert n['field'].shape==(2000,400) and n['R'].shape==(10000,) and n['drift'].shape==(500,4)
        if K==9.5:
            old=model_read.OUT;model_read.OUT=MODEL
            try:m=model_read.load_case(name,sizes,masks,counts,proj)
            finally:model_read.OUT=old
        else:m=dict(np.load(ROOT/'mean_exit_interval/analysis'/f'{name}_readouts.npz'))
        if native_inputs is None:native_inputs=inputs
        else:assert np.array_equal(native_inputs,inputs)
        rng=implementation.native.read_pickle(folder/'checkpoint.pkl')['engine']['rng_state']
        if native_rng is None:native_rng=rng
        else:assert native_rng==rng
        windows=[]
        for lo,hi in [(0,5),(5,10)]:
            s=slice(lo*1000,hi*1000);ns=slice(lo*200,hi*200);ds=slice(lo*50,hi*50)
            delta=m['field'][s].mean(0)-n['field'][ns].mean(0)
            windows.append(dict(interval_s=[lo,hi],model_rate_Hz=m['rate'][s].mean(0).tolist(),native_rate_Hz=n['rate'][ns].mean(0).tolist(),
                rate_difference_Hz=(m['rate'][s].mean(0)-n['rate'][ns].mean(0)).tolist(),
                weighted_field_RMS_Hz=float(np.sqrt(np.average(delta**2,weights=counts))),
                model_Zdot=m['drift'][s].mean(0).tolist(),native_Zdot=n['drift'][ds].mean(0).tolist(),
                Zdot_difference=(m['drift'][s].mean(0)-n['drift'][ds].mean(0)).tolist()))
        tn=first_low(n['time'],n['R']);tm=first_low(m['time'],m['R'])
        lowmatch=(tn is None and tm is None) or (tn is not None and tm is not None and abs(tn-tm)<=.25)
        guards=dict(both_windows_spatial=all(w['weighted_field_RMS_Hz']<=10 for w in windows),
            both_windows_core_rate=all(max(abs(x) for x in w['rate_difference_Hz'][1:3])<=10 for w in windows),
            both_windows_core_drift=all(max(abs(x) for x in w['Zdot_difference'][1:3])<=.01 for w in windows),
            no_wrong_R_gate=bool((n['R']<200).all() and (m['R']<200).all()),
            both_G_below_point1=bool((n['G']<.1).all() and (m['G']<.1).all()),low_transition_match=bool(lowmatch))
        rows.append(dict(name=name,K=K,windows=windows,first100ms_causal_R_le5_s=dict(native=tn,model=tm),
            guards=guards,retained=all(guards.values()),native_censoring=row['censoring']))
        data[name]=(n,m)
        np.savez_compressed(dest/f'{name}_readouts.npz',**{'native_'+k:v for k,v in n.items()},**{'model_'+k:v for k,v in m.items()})
    result=dict(status='COMPLETE_SAME_HISTORY_UPPER_INTERVAL_COMPARISON',rows=rows,both_conditions_retained=all(r['retained'] for r in rows),
        native_future_inputs_and_final_RNG_paired=True,model_future_RNG_paired_with_four_previous_probes=True,
        scope='Conditional heldfields at82s nativehistory/model100000clock. No independently sampled topology, seed or autonomousloop. A finite-time transition bracket is not a proof of stable/unstablebranches or Fold/Hopf.',
        formal_bifurcation_allowed=False,agent_visual='PENDING',human_visual='PENDING',producer_sha256=sha(__file__))
    write(dest/'result.json',result);plot(data,rows);branch_figure(data,rows)
    shutil.copy2(__file__,dest/'producer.py');write(dest/'progress.json',dict(status='COMPLETE',updated_epoch=time.time()))
    print(result,flush=True)


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    return plt


def plot(data,rows):
    plt=style();from matplotlib.patches import Circle
    fig,ax=plt.subplots(2,4,figsize=(14,7),layout='constrained')
    centers=np.load(ROOT/'native_slices/geometry.npz')['centers_mm']
    for j,row in enumerate(rows):
        n,m=data[row['name']]
        for key,color,d in [('Native','black',n),('Copied model','#8266ad',m)]:
            ax[0,2*j].plot(d['time'],d['R'],color=color,lw=.9,label=key)
            for core,ls in [(1,'-'),(2,'--')]:ax[0,2*j+1].plot(d.get('drift_time',d['time']),d['drift'][:,core],color=color,lw=.9,ls=ls)
        ax[0,2*j].set(title=f"K = {row['K']:g}",ylabel='Causal E rate (Hz)',ylim=(0,205))
        ax[0,2*j].spines['bottom'].set_position(('outward',4));ax[0,2*j].legend(frameon=False,fontsize=8)
        ax[0,2*j+1].set(title='Core A solid; B dashed',ylabel='dZ/dt if released (1/s)');ax[0,2*j+1].axhline(0,color='.6',lw=.7)
        for i,(key,d) in enumerate([('Native',n),('Copied model',m)]):
            field=d['field'][len(d['field'])//2:].mean(0)
            im=ax[1,2*j+i].imshow(field.reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0,vmax=500,interpolation='nearest')
            for xy in centers:ax[1,2*j+i].add_patch(Circle(xy,1.5,fill=False,edgecolor='#00c3c5',lw=.9))
            ax[1,2*j+i].set(title=f"{key}; K {row['K']:g}",xlabel='x (mm)',ylabel='y (mm)',xticks=[0,10,20],yticks=[0,10,20])
    for axis in ax[0]:axis.set(xlabel='Time after K change (s)',xlim=(0,10))
    fig.colorbar(im,ax=ax[1].tolist(),shrink=.7,label='E rate, 5–10 s (Hz)')
    fig.suptitle('Same-history exit interval: native spatial correspondence',weight='bold')
    for ext in ['png','svg']:fig.savefig(ROOT/f'figures/native_mean_exit_interval.{ext}',dpi=180)
    plt.close(fig)
    append_readme('native_mean_exit_interval','从同一82秒完整高史，原生网络核对K9.4625与9.5；候选模型的两个点也共用同一完整起态及未来数值随机流。图示整个转换、双核收支及相同后五秒的空间场。')


def branch_figure(data,rows):
    plt=style();from matplotlib.lines import Line2D
    fig,ax=plt.subplots(1,2,figsize=(10.4,4.9),layout='constrained')
    colors=['#8a63b4','#d63378','#008eb3'];labels=['All E','Core A','Core B']
    prev=read(ROOT/'mean_exit_interval/analysis/result.json')['rows']
    points=[]
    for row in prev:
        t=row['windows'][-1]
        d=dict(np.load(ROOT/'mean_exit_interval/analysis'/f"{row['name']}_readouts.npz"))
        points.append((row['held_mean_K'],t['mean_rate_Hz'],d['drift'][5000:].mean(0),'s' if row['history']=='quiet' else 'o',False))
    m=data['high_K9p5'][1];points.append((9.5,m['rate'][5000:].mean(0),m['drift'][5000:].mean(0),'o',False))
    for row in rows:
        t=row['windows'][-1];points.append((row['K'],t['native_rate_Hz'],t['native_Zdot'],'o',True))
    for K,rate,drift,marker,filled in points:
        for j,color in enumerate(colors):
            kw=dict(marker=marker,s=62 if not filled else 25,edgecolors=color,facecolors=color if filled else 'none',linewidths=1.1,zorder=4)
            ax[0].scatter(K,rate[j],**kw);ax[1].scatter(K,drift[j],**kw)
    # Annotate only an observed, matched transition; do not interpolate an
    # unstable branch or assert a bifurcation from these finite responses.
    low=rows[0]['windows'][-1];high=rows[1]['windows'][-1]
    bracket=all(r['retained'] for r in rows) and min(low['native_rate_Hz'][1:3])>100 and max(high['native_rate_Hz'][:3])<5
    if bracket:
        for axis in ax:axis.axvspan(9.4625,9.5,color='.94',zorder=0)
    ax[0].set(ylabel='E rate, 5–10 s (Hz)',ylim=(-15,505))
    ax[1].set(ylabel='dZ/dt if released (1/s)',ylim=(-.05,.18));ax[1].axhline(0,color='.6',lw=.7)
    for axis in ax:axis.set(xlabel=r'Held mean $K$ ($g_K/g_L$)',xlim=(9.337,9.513),xticks=[9.35,9.40,9.45,9.50])
    ax[0].legend([Line2D([],[],color=c,lw=2) for c in colors],labels,frameon=False,loc='center left',fontsize=9)
    legend=[Line2D([],[],color='.3',marker='o',ls='none',markerfacecolor='none'),Line2D([],[],color='.3',marker='o',ls='none'),Line2D([],[],color='.3',marker='s',ls='none',markerfacecolor='none')]
    ax[1].legend(legend,['Copied model: high history','Native: high history','Copied model: quiet history'],frameon=False,loc='center left',fontsize=8)
    fig.suptitle(r'Exit-related conditional states: actual field family, $\bar Z=0.21$',fontsize=12)
    note='Measured 10 s responses; no stability or critical-type certification.'
    if bracket:note+=' Shading: matched transition interval.'
    fig.text(.5,-.022,note,ha='center',fontsize=9)
    for ext in ['png','svg','pdf']:fig.savefig(ROOT/f'figures/exit_conditional_branch_candidate.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    write(OUT/'analysis/branch_figure.json',dict(status='COMPLETE_CANDIDATE_REVIEW_PENDING',same_history_transition_interval=[9.4625,9.5] if bracket else None,
        point_count=len(points),formal_bifurcation_allowed=False,agent_visual='PENDING',human_visual='PENDING'))
    append_readme('exit_conditional_branch_candidate','固定实际退出空间场族、平均Z=0.21时，展示同起态与配对未来流下的高历史响应，以及静默历史返回K9.35的响应。候选模型以空心、原生核对点以实心表示；不同区域沿用All E紫、Core A粉、Core B蓝。')


def append_readme(stem,text):
    with (ROOT/'figures/README.md').open('a') as f:f.write(f'\n\n### {stem}.png / {stem}.svg\n{text}\n**关注点**：条件钳制不算自主退出或资源恢复；仅画实际测点。若有阴影，它是本十秒协议的转换区间，未经稳定/不稳定支或分岔类型认证，人工待审。\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)
