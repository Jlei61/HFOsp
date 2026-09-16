#!/usr/bin/env python3
"""Compare completed transfer-correction trajectories using the fixed readouts."""
from topic4_spatial_boundary_common import OUT,OLD,REFERENCE,read,write,observables
from plot_topic4_spatial_boundary_results import save
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse


def main(wait_for_result=False):
    if wait_for_result:
        while True:
            p=OUT/'mixed_timescale_status.json';status=read(p) if p.exists() else {}
            if status.get('status')=='COMPLETE':break
            if status.get('status')=='FAILED':raise RuntimeError(status)
            os.kill(read(OUT/'mixed_timescale_process.json')['pid'],0);time.sleep(15)
    native=np.load(REFERENCE/'trajectory.npz');src=np.load(OLD/'external_input.npz');weights=src['count_e']
    ag=np.zeros((100,400));ag[src['cell_e'],native['cell_e']]=1
    native_counts=native['field_e_count_1ms']@ag.T;native_rates=native_counts/weights*1000
    zmap=np.bincount(src['cell_e']*400+native['cell_e'],minlength=40000).reshape(100,400)/weights[:,None]
    native_z=native['z_field_10ms']@zmap.T
    fig,axs=plt.subplots(4,2,figsize=(14,12),layout='constrained');rows=[]
    e_native=native['rate_e_hz'].reshape(-1,50).mean(1);i_native=native['rate_i_hz'].reshape(-1,50).mean(1)
    windows=[('baseline',500,1000),('recurrent_event_window',5000,7000),('boundary_window',8000,9400),('high_window',10180,10680),('restored',12680,13680)]
    for col,mode in enumerate(('native_replay','autonomous_gaussian')):
        t=(np.arange(len(e_native))+.5)*.005
        axs[0,col].plot(t,e_native,c='k',lw=.8,label='Native SNN');axs[1,col].plot(t,i_native,c='k',lw=.8,label='Native SNN')
        axs[2,col].plot(native['z_time_ms']/1000,native['z_stats'][:,0],c='k',lw=1.2,label='Native SNN')
        sm_native=native_rates.reshape(-1,10,100).mean(1)
        tm=[];coverage=[]
        for end in range(100, len(sm_native)+1,25):
            duty=np.mean(sm_native[end-100:end]>50,axis=0);tm.append(end*.01);coverage.append(np.average(duty>=.8,weights=weights))
        axs[3,col].plot(tm,coverage,c='k',lw=1.2,label='Native SNN')
        for label,path,color in [('Previous closure',OLD/'rate'/f'{mode}_expected.npz','#ae3a6b'),
                                  ('Mixed-timescale candidate',OUT/'mixed_timescale'/f'{mode}.npz','#387f9f')]:
            a=np.load(path);fields=a['fields_hz'];e=np.average(fields[:,0],axis=1,weights=a['count_e']);i=np.average(fields[:,1],axis=1,weights=a['count_i']);z=a['z']
            axs[0,col].plot((np.arange(len(e)//5)+.5)*.005,e.reshape(-1,5).mean(1),c=color,lw=.8,label=label)
            axs[1,col].plot((np.arange(len(i)//5)+.5)*.005,i.reshape(-1,5).mean(1),c=color,lw=.8,label=label)
            axs[2,col].plot((np.arange(len(z))+1)*.001,np.average(z,axis=1,weights=weights),c=color,lw=1.2,label=label)
            sm=fields[:,0].reshape(-1,10,100).mean(1);cov=[]
            for end in range(100,len(sm)+1,25):cov.append(np.average(np.mean(sm[end-100:end]>50,axis=0)>=.8,weights=weights))
            axs[3,col].plot(tm,cov,c=color,lw=1.2,label=label)
            counts=fields[:,0]*weights/1000;metrics={name:observables(counts[lo:hi],weights,hi-lo) for name,lo,hi in windows}
            e10=e.reshape(-1,10).mean(1);hit=np.flatnonzero(np.convolve((e10>=200).astype(int),np.ones(20,dtype=int),'valid')==20)
            # Rate Z is recorded just before its last substep update; match nearest native 10-ms state.
            ix=np.arange(10,10681,10);diff=z[ix-1]-native_z[ix//10]
            early=ix<=6000
            rows.append({'mode':mode,'closure':label,'path':str(path),'high_trigger_ms':float((hit[0]+20)*10) if len(hit) else None,
                'mean_Z_RMSE_10ms_to_6s':float(np.sqrt(np.mean(np.average(diff[early],axis=1,weights=weights)**2))),
                'spatial_Z_RMSE_10ms_to_6s':float(np.sqrt(np.mean(np.average(diff[early]**2,axis=1,weights=weights)))),
                'end_spatial_Z_RMSE_10680ms':float(np.sqrt(np.average(diff[-1]**2,weights=weights))), 'windows':metrics})
        axs[0,col].set_title(('Prescribed native Z','Autonomous original-Z closure')[col])
        for row,ylabel in enumerate(['Mean E rate (Hz)','Mean I rate (Hz)','Mean E-target Z','Persistent spatial fraction\n(>50 Hz; >=80% duty in trailing 1 s)']):
            axs[row,col].set(xlabel='Time (s)',ylabel=ylabel,xlim=(0,13.68));axs[row,col].axvspan(10.68,11.68,color='#68ab88',alpha=.13)
            axs[row,col].legend(fontsize=8,ncol=1)
        axs[0,col].set_ylim(0,500);axs[1,col].set_ylim(0,1000);axs[2,col].set_ylim(.25,1.02);axs[3,col].set_ylim(-.025,1.025)
    fig.suptitle('Does correcting the transfer approximation restore the original network transition?',fontsize=15)
    save(fig,'mixed_timescale_full_trajectory_comparison',
        '比较原SNN、旧传递函数及未拟合参数的快AMPA/慢GABA候选在原Z回放与自主Z两种协议下的完整轨迹。各列同时展示全E/I率、平均Z及共同10×10格的持续空间占据，绿色区域为相同的外部Z恢复。',
        '早期Z或静态传递率改善不能代替完整自主边界匹配；持续占据使用固定的50Hz与80%时间标准，并非患者传播模式分类。')
    write(OUT/'mixed_timescale_validation_comparison.json',{'status':'COMPLETE_PENDING_SCIENTIFIC_REVIEW',
        'rows':rows,'native_high_trigger_ms':10180,'native_windows':{name:observables(native_counts[lo:hi],weights,hi-lo) for name,lo,hi in windows},
        'source_scope':'One fixed C network and input realization; projection, timing convention and thresholds held fixed.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');args=p.parse_args()
    try:main(args.wait)
    except Exception as exc:
        write(OUT/'mixed_timescale_analysis_status.json',{'status':'FAILED','error':repr(exc)});raise
