#!/usr/bin/env python3
"""Matched six-second spatial readouts, without assuming a bifurcation."""
from topic4_spatial_boundary_common import OUT, OLD, read, write, observables
from plot_topic4_spatial_boundary_results import save
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse


def main(wait=False):
    if wait:
        while True:
            p=OUT/'mixed_boundary_status.json';r=read(p) if p.exists() else {}
            if r.get('status')=='COMPLETE':break
            if r.get('status')=='FAILED':raise RuntimeError(r)
            os.kill(read(OUT/'mixed_boundary_process.json')['pid'],0);time.sleep(15)
    src=np.load(OLD/'external_input.npz');weights=src['count_e']
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained');rows=[]
    for row,zt in enumerate((8800,9400)):
        name=f'z{zt}_history8000';a=np.load(OUT/'native'/f'{name}.npz');b=np.load(OUT/'native'/f'{name}_extend4s.npz')
        native=np.concatenate([a['field_e_count_1ms'],b['field_e_count_1ms']])
        mapping=np.zeros((100,400));mapping[src['cell_e'],a['cell_e']]=1
        counts=native@mapping.T;rates=counts/weights*1000
        candidate=np.load(OUT/'mixed_boundary'/f'{name}.npz');other=candidate['fields_hz'][:,0]
        assert len(rates)==len(other)==6000 and np.array_equal(candidate['count_e'],weights)
        record={'name':name,'mean_Z':float(np.average(candidate['z'],weights=weights))}
        for j,(label,field,color) in enumerate((('Native SNN',rates,'#252525'),('Corrected rate',other,'#387f9f'))):
            e=np.average(field,axis=1,weights=weights);e5=e.reshape(-1,5).mean(1)
            axs[row,0].plot((np.arange(len(e5))+.5)*.005,e5,c=color,lw=.8,label=label)
            duty=np.mean(field[-1000:].reshape(100,10,100).mean(1)>50,axis=0)
            im=axs[row,j+1].imshow(duty.reshape(10,10),origin='lower',extent=[0,20,0,20],vmin=0,vmax=1,cmap='magma')
            axs[row,j+1].set(title=f'{label}: last 1 s',xlabel='x (mm)',ylabel='y (mm)')
            record[label]=[observables(field[k:k+1000]*weights/1000,weights) for k in range(0,6000,1000)]
        axs[row,0].set(title=f'Fixed native Z from {zt/1000:g} s',xlabel='Time after Z fixation (s)',ylabel='Mean E rate (Hz)',xlim=(0,6),ylim=(0,500))
        axs[row,0].legend(fontsize=9)
        rows.append(record)
    fig.colorbar(im,ax=axs[:,1:],label='Fraction of 10-ms bins >50 Hz',shrink=.65)
    fig.suptitle('Does the corrected fast model reproduce both frozen-Z spatial regimes?',fontsize=15)
    save(fig,'mixed_timescale_frozen_boundary_comparison',
        '在同一个8秒快速历史来源、9.4秒未来输入下，分别冻结原8.8秒与9.4秒Z场，比较原SNN和修正rate model连续6秒的放电率及最后1秒空间占据。原生20格计数精确合并为共同10格；rate的初态来自它自己的原Z回放。',
        '分别检验自限侧与持续侧；相同Z和输入不等于相同微观初态，有限时间匹配也不等于已经识别渐近吸引子或分岔。')
    write(OUT/'mixed_boundary_comparison.json',{'status':'COMPLETE_PENDING_SCIENTIFIC_REVIEW','rows':rows,
        'scope':'Two finite-time frozen-field checks, not a bifurcation proof or autonomous-Z validation.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    try:main(args.wait)
    except Exception as exc:
        write(OUT/'mixed_boundary_analysis_status.json',{'status':'FAILED','error':repr(exc)});raise
