#!/usr/bin/env python3
"""Read the complete bounded spectral pilot; do not relabel iterations stability."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time,shutil
from pathlib import Path
import numpy as np
from scipy import sparse
from campaign import ROOT,read,write,sha
from prepare_spectral_closure_pilot import OUT,N,GENERATIONS
from conditional_density_inputs import OPS
from observe_source_aggregation import OUT as OBS


def main():
    while not (OUT/'generation_3/complete.json').exists():
        if read(OUT/'supervisor.json')['status']=='FAILED':raise RuntimeError('Worker failure')
        time.sleep(10)
    raw=dict(np.load(OUT/'parameters.npz'));E=np.arange(40000)<32000;params=read(OPS/'prepared.json')['params']
    H=np.load(OUT/'filter_power.npy');w=np.full(N//2+1,2.);w[[0,-1]]=1
    W=[sparse.load_npz(OUT/f'original_{kind}_jump.npz') for kind in ['ampa','gaba']]
    sq=[a.multiply(a) for a in W]
    # Exact white external shot-noise variance at the pilot's declared constant input.
    external=raw['nu_per_ms']*.1*raw['jump_external']**2*np.dot(H[0],w)/N
    native=dict(np.load(OBS/'input_analysis/input_reconstruction.npz'))
    edge=read(OBS/'contract.json')['edge_cells'];edge=[q['cell'] for q in edge]
    masks=[('allE',E),('coreA',E&(raw['region']==0)),('coreB',E&(raw['region']==1)),
        ('surroundE',E&(raw['region']==2)),('I',~E),('selected_edges',E&np.isin(raw['display'],edge))]
    spectra=[];predictions=[]
    for gen in range(GENERATIONS+1):
        S=np.load(OUT/f'generation_{gen}/source_PSD.npy',mmap_mode='r')
        sourcevar=np.stack([S@(H[q]*w/N**2) for q in range(2)])
        rec=np.stack([sq[0]@sourcevar[0,:32000],sq[1]@sourcevar[1,32000:]])
        total=rec.copy();total[0]+=external;predictions.append(total)
        spectra.append(dict(generation=gen,rows=[dict(region=name,IE_II_variance=total[:,mask].mean(1).tolist()) for name,mask in masks]))
    results=[read(OUT/f'generation_{g}/complete.json') for g in range(1,4)]
    # Numerical uncertainty here covers replicas only, not closure/native-noise uncertainty.
    uncertainty=[]
    for gen in range(1,4):
        a=[dict(np.load(OUT/f'generation_{gen}/part{part}_statistics.npz')) for part in range(2)]
        samples=np.concatenate([d['replica_statistics'] for d in a]);r=samples[:,:,:2].sum(2)/2
        stderr=r.std(1,ddof=1)/np.sqrt(r.shape[1]);rows=[]
        for name,mask in masks:
            rows.append(dict(region=name,independent_replica_meanrate_SEM_Hz=float(np.sqrt(np.sum(stderr[mask]**2))/mask.sum()),
                target_MC_SEM_RMS_Hz=float(np.sqrt(np.mean(stderr[mask]**2))),
                native_IE_II_variance=native['observed_variance_IE_II'][:,mask].mean(1).tolist(),
                sampled_IE_II_variance=np.mean(samples[mask,:,7:9]-samples[mask,:,5:7]**2,axis=(0,1)).tolist()))
        uncertainty.append(dict(generation=gen,rows=rows))
    np.savez_compressed(OUT/'review_arrays.npz',predicted_variance=np.array(predictions))
    result=dict(status='COMPLETE_BOUNDED_SPECTRAL_PILOT_REVIEW',generations=results,
        next_generation_input_variances=spectra,numerical_replica_uncertainty=uncertainty,
        native_correspondence_certified=False,root_established=False,formal_bifurcation_allowed=False,
        physical_stability_established=False,producer_sha256=sha(__file__))
    write(OUT/'review.json',result);shutil.copy2(__file__,OUT/'review_producer.py')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(12,6.5),layout='constrained');grid=fig.add_gridspec(2,5,width_ratios=[1,1,1,1,.045])
    ax=fig.add_subplot(grid[0,:2]);bx=fig.add_subplot(grid[0,2:4])
    colors=['#8a63b4','#d63378','#008eb3'];labels=['All E','Core A','Core B']
    init=np.load(OUT/'generation_0/source_rate_Hz.npy')
    for j,(label,color,mask) in enumerate(zip(labels,colors,[E,E&(raw['region']==0),E&(raw['region']==1)])):
        vals=[float(init[mask].mean())]+[r['rows'][j]['output_rate_Hz'] for r in results]
        ax.plot(range(4),vals,'o-',label=label,color=color,lw=1.2)
    ax.set(xlabel='Spectral iteration (not physical time)',ylabel='Mean rate (Hz)',xticks=range(4),title='A  Conditional output rates')
    ax.legend(frameon=False,fontsize=9)
    vals=[r['development_native_field_weighted_RMS_Hz'] for r in results]
    bx.plot(range(1,4),vals,'o-',color='.25',label='E field vs native reference')
    for j,label,color in [(0,'E cell change per iteration','#8a63b4'),(4,'I cell change per iteration','#bf7938')]:
        bx.plot(range(1,4),[r['rows'][j]['individual_rate_RMS_change_Hz'] for r in results],
            's--',color=color,label=label)
    bx.set(xlabel='Spectral iteration',ylabel='RMS difference (Hz)',xticks=range(1,4),
        yscale='log',ylim=(.02,10),title='B  Small E-field error; I iteration grows')
    bx.legend(frameon=False,fontsize=8,loc='upper left')
    geo=dict(np.load(ROOT/'native_K9p35_held_history/geometry.npz'));display=raw['display'][:32000];n=np.bincount(display,minlength=400)
    for gen in range(4):
        r=np.load(OUT/f'generation_{gen}/source_rate_Hz.npy');field=np.bincount(display,weights=r[:32000],minlength=400)/np.maximum(n,1)
        a=fig.add_subplot(grid[1,gen]);im=a.imshow(field.reshape(20,20),origin='lower',extent=(0,20,0,20),vmin=0,vmax=500,cmap='magma',interpolation='nearest')
        for lab,xy in zip(['A','B'],geo['centers_mm']):
            a.add_patch(Circle(xy,float(geo['core_radius_mm']),fill=False,color='#00c3c5',lw=1))
            a.text(xy[0],xy[1]+2,lab,color='#00c3c5',ha='center',fontsize=8)
        a.set(xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20],title='Native initial reference' if gen==0 else f'Own-output iteration {gen}')
        if gen==0:a.set_ylabel('y (mm)')
    fig.colorbar(im,cax=fig.add_subplot(grid[1,4]),label='E rate (Hz)')
    fig.suptitle(r'Individual-source spectral pilot: held $\bar Z=0.21$, $\bar K=9.35$; fixed mean external drive',fontsize=11)
    for ext in ['png','svg']:fig.savefig(ROOT/f'figures/individual_source_spectral_pilot.{ext}',dpi=190)
    plt.close(fig)
    import xml.etree.ElementTree as ET
    ET.parse(ROOT/'figures/individual_source_spectral_pilot.svg')
    write(OUT/'figure_qa.json',dict(SVG_XML='PASS',agent_visual_review='PENDING',human_visual_review='PENDING'))
    readme=ROOT/'figures/README.md'
    if '### individual_source_spectral_pilot.png' not in readme.read_text():
        with readme.open('a') as f:
            f.write('\n### individual_source_spectral_pilot.png / .svg\n单个实际退出场上的三轮个体源频谱自洽试验：原生源率及频谱仅用于初始化，后续递归输入由上一轮模型输出生成。显示核率、与原生参考的空间差异和空间场；迭代编号不代表物理时间，试验还将外源期望强度固定为原生窗口均值。\n**关注点**：检查保留源时间结构后能否保留不对称招募；没有认证自洽根、物理稳定性或分岔，人工待审。\n')
    print('COMPLETE spectral pilot review',flush=True)


if __name__=='__main__':main()
