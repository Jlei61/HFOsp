#!/usr/bin/env python3
"""Finite-map residual and spectrum diagnostics after the bounded damped run."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time,shutil
import numpy as np
from scipy import sparse
from campaign import ROOT,read,write,sha
from damped_spectral_pilot import OUT,SOURCE,N
from conditional_density_inputs import OPS


def main():
    while not (OUT/'result.json').exists():
        if read(OUT/'supervisor.json')['status']=='FAILED':raise RuntimeError('Sampler failed')
        time.sleep(10)
    d=read(OUT/'result.json');rows=d['updates'];g=len(rows)
    assert 0<g<=12
    raw=dict(np.load(OUT/'parameters.npz'));p=read(OPS/'prepared.json')['params']
    final=OUT/f'generation_{g}';previous=OUT/f'generation_{g-1}'
    x=np.load(previous/'source_PSD.npy',mmap_mode='r');y=np.load(final/'response_PSD.npy',mmap_mode='r')
    r=np.load(previous/'source_rate_Hz.npy');rr=np.load(final/'response_rate_Hz.npy')
    H=np.load(OUT/'filter_power.npy');w=np.full(N//2+1,2.);w[[0,-1]]=1
    v=np.zeros((2,40000));delta=v.copy()
    for lo in range(0,40000,128):
        a=np.asarray(x[lo:lo+128]);b=np.asarray(y[lo:lo+128])
        for q in range(2):
            v[q,lo:lo+len(a)]=a@(H[q]*w/N**2)
            delta[q,lo:lo+len(a)]=abs(b-a)@(H[q]*w/N**2)
    current_mean=[];current_var=[];current_l1=[]
    for q,(kind,sources) in enumerate([('ampa',slice(0,32000)),('gaba',slice(32000,40000))]):
        W=sparse.load_npz(SOURCE/f'original_{kind}_jump.npz')
        current_mean.append(W@((rr[sources]-r[sources])*.0001)/(1-np.exp(-.1/p['tau_r_'+kind.upper()])))
        Q=W.multiply(W);current_var.append(Q@v[q,sources]);current_l1.append(Q@delta[q,sources])
    current_mean=np.array(current_mean);current_var=np.array(current_var);current_l1=np.array(current_l1)
    stats=np.concatenate([np.load(final/f'part{i}_statistics.npz')['replica_statistics'] for i in range(2)])
    samples=stats[:,:,:2].sum(2)/2;sem=samples.std(1,ddof=1)/4
    E=np.arange(40000)<32000
    regional=[]
    for label,mask in [('allE',E),('coreA',E&(raw['region']==0)),('coreB',E&(raw['region']==1)),('surroundE',E&(raw['region']==2)),('I',~E),('activeE',E&(rr>1)),('inactiveE',E&(rr<=1))]:
        regional.append(dict(region=label,count=int(mask.sum()),
            undamped_cell_rate_residual_RMS_Hz=float(np.sqrt(np.mean((rr[mask]-r[mask])**2))),
            output_MC_SEM_RMS_Hz=float(np.sqrt(np.mean(sem[mask]**2))),
            recurrent_IE_II_mean_residual_RMS_mV=np.sqrt(np.mean(current_mean[:,mask]**2,axis=1)).tolist(),
            recurrent_IE_II_variance_mean=current_var[:,mask].mean(1).tolist(),
            filtered_PSD_L1_current_upper_bound_mV2=current_l1[:,mask].mean(1).tolist(),
            negative_II_fraction=float(stats[mask,:,12].mean())))
    np.savez_compressed(OUT/'final_residual_arrays.npz',current_mean_residual=current_mean,
        recurrent_input_variance=current_var,filtered_spectral_L1_bound=current_l1,rate_output_SEM_Hz=sem)
    result=dict(status='COMPLETE_DAMPED_PILOT_REVIEW',updates=g,regional_final_residuals=regional,
        last=rows[-1],root_certified=False,physical_stability_established=False,formal_bifurcation_allowed=False,
        sampling='MCSEM describes output replicates at one supplied candidate input. It does not include fixed-point input uncertainty, finite2s bias, or closure/native error.',producer_sha256=sha(__file__))
    write(OUT/'review.json',result);shutil.copy2(__file__,OUT/'review_producer.py')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(12,3.7),layout='constrained');iterations=np.arange(1,g+1)
    for j,label,color in [(0,'All E','#8a63b4'),(4,'I','#bf7938')]:
        axs[0].semilogy(iterations,[a['rows'][j]['individual_rate_RMS_change_Hz'] for a in rows],'o-',color=color,label=label,ms=4)
    axs[0].set(title='A  Undamped map residual',xlabel='Numerical update',ylabel='Cell-rate RMS residual (Hz)');axs[0].legend(frameon=False)
    axs[1].plot(iterations,[a['actual_native_field_RMS_Hz'] for a in rows],'o-',color='.25',ms=4)
    axs[1].set(title='B  Native development reference',xlabel='Numerical update',ylabel='E-field RMS difference (Hz)')
    for j,label,color in [(1,'Core A','#d63378'),(2,'Core B','#008eb3')]:
        axs[2].plot(iterations,[a['rows'][j]['output_rate_Hz'] for a in rows],'o-',label=label,color=color,ms=4)
    axs[2].set(title='C  Conditional core outputs',xlabel='Numerical update',ylabel='Rate (Hz)');axs[2].legend(frameon=False)
    fig.suptitle('Same spectral map; numerical damping = 0.1. Iterations are not physical time.',fontsize=11)
    for ext in ['png','svg']:fig.savefig(ROOT/f'figures/damped_spectral_pilot.{ext}',dpi=190)
    plt.close(fig)
    import xml.etree.ElementTree as ET
    ET.parse(ROOT/'figures/damped_spectral_pilot.svg')
    write(OUT/'figure_qa.json',dict(SVG_XML='PASS',agent_visual_review='PENDING',human_visual_review='PENDING'))
    readme=ROOT/'figures/README.md'
    if '### damped_spectral_pilot.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### damped_spectral_pilot.png / .svg\n保留同一物理输入—输出映射，仅用0.1数值混合检查谱迭代；图示未乘阻尼系数的真实残差、原生空间参考差异及两核率。迭代不代表物理时间，固定平均外源、有限谱窗和源间相关近似均保留。\n**关注点**：求解器残差变化不等同网络稳定性，没有认证固定点或分岔，人工待审。\n')
    print(result,flush=True)


if __name__=='__main__':main()
