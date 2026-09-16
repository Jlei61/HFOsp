#!/usr/bin/env python3
"""Spectral energy from native-rate current proxies, with no projected field.

Explicit equal-length windows avoid comparing a transient to an unfiltered DC
plateau. This is a model-only diagnostic, not the clinical Fig3 estimator.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse,json
from pathlib import Path
import numpy as np
from scipy.signal import periodogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FS=10000.
KEYS=['effective_proxy','excitation_proxy','effective_global_proxy',
      'native_effective_grid','native_excitation_grid','native_global_grid']

def spectral_power(x,fs=FS,band=(1.,150.)):
    f,p=periodogram(x,fs=fs,window='hann',detrend='constant',axis=0,scaling='density')
    selected=(f>=band[0])&(f<=band[1])
    return p[selected].sum(0)*(f[1]-f[0])

def window(folder,start,duration):
    first=round(start*FS);last=first+round(duration*FS)
    times=[];values={k:[] for k in KEYS}
    for path in sorted((folder/'dense_chunks').glob('*.npz')):
        if '.tmp.' in path.name:continue
        with np.load(path) as q:
            t=np.rint(q['contact_time_ms']*10).astype(np.int64)
            mask=(t>=first)&(t<last)
            if not mask.any():continue
            times.append(t[mask])
            for k in KEYS:values[k].append(q[k][mask])
    if not times:raise ValueError('Requested window has no native-rate observation')
    times=np.concatenate(times)
    if not np.array_equal(times,np.arange(first,last)):
        raise ValueError('Requested window is incomplete or has a recording gap')
    return {k:np.concatenate(v).astype(float) for k,v in values.items()}

def main(root,name,baseline,early,duration):
    if duration<1:raise ValueError('At least1 second is needed for this1Hz-grid spectral diagnostic')
    qa=json.loads((root/'observation_qa.json').read_text())
    if qa.get('status')!='PASS' or not qa.get('entire_checkpoint_bitwise'):
        raise ValueError('Source-trajectory equality must pass before spectral figure production')
    folder=root/'runs'/name
    b=window(folder,baseline,duration);e=window(folder,early,duration)
    bp={k:spectral_power(v) for k,v in b.items()};ep={k:spectral_power(v) for k,v in e.items()}
    delta={k:ep[k]-bp[k] for k in KEYS}
    # Components share a signal: their powers need not sum to total power.
    geo=np.load(root/'geometry.npz');xy=geo['contact_xy'];centers=geo['centers_mm']
    d=folder/'native_energy';d.mkdir(exist_ok=True);figdir=d/'figures';figdir.mkdir(exist_ok=True)
    np.savez_compressed(d/'spectral_power.npz',**{'baseline_'+k:v for k,v in bp.items()},
        **{'early_'+k:v for k,v in ep.items()},**{'delta_'+k:v for k,v in delta.items()})
    vmax=max(float(np.abs(delta[k]).max()) for k in ['native_effective_grid','native_excitation_grid','effective_proxy'])
    vmax=max(vmax,1e-9)
    # Scale display units only; the saved spectra stay in unscaled power units.
    display_scale=1e5
    fig=plt.figure(figsize=(19.5,5.6))
    layout=fig.add_gridspec(1,4,width_ratios=[1,1,.055,1.3],
        left=.06,right=.98,bottom=.24,top=.91,wspace=.56)
    axes=[fig.add_subplot(layout[k]) for k in [0,1,3]]
    maps=[delta['native_effective_grid'],delta['native_excitation_grid']]
    for i,(ax,z) in enumerate(zip(axes[:2],maps)):
        im=ax.imshow(z.reshape(20,20)/display_scale,origin='lower',extent=[0,20,0,20],
            cmap='RdBu_r',vmin=-vmax/display_scale,vmax=vmax/display_scale,interpolation='nearest')
        if i==0:ax.scatter(*xy.T,c=delta['effective_proxy']/display_scale,cmap='RdBu_r',vmin=-vmax/display_scale,vmax=vmax/display_scale,
            s=42,edgecolors='black',linewidths=.8,zorder=5)
        for core,c in zip('AB',centers):
            ax.add_patch(plt.Circle(c,float(geo['core_radius_mm']),fill=False,ec='#31a5a9',lw=1.4))
            ax.text(*c,core,color='#20777c',weight='bold',ha='center')
        ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
    cbar=fig.colorbar(im,cax=fig.add_subplot(layout[2]))
    cbar.ax.set_title('ΔPower\n[10⁵ (mV equiv.)²]',fontsize=11,pad=13)
    cbar.ax.tick_params(labelsize=12)
    xx=np.arange(len(xy))
    for y,col,label in [(delta['effective_proxy'],'#3e719c','Total synaptic proxy'),
                         (delta['excitation_proxy'],'#c4654d','Excitatory proxy'),
                         (delta['effective_global_proxy'],'#4e9479','Global inhibitory proxy')]:
        axes[2].plot(xx,y/display_scale,'o-',ms=4,lw=1.1,c=col,label=label)
    axes[2].axhline(0,color='.5',lw=.7);axes[2].set_xticks(xx,geo['contact_names'],rotation=65,ha='right')
    axes[2].set_ylabel('ΔPower [10⁵ (mV equiv.)²]');axes[2].legend(frameon=False,fontsize=11)
    for i,ax in enumerate(axes):
        ax.text(-.15,1.05,'ABC'[i],transform=ax.transAxes,fontsize=22,weight='bold')
        ax.tick_params(labelsize=13);ax.xaxis.label.set_fontsize(17);ax.yaxis.label.set_fontsize(17)
    fig.canvas.draw()
    position=cbar.ax.get_position();mp=axes[1].get_position()
    cbar.ax.set_position([position.x0,mp.y0,position.width,mp.height])
    for ext in ['png','pdf']:fig.savefig(figdir/f'native_early_band_energy.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    metadata=dict(source=str(folder),baseline_s=[baseline,baseline+duration],early_s=[early,early+duration],
        sample_rate_Hz=FS,frequency_resolution_Hz=1/duration,band_Hz=[1,150],
        estimator='Hann-window periodogram, per-window mean removed; sum density times frequency-bin width over1–150Hz. Native10kHz samples, no pre-FFT downsampling.',
        observable='Current-magnitude proxy |IE|+|Z*deliveredII|, not raw measured SEEG. Native fields average actual cells within1mm bins; contact dots use the independently weighted electrode readout.',
        limitation='An onset ramp contributes low-frequency spectral energy. Band energy alone does not demonstrate sustained oscillation. Component powers are not additive because of cross-covariance. This model-only estimator does not replace the canonical clinical Fig3 pipeline.',
        negative_values='Negative delta denotes reduced power relative to baseline; it is not negative absolute power. Do not clip reductions or recolor them as increases.',human_review='PENDING')
    (d/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (figdir/'README.md').write_text('### native_early_band_energy.png / .pdf\nA为直接细胞网格的实际突触电流幅值代理谱能量变化，电极圆点独立使用对应接触读出；B单列兴奋分量，C列出接触总代理与分量。相同长度的早期及基线窗口均用原生10kHz计算1–150Hz谱能量，未将DC平台或电极平滑插值当成底层空间场。**关注点**：带内功率变化不等于持续振荡；正负变化、频谱分辨率及新增全局反馈的直接读出贡献均需保留。\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True)
    p.add_argument('--baseline',type=float,default=.5);p.add_argument('--early',type=float,required=True)
    p.add_argument('--duration',type=float,default=1.);a=p.parse_args();main(a.root,a.name,a.baseline,a.early,a.duration)
