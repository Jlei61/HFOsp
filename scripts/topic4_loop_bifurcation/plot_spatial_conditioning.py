#!/usr/bin/env python3
"""Observed state fields versus the conditional template at identical mean Z."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from campaign import ROOT,NATIVE,read,write
import run_topic4_loop_zk_conditional as native


def main():
    assert read(ROOT/'exit_state_continuation_qa/gate.json')['status']=='PASS'
    g=np.load(NATIVE/'geometry.npz');positions=g['positions_e'];centers=g['centers_mm'];radius=float(g['core_radius_mm'])
    indices=np.clip(np.floor(positions*2).astype(int),0,39);cells=indices[:,0]+40*indices[:,1]
    counts=np.bincount(cells,minlength=1600)
    def field(x):return (np.bincount(cells,weights=x,minlength=1600)/np.maximum(counts,1)).reshape(40,40)
    masks=[np.linalg.norm(positions-c,axis=1)<1.75 for c in centers]
    assert [int(m.sum()) for m in masks]==g['region_counts'][:2].tolist()
    t10=native.SOURCE/'runs'/native.NAME/'states/t10s.pkl'
    t167=ROOT/'exit_state_reconstruction/runs/source10_to16p70/checkpoint.pkl'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,3,figsize=(10.8,7.8),layout='constrained',sharex=True,sharey=True)
    rows=[]
    for row,(path,label) in enumerate([(t10,'Entry vicinity: 10.00 s'),(t167,'Exit vicinity: 16.70 s')]):
        e=native.read_pickle(path)['engine'];z=e['slow']['z'][:32000];k=e['termination_mechanism']['sahp_g']
        tz,_=native.fields(float(z.mean()),float(k.mean()));numbers=[]
        for col,values in enumerate([z,tz,z-tz]):
            ax=axes[row,col];im=ax.imshow(field(values),origin='lower',extent=[0,20,0,20],interpolation='nearest',cmap='RdBu_r' if col==2 else 'viridis',vmin=-.12 if col==2 else 0.,vmax=.12 if col==2 else 1.)
            for j,c in enumerate(centers):
                ax.add_patch(Circle(c,radius,fill=False,ec='black' if col==2 else 'white',lw=1.))
                ax.add_patch(Circle(c,1.75,fill=False,ec='black' if col==2 else 'white',lw=.7,ls=':'))
                ax.text(c[0],c[1]+radius+.6,'AB'[j],ha='center',va='bottom',color='black' if col==2 else 'white',weight='bold')
            if row==0:ax.set_title(['Natural spatial field','Common-template field','Natural − template'][col])
            if col==0:ax.set_ylabel(label+'\ny (mm)')
            ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])
            core=[float(values[m].mean()) for m in masks]
            label=f'Core A {core[0]:+.3f}   Core B {core[1]:+.3f}' if col==2 else f'Core A {core[0]:.3f}   Core B {core[1]:.3f}'
            ax.set_xlabel(('x (mm)\n' if row==1 else '')+label,fontsize=10)
            numbers.append(core)
            if col==1 and row==1:fig.colorbar(im,ax=axes[:,:2],shrink=.75,label='Resource Z')
            if col==2 and row==1:fig.colorbar(im,ax=axes[:,2],shrink=.75,label='Difference in Z')
        rows.append(dict(time_s=e['step']*.0001,allEmeanZ=float(z.mean()),core_values_natural_template_difference=numbers,source=str(path)))
    fig.supxlabel('Matched all-E means: Z = 0.736 (top), 0.213 (bottom). Conditional template: 20-s field, logit shifted.\nSolid circles: physical cores; dotted circles: unchanged 1.75-mm observers used for the values.\nSpatial differences are descriptive; transition effects require paired native probes.',fontsize=9)
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    for ext in ['png','svg']:fig.savefig(out/f'spatial_Z_conditioning.{ext}',dpi=180)
    plt.close(fig)
    write(out/'spatial_Z_conditioning_metadata.json',dict(rows=rows,display_grid=40,display_operation='Per-bin mean over original E neurons; no spatial smoothing.',physical_core_radius_mm=radius,regional_observer_radius_mm=1.75,observer_neurons=[int(m.sum()) for m in masks],interpretation='Equal globalZ does not specify coreZ; no transition effect inferred from this picture alone.',exit_checkpoint_lineage='All168observationarrays and continued20s fullphysicalenginebitwise.',human_review='PENDING'))
    path=out/'README.md';text=path.read_text();marker='\n### spatial_Z_conditioning.png / spatial_Z_conditioning.svg'
    if marker in text:text=text.split(marker)[0]
    path.write_text(text+marker+'\n自然进入附近10秒、退出附近16.70秒的Z空间场，与相同全E均值下的共同20秒模板并列。右列是逐细胞差值的空间均值；数值沿用原1.75mm观测区域（虚圈），实圈为物理core半径1.5mm。退出状态已通过原观测及续到20秒的完整物理状态逐位核验，10秒比操作性进入晚60ms。\n**关注点**：平均Z相同仍可掩盖双核内资源差异；这张图只定位缺失的空间条件，转换后果需看配对原生分支。\n')


if __name__=='__main__':main()
