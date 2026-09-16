#!/usr/bin/env python3
"""Independent native-field diagnostic: where persistent activity first latches."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import analyze_topic4_autonomous_recovery as audit

def first_persistent(rate,threshold=100.,bins=10):
    hit=rate>=threshold
    sums=np.cumsum(np.pad(hit.astype(np.int32),((1,0),(0,0))),axis=0)
    good=sums[bins:]-sums[:-bins]>=bins
    out=np.full(rate.shape[1],np.nan);mask=good.any(0)
    out[mask]=np.argmax(good[:,mask],axis=0)*.02
    return out

def main(root):
    geo=np.load(root/'geometry.npz');nr=geo['region_counts'];nc=geo['cell_e_counts'];centers=geo['centers_mm']
    records=[];maps=[]
    for folder in sorted((root/'runs').iterdir()):
        if folder.name.startswith('qa_') or not (folder/'chunks').exists():continue
        a=audit.load(folder,['spikes_1ms','regions_1ms','field_5ms'])
        if a is None:continue
        n=len(a['field_5ms'])//4
        fields=a['field_5ms'][:n*4].reshape(n,4,400).sum(1)/nc/.02
        regions=a['regions_1ms'][:n*20,:3].reshape(n,20,3).sum(1)/nr[:3]/.02
        onset=first_persistent(fields);regional=first_persistent(regions)
        gx,gy=np.meshgrid(np.arange(20)+.5,np.arange(20)+.5)
        positions=np.c_[gx.ravel(),gy.ravel()]
        # The engine's field index is iy*n_grid+ix; keep y,x reshape consistent.
        distance=np.linalg.norm(positions[:,None]-centers[None],axis=2).min(1)
        earliest=float(np.nanmin(onset)) if np.isfinite(onset).any() else None
        initial=np.isfinite(onset)&(onset<=earliest+.1) if earliest is not None else np.zeros(400,bool)
        records.append(dict(name=folder.name,observed_s=n*.02,first_persistent_cell_s=earliest,
            core_A_B_surround_first_persistent_s=[float(v) if np.isfinite(v) else None for v in regional],
            first100ms_persistent_cells=int(initial.sum()),first100ms_fraction_outside_analysis_core_neighborhoods=float((distance[initial]>1.75).mean()) if initial.any() else None,
            interpretation='First sustained firing is distinct from event initiation. A finite core burst may launch propagation before a remote region first latches. This diagnostic alone cannot label an origin ectopic or reject a propagation mechanism.'))
        maps.append((folder.name,onset.reshape(20,20),n*.02))
    audit.write(root/'persistent_recruitment_audit.json',dict(definition='Native 1mm field cells, 20ms spike-count bins, at least100Hz in every bin for200ms; onset is the first bin start. Regional criterion uses actual neuron counts. Independent diagnostic, not a training loss or replacement high-state endpoint.',records=records))
    if not maps:return
    cols=4;rows=(len(maps)+cols-1)//cols
    fig,axs=plt.subplots(rows,cols,figsize=(4.3*cols,4*rows),squeeze=False)
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#eeeeee');maximum=max(m[2] for m in maps)
    for ax,(name,values,duration) in zip(axs.flat,maps):
        im=ax.imshow(values,origin='lower',extent=[0,20,0,20],vmin=0,vmax=maximum,cmap=cmap)
        for i,c in enumerate(centers):ax.add_patch(Circle(c,1.5,fill=False,edgecolor='#e87535',lw=1.5));ax.text(c[0],c[1]+1.8,'AB'[i],ha='center',fontsize=12)
        ax.set_title(name.replace('_s9108401','')+f'\n0–{duration:g} s',fontsize=11)
        ax.set(xlabel='x (mm)',ylabel='y (mm)',xticks=[0,10,20],yticks=[0,10,20]);ax.tick_params(labelsize=12)
    for ax in list(axs.flat)[len(maps):]:ax.remove()
    fig.subplots_adjust(left=.05,right=.88,bottom=.06,top=.94,hspace=.5,wspace=.36)
    ca=fig.add_axes([.92,.16,.017,.7]);cb=fig.colorbar(im,cax=ca);cb.set_label('First persistent activity (s)',fontsize=15)
    dest=root/'recruitment_origin/figures';dest.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(dest/f'first_persistent_activity.{ext}',dpi=170)
    plt.close(fig)
    (dest/'README.md').write_text('### first_persistent_activity.png\n直接从原生二维细胞计数计算，每个1毫米空间格第一次连续200毫秒保持至少100Hz的时间；灰色表示在已保存时长内未出现。圆圈是静态阈值核的实际1.5毫米半径。**关注点**：持续活动首先锁定的位置不等同于最初事件的起源；传播可由更早的有限核内事件触发。\n\n### first_persistent_activity.pdf\n同一诊断的矢量版本。**关注点**：不同条件当前观测时长标在各图上，不能把未观测尾部视作稳定。\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=audit.OUT);args=ap.parse_args();main(args.root)
