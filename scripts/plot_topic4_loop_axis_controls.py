#!/usr/bin/env python3
"""Achieved spatial structure, not requested kernel parameters or dynamics."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json
import numpy as np
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from run_topic4_loop_axis_native import ROOT, graph_folder


def main():
    with np.load(ROOT.parent/'geometry.npz') as f:
        pos=f['positions_e']
    bins=np.linspace(-3.,3.,81);angles=np.linspace(0,2*np.pi,73)
    rows=[]
    for condition in ['reference','rotated','isotropic']:
        folder=graph_folder(condition);audit=json.loads((folder/'audit.json').read_text())
        hist=np.zeros((80,80));polar=np.zeros(72);total=0.
        for path in sorted((folder/'ampa_by_delay').glob('*.npz')):
            coo=sparse.load_npz(path).tocoo()
            mask=coo.row<32000
            rr,cc,w=coo.row[mask],coo.col[mask],coo.data[mask]
            inside=np.all((pos[rr]>=5)&(pos[rr]<=15),axis=1)
            d=pos[cc[inside]]-pos[rr[inside]];weights=w[inside]
            hist+=np.histogram2d(d[:,0],d[:,1],bins=[bins,bins],weights=weights)[0]
            polar+=np.histogram(np.arctan2(d[:,1],d[:,0])%(2*np.pi),bins=angles,weights=weights)[0]
            total+=weights.sum()
        rows.append((condition,audit['after']['central_targets'],hist/total,polar/total))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(10,6),layout='constrained')
    gs=fig.add_gridspec(2,4,width_ratios=[1,1,1,.055],height_ratios=[1,.85])
    titles=['Current axis','Rotated partners','Isotropic partners']
    vmax=max(row[2].max() for row in rows)
    for i,((condition,tensor,hist,polar),title) in enumerate(zip(rows,titles)):
        ax=fig.add_subplot(gs[0,i]);im=ax.imshow(hist.T,origin='lower',extent=(-3,3,-3,3),cmap='magma',
            interpolation='nearest',norm=LogNorm(vmin=1e-5,vmax=vmax))
        ax.set(xlabel='Source − target x (mm)',ylabel='Source − target y (mm)' if i==0 else '',
               title=f'{title}\ncentral axis ratio {tensor["axis_ratio"]:.2f}')
        if condition!='isotropic':
            angle=np.deg2rad(tensor['angle_deg_mod180']);v=np.array([np.cos(angle),np.sin(angle)])*2.6
            ax.plot([-v[0],v[0]],[-v[1],v[1]],color='#55eeee',lw=1.2)
        pax=fig.add_subplot(gs[1,i],projection='polar')
        mids=(angles[1:]+angles[:-1])/2
        pax.plot(np.r_[mids,mids[0]+2*np.pi],np.r_[polar,polar[0]],c='#226b8e',lw=1.5)
        pax.set_ylim(0,.043);pax.set_yticks([.01,.03]);pax.set_yticklabels(['1%','3%'],fontsize=8)
        pax.set_xticks(np.deg2rad([0,90,180,270]));pax.set_xticklabels(['0°','90°','180°','270°'])
        pax.spines['polar'].set_visible(False)
    fig.colorbar(im,cax=fig.add_subplot(gs[0,3]),label='Fraction of incoming EE weight / bin')
    fig.suptitle('Achieved connection geometry • central targets (5–15 mm)\nIncoming weights, source-region totals and exact delay bins matched',fontsize=12)
    dest=ROOT/'figures';dest.mkdir(exist_ok=True)
    fig.savefig(dest/'axis_geometry_controls.png',dpi=180)
    fig.savefig(dest/'axis_geometry_controls.pdf')
    plt.close(fig)
    np.savez_compressed(dest/'axis_geometry_inputs.npz',spatial_bins=bins,angle_bins=angles,
        histograms=np.stack([r[2] for r in rows]),angle_histograms=np.stack([r[3] for r in rows]))
    (dest/'README.md').write_text('''### axis_geometry_controls.png

展示实际图的中央目标细胞所接收的 E→E 位移权重分布；上行二维直方图，下行方向分布，均使用共同标尺。旋转对照中央主轴改变88.3°，但轴比由1.94减至1.58；各向同性对照轴比1.03。每个目标×来源区域×物理延迟层保留原权重多重集，输出度未匹配；这张图只验收结构，尚不表示传播或闭环动力学结果。

**关注点**：达到的几何改变与残余各向异性，不能把名义旋转当成完美刚体旋转；图待作者目视。
''')
    print(dest/'axis_geometry_controls.png',flush=True)


if __name__=='__main__':main()
