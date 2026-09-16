#!/usr/bin/env python3
"""Actual source-design rate field: no readout interpolation or invented carrier."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from run_topic4_liou_original_reference import OUT

folder=OUT/'reference_runs/exp1_2d';a=np.load(folder/'traces.npz')
field=np.load(folder/'field_Hz.npy',mmap_mode='r');state=np.load(folder/'state.npy',mmap_mode='r')
mask=a['mask'];time=a['field_time_ms']/1000
plt.rcParams.update({'font.size':13,'axes.labelsize':15,'axes.titlesize':15})
fig=plt.figure(figsize=(12,5.5));gs=fig.add_gridspec(2,3,height_ratios=[3,1],hspace=.4,left=.06,right=.96,top=.86,bottom=.11,wspace=.4)
ims=[]
for col,(title,vmin,vmax,cmap) in enumerate([('Rate (Hz)',0,150,'magma'),('Chloride (mM)',6,22,'viridis'),('Slow K (nS)',0,7,'plasma')]):
    ax=fig.add_subplot(gs[0,col]);im=ax.imshow(np.ma.masked_array(np.zeros((100,100)),~mask).T,origin='lower',extent=[0,1,0,1],interpolation='nearest',cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set(xlabel='x',ylabel='y',title=title,xticks=[0,.5,1],yticks=[0,.5,1]);fig.colorbar(im,ax=ax,fraction=.045,pad=.03);ims.append(im)
ax=fig.add_subplot(gs[1,:]);ax.plot(a['time_ms']/1000,a['trace'][:,0],c='.25',lw=.6);ax.axvspan(2,5,color='#b87b44',alpha=.2)
ax.set(xlabel='Time (s)',ylabel='Mean rate (Hz)',xlim=(0,100));line=ax.axvline(0,c='#c73552',lw=1.3)
caption=fig.text(.5,.975,'',ha='center',va='top',fontsize=16)
def update(idx):
    vals=[field[idx],state[idx,2].reshape(100,100),state[idx,3].reshape(100,100)/.2]
    for im,val in zip(ims,vals):im.set_data(np.ma.masked_array(val,~mask).T)
    line.set_xdata([time[idx],time[idx]]);caption.set_text(f'Original Exp1 rate model · {time[idx]:.2f} s');return ims+[line,caption]
for name,indices,fps in [('original_2d_overview',np.arange(0,len(time),50),12),('original_2d_wave_detail',np.flatnonzero((time>=44)&(time<=46))[::2],20)]:
    anim=FuncAnimation(fig,update,frames=indices,blit=False)
    anim.save(OUT/'figures'/f'{name}.gif',writer=PillowWriter(fps=fps),dpi=85)
plt.close(fig)
(OUT/'animation_metadata.json').write_text(json.dumps({'source':'Exp1 100x100 native rate field','spatial_interpolation':'none','overview_sample_s':.5,'detail_window_s':[44,46],'detail_sample_s':.02,'not_a_SNN_raster':True,'human_visual_acceptance':'PENDING'},indent=2)+'\n')
if '### original_2d_overview.gif' not in (OUT/'figures/README.md').read_text():
  with (OUT/'figures/README.md').open('a') as h:
    h.write('\n### original_2d_overview.gif\n原文二维rate模型的全过程，0.5秒采样，同步显示原生率、氯离子和慢钾电导；没有电极平滑或人工振荡载波。**关注点**：局部触发、空间扩张和最终停止是否与静态图一致。\n\n### original_2d_wave_detail.gif\n同一模型44–46秒的20毫秒采样动画，坐标和颜色范围保持固定。**关注点**：实际二维波动和慢变量场的时间尺度差异；这不是spiking神经元raster。\n')
