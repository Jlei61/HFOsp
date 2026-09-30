#!/usr/bin/env python3
"""Complete native30s initial-G comparison, preserving held-state semantics."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from campaign import ROOT,read,write,sha

OUT=ROOT/'exit_actual_G_history_probes/G_history_comparison'
result=read(OUT/'analysis.json')
assert result['status']=='COMPLETE_ACTUAL_FIELD_G_HISTORY_COMPARISON'
geo=np.load(ROOT/'native_slices/geometry.npz');centers=geo['centers_mm']
colors=['#8a63b4','#d63378','#008eb3'];labels=['All E','Core A','Core B']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(13,10),layout='constrained')
grid=fig.add_gridspec(3,4,width_ratios=[1,1,1,.035],height_ratios=[1,1,1.25])
metadata=[]
for col,row in enumerate(result['rows']):
    assert row['complete_30s'] and row['recorded_future_input_samples_bitwise']==300
    with np.load(OUT/f"{row['name']}.npz") as z:d={k:z[k] for k in z.files}
    ax=fig.add_subplot(grid[0,col]);time=d['time5_s'].reshape(-1,4).mean(1)
    rates=d['rate5_Hz'].reshape(-1,4,4).mean(1)
    for j in range(3):ax.plot(time,rates[:,j],color=colors[j],lw=1.1,label=labels[j])
    ax.set(title=rf"Initial $G_{{\rm raw}}$ = {row['initial_Graw']:.2f}",ylim=(-5,510),xlim=(0,30),xticks=[0,10,20,30],xlabel='Elapsed time (s)')
    ax.axvspan(20,30,color='.5',alpha=.08)
    if col==0:ax.set_ylabel('Rate (Hz)');ax.legend(frameon=False,fontsize=8,loc='lower left')
    ax=fig.add_subplot(grid[1,col]);ax.plot(d['time1_s'],d['Graw'],color='#b27228',lw=1.2)
    bound=95.19851312666987/(18+17.662847938268442)
    ax.axhline(bound,color='#7a8b72',ls=':',lw=1)
    ax.axvspan(20,30,color='.5',alpha=.08)
    ax.set(ylim=(-.1,12),xlim=(0,30),xticks=[0,10,20,30],xlabel='Elapsed time (s)')
    if col==0:ax.set_ylabel(r'$G_{\rm raw}$');ax.text(1,3.15,'Z-eligibility bound',fontsize=8,color='#607456')
    mask=(d['time5_s']>=20)&(d['time5_s']<30);field=d['field5_Hz'][mask].mean(0)
    drift=row['windows'][-1]['mean_counterfactual_dZ_per_s_allE_A_B_surround']
    ax=fig.add_subplot(grid[2,col]);im=ax.imshow(field.reshape(20,20),origin='lower',extent=(0,20,0,20),vmin=0,vmax=500,cmap='magma',interpolation='nearest')
    for label,xy in zip(['A','B'],centers):
        ax.add_patch(Circle(xy,1.5,fill=False,edgecolor='#00c3c5',lw=1))
        ax.text(xy[0],xy[1]+1.75,label,ha='center',color='#00c3c5',fontsize=8)
    direction='positive' if drift[0]>0 else 'negative'
    assert drift[1]<0 and drift[2]<0
    ax.set(title=f'20–30 s mean field\nZ drift: all E {direction}; cores negative',xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20])
    if col==0:ax.set_ylabel('y (mm)')
    metadata.append(dict(name=row['name'],initial_Graw=row['initial_Graw'],tail_field_Hz=field.tolist(),
        tail_mean_counterfactual_Z_drift=drift,tail_rates_Hz=row['windows'][-1]['rates_Hz_allE_A_B_surround']))
fig.colorbar(im,cax=fig.add_subplot(grid[2,3]),label='E rate (Hz)')
fig.suptitle('Initial global feedback changes spatial recruitment; both cores remain active',fontsize=13,weight='bold')
fig.text(.5,-.025,'Native network: same full initial state and future input; only initial G differs. Z/K held at means 0.21/9.\nZ drift is counterfactual; neither actual Z recovery nor autonomous termination is measured in these controls.',ha='center',fontsize=9)
for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'native_initial_G_history.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
write(OUT/'figure_metadata.json',dict(producer_sha256=sha(__file__),input_sha256=sha(OUT/'analysis.json'),
    rows=metadata,statistical_unit=result['statistical_unit'],agent_visual='PENDING',human_review='PENDING',formal_Fig5_replaced=False))
readme=ROOT/'figures/README.md';entry='### native_initial_G_history.png / .svg'
if entry not in readme.read_text():
    with readme.open('a') as f:f.write('\n'+entry+'\n原生网络从同一完整内源状态和相同未来外源输入出发，仅改变初始全局G，展示完整30秒的群体活动、G以及末20–30秒空间场。三条均没有两核退出，但初始强G改变了空间招募；其全网Z恢复方向转正，两核却仍消耗Z。Z和K固定，因此这里不计自主终止，也不把反事实Z漂移当作实际恢复；一条共享历史和输入不代表三个独立种子。\n**关注点**：全网均率和平均Z会掩盖核心未退出；同一Z/K下的空间状态取决于完整历史。\n')
print('NATIVE G HISTORY FIGURE WRITTEN',flush=True)
