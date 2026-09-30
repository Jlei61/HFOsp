#!/usr/bin/env python3
"""Review the measured K transition and its spatial contraction.

This post hoc display adds transition-focused snapshots without changing the
registered-window figure, classifications or any simulation.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from campaign import ROOT,read,write,sha
from continue_target_high_state import OUT

p=argparse.ArgumentParser();p.add_argument('--include-holds',action='store_true');p.add_argument('--include-lower-holds',action='store_true');args=p.parse_args()
if args.include_lower_holds:args.include_holds=True
basename='carried_K_transition_bracket' if args.include_lower_holds else 'carried_K_ramps_and_holds' if args.include_holds else 'carried_high_state_K_transition_zoom'
dest=OUT/('transition_bracket' if args.include_lower_holds else 'transition_zoom_with_holds' if args.include_holds else 'transition_zoom');dest.mkdir(exist_ok=True)
assert read(OUT/'analysis/result.json')['status']=='COMPLETE'
data={name:dict(np.load(OUT/'analysis'/f'{name}.npz')) for name in ['ramp2s_K10p5','ramp10s_K10p5']}
colors=['#8a63b4','#d63378','#008eb3'];labels=['All E','Core A','Core B']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(14,7.2),layout='constrained')
outer=fig.add_gridspec(2,1,height_ratios=[1.35,1])
top=outer[0].subgridspec(1,3);axes=[fig.add_subplot(top[0,j]) for j in range(3)]
styles={'ramp2s_K10p5':(0,(4,2)),'ramp10s_K10p5':'-'}
for name,d in data.items():
    K=d['K'].reshape(-1,20).mean(1);rates=d['rate_Hz'].reshape(-1,20,4).mean(1)
    drift=d['drift_per_s'].reshape(-1,20,4).mean(1);G=d['Graw'].reshape(-1,20).mean(1)
    for j in range(3):
        axes[0].plot(K,rates[:,j],color=colors[j],ls=styles[name],lw=1.5)
        axes[2].plot(K,drift[:,j],color=colors[j],ls=styles[name],lw=1.3)
    axes[1].plot(K,G,color='#b27228',ls=styles[name],lw=1.5)
axes[0].set(title='A  Population activity',ylabel='Rate (Hz)',ylim=(-8,510))
axes[1].set(title='B  Global feedback',ylabel=r'$G_{\rm raw}$',ylim=(-.08,4.9))
axes[2].set(title='C  Z recovery direction',ylabel=r'Counterfactual $dZ/dt$ (s$^{-1}$)',ylim=(-.055,.19))
axes[1].axhline(95.19851312666987/(18+17.662847938268442),ls=':',color='#7a8b72',lw=.9)
axes[1].text(9.04,2.78,'Native Z-eligibility bound',fontsize=8,color='#607456')
axes[2].axhline(0,color='.5',lw=.8,ls=':')
for ax in axes:ax.set(xlabel=r'Mean imposed $K$ ($g_K/g_L$)',xlim=(8.98,10.52),xticks=[9,9.5,10,10.5])
legend1=[Line2D([0],[0],color=c,lw=1.6,label=l) for c,l in zip(colors,labels)]
axes[0].legend(handles=legend1,frameon=False,fontsize=8,loc='upper right')
axes[1].legend(handles=[Line2D([0],[0],color='.2',ls=styles[n],label=l) for n,l in
    [('ramp10s_K10p5','10 s ramp'),('ramp2s_K10p5','2 s ramp')]],frameon=False,fontsize=9,loc='upper right')
# Native finite held-state means are separate markers, never connected as roots.
native=[]
for root,k,h in [('exit_return_probes',9,'high'),('exit_return_probes',9,'recovery'),('exit_midpoint_probes',10.5,'high')]:
    name=f'exit_z0.21_k{k:g}_fields16p7_{h}';r=read(ROOT/root/'extended_analysis'/f'{name}.json')
    q=r['tail_mean_Hz'];native.append(dict(root=root,name=name,K=k,history=h,tail20to30s_rates=q))
    if h=='high' and k==9:
        for j in range(3):axes[0].scatter(k,q[j],marker='s',s=25,facecolors='white',edgecolors=colors[j],zorder=5)
    else:axes[0].scatter(k,q[0],marker='s',s=25,facecolors='white',edgecolors='.25',zorder=5)
axes[0].text(9.04,75,'Squares: native held-state tails\n(20–30 s; distinct initial histories)',fontsize=8,color='.3')
held=[]
if args.include_holds:
    hold_sources=[('carried_exit_fixed_holds',n) for n in ['held_K9p5','held_K9p65']]
    if args.include_lower_holds:
        hold_sources=[('carried_exit_lower_holds',n) for n in ['held_K9p2','held_K9p35']]+hold_sources
    for folder,name in hold_sources:
        q=read(ROOT/folder/'analysis'/f'{name}.json')
        held.append(q)
        if args.include_lower_holds:
            for j in range(3):
                axes[0].scatter(q['K'],q['final3s_rates_allE_A_B_surround_Hz'][j],marker='^',s=43,color=colors[j],edgecolors='white',linewidths=.45,zorder=8)
                axes[2].scatter(q['K'],q['final3s_dZ_per_s'][j],marker='^',s=43,color=colors[j],edgecolors='white',linewidths=.45,zorder=8)
            axes[1].scatter(q['K'],q['final3s_Graw'],marker='^',s=43,color='#b27228',edgecolors='white',linewidths=.45,zorder=8)
        else:axes[0].scatter(q['K'],q['final3s_rates_allE_A_B_surround_Hz'][0],marker='^',s=45,color='black',zorder=8)
    axes[0].text(9.24,24,'Triangles: final 3 s of 8 s holds' if args.include_lower_holds else 'Triangles: quiet after 8 s holds',fontsize=8,color='.1')
bottom=outer[1].subgridspec(1,6,width_ratios=[1,1,1,1,1,.045])
windows=[(0,.2),(1.5,1.7),(3.5,3.7),(4.35,4.45),(4.7,4.9)]
slow=data['ramp10s_K10p5'];centers=np.load(ROOT/'native_slices/geometry.npz')['centers_mm'];snapshots=[]
for col,(lo,hi) in enumerate(windows):
    ax=fig.add_subplot(bottom[0,col]);m=(slow['time_s']>lo)&(slow['time_s']<=hi)
    field=slow['field_Hz'][m].mean(0);K=float(slow['K'][m].mean())
    im=ax.imshow(field.reshape(20,20),origin='lower',extent=(0,20,0,20),vmin=0,vmax=500,cmap='magma',interpolation='nearest')
    for label,xy in zip(['A','B'],centers):
        ax.add_patch(Circle(xy,1.5,fill=False,edgecolor='#00c3c5',lw=.9))
        ax.text(xy[0],xy[1]+1.75,label,ha='center',fontsize=8,color='#00c3c5')
    ax.set(title=f'$K$ ≈ {K:.2f}\n{lo:g}–{hi:g} s',xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20])
    if col==0:ax.set_ylabel('y (mm)')
    snapshots.append(dict(window_s=[lo,hi],meanK=K,field_Hz=field.tolist()))
fig.colorbar(im,cax=fig.add_subplot(bottom[0,5]),label='E rate (Hz)')
fig.suptitle('Dynamic transition and fixed-K checks' if args.include_holds else 'Measured conditional transition: finite speed shifts the exit location',fontsize=14,weight='bold')
fig.text(.5,-.02,'Z field held at mean 0.21. Prescribed K, not autonomous recovery or certified equilibrium branches. Lower row: transition-focused snapshots of the slow ramp.',ha='center',fontsize=9)
for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'{basename}.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
write(dest/'metadata.json',dict(agent_visual='PENDING',human_review='PENDING',producer_sha256=sha(__file__),
    native_markers=native,snapshots=snapshots,fixed_K_checks=held,
    display_selection='Posthocwindowsselectedaftercompletepairedramps to show recruitment contraction before the core collapse. Original registered-window figure retained; no reclassification or data exclusion.',
    native_limit='Native held-state markers use original timevarying exogenous drive and original initial histories; not matched dynamic-ramp trials. Their means do not certify branches.',
    formal_bifurcation=False,formal_Fig5_replaced=False))
readme=ROOT/'figures/README.md'
if f'### {basename}.png' not in readme.read_text():
    with readme.open('a') as f:
        f.write(f'\n### {basename}.png / .svg\n同一完整高态出发，比较两种K增加速度下的群体活动、全局G以及固定Z处的恢复方向，并放大慢变化期间核外招募收缩到两核退出的空间过程。下排窗口在完整结果后选择，仅为解释转换顺序；预定窗口的原图保留在carried_high_state_K_ramps。方块是此前原生固定参数轨迹的20–30秒均率，具有不同初态和外源路径，不能当成与变化曲线同一协议的平衡点；虚线只表示较快的人为K变化，**不是不稳定支**。'+('三角为四条保持8秒后的末3秒均值：K9.2/9.35两核仍活跃，而9.5/9.65静默；有限持久性不等于稳定平衡。' if args.include_lower_holds else '黑三角表示两条K9.5/9.65保持8秒的密度状态末3秒均率均为0，因此变化过程中看到的高率不能当成稳定支。' if held else '')+'候选尚待人工目视检查。\n**关注点**：较慢变化在更低K处退出；全网Z恢复方向转正早于两核；空间招募收缩是否与核心失去持续活动相连。\n')
print('TRANSITION ZOOM WRITTEN',flush=True)
