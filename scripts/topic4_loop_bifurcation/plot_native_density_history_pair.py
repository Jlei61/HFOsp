#!/usr/bin/env python3
"""Four complete paired-history states; no attractor or bifurcation labels."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, time, shutil
import numpy as np
from campaign import ROOT, read, write, sha
from analyze_actual_G_history import load as native_load
from analyze_high_state_continuation import load as density_load
from coupled_density_exit import ADAPTED

OUT = ROOT / 'native_density_history_pair'


def main(wait,redraw=False):
    OUT.mkdir(exist_ok=True); assert redraw or not (OUT / 'result.json').exists()
    files = [ROOT/'native_K9p35_held_history/history_comparison/result.json',
             ROOT/'density_K9p35_held_history/comparison/result.json']
    while not all(p.exists() for p in files):
        write(OUT/'progress.json',dict(status='WAITING_COMPLETE_COMPARISONS',pid=os.getpid(),updated_epoch=time.time()))
        if not wait: return
        time.sleep(15)
    for p in files: assert read(p)['status']=='COMPLETE'
    geo = dict(np.load(ADAPTED/'geometry.npz'))
    geometry = dict(np.load(ROOT/'native_K9p35_held_history/geometry.npz'))
    names = ['exit_z0.21_k9.35_fields16p7_high','exit_z0.21_k9.35_fields16p7_held_K9_history']
    nroots = [ROOT/'native_exit_K_bracket', ROOT/'native_K9p35_held_history']
    droots = [ROOT/'density_exit_bracket_protocol', ROOT/'density_K9p35_held_history']
    data=[]; rows=[]
    for history,name,nroot,droot in zip(['Original evolving history','Held K9 history'],names,nroots,droots):
        n=native_load(nroot,name); d=density_load(droot/name,geo)
        for kind,s in [('Native',n),('Density',d)]:
            if kind=='Native':
                t,r,tg,G,R,fields=s['time5'],s['rate'],s['time1'],s['G'],s['R'],s['field']
            else:
                t,r,tg,G,R,fields=s['time_s'],s['rate_Hz'],s['time_s'],s['Graw'],s['causal_R_Hz'],s['field_Hz']
            m=(t>=5)&(t<10); mg=(tg>=5)&(tg<10)
            data.append(dict(kind=kind,history=history,t=t,r=r,tg=tg,G=G,field=fields[m].mean(0)))
            rows.append(dict(kind=kind,history=history,window_s=[5,10],rate_Hz=r[m].mean(0).tolist(),
                mean_Graw=float(G[mg].mean()),mean_causal_R_Hz=float(R[mg].mean()),
                min_causal_R_Hz=float(R[mg].min()),max_causal_R_Hz=float(R[mg].max()),
                sampled_R_above200_fraction=float((R[mg]>200).mean())))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(15,8.7),layout='constrained')
    grid=fig.add_gridspec(3,5,width_ratios=[1,1,1,1,.035],height_ratios=[1,.65,1.15])
    colors=['#8a63b4','#d63378','#008eb3'];labels=['All E','Core A','Core B']
    for col,s in enumerate(data):
        ax=fig.add_subplot(grid[0,col]); step=4 if s['kind']=='Native' else 20
        num=len(s['t'])//step*step
        t=s['t'][:num].reshape(-1,step).mean(1); rate=s['r'][:num].reshape(-1,step,4).mean(1)
        for j in range(3):ax.plot(t,rate[:,j],color=colors[j],lw=1,label=labels[j])
        ax.set(xlim=(0,10),ylim=(-5,510),xticks=[0,5,10],xlabel='Elapsed time (s)')
        ax.set_title(chr(65+col)+'  '+s['kind']+'\n'+s['history'],loc='left',fontsize=10)
        ax.axvspan(5,10,color='.4',alpha=.06)
        if col==0:ax.set_ylabel('Rate (Hz)');ax.legend(frameon=False,fontsize=8,loc='center right')
        ax=fig.add_subplot(grid[1,col]);ax.plot(s['tg'],np.maximum(s['G'],1e-10),color='#b27228',lw=1)
        ax.set(xlim=(0,10),ylim=(1e-8,10),yscale='log',xticks=[0,5,10],yticks=[1e-8,1e-4,1],xlabel='Elapsed time (s)')
        if col==0:ax.set_ylabel(r'$G_{\rm raw}$')
        ax.text(.98,.94,r'$R$ = '+f"{rows[col]['mean_causal_R_Hz']:.1f} Hz",ha='right',va='top',transform=ax.transAxes,fontsize=9)
        ax=fig.add_subplot(grid[2,col]);im=ax.imshow(s['field'].reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0,vmax=500,interpolation='nearest')
        for label,xy in zip(['A','B'],geometry['centers_mm']):
            ax.add_patch(Circle(xy,float(geometry['core_radius_mm']),fill=False,edgecolor='#00c3c5',lw=1))
            ax.text(xy[0],xy[1]+2,label,color='#00c3c5',ha='center',fontsize=9)
        ax.set(xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20])
        if col==0:ax.set_ylabel('y (mm)')
    fig.colorbar(im,cax=fig.add_subplot(grid[2,4]),label='E rate (Hz)')
    fig.suptitle(r'Held $\bar Z=0.21$, $\bar K=9.35$: same future input, two complete initial histories',fontsize=12)
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    for ext in ['png','svg']:fig.savefig(out/f'native_density_history_pair.{ext}',dpi=190)
    plt.close(fig)
    import xml.etree.ElementTree as ET
    ET.parse(out/'native_density_history_pair.svg')
    result=dict(status='COMPLETE',rows=rows,source_comparisons=[str(p) for p in files],
        scope='Finite conditional history selection, identical Z/K and future input; native30s anddensity10s complete. Figure uses common5-10sfield/summary. Distinct finite states are not certified attractors, bistability, an autonomous loop, or a bifurcation.',
        field_geometry='20x20 display;1.5mm substrate circles, original1.75mmnearestcore rate grouping unchanged.',
        native_correspondence_certified=False,formal_bifurcation_allowed=False,
        agent_visual_review='PENDING',human_visual_review='PENDING',SVG_XML='PASS',producer_sha256=sha(__file__))
    write(OUT/'result.json',result);shutil.copy2(__file__,OUT/'figure_producer.py')
    readme=out/'README.md'
    if '### native_density_history_pair.png' not in readme.read_text():
        with readme.open('a') as f:
            f.write('\n### native_density_history_pair.png / .svg\n四列比较相同Z/K与未来外源下，两种完整原生初始历史的原生和密度结果；率、G及5–10秒空间场分别展示条件状态及反馈分段。原生30秒、密度10秒均已完整结束；圆圈是1.5毫米底物标志，率分组仍用原1.75毫米近核口径。\n**关注点**：完整历史可选择不同有限时窗状态，但高态的原生与密度可能位于G激活门槛两侧；不能据此认证稳定支或分岔，人工待审。\n')
    write(OUT/'progress.json',dict(status='COMPLETE',updated_epoch=time.time()))
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');p.add_argument('--redraw',action='store_true');a=p.parse_args();main(a.wait,a.redraw)
