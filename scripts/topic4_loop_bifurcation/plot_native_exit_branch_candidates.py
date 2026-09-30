#!/usr/bin/env python3
"""Measured native conditional branch candidates, without stability labels."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time,shutil
import numpy as np
from campaign import ROOT,read,write,sha
from analyze_actual_G_history import load

OUT=ROOT/'native_exit_branch_candidates'


def main(redraw=False):
    OUT.mkdir(exist_ok=True);assert redraw or not (OUT/'result.json').exists()
    specs=[]
    for K in [9,9.35,9.5,10.5,12]:
        root=ROOT/('native_exit_K_bracket' if K in [9.35,9.5] else 'exit_midpoint_probes' if K==10.5 else 'exit_return_probes')
        specs.append((root,f'exit_z0.21_k{K}_fields16p7_high','evolving',K))
        if K in [9,10.5,12]:specs.append((root,f'exit_z0.21_k{K}_fields16p7_recovery','recovery',K))
    specs.append((ROOT/'native_K9p35_held_history','exit_z0.21_k9.35_fields16p7_held_K9_history','heldK9',9.35))
    data=[];rows=[];Z0=None;shape0=None;drive=None
    for root,name,history,K in specs:
        d=load(root,name);job=d['job']
        with np.load(job['held_fields_file']) as z:Z=z['Z'].copy();ks=z['K'].copy()/K
        if Z0 is None:Z0=Z;shape0=ks;drive=d['inputs']
        assert np.array_equal(Z,Z0) and np.allclose(ks,shape0,rtol=1e-13,atol=1e-13)
        assert np.array_equal(d['inputs'],drive)
        m=(d['time5']>=20)&(d['time5']<30);md=(d['drift_time']>20)&(d['drift_time']<=30)
        r=d['rate'][m].mean(0);drift=d['drift'][md].mean(0)
        field=d['field'][m].mean(0)
        rows.append(dict(K=K,history=history,root=str(root),name=name,rate_Hz=r.tolist(),
            counterfactual_dZ_per_s=drift[:,0].tolist(),counterfactual_dK_per_s=drift[:,1].tolist(),
            complete_native30s=True,tail_brief_events=d['generic']['tail_brief_events']))
        data.append(dict(K=K,history=history,field=field))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(12,7.1),layout='constrained')
    grid=fig.add_gridspec(2,5,width_ratios=[1,1,1,1,.035],height_ratios=[1,1])
    a=fig.add_subplot(grid[0,:2]);b=fig.add_subplot(grid[0,2:4])
    colors=['#8a63b4','#d63378','#008eb3'];labels=['All E','Core A','Core B'];markers={'evolving':'o','recovery':'^','heldK9':'s'}
    for q in rows:
        for j,color in enumerate(colors):
            kw=dict(marker=markers[q['history']],s=42,edgecolors=color,facecolors='none' if q['history']=='recovery' else color,linewidths=1.1,zorder=3)
            a.scatter(q['K'],q['rate_Hz'][j],**kw);b.scatter(q['K'],q['counterfactual_dZ_per_s'][j],**kw)
    for ax in [a,b]:ax.set(xlabel=r'Held mean $K$ ($g_K/g_L$)',xlim=(8.85,12.15),xticks=[9,9.5,10.5,12])
    a.set(ylabel='E rate (Hz)',ylim=(-15,510));b.set(ylabel=r'Counterfactual $dZ/dt$ (s$^{-1}$)',ylim=(-.055,.18))
    a.set_title('A  Native conditional states',loc='left');b.set_title('B  Native resource drift at held states',loc='left')
    b.axhline(0,color='.6',lw=.8,ls=':')
    a.legend([Line2D([],[],color=c,marker='o',ls='none') for c in colors],labels,frameon=False,loc='upper right',fontsize=9)
    legend=[Line2D([],[],color='.25',marker=markers[h],markerfacecolor='none' if h=='recovery' else '.25',ls='none') for h in ['evolving','recovery','heldK9']]
    b.legend(legend,['Evolving high history','Recovery history','Held K9 history'],frameon=False,loc='lower right',fontsize=8)
    chosen=[(9,'evolving'),(9.35,'evolving'),(9.35,'heldK9'),(9.5,'evolving')]
    geo=np.load(ROOT/'native_K9p35_held_history/geometry.npz')
    for col,(K,h) in enumerate(chosen):
        q=next(x for x in data if x['K']==K and x['history']==h)
        ax=fig.add_subplot(grid[1,col]);im=ax.imshow(q['field'].reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0,vmax=500,interpolation='nearest')
        for label,xy in zip(['A','B'],geo['centers_mm']):
            ax.add_patch(Circle(xy,float(geo['core_radius_mm']),fill=False,edgecolor='#00c3c5',lw=1))
            ax.text(xy[0],xy[1]+2,label,color='#00c3c5',ha='center',fontsize=8)
        ax.set(title=f"K = {K}\n"+('Held K9 history' if h=='heldK9' else 'Evolving high history'),xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20])
        if col==0:ax.set_ylabel('y (mm)')
    fig.colorbar(im,cax=fig.add_subplot(grid[1,4]),label='E rate (Hz)')
    fig.suptitle(r'Actual exit-field family, held $\bar Z=0.21$; native 20–30 s means',fontsize=12)
    figures=ROOT/'figures'
    for ext in ['png','svg']:fig.savefig(figures/f'native_exit_branch_candidates.{ext}',dpi=190)
    plt.close(fig)
    import xml.etree.ElementTree as ET
    ET.parse(figures/'native_exit_branch_candidates.svg')
    np.savez_compressed(OUT/'fields.npz',fields_Hz=np.array([x['field'] for x in data]))
    result=dict(status='COMPLETE',rows=rows,QA=dict(all_Z_fields_bitwise=True,K_spatial_shape_common=True,all_future_external_records_bitwise=True),
        scope='Nine complete native conditional interventions on one shared seed/futureinput. Measured finite-horizon response branches, not certified equilibria, stable/unstable branches or formal criticalpoints. Only discrete observedpoints; no interpolated disappearance boundary. Conditional Z/K clamps do not count as autonomous exits/recoveries.',
        drift='Averaged counterfactual drift evaluated at held states; panelB is not an actual release experiment.',
        field_geometry='1.5mm substrate circles; original1.75mmnearestcore statistical masks.',
        formal_bifurcation_allowed=False,agent_visual_review='PENDING',human_visual_review='PENDING',SVG_XML='PASS',producer_sha256=sha(__file__))
    write(OUT/'result.json',result);shutil.copy2(__file__,OUT/'figure_producer.py')
    if '### native_exit_branch_candidates.png' not in (figures/'README.md').read_text():
        with (figures/'README.md').open('a') as f:
            f.write('\n### native_exit_branch_candidates.png / .svg\n使用相同实际退出Z/K场族、相同未来外源的九条完整原生30秒条件轨迹，显示20–30秒放电率及反事实Z漂移；下排给出K9、K9.35两种历史、K9.5的空间场。只画实际测点，不连接未经验证的稳定/不稳定分支；不同历史是同一种子下的干预，不是独立种子。\n**关注点**：同K9.35可保留不同空间活动，全E资源漂移转正时核心仍可能消耗；尚无正式分岔认证，钳制结果不算自主退出或恢复，人工待审。\n')
    print('COMPLETE nine native conditional state points',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--redraw',action='store_true');main(p.parse_args().redraw)
