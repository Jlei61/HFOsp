#!/usr/bin/env python3
"""Add the missing upper-point history control without fabricating branches."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,shutil,time
import numpy as np
from campaign import ROOT,read,write,sha
from native_high_history_upper_point import OUT as NATIVE,NAME
from analyze_actual_G_history import load

OUT=ROOT/'native_exit_upper_history_figure_v2'
STEM='native_exit_upper_history_v2'


def main(wait):
    OUT.mkdir(exist_ok=True);assert not (OUT/'result.json').exists()
    while not (NATIVE/'comparison.json').exists():
        if (NATIVE/'status.json').exists() and read(NATIVE/'status.json')['stage']=='FAILED':
            write(OUT/'progress.json',dict(status='STOPPED_NATIVE_FAILURE'));raise RuntimeError('Nativeupperpointfailed')
        write(OUT/'progress.json',dict(status='WAITING_COMPLETE_NATIVE_AND_ANALYSIS',pid=os.getpid(),updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    assert read(NATIVE/'comparison.json')['recorded_future_inputs_exact']
    old=read(ROOT/'native_exit_branch_candidates/result.json')
    fields=np.load(ROOT/'native_exit_branch_candidates/fields.npz')['fields_Hz']
    rows=[];fieldmap={}
    for i,row in enumerate(old['rows']):
        if row['K']>9.5:continue
        rows.append(row);fieldmap[(row['K'],row['history'])]=fields[i]
    d=load(NATIVE,NAME);m=(d['time5']>=20)&(d['time5']<30);md=(d['drift_time']>20)&(d['drift_time']<=30)
    new=dict(K=9.5,history='heldK9',name=NAME,root=str(NATIVE),rate_Hz=d['rate'][m].mean(0).tolist(),
        counterfactual_dZ_per_s=d['drift'][md,:,0].mean(0).tolist(),counterfactual_dK_per_s=d['drift'][md,:,1].mean(0).tolist(),
        complete_native30s=True,tail_brief_events=d['generic']['tail_brief_events'])
    rows.append(new);fieldmap[(9.5,'heldK9')]=d['field'][m].mean(0)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(12,7.3),layout='constrained');grid=fig.add_gridspec(2,5,width_ratios=[1,1,1,1,.035])
    ax=fig.add_subplot(grid[0,:2]);az=fig.add_subplot(grid[0,2:4]);colors=['#8a63b4','#d63378','#008eb3']
    markers={'evolving':'o','recovery':'^','heldK9':'s'}
    for row in rows:
        for j,c in enumerate(colors):
            kw=dict(marker=markers[row['history']],s=44,edgecolors=c,facecolors='none' if row['history']=='recovery' else c,linewidths=1.1)
            ax.scatter(row['K'],row['rate_Hz'][j],**kw);az.scatter(row['K'],row['counterfactual_dZ_per_s'][j],**kw)
    for a in [ax,az]:a.set(xlabel=r'Held mean $K$ ($g_K/g_L$)',xlim=(8.96,9.54),xticks=[9,9.35,9.5])
    ax.set(ylabel='E rate (Hz)',ylim=(-15,510),title='A  Native conditional responses')
    az.set(ylabel=r'Counterfactual $dZ/dt$ (s$^{-1}$)',ylim=(-.055,.18),title='B  Resource drift at held states')
    az.axhline(0,color='.6',lw=.8,ls=':')
    ax.legend([Line2D([],[],color=c,marker='o',ls='none') for c in colors],['All E','Core A','Core B'],frameon=False,fontsize=9,loc='center left',bbox_to_anchor=(.12,.52))
    ax.annotate('Both high histories;\nall three regions quiet',xy=(9.5,0),xytext=(9.40,53),ha='center',fontsize=8,
        arrowprops=dict(arrowstyle='-',color='.4',lw=.7))
    az.legend([Line2D([],[],color='.25',marker=markers[h],markerfacecolor='none' if h=='recovery' else '.25',ls='none') for h in ['evolving','recovery','heldK9']],
        ['Evolving high history','Recovery history','Held K9 history'],frameon=False,fontsize=8,loc='center left')
    chosen=[(9.35,'evolving'),(9.35,'heldK9'),(9.5,'evolving'),(9.5,'heldK9')]
    geo=np.load(NATIVE/'geometry.npz')
    for i,(K,h) in enumerate(chosen):
        a=fig.add_subplot(grid[1,i]);im=a.imshow(fieldmap[(K,h)].reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='magma',vmin=0,vmax=500,interpolation='nearest')
        for label,xy in zip(['A','B'],geo['centers_mm']):
            a.add_patch(Circle(xy,float(geo['core_radius_mm']),fill=False,edgecolor='#00c3c5',lw=1))
            a.text(xy[0],xy[1]+2,label,color='#00c3c5',ha='center',fontsize=8)
        a.set(title=f'K = {K}\n'+('Held K9 history' if h=='heldK9' else 'Evolving high history'),xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20])
        if i==0:a.set_ylabel('y (mm)')
    fig.colorbar(im,cax=fig.add_subplot(grid[1,4]),label='E rate (Hz)')
    fig.suptitle(r'Actual exit-field family, held $\bar Z=0.21$; native 20–30 s means',fontsize=12)
    for ext in ['png','svg']:fig.savefig(ROOT/f'figures/{STEM}.{ext}',dpi=180)
    plt.close(fig)
    import xml.etree.ElementTree as ET
    ET.parse(ROOT/f'figures/{STEM}.svg')
    np.savez_compressed(OUT/'field_inputs.npz',K=np.array([q[0] for q in chosen]),history=np.array([q[1] for q in chosen]),fields_Hz=np.array([fieldmap[q] for q in chosen]))
    write(OUT/'result.json',dict(status='COMPLETE_NATIVE_UPPER_HISTORY_CANDIDATE_FIGURE',rows=rows,
        new_point=new,recorded_future_inputs_exact=True,SVG_XML='PASS',agent_visual_review='PENDING',human_visual_review='PENDING',
        scope='Actualfiniteconditionalobservations only, no joined stable/unstable branches or certifiedcriticalK. Z/Kclamps are interventions, notautonomous exit or observedZrecovery.',producer_sha256=sha(__file__)))
    shutil.copy2(__file__,OUT/'producer.py')
    title=f'### {STEM}.png / {STEM}.svg';readme=ROOT/'figures/README.md'
    if title not in readme.read_text():
        with readme.open('a') as f:f.write('\n\n'+title+'\n在既有原生条件测点中加入缺失的K9.5/已维持K9高历史对照，显示同一实际退出Z/K空间场族及配对未来输入下的历史差别；所有点来自完整30秒观测的最后10秒。上排为真实均率与钳制时反事实资源漂移，下排对照K9.35和9.5的两种高活动历史，均未连接为稳定或不稳定支。\n**关注点**：检验K9.5安静是否仅由原演化初态造成；有限窗响应不是认证分岔，反事实正漂移也不是实际资源恢复，人工待审。\n')
    write(OUT/'progress.json',dict(status='COMPLETE_FIGURE_PENDING_VISUAL_REVIEW',updated_epoch=time.time()))
    print(new,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args();main(a.wait)
