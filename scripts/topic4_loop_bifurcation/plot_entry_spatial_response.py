#!/usr/bin/env python3
"""Actual-stage Z pattern versus common-template entry responses."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from campaign import ROOT,NATIVE,read,write,sha


def main():
    source=ROOT/'entry_spatial_probes/extended_analysis_summary.json'
    data=read(source);assert data['status']=='COMPLETE' and data['completed']==4
    common=[r for r in read(NATIVE/'extended_analysis_summary.json')['rows'] if r['job']['local_cut']=='entry']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,2,figsize=(11.5,7),layout='constrained',sharex='col',sharey='row')
    colors=['#74398f','#d34e99','#249ac1'];rows=[]
    for col,history in enumerate(['high','interictal']):
        ax[0,col].set_title(history.capitalize()+' initial history')
        for family,items,style,marker in [('Common t20 pattern',common,'-','o'),('Entry t10 pattern',data['rows'],'--','s')]:
            selected=sorted([r for r in items if r['job']['source_history']==history],key=lambda r:r['job']['target_Z'])
            x=[r['job']['target_Z'] for r in selected]
            for j,color in enumerate(colors):ax[0,col].plot(x,[r['tail_mean_Hz'][j] for r in selected],color=color,ls=style,marker=marker,ms=5,lw=1.4)
            ax[1,col].plot(x,[r['tail_brief_events']/10 for r in selected],color='#237d65',ls=style,marker=marker,ms=5,lw=1.4)
            for r in selected:rows.append(dict(name=r['name'],family=family,history=history,mean_Z=r['job']['target_Z'],tail_mean_Hz=r['tail_mean_Hz'],brief_events=r['tail_brief_events'],joint_quiet_fraction=r['tail_joint_quiet_fraction'],censoring=r['censoring']))
        ax[0,col].set_ylim(-8,510);ax[1,col].set_ylim(-.15,5.5);ax[1,col].set_xlabel('Held all-E mean Z')
        for a in ax[:,col]:a.grid(axis='y',alpha=.15);a.set_xlim(.64,.86)
    ax[0,0].set_ylabel('Native E rate (Hz)');ax[1,0].set_ylabel('Complete brief events / s')
    handles=[Line2D([],[],color=c,label=l) for c,l in zip(colors,['All E','Core A','Core B'])]
    handles += [Line2D([],[],color='#444',ls=s,marker=m,label=l) for l,s,m in [('Common t20 Z pattern','-','o'),('Actual-entry t10 Z pattern','--','s')]]
    fig.legend(handles=handles,ncol=3,loc='outside upper center',frameon=False)
    fig.supxlabel('Same mean Z and K = 0.0002; only the held Z spatial pattern changes within each history.\nFinal 10 s of paired 30-s native branches; guide lines are not equilibrium or periodic branches.',fontsize=9)
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    for suffix in ['png','svg']:fig.savefig(out/f'entry_spatial_response.{suffix}',dpi=180)
    plt.close(fig)
    write(out/'entry_spatial_response_metadata.json',dict(source=str(source),rows=rows,producer_sha256=sha(__file__),
        human_review='PENDING',agent_visual_review='PENDING',formal_bifurcation=False,
        interpretation='Same mean-Z coordinates need a specified spatial field family. t10 is60ms after the operational entry marker, not an exact bifurcation state.'))
    path=out/'README.md';text=path.read_text() if path.exists() else ''
    marker='### entry_spatial_response.png / entry_spatial_response.svg\n'
    section=marker+'同一未来噪声、同一平均Z/K及同一内源初值下，只改变被固定的Z空间分布。左右分别为高态史和间期史，颜色沿用全E/核A/核B，线型区分共同t20模板与实际进入附近t10场；下排保留全部完整短事件频率。每点30秒、显示末10秒，是条件响应而非认证的分岔分支。\n**关注点**：更深的核内疲劳能否在相同全局平均Z下消除分离短事件；两个初值是否支持同一空间效应。\n'
    if marker in text:
        before,after=text.split(marker,1);tail=after.find('\n### ');text=before+section+(after[tail:] if tail>=0 else '')
    else:text=text.rstrip()+'\n\n'+section
    path.write_text(text);print('ENTRY_SPATIAL_RESPONSE_GENERATED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args()
    while a.wait:
        source=ROOT/'entry_spatial_probes/extended_analysis_summary.json'
        if source.exists() and read(source)['status']=='COMPLETE':break
        time.sleep(30)
    main()
