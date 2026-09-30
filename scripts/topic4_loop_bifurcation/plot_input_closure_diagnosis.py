#!/usr/bin/env python3
"""Completed input and local-response diagnostics, separate from formal Fig5."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time,shutil
import numpy as np
from campaign import ROOT,read,write,sha
from observe_source_time_structure import OUT as TEMPORAL
from observe_source_aggregation import OUT as SOURCE

OUT=SOURCE/'diagnostic_figure'


def main(wait,exact_thresholds=False):
    global OUT
    if exact_thresholds:OUT=SOURCE/'diagnostic_figure_exact_thresholds'
    OUT.mkdir(exist_ok=True);assert not (OUT/'result.json').exists()
    spectrum=TEMPORAL/'spectral_analysis/result.json'
    while not spectrum.exists():
        write(OUT/'progress.json',dict(status='WAITING_COMPLETE_SOURCE_SPECTRA',pid=os.getpid(),updated_epoch=time.time()))
        if not wait:return
        time.sleep(15)
    s=read(spectrum);assert s['status']=='COMPLETE_SOURCE_AUTOCORRELATION_DIAGNOSTIC'
    local=SOURCE/('local_response_factorial_exact_thresholds' if exact_thresholds else 'local_response_factorial')/'result.json'
    d=read(local)
    a=np.array([q['rates_Hz'] for q in d['effects']]);r=np.array([q['native_rate_Hz'] for q in d['effects']])
    edge=np.array([q['selected_by'][0].startswith('edge') for q in d['effects']])
    errors=a-r[:,None]
    row=next(x for x in s['rows'] if x['region']=='selected10edge_cells_E')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(10.3,4.5),layout='constrained')
    values=np.array([row['measured_IE_II_variance'],row['white_IE_II_variance'],row['source_auto_IE_II_variance']])
    colors=['#444444','#967bb6','#dc973f'];labels=['Native current','Independent white sources','Observed source autocorrelation']
    ax=axs[0]
    for j in range(3):
        bars=ax.bar(np.arange(2)+(j-1)*.23,values[j],width=.21,color=colors[j],label=labels[j])
        for b,v in zip(bars,values[j]):ax.text(b.get_x()+b.get_width()/2,v+4,f'{v:.1f}',ha='center',va='bottom',fontsize=8)
    ax.set(xticks=[0,1],xticklabels=[r'$I_E$',r'$I_I$'],ylabel=r'Current variance (mV$^2$)',ylim=(0,values.max()*1.32),title='Input fluctuations at the recruitment edge')
    ax.legend(frameon=False,fontsize=8,loc='upper right');ax.text(-.11,1.04,'A',transform=ax.transAxes,weight='bold')
    ax=axs[1];jitter=np.linspace(-.1,.1,edge.sum())
    for j in range(4):
        ax.scatter(j+jitter,errors[edge,j],s=18,color='#398ca5',alpha=.7,label='30 edge targets' if j==0 else None)
        ax.scatter(j+np.linspace(-.1,.1,(~edge).sum()),errors[~edge,j],s=28,marker='x',color='#a7613b',label='9 core / I controls' if j==0 else None)
    ax.axhline(0,color='.5',ls=':',lw=.8)
    ax.set(xticks=range(4),xticklabels=['Group mean\nWhite variance','Native mean\nWhite variance','Group mean\nMeasured variance','Native mean\nMeasured variance'],ylabel='Local rate − native rate (Hz)',title='Two input approximations tested separately')
    ax.tick_params(axis='x',labelsize=8);ax.legend(frameon=False,fontsize=8,loc='lower left');ax.text(-.11,1.04,'B',transform=ax.transAxes,weight='bold')
    figures=ROOT/'figures'
    basename='input_closure_diagnosis_exact_thresholds' if exact_thresholds else 'input_closure_diagnosis'
    for ext in ['png','svg']:fig.savefig(figures/f'{basename}.{ext}',dpi=190)
    plt.close(fig)
    import xml.etree.ElementTree as ET
    ET.parse(figures/f'{basename}.svg')
    result=dict(status='COMPLETE',source_spectra=str(spectrum),local_response=str(local),exact_native_thresholds=exact_thresholds,
        edge_response_RMS_errors_Hz=np.sqrt((errors[edge]**2).mean(0)).tolist(),
        core_I_response_RMS_errors_Hz=np.sqrt((errors[~edge]**2).mean(0)).tolist(),
        plotted_input_variances=values.tolist(),
        scope='PanelA mean variance over785Etargets in10error-selected displaycells, one2snativetrajectory. PanelB30selectededgetargets plus9controls; fixedM andprescribed moments with1024numericalreplicaseach. These are developmentdiagnostics, not independentnative seeds or avalidatedautonomousclosure.',
        formal_bifurcation_allowed=False,agent_visual_review='PENDING',human_visual_review='PENDING',SVG_XML='PASS',producer_sha256=sha(__file__))
    write(OUT/'result.json',result);shutil.copy2(__file__,OUT/'figure_producer.py')
    with (figures/'README.md').open('a') as f:
        note='本版全部目标恢复实际原生个体阈值，替代前版核心群平均阈值的比较。' if exact_thresholds else '核心仍使用密度群平均阈值，后续应看exact_thresholds版本。'
        f.write(f'\n### {basename}.png / .svg\n左图比较误差边缘785个E目标的实测电流方差、独立白源预测及保留实测源自相关后的预测，后者仍不包含不同源之间的互谱。右图以选定30个边缘目标和9个核心/I对照分别检验平均输入与方差近似对固定M局部放电的影响。{note}\n**关注点**：这些输入来自原生数据，是误差定位而非自主闭合修复；目标是诊断单位，不是独立种子，正式Fig5未替换，人工待审。\n')
    write(OUT/'progress.json',dict(status='COMPLETE',updated_epoch=time.time()));print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');p.add_argument('--exact-thresholds',action='store_true');a=p.parse_args();main(a.wait,a.exact_thresholds)
