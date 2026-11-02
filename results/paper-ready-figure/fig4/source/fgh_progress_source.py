#!/usr/bin/env python3
"""Read-only progress figures using frozen live scores and the original reducer.

This does not change any running campaign dependency or promote Figure 4.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO=Path(__file__).resolve().parents[2]
DATA=Path('/data/hfosp/topic4_sef_hfo')
COLORS={1:'#397DAA',2:'#8C4B20',3:'#8073AC',4:'#4F8A64'}
GROUPS=('rank_pattern','local_and_interrod_timing','participation')
DEFAULT_OUT=REPO/'results/paper-ready-figure/fig4/candidates/core_count_single_vs_dual_20260929'

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,a):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(a,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)

def original_reducer():
    folder=REPO/'scripts/topic4_core_count_optimization'
    sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location('frozen_core_count_plotting',folder/'report.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.trajectory

def load_case(label,root,sid,reduce,manifest,*,core_counts=(1,2)):
    plan=read(root/'plan.json');repeats=plan['restarts'];epochs=plan['epochs']
    is_e1146=sid=='epilepsiae_1146'
    base=root/'per_subject'/sid
    if not is_e1146:base=base/'runs'
    arrays={};records={};prefix={};completed={};unscorable={}
    for k in core_counts:
        arrays[k]=[];records[k]=[];prefix[k]=[];completed[k]=0;unscorable[k]=0
        for restart in range(repeats):
            rows=[]
            for p in sorted((base/f'core{k}'/f'restart{restart:02d}'/'scores').glob('*.json')):
                raw=p.read_bytes();r=json.loads(raw)
                assert r['phase']=='train' and r['core_count']==k and r['restart']==restart
                assert r['noise']==plan['training_noise']
                assert r['J'] is None or np.isfinite(r['J'])
                manifest[str(p)]=sha_bytes(raw);rows.append(r)
            rows.sort(key=lambda r:r['epoch'])
            assert len({r['epoch'] for r in rows})==len(rows)
            arr,n=reduce(rows,epochs);arrays[k].append(arr);records[k].append(rows);prefix[k].append(n)
            completed[k]+=len(rows);unscorable[k]+=sum(r['J'] is None for r in rows)
        arrays[k]=np.asarray(arrays[k])
    common=min(n for counts in prefix.values() for n in counts)
    comparison={};chosen={}
    for k in core_counts:
        good=[];chosen[k]=[]
        for rows in records[k]:
            candidates=[r for r in rows if r['epoch']<=common and r['J'] is not None]
            best=min(candidates,key=lambda r:(r['J'],r['epoch'])) if candidates else None
            chosen[k].append(best)
            if best is not None:good.append(best['J'])
        comparison[k]=float(np.median(good)) if len(good)==repeats else None
        if comparison[k] is not None:
            np.testing.assert_allclose(np.median(arrays[k][:,common-1,0]),comparison[k],rtol=0,atol=1e-12)
    single,dual=comparison[1],comparison[2]
    summary=dict(subject=sid,display_id=label,core_counts=list(core_counts),planned_epochs=epochs,restarts=repeats,completed=completed,
        unscorable=unscorable,prefix_by_restart=prefix,common_epoch=common,initial_conditions=plan['initial_conditions'],
        median_best_loss=comparison,two_minus_one=dual-single if single is not None and dual is not None else None,
        reduction_percent=100*(single-dual)/single if single is not None and dual is not None and single>0 else None,
        stage='initial_sampling' if common<=plan['initial_conditions'] else 'adaptive_search',
        chosen_at_common_epoch={k:[None if r is None else dict(candidate=r['candidate'],epoch=r['epoch'],J=r['J'],N=r['N']) for r in chosen[k]] for k in core_counts})
    return dict(summary=summary,arrays=arrays,records=records)

def mean_std_trajectory(block, common):
    """Mean +/- sample SD across every repeat, at jointly observed epochs."""
    if block.shape[0] < 2:
        raise ValueError('Sample standard deviation requires at least two repeats.')
    shared = np.flatnonzero(np.isfinite(block[:, :common]).all(axis=0))
    values = block[:, shared]
    mean = np.mean(values, axis=0)
    sd = np.std(values, axis=0, ddof=1)
    return shared + 1, mean, sd


def draw(ax,case,component=0,*,show_title=True,paper_style=None,show_legend=False,
         summary_style='median_iqr'):
    """Draw the frozen trajectories directly at their final panel size.

    ``paper_style`` uses the complete figure's label/tick/legend point sizes;
    no title or figure-level progress caption is needed in an assembled panel.
    ``mean_std`` draws only means and sample SD bands on the common prefix;
    the legacy default preserves existing published-figure consumers.
    """
    if summary_style not in ('median_iqr', 'mean_std'):
        raise ValueError(f'Unknown summary style: {summary_style}')
    thin_lw,thin_ms,median_lw,median_ms=(.65,2.,1.45,3.2) if paper_style else (1.,3.8,2.3,5.5)
    s=case['summary'];common=s['common_epoch'];max_epoch=max(n for counts in s['prefix_by_restart'].values() for n in counts)
    mean_sd = summary_style == 'mean_std'
    summary_lw,summary_ms=((.85,1.5) if paper_style else (1.2,2.2)) if mean_sd else (median_lw,median_ms)
    lower_bounds=[]
    if mean_sd:
        max_epoch=common
    for k in sorted(case['arrays']):
        block=case['arrays'][k][:,:,component]
        if mean_sd:
            x,mean,sd=mean_std_trajectory(block,common)
            if len(x):
                lower_bounds.append(float(np.min(mean-sd)))
                band=ax.fill_between(x,mean-sd,mean+sd,color=COLORS[k],alpha=.14,linewidth=0,zorder=1)
                band.set_gid(f'core{k}_sample_sd')
                if len(x)==1:
                    ax.vlines(x,mean-sd,mean+sd,color=COLORS[k],alpha=.35,lw=.7,zorder=2)
                line,=ax.plot(x,mean,color=COLORS[k],lw=summary_lw,marker='o',ms=summary_ms,
                             markeredgewidth=0,zorder=3)
                line.set_gid(f'core{k}_mean')
            continue
        for row in block:
            finite=np.flatnonzero(np.isfinite(row))
            ax.plot(finite+1,row[finite],color=COLORS[k],alpha=.28,lw=thin_lw,marker='o',ms=thin_ms)
        # Compare all displayed arms only where every planned restart has finished.
        shared=np.isfinite(block[:,:common]).all(0);x=np.flatnonzero(shared)+1
        if len(x):
            q=np.quantile(block[:,x-1],[.25,.5,.75],axis=0)
            if len(x)>1:ax.fill_between(x,q[0],q[2],color=COLORS[k],alpha=.13,linewidth=0)
            else:ax.errorbar(x,q[1],yerr=np.stack([q[1]-q[0],q[2]-q[1]]),color=COLORS[k],fmt='none',capsize=5,lw=2)
            ax.plot(x,q[1],color=COLORS[k],lw=median_lw,marker='o',ms=median_ms)
    if max_epoch<=1:
        ax.set_xlim(.7,1.3);ax.set_xticks([1])
    else:
        ax.set_xlim(.85,max_epoch+.3)
        ax.set_xticks(np.unique(np.linspace(1,max_epoch,min(8,max_epoch),dtype=int)))
    ymin=min(lower_bounds,default=0.) if mean_sd else min((float(np.nanmin(a[:,:,component])) for a in case['arrays'].values() if np.isfinite(a[:,:,component]).any()),default=0.)
    if ymin>=0:ax.set_ylim(bottom=0)
    ax.set_xlabel('Epochs');ax.set_ylabel('Loss' if component==0 else 'Scaled error')
    ax.set_title(s['display_id'] if show_title else '',fontweight='bold',pad=11)
    ax.tick_params(direction='out',length=4,width=.8)
    if paper_style:
        ax.xaxis.label.set_fontsize(paper_style['label'])
        ax.yaxis.label.set_fontsize(paper_style['label'])
        ax.tick_params(labelsize=paper_style['tick'],length=2.4,width=.7,pad=2)
        ax.xaxis.labelpad=ax.yaxis.labelpad=3
        ax.spines[['top','right']].set_visible(False)
    if show_legend:
        handles=[Line2D([],[],color=COLORS[k],lw=summary_lw,marker='o',ms=summary_ms,
                       markeredgewidth=0 if mean_sd else plt.rcParams['lines.markeredgewidth'],
                       label=f'{k} core'+('s' if k>1 else '')) for k in sorted(case['arrays'])]
        ax.legend(handles=handles,loc='upper right',frameon=False,
                  fontsize=paper_style['legend'] if paper_style else 11,
                  handlelength=1.3,handletextpad=.45,borderaxespad=.35,labelspacing=.3)

def render(out,cases,stamp):
    figdir=out/'figures';figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':12,'axes.titlesize':13,
        'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':11,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    handles=[Line2D([],[],color=COLORS[k],lw=2.3,marker='o',ms=5,label=f'{k} core'+('s' if k==2 else '')) for k in (1,2)]
    generated=[]
    def save(fig,stem):
        for ext in ('png','svg','pdf'):
            p=figdir/f'{stem}.{ext}';fig.savefig(p,dpi=220,facecolor='white');generated.append(str(p))
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.8,4.4));draw(ax,cases[0]);ax.legend(handles=handles,frameon=False,loc='upper right')
    s=cases[0]['summary'];fig.subplots_adjust(left=.14,right=.97,top=.87,bottom=.22)
    fig.text(.14,.07,f"Ongoing · common epoch {s['common_epoch']}/{s['planned_epochs']} · {s['restarts']} restarts",fontsize=10,color='.4')
    save(fig,'e1146_single_vs_dual_loss')
    fig,axes=plt.subplots(1,3,figsize=(11.6,4.1))
    for ax,case in zip(axes,cases[1:]):
        draw(ax,case);s=case['summary']
        ax.text(.5,-.3,f"{s['common_epoch']}/{s['planned_epochs']} epochs · {s['restarts']} restarts",transform=ax.transAxes,ha='center',fontsize=10,color='.4')
    fig.subplots_adjust(left=.075,right=.98,top=.75,bottom=.27,wspace=.36)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.99),ncol=2,frameon=False)
    save(fig,'representative_single_vs_dual_loss')
    fig,axes=plt.subplots(4,3,figsize=(11.6,12.5))
    for i,case in enumerate(cases):
        for j,name in enumerate(('Rank distribution','Order / timing','Participation')):
            draw(axes[i,j],case,j+1);axes[i,j].set_title(f"{case['summary']['display_id']} · {name}",fontsize=12)
    fig.subplots_adjust(left=.08,right=.98,bottom=.06,top=.92,wspace=.36,hspace=.62)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.99),ncol=2,frameon=False)
    save(fig,'single_vs_dual_loss_components')
    return generated

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT_OUT);args=p.parse_args();out=args.out
    out.mkdir(parents=True,exist_ok=True);manifest={};reduce=original_reducer();stamp=time.strftime('%Y-%m-%d %H:%M:%S %Z')
    cases=[load_case('E1146',DATA/'e1146_core_count_optimization_20260928','epilepsiae_1146',reduce,manifest)]
    root=DATA/'representative_core_count_20260929';selection=read(root/'selection.json')
    for row in selection['selected']:
        label=row['display_id']+' (Zhaochenxi)' if row['subject']=='yuquan_zhaochenxi' else row['display_id']
        cases.append(load_case(label,root,row['subject'],reduce,manifest))
    snapshot=dict(updated_at=stamp,scope='single versus dual core; ongoing training, not final validation',cases=[dict(summary=c['summary'],records=c['records']) for c in cases],score_sources_sha256=manifest,
        reducer=str(REPO/'scripts/topic4_core_count_optimization/report.py'),reducer_sha256=sha_bytes((REPO/'scripts/topic4_core_count_optimization/report.py').read_bytes()),
        semantics='Original best-so-far reducer; common-prefix medians/IQR across all planned restarts in both arms. Thin lines may extend to later completed epochs. No arm-specific normalization. A single epoch is drawn as points, never an invented trajectory.')
    snapdir=out/'snapshots'/time.strftime('%Y%m%d_%H%M%S');snapdir.mkdir(parents=True)
    write(snapdir/'input_snapshot.json',snapshot);write(out/'input_snapshot.json',snapshot)
    fields=['subject','display_id','common_epoch','planned_epochs','restarts','single_core_median','dual_core_median','two_minus_one','reduction_percent','stage']
    with (out/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for c in cases:
            s=c['summary'];w.writerow({k:s[k] for k in fields if k in s}|dict(single_core_median=s['median_best_loss'][1],dual_core_median=s['median_best_loss'][2]))
    files=render(out,cases,stamp)
    write(out/'plot_validation.json',dict(status='NUMERICAL_PASS_VISUAL_PENDING',source_files=len(manifest),
        common_epoch_medians_recomputed_independently=True,original_reducer_reused=True,negative_losses_retained=True,
        no_pending_epoch_interpolation=True,score_sources_unchanged=all(sha_bytes(Path(p).read_bytes())==h for p,h in manifest.items()),
        generated_files=files,human_visual_acceptance='PENDING',updated_at=stamp))
    (out/'figures/README.md').write_text('''### e1146_single_vs_dual_loss.png
E1146 单 core 与双 core 的最新训练 loss 曲线，另有同源 PDF/SVG。浅线及小点为四次随机优化各自的已完成前缀；粗线与阴影为两组全部重复共同完成的 epoch 上的中位数及四分位范围，使用共同原始 loss 尺度。
**关注点**：横轴放大至当前已评估范围，完整计划为每次96个epoch；目前属于初始采样，不代表优化完成或噪声复测结论。

### representative_single_vs_dual_loss.png
E635、E958、Y9（赵晨曦）的单核／双核训练结果，另有同源 PDF/SVG。每组两次随机优化，符号和短竖线在只有一个epoch时仍显示实际值、中位数及四分位范围，不连接或外推不存在的趋势。
**关注点**：各患者使用自身冻结loss尺度，不能直接将患者间纵轴数值当作病情或模型能力排序；当前首个配置尚不支持优化效果判断。

### single_vs_dual_loss_components.png
四位患者的rank分布、order/timing和participation三个分量，另有同源 PDF/SVG。每个epoch的三个分量来自同一个总loss最佳候选，复用原运行报告的选择算法。
**关注点**：误差分量不是分别选择最低值，某一项可随总loss下降而上升；本候选待作者目视检查。
''')
    note=[f'# 单核／双核最新进度：{stamp}','',
        '这份候选仅更新 core 数量实验的独立对照图，输入快照和同状态 PNG/PDF/SVG 已保存。当前正式 Fig4 A–I 的身份不变，图像尚待作者目视检查。','',
        '| Subject | 共同 epoch | 单核中位数 | 双核中位数 | 双核相对降低 |','|---|---:|---:|---:|---:|']
    for c in cases:
        s=c['summary'];a=s['median_best_loss'][1];b=s['median_best_loss'][2];pct=s['reduction_percent']
        note.append(f"| {s['display_id']} | {s['common_epoch']}/{s['planned_epochs']} | {a:.4f} | {b:.4f} | {pct:.1f}% |" if a is not None and b is not None and pct is not None else f"| {s['display_id']} | {s['common_epoch']}/{s['planned_epochs']} | NA | NA | NA |")
    note += ['', '百分比仅描述当前共同预算下的最佳训练loss中位数差，不是跨记录验证结果。E1146的4次重启仍处于初始24点采样，另外三人的2次重启仍处于初始16点采样；噪声复测和完整预算比较尚未完成。代表患者按既有共享轴观测选择，不按模型胜负选择。']
    (out/'scientific_note.md').write_text('\n'.join(note)+'\n')
    print(json.dumps(dict(output=str(out),snapshot=str(snapdir),cases=[c['summary'] for c in cases]),ensure_ascii=False))

if __name__=='__main__':main()
