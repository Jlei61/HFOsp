#!/usr/bin/env python3
"""Audit unequal follow-up and compare supported1000s and3000s boundaries."""
import csv
import copy
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterSciNotation
from scipy.interpolate import PchipInterpolator
import run_topic4_fig5_long_boundary as run
from analyze_topic4_fig5_entry_progress import audit_counts


def collect():
    p=run.prepare()
    assert run.base.sha(run.OUT/'previous_1000s_snapshot.json')==p['previous_snapshot_sha256']
    grid=run.base.read(run.OUT/'previous_1000s_snapshot.json')['grid']
    original=grid['base_grid']['records']+grid['refinement_records']
    records={}
    for r in original:
        pair=(r['job']['tau_M_s'],r['job']['eta_m'])
        records[pair]=dict(tau_M_s=pair[0],eta_M=pair[1],seed=run.SEED,
            event_observed=r['event_observed'],confirmation_s=r['first_entry']['confirmation_s'] if r['event_observed'] else None,
            followup_s=r['followup_s'],source=r['source'],long_job=False,complete=True)
    jobs=[]
    for job in p['jobs']:
        folder=run.OUT/'runs'/job['name']
        chunks=[q for q in (folder/'chunks').glob('*.npz') if '.tmp.' not in q.name]
        a=audit_counts(folder) if chunks else dict(first_entry=None,followup_s=0.)
        done=(folder/'result.json').exists()
        if done:
            result=run.base.read(folder/'result.json')
            assert result['job']==job and result['identity']==p['identity']
            assert np.isclose(a['followup_s'],result['elapsed_s'])
            assert (a['first_entry'] is not None)==result['event_observed']
            if result['event_observed']:
                for key in ('onset_s','confirmation_s'):assert np.isclose(a['first_entry'][key],result['first_entry'][key])
            else:assert a['followup_s']==3000
        entered=a['first_entry'] is not None
        row=dict(name=job['name'],tau_M_s=job['tau_M_s'],eta_M=job['eta_m'],seed=run.SEED,
            event_observed=entered,confirmation_s=a['first_entry']['confirmation_s'] if entered else None,
            followup_s=a['followup_s'],source=str(folder),long_job=True,complete=done,
            kind=next(d['kind'] for d in p['design'] if d['name']==job['name']))
        jobs.append(row)
        pair=(job['tau_M_s'],job['eta_m'])
        if entered or a['followup_s']>0:
            if pair in records:assert a['followup_s']>=records[pair]['followup_s']
            records[pair]=row
    return p,grid,sorted(records.values(),key=lambda r:(r['tau_M_s'],r['eta_M'])),jobs


def at_horizon(records,horizon):
    selected=[]
    for record in records:
        entered=record['event_observed'] and record['confirmation_s']<=horizon
        if not entered and record['followup_s']<horizon:continue
        selected.append(dict(record,entered_by_horizon=entered,
                             plot_time_s=record['confirmation_s'] if entered else horizon))
    return selected


def bracket_curve(records,horizon,all_taus):
    qualified=at_horizon(records,horizon)
    brackets=[];issues=[]
    for tau in all_taus:
        column=[r for r in qualified if r['tau_M_s']==tau]
        entered=[r['eta_M'] for r in column if r['entered_by_horizon']]
        censored=[r['eta_M'] for r in column if not r['entered_by_horizon']]
        if not entered or not censored:continue
        lo,hi=max(entered),min(censored)
        if lo>=hi:
            issues.append(dict(tau_M_s=tau,reason='Nonmonotone measured entry labels; no boundary imposed.'))
            continue
        brackets.append(dict(tau_M_s=tau,eta_enter=lo,eta_censored=hi,eta_mid=float(np.sqrt(lo*hi))))
    # Only connect adjacent qualified columns; unsupported columns break the line.
    lookup={b['tau_M_s']:b for b in brackets}
    segments=[];group=[]
    for tau in all_taus:
        if tau in lookup:group.append(lookup[tau])
        else:
            if len(group)>=2:segments.append(group)
            group=[]
    if len(group)>=2:segments.append(group)
    curves=[]
    for group in segments:
        x=np.log10([b['tau_M_s'] for b in group]);y=np.log10([b['eta_mid'] for b in group])
        f=PchipInterpolator(x,y,extrapolate=False)
        gx=np.linspace(x.min(),x.max(),501)
        curves.append(dict(tau=(10**gx).tolist(),eta=(10**f(gx)).tolist()))
    return dict(horizon_s=horizon,qualified_points=len(qualified),brackets=brackets,curves=curves,issues=issues)


def report():
    p,grid,records,jobs=collect()
    # Only the original eight columns have an entered/censored pair by design.
    # Gap points add measured information to the color surface; they do not
    # create a two-sided bracket unless both outcomes actually become available.
    all_taus=sorted({b['tau_M_s'] for b in grid['continuous_surface']['boundary_brackets']})
    curves={str(h):bracket_curve(records,h,all_taus) for h in (1000,3000)}
    common=at_horizon(records,1000)
    xy=np.log10([[r['tau_M_s'],r['eta_M']] for r in common])
    values=np.array([r['plot_time_s'] for r in common])
    tri=mtri.Triangulation(*xy.T)
    gx=np.linspace(0,4,501);gy=np.linspace(-4,1,501);xx,yy=np.meshgrid(gx,gy)
    colors=10**mtri.LinearTriInterpolator(tri,np.log10(values))(xx,yy)
    # Reject curve pieces inconsistent with any qualified intermediate point.
    for horizon,collection in curves.items():
        for curve in collection['curves']:
            tx=np.log10(curve['tau']);by=np.log10(curve['eta'])
            for r in at_horizon(records,float(horizon)):
                x=np.log10(r['tau_M_s'])
                if x<tx[0] or x>tx[-1]:continue
                y=np.interp(x,tx,by)
                if (np.log10(r['eta_M'])<y)!=r['entered_by_horizon']:
                    left=max(t for t in all_taus if t<=r['tau_M_s'])
                    right=min(t for t in all_taus if t>=r['tau_M_s'])
                    by[(tx>=np.log10(left))&(tx<=np.log10(right))]=np.nan
                    collection['issues'].append(dict(tau_M_s=r['tau_M_s'],reason='Intermediate measured point contradicts interpolation; segment omitted.'))
            curve['eta']=[float(v) if np.isfinite(v) else None for v in 10**by]
    plt.rcParams.update({'font.size':14,'axes.labelsize':17,'xtick.labelsize':14,'ytick.labelsize':14,'svg.fonttype':'none','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(9,8));ax.set_box_aspect(1)
    im=ax.pcolormesh(10**gx,10**gy,colors,cmap='viridis',norm=LogNorm(1,1000),shading='nearest',rasterized=True)
    for horizon,color,style in [('1000','black','-'),('3000','#d02d9c','--')]:
        for curve in curves[horizon]['curves']:
            ax.plot(curve['tau'],np.array([np.nan if v is None else v for v in curve['eta']]),color=color,ls=style,lw=2.1,zorder=4)
    for r in common:
        ax.scatter([r['tau_M_s']],[r['eta_M']],s=18,marker='o' if r['entered_by_horizon'] else '^',
                   facecolors=im.to_rgba(r['plot_time_s']) if r['entered_by_horizon'] else 'none',
                   edgecolors='white' if r['entered_by_horizon'] else '#555555',linewidths=.65,zorder=5,clip_on=False)
    late=[r for r in jobs if r['event_observed'] and r['confirmation_s']>1000]
    for r in late:
        ax.scatter([r['tau_M_s']],[r['eta_M']],marker='D',s=48,facecolors='none',edgecolors='#d02d9c',lw=1.3,zorder=8)
        ax.annotate(f'{r["confirmation_s"]:.0f} s',(r['tau_M_s'],r['eta_M']),xytext=(4,6),textcoords='offset points',fontsize=9,color='#9c1676')
    waiting=[r for r in jobs if not r['complete'] and r['kind']!='continue_nearest_censored']
    for r in waiting:
        if r['followup_s']<1000 and not r['event_observed']:
            ax.scatter([r['tau_M_s']],[r['eta_M']],marker='+',s=38,color='#eeeeee',linewidth=1.1,zorder=7)
    ax.set(xscale='log',yscale='log',xlim=(1,1e4),ylim=(1e-4,10),xlabel=r'$\tau_M$ (s)',ylabel=r'$\eta_M$')
    ax.minorticks_off()
    handles=[Line2D([],[],color='black',lw=2,label='1000 s boundary'),
             Line2D([],[],color='#d02d9c',lw=2,ls='--',label='3000 s boundary (supported only)')]
    if late:handles.append(Line2D([],[],marker='D',mfc='none',mec='#d02d9c',ls='',label='Entry after 1000 s'))
    ax.legend(handles=handles,loc='upper right',fontsize=10,frameon=True,facecolor='white',edgecolor='none')
    cb=fig.colorbar(im,ax=ax,pad=.035,fraction=.047)
    cb.set_label('Entry time / lower bound (s)');cb.set_ticks([1,10,100,1000])
    cb.formatter=LogFormatterSciNotation();cb.update_ticks();cb.ax.minorticks_off()
    fig.tight_layout();fig.canvas.draw()
    assert np.isclose(ax.bbox.width,ax.bbox.height)
    figures=run.OUT/'figures';figures.mkdir(exist_ok=True)
    for ext in ('png','pdf','svg'):fig.savefig(figures/f'long_boundary.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    summary=dict(completed=sum(r['complete'] for r in jobs),total=19,
        continuations_completed=sum(r['complete'] and r['kind']=='continue_nearest_censored' for r in jobs),
        fresh_completed=sum(r['complete'] and r['kind']!='continue_nearest_censored' for r in jobs),
        entered=sum(r['event_observed'] for r in jobs),late_entries=len(late),
        censored3000=sum(r['complete'] and not r['event_observed'] for r in jobs),
        eligible1000_points=len(common),eligible3000_points=len(at_horizon(records,3000)),
        human_review='PENDING')
    run.base.write(run.OUT/'analysis.json',dict(summary=summary,records=records,jobs=jobs,boundaries=curves,updated_at=time.time()))
    with (run.OUT/'first_entry_points.csv').open('w') as file:
        w=csv.DictWriter(file,fieldnames=list(jobs[0]));w.writeheader();w.writerows(jobs)
    (figures/'README.md').write_text(
        '### long_boundary.png / .pdf / .svg\n'
        '底色与黑线为共同1000秒观察证据的连续进入图；仅把已进入或完整观察到1000秒的新点加入色面。紫色虚线只在进入侧与完整3000秒未进入侧均有实测支持的相邻截面间连接；紫色菱形和数值标记1000秒后进入，色条保持原log1–1000秒。白色加号为尚未达到共同1000秒终点的新采样位置。\n'
        '**关注点**：未延长的旧删失点不能作为3000秒未进入，缺乏夹区或与实测点矛盾处不连接；本候选不自动覆盖paper-ready Fig5。\n')
    (run.OUT/'README.md').write_text(
        '# Fig5边界附近的3000秒单种子实验\n\n'
        f'固定seed9108401，共19条（8条从1000秒续跑，11个新参数点）。已完成{summary["completed"]}/19，'
        f'进入{summary["entered"]}条，其中1000秒后进入{summary["late_entries"]}条，3000秒未进入{summary["censored3000"]}条。\n\n'
        '[实时状态](status.json) · [参数清单](parameter_points.csv) · [执行方案](execution_plan.md) · '
        '[连续边界候选图](figures/long_boundary.png) · [逐点结果](first_entry_points.csv) · [检查点核对](continuation_qa.json)\n\n'
        '原59点1000秒paper-ready快照保留；本目录每10分钟或新终点刷新候选。实线表示1000秒边界，紫色虚线仅连接具有双侧合格支持的3000秒边界；colorbar保持原1–1000秒，晚进入另行标数值。'
        '19条完成后停止，不增加种子、参数或观察窗。候选待人工检查。\n')


if __name__=='__main__':
    report()
