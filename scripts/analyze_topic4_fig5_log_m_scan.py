#!/usr/bin/env python3
"""Build a log-coordinate first-passage map from completed observed endpoints."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogFormatterSciNotation
import run_topic4_fig5_log_m_scan as scan
from plot_topic4_m_parameter_modes import safe
OUT=scan.OUT

def collect():
    p=scan.prepare();records=[]
    for ref in p['references']:
        assert scan.base.sha(Path(ref['source'])/'result.json')==ref['result_sha256']
        records.append(dict(job=ref['job'],event_observed=True,first_entry=ref['first_entry'],source=ref['source'],reused=True))
    pending=[]
    for job in p['jobs']:
        path=OUT/'runs'/job['name']/'result.json'
        if not path.exists():pending.append(job['name']);continue
        r=scan.base.read(path);assert r['job']==job and r['identity']==p['identity']
        assert r['event_observed'] or r['elapsed_s']>=p['horizon_s']
        records.append(dict(job=job,event_observed=r['event_observed'],first_entry=r['first_entry'],source=str(path.parent),reused=False))
    shape=(len(scan.ETAS),len(scan.TAUS));count=np.zeros(shape,int);entered=np.zeros(shape,int);values=np.full(shape,np.nan)
    for i,eta in enumerate(scan.ETAS):
        for j,tau in enumerate(scan.TAUS):
            cell=[r for r in records if r['job']['eta_m']==eta and r['job']['tau_M_s']==tau]
            count[i,j]=len(cell);entered[i,j]=sum(r['event_observed'] for r in cell)
            if len(cell)==2:values[i,j]=np.mean([r['first_entry']['confirmation_s'] if r['event_observed'] else p['horizon_s'] for r in cell])
    return dict(eta_M=scan.ETAS,tau_M_s=scan.TAUS,mean=values,count=count,entered=entered,horizon_s=p['horizon_s'],
        completed_new=len(records)-len(p['references']),total_new=p['new_jobs'],reused=len(p['references']),pending=pending,
        records=records,scale='Logarithmic eta and tau axes; log color for restricted mean confirmation time.',
        endpoint=p['endpoint'],interpretation=p['interpretation'],all_complete=not pending)

def edges(centers):
    x=np.log10(centers);mid=(x[:-1]+x[1:])/2
    return 10**np.r_[x[0]-(x[1]-x[0])/2,mid,x[-1]+(x[-1]-x[-2])/2]

def draw(fig,spec,g,letter='E',working_point=(1.,.0005)):
    ax=fig.add_subplot(spec);x=edges(g['tau_M_s']);y=edges(g['eta_M'])
    # Fix physical width/height, while retaining the two logarithmic data axes.
    ax.set_box_aspect(1)
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#ededed')
    im=ax.pcolormesh(x,y,np.ma.masked_invalid(g['mean']),cmap=cmap,norm=LogNorm(1,g['horizon_s']),edgecolors='#ffffff99',lw=.6)
    ax.set(xscale='log',yscale='log',xticks=g['tau_M_s'],yticks=g['eta_M'],xlabel=r'$\tau_M$ (s)',ylabel=r'$\eta_M$')
    ax.set_xticklabels([f'{v:g}' for v in g['tau_M_s']]);ax.set_yticklabels([f'{v:g}' for v in g['eta_M']]);ax.minorticks_off()
    for i,eta in enumerate(g['eta_M']):
        for j,tau in enumerate(g['tau_M_s']):
            n=int(g['count'][i,j]);success=int(g['entered'][i,j])
            if n<2:text='…' if n==0 else '1/2 done';color='#777'
            else:
                val=g['mean'][i,j];text=(f'≥{g["horizon_s"]:g}' if success==0 else f'{val:.1f}')+f'\n{success}/2';color='white' if val<80 else '#111'
                if success<2:ax.add_patch(Rectangle((x[j],y[i]),x[j+1]-x[j],y[i+1]-y[i],fc='none',ec='#55555588',lw=0,hatch='///'))
            # Labels belong at cell centers; irregular log spacing otherwise
            # crowds the eta=0.0005 and 0.001 annotations in the square panel.
            ax.text(np.sqrt(x[j]*x[j+1]),np.sqrt(y[i]*y[i+1]),text,
                ha='center',va='center',fontsize=12,color=color)
    # Mark the exact measured working cell, not an interpolated value.
    tau,eta=working_point
    if eta in g['eta_M'] and tau in g['tau_M_s']:
        i=g['eta_M'].index(eta);j=g['tau_M_s'].index(tau)
        ax.add_patch(Rectangle((x[j],y[i]),x[j+1]-x[j],y[i+1]-y[i],fc='none',ec='black',lw=2))
    slot=spec.get_position(fig)
    fig.text(slot.x0,slot.y1+.012,letter,weight='bold',fontsize=24,ha='left')
    # An inset colorbar follows the square axes instead of shrinking its width.
    cb=fig.colorbar(im,cax=ax.inset_axes([1.06,0,.055,1]))
    cb.ax.set_yscale('log')
    ticks=[10.**k for k in range(int(np.floor(np.log10(g['horizon_s'])))+1)]
    if not np.isclose(ticks[-1],g['horizon_s']):ticks.append(g['horizon_s'])
    cb.set_label('Restricted mean entry time (s)')
    cb.set_ticks(ticks);cb.formatter=LogFormatterSciNotation(base=10,labelOnlyBase=False,
        minor_thresholds=(np.inf,np.inf))
    cb.update_ticks();cb.ax.minorticks_off()
    return ax

def main():
    g=collect();scan.base.write(OUT/'measured_grid.json',safe(g))
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    fig=plt.figure(figsize=(11,9));draw(fig,fig.add_gridspec(1,1)[0],g)
    for ax in list(fig.axes)+[child for parent in fig.axes for child in parent.child_axes]:ax.tick_params(labelsize=16);ax.xaxis.label.set_fontsize(20);ax.yaxis.label.set_fontsize(20)
    for ext in ['png','pdf']:fig.savefig(dest/f'log_m_first_entry.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    (dest/'README.md').write_text('### log_m_first_entry.png / .pdf\n正方形绘图区与等高色条；同一手放双核与Z动力学下，τM从1到1000秒、ηM从0.0001到1的对数参数网格；每格两个噪声种子。颜色为300秒观察窗内首次高态确认时间的限制均值，完成格显示进入数/2；灰格仍在计算，斜线表示有右删失。\n**关注点**：τM=1000秒在300秒内未充分恢复，不能将未进入解释成永久稳定；此图是有限时间首次进入图，不是严格分岔图。\n')
    # Rendering reads only existing left-side data; incremental E updates are autonomous.
    script=scan.ROOT/'scripts/plot_topic4_fig5_clean_panels.py'
    if script.exists():
        import subprocess,sys
        r=subprocess.run([sys.executable,str(script),'--eta','.0005'],cwd=scan.ROOT)
        if r.returncode:raise RuntimeError('Automatic complete-figure rendering failed')
    # Audit observed endpoints directly from the stored10ms counts, including censoring.
    checks=[]
    for record in g['records']:
        if record['reused']:continue
        rows=[];last=0
        for path in sorted((Path(record['source'])/'chunks').glob('*.npz')):
            if '.tmp.' in path.name:continue
            with np.load(path) as a:
                assert int(a['start_step'])==last;last=int(a['end_step'])
                assert np.array_equal(a['spikes_10ms'][:,0],a['regions_10ms'][:,:3].sum(1))
                rows.append(a['spikes_10ms'])
        pop=np.concatenate(rows);rate=pop[:,0]/320
        bits=np.diff(np.r_[False,rate>=200,False].astype(int))
        intervals=[(lo,hi) for lo,hi in zip(np.flatnonzero(bits==1),np.flatnonzero(bits==-1)) if hi-lo>=20]
        if record['event_observed']:
            assert intervals
            assert np.isclose(intervals[0][0]*.01,record['first_entry']['onset_s'])
            assert np.isclose((intervals[0][0]+20)*.01,record['first_entry']['confirmation_s'])
        else:
            assert not intervals and last*.0001>=g['horizon_s']
        checks.append(dict(name=record['job']['name'],status='PASS',observed=record['event_observed'],
            full_count_coverage=True,late_mean_E_Hz=float(rate[-min(100,len(rate)):].mean()),
            late_quiet_fraction=float((rate[-min(100,len(rate)):]<5).mean())))
    scan.base.write(OUT/'endpoint_audit.json',dict(status='PASS_AVAILABLE_COMPLETED_RUNS',all_complete=g['all_complete'],
        checked_new=len(checks),checks=checks,reused_exact_endpoints=g['reused'],human_review='PENDING'))
    lines=['# 对数M动力学首次进入扫描','',f'当前完成{g["completed_new"]}/{g["total_new"]}条新仿真，复用{g["reused"]}条已观测到的精确首次终点。',
        '', 'τM=1、10、100、1000秒；ηM=0.0001、0.0005、0.001、0.01、0.1、1。每格两个噪声种子，固定原手放双核、原生OU、Z动力学；无人工reset。',
        '', '共同观察窗300秒；首次全E在10ms分箱下≥200Hz持续200ms，保留起点与确认时间，颜色使用确认时间的限制均值。完成格显示进入数/2；未完成格灰色省略号，有删失格加斜线，不能把灰格当成未进入。',
        '', 'τM远大于首次转变时间时，M近似累积放电；稳态有效反馈取决于ηM×τM。这个参数图只测有限时间首次进入，不证明严格分岔或永久稳定，尤其τM=1000秒的未进入条件尚未完整弛豫。',
        '', '[当前E图](figures/log_m_first_entry.png) · [完整新版Fig5](../fig5_preentry_event_audit_20260914/clean_panels_v2/README.md) · [逐终点核对](endpoint_audit.json)',
        '', '每批结果完成后自动核对终点、更新参数图并重绘完整Fig5；整轮完成后停在人工审阅点，不再自动扩展参数。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(g['completed_new'],g['total_new'],g['all_complete'],flush=True)
if __name__=='__main__':main()
