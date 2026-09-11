#!/usr/bin/env python3
"""Read completed factorial workers; no refitting and no SNN execution."""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path(__file__).resolve().parents[1]
DEFAULT=Path('/data/hfosp/topic4_sef_hfo/state_s_native_z_pilot_20260909')
COLORS=['#2878b5','#d34f37','#47955a']
IS_FIXTURE=False
PARTIAL_LABEL=''

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvwrite(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with p.open('w') as f:
        wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(rows)
def save(fig,p):
    if IS_FIXTURE:fig.text(.5,.005,'SYNTHETIC FIXTURE — NOT A SCIENTIFIC RESULT',ha='center',color='red',fontsize=10)
    if PARTIAL_LABEL and fig._suptitle is not None:fig._suptitle.set_text(fig._suptitle.get_text()+'\n'+PARTIAL_LABEL)
    fig.savefig(p.with_suffix('.png'),dpi=170,bbox_inches='tight');fig.savefig(p.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)
def load_arrays(out,j):
    with np.load(out/'workers'/f'{j}.npz') as z:return {k:z[k] for k in z.files}
def binrate(x,dt=.1,width=5):
    bs=round(width/dt);n=len(x)//bs;return np.arange(n)*width/1000+width/2000,np.asarray(x[:n*bs]).reshape(n,bs).mean(1)

def analyze(out,partial=False,fixture=False):
    global IS_FIXTURE, PARTIAL_LABEL
    IS_FIXTURE=fixture
    d=read(out/'design.json');records=[];missing=[]
    for j in d['jobs']:
        p=out/'workers'/f"{j['id']}.json"
        if not p.exists():missing.append(j['id']);continue
        r=read(p);assert r['status']=='COMPLETE';assert sha(p.with_suffix('.npz'))==r['arrays_sha256'];records.append(r)
    if missing and not partial:raise RuntimeError('Incomplete jobs: '+str(missing))
    if not records:raise RuntimeError('No completed workers')
    PARTIAL_LABEL=f'INTERIM: {len(records)}/{len(d["jobs"])} completed; pending conditions are not negative results' if missing else ''
    figdir=out/'figures';figdir.mkdir(exist_ok=True);rows=[];events=[];lookup={};checks=[]
    for r in records:
        j=r['job'];a=load_arrays(out,j['id']);lookup[(j['Z_dynamic'],j['s'],j['seed'])]=(r,a)
        eligible=[e for e in r['events'] if e['pilot_interictal_eligible']]
        names=a['contact_names'].tolist();scl=np.array([str(n).startswith('SCL') for n in names]);counts={m:sum(e['mode']==m for e in eligible) for m in (0,1)}
        nscl={m:sum(not np.isfinite(a['centroid_ms'][e['index'],scl]).any() for e in eligible if e['mode']==m) for m in (0,1)}
        row=dict(job=j['id'],seed=j['seed'],s=j['s'],Z_dynamic=j['Z_dynamic'],detected_s=None if r['detected_ms'] is None else r['detected_ms']/1000,right_censored=r['latency_right_censored'],observed_s=r['actual_duration_ms']/1000,n_interictal=len(eligible),M0=counts[0],M1=counts[1],M0_fraction=counts[0]/sum(counts.values()) if sum(counts.values()) else None,M0_no_SCL=nscl[0],M1_no_SCL=nscl[1],Z_end=float(a['Z_final'][:len(a['positions_E'])].mean()))
        t,rate=binrate(a['rate_E'],width=10);last=rate[-100:];row['final1s_E_Hz']=float(last.mean());row['final1s_quiet_fraction']=float(np.mean(last<1));rows.append(row)
        for e in r['events']:events.append(dict(job=j['id'],seed=j['seed'],s=j['s'],Z_dynamic=j['Z_dynamic'],**e))
    csvwrite(out/'run_summary.csv',rows);csvwrite(out/'event_summary.csv',events)
    # Exogenous input stream comparison, including different stopping times.
    for seed in sorted({r['job']['seed'] for r in records}):
        selected=[r for r in records if r['job']['seed']==seed]
        prefix={r['input_audit'].get('legacy_first_6s_sha256') for r in selected if r['actual_duration_ms']>=6000}
        checks.append(dict(seed=seed,check='first6s_master_inputs_match',pass_value=len(prefix)<=1 and None not in prefix,n_runs=sum(r['actual_duration_ms']>=6000 for r in selected)))
        for uz in (False,True):
            subset=[r for r in selected if r['job']['Z_dynamic']==uz]
            checks.append(dict(seed=seed,Z_dynamic=uz,check='first1s_before_state_all_spikes_match',pass_value=len({r['first1s_spike_sha256'] for r in subset})<=1))
    if not all(c['pass_value'] for c in checks):raise RuntimeError('Paired-input or warmup check failed')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
    seeds=sorted({r['seed'] for r in rows})
    for uz,ls in ((False,'--'),(True,'-')):
        for i,seed in enumerate(seeds):
            rr=sorted([r for r in rows if r['seed']==seed and r['Z_dynamic']==uz],key=lambda r:r['s'])
            xs=[r['s'] for r in rr];ys=[r['detected_s'] if r['detected_s'] is not None else r['observed_s'] for r in rr]
            axes[0].plot(xs,ys,ls,color=COLORS[i%3],alpha=.8,label=f'{seed}: '+('native Z' if uz else 'Z=1'))
            for x,y,r in zip(xs,ys,rr):axes[0].plot(x,y,'^' if r['right_censored'] else 'o',color=COLORS[i%3],mfc='white' if r['right_censored'] else COLORS[i%3])
            axes[1].plot(xs,[r['M0_fraction'] if r['M0_fraction'] is not None else np.nan for r in rr],ls+'o',color=COLORS[i%3])
            axes[2].plot(xs,[r['n_interictal'] for r in rr],ls+'o',color=COLORS[i%3])
    for ax in axes:ax.set_xlabel('Imposed state s');ax.set_xticks([-.5,0,.5]);ax.grid(alpha=.15)
    axes[0].set(title='Sustained-high detection',ylabel='Confirmation time (s)');axes[0].legend(fontsize=7)
    axes[1].set(title='Before sustained activity',ylabel='M0 / classified eligible events',ylim=(-.05,1.05))
    axes[2].set(title='Available propagation events',ylabel='Eligible event count')
    fig.suptitle('Fixed historical graph; M off; no manual restoration\nDashed: Z=1; solid: native Z; open triangles: no detection before observation ended')
    save(fig,figdir/'factorial_outcomes')
    rep=seeds[0];fig,axs=plt.subplots(2,3,figsize=(16,7),sharex=True,sharey=True,layout='constrained')
    for row,uz in enumerate((False,True)):
        for col,sval in enumerate((-.5,0.,.5)):
            ax=axs[row,col];item=lookup.get((uz,sval,rep))
            if item is None:
                ax.set_title(('Native Z' if uz else 'Z fixed at 1')+f'; s={sval:+.1f}')
                ax.text(.5,.5,'Pending',transform=ax.transAxes,ha='center');continue
            r,a=item;sp=a['sample_spikes'];xx,yy=np.nonzero(sp);ax.scatter(xx*.0001,yy,s=.4,c=np.where(a['sample_groups'][yy]==3,'#ed8d37','#4484b7'),rasterized=True,linewidths=0)
            for y in (60,120,240):ax.axhline(y,color='.7',lw=.5)
            ax.set_title(('Native Z' if uz else 'Z fixed at 1')+f'; s={sval:+.1f}')
            if r['detected_ms'] is not None:ax.axvline(r['detected_ms']/1000,c='#ad3333',ls='--',lw=1)
            ax.axvspan(r['actual_duration_ms']/1000,20,color='.94');ax.set_xlim(0,20);ax.set_yticks([30,90,180,270],['Core A E','Core B E','Other E','I']);ax.set_xlabel('Time (s)')
    fig.suptitle(f'Actual spike raster; predetermined representative noise seed {rep}\nRed line: detection confirmation; gray: not simulated after the observation stop')
    save(fig,figdir/'representative_rasters')
    fig,axs=plt.subplots(4,3,figsize=(16,11),sharex=True,sharey='row',layout='constrained')
    for col,sval in enumerate((-.5,0.,.5)):
        item=lookup.get((True,sval,rep))
        if item is None:continue
        r,a=item;tr=a['trace'];tm=tr[:,0]/1000
        axs[0,col].plot(tm,tr[:,4],label='Core A',color='#bd3e8e');axs[0,col].plot(tm,tr[:,9],label='Core B',color='#2878b5');axs[0,col].plot(tm,tr[:,32],label='All E',color='.3');axs[0,col].set_ylim(0,1.05);axs[0,col].set_title(f'Native Z; s={sval:+.1f}')
        for key,color in (('rate_E','#2878b5'),('rate_I','#e18a31')):
            t,rates=binrate(a[key]);axs[1,col].plot(t,rates,color=color,label=key[-1])
        for j,color in ((0,'#bd3e8e'),(1,'#2878b5')):
            axs[2,col].plot(tm,tr[:,2+j*5],c=color,ls=':',label=f'{"AB"[j]} raw I')
            axs[2,col].plot(tm,tr[:,3+j*5],c=color,label=f'{"AB"[j]} effective I')
            axs[3,col].plot(tm,tr[:,5+j*5],c=color,label=f'Core {"AB"[j]}')
        for ax in axs[:,col]:
            ax.set_xlim(0,20);ax.axvspan(r['actual_duration_ms']/1000,20,color='.94');ax.grid(alpha=.13)
            if r['detected_ms'] is not None:ax.axvline(r['detected_ms']/1000,c='#ad3333',ls='--',lw=.8)
        axs[3,col].set_xlabel('Time (s)');axs[3,col].set_ylim(0,1.05)
    for i,l in enumerate(['Inhibition coefficient Z','Population rate (Hz)','Inhibitory current (model units)','Fraction above Z threshold']):axs[i,0].set_ylabel(l);axs[i,0].legend(fontsize=8,ncol=2)
    fig.suptitle(f'State, local inhibitory load, and sustained activity; noise seed {rep}\nZ and currents sampled every 10 ms; rates averaged over 5 ms; no M or refill')
    save(fig,figdir/'representative_resource_currents')
    readme='''### factorial_outcomes.png / .pdf
同一历史手放基底上，比较三种固定 s 与 Z 固定/原生演化的持续高活动进入时间和进入前模式计数。颜色对应独立噪声重演，空三角为未触发的右截尾记录。
**关注点**：统计单位是运行；合格事件数与观察时间不同，不能直接解释成事件率或患者匹配程度。

### representative_rasters.png / .pdf
预先选第一条噪声的六条件真实样本 raster，共用样本身份和时间轴。红线为持续高活动确认时刻，灰区表示预算规则结束后没有继续模拟。
**关注点**：本轮没有人工补回；观察截止不等于活动终止。

### representative_resource_currents.png / .pdf
同一代表噪声的三种 s 下，展示局部及全 E 的 Z、群体发放率、原始与有效抑制和超阈值负荷比例。原始抑制推动 Z，实际作用于 E 的抑制为 Z 乘以原始抑制。
**关注点**：s 对即时抑制和延迟资源变化可能有不同作用；M 关闭，不能据此判断适应恢复机制。
'''
    (figdir/'README.md').write_text(readme)
    lines=['# 固定状态 s × 原生 Z pilot','',f'状态：{"PARTIAL" if missing else "ROUND_COMPLETE_PENDING_SCIENTIFIC_REVIEW"}；{len(records)}/{len(d["jobs"])} 条完成。','', '原图的历史 manual_hard 阈值、图和 C 快速连接保持；M 关闭；没有临床标签驱动的重置或人工 Z 补回。s 固定，未测试 s 自主演化。','', '| Z | s | seed | 高活动确认(s) | M0/M1 | 末端 Z |','|---|---|---|---|---|---|']
    for r in rows:
        timing=f"{r['detected_s']:.2f}" if r['detected_s'] is not None else f">{r['observed_s']:.2f} (截尾)"
        lines.append(f"| {'原生' if r['Z_dynamic'] else '固定1'} | {r['s']:+.1f} | {r['seed']} | {timing} | {r['M0']}/{r['M1']} | {r['Z_end']:.3f} |")
    lines+=['','![汇总](figures/factorial_outcomes.png)','![代表 raster](figures/representative_rasters.png)','![局部资源与抑制](figures/representative_resource_currents.png)','', '进入前事件沿用冻结观察器，只保留窗口完整位于1.5秒之后且持续高活动起点之前的合格事件。M0/M1是冻结患者模板标签，不等于起始core或患者传播验收。event_summary.csv 的core rise20是窗口内2毫秒分箱率首次达到自身峰值20%的时间，不是单神经元起燃证据。','', '未触发保留为20秒右截尾；触发后最多继续2秒，持续时间因观察停止可能只有下界。三次噪声重演只能提供小规模机制诊断，不作事件级独立显著性检验。s符号与Z变化是本轮因素，不改变结构、不扩展M、不判定发作分岔。','', '配对输入和状态关闭期一致性检查见 analysis_qa.json；图仍待人工科学审阅。']
    if fixture:lines.insert(0,'SYNTHETIC FIXTURE: NOT A SCIENTIFIC RESULT.\n')
    (out/'scientific_report.md').write_text('\n'.join(lines)+'\n')
    write(out/'analysis_qa.json',dict(status='PASS',fixture=fixture,n_runs=len(records),missing=missing,checks=checks,representative_seed=rep,figure_files={p.name:sha(p) for p in figdir.iterdir() if p.suffix in ('.png','.pdf')},human_visual_review_pending=True))
    if not fixture:
        from review_topic4_state_s_native_z_pilot import review
        review(out)
    print(json.dumps({'analyzed':len(records),'out':str(out),'fixture':fixture}))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output-root',type=Path,default=DEFAULT);ap.add_argument('--allow-incomplete',action='store_true');ap.add_argument('--fixture',action='store_true');a=ap.parse_args();analyze(a.output_root,a.allow_incomplete,a.fixture)
