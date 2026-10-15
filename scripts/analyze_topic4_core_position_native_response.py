"""Continuous activity distinguishes silent core from missing contact patterns.

Uses every post-burn-in native sample; does not filter by event qualification.
"""
import sys,json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT)]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import run_topic4_core_position_response as run
run.configure()
OUT=run.OUT/'analysis';F=OUT/'continuous_native/figures'

def main():
    F.mkdir(parents=True,exist_ok=True)
    plan=run.rt.read(run.OUT/'plan.json');rows=[];traces={};timing=[]
    for c in plan['candidates']:
        for seed in plan['seeds']:
            result=run.rt.read(run.core.result_path(c['id'],seed))
            with np.load(run.core.result_path(c['id'],seed).with_suffix('.npz')) as a:
                tt=a['trace_time_ms'];dt=float(np.median(np.diff(tt)))
                mask=(tt>=plan['analysis_burnin_ms'])&(tt<plan['duration_ms'])
                duration_s=float(mask.sum()*dt/1000)
                for group in ['coreAE','coreBE','surroundE']:
                    counts=a['trace_'+group+'_spikes'];n=len(a['group_'+group])
                    rows.append(dict(candidate=c['id'],label=c['label'],seed=seed,group=group,n_neurons=n,
                        observed_seconds=duration_s,n_spikes=int(counts[mask].sum()),mean_rate_hz=float(counts[mask].sum()/n/duration_s)))
                    k=max(1,round(20/dt));length=len(counts)//k*k
                    traces[(c['id'],seed,group)]=(tt[:length].reshape(-1,k).mean(1)/1000,counts[:length].reshape(-1,k).sum(1)/n/(k*dt/1000))
                for i in a['primary_event_indices']:
                    lo,hi=result['events'][i]['window_ms']
                    if lo<plan['analysis_burnin_ms'] or hi>plan['duration_ms']:continue
                    mask=(tt>=lo)&(tt<hi);times=[];masses=[]
                    for group in ['coreAE','coreBE']:
                        counts=a['trace_'+group+'_spikes'][mask];mass=int(counts.sum());masses.append(mass)
                        times.append(float(tt[mask][np.searchsorted(np.cumsum(counts),.1*mass)]) if mass else None)
                    timing.append(dict(candidate=c['id'],seed=seed,event=int(i),mode='TA' if a['event_mode'][i]==1 else 'TB',
                        A_mass=masses[0],B_mass=masses[1],A_q10_ms=times[0],B_q10_ms=times[1],A_minus_B_q10_ms=times[0]-times[1] if None not in times else None))
    with (OUT/'continuous_native_activity.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    with (OUT/'event_core_mass_timing.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=timing[0]);writer.writeheader();writer.writerows(timing)
    summaries=[]
    for c in plan['candidates']:
        for seed in plan['seeds']:
            for mode in ['ALL','TA','TB']:
                d=np.asarray([r['A_minus_B_q10_ms'] for r in timing if r['candidate']==c['id'] and r['seed']==seed and (mode=='ALL' or r['mode']==mode) and r['A_minus_B_q10_ms'] is not None])
                summaries.append(dict(candidate=c['id'],seed=seed,mode=mode,n=len(d),median_A_minus_B_q10_ms=float(np.median(d)) if len(d) else None,
                    A_q10_earlier_fraction=float(np.mean(d<0)) if len(d) else None,q05_ms=float(np.quantile(d,.05)) if len(d) else None,q95_ms=float(np.quantile(d,.95)) if len(d) else None))
    with (OUT/'core_mass_timing_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=summaries[0]);writer.writeheader();writer.writerows(summaries)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':42})
    cases=[c for c in plan['candidates'] if not c['retain_I_state'] and c['y_shift_mm'] is not None]
    cases.sort(key=lambda c:c['y_shift_mm'])
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,group,title in zip(axes,['coreAE','coreBE','surroundE'],['左核 A','右核 B','核外 E']):
        for seed in plan['seeds']:
            values=[next(r['mean_rate_hz'] for r in rows if r['candidate']==c['id'] and r['seed']==seed and r['group']==group) for c in cases]
            ax.plot([c['y_shift_mm'] for c in cases],values,'o-',label=f'噪声 {seed}')
        ax.set(title=title,xlabel='左核上移量 (mm)',ylabel='每神经元平均放电率 (Hz)',xticks=[0,1.5,3,4.5])
    axes[0].legend(fontsize=8);fig.suptitle('完整原生活动：1.5–20秒所有发放，未按事件/TA/TB筛选\n回答核是否仍活跃；平均率相似不证明burst形态、模式或传播相同')
    for ext in ['png','pdf']:fig.savefig(F/f'continuous_native_rate_response.{ext}',dpi=160)
    plt.close(fig)
    selected=['base_off','A_up_3p0_off','A_up_4p5_off','A_near_upper_SCL_off']
    fig,axes=plt.subplots(4,2,figsize=(14,9),layout='constrained',sharex=True,sharey=True)
    for row,cid in enumerate(selected):
        case=next(c for c in plan['candidates'] if c['id']==cid)
        for col,seed in enumerate(plan['seeds']):
            ax=axes[row,col]
            for group,color,label in [('coreAE','#c55154','左核A'),('coreBE','#367fa8','右核B')]:
                t,rate=traces[(cid,seed,group)];ax.plot(t,rate,color=color,lw=.65,label=label)
            ax.axvspan(0,1.5,color='gray',alpha=.12);ax.set(title=f'{case["label"]} · 噪声 {seed}',xlim=(0,20),ylabel='发放率 (Hz)')
            if row==3:ax.set_xlabel('连续时间 (s)')
    axes[0,0].legend(fontsize=8);fig.suptitle('同网两核的完整连续发放：20 ms固定分箱，无事件筛选\n灰区为排除的启动期；两个核都活跃不等于两种患者传播都被恢复')
    for ext in ['png','pdf']:fig.savefig(F/f'continuous_core_activity.{ext}',dpi=160)
    plt.close(fig)
    with (F/'README.md').open('w') as f:
        for stem,desc in [('continuous_native_rate_response','完整1.5–20秒的左核、右核和核外E放电率对位置的响应。按实际神经元数和持续时间归一化，不使用事件窗口筛选。'),('continuous_core_activity','四个固定位置、两条噪声的完整两核发放率，20ms固定分箱。灰区为不计入均值的启动期，保留未经资格过滤的所有活动。')]:
            for ext in ['png','pdf']:f.write(f'\n\n### {stem}.{ext}\n\n{desc}\n\n**关注点**：区分核未活跃与活动未被电极/模式读到，不能用平均率替代传播形态。\n')
    (OUT/'continuous_native_activity_note.md').write_text('# 连续原生活动\n\n按每个实际分组的神经元数归一化，1.5–20秒所有原生发放除以18.5秒。统计未要求被检测或归类为事件；连续图用20ms固定分箱，无平滑重定时。此诊断仅用于区别核整体静默与电极传播缺失，不能由平均率推断两核因果驱动、burst同形或TA/TB能力。\n')
    with (OUT/'continuous_native_activity_note.md').open('a') as f:
        f.write('\n另一张event_core_mass_timing.csv仅对合格事件窗口分别计算两核达到本核窗口活动质量10%的时间，汇总见core_mass_timing_summary.csv。A−B为负表示A较早；这是质量分位时间，不是组织首次起燃时间，也不是因果先导核。它受窗口截断、背景发放及各核活动量影响；少数A先到10%但被归为TB的情况不能删去。此事件条件诊断与上面的全时段平均率分开。\n')

if __name__=='__main__':main()
