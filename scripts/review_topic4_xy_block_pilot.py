"""Read-only scientific review: fixed-geometry contrasts and SCL order residuals.

TA/TB and SCL sequence classifications are diagnostics, never optimization input.
All four SCL contacts must participate for the sequence summary; both conditional
and all-event denominators are retained. Centroid order is not onset or causality.
"""
from pathlib import Path
import csv,json,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import run_topic4_xy_block_pilot as run
from scripts import analyze_topic4_three_observable_bo as source
rt=run.rt


def pattern(t,names):
    ix=[names.index(n) for n in ['SCL9','SCL8','SCL7','SCL6']]
    full=np.isfinite(t[:,ix]).all(1); z=t[full][:,ix]; d=np.diff(z,axis=1); n=len(z)
    out=dict(N_events=len(t),N_SCL4=n,all_four_fraction=float(full.mean()) if len(t) else None,
        median_adjacent_lags_ms=np.median(d,axis=0).tolist() if n else None,
        median_SCL_span_ms=float(np.median(np.ptp(z,axis=1))) if n else None)
    for eps in [0,2,5]:
        a=int(np.all(d>eps,axis=1).sum()); b=int(np.all(d < -eps,axis=1).sum())
        out[f'separation_gt_{eps}_ms']=dict(SCL9_to6=a,SCL6_to9=b,other=n-a-b,
            reverse_conditional=b/n if n else None,reverse_all_events=b/len(t) if len(t) else None)
    return out


def unit_patterns(score,names):
    src=Path(score['source']); r,t,cn=run.dense.load_times(src);assert names==cn
    with np.load(src.with_suffix('.npz')) as z:
        ids=[int(i) for i in z['primary_event_indices'] if r['events'][i]['window_ms'][0]>=1500 and r['events'][i]['window_ms'][1]<=r['actual_duration_ms']]
        labels=z['event_mode'][ids]
    return {mode:pattern(t if k is None else t[labels==k],names) for mode,k in [('ALL',None),('TA',1),('TB',0)]}


def main():
    out=run.OUT; dest=out/'analysis/propagation_review';dest.mkdir(exist_ok=True)
    plan=rt.read(out/'plan.json');scores={f.stem:rt.read(f) for f in (out/'scores').glob('*.json')}
    ev,names,_=source.patient()
    pat={mode:pattern(ev.fit if k is None else ev.fit[ev.fit_labels==k],names) for mode,k in [('ALL',None),('TA',1),('TB',0)]}
    cache=dest/'SCL_sequence_patterns.json'
    data=rt.read(cache) if cache.exists() else dict(patient=pat,current={},historical={})
    for cid,sc in scores.items():
        if cid not in data['current']:data['current'][cid]=unit_patterns(sc,names)
    oldplan=rt.read(run.SOURCE/'plan.json'); olddiag=rt.read(run.SOURCE/'analysis/diagnostic_observables.json')
    history=[]
    for axis in [4,5]:
        line=oldplan['response_lines'][axis]
        for value,cid in zip(line['values'],line['ids']):
            sc=rt.read(run.SOURCE/'scores'/f'{cid}.json')
            if cid not in data['historical']:data['historical'][cid]=unit_patterns(sc,names)
            for mode in ['ALL','TA','TB']:
                q=data['historical'][cid][mode]; obs=olddiag['runs'][cid][mode];p=olddiag['patient'][mode]
                errors=[abs(v['order_probability']-p['pairs'][k]['order_probability']) for k,v in obs['pairs'].items()
                    if v['shaft']=='SCL' and v['order_probability'] is not None and p['pairs'][k]['order_probability'] is not None]
                history.append(dict(axis='EE_out' if axis==4 else 'EE_angle',value=value,candidate=cid,mode=mode,J=sc['J'],N=q['N_events'],
                    N_SCL4=q['N_SCL4'],SCL_order_error=float(np.mean(errors)),rod_lag_ms=obs['rod_lag_ms']['median'],
                    reverse_fraction=q['separation_gt_0_ms']['reverse_conditional'],
                    reverse_separated_2ms=q['separation_gt_2_ms']['reverse_conditional'],
                    SCL_span_median_ms=q['median_SCL_span_ms']))
    rt.write(cache,data)
    with (dest/'historical_EE_angle_response.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    rows=list(csv.DictReader((out/'analysis/observables.csv').open()));lookup={(r['candidate'],r['mode']):r for r in rows}
    # Use the actual grid IDs: some centreline conditions reuse historical IDs.
    contrasts=[]
    for grid in plan['grids']:
        for axis in [0,1]:
            for other in [1,2,3]:
                a=grid['ids'][other][1] if axis==0 else grid['ids'][1][other]
                b=grid['ids'][other][3] if axis==0 else grid['ids'][3][other]
                if (a,'ALL') not in lookup or (b,'ALL') not in lookup:continue
                for mode in ['ALL','TA','TB']:
                    aa,bb=lookup[a,mode],lookup[b,mode]
                    contrasts.append(dict(core=grid['core'],axis='x' if axis==0 else 'y',other_offset_mm=[-.25,0,.25][other-1],
                        low=a,high=b,displacement_mm=.5,mode=mode,
                        **{f'delta_{k}':float(bb[k])-float(aa[k]) for k in ['J','rank_error','within_rod_order_error','SCL_order_error','rod_lag_ms','participation_error']}))
    rt.write(dest/'matched_position_contrasts.json',contrasts)
    best=min([s for s in scores.values() if s['J'] is not None],key=lambda s:s['J']);ref=scores[run.REF]
    summary=dict(complete=len(scores),best=best['candidate'],J_best=best['J'],J_reference=ref['J'],
        relative_J_change=best['J']/ref['J']-1,
        training_component_delta={g:best['groups'][g]['scaled']-ref['groups'][g]['scaled'] for g in ref['groups']},
        best_observables=[lookup[best['candidate'],mode] for mode in ['ALL','TA','TB']],
        reference_observables=[lookup[run.REF,mode] for mode in ['ALL','TA','TB']],
        best_sequence=data['current'][best['candidate']],patient_sequence=pat,
        interpretation='single fixed topology and noise; incomplete grid if fewer than 49; no sequence metric enters optimization',
        historical_background='EE/angle curves hold right y=4.7155 mm; current XY reference right y=5.4655 mm. Do not pool as one response surface.',
        time=time.time())
    rt.write(dest/'summary.json',summary)
    current=summary['best_sequence']['TB'];patient=pat['TB'];rev=current['separation_gt_0_ms'];pref=patient['separation_gt_0_ms']
    obs=lookup[best['candidate'],'TB'];baseobs=lookup[run.REF,'TB']
    differences=[r for r in contrasts if r['core']==1 and r['axis']=='y' and r['mode']=='TB']
    grid_ids={cid for grid in plan['grids'] for row in grid['ids'] for cid in row}
    completed_grid=sum(cid in scores for cid in grid_ids)
    note=f'''# 二维位置实验：阶段科学审阅

本次统计覆盖 {completed_grid}/49 个网格条件及 {len(scores)-completed_grid} 个双核组合，均为拓扑2511、噪声847401、60秒轨迹；不是多个网络重复。训练和组合选择仍只读取原冻结无标签损失，以下TA/TB诊断及序列比例不进入优化。

当前评分首名 `{best['candidate']}`，J={best['J']:.6f}，原参考J={ref['J']:.6f}。这不是完整双模式恢复的验收。三个训练分项的变化见summary.json：分布损失与图中均值/先后概率摘要不是同一个量，不能用总分下降替代每项改善。

## TB的具体残差

SCL先后概率误差由 {float(baseobs['SCL_order_error']):.3f} 到 {float(obs['SCL_order_error']):.3f}；SCL−ICL中位质心时差由 {float(baseobs['rod_lag_ms']):.2f} 到 {float(obs['rod_lag_ms']):.2f} ms，患者约1.19 ms。正值表示SCL较晚。

只在四个SCL触点都参与的TB事件内计算严格序列：当前模型SCL6→7→8→9为{rev['SCL6_to9']}/{current['N_SCL4']}，患者为{pref['SCL6_to9']}/{patient['N_SCL4']}。二者四触点参与率本身也不同，完整分母分别为{current['N_events']}和{patient['N_events']}个TB事件，不能把条件比例直接当成全部事件占比。

把三个相邻时差均要求大于2 ms，模型逆序比例为{current['separation_gt_2_ms']['reverse_conditional']:.1%}、患者为{patient['separation_gt_2_ms']['reverse_conditional']:.1%}；仅排除2 ms以内的顺序抖动仍保留差距。四触点最大减最小质心跨度中位数为模型{current['median_SCL_span_ms']:.2f} ms、患者{patient['median_SCL_span_ms']:.2f} ms。这是质心时间与参与条件下的差别，不等于起燃位置、连续路径或真实机制已被识别；HFO与模型发放包络的表征差别仍存在。

## 参数作用的配对证据

右核向上移动0.5 mm、横坐标分别固定于三个局部取值时，TB杆间时差的变化分别为{[round(x['delta_rod_lag_ms'],2) for x in differences]} ms。SCL内部先后误差变化并非全部同方向；不称为跨seed稳健效应。

另复核既有13点方向和13点向外EE扫描。它们固定另一套旧背景（右核y=4.7155 mm），不能与本轮位置网格直接拼成同一响应面。在旧背景上，EE方向0°→+13.33°使TB SCL先后误差0.432→0.179，但杆间时差33.26→44.27 ms，总损失1.622→1.930。核向外EE×1.25→×1.5也使TB SCL误差0.432→0.297，但杆间时差几乎不变（33.26→32.82 ms），总损失变差至2.233。方向能改变局部顺序，不代表不付出其他代价。

接受：已有方向与位置对不同观测的配对作用线索。暂不接受：本轮已恢复完整患者TB，或者只是增加搜索次数就一定解决。当前继续完成固定预算的网格和最多4个双核组合；下一物理问题应是位置与方向的交互，先保留对照和固定单seed，不据此自动扩大预算或更换loss。

## 图件

[二维参数响应](../../figures/xy_response.png)；当前最佳点多事件GIF和患者模板/原生场快照由key_results/current.json定位。GIF按每类最早三个合格事件选例，不按患者相似度挑选；患者左图是32+32冻结小包的平均HFO包络，不是平均频谱，也不估计自然类别比例。完整FIT作为数值参考，二者身份分开。
'''
    (dest/'scientific_review.md').write_text(note)
    for axis in ['EE_out','EE_angle']:
        rr=[r for r in history if r['axis']==axis and r['mode']=='TB']
        print(axis,[{k:r[k] for k in ['value','J','SCL_order_error','rod_lag_ms','reverse_fraction','reverse_separated_2ms']} for r in rr],flush=True)
    print('current best sequence',summary['best_sequence']['TB'],flush=True)
    print('patient sequence',summary['patient_sequence']['TB'],flush=True)


if __name__=='__main__':main()
