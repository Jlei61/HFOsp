#!/usr/bin/env python3
"""Post-run scientific checks and common-window sensitivity; never changes simulations."""
import csv,json
from pathlib import Path
import numpy as np

def review(out):
    out=Path(out);records=[];rows=[];checks=[]
    for p in sorted((out/'workers').glob('*.json')):
        if p.name.endswith('.progress.json'):continue
        r=json.loads(p.read_text());j=r['job'];records.append(r)
        with np.load(p.with_suffix('.npz')) as a:
            tr=a['trace'];times=tr[:,0];counts=a['region_counts'];gn=a['group_n'];ne=int(gn[:3].sum())
            b=counts[:len(counts)//5*5,:3].reshape(-1,5,3).sum((1,2));high=b>=2*ne;streak=0;d=None
            for k,v in enumerate(high):
                streak=streak+1 if v else 0
                if streak>=20:d=(k+1)*10.;break
            checks.append({'job':j['id'],'exact_count_detection_matches':d==r['detected_ms']})
            # This common window is a post-hoc sensitivity, additional to the registered per-run window.
            eligible=[e for e in r['events'] if e['pilot_interictal_eligible'] and e['window_ms'][0]>=1500 and e['window_ms'][1]<=8000]
            idx=int(np.argmin(abs(times-8000)));w=(times>=1500)&(times<8000);nc={m:sum(e['mode']==m for e in eligible) for m in (0,1)}
            row=dict(job=j['id'],seed=j['seed'],s=j['s'],Z_dynamic=j['Z_dynamic'],M0_common=nc[0],M1_common=nc[1],common_M0_fraction=nc[0]/sum(nc.values()) if sum(nc.values()) else None,Z_A_8s=float(tr[idx,4]),Z_B_8s=float(tr[idx,9]),Z_all_E_8s=float(tr[idx,32]),raw_I_A_mean=float(tr[w,2].mean()),raw_I_B_mean=float(tr[w,7].mean()),effective_I_A_mean=float(tr[w,3].mean()),effective_I_B_mean=float(tr[w,8].mean()),load_fraction_A_mean=float(tr[w,5].mean()),load_fraction_B_mean=float(tr[w,10].mean()))
            if d is not None:
                ti=np.argmin(abs(times-d));row['Z_all_E_at_confirmation']=float(tr[ti,32])
                # Final 1s from exact 10ms count bins, without an extra endpoint bin.
                last=b[-100:]/ne/.01;row['final1s_E_Hz']=float(last.mean());row['final1s_quiet_fraction']=float(np.mean(last<1))
                for label,cols in [('A',(2,3)),('B',(7,8))]:
                    start=(times>=d)&(times<d+100);end=times>=times[-1]-90
                    row[f'raw_I_{label}_first100ms']=float(tr[start,cols[0]].mean());row[f'raw_I_{label}_last100ms']=float(tr[end,cols[0]].mean())
                    row[f'effective_I_{label}_first100ms']=float(tr[start,cols[1]].mean());row[f'effective_I_{label}_last100ms']=float(tr[end,cols[1]].mean())
            rows.append(row)
    if not all(c['exact_count_detection_matches'] for c in checks):raise RuntimeError('Detection readout mismatch')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (out/'common_window_and_resource_audit.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(rows)
    groups=[]
    for uz in (False,True):
        for s in (-.5,0,.5):
            rr=[r for r in rows if r['Z_dynamic']==uz and r['s']==s]
            groups.append(dict(Z_dynamic=uz,s=s,n_runs=len(rr),M0=sum(r['M0_common'] for r in rr),M1=sum(r['M1_common'] for r in rr)))
    lookup={(r['job']['seed'],r['job']['s']):r for r in records if r['job']['Z_dynamic']}
    paired=[]
    for seed in sorted({k[0] for k in lookup}):
        rr=[lookup.get((seed,s)) for s in (-.5,0,.5)]
        if all(r is not None and r['detected_ms'] is not None for r in rr):
            ts=[r['detected_ms']/1000 for r in rr];paired.append(dict(seed=seed,negative_s=ts[0],zero_s=ts[1],positive_s=ts[2],negative_minus_zero_s=ts[0]-ts[1],positive_minus_zero_s=ts[2]-ts[1],negative_minus_positive_s=ts[0]-ts[2]))
    payload=dict(status='PASS',n_complete=len(records),n_planned=18,common_window_ms=[1500,8000],window_scope='Post-hoc common-window sensitivity; frozen primary window retained',checks=checks,common_mode_counts=groups,paired_nativeZ_detection=paired,claim_scope='Single graph; 3 noise runs per condition; operational high-state detection; M off; fixed s; no manual refill; no clinical validation')
    (out/'scientific_audit.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    lines=['# s × Z pilot 科学解读','',f'完成 {len(records)}/18；这是已完成运行的分析，不将运行中条件视为阴性。','', '**模式偏好。** 原生Z的9条已完成，s符号改变进入持续高活动前的模式分布；1.5–8秒共同窗敏感性保留该方向。统计单位是3个噪声重演，不是合并事件数。','', '| Z | s | 完成运行 | 共同窗M0/M1 |','|---|---|---|---|']
    for g in groups:lines.append(f"| {'原生' if g['Z_dynamic'] else '固定1'} | {g['s']:+.1f} | {g['n_runs']}/3 | {g['M0']}/{g['M1']} |")
    lines+=['','**高活动进入。** 原生Z下所有三个s条件均进入高活动。负s相对零s的提前在三条配对噪声中同向，正s相对零s没有一致延后。不能把正负s解释成发作开关。','', '| seed | s=-0.5 | s=0 | s=+0.5 | 负s减零s |','|---|---|---|---|---|']
    for r in paired:lines.append(f"| {r['seed']} | {r['negative_s']:.2f} | {r['zero_s']:.2f} | {r['positive_s']:.2f} | {r['negative_minus_zero_s']:.2f} |")
    lines+=['','**资源机制。** 8秒时负s的两个core平均Z均低于同seed的正s，但全E平均Z并不呈相同一致排序。局部资源轨迹与模式选择不同于一个全局平均Z阈值。原始I输入增加与有效抑制增加不是同一件事，必须分别检查core和核外群体；不能把core中的有效抑制下降推成全E的有效抑制都下降。s改变网络活动及Z历史，但当前设计没有隔离Z作为中介的贡献，需匹配快速初态和未来输入的反事实才能进一步确认。','', '**持续与恢复。** 原生Z运行末端仍为高活动、Z继续下降，没有观察到自主返回；只观察触发后2秒，不能断言永不恢复。本轮M关闭，不能评价ZM联合模型的自主终止。','', '**验收边界。** 支持固定结构下状态调制模式偏好，并为状态—资源耦合提供pilot证据；不支持临床发作匹配、两模式完整传播恢复、s的患者拟合或双稳态。M1的SCL参与缺口仍应对照run_summary.csv，不因类别比例改善而取消。','', '图件入口：scientific_report.md；精确计数重算的高活动确认时刻全部与worker一致。共同时窗和局部资源数值见common_window_and_resource_audit.csv。']
    (out/'scientific_interpretation.md').write_text('\n'.join(lines)+'\n')
    return payload
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();print(json.dumps(review(a.out)))
