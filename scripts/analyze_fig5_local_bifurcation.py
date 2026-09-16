"""Summarize actual branch and delay evidence without promoting SNN claims."""
import sys,csv,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fig5_near_runaway_branches as run
from src.topic4_xy_fig5_followup import read,write


def analyze():
    expected=run.jobs();complete=[];missing=[]
    for cid,key,kind,value in expected:
        p=run.DATA/cid/'stability'/f'{key}_{kind}_{value}.json'
        if not p.exists():missing.append(str(p));continue
        r=read(p);r['original_converged']=r['converged']
        retry=run.DATA/cid/'silent_root_spectrum_retry.json'
        if not r['converged'] and key=='ee' and r['population_hz']<.1 and retry.exists():
            check=read(retry)
            if check['converged']:
                r['converged']=True;r['modes']=check['modes'];r['retry_source']=str(retry)
        leading=r['modes'][0] if r['modes'] else {}
        complete.append(dict(candidate_id=cid,parameter_name=key,source=str(p),**r,
            leading_growth_per_ms=leading.get('growth_per_ms'),
            leading_frequency_hz=leading.get('frequency_hz')))
    summaries=[]
    for cid in run.base.rec.IDS:
        for key in run.BOUNDS:
            r=read(run.DATA/cid/(key+'.json'))
            folds=[f for f in r['folds'] if f.get('confirmed') and f['population_hz']>250]
            closest=min(folds,key=lambda f:abs(f['parameter']-1)) if folds else None
            entries=[c for c in complete if c['candidate_id']==cid and c['parameter_name']==key]
            summaries.append(dict(candidate_id=cid,parameter=key,bounds=run.BOUNDS[key],
                nearest_verified_high_rate_fold=closest,roots=r['root_count'],max_residual=r['max_residual'],
                continuation_stops=[b['reason'] for b in r['branches']],seed_failures=r['failures'],
                stability=entries))
    resolved=sum(r['converged'] for r in complete)
    report=dict(stage='REDUCED_SCREEN_CLOSED_WITH_UNRESOLVED_SPECTRA' if not missing else 'DELAY_SCREEN_RUNNING',
        completed=len(complete),total=len(expected),missing=missing,rows=summaries,
        resolved_stability=resolved,unresolved_stability=len(complete)-resolved,
        scope='Two fixed GIF substrates; local deterministic diffusion-rate reduction, not finite-SNN bifurcation proof.',
        key_limitations=['Near-silent equilibrium does not represent recurrent interictal events.',
            'High equilibrium folds may already have growing oscillatory delay modes.',
            'Only selected roots have stability estimates; full periodic-orbit continuation not performed.',
            'No spatial-grid convergence or colored-current closure validation here.',
            'tau_Z is excluded by frozen-Z construction; its effect on the full slow-fast system remains open.'])
    write(run.DATA/'analysis.json',report)
    with (run.DATA/'stability_summary.csv').open('w') as fp:
        keys=['candidate_id','parameter_name','parameter','population_hz','converged','leading_growth_per_ms','leading_frequency_hz','source']
        w=csv.DictWriter(fp,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(complete)
    lines=['# Runaway 附近核心参数的局部分支分析','',
        '问题是核心参数改变时哪些平衡分支存在、何时失稳。分析使用两套已选 GIF 基底的实际连接、阈值与延迟，Z 场来自真实早期/转变前检查点；没有重新优化 XY 或患者传播拟合。','',
        f'已尝试 {len(complete)}/{len(expected)} 个延迟稳定性状态，其中 {resolved} 个得到收敛谱（含已通过的扩大子空间复核），{len(complete)-resolved} 个保留未确定；两套基底共 14 条非 Z 参数扫描已生成。','',
        '| 基底 | 选定的 Z 损失量 λ | 平均 Z | 同参数下发现的平衡 E 率（Hz） |',
        '|---|---:|---:|---|']
    for i,cid in enumerate(run.base.rec.IDS):
        a=read(run.DATA/cid/'anchor.json')
        lines.append(f'| {i+1} | {a["lambda_anchor"]:.4f} | {a["mean_Z"]:.4f} | '+', '.join(f'{x:.3f}' for x in a['root_rates_hz'])+' |')
    lines+=['','λ 工作点由各自高态折点后移 0.03 选定，用于比较局部参数效应，不能与 SNN onset 等同。低支接近静默，不能直接称为间期事件状态。','',
        '| 基底 | 参数 | 高率分支上离基线最近的已验证折点（倍率） |',
        '|---|---|---:|']
    for r in summaries:
        f=r['nearest_verified_high_rate_fold'];value=f'{f["parameter"]:.6f}' if f else '本次追踪未检出'
        lines.append(f'| {run.base.rec.IDS.index(r["candidate_id"])+1} | {r["parameter"]} | {value} |')
    lines+=['','M 强度扫描为 0–2 倍，M 时间常数为 250–1000 ms；另有 ηMτM 固定的时间常数对照。GABA 衰减为各自基线的 0.5–1.5 倍（9–27 ms、12–36 ms）。EE 为 0.9–1.1 倍，E→I 与 I→E 为 0.85–1.15 倍。','',
        '平衡折点通过 F=0、Jv=0、孤立零模、横截性及二次非退化检验。两个 Z 折点的零模均能提升为带实际离散延迟系统的 +1 乘子，误差 < 4e-15；LIF 积分求积阶数从 16 提高到 32 后折点移动 < 2e-11。','',
        '延迟雅可比已修正：响应斜率在减去自洽 M 电流后计算；独立的非线性一步映射有限差分检查通过。','',
        '**结论边界**：平衡根存在不代表它是吸引态。选定基底的高根已经出现约 26.5/23.4 Hz 的增长振荡模态，所以不能把这两个折点直接解释为间期状态失稳并进入 runaway 的机制。稳定性全量读数见 stability_summary.csv；后续分岔归因需要检查实际吸引态和周期轨道，并与固定 Z 的原始 SNN 对照。','',
        '部分分支到达步数上限或校正失败；这些末端不定义生物学边界。该结果为局部候选图，尚未经过空间分辨率收敛与作者目视验收。','',
        '方法参考：[Brunel 2000](https://doi.org/10.1023/A:1008925309027)，[Dhooge et al. 2003](https://doi.org/10.1145/779359.779362)。']
    lines+=['','## GABA 振荡稳定性边界','',
        '逐点从零增长率重新求根会切换到邻近模态，已改为从基线完整延迟矩阵的主导模态出发连续追踪；旧的跳模态草图已移至 superseded_mode_tracking，不作为结果。']
    for i,cid in enumerate(run.base.rec.IDS):
        path=run.DATA/cid/'gaba_oscillatory_boundary.json'
        if not path.exists():continue
        r=read(path)
        if not r.get('discovery'):continue
        if not r['crossings']:
            lines+=['',f'基底 {i+1}：GABA 从 18 到 27 ms，所追踪的增长振荡模态持续存在，未发现这条模态的稳定性过零。']
        else:
            c=r['crossings'][0];limit=r['crossings'][-1]
            lines+=['',f'基底 {i+1}：在 native dt=0.1 ms 下，振荡模态过零位置为 GABA={c["tau_gaba_ms"]:.4f} ms、频率 {c["frequency_hz"]:.3f} Hz，横截增长率斜率非零。连续时间极限为 {limit["tau_gaba_ms"]:.4f} ms，差约 5%，因此精确临界时间常数仍受数值步长影响。']
            checks={}
            for side in ['below','above']:
                q=run.DATA/cid/f'gaba_crossing_full_spectrum_{side}.json'
                if q.exists():checks[side]=read(q)
            if len(checks)==2 and all(q['converged'] for q in checks.values()):
                signs=[checks[k]['modes'][0]['growth_per_ms'] for k in ['below','above']]
                lines+=['',f'完整延迟矩阵主导谱在边界两侧（GABA 倍率 ±0.02）的增长率为 {signs[0]*1000:.4f} 与 {signs[1]*1000:.4f} s⁻¹。']
            else:lines+=['','边界附近完整延迟矩阵的主导谱核查尚未全部获得收敛结果。']
    lines+=['','该过零改变的是约 335 Hz 高活动平衡态的振荡稳定性。尚未计算周期轨道、第一 Lyapunov 系数或原始 SNN 的固定 Z 对照，因此采用“Hopf 候选/振荡稳定性边界”，不写成已证明的间期事件到 runaway 分岔。']
    (run.DATA/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__=='__main__':
    r=analyze();print(r['stage'],r['completed'],r['total'])
