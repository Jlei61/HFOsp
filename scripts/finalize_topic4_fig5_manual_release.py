#!/usr/bin/env python3
"""Finish the already authorized bounded batch and save reviewable figures."""
from pathlib import Path
import time
import json
import subprocess
import sys
import hashlib

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/fig5_manual_core_release_v1'


def write(name,value):
    p=OUT/name;temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(p)


def write_report():
    analysis=json.loads((OUT/'analysis.json').read_text())
    metadata=json.loads((OUT/'figure_metadata.json').read_text())
    run=metadata['main_run'];lat=metadata['latency']
    lines=['# 手放双核 Figure 5：连续读出、补回后释放 Z 与参数—转变时间',
           '', '状态：有界批次和图件生成完成，数值QA及Agent图面自查通过；待用户科学审阅。',
           '', '## 本次改变与保留',
           '按用户新布局，左侧依次为未滤波虚拟SEEG电流proxy（实际施加的|AMPA|+|Z×GABA|）、六个250毫秒raster窗、逐神经元Z的统计、同窗全E空间活动；右侧为实际E–I–Z三维轨迹及参数—转变时间。所有左侧读出取自一条连续SNN，不再用冻结后的续跑代替时间快照。',
           '底物沿用historical_manual_hard_native_z_v1：历史手放双核/阈值场，当前C快速连接、全局及空间OU、Poisson；不切换到另一个任务正在验证的core驱动输入版本。M保持关闭。',
           '仅改变补回后的协议：第一次触发持续高活动后等待500毫秒，线性补回Z一秒，随后立即恢复原Z方程，不再次补回，不重置快速状态或噪声。',
           '', '## 连续轨迹',
           f'首次达到操作性高活动判据为{run["first_trigger_ms"]/1000:.2f}秒；补回{run["restore_start_ms"]/1000:.2f}–{run["release_ms"]/1000:.2f}秒；随后原生演化至26秒。全部高态触发时刻(ms)：{run["all_trigger_times_ms"]}。',
           '', '| 窗口 | E均率(Hz) | I均率(Hz) | 安静比例 | 平均Z |',
           '|---|---:|---:|---:|---:|']
    for s in analysis['stages']:
        lines.append(f'| {s["label"]} {s["window_s"]} s | {s["E_mean_hz"]:.2f} | {s["I_mean_hz"]:.2f} | {s["E_quiet_fraction"]:.3f} | {s["Z_mean"]:.3f} |')
    lines+=['','## 参数图',
            'tauZ=2.5、5、10秒；耗竭电流阈值=75、95.1985、120；每格三条噪声9108401/2/3，拓扑6101固定。代表连续轨迹贡献中心格的第一条首次进入时间，其补回发生在该读出之后，不影响首入时间。',
            '全E率10毫秒分箱连续200毫秒≥200Hz为操作性高态触发；观察期24秒。左图为进入比例，右图为mean(min(首入时间,24秒))，即24秒限制平均未转变时间；未进入者不剔除，也不伪造更大的发生时间。',
            'tauZ同时影响耗竭和恢复；Ith为原始GABA电流的阈值，不是每事件耗竭量。统计单位为噪声运行，不是事件或患者。',
            '', '| Ith | tau=2.5s：比例 / 限制均时 | tau=5s | tau=10s |','|---|---|---|---|']
    for j,h in enumerate(lat['thresholds']):
        vals=[f'{lat["transition_probability"][j][k]:.2f} / {lat["restricted_mean_transition_free_time_s"][j][k]:.2f}s' for k in range(3)]
        lines.append(f'| {h:.4f} | '+ ' | '.join(vals)+' |')
    lines+=['','## 相图及机制边界',
            '三维轨迹来自全E/I群体的5毫秒分箱率（绘图平滑sigma=5毫秒）与实际平均Z，原仿真保留每个神经元的Z。方向场为各Z带内有采样支持的条件平均漂移；外部补回阶段排除于漂移估计。未采样区域不外推，不把投影轨迹的回环命名为极限环。',
            '旧近似模型未通过原SNN状态边界校验，本次未把其固定点或特征值移植到手放底物。当前图不是完整SNN的nullcline或Hopf证明。空间50毫秒图为本地放电率，亮度同时反映参与与频率，不能独自证明逐神经元首招募顺序。',
            '第二次进入后的25–26秒，全E均率约474.44Hz，5毫秒分箱率的变异系数约0.0028：这是接近饱和的持续高率，不能称为已验证的持续群体振荡。前段自限事件与轨迹回环也不能替代极限环或分岔证据。',
            '另交付同一8秒/9.4秒检查点的自然Z与冻结Z对照，以及同次补回后维持Z=1与释放Z的对照。冻结不同程度的Z可能限制高态强度而未消除持续放电；不能统一改写为“钳制Z必然阻止所有转变”。',
            '', '## 文件',
            f'结果目录：`{OUT}`。图见`figures/README.md`；参数逐运行表为`transition_times.csv`；相图箭头原数组为`phase_drift_data.npz`；科学读出见`analysis.json`与`paired_control_summary.json`；数值QA见`artifact_qa.json`。',
            '所有图仅为当前候选，不覆盖正式paper-ready Figure5。']
    report=ROOT/'docs/archive/topic4/fig5_manual_core_release_2026-09-10.md'
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    started=time.time()
    while True:
        status=json.loads((OUT/'status.json').read_text())
        if status['status']=='SIMULATIONS_COMPLETE':break
        failures=[]
        for f in (OUT/'progress').glob('*.json'):
            r=json.loads(f.read_text())
            if r.get('status')=='FAILED':failures.append(r)
        if failures:raise RuntimeError(failures)
        if time.time()-started>14400:raise RuntimeError('Bounded finalizer timed out after 4h; no new jobs launched.')
        time.sleep(5)
    for script in ['analyze_topic4_fig5_manual_release.py','plot_topic4_fig5_manual_release.py']:
        with (OUT/'logs'/(script+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    files=[ROOT/'scripts'/s for s in ['run_topic4_fig5_manual_release.py','analyze_topic4_fig5_manual_release.py','plot_topic4_fig5_manual_release.py','finalize_topic4_fig5_manual_release.py']]
    write('producer_manifest.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    write_report()
    write('delivery_status.json',dict(status='ARTIFACTS_COMPLETE_PENDING_AGENT_VISUAL_REVIEW',
                                    figures=[str(p) for p in sorted((OUT/'figures').glob('*.png'))],
                                    automatic_new_batch=False,human_acceptance='PENDING_USER_REVIEW'))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write('finalizer_failure.json',dict(error=repr(exc)));raise
