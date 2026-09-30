#!/usr/bin/env python3
"""Distinguish reaching the low-rate regime from staying long enough to recover."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import shutil
import numpy as np
from campaign import ROOT, read, write, sha
from analyze_natural_exit_mediators import DEST, START, CONTROLS


def main():
    result = read(DEST/'result.json');rows = []
    for row in result['rows']:
        with np.load(DEST/f"{row['name']}.npz") as d:
            t, R, K, G, q = [d[k] for k in ['time_s', 'causal_R_Hz', 'K_mean', 'Graw', 'gate']]
            on = np.flatnonzero(q > 0)
            i = int(on[0]) if len(on) else None
            stop = i if i is not None else len(t)
            nadir = int(np.argmin(R[:stop]))
            # For any intervening time s, R(s)>=R(left)*exp(-1ms/15ms).
            # This lower bound follows from nonnegative spike increments.
            lower = float(R[:stop].min()*np.exp(-.001/.015))
            z_i = max(0, np.searchsorted(d['slow_time_s'], t[i] if i is not None else t[-1], side='right')-1)
            causal = dict(name=row['name'], first_q_reopens_s=None if i is None else float(t[i]),
                R_min_before_q_reopens_Hz=float(R[nadir]), R_min_time_s=float(t[nadir]),
                K_at_R_min=float(K[nadir]), continuous_R_lower_bound_Hz=lower,
                no_R_at_or_below5_before_q_reopens_certified=bool(i is not None and lower > 5),
                core_Z_last_sample_before_q_reopens=d['Z_allE_A_B_other'][z_i, 1:3].tolist() if i is not None else None,
                core_Z_sample_time_s=float(d['slow_time_s'][z_i]) if i is not None else None,
                K_at_q_reopens=None if i is None else float(K[i]),
                first_low_s=row['first_R_at_or_below5_for100ms_s'],
                both_core_reference_s=row['first_both_core_Z_reference_s'],
                final_core_Z=row['final_Z_allE_A_B_other'][1:3],
                recurrent_G_is_allowed=True, counts_as_autonomous_loop=False)
            rows.append(causal)
    baseline, remove_g, fast_k = rows
    g_support = remove_g['no_R_at_or_below5_before_q_reopens_certified']
    fast_k_initial_exit_same = fast_k['first_low_s'] == baseline['first_low_s']
    fast_k_early_reactivation = (fast_k['first_q_reopens_s'] is not None and
        np.any(np.asarray(fast_k['core_Z_last_sample_before_q_reopens']) < result['rows'][2]['core_reference']))
    review = dict(status='COMPLETE_PAIRED_NATIVE_CAPTURE_REVIEW', rows=rows,
        existing_G_tail_supports_first_low_rate_capture_at_this_state=g_support,
        shortening_low_K_retention_preserves_initial_low_transition=fast_k_initial_exit_same,
        shortened_K_retention_reactivates_before_both_core_reference=bool(fast_k_early_reactivation),
        scope='Two interventions at one complete endogenous state and one shared future noise path. First capture, later recovery and brief propagation return are distinct. G can regenerate after its one-time removal; no claim of universal necessity or formal bifurcation.',
        producer_sha256=sha(__file__))
    write(DEST/'capture_review.json', review);shutil.copy2(__file__, DEST/'capture_review_producer.py')
    labels = ['原轨迹', '一次移除已有 G', '缩短低率 K 保留']
    lines = ['# 真实退出：进入低率段与保留恢复窗口','',
        '本次问题是：G 的已有尾迹帮助什么，低率 K 的保留帮助什么？两条干预从完全相同的 16.8 秒内部状态和未来输入出发，Z/K/M 正常演化；原始轨迹作为配对基线。干预不计入自主闭环。','',
        '|条件|第一次低率段开始 s|q 再开启 s|两核恢复到原参考 s|', '|---|---|---|---|']
    def fmt(v):return '本窗未观察到' if v is None else f'{v:.3f}'
    for label, row in zip(labels, rows):
        lines.append(f"|{label}|{fmt(row['first_low_s'])}|{fmt(row['first_q_reopens_s'])}|{fmt(row['both_core_reference_s'])}|")
    lines += ['', '低率开始沿用因果 R≤5 Hz 持续 100 ms 的诊断读出；物理 K 衰减切换仍逐 0.1 ms 依据步前 R。q 再开启表示新增高率反馈重新被招募，不能单凭它认定完整发作进入。两核参考值沿用原始间期记录，未新设释放开关。','']
    if g_support:
        lines += [f"一次去掉 G 后，第一段最低 R 为 {remove_g['R_min_before_q_reopens_Hz']:.3f} Hz，发生在 {remove_g['R_min_time_s']:.3f} s。采样间的严格下界仍为 {remove_g['continuous_R_lower_bound_Hz']:.3f} Hz，高于 5 Hz，排除了漏看短暂低率切换的解释。K 因而留在快速消退段，之后活动重新招募 q。G 尾迹在这个状态下帮助第一次及时进入低率段。", '']
    if fast_k_initial_exit_same and fast_k_early_reactivation:
        lines += ['缩短 K 保留没有改变第一次下降到低率段的时刻，却让高率反馈在两核尚未回到原参考时重新开启。这直接区分了终止最初的高活动与获得充分资源恢复：两者需要的条件不同。', '']
    if fast_k['both_core_reference_s'] is not None:
        lines += [f"但缩短 K 保留后，两核仍在 {fast_k['both_core_reference_s']:.2f} s 达到参考，原轨迹为 {baseline['both_core_reference_s']:.2f} s。因此本次反证否定了‘5秒保留是充分Z恢复的必要条件’这一强说法。它提供更连续的安静恢复并推迟活动返回，但短K情况下也能经过活动与恢复交替逐步恢复；是否重现原生间期传播及持续序列仍需另验。", '']
    lines += ['一次移除 G 后，模型仍可重新产生 G；后续再退出不能解释成“G 不重要”，也不能声称该干预永久阻止退出。短事件计数及截尾保存在完整分析，传播身份仍不能由计数替代。','',
        '分岔解释需要据此调整：固定 Z/K 的高态变化说明持续活动的条件范围；真实退出还受 G 的携带状态及 K 进入不同衰减段的先后控制。这里没有把 5 Hz 切换或有限窗反弹自动命名为 Hopf、fold 或边界分岔。正式分支仍需物理动态对应与稳定性认证。','',
        '验核包括 20 s 完整原轨迹续演逐位复现、两干预规定之外初态逐位一致、100 条未来输入记录与末态完整外源状态配对、逐区 Z 预算守恒及实际低率 K 指数式。低率 K 首次任务在任何仿真步前因类名元数据不匹配失败，修复记录与原 checkpoint 留存，物理初态未变。']
    (ROOT/'natural_exit_capture_review.md').write_text('\n'.join(lines)+'\n')
    print(review, flush=True)


if __name__ == '__main__':main()
