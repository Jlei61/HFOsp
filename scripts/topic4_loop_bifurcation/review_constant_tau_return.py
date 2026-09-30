#!/usr/bin/env python3
"""Complete the bounded return review without changing its acceptance rules."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import shutil
import numpy as np
from campaign import ROOT, read, write, sha
import analyze_constant_tau_return as analysis
import zoom_topic4_return_core_propagation as zoom

OUT = analysis.OUT/'scientific_review'


def state_at(t, times, Z, K, mk):
    i = np.searchsorted(times, t, side='right')-1;j = np.searchsorted(mk, t, side='right')-1
    return dict(event_s=t, preceding_Z_sample_s=float(times[i]),
        Z_allE_coreA_coreB=Z[i, [0, 5, 6]].tolist(), K_mean=float(K[j]))


def main():
    OUT.mkdir(exist_ok=True)
    r = read(analysis.DEST/'result.json');d = dict(np.load(analysis.DEST/'readouts.npz'))
    e = r['episodes'][0];events = e['after_reference_events']['brief_events'];assert len(events) == 13
    z = analysis.part(analysis.BASE, 'chunks', 'slow_time_ms', ['Z'], 0, 56.8)
    k = analysis.part(analysis.BASE, 'intrinsic_adaptation_chunks', 'time_ms', ['sahp_mean_conductance_ratio'], 0, 56.8)
    original_first = state_at(49.31, z['slow_time_ms']/1000, z['Z'], k['sahp_mean_conductance_ratio'], k['time_ms']/1000)
    candidate_first = state_at(events[0]['start_s'], d['slow_time_s'], d['Z'], d['K_mean'], d['time1_s'])
    reentry = state_at(e['window_end_s'], d['slow_time_s'], d['Z'], d['K_mean'], d['time1_s'])
    assert original_first['Z_allE_coreA_coreB'][1] > .99
    weak = [ev for ev, metric in zip(events, e['spatial_event_metrics']) if max(metric['peak_5ms_Hz'][1:3]) < 100]
    examples = [('Original interictal reference', .57)] + [(f'Returned event {i+1}', ev['start_s']) for i, ev in enumerate(events[:3])]
    if weak:examples.append(('First weak returned event', weak[0]['start_s']))
    previous = zoom.OUT;zoom.OUT = ROOT
    try:zoom.storyboard(dict(rates=d['rates5_Hz'], field_rates=d['field5_Hz']),
        dict(np.load(analysis.OUT/'geometry.npz')), examples, 'constant_tau_return_spatial_sequence.png')
    finally:zoom.OUT = previous
    result = dict(status='COMPLETE_RETURN_REVIEW_CANDIDATE', original_first_return=original_first,
        candidate_first_return=candidate_first, candidate_reentry=reentry,
        after_reference_finite_events=e['after_reference_events']['finite_count'], brief_events=13,
        brief_fraction=e['after_reference_events']['brief_fraction'],
        brief_span_s=e['after_reference_events']['event_span_s'], minimum_required_span_s=5.,
        reference_feature_ratios=e['reference_ratios'], spatial_summary=e['spatial_summary'],
        selected_spatial_examples=examples, first_weak_events=weak,
        return_screen='FAIL_INSUFFICIENT_EVENT_SPAN', original_fig5_replaced=False,
        mechanism='The constant-tau parameter trajectory permits full core-Z reference crossing and13 brief returns, but begins returning nearcoreZ0.79 rather than the original first-return coreZ0.999. It reenters after a2.97s brief span. This is consistent with a smaller recovered resource reserve and faster recurrent depletion, not proof that Z reserve alone determines return duration: K/G/M and time-dependent drive differ at the two return times.',
        interpretation='Same-seed paired parameter trajectory. The0-16.8s invariant prefix makes this consistent with the constant-tau model fromt0; do not add it as an independent seed or to original-model confirmed loop counts. NativeZ/G/M/K and all fast states evolve without release timer or clamp.',
        agent_spatial_review='PENDING', human_review='PENDING', producer_sha256=sha(__file__))
    write(OUT/'result.json', result);shutil.copy2(__file__, OUT/'producer.py')
    (OUT/'review.md').write_text('''# 恒定K衰减：有真实短事件返回，但恢复余量较小\n\n30秒延长已完整结束，拼接后的观察窗为0–56.8秒。原始8秒数组及raster一致，568条外源记录与原轨迹配对一致，所有区域/空间计数、Z收支和实际K衰减式均通过核对。它是一个同种子的恒定0.5秒tau参数替代轨迹；前16.8秒已证明对此改动不敏感，因此轨迹与从起点使用该参数相容，但不新增独立种子或原模型闭环计数。\n\n第一次两核Z达到原参考后出现13个完整短事件；持续时间、事件间隔和全E峰值中位数均在原先规定的0.5–2倍范围内，短事件比例也通过。其跨度只有2.97秒，29.87秒再次进入高活动，因此未通过预定至少5秒的持续返回门。后两次低活动后再次进入前，两核Z没有同时达到原参考。观察终点保留再次进入，不选择性截掉。\n\n11个短事件达到明显双核参与，先后顺序有A先也有B先；还有弱事件，全部保留。空间图按时间显示最初三个事件及首个弱事件，共同150毫秒窗口和5毫秒原生场，不把所有事件强称为同一种核间传播。原有raster/率/Z放大格式保持。\n\n原模型首次返回前两核Z约0.999，恒定tau首次返回前约0.79，随后逐步消耗并再次进入。与原式K尾部较慢消退的结果一起看，K保留并非充分Z恢复或短事件返回的必要条件，但延迟返回可以留下更大资源余量。后一条是由方程和轨迹支持的机制解释；并未通过只交换Z的对照证明它是返回时长差异的唯一中介。\n\n这一候选不替换已认可的Fig5，不放宽持续返回标准。图已生成，人工验收待定；正式分岔问题仍独立未完成。\n''')
    title = '### constant_tau_return_spatial_sequence.png';p = ROOT/'figures/README.md'
    if title not in p.read_text():
        with p.open('a') as f:f.write('\n\n'+title+'\n按时间展示原间期固定示例、恒定tau候选充分Z恢复后的最初三个短事件以及首个弱事件，沿用原生5ms/1mm空间读出和共同150ms窗口。所有13个短事件仍计入总表，示例不按效果挑选。\n**关注点**：区分核起始、双核招募、核外传播和弱事件；一次事件的空间图不能替代持续返回门，人工待审。\n')
    print(result, flush=True)


if __name__ == '__main__':main()
