#!/usr/bin/env python3
"""Write the scientific readout of this fixed native figure experiment."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/topic4_sef_hfo/fig5_manual_core_release_v1'
OUT=BASE/'layout_v11'


def main():
    native=json.loads((BASE/'native_transition_v2/analysis_summary.json').read_text())
    audit=json.loads((BASE/'native_transition_v2/spikes_broadband_audit.json').read_text())
    spectrum=json.loads((BASE/'native_transition_v2/spectrum_robustness.json').read_text())
    lines=['# Fig. 5 v11：原生空间场与 Z 参数加密', '',
        '本版回答两个问题：状态2到3的空间招募是否真实存在于SNN神经元场，以及原有Z动力学参数能否改变进入高率状态的时间。所有结果使用同一历史手放双核、同一连接与阈值；未切换到其他任务的新噪声/连接模型。', '',
        '## 原生场实际显示什么', '',
        '0.5毫米网格直接覆盖40,000个神经元的20×20毫米平面。E/I放电率直接来自spikes；频带功率来自每格E细胞平均的 |AMPA|+|Z·GABA| 电流。没有电极插值或空间平滑，仍是模型电流代理，不能当作已验证的生物物理SEEG。实际core半径为1.5毫米；原连续raster的分组邻域沿用1.75毫米，二者含义不同。', '',
        '| 窗口（秒） | 全E平均率（Hz） | Core A 20–150 Hz变化（dB） | Core B 20–150 Hz变化（dB） |',
        '|---|---:|---:|---:|']
    for k,w in enumerate(native['windows_s']):
        rows=[r for r in native['regional_readouts'] if r['window_start_s']==w[0]]
        values={r['region']:r['band_20_150_dB'] for r in rows}
        lines.append(f"| {w[0]:.2f}–{w[1]:.2f} | {native['global_E_rates'][k]:.1f} | {values['Core A']:.1f} | {values['Core B']:.1f} |")
    lines += ['',
        '放电形成跨越两核并向侧方扩张的活动带，支持真实空间招募。20–150 Hz短窗功率在核心区明显下降；换成仅去均值或较低活动基线仍保留这一结果。10.50–10.75秒，两核250–500 Hz功率分别增加约7.2/10.1 dB，但局部电流RMS相对均值均约0.5%。固定抽样的48个A-core和40个B-core神经元此窗全部ISI CV<0.05，支持规则连续放电；这不等同于群体相位同步或大幅持续振荡。', '',
        '直接核对0.1毫秒原始全群体spike计数，最后两个窗E群体250–500 Hz带内RMS约为均值的0.95%/1.11%，I群体约0.79%/0.73%。因此小幅高频波动也不只是电流低通滤波的表象；全群体平均仍可能弱化局部相位差，不能据此证明不存在任何局部振荡模态。', '',
        '## 与患者早期能量场的观测对应', '',
        '患者Figure 3采用1–150 Hz、1秒PSD窗与基线robust-z。E2的250毫秒窗用于分开快速转变，主带20–150 Hz，不能将它的负结果外推成所有频带均无早期增强。补充1秒窗分析在9.5–10.5秒测得A/B核心1–150 Hz功率分别增加4.34/3.33 dB；10–11秒则为−2.81/+1.88 dB。较长窗混合了低频上升、点火和高率状态，显示短暂早期增强与随后中频减弱可同时成立，尚未证明与患者间期rank/发作能量分布同型。', '',
        '9.5–10.5秒，两核1–19 Hz占1–150 Hz合计功率约98%；同一1秒窗的20–150 Hz功率仍略降约0.6 dB。改为线性去趋势后1–150 Hz增强仍保留。因此更准确的描述是低频占主导的早期合计功率增强，并非带内各频率都增强。', '',
        '短窗主结果及1秒补充结果均保留。补充使用同一0.5–8秒基线，14个1秒重叠窗，不满足临床合同的至少50帧资格；图示dB而非冒充临床robust-z统计。其加入依据是现有患者图的频带合同，属于本轮看到短窗结果后的敏感性检查。', '',
        '## Z参数与进入时间', '',
        '原生方程为 τZ·dZⱼ/dt = 1[I_GABA,ⱼ < Ith] − Zⱼ（E靶细胞），I细胞Z=1，M关闭。记 d=1−mean(Z)、u=超过原始GABA耗竭阈值的E细胞比例，则 τZ·dd/dt=u−d。Z反映带恢复的耗竭驱动累积，并非无恢复的放电总计数；τZ同时控制耗竭和恢复，Ith是耗竭电流阈值，不是抑制连接强度。', '']
    path=BASE/'latency_dense_v1/analysis_summary.json'
    if path.exists():
        dense=json.loads(path.read_text());a=np.load(path.with_name('analysis_arrays.npz'))
        lines += [f"真实7×7节点，每格3条配对噪声，共147条（新增120、复用27）。首次连续200毫秒满足全E的10毫秒bin≥200 Hz时记为进入；未在24秒内达到者为右删失，共{dense['n_censored']}条。F显示mean(min(T,24s))，不是只对已进入者求均值；节点间未插值。", '',
            '| τZ（秒），Ith=95.2 | 限制平均进入时间（秒） |','|---:|---:|']
        for x,y in zip(dense['tau_s'],dense['tau_center_threshold_means_s']):lines.append(f'| {x:.3f} | {y:.2f} |')
        lines += ['', '| Ith，τZ=5秒 | 限制平均进入时间（秒） |','|---:|---:|']
        for x,y in zip(dense['threshold'],dense['threshold_center_tau_means_s']):lines.append(f'| {x:.3f} | {y:.2f} |')
        z=a['Z_at_end'][a['observed']]
        lines += ['', f"相邻均值中，沿τZ有{dense['adjacent_tau_mean_reversals']}处反向变化，沿Ith有{dense['adjacent_threshold_mean_reversals']}处反向变化，不能把小样本表面画成强制单调。进入时平均Z的实际范围为{z.min():.3f}–{z.max():.3f}；平均Z不能单独代表完整空间资源场和快状态。", '',
            '此扫描在固定网络和配对噪声下检验Z参数改变进入时间，支持模型内的参数作用。它没有改变I→E连接权重，也没有证明临床发作潜伏期、自主终止或Hopf分岔。人工补回只出现于左侧示例，F只统计首次进入，不混入补回干预。']
    else:lines += ['7×7×3加密仍在运行。完成后自动读取全部147条结果，补入数值表、删失数及实际的单调性检查；不以未完成网格作结论。']
    lines += ['', '## 交付与验收边界', '',
        '原生重放与原轨迹的spikes、E/I计数和虚拟SEEG逐点一致性检查通过。0.5毫米网格与原1毫米网格的计数守恒通过；3D轨迹保持原坐标和时间，灰色段落注释移出图面。新图为候选版本，人工目视验收仍待用户完成，未替换正式paper-ready图。', '']
    OUT.mkdir(exist_ok=True);(OUT/'scientific_review.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
