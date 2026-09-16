"""Assemble the completed bounded experiment; never report a partial batch as final."""
from pathlib import Path
import json,sys
import numpy as np
from analyze import OUT,OLD,read,write

def span(values,digits=3):
    vals=[v for v in values if v is not None]
    if not vals:return '不可估计'
    return f'{min(vals):.{digits}f}–{max(vals):.{digits}f}'

def main():
    base=read(OUT/'baseline_observables.json');noise=read(OUT/'noise_comparisons.json')
    valid=read(OUT/'native_validation.json');quiet=read(OUT/'quiescent_branch.json')
    returns=read(OUT/'deterministic_return.json');rec=read(OUT/'native_recovery_diagnostics.json')
    cell=read(OUT/'single_cell_irregularity.json')
    assert valid['completed']==valid['expected']==16
    assert all(r['duration_ms']==20000 and r['runaway_ms'] is None and all(r['prefix_bitwise'].values()) for r in valid['rows'])
    bt=[];nt=[];ct=[];vt=[]
    for ee in [.5,.7,.85,1.,1.2]:
        rr=[r for r in base if r['depth']==1 and r['ee']==ee]
        bt.append('|'+f'{ee:g}|'+span([r['metric']['mean_rate_hz'] for r in rr])+ '|'+span([r['metric']['n_bursts'] for r in rr],0)+'|'+span([r['metric']['cv'] for r in rr])+ '|'+span([r['metric']['median_peak_active_fraction_2ms'] for r in rr])+'|')
    for ee in [.7,.85,1.,1.2]:
        a=[r for r in noise if r['ee']==ee and r['contrast']=='ou_removed' and r['context']=='baseline']
        b=[r for r in noise if r['ee']==ee and r['contrast']=='ou_removed' and r['context']=='intervention']
        assert len(a)==len(b)==4
        nt.append('|'+f'{ee:g}|'+span([r['metric']['n_bursts'] for r in a],0)+'|'+span([r['metric']['n_bursts'] for r in b],0)+'|'+span([r['metric']['cv'] for r in b if r['metric']['n_bursts']>=8])+'|'+span([r['metric']['mean_rate_hz'] for r in b])+'|')
        cr=[r for r in cell if r['ee']==ee and r['window']=='after']
        ct.append('|'+f'{ee:g}|'+span([r['eligible_cells'] for r in cr],0)+' / '+span([r['sampled_cells'] for r in cr],0)+'|'+span([float(np.median(r['cell_isi_cv'])) for r in cr if r['cell_isi_cv']])+'|')
        vr=[r for r in rec if r['ee']==ee and r['window']=='after']
        vt.append('|'+f'{ee:g}|'+span([r['minimum_population_mean_V_mV'] for r in vr],1)+'|'+span([r['peak_local_I_rate_hz'] for r in vr],1)+'|')
    det=[r for r in noise if r['arm']=='all_off_probe' and r['context']=='intervention']
    det_spikes=[r for r in returns if 'group_spike_counts' in r]
    all_quiet=all(all(v==0 for v in r['group_spike_counts'].values()) for r in det_spikes)
    assert all_quiet, 'Adapt interpretation: deterministic activity persisted.'
    pulse=[r for r in returns if 'total_spikes' in r]
    a_pulse=[r for r in pulse if r['group']=='coreAE']
    b_pulse=[r for r in pulse if r['group']=='coreBE']
    native_eq_error=max(r['final_second_max_voltage_eq_error_mV'] for r in pulse)
    # Eligibility and sample identities remain per core/run, never pooled as network replicates.
    representative=next(r for r in base if r['ee']==.85 and r['depth']==1 and r['seed']==848101 and r['group']=='coreAE')
    q=np.quantile([e['peak_active_fraction_10ms'] for e in representative['events']],[.1,.5,.9])
    strong=[r for r in noise if r['ee']>=.85 and r['contrast']=='ou_removed' and r['context']=='intervention']
    summary=dict(status='BOUNDED_BATCH_COMPLETE',baseline_native_runs=50,new_native_runs=16,topology=2511,
        duration_s=20,active_scope='core burst onset; no seizure/Z/M analysis',
        deterministic_all_silent_after_removal_and_late_pulse=all_quiet,
        constant_input_equilibrium_native_voltage_error_mV=native_eq_error,
        poisson_only_strong_n_bursts_range=[min(r['metric']['n_bursts'] for r in strong),max(r['metric']['n_bursts'] for r in strong)],
        deterministic_silent_branch_local_bifurcation='ABSENT_ALONG_TESTED_EE_AT_DEPTH_1',
        noisy_population_bifurcation_type='NOT_IDENTIFIED',
        intermediate_third_attractor='NOT_ESTABLISHED',
        material_model_limitation='Deep post-burst hyperpolarization and high local I firing in the current-based LIF; recovery may influence the burst rhythm.',
        human_visual_acceptance='PENDING')
    write(OUT/'summary.json',summary)
    report=fr'''# Core burst 如何开始：Brunel–Hakim 启发的第一版结果

本版检索并阅读六篇相关原始研究，复用原有50条原生SNN网格轨迹，新增16条20秒连续状态干预；全部完成。范围是core内部burst起始与中间表型，未启用Z/M，也不研究发作转变。

**当前判断：本模型能显示从低活动、小范围不规则招募到强群体burst的变化；共享慢OU不是重复burst的必要条件。恒定输入下的全静默分支在EE扫描中始终局部稳定，本次活动前史和有限刺激均未维持无噪声重复burst。噪声背景下是否存在群体Hopf仍未判定，中间表型也未证明是第三个吸引子。**

本版另发现会影响机制解释的限制：强burst后，电流型LIF的core平均电位下降数百mV，局部I出现很高峰率。恢复过程可能塑造低频重复间隔，应先核查这一模型特征，再接受其生理机制解释。

## 原始研究怎样指导这版

[Brunel & Hakim 1999](https://webhome.phy.duke.edu/~nb170/pdfs/brunel99.pdf)将稳定性推导、临界附近非线性展开和有限规模模拟结合，说明振荡自相关不能独自确认Hopf。[Brunel 2000](https://webhome.phy.duke.edu/~nb170/pdfs/brunel00JCNS.pdf)的参数图与配套rate/raster最接近本任务，但其“synchronous irregular”主要描述单细胞发放，而不是群体burst事件间隔。

[Brunel & Wang 2003](https://webhome.phy.duke.edu/~nb170/pdfs/brunel03JNP.pdf)及[Ledoux & Brunel 2011](https://webhome.phy.duke.edu/~nb170/pdfs/ledoux11.pdf)提示需要保留神经元动态响应、突触滤波与不同反馈路径；静态平均率吻合不足以认定相同分岔。[Brunel & Hansel 2006](https://webhome.phy.duke.edu/~nb170/pdfs/brunel06.pdf)进一步区分聚簇和群体率振荡失稳。逐篇适用条件、阅读位置及独立基准推导见[literature_review.md](literature_review.md)。

由此采用三层证据：原生SNN的实际波形与区域；拆开共享OU/私有Poisson、保留连续状态的干预；可以严格计算的静默分支稳定性。独立复算的Brunel–Hakim 2008时延率例子单独成页，未当作当前SNN降阶模型。

## 参数、观测与统计单位

网络为32,000 E + 8,000 I；Core A有720个E细胞，Core B有742个。仿真包含完整空间网络，两个core仍与周边和彼此耦合。固定拓扑2511、原有双核几何、GABA衰减18ms、原生0.1ms积分和只降低E阈值的空间场。核外输入保留原确定性形式。

- **X轴**：同一core内部E→E突触权重倍率；不同时缩放E→I或GABA。
- **Y轴**：原逐细胞降阈值场的整体幅度，\(V_{{\theta,i}}=V_{{\theta,i}}^{{bg}}-d\,\Delta_i\)，\(\Delta_i\geq0\)。它同时改变降幅均值与离散度，**不是固定均值的阈值异质性轴**。连接异质性本版固定。
- 群体率：每2ms spike数除以该core全部E细胞数和bin时长，单位Hz/细胞。raster只展示固定100或30个细胞，不能用raster抽样数作群体率分母。
- 事件：10ms内活跃E细胞比例达到10%，3%作为持续阈值，间隙不超过20ms合并。首尾截断按原规则处理，至少8个非左截断事件才解释IEI-CV。7.5%/12.5%启动阈值另存敏感性结果。
- 独立随机实现是每条网络轨迹；事件和细胞均是运行内读出。当前两噪声种子、同一张拓扑，两个core不能当成独立网络重复。表中范围保留四个core×seed读出，不合并事件估算CV。

主网格为EE=0.5/0.7/0.85/1/1.2、降幅=0/0.4/0.7/1/1.3，各两个种子；分析2–20秒。下文一维分析固定降幅1。

## 原生连接扫描：burst增强与规则化

|核内EE倍率|平均率范围，Hz/细胞|18秒burst数|IEI-CV|每运行burst峰值2ms参与比例的中位数|
|---|---|---|---|---|
{chr(10).join(bt)}

0.5未检测到burst，0.7出现不规则小范围招募；0.85的平均率、峰率和招募范围明显上升，1/1.2的事件间隔与招募更一致。完整群体率的均值和时间标准差也随EE变化，说明上述变化并非仅由事件检测阈值产生。这里只有离散参数响应，尚未定位数学临界点。

中间条件0.85并非额外预设的模型状态。代表Core A、种子848101的10ms峰值参与比例10/50/90%分位数为 **{q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}**，实际混合部分招募和接近全核招募。四个读出的CV约0.278–0.350，活动自相关峰约0.068–0.199，未同时满足既定规则性条件。以Core B种子848101为例，CV已低于0.30，但自相关峰仅0.068，因此仍归为中间表型。这说明标签依赖多个连续读出，不能推导第三个吸引子、混沌或准周期分岔。

**合理的工作解释**是：提高EE增强了有限涨落触发的递归招募，强事件的招募与恢复越来越一致。原生有限脉冲后可见兴奋招募、随后占优的抑制输入和活动停止；这是机制线索。当前未独立扰动该抑制反馈，也未计算噪声背景群体特征根，因此不把这一解释当成已证明的分岔类型。

## 共享OU与私有Poisson的分离

新增协议在6秒移除同一core共享的150ms OU，仅保留固定强度的每细胞Poisson输入。强度匹配整流OU输入的平稳期均值；对应旧基线使用同样8–20秒比较窗。干预前6秒的群体计数、参与比例和raster逐位一致，干预后仅保留共同前史/种子，不声明逐一相同的Poisson创新。

|核内EE倍率|原输入：12秒burst数|仅Poisson：12秒burst数|仅Poisson：合格读出的IEI-CV|仅Poisson：平均率，Hz/细胞|
|---|---|---|---|---|
{chr(10).join(nt)}

EE≥0.85时去OU后仍持续出现重复burst，支持慢共享OU不是逐次事件的必要触发器；EE=0.7仍有少量burst，但部分读出不足8事件。**保留Poisson并不等于无噪声极限环**：私有涨落仍维持亚阈值神经元的发放，并可能与网络反馈共同形成群体节律。

弱连接条件的事件数对检测阈值很敏感：仅Poisson、EE=0.7时，7.5%阈值检出20–27次，12.5%阈值仅2–5次。因此不能把10%阈值下的次数减少直接解释为潜在触发次数减少，也不能稳定估计其去OU后的CV。EE=1/1.2的次数在三种阈值下逐条相同，EE=0.85在三种阈值下均保留大量重复事件，支持强连接部分的上述判断。

## 恒定输入下的返回与单次点火

另一协议在6秒把OU和Poisson随机性均换成匹配均值，保留当时全部动态状态；12秒强制Core A每个E细胞各发一个spike，再无额外刺激。四个EE×两个种子共8条轨迹中，8–12秒及14–20秒窗口的**全E、全I spike计数均为0**。

单次点火确实诱发了后续发放，并非只有输入的720个spike：Core A在12–14秒共{span([r['total_spikes'] for r in a_pulse],0)}个spike，其中720个为强制。Core A最后有spike的2ms bin在刺激后{span([r['last_occupied_bin_end_after_pulse_ms'] for r in a_pulse],0)}ms内；另一个core也被招募，最后发放在{span([r['last_occupied_bin_end_after_pulse_ms'] for r in b_pulse],0)}ms内。这里是单次空间脉冲的有限响应，未证明单核与外部网络解耦。

因此，本次访问的活动状态在完全去噪后未自维持；一次较强点火产生自限响应后返回。它不排除未测试初态、其他脉冲或其他参数存在活动吸引子，也不等于排除了噪声背景群体Hopf。

同一EE的两个种子在刺激前已收敛至同一确定性平衡点，因而诱发响应相同；这不构成两个独立网络的重复证据。

## 本版真正完成的原生稳定性计算

在上述恒定均值输入下，直接由原生离散方程计算全静默平衡点。无递归spike输入时，令外部输入率为\(\nu_i\)、门控跃增为\(a_i\)，则

\[
s_{{E,i}}^*=I_{{E,i}}^*=V_i^*=\frac{{\nu_i\Delta t\,a_i}}{{1-e^{{-\Delta t/\tau_{{r,E}}}}}},\qquad s_{{I,i}}^*=I_{{I,i}}^*=0.
\]

全部40,000细胞均低于实际阈值。Core A/B的\(V^*\)均为12.22237mV，最小阈值裕量分别 **2.06902/2.42909mV**。因为存在严格的阈下邻域，spike输出映射在该邻域的导数为0，递归EE权重不进入局部雅可比。神经元/突触连续状态的特征值为对应的膜与滤波衰减因子，有限延迟队列移位为幂零；最大模为 **0.99501248**，等效主导衰减率 **−50 s⁻¹**，沿EE不变。

因此，**在降幅1和所扫EE范围内，这条确定性全静默分支没有局部Hopf或saddle-node**。这是原生方程的局部结论，不是由“去噪没有spike”猜出的稳定性。一步平衡残差为0，独立中心差分雅可比/特征值最大误差低于6×10⁻¹³；去噪轨迹末秒的core平均电位与该平衡点最大差为{native_eq_error:.2g}mV。

它没有回答非零噪声群体工作点的稳定性。后者需要维持原输入方差、阈值分布和空间反馈，检验小扰动的群体响应，并与可信的群体近似匹配；不能用这条0-spike分支代替。

## 会改变接受判断的恢复过程

新增原生电流/电位记录显示，在仅保留Poisson的8–20秒窗口：

|核内EE倍率|每core运行的最低群体平均V，模型mV|每core运行的局部I峰率，Hz/细胞|
|---|---|---|
{chr(10).join(vt)}

执行器采用\(\tau_m\dot V=-V+I_E-I_I\)对应的离散更新，当前路径没有GABA反转电位所提供的下界。上述V采用模型参考坐标，reset=11mV；不能把数百mV的下降当作真实细胞的正常膜电位范围。图14展示自发burst后深度超极化与缓慢返回，**不是强制点火才出现的现象**。

这给低频重复周期提供了必须检验的替代解释：强递归招募造成大的抑制尾部和电位偏离，下一次点火受恢复过程制约。尚未量化恢复对周期的因果贡献，本版也没有修改该物理路径。强同步、多次spike/burst的当前条件同时偏离经典稀疏弱相关、每周期少量spike的近似，不能照搬Brunel的Hopf临界公式。

## 单细胞ISI与群体IEI

新记录器保存固定样本的真实0.1ms spike时刻。仅Poisson窗口中，至少21个spike的细胞才计算ISI-CV；这些细胞的中位数在四个core×seed读出中的范围如下。

|核内EE倍率|合格细胞数 / 实际采样细胞数范围|每运行合格细胞ISI-CV中位数|
|---|---|---|
{chr(10).join(ct)}

这些ISI包含burst内部短间隔和burst之间长静默，**高ISI-CV不等于burst内部随机**，也不证明经典的稀疏同步态。低率条件的细胞资格筛选偏差已保留，图10只用于说明两种irregularity不能混用。

## 接受边界与后续路线

接受本版作为当前原生模型的burst起始、招募与噪声依赖诊断图；暂不接受“中间态由某种特定分岔产生”或“规则burst是已确认自主极限环”的结论。颜色图中的中间区域和连线均不是数学分岔边界。

下一版的决定性先后关系是：先核查限制过深超极化的生物物理实现能否在匹配低活动工作点下保留招募和规则化；若保留，再在维持私有涨落的背景下测动态响应，匹配空间/分区群体近似并延拓其平衡与周期分支。若规则burst主要随恢复路径修正而消失，就应优先解释当前模型的恢复机制，不继续给旧区域贴Hopf标签。若保留且出现共轭特征根过零，才进入Hopf及其超/亚临界类型检验。本轮16条轨迹已结束，未自动开展新的物理版本或扫描。

## 交付与验证

- [完整图册](figures/core_burst_onset_v1_booklet.pdf)：二维区域、连续指标、四状态100/30细胞大图、噪声干预、有限点火、恢复过程、静默稳定性与独立文献基准。
- [逐图说明](figures/README.md)、[逐运行比较表](noise_comparisons.csv)、[摘要](summary.json)。
- [原生一致性检查](native_validation.json)、[静默平衡与差分检查](quiescent_branch.json)、[完整去噪返回计数](deterministic_return.json)、[恢复读出](native_recovery_diagnostics.json)。
- [独立文献基准](literature_hopf_benchmark.json)：Brunel–Hakim 2008图2时延rate方程，Kc=8.502425、频率134.381Hz，步长减半幅度差约0.008%；这些值不是当前SNN的EE阈值或burst频率。

16/16条新轨迹均完整20秒，无runaway提前停止，干预前记录逐位复现；常量输入与精确spike记录通过检查。图像的Agent目视结果见visual_review.json，用户目视验收待定。脚本位于`{OUT.parents[2] / 'scripts/topic4_core_burst_onset_v1'}`；完整命令和结果入口见README.md。原执行器、旧图和旧结果未替换。
'''
    (OUT/'scientific_report.md').write_text(report)
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
