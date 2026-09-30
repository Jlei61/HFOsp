"""Scientific handover decision; no new simulation or model approval."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json
import numpy as np
from audit_topic4_kinetic_acceptance import ROOT,BASE,NATIVE,CAND,OUT

def main():
    a=json.loads((OUT/'independent_comparison.json').read_text())
    early=a['windows']['early_interictal'];n=early['SNN_9108401'];c=early['candidate_9108401'];n2=early['SNN_9108402']
    nt=a['transition']['SNN_9108401'];ct=a['transition']['candidate_9108401']
    fields=[('完整自限事件数','event_count'),('静息占比','quiet_fraction'),('平均 E 率 Hz/neuron','mean_E_hz'),
            ('事件时长中位数 ms','duration_median_ms'),('事件峰率中位数 Hz/neuron','peak_median_hz'),
            ('起始间隔 CV','IEI_CV'),('单事件招募 E 加权面积比例中位数','recruited_E_fraction_median')]
    table='\n'.join(f"| {label} | {n[key]:.4g} | {c[key]:.4g} | {n2[key]:.4g} |" for label,key in fields)
    prop='\n'.join(f"| {name} | {row['native_candidate']['IoU']:.3f} / {row['native_noise_baseline']['IoU']:.3f} | {row['native_candidate']['arrival_rho']:.3f} / {row['native_noise_baseline']['arrival_rho']:.3f} | {row['native_candidate']['offset_removed_error_ms']:.2f} / {row['native_noise_baseline']['offset_removed_error_ms']:.2f} |" for name,row in a['propagation_comparisons'].items())
    ck=list(np.load(CAND/'run/end_state.npz').files)
    report=f'''# 当前空间动理学候选：接手后的独立科学审阅

## 判定

接受“同参数空间通信近似可以自主产生早期间期样自限事件并走向持续高率”的可行性结果；暂不接受“与原 SNN 的间期、进入过程和二维传播已等效”，也不接受“已具备旧图那样的平衡/周期分岔延拓能力”。两个缺口分别属于物理对应与数学计算对象，不能用其中一项替代另一项。

接手时完整 0–12.5 s 轨迹和图均已完成，未发现仍在运行的该批仿真进程。本次没有重复启动仿真，而是读取原 SNN 两条完整参考轨迹及候选一条完整轨迹，独立复算事件、二维招募、到达顺序与慢变量，并生成新的对照图。上游结果与正式图保持原样。

## 模型身份与能否做分岔

当前候选是 40×40（0.5 mm）mean 通信闭合，32000 E + 8000 I 粒子，0.1 ms 时步，原实际图的 358 个延迟步，q_IE=1 原物理参数。Z、M 都逐粒子更新：E 净电流 I_A−Z I_G−0.0005 M；τZ=5 s、τM=1 s。外部全局/空间 OU 与逐粒子 Poisson 保留，当前完整轨迹从零开始；全程不输入原 SNN 的未来放电、Z 或 M。RNG/OU 在原生多个检查点一致，计数守恒已重新检查。

每个粒子保留 V、不应计数、四个突触变量、Z、M，另有空间延迟队列与外部 OU 状态。现阶段主要减少连接通信细节，**没有把 40000 个微观状态变成少量光滑率变量**。这依然是有脉冲阈值、复位和随机驱动的高维系统。

因此它可用于自主动力学、冻结资源的状态/响应图、噪声下的转变概率等分析，也可以作为将来群体分布方程或 equation-free 方法的微观推进器；但本交付没有实现可延续的群体方程、合格粗状态、提升/投影算子及稳定的导数估计。不能直接复用旧静态 Phi 模型的平衡残差、Jacobian 或周期 Floquet 程序；旧 q_IE=1.25 的 H/NS/SN 位置全部不属于当前 q_IE=1 候选。

在高维或随机系统上做 equation-free 分析有原始方法依据，但需要检验宏观状态是否闭合、微观初始化记忆是否衰减以及时间步映射是否收敛：[Sieber, Marschler & Starke, 2018](https://arxiv.org/abs/1701.08999)。有限粒子轨迹的随机进入、P-bifurcation/统计变化与确定性群体极限的 Hopf/fold 不是同一对象。若采用确定性极限，必须说明如何处理共同 OU 与有限粒子涨落，重新验证该对象的动力学，不能简单关掉噪声获得一个方便求根但物理不同的系统。

记录方面还缺一个具体接口：现有末态 NPZ 只有 {', '.join(ck)}，没有待到达延迟队列、绝对步号、Poisson RNG、全局 OU、空间 OU 状态/缓存及更新时钟。它不是可精确续接的完整检查点。当前轨迹有效，但不足以从任意时刻开始组成可重复的短时推进算子。

## 间期动力学：存在性已超过原先 8s 续接证据，等效尚未通过

窗口统一为 0.5–8 s；全局率按 10 ms 算 E 神经元加权平均。完整事件由低于 5 Hz 持续至少 20 ms 的低活动间隔分隔，事件峰率至少 20 Hz、事件段至少 20 ms；边界截断事件不计。下表不是独立网络的总体估计。

| 读出 | 原 SNN 输入1 | 候选同输入1 | 原 SNN 输入2 |
|---|---:|---:|---:|
{table}

候选的事件数、静息比例与 IEI-CV 接近参考，支持自主自限活动存在。CV≈0.36 仅表示该有限慢变窗口内的起始间隔有变异，不能独自证明混沌、排除噪声驱动振荡或证明稳定间期吸引子；此轨迹的 Z 持续漂移，并非在固定工作点上的长时平稳运行。

单事件空间招募读出为：1 mm 观察格的 5 ms 尾随 E 率至少 50 Hz 且持续 5 ms；事件开始前 10 ms 已达阈值的格左删失。比例按格内 E 细胞数加权，它不是逐神经元放电参与率。候选的招募比例中位数低于两条原生参考。把左删失格也纳入上界，三者中位数仍约为 69.7% / 49.1% / 64.1%，差异没有被左删失规则解释掉。

两核早期活动类别计数：原生1 A/B/both={n['early_core_counts'].get('A',0)}/{n['early_core_counts'].get('B',0)}/{n['early_core_counts'].get('both',0)}；候选={c['early_core_counts'].get('A',0)}/{c['early_core_counts'].get('B',0)}/{c['early_core_counts'].get('both',0)}。这是前 30 ms 两核率比的描述，不等于已确定事件起源。它提示不能只在给定类别下比较传播，再忽略类别频率本身的偏差。

## 二维传播：部分接近，但不能由一张空间快照验收

按相同早期核活动类别纳入所有跨事件对，不挑最相似事件；到达秩相关与去整体时间偏移后的误差只在双方均招募的格计算。每项为“原生1—候选 / 原生1—原生2”的所有配对中位数；配对和格点不独立，原生两条输入差异不是预先确定的等效容限。

| 窗口 | 招募集合 IoU | 到达秩相关 | 到达误差 ms |
|---|---:|---:|---:|
{prop}

全部间期事件池的条件化传播指标与原生跨噪声差异同量级，支持该路线保留了部分二维传播；但更早窗口的顺序、后段的类别组成、单事件招募大小和临近进入的传播时间仍不一致。整体池的相近指标不能抵消这些具体缺口。

## 进入持续高率：时钟、资源与传播须分开

操作性进入规则为全局 E 率连续 200 ms 至少 200 Hz。原生1/2 均为 9.87 s，完整候选为 {ct['high_onset_s']:.2f} s，晚 {ct['high_onset_s']-nt['high_onset_s']:.2f} s。原先“8s 原生状态续接”的 10.18/10.30 s 不是这条从零开始轨迹的进入时间。

在进入率窗开始之前最近的 5 ms 状态，原生1 D={nt['D_before_onset']:.6f}、M={nt['M_before_onset']:.3f}；候选 D={ct['D_before_onset']:.6f}、M={ct['M_before_onset']:.3f}。候选本次完整轨迹不应继续套用短续接的 D≈0.2581。全局均值 D 接近也不能替代完整 Z 场和快速历史的对应；当前完整轨迹只存了 Z/M 统计摘要，没有早期逐粒子 Z 场检查点。

在各自进入后的 0.5–1.5 s，原生1/候选平均率为 {nt['aligned_high_mean_E_hz']:.2f}/{ct['aligned_high_mean_E_hz']:.2f} Hz，达到局部 50 Hz 的 E 加权面积占比为 {nt['aligned_high_occupied_E_fraction']:.3f}/{ct['aligned_high_occupied_E_fraction']:.3f}。绝对时钟比较显示更大差距，不能把所有差异都归为形态错误，也不能通过对齐消除真实的资源推进偏差。

从各自最后完整低活动间隔结束到进入后 0.2 s，首次持续局部招募的空间 IoU≈0.900、到达秩相关≈0.467、去整体偏移误差中位数 118 ms；原生两条输入分别约 0.935、0.312、69 ms。这是多次活动共同构成的招募过程，不是单个波的传播速度，也不是同步指标。不能将高率标签直接称为已恢复发作同步性。

## 接手后的验收决定

1. 接受当前模型作为后续研究候选，并接受从零开始自主自限事件的存在性、原参数/双慢变量和输入一致性。
2. 动力学与传播仅部分恢复：当前候选完整轨迹仅一条，且空间招募范围与进入时序有实测偏差。先补同版本快响应和独立输入验证，不接受“科学等效 PASS”。
3. 经典分岔计算对象尚未完成。优先补可精确恢复的推进器，再判断粗状态闭合是否成立；若失败就增加必要状态/记忆，不能直接使用三变量率/Z/M 的均值闭合。
4. 原 20×20 shot 的冻结 Z 两端结果，不计作本 40×40 mean 的验证。旧率模型任何鞍结、Hopf/NS 与 separatrix 标签都不得迁移。

下一里程碑的范围、判断与停止条件见 `next_milestone.md`。本轮完成的是审阅和既有数据再分析，没有偷偷更改参数或启动新的分岔批次。全部图已 Agent 目检，用户人工验图待完成。
'''
    (OUT/'scientific_review.md').write_text(report)
    plan='''# 下一里程碑：先验收同一模型，再决定分岔对象

## 固定科学问题

原 Fig.5 参数 q_IE=1、同一实际连接图6101。候选固定40×40 mean、40000粒子、原阈值/电流/延迟/外驱、Z和M双动态；不再沿旧的 q_IE=1.25 率方程调分岔位置。先区分三件事：间期事件/传播是否对应，资源推进为何滞后，以及是否存在合格的可延续群体演化算子。

## K0：可恢复推进器与证据记录

把当前同一积分代码封装为 initialize/step/save_state/load_state，保存 V/ref/s_E/I_E/s_I/I_I/Z/M、延迟 mean ring（shot不进入本里程碑）、当前环位置/绝对步号、全局RNG和xi、空间OU随机状态/场/缓存/更新时钟。观察器只读；新代码与旧已保存canary逐位核对。分别在低活动、事件中、进入附近、高率期验证连续运行与存储恢复后分段运行逐位一致；不以总率相近替代。

完整轨迹记录至少包含0.5、4.025、8、9.30、9.42、9.87、10.37s的可恢复状态，以及自身最后低活动/进入状态。原生与候选都有对应记录。先通过恢复检查，再开展短时响应；缺队列的旧end_state不得直接恢复。

## K1：当前mean版本的固定Z配对响应

先用已有原生参考中可复用的 Z=8.00/9.42/9.87s 三张逐细胞场，8.00/10.37s 两种完整快速历史，各配两套已有外部输入，共12个配对条件。外部时钟统一原生协议10.37s，M保留历史初值且动态，Z冻结后仍乘GABA电流；不能把Z=1或禁用Z当作冻结。先比较共同2s窗口；仅对分类未决条件延长到与已有原生共同的10s窗口，不因某次例图漂亮改选条件。

读出沿用自限事件、低活动占比、局部持续/全局招募、核A/B先后、二维到达与资源条件，完整记录M与抑制电流过阈占比。判定首先看两端是否保留同样动力学，以及9.42s是否恢复原生局部/交替活动；若快响应不同，优先查格内连接异质性和共同输入近似。若冻结Z快响应相符而双动态运行仍滞后，才把问题定位到慢反馈积累及历史相关；不直接调整τZ来消除时间差。

## K2：从零开始的独立输入验收

第二条已有输入完整轨迹用于补充开发诊断，不冒充留出验收。冻结模型后先预定4对新的输入实现（同一拓扑，原生和候选配对）从0到12.5s；该数量是第一批精度检查，不保证统计功效，也不外推跨拓扑。

在运行前写定科学允许误差，联合原生跨输入分布报告，不以不显著差异宣布等效，不以本轮观察到的偏差反向放宽界限。主读出为分段0.5–3/3–6/6–8s的事件数、时长和起始间隔分布、核参与类别比例、空间招募比例/概率场、类别内到达顺序与时延；进入过程同时报告绝对时间、D与完整Z/M场、按进入对齐的招募。固定工作点间期长时行为要由冻结Z试验支持，不能用资源漂移轨迹代替。需要全局同步性结论时，另定义本地率共同起落和延迟结构；全局200Hz阈值只作高率状态。

若空间招募系统性不足或某类传播模式丢失，返回模型闭合层；不要靠相位对齐、挑事件、增加q_IE、缩放空间图或只报告全局率通过验收。两端通过但中间失败，不能启动目标边界分岔解释。

## K3：分岔对象可行性，只做局部验证

在K1/K2通过的物理工作点固定完整Z_i(D)路径（全部0≤Z_i≤1，D仍为1−mean_E Z），M继续动态。先明确定义目标：确定性群体分布极限的分岔，或有限噪声过程的统计/随机稳定性；两者不得混称。

优先以保留V/不应年龄/突触电流/M联合分布及延迟记忆的演化对象开始，不能假定只保留平均率足够。若用equation-free路线，明确restriction/lifting并检查同一粗状态的不同微观重建是否给出一致未来分布；检查healing时间和推进窗变化、粒子数/重复数与扰动幅度下的一致性。宏观映射、导数及主要谱模态必须在噪声误差内收敛；简单固定随机数做一次轨迹差分不足以证明。

至少在一个低态工作点和一个进入附近工作点，分别与候选直接模拟及原SNN小扰动响应核对。若粗状态保留不了必要记忆，增加状态或停止该降阶对象；若只能给转变概率/寿命，交付统计状态图，不伪造平衡/周期分支。通过后再开下一批pseudo-arclength、周期延拓和稳定性判定，临界点一律重新计算。

## 交回审阅的交付

同版本参数/输入身份、完整恢复QA；逐条件原生—候选读出和例图；所有中间/负结果；分开的动力学对应与分岔对象判定表；可运行脚本及版本固定。未通过对应性前不以旧星号、旧周期包络或D≈0.26作为机制结论。当前计划未自动派发新的仿真批次。
'''
    (OUT/'next_milestone.md').write_text(plan)
    decision=dict(status='REVIEW_COMPLETE',original_full_simulation='COMPLETE_0_TO_12_5S',
        candidate_identity='g40_mean_qIE1_40000_particles',
        autonomous_interictal_events='CONFIRMED_SINGLE_FULL_TRAJECTORY',
        original_parameters_and_dynamic_Z_M='CONFIRMED',
        dynamics_and_propagation_equivalence='PARTIAL_NOT_ACCEPTED_AS_EQUIVALENT',
        direct_classical_bifurcation_readiness='NOT_READY',
        current_end_state_restart='MISSING_DELAY_AND_INPUT_STATES',
        recommended_route='validate correspondence, construct and test resumable/coarse evolution object, then continuation',
        old_bifurcation_labels_transferable=False,new_simulation_launched=False,
        independent_postprocessing='COMPLETE',human_visual_acceptance='PENDING')
    (OUT/'review_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print('REVIEW WRITTEN',flush=True)

if __name__=='__main__':main()
